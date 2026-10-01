//! The `R_162` slot fold of `fold_commitment` (`sum_j c_j C_j` per modulus): upstream's
//! `simd/slots.rs`, ported. The Horner-per-slot scalar path (a 162-term Horner per slot per
//! challenge) is replaced by a precomputed **slot table**: the transformed slots of the 162 unit
//! challenges `e_p` of `R_162`, so any short challenge's slots are a signed sum of table rows —
//! linearity of the embedding `c -> c(-X^4)` and of the slot map — accumulated in i32 lanes and
//! reduced once. The fold itself then runs as `vpmaddwd` MACs of the challenge slots against the
//! commitment columns' 162-slot rows, folded back on [`slot_period`], with the interleaved
//! even/odd dword form `vpmaddwd` consumes directly.
//!
//! Bit-exactness against the scalar `challenge_r162_slots` + Horner path is pinned by
//! `tests/round.rs` (the commitment fold feeds the same transcripts).

use crate::challenge::ShortChallenge;
use crate::ring::N162;
use crate::simd::commit as cm;
use core::arch::x86_64::*;

pub const LANES: usize = 192;
pub const VECS: usize = LANES / 16;
pub const BLOCKS: usize = LANES / 32;
const TAIL: __mmask32 = ((1u64 << (N162 - 32 * (BLOCKS - 1))) - 1) as __mmask32;

#[repr(C, align(64))]
pub struct SlotTable {
    pub rows: [[i16; LANES]; 2 * N162],
}

impl SlotTable {
    pub fn zero() -> Box<SlotTable> {
        // Safety: all-zero bytes are a valid SlotTable (i16 lanes all zero).
        unsafe {
            let mut b = Box::<SlotTable>::new_uninit();
            core::ptr::write_bytes(
                b.as_mut_ptr() as *mut u8,
                0,
                core::mem::size_of::<SlotTable>(),
            );
            b.assume_init()
        }
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Ch {
    pub e: [[i16; 32]; BLOCKS],
    pub o: [[i16; 32]; BLOCKS],
}

impl Ch {
    pub fn zero() -> Ch {
        Ch {
            e: [[0i16; 32]; BLOCKS],
            o: [[0i16; 32]; BLOCKS],
        }
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct SlotAcc {
    pub v: [[i32; 16]; 2 * BLOCKS],
}

impl SlotAcc {
    pub fn zero() -> SlotAcc {
        SlotAcc {
            v: [[0i32; 16]; 2 * BLOCKS],
        }
    }
}

pub const fn slot_period(q: u16) -> usize {
    cm::period_for(q, cm::a_bound(q) * cm::a_bound(q))
}

/// The slots of one short challenge, accumulated from the table rows and fully reduced.
///
/// # Safety
/// AVX-512 F/BW/DQ; `t` 64-byte aligned covering `2 * N162` rows of `LANES` i16.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq")]
pub unsafe fn challenge_slots<const Q: u16>(t: &SlotTable, c: &ShortChallenge, out: &mut Ch) {
    let mut acc = [_mm512_setzero_si512(); VECS];
    for i in 0..c.weight {
        let r = t.rows[2 * c.positions[i] as usize + ((c.signs >> i) & 1) as usize].as_ptr();
        for (j, a) in acc.iter_mut().take(VECS - 1).enumerate() {
            let x = _mm512_cvtepi16_epi32(_mm256_load_si256(r.add(16 * j) as *const __m256i));
            *a = _mm512_add_epi32(*a, x);
        }
    }
    let half = _mm512_set1_epi32(((Q - 1) / 2) as i32);
    let qv = _mm512_set1_epi32(Q as i32);
    let mut red = [_mm256_setzero_si256(); VECS];
    for (j, a) in acc.iter().enumerate() {
        let x = cm::mod_q::<Q>(*a);
        let x = _mm512_mask_sub_epi32(x, _mm512_cmpgt_epi32_mask(x, half), x, qv);
        red[j] = _mm512_cvtepi32_epi16(x);
    }
    let em = _mm512_set1_epi32(0x0000ffffu32 as i32);
    let om = _mm512_set1_epi32(0xffff0000u32 as i32);
    for b in 0..BLOCKS {
        let v = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(red[2 * b]), red[2 * b + 1]);
        _mm512_store_si512(
            out.e[b].as_mut_ptr() as *mut __m512i,
            _mm512_and_si512(v, em),
        );
        _mm512_store_si512(
            out.o[b].as_mut_ptr() as *mut __m512i,
            _mm512_and_si512(v, om),
        );
    }
}

/// `acc += ch . column` over the 162 slots (the last block masked to the true length).
///
/// # Safety
/// AVX-512 F/BW; `e` covers `N162` i16, `acc`/`ch` 64-byte aligned per their layout.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn mac_slots(acc: &mut SlotAcc, ch: &Ch, e: *const i16) {
    for b in 0..BLOCKS {
        let x = if b + 1 == BLOCKS {
            _mm512_maskz_loadu_epi16(TAIL, e.add(32 * b))
        } else {
            _mm512_loadu_si512(e.add(32 * b) as *const __m512i)
        };
        let pe = _mm512_madd_epi16(
            _mm512_load_si512(ch.e[b].as_ptr() as *const __m512i),
            x,
        );
        let po = _mm512_madd_epi16(
            _mm512_load_si512(ch.o[b].as_ptr() as *const __m512i),
            x,
        );
        let pa = acc.v[2 * b].as_mut_ptr() as *mut __m512i;
        let pb = acc.v[2 * b + 1].as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(pa, _mm512_add_epi32(_mm512_load_si512(pa), pe));
        _mm512_store_si512(pb, _mm512_add_epi32(_mm512_load_si512(pb), po));
    }
}

/// The exact fold-back of [`crate::simd::commit`] over the slot accumulator.
///
/// # Safety
/// AVX-512 F/BW.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn reduce_slots<const Q: u16>(acc: &mut SlotAcc) {
    for v in acc.v.iter_mut() {
        let p = v.as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(p, cm::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i)));
    }
}

/// The accumulator drained to the four centered `R_162` rows of the folded commitment.
///
/// # Safety
/// AVX-512 F/BW/DQ; `acc` per its layout, `out` covering `N162` i16.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq")]
pub unsafe fn finish_slots<const Q: u16>(acc: &SlotAcc, out: &mut [i16; N162]) {
    let half = _mm512_set1_epi32(((Q - 1) / 2) as i32);
    let qv = _mm512_set1_epi32(Q as i32);
    let mut tmp = [[0i16; 16]; 2 * BLOCKS];
    for (j, v) in acc.v.iter().enumerate() {
        let x = cm::mod_q::<Q>(_mm512_load_si512(v.as_ptr() as *const __m512i));
        let x = _mm512_mask_sub_epi32(x, _mm512_cmpgt_epi32_mask(x, half), x, qv);
        _mm256_storeu_si256(tmp[j].as_mut_ptr() as *mut __m256i, _mm512_cvtepi32_epi16(x));
    }
    for b in 0..BLOCKS {
        for k in 0..16 {
            let s = 32 * b + 2 * k;
            if s < N162 {
                out[s] = tmp[2 * b][k];
            }
            if s + 1 < N162 {
                out[s + 1] = tmp[2 * b + 1][k];
            }
        }
    }
}
