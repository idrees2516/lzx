//! The Ajtai commitment multiply-accumulate, in the NTT domain: one row of an inner product
//!
//! ```text
//!     y[j] = sum_i A_i[j] * NTT_q(w_i)[j]  mod q,      j = 0..648,
//! ```
//!
//! where the `w_i` are the ring elements lifted from a stream of `F162` and `A` is a fixed row
//! of uniform NTT-domain ring elements. Port of upstream `simd/commit.rs` (the pure-intrinsics
//! parts; the block-sink fusion and the A-prefetch machinery of upstream's `asm!` production
//! path are simplified away — this port materialises each batch's transform and then multiplies
//! it, which upstream measures at 874 against 630 cycles per ring element for the fused sink,
//! both far ahead of the scalar path).
//!
//! # The layout of A
//!
//! `A` is a slice of [`Batch32`] in exactly the layout the transform writes: `a.v[j][p]` is
//! slot `j` of `A_{32b + p}`, one 64-byte vector per slot per batch, 41472 bytes per batch.
//! Entries are **centered**, `|A| <= (q-1)/2`.
//!
//! # Raw accumulation
//!
//! A slot product is never reduced. A `vpmaddwd` per slot per batch does `acc32 += W_j * A_j`
//! on adjacent pairs of 16-bit lanes, so a 32-bit lane carries the running sum of two ring
//! elements' products: one multiply-port uop per slot per batch, against the 4 multiply-port
//! uops a Montgomery slot product would cost. Nothing but the sum is ever needed, and
//! `sum_i A_i[j] W_i[j]` is congruent mod q whatever representatives the transform leaves, so
//! the only question is overflow.
//!
//! # The exact fold-back
//!
//! The transform's output is lazily reduced to [`w_bound`] (7.5 q for q = 3889, 2.294 q for
//! q = 9721) and `|A| <= (q-1)/2`, so one batch adds at most [`acc_per_batch`] to a lane. Every
//! [`red_period`] batches the accumulator is folded back into `|acc| <= 2^15 (1 + R)`
//! ([`acc_after_reduce`], `R = 2^16 mod q`) by [`reduce_acc_i32`]: three uops per accumulator
//! vector, one of them on the multiply port, amortised over 8 (q = 3889) or 4 (q = 9721)
//! batches. The period is the largest power of two for which `acc_after_reduce + period *
//! acc_per_batch` still fits `i32`, which the `fits` assertions below check at compile time.
//!
//! # The packed accumulator
//!
//! Folding the 16 lanes of a slot down to 8 and packing two slots into one vector
//! ([`pack2`]: two `vshufi64x2` and one `vpaddd`) cuts the accumulator to 21.5 KB and its
//! traffic by a third for the same uop count. Layout: the 648 slots are 24 consecutive 27-slot
//! blocks; inside a block, vector `p < 13` carries slot `2p` in lanes 0..8 and slot `2p+1` in
//! lanes 8..16; vector 13 carries the odd slot 26 in lanes 0..8 and a harmless duplicate of it
//! in lanes 8..16 ([`slot_lane`]). A lane carries four products per batch.
//!
//! # Quadratic-slot limbs
//!
//! For q in `QS_QUAD` the ring does not split completely: the transform ends at 324 quadratic
//! leaves `Z_q[X]/(X^2 - c_j)`, and the commitment wants the sum over ring elements of the
//! quadratic product, so the three sums `P_0 = sum a_0 b_0`, `P_1 = sum a_1 b_1`,
//! `P_2 = sum (a_0 b_1 + a_1 b_0)` are accumulated raw and combined once per leaf at the end:
//! `y[2j] = P_0 + c_j P_1`, `y[2j+1] = P_2`. `P_2` is one `vpmaddwd` instead of two when formed
//! as `(a_0 + a_1)(b_0 + b_1) - P_0 - P_1` (Karatsuba; possible when `2 * w_bound_quad <=
//! 2^15`, i.e. q = 2917).

use crate::params::*;
use crate::simd::ntt_quad;
use crate::simd::ntt_small;
use crate::simd::transpose::BinaryIndex32;
use crate::simd::Batch32;
use core::arch::x86_64::*;

// =============================================================================================
// bounds
// =============================================================================================

/// `R = 2^16 mod q` (3312 for q = 3889, 7210 for q = 9721): the weight the high half of an i32
/// accumulator lane carries into the low half.
pub const fn r16(q: u16) -> i32 {
    (65536 % q as u32) as i32
}

/// Bound on one lane of the transform's output: the split kernel's 7.5 q (3889) and 2.294 q
/// (9721), the large-prime kernel's declared bound (17497 / 19441), the quad kernel's declared
/// bound (2917 / 4861 / 12637).
pub const fn w_bound(q: u16) -> i64 {
    if quadratic_slots(q) {
        ntt_quad::output_bound(q) as i64
    } else if crate::simd::ntt_large::is_large(q) {
        crate::simd::ntt_large::output_bound(q) as i64
    } else {
        (ntt_small::output_bound_milli_q(q) as i64 * q as i64) / 1000
    }
}

/// Bound on one lane of A: the matrix is stored centered.
pub const fn a_bound(q: u16) -> i64 {
    ((q - 1) / 2) as i64
}

/// What one batch adds to one accumulator lane: four products.
pub const fn acc_per_batch(q: u16) -> i64 {
    4 * w_bound(q) * a_bound(q)
}

/// Bound on a lane straight after [`reduce_acc_i32`]: `|l| <= 2^15` and `|h + c| <= 2^15`, so
/// `|l + (h + c) R| <= 2^15 (1 + R)`.
pub const fn acc_after_reduce(q: u16) -> i64 {
    32768 * (1 + r16(q) as i64)
}

/// Batches accumulated between two fold-backs: the largest power of two P with
/// `acc_after_reduce + P * acc_per_batch <= i32::MAX` (8 for 3889, 4 for 9721).
pub const fn red_period(q: u16) -> usize {
    period_for(q, acc_per_batch(q))
}

pub const fn period_for(q: u16, per: i64) -> usize {
    let mut p = 1usize;
    while acc_after_reduce(q) + 2 * (p as i64) * per <= i32::MAX as i64 {
        p *= 2;
    }
    p
}

const fn fits(q: u16) -> bool {
    acc_after_reduce(q) + (red_period(q) as i64) * acc_per_batch(q) <= i32::MAX as i64
}
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!(fits(QS[i]));
        i += 1;
    }
};
const _: () = assert!(red_period(3889) == 8 && red_period(9721) == 4);
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!(fits(QS_LARGE[i]) && red_period(QS_LARGE[i]) >= 1);
        i += 1;
    }
};

/// The exact fold-back, lane-wise (the scalar model of `reduce_vec`).
///
/// Write the i32 lane as `x = 2^16 h + u`, `h = x >> 16` (arithmetic), `u = x & 0xffff`, and let
/// `l` be `u` read as an i16. Then `x = l + (h + c) R (mod q)` exactly with
/// `c = [bit 15 of x]`, and the result satisfies `|l + (h + c) R| <= 2^15 (1 + R)`.
///
/// In vector form this is `vpmaddwd(acc, [1, R])` plus `+ R` on the lanes whose bit 15 is set
/// (`vptestmd` + masked `vpaddd`): three uops per vector, one on the multiply port.
#[inline]
pub fn reduce_acc_i32(x: i32, q: u16) -> i32 {
    let l = ((x as u32) as u16) as i16 as i32;
    let h = x >> 16;
    let c = (x >> 15) & 1;
    l + (h + c) * r16(q)
}

/// The vector form of [`reduce_acc_i32`] (three uops, one on the multiply port).
///
/// # Safety
/// AVX-512 F/BW.
#[inline(always)]
pub unsafe fn reduce_vec<const Q: u16>(x: __m512i) -> __m512i {
    let k = _mm512_set1_epi32(1 | (r16(Q) << 16));
    let r = _mm512_madd_epi16(x, k);
    let m = _mm512_test_epi32_mask(x, _mm512_set1_epi32(0x8000));
    _mm512_mask_add_epi32(r, m, r, _mm512_set1_epi32(r16(Q)))
}

// =============================================================================================
// the packed accumulator
// =============================================================================================

/// Accumulator vectors per 27-slot block, and in total (21.5 KB).
pub const ACC_PER_BLK: usize = 14;
pub const ACC_VECS: usize = 24 * ACC_PER_BLK;

#[repr(C, align(64))]
pub struct Acc {
    pub v: [[i32; 16]; ACC_VECS],
}

impl Acc {
    pub fn zero() -> Box<Acc> {
        unsafe {
            let mut b = Box::<Acc>::new_uninit();
            core::ptr::write_bytes(b.as_mut_ptr() as *mut u8, 0, core::mem::size_of::<Acc>());
            b.assume_init()
        }
    }
    pub fn clear(&mut self) {
        unsafe {
            core::ptr::write_bytes(
                self.v.as_mut_ptr() as *mut u8,
                0,
                core::mem::size_of::<Acc>(),
            );
        }
    }
}

/// Accumulator vector and lane group (0 = lanes 0..8, 1 = lanes 8..16) holding slot `s`.
pub const fn slot_lane(s: usize) -> (usize, usize) {
    let (bl, r) = (s / 27, s % 27);
    if r == 26 {
        (ACC_PER_BLK * bl + 13, 0)
    } else {
        (ACC_PER_BLK * bl + r / 2, r % 2)
    }
}

/// Two 16-lane product vectors folded to 8 lanes each and packed into one vector: lanes 0..8
/// carry `t0`, lanes 8..16 carry `t1` (two `vshufi64x2` and one `vpaddd`).
#[inline(always)]
unsafe fn pack2(t0: __m512i, t1: __m512i) -> __m512i {
    _mm512_add_epi32(
        _mm512_shuffle_i64x2::<0x44>(t0, t1),
        _mm512_shuffle_i64x2::<0xEE>(t0, t1),
    )
}

/// The two slots `w0, w1` times `a0, a1`, folded to 8 lanes each and packed into one vector.
#[inline(always)]
unsafe fn fold_pair(w0: *const i16, w1: *const i16, a0: *const i16, a1: *const i16) -> __m512i {
    let t0 = _mm512_madd_epi16(
        _mm512_load_si512(w0 as *const __m512i),
        _mm512_load_si512(a0 as *const __m512i),
    );
    let t1 = _mm512_madd_epi16(
        _mm512_load_si512(w1 as *const __m512i),
        _mm512_load_si512(a1 as *const __m512i),
    );
    pack2(t0, t1)
}

/// One 27-slot block of one batch into 14 accumulator vectors.
///
/// # Safety
/// `w`, `a` and `acc` must be 64-byte aligned; `w` and `a` must cover 27 vectors, `acc`
/// [`ACC_PER_BLK`]. AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac27(w: *const i16, a: *const i16, acc: *mut i32) {
    for p in 0..13 {
        let s = _mm512_load_si512(acc.add(16 * p) as *const __m512i);
        let d = fold_pair(w.add(64 * p), w.add(64 * p + 32), a.add(64 * p), a.add(64 * p + 32));
        _mm512_store_si512(acc.add(16 * p) as *mut __m512i, _mm512_add_epi32(s, d));
    }
    let s = _mm512_load_si512(acc.add(16 * 13) as *const __m512i);
    let d = fold_pair(w.add(32 * 26), w.add(32 * 26), a.add(32 * 26), a.add(32 * 26));
    _mm512_store_si512(acc.add(16 * 13) as *mut __m512i, _mm512_add_epi32(s, d));
}

/// One whole batch: the same 24 blocks, over a materialised transform.
///
/// # Safety
/// See [`mac27`]; `w` and `a` must cover 648 vectors and `acc` [`ACC_VECS`]. AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac_batch(w: *const i16, a: *const i16, acc: *mut i32) {
    for bl in 0..24 {
        mac27(
            w.add(32 * 27 * bl),
            a.add(32 * 27 * bl),
            acc.add(16 * ACC_PER_BLK * bl),
        );
    }
}

/// The periodic fold-back over the whole accumulator.
///
/// # Safety
/// `acc` must be 64-byte aligned and cover [`ACC_VECS`] vectors. AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn reduce_acc<const Q: u16>(acc: *mut i32) {
    for j in 0..ACC_VECS {
        let s = _mm512_load_si512(acc.add(16 * j) as *const __m512i);
        _mm512_store_si512(acc.add(16 * j) as *mut __m512i, reduce_vec::<Q>(s));
    }
}

// =============================================================================================
// the horizontal finish
// =============================================================================================

const HS_IDX: [[i32; 16]; 6] = [
    [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27],
    [4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31],
    [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29],
    [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31],
    [0, 4, 2, 6, 8, 12, 10, 14, 16, 20, 18, 22, 24, 28, 26, 30],
    [1, 5, 3, 7, 9, 13, 11, 15, 17, 21, 19, 23, 25, 29, 27, 31],
];

/// One stage of [`hsum8`]: halve the width of every partial sum in `a` and `b` at once.
#[inline(always)]
unsafe fn hs<const S: usize>(a: __m512i, b: __m512i) -> __m512i {
    let lo = _mm512_loadu_si512(HS_IDX[2 * S].as_ptr() as *const __m512i);
    let hi = _mm512_loadu_si512(HS_IDX[2 * S + 1].as_ptr() as *const __m512i);
    _mm512_add_epi32(
        _mm512_permutex2var_epi32(a, lo, b),
        _mm512_permutex2var_epi32(a, hi, b),
    )
}

/// Does one fold-back leave eight lanes summable inside i32? `reduce_vec` caps a lane at
/// `2^15 (1 + R)`, so this asks `2^18 (1 + R) <= i32::MAX`. 3889, 9721 and the quad primes
/// clear it.
pub const fn hsum_double(q: u16) -> bool {
    8 * acc_after_reduce(q) > i32::MAX as i64
}

/// Bound on a lane going into the eight-lane sum of [`hsum8`].
pub const fn acc_after_hsum(q: u16) -> i64 {
    let a = acc_after_reduce(q);
    if hsum_double(q) {
        32768 + (a / 65536 + 1) * r16(q) as i64
    } else {
        a
    }
}

const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!(8 * acc_after_hsum(QS[i]) <= i32::MAX as i64);
        i += 1;
    }
};
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(8 * acc_after_hsum(QS_QUAD[i]) <= i32::MAX as i64);
        i += 1;
    }
};

/// The 16 lane-group sums of 8 consecutive accumulator vectors: lane `2k` is the sum of lanes
/// 0..8 of vector `k`, lane `2k + 1` the sum of its lanes 8..16, each folded back first.
///
/// # Safety
/// `p` must be 64-byte aligned and cover 8 vectors. AVX-512 F/BW.
#[inline(always)]
unsafe fn hsum8<const Q: u16>(p: *const i32) -> __m512i {
    let v = |k: usize| {
        let x = reduce_vec::<Q>(_mm512_load_si512(p.add(16 * k) as *const __m512i));
        if hsum_double(Q) {
            reduce_vec::<Q>(x)
        } else {
            x
        }
    };
    let r0 = hs::<0>(v(0), v(1));
    let r1 = hs::<0>(v(2), v(3));
    let r2 = hs::<0>(v(4), v(5));
    let r3 = hs::<0>(v(6), v(7));
    hs::<2>(hs::<1>(r0, r1), hs::<1>(r2, r3))
}

/// `x mod q` in [0, q) for eight i32 lanes held as doubles.
#[inline(always)]
unsafe fn mod_q_pd<const Q: u16>(v: __m512d) -> __m256i {
    let q = _mm512_set1_pd(Q as f64);
    let t = _mm512_roundscale_pd::<0x09>(_mm512_mul_pd(v, _mm512_set1_pd(1.0 / Q as f64)));
    let r = _mm512_fnmadd_pd(t, q, v);
    let r = _mm512_mask_add_pd(r, _mm512_cmp_pd_mask::<_CMP_LT_OQ>(r, _mm512_setzero_pd()), r, q);
    let r = _mm512_mask_sub_pd(r, _mm512_cmp_pd_mask::<_CMP_NLT_UQ>(r, q), r, q);
    _mm512_cvttpd_epi32(r)
}

/// `x mod q` in [0, q) for 16 i32 lanes, `|x| < 2^31`.
///
/// # Safety
/// AVX-512 F.
#[inline(always)]
pub unsafe fn mod_q<const Q: u16>(x: __m512i) -> __m512i {
    let lo = mod_q_pd::<Q>(_mm512_cvtepi32_pd(_mm512_castsi512_si256(x)));
    let hi = mod_q_pd::<Q>(_mm512_cvtepi32_pd(_mm512_extracti64x4_epi64::<1>(x)));
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// Sum of the 8 lanes of every slot, reduced to [0, q).
///
/// # Safety
/// AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn finish_vec<const Q: u16>(acc: &Acc) -> [u32; N] {
    const HS_SHIFT4: [i32; 16] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0, 0];
    let shift = _mm512_loadu_si512(HS_SHIFT4.as_ptr() as *const __m512i);
    let mut y = [0u32; N];
    let p = acc.v.as_ptr() as *const i32;
    for bl in 0..24 {
        let base = 16 * ACC_PER_BLK * bl;
        let lo = mod_q::<Q>(hsum8::<Q>(p.add(base)));
        let hi = mod_q::<Q>(hsum8::<Q>(p.add(base + 16 * 6)));
        let o = y.as_mut_ptr().add(27 * bl) as *mut i32;
        _mm512_storeu_si512(o as *mut __m512i, lo);
        _mm512_mask_storeu_epi32(o.add(16), 0x07ff, _mm512_permutexvar_epi32(shift, hi));
    }
    y
}

/// The split-limb finish (safe wrapper of [`finish_vec`]).
pub fn finish<const Q: u16>(acc: &Acc) -> [u32; N] {
    unsafe { finish_vec::<Q>(acc) }
}

// =============================================================================================
// quadratic-slot limbs
// =============================================================================================

/// Bound on one lane of the quadratic kernel's output.
pub const fn w_bound_quad(q: u16) -> i64 {
    ntt_quad::output_bound(q) as i64
}

/// Can the Karatsuba sum `a_0 + a_1` of two output rows live in an i16 lane? (2917: yes.)
pub const fn karatsuba(q: u16) -> bool {
    2 * w_bound_quad(q) <= 32767
}
const _: () = assert!(!karatsuba(2917) && karatsuba(4861) && !karatsuba(12637));

/// What one batch adds to a lane of the `P_0 | P_1` accumulator: four products of `|W| |A|`.
pub const fn acc_per_batch_quad01(q: u16) -> i64 {
    4 * w_bound_quad(q) * a_bound(q)
}

/// What one batch adds to a lane of the `P_2` accumulator: four products of `2|W|` by
/// `|b_0 + b_1| <= q - 1` with Karatsuba, eight of `|W| |A|` without.
pub const fn acc_per_batch_quad2(q: u16) -> i64 {
    if karatsuba(q) {
        4 * (2 * w_bound_quad(q)) * (2 * a_bound(q))
    } else {
        8 * w_bound_quad(q) * a_bound(q)
    }
}

/// Batches between two fold-backs of the `P_0 | P_1` accumulator (16 / 8 / 2).
pub const fn red_period_quad01(q: u16) -> usize {
    period_for(q, acc_per_batch_quad01(q))
}
/// Batches between two fold-backs of the `P_2` accumulator (8 / 2 / 1).
pub const fn red_period_quad2(q: u16) -> usize {
    period_for(q, acc_per_batch_quad2(q))
}

const fn fits_quad(q: u16) -> bool {
    acc_after_reduce(q) + (red_period_quad01(q) as i64) * acc_per_batch_quad01(q) <= i32::MAX as i64
        && acc_after_reduce(q) + (red_period_quad2(q) as i64) * acc_per_batch_quad2(q)
            <= i32::MAX as i64
}
const _: () = assert!(fits_quad(2917) && fits_quad(4861) && fits_quad(12637));

/// Accumulator vectors per 18-row block: 9 for `P_0 | P_1`, 5 for `P_2`.
pub const QACC01_PER_BLK: usize = 9;
pub const QACC2_PER_BLK: usize = 5;
/// Blocks the quadratic kernel hands out (36 x 18 rows = 648).
pub const QBLOCKS: usize = 36;

/// Vectors of zero padding after each quadratic accumulator: neither 324 nor 180 is a multiple
/// of the eight vectors `hsum8` folds at a time, and the last group of each reads past the end.
pub const QPAD: usize = 4;

#[repr(C, align(64))]
pub struct QuadAcc {
    /// `p01[9 blk + j]`: lanes 0..8 are `P_0` of leaf `9 blk + j`, lanes 8..16 its `P_1`.
    pub p01: [[i32; 16]; QBLOCKS * QACC01_PER_BLK + QPAD],
    /// `p2[5 blk + j/2]`: lanes `8 (j % 2) ..` are `P_2` of leaf `9 blk + j` (leaf 8 in lanes
    /// 0..8, its duplicate in 8..16).
    pub p2: [[i32; 16]; QBLOCKS * QACC2_PER_BLK + QPAD],
}

impl QuadAcc {
    pub fn zero() -> Box<QuadAcc> {
        unsafe {
            let mut b = Box::<QuadAcc>::new_uninit();
            core::ptr::write_bytes(
                b.as_mut_ptr() as *mut u8,
                0,
                core::mem::size_of::<QuadAcc>(),
            );
            b.assume_init()
        }
    }
    pub fn clear(&mut self) {
        unsafe {
            core::ptr::write_bytes(
                self.p01.as_mut_ptr() as *mut u8,
                0,
                core::mem::size_of::<QuadAcc>(),
            );
        }
    }
}

/// One leaf: the packed `P_0 | P_1` contribution and the `P_2` one, off four loads.
///
/// The four vectors are loaded once and used by both products — with the accumulator stores in
/// between, LLVM has to assume they alias and reloads them, which measures 30% of the sink.
#[inline(always)]
unsafe fn leaf<const Q: u16>(w: *const i16, a: *const i16) -> (__m512i, __m512i) {
    let w0 = _mm512_load_si512(w as *const __m512i);
    let w1 = _mm512_load_si512(w.add(32) as *const __m512i);
    let a0 = _mm512_load_si512(a as *const __m512i);
    let a1 = _mm512_load_si512(a.add(32) as *const __m512i);
    let p01 = pack2(_mm512_madd_epi16(w0, a0), _mm512_madd_epi16(w1, a1));
    let p2 = if karatsuba(Q) {
        _mm512_madd_epi16(_mm512_add_epi16(w0, w1), _mm512_add_epi16(a0, a1))
    } else {
        _mm512_add_epi32(_mm512_madd_epi16(w0, a1), _mm512_madd_epi16(w1, a0))
    };
    (p01, p2)
}

/// One 18-row block (9 leaves) of one batch into its 14 accumulator vectors.
///
/// # Safety
/// `w`, `a`, `acc01` and `acc2` must be 64-byte aligned; `w` and `a` must cover 18 vectors,
/// `acc01` [`QACC01_PER_BLK`] and `acc2` [`QACC2_PER_BLK`]. AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac_quad18<const Q: u16>(w: *const i16, a: *const i16, acc01: *mut i32, acc2: *mut i32) {
    for m in 0..4 {
        let (j0, j1) = (2 * m, 2 * m + 1);
        let (d0, t0) = leaf::<Q>(w.add(64 * j0), a.add(64 * j0));
        let (d1, t1) = leaf::<Q>(w.add(64 * j1), a.add(64 * j1));
        let s0 = _mm512_load_si512(acc01.add(16 * j0) as *const __m512i);
        let s1 = _mm512_load_si512(acc01.add(16 * j1) as *const __m512i);
        let s2 = _mm512_load_si512(acc2.add(16 * m) as *const __m512i);
        _mm512_store_si512(acc01.add(16 * j0) as *mut __m512i, _mm512_add_epi32(s0, d0));
        _mm512_store_si512(acc01.add(16 * j1) as *mut __m512i, _mm512_add_epi32(s1, d1));
        _mm512_store_si512(
            acc2.add(16 * m) as *mut __m512i,
            _mm512_add_epi32(s2, pack2(t0, t1)),
        );
    }
    let (d, t) = leaf::<Q>(w.add(64 * 8), a.add(64 * 8));
    let s0 = _mm512_load_si512(acc01.add(16 * 8) as *const __m512i);
    let s2 = _mm512_load_si512(acc2.add(16 * 4) as *const __m512i);
    _mm512_store_si512(acc01.add(16 * 8) as *mut __m512i, _mm512_add_epi32(s0, d));
    _mm512_store_si512(
        acc2.add(16 * 4) as *mut __m512i,
        _mm512_add_epi32(s2, pack2(t, t)),
    );
}

/// One whole batch through [`mac_quad18`], over a materialised transform.
///
/// # Safety
/// See [`mac_quad18`]; `w` and `a` must cover 648 vectors and the accumulators a whole
/// [`QuadAcc`]. AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn mac_quad_batch<const Q: u16>(
    w: *const i16,
    a: *const i16,
    acc01: *mut i32,
    acc2: *mut i32,
) {
    for bl in 0..QBLOCKS {
        mac_quad18::<Q>(
            w.add(32 * 18 * bl),
            a.add(32 * 18 * bl),
            acc01.add(16 * QACC01_PER_BLK * bl),
            acc2.add(16 * QACC2_PER_BLK * bl),
        );
    }
}

/// The fold-back over both quadratic accumulators.
///
/// # Safety
/// AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn reduce_quad_acc<const Q: u16>(acc: &mut QuadAcc) {
    reduce_quad_part::<Q>(acc.p01.as_mut_ptr() as *mut i32, QBLOCKS * QACC01_PER_BLK);
    reduce_quad_part::<Q>(acc.p2.as_mut_ptr() as *mut i32, QBLOCKS * QACC2_PER_BLK);
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn reduce_quad_part<const Q: u16>(acc: *mut i32, vecs: usize) {
    for j in 0..vecs {
        let p = acc.add(16 * j);
        _mm512_store_si512(
            p as *mut __m512i,
            reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i)),
        );
    }
}

/// The three sums per leaf combined into the 648 rows of the commitment, reduced to `[0, q)`:
/// `y[2j] = P_0 + c_j P_1`, `y[2j+1] = P_2` (minus `P_0 + P_1` when the Karatsuba product was
/// accumulated).
///
/// # Safety
/// AVX-512 F/BW.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn finish_quad_vec<const Q: u16>(acc: &QuadAcc) -> [u32; N] {
    const G01: usize = (QBLOCKS * QACC01_PER_BLK).div_ceil(8);
    const G2: usize = (QBLOCKS * QACC2_PER_BLK).div_ceil(8);
    let mut s01 = [0i32; 16 * G01];
    let mut s2 = [0i32; 16 * G2];
    let p = acc.p01.as_ptr() as *const i32;
    for g in 0..G01 {
        let v = mod_q::<Q>(hsum8::<Q>(p.add(128 * g)));
        _mm512_storeu_si512(s01.as_mut_ptr().add(16 * g) as *mut __m512i, v);
    }
    let p = acc.p2.as_ptr() as *const i32;
    for g in 0..G2 {
        let v = mod_q::<Q>(hsum8::<Q>(p.add(128 * g)));
        _mm512_storeu_si512(s2.as_mut_ptr().add(16 * g) as *mut __m512i, v);
    }

    let q = Q as i32;
    let mut y = [0u32; N];
    for blk in 0..QBLOCKS {
        for j in 0..QACC01_PER_BLK {
            let leaf = QACC01_PER_BLK * blk + j;
            let (p0, p1) = (s01[2 * leaf], s01[2 * leaf + 1]);
            let p2 = s2[2 * QACC2_PER_BLK * blk + j];
            let c = ParamsQ::<Q>::LEAF_C[leaf] as i32;
            y[2 * leaf] = ((p0 + c * p1) % q) as u32;
            y[2 * leaf + 1] = if karatsuba(Q) {
                (p2 - p0 - p1).rem_euclid(q) as u32
            } else {
                p2 as u32
            };
        }
    }
    y
}

/// The quad-limb finish (safe wrapper of [`finish_quad_vec`]).
pub fn finish_quad<const Q: u16>(acc: &QuadAcc) -> [u32; N] {
    unsafe { finish_quad_vec::<Q>(acc) }
}

// =============================================================================================
// the transform extraction (base limb, for the fold)
// =============================================================================================

/// One slot row of the materialised transform, lazily reduced i16, converted to fully reduced
/// `u32` in `[0, q)` and **scattered** into the element-major layout the fold consumes:
/// `dst[p][j] = mod_q(src[j][p])` for the 32 elements of one batch.
///
/// # Safety
/// AVX-512 F/BW. `src` is 64-byte aligned and covers 648 vectors; `dst` covers 32 * `N` u32.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn store_transform<const Q: u16>(src: *const i16, dst: *mut u32) {
    // lane p of the low index vector: element p; lane p of the high one: element 16 + p.
    let mut idx = [0i32; 16];
    let mut idx2 = [0i32; 16];
    let mut p = 0;
    while p < 16 {
        idx[p] = (p * N) as i32;
        idx2[p] = ((16 + p) * N) as i32;
        p += 1;
    }
    let iv0 = _mm512_loadu_si512(idx.as_ptr() as *const __m512i);
    let iv1 = _mm512_loadu_si512(idx2.as_ptr() as *const __m512i);
    for j in 0..N {
        let x = _mm512_load_si512(src.add(32 * j) as *const __m512i);
        let lo = mod_q::<Q>(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<0>(x)));
        let hi = mod_q::<Q>(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(x)));
        let j0 = _mm512_add_epi32(iv0, _mm512_set1_epi32(j as i32));
        let j1 = _mm512_add_epi32(iv1, _mm512_set1_epi32(j as i32));
        _mm512_i32scatter_epi32::<4>(dst as *mut i32, j0, lo);
        _mm512_i32scatter_epi32::<4>(dst as *mut i32, j1, hi);
    }
}

/// Copy one finished batch into the kept vertical transform with non-temporal stores
/// (upstream's `MacKeep` write-out): the 41.5 KB per batch leaves no cache footprint — the
/// write-combining buffers absorb it while the next batch's transform runs. The caller fences
/// once after the whole commitment ([`sfence`]).
///
/// # Safety
/// AVX-512 F; `src`/`dst` 64-byte aligned, valid `Batch32`.
#[target_feature(enable = "avx512f")]
pub unsafe fn stream_copy_batch32(dst: *mut Batch32, src: *const Batch32) {
    let s = (*src).v.as_ptr() as *const __m512i;
    let d = (*dst).v.as_mut_ptr() as *mut __m512i;
    for j in 0..N {
        _mm512_stream_si512(d.add(j) as *mut __m512i, _mm512_load_si512(s.add(j) as *const __m512i));
    }
}

/// The fence the non-temporal kept-transform stores must be followed by before anyone reads
/// them back (once per commitment, not per batch).
///
/// # Safety
/// none (x86-64).
#[inline(always)]
pub fn sfence() {
    #[cfg(target_arch = "x86_64")]
    // Safety: sfence is safe on any x86-64 CPU.
    unsafe {
        _mm_sfence();
    }
}

// =============================================================================================
// per-batch entry points (runtime q dispatch)
// =============================================================================================

/// One split limb's batch: transform materialised into `out`, multiplied into `acc`, fold-back
/// on the limb's period.
///
/// # Safety
/// AVX-512 PCS feature set; `idx`/`a`/`out`/`acc` per the kernels' contracts.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
pub unsafe fn split_batch<const Q: u16>(
    idx: &BinaryIndex32,
    a: &Batch32,
    out: &mut Batch32,
    acc: &mut Acc,
    done: usize,
) {
    ntt_small::ntt_bin_batch32::<Q>(idx, out);
    mac_batch(
        out.v.as_ptr() as *const i16,
        a.v.as_ptr() as *const i16,
        acc.v.as_mut_ptr() as *mut i32,
    );
    if done.is_multiple_of(red_period(Q)) {
        reduce_acc::<Q>(acc.v.as_mut_ptr() as *mut i32);
    }
}

/// One *large*-prime split limb's batch (`17497`/`19441`): the `ntt_large` binary kernel, the
/// same MAC, fold-back on the large primes' own (narrower) period.
///
/// # Safety
/// AVX-512 PCS feature set; `idx`/`a`/`out`/`acc` per the kernels' contracts.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
pub unsafe fn split_large_batch<const Q: u16>(
    idx: &BinaryIndex32,
    a: &Batch32,
    out: &mut Batch32,
    acc: &mut Acc,
    done: usize,
) {
    crate::simd::ntt_large::ntt_bin_batch32::<Q>(idx, out);
    mac_batch(
        out.v.as_ptr() as *const i16,
        a.v.as_ptr() as *const i16,
        acc.v.as_mut_ptr() as *mut i32,
    );
    if done.is_multiple_of(red_period(Q)) {
        reduce_acc::<Q>(acc.v.as_mut_ptr() as *mut i32);
    }
}

/// One quadratic limb's batch: transform materialised into `out`, multiplied into the two
/// accumulators, fold-backs on their own periods.
///
/// # Safety
/// AVX-512 PCS feature set; `idx`/`a`/`out`/`acc` per the kernels' contracts.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
pub unsafe fn quad_batch<const Q: u16>(
    idx: &BinaryIndex32,
    a: &Batch32,
    out: &mut Batch32,
    acc: &mut QuadAcc,
    done: usize,
) {
    ntt_quad::ntt_quad_bin_batch32::<Q>(idx, out);
    mac_quad_batch::<Q>(
        out.v.as_ptr() as *const i16,
        a.v.as_ptr() as *const i16,
        acc.p01.as_mut_ptr() as *mut i32,
        acc.p2.as_mut_ptr() as *mut i32,
    );
    if done.is_multiple_of(red_period_quad01(Q)) {
        reduce_quad_part::<Q>(acc.p01.as_mut_ptr() as *mut i32, QBLOCKS * QACC01_PER_BLK);
    }
    if done.is_multiple_of(red_period_quad2(Q)) {
        reduce_quad_part::<Q>(acc.p2.as_mut_ptr() as *mut i32, QBLOCKS * QACC2_PER_BLK);
    }
}


// =============================================================================================
// the fold's u64 MAC (non-binary rows: the amortised witness and the verifier's A v)
// =============================================================================================

/// One row of the fold's length-`r` inner product: `acc += ch * w` over 648 slots, products and
/// sums in u64 lanes (values below `q < 2^15`, so `r * q^2` overflows nothing the suites use;
/// the caller reduces once per row). Eight slots per vector.
///
/// # Safety
/// AVX-512 F/DQ; `ch`, `w` and `acc` must cover `N` u32/u64 respectively, `acc` 64-byte
/// aligned.
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn mac_row_u64(ch: *const u32, w: *const u32, acc: *mut u64) {
    for j in 0..(N / 8) {
        let c8 = _mm256_loadu_si256(ch.add(8 * j) as *const __m256i);
        let w8 = _mm256_loadu_si256(w.add(8 * j) as *const __m256i);
        let c64 = _mm512_cvtepu32_epi64(c8);
        let w64 = _mm512_cvtepu32_epi64(w8);
        let p = _mm512_mullo_epi64(c64, w64);
        // unaligned: `acc` is a caller stack array, aligned to u64 only
        let a = _mm512_loadu_si512(acc.add(8 * j) as *const __m512i);
        _mm512_storeu_si512(acc.add(8 * j) as *mut __m512i, _mm512_add_epi64(a, p));
    }
}

/// The portable form of [`mac_row_u64`]: identical results, scalar u64 arithmetic.
pub fn mac_row_u64_soft(ch: &[u32], w: &[u32], acc: &mut [u64]) {
    for u in 0..N {
        acc[u] += ch[u] as u64 * w[u] as u64;
    }
}

/// `acc += ch * w` over 648 slots, AVX-512 when available.
pub fn mac_row(ch: &[u32], w: &[u32], acc: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::hw::avx512_pcs() {
            // Safety: the feature gate was just checked; slices cover N as required.
            unsafe {
                mac_row_u64(ch.as_ptr(), w.as_ptr(), acc.as_mut_ptr());
            }
            return;
        }
    }
    mac_row_u64_soft(ch, w, acc);
}
