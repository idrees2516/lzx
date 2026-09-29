//! The folding step `v = sum_j c_j W_j`, entirely in the 648-slot NTT domain of `R_648`, plus
//! the verifier's `A v` recomputation and the commitment fold `sum_j c_j C_j`.
//! Port of `labinius` `fold.rs` — both the scalar reference path and the vertical AVX-512 path.
//!
//! A challenge `c_j` (an element of the subring `R_162`) enters `R_648` as `c_j(-X^4)`
//! (coefficient of `X^{4m}` is `(-1)^m c_{j,m}`, everything else zero), so the fold is a single
//! length-`r` inner product per slot with no ring multiplication anywhere. For a quadratic-slot
//! base limb, `c(-X^4)` is the scalar `c(-theta^v)` in leaf `X^2 - psi'^v`, written into both
//! rows of the leaf.
//!
//! # The vertical path (upstream parity)
//!
//! When the commitment ran the AVX-512 backend, `AuxData` keeps the witness transform in the
//! vertical `Batch32` layout the kernels produced — no `store_transform` scatter — and this
//! module folds it in place: the challenges are embedded vertically and transformed with the
//! generic-input kernels (`gen_small`/`gen_large`/`gen_quad`), packed into dword pairs so each
//! chunk pair costs one `vpunpck` + one `vpmaddwd` per slot vector against a `G`-sized
//! i32 accumulator that stays in L2, folded back exactly every [`fold_period`] chunks; the
//! drained accumulator is inverted by the same generic kernels. The verifier's `A v` reuses the
//! commitment's own packed accumulator (`vpmaddwd`, fold-back every [`av_period`] batches) against
//! the vertical `A`. The commitment fold `sum_j c_j C_j` runs through the slot tables of
//! [`crate::simd::slots`].

use crate::challenge::ShortChallenge;
use crate::key::AuxData;
use crate::params::{quadratic_slots, N, QUAD_CLASS_SLOT, QUAD_SLOTS, QS, QS_LARGE, QS_QUAD};
use crate::ring::{components_of, PowerOfThreeRing, N162, SLOT_648};
use crate::scalar::Coeffs;
use crate::simd::commit as mac;
use crate::simd::ntt_large;
use crate::simd::slots;
use crate::simd::{gen_large, gen_quad, gen_small, Batch32};
use core::arch::x86_64::*;
use std::sync::OnceLock;

// =============================================================================================
// the scalar reference path (kept verbatim: bd, tests, and the no-SIMD fallback)
// =============================================================================================

/// Embed `c` as `c(-X^4)` into `R_648` coefficients (degree-162 multiples of 4 only).
fn embed(c: &ShortChallenge) -> [i64; N] {
    let mut out = [0i64; N];
    for i in 0..c.weight {
        let m = c.positions[i] as usize;
        let co = 1 - 2 * ((c.signs >> i) & 1) as i64;
        out[4 * m] = if m.is_multiple_of(2) { co } else { -co };
    }
    out
}

/// `NTT(c(-X^4))` for a splitting prime: the 648 tree-order slots, centered i16.
fn challenge_ntt<const Q: u16>(c: &ShortChallenge) -> [i16; N] {
    let e = embed(c);
    let mut coeffs = [0u32; N];
    for (i, &x) in e.iter().enumerate() {
        coeffs[i] = (x.rem_euclid(Q as i64) as u64 % Q as u64) as u32;
    }
    let t = crate::scalar::ntt::<Q>(&coeffs);
    let half = (Q as i32 - 1) / 2;
    let mut out = [0i16; N];
    for u in 0..N {
        out[u] = if t[u] as i32 > half {
            t[u] as i32 - Q as i32
        } else {
            t[u] as i32
        } as i16;
    }
    out
}

/// The quadratic-slot analogue: leaf image of `c(-X^4)` is the scalar `c(-theta^v)` in both rows.
fn challenge_ntt_quad<const Q: u16>(c: &ShortChallenge) -> [i16; N] {
    let e = embed(c);
    let mut coeffs = [0u32; N];
    for (i, &x) in e.iter().enumerate() {
        coeffs[i] = (x.rem_euclid(Q as i64) as u64 % Q as u64) as u32;
    }
    let t = crate::scalar::ntt_quad::<Q>(&coeffs);
    let half = (Q as i32 - 1) / 2;
    let mut out = [0i16; N];
    for u in 0..N {
        out[u] = if t[u] as i32 > half {
            t[u] as i32 - Q as i32
        } else {
            t[u] as i32
        } as i16;
    }
    // duplicate the leaf scalars into both rows: the transform put the scalar in row 2j and 0 in
    // row 2j+1 (X^4 = psi'^{2u} is a constant per leaf); verify and copy.
    let mut scalars = [0i16; QUAD_SLOTS];
    for j in 0..QUAD_SLOTS {
        debug_assert_eq!(out[2 * j + 1], 0, "a challenge leaf is not a scalar");
        scalars[j] = out[2 * j];
    }
    for j in 0..QUAD_SLOTS {
        out[2 * j + 1] = scalars[j];
    }
    out
}

/// The transformed challenge for any limb prime (scalar path).
pub fn challenge_slots(q: u16, c: &ShortChallenge) -> [i16; N] {
    match q {
        3889 => challenge_ntt::<3889>(c),
        9721 => challenge_ntt::<9721>(c),
        17497 => challenge_ntt::<17497>(c),
        19441 => challenge_ntt::<19441>(c),
        2917 => challenge_ntt_quad::<2917>(c),
        4861 => challenge_ntt_quad::<4861>(c),
        12637 => challenge_ntt_quad::<12637>(c),
        _ => unreachable!("unknown limb prime {q}"),
    }
}

/// `v = sum_j c_j W_j` in coefficient form modulo the base prime, centered — the amortised
/// witness. Slot-wise over the kept witness transform, then the base limb's inverse transform.
/// A coefficient is a sum of `r * w` signed 0/1 terms (standard deviation `sqrt(r w / 2)`), two
/// orders below `q/2`.
///
/// The scalar path's challenges are converted to `[0, q)` once and the accumulation runs through
/// the AVX-512 u64-lane MAC when the CPU has it; the vertical path (SIMD commitment) folds the
/// kept `Batch32` transform directly.
pub fn fold_witness(aux: &AuxData, challenges: &[ShortChallenge], q: u16) -> Vec<[i16; N]> {
    if !aux.vertical.is_empty() && crate::simd::available() {
        return fold_witness_simd(aux, challenges, q);
    }
    fold_witness_scalar(aux, challenges, q)
}

/// The scalar reference fold over the element-major kept transform.
fn fold_witness_scalar(aux: &AuxData, challenges: &[ShortChallenge], q: u16) -> Vec<[i16; N]> {
    assert_eq!(challenges.len(), aux.chunks, "one challenge per chunk");
    let nr = aux.batches.len() / aux.chunks;
    let ch: Vec<[u32; N]> = challenges
        .iter()
        .map(|c| {
            let t = challenge_slots(q, c);
            let mut u = [0u32; N];
            for (x, &v) in u.iter_mut().zip(t.iter()) {
                *x = v.rem_euclid(q as i16) as u32;
            }
            u
        })
        .collect();
    let q64 = q as u64;
    let mut out = Vec::with_capacity(nr);
    let mut acc = [0u64; N];
    for i in 0..nr {
        acc.fill(0);
        for c in 0..aux.chunks {
            crate::simd::commit::mac_row(&ch[c], &aux.batches[c * nr + i], &mut acc);
        }
        let mut coeffs = [0u32; N];
        for (x, &a) in coeffs.iter_mut().zip(acc.iter()) {
            *x = (a % q64) as u32;
        }
        let v = if quadratic_slots(q) {
            crate::ring::intt_quad_of(q, &coeffs)
        } else {
            crate::ring::intt_of(q, &coeffs)
        };
        let half = (q as i32 - 1) / 2;
        let mut e = [0i16; N];
        for u in 0..N {
            e[u] = if v[u] as i32 > half {
                v[u] as i32 - q as i32
            } else {
                v[u] as i32
            } as i16;
        }
        out.push(e);
    }
    out
}

/// `A v` for one limb from centered coefficient batches: forward-transform `v`, pointwise inner
/// product against the key rows. Returns the 648-row raw commitment form (scalar path).
pub fn a_times_v(q: u16, a: &[[i16; N]], v: &[[i16; N]]) -> Coeffs {
    assert_eq!(a.len(), v.len());
    let q64 = q as u64;
    let n = v.len();
    let mut a_u32: Vec<[u32; N]> = Vec::with_capacity(n);
    for row in a {
        let mut r = [0u32; N];
        for (x, &s) in r.iter_mut().zip(row.iter()) {
            *x = s.rem_euclid(q as i16) as u32;
        }
        a_u32.push(r);
    }
    let mut acc = [0u64; N];
    for i in 0..n {
        let mut coeffs = [0u32; N];
        for (x, &s) in coeffs.iter_mut().zip(v[i].iter()) {
            *x = s.rem_euclid(q as i16) as u32;
        }
        let t = crate::ring::ntt_of(q, &coeffs);
        crate::simd::commit::mac_row(&a_u32[i], &t, &mut acc);
    }
    let mut out = [0u32; N];
    for u in 0..N {
        out[u] = (acc[u] % q64) as u32;
    }
    out
}

// =============================================================================================
// the vertical path: bounds
// =============================================================================================

/// Bound on one lane of the kept transform: the base kernel's declared output bound, whichever
/// tree it runs ([`mac::w_bound`] for a splitting prime, [`mac::w_bound_quad`] for a quadratic
/// one).
pub const fn witness_bound(q: u16) -> i64 {
    if quadratic_slots(q) {
        mac::w_bound_quad(q)
    } else {
        mac::w_bound(q)
    }
}

/// What one chunk adds to one accumulator lane: `|W| <= witness_bound(q)` times
/// `|c| <= (q-1)/2` — one product, unlike the commitment's four, because a lane of this
/// accumulator carries one ring element rather than four.
pub const fn fold_per_chunk(q: u16) -> i64 {
    witness_bound(q) * mac::a_bound(q)
}

/// Chunks accumulated between two fold-backs of a base limb's accumulator. The chunks are
/// accumulated in pairs, so the period must be even.
pub const fn fold_period(q: u16) -> usize {
    mac::period_for(q, fold_per_chunk(q))
}

const fn fold_fits(q: u16, p: usize) -> bool {
    mac::acc_after_reduce(q) + (p as i64) * fold_per_chunk(q) <= i32::MAX as i64
}
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(fold_fits(QS_QUAD[i], fold_period(QS_QUAD[i])) && fold_period(QS_QUAD[i]) % 2 == 0);
        i += 1;
    }
    let mut i = 0;
    while i < 2 {
        assert!(fold_fits(QS[i], fold_period(QS[i])) && fold_period(QS[i]) % 2 == 0);
        assert!(fold_fits(QS_LARGE[i], fold_period(QS_LARGE[i])) && fold_period(QS_LARGE[i]) % 2 == 0);
        i += 1;
    }
};

/// Batches of `A v` between two fold-backs of a splitting limb's accumulator: both operands are
/// centered, so a lane grows by `4 ((q-1)/2)^2` per batch.
pub const fn av_period(q: u16) -> usize {
    mac::period_for(q, 4 * mac::a_bound(q) * mac::a_bound(q))
}

const fn av_fits(q: u16) -> bool {
    mac::acc_after_reduce(q) + (av_period(q) as i64) * 4 * mac::a_bound(q) * mac::a_bound(q)
        <= i32::MAX as i64
}
const _: () = {
    let mut i = 0;
    while i < 2 {
        assert!(av_fits(QS[i]) && av_period(QS[i]) >= 1);
        assert!(av_fits(QS_LARGE[i]) && av_period(QS_LARGE[i]) >= 1);
        i += 1;
    }
};

/// Batches of `A v` between two fold-backs of a quadratic limb's accumulators.
pub const fn av_period_quad(q: u16) -> usize {
    let per = if mac::karatsuba(q) {
        16 * mac::a_bound(q) * mac::a_bound(q)
    } else {
        8 * mac::a_bound(q) * mac::a_bound(q)
    };
    mac::period_for(q, per)
}
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(av_period_quad(QS_QUAD[i]) >= 1);
        i += 1;
    }
};

// =============================================================================================
// the vertical path: small vector helpers
// =============================================================================================

/// `floor(2^43 / q)`, the Barrett magic of [`barrett_u31`].
const fn barrett_magic(q: u16) -> u64 {
    (1u64 << 43) / q as u64
}

/// The multiple of `q` added before [`barrett_u31`] to make a folded-back lane non-negative.
const fn shift_up(q: u16) -> i32 {
    let a = mac::acc_after_reduce(q);
    let k = (a + q as i64 - 1) / q as i64;
    (k * q as i64) as i32
}

/// What [`barrett_u31`] needs of a base limb: the shifted lane is a non-negative i32, and the
/// 32x32 product against the magic stays inside the 64-bit lane it is formed in.
const fn drain_fits(q: u16) -> bool {
    let p = 2 * shift_up(q) as u64;
    p < (1u64 << 31) && p * barrett_magic(q) < (1u64 << 63)
}
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(drain_fits(QS_QUAD[i]));
        i += 1;
    }
    let mut i = 0;
    while i < 2 {
        assert!(drain_fits(QS[i]) && drain_fits(QS_LARGE[i]));
        i += 1;
    }
};

/// `p mod q` for `0 <= p < 2^31` with `p * barrett_magic(q) < 2^63`, 16 lanes at a time: the
/// quotient estimate is short by at most `p / 2^43 < 1`, so one conditional subtract finishes it.
#[inline(always)]
unsafe fn barrett_u31<const Q: u16>(p: __m512i) -> __m512i {
    let q = _mm512_set1_epi32(Q as i32);
    let mag = _mm512_set1_epi64(barrett_magic(Q) as i64);
    let lo = _mm512_set1_epi64(0xFFFF_FFFFu32 as i64);
    let he = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_and_si512(p, lo), mag));
    let ho = _mm512_srli_epi64::<43>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(p), mag));
    let t = _mm512_or_si512(he, _mm512_slli_epi64::<32>(ho));
    let r = _mm512_sub_epi32(p, _mm512_mullo_epi32(t, q));
    _mm512_min_epu32(r, _mm512_sub_epi32(r, q))
}

/// Declared output bound of the generic-input kernel of `q` — `gen_small` for a small splitting
/// prime, `gen_large` above `2^14`, `gen_quad` for a quadratic-slot one.
pub const fn gen_bound(q: u16) -> i32 {
    if q == 3889 {
        gen_small::Tw::<3889>::OUTPUT_BOUND
    } else if q == 9721 {
        gen_small::Tw::<9721>::OUTPUT_BOUND
    } else if ntt_large::is_large(q) {
        gen_large::output_bound(q)
    } else {
        gen_quad::output_bound(q)
    }
}

/// The shift [`center_epi16`] uses: the smallest power of two `K` with `K q >= gen_bound(q)`.
pub const fn center_k(q: u16) -> u32 {
    let mut k = 1u32;
    while (k * q as u32) < gen_bound(q) as u32 {
        k *= 2;
    }
    k
}

/// `K q + gen_bound(q)` must stay inside a u16 lane, which is what makes the unsigned trick work.
const fn center_fits(q: u16) -> bool {
    center_k(q) * q as u32 + gen_bound(q) as u32 <= 65535
}
const _: () = assert!(center_fits(3889) && center_fits(9721));
const _: () = assert!(center_fits(2917) && center_fits(4861) && center_fits(12637));
const _: () = assert!(center_fits(QS_LARGE[0]) && center_fits(QS_LARGE[1]));
const _: () = assert!(center_k(QS_LARGE[0]) == 1 && center_k(QS_LARGE[1]) == 1);

/// `x mod q` centered into `[-(q-1)/2, (q-1)/2]` for 32 i16 lanes with `|x| <= K q`: shift by
/// `K q` into `[0, 2 K q) < 2^16`, `log2 K + 1` unsigned conditional subtracts, then the
/// centering subtract.
#[inline(always)]
unsafe fn center_epi16<const Q: u16>(x: __m512i) -> __m512i {
    let q = Q as u32;
    let mut k = center_k(Q);
    let mut v = _mm512_add_epi16(x, _mm512_set1_epi16((k * q) as i16));
    while k >= 1 {
        let s = _mm512_sub_epi16(v, _mm512_set1_epi16((k * q) as i16));
        v = _mm512_min_epu16(v, s);
        k /= 2;
    }
    let hi = _mm512_cmpgt_epu16_mask(v, _mm512_set1_epi16(((q - 1) / 2) as i16));
    _mm512_mask_sub_epi16(v, hi, v, _mm512_set1_epi16(q as i16))
}

/// Every slot of a batch fully reduced and centered (`|v| <= (q-1)/2`); the input must satisfy
/// `|v| <= center_k(Q) * q`, which every generic kernel's output bound does.
///
/// # Safety
/// AVX-512 F/BW; `b` 64-byte aligned per `Batch32`.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub(crate) unsafe fn center_batch<const Q: u16>(b: &mut Batch32) {
    for j in 0..N {
        let p = b.v[j].as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(p, center_epi16::<Q>(_mm512_load_si512(p as *const __m512i)));
    }
}

// =============================================================================================
// the vertical path: the challenges
// =============================================================================================

/// The `r` challenges transformed, packed so that the scalar pair `(c_{2i}[u], c_{2i+1}[u])` is
/// one dword — a `vpbroadcastd` memory operand, and exactly the operand `vpmaddwd` wants against
/// two interleaved witness rows.
pub(crate) struct ChallengeNtt {
    /// `pair[i][u]` = `c_{2i}[u] as u16 | (c_{2i+1}[u] as u16) << 16`.
    pair: Vec<[u32; N]>,
}

/// The `r` challenges embedded as `c(-X^4)` into as many `Batch32` of coefficients as they need.
fn embed_v(challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let nb = challenges.len().div_ceil(32);
    let mut out: Vec<Batch32> = (0..nb).map(|_| Batch32::zero()).collect();
    for (j, c) in challenges.iter().enumerate() {
        for i in 0..c.weight {
            let m = c.positions[i] as usize;
            let co = 1 - 2 * ((c.signs >> i) & 1) as i16;
            out[j / 32].v[4 * m][j % 32] = if m.is_multiple_of(2) { co } else { -co };
        }
    }
    out
}

/// Which generic-input kernel a splitting prime runs, as an associated const so the choice is
/// made before the branches are emitted.
struct Gen<const Q: u16>;

impl<const Q: u16> Gen<Q> {
    const LARGE: bool = ntt_large::is_large(Q);
}

/// `NTT(b)` for a splitting prime, fully reduced and centered: `gen_small` below `2^14`,
/// `gen_large` above it.
///
/// # Safety
/// AVX-512 F/BW/VL/VBMI; `b` holds coefficients with `|x| <= q`.
#[inline(always)]
unsafe fn forward_split<const Q: u16>(b: &mut Batch32) {
    if Gen::<Q>::LARGE {
        gen_large::ntt_gen_batch32::<Q>(b);
    } else {
        gen_small::ntt_gen_batch32::<Q>(b);
    }
    center_batch::<Q>(b);
}

/// The inverse transform of a splitting base limb: `gen_small` below `2^14`, `gen_large` above
/// it, chosen before the branches are emitted.
///
/// # Safety
/// AVX-512 F/BW/VL/VBMI; `b` is a centered transform.
#[inline(always)]
unsafe fn inverse_split<const Q: u16>(b: &mut Batch32) {
    if Gen::<Q>::LARGE {
        gen_large::intt_gen_batch32::<Q>(b);
    } else {
        gen_small::intt_gen_batch32::<Q>(b);
    }
}

/// Transform the embedded challenges modulo `Q` and fully reduce them to centered slots, which
/// is what the accumulation bound of [`fold_period`] assumes.
fn challenge_ntt_v<const Q: u16>(challenges: &[ShortChallenge]) -> ChallengeNtt {
    let mut bs = embed_v(challenges);
    unsafe {
        for b in bs.iter_mut() {
            forward_split::<Q>(b);
        }
    }
    pack(&bs, challenges.len())
}

/// Each leaf's scalar written into both of the leaf's rows.
fn duplicate_leaf_scalars(b: &mut Batch32) {
    for j in 0..QUAD_SLOTS {
        debug_assert!(
            b.v[2 * j + 1] == [0i16; 32],
            "a challenge leaf is not a scalar"
        );
        b.v[2 * j + 1] = b.v[2 * j];
    }
}

/// The challenge transform a quadratic-slot *base* limb folds against.
fn challenge_ntt_quad_base_v<const Q: u16>(challenges: &[ShortChallenge]) -> ChallengeNtt {
    let mut bs = embed_v(challenges);
    unsafe {
        for b in bs.iter_mut() {
            gen_quad::ntt_quad_gen_batch32::<Q>(b);
            center_batch::<Q>(b);
        }
    }
    bs.iter_mut().for_each(duplicate_leaf_scalars);
    pack(&bs, challenges.len())
}

/// The transformed batches read out per challenge and packed into the dword pairs the
/// accumulation wants.
fn pack(bs: &[Batch32], r: usize) -> ChallengeNtt {
    assert!(
        r >= 2 && r % 2 == 0,
        "the fold pairs the chunks: r must be even"
    );
    let mut pair = vec![[0u32; N]; r / 2];
    for i in 0..r / 2 {
        for u in 0..N {
            let lo = bs[(2 * i) / 32].v[u][(2 * i) % 32] as u16;
            let hi = bs[(2 * i + 1) / 32].v[u][(2 * i + 1) % 32] as u16;
            pair[i][u] = lo as u32 | ((hi as u32) << 16);
        }
    }
    ChallengeNtt { pair }
}

/// `NTT(v)` for one limb, in place on centered coefficient batches, fully reduced and centered.
pub(crate) fn forward_limb(q: u16, bs: &mut [Batch32]) {
    unsafe {
        for b in bs.iter_mut() {
            match q {
                3889 => forward_split::<3889>(b),
                9721 => forward_split::<9721>(b),
                17497 => forward_split::<17497>(b),
                19441 => forward_split::<19441>(b),
                2917 => {
                    gen_quad::ntt_quad_gen_batch32::<2917>(b);
                    center_batch::<2917>(b);
                }
                4861 => {
                    gen_quad::ntt_quad_gen_batch32::<4861>(b);
                    center_batch::<4861>(b);
                }
                12637 => {
                    gen_quad::ntt_quad_gen_batch32::<12637>(b);
                    center_batch::<12637>(b);
                }
                _ => unreachable!("unknown limb prime {q}"),
            }
        }
    }
}

/// The challenges transformed for any limb prime, as vertical batches (slot-table builder and
/// tests).
pub(crate) fn challenge_batches_v(q: u16, challenges: &[ShortChallenge]) -> Vec<Batch32> {
    let mut bs = embed_v(challenges);
    forward_limb(q, &mut bs);
    bs
}

// =============================================================================================
// the vertical path: the accumulator
// =============================================================================================

/// One 64-byte aligned accumulator vector.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct AccVec([i32; 16]);

/// `acc[(b * 648 + u) * 2 + h]`: batch position `b` inside a chunk, slot `u`, half `h`.
///
/// Half 0 is `vpunpcklwd` of the two witness rows, half 1 is `vpunpckhwd`, so lane `t` of half
/// `h` carries ring element [`lane_of`]`(h, t)` of the chunk. The permutation is undone once,
/// when the accumulator is read out.
const fn lane_of(h: usize, t: usize) -> usize {
    8 * (t / 4) + 4 * h + t % 4
}

/// The `vpermi2d` indices that undo [`lane_of`]: `NAT[k]` gathers lanes `16k .. 16k+16` of the
/// natural element order out of (half 0, half 1).
const NAT: [[i32; 16]; 2] = {
    let mut nat = [[0i32; 16]; 2];
    let mut h = 0;
    while h < 2 {
        let mut t = 0;
        while t < 16 {
            let p = lane_of(h, t);
            nat[p / 16][p % 16] = (16 * h + t) as i32;
            t += 1;
        }
        h += 1;
    }
    nat
};

/// Two consecutive chunks into the accumulator of one batch position: 648 slot vectors, one
/// `vpunpck` and one `vpmaddwd` per half.
///
/// # Safety
/// AVX-512 F/BW; `w0`, `w1` and `acc` cover 648 vectors (`acc` 648 pairs), 64-byte aligned.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn accumulate_pair(w0: *const i16, w1: *const i16, cp: *const u32, acc: *mut i32) {
    for u in 0..N {
        let a = _mm512_load_si512(w0.add(32 * u) as *const __m512i);
        let b = _mm512_load_si512(w1.add(32 * u) as *const __m512i);
        let c = _mm512_set1_epi32(*cp.add(u) as i32);
        let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), c);
        let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), c);
        let d = acc.add(32 * u);
        _mm512_store_si512(
            d as *mut __m512i,
            _mm512_add_epi32(_mm512_load_si512(d as *const __m512i), lo),
        );
        _mm512_store_si512(
            d.add(16) as *mut __m512i,
            _mm512_add_epi32(_mm512_load_si512(d.add(16) as *const __m512i), hi),
        );
    }
}

/// The exact fold-back of [`crate::simd::commit`] over one batch position's accumulator.
///
/// # Safety
/// AVX-512 F/BW; `acc` covers `vecs` vectors, 64-byte aligned.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn fold_back<const Q: u16>(acc: *mut i32, vecs: usize) {
    for j in 0..vecs {
        let p = acc.add(16 * j);
        _mm512_store_si512(
            p as *mut __m512i,
            mac::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i)),
        );
    }
}

/// The accumulator of one batch position, folded back a last time, reduced to `[0, q)` and
/// written out in the natural element order as centered slots of `out`.
///
/// # Safety
/// AVX-512 F/BW/DQ; `acc` per the layout above, `out` 64-byte aligned.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512dq")]
unsafe fn drain<const Q: u16>(acc: *const i32, out: &mut Batch32) {
    let up = _mm512_set1_epi32(shift_up(Q));
    let half = _mm512_set1_epi32((Q as i32 - 1) / 2);
    let q = _mm512_set1_epi32(Q as i32);
    let i0 = _mm512_loadu_si512(NAT[0].as_ptr() as *const __m512i);
    let i1 = _mm512_loadu_si512(NAT[1].as_ptr() as *const __m512i);
    for u in 0..N {
        let p = acc.add(32 * u);
        let lo = mac::reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i));
        let hi = mac::reduce_vec::<Q>(_mm512_load_si512(p.add(16) as *const __m512i));
        let lo = barrett_u31::<Q>(_mm512_add_epi32(lo, up));
        let hi = barrett_u31::<Q>(_mm512_add_epi32(hi, up));
        let n0 = _mm512_permutex2var_epi32(lo, i0, hi);
        let n1 = _mm512_permutex2var_epi32(lo, i1, hi);
        let c0 = _mm512_mask_sub_epi32(n0, _mm512_cmpgt_epi32_mask(n0, half), n0, q);
        let c1 = _mm512_mask_sub_epi32(n1, _mm512_cmpgt_epi32_mask(n1, half), n1, q);
        let d = out.v[u].as_mut_ptr();
        _mm256_storeu_si256(d as *mut __m256i, _mm512_cvtepi32_epi16(c0));
        _mm256_storeu_si256(d.add(16) as *mut __m256i, _mm512_cvtepi32_epi16(c1));
    }
}

/// Batch positions accumulated together: 324 KB of accumulator, and `2 G` witness streams.
const G: usize = 4;

/// `v_ntt = sum_j c_j o W_j` modulo the base prime `Q`, centered, one `Batch32` per batch
/// position.
fn accumulate<const Q: u16>(aux: &AuxData, ch: &ChallengeNtt, bpc: usize) -> Vec<Batch32> {
    let pairs = ch.pair.len();
    let g = G.min(bpc);
    let mut acc = vec![AccVec([0i32; 16]); g * N * 2];
    let base = acc.as_mut_ptr() as *mut i32;
    let row = |j: usize, b: usize| aux.vertical[j * bpc + b].v.as_ptr() as *const i16;
    let mut out: Vec<Batch32> = (0..bpc).map(|_| Batch32::zero()).collect();
    unsafe {
        for b0 in (0..bpc).step_by(g) {
            acc.fill(AccVec([0i32; 16]));
            for i in 0..pairs {
                for b in 0..g {
                    accumulate_pair(
                        row(2 * i, b0 + b),
                        row(2 * i + 1, b0 + b),
                        ch.pair[i].as_ptr(),
                        base.add(32 * N * b),
                    );
                }
                if (2 * (i + 1)) % fold_period(Q) == 0 {
                    fold_back::<Q>(base, g * N * 2);
                }
            }
            for b in 0..g {
                drain::<Q>(base.add(32 * N * b), &mut out[b0 + b]);
            }
        }
    }
    out
}

/// The fold over a splitting base limb: transform the challenges, accumulate, invert.
fn fold_split<const Q: u16>(aux: &AuxData, challenges: &[ShortChallenge], bpc: usize) -> Vec<Batch32> {
    let ch = challenge_ntt_v::<Q>(challenges);
    let mut vb = accumulate::<Q>(aux, &ch, bpc);
    unsafe {
        for b in vb.iter_mut() {
            inverse_split::<Q>(b);
        }
    }
    vb
}

/// The same over a quadratic-slot base limb.
fn fold_quad<const Q: u16>(aux: &AuxData, challenges: &[ShortChallenge], bpc: usize) -> Vec<Batch32> {
    let ch = challenge_ntt_quad_base_v::<Q>(challenges);
    let mut vb = accumulate::<Q>(aux, &ch, bpc);
    unsafe {
        for b in vb.iter_mut() {
            gen_quad::intt_quad_gen_batch32::<Q>(b);
        }
    }
    vb
}

/// Element `p` of batch `b` as an element-major `[i16; N]` (the column of the vertical layout).
fn column_of(b: &Batch32, p: usize) -> [i16; N] {
    let mut e = [0i16; N];
    for (j, x) in e.iter_mut().enumerate() {
        *x = b.v[j][p];
    }
    e
}

/// The vertical fold: dispatch on the base limb's prime.
fn fold_witness_simd(aux: &AuxData, challenges: &[ShortChallenge], q: u16) -> Vec<[i16; N]> {
    assert_eq!(challenges.len(), aux.chunks, "one challenge per chunk");
    assert_eq!(challenges.len() % 2, 0, "the fold pairs the chunks");
    let bpc = aux.vertical.len() / aux.chunks;
    assert_eq!(bpc * aux.chunks, aux.vertical.len());
    let vb = match q {
        3889 => fold_split::<3889>(aux, challenges, bpc),
        9721 => fold_split::<9721>(aux, challenges, bpc),
        17497 => fold_split::<17497>(aux, challenges, bpc),
        19441 => fold_split::<19441>(aux, challenges, bpc),
        2917 => fold_quad::<2917>(aux, challenges, bpc),
        4861 => fold_quad::<4861>(aux, challenges, bpc),
        12637 => fold_quad::<12637>(aux, challenges, bpc),
        _ => unreachable!("unknown limb prime {q}"),
    };
    let mut out = Vec::with_capacity(32 * bpc);
    for b in 0..bpc {
        for p in 0..32 {
            out.push(column_of(&vb[b], p));
        }
    }
    out
}

// =============================================================================================
// A v (the verifier), vertical
// =============================================================================================

/// `32 * v.len().div_ceil(32)` centered coefficient elements repacked into vertical batches.
fn pack_vertical(v: &[[i16; N]]) -> Vec<Batch32> {
    let nb = v.len().div_ceil(32);
    let mut bs: Vec<Batch32> = (0..nb).map(|_| Batch32::zero()).collect();
    for (i, e) in v.iter().enumerate() {
        for (j, &x) in e.iter().enumerate() {
            bs[i / 32].v[j][i % 32] = x;
        }
    }
    bs
}

/// `y[u] = sum_i A_i[u] v_i[u] mod q` for a splitting limb, on the commitment's own packed
/// accumulator: `v` in centered coefficients, forwarded per batch; `A` in the vertical layout.
///
/// # Safety
/// AVX-512 PCS feature set; `a`/`v` same length, 64-byte aligned.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
unsafe fn a_times_v_fwd<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> Coeffs {
    let mut acc = mac::Acc::zero();
    let ap = acc.v.as_mut_ptr() as *mut i32;
    let mut w = Box::new(Batch32::zero());
    for b in 0..a.len() {
        w.v.copy_from_slice(&v[b].v);
        forward_split::<Q>(&mut w);
        let ar = a[b].v.as_ptr() as *const i16;
        mac::mac_batch(w.v.as_ptr() as *const i16, ar, ap);
        if (b + 1) % av_period(Q) == 0 {
            mac::reduce_acc::<Q>(ap);
        }
    }
    mac::finish::<Q>(&acc)
}

/// The same for a quadratic-slot limb, on the quadratic accumulator.
///
/// # Safety
/// AVX-512 PCS feature set; `a`/`v` same length, 64-byte aligned.
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vl", enable = "avx512vbmi", enable = "avx512vbmi2", enable = "avx512vnni", enable = "gfni")]
unsafe fn a_times_v_fwd_quad<const Q: u16>(a: &[Batch32], v: &[Batch32]) -> Coeffs {
    let mut acc = mac::QuadAcc::zero();
    let (p01, p2) = (
        acc.p01.as_mut_ptr() as *mut i32,
        acc.p2.as_mut_ptr() as *mut i32,
    );
    let mut w = Box::new(Batch32::zero());
    for b in 0..a.len() {
        w.v.copy_from_slice(&v[b].v);
        gen_quad::ntt_quad_gen_batch32::<Q>(&mut w);
        center_batch::<Q>(&mut w);
        let ar = a[b].v.as_ptr() as *const i16;
        mac::mac_quad_batch::<Q>(w.v.as_ptr() as *const i16, ar, p01, p2);
        if (b + 1) % av_period_quad(Q) == 0 {
            mac::reduce_quad_acc::<Q>(&mut acc);
        }
    }
    mac::finish_quad::<Q>(&acc)
}

/// `A v` for one limb from centered coefficient batches against the vertical `A`, dispatched on
/// the prime. Falls back to the scalar path when SIMD is not available.
pub fn a_times_v_forward(q: u16, a: &[Batch32], v: &[[i16; N]]) -> Coeffs {
    if !crate::simd::available() {
        // unreachable in practice (callers gate on availability); keeps the fn callable in tests
        let flat: Vec<[i16; N]> = (0..a.len() * 32)
            .map(|i| column_of(&a[i / 32], i % 32))
            .collect();
        return a_times_v(q, &flat, v);
    }
    let vb = pack_vertical(v);
    assert_eq!(a.len(), vb.len(), "the key and v disagree on the chunk length");
    unsafe {
        match q {
            3889 => a_times_v_fwd::<3889>(a, &vb),
            9721 => a_times_v_fwd::<9721>(a, &vb),
            17497 => a_times_v_fwd::<17497>(a, &vb),
            19441 => a_times_v_fwd::<19441>(a, &vb),
            2917 => a_times_v_fwd_quad::<2917>(a, &vb),
            4861 => a_times_v_fwd_quad::<4861>(a, &vb),
            12637 => a_times_v_fwd_quad::<12637>(a, &vb),
            _ => unreachable!("unknown limb prime {q}"),
        }
    }
}

// =============================================================================================
// the commitment fold (sum_j c_j C_j) — the slot-table path
// =============================================================================================

const PRIMES: [u16; 7] = [3889, 9721, 17497, 19441, 2917, 4861, 12637];

const QUAD_COMPONENT_SLOT: [[u16; N162]; 4] = {
    let mut m = [[0u16; N162]; 4];
    let mut s = 0;
    while s < N162 {
        m[0][s] = 2 * QUAD_CLASS_SLOT[0][s];
        m[1][s] = 2 * QUAD_CLASS_SLOT[0][s] + 1;
        m[2][s] = 2 * QUAD_CLASS_SLOT[1][s];
        m[3][s] = 2 * QUAD_CLASS_SLOT[1][s] + 1;
        s += 1;
    }
    m
};

const _: () = {
    let mut seen = [false; N];
    let mut t = 0;
    while t < 4 {
        let mut s = 0;
        while s < N162 {
            let u = QUAD_COMPONENT_SLOT[t][s] as usize;
            assert!(!seen[u]);
            seen[u] = true;
            s += 1;
        }
        t += 1;
    }
};

/// Which 648-slots hold the four `R_162` components, per tree.
pub(crate) fn component_slots(quad: bool) -> &'static [[u16; N162]; 4] {
    if quad {
        &QUAD_COMPONENT_SLOT
    } else {
        &SLOT_648
    }
}

static SLOT_TABLES: [OnceLock<Box<slots::SlotTable>>; PRIMES.len()] =
    [const { OnceLock::new() }; PRIMES.len()];

fn slot_prime_index(q: u16) -> usize {
    PRIMES
        .iter()
        .position(|&p| p == q)
        .unwrap_or_else(|| unreachable!("no limb with q = {q}"))
}

/// The transformed slots of the 162 unit challenges of `R_162` (component 0: a subring element's
/// four components are equal), each with its negation.
fn build_slot_table(q: u16) -> Box<slots::SlotTable> {
    let units: Vec<ShortChallenge> = (0..N162)
        .map(|p| {
            let mut c = ShortChallenge::zero();
            c.positions[0] = p as u8;
            c.weight = 1;
            c
        })
        .collect();
    let quad = quadratic_slots(q);
    let mut bs = challenge_batches_v(q, &units);
    if quad {
        bs.iter_mut().for_each(duplicate_leaf_scalars);
    }
    let map = component_slots(quad);
    let mut t = slots::SlotTable::zero();
    for p in 0..N162 {
        for s in 0..N162 {
            let x = bs[p / 32].v[map[0][s] as usize][p % 32];
            t.rows[2 * p][s] = x;
            t.rows[2 * p + 1][s] = -x;
        }
    }
    t
}

fn slot_table(q: u16) -> &'static slots::SlotTable {
    SLOT_TABLES[slot_prime_index(q)].get_or_init(|| build_slot_table(q))
}

/// The commitment fold of one limb over the slot tables.
fn fold_columns_limb<const Q: u16>(
    challenges: &[ShortChallenge],
    columns: &[[*const i16; 4]],
    out: &mut [[i16; N162]; 4],
) {
    let table = slot_table(Q);
    let mut acc = [slots::SlotAcc::zero(); 4];
    let mut ch = slots::Ch::zero();
    let period = slots::slot_period(Q);
    unsafe {
        for (j, c) in challenges.iter().enumerate() {
            slots::challenge_slots::<Q>(table, c, &mut ch);
            for (t, a) in acc.iter_mut().enumerate() {
                slots::mac_slots(a, &ch, columns[j][t]);
            }
            if (j + 1) % period == 0 {
                for a in acc.iter_mut() {
                    slots::reduce_slots::<Q>(a);
                }
            }
        }
        for (t, a) in acc.iter().enumerate() {
            slots::finish_slots::<Q>(a, &mut out[t]);
        }
    }
}

/// The commitment fold dispatched on the prime.
fn fold_columns_slots(
    q: u16,
    challenges: &[ShortChallenge],
    columns: &[[*const i16; 4]],
    out: &mut [[i16; N162]; 4],
) {
    match q {
        3889 => fold_columns_limb::<3889>(challenges, columns, out),
        9721 => fold_columns_limb::<9721>(challenges, columns, out),
        17497 => fold_columns_limb::<17497>(challenges, columns, out),
        19441 => fold_columns_limb::<19441>(challenges, columns, out),
        2917 => fold_columns_limb::<2917>(challenges, columns, out),
        4861 => fold_columns_limb::<4861>(challenges, columns, out),
        12637 => fold_columns_limb::<12637>(challenges, columns, out),
        _ => unreachable!("unknown limb prime {q}"),
    }
}

/// `sum_j c_j C_j` per modulus: multiplication by a challenge acts on the four `R_162`
/// components of a commitment alike and slot-wise in the `R_162` transform, so this is four
/// length-`r` inner products per slot and modulus. Returns the four folded rows.
///
/// The AVX-512 slot-table path when the CPU has it, the scalar Horner path otherwise; both are
/// bit-identical (exact modular arithmetic, the same centered representative).
pub fn fold_commitment(
    q: u16,
    challenges: &[ShortChallenge],
    columns: &[Vec<PowerOfThreeRing>],
) -> [PowerOfThreeRing; 4] {
    if crate::simd::available() {
        let cols: Vec<[*const i16; 4]> = columns
            .iter()
            .map(|c| {
                [
                    c[0].v.as_ptr(),
                    c[1].v.as_ptr(),
                    c[2].v.as_ptr(),
                    c[3].v.as_ptr(),
                ]
            })
            .collect();
        let mut out = [[0i16; N162]; 4];
        fold_columns_slots(q, challenges, &cols, &mut out);
        return [
            PowerOfThreeRing { v: out[0] },
            PowerOfThreeRing { v: out[1] },
            PowerOfThreeRing { v: out[2] },
            PowerOfThreeRing { v: out[3] },
        ];
    }
    fold_commitment_scalar(q, challenges, columns)
}

/// The scalar reference commitment fold (the original port's path).
fn fold_commitment_scalar(
    q: u16,
    challenges: &[ShortChallenge],
    columns: &[Vec<PowerOfThreeRing>],
) -> [PowerOfThreeRing; 4] {
    // columns[j][row] — the 4 components of column j (one modulus)
    // The challenge slots are converted to [0, q) once (they used to be re-centred per
    // (row, slot, column) — 648 times per value).
    let ch: Vec<Vec<u32>> = challenges
        .iter()
        .map(|c| {
            challenge_r162_slots(q, c)
                .iter()
                .map(|&s| s.rem_euclid(q as i16) as u32)
                .collect()
        })
        .collect();
    let q64 = q as u64;
    let half = (q as i64 - 1) / 2;
    let mut out = [PowerOfThreeRing::zero(); 4];
    for row in 0..4 {
        for s in 0..N162 {
            let mut acc = 0u64;
            for (j, c) in columns.iter().enumerate() {
                let a = (c[row].v[s] as i64).rem_euclid(q64 as i64) as u64;
                acc += a * ch[j][s] as u64;
            }
            let r = (acc % q64) as i64;
            out[row].v[s] = if r > half { (r - q as i64) as i16 } else { r as i16 };
        }
    }
    out
}

/// The `R_162` slot multiplier of a challenge for modulus q: slot s holds `c(-theta^{v_s})` —
/// the challenge enters `R_648` as `c(-X^4)`, i.e. as the `Z`-basis element read in the `Y`
/// basis (`Z = -Y`), so the component slots (Y-basis evaluations at `theta^{v_s}`) are scaled
/// by `c` evaluated at `-theta^{v_s}`: coefficient m picks up `(-1)^m`.
pub fn challenge_r162_slots(q: u16, c: &ShortChallenge) -> [i16; N162] {
    let mut coeffs = c.coeffs64();
    for m in 0..N162 {
        if m % 2 == 1 {
            coeffs[m] = -coeffs[m];
        }
    }
    let q64 = q as u64;
    let theta = if quadratic_slots(q) {
        let psi = match q {
            2917 => crate::params::ParamsQ::<2917>::PSI972 as u64,
            4861 => crate::params::ParamsQ::<4861>::PSI972 as u64,
            12637 => crate::params::ParamsQ::<12637>::PSI972 as u64,
            _ => unreachable!(),
        };
        psi * psi % q64
    } else {
        let psi = match q {
            3889 => crate::params::Params::<3889>::PSI as u64,
            9721 => crate::params::Params::<9721>::PSI as u64,
            17497 => crate::params::Params::<17497>::PSI as u64,
            19441 => crate::params::Params::<19441>::PSI as u64,
            _ => unreachable!(),
        };
        crate::params::pow_mod(psi, 4, q64)
    };
    let half = (q as i64 - 1) / 2;
    let mut out = [0i16; N162];
    for s in 0..N162 {
        let v = crate::ring::POW3_SLOT_EXP[s] as u64;
        let x = crate::params::pow_mod(theta, v, q64);
        let mut acc = 0u64;
        for k in (0..N162).rev() {
            // Barrett for every Horner step: acc < q, x < q, so acc*x + coeff < q^2 + q,
            // the range params::barrett_mod_u64 is exhaustively checked on.
            acc = crate::params::barrett_mod_u64(acc * x + coeffs[k].rem_euclid(q64 as i64) as u64, q);
        }
        let r = acc as i64;
        out[s] = if r > half { (r - q as i64) as i16 } else { r as i16 };
    }
    out
}

/// Slot indices for the fold of a quad-limb commitment (tests and bd).
pub fn quad_component_slot(t: usize, s: usize) -> usize {
    let base = QUAD_CLASS_SLOT[t / 2][s] as usize;
    2 * base + t % 2
}

/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4m + k` is component `k` of packed ring element `m`, its bit `p` the parity of coefficient
/// `4p + k` of that element.
pub fn components_mod_2(v: &[[i16; N]]) -> Vec<crate::binfield::F162> {
    let mut out = vec![crate::binfield::F162::ZERO; 4 * v.len()];
    for (m, e) in v.iter().enumerate() {
        for p in 0..N162 {
            for k in 0..4 {
                if e[4 * p + k] & 1 == 1 {
                    out[4 * m + k].0[p >> 6] |= 1u64 << (p & 63);
                }
            }
        }
    }
    out
}

/// `A v` for one limb, dispatched (tests / verifier).
pub fn a_times_v_of(q: u16, a: &[[i16; N]], v: &[[i16; N]]) -> Coeffs {
    if quadratic_slots(q) {
        // quadratic limb: slot product is per-leaf bilinear; use mul_quad_slots per element
        let q64 = q as u64;
        let mut acc = [0u64; N];
        for i in 0..v.len() {
            let mut coeffs = [0u32; N];
            for u in 0..N {
                coeffs[u] = (v[i][u].rem_euclid(q as i16) as i64 as u32) % q as u32;
            }
            let t = crate::ring::ntt_of(q, &coeffs);
            let mut acoef = [0u32; N];
            for u in 0..N {
                acoef[u] = (a[i][u].rem_euclid(q as i16) as i64 as u32) % q as u32;
            }
            let prod = match q {
                2917 => crate::scalar::mul_quad_slots::<2917>(&acoef, &t),
                4861 => crate::scalar::mul_quad_slots::<4861>(&acoef, &t),
                12637 => crate::scalar::mul_quad_slots::<12637>(&acoef, &t),
                _ => unreachable!(),
            };
            for u in 0..N {
                acc[u] += prod[u] as u64;
            }
        }
        let mut out = [0u32; N];
        for u in 0..N {
            out[u] = (acc[u] % q64) as u32;
        }
        out
    } else {
        a_times_v(q, a, v)
    }
}

/// Rebuild the raw 648-row commitment of one folded chunk for one limb (tests / verifier):
/// `A v` then decompose. Returns the four `R_162` rows.
pub fn a_times_v_components(q: u16, a: &[[i16; N]], v: &[[i16; N]]) -> [PowerOfThreeRing; 4] {
    components_of(q, &a_times_v_of(q, a, v))
}
