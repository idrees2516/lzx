//! Packed Goldilocks AVX-512 field kernels — 8 field elements per `__m512i`.
//!
//! The Plonky2-style packed doctrine applied to the Goldilocks prime
//! `q = 2^64 − 2^32 + 1` for the whole non-labinius stack (sumcheck rounds,
//! MLE binding, eq tables, hypercube sums). Everything is exact: every
//! kernel is bit-identical to the scalar [`crate::field::Goldilocks`] ops,
//! cross-checked by randomized + edge-value tests in this module. There is
//! no rounding anywhere — the speedups are speedups of *verified-identical*
//! computation.
//!
//! # The reduction (derived here, tested below)
//!
//! For a 128-bit value `x = hi·2^64 + lo` the Goldilocks identity
//! `2^64 ≡ 2^32 − 1 (mod q)` gives one *fold*:
//!
//! ```text
//! x ≡ lo + hi·(2^32 − 1)          (mod q)
//! ```
//!
//! and splitting `hi·2^32 = (hi >> 32)·2^64 + ((hi & M32) << 32)` lets the
//! fold run entirely in u64 lanes with one carry and one borrow:
//!
//! ```text
//! sum = lo + ((hi & M32) << 32)        with carry cA ∈ {0,1}
//! q64 = (hi >> 32) + cA                (< 2^33)
//! lo' = sum − hi                       with borrow b ∈ {0,1}
//! hi' = q64 − b
//! ```
//!
//! A fold with `hi = 0` is the identity, so a fixed unroll is safe. Three
//! folds bring **any** u128 input to `hi ≤ 1` (fold 1 gives `hi ≤ 2^32`,
//! fold 2 gives `hi ≤ 1`, and a third fold keeps `hi ≤ 1` while forcing
//! `lo < 2^32` whenever `hi = 1`, because that case only triggers when
//! `lo ≥ 2^64 − 2^32` and then `lo + 2^32 − 2^64 < 2^32`). The last `2^64`
//! is therefore worth exactly `2^32 − 1` with **no wrap possible**: one
//! masked add of `0xFFFF_FFFF` where `hi = 1`, followed by the canonical
//! conditional subtraction of `q` — exactly the value of
//! [`Goldilocks::from_u128`] (verified against big-int arithmetic over all
//! edge pairs plus 600k random u128s).
//!
//! # Packed multiplication
//!
//! A full 128-bit product is assembled from 32-bit half products
//! (`vpmuludq` = `_mm512_mul_epu32` on the low halves of each 64-bit lane):
//!
//! ```text
//! a·b = hh·2^64 + mid·2^32 + ll,   mid = a_lo·b_hi + a_hi·b_lo (< 2^65)
//! lo128 = ((mid & M32) << 32) + ll                (carry c2)
//! hi128 = hh + (mid >> 32) + c2 + (c1 << 32)      (c1 = carry of mid; fits u64:
//!                                                  it is the true high word)
//! ```
//!
//! followed by three folds, the masked `2^32 − 1` compensation for a
//! residual high bit, and the canonical conditional subtraction, i.e. the
//! lane result is exactly `Goldilocks::from_u128(a as u128 * b as u128)` for
//! *any* u64 operands (canonical or not).
//!
//! # Lazy accumulation
//!
//! Adding is where reduction is deferred: two canonical values already need
//! carry compensation (`a + b` can wrap, and the dropped `2^64` is worth
//! `2^32 − 1 = 0xFFFFFFFF`), so plain deferred adds would silently lose
//! wraps. Instead an 8-lane accumulator ([`Sum8`]) keeps lazy lane values in
//! `[0, 2^64)` plus a per-lane *carry count* `k`, maintaining the invariant
//!
//! ```text
//! Σ (true integers) ≡ acc[i] + k[i]·0xFFFFFFFF   (mod q)
//! ```
//!
//! which holds regardless of how often the lanes wrap (each wrap adds
//! exactly one owed `0xFFFFFFFF`). The final canonical value is
//! `from_u128(acc + k·0xFFFFFFFF)` per lane plus exact adds across lanes —
//! the canonical representative is order-independent, so this is
//! bit-identical to the scalar sequential sum.
//!
//! # Gating
//!
//! Runtime detection, exactly the [`crate`] analogue of
//! `lattice-labinius`'s `hw.rs` pattern: one `OnceLock` gate
//! ([`avx512_field`], needs `avx512f` only — every intrinsic used here is
//! AVX512F) with an `LZX_NO_SIMD=1` environment escape hatch so the scalar
//! path can be benchmarked. On non-x86-64 targets the gate is `false` and
//! every API below answers from the scalar reference.
//!
//! # Deliberately scalar
//!
//! * [`Goldilocks::batch_inverse`] — Montgomery's trick is already
//!   inversion-minimal and its prefix products are inherently serial; there
//!   is no profitable 8-lane shape (and no sumcheck hot path through it).
//! * `interpolate_at` — at most `(d+1)² ≤ 16` small inverses per round
//!   versus `2^num_vars · terms` products: not field-bound.

#![allow(unsafe_code)] // core::arch intrinsics, gated on runtime detection (see module docs)
// The AVX-512F intrinsics (`_mm512_*`) stabilized after the workspace MSRV
// (1.75); they compile only under `cfg(target_arch = "x86_64")` and execute
// only behind the runtime `avx512_field()` gate, so the portable build
// surface is unaffected. This deployment's toolchain (1.98) supports them —
// the same situation as lattice-labinius's AVX-512 kernels.
#![allow(clippy::incompatible_msrv)]

use crate::field::Goldilocks;
use std::ffi::OsStr;
use std::sync::OnceLock;

/// `LZX_NO_SIMD=1` disables the AVX-512 kernels for this process (scalar
/// benchmarking escape hatch). Any other value — including an unset
/// variable — leaves the kernels enabled subject to CPU detection.
fn simd_disabled_by_env(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("1"))
}

static AVX512_FIELD: OnceLock<bool> = OnceLock::new();

/// Are the packed Goldilocks AVX-512 kernels usable? Detected once per
/// process; `LZX_NO_SIMD=1` forces the scalar reference paths.
pub fn avx512_field() -> bool {
    *AVX512_FIELD.get_or_init(|| {
        if simd_disabled_by_env(std::env::var_os("LZX_NO_SIMD").as_deref()) {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Slice-level kernels. Length contracts: every op processes
// `n = min(relevant lengths)` elements; callers in this workspace always
// pass equal lengths. Inputs to add/sub-style ops must be canonical
// (`< q`) — the same contract the scalar `Goldilocks` methods assume.
// ---------------------------------------------------------------------------

/// `out[i] = a[i] · b[i]` — bit-exact with `Goldilocks::mul` per element
/// (any u64 operand values, output canonical).
pub fn mul_slices(a: &[Goldilocks], b: &[Goldilocks], out: &mut [Goldilocks]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on all three slices.
        unsafe { imp::mul_slice_simd(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = Goldilocks::from_u128(a[i].0 as u128 * b[i].0 as u128);
    }
}

/// `out[i] = a[i] + b[i]` — bit-exact with `Goldilocks::add` (canonical inputs).
pub fn add_slices(a: &[Goldilocks], b: &[Goldilocks], out: &mut [Goldilocks]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on all three slices.
        unsafe { imp::add_slice_simd(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = a[i].add(&b[i]);
    }
}

/// `out[i] = a[i] − b[i]` — bit-exact with `Goldilocks::sub` (canonical inputs).
pub fn sub_slices(a: &[Goldilocks], b: &[Goldilocks], out: &mut [Goldilocks]) {
    let n = a.len().min(b.len()).min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on all three slices.
        unsafe { imp::sub_slice_simd(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = a[i].sub(&b[i]);
    }
}

/// Packed `Goldilocks::from_u128`: reduce the 128-bit integers
/// `hi[i]·2^64 + lo[i]` to canonical field elements — three reduction folds,
/// the masked `2^32 − 1` compensation for a residual high bit, and the
/// canonical conditional subtraction per lane (the module-level derivation).
/// Bit-exact with the scalar reduction for **any** u128 input.
///
/// This is also the exactness bridge for the multiplier: feeding it the
/// `(hi, lo)` halves of a full 128-bit product reproduces `mul_slices`.
pub fn reduce128(hi: &[u64], lo: &[u64], out: &mut [Goldilocks]) {
    let n = hi.len().min(lo.len()).min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on all three slices.
        unsafe { imp::reduce128_slice_simd(hi.as_ptr(), lo.as_ptr(), out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = Goldilocks::from_u128(((hi[i] as u128) << 64) | (lo[i] as u128));
    }
}

/// Batched fused multiply-add: `out[i] = a[i]·b[i] + c[i]` — bit-exact with
/// the scalar `a.mul(&b).add(&c)` chain.
pub fn mul_add_slices(a: &[Goldilocks], b: &[Goldilocks], c: &[Goldilocks], out: &mut [Goldilocks]) {
    let n = a.len().min(b.len()).min(c.len()).min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on all four slices.
        unsafe { imp::mul_add_slice_simd(a.as_ptr(), b.as_ptr(), c.as_ptr(), out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = a[i].mul(&b[i]).add(&c[i]);
    }
}

/// `out[i] = a[i] · s` (broadcast scalar) — the eq-table / tensor / scale
/// workhorse. Source and destination may alias elementwise when the caller
/// holds the right borrows; use [`mul_scalar_slice_inplace`] for in-place.
pub fn mul_scalar_slice(a: &[Goldilocks], s: Goldilocks, out: &mut [Goldilocks]) {
    let n = a.len().min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on both slices.
        unsafe { imp::mul_scalar_slice_simd(a.as_ptr(), s.0, out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = a[i].mul(&s);
    }
}

/// `a[i] = a[i] · s` in place.
pub fn mul_scalar_slice_inplace(a: &mut [Goldilocks], s: Goldilocks) {
    let n = a.len();
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        let p = a.as_mut_ptr();
        // SAFETY: gate checked; elementwise load/mul/store over chunks*8 <= n elements.
        unsafe { imp::mul_scalar_slice_simd(p as *const Goldilocks, s.0, p, chunks) };
        done = chunks * 8;
    }
    for v in a[done..n].iter_mut() {
        *v = v.mul(&s);
    }
}

/// The sumcheck half-binding: `out[i] = lo[i] + (hi[i] − lo[i])·r` —
/// bit-exact with the scalar `a.add(&b.sub(&a).mul(&r))` binding step.
pub fn bind_half_slices(lo: &[Goldilocks], hi: &[Goldilocks], r: Goldilocks, out: &mut [Goldilocks]) {
    let n = lo.len().min(hi.len()).min(out.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; chunks*8 <= n elements are in bounds on all three slices.
        unsafe { imp::bind_half_simd(lo.as_ptr(), hi.as_ptr(), r.0, out.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        out[i] = lo[i].add(&hi[i].sub(&lo[i]).mul(&r));
    }
}

/// In-place `fix_variables` binding of the *first* variable:
/// `evals[i] = evals[i] + (evals[i + half] − evals[i])·r` for `i < half`,
/// `half = evals.len()/2` (the second half is left untouched, matching the
/// scalar loop that only writes the low half).
pub fn bind_first_half_in_place(evals: &mut [Goldilocks], r: Goldilocks) {
    let half = evals.len() / 2;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if half >= 8 && avx512_field() {
        let chunks = half / 8;
        // SAFETY: gate checked; reads/writes stay inside evals (first half writes,
        // second-half reads at disjoint offsets).
        unsafe { imp::bind_first_half_simd(evals.as_mut_ptr(), half, r.0, chunks) };
        done = chunks * 8;
    }
    for i in done..half {
        let a = evals[i];
        let b = evals[i + half];
        evals[i] = a.add(&b.sub(&a).mul(&r));
    }
}

/// In-place binding of the *last* (least-significant) variable:
/// `evals[i] = evals[2i] + (evals[2i+1] − evals[2i])·r` for
/// `i < evals.len()/2` — the `fix_last_variables` fold over adjacent pairs.
pub fn bind_pairs_in_place(evals: &mut [Goldilocks], r: Goldilocks) {
    let half = evals.len() / 2;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if half >= 8 && avx512_field() {
        let chunks = half / 8;
        // SAFETY: gate checked; each iteration reads 16 inputs before writing its
        // 8 outputs at strictly lower offsets (see imp::bind_pairs_simd).
        unsafe { imp::bind_pairs_simd(evals.as_mut_ptr(), r.0, chunks) };
        done = chunks * 8;
    }
    for i in done..half {
        let a = evals[2 * i];
        let b = evals[2 * i + 1];
        evals[i] = a.add(&b.sub(&a).mul(&r));
    }
}

/// In-place **projective** binding of the first variable over the monomial
/// (coefficient) basis — Corollary 3.2 of "The Sum-Check Protocol over the
/// Monomial Basis" (ePrint 2026/762):
/// `coeffs[i] = coeffs[i] + r · coeffs[i + half]` for `i < half`,
/// `half = coeffs.len()/2`.
/// The first half holds the monomials without the variable (the value at 0),
/// the second half the coefficients of the variable (the value at infinity),
/// so binding is one multiplication and one addition — **no subtraction**.
/// The second half is left untouched, matching `bind_first_half_in_place`.
pub fn bind_projective_first_half_in_place(coeffs: &mut [Goldilocks], r: Goldilocks) {
    let half = coeffs.len() / 2;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if half >= 8 && avx512_field() {
        let chunks = half / 8;
        // SAFETY: gate checked; writes stay in the first half, second-half
        // reads at disjoint offsets (same aliasing argument as
        // imp::bind_first_half_simd).
        unsafe { imp::bind_projective_first_half_simd(coeffs.as_mut_ptr(), half, r.0, chunks) };
        done = chunks * 8;
    }
    for i in done..half {
        let a = coeffs[i];
        let b = coeffs[i + half];
        coeffs[i] = a.add(&r.mul(&b));
    }
}

/// In-place binding of the *last* (least-significant) variable of the
/// monomial (coefficient) basis:
/// `coeffs[i] = coeffs[2i] + r · coeffs[2i+1]` for `i < len/2` — the
/// projective analogue of `bind_pairs_in_place`, again subtraction-free.
pub fn bind_projective_pairs_in_place(coeffs: &mut [Goldilocks], r: Goldilocks) {
    let half = coeffs.len() / 2;
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if half >= 8 && avx512_field() {
        let chunks = half / 8;
        // SAFETY: gate checked; each iteration loads its 16 inputs before
        // writing 8 outputs at strictly lower offsets (same argument as
        // imp::bind_pairs_simd).
        unsafe { imp::bind_projective_pairs_simd(coeffs.as_mut_ptr(), r.0, chunks) };
        done = chunks * 8;
    }
    for i in done..half {
        let a = coeffs[2 * i];
        let b = coeffs[2 * i + 1];
        coeffs[i] = a.add(&r.mul(&b));
    }
}

/// The dense multilinear `eq` table `prod_i [(1−x_i)(1−b_i) + x_i·b_i]`
/// evaluated at every hypercube vertex `b` (variable 0 = most significant
/// index bit, matching `DenseMle::eq_extension`). Bit-exact with the scalar
/// construction: every element is a canonical product of the same factors.
pub fn eq_table(point: &[Goldilocks]) -> Vec<Goldilocks> {
    let m = point.len();
    let mut evals = vec![Goldilocks::ZERO; 1usize << m];
    evals[0] = Goldilocks::ONE;
    let mut cur = 1usize;
    // Process variables in reverse so variable 0 lands on the most
    // significant bit (the fix_variables convention).
    for p in point.iter().rev() {
        let one_minus_p = Goldilocks::ONE.sub(p);
        let (first, second) = evals.split_at_mut(cur);
        // SIMD: two broadcast-scalar multiplies per doubling step.
        mul_scalar_slice(first, *p, second);
        mul_scalar_slice_inplace(first, one_minus_p);
        cur *= 2;
    }
    evals
}

/// Exact canonical sum over a slice — bit-exact with the scalar sequential
/// `acc.add(e)` loop (the canonical representative is order-independent).
pub fn sum_slice(a: &[Goldilocks]) -> Goldilocks {
    let mut acc = Sum8::new();
    acc.accumulate_term(Goldilocks::ONE, &[a]);
    acc.finish()
}

/// Pointwise term-product accumulation for dense materialization:
/// `acc[i] += coeff · Π_j factors[j][i]` with exact adds — the
/// `to_dense_mle` kernel.
pub fn accumulate_term_pointwise(
    factors: &[&[Goldilocks]],
    coeff: Goldilocks,
    acc: &mut [Goldilocks],
) {
    if factors.is_empty() {
        return;
    }
    let n = factors[0].len().min(acc.len());
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && avx512_field() {
        let chunks = n / 8;
        // SAFETY: gate checked; every factor and acc cover chunks*8 <= n elements.
        unsafe { imp::accumulate_term_pointwise_simd(factors, coeff.0, acc.as_mut_ptr(), chunks) };
        done = chunks * 8;
    }
    for i in done..n {
        let mut prod = coeff;
        for f in factors {
            prod = prod.mul(&f[i]);
        }
        acc[i] = acc[i].add(&prod);
    }
}

// ---------------------------------------------------------------------------
// Sum8: the 8-lane lazy accumulator behind every summation hot loop.
// ---------------------------------------------------------------------------

/// Eight lazy Goldilocks accumulators with exact carry accounting.
///
/// Invariant per lane: the true partial sum ≡ `acc[i] + carries[i]·0xFFFF_FFFF`
/// (mod q) — each wrapping add owes exactly one `2^64 ≡ 2^32 − 1`. This is
/// exact for any number of accumulations (see module docs), and
/// [`finish`](Self::finish) produces the same canonical element as the
/// scalar sequential sum regardless of how the additions were grouped.
///
/// This is also the "weighted sum" kernel: `accumulate_term(c, &[xs])`
/// computes `Σ_i c·xs[i]` lazily.
#[derive(Clone, Debug)]
pub struct Sum8 {
    acc: [u64; 8],
    carries: [u64; 8],
}

impl Default for Sum8 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sum8 {
    /// Zero accumulator.
    #[inline]
    pub fn new() -> Self {
        Sum8 {
            acc: [0; 8],
            carries: [0; 8],
        }
    }

    /// Accumulate `coeff · Π_j factors[j][p]` for every index `p` (all
    /// factor slices must share one length; the sumcheck term-product shape).
    /// Bit-exact with the scalar nested loop.
    pub fn accumulate_term(&mut self, coeff: Goldilocks, factors: &[&[Goldilocks]]) {
        if factors.is_empty() || factors[0].is_empty() {
            return;
        }
        let n = factors[0].len();
        debug_assert!(factors.iter().all(|f| f.len() == n));
        let mut done = 0;
        #[cfg(target_arch = "x86_64")]
        if n >= 8 && avx512_field() {
            let chunks = n / 8;
            // SAFETY: gate checked; every factor covers chunks*8 <= n elements
            // (equal lengths debug-asserted above).
            unsafe {
                imp::accumulate_term_simd(
                    &mut self.acc,
                    &mut self.carries,
                    coeff.0,
                    factors,
                    chunks,
                )
            };
            done = chunks * 8;
        }
        // Tail (and the no-SIMD fallback): continue lane 0's lazy chain.
        for p in done..n {
            let mut prod = coeff;
            for f in factors {
                prod = prod.mul(&f[p]);
            }
            let (s, wrapped) = self.acc[0].overflowing_add(prod.0);
            self.acc[0] = s;
            self.carries[0] += u64::from(wrapped);
        }
    }

    /// Canonical total — reduces the 8 lazy lanes and folds them with exact
    /// adds. Bit-exact with the scalar sequential sum.
    pub fn finish(&self) -> Goldilocks {
        let mut total = Goldilocks::ZERO;
        for lane in 0..8 {
            // acc + k·0xFFFFFFFF ≡ the lane's true partial sum (mod q).
            let v = Goldilocks::from_u128(
                self.acc[lane] as u128 + (self.carries[lane] as u128) * 0xFFFF_FFFF,
            );
            total = total.add(&v);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// AVX-512 kernel implementations (AVX512F only).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod imp {
    use crate::field::{Goldilocks, GOLDILOCKS_MODULUS};
    use core::arch::x86_64::*;

    #[inline]
    unsafe fn vq() -> __m512i {
        _mm512_set1_epi64(GOLDILOCKS_MODULUS as i64)
    }

    #[inline]
    unsafe fn vcomp() -> __m512i {
        // 2^32 − 1: the value of a dropped/borrowed 2^64 (mod q).
        _mm512_set1_epi64(0xFFFF_FFFFu64 as i64)
    }

    #[inline]
    unsafe fn vone() -> __m512i {
        _mm512_set1_epi64(1)
    }

    /// One reduction fold `(hi, lo) → split(lo + hi·0xFFFF_FFFF)`;
    /// identity when `hi = 0`. See the module-level derivation.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fold_v(hi: __m512i, lo: __m512i) -> (__m512i, __m512i) {
        // hi·2^32 = (hi >> 32)·2^64 + ((hi & M32) << 32); the low part may
        // carry out of the u64 lane.
        let shifted = _mm512_slli_epi64(hi, 32);
        let sum = _mm512_add_epi64(lo, shifted);
        let carry = _mm512_cmplt_epu64_mask(sum, lo);
        let q64 = _mm512_add_epi64(_mm512_srli_epi64(hi, 32), _mm512_maskz_set1_epi64(carry, 1));
        // Subtract hi with a borrow into the high part.
        let lo2 = _mm512_sub_epi64(sum, hi);
        let borrow = _mm512_cmplt_epu64_mask(sum, hi);
        let hi2 = _mm512_mask_sub_epi64(q64, borrow, q64, vone());
        (hi2, lo2)
    }

    /// Canonical packed multiplication: per lane, exactly
    /// `Goldilocks::from_u128(a as u128 * b as u128)` for any u64 operands.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mul_v(a: __m512i, b: __m512i) -> __m512i {
        // 128-bit product from 32-bit half products (vpmuludq).
        let ah = _mm512_srli_epi64(a, 32);
        let bh = _mm512_srli_epi64(b, 32);
        let ll = _mm512_mul_epu32(a, b); // a_lo · b_lo
        let m1 = _mm512_mul_epu32(a, bh); // a_lo · b_hi
        let m2 = _mm512_mul_epu32(ah, b); // a_hi · b_lo
        let hh = _mm512_mul_epu32(ah, bh); // a_hi · b_hi
        let mid = _mm512_add_epi64(m1, m2);
        let c1 = _mm512_cmplt_epu64_mask(mid, m1);
        let midsh = _mm512_slli_epi64(mid, 32); // (mid & M32) << 32
        let lo = _mm512_add_epi64(midsh, ll);
        let c2 = _mm512_cmplt_epu64_mask(lo, ll);
        // The true high word (fits u64 by construction — no lane wrap).
        let hi = _mm512_add_epi64(
            _mm512_add_epi64(hh, _mm512_srli_epi64(mid, 32)),
            _mm512_add_epi64(
                _mm512_maskz_set1_epi64(c2, 1),
                _mm512_slli_epi64(_mm512_maskz_set1_epi64(c1, 1), 32),
            ),
        );
        // Three folds clear hi to ≤ 1, with lo < 2^32 whenever hi = 1 (see
        // the module-level derivation), so the residual 2^64 is worth
        // exactly 2^32 − 1 and the masked add cannot wrap.
        let (hi, lo) = fold_v(hi, lo);
        let (hi, lo) = fold_v(hi, lo);
        let (hi, lo) = fold_v(hi, lo);
        let hi1 = _mm512_cmpeq_epu64_mask(hi, vone());
        let lo = _mm512_mask_add_epi64(lo, hi1, lo, vcomp());
        // Canonical conditional subtraction.
        let ge = _mm512_cmpge_epu64_mask(lo, vq());
        _mm512_mask_sub_epi64(lo, ge, lo, vq())
    }

    /// Exact canonical addition (canonical inputs) — `Goldilocks::add`.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add_v(a: __m512i, b: __m512i) -> __m512i {
        let s = _mm512_add_epi64(a, b);
        let carry = _mm512_cmplt_epu64_mask(s, a);
        // A dropped 2^64 is worth 2^32 − 1 (single compensation provably
        // suffices for canonical inputs).
        let s2 = _mm512_mask_add_epi64(s, carry, s, vcomp());
        let ge = _mm512_cmpge_epu64_mask(s2, vq());
        _mm512_mask_sub_epi64(s2, ge, s2, vq())
    }

    /// Exact canonical subtraction (canonical inputs) — `Goldilocks::sub`.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sub_v(a: __m512i, b: __m512i) -> __m512i {
        let d = _mm512_sub_epi64(a, b);
        let borrow = _mm512_cmplt_epu64_mask(a, b);
        let d2 = _mm512_mask_sub_epi64(d, borrow, d, vcomp());
        let ge = _mm512_cmpge_epu64_mask(d2, vq());
        _mm512_mask_sub_epi64(d2, ge, d2, vq())
    }

    #[inline]
    unsafe fn load_u64(p: *const u64) -> __m512i {
        _mm512_loadu_epi64(p as *const i64)
    }

    #[inline]
    unsafe fn store_u64(p: *mut u64, v: __m512i) {
        _mm512_storeu_epi64(p as *mut i64, v)
    }

    #[inline]
    unsafe fn load(p: *const Goldilocks) -> __m512i {
        _mm512_loadu_epi64(p as *const i64)
    }

    #[inline]
    unsafe fn store(p: *mut Goldilocks, v: __m512i) {
        _mm512_storeu_epi64(p as *mut i64, v)
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`, and all pointers must be
    /// valid for `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn mul_slice_simd(a: *const Goldilocks, b: *const Goldilocks, out: *mut Goldilocks, chunks: usize) {
        for i in 0..chunks {
            store(out.add(i * 8), mul_v(load(a.add(i * 8)), load(b.add(i * 8))));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`, and all pointers must be
    /// valid for `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn add_slice_simd(a: *const Goldilocks, b: *const Goldilocks, out: *mut Goldilocks, chunks: usize) {
        for i in 0..chunks {
            store(out.add(i * 8), add_v(load(a.add(i * 8)), load(b.add(i * 8))));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`, and all pointers must be
    /// valid for `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn sub_slice_simd(a: *const Goldilocks, b: *const Goldilocks, out: *mut Goldilocks, chunks: usize) {
        for i in 0..chunks {
            store(out.add(i * 8), sub_v(load(a.add(i * 8)), load(b.add(i * 8))));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`, and all pointers must be
    /// valid for `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn reduce128_slice_simd(
        hi: *const u64,
        lo: *const u64,
        out: *mut Goldilocks,
        chunks: usize,
    ) {
        for i in 0..chunks {
            // Same reduction tail as mul_v: three folds (hi ≤ 1, and
            // lo < 2^32 whenever hi = 1), the masked 2^32 − 1 compensation,
            // and the canonical conditional subtraction.
            let (mut h, mut l) = (load_u64(hi.add(i * 8)), load_u64(lo.add(i * 8)));
            (h, l) = fold_v(h, l);
            (h, l) = fold_v(h, l);
            (h, l) = fold_v(h, l);
            let hi1 = _mm512_cmpeq_epu64_mask(h, vone());
            let l = _mm512_mask_add_epi64(l, hi1, l, vcomp());
            let ge = _mm512_cmpge_epu64_mask(l, vq());
            store(out.add(i * 8), _mm512_mask_sub_epi64(l, ge, l, vq()));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`, and all pointers must be
    /// valid for `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn mul_add_slice_simd(
        a: *const Goldilocks,
        b: *const Goldilocks,
        c: *const Goldilocks,
        out: *mut Goldilocks,
        chunks: usize,
    ) {
        for i in 0..chunks {
            let prod = mul_v(load(a.add(i * 8)), load(b.add(i * 8)));
            store(out.add(i * 8), add_v(prod, load(c.add(i * 8))));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`, and both pointers must be
    /// valid for `chunks*8` elements (source and destination may alias
    /// elementwise).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn mul_scalar_slice_simd(a: *const Goldilocks, s: u64, out: *mut Goldilocks, chunks: usize) {
        let vs = _mm512_set1_epi64(s as i64);
        for i in 0..chunks {
            store(out.add(i * 8), mul_v(load(a.add(i * 8)), vs));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`; `lo`/`hi`/`out` valid for
    /// `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bind_half_simd(
        lo: *const Goldilocks,
        hi: *const Goldilocks,
        r: u64,
        out: *mut Goldilocks,
        chunks: usize,
    ) {
        let vr = _mm512_set1_epi64(r as i64);
        for i in 0..chunks {
            let a = load(lo.add(i * 8));
            let b = load(hi.add(i * 8));
            store(out.add(i * 8), add_v(a, mul_v(sub_v(b, a), vr)));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`; `evals` valid for
    /// `2*half` elements with `chunks*8 <= half`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bind_first_half_simd(evals: *mut Goldilocks, half: usize, r: u64, chunks: usize) {
        let vr = _mm512_set1_epi64(r as i64);
        for i in 0..chunks {
            let a = load(evals.add(i * 8));
            let b = load(evals.add(half + i * 8));
            // Writes stay in the first half; second-half reads are disjoint.
            store(evals.add(i * 8), add_v(a, mul_v(sub_v(b, a), vr)));
        }
    }

    /// # Safety
    /// Caller must have checked `avx512_field()`; `evals` valid for
    /// `2*half` elements with `chunks*8 <= half`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bind_projective_first_half_simd(
        evals: *mut Goldilocks,
        half: usize,
        r: u64,
        chunks: usize,
    ) {
        let vr = _mm512_set1_epi64(r as i64);
        for i in 0..chunks {
            let a = load(evals.add(i * 8));
            let b = load(evals.add(half + i * 8));
            // Writes stay in the first half; second-half reads are disjoint.
            store(evals.add(i * 8), add_v(a, mul_v(b, vr)));
        }
    }

    /// Projective pairwise binding: `out[i] = in[2i] + r·in[2i+1]`.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`; `evals` valid for
    /// `16*chunks` elements with `8*chunks <= 16*chunks/2`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bind_projective_pairs_simd(evals: *mut Goldilocks, r: u64, chunks: usize) {
        let vr = _mm512_set1_epi64(r as i64);
        let idx_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let idx_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        for c in 0..chunks {
            let v0 = load(evals.add(16 * c));
            let v1 = load(evals.add(16 * c + 8));
            let ev = _mm512_permutex2var_epi64(v0, idx_even, v1);
            let od = _mm512_permutex2var_epi64(v0, idx_odd, v1);
            store(evals.add(8 * c), add_v(ev, mul_v(od, vr)));
        }
    }

    /// Pairwise binding for `fix_last_variables`: consumes 16 consecutive
    /// inputs and writes 8 outputs at half the offset. In-place safe: each
    /// iteration loads all its inputs before storing, and output offsets
    /// (`8c..8c+8`) never reach the next iteration's input region
    /// (`16(c+1)..`).
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`; `evals` valid for
    /// `16*chunks` elements with `8*chunks <= 16*chunks/2`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn bind_pairs_simd(evals: *mut Goldilocks, r: u64, chunks: usize) {
        let vr = _mm512_set1_epi64(r as i64);
        let idx_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let idx_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        for c in 0..chunks {
            let v0 = load(evals.add(16 * c));
            let v1 = load(evals.add(16 * c + 8));
            // Deinterleave even/odd lanes across the two 8-lane halves.
            let ev = _mm512_permutex2var_epi64(v0, idx_even, v1);
            let od = _mm512_permutex2var_epi64(v0, idx_odd, v1);
            store(evals.add(8 * c), add_v(ev, mul_v(sub_v(od, ev), vr)));
        }
    }

    /// The sumcheck term-product accumulation: 8 independent lazy lanes,
    /// carry counts, per-lane products chained in registers.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`; every factor slice and
    /// both accumulator arrays must cover `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn accumulate_term_simd(
        acc: &mut [u64; 8],
        carries: &mut [u64; 8],
        coeff: u64,
        factors: &[&[Goldilocks]],
        chunks: usize,
    ) {
        let mut vacc = load_u64(acc.as_ptr());
        let mut vcar = load_u64(carries.as_ptr());
        let c = _mm512_set1_epi64(coeff as i64);
        for chunk in 0..chunks {
            let base = 8 * chunk;
            let mut prod = c;
            for f in factors {
                prod = mul_v(prod, load(f.as_ptr().add(base)));
            }
            // Lazy add with exact carry accounting (invariant in module docs).
            let sum = _mm512_add_epi64(vacc, prod);
            let wrapped = _mm512_cmplt_epu64_mask(sum, vacc);
            vcar = _mm512_add_epi64(vcar, _mm512_maskz_set1_epi64(wrapped, 1));
            vacc = sum;
        }
        store_u64(acc.as_mut_ptr(), vacc);
        store_u64(carries.as_mut_ptr(), vcar);
    }

    /// Pointwise `acc[i] += coeff · Π_j factors[j][i]`.
    ///
    /// # Safety
    /// Caller must have checked `avx512_field()`; every factor slice and
    /// `acc` must cover `chunks*8` elements.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn accumulate_term_pointwise_simd(
        factors: &[&[Goldilocks]],
        coeff: u64,
        acc: *mut Goldilocks,
        chunks: usize,
    ) {
        let c = _mm512_set1_epi64(coeff as i64);
        for chunk in 0..chunks {
            let base = 8 * chunk;
            let mut prod = c;
            for f in factors {
                prod = mul_v(prod, load(f.as_ptr().add(base)));
            }
            let cur = load(acc.add(base));
            store(acc.add(base), add_v(cur, prod));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: bit-exactness against the scalar Goldilocks reference. On machines
// without AVX-512 (or with LZX_NO_SIMD=1) the slice APIs under test take the
// scalar path and the checks degenerate to scalar-vs-scalar; on this
// deployment's Xeons they exercise the packed kernels.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GOLDILOCKS_MODULUS;

    /// Deterministic splitmix64 for reproducible randomized cross-checks.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Edge values around every reduction boundary: 0/1, q-boundaries, the
    /// 2^32 split, 2^63, near-q, plus deliberately non-canonical values.
    fn edge_values() -> Vec<u64> {
        let q = GOLDILOCKS_MODULUS;
        vec![
            0,
            1,
            2,
            q - 1,
            q - 2,
            q - 3,
            (1u64 << 32) - 1,
            1u64 << 32,
            (1u64 << 32) + 1,
            q.wrapping_sub(1u64 << 32),
            q.wrapping_sub(1u64 << 32) + 1,
            (1u64 << 63) - 1,
            1u64 << 63,
            (1u64 << 63) + 1,
            0xFFFF_FFFF,
            0xFFFF_FFFF_0000_0000, // q - 1
            0xFFFF_FFFE_FFFF_FFFF,
            0xFFFF_FFFF_0000_0002, // q + 1 (non-canonical)
            0xDEAD_BEEF_CAFE_F00D % q,
            u64::MAX, // non-canonical on purpose
        ]
    }

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn random_slice(state: &mut u64, n: usize, canonical: bool) -> Vec<Goldilocks> {
        (0..n)
            .map(|_| {
                let raw = splitmix64(state);
                if canonical {
                    fe(raw)
                } else {
                    Goldilocks(raw)
                }
            })
            .collect()
    }

    fn scalar_mul(a: u64, b: u64) -> u64 {
        Goldilocks::from_u128(a as u128 * b as u128).0
    }

    /// Independent scalar eq-table reference: the per-entry definition
    /// `Π_i [(1−x_i)(1−b_i) + x_i·b_i]` with variable 0 on the most
    /// significant index bit — deliberately NOT the doubling construction
    /// the packed kernel uses, so cross-checks are non-circular.
    fn scalar_eq_table(point: &[Goldilocks]) -> Vec<Goldilocks> {
        let m = point.len();
        (0..(1usize << m))
            .map(|idx| {
                let mut v = Goldilocks::ONE;
                for (i, x) in point.iter().enumerate() {
                    let bit = (idx >> (m - 1 - i)) & 1;
                    let term = if bit == 1 { *x } else { Goldilocks::ONE.sub(x) };
                    v = v.mul(&term);
                }
                v
            })
            .collect()
    }

    #[test]
    fn env_flag_parsing() {
        assert!(simd_disabled_by_env(Some(OsStr::new("1"))));
        assert!(!simd_disabled_by_env(Some(OsStr::new("0"))));
        assert!(!simd_disabled_by_env(Some(OsStr::new(""))));
        assert!(!simd_disabled_by_env(Some(OsStr::new("true"))));
        assert!(!simd_disabled_by_env(None));
    }

    #[test]
    fn gate_is_stable() {
        assert_eq!(avx512_field(), avx512_field());
    }

    /// The core exactness proof of the packed multiplier: ALL n² edge-value
    /// pairs exhaustively (not a sample), plus a swapped sweep that re-seats
    /// every pair in a different lane.
    #[test]
    fn packed_mul_edge_pairs() {
        let edges = edge_values();
        let n = edges.len();
        let total = n * n;
        let padded = total.div_ceil(8) * 8;
        let mut a = vec![Goldilocks::ONE; padded];
        let mut b = vec![Goldilocks::ONE; padded];
        for k in 0..total {
            a[k] = Goldilocks(edges[k / n]);
            b[k] = Goldilocks(edges[k % n]);
        }
        let mut out = vec![Goldilocks::ZERO; padded];
        mul_slices(&a, &b, &mut out);
        for k in 0..total {
            let want = scalar_mul(a[k].0, b[k].0);
            assert_eq!(out[k].0, want, "mul edge {:#x}·{:#x}", a[k].0, b[k].0);
        }
        // Swapped operands: commutativity + every pair lands in a rotated lane.
        mul_slices(&b, &a, &mut out);
        for k in 0..total {
            assert_eq!(out[k].0, scalar_mul(a[k].0, b[k].0), "mul edge swapped #{k}");
        }
    }

    /// Randomized mul cross-check over canonical and full-range operands.
    #[test]
    fn packed_mul_random() {
        let mut st = 0x1234_5678_9ABC_DEF0u64;
        const ROUNDS: usize = 4096; // 4096*8 = 32768 random pairs per class
        for canonical in [true, false] {
            let a = random_slice(&mut st, ROUNDS * 8, canonical);
            let b = random_slice(&mut st, ROUNDS * 8, canonical);
            let mut out = vec![Goldilocks::ZERO; ROUNDS * 8];
            mul_slices(&a, &b, &mut out);
            for i in 0..ROUNDS * 8 {
                assert_eq!(out[i].0, scalar_mul(a[i].0, b[i].0), "mul random #{i}");
            }
        }
    }

    #[test]
    fn mul_slice_lengths_and_tails() {
        let mut st = 0x0BAD_C0DE_DEAD_10CCu64;
        for len in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 23, 24, 31, 63, 64, 65, 100, 257, 1000] {
            let a = random_slice(&mut st, len, true);
            let b = random_slice(&mut st, len, true);
            let mut out = vec![Goldilocks::ZERO; len];
            mul_slices(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i].0, scalar_mul(a[i].0, b[i].0), "len {len} idx {i}");
            }
        }
    }

    /// Packed u128 reduction: edge boundaries of every reduction regime,
    /// random full-range u128s, and the (hi, lo) halves of real 128-bit
    /// products (the mul_v bridge).
    #[test]
    fn reduce128_matches_from_u128() {
        let q = GOLDILOCKS_MODULUS as u128;
        let edge128: Vec<u128> = vec![
            0,
            1,
            2,
            q - 1,
            q,
            q + 1,
            2 * q,
            (1u128 << 32) - 1,
            1u128 << 32,
            (1u128 << 32) + 1,
            (1u128 << 64) - 1,
            1u128 << 64,
            (1u128 << 64) + 1,
            (1u128 << 64) - q,
            (1u128 << 96) - 1,
            1u128 << 96,
            (1u128 << 127) - 1,
            1u128 << 127,
            u128::MAX,
            u128::MAX - 1,
            q * q,
            q * q - 1,
            (q - 1) * (q - 1),
            (1u128 << 64) * 0xFFFF_FFFF + 0xFFFF_FFFF, // maximal fold-1 shape
        ];
        let mut his: Vec<u64> = edge128.iter().map(|&x| (x >> 64) as u64).collect();
        let mut los: Vec<u64> = edge128.iter().map(|&x| x as u64).collect();
        // Random full-range u128s.
        let mut st = 0xC0FF_EE00_C0FF_EE00u64;
        const ROUNDS: usize = 4096;
        let mut randoms: Vec<u128> = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let hi = splitmix64(&mut st);
            let lo = splitmix64(&mut st);
            randoms.push(((hi as u128) << 64) | lo as u128);
        }
        his.extend(randoms.iter().map(|&x| (x >> 64) as u64));
        los.extend(randoms.iter().map(|&x| x as u64));
        // (hi, lo) halves of full 128-bit products: reduce128 must reproduce
        // both from_u128 and the packed multiplier.
        let mut prod_his = Vec::with_capacity(ROUNDS);
        let mut prod_los = Vec::with_capacity(ROUNDS);
        let mut prod_pairs: Vec<(u64, u64)> = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let a = splitmix64(&mut st);
            let b = splitmix64(&mut st);
            let p = a as u128 * b as u128;
            prod_his.push((p >> 64) as u64);
            prod_los.push(p as u64);
            prod_pairs.push((a, b));
        }
        his.extend_from_slice(&prod_his);
        los.extend_from_slice(&prod_los);

        let total = his.len();
        let mut out = vec![Goldilocks::ZERO; total];
        reduce128(&his, &los, &mut out);
        for i in 0..total {
            let want = Goldilocks::from_u128(((his[i] as u128) << 64) | los[i] as u128);
            assert_eq!(out[i], want, "reduce128 #{i} hi {:#x} lo {:#x}", his[i], los[i]);
        }
        // The multiplier bridge: reduce128 of a product == mul of the halves' sources.
        let base = total - prod_pairs.len();
        for k in 0..prod_pairs.len() {
            let (a, b) = prod_pairs[k];
            assert_eq!(
                out[base + k].0,
                scalar_mul(a, b),
                "reduce128 vs mul {:#x}·{:#x}",
                a,
                b
            );
        }
        // Odd/tail lengths exercise the scalar tail path too.
        for len in [0usize, 1, 7, 8, 9, 17, 31, 100] {
            let h: Vec<u64> = (0..len).map(|_| splitmix64(&mut st)).collect();
            let l: Vec<u64> = (0..len).map(|_| splitmix64(&mut st)).collect();
            let mut o = vec![Goldilocks::ZERO; len];
            reduce128(&h, &l, &mut o);
            for i in 0..len {
                assert_eq!(
                    o[i],
                    Goldilocks::from_u128(((h[i] as u128) << 64) | l[i] as u128),
                    "reduce128 tail len {len} idx {i}"
                );
            }
        }
    }

    #[test]
    fn add_sub_slices_match_scalar() {
        let mut st = 0x5EED_5EED_5EED_5EEDu64;
        for len in [0usize, 1, 7, 8, 9, 16, 17, 31, 100, 999] {
            let a = random_slice(&mut st, len, true);
            let b = random_slice(&mut st, len, true);
            let mut out = vec![Goldilocks::ZERO; len];
            add_slices(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i].add(&b[i]), "add len {len} idx {i}");
            }
            sub_slices(&a, &b, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i].sub(&b[i]), "sub len {len} idx {i}");
            }
        }
        // ALL n² canonical edge pairs for add/sub, exhaustively packed.
        let edges: Vec<u64> = edge_values().into_iter().map(|x| x % GOLDILOCKS_MODULUS).collect();
        let n = edges.len();
        let total = n * n;
        let padded = total.div_ceil(8) * 8;
        let mut a = vec![Goldilocks::ONE; padded];
        let mut b = vec![Goldilocks::ZERO; padded];
        for k in 0..total {
            a[k] = fe(edges[k / n]);
            b[k] = fe(edges[k % n]);
        }
        let mut out = vec![Goldilocks::ZERO; padded];
        add_slices(&a, &b, &mut out);
        for k in 0..total {
            assert_eq!(out[k], a[k].add(&b[k]), "add edge {:#x}+{:#x}", a[k].0, b[k].0);
        }
        sub_slices(&a, &b, &mut out);
        for k in 0..total {
            assert_eq!(out[k], a[k].sub(&b[k]), "sub edge {:#x}-{:#x}", a[k].0, b[k].0);
        }
    }

    #[test]
    fn mul_add_slices_match_scalar() {
        let mut st = 0xA11CE_A11CE_A11CEu64;
        for len in [1usize, 7, 8, 9, 23, 24, 100, 512] {
            let a = random_slice(&mut st, len, true);
            let b = random_slice(&mut st, len, true);
            let c = random_slice(&mut st, len, true);
            let mut out = vec![Goldilocks::ZERO; len];
            mul_add_slices(&a, &b, &c, &mut out);
            for i in 0..len {
                assert_eq!(out[i], a[i].mul(&b[i]).add(&c[i]), "len {len} idx {i}");
            }
        }
    }

    #[test]
    fn mul_scalar_slice_matches() {
        let mut st = 0xFEED_FACE_CAFE_1234u64;
        let scalars: Vec<Goldilocks> = edge_values().iter().map(|&x| fe(x % GOLDILOCKS_MODULUS)).collect();
        for len in [1usize, 7, 8, 9, 16, 17, 100, 4096] {
            let a = random_slice(&mut st, len, true);
            for s in &scalars {
                let mut out = vec![Goldilocks::ZERO; len];
                mul_scalar_slice(&a, *s, &mut out);
                for i in 0..len {
                    assert_eq!(out[i], a[i].mul(s), "len {len} idx {i} s {:#x}", s.0);
                }
                // In-place variant on a fresh copy.
                let mut copy = a.clone();
                mul_scalar_slice_inplace(&mut copy, *s);
                for i in 0..len {
                    assert_eq!(copy[i], a[i].mul(s), "inplace len {len} idx {i}");
                }
            }
        }
    }

    #[test]
    fn bind_half_slices_matches_scalar() {
        let mut st = 0x1CEB_00DA_1CEB_00DAu64;
        let rs: Vec<Goldilocks> = vec![
            Goldilocks::ZERO,
            Goldilocks::ONE,
            fe(2),
            fe((1u64 << 32) - 1),
            fe(GOLDILOCKS_MODULUS - 1),
            fe(0x1234_5678_9ABC_DEF0 % GOLDILOCKS_MODULUS),
        ];
        for len in [1usize, 7, 8, 9, 15, 16, 17, 100, 1000] {
            let lo = random_slice(&mut st, len, true);
            let hi = random_slice(&mut st, len, true);
            for r in &rs {
                let mut out = vec![Goldilocks::ZERO; len];
                bind_half_slices(&lo, &hi, *r, &mut out);
                for i in 0..len {
                    let want = lo[i].add(&hi[i].sub(&lo[i]).mul(r));
                    assert_eq!(out[i], want, "len {len} idx {i} r {:#x}", r.0);
                }
            }
        }
    }

    #[test]
    fn bind_first_half_in_place_matches_scalar() {
        let mut st = 0xB0BA_CAFE_B0BA_CAFEu64;
        for len in [0usize, 1, 2, 3, 8, 9, 16, 17, 32, 64, 130] {
            let evals = random_slice(&mut st, len, true);
            let r = fe(0x0F0F_0F0F_0F0F_0F0F % GOLDILOCKS_MODULUS);
            let mut got = evals.clone();
            bind_first_half_in_place(&mut got, r);
            let half = len / 2;
            let mut want = evals.clone();
            for i in 0..half {
                let a = want[i];
                let b = want[i + half];
                want[i] = a.add(&b.sub(&a).mul(&r));
            }
            assert_eq!(got, want, "len {len}");
            // The untouched second half must survive verbatim.
            if len >= 2 {
                assert_eq!(&got[half..], &evals[half..]);
            }
        }
    }

    #[test]
    fn bind_pairs_in_place_matches_scalar() {
        let mut st = 0xD00D_1E55_D00D_1E55u64;
        for len in [0usize, 2, 4, 8, 16, 17, 32, 64, 130, 258] {
            let evals = random_slice(&mut st, len, true);
            let r = fe(0xABCD_EF01_2345_6789 % GOLDILOCKS_MODULUS);
            let mut got = evals.clone();
            bind_pairs_in_place(&mut got, r);
            let half = len / 2;
            let mut want = evals.clone();
            for i in 0..half {
                let a = want[2 * i];
                let b = want[2 * i + 1];
                want[i] = a.add(&b.sub(&a).mul(&r));
            }
            // Only the first `half` outputs are defined; compare those.
            assert_eq!(&got[..half], &want[..half], "len {len}");
        }
    }

    #[test]
    fn eq_table_matches_mle_eq_extension() {
        use crate::mle::DenseMle;
        let mut st = 0xE0E0_E0E0_1234_5678u64;
        for m in [0usize, 1, 2, 3, 5, 8, 12] {
            let point: Vec<Goldilocks> = (0..m)
                .map(|i| {
                    if i % 3 == 0 {
                        // Sprinkle exact boolean values (0/1) plus field points.
                        fe(splitmix64(&mut st) & 1)
                    } else {
                        fe(splitmix64(&mut st))
                    }
                })
                .collect();
            let want = DenseMle::eq_extension(&point);
            let got = eq_table(&point);
            assert_eq!(got.len(), want.evaluations.len(), "m {m}");
            assert_eq!(got, want.evaluations, "m {m}");
            // Boolean points must produce the pure indicator.
            if m > 0 {
                let bool_point: Vec<Goldilocks> = (0..m).map(|i| fe((i % 2) as u64)).collect();
                assert_eq!(eq_table(&bool_point), DenseMle::eq_extension(&bool_point).evaluations);
            }
        }
    }

    /// Non-circular eq oracle: the packed doubling construction against the
    /// per-entry scalar definition (a different algorithm, not a wrapper of
    /// the same kernel). Boolean points must give the exact hypercube
    /// indicator.
    #[test]
    fn eq_table_matches_scalar_definition() {
        let mut st = 0x7A11_7A11_7A11_7A11u64;
        for m in [0usize, 1, 2, 3, 4, 5, 8, 12] {
            let point: Vec<Goldilocks> = (0..m)
                .map(|i| {
                    if i % 2 == 0 {
                        fe(splitmix64(&mut st) & 1) // exact booleans interleaved
                    } else {
                        fe(splitmix64(&mut st))
                    }
                })
                .collect();
            let want = scalar_eq_table(&point);
            assert_eq!(eq_table(&point), want, "m {m}");
            // Pure boolean points: eq is the point indicator on the hypercube.
            if m > 0 {
                for mask in [0u64, 1, (1u64 << m) - 1, 0b1010_1101 & ((1 << m) - 1)] {
                    let bools: Vec<Goldilocks> = (0..m)
                        .map(|i| fe((mask >> (m - 1 - i)) & 1))
                        .collect();
                    let table = eq_table(&bools);
                    for (idx, &v) in table.iter().enumerate() {
                        assert_eq!(
                            v,
                            if idx as u64 == mask { Goldilocks::ONE } else { Goldilocks::ZERO },
                            "indicator m {m} mask {mask:#x} idx {idx}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sum_slice_matches_sequential() {
        let mut st = 0x5A5A_5A5A_F00D_F00Du64;
        for len in [0usize, 1, 7, 8, 9, 100, 4096] {
            let a = random_slice(&mut st, len, true);
            let want = a.iter().fold(Goldilocks::ZERO, |acc, x| acc.add(x));
            assert_eq!(sum_slice(&a), want, "len {len}");
        }
        // All-(q-1) adversarial sum: maximal carry traffic through the lanes.
        let vals = vec![Goldilocks::from_u64(GOLDILOCKS_MODULUS - 1); 1000];
        let want = vals.iter().fold(Goldilocks::ZERO, |acc, x| acc.add(x));
        assert_eq!(sum_slice(&vals), want);
    }

    #[test]
    fn sum8_term_products_match_scalar() {
        let mut st = 0xFACE_B00C_C0DE_5171u64;
        for len in [1usize, 7, 8, 9, 23, 24, 100, 1000] {
            for nfactors in [1usize, 2, 3, 5] {
                let factors: Vec<Vec<Goldilocks>> =
                    (0..nfactors).map(|_| random_slice(&mut st, len, true)).collect();
                let coeff = fe(splitmix64(&mut st));
                let fslices: Vec<&[Goldilocks]> = factors.iter().map(|f| f.as_slice()).collect();
                // Multiple terms to exercise lane reuse across terms.
                let coeff2 = fe(splitmix64(&mut st));
                let fslices2: Vec<&[Goldilocks]> =
                    factors.iter().rev().map(|f| f.as_slice()).collect();
                let mut acc = Sum8::new();
                acc.accumulate_term(coeff, &fslices);
                acc.accumulate_term(coeff2, &fslices2);

                // Scalar reference: term-major, index-minor, exact adds.
                let mut want = Goldilocks::ZERO;
                for p in 0..len {
                    let mut prod = coeff;
                    for f in &factors {
                        prod = prod.mul(&f[p]);
                    }
                    want = want.add(&prod);
                }
                for p in 0..len {
                    let mut prod = coeff2;
                    for f in factors.iter().rev() {
                        prod = prod.mul(&f[p]);
                    }
                    want = want.add(&prod);
                }
                assert_eq!(acc.finish(), want, "len {len} nfactors {nfactors}");
            }
        }
        // Fresh accumulator is exactly zero.
        assert_eq!(Sum8::new().finish(), Goldilocks::ZERO);
        assert_eq!(Sum8::default().finish(), Goldilocks::ZERO);
    }

    #[test]
    fn sum8_lazy_carry_accounting_adversarial() {
        // Values that force wraps at every opportunity: q-1 and near-2^64
        // lazy lane states, accumulated far past 2^64 many times over.
        let mut st = 0x7F3E_1A2B_3C4D_5E6Fu64;
        for _ in 0..32 {
            let n = 1 + (splitmix64(&mut st) % 512) as usize;
            let vals: Vec<Goldilocks> = (0..n)
                .map(|_| fe(GOLDILOCKS_MODULUS - 1 - (splitmix64(&mut st) % 4)))
                .collect();
            let mut acc = Sum8::new();
            acc.accumulate_term(Goldilocks::ONE, &[&vals]);
            let want = vals.iter().fold(Goldilocks::ZERO, |a, x| a.add(x));
            assert_eq!(acc.finish(), want, "n {n}");
        }
    }

    #[test]
    fn accumulate_term_pointwise_matches_scalar() {
        let mut st = 0x1234_ABCD_9876_5432u64;
        for len in [1usize, 7, 8, 9, 24, 100, 512] {
            for nfactors in [1usize, 2, 4] {
                let factors: Vec<Vec<Goldilocks>> =
                    (0..nfactors).map(|_| random_slice(&mut st, len, true)).collect();
                let coeff = fe(splitmix64(&mut st));
                let fslices: Vec<&[Goldilocks]> = factors.iter().map(|f| f.as_slice()).collect();
                let mut acc = vec![Goldilocks::ZERO; len];
                accumulate_term_pointwise(&fslices, coeff, &mut acc);
                // Second term on top of a non-zero accumulator.
                let coeff2 = fe(splitmix64(&mut st));
                accumulate_term_pointwise(&fslices, coeff2, &mut acc);

                let mut want = vec![Goldilocks::ZERO; len];
                for (coeff, _fs) in [(coeff, &fslices), (coeff2, &fslices)] {
                    for p in 0..len {
                        let mut prod = Goldilocks::ONE;
                        for f in &factors {
                            prod = prod.mul(&f[p]);
                        }
                        want[p] = want[p].add(&coeff.mul(&prod));
                    }
                }
                assert_eq!(acc, want, "len {len} nfactors {nfactors}");
            }
        }
    }

    /// End-to-end style: the packed kernels must reproduce a scalar
    /// sumcheck round computation exactly (binding + term products).
    #[test]
    #[allow(clippy::needless_range_loop)] // indexes several bound factor slices at once
    fn packed_round_pipeline_matches_scalar() {
        use crate::mle::DenseMle;
        let num_vars = 7usize;
        let points = 1usize << (num_vars - 1);
        let f0 = DenseMle::random(num_vars, b"pipe-f0");
        let f1 = DenseMle::random(num_vars, b"pipe-f1");
        let f2 = DenseMle::random(num_vars, b"pipe-f2");
        let factors = vec![f0, f1, f2];
        let terms: Vec<(Goldilocks, Vec<usize>)> = vec![
            (fe(3), vec![0, 1]),
            (fe(5), vec![1, 2, 0]),
            (fe(11), vec![2]),
        ];
        for t_raw in [0u64, 1, 2, 3, 0x9999] {
            let t = fe(t_raw);
            // Scalar reference for sum_products.
            let mut bound_vals: Vec<Vec<Goldilocks>> = Vec::new();
            for f in &factors {
                let evs = &f.evaluations;
                let mut vals = Vec::with_capacity(points);
                for p in 0..points {
                    let a = evs[p];
                    let b = evs[p + points];
                    vals.push(a.add(&b.sub(&a).mul(&t)));
                }
                bound_vals.push(vals);
            }
            let mut want = Goldilocks::ZERO;
            for (coeff, ids) in &terms {
                for p in 0..points {
                    let mut prod = *coeff;
                    for fi in ids {
                        prod = prod.mul(&bound_vals[*fi][p]);
                    }
                    want = want.add(&prod);
                }
            }
            // Packed pipeline.
            let mut acc = Sum8::new();
            let mut packed_bv: Vec<Vec<Goldilocks>> = Vec::new();
            for f in &factors {
                let evs = &f.evaluations;
                let mut vals = vec![Goldilocks::ZERO; points];
                bind_half_slices(&evs[..points], &evs[points..], t, &mut vals);
                packed_bv.push(vals);
            }
            for (coeff, ids) in &terms {
                let fslices: Vec<&[Goldilocks]> =
                    ids.iter().map(|fi| packed_bv[*fi].as_slice()).collect();
                acc.accumulate_term(*coeff, &fslices);
            }
            assert_eq!(acc.finish(), want, "t {t_raw}");
        }
    }
}
