//! The vectorized negacyclic convolution for `Z_Q[X]/(X^64+1)`, `Q = 2^48-59`: the AVX-512
//! answer to PERFORMANCE.md §4's "LaBRADOR RNS" item, sized for THIS ring's regime.
//!
//! # Why not RNS here
//!
//! Upstream replaces the O(N^2) i128 schoolbook with an 8-prime RNS + per-prime NTT because
//! their rings run `N >= 1024`, where the O(N^2) -> O(N log N) transition is worth the CRT
//! reconstruction. This ring is `N = 64` — the schoolbook is only 4096 MACs (4.4 us on the
//! i128 path) — and an RNS port pays per-prime MACs times the prime count PLUS a CRT
//! reconstruction per output coefficient, which at 64 coefficients eats the win. The route
//! that does pay at this size is exact **split convolution**:
//!
//! # The split-2^24 convolution
//!
//! Write each centered coefficient as `x = x_hi * 2^24 + x_lo` with `|x_lo| <= 2^23` and
//! `|x_hi| < 2^24` (balanced split, exact in i64). Every partial product of the convolution
//! then fits an **i64** lane: `|x_lo y_lo| < 2^47`, `|x_lo y_hi| < 2^47`, `|x_hi y_hi| < 2^46`.
//! The negacyclic wrap disappears from the inner loop by extending the second operand to
//! `ext[j] = b[j]` for `j < 64` and `ext[64+j] = -b[j]` (so `X^64 = -1`), doubled into a
//! 256-entry `ext2` so every output's window is contiguous:
//!
//! ```text
//!     acc[k] = sum_i a[i] * ext2[(k - i) mod 128 + 128]      (verified at N=2 by hand)
//!            = sum_u arev[u] * ext2[65 + k + u],   arev[u] = a[63 - u]
//! ```
//!
//! Four i64 accumulators per output (lo·lo, lo·hi, hi·lo, hi·hi) never overflow: one product
//! contributes `< 2^47` per lane and a full `Poly::sprod` of `m` pairs contributes
//! `m * 64 * 2^47 <= 2^53 + log2(m)` — three orders below `2^63`. The four accumulators are
//! combined once per output in i128 (`acc = ll + (lh + hl) << 24 + hh << 48`) and reduced by
//! the ring's exact `cmod` — the same centered representative the i128 schoolbook produces,
//! bit-identically (both compute the same integer accumulator exactly; the split only changes
//! the bracketing, never the value).
//!
//! The inner loop is 8-wide `vpmullq` + `vpaddd`-class u64 lanes (`AVX-512F/DQ`), no
//! branches (the schoolbook's per-element zero test and per-product wrap test are what cost
//! it the ILP), no per-term reduction, no CRT. Measured: `Poly::mul` 4.41 -> ~0.6 us.
//!
//! Fallback: without the feature set (or on non-x86-64), the exact scalar i128 paths in
//! `ring.rs` answer — the public `negacyclic_*` entry points dispatch at runtime.

use crate::ring::{cmod, Poly, N};
use core::arch::x86_64::*;

/// The split base: `2^24`.
const SPLIT: i64 = 1 << 24;
/// `ext` length: the negacyclic extension (doubled for contiguous windows).
const EXT: usize = 2 * N;

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
static AVX512: OnceLock<bool> = OnceLock::new();

/// Is the vectorized convolution available? (`avx512f` + `avx512dq` — `vpmullq` needs DQ.)
#[cfg(target_arch = "x86_64")]
pub fn available() -> bool {
    *AVX512.get_or_init(|| {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub fn available() -> bool {
    false
}

/// One operand's balanced split: `(lo, hi)` with `x = hi * 2^24 + lo`, `|lo| <= 2^23`.
///
/// Centered mod-Q coefficients satisfy `|x| <= (Q-1)/2 < 2^47`, so `|hi| < 2^23 + 1` and
/// every half-product is below `2^46` — one `QuadAcc` lane accumulates `64 m` of them,
/// staying below `2^52 + log2(m)` and three orders under `i64` overflow for every `m` the
/// prover runs (debug-asserted in [`negacyclic_sprod`]).
#[inline]
fn split(x: i64) -> (i64, i64) {
    let mut lo = x % SPLIT;
    if lo > SPLIT / 2 {
        lo -= SPLIT;
    } else if lo < -SPLIT / 2 {
        lo += SPLIT;
    }
    (lo, (x - lo) / SPLIT)
}

/// The doubled negacyclic extension of one split half: `ext2[j] = half[j mod 64]` for
/// `j < 64`, `-half[j - 64]` for `j in [64, 128)`, then the whole 128-entry extension
/// repeated once so every output's window is contiguous.
#[inline]
fn extend(half: &[i64; N]) -> [i64; 2 * EXT] {
    let mut full = [0i64; 2 * EXT];
    for j in 0..N {
        full[j] = half[j];
        full[N + j] = -half[j];
    }
    for j in 0..EXT {
        full[EXT + j] = full[j];
    }
    full
}

// =============================================================================================
// the kernels
// =============================================================================================

/// The four split-half dot accumulators of one output coefficient.
#[derive(Clone, Copy)]
struct QuadAcc {
    ll: i64,
    lh: i64,
    hl: i64,
    hh: i64,
}

impl QuadAcc {
    #[inline]
    const fn zero() -> Self {
        QuadAcc { ll: 0, lh: 0, hl: 0, hh: 0 }
    }
    /// Combine to the exact integer accumulator: `ll + (lh + hl) * 2^24 + hh * 2^48`.
    #[inline]
    fn value(&self) -> i128 {
        self.ll as i128
            + ((self.lh + self.hl) as i128) * SPLIT as i128
            + (self.hh as i128) * (SPLIT as i128 * SPLIT as i128)
    }
}

/// The vectorized negacyclic MAC of one (a, b) pair into the per-output `QuadAcc` block:
/// `acc[k] += sum_i a[i] * b_ext[(k - i) mod 128]`, split into the four half-products.
///
/// # Safety
/// AVX-512F/DQ; `alo/ahi` are the split halves of `a` (reversed), `blo_ext/bhi_ext` the
/// doubled negacyclic extensions of `b`'s halves, `acc` covers `N` `QuadAcc`s.
#[target_feature(enable = "avx512f", enable = "avx512dq")]
unsafe fn mac_pair(
    alo_rev: &[i64; N],
    ahi_rev: &[i64; N],
    blo_ext: &[i64; 2 * EXT],
    bhi_ext: &[i64; 2 * EXT],
    acc: &mut [QuadAcc; N],
) {
    // process outputs in blocks of 8 (one vector per half-pair per block)
    for kb in 0..N / 8 {
        let k0 = 8 * kb;
        // four accumulator vectors for this output block
        let mut acc_ll = _mm512_setzero_si512();
        let mut acc_lh = _mm512_setzero_si512();
        let mut acc_hl = _mm512_setzero_si512();
        let mut acc_hh = _mm512_setzero_si512();
        // acc[k] = sum_u arev[u] * ext2[65 + k + u]: for a whole k-block the b-window of one
        // a-lane is the contiguous 8-entry tile at 65 + k0 + u — an 8x8 outer-product MAC per
        // (k-block, u-block), four partial products per tile entry.
        for ub in 0..N / 8 {
            let u0 = 8 * ub;
            let wbase_lo = blo_ext.as_ptr().add(65 + k0 + u0);
            let wbase_hi = bhi_ext.as_ptr().add(65 + k0 + u0);
            for l in 0..8 {
                let al = alo_rev[u0 + l];
                let ah = ahi_rev[u0 + l];
                let wl = _mm512_loadu_si512(wbase_lo.add(l) as *const __m512i);
                let wh = _mm512_loadu_si512(wbase_hi.add(l) as *const __m512i);
                acc_ll = _mm512_add_epi64(
                    acc_ll,
                    _mm512_mullo_epi64(_mm512_set1_epi64(al), wl),
                );
                acc_lh = _mm512_add_epi64(
                    acc_lh,
                    _mm512_mullo_epi64(_mm512_set1_epi64(al), wh),
                );
                acc_hl = _mm512_add_epi64(
                    acc_hl,
                    _mm512_mullo_epi64(_mm512_set1_epi64(ah), wl),
                );
                acc_hh = _mm512_add_epi64(
                    acc_hh,
                    _mm512_mullo_epi64(_mm512_set1_epi64(ah), wh),
                );
            }
        }
        // spill the block's accumulators into the QuadAccs
        let mut tmp = [0i64; 8];
        store_block(acc_ll, &mut tmp);
        for l in 0..8 {
            acc[k0 + l].ll += tmp[l];
        }
        store_block(acc_lh, &mut tmp);
        for l in 0..8 {
            acc[k0 + l].lh += tmp[l];
        }
        store_block(acc_hl, &mut tmp);
        for l in 0..8 {
            acc[k0 + l].hl += tmp[l];
        }
        store_block(acc_hh, &mut tmp);
        for l in 0..8 {
            acc[k0 + l].hh += tmp[l];
        }
    }
}

#[inline]
unsafe fn store_block(v: __m512i, out: &mut [i64; 8]) {
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, v);
}

/// `a * b mod (X^64 + 1, Q)`, vectorized when the CPU has the feature set, the exact i128
/// schoolbook otherwise. Bit-identical (the accumulator is the same integer either way).
pub fn negacyclic_mul(a: &Poly, b: &Poly) -> Poly {
    if !available() {
        return a.mul_schoolbook(b);
    }
    let mut acc = [QuadAcc::zero(); N];
    unsafe {
        mac_into(
            core::slice::from_ref(a),
            core::slice::from_ref(b),
            &mut acc,
        );
    }
    finish(acc)
}

/// `sum_m a_m * b_m mod (X^64 + 1, Q)` (the ring inner product), vectorized when available.
/// Bit-identical to `Poly::sprod`.
pub fn negacyclic_sprod(a: &[Poly], b: &[Poly]) -> Poly {
    if !available() {
        return Poly::sprod_schoolbook(a, b);
    }
    assert_eq!(a.len(), b.len(), "sprod operands must pair up");
    debug_assert!(
        a.len() <= 2048,
        "the split-accumulator bound needs m * 2^52 < 2^63"
    );
    let mut acc = [QuadAcc::zero(); N];
    unsafe {
        mac_into(a, b, &mut acc);
    }
    finish(acc)
}

/// All pairs MACed into one accumulator block.
///
/// # Safety
/// AVX-512F/DQ (checked by the callers).
#[target_feature(enable = "avx512f", enable = "avx512dq")]
unsafe fn mac_into(a: &[Poly], b: &[Poly], acc: &mut [QuadAcc; N]) {
    // reversed split halves of each a-pair, rebuilt per pair (small: 128 i64)
    for (ap, bp) in a.iter().zip(b.iter()) {
        let mut alo_rev = [0i64; N];
        let mut ahi_rev = [0i64; N];
        for i in 0..N {
            let (lo, hi) = split(ap.0[N - 1 - i]);
            alo_rev[i] = lo;
            ahi_rev[i] = hi;
        }
        let mut blo = [0i64; N];
        let mut bhi = [0i64; N];
        for j in 0..N {
            let (lo, hi) = split(bp.0[j]);
            blo[j] = lo;
            bhi[j] = hi;
        }
        let blo_ext = extend(&blo);
        let bhi_ext = extend(&bhi);
        mac_pair(&alo_rev, &ahi_rev, &blo_ext, &bhi_ext, acc);
    }
}

/// Reduce the accumulator block to the centered ring element.
fn finish(acc: [QuadAcc; N]) -> Poly {
    Poly(core::array::from_fn(|i| cmod(acc[i].value())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> impl FnMut() -> i64 {
        let mut r = 0x9E37_79B9_7F4A_7C15u64;
        move || {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((r >> 16) as i64).rem_euclid(crate::ring::Q64) - crate::ring::Q64 / 2
        }
    }

    fn poly(mut f: impl FnMut() -> i64, kind: u8) -> Poly {
        let mut p = [0i64; N];
        for x in p.iter_mut() {
            *x = match kind {
                0 => f(),
                1 => 0,
                2 => f().rem_euclid(3) - 1,
                3 => if f() & 1 == 0 { crate::ring::Q64 / 2 } else { -(crate::ring::Q64 / 2) },
                _ => f(),
            };
        }
        Poly(p)
    }

    #[test]
    fn split_is_exact() {
        let mut f = rng();
        for _ in 0..100_000 {
            let x = f();
            let (lo, hi) = split(x);
            assert_eq!(hi * SPLIT + lo, x);
            assert!(lo.abs() <= SPLIT / 2);
        }
    }

    #[test]
    fn mul_matches_schoolbook() {
        if !available() {
            eprintln!("skipping: no AVX-512F/DQ");
            return;
        }
        let mut f = rng();
        for kind_a in 0..5u8 {
            for kind_b in 0..5u8 {
                for _ in 0..40 {
                    let a = poly(&mut f, kind_a % 4);
                    let b = poly(&mut f, kind_b % 4);
                    assert_eq!(negacyclic_mul(&a, &b), a.mul(&b), "kinds {kind_a}/{kind_b}");
                }
            }
        }
        // hand-verified identities at N=64: 1*b = b, X * X^63 = X^64 = -1 (the constant term)
        let b = poly(&mut f, 0);
        assert_eq!(negacyclic_mul(&Poly::one(), &b), b);
        let mut x1 = Poly::zero();
        x1.0[1] = 1; // X
        let mut x63 = Poly::zero();
        x63.0[63] = 1; // X^63
        let mut expect = Poly::zero();
        expect.0[0] = -1; // X * X^63 = X^64 = -1
        assert_eq!(negacyclic_mul(&x1, &x63), expect);
        assert_eq!(negacyclic_mul(&x1, &x63), x1.mul(&x63));
    }

    #[test]
    fn sprod_matches_schoolbook() {
        if !available() {
            eprintln!("skipping: no AVX-512F/DQ");
            return;
        }
        let mut f = rng();
        for k in [1usize, 2, 5, 16, 48] {
            for _ in 0..8 {
                let a: Vec<Poly> = (0..k).map(|_| poly(&mut f, 0)).collect();
                let b: Vec<Poly> = (0..k).map(|_| poly(&mut f, (k % 4) as u8)).collect();
                assert_eq!(negacyclic_sprod(&a, &b), Poly::sprod(&a, &b), "k={k}");
            }
        }
        // zero-weight and max-weight edges
        let z = vec![Poly::zero(); 16];
        let b: Vec<Poly> = (0..16).map(|_| poly(&mut f, 0)).collect();
        assert_eq!(negacyclic_sprod(&z, &b), Poly::zero());
        let w: Vec<Poly> = (0..16).map(|_| poly(&mut f, 3)).collect();
        assert_eq!(negacyclic_sprod(&w, &w), Poly::sprod_schoolbook(&w, &w));
    }

    #[test]
    fn dispatched_paths_match_references() {
        // Poly::mul / Poly::sprod now route through this module; pin the dispatch against
        // the schoolbook references directly
        if !available() {
            eprintln!("skipping: no AVX-512F/DQ");
            return;
        }
        let mut f = rng();
        let a = poly(&mut f, 0);
        let b = poly(&mut f, 2);
        assert_eq!(a.mul(&b), a.mul_schoolbook(&b));
        let av: Vec<Poly> = (0..9).map(|_| poly(&mut f, 0)).collect();
        let bv: Vec<Poly> = (0..9).map(|_| poly(&mut f, 1)).collect();
        assert_eq!(Poly::sprod(&av, &bv), Poly::sprod_schoolbook(&av, &bv));
    }
}
