//! Bit-exactness tests for `simd::gen_large` and `simd::gen_quad` — the generic-input NTT
//! kernels of the large splitting primes (17497, 19441) and the quadratic-slot primes
//! (2917, 4861, 12637), ported from upstream `tests/ntt.rs::gen_large` and the `quad` module's
//! generic-kernel parts.
//!
//! Three layers of checking per kernel:
//! * an exact i32 *shadow* of the kernel (same operation order, same reduction placement, every
//!   value kept as i32 so an i16 overflow would be observable): the kernel lanes must equal the
//!   shadow bit for bit;
//! * the scalar reference (`scalar::ntt` / `scalar::intt` / `scalar::ntt_quad` /
//!   `ring::intt_quad_of`) of the same input: the lanes must be congruent mod q;
//! * the declared bounds (`output_bound`, the `const` recursions `fwd_model` / `inv_model` /
//!   `gen_model` / `inv_bound`): every lane and every intermediate must respect them.
//!
//! Round trips close the loop: `intt(ntt(x)) == centered x`, `ntt(intt(y)) == y mod q`, and an
//! end-to-end identity over several random batches. The `gen_large` round trip feeds the inverse
//! its own forward output (`<= 0.53 q`) rather than the declared `(q-1)/2`: every inverse
//! butterfly reduces its inputs on arrival, so anything inside the lookup Barrett's own bound is
//! exactly as safe as the declared one — upstream's round-trip test does the same. A final test
//! prints the per-batch medians.
//!
//! One soundness caveat the `gen_large` inverse checks have to respect (upstream's own inverse
//! test never replays the per-level model on data, so it cannot see it): the model's level-6
//! entry `1.50 q` is derived from `red_bound((q-1)/2) = (q-1)/2`, i.e. from the assumption that
//! the lookup Barrett leaves an already centered lane alone. The 2^11-window LUT does not
//! guarantee that — for 17497 the window `[8192, 10240]` rounds onto `k = 1`, so centered inputs
//! in `[8192, 8748]` spill to `[-9305, -8749]` (and mirrored on the negative side), making the
//! real first-level data bound `3 * 9305 = 27915`; for 19441 the same window has `k = 0` and
//! nothing spills. The test checks that sound superset at level 6 and the model's own claims at
//! every deeper level, whose inputs are all past the LUT's own bound, where the reduction clamps
//! exactly as modeled.
//!
//! The tests skip silently on machines without the AVX-512 PCS feature set.

// the kernels and their shadows index positions with computed strides
#![allow(clippy::needless_range_loop)]

use lattice_labinius::binfield::Rng;
use lattice_labinius::params::{inv_mod, mont_mul_i16, Params, ParamsQ, N};
use lattice_labinius::ring;
use lattice_labinius::scalar::{self, Coeffs};
use lattice_labinius::simd::ntt_quad::barrett_lut_i16;
use lattice_labinius::simd::{gen_large, gen_quad, Batch32};
use std::time::Instant;

fn gate() -> bool {
    if lattice_labinius::simd::available() {
        true
    } else {
        eprintln!("skipping: no AVX-512 PCS feature set on this machine");
        false
    }
}

// ------------------------------------------------------------------ shared input builders

fn to_batch(cols: &[[i16; N]; 32]) -> Batch32 {
    let mut b = Batch32::zero();
    for p in 0..32 {
        for j in 0..N {
            b.v[j][p] = cols[p][j];
        }
    }
    b
}

/// Coefficient columns with lanes in the forward kernels' declared input range `|x| <= q`.
fn random_cols<const Q: u16>(rng: &mut Rng) -> [[i16; N]; 32] {
    std::array::from_fn(|_| {
        std::array::from_fn(|_| (rng.below(2 * Q as u32 + 1) as i32 - Q as i32) as i16)
    })
}

/// Binary columns: the input a commitment's generic-side kernel sees when it is fed a lift.
fn binary_cols(rng: &mut Rng) -> [[i16; N]; 32] {
    std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() & 1) as i16))
}

/// Adversarial coefficient batches at the corners of the input range.
fn adversarial<const Q: u16>() -> Vec<[[i16; N]; 32]> {
    let q = Q as i16;
    let mut out = Vec::new();
    out.push([[0i16; N]; 32]);
    out.push([[q; N]; 32]);
    out.push([[-q; N]; 32]);
    out.push(std::array::from_fn(|_| {
        std::array::from_fn(|j| if j % 2 == 0 { q } else { -q })
    }));
    out.push(std::array::from_fn(|p| {
        std::array::from_fn(|j| if (j + p) % 2 == 0 { q } else { -q })
    }));
    for &m in &[0usize, 161, 162, 323, 324, 486, 647] {
        for &val in &[1i16, q, -q] {
            let mut c = [[0i16; N]; 32];
            for p in 0..32 {
                c[p][m] = val;
            }
            out.push(c);
        }
    }
    // one monomial per polynomial, all different positions
    let mut c = [[0i16; N]; 32];
    for p in 0..32 {
        c[p][(p * 21) % N] = q;
    }
    out.push(c);
    out
}

/// NTT-domain columns with `|v| <= bound`.
fn random_ntt_cols<const Q: u16>(rng: &mut Rng, bound: i32) -> [[i16; N]; 32] {
    std::array::from_fn(|_| {
        std::array::from_fn(|_| (rng.below(2 * bound as u32 + 1) as i32 - bound) as i16)
    })
}

/// Adversarial transform-domain batches at the corners of an input bound.
fn adversarial_ntt<const Q: u16>(bound: i32) -> Vec<[[i16; N]; 32]> {
    let m = bound as i16;
    let mut out = Vec::new();
    out.push([[0i16; N]; 32]);
    out.push([[m; N]; 32]);
    out.push([[-m; N]; 32]);
    out.push(std::array::from_fn(|_| {
        std::array::from_fn(|j| if j % 2 == 0 { m } else { -m })
    }));
    out.push(std::array::from_fn(|_| {
        std::array::from_fn(|j| if j % 3 == 0 { m } else { -m })
    }));
    out.push(std::array::from_fn(|p| {
        std::array::from_fn(|j| if (j / 27 + p) % 2 == 0 { m } else { -m })
    }));
    for &u in &[
        0usize, 1, 2, 26, 27, 53, 54, 80, 81, 161, 162, 323, 324, 485, 486, 647,
    ] {
        let mut c = [[0i16; N]; 32];
        for p in 0..32 {
            c[p][u] = if p % 2 == 0 { m } else { -m };
        }
        out.push(c);
    }
    out
}

// ==============================================================================================
// gen_large: the large splitting primes 17497 / 19441
// ==============================================================================================

/// The scalar lane arithmetic the kernel's helpers implement: a Montgomery product against a
/// plain twiddle, and the shuffle-port lookup Barrett.
struct Ops<const Q: u16>;

impl<const Q: u16> Ops<Q> {
    fn mont(a: i32, x: u16) -> i32 {
        assert!(
            a.abs() < 32768,
            "q={Q} i16 overflow feeding a multiplication: {a}"
        );
        let w = Params::<Q>::to_mont(x);
        mont_mul_i16(a as i16, w, Params::<Q>::mont_pre(w), Q) as i32
    }
    fn red(a: i32) -> i32 {
        assert!(a.abs() < 32768, "q={Q} i16 overflow before reduction: {a}");
        barrett_lut_i16(a as i16, Q) as i32
    }
    fn ck(a: i32) -> i32 {
        assert!(a.abs() < 32768, "q={Q} i16 overflow: {a}");
        a
    }
}

/// Exact i32 mirror of `gen_large::ntt_gen_batch32`: same operation order, same reduction
/// placement (a radix-3 butterfly reduces all of `a0`, `t1`, `t2`, `u`; level 2 reduces the
/// three inputs it does not multiply; level 6 reduces its three outputs). `lmax[i]` = max |value|
/// after pass `i` (`[levels 0+1, 2+3, 4, 5, 6]`), matching `gen_large::fwd_model`.
fn shadow_large<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 5]) -> [i32; N] {
    let q = Q as u64;
    let mut v = [0i32; N];
    for i in 0..N {
        v[i] = input[i] as i32;
        assert!(v[i].abs() <= Q as i32, "q={Q} input bound: {}", v[i]);
    }
    let om = Params::<Q>::OMEGA;
    // one radix-3 butterfly with every term reduced
    let r3 = |a0: i32, a1: i32, a2: i32, z: u16| -> (i32, i32, i32) {
        let z2 = (z as u64 * z as u64 % q) as u16;
        let t1 = Ops::<Q>::red(Ops::<Q>::mont(a1, z));
        let t2 = Ops::<Q>::red(Ops::<Q>::mont(a2, z2));
        let u = Ops::<Q>::red(Ops::<Q>::mont(Ops::<Q>::ck(t1 - t2), om));
        let a0 = Ops::<Q>::red(a0);
        (
            Ops::<Q>::ck(a0 + t1 + t2),
            Ops::<Q>::ck(a0 - t2 + u),
            Ops::<Q>::ck(a0 - t1 - u),
        )
    };
    // pass A: level 0 as the two products a0 + zeta6 a1 / a0 + zeta6^-1 a1, then level 1.
    let z6 = Params::<Q>::ZETA6;
    let z6i = ((1 + q - z6 as u64) % q) as u16;
    let z10 = Params::<Q>::zeta(1, 0);
    let z11 = Params::<Q>::zeta(1, 1);
    for i in 0..162 {
        let (a0, a1) = (v[i], v[i + 324]);
        let (b0, b1) = (v[i + 162], v[i + 486]);
        let c0 = Ops::<Q>::red(Ops::<Q>::ck(a0 + Ops::<Q>::mont(a1, z6)));
        let c1 = Ops::<Q>::ck(b0 + Ops::<Q>::mont(b1, z6));
        let c2 = Ops::<Q>::red(Ops::<Q>::ck(a0 + Ops::<Q>::mont(a1, z6i)));
        let c3 = Ops::<Q>::ck(b0 + Ops::<Q>::mont(b1, z6i));
        let u = Ops::<Q>::mont(c1, z10);
        let w = Ops::<Q>::mont(c3, z11);
        v[i] = Ops::<Q>::ck(c0 + u);
        v[i + 162] = Ops::<Q>::ck(c0 - u);
        v[i + 324] = Ops::<Q>::ck(c2 + w);
        v[i + 486] = Ops::<Q>::ck(c2 - w);
    }
    lmax[0] = v.iter().map(|x| x.abs()).max().unwrap();
    // pass B: level 2 reduces the three inputs it does not multiply, then level 3.
    for blk in 0..4 {
        let base = 162 * blk;
        let z2 = Params::<Q>::zeta(2, blk);
        let z3a = Params::<Q>::zeta(3, 2 * blk);
        let z3b = Params::<Q>::zeta(3, 2 * blk + 1);
        for j in 0..27 {
            let b = base + j;
            let x0 = Ops::<Q>::red(v[b]);
            let x1 = Ops::<Q>::red(v[b + 27]);
            let x2 = Ops::<Q>::red(v[b + 54]);
            let t0 = Ops::<Q>::mont(v[b + 81], z2);
            let t1 = Ops::<Q>::mont(v[b + 108], z2);
            let t2 = Ops::<Q>::mont(v[b + 135], z2);
            let (y0, y1, y2) = r3(
                Ops::<Q>::ck(x0 + t0),
                Ops::<Q>::ck(x1 + t1),
                Ops::<Q>::ck(x2 + t2),
                z3a,
            );
            let (w0, w1, w2) = r3(
                Ops::<Q>::ck(x0 - t0),
                Ops::<Q>::ck(x1 - t1),
                Ops::<Q>::ck(x2 - t2),
                z3b,
            );
            v[b] = y0;
            v[b + 27] = y1;
            v[b + 54] = y2;
            v[b + 81] = w0;
            v[b + 108] = w1;
            v[b + 135] = w2;
        }
    }
    lmax[1] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 4
    for k4 in 0..24 {
        let base = 27 * k4;
        let z4 = Params::<Q>::zeta(4, k4);
        for i in 0..9 {
            let b = base + i;
            let (y0, y1, y2) = r3(v[b], v[b + 9], v[b + 18], z4);
            v[b] = y0;
            v[b + 9] = y1;
            v[b + 18] = y2;
        }
    }
    lmax[2] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 5
    for k4 in 0..24 {
        let base = 27 * k4;
        for bb in 0..3 {
            let z5 = Params::<Q>::zeta(5, 3 * k4 + bb);
            for j in 0..3 {
                let b = base + 9 * bb + j;
                let (y0, y1, y2) = r3(v[b], v[b + 3], v[b + 6], z5);
                v[b] = y0;
                v[b + 3] = y1;
                v[b + 6] = y2;
            }
        }
    }
    lmax[3] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 6, whose three outputs are reduced
    for k4 in 0..24 {
        let base = 27 * k4;
        for g in 0..9 {
            let z6l = Params::<Q>::zeta(6, 9 * k4 + g);
            let b = base + 3 * g;
            let (y0, y1, y2) = r3(v[b], v[b + 1], v[b + 2], z6l);
            v[b] = Ops::<Q>::red(y0);
            v[b + 1] = Ops::<Q>::red(y1);
            v[b + 2] = Ops::<Q>::red(y2);
        }
    }
    lmax[4] = v.iter().map(|x| x.abs()).max().unwrap();
    v
}

/// Exact i32 mirror of `gen_large::intt_gen_batch32`. `lmax[l]` = max |value| after inverse
/// level `l` (levels 6..1, and 0 = before centering) — the reverse of `gen_large::inv_model`'s
/// `[after levels 6, 5, .., 0]` ordering; the check loop maps one to the other.
fn shadow_large_inv<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 7]) -> [i32; N] {
    let q = Q as u64;
    let mut v = [0i32; N];
    for i in 0..N {
        v[i] = input[i] as i32;
        // the declared contract is |x| <= in_bound; the forward kernel's own output (<= the
        // lookup Barrett's bound, below q for the large primes) is also accepted — every
        // butterfly reduces its inputs on arrival, so anything inside i16 is exactly as safe
        // (upstream's round-trip test feeds the same off-label input).
        assert!(
            v[i].abs() < 32768,
            "q={Q} inverse input overflows i16: {}",
            v[i]
        );
    }
    let om = Params::<Q>::OMEGA;
    let zi =
        |level: usize, k: usize| -> u16 { inv_mod(Params::<Q>::zeta(level, k) as u64, q) as u16 };
    // inverse radix-3: the three inputs and `u` are reduced, normalisation deferred
    let ir3 = |y0: i32, y1: i32, y2: i32, z: u16| -> (i32, i32, i32) {
        let (y0, y1, y2) = (Ops::<Q>::red(y0), Ops::<Q>::red(y1), Ops::<Q>::red(y2));
        let u = Ops::<Q>::red(Ops::<Q>::mont(Ops::<Q>::ck(y2 - y1), om));
        let s = Ops::<Q>::ck(y0 + y1 + y2);
        let z2 = (z as u64 * z as u64 % q) as u16;
        let a1 = Ops::<Q>::mont(Ops::<Q>::ck(y0 - y1 + u), z);
        let a2 = Ops::<Q>::mont(Ops::<Q>::ck(y0 - y2 - u), z2);
        (s, a1, a2)
    };
    // inverse radix-2: both inputs reduced, normalisation deferred
    let ir2 = |y0: i32, y1: i32, z: u16| -> (i32, i32) {
        let (y0, y1) = (Ops::<Q>::red(y0), Ops::<Q>::red(y1));
        (
            Ops::<Q>::ck(y0 + y1),
            Ops::<Q>::mont(Ops::<Q>::ck(y0 - y1), z),
        )
    };
    // level 6 (pass D)
    for k4 in 0..24 {
        let base = 27 * k4;
        for g in 0..9 {
            let b = base + 3 * g;
            let (s, a1, a2) = ir3(v[b], v[b + 1], v[b + 2], zi(6, 9 * k4 + g));
            v[b] = s;
            v[b + 1] = a1;
            v[b + 2] = a2;
        }
    }
    lmax[6] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 5 (pass C5)
    for k4 in 0..24 {
        let base = 27 * k4;
        for bb in 0..3 {
            let z5 = zi(5, 3 * k4 + bb);
            for j in 0..3 {
                let b = base + 9 * bb + j;
                let (s, a1, a2) = ir3(v[b], v[b + 3], v[b + 6], z5);
                v[b] = s;
                v[b + 3] = a1;
                v[b + 6] = a2;
            }
        }
    }
    lmax[5] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 4 (pass C4)
    for k4 in 0..24 {
        let base = 27 * k4;
        let z4 = zi(4, k4);
        for i in 0..9 {
            let b = base + i;
            let (s, a1, a2) = ir3(v[b], v[b + 9], v[b + 18], z4);
            v[b] = s;
            v[b + 9] = a1;
            v[b + 18] = a2;
        }
    }
    lmax[4] = v.iter().map(|x| x.abs()).max().unwrap();
    // levels 3 + 2 (pass B)
    for blk in 0..4 {
        let base = 162 * blk;
        let z3a = zi(3, 2 * blk);
        let z3b = zi(3, 2 * blk + 1);
        let z2 = zi(2, blk);
        for j in 0..27 {
            let b = base + j;
            let (n0, n1, n2) = ir3(v[b], v[b + 27], v[b + 54], z3a);
            let (m0, m1, m2) = ir3(v[b + 81], v[b + 108], v[b + 135], z3b);
            lmax[3] = lmax[3].max(n0.abs()).max(n1.abs()).max(n2.abs());
            lmax[3] = lmax[3].max(m0.abs()).max(m1.abs()).max(m2.abs());
            let (s0, t0) = ir2(n0, m0, z2);
            let (s1, t1) = ir2(n1, m1, z2);
            let (s2, t2) = ir2(n2, m2, z2);
            v[b] = s0;
            v[b + 27] = s1;
            v[b + 54] = s2;
            v[b + 81] = t0;
            v[b + 108] = t1;
            v[b + 135] = t2;
        }
    }
    lmax[2] = v.iter().map(|x| x.abs()).max().unwrap();
    // levels 1 + 0 (pass A): the level-1 outputs, then the Phi_6 recombination + centering.
    let z10i = zi(1, 0);
    let z11i = zi(1, 1);
    let half = (Q as i32 - 1) / 2;
    let mut raw = 0i32;
    let mut out = [0i32; N];
    let z6 = Params::<Q>::ZETA6 as u64;
    let det = inv_mod((2 * z6 + q - 1) % q, q);
    let ka = (det * inv_mod(324, q) % q) as u16;
    let kb = inv_mod(648, q) as u16;
    let kc = ((q - det * inv_mod(648, q) % q) % q) as u16;
    let center = |mut x: i32| {
        if x > half {
            x -= Q as i32;
        }
        if x < -half {
            x += Q as i32;
        }
        x
    };
    for i in 0..162 {
        let (c0, c1) = ir2(v[i], v[i + 162], z10i);
        let (c2, c3) = ir2(v[i + 324], v[i + 486], z11i);
        lmax[1] = lmax[1]
            .max(c0.abs())
            .max(c1.abs())
            .max(c2.abs())
            .max(c3.abs());
        let (c0, c2) = (Ops::<Q>::red(c0), Ops::<Q>::red(c2));
        let d0 = Ops::<Q>::ck(c0 - c2);
        let a1 = Ops::<Q>::mont(d0, ka);
        let a0 = Ops::<Q>::ck(Ops::<Q>::mont(Ops::<Q>::ck(c0 + c2), kb) + Ops::<Q>::mont(d0, kc));
        let d1 = Ops::<Q>::ck(c1 - c3);
        let b1 = Ops::<Q>::mont(d1, ka);
        let b0 = Ops::<Q>::ck(Ops::<Q>::mont(Ops::<Q>::ck(c1 + c3), kb) + Ops::<Q>::mont(d1, kc));
        raw = raw.max(a0.abs()).max(a1.abs()).max(b0.abs()).max(b1.abs());
        out[i] = center(a0);
        out[i + 162] = center(b0);
        out[i + 324] = center(a1);
        out[i + 486] = center(b1);
    }
    lmax[0] = raw;
    out
}

// ------------------------------------------------------------------ gen_large checks

/// Max `|barrett_lut_i16(a, Q)|` over `|a| <= bound`: what the 2^11-window lookup Barrett can do
/// to an already-centered lane. Equal to `bound` when no window straddling `+-q/2` rounds onto a
/// non-zero `k` (19441: 9720), up to `q/2 + 2^10`-ish when one does (17497: 9305).
fn red_in_max<const Q: u16>(bound: i32) -> i32 {
    let mut m = 0i32;
    let mut a = -bound;
    while a <= bound {
        let r = barrett_lut_i16(a as i16, Q) as i32;
        let r = if r < 0 { -r } else { r };
        if r > m {
            m = r;
        }
        a += 1;
    }
    m
}

fn check_large<const Q: u16>(cols: &[[i16; N]; 32], what: &str, worst: &mut [i32; 5]) {
    let mut b = to_batch(cols);
    unsafe { gen_large::ntt_gen_batch32::<Q>(&mut b) };
    let bound = gen_large::output_bound(Q);
    for p in 0..32 {
        let mut lmax = [0i32; 5];
        let want_shadow = shadow_large::<Q>(&cols[p], &mut lmax);
        for l in 0..5 {
            worst[l] = worst[l].max(lmax[l]);
        }
        let mut coeffs: Coeffs = [0u32; N];
        for j in 0..N {
            coeffs[j] = (cols[p][j] as i32).rem_euclid(Q as i32) as u32;
        }
        let want = scalar::ntt::<Q>(&coeffs);
        for j in 0..N {
            let got = b.v[j][p] as i32;
            assert_eq!(
                got, want_shadow[j],
                "{what} q={Q} poly {p} slot {j}: shadow mismatch"
            );
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} slot {j}"
            );
            assert!(
                got.abs() <= bound,
                "{what} q={Q} poly {p} slot {j}: |{got}| > {bound}"
            );
        }
    }
}

fn check_large_inv<const Q: u16>(cols: &[[i16; N]; 32], what: &str, worst: &mut Option<[i32; 7]>) {
    let mut b = to_batch(cols);
    unsafe { gen_large::intt_gen_batch32::<Q>(&mut b) };
    let half = (Q as i32 - 1) / 2;
    for p in 0..32 {
        let mut lmax = [0i32; 7];
        let want_shadow = shadow_large_inv::<Q>(&cols[p], &mut lmax);
        // the per-level model is claimed for the declared input bound `(q-1)/2` only; the
        // off-label forward-output inputs (below q, reduced on arrival) stay bit-exact and
        // inside i16 but can run `3 r` wide at level 6, past `3 (q-1)/2`.
        if let Some(w) = worst.as_mut() {
            for l in 0..7 {
                w[l] = w[l].max(lmax[l]);
            }
        }
        let mut coeffs: Coeffs = [0u32; N];
        for j in 0..N {
            coeffs[j] = (cols[p][j] as i32).rem_euclid(Q as i32) as u32;
        }
        let want = scalar::intt::<Q>(&coeffs);
        for j in 0..N {
            let got = b.v[j][p] as i32;
            assert_eq!(
                got, want_shadow[j],
                "{what} q={Q} poly {p} coeff {j}: shadow mismatch"
            );
            assert!(
                got.abs() <= half,
                "{what} q={Q} poly {p} coeff {j}: |{got}| > {half}"
            );
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} coeff {j}"
            );
        }
    }
}

fn run_large<const Q: u16>() {
    let mut rng = Rng::new(0x5ee1 ^ Q as u64);
    let mut worst = [0i32; 5];
    for (i, c) in adversarial::<Q>().iter().enumerate() {
        check_large::<Q>(c, &format!("adversarial#{i}"), &mut worst);
    }
    for i in 0..16 {
        check_large::<Q>(&binary_cols(&mut rng), &format!("binary#{i}"), &mut worst);
    }
    for i in 0..16 {
        check_large::<Q>(
            &random_cols::<Q>(&mut rng),
            &format!("random#{i}"),
            &mut worst,
        );
    }
    let (claim, peak) = gen_large::fwd_model(Q);
    for l in 0..5 {
        assert!(
            worst[l] <= claim[l],
            "q={Q} forward pass {l}: observed {} > claimed {}",
            worst[l],
            claim[l]
        );
        assert!(
            claim[l] < 32768,
            "q={Q} forward pass {l}: claimed {} exceeds i16",
            claim[l]
        );
        println!(
            "q={Q} forward pass {l}: observed {} ({:.3} q), claimed {} ({:.4} q)",
            worst[l],
            worst[l] as f64 / Q as f64,
            claim[l],
            claim[l] as f64 / Q as f64
        );
    }
    println!(
        "q={Q} forward: model peak {} of 32767, output bound {} = {:.4} q",
        peak,
        gen_large::output_bound(Q),
        gen_large::output_bound(Q) as f64 / Q as f64
    );
}

fn run_large_inv<const Q: u16>() {
    let mut rng = Rng::new(0x1177 ^ Q as u64);
    let bound = gen_large::in_bound(Q);
    let mut worst = Some([0i32; 7]);
    for (i, c) in adversarial_ntt::<Q>(bound).iter().enumerate() {
        check_large_inv::<Q>(c, &format!("adversarial#{i}"), &mut worst);
    }
    for i in 0..12 {
        let c = random_ntt_cols::<Q>(&mut rng, bound);
        check_large_inv::<Q>(&c, &format!("lazy#{i}"), &mut worst);
    }
    for i in 0..12 {
        let c = random_ntt_cols::<Q>(&mut rng, (Q as i32 - 1) / 2);
        check_large_inv::<Q>(&c, &format!("centered#{i}"), &mut worst);
    }
    // the real inputs: the forward kernel's own output (inside the lookup Barrett's bound, so
    // exactly as safe as the declared one: every butterfly reduces its inputs on arrival) —
    // bit-exactness only, outside the declared per-level model.
    let mut off_label = None;
    for i in 0..8 {
        let cols = random_cols::<Q>(&mut rng);
        let mut b = to_batch(&cols);
        unsafe { gen_large::ntt_gen_batch32::<Q>(&mut b) };
        let ntt: [[i16; N]; 32] = std::array::from_fn(|p| std::array::from_fn(|j| b.v[j][p]));
        check_large_inv::<Q>(&ntt, &format!("forward#{i}"), &mut off_label);
    }
    let worst = worst.unwrap();
    let (claim, peak) = gen_large::inv_model(Q);
    // claim[l] = after inverse level 6-l (the model is ordered [levels 6, 5, .., 0]); the shadow
    // records lmax[l] = after level l.
    //
    // Level 6 is the one place the model is optimistic on data: its `1.50 q` comes from
    // `red_bound((q-1)/2) = (q-1)/2`, but the 2^11-window lookup Barrett can spill a centered
    // lane in the window straddling q/2 up to `red_in_max` (9305 at 17497; nothing spills at
    // 19441, where the same window has k = 0), so the sound first-level data bound is
    // `3 * red_in_max` (27915 at 17497, 29160 = the model's own claim at 19441). Every deeper
    // level's inputs are past the LUT's own bound, so the reduction clamps to BLM exactly as
    // modeled and the model's claims there are sound as stated.
    let r_in = red_in_max::<Q>(bound);
    let c6 = claim[0].max(3 * r_in);
    assert!(
        worst[6] <= c6,
        "q={Q} inverse level 6: observed {} > sound bound {c6}",
        worst[6]
    );
    assert!(
        3 * r_in < 32768,
        "q={Q} inverse level 6: sound bound {} exceeds i16",
        3 * r_in
    );
    println!(
        "q={Q} inverse level 6: observed {} ({:.3} q), model claim {} ({:.3} q), sound bound {} = 3 x red_in_max {} ({:.3} q)",
        worst[6],
        worst[6] as f64 / Q as f64,
        claim[0],
        claim[0] as f64 / Q as f64,
        3 * r_in,
        r_in,
        (3 * r_in) as f64 / Q as f64
    );
    for l in (0..6).rev() {
        let c = claim[6 - l];
        assert!(
            worst[l] <= c,
            "q={Q} inverse level {l}: observed {} > claimed {c}",
            worst[l]
        );
        assert!(
            c < 32768,
            "q={Q} inverse level {l}: claimed {c} exceeds i16"
        );
        println!(
            "q={Q} inverse level {l}: observed {} ({:.3} q), claimed {c} ({:.4} q)",
            worst[l],
            worst[l] as f64 / Q as f64,
            c as f64 / Q as f64
        );
    }
    println!(
        "q={Q} inverse: model peak {} of 32767 (level-6 spill-aware peak {})",
        peak,
        peak.max(3 * r_in)
    );
}

#[test]
fn large_forward_17497() {
    if !gate() {
        return;
    }
    run_large::<17497>();
}

#[test]
fn large_forward_19441() {
    if !gate() {
        return;
    }
    run_large::<19441>();
}

#[test]
fn large_inverse_17497() {
    if !gate() {
        return;
    }
    run_large_inv::<17497>();
}

#[test]
fn large_inverse_19441() {
    if !gate() {
        return;
    }
    run_large_inv::<19441>();
}

// ==============================================================================================
// gen_quad: the quadratic-slot primes 2917 / 4861 / 12637
// ==============================================================================================

/// The scalar lane arithmetic of the quad kernels (same Montgomery form, same lookup Barrett).
struct OpsQ<const Q: u16>;

impl<const Q: u16> OpsQ<Q> {
    fn mont(a: i32, x: u16) -> i32 {
        assert!(
            a.abs() < 32768,
            "q={Q} i16 overflow feeding a multiplication: {a}"
        );
        let w = Params::<Q>::to_mont(x);
        mont_mul_i16(a as i16, w, Params::<Q>::mont_pre(w), Q) as i32
    }
    fn red(a: i32) -> i32 {
        assert!(a.abs() < 32768, "q={Q} i16 overflow before reduction: {a}");
        barrett_lut_i16(a as i16, Q) as i32
    }
    fn ck(a: i32) -> i32 {
        assert!(a.abs() < 32768, "q={Q} i16 overflow: {a}");
        a
    }
}

/// Exact i32 mirror of `gen_quad::ntt_quad_gen_batch32`, reduction placement included
/// (`gen_flags`: the loaded `a1`/`b1` of pass A, its level-0 output, the untwiddled `a0` of
/// levels 2..5). `lmax[i]` = max |value| after `[levels 0+1, level 2, level 3, level 4, level 5]`,
/// matching `gen_quad::gen_model`.
fn shadow_quad<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 5]) -> [i32; N] {
    let q = Q as u64;
    let (bi, ba, bl) = gen_quad::gen_flags(Q);
    let mut v = [0i32; N];
    for i in 0..N {
        v[i] = input[i] as i32;
        assert!(v[i].abs() <= Q as i32, "q={Q} input bound: {}", v[i]);
    }
    let om = ParamsQ::<Q>::OMEGA;
    // one radix-3 butterfly, only the untwiddled `a0` optionally reduced
    let r3 = |a0: i32, a1: i32, a2: i32, z: u16, bar: bool| -> (i32, i32, i32) {
        let z2 = (z as u64 * z as u64 % q) as u16;
        let t1 = OpsQ::<Q>::mont(a1, z);
        let t2 = OpsQ::<Q>::mont(a2, z2);
        let u = OpsQ::<Q>::mont(OpsQ::<Q>::ck(t1 - t2), om);
        let a0 = if bar { OpsQ::<Q>::red(a0) } else { a0 };
        (
            OpsQ::<Q>::ck(a0 + t1 + t2),
            OpsQ::<Q>::ck(a0 - t2 + u),
            OpsQ::<Q>::ck(a0 - t1 - u),
        )
    };
    // pass A: levels 0 + 1
    let z6 = ParamsQ::<Q>::ZETA6;
    let z10 = ParamsQ::<Q>::zeta(1, 0);
    let z11 = ParamsQ::<Q>::zeta(1, 1);
    for i in 0..162 {
        let a0 = v[i];
        let b0 = v[i + 162];
        let mut a1 = v[i + 324];
        let mut b1 = v[i + 486];
        if bi {
            a1 = OpsQ::<Q>::red(a1);
            b1 = OpsQ::<Q>::red(b1);
        }
        let t = OpsQ::<Q>::mont(a1, z6);
        let s = OpsQ::<Q>::mont(b1, z6);
        let c0 = OpsQ::<Q>::ck(a0 + t);
        let c1 = OpsQ::<Q>::ck(b0 + s);
        let mut c2 = OpsQ::<Q>::ck(a0 + a1 - t);
        let mut c3 = OpsQ::<Q>::ck(b0 + b1 - s);
        if ba {
            c2 = OpsQ::<Q>::red(c2);
            c3 = OpsQ::<Q>::red(c3);
        }
        let u = OpsQ::<Q>::mont(c1, z10);
        let w = OpsQ::<Q>::mont(c3, z11);
        v[i] = OpsQ::<Q>::ck(c0 + u);
        v[i + 162] = OpsQ::<Q>::ck(c0 - u);
        v[i + 324] = OpsQ::<Q>::ck(c2 + w);
        v[i + 486] = OpsQ::<Q>::ck(c2 - w);
    }
    lmax[0] = v.iter().map(|x| x.abs()).max().unwrap();
    // pass B: levels 2 + 3, fused over groups of 9 (the level-2 outputs live in registers)
    for blk in 0..4 {
        let base = 162 * blk;
        let z2 = ParamsQ::<Q>::zeta(2, blk);
        for i0 in 0..18 {
            let mut y = [0i32; 9];
            for a in 0..3 {
                let b = base + i0 + 18 * a;
                let (u0, u1, u2) = r3(v[b], v[b + 54], v[b + 108], z2, bl[0]);
                y[a] = u0;
                y[3 + a] = u1;
                y[6 + a] = u2;
            }
            lmax[1] = lmax[1]
                .max(y[0].abs())
                .max(y[1].abs())
                .max(y[2].abs())
                .max(y[3].abs())
                .max(y[4].abs())
                .max(y[5].abs())
                .max(y[6].abs())
                .max(y[7].abs())
                .max(y[8].abs());
            for s in 0..3 {
                let z3 = ParamsQ::<Q>::zeta(3, 3 * blk + s);
                let (v0, v1, v2) = r3(y[3 * s], y[3 * s + 1], y[3 * s + 2], z3, bl[1]);
                let b = base + 54 * s + i0;
                v[b] = v0;
                v[b + 18] = v1;
                v[b + 36] = v2;
            }
        }
    }
    lmax[2] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 4
    for k4 in 0..36 {
        let base = 18 * k4;
        let z4 = ParamsQ::<Q>::zeta(4, k4);
        for i in 0..6 {
            let b = base + i;
            let (y0, y1, y2) = r3(v[b], v[b + 6], v[b + 12], z4, bl[2]);
            v[b] = y0;
            v[b + 6] = y1;
            v[b + 12] = y2;
        }
    }
    lmax[3] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 5
    for k4 in 0..36 {
        let base = 18 * k4;
        for g in 0..3 {
            let z5 = ParamsQ::<Q>::zeta(5, 3 * k4 + g);
            for i in 0..2 {
                let b = base + 6 * g + i;
                let (y0, y1, y2) = r3(v[b], v[b + 2], v[b + 4], z5, bl[3]);
                v[b] = y0;
                v[b + 2] = y1;
                v[b + 4] = y2;
            }
        }
    }
    lmax[4] = v.iter().map(|x| x.abs()).max().unwrap();
    v
}

/// Exact i32 mirror of `gen_quad::intt_quad_gen_batch32`, reduction placement included
/// (`inv_flags`: the level-5 inputs, the sums of levels 5, 4, 3, 2, 1). `lmax[l]` = max |value|
/// after inverse level `l` (levels 5..1, and 0 = before centering) — the reverse of
/// `gen_quad::inv_bound`'s `[after levels 5, 4, .., 0]` ordering; the check loop maps one to
/// the other.
fn shadow_quad_inv<const Q: u16>(input: &[i16; N], lmax: &mut [i32; 6]) -> [i32; N] {
    let q = Q as u64;
    let f = gen_quad::inv_flags(Q);
    let mut v = [0i32; N];
    for i in 0..N {
        v[i] = input[i] as i32;
        // the declared contract is |x| <= in_bound, the widest transform of this tree the
        // crate produces; the assertion below is the hard requirement (i16).
        assert!(
            v[i].abs() < 32768,
            "q={Q} inverse input overflows i16: {}",
            v[i]
        );
    }
    let om = ParamsQ::<Q>::OMEGA;
    let zi =
        |level: usize, k: usize| -> u16 { inv_mod(ParamsQ::<Q>::zeta(level, k) as u64, q) as u16 };
    // inverse radix-3: `IN` reduces the three loaded values, `BAR` the untwiddled sum
    let ir3 = |y0: i32, y1: i32, y2: i32, z: u16, iin: bool, bar: bool| -> (i32, i32, i32) {
        let (y0, y1, y2) = if iin {
            (OpsQ::<Q>::red(y0), OpsQ::<Q>::red(y1), OpsQ::<Q>::red(y2))
        } else {
            (y0, y1, y2)
        };
        let u = OpsQ::<Q>::mont(OpsQ::<Q>::ck(y2 - y1), om);
        let s = OpsQ::<Q>::ck(y0 + y1 + y2);
        let z2 = (z as u64 * z as u64 % q) as u16;
        let a1 = OpsQ::<Q>::mont(OpsQ::<Q>::ck(y0 - y1 + u), z);
        let a2 = OpsQ::<Q>::mont(OpsQ::<Q>::ck(y0 - y2 - u), z2);
        (if bar { OpsQ::<Q>::red(s) } else { s }, a1, a2)
    };
    // inverse radix-2, `BAR` on the sum
    let ir2 = |y0: i32, y1: i32, z: u16, bar: bool| -> (i32, i32) {
        let s = OpsQ::<Q>::ck(y0 + y1);
        (
            if bar { OpsQ::<Q>::red(s) } else { s },
            OpsQ::<Q>::mont(OpsQ::<Q>::ck(y0 - y1), z),
        )
    };
    // level 5 (the first inverse level)
    for k4 in 0..36 {
        let base = 18 * k4;
        for g in 0..3 {
            let z5 = zi(5, 3 * k4 + g);
            for i in 0..2 {
                let b = base + 6 * g + i;
                let (s, a1, a2) = ir3(v[b], v[b + 2], v[b + 4], z5, f[0], f[1]);
                v[b] = s;
                v[b + 2] = a1;
                v[b + 4] = a2;
            }
        }
    }
    lmax[5] = v.iter().map(|x| x.abs()).max().unwrap();
    // level 4
    for k4 in 0..36 {
        let base = 18 * k4;
        let z4 = zi(4, k4);
        for i in 0..6 {
            let b = base + i;
            let (s, a1, a2) = ir3(v[b], v[b + 6], v[b + 12], z4, false, f[2]);
            v[b] = s;
            v[b + 6] = a1;
            v[b + 12] = a2;
        }
    }
    lmax[4] = v.iter().map(|x| x.abs()).max().unwrap();
    // levels 3 + 2 (the mirror of pass B: level 3 then level 2, fused over groups of 9)
    for blk in 0..4 {
        let base = 162 * blk;
        let z2 = zi(2, blk);
        for i0 in 0..18 {
            let mut y = [0i32; 9];
            for s in 0..3 {
                let z3 = zi(3, 3 * blk + s);
                let b = base + 54 * s + i0;
                let (u0, u1, u2) = ir3(v[b], v[b + 18], v[b + 36], z3, false, f[3]);
                y[3 * s] = u0;
                y[3 * s + 1] = u1;
                y[3 * s + 2] = u2;
            }
            lmax[3] = lmax[3]
                .max(y[0].abs())
                .max(y[1].abs())
                .max(y[2].abs())
                .max(y[3].abs())
                .max(y[4].abs())
                .max(y[5].abs())
                .max(y[6].abs())
                .max(y[7].abs())
                .max(y[8].abs());
            for a in 0..3 {
                let (v0, v1, v2) = ir3(y[a], y[3 + a], y[6 + a], z2, false, f[4]);
                let b = base + i0 + 18 * a;
                v[b] = v0;
                v[b + 54] = v1;
                v[b + 108] = v2;
            }
        }
    }
    lmax[2] = v.iter().map(|x| x.abs()).max().unwrap();
    // levels 1 + 0: the two inverse radix-2 butterflies, then the Phi_6 recombination
    let z10i = zi(1, 0);
    let z11i = zi(1, 1);
    let half = (Q as i32 - 1) / 2;
    let mut raw = 0i32;
    let mut out = [0i32; N];
    let z6 = ParamsQ::<Q>::ZETA6 as u64;
    let det = inv_mod((2 * z6 + q - 1) % q, q);
    let ka = (det * inv_mod(162, q) % q) as u16;
    let kb = inv_mod(324, q) as u16;
    let kc = ((q - det * inv_mod(324, q) % q) % q) as u16;
    let center = |mut x: i32| {
        if x > half {
            x -= Q as i32;
        }
        if x < -half {
            x += Q as i32;
        }
        x
    };
    for i in 0..162 {
        let (c0, c1) = ir2(v[i], v[i + 162], z10i, f[5]);
        let (c2, c3) = ir2(v[i + 324], v[i + 486], z11i, f[5]);
        lmax[1] = lmax[1]
            .max(c0.abs())
            .max(c1.abs())
            .max(c2.abs())
            .max(c3.abs());
        let d0 = OpsQ::<Q>::ck(c0 - c2);
        let a1 = OpsQ::<Q>::mont(d0, ka);
        let a0 =
            OpsQ::<Q>::ck(OpsQ::<Q>::mont(OpsQ::<Q>::ck(c0 + c2), kb) + OpsQ::<Q>::mont(d0, kc));
        let d1 = OpsQ::<Q>::ck(c1 - c3);
        let b1 = OpsQ::<Q>::mont(d1, ka);
        let b0 =
            OpsQ::<Q>::ck(OpsQ::<Q>::mont(OpsQ::<Q>::ck(c1 + c3), kb) + OpsQ::<Q>::mont(d1, kc));
        raw = raw.max(a0.abs()).max(a1.abs()).max(b0.abs()).max(b1.abs());
        out[i] = center(a0);
        out[i + 162] = center(b0);
        out[i + 324] = center(a1);
        out[i + 486] = center(b1);
    }
    lmax[0] = raw;
    out
}

// ------------------------------------------------------------------ gen_quad checks

fn check_quad<const Q: u16>(cols: &[[i16; N]; 32], what: &str, worst: &mut [i32; 5]) {
    let mut b = to_batch(cols);
    unsafe { gen_quad::ntt_quad_gen_batch32::<Q>(&mut b) };
    let bound = gen_quad::output_bound(Q);
    for p in 0..32 {
        let mut lmax = [0i32; 5];
        let want_shadow = shadow_quad::<Q>(&cols[p], &mut lmax);
        for l in 0..5 {
            worst[l] = worst[l].max(lmax[l]);
        }
        let mut coeffs: Coeffs = [0u32; N];
        for j in 0..N {
            coeffs[j] = (cols[p][j] as i32).rem_euclid(Q as i32) as u32;
        }
        let want = scalar::ntt_quad::<Q>(&coeffs);
        for j in 0..N {
            let got = b.v[j][p] as i32;
            assert_eq!(
                got, want_shadow[j],
                "{what} q={Q} poly {p} slot {j}: shadow mismatch"
            );
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} slot {j}"
            );
            assert!(
                got.abs() <= bound,
                "{what} q={Q} poly {p} slot {j}: |{got}| > {bound}"
            );
        }
    }
}

fn check_quad_inv<const Q: u16>(cols: &[[i16; N]; 32], what: &str, worst: &mut [i32; 6]) {
    let mut b = to_batch(cols);
    unsafe { gen_quad::intt_quad_gen_batch32::<Q>(&mut b) };
    let half = (Q as i32 - 1) / 2;
    for p in 0..32 {
        let mut lmax = [0i32; 6];
        let want_shadow = shadow_quad_inv::<Q>(&cols[p], &mut lmax);
        for l in 0..6 {
            worst[l] = worst[l].max(lmax[l]);
        }
        let mut coeffs: Coeffs = [0u32; N];
        for j in 0..N {
            coeffs[j] = (cols[p][j] as i32).rem_euclid(Q as i32) as u32;
        }
        let want = ring::intt_quad_of(Q, &coeffs);
        for j in 0..N {
            let got = b.v[j][p] as i32;
            assert_eq!(
                got, want_shadow[j],
                "{what} q={Q} poly {p} coeff {j}: shadow mismatch"
            );
            assert!(
                got.abs() <= half,
                "{what} q={Q} poly {p} coeff {j}: |{got}| > {half}"
            );
            assert_eq!(
                got.rem_euclid(Q as i32) as u32,
                want[j],
                "{what} q={Q} poly {p} coeff {j}"
            );
        }
    }
}

fn run_quad<const Q: u16>() {
    let mut rng = Rng::new(0x3ead ^ Q as u64);
    let mut worst = [0i32; 5];
    for (i, c) in adversarial::<Q>().iter().enumerate() {
        check_quad::<Q>(c, &format!("adversarial#{i}"), &mut worst);
    }
    for i in 0..16 {
        check_quad::<Q>(&binary_cols(&mut rng), &format!("binary#{i}"), &mut worst);
    }
    for i in 0..16 {
        check_quad::<Q>(
            &random_cols::<Q>(&mut rng),
            &format!("random#{i}"),
            &mut worst,
        );
    }
    let flags = gen_quad::gen_flags(Q);
    let (claim, peak) = gen_quad::gen_model(Q, flags.0, flags.1, flags.2);
    for l in 0..5 {
        assert!(
            worst[l] <= claim[l],
            "q={Q} forward level {l}: observed {} > claimed {}",
            worst[l],
            claim[l]
        );
        assert!(
            claim[l] < 32768,
            "q={Q} forward level {l}: claimed {} exceeds i16",
            claim[l]
        );
        println!(
            "q={Q} forward level {l}: observed {} ({:.3} q), claimed {} ({:.4} q)",
            worst[l],
            worst[l] as f64 / Q as f64,
            claim[l],
            claim[l] as f64 / Q as f64
        );
    }
    println!(
        "q={Q} forward: flags {:?}, model peak {} of 32767, output bound {} = {:.4} q",
        flags,
        peak,
        gen_quad::output_bound(Q),
        gen_quad::output_bound(Q) as f64 / Q as f64
    );
}

fn run_quad_inv<const Q: u16>() {
    let mut rng = Rng::new(0x11d5 ^ Q as u64);
    let bound = gen_quad::in_bound(Q);
    let mut worst = [0i32; 6];
    for (i, c) in adversarial_ntt::<Q>(bound).iter().enumerate() {
        check_quad_inv::<Q>(c, &format!("adversarial#{i}"), &mut worst);
    }
    for i in 0..12 {
        let c = random_ntt_cols::<Q>(&mut rng, bound);
        check_quad_inv::<Q>(&c, &format!("lazy#{i}"), &mut worst);
    }
    for i in 0..12 {
        let c = random_ntt_cols::<Q>(&mut rng, (Q as i32 - 1) / 2);
        check_quad_inv::<Q>(&c, &format!("centered#{i}"), &mut worst);
    }
    // the real inputs: the forward kernel's own output, the widest transform of this tree
    for i in 0..8 {
        let cols = random_cols::<Q>(&mut rng);
        let mut b = to_batch(&cols);
        unsafe { gen_quad::ntt_quad_gen_batch32::<Q>(&mut b) };
        let ntt: [[i16; N]; 32] = std::array::from_fn(|p| std::array::from_fn(|j| b.v[j][p]));
        check_quad_inv::<Q>(&ntt, &format!("forward#{i}"), &mut worst);
    }
    let flags = gen_quad::inv_flags(Q);
    let claim = gen_quad::inv_bound(Q);
    // claim[i] = after inverse level 5-i (the model is ordered [levels 5, 4, .., 0]); the
    // shadow records lmax[l] = after level l.
    for l in (0..6).rev() {
        let c = claim[5 - l];
        assert!(
            worst[l] <= c,
            "q={Q} inverse level {l}: observed {} > claimed {c}",
            worst[l]
        );
        assert!(
            c < 32768,
            "q={Q} inverse level {l}: claimed {c} exceeds i16"
        );
        println!(
            "q={Q} inverse level {l}: observed {} ({:.3} q), claimed {c} ({:.4} q)",
            worst[l],
            worst[l] as f64 / Q as f64,
            c as f64 / Q as f64
        );
    }
    println!(
        "q={Q} inverse: flags {:?}, input bound {} = {:.3} q",
        flags,
        bound,
        bound as f64 / Q as f64
    );
}

#[test]
fn quad_forward_2917() {
    if !gate() {
        return;
    }
    run_quad::<2917>();
}

#[test]
fn quad_forward_4861() {
    if !gate() {
        return;
    }
    run_quad::<4861>();
}

#[test]
fn quad_forward_12637() {
    if !gate() {
        return;
    }
    run_quad::<12637>();
}

#[test]
fn quad_inverse_2917() {
    if !gate() {
        return;
    }
    run_quad_inv::<2917>();
}

#[test]
fn quad_inverse_4861() {
    if !gate() {
        return;
    }
    run_quad_inv::<4861>();
}

#[test]
fn quad_inverse_12637() {
    if !gate() {
        return;
    }
    run_quad_inv::<12637>();
}

/// The reduction placements the upstream module comments document, reproduced by this port's
/// `const` searches — a fidelity check on the schedule machinery itself.
#[test]
fn schedules_match_upstream() {
    // gen_quad forward: 2917 nothing; 4861 the level-4 `a0`; 12637 the pass-A inputs and
    // output plus every level.
    assert_eq!(gen_quad::gen_flags(2917), (false, false, [false; 4]));
    assert_eq!(
        gen_quad::gen_flags(4861),
        (false, false, [false, false, true, false])
    );
    assert_eq!(
        gen_quad::gen_flags(12637),
        (true, true, [true, true, true, true])
    );
    // gen_quad inverse: 2917 the level-5 inputs and the sums of levels 5 and 3 (1080
    // reductions); 4861 the level-5 inputs and the sums of levels 4 and 2 (1080); 12637
    // everything (1836).
    assert_eq!(
        gen_quad::inv_flags(2917),
        [true, true, false, true, false, false]
    );
    assert_eq!(
        gen_quad::inv_flags(4861),
        [true, false, true, false, true, false]
    );
    assert_eq!(
        gen_quad::inv_flags(12637),
        [true, true, true, true, true, true]
    );
}

// ------------------------------------------------------------------ round trips

/// `intt(ntt(x)) == x` for the centered representative of `x mod q`.
fn round_trip_large<const Q: u16>() {
    let mut rng = Rng::new(0x0d0d ^ Q as u64);
    let half = (Q as i32 - 1) / 2;
    let mut cases: Vec<[[i16; N]; 32]> = adversarial::<Q>();
    for _ in 0..8 {
        cases.push(binary_cols(&mut rng));
        cases.push(random_cols::<Q>(&mut rng));
    }
    for (c, cols) in cases.iter().enumerate() {
        let mut b = to_batch(cols);
        unsafe {
            gen_large::ntt_gen_batch32::<Q>(&mut b);
            gen_large::intt_gen_batch32::<Q>(&mut b);
        }
        for p in 0..32 {
            for j in 0..N {
                let mut want = (cols[p][j] as i32).rem_euclid(Q as i32);
                if want > half {
                    want -= Q as i32;
                }
                assert_eq!(
                    b.v[j][p] as i32, want,
                    "round trip #{c} q={Q} poly {p} coeff {j}"
                );
            }
        }
    }
}

fn round_trip_quad<const Q: u16>() {
    let mut rng = Rng::new(0x0d0d ^ Q as u64);
    let half = (Q as i32 - 1) / 2;
    let mut cases: Vec<[[i16; N]; 32]> = adversarial::<Q>();
    for _ in 0..8 {
        cases.push(binary_cols(&mut rng));
        cases.push(random_cols::<Q>(&mut rng));
    }
    for (c, cols) in cases.iter().enumerate() {
        let mut b = to_batch(cols);
        unsafe {
            gen_quad::ntt_quad_gen_batch32::<Q>(&mut b);
            gen_quad::intt_quad_gen_batch32::<Q>(&mut b);
        }
        for p in 0..32 {
            for j in 0..N {
                let mut want = (cols[p][j] as i32).rem_euclid(Q as i32);
                if want > half {
                    want -= Q as i32;
                }
                assert_eq!(
                    b.v[j][p] as i32, want,
                    "round trip #{c} q={Q} poly {p} coeff {j}"
                );
            }
        }
    }
}

#[test]
fn round_trip_17497() {
    if !gate() {
        return;
    }
    round_trip_large::<17497>();
}

#[test]
fn round_trip_19441() {
    if !gate() {
        return;
    }
    round_trip_large::<19441>();
}

#[test]
fn round_trip_2917() {
    if !gate() {
        return;
    }
    round_trip_quad::<2917>();
}

#[test]
fn round_trip_4861() {
    if !gate() {
        return;
    }
    round_trip_quad::<4861>();
}

#[test]
fn round_trip_12637() {
    if !gate() {
        return;
    }
    round_trip_quad::<12637>();
}

/// `ntt(intt(y)) == y mod q`: the forward kernel accepts the inverse's centered output
/// (`|v| <= (q-1)/2 <= q`) and reproduces the input's residues, within its declared bound.
fn forward_of_inverse_large<const Q: u16>() {
    let mut rng = Rng::new(0x2211 ^ Q as u64);
    let bound = gen_large::output_bound(Q);
    for c in 0..8 {
        let cols = random_ntt_cols::<Q>(&mut rng, gen_large::in_bound(Q));
        let mut b = to_batch(&cols);
        unsafe { gen_large::intt_gen_batch32::<Q>(&mut b) };
        for p in 0..32 {
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert!(
                    got.abs() <= (Q as i32 - 1) / 2,
                    "inverse output not centered: q={Q} #{c} poly {p} coeff {j}"
                );
            }
        }
        unsafe { gen_large::ntt_gen_batch32::<Q>(&mut b) };
        for p in 0..32 {
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert_eq!(
                    got.rem_euclid(Q as i32),
                    (cols[p][j] as i32).rem_euclid(Q as i32),
                    "forward(inverse(x)) q={Q} #{c} poly {p} slot {j}"
                );
                assert!(
                    got.abs() <= bound,
                    "forward(inverse(x)) q={Q} #{c} poly {p} slot {j}: |{got}| > {bound}"
                );
            }
        }
    }
}

fn forward_of_inverse_quad<const Q: u16>() {
    let mut rng = Rng::new(0x2211 ^ Q as u64);
    let bound = gen_quad::output_bound(Q);
    for c in 0..8 {
        let cols = random_ntt_cols::<Q>(&mut rng, gen_quad::in_bound(Q));
        let mut b = to_batch(&cols);
        unsafe { gen_quad::intt_quad_gen_batch32::<Q>(&mut b) };
        for p in 0..32 {
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert!(
                    got.abs() <= (Q as i32 - 1) / 2,
                    "inverse output not centered: q={Q} #{c} poly {p} coeff {j}"
                );
            }
        }
        unsafe { gen_quad::ntt_quad_gen_batch32::<Q>(&mut b) };
        for p in 0..32 {
            for j in 0..N {
                let got = b.v[j][p] as i32;
                assert_eq!(
                    got.rem_euclid(Q as i32),
                    (cols[p][j] as i32).rem_euclid(Q as i32),
                    "forward(inverse(x)) q={Q} #{c} poly {p} slot {j}"
                );
                assert!(
                    got.abs() <= bound,
                    "forward(inverse(x)) q={Q} #{c} poly {p} slot {j}: |{got}| > {bound}"
                );
            }
        }
    }
}

#[test]
fn forward_of_inverse_17497() {
    if !gate() {
        return;
    }
    forward_of_inverse_large::<17497>();
}

#[test]
fn forward_of_inverse_19441() {
    if !gate() {
        return;
    }
    forward_of_inverse_large::<19441>();
}

#[test]
fn forward_of_inverse_2917() {
    if !gate() {
        return;
    }
    forward_of_inverse_quad::<2917>();
}

#[test]
fn forward_of_inverse_4861() {
    if !gate() {
        return;
    }
    forward_of_inverse_quad::<4861>();
}

#[test]
fn forward_of_inverse_12637() {
    if !gate() {
        return;
    }
    forward_of_inverse_quad::<12637>();
}

/// End-to-end identity over several random batches: forward then inverse on full `Batch32`es
/// with lanes in the documented input range `|x| <= q`, every lane exactly the centered original.
fn end_to_end_identity_large<const Q: u16>(batches: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    for c in 0..batches {
        let cols = random_cols::<Q>(&mut rng);
        let mut b = to_batch(&cols);
        unsafe {
            gen_large::ntt_gen_batch32::<Q>(&mut b);
            gen_large::intt_gen_batch32::<Q>(&mut b);
        }
        let half = (Q as i32 - 1) / 2;
        for p in 0..32 {
            for j in 0..N {
                let mut want = (cols[p][j] as i32).rem_euclid(Q as i32);
                if want > half {
                    want -= Q as i32;
                }
                assert_eq!(
                    b.v[j][p] as i32, want,
                    "identity #{c} q={Q} poly {p} coeff {j}"
                );
            }
        }
    }
}

fn end_to_end_identity_quad<const Q: u16>(batches: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    for c in 0..batches {
        let cols = random_cols::<Q>(&mut rng);
        let mut b = to_batch(&cols);
        unsafe {
            gen_quad::ntt_quad_gen_batch32::<Q>(&mut b);
            gen_quad::intt_quad_gen_batch32::<Q>(&mut b);
        }
        let half = (Q as i32 - 1) / 2;
        for p in 0..32 {
            for j in 0..N {
                let mut want = (cols[p][j] as i32).rem_euclid(Q as i32);
                if want > half {
                    want -= Q as i32;
                }
                assert_eq!(
                    b.v[j][p] as i32, want,
                    "identity #{c} q={Q} poly {p} coeff {j}"
                );
            }
        }
    }
}

#[test]
fn end_to_end_identity_large_primes() {
    if !gate() {
        return;
    }
    end_to_end_identity_large::<17497>(8, 0x5EED_1A01);
    end_to_end_identity_large::<19441>(8, 0x5EED_1A02);
}

#[test]
fn end_to_end_identity_quad_primes() {
    if !gate() {
        return;
    }
    end_to_end_identity_quad::<2917>(8, 0x5EED_1A03);
    end_to_end_identity_quad::<4861>(8, 0x5EED_1A04);
    end_to_end_identity_quad::<12637>(8, 0x5EED_1A05);
}

// ------------------------------------------------------------------ timing

/// Median of 9 runs of each kernel on one full batch (the copies that reset the batch run
/// outside the timed region, so the measurement is the transform alone). The warmup needs no
/// reset: the forward output is a valid inverse input and the inverse output a valid forward
/// input (centered, |x| <= q), so the two kernels can chase each other in place.
fn time_batch_large<const Q: u16>() {
    let mut rng = Rng::new(0x7107 ^ Q as u64);
    let saved = to_batch(&random_cols::<Q>(&mut rng));
    let mut b = saved;
    for _ in 0..3 {
        unsafe { gen_large::ntt_gen_batch32::<Q>(&mut b) };
        unsafe { gen_large::intt_gen_batch32::<Q>(&mut b) };
    }
    let mut fwd = [std::time::Duration::ZERO; 9];
    for i in 0..9 {
        b = saved;
        let t0 = Instant::now();
        unsafe { gen_large::ntt_gen_batch32::<Q>(&mut b) };
        fwd[i] = t0.elapsed();
    }
    let mut ntt_saved = saved;
    unsafe { gen_large::ntt_gen_batch32::<Q>(&mut ntt_saved) };
    let mut inv = [std::time::Duration::ZERO; 9];
    for i in 0..9 {
        b = ntt_saved;
        let t0 = Instant::now();
        unsafe { gen_large::intt_gen_batch32::<Q>(&mut b) };
        inv[i] = t0.elapsed();
    }
    fwd.sort();
    inv.sort();
    println!(
        "gen_large q={Q} (32 polys x {N} slots): forward median {:.1} us/batch, inverse median {:.1} us/batch",
        fwd[4].as_secs_f64() * 1e6,
        inv[4].as_secs_f64() * 1e6
    );
}

fn time_batch_quad<const Q: u16>() {
    let mut rng = Rng::new(0x7107 ^ Q as u64);
    let saved = to_batch(&random_cols::<Q>(&mut rng));
    let mut b = saved;
    for _ in 0..3 {
        unsafe { gen_quad::ntt_quad_gen_batch32::<Q>(&mut b) };
        unsafe { gen_quad::intt_quad_gen_batch32::<Q>(&mut b) };
    }
    let mut fwd = [std::time::Duration::ZERO; 9];
    for i in 0..9 {
        b = saved;
        let t0 = Instant::now();
        unsafe { gen_quad::ntt_quad_gen_batch32::<Q>(&mut b) };
        fwd[i] = t0.elapsed();
    }
    let mut ntt_saved = saved;
    unsafe { gen_quad::ntt_quad_gen_batch32::<Q>(&mut ntt_saved) };
    let mut inv = [std::time::Duration::ZERO; 9];
    for i in 0..9 {
        b = ntt_saved;
        let t0 = Instant::now();
        unsafe { gen_quad::intt_quad_gen_batch32::<Q>(&mut b) };
        inv[i] = t0.elapsed();
    }
    fwd.sort();
    inv.sort();
    println!(
        "gen_quad q={Q} (32 polys x {N} slots): forward median {:.1} us/batch, inverse median {:.1} us/batch",
        fwd[4].as_secs_f64() * 1e6,
        inv[4].as_secs_f64() * 1e6
    );
}

#[test]
fn timing() {
    if !gate() {
        return;
    }
    time_batch_large::<17497>();
    time_batch_large::<19441>();
    time_batch_quad::<2917>();
    time_batch_quad::<4861>();
    time_batch_quad::<12637>();
}
