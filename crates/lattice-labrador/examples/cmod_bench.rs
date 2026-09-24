//! Reduction and product benchmarks for the LaBRADOR ring: the folded `cmod` against the
//! division form, and the negacyclic product / multi-product inner product the prover's
//! commitments are made of.
//!
//! `Q = 2^48 - 59` gives `2^48 ≡ 59 (mod Q)`, so every reduction is three-to-four
//! shift-and-multiply folds instead of an i128 `rem_euclid` (~40 cycles of division
//! latency each). The exactness is checked against the division form by
//! `lattice_labrador::ring::tests` and the round trip by `lattice-labinius`'s
//! `labrador_round_trip`.

use lattice_labrador::ring::{cmod, cmod_div, Poly};
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn bench<F: FnMut()>(mut f: F, runs: usize) -> f64 {
    let mut ts = vec![0f64; runs];
    for t in ts.iter_mut() {
        let s = Instant::now();
        f();
        *t = s.elapsed().as_secs_f64();
    }
    median(&mut ts)
}

fn main() {
    let runs = 7;
    // pseudo-random i128 magnitudes up to the sprod accumulator's worst case
    let mut r = 0x9E37_79B9_7F4A_7C15i128;
    let mut xs = [0i128; 4096];
    for x in xs.iter_mut() {
        r = r
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = (r >> 20) % (1i128 << 100);
    }
    {
        let mut acc = 0i64;
        let fold = bench(
            || {
                for &x in xs.iter() {
                    acc = acc.wrapping_add(cmod(x));
                }
            },
            runs,
        );
        let div = bench(
            || {
                for &x in xs.iter() {
                    acc = acc.wrapping_add(cmod_div(x));
                }
            },
            runs,
        );
        println!(
            "  cmod x4096      fold {:>9.3} ms   div {:>9.3} ms   speedup {:>5.1}x   (acc {acc})",
            fold * 1e3,
            div * 1e3,
            div / fold
        );
    }
    // the negacyclic product: 4096 i128 mults + 64 reductions
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut elt = || {
        let mut p = [0i64; 64];
        for c in p.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *c = ((seed >> 16) % (1 << 47)) as i64 - (1 << 46);
        }
        Poly(p)
    };
    let a: Vec<Poly> = (0..16).map(|_| elt()).collect();
    let b: Vec<Poly> = (0..16).map(|_| elt()).collect();
    {
        let mut acc = Poly::zero();
        let mul = bench(
            || {
                for i in 0..16 {
                    acc = a[i].mul(&b[i]);
                }
            },
            runs,
        );
        let sprod = bench(
            || {
                let _ = Poly::sprod(&a, &b);
            },
            runs,
        );
        println!(
            "  Poly::mul x16        {:>9.3} ms   (64x64 negacyclic each)",
            mul * 1e3
        );
        println!(
            "  Poly::sprod x16      {:>9.3} ms   (16-element inner product)",
            sprod * 1e3
        );
        let _ = acc;
    }
}
