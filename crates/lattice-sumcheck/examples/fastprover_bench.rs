//! Sumcheck prover benchmark: the baseline linear-time engine vs the
//! round-batched evaluation-grid fast prover (ePrint 2026/587 §5 /
//! 2025/1117 §4–5), with multiplication-count instrumentation.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::fastprover::{prove_fast_with_opts, take_last_stats, FastProverOpts};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

fn rand_mle(num_vars: usize, seed: &mut u64, bound: u64) -> DenseMle {
    let mut x = *seed | 1;
    let evals: Vec<Goldilocks> = (0..(1usize << num_vars))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            fe(x % bound)
        })
        .collect();
    *seed = x;
    DenseMle::new(evals).ok().unwrap()
}

fn bench(num_vars: usize, factors_per_term: usize, tag: &str) {
    let mut seed = 0xDEAD_BEEFu64;
    let mut vp = VirtualPolynomial::new(num_vars);
    for _ in 0..factors_per_term {
        let f = rand_mle(num_vars, &mut seed, 1 << 20);
        vp.add_factor(f).ok().unwrap();
    }
    vp.add_term(fe(1), (0..factors_per_term).collect()).ok().unwrap();
    let n = 1usize << num_vars;
    let mut claim = Goldilocks::ZERO;
    for i in 0..n {
        let mut prod = fe(1);
        for f in &vp.factors {
            prod = prod.mul(&f.evaluations[i]);
        }
        claim = claim.add(&prod);
    }

    // Baseline.
    let t0 = std::time::Instant::now();
    let mut t1 = Transcript::new_default(b"bench");
    let out1 = sumcheck::prove(&vp, claim, &mut t1).ok().unwrap();
    let base_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Fast, per window.
    print!(
        "{tag}: vars={num_vars} d={factors_per_term} M={n} | baseline {base_ms:.1} ms"
    );
    for window in [1usize, 2, 3, 4] {
        if window > num_vars {
            continue;
        }
        let t0 = std::time::Instant::now();
        let mut t2 = Transcript::new_default(b"bench");
        let opts = FastProverOpts { window, collect_stats: true };
        let out2 = prove_fast_with_opts(&vp, claim, &mut t2, &opts, &[], &[])
            .ok()
            .unwrap();
        let fast_ms = t0.elapsed().as_secs_f64() * 1e3;
        let identical = out1.proof == out2.proof;
        let stats = take_last_stats().unwrap();
        println!(
            " | w={window}: {fast_ms:.1} ms ({:.2}x) identical={identical} bb={} sb={} grid(bb={},sb={})",
            base_ms / fast_ms,
            stats.bb_mults,
            stats.sb_mults,
            stats.grid.bb_mults,
            stats.grid.sb_mults
        );
    }
    println!();
}

fn main() {
    bench(14, 2, "d2");
    bench(14, 3, "d3");
    bench(12, 4, "d4");
    bench(10, 8, "d8");
    bench(8, 16, "d16");
}
