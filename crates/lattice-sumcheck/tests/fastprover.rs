//! Tests for the fast (round-batched, evaluation-grid) sumcheck prover.
//!
//! The contract: `prove_fast` produces **byte-identical transcripts and
//! proofs** to the baseline `sumcheck::prove` (the same polynomial, the
//! same Fiat–Shamir flow), while restructuring the prover's work per
//! ePrint 2026/587 §5 / 2025/1117 §4–5.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::fastprover::{
    prove_fast, prove_fast_with_eq, prove_fast_with_opts, take_last_stats, FastProverOpts,
};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// Deterministic pseudo-random MLE with small values.
fn small_mle(num_vars: usize, seed: u64, bound: u64) -> DenseMle {
    let mut x = seed | 1;
    let evals: Vec<Goldilocks> = (0..(1usize << num_vars))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            fe(x % bound)
        })
        .collect();
    DenseMle::new(evals).ok().unwrap()
}

struct VpSpec {
    num_vars: usize,
    /// Terms: (coefficient, number of factors).
    terms: Vec<(u64, usize)>,
    seed: u64,
}

fn build_vp(spec: &VpSpec) -> VirtualPolynomial {
    let mut vp = VirtualPolynomial::new(spec.num_vars);
    let mut s = spec.seed;
    let mut starts = Vec::new();
    let mut start = 0usize;
    for &(coeff, nf) in &spec.terms {
        let mut ids = Vec::with_capacity(nf);
        for _ in 0..nf {
            s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) + 1;
            let idx = vp.add_factor(small_mle(spec.num_vars, s, 32)).ok().unwrap();
            ids.push(idx);
        }
        vp.add_term(fe(coeff), ids.clone()).ok().unwrap();
        starts.push(start);
        start += nf;
    }
    vp
}

fn vp_sum(vp: &VirtualPolynomial) -> Goldilocks {
    // Σ_{x ∈ {0,1}^ℓ} Σ_term c·Π_k f_k(x) — the honest sumcheck claim,
    // computed directly over the hypercube.
    let n = 1usize << vp.num_vars;
    let mut total = Goldilocks::ZERO;
    for x in 0..n {
        for (coeff, ids) in &vp.terms {
            let mut prod = *coeff;
            for &k in ids {
                prod = prod.mul(&vp.factors[k].evaluations[x]);
            }
            total = total.add(&prod);
        }
    }
    total
}

fn transcripts_equal(a: &mut Transcript, b: &mut Transcript) -> bool {
    let ca = a.challenge_field(b"ident").ok().unwrap();
    let cb = b.challenge_field(b"ident").ok().unwrap();
    ca == cb
}

#[test]
fn fast_prover_bit_identical_to_baseline() {
    let specs = vec![
        VpSpec {
            num_vars: 6,
            terms: vec![(1, 2)],
            seed: 0x1111,
        },
        VpSpec {
            num_vars: 6,
            terms: vec![(3, 3)],
            seed: 0x2222,
        },
        VpSpec {
            num_vars: 7,
            terms: vec![(1, 4)],
            seed: 0x3333,
        },
        VpSpec {
            num_vars: 8,
            terms: vec![(5, 2), (7, 3)],
            seed: 0x4444,
        },
        VpSpec {
            num_vars: 5,
            terms: vec![(2, 5)],
            seed: 0x5555,
        },
    ];
    for spec in &specs {
        let vp = build_vp(spec);
        let claim = vp_sum(&vp);
        for window in 0..=3usize {
            let mut t1 = Transcript::new_default(b"sc-ident");
            let out1 = sumcheck::prove(&vp, claim, &mut t1).ok().unwrap();
            let mut t2 = Transcript::new_default(b"sc-ident");
            let opts = FastProverOpts {
                window,
                collect_stats: false,
            };
            let out2 = prove_fast_with_opts(&vp, claim, &mut t2, &opts, &[], &[])
                .ok()
                .unwrap();
            assert_eq!(
                out1.proof, out2.proof,
                "proof mismatch: vars={} terms={:?} window={window}",
                spec.num_vars, spec.terms
            );
            assert_eq!(out1.challenges, out2.challenges, "challenges mismatch");
            assert_eq!(out1.final_claim, out2.final_claim);
            assert_eq!(out1.factor_claims, out2.factor_claims);
            assert!(transcripts_equal(&mut t1, &mut t2));
        }
    }
}

#[test]
fn fast_prover_zerocheck_style_with_eq_split() {
    // The zerocheck shape: P(x)·eq(r, x) with the eq factor split-
    // optimised (never materialised in the fast path).
    let num_vars = 7usize;
    let poly = small_mle(num_vars, 0xAAAA, 8);
    let r: Vec<Goldilocks> = (0..num_vars)
        .map(|i| fe((i as u64 * 2654435761 + 7) % (1 << 20)))
        .collect();
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(num_vars);
    let pi = vp.add_factor(poly.clone()).ok().unwrap();
    let ei = vp.add_factor(eq).ok().unwrap();
    vp.add_term(fe(1), vec![pi, ei]).ok().unwrap();
    let claim = poly.evaluate(&r).ok().unwrap();

    // Baseline (materialised eq table).
    let mut t1 = Transcript::new_default(b"sc-eq");
    let out1 = sumcheck::prove(&vp, claim, &mut t1).ok().unwrap();

    // Fast with the eq factor split: pass the eq factor's index + w = r.
    let mut t2 = Transcript::new_default(b"sc-eq");
    let opts = FastProverOpts {
        window: 3,
        collect_stats: false,
    };
    let out2 = prove_fast_with_eq(&vp, claim, &mut t2, &opts, &[ei], &r)
        .ok()
        .unwrap();
    assert_eq!(out1.proof, out2.proof, "eq-split proof mismatch");
    assert_eq!(out1.challenges, out2.challenges);
    assert_eq!(out1.factor_claims, out2.factor_claims);
}

#[test]
fn fast_prover_window_zero_is_optimized_tail() {
    let spec = VpSpec {
        num_vars: 6,
        terms: vec![(1, 3)],
        seed: 0x6666,
    };
    let vp = build_vp(&spec);
    let claim = vp_sum(&vp);
    let mut t1 = Transcript::new_default(b"sc-tail");
    let out1 = sumcheck::prove(&vp, claim, &mut t1).ok().unwrap();
    let mut t2 = Transcript::new_default(b"sc-tail");
    let opts = FastProverOpts {
        window: 0,
        collect_stats: false,
    };
    let out2 = prove_fast_with_opts(&vp, claim, &mut t2, &opts, &[], &[])
        .ok()
        .unwrap();
    assert_eq!(out1.proof, out2.proof);
}

#[test]
fn fast_prover_stats_reported() {
    let spec = VpSpec {
        num_vars: 8,
        terms: vec![(1, 3)],
        seed: 0x7777,
    };
    let vp = build_vp(&spec);
    let claim = vp_sum(&vp);
    let mut t = Transcript::new_default(b"sc-stats");
    let opts = FastProverOpts {
        window: 3,
        collect_stats: true,
    };
    prove_fast_with_opts(&vp, claim, &mut t, &opts, &[], &[])
        .ok()
        .unwrap();
    let stats = take_last_stats().unwrap();
    assert!(stats.bb_mults > 0);
    assert!(stats.grid.bb_mults > 0 || stats.grid.sb_mults > 0);
}

#[test]
fn fast_prover_default_drop_in() {
    let spec = VpSpec {
        num_vars: 6,
        terms: vec![(9, 2)],
        seed: 0x8888,
    };
    let vp = build_vp(&spec);
    let claim = vp_sum(&vp);
    let mut t1 = Transcript::new_default(b"sc-drop");
    let out1 = sumcheck::prove(&vp, claim, &mut t1).ok().unwrap();
    let mut t2 = Transcript::new_default(b"sc-drop");
    let out2 = prove_fast(&vp, claim, &mut t2).ok().unwrap();
    assert_eq!(out1.proof, out2.proof);
}
