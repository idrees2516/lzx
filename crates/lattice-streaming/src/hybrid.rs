//! **The hybrid prover** — the paper's space/time switch (§1.2 item 5,
//! §3.1.2's closing remark):
//!
//! > "the space required by the standard linear-time sum-check proving
//! > algorithm halves in each round. Hence, after enough rounds have
//! > passed, the prover [can] switch from a small-space prover
//! > implementation to the standard linear-time prover implementation,
//! > without increasing its space requirements. This saves a meaningful
//! > constant factor in prover time, while keeping the prover's space
//! > bounded by, e.g., `T^{1/2}`."
//!
//! With a memory budget of `2^c` field elements per factor:
//!
//! * rounds `0..n−c` run as Algorithm-1 sweeps — `O(n + ℓ²)` space,
//!   `O(ℓ·2^n)` oracle queries each (checkpointed regeneration serves
//!   them client-side);
//! * then **one materializing pass** builds the bound arrays at size
//!   `2^c`: `bound_f[m] = Σ_{j∈{0,1}^{n−c}} eq(r_{<n−c}, tobits(j))·f[(j, m)]`
//!   — a single sequential walk with a Gray-coded eq weight (Claim 3.2
//!   read as a stream transformation);
//! * the remaining `c` rounds run the standard in-memory linear-time
//!   algorithm (the Boolean binding `a + r·(b−a)` and SIMD term sums).
//!
//! Peak prover space: `O(ℓ·2^c + n)` field elements. `c = n/2` is the
//! paper's `O(√T)` regime; `c = n` degenerates to the in-memory engine;
//! `c = 0` to Algorithm 1. Round messages stay bit-identical to both.

// Index-arithmetic loops (MSB-first bit extraction, limb walks,
// prefix/suffix products) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]
use crate::oracle::IndexOracle;
use crate::small_space::{
    interpolate_nodes, round_evals_sweep, terminal_claims_sweep, EqWalk, SmallSpaceError,
    SmallSpaceOutput,
};
use lattice_core::field_simd::{self, Sum8};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;

/// Prove `Σ_{x∈{0,1}^n} Σ_j c_j·Π_k f_{j,k}(x) = claim` with a
/// `2^c`-element-per-factor memory budget.
pub fn prove_hybrid(
    num_vars: usize,
    terms: &[(Goldilocks, Vec<usize>)],
    oracles: &mut [&mut dyn IndexOracle],
    claim: Goldilocks,
    c: usize,
    transcript: &mut Transcript,
) -> Result<SmallSpaceOutput, SmallSpaceError> {
    let n = num_vars;
    let ell = oracles.len();
    let d = terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1);
    if ell == 0 {
        return Err(SmallSpaceError::BadShape {
            expected: 1,
            got: 0,
        });
    }
    if oracles.iter().any(|o| o.len() != 1u64 << n) {
        return Err(SmallSpaceError::BadShape {
            expected: 1 << n,
            got: oracles[0].len() as usize,
        });
    }
    let streamed_rounds = n.saturating_sub(c);
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut bound_r: Vec<Goldilocks> = Vec::with_capacity(n);

    // ------------------------------------------------------------------
    // Phase A: Algorithm-1 sweeps, O(n + ℓ²) space.
    // ------------------------------------------------------------------
    for _ in 0..streamed_rounds {
        let evals_at = round_evals_sweep(oracles, terms, &bound_r, n, d);
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(SmallSpaceError::RoundCheckFailed {
                round: bound_r.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(SmallSpaceError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(SmallSpaceError::Transcript)?;
        current_claim = interpolate_nodes(&evals_at, &r);
        bound_r.push(r);
        rounds.push(evals_at);
        challenges.push(r);
    }

    // ------------------------------------------------------------------
    // Phase B: materialize the bound arrays at size 2^c — one sequential
    // pass per factor with a Gray-coded eq walk over the bound prefix.
    // Space: ℓ · 2^c (the caller's budget).
    // ------------------------------------------------------------------
    let c_eff = n - streamed_rounds;
    let material_len = 1usize << c_eff;
    let mut bound: Vec<Vec<Goldilocks>> = Vec::with_capacity(ell);
    if c_eff > 0 {
        for fi in 0..ell {
            let mut acc = vec![Goldilocks::ZERO; material_len];
            let mut walk = EqWalk::new(&bound_r);
            for j in 0..(1u64 << streamed_rounds) {
                let w = walk.weight();
                let base = j << c_eff;
                for m in 0..material_len {
                    let v = oracles[fi].eval(base + m as u64);
                    acc[m] = acc[m].add(&w.mul(&v));
                }
                if j + 1 < (1u64 << streamed_rounds) {
                    walk.advance();
                }
            }
            bound.push(acc);
        }
    } else {
        // c = 0: no in-memory rounds; terminal claims via the sweep.
        let factor_claims = terminal_claims_sweep(oracles, &bound_r, n);
        let mut final_claim = Goldilocks::ZERO;
        for (cf, ids) in terms {
            let mut prod = *cf;
            for fi in ids {
                prod = prod.mul(&factor_claims[*fi]);
            }
            final_claim = final_claim.add(&prod);
        }
        if final_claim != current_claim {
            return Err(SmallSpaceError::ClaimMismatch);
        }
        return Ok(SmallSpaceOutput {
            rounds,
            challenges,
            final_claim,
            factor_claims,
        });
    }

    // ------------------------------------------------------------------
    // Phase C: in-memory linear-time rounds (the standard algorithm).
    // ------------------------------------------------------------------
    let n_points = d + 1;
    for _ in 0..c_eff {
        let cur_len = bound[0].len();
        let half = cur_len / 2;
        let mut evals_at = vec![Goldilocks::ZERO; n_points];
        // Evaluate the round polynomial at nodes 0..=d.
        for (pi, t) in (0..=d as u64).enumerate() {
            let tf = Goldilocks::from_u64(t);
            // Per-factor half-bound values at X = t.
            let mut vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(ell);
            for f in &bound {
                let mut v = vec![Goldilocks::ZERO; half];
                if t == 0 {
                    v.copy_from_slice(&f[..half]);
                } else if t == 1 {
                    v.copy_from_slice(&f[half..]);
                } else {
                    // f_t(p) = f[p] + t·(f[p+half] − f[p]).
                    field_simd::bind_half_slices(&f[..half], &f[half..], tf, &mut v);
                }
                vals.push(v);
            }
            let slices: Vec<&[Goldilocks]> = vals.iter().map(|v| v.as_slice()).collect();
            let mut acc = Sum8::new();
            let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
            for (cf, ids) in terms {
                fslices.clear();
                fslices.extend(ids.iter().map(|fi| slices[*fi]));
                acc.accumulate_term(*cf, &fslices);
            }
            evals_at[pi] = acc.finish();
        }
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(SmallSpaceError::RoundCheckFailed {
                round: bound_r.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(SmallSpaceError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(SmallSpaceError::Transcript)?;
        current_claim = interpolate_nodes(&evals_at, &r);
        bound_r.push(r);
        challenges.push(r);
        // Standard in-place Boolean binding of every factor.
        for f in bound.iter_mut() {
            field_simd::bind_first_half_in_place(f, r);
            f.truncate(half);
        }
        rounds.push(evals_at);
    }

    // Terminal claims from the shrunken arrays.
    let factor_claims: Vec<Goldilocks> = bound.iter().map(|f| f[0]).collect();
    let mut final_claim = Goldilocks::ZERO;
    for (cf, ids) in terms {
        let mut prod = *cf;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(SmallSpaceError::ClaimMismatch);
    }

    Ok(SmallSpaceOutput {
        rounds,
        challenges,
        final_claim,
        factor_claims,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use crate::small_space::prove_small_space;
    use crate::small_space::SmallSpaceInstance;
    use lattice_core::DenseMle;

    /// Every budget `c` produces bit-identical round messages to both the
    /// fully-streamed Algorithm 1 and the in-memory engine.
    #[test]
    fn hybrid_matches_all_budgets() {
        let n = 7;
        let df = DenseMle::random(n, b"h-f");
        let dh = DenseMle::random(n, b"h-h");
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| df.evaluations[i].mul(&dh.evaluations[i]))
            .fold(Goldilocks::ZERO, |a, v| a.add(&v));
        let terms = vec![(Goldilocks::ONE, vec![0usize, 1])];

        // Reference: fully streamed.
        let mut o0 = OwnedOracle::new(df.evaluations.clone());
        let mut o1 = OwnedOracle::new(dh.evaluations.clone());
        let mut inst = SmallSpaceInstance {
            num_vars: n,
            factors: vec![&mut o0, &mut o1],
            terms: terms.clone(),
        };
        let mut ts = Transcript::new_default(b"h-seed");
        let reference = prove_small_space(&mut inst, claim, &mut ts).unwrap();

        for c in [0usize, 1, 3, 5, 7] {
            let mut o0 = OwnedOracle::new(df.evaluations.clone());
            let mut o1 = OwnedOracle::new(dh.evaluations.clone());
            let mut oracles: [&mut dyn IndexOracle; 2] = [&mut o0, &mut o1];
            let mut ts = Transcript::new_default(b"h-seed");
            let out = prove_hybrid(n, &terms, &mut oracles, claim, c, &mut ts).unwrap();
            assert_eq!(out.rounds, reference.rounds, "rounds at c={c}");
            assert_eq!(out.challenges, reference.challenges, "challenges at c={c}");
            assert_eq!(out.final_claim, reference.final_claim, "final at c={c}");
            assert_eq!(
                out.factor_claims, reference.factor_claims,
                "claims at c={c}"
            );
        }
    }

    /// A degree-3 mixed-term instance roundtrips at every budget.
    #[test]
    fn hybrid_mixed_degree() {
        let n = 6;
        let df0 = DenseMle::random(n, b"x-f0");
        let df1 = DenseMle::random(n, b"x-f1");
        let df2 = DenseMle::random(n, b"x-f2");
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| {
                Goldilocks::from_u64(2)
                    .mul(
                        &df0.evaluations[i]
                            .mul(&df1.evaluations[i])
                            .mul(&df2.evaluations[i]),
                    )
                    .add(&Goldilocks::from_u64(7).mul(&df0.evaluations[i]))
            })
            .fold(Goldilocks::ZERO, |a, v| a.add(&v));
        let terms = vec![
            (Goldilocks::from_u64(2), vec![0usize, 1, 2]),
            (Goldilocks::from_u64(7), vec![0usize]),
        ];
        for c in [0usize, 2, 4, 6] {
            let mut o0 = OwnedOracle::new(df0.evaluations.clone());
            let mut o1 = OwnedOracle::new(df1.evaluations.clone());
            let mut o2 = OwnedOracle::new(df2.evaluations.clone());
            let mut oracles: [&mut dyn IndexOracle; 3] = [&mut o0, &mut o1, &mut o2];
            let mut ts = Transcript::new_default(b"x-seed");
            let out = prove_hybrid(n, &terms, &mut oracles, claim, c, &mut ts).unwrap();
            // Terminal claims equal direct MLE evaluations at the point.
            assert_eq!(
                out.factor_claims[0],
                df0.evaluate(&out.challenges).unwrap(),
                "f0 at c={c}"
            );
            assert_eq!(
                out.factor_claims[1],
                df1.evaluate(&out.challenges).unwrap(),
                "f1 at c={c}"
            );
        }
    }
}
