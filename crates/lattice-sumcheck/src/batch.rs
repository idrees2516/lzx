//! Claim batching: combine `k` sumcheck statements into one via a random
//! linear combination sampled from the transcript (SALSA's batched
//! verification; Jolt's multi-claim stage fusion).
//!
//! Given claims `Σ P_i = c_i`, sample `ρ^0..ρ^{k-1}` and prove
//! `Σ (Σ_i ρ^i P_i) = Σ_i ρ^i c_i` in a single sumcheck. The combined
//! virtual polynomial shares the union of factor indices across claims,
//! and the returned per-factor claims map back to each input polynomial's
//! factors — so one PCS opening round authenticates every original claim.

use crate::sumcheck::{self, SumcheckError, SumcheckOutput, SumcheckProof};
use crate::virtual_poly::{VirtualPolyError, VirtualPolynomial};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    Sumcheck(SumcheckError),
    VariableCountMismatch { expected: usize, got: usize },
    EmptyBatch,
    ChallengeCount,
}

/// A single claim: virtual polynomial + asserted sum.
pub struct BatchClaim<'a> {
    pub poly: &'a VirtualPolynomial,
    pub claimed_sum: Goldilocks,
}

/// Prove a batch of claims in one sumcheck. All polynomials must share the
/// same variable count. Returns the combined proof, per-claim challenges
/// (shared), and combined factor claims in *global* factor indexing.
pub fn prove_batch<'a>(
    claims: &[BatchClaim<'a>],
    transcript: &mut Transcript,
) -> Result<(SumcheckOutput, Vec<Goldilocks>), BatchError> {
    if claims.is_empty() {
        return Err(BatchError::EmptyBatch);
    }
    let num_vars = claims[0].poly.num_vars;
    if claims.iter().any(|c| c.poly.num_vars != num_vars) {
        return Err(BatchError::VariableCountMismatch {
            expected: num_vars,
            got: claims.iter().map(|c| c.poly.num_vars).max().unwrap_or(0),
        });
    }
    // Combined challenges rho^i.
    let rhos = transcript
        .challenge_fields(b"batch-rho", claims.len())
        .map_err(|_| BatchError::ChallengeCount)?;

    // Build the combined virtual polynomial with a global factor pool.
    let mut combined = VirtualPolynomial::new(num_vars);
    let mut global_factor_claims_map: Vec<(usize, usize)> = Vec::new(); // (claim_idx, local factor idx)
    let mut combined_claim = Goldilocks::ZERO;
    for (ci, claim) in claims.iter().enumerate() {
        let rho = rhos[ci];
        combined_claim = combined_claim.add(&rho.mul(&claim.claimed_sum));
        for factor in &claim.poly.factors {
            let gi = combined.add_factor(factor.clone()).map_err(var_count_err)?;
            global_factor_claims_map.push((ci, gi));
        }
    }
    for (ci, claim) in claims.iter().enumerate() {
        let rho = rhos[ci];
        let base: usize = claims.iter().take(ci).map(|c| c.poly.factors.len()).sum();
        for (coeff, ids) in &claim.poly.terms {
            let global_ids: Vec<usize> = ids.iter().map(|li| base + li).collect();
            combined.add_term(rho.mul(coeff), global_ids).map_err(|_| {
                BatchError::Sumcheck(SumcheckError::VirtualPoly(VirtualPolyError::EmptyProduct))
            })?;
        }
    }

    let out =
        sumcheck::prove(&combined, combined_claim, transcript).map_err(BatchError::Sumcheck)?;
    Ok((out, rhos))
}

fn var_count_err(e: VirtualPolyError) -> BatchError {
    match e {
        VirtualPolyError::VariableCountMismatch { expected, got } => {
            BatchError::VariableCountMismatch { expected, got }
        }
        other => BatchError::Sumcheck(SumcheckError::VirtualPoly(other)),
    }
}

// ---------------------------------------------------------------------------
// SALSAA D3 — Π^batch* : row-count-preserving batch folding (Theorem 4)
// ---------------------------------------------------------------------------

/// SALSAA D3 (ePrint 2025/2124, Theorem 4): batch `m` sumcheck statements
/// into ONE sumcheck whose round-message count (the "row count") is the
/// MAXIMUM variable count across claims — never the sum — so batching `m`
/// legs costs `m` data sweeps but only `max(num_vars)` rounds, and claims
/// with heterogeneous variable counts are padded instead of rejected.
///
/// Padding rule: a factor over `r` variables lifts to `r + k` variables by
/// ignoring the last `k` bits — the evaluation table duplicates every entry
/// `2^k` times (`f'[j·2^k + s] = f[j]`), so the hypercube sum scales by
/// exactly `2^k` and the padded claim is `2^k · c`. The lifted factor's MLE
/// at any point `(p_1..p_{r+k})` equals the original factor's MLE at
/// `(p_1..p_r)`, so per-claim evaluation claims map straight back through.
#[derive(Clone, Debug)]
pub struct BatchStarOutput {
    /// The single combined sumcheck proof (row count = max num_vars).
    pub proof: SumcheckProof,
    /// The shared challenge point (max num_vars entries).
    pub challenges: Vec<Goldilocks>,
    /// The rho combiners `ρ^0..ρ^{m-1}`.
    pub rhos: Vec<Goldilocks>,
    /// Per-claim evaluation claims for the PCS layer, in each claim's own
    /// factor indexing, evaluated at that claim's (truncated) point.
    pub per_claim: Vec<PerClaimClaims>,
}

/// Per-claim PCS claims produced by `prove_batch_star`.
#[derive(Clone, Debug)]
pub struct PerClaimClaims {
    /// The claim's original (unpadded) asserted sum.
    pub claimed_sum: Goldilocks,
    /// The claim's variable count (its slice of the shared point).
    pub num_vars: usize,
    /// The claim's factor-indexed evaluation claims at its point.
    pub factor_claims: Vec<Goldilocks>,
    /// The claim's evaluation point (the first `num_vars` shared challenges).
    pub point: Vec<Goldilocks>,
}

/// Lift a factor from `r` to `r + k` variables by ignoring the last `k`
/// bits: every evaluation entry is duplicated `2^k` times consecutively.
fn pad_factor(factor: &DenseMle, k: usize) -> DenseMle {
    if k == 0 {
        return factor.clone();
    }
    let dup = 1usize << k;
    let mut evals = Vec::with_capacity(factor.evaluations.len() * dup);
    for e in &factor.evaluations {
        for _ in 0..dup {
            evals.push(*e);
        }
    }
    DenseMle {
        num_vars: factor.num_vars + k,
        evaluations: evals,
    }
}

/// 2^k as a Goldilocks element (exact: the modulus exceeds 2^32 for k ≤ 31
/// only when k is small — computed by repeated doubling with early gate).
fn pow2(k: usize) -> Goldilocks {
    let mut v = Goldilocks::ONE;
    for _ in 0..k {
        v = v.add(&v);
    }
    v
}

/// Prove `m` claims — possibly with different variable counts — in ONE
/// row-count-preserving sumcheck. The transcript must already contain the
/// statement material.
pub fn prove_batch_star<'a>(
    claims: &[BatchClaim<'a>],
    transcript: &mut Transcript,
) -> Result<BatchStarOutput, BatchError> {
    if claims.is_empty() {
        return Err(BatchError::EmptyBatch);
    }
    let num_vars = claims
        .iter()
        .map(|c| c.poly.num_vars)
        .max()
        .ok_or(BatchError::EmptyBatch)?;
    let rhos = transcript
        .challenge_fields(b"batch-star-rho", claims.len())
        .map_err(|_| BatchError::ChallengeCount)?;

    // Combined polynomial over the union factor pool, every claim padded
    // to num_vars (row-count preservation: the pad is zero-allocation in
    // the transcript — the round count is exactly num_vars).
    let mut combined = VirtualPolynomial::new(num_vars);
    let mut factor_base: Vec<usize> = Vec::with_capacity(claims.len() + 1);
    let mut combined_claim = Goldilocks::ZERO;
    for (ci, claim) in claims.iter().enumerate() {
        let rho = rhos[ci];
        let k = num_vars - claim.poly.num_vars;
        // Padded claim: the ignored k variables double the sum k times.
        let scale = pow2(k);
        combined_claim = combined_claim.add(&rho.mul(&scale).mul(&claim.claimed_sum));
        factor_base.push(combined.factors.len());
        for factor in &claim.poly.factors {
            combined
                .add_factor(pad_factor(factor, k))
                .map_err(var_count_err)?;
        }
        for (coeff, ids) in &claim.poly.terms {
            let base = factor_base[ci];
            let global_ids: Vec<usize> = ids.iter().map(|li| base + li).collect();
            combined.add_term(rho.mul(coeff), global_ids).map_err(|_| {
                BatchError::Sumcheck(SumcheckError::VirtualPoly(VirtualPolyError::EmptyProduct))
            })?;
        }
    }

    let out =
        sumcheck::prove(&combined, combined_claim, transcript).map_err(BatchError::Sumcheck)?;
    // Map the combined factor claims back per claim (padded factors bind to
    // the original value at the truncated point — every duplicate is equal,
    // so the constant after binding is the original MLE value).
    let mut per_claim = Vec::with_capacity(claims.len());
    for (ci, claim) in claims.iter().enumerate() {
        let base = factor_base[ci];
        let n = claim.poly.factors.len();
        per_claim.push(PerClaimClaims {
            claimed_sum: claim.claimed_sum,
            num_vars: claim.poly.num_vars,
            factor_claims: out.factor_claims[base..base + n].to_vec(),
            point: out.challenges[..claim.poly.num_vars].to_vec(),
        });
    }
    Ok(BatchStarOutput {
        proof: out.proof,
        challenges: out.challenges,
        rhos,
        per_claim,
    })
}

/// Verify a `prove_batch_star` proof: replays the transcript, reconstructs
/// the padded combined claim, and verifies the single row-count-preserving
/// sumcheck. `expected_combined_final`: the PCS-authenticated terminal —
/// `Σ_j ρ^j · P_j(r_j)` where `P_j(r_j)` is recomputed from the caller's
/// per-claim factor openings and term structure (the caller knows the
/// structure; this layer does not — note the padding scale lives on the
/// claim only, never on the terminal). Pass `None` to leave the terminal
/// to the PCS composition (the single-claim convention).
pub fn verify_batch_star(
    proof: &crate::sumcheck::SumcheckProof,
    num_vars: &[usize],
    max_degree: usize,
    claimed_sums: &[Goldilocks],
    transcript: &mut Transcript,
    expected_combined_final: Option<Goldilocks>,
) -> Result<Vec<Goldilocks>, BatchError> {
    if num_vars.is_empty() || num_vars.len() != claimed_sums.len() {
        return Err(BatchError::VariableCountMismatch {
            expected: claimed_sums.len(),
            got: num_vars.len(),
        });
    }
    let rhos = transcript
        .challenge_fields(b"batch-star-rho", claimed_sums.len())
        .map_err(|_| BatchError::ChallengeCount)?;
    verify_batch_star_with_rhos(
        proof,
        num_vars,
        max_degree,
        claimed_sums,
        &rhos,
        transcript,
        expected_combined_final,
    )
}

/// The `verify_batch_star` core with pre-sampled rhos: callers that need
/// the rhos to compose the terminal (e.g. the SALSAA norm batch) replay
/// them right after their statement absorption and pass them here — the
/// transcript must be positioned exactly where the rhos were sampled.
pub fn verify_batch_star_with_rhos(
    proof: &crate::sumcheck::SumcheckProof,
    num_vars: &[usize],
    max_degree: usize,
    claimed_sums: &[Goldilocks],
    rhos: &[Goldilocks],
    transcript: &mut Transcript,
    expected_combined_final: Option<Goldilocks>,
) -> Result<Vec<Goldilocks>, BatchError> {
    if num_vars.is_empty() || num_vars.len() != claimed_sums.len() {
        return Err(BatchError::VariableCountMismatch {
            expected: claimed_sums.len(),
            got: num_vars.len(),
        });
    }
    if rhos.len() != claimed_sums.len() {
        return Err(BatchError::ChallengeCount);
    }
    let max_vars = num_vars.iter().copied().max().ok_or(BatchError::EmptyBatch)?;
    let mut combined = Goldilocks::ZERO;
    for ((rho, c), nv) in rhos
        .iter()
        .zip(claimed_sums.iter())
        .zip(num_vars.iter())
    {
        combined = combined.add(&rho.mul(&pow2(max_vars - nv)).mul(c));
    }
    proof
        .verify(max_vars, max_degree, combined, transcript, expected_combined_final)
        .map(|v| v.point)
        .map_err(BatchError::Sumcheck)
}

/// Verify a batch proof: replays the same transcript, checks the combined
/// claim against the per-claim sums recomputed with the same rhos.
pub fn verify_batch(
    proof: &crate::sumcheck::SumcheckProof,
    num_vars: usize,
    max_degree: usize,
    claimed_sums: &[Goldilocks],
    transcript: &mut Transcript,
    expected_combined_final: Option<Goldilocks>,
) -> Result<Vec<Goldilocks>, BatchError> {
    let rhos = transcript
        .challenge_fields(b"batch-rho", claimed_sums.len())
        .map_err(|_| BatchError::ChallengeCount)?;
    let mut combined = Goldilocks::ZERO;
    for (rho, c) in rhos.iter().zip(claimed_sums.iter()) {
        combined = combined.add(&rho.mul(c));
    }
    proof
        .verify(
            num_vars,
            max_degree,
            combined,
            transcript,
            expected_combined_final,
        )
        .map(|v| v.point)
        .map_err(BatchError::Sumcheck)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_poly::VirtualPolynomial;
    use lattice_core::DenseMle;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn build(num_vars: usize, tag: &[u8]) -> VirtualPolynomial {
        let mut vp = VirtualPolynomial::new(num_vars);
        let f = vp.add_factor(DenseMle::random(num_vars, tag)).ok().unwrap();
        let g = vp
            .add_factor(DenseMle::random(num_vars, b"g"))
            .ok()
            .unwrap();
        vp.add_term(fe(2), vec![f, g]).ok().unwrap();
        vp.add_term(fe(9), vec![f]).ok().unwrap();
        vp
    }

    #[test]
    fn batch_prove_verify() {
        let vps = [build(5, b"a"), build(5, b"b"), build(5, b"c")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-test");
        let (out, rhos) = prove_batch(&claims, &mut t).ok().unwrap();
        assert_eq!(rhos.len(), 3);

        let mut vt = Transcript::new_default(b"lzx-batch-test");
        // Expected final: combined P evaluated at challenges.
        let expected = {
            let mut acc = Goldilocks::ZERO;
            for (rho, vp) in rhos.iter().zip(vps.iter()) {
                acc = acc.add(&rho.mul(&vp.evaluate(&out.challenges).ok().unwrap()));
            }
            acc
        };
        let point = verify_batch(&out.proof, 5, 2, &sums, &mut vt, Some(expected))
            .ok()
            .unwrap();
        assert_eq!(point, out.challenges);
    }

    #[test]
    fn batch_wrong_sums_rejected() {
        let vps = [build(4, b"a"), build(4, b"b")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-test");
        let (out, _) = prove_batch(&claims, &mut t).ok().unwrap();
        let mut bad_sums = sums.clone();
        bad_sums[0] = bad_sums[0].add(&fe(1));
        let mut vt = Transcript::new_default(b"lzx-batch-test");
        assert!(verify_batch(&out.proof, 4, 2, &bad_sums, &mut vt, None).is_err());
    }

    #[test]
    fn mixed_variable_counts_rejected() {
        let vp1 = build(3, b"a");
        let vp2 = build(4, b"b");
        let claims = [
            BatchClaim {
                poly: &vp1,
                claimed_sum: Goldilocks::ZERO,
            },
            BatchClaim {
                poly: &vp2,
                claimed_sum: Goldilocks::ZERO,
            },
        ];
        let mut t = Transcript::new_default(b"lzx-batch-test");
        assert!(matches!(
            prove_batch(&claims, &mut t),
            Err(BatchError::VariableCountMismatch { .. })
        ));
        assert!(matches!(
            prove_batch(&[], &mut t),
            Err(BatchError::EmptyBatch)
        ));
    }

    // -------------------------------------------------------------------
    // SALSAA D3 — Pi-batch-star tests
    // -------------------------------------------------------------------

    #[test]
    fn batch_star_row_count_is_max_not_sum() {
        // Theorem-4 property: m claims with heterogeneous variable counts
        // fold into ONE sumcheck with rounds == max(num_vars).
        let vps = [build(3, b"a"), build(5, b"b"), build(5, b"c"), build(4, b"d")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-star-test");
        let out = prove_batch_star(&claims, &mut t).ok().unwrap();
        // Row-count preservation: rounds = max(3, 5, 5, 4) = 5, NOT 17.
        assert_eq!(out.proof.rounds.len(), 5);
        assert_eq!(out.challenges.len(), 5);

        // Verify with the same transcript.
        let mut vt = Transcript::new_default(b"lzx-batch-star-test");
        let point = verify_batch_star(
            &out.proof,
            &[3, 5, 5, 4],
            2,
            &sums,
            &mut vt,
            None,
        )
        .ok()
        .unwrap();
        assert_eq!(point, out.challenges);

        // Per-claim claims: factor claims at the truncated points.
        for (pc, vp) in out.per_claim.iter().zip(vps.iter()) {
            assert_eq!(pc.point.len(), vp.num_vars);
            assert_eq!(pc.factor_claims.len(), vp.factors.len());
            // The per-claim point is the shared point truncated.
            assert_eq!(pc.point, out.challenges[..vp.num_vars]);
        }
    }

    #[test]
    fn batch_star_terminal_from_per_claim_openings() {
        // The PCS-composed terminal: combined final = sum_j rho^j 2^k_j P_j(r_j),
        // recomputed from per-claim factor claims (padded factor values).
        let vps = [build(3, b"a"), build(5, b"b"), build(5, b"c")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-star-test");
        let out = prove_batch_star(&claims, &mut t).ok().unwrap();

        let mut expected = Goldilocks::ZERO;
        for (j, (rho, vp)) in out.rhos.iter().zip(vps.iter()).enumerate() {
            // The padding scale lives on the CLAIM only (the hypercube sum
            // doubles per ignored variable); the TERMINAL never scales —
            // the padded factor's value at the shared point equals the
            // original factor's MLE at the truncated point.
            let pj = vp
                .evaluate(&out.per_claim[j].point)
                .ok()
                .unwrap();
            expected = expected.add(&rho.mul(&pj));
        }

        let mut vt = Transcript::new_default(b"lzx-batch-star-test");
        let point = verify_batch_star(
            &out.proof,
            &[3, 5, 5],
            2,
            &sums,
            &mut vt,
            Some(expected),
        )
        .ok()
        .unwrap();
        assert_eq!(point, out.challenges);
    }

    #[test]
    fn batch_star_wrong_claim_rejected() {
        let vps = [build(4, b"a"), build(3, b"b")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-star-test");
        let out = prove_batch_star(&claims, &mut t).ok().unwrap();
        let mut bad = sums.clone();
        bad[0] = bad[0].add(&fe(1));
        let mut vt = Transcript::new_default(b"lzx-batch-star-test");
        assert!(verify_batch_star(&out.proof, &[4, 3], 2, &bad, &mut vt, None).is_err());
    }

    #[test]
    fn batch_star_tampered_round_rejected() {
        let vps = [build(4, b"a"), build(4, b"b")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-star-test");
        let mut out = prove_batch_star(&claims, &mut t).ok().unwrap();
        if let Some(r0) = out.proof.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&fe(1));
            }
        }
        let mut vt = Transcript::new_default(b"lzx-batch-star-test");
        assert!(verify_batch_star(&out.proof, &[4, 4], 2, &sums, &mut vt, None).is_err());
    }

    #[test]
    fn batch_star_matches_same_var_batch() {
        // When all claims share num_vars, batch-star must agree with the
        // classic prove_batch claim arithmetic (no padding scale).
        let vps = [build(5, b"a"), build(5, b"b")];
        let sums: Vec<Goldilocks> = vps.iter().map(|vp| vp.sum_over_hypercube()).collect();
        let claims: Vec<BatchClaim> = vps
            .iter()
            .zip(sums.iter())
            .map(|(vp, s)| BatchClaim {
                poly: vp,
                claimed_sum: *s,
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-batch-star-test");
        let out = prove_batch_star(&claims, &mut t).ok().unwrap();
        // Transcript labels differ from prove_batch, so compare the
        // STRUCTURE: same rounds count, valid verify.
        assert_eq!(out.proof.rounds.len(), 5);
        let mut vt = Transcript::new_default(b"lzx-batch-star-test");
        assert!(verify_batch_star(&out.proof, &[5, 5], 2, &sums, &mut vt, None).is_ok());
    }
}
