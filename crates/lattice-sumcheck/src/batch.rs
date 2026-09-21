//! Claim batching: combine `k` sumcheck statements into one via a random
//! linear combination sampled from the transcript (SALSA's batched
//! verification; Jolt's multi-claim stage fusion).
//!
//! Given claims `Σ P_i = c_i`, sample `ρ^0..ρ^{k-1}` and prove
//! `Σ (Σ_i ρ^i P_i) = Σ_i ρ^i c_i` in a single sumcheck. The combined
//! virtual polynomial shares the union of factor indices across claims,
//! and the returned per-factor claims map back to each input polynomial's
//! factors — so one PCS opening round authenticates every original claim.

use crate::sumcheck::{self, SumcheckError, SumcheckOutput};
use crate::virtual_poly::{VirtualPolyError, VirtualPolynomial};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;

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
}
