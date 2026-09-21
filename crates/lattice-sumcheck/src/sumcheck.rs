//! The multilinear sumcheck protocol with Fiat–Shamir compilation.
//!
//! Statement: `Σ_{x ∈ {0,1}^m} P(x) = claim` for a virtual polynomial P
//! given as a product structure over MLE factors.
//!
//! * Prover binds factors round by round; round `j` sends the univariate
//!   polynomial `g_j(X)` (degree ≤ d) evaluated at `X = 0..d` — the
//!   compressed (coefficient-free) form.
//! * Verifier checks `g_0(0) + g_0(1) = claim`, `g_{j+1}(0) + g_{j+1}(1) =
//!   g_j(r_j)`, samples `r_j` from the transcript, and finally checks the
//!   terminal identity `g_{m-1}(r_{m-1}) = P(r_0..r_{m-1})` against
//!   *claimed factor evaluations* — which the PCS layer must then
//!   authenticate (see lattice-akita). The engine deliberately does NOT
//!   trust locally-evaluated factors: it returns the binding point and
//!   expected evaluations as claims.

use crate::virtual_poly::{VirtualPolyError, VirtualPolynomial};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};

/// A sumcheck proof: per-round compressed univariate evaluations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof {
    /// Round polynomials: round j has degree ≤ d_j entries g_j(0..d_j).
    pub rounds: Vec<Vec<Goldilocks>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumcheckError {
    VirtualPoly(VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    Transcript(TranscriptError),
    /// Round polynomial degree/length invalid.
    BadRoundShape {
        round: usize,
        got: usize,
    },
    /// Round-sum identity failed.
    RoundCheckFailed {
        round: usize,
    },
    /// Terminal identity failed.
    FinalCheckFailed,
    /// Declared claim does not match the polynomial.
    ClaimMismatch,
}

/// Prover-side result: proof + the final evaluation claims.
#[derive(Clone, Debug)]
pub struct SumcheckOutput {
    pub proof: SumcheckProof,
    /// The random point r (one challenge per round).
    pub challenges: Vec<Goldilocks>,
    /// P(r): the final claimed evaluation of the whole virtual polynomial.
    pub final_claim: Goldilocks,
    /// Per-factor claimed evaluations at r (indexed like vp.factors).
    pub factor_claims: Vec<Goldilocks>,
}

/// Prove `Σ P = claim`, deriving challenges from the given transcript.
/// The transcript must already contain all public statement material.
pub fn prove(
    vp: &VirtualPolynomial,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<SumcheckOutput, SumcheckError> {
    if vp.terms.is_empty() {
        // Zero polynomial: sum is zero. Claim must be zero. Emit well-formed
        // degree-1 zero rounds so the verifier's shape checks pass.
        if !claim.is_zero() {
            return Err(SumcheckError::ClaimMismatch);
        }
        let mut challenges = Vec::with_capacity(vp.num_vars);
        let mut rounds = Vec::with_capacity(vp.num_vars);
        for _ in 0..vp.num_vars {
            let evals = vec![Goldilocks::ZERO, Goldilocks::ZERO];
            transcript
                .append_field_slice(b"sumcheck-round", &evals)
                .map_err(SumcheckError::Transcript)?;
            let r = transcript
                .challenge_field(b"sumcheck-challenge")
                .map_err(SumcheckError::Transcript)?;
            challenges.push(r);
            rounds.push(evals);
        }
        return Ok(SumcheckOutput {
            proof: SumcheckProof { rounds },
            challenges,
            final_claim: Goldilocks::ZERO,
            factor_claims: Vec::new(),
        });
    }
    let m = vp.num_vars;
    let d = vp.max_degree();
    // Working bound copies of all factors (evaluations shrink each round).
    let mut bound: Vec<DenseMle> = vp.factors.clone();
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(m);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(m);

    for round in 0..m {
        // g(X) evaluated at X = 0..=d: bind the current variable of every
        // factor to t, then accumulate term products over the remaining
        // hypercube (all terms at once inside sum_products).
        let mut evals_at = Vec::with_capacity(d + 1);
        for t in 0..=d {
            let t_fe = Goldilocks::from_u64(t as u64);
            evals_at.push(sum_products(&bound, &vp.terms, t_fe));
        }
        // Absorb round polynomial, get challenge, bind all factors.
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(SumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(SumcheckError::Transcript)?;
        challenges.push(r);

        // Prover-side consistency guard: g(0) + g(1) must equal the
        // running claim (catches construction bugs before transcription).
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(SumcheckError::ClaimMismatch);
        }
        current_claim = interpolate_at(&evals_at, &r);

        for b in bound.iter_mut() {
            *b = b.fix_variables(&[r]).map_err(SumcheckError::Mle)?;
        }
        rounds.push(evals_at);
        let _ = round;
    }

    // All variables bound: each factor is a constant. Compute P(r).
    let mut factor_claims = Vec::with_capacity(bound.len());
    for b in &bound {
        factor_claims.push(b.evaluate(&[]).map_err(SumcheckError::Mle)?);
    }
    let mut final_claim = Goldilocks::ZERO;
    for (coeff, ids) in &vp.terms {
        let mut prod = *coeff;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(SumcheckError::FinalCheckFailed);
    }

    Ok(SumcheckOutput {
        proof: SumcheckProof { rounds },
        challenges,
        final_claim,
        factor_claims,
    })
}

/// g(t): sum over the remaining hypercube of the virtual polynomial with
/// the current (first) variable of every factor set to t. Factors are
/// half-bound on the fly: `f_t(p) = f[p] + t·(f[p + points] − f[p])`.
fn sum_products(
    bound: &[DenseMle],
    terms: &[(Goldilocks, Vec<usize>)],
    t: Goldilocks,
) -> Goldilocks {
    // Number of remaining points after binding this variable.
    if bound.is_empty() {
        return Goldilocks::ZERO;
    }
    let rem_vars = bound[0].num_vars; // before binding
    let points = 1usize << (rem_vars - 1);
    let mut acc = Goldilocks::ZERO;
    // Precompute per-factor half-bindings: for factor f with 2*points
    // evaluations, bound value at point p with first var = t:
    // f_val(p) = f[p] + t * (f[p + points] - f[p]).
    let mut bound_vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(bound.len());
    for f in bound {
        let evs = &f.evaluations;
        let mut vals = Vec::with_capacity(points);
        for p in 0..points {
            let a = evs[p];
            let b = evs[p + points];
            vals.push(a.add(&b.sub(&a).mul(&t)));
        }
        bound_vals.push(vals);
    }
    #[allow(clippy::needless_range_loop)]
    for (coeff, ids) in terms {
        for p in 0..points {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&bound_vals[*fi][p]);
            }
            acc = acc.add(&prod);
        }
    }
    acc
}

/// Lagrange-evaluate the round polynomial (given its values at 0..d) at r.
#[allow(clippy::needless_range_loop)]
fn interpolate_at(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
    // Univariate Lagrange interpolation over nodes 0..d.
    let n = evals.len();
    let mut acc = Goldilocks::ZERO;
    for i in 0..n {
        let mut weight = Goldilocks::ONE;
        let xi = Goldilocks::from_u64(i as u64);
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            // (r - xj) / (xi - xj)
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            weight = weight.mul(&num.mul(&den.inverse().unwrap_or(Goldilocks::ZERO)));
        }
        acc = acc.add(&evals[i].mul(&weight));
    }
    acc
}

/// Verifier state returned to the caller for PCS binding.
#[derive(Clone, Debug)]
pub struct SumcheckVerifier {
    /// Random point sampled during verification.
    pub point: Vec<Goldilocks>,
    /// Claimed P(r).
    pub final_claim: Goldilocks,
}

impl SumcheckProof {
    /// Verify the proof against a claimed sum. Returns the evaluation
    /// binding (point + claimed final value) for the PCS layer.
    ///
    /// `expected_final`: if provided, must equal the derived final claim —
    /// callers with a locally computable P(r) pass it here; PCS-authenticated
    /// callers instead take the returned claims and open them.
    pub fn verify(
        &self,
        num_vars: usize,
        max_degree: usize,
        claim: Goldilocks,
        transcript: &mut Transcript,
        expected_final: Option<Goldilocks>,
    ) -> Result<SumcheckVerifier, SumcheckError> {
        if self.rounds.len() != num_vars {
            return Err(SumcheckError::BadRoundShape {
                round: 0,
                got: self.rounds.len(),
            });
        }
        let mut current = claim;
        let mut point = Vec::with_capacity(num_vars);
        for (round, evals) in self.rounds.iter().enumerate() {
            if evals.is_empty() || evals.len() > max_degree + 1 {
                return Err(SumcheckError::BadRoundShape {
                    round,
                    got: evals.len(),
                });
            }
            transcript
                .append_field_slice(b"sumcheck-round", evals)
                .map_err(SumcheckError::Transcript)?;
            let r = transcript
                .challenge_field(b"sumcheck-challenge")
                .map_err(SumcheckError::Transcript)?;
            // g(0) + g(1) == current claim.
            let sum01 = evals[0].add(&evals[1]);
            if sum01 != current {
                return Err(SumcheckError::RoundCheckFailed { round });
            }
            current = interpolate_at(evals, &r);
            point.push(r);
        }
        if let Some(expected) = expected_final {
            if current != expected {
                return Err(SumcheckError::FinalCheckFailed);
            }
        }
        Ok(SumcheckVerifier {
            point,
            final_claim: current,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_poly::VirtualPolynomial;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn build_vp(num_vars: usize) -> VirtualPolynomial {
        let mut vp = VirtualPolynomial::new(num_vars);
        let f = vp
            .add_factor(DenseMle::random(num_vars, b"sc-f"))
            .ok()
            .unwrap();
        let g = vp
            .add_factor(DenseMle::random(num_vars, b"sc-g"))
            .ok()
            .unwrap();
        let h = vp
            .add_factor(DenseMle::random(num_vars, b"sc-h"))
            .ok()
            .unwrap();
        vp.add_term(fe(3), vec![f, g]).ok().unwrap();
        vp.add_term(fe(5), vec![g, h, f]).ok().unwrap();
        vp.add_term(fe(11), vec![h]).ok().unwrap();
        vp
    }

    #[test]
    fn prove_verify_happy_path() {
        for num_vars in [1usize, 2, 5, 8] {
            let vp = build_vp(num_vars);
            let claim = vp.sum_over_hypercube();
            let mut prover_t = Transcript::new_default(b"lzx-sumcheck-test");
            let out = prove(&vp, claim, &mut prover_t).ok().unwrap();

            // Verifier replays the same transcript.
            let mut verifier_t = Transcript::new_default(b"lzx-sumcheck-test");
            // Expected final: evaluate P at the challenge point locally.
            let expected = vp.evaluate(&out.challenges).ok().unwrap();
            let verdict = out
                .proof
                .verify(
                    vp.num_vars,
                    vp.max_degree(),
                    claim,
                    &mut verifier_t,
                    Some(expected),
                )
                .ok()
                .unwrap();
            assert_eq!(verdict.point, out.challenges);
            assert_eq!(verdict.final_claim, out.final_claim);
        }
    }

    #[test]
    fn wrong_claim_rejected() {
        let vp = build_vp(4);
        let claim = vp.sum_over_hypercube();
        let mut t = Transcript::new_default(b"lzx-sumcheck-test");
        let out = prove(&vp, claim, &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"lzx-sumcheck-test");
        let bad_claim = claim.add(&fe(1));
        assert!(out
            .proof
            .verify(vp.num_vars, vp.max_degree(), bad_claim, &mut t2, None)
            .is_err());
    }

    #[test]
    fn tampered_round_rejected() {
        let vp = build_vp(4);
        let claim = vp.sum_over_hypercube();
        let mut t = Transcript::new_default(b"lzx-sumcheck-test");
        let mut out = prove(&vp, claim, &mut t).ok().unwrap();
        // Tamper with round 0 evaluation.
        if let Some(r0) = out.proof.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&fe(1));
            }
        }
        let mut t2 = Transcript::new_default(b"lzx-sumcheck-test");
        // Challenges desync -> round check or transcript mismatch.
        let res = out
            .proof
            .verify(vp.num_vars, vp.max_degree(), claim, &mut t2, None);
        assert!(res.is_err() || res.ok().map(|v| v.final_claim) != Some(out.final_claim));
    }

    #[test]
    fn tampered_final_rejected() {
        let vp = build_vp(4);
        let claim = vp.sum_over_hypercube();
        let mut t = Transcript::new_default(b"lzx-sumcheck-test");
        let out = prove(&vp, claim, &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"lzx-sumcheck-test");
        // Wrong expected final evaluation must fail.
        assert!(out
            .proof
            .verify(
                vp.num_vars,
                vp.max_degree(),
                claim,
                &mut t2,
                Some(out.final_claim.add(&fe(1)))
            )
            .is_err());
    }

    #[test]
    fn zero_polynomial_edge() {
        let vp = VirtualPolynomial::new(3);
        let mut t = Transcript::new_default(b"lzx-sumcheck-test");
        let out = prove(&vp, Goldilocks::ZERO, &mut t).ok().unwrap();
        assert_eq!(out.proof.rounds.len(), 3);
        assert!(out
            .proof
            .rounds
            .iter()
            .all(|r| r.iter().all(|e| e.is_zero())));
        let mut t2 = Transcript::new_default(b"lzx-sumcheck-test");
        assert!(out
            .proof
            .verify(3, 1, Goldilocks::ZERO, &mut t2, Some(Goldilocks::ZERO))
            .is_ok());
    }

    #[test]
    fn interpolation_matches_direct() {
        // g evaluated at 0,1,2 (degree-2) — interpolate at 5 and compare
        // with the true polynomial x^2 + x + 1.
        let evals = [fe(1), fe(3), fe(7)]; // x=0,1,2
        let r = fe(5);
        assert_eq!(interpolate_at(&evals, &r), fe(31)); // 25+5+1
    }
}
