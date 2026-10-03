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

    for _round in 0..m {
        // g(X) evaluated at X = 0..=d — the D6 single-binding discipline:
        // the per-factor half-binding structure (lo/hi slices) is computed
        // ONCE per round; t = 0 and t = 1 borrow the raw halves (zero-copy —
        // a + (b−a)·0 = a, a + (b−a)·1 = b are canonical), and every other t
        // binds through one reusable packed kernel buffer. Byte-identical
        // round values to the per-t half-binding form.
        let evals_at = round_evals_single_bind(&bound, &vp.terms, d);
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

        // D6 in-place binding: fold every factor's first variable directly
        // in its own buffer (the same `bind_first_half_in_place` kernel
        // `fix_variables` uses on its clone) — no per-round allocation and
        // no per-round full-table copy.
        for b in bound.iter_mut() {
            let len = b.evaluations.len();
            lattice_core::field_simd::bind_first_half_in_place(&mut b.evaluations, r);
            b.evaluations.truncate(len / 2);
            b.num_vars -= 1;
        }
        rounds.push(evals_at);
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

/// The D6 single-binding round evaluation: `g(t)` for every `t = 0..=d`
/// from ONE half-binding pass per factor per round.
///
/// `g(t) = Σ_p Σ_terms coeff · Π_i (a_i(p) + t·(b_i(p) − a_i(p)))` where
/// `a = evs[..points]` and `b = evs[points..]` are the factor's raw halves.
/// t = 0 borrows `a`, t = 1 borrows `b` (zero copies); each other t binds
/// into a per-factor reusable buffer with the packed half-binding kernel.
/// Values are bit-identical to the historical per-t `sum_products` form.
fn round_evals_single_bind(
    bound: &[DenseMle],
    terms: &[(Goldilocks, Vec<usize>)],
    d: usize,
) -> Vec<Goldilocks> {
    if bound.is_empty() {
        return vec![Goldilocks::ZERO; d + 1];
    }
    let rem_vars = bound[0].num_vars;
    let points = 1usize << (rem_vars - 1);
    let num_factors = bound.len();
    // One reusable binding buffer per factor, sized once per round.
    let mut bind_bufs: Vec<Vec<Goldilocks>> =
        vec![vec![Goldilocks::ZERO; points]; num_factors];
    let mut evals_at = Vec::with_capacity(d + 1);
    for t in 0..=d {
        let t_fe = Goldilocks::from_u64(t as u64);
        // Per-factor value slices at this t.
        let mut bound_vals: Vec<&[Goldilocks]> = Vec::with_capacity(num_factors);
        if t_fe.is_zero() {
            for f in bound {
                bound_vals.push(&f.evaluations[..points]);
            }
        } else if t_fe == Goldilocks::ONE {
            for f in bound {
                bound_vals.push(&f.evaluations[points..]);
            }
        } else {
            // Bind every factor into its reusable buffer first (mutable
            // borrows end here), then collect the immutable slices.
            for (fi, f) in bound.iter().enumerate() {
                let evs = &f.evaluations;
                lattice_core::field_simd::bind_half_slices(
                    &evs[..points],
                    &evs[points..],
                    t_fe,
                    &mut bind_bufs[fi],
                );
            }
            for buf in bind_bufs.iter() {
                bound_vals.push(buf.as_slice());
            }
        }
        // SIMD: 8-lane lazy term-product accumulation with exact carry
        // accounting (bit-identical to the scalar sequential sum).
        let mut acc = lattice_core::field_simd::Sum8::new();
        let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
        for (coeff, ids) in terms {
            fslices.clear();
            fslices.extend(ids.iter().map(|fi| bound_vals[*fi]));
            acc.accumulate_term(*coeff, &fslices);
        }
        evals_at.push(acc.finish());
    }
    evals_at
}

/// g(t): sum over the remaining hypercube of the virtual polynomial with
/// the current (first) variable of every factor set to t. Factors are
/// half-bound on the fly: `f_t(p) = f[p] + t·(f[p + points] − f[p])`.
///
/// Kept as the reference form of the round evaluation (the tests pin it
/// against `round_evals_single_bind`); the prover hot path uses the
/// single-binding variant above.
#[allow(dead_code)]
pub(crate) fn sum_products(
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
    // Precompute per-factor half-bindings: for factor f with 2*points
    // evaluations, bound value at point p with first var = t:
    // f_val(p) = f[p] + t * (f[p + points] - f[p]).
    let mut bound_vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(bound.len());
    for f in bound {
        let evs = &f.evaluations;
        let mut vals = vec![Goldilocks::ZERO; points];
        // SIMD: the t = 0 / t = 1 bindings are exactly the raw halves
        // (a + (b−a)·0 = a, a + (b−a)·1 = b — canonical), so copy instead
        // of multiplying; every other t goes through the packed half-binding
        // kernel (8 field elements per chunk).
        if t.is_zero() {
            vals.copy_from_slice(&evs[..points]);
        } else if t == Goldilocks::ONE {
            vals.copy_from_slice(&evs[points..]);
        } else {
            lattice_core::field_simd::bind_half_slices(&evs[..points], &evs[points..], t, &mut vals);
        }
        bound_vals.push(vals);
    }
    // SIMD: 8-lane lazy term-product accumulation with exact carry
    // accounting (bit-identical to the scalar sequential sum).
    let mut acc = lattice_core::field_simd::Sum8::new();
    let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
    for (coeff, ids) in terms {
        fslices.clear();
        fslices.extend(ids.iter().map(|fi| bound_vals[*fi].as_slice()));
        acc.accumulate_term(*coeff, &fslices);
    }
    acc.finish()
}

/// Lagrange-evaluate the round polynomial (given its values at 0..d) at r.
#[allow(clippy::needless_range_loop)]
pub(crate) fn interpolate_at(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
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

    #[test]
    fn single_binding_round_values_match_reference() {
        // D6: the single-binding round evaluation must be bit-identical to
        // the historical per-t half-binding reference at every t of every
        // round (the byte-identical-transcript invariant).
        for num_vars in [2usize, 4, 6] {
            let vp = build_vp(num_vars);
            let claim = vp.sum_over_hypercube();
            let mut bound: Vec<DenseMle> = vp.factors.clone();
            let d = vp.max_degree();
            for round in 0..num_vars {
                let fast = round_evals_single_bind(&bound, &vp.terms, d);
                let mut reference = Vec::with_capacity(d + 1);
                for t in 0..=d {
                    reference.push(sum_products(&bound, &vp.terms, fe(t as u64)));
                }
                assert_eq!(fast, reference, "round {round} of {num_vars} vars");
                // bind to the SAME next state both ways: in-place vs
                // fix_variables.
                let mut inplace = bound.clone();
                let r = fe((round as u64) % 5 + 2); // a non-canonical t
                for b in inplace.iter_mut() {
                    let len = b.evaluations.len();
                    lattice_core::field_simd::bind_first_half_in_place(
                        &mut b.evaluations,
                        r,
                    );
                    b.evaluations.truncate(len / 2);
                    b.num_vars -= 1;
                }
                let cloned: Vec<DenseMle> = bound
                    .iter()
                    .map(|b| b.fix_variables(&[r]).ok().unwrap())
                    .collect();
                for (a, b) in inplace.iter().zip(cloned.iter()) {
                    assert_eq!(a.num_vars, b.num_vars);
                    assert_eq!(a.evaluations, b.evaluations);
                }
                bound = cloned;
            }
            let _ = claim;
        }
    }
}
