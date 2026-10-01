//! Multilinear sumcheck over `F_{q^2}` (Wave 7 substrate).
//!
//! Four Wave-7 protocols communicate their rounds as single **F_{q^e}**
//! elements per the papers: Cyclo Π^range (Fig 1: sumcheck over F_{q^e},
//! individual degree 2b+2), LatticeFold+ Π^mon (Construction 4.2:
//! degree-3 sumcheck over C ⊇ F_{q^u}), Symphony Π_had (Fig 1: degree-3
//! sumcheck over K = F_{q^2}) and PikkuFold RingSC (Def 15: sumcheck over
//! F_{q^a} with subfield batching). `lattice-sumcheck` is Goldilocks-only,
//! so this module provides the extension-field engine over
//! [`lattice_core::extension::Fq2`] with the same discipline as the
//! Goldilocks engine:
//!
//! * the prover binds dense factors round by round and sends the
//!   univariate round polynomial `g_j` evaluated at `X = 0..=D`
//!   (compressed form, D = max individual degree),
//! * the verifier checks `g_j(0) + g_j(1)` equals the running claim,
//!   samples the round challenge from the transcript, interpolates,
//! * the terminal check runs against a caller-supplied expected final
//!   evaluation (each protocol derives it from prover claims — Cyclo's
//!   `eq(u,η)·Π_{j=−b}^{b}(t−j)` leaf check, LF+'s `ev(β)² − ev(β²)`
//!   check, Symphony's `U`-matrix check).
//!
//! Determinism: round values are absorbed as canonical [`Fq2::to_bytes`]
//! concatenations and challenges drawn with
//! [`lattice_core::extension::challenge_fq2`] — prover and verifier
//! replay identical transcripts.

use lattice_core::extension::{challenge_fq2, Fq2};
use lattice_core::transcript::{Transcript, TranscriptError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fq2SumcheckError {
    /// A factor's evaluation table does not have `2^num_vars` entries.
    BadFactorShape { factor: usize, got: usize },
    /// A term references an unknown factor.
    BadTerm { factor: usize },
    /// Round polynomial length/degree invalid.
    BadRoundShape { round: usize, got: usize },
    /// Round-sum identity failed.
    RoundCheckFailed { round: usize },
    /// Terminal identity failed.
    FinalCheckFailed,
    /// Declared claim does not match the polynomial.
    ClaimMismatch,
    Transcript(TranscriptError),
}

/// A virtual polynomial over `F_{q^2}`: dense factors (hypercube
/// evaluations) plus product terms with coefficients.
#[derive(Clone, Debug, Default)]
pub struct Fq2VirtualPoly {
    pub num_vars: usize,
    pub factors: Vec<Vec<Fq2>>,
    pub terms: Vec<(Fq2, Vec<usize>)>,
}

impl Fq2VirtualPoly {
    pub fn new(num_vars: usize) -> Self {
        Fq2VirtualPoly {
            num_vars,
            factors: Vec::new(),
            terms: Vec::new(),
        }
    }

    /// Add a dense factor (its evaluations over `{0,1}^num_vars`).
    pub fn add_factor(&mut self, evals: Vec<Fq2>) -> Result<usize, Fq2SumcheckError> {
        if evals.len() != 1usize << self.num_vars {
            return Err(Fq2SumcheckError::BadFactorShape {
                factor: self.factors.len(),
                got: evals.len(),
            });
        }
        self.factors.push(evals);
        Ok(self.factors.len() - 1)
    }

    /// Add a product term `coeff · Π_ids factor[id]`.
    pub fn add_term(&mut self, coeff: Fq2, ids: Vec<usize>) -> Result<(), Fq2SumcheckError> {
        if ids.iter().any(|id| *id >= self.factors.len()) {
            return Err(Fq2SumcheckError::BadTerm {
                factor: ids.iter().copied().max().unwrap_or(0),
            });
        }
        self.terms.push((coeff, ids));
        Ok(())
    }

    /// Maximum individual degree of the virtual polynomial.
    pub fn max_degree(&self) -> usize {
        self.terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1)
    }
}

/// A sumcheck proof over `F_{q^2}`: per-round compressed univariate
/// evaluations (round j has degree ≤ D, so D+1 entries).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fq2SumcheckProof {
    pub rounds: Vec<Vec<Fq2>>,
}

/// Prover-side result.
#[derive(Clone, Debug)]
pub struct Fq2SumcheckOutput {
    pub proof: Fq2SumcheckProof,
    /// Random point (one challenge per round).
    pub challenges: Vec<Fq2>,
    /// P(r) — the final claimed evaluation.
    pub final_claim: Fq2,
    /// Per-factor claimed evaluations at r.
    pub factor_claims: Vec<Fq2>,
}

/// Half-bind a dense factor's first variable at `t`:
/// `f_t(p) = f[p] + t·(f[p + points] − f[p])` over the remaining points.
fn half_bind(evals: &[Fq2], t: Fq2) -> Vec<Fq2> {
    let points = evals.len() / 2;
    let mut out = Vec::with_capacity(points);
    for p in 0..points {
        let a = evals[p];
        let b = evals[p + points];
        out.push(a.add(&b.sub(&a).mul(&t)));
    }
    out
}

/// Sum of term products over the remaining hypercube, with every factor
/// half-bound at `t`.
fn sum_products(bound: &[Vec<Fq2>], terms: &[(Fq2, Vec<usize>)]) -> Fq2 {
    let mut acc = Fq2::ZERO;
    for (coeff, ids) in terms {
        if ids.is_empty() {
            // Constant term: contributes `coeff` at each remaining point.
            let pts = bound.first().map(|f| f.len()).unwrap_or(0);
            for _ in 0..pts {
                acc = acc.add(coeff);
            }
            continue;
        }
        let pts = bound[ids[0]].len();
        #[allow(clippy::needless_range_loop)]
        for p in 0..pts {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&bound[*fi][p]);
            }
            acc = acc.add(&prod);
        }
    }
    acc
}

/// Lagrange-evaluate the round polynomial (values at nodes 0..n−1) at `r`.
fn interpolate_fq2(evals: &[Fq2], r: &Fq2) -> Fq2 {
    let n = evals.len();
    let mut acc = Fq2::ZERO;
    for (i, v) in evals.iter().enumerate() {
        let xi = Fq2::from_base(lattice_core::Goldilocks::from_u64(i as u64));
        let mut weight = Fq2::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Fq2::from_base(lattice_core::Goldilocks::from_u64(j as u64));
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            if let Some(inv) = den.inverse() {
                weight = weight.mul(&num.mul(&inv));
            }
        }
        acc = acc.add(&v.mul(&weight));
    }
    acc
}

/// Absorb a slice of Fq2 values into the transcript (canonical bytes).
fn absorb_fq2_slice(
    transcript: &mut Transcript,
    label: &[u8],
    values: &[Fq2],
) -> Result<(), TranscriptError> {
    let mut buf = Vec::with_capacity(values.len() * 16);
    for v in values {
        buf.extend_from_slice(&v.to_bytes());
    }
    transcript.append_bytes(label, &buf)
}

/// Prove `Σ_{x ∈ {0,1}^m} P(x) = claim` over F_{q^2}, deriving challenges
/// from the given transcript (which must already contain the statement).
pub fn prove(
    vp: &Fq2VirtualPoly,
    claim: Fq2,
    transcript: &mut Transcript,
) -> Result<Fq2SumcheckOutput, Fq2SumcheckError> {
    let m = vp.num_vars;
    let d = vp.max_degree();
    if vp.terms.is_empty() {
        // Zero polynomial: the claim must be zero; emit well-formed
        // degree-1 zero rounds so the verifier's shape checks pass.
        if !claim.is_zero() {
            return Err(Fq2SumcheckError::ClaimMismatch);
        }
        let mut challenges = Vec::with_capacity(m);
        let mut rounds = Vec::with_capacity(m);
        for _ in 0..m {
            let evals = vec![Fq2::ZERO, Fq2::ZERO];
            absorb_fq2_slice(transcript, b"fq2-sumcheck-round", &evals)
                .map_err(Fq2SumcheckError::Transcript)?;
            let r = challenge_fq2(transcript, b"fq2-sumcheck-challenge")
                .map_err(Fq2SumcheckError::Transcript)?;
            challenges.push(r);
            rounds.push(evals);
        }
        return Ok(Fq2SumcheckOutput {
            proof: Fq2SumcheckProof { rounds },
            challenges,
            final_claim: Fq2::ZERO,
            factor_claims: Vec::new(),
        });
    }
    let mut bound: Vec<Vec<Fq2>> = vp.factors.clone();
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Fq2>> = Vec::with_capacity(m);
    let mut challenges: Vec<Fq2> = Vec::with_capacity(m);

    for _round in 0..m {
        // g(t) for t = 0..=D: half-bind every factor at t and accumulate.
        let mut evals_at = Vec::with_capacity(d + 1);
        for t in 0..=d {
            let t_fe = Fq2::from_base(lattice_core::Goldilocks::from_u64(t as u64));
            let bound_at: Vec<Vec<Fq2>> = bound.iter().map(|f| half_bind(f, t_fe)).collect();
            evals_at.push(sum_products(&bound_at, &vp.terms));
        }
        absorb_fq2_slice(transcript, b"fq2-sumcheck-round", &evals_at)
            .map_err(Fq2SumcheckError::Transcript)?;
        let r = challenge_fq2(transcript, b"fq2-sumcheck-challenge")
            .map_err(Fq2SumcheckError::Transcript)?;
        challenges.push(r);

        // Prover-side consistency guard: g(0) + g(1) must equal the
        // running claim (catches construction bugs before transcription).
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(Fq2SumcheckError::ClaimMismatch);
        }
        current_claim = interpolate_fq2(&evals_at, &r);
        for b in bound.iter_mut() {
            *b = half_bind(b, r);
        }
        rounds.push(evals_at);
    }

    // All variables bound: each factor is a single evaluation.
    let factor_claims: Vec<Fq2> = bound
        .iter()
        .map(|f| f.first().copied().unwrap_or(Fq2::ZERO))
        .collect();
    let mut final_claim = Fq2::ZERO;
    for (coeff, ids) in &vp.terms {
        let mut prod = *coeff;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(Fq2SumcheckError::FinalCheckFailed);
    }

    Ok(Fq2SumcheckOutput {
        proof: Fq2SumcheckProof { rounds },
        challenges,
        final_claim,
        factor_claims,
    })
}

/// Verifier state returned for protocol-layer terminal checks.
#[derive(Clone, Debug)]
pub struct Fq2SumcheckVerifier {
    /// Random point sampled during verification.
    pub point: Vec<Fq2>,
    /// Claimed P(r) (derived by interpolation).
    pub final_claim: Fq2,
}

impl Fq2SumcheckProof {
    /// Verify the proof against a claimed sum. `expected_final`: if
    /// provided, must equal the derived final claim — callers with a
    /// locally computable P(r) (from prover-supplied leaf claims) pass it
    /// here; PCS-authenticated callers take the returned binding instead.
    pub fn verify(
        &self,
        num_vars: usize,
        max_degree: usize,
        claim: Fq2,
        transcript: &mut Transcript,
        expected_final: Option<Fq2>,
    ) -> Result<Fq2SumcheckVerifier, Fq2SumcheckError> {
        if self.rounds.len() != num_vars {
            return Err(Fq2SumcheckError::BadRoundShape {
                round: 0,
                got: self.rounds.len(),
            });
        }
        let mut current = claim;
        let mut point = Vec::with_capacity(num_vars);
        for (round, evals) in self.rounds.iter().enumerate() {
            if evals.is_empty() || evals.len() > max_degree + 1 {
                return Err(Fq2SumcheckError::BadRoundShape {
                    round,
                    got: evals.len(),
                });
            }
            absorb_fq2_slice(transcript, b"fq2-sumcheck-round", evals)
                .map_err(Fq2SumcheckError::Transcript)?;
            let r = challenge_fq2(transcript, b"fq2-sumcheck-challenge")
                .map_err(Fq2SumcheckError::Transcript)?;
            let sum01 = evals[0].add(&evals[1]);
            if sum01 != current {
                return Err(Fq2SumcheckError::RoundCheckFailed { round });
            }
            current = interpolate_fq2(evals, &r);
            point.push(r);
        }
        if let Some(expected) = expected_final {
            if current != expected {
                return Err(Fq2SumcheckError::FinalCheckFailed);
            }
        }
        Ok(Fq2SumcheckVerifier {
            point,
            final_claim: current,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::Goldilocks;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn random_fq2_vec(num_vars: usize, seed: &[u8]) -> Vec<Fq2> {
        let bytes = Transcript::xof(b"fq2-sc-random", seed, (1 << num_vars) * 16);
        bytes
            .chunks(16)
            .map(|c| {
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&c[..16.min(c.len())]);
                Fq2::new(
                    Goldilocks::from_u64(u64::from_le_bytes(arr[..8].try_into().ok().unwrap())),
                    Goldilocks::from_u64(u64::from_le_bytes(arr[8..16].try_into().ok().unwrap())),
                )
            })
            .take(1 << num_vars)
            .collect()
    }

    fn build_vp(num_vars: usize) -> Fq2VirtualPoly {
        let mut vp = Fq2VirtualPoly::new(num_vars);
        let f = vp.add_factor(random_fq2_vec(num_vars, b"f")).ok().unwrap();
        let g = vp.add_factor(random_fq2_vec(num_vars, b"g")).ok().unwrap();
        let h = vp.add_factor(random_fq2_vec(num_vars, b"h")).ok().unwrap();
        vp.add_term(Fq2::new(fe(3), fe(5)), vec![f, g]).ok().unwrap();
        vp.add_term(Fq2::new(fe(0), fe(7)), vec![g, h, f]).ok().unwrap();
        vp.add_term(Fq2::from_base(fe(11)), vec![h]).ok().unwrap();
        vp
    }

    fn vp_sum(vp: &Fq2VirtualPoly) -> Fq2 {
        let mut acc = Fq2::ZERO;
        for (coeff, ids) in &vp.terms {
            for p in 0..(1usize << vp.num_vars) {
                let mut prod = *coeff;
                for fi in ids {
                    prod = prod.mul(&vp.factors[*fi][p]);
                }
                acc = acc.add(&prod);
            }
        }
        acc
    }

    fn vp_eval(vp: &Fq2VirtualPoly, point: &[Fq2]) -> Fq2 {
        // P is NOT multilinear (terms can be degree > 1 per variable), so
        // evaluate each FACTOR's MLE at the point (factors are multilinear
        // — eq-weight sums are valid there) and combine the terms.
        let factor_at: Vec<Fq2> = vp
            .factors
            .iter()
            .map(|f| {
                let mut acc = Fq2::ZERO;
                for (p, v) in f.iter().enumerate() {
                    let mut w = Fq2::ONE;
                    for (var, rv) in point.iter().enumerate() {
                        let bit = (p >> (vp.num_vars - 1 - var)) & 1;
                        let term = if bit == 1 { *rv } else { Fq2::ONE.sub(rv) };
                        w = w.mul(&term);
                    }
                    acc = acc.add(&v.mul(&w));
                }
                acc
            })
            .collect();
        let mut acc = Fq2::ZERO;
        for (coeff, ids) in &vp.terms {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&factor_at[*fi]);
            }
            acc = acc.add(&prod);
        }
        acc
    }

    #[test]
    fn prove_verify_happy_path() {
        for num_vars in [1usize, 2, 4, 6] {
            let vp = build_vp(num_vars);
            let claim = vp_sum(&vp);
            let mut pt = Transcript::new_default(b"fq2-sc-test");
            let out = prove(&vp, claim, &mut pt).ok().unwrap();
            let mut vt = Transcript::new_default(b"fq2-sc-test");
            let expected = vp_eval(&vp, &out.challenges);
            let verdict = out
                .proof
                .verify(num_vars, vp.max_degree(), claim, &mut vt, Some(expected))
                .ok()
                .unwrap();
            assert_eq!(verdict.point, out.challenges);
            assert_eq!(verdict.final_claim, out.final_claim);
        }
    }

    #[test]
    fn wrong_claim_rejected() {
        let vp = build_vp(4);
        let claim = vp_sum(&vp);
        let mut t = Transcript::new_default(b"fq2-sc-test");
        let out = prove(&vp, claim, &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"fq2-sc-test");
        let bad = claim.add(&Fq2::from_base(fe(1)));
        assert!(out
            .proof
            .verify(4, vp.max_degree(), bad, &mut t2, None)
            .is_err());
    }

    #[test]
    fn tampered_round_rejected() {
        let vp = build_vp(4);
        let claim = vp_sum(&vp);
        let mut t = Transcript::new_default(b"fq2-sc-test");
        let mut out = prove(&vp, claim, &mut t).ok().unwrap();
        if let Some(r0) = out.proof.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&Fq2::from_base(fe(1)));
            }
        }
        let mut t2 = Transcript::new_default(b"fq2-sc-test");
        let res = out.proof.verify(4, vp.max_degree(), claim, &mut t2, None);
        assert!(res.is_err() || res.ok().map(|v| v.final_claim) != Some(out.final_claim));
    }

    #[test]
    fn tampered_final_rejected() {
        let vp = build_vp(3);
        let claim = vp_sum(&vp);
        let mut t = Transcript::new_default(b"fq2-sc-test");
        let out = prove(&vp, claim, &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"fq2-sc-test");
        assert!(out
            .proof
            .verify(3, vp.max_degree(), claim, &mut t2, Some(out.final_claim.add(&Fq2::I)))
            .is_err());
    }

    #[test]
    fn zero_polynomial_edge() {
        let vp = Fq2VirtualPoly::new(3);
        let mut t = Transcript::new_default(b"fq2-sc-test");
        let out = prove(&vp, Fq2::ZERO, &mut t).ok().unwrap();
        assert_eq!(out.proof.rounds.len(), 3);
        let mut t2 = Transcript::new_default(b"fq2-sc-test");
        assert!(out
            .proof
            .verify(3, 1, Fq2::ZERO, &mut t2, Some(Fq2::ZERO))
            .is_ok());
    }

    #[test]
    fn interpolation_matches_direct() {
        // g evaluated at 0,1,2 (degree-2): interpolate at 5 → 31 for
        // x² + x + 1.
        let evals = [Fq2::from_base(fe(1)), Fq2::from_base(fe(3)), Fq2::from_base(fe(7))];
        let r = Fq2::from_base(fe(5));
        assert_eq!(interpolate_fq2(&evals, &r), Fq2::from_base(fe(31)));
    }

    #[test]
    fn factor_claims_match_direct_evaluation() {
        let vp = build_vp(3);
        let claim = vp_sum(&vp);
        let mut t = Transcript::new_default(b"fq2-sc-test");
        let out = prove(&vp, claim, &mut t).ok().unwrap();
        for (i, factor) in vp.factors.iter().enumerate() {
            // Brute-force MLE evaluation of the factor at the challenge.
            let mut acc = Fq2::ZERO;
            for (p, fv) in factor.iter().enumerate() {
                let mut w = Fq2::ONE;
                for (var, rv) in out.challenges.iter().enumerate() {
                    let bit = (p >> (vp.num_vars - 1 - var)) & 1;
                    let term = if bit == 1 { *rv } else { Fq2::ONE.sub(rv) };
                    w = w.mul(&term);
                }
                acc = acc.add(&fv.mul(&w));
            }
            assert_eq!(acc, out.factor_claims[i], "factor {i}");
        }
    }
}
