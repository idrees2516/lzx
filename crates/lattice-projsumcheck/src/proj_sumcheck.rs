//! The **projective sum-check protocol** — Figure 3 of ePrint 2026/762.
//!
//! Statement: `Σ_{b ∈ {0,∞}^n} g(b) = C` for `g` a product (or virtual
//! product) of multilinear factors stored in monomial coefficient form.
//! Because the monomial coefficients *are* the truth table, this is the
//! same discrete statement the Boolean sum-check proves for the same data:
//! `Σ_{x∈{0,1}^n} g(x) = C`.
//!
//! ## What changes relative to the Boolean protocol (Figure 2)
//!
//! * Round identity: `s_i(0) + s_i(∞) = C_{i−1}` — the verifier derives
//!   `s_i(0) := C_{i−1} − s_i(∞)` from the **leading coefficient**
//!   `s_i(∞)` instead of deriving `s_i(1)` from `s_i(0)`.
//! * The prover's message points are `Ū_d = {∞} ∪ {1, …, d−1}` — the
//!   degree-`d` univariate is specified by its leading coefficient plus
//!   `d−1` finite evaluations, and the constant term is recovered by the
//!   verifier. This is a one-field-element-per-round **proof compression**
//!   relative to sending evaluations at `0..d` (§B.1 of the paper).
//! * Binding is subtraction-free: `p(r, x') = p(0, x') + r·p(∞, x')`
//!   (Corollary 3.2) — the evaluation pass below computes `s_i(1)` with
//!   *additions only* (`f(1) = f(0) + f(∞)` per factor per pair).
//! * Soundness is unchanged: round-by-round soundness error `d/|F|`
//!   (Theorem 3.3 — the state-function argument carries over verbatim).
//!
//! ## Terminal claims
//!
//! After `n` rounds the factors are constants; the engine returns the
//! opening point `r` and per-factor claimed evaluations of the
//! **coefficient-form** polynomials at `r` — exactly the representation
//! the compact Ajtai opening (`lattice-zkvm::compact`) commits to, so no
//! basis conversion is needed between sum-check and PCS (§4.3).

use crate::proj_mle::{MonomialMle, ProjMleError};
use lattice_core::field_simd::{self, Sum8};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;

/// A projective sum-check proof: round `j` stores the compressed univariate
/// evaluations `[s_j(∞), s_j(1), …, s_j(d−1)]` (`d` entries; the value at
/// `0` is derived by the verifier from the round identity).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjSumcheckProof {
    pub rounds: Vec<Vec<Goldilocks>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjSumcheckError {
    Mle(ProjMleError),
    Transcript(TranscriptError),
    BadRoundShape {
        round: usize,
        got: usize,
        expected: usize,
    },
    RoundCheckFailed {
        round: usize,
    },
    ClaimMismatch,
    FinalCheckFailed,
    EmptyInstance,
}

impl core::fmt::Display for ProjSumcheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProjSumcheckError::Mle(e) => write!(f, "monomial MLE error: {e}"),
            ProjSumcheckError::Transcript(e) => write!(f, "transcript error: {e}"),
            ProjSumcheckError::BadRoundShape {
                round,
                got,
                expected,
            } => {
                write!(f, "round {round} length {got} != expected {expected}")
            }
            ProjSumcheckError::RoundCheckFailed { round } => {
                write!(f, "projective round identity failed at round {round}")
            }
            ProjSumcheckError::ClaimMismatch => write!(f, "claimed sum does not match polynomial"),
            ProjSumcheckError::FinalCheckFailed => write!(f, "terminal identity failed"),
            ProjSumcheckError::EmptyInstance => write!(f, "empty product term in instance"),
        }
    }
}

/// A virtual polynomial over shared monomial-form factors:
/// `P(x) = Σ_j c_j · Π_{k ∈ term_j} f_k(x)`.
#[derive(Clone, Debug)]
pub struct ProjVirtualPolynomial {
    pub num_vars: usize,
    pub factors: Vec<MonomialMle>,
    pub terms: Vec<(Goldilocks, Vec<usize>)>,
}

impl ProjVirtualPolynomial {
    pub fn new(num_vars: usize) -> Self {
        ProjVirtualPolynomial {
            num_vars,
            factors: Vec::new(),
            terms: Vec::new(),
        }
    }

    pub fn add_factor(&mut self, factor: MonomialMle) -> Result<usize, ProjSumcheckError> {
        if factor.num_vars != self.num_vars {
            return Err(ProjSumcheckError::Mle(
                ProjMleError::WrongCoefficientCount {
                    expected: 1 << self.num_vars,
                    got: factor.coeffs.len(),
                },
            ));
        }
        self.factors.push(factor);
        Ok(self.factors.len() - 1)
    }

    pub fn add_term(
        &mut self,
        coeff: Goldilocks,
        indices: Vec<usize>,
    ) -> Result<(), ProjSumcheckError> {
        if indices.is_empty() {
            return Err(ProjSumcheckError::EmptyInstance);
        }
        if indices.iter().any(|i| *i >= self.factors.len()) {
            return Err(ProjSumcheckError::Mle(
                ProjMleError::WrongCoefficientCount {
                    expected: self.factors.len(),
                    got: indices.iter().max().copied().unwrap_or(0) + 1,
                },
            ));
        }
        self.terms.push((coeff, indices));
        Ok(())
    }

    /// Single product of factors (the canonical degree-`ℓ` instance).
    pub fn product(factors: Vec<MonomialMle>) -> Result<Self, ProjSumcheckError> {
        let num_vars = factors.first().map(|f| f.num_vars).unwrap_or(0);
        let mut vp = ProjVirtualPolynomial::new(num_vars);
        for f in factors {
            vp.add_factor(f)?;
        }
        let ids: Vec<usize> = (0..vp.factors.len()).collect();
        vp.add_term(Goldilocks::ONE, ids)?;
        Ok(vp)
    }

    pub fn max_degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(_, ids)| ids.len())
            .max()
            .unwrap_or(1)
    }

    /// Total sum over the infinity hypercube — which equals the
    /// Boolean-cube sum of the underlying truth tables:
    /// `Σ_m Σ_j c_j · Π_k F_k[m]` (the sum of pointwise term products).
    pub fn total_sum(&self) -> Goldilocks {
        if self.terms.is_empty() || self.factors.is_empty() {
            return Goldilocks::ZERO;
        }
        let n = self.factors[0].coeffs.len();
        let mut dense = vec![Goldilocks::ZERO; n];
        for (c, ids) in &self.terms {
            let slices: Vec<&[Goldilocks]> = ids
                .iter()
                .map(|fi| self.factors[*fi].coeffs.as_slice())
                .collect();
            field_simd::accumulate_term_pointwise(&slices, *c, &mut dense);
        }
        field_simd::sum_slice(&dense)
    }
}

/// Prover-side result: proof + terminal evaluation claims.
#[derive(Clone, Debug)]
pub struct ProjSumcheckOutput {
    pub proof: ProjSumcheckProof,
    /// The random point (one challenge per round, variable 0 first).
    pub challenges: Vec<Goldilocks>,
    /// `P(r)` — claimed evaluation of the whole virtual polynomial.
    pub final_claim: Goldilocks,
    /// Per-factor claimed coefficient-form evaluations at `r`.
    pub factor_claims: Vec<Goldilocks>,
}

/// Evaluate a degree-`d` univariate at `r`, given its values at the finite
/// nodes `0, 1, …, d−1` (`finite[k] = s(k)`) and its leading coefficient
/// `s(∞)` — Lemma 2.2:
/// `s(X) = s(∞)·Π_k (X − k) + Σ_k s(k)·L_k(X)`.
pub(crate) fn interpolate_with_infinity(
    finite: &[Goldilocks],
    leading: Goldilocks,
    r: Goldilocks,
) -> Goldilocks {
    let d = finite.len();
    // Leading-coefficient term: s(∞) · Π_k (r − k).
    let mut lead_term = leading;
    for k in 0..d {
        lead_term = lead_term.mul(&r.sub(&Goldilocks::from_u64(k as u64)));
    }
    // Lagrange part over the d finite nodes.
    let mut lag = Goldilocks::ZERO;
    for (k, &fk) in finite.iter().enumerate() {
        let xk = Goldilocks::from_u64(k as u64);
        let mut weight = Goldilocks::ONE;
        for j in 0..d {
            if j == k {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xk.sub(&xj);
            weight = weight.mul(&num.mul(&den.inverse().unwrap_or(Goldilocks::ZERO)));
        }
        lag = lag.add(&fk.mul(&weight));
    }
    lead_term.add(&lag)
}

/// Standard finite-node Lagrange: evaluate the degree-`D` univariate
/// given its values at the nodes `0, 1, …, D` at the point `r`.
fn interpolate_fin(evals: &[Goldilocks], r: Goldilocks) -> Goldilocks {
    let n = evals.len();
    let mut acc = Goldilocks::ZERO;
    for (i, &ei) in evals.iter().enumerate() {
        let xi = Goldilocks::from_u64(i as u64);
        let mut weight = Goldilocks::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            weight = weight.mul(&num.mul(&den.inverse().unwrap_or(Goldilocks::ZERO)));
        }
        acc = acc.add(&ei.mul(&weight));
    }
    acc
}

/// `Σ_j c_j · Π_k first_halves[k]` — the round value at `X = 0`.
fn sum_terms_over_slices(
    terms: &[(Goldilocks, Vec<usize>)],
    slices: &[&[Goldilocks]],
) -> Goldilocks {
    let mut acc = Sum8::new();
    let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
    for (c, ids) in terms {
        fslices.clear();
        fslices.extend(ids.iter().map(|fi| slices[*fi]));
        acc.accumulate_term(*c, &fslices);
    }
    acc.finish()
}

/// Materialize, for the finite point `t ∈ {1, …, d−1}`, the bound values
/// `vals_k[m] = α_k[m] + t·β_k[m]` into `buffers` (one buffer per factor).
///
/// `t = 1` uses pure addition — `f(1) = f(0) + f(∞)` — which is precisely
/// where the Boolean prover's per-pair subtraction disappears (§4.1).
fn materialize_bound_values(bound: &[MonomialMle], t: u64, buffers: &mut [Vec<Goldilocks>]) {
    let half = bound[0].coeffs.len() / 2;
    for (k, f) in bound.iter().enumerate() {
        let (lo, hi) = f.coeffs.split_at(half);
        let out = &mut buffers[k];
        if t == 1 {
            field_simd::add_slices(lo, hi, out);
        } else {
            let tf = Goldilocks::from_u64(t);
            // out = lo + t*hi, computed chunk-wise so the source and
            // destination never alias within a chunk.
            let n = out.len();
            let mut done = 0;
            const CHUNK: usize = 256;
            while done < n {
                let end = (done + CHUNK).min(n);
                field_simd::mul_scalar_slice(&hi[done..end], tf, &mut out[done..end]);
                for k in done..end {
                    out[k] = lo[k].add(&out[k]);
                }
                done = end;
            }
        }
    }
}

/// Prove `Σ_{b∈{0,∞}^n} P(b) = claim` with Fiat–Shamir challenges drawn
/// from `transcript` (which must already hold the public statement).
pub fn prove(
    vp: &ProjVirtualPolynomial,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<ProjSumcheckOutput, ProjSumcheckError> {
    if vp.terms.is_empty() {
        // Zero polynomial: emit well-formed degree-1 zero rounds.
        if !claim.is_zero() {
            return Err(ProjSumcheckError::ClaimMismatch);
        }
        let mut challenges = Vec::with_capacity(vp.num_vars);
        let mut rounds = Vec::with_capacity(vp.num_vars);
        for _ in 0..vp.num_vars {
            // H(X) = 0: ∞-value 0, H(1) = 0 (the d=1 mixed shape).
            let evals = vec![Goldilocks::ZERO, Goldilocks::ZERO];
            transcript
                .append_field_slice(b"projsumcheck-round", &evals)
                .map_err(ProjSumcheckError::Transcript)?;
            let r = transcript
                .challenge_field(b"projsumcheck-challenge")
                .map_err(ProjSumcheckError::Transcript)?;
            challenges.push(r);
            rounds.push(evals);
        }
        return Ok(ProjSumcheckOutput {
            proof: ProjSumcheckProof { rounds },
            challenges,
            final_claim: Goldilocks::ZERO,
            factor_claims: Vec::new(),
        });
    }

    let m = vp.num_vars;
    let d = vp.max_degree();
    // Pure products (every term of degree exactly d): the round
    // polynomial's ∞-value equals its leading coefficient, so the
    // paper's compressed message {s(∞), s(1..d−1)} serves both the round
    // identity and the Lemma 2.2 interpolation. Mixed-degree instances
    // send one extra entry: the ∞-value carries the identity
    // `s(0) + s(∞) = C` while the finite points 1..d carry the
    // interpolation (the ∞-value is then NOT the leading coefficient —
    // lower-degree terms contribute their own X^deg coefficient to it).
    let pure = vp.terms.iter().all(|(_, ids)| ids.len() == d);
    let msg_len = if pure { d } else { d + 1 };
    let mut bound: Vec<MonomialMle> = vp.factors.clone();
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(m);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(m);

    for _round in 0..m {
        // s(0): first halves — the raw monomials independent of X_0.
        let first: Vec<&[Goldilocks]> = bound
            .iter()
            .map(|f| &f.coeffs[..f.coeffs.len() / 2])
            .collect();
        let s0 = sum_terms_over_slices(&vp.terms, &first);

        // s(∞): second halves — the coefficients of X_0.
        let second: Vec<&[Goldilocks]> = bound
            .iter()
            .map(|f| &f.coeffs[f.coeffs.len() / 2..])
            .collect();
        let s_inf = sum_terms_over_slices(&vp.terms, &second);

        // Prover-side consistency guard: s(0) + s(∞) = C_{i−1}.
        if s0.add(&s_inf) != current_claim {
            return Err(ProjSumcheckError::ClaimMismatch);
        }

        let mut evals = Vec::with_capacity(msg_len);
        evals.push(s_inf);
        for t in 1..=msg_len - 1 {
            // Per-round scratch (factors halve every round — a shared
            // buffer would carry stale tails into the sums).
            let mut buffers: Vec<Vec<Goldilocks>> = bound
                .iter()
                .map(|f| vec![Goldilocks::ZERO; f.coeffs.len() / 2])
                .collect();
            materialize_bound_values(&bound, t as u64, &mut buffers);
            let slices: Vec<&[Goldilocks]> = buffers.iter().map(|b| b.as_slice()).collect();
            evals.push(sum_terms_over_slices(&vp.terms, &slices));
        }

        transcript
            .append_field_slice(b"projsumcheck-round", &evals)
            .map_err(ProjSumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"projsumcheck-challenge")
            .map_err(ProjSumcheckError::Transcript)?;
        challenges.push(r);

        // C_i = s_i(r).
        if pure {
            // Compressed: finite nodes 0..d−1 with the derived s(0), and
            // the ∞-value doubling as the leading coefficient.
            let mut finite = vec![s0];
            finite.extend(evals.iter().skip(1).copied());
            current_claim = interpolate_with_infinity(&finite, s_inf, r);
        } else {
            // Mixed: standard finite Lagrange over the nodes 0..d.
            let mut finite = vec![s0];
            finite.extend(evals.iter().skip(1).copied());
            current_claim = interpolate_fin(&finite, r);
        }

        // Subtraction-free binding (Corollary 3.2) of every factor.
        for f in bound.iter_mut() {
            MonomialMle::bind_first_slice_in_place(&mut f.coeffs, r);
            let half = f.coeffs.len() / 2;
            f.coeffs.truncate(half);
            f.num_vars -= 1;
        }
        rounds.push(evals);
    }

    // Terminal: factors are single constants.
    let factor_claims: Vec<Goldilocks> = bound.iter().map(|f| f.coeffs[0]).collect();
    let mut final_claim = Goldilocks::ZERO;
    for (c, ids) in &vp.terms {
        let mut prod = *c;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(ProjSumcheckError::FinalCheckFailed);
    }

    Ok(ProjSumcheckOutput {
        proof: ProjSumcheckProof { rounds },
        challenges,
        final_claim,
        factor_claims,
    })
}

/// Verifier-side reduction result for the PCS layer.
#[derive(Clone, Debug)]
pub struct ProjSumcheckVerifier {
    /// The random point sampled during verification (variable 0 first).
    pub point: Vec<Goldilocks>,
    /// Claimed `P(r)` — the terminal value the caller's PCS must bind.
    pub final_claim: Goldilocks,
}

impl ProjSumcheckProof {
    /// Verify against a claimed sum, returning the terminal binding.
    ///
    /// The caller must authenticate the terminal factor evaluations with
    /// its PCS (the engine deliberately does not trust locally evaluated
    /// factors, mirroring `lattice-sumcheck`).
    pub fn verify(
        &self,
        claim: Goldilocks,
        num_vars: usize,
        max_degree: usize,
        transcript: &mut Transcript,
    ) -> Result<ProjSumcheckVerifier, ProjSumcheckError> {
        if self.rounds.len() != num_vars {
            return Err(ProjSumcheckError::BadRoundShape {
                round: usize::MAX,
                got: self.rounds.len(),
                expected: num_vars,
            });
        }
        let pure = self.rounds.iter().all(|r| r.len() == max_degree);
        let mixed = self.rounds.iter().all(|r| r.len() == max_degree + 1);
        if !pure && !mixed {
            return Err(ProjSumcheckError::BadRoundShape {
                round: usize::MAX,
                got: self.rounds.first().map(|r| r.len()).unwrap_or(0),
                expected: max_degree,
            });
        }
        let mut current = claim;
        let mut point = Vec::with_capacity(num_vars);
        for (i, round) in self.rounds.iter().enumerate() {
            let expected = if pure { max_degree } else { max_degree + 1 };
            if round.len() != expected || round.is_empty() {
                return Err(ProjSumcheckError::BadRoundShape {
                    round: i,
                    got: round.len(),
                    expected,
                });
            }
            transcript
                .append_field_slice(b"projsumcheck-round", round)
                .map_err(ProjSumcheckError::Transcript)?;
            let r = transcript
                .challenge_field(b"projsumcheck-challenge")
                .map_err(ProjSumcheckError::Transcript)?;
            // Derive s(0) = C_{i−1} − s(∞) (the projective round identity —
            // the ∞ entry is always the ∞-value).
            let s_inf = round[0];
            let s0 = current.sub(&s_inf);
            let mut finite = vec![s0];
            finite.extend(round.iter().skip(1).copied());
            current = if pure {
                // ∞-value == leading coefficient: Lemma 2.2.
                interpolate_with_infinity(&finite, s_inf, r)
            } else {
                // Mixed degrees: plain finite Lagrange over 0..d.
                interpolate_fin(&finite, r)
            };
            point.push(r);
        }
        Ok(ProjSumcheckVerifier {
            point,
            final_claim: current,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::DenseMle;

    fn g(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// End-to-end prove/verify roundtrip on a degree-2 instance, with the
    /// total claim derived from the truth tables (so the Boolean and
    /// projective statements coincide).
    #[test]
    fn degree2_roundtrip() {
        let f = MonomialMle::random(6, b"projsc-f");
        let h = MonomialMle::random(6, b"projsc-h");
        let vp = ProjVirtualPolynomial::product(vec![f, h]).unwrap();
        let claim = vp.total_sum();
        let mut prover_ts = Transcript::new_default(b"projsc-prove");
        let out = prove(&vp, claim, &mut prover_ts).unwrap();
        // Prover-side claims must satisfy P(r) = Π f_k(r) coefficient-form.
        assert_eq!(out.final_claim, {
            let mut prod = Goldilocks::ONE;
            for (k, c) in out.factor_claims.iter().enumerate() {
                let _ = k;
                prod = prod.mul(c);
            }
            prod
        });
        let mut verifier_ts = Transcript::new_default(b"projsc-prove");
        let v = out.proof.verify(claim, 6, 2, &mut verifier_ts).unwrap();
        assert_eq!(v.point, out.challenges);
        assert_eq!(v.final_claim, out.final_claim);
    }

    /// A degree-3 virtual polynomial with two terms roundtrips.
    #[test]
    fn degree3_virtual_roundtrip() {
        let f0 = MonomialMle::random(5, b"v-f0");
        let f1 = MonomialMle::random(5, b"v-f1");
        let f2 = MonomialMle::random(5, b"v-f2");
        let mut vp = ProjVirtualPolynomial::new(5);
        let i0 = vp.add_factor(f0).unwrap();
        let i1 = vp.add_factor(f1).unwrap();
        let i2 = vp.add_factor(f2).unwrap();
        vp.add_term(g(3), vec![i0, i1, i2]).unwrap();
        vp.add_term(g(5), vec![i0]).unwrap();
        let claim = vp.total_sum();
        let mut ts = Transcript::new_default(b"v-prove");
        let out = prove(&vp, claim, &mut ts).unwrap();
        let mut ts2 = Transcript::new_default(b"v-prove");
        let v = out.proof.verify(claim, 5, 3, &mut ts2).unwrap();
        assert_eq!(v.final_claim, out.final_claim);
    }

    /// Tamper: a modified round message must change the derived challenge
    /// path and break the terminal identity.
    #[test]
    fn tampered_round_rejected() {
        let f = MonomialMle::random(6, b"t-f");
        let h = MonomialMle::random(6, b"t-h");
        let vp = ProjVirtualPolynomial::product(vec![f, h]).unwrap();
        let claim = vp.total_sum();
        let mut ts = Transcript::new_default(b"t-prove");
        let mut out = prove(&vp, claim, &mut ts).unwrap();
        // Corrupt one round evaluation.
        out.proof.rounds[2][1] = out.proof.rounds[2][1].add(&Goldilocks::ONE);
        let mut ts2 = Transcript::new_default(b"t-prove");
        let v = out.proof.verify(claim, 6, 2, &mut ts2).unwrap();
        // The corrupted transcript derives a different terminal claim: the
        // caller's PCS check fails in practice; here the values differ.
        assert_ne!(v.final_claim, out.final_claim);
    }

    /// Wrong claim is caught by the prover's own guard (claim mismatch).
    #[test]
    fn wrong_claim_rejected() {
        let f = MonomialMle::random(4, b"w-f");
        let h = MonomialMle::random(4, b"w-h");
        let vp = ProjVirtualPolynomial::product(vec![f, h]).unwrap();
        let claim = vp.total_sum().add(&Goldilocks::ONE);
        let mut ts = Transcript::new_default(b"w-prove");
        assert!(matches!(
            prove(&vp, claim, &mut ts),
            Err(ProjSumcheckError::ClaimMismatch)
        ));
    }

    /// The `×eq` shape: `Σ f·g·eq_b(r, ·)` — the paper's degree-2 × eq
    /// benchmark instance, with the projective eq table as a factor.
    #[test]
    fn degree2_times_eq() {
        let n = 7;
        let r: Vec<Goldilocks> = (1..=n as u64)
            .map(|i| Goldilocks::from_u64(1_000_003 * i))
            .collect();
        let f = MonomialMle::random(n, b"e-f");
        let h = MonomialMle::random(n, b"e-h");
        let eq = MonomialMle::eq_projective(&r);
        let vp = ProjVirtualPolynomial::product(vec![f, h, eq]).unwrap();
        let claim = vp.total_sum();
        let mut ts = Transcript::new_default(b"e-prove");
        let out = prove(&vp, claim, &mut ts).unwrap();
        let mut ts2 = Transcript::new_default(b"e-prove");
        let v = out.proof.verify(claim, n, 3, &mut ts2).unwrap();
        assert_eq!(v.final_claim, out.final_claim);
    }

    /// Cross-validation with the Boolean engine: the total claim equals
    /// the Boolean sum-check's claim for the same truth tables, and the
    /// projective final claim satisfies
    /// `P(r)·Π(1+r_i) = f̂(φ(r))·ĝ(φ(r))` (the Möbius bridge).
    #[test]
    fn boolean_cross_validation() {
        use lattice_sumcheck::sumcheck::prove as bool_prove;
        use lattice_sumcheck::virtual_poly::VirtualPolynomial;

        let n = 5;
        let df = DenseMle::random(n, b"x-f");
        let dh = DenseMle::random(n, b"x-h");
        let mut vp_bool = VirtualPolynomial::new(n);
        let a = vp_bool.add_factor(df.clone()).unwrap();
        let b = vp_bool.add_factor(dh.clone()).unwrap();
        vp_bool.add_term(Goldilocks::ONE, vec![a, b]).unwrap();
        let claim_bool = {
            // Σ f̂·ĝ over the Boolean cube, computed directly.
            let mut acc = Goldilocks::ZERO;
            for i in 0..(1 << n) {
                acc = acc.add(&df.evaluations[i].mul(&dh.evaluations[i]));
            }
            acc
        };
        let mf = MonomialMle::from_dense(&df);
        let mh = MonomialMle::from_dense(&dh);
        let vp_proj = ProjVirtualPolynomial::product(vec![mf, mh]).unwrap();
        let claim_proj = vp_proj.total_sum();
        assert_eq!(claim_bool, claim_proj);
        // The Boolean protocol's own prover accepts the same claim.
        let mut ts = Transcript::new_default(b"x-bool");
        let out_bool = bool_prove(&vp_bool, claim_bool, &mut ts).unwrap();
        assert_eq!(out_bool.final_claim, {
            let pt = &out_bool.challenges;
            df.evaluate(pt).unwrap().mul(&dh.evaluate(pt).unwrap())
        });
        // Projective final claim bridges via the Möbius map.
        let mut ts2 = Transcript::new_default(b"x-proj");
        let out_proj = prove(&vp_proj, claim_proj, &mut ts2).unwrap();
        let r = &out_proj.challenges;
        let mut phi = Vec::with_capacity(n);
        let mut denom = Goldilocks::ONE;
        for ri in r {
            let one_plus = Goldilocks::ONE.add(ri);
            phi.push(ri.mul(&one_plus.inverse().unwrap()));
            denom = denom.mul(&one_plus);
        }
        let want = df.evaluate(&phi).unwrap().mul(&dh.evaluate(&phi).unwrap());
        // P(r) = f̂(φ)·ĝ(φ)·Π(1+r_i)²  (per factor).
        let denom_sq = denom.mul(&denom);
        assert_eq!(out_proj.final_claim, want.mul(&denom_sq));
    }
}
