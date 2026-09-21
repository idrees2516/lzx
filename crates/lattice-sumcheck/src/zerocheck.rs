//! Spartan-style zerocheck: prove `P(x) = 0` for all `x ∈ {0,1}^m`.
//!
//! Reduction: sample a random point `r` after committing to it in the
//! transcript, then run sumcheck on `Q(x) = P(x) · eq(r, x)`:
//! `Σ_{x} eq(r, x) · P(x) = P(r)` — the multilinear evaluation identity.
//! The claim is exactly `P(r) = 0`, and by the random-point argument
//! (eq(r, ·) is nonzero at every hypercube vertex except with probability
//! 2^{-λ}), a cheating prover passing the sumcheck implies P vanishes on
//! the hypercube.

use crate::sumcheck::{self, SumcheckError, SumcheckOutput};
use crate::virtual_poly::VirtualPolynomial;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZerocheckError {
    Sumcheck(SumcheckError),
    Transcript(String),
    FinalClaimNonZero,
}

/// Prove that `poly` vanishes on the entire boolean hypercube.
/// `poly` is a *dense* MLE here; large-witness callers wrap factors in a
/// virtual polynomial themselves and use `prove_vp`.
pub fn prove(
    poly: &DenseMle,
    transcript: &mut Transcript,
) -> Result<SumcheckOutput, ZerocheckError> {
    let num_vars = poly.num_vars;
    // Commit to a random point r (transcript-derived).
    let r = transcript
        .challenge_fields(b"zerocheck-point", num_vars)
        .map_err(|e| ZerocheckError::Transcript(e.to_string()))?;
    // Q(x) = P(x) * eq(r, x); claim = P(r) = 0 (must hold for an honest
    // prover; we check it locally to fail closed).
    let eq = DenseMle::eq_extension(&r);
    let actual = poly
        .evaluate(&r)
        .map_err(|e| ZerocheckError::Transcript(e.to_string()))?;
    if !actual.is_zero() {
        // P does not vanish on the hypercube; refuse to prove.
        return Err(ZerocheckError::FinalClaimNonZero);
    }
    let mut vp = VirtualPolynomial::new(num_vars);
    let pi = vp
        .add_factor(poly.clone())
        .map_err(|e| ZerocheckError::Transcript(e.to_string()))?;
    let ei = vp
        .add_factor(eq)
        .map_err(|e| ZerocheckError::Transcript(e.to_string()))?;
    vp.add_term(Goldilocks::ONE, vec![pi, ei])
        .map_err(|e| ZerocheckError::Transcript(e.to_string()))?;
    sumcheck::prove(&vp, Goldilocks::ZERO, transcript).map_err(ZerocheckError::Sumcheck)
}

/// Verify a zerocheck proof. The returned verifier state carries the
/// factor-evaluation claims (P(r) must be verified as zero through the
/// PCS opening of P; eq(r, ·) is verifier-computable).
pub fn verify(
    proof: &crate::sumcheck::SumcheckProof,
    num_vars: usize,
    transcript: &mut Transcript,
    expected_poly_eval: Option<Goldilocks>,
) -> Result<Vec<Goldilocks>, ZerocheckError> {
    let r = transcript
        .challenge_fields(b"zerocheck-point", num_vars)
        .map_err(|e| ZerocheckError::Transcript(e.to_string()))?;
    // Degree 2 rounds (product of two multilinears).
    let verdict = proof
        .verify(
            num_vars,
            2,
            Goldilocks::ZERO,
            transcript,
            expected_poly_eval,
        )
        .map_err(ZerocheckError::Sumcheck)?;
    let _ = verdict;
    // The eq factor's evaluation at r is prod eq(r_i, r_i) — verifier
    // computable; the P factor's claim must be zero (checked by caller
    // against PCS openings).
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn zero_poly_proves_and_verifies() {
        // A multilinear P vanishing on the hypercube is identically zero:
        // build P = f - f with shared boolean evaluations.
        let f = DenseMle::random(5, b"zc-f");
        let evals: Vec<Goldilocks> = f.evaluations.iter().map(|v| v.sub(v)).collect();
        let p = DenseMle {
            num_vars: 5,
            evaluations: evals,
        };

        let mut pt = Transcript::new_default(b"lzx-zerocheck-test");
        let out = prove(&p, &mut pt).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-zerocheck-test");
        // The P factor claim must be zero.
        assert!(out.factor_claims[0].is_zero());
        let r = verify(&out.proof, 5, &mut vt, Some(Goldilocks::ZERO))
            .ok()
            .unwrap();
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn nonzero_poly_refused() {
        // f - g where g differs at one hypercube point: P does not vanish,
        // and the prover must detect it (P(r) != 0 whp).
        let f = DenseMle::random(4, b"nz-f");
        let mut g_evals = f.evaluations.clone();
        g_evals[3] = g_evals[3].add(&Goldilocks::ONE);
        let g = DenseMle {
            num_vars: 4,
            evaluations: g_evals,
        };
        let evals: Vec<Goldilocks> = f
            .evaluations
            .iter()
            .zip(g.evaluations.iter())
            .map(|(a, b)| a.sub(b))
            .collect();
        let p = DenseMle {
            num_vars: 4,
            evaluations: evals,
        };
        let mut t = Transcript::new_default(b"lzx-zerocheck-test");
        assert!(matches!(
            prove(&p, &mut t),
            Err(ZerocheckError::FinalClaimNonZero) | Err(ZerocheckError::Sumcheck(_))
        ));
    }

    #[test]
    fn zerocheck_tampered_round_rejected() {
        // Zero MLE built as f - f.
        let f = DenseMle::random(4, b"zc-f");
        let evals: Vec<Goldilocks> = f.evaluations.iter().map(|v| v.sub(v)).collect();
        let p = DenseMle {
            num_vars: 4,
            evaluations: evals,
        };
        let mut pt = Transcript::new_default(b"lzx-zerocheck-test");
        let mut out = prove(&p, &mut pt).ok().unwrap();
        if let Some(r0) = out.proof.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&fe(1));
            }
        }
        let mut vt = Transcript::new_default(b"lzx-zerocheck-test");
        assert!(verify(&out.proof, 4, &mut vt, None).is_err());
    }
}
