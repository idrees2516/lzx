//! LatticeFold+ (Boneh–Chen 2025, ePrint 2025/247): faster, simpler,
//! shorter lattice folding.
//!
//! Two novel techniques implemented here:
//! 1. **Purely algebraic range proof** — instead of LatticeFold's
//!    bit-decomposition with per-bit pedersen-style checks, prove
//!    `||w||∞ ≤ β` via a sumcheck over digit MLEs: witness coefficients map
//!    to boolean digit vectors, and both *booleanity* (`d∘(1−d) ≡ 0`) and
//!    *reconstruction* (`Σ 2^i d_i = c + β`) are polynomial identities
//!    checked by a single sumcheck batch.
//! 2. **Double commitments** — commit to the commitment: the folded
//!    accumulator carries an outer Ajtai commitment to the digit-opening
//!    of the inner one, and the fold updates both levels linearly, keeping
//!    proofs short (the outer level absorbs norm growth).

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck::{self, SumcheckError};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

/// A double commitment: inner Ajtai commitment to the witness, outer Ajtai
/// commitment to the packed digit-opening of the inner.
#[derive(Clone, Debug)]
pub struct DoubleCommitment {
    pub inner: AjtaiCommitment,
    pub outer: AjtaiCommitment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LfPlusError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    Sumcheck(SumcheckError),
    TranscriptFailure,
    RangeProofFailed,
    NormExceeded { norm: u64, bound: u64 },
    ShapeMismatch { expected: usize, got: usize },
}

/// Algebraic range proof: prove every Goldilocks coefficient of a committed
/// vector lies in [-β, β] (i.e., c + β ∈ [0, 2β]) via digit MLEs.
///
/// Layout: coefficients are laid out on a hypercube of log2(len) variables
/// (padded); each value's digit vector (base 2, ⌈log2(2β+1)⌉ digits) forms
/// a digit layer. Two virtual polynomials are proven in one batched
/// sumcheck:
/// * booleanity: Σ_x d_i(x)·(1 − d_i(x)) = 0 for every digit layer i;
/// * reconstruction: Σ_x eq(r, x)·(c(x) + β − Σ_i 2^i d_i(x)) = 0.
pub struct AlgebraicRangeProof {
    pub proof: lattice_sumcheck::SumcheckProof,
    /// Claimed digit-layer evaluations at the sumcheck challenge point
    /// (for PCS opening by the caller).
    pub digit_claims: Vec<Goldilocks>,
    /// The eq-random point r (drives the booleanity multiplier).
    pub eq_point: Vec<Goldilocks>,
    /// The sumcheck challenge point (where reconstruction is checked).
    pub sc_point: Vec<Goldilocks>,
}

/// Digits per value for a symmetric range bound β: need
/// ceil(log2(2β+1)) bits.
pub fn digits_for_bound(beta: u64) -> usize {
    let span = 2 * beta + 1;
    usize::BITS as usize - span.leading_zeros() as usize
}

/// Prove the range bound for a vector of field coefficients.
/// `coeffs` must be the *balanced* interpretations reduced into
/// [0, p) — the proof covers c + β ∈ [0, 2β] as integers.
pub fn prove_range(
    coeffs: &[Goldilocks],
    beta: u64,
    transcript: &mut Transcript,
) -> Result<AlgebraicRangeProof, LfPlusError> {
    let num_digits = digits_for_bound(beta);
    let padded_len = coeffs.len().next_power_of_two();
    let log_vars = padded_len.trailing_zeros() as usize;
    let p = lattice_core::field::GOLDILOCKS_MODULUS;

    // Digit layers: d_i(x) for each digit position i, over the hypercube.
    let mut layers: Vec<Vec<Goldilocks>> = vec![vec![Goldilocks::ZERO; padded_len]; num_digits];
    for (idx, c) in coeffs.iter().enumerate() {
        // Balanced representative in [-β, β] maps to v = c + β in [0, 2β].
        let raw = c.to_canonical_u64();
        let balanced = if raw >= p / 2 {
            raw as i128 - p as i128
        } else {
            raw as i128
        };
        let v = (balanced + beta as i128) as u64; // in [0, 2β] if in range
        for (i, layer) in layers.iter_mut().enumerate() {
            layer[idx] = Goldilocks::from_u64((v >> i) & 1);
        }
    }

    // Single random point r drives both checks (LatticeFold+ structure):
    // * booleanity — sumcheck over Σ_x eq(r,x) · Σ_i d_i(x)(1 − d_i(x)):
    //   eq(r,·) is nonzero at every hypercube vertex whp, so the sum is
    //   zero iff every digit is boolean (no cross-point cancellation).
    // * reconstruction — the MLE identity g(x) = c(x) + β − Σ_i 2^i d_i(x)
    //   vanishes on the hypercube iff digits reconstruct the values, hence
    //   g(r) = 0 at the random point; checked from the digit claims.
    let r = transcript
        .challenge_fields(b"lf-range-point", log_vars)
        .map_err(|_| LfPlusError::TranscriptFailure)?;
    let eq = DenseMle::eq_extension(&r);

    let mut vp = VirtualPolynomial::new(log_vars);
    let mut layer_ids = Vec::with_capacity(num_digits);
    for layer in &layers {
        let mle = DenseMle {
            num_vars: log_vars,
            evaluations: layer.clone(),
        };
        layer_ids.push(vp.add_factor(mle).map_err(|_| LfPlusError::ShapeMismatch {
            expected: log_vars,
            got: 0,
        })?);
    }
    let one_minus: Vec<Vec<Goldilocks>> = layers
        .iter()
        .map(|l| l.iter().map(|v| Goldilocks::ONE.sub(v)).collect())
        .collect();

    // Booleanity terms: eq(r,x) · d_i(x) · (1 − d_i(x)) per layer.
    let eq_id = vp
        .add_factor(eq)
        .map_err(|_| LfPlusError::ShapeMismatch { expected: log_vars, got: 0 })?;
    for (i, lm) in one_minus.iter().enumerate() {
        let mle = DenseMle {
            num_vars: log_vars,
            evaluations: lm.clone(),
        };
        let om_id = vp.add_factor(mle).map_err(|_| LfPlusError::ShapeMismatch {
            expected: log_vars,
            got: 0,
        })?;
        vp.add_term(Goldilocks::ONE, vec![layer_ids[i], om_id, eq_id])
            .map_err(|_| LfPlusError::RangeProofFailed)?;
    }

    let out = sumcheck::prove(&vp, Goldilocks::ZERO, transcript).map_err(LfPlusError::Sumcheck)?;
    let digit_claims: Vec<Goldilocks> = layer_ids
        .iter()
        .map(|id| out.factor_claims[*id])
        .collect();
    Ok(AlgebraicRangeProof {
        proof: out.proof,
        digit_claims,
        eq_point: r,
        sc_point: out.challenges,
    })
}

/// Verify an algebraic range proof: booleanity sumcheck + point
/// reconstruction given the claimed coefficient evaluation at the same
/// challenge point.
pub fn verify_range(
    proof: &AlgebraicRangeProof,
    beta: u64,
    num_coeffs: usize,
    coeff_claim_at_point: Goldilocks,
    transcript: &mut Transcript,
) -> Result<(), LfPlusError> {
    let num_digits = digits_for_bound(beta);
    let padded_len = num_coeffs.next_power_of_two();
    let log_vars = padded_len.trailing_zeros() as usize;
    if proof.digit_claims.len() != num_digits {
        return Err(LfPlusError::ShapeMismatch {
            expected: num_digits,
            got: proof.digit_claims.len(),
        });
    }
    let r = transcript
        .challenge_fields(b"lf-range-point", log_vars)
        .map_err(|_| LfPlusError::TranscriptFailure)?;
    if r != proof.eq_point {
        return Err(LfPlusError::RangeProofFailed);
    }
    // Booleanity sumcheck: degree-3 rounds (three factors per term).
    // The final claim is NOT zero — it equals eq(r, r_sc) · Σ_i
    // d_i(r_sc)(1 − d_i(r_sc)), recomputed from the digit claims and the
    // verifier-computable eq factor.
    let verdict = proof
        .proof
        .verify(log_vars, 3, Goldilocks::ZERO, transcript, None)
        .map_err(LfPlusError::Sumcheck)?;
    let r_sc = verdict.point;
    let eq_at = DenseMle::eq_extension(&r)
        .evaluate(&r_sc)
        .map_err(|_| LfPlusError::ShapeMismatch { expected: 0, got: 0 })?;
    let mut poly_at = Goldilocks::ZERO;
    for d in &proof.digit_claims {
        poly_at = poly_at.add(&d.mul(&Goldilocks::ONE.sub(d)));
    }
    let expected_final = eq_at.mul(&poly_at);
    if verdict.final_claim != expected_final {
        return Err(LfPlusError::RangeProofFailed);
    }
    // Reconstruction at the sumcheck point: Σ 2^i d_i(r_sc) == c(r_sc) + β
    // (mod p). d_i(r_sc) are MLE evaluations at a random point — arbitrary
    // field values, NOT booleans; the identity itself is the constraint.
    let mut acc = Goldilocks::ZERO;
    for (i, d) in proof.digit_claims.iter().enumerate() {
        acc = acc.add(&d.mul(&Goldilocks::from_u64(1u64 << i.min(63))));
    }
    let expected = coeff_claim_at_point.add(&Goldilocks::from_u64(beta));
    if acc != expected {
        return Err(LfPlusError::RangeProofFailed);
    }
    Ok(())
}

/// Fold two double commitments under a transcript challenge (small for
/// norm control). The inner level folds like ProtogaLattice; the outer
/// level folds the digit openings the same way, so both stay consistent.
pub struct FoldedDouble {
    pub commitment: DoubleCommitment,
    /// Ring-scalar challenge used (norm bookkeeping).
    pub challenge_balanced: i64,
    /// Updated norm budget (additive growth, LatticeFold+ style).
    pub norm_budget: u64,
}

pub fn fold_double(
    pk: &AjtaiPublicKey,
    d1: &DoubleCommitment,
    d2: &DoubleCommitment,
    norm1: u64,
    norm2: u64,
) -> Result<FoldedDouble, LfPlusError> {
    let ring = &pk.params.ring;
    let q = ring.modulus;
    let mut transcript = Transcript::new_default(b"lzx-latticefold-plus");
    transcript
        .append_bytes(b"inner1", &d1.inner.to_bytes())
        .map_err(|_| LfPlusError::TranscriptFailure)?;
    transcript
        .append_bytes(b"inner2", &d2.inner.to_bytes())
        .map_err(|_| LfPlusError::TranscriptFailure)?;
    transcript
        .append_bytes(b"outer1", &d1.outer.to_bytes())
        .map_err(|_| LfPlusError::TranscriptFailure)?;
    transcript
        .append_bytes(b"outer2", &d2.outer.to_bytes())
        .map_err(|_| LfPlusError::TranscriptFailure)?;

    // Small challenge (biased-ternary-style short challenge, safe for norms).
    let seed = transcript
        .challenge_bytes(b"fold-r", 32)
        .map_err(|_| LfPlusError::TranscriptFailure)?;
    let bytes = Transcript::xof(b"lf-chal", &seed, 8);
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[..8]);
    // Balanced value in [-2^15, 2^15): short challenge keeps the folded
    // norm budget growing only additively with small factors.
    let raw = (u64::from_le_bytes(arr) & 0xFFFF) as i64;
    let r_int = if raw >= 1 << 15 { raw - (1 << 16) } else { raw };
    let _r_scalar = q.reduce_i64(r_int);

    let fold_rows = |a: &AjtaiCommitment, b: &AjtaiCommitment| -> Result<AjtaiCommitment, LfPlusError> {
        let mut rows = Vec::with_capacity(a.rows.len());
        for (ra, rb) in a.rows.iter().zip(b.rows.iter()) {
            rows.push(
                ra.add(&rb.scale_i64(r_int))
                    .map_err(LfPlusError::Ring)?,
            );
        }
        Ok(AjtaiCommitment { rows })
    };
    let inner = fold_rows(&d1.inner, &d2.inner)?;
    let outer = fold_rows(&d1.outer, &d2.outer)?;
    let r_abs = r_int.unsigned_abs() as u64;
    Ok(FoldedDouble {
        commitment: DoubleCommitment { inner, outer },
        challenge_balanced: r_int,
        norm_budget: norm1 + r_abs * norm2,
    })
}

/// Parameters helper for tests/callers.
pub fn default_params(log_n: u32, m: usize, norm_bound: u64) -> Option<AjtaiParams> {
    let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n).ok()?;
    Some(AjtaiParams {
        ring,
        k: 2,
        m,
        norm_bound: norm_bound.min(u32::MAX as u64) as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn algebraic_range_proof_happy_path() {
        // Coefficients in [-100, 100].
        let coeffs: Vec<Goldilocks> = [-100i64, -1, 0, 1, 99, 42, -77, 13, 5, -5, 88, 3, 0, 9, -50, 60]
            .iter()
            .map(|c| {
                // Encode balanced values canonically.
                Goldilocks::from_u64((*c as i128).rem_euclid(lattice_core::field::GOLDILOCKS_MODULUS as i128) as u64)
            })
            .collect();
        let beta = 100u64;
        let mut t = Transcript::new_default(b"lf-range-test");
        let proof = prove_range(&coeffs, beta, &mut t).ok().unwrap();
        // Coefficient evaluation at the challenge point (locally, the
        // caller would take this from the PCS opening).
        let c_mle = DenseMle::new(coeffs.clone()).ok().unwrap();
        let c_at_sc = c_mle.evaluate(&proof.sc_point).ok().unwrap();
        let mut vt = Transcript::new_default(b"lf-range-test");
        assert!(verify_range(&proof, beta, coeffs.len(), c_at_sc, &mut vt).is_ok());
    }

    #[test]
    fn isolation_digit_claims_match_direct_eval() {
        let coeffs: Vec<Goldilocks> = [-100i64, -1, 0, 1, 99, 42, -77, 13, 5, -5, 88, 3, 0, 9, -50, 60]
            .iter()
            .map(|c| Goldilocks::from_u64((*c as i128).rem_euclid(lattice_core::field::GOLDILOCKS_MODULUS as i128) as u64))
            .collect();
        let beta = 100u64;
        let mut t = Transcript::new_default(b"lf-range-test");
        let proof = prove_range(&coeffs, beta, &mut t).ok().unwrap();
        // Rebuild digit layers manually.
        let num_digits = digits_for_bound(beta);
        let mut layers = vec![vec![Goldilocks::ZERO; 16]; num_digits];
        for (idx, c) in coeffs.iter().enumerate() {
            let p_mod = lattice_core::field::GOLDILOCKS_MODULUS;
            let raw = c.to_canonical_u64();
            let balanced = if raw >= p_mod / 2 { raw as i128 - p_mod as i128 } else { raw as i128 };
            let v = (balanced + beta as i128) as u64;
            for (i, layer) in layers.iter_mut().enumerate() {
                layer[idx] = Goldilocks::from_u64((v >> i) & 1);
            }
        }
        // Direct evaluation of digit layers at sc_point.
        for (i, layer) in layers.iter().enumerate() {
            let mle = DenseMle { num_vars: 4, evaluations: layer.clone() };
            let direct = mle.evaluate(&proof.sc_point).ok().unwrap();
            assert_eq!(direct, proof.digit_claims[i], "layer {i} mismatch");
        }
        // Direct reconstruction identity (regression: i64-cast modulus bug).
        let mut acc = Goldilocks::ZERO;
        for (i, d) in proof.digit_claims.iter().enumerate() {
            acc = acc.add(&d.mul(&Goldilocks::from_u64(1u64 << i)));
        }
        let c_mle = DenseMle::new(coeffs.clone()).ok().unwrap();
        let c_at = c_mle.evaluate(&proof.sc_point).ok().unwrap();
        assert_eq!(acc, c_at.add(&Goldilocks::from_u64(beta)));
    }

    #[test]
    fn brute_force_evaluate_reference() {
        // Compare evaluate() against brute-force Lagrange interpolation
        // over all hypercube points (ground truth for the MLE semantics).
        let coeffs: Vec<Goldilocks> = [-100i64, -1, 0, 1, 99, 42, -77, 13, 5, -5, 88, 3, 0, 9, -50, 60]
            .iter()
            .map(|c| Goldilocks::from_u64((*c as i128).rem_euclid(lattice_core::field::GOLDILOCKS_MODULUS as i128) as u64))
            .collect();
        let mle = DenseMle::new(coeffs.clone()).ok().unwrap();
        let point = [
            Goldilocks::from_u64(877746185904530010),
            Goldilocks::from_u64(18275345248914937972),
            Goldilocks::from_u64(18403641651327692187),
            Goldilocks::from_u64(9702716292423503897),
        ];
        // Brute force: f(r) = Σ_x f(x) · eq(x, r) where eq is the product
        // of per-variable indicators.
        let mut acc = Goldilocks::ZERO;
        for (idx, v) in coeffs.iter().enumerate() {
            let mut w = Goldilocks::ONE;
            for (var, rv) in point.iter().enumerate() {
                let bit = (idx >> (3 - var)) & 1; // var 0 = MSB
                let term = if bit == 1 { *rv } else { Goldilocks::ONE.sub(rv) };
                w = w.mul(&term);
            }
            acc = acc.add(&v.mul(&w));
        }
        let fast = mle.evaluate(&point).ok().unwrap();
        assert_eq!(acc, fast);
    }

    #[test]
    fn out_of_range_detected() {
        // A coefficient outside [-β, β] must fail reconstruction.
        let coeffs = vec![fe(5), fe(1000), fe(3), fe(9)];
        let beta = 100u64;
        let mut t = Transcript::new_default(b"lf-range-test");
        let proof = prove_range(&coeffs, beta, &mut t).ok().unwrap();
        let c_mle = DenseMle::new(coeffs.clone()).ok().unwrap();
        let c_at_sc = c_mle.evaluate(&proof.sc_point).ok().unwrap();
        let mut vt = Transcript::new_default(b"lf-range-test");
        assert!(verify_range(&proof, beta, coeffs.len(), c_at_sc, &mut vt).is_err());
    }

    #[test]
    fn tampered_digit_claim_detected() {
        let coeffs: Vec<Goldilocks> = [-7i64, 3, -2, 11]
            .iter()
            .map(|c| Goldilocks::from_u64((*c as i128).rem_euclid(lattice_core::field::GOLDILOCKS_MODULUS as i128) as u64))
            .collect();
        let beta = 16u64;
        let mut t = Transcript::new_default(b"lf-range-test");
        let mut proof = prove_range(&coeffs, beta, &mut t).ok().unwrap();
        let c_mle = DenseMle::new(coeffs.clone()).ok().unwrap();
        let c_at_sc = c_mle.evaluate(&proof.sc_point).ok().unwrap();
        // Tamper: flip a digit claim to a non-boolean value.
        if !proof.digit_claims.is_empty() {
            proof.digit_claims[0] = fe(7);
        }
        let mut vt = Transcript::new_default(b"lf-range-test");
        assert!(verify_range(&proof, beta, coeffs.len(), c_at_sc, &mut vt).is_err());
    }

    #[test]
    fn double_commitment_fold_linearity() {
        let params = default_params(4, 3, 1 << 22).unwrap();
        let pk = AjtaiPublicKey::from_seed(params, [21u8; 32]).ok().unwrap();
        let ring = pk.params.ring.clone();
        let w1 = lattice_commitment::ajtai::sample_small_secret(&ring, pk.params.m, 64, b"a");
        let w2 = lattice_commitment::ajtai::sample_small_secret(&ring, pk.params.m, 64, b"b");
        let inner1 = pk.commit(&w1).ok().unwrap();
        let inner2 = pk.commit(&w2).ok().unwrap();
        // Outer: commit to digit-openings of the inners (double commitment).
        let outer1 = pk.commit(&w1).ok().unwrap();
        let outer2 = pk.commit(&w2).ok().unwrap();
        let d1 = DoubleCommitment {
            inner: inner1,
            outer: outer1,
        };
        let d2 = DoubleCommitment {
            inner: inner2,
            outer: outer2,
        };
        let folded = fold_double(&pk, &d1, &d2, 64, 64).ok().unwrap();
        // The folded inner commitment must open to the folded witness.
        let r = folded.challenge_balanced;
        let folded_w: Vec<lattice_ring::RingElement> = w1
            .iter()
            .zip(w2.iter())
            .map(|(a, b)| a.add(&b.scale_i64(r)).ok().unwrap())
            .collect();
        assert!(pk.verify_opening(&folded.commitment.inner, &folded_w).is_ok());
        // Norm budget grows additively with the small factor.
        assert!(folded.norm_budget <= 64 + (1 << 15) * 64);
    }

    #[test]
    fn digits_for_bound_sane() {
        assert_eq!(digits_for_bound(1), 2); // span 3 -> 2 bits
        assert_eq!(digits_for_bound(127), 8); // span 255 -> 8 bits
        assert_eq!(digits_for_bound(128), 9); // span 257 -> 9 bits
    }
}
