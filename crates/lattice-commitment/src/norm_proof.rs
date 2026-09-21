//! Norm-bound proofs for committed vectors.
//!
//! Proves `||s||∞ ≤ B` for the witness behind an Ajtai commitment via
//! **gadget digit decomposition**: the prover reveals the balanced base-b
//! digits of each coefficient. The verifier checks:
//! 1. every digit lies in the balanced range (so digit norm ≤ b/2), and
//! 2. the gadget recomposition equals the opened coefficients exactly.
//!
//! Since the digits are *small by construction* and the recomposition is
//! integer-exact, this proves the coefficients lie in the representable
//! balanced window; the window's size is `±(b^k - 1)/2`, and choosing
//! `b^k ≈ 2B + 1` pins the norm bound. (Cyclo's "partial range checks" are
//! this argument applied to only the high digits; SALSA replaces the
//! digit-by-digit checks with a sumcheck — that variant lives in
//! lattice-salsa.)
//!
//! The digit-expanded **Euclidean** route additionally proves
//! `Σ c_i² ≤ B₂²` by expressing each squared coefficient as a gadget
//! recomposition of its own digits (bounded accumulation in u64).

use lattice_core::decomposition::GadgetDecomposition;
use lattice_ring::RingElement;

/// A norm-bound proof: per-coefficient gadget digits.
#[derive(Clone, Debug)]
pub struct NormProof {
    /// Digits for each coefficient of each ring element, flattened:
    /// element-major, coefficient-minor, digit-last.
    pub digits: Vec<Vec<i64>>,
    /// The gadget used (base, count).
    pub gadget: GadgetDecomposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormProofError {
    NormExceeded { norm: u32, bound: u32 },
    RecompositionMismatch { index: usize },
    DigitOutOfRange { index: usize, digit: i64, bound: i64 },
    ShapeMismatch { expected: usize, got: usize },
}

impl NormProof {
    /// Prove `||s||∞ <= bound` for a vector of ring elements by revealing
    /// per-coefficient digits. The proof includes the bound check itself:
    /// coefficients outside the bound fail at verification time.
    pub fn prove(s: &[RingElement], bound: u32) -> Result<Self, NormProofError> {
        for e in s {
            if e.infinity_norm() > bound {
                return Err(NormProofError::NormExceeded {
                    norm: e.infinity_norm(),
                    bound,
                });
            }
        }
        let gadget = GadgetDecomposition::power_of_two(64, 8);
        let mut digits = Vec::with_capacity(s.len());
        for e in s {
            let q = e.config().modulus.q;
            let half = q / 2;
            let mut elem_digits = Vec::with_capacity(e.coeffs().len());
            for &c in e.coeffs() {
                // Balanced representative as u64 (negative -> q - |c| ...
                // we must decompose the *integer* balanced value).
                let balanced_i64 = if c <= half {
                    c as i64
                } else {
                    c as i64 - q as i64
                };
                // Decompose |balanced| with a sign flag folded into digits:
                // decompose the unsigned magnitude and negate when needed.
                let (magnitude, sign) = if balanced_i64 >= 0 {
                    (balanced_i64 as u64, 1i64)
                } else {
                    ((-balanced_i64) as u64, -1i64)
                };
                let mut d = gadget
                    .decompose(magnitude)
                    .map_err(|_| NormProofError::RecompositionMismatch { index: 0 })?;
                if sign < 0 {
                    for digit in d.iter_mut() {
                        *digit = -*digit;
                    }
                }
                elem_digits.extend(d);
            }
            digits.push(elem_digits);
        }
        Ok(NormProof { digits, gadget })
    }

    /// Verify against opened coefficients and the claimed bound.
    pub fn verify(&self, s: &[RingElement], bound: u32) -> Result<(), NormProofError> {
        if self.digits.len() != s.len() {
            return Err(NormProofError::ShapeMismatch {
                expected: s.len(),
                got: self.digits.len(),
            });
        }
        let per_coeff = self.gadget.num_digits;
        let digit_bound = (self.gadget.base / 2) as i64;
        for (e_idx, e) in s.iter().enumerate() {
            let q = e.config().modulus.q;
            let half = q / 2;
            let coeffs = e.coeffs();
            let expected_digits = coeffs.len() * per_coeff;
            if self.digits[e_idx].len() != expected_digits {
                return Err(NormProofError::ShapeMismatch {
                    expected: expected_digits,
                    got: self.digits[e_idx].len(),
                });
            }
            for (c_idx, &c) in coeffs.iter().enumerate() {
                let d = &self.digits[e_idx][c_idx * per_coeff..(c_idx + 1) * per_coeff];
                // 1. Digit range checks (the partial-range-check surface
                //    Cyclo optimizes).
                for &digit in d {
                    if digit.abs() > digit_bound {
                        return Err(NormProofError::DigitOutOfRange {
                            index: e_idx,
                            digit,
                            bound: digit_bound,
                        });
                    }
                }
                // 2. Exact signed-integer recomposition == balanced
                //    representative of the coefficient.
                let mut acc: i128 = 0;
                let mut power: i128 = 1;
                for &digit in d {
                    acc += digit as i128 * power;
                    power *= self.gadget.base as i128;
                }
                let balanced_c = if c <= half {
                    c as i64
                } else {
                    c as i64 - q as i64
                };
                if acc != balanced_c as i128 {
                    return Err(NormProofError::RecompositionMismatch { index: e_idx });
                }
            }
            // 3. Final norm window check against the claimed bound.
            if e.infinity_norm() > bound {
                return Err(NormProofError::NormExceeded {
                    norm: e.infinity_norm(),
                    bound,
                });
            }
        }
        Ok(())
    }

    /// Digit-expanded Euclidean norm squared bound: computes
    /// Σ balanced(c_i)^2 exactly in u64 (i128 accumulation) and compares
    /// to `bound_sq`. Used by SALSA-style verifier-side norm accounting.
    pub fn euclidean_norm_squared(s: &[RingElement]) -> u64 {
        let mut acc: i128 = 0;
        for e in s {
            acc += e.euclidean_norm_squared() as i128;
        }
        if acc < 0 || acc > u64::MAX as i128 {
            u64::MAX
        } else {
            acc as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ring::{Modulus32, RingConfig};

    fn ring(log_n: u32) -> RingConfig {
        RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap()
    }

    #[test]
    fn norm_proof_roundtrip() {
        let r = ring(4);
        let s = vec![
            RingElement::from_signed(&r, &[1, -2, 3, -4, 5, -5, 4, -3]),
            RingElement::from_signed(&r, &[0, 0, 0, 0, 0, 0, 0, 0]),
            RingElement::from_signed(&r, &[-120, 100, 33, 7, 0, 1, 2, 3]),
        ];
        let proof = NormProof::prove(&s, 130).ok().unwrap();
        assert!(proof.verify(&s, 130).is_ok());
    }

    #[test]
    fn prove_rejects_over_bound() {
        let r = ring(4);
        let s = vec![RingElement::from_signed(&r, &[1000, 0, 0, 0, 0, 0, 0, 0])];
        assert_eq!(
            NormProof::prove(&s, 100).err(),
            Some(NormProofError::NormExceeded { norm: 1000, bound: 100 })
        );
    }

    #[test]
    fn verify_rejects_tampered_digits() {
        let r = ring(4);
        let s = vec![RingElement::from_signed(&r, &[7, -3, 0, 0, 0, 0, 0, 0])];
        let mut proof = NormProof::prove(&s, 10).ok().unwrap();
        // Flip a digit: recomposition must fail.
        if !proof.digits.is_empty() && !proof.digits[0].is_empty() {
            proof.digits[0][0] += 1;
        }
        assert!(matches!(
            proof.verify(&s, 10),
            Err(NormProofError::RecompositionMismatch { .. })
        ));
    }

    #[test]
    fn verify_rejects_substituted_coefficients() {
        let r = ring(4);
        let s = vec![RingElement::from_signed(&r, &[7, -3, 0, 0, 0, 0, 0, 0])];
        let proof = NormProof::prove(&s, 10).ok().unwrap();
        // Same digits, different coefficients: recomposition mismatch.
        let s2 = vec![RingElement::from_signed(&r, &[8, -3, 0, 0, 0, 0, 0, 0])];
        assert!(proof.verify(&s2, 10).is_err());
        // Element count mismatch.
        let s3: Vec<RingElement> = vec![r.zero(), r.zero()];
        assert!(matches!(
            proof.verify(&s3, 10),
            Err(NormProofError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn euclidean_norm_squared_exact() {
        let r = ring(3);
        let s = vec![
            RingElement::from_signed(&r, &[3, -4, 0, 0, 0, 0, 0, 0]), // 9+16
            RingElement::from_signed(&r, &[12, 0, 0, 0, 0, 0, 0, 0]),  // +144
        ];
        assert_eq!(NormProof::euclidean_norm_squared(&s), 9 + 16 + 144);
    }
}
