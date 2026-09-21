//! Gadget decomposition utilities.
//!
//! Lattice commitments open small-norm secrets via a gadget matrix
//! `G = [1, b, b^2, ..., b^{d-1}]`: a value `x mod q` is decomposed into
//! signed digits in `[-b/2, b/2)` so that `<G, digits(x)> = x mod q` while
//! the digit vector has infinity-norm at most `b/2`.
//!
//! This is the exact mechanism used by:
//! * Ajtai commitment openings (Akita §commitment, norm checks),
//! * folding schemes' norm-control (LatticeFold+, Cyclo partial range
//!   checks, PikkuFold, ProtogaLattice),
//! * SALSA's digit decomposition in the norm-check sumcheck.

use crate::field::Goldilocks;

/// Balanced signed-digit decomposition of `x` (mod 2^64 range) in base `b`.
///
/// Returns digits `d_0..d_{k-1}` with each `d_i in [-(b/2), b/2)` (balanced)
/// such that `sum d_i b^i = x`, and `k` covers the full range of `x`
/// (`k = ceil(bits / log2(b))` where bits is the bit width of x).
///
/// The verifier reconstructs `sum d_i b^i` and checks equality, plus checks
/// each digit lies in the balanced range (the range check itself is what
/// folding schemes like Cyclo partially offload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GadgetDecomposition {
    pub base: u64,
    pub num_digits: usize,
}

impl GadgetDecomposition {
    /// Gadget with base `b` and `k` digits.
    pub fn new(base: u64, num_digits: usize) -> Self {
        GadgetDecomposition { base, num_digits }
    }

    /// Power-of-two base gadget with enough digits to cover `bits` bits
    /// under *balanced* digits (one extra digit beyond the unsigned need,
    /// because balanced range is symmetric around zero).
    pub fn power_of_two(bits: u32, log_base: u32) -> Self {
        let base = 1u64 << log_base;
        let num_digits = ((bits + 1 + log_base - 1) / log_base) as usize;
        GadgetDecomposition { base, num_digits }
    }

    /// Decompose a u64 into balanced digits (each in `[-base/2, base/2)`).
    /// `None` when the value cannot be represented with the configured
    /// digit count (caller must reject, never wrap).
    pub fn decompose(&self, x: u64) -> Result<Vec<i64>, DecompositionError> {
        let mut rem: u128 = x as u128;
        let mut out = Vec::with_capacity(self.num_digits);
        let half = (self.base / 2) as i64;
        for _ in 0..self.num_digits {
            // Least significant base-b digit, mapped into the balanced range
            // by borrowing when it exceeds base/2. u128 intermediate so the
            // borrow never overflows.
            let digit = (rem % self.base as u128) as i64;
            let d = if digit > half { digit - self.base as i64 } else { digit };
            // i128 intermediate: a negative d would wrap if cast to u128.
            rem = ((rem as i128 - d as i128) / self.base as i128) as u128;
            out.push(d);
        }
        if rem != 0 {
            return Err(DecompositionError::ValueTooLarge);
        }
        Ok(out)
    }

    /// Recompose digits into the original value (exact integer arithmetic).
    pub fn recompose(&self, digits: &[i64]) -> Result<u64, DecompositionError> {
        if digits.len() != self.num_digits {
            return Err(DecompositionError::DigitCountMismatch);
        }
        let mut acc: i128 = 0;
        let mut power: i128 = 1;
        for &d in digits {
            let half = (self.base / 2) as i64;
            if d < -half || d > half {
                return Err(DecompositionError::DigitOutOfRange { digit: d, bound: half });
            }
            acc += d as i128 * power;
            power *= self.base as i128;
        }
        if acc < 0 || acc > u64::MAX as i128 {
            return Err(DecompositionError::ValueOutOfRange);
        }
        Ok(acc as u64)
    }

    /// Decompose a Goldilocks element (canonical u64 residue).
    pub fn decompose_field(&self, x: &Goldilocks) -> Result<Vec<i64>, DecompositionError> {
        self.decompose(x.to_canonical_u64())
    }

    /// Recompose into a Goldilocks element (reducing mod p).
    pub fn recompose_field(&self, digits: &[i64]) -> Result<Goldilocks, DecompositionError> {
        Ok(Goldilocks::from_u64(self.recompose(digits)?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionError {
    DigitCountMismatch,
    DigitOutOfRange { digit: i64, bound: i64 },
    ValueOutOfRange,
    ValueTooLarge,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_boundaries() {
        let g = GadgetDecomposition::power_of_two(64, 8); // base 256, 8 digits
        for &x in &[
            0u64,
            1,
            127,
            128, // boundary: becomes -128 with carry
            129,
            255,
            256,
            65535,
            u32::MAX as u64,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let digits = g.decompose(x).unwrap();
            assert_eq!(g.recompose(&digits).unwrap(), x);
        }
    }

    #[test]
    fn balanced_digits_bounded() {
        let g = GadgetDecomposition::power_of_two(64, 4);
        for x in (0..100_000u64).step_by(977) {
            let digits = g.decompose(x).unwrap();
            for &d in &digits {
                assert!(d >= -8 && d <= 8, "digit {d} out of balanced range");
            }
        }
    }

    #[test]
    fn rejects_out_of_range_digit() {
        let g = GadgetDecomposition::power_of_two(64, 8); // 9 balanced digits
        let mut digits = [0i64; 9];
        digits[0] = 200;
        assert_eq!(
            g.recompose(&digits).err(),
            Some(DecompositionError::DigitOutOfRange { digit: 200, bound: 128 })
        );
        assert_eq!(
            g.recompose(&[0; 7]).err(),
            Some(DecompositionError::DigitCountMismatch)
        );
    }

    #[test]
    fn field_roundtrip() {
        let g = GadgetDecomposition::power_of_two(64, 11);
        let x = Goldilocks::from_u64(0xDEAD_BEEF_CAFE_F00D % crate::field::GOLDILOCKS_MODULUS);
        let digits = g.decompose_field(&x).unwrap();
        assert_eq!(g.recompose_field(&digits).unwrap(), x);
    }

    #[test]
    fn sum_of_digits_relation() {
        // <G, digits> must equal the field element: the core gadget identity
        // relied upon by Ajtai openings and norm checks.
        let g = GadgetDecomposition::power_of_two(64, 16);
        for x in (1..50_000u64).step_by(1234) {
            let digits = g.decompose(x).unwrap();
            let mut sum = Goldilocks::ZERO;
            let mut pow = Goldilocks::ONE;
            for &d in &digits {
                let term = if d >= 0 {
                    Goldilocks::from_u64(d as u64).mul(&pow)
                } else {
                    Goldilocks::ZERO.sub(&Goldilocks::from_u64((-d) as u64).mul(&pow))
                };
                sum = sum.add(&term);
                pow = pow.mul(&Goldilocks::from_u64(g.base));
            }
            assert_eq!(sum, Goldilocks::from_u64(x));
        }
    }
}
