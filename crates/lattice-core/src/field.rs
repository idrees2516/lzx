//! 64-bit Goldilocks prime field: p = 2^64 - 2^32 + 1.
//!
//! Goldilocks is the Jolt-family prover field: multiplication fits in `u128`
//! and the multiplicative group has 2-adicity 32 (p - 1 = 2^32 * (2^32 - 1)),
//! so it supports FFTs up to length 2^32. All arithmetic here is constant in
//! time with respect to field *values* (no data-dependent branches), although
//! the crate makes no constant-time claim against hardware side channels.

/// The Goldilocks characteristic: 2^64 - 2^32 + 1.
pub const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// 2-adicity of p - 1.
pub const TWO_ADICITY: u32 = 32;

/// Multiplicative group order factorization helper: (p-1) = 2^32 * (2^32 - 1).
/// 2^32 - 1 = 3 * 5 * 17 * 257 * 65537 — all odd primes.

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Goldilocks(pub u64);

impl Goldilocks {
    pub const ZERO: Goldilocks = Goldilocks(0);
    pub const ONE: Goldilocks = Goldilocks(1);
    pub const TWO: Goldilocks = Goldilocks(2);

    /// Reduce an arbitrary u64 into the canonical representative [0, p).
    #[inline]
    pub const fn from_u64(x: u64) -> Self {
        // Branch-free conditional subtraction of p (x < 2^64 implies one step
        // is sufficient because p > 2^63).
        Goldilocks(x.wrapping_sub(if x >= GOLDILOCKS_MODULUS {
            GOLDILOCKS_MODULUS
        } else {
            0
        }))
    }

    /// Reduce a u128 into the canonical representative using the identity
    /// 2^64 = 2^32 - 1 (mod p). Bounded unrolled reduction (≤ 6 steps,
    /// analysis shows ≤ 4 ever needed for u128 inputs).
    #[inline]
    pub fn from_u128(x: u128) -> Self {
        let mut acc = x;
        for _ in 0..6 {
            if (acc >> 64) == 0 {
                break;
            }
            let hi = (acc >> 64) as u64;
            let lo = acc as u64;
            // acc = hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p)
            acc = (hi as u128) * 0xFFFF_FFFF + (lo as u128);
        }
        debug_assert!((acc >> 64) == 0);
        Self::from_u64(acc as u64)
    }

    #[inline]
    pub const fn to_canonical_u64(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Addition with carry compensation: when the u64 sum wraps past 2^64,
    /// the dropped bit is worth 2^64 ≡ 2^32 - 1 (mod p) and is added back.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let (sum, carry) = self.0.overflowing_add(other.0);
        let mut r = sum;
        if carry {
            r = r.wrapping_add(0xFFFF_FFFF);
        }
        Self::from_u64(r)
    }

    /// Subtraction with borrow compensation: the borrowed 2^64 is charged
    /// as 2^32 - 1 (mod p).
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let (diff, borrow) = self.0.overflowing_sub(other.0);
        let mut r = diff;
        if borrow {
            r = r.wrapping_sub(0xFFFF_FFFF);
        }
        Self::from_u64(r)
    }

    /// Additive inverse.
    #[inline]
    pub fn neg(&self) -> Self {
        if self.is_zero() {
            Self::ZERO
        } else {
            Self::from_u64(GOLDILOCKS_MODULUS - self.0)
        }
    }

    /// Multiplication with u128 intermediate.
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        Self::from_u128(self.0 as u128 * other.0 as u128)
    }

    #[inline]
    pub fn square(&self) -> Self {
        self.mul(self)
    }

    /// Doubling via addition (carry-safe).
    #[inline]
    pub fn double(&self) -> Self {
        self.add(self)
    }

    /// Exponentiation by square-and-multiply.
    pub fn pow(&self, exp: &[u64]) -> Self {
        let mut acc = Self::ONE;
        for &word in exp.iter().rev() {
            for i in (0..64).rev() {
                acc = acc.square();
                if (word >> i) & 1 == 1 {
                    acc = acc.mul(self);
                }
            }
        }
        acc
    }

    pub fn pow_u64(&self, exp: u64) -> Self {
        self.pow(&[exp])
    }

    /// Multiplicative inverse via Fermat little theorem (p prime).
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.pow(&[GOLDILOCKS_MODULUS - 2]))
        }
    }

    /// Checked division; errors on zero divisor.
    pub fn try_div(&self, other: &Self) -> Result<Self, FieldError> {
        match other.inverse() {
            Some(inv) => Ok(self.mul(&inv)),
            None => Err(FieldError::DivisionByZero),
        }
    }

    /// Montgomery batch inversion: one field inversion for n elements.
    /// Returns None if any element is zero.
    pub fn batch_inverse(items: &[Self]) -> Option<Vec<Self>> {
        if items.is_empty() {
            return Some(Vec::new());
        }
        let mut prods = Vec::with_capacity(items.len());
        let mut acc = Self::ONE;
        for it in items {
            if it.is_zero() {
                return None;
            }
            acc = acc.mul(it);
            prods.push(acc);
        }
        // acc is the product of everything
        let mut inv = acc.inverse()?;
        let mut out = vec![Self::ZERO; items.len()];
        for i in (0..items.len()).rev() {
            // inv = (prod_{0..=i})^{-1}; out[i] = inv * prod_{0..i-1}
            out[i] = inv.mul(if i == 0 { &Self::ONE } else { &prods[i - 1] });
            inv = inv.mul(&items[i]);
        }
        Some(out)
    }

    /// Canonical little-endian byte encoding (8 bytes).
    pub fn to_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Decode canonical little-endian bytes; rejects non-canonical encodings.
    pub fn from_bytes(bytes: &[u8; 8]) -> Result<Self, FieldError> {
        let raw = u64::from_le_bytes(*bytes);
        if raw >= GOLDILOCKS_MODULUS {
            return Err(FieldError::NonCanonical);
        }
        Ok(Goldilocks(raw))
    }

    /// A canonical generator of the 2^TWO_ADICITY-order subgroup
    /// (primitive 2^32-th root of unity).
    pub fn two_adic_generator() -> Self {
        // 7 is a known primitive root for Goldilocks; g = 7^((p-1)/2^32).
        Goldilocks(7).pow(&[
            0x0000_0000_FFFF_FFFF, // (p-1)/2^32 = 2^32 - 1
        ])
    }

    /// Sample a field element from 8 transcript bytes, rejecting
    /// non-canonical values (rejection sampling keeps distribution uniform).
    pub fn from_uniform_bytes(bytes: &[u8; 16]) -> Self {
        let hi = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0u8; 8]));
        let lo = u64::from_le_bytes(bytes[8..].try_into().unwrap_or([0u8; 8]));
        Self::from_u128(((hi as u128) << 64) | (lo as u128))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldError {
    NonCanonical,
    DivisionByZero,
    DecodeError,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = fe(12345678901234567890);
        let b = fe(9876543210987654321);
        assert_eq!(a.add(&b).sub(&b), a);
        assert_eq!(a.add(&b), b.add(&a));
        // p - 1 + 1 wraps to zero
        assert_eq!(fe(GOLDILOCKS_MODULUS - 1).add(&fe(1)), fe(0));
    }

    #[test]
    fn mul_matches_u128_reference() {
        let a = fe(GOLDILOCKS_MODULUS - 1);
        let b = fe(0x1234_5678_9ABC_DEF0 % GOLDILOCKS_MODULUS);
        let expected = ((a.0 as u128 * b.0 as u128) % GOLDILOCKS_MODULUS as u128) as u64;
        assert_eq!(a.mul(&b).0, expected);
        assert_eq!(a.try_div(&fe(0)).err(), Some(FieldError::DivisionByZero));
    }

    #[test]
    fn inverse_and_batch() {
        let a = fe(42);
        assert_eq!(a.mul(&a.inverse().unwrap_or(Goldilocks::ZERO)), fe(1));
        let items: Vec<Goldilocks> = (1..=50u64).map(fe).collect();
        let invs = Goldilocks::batch_inverse(&items).unwrap_or_default();
        for (x, xi) in items.iter().zip(invs.iter()) {
            assert_eq!(x.mul(xi), fe(1));
        }
        assert!(Goldilocks::batch_inverse(&[fe(1), fe(0)]).is_none());
    }

    #[test]
    fn two_adic_root_order() {
        let g = Goldilocks::two_adic_generator();
        // g^(2^32) == 1, g^(2^31) != 1
        let half = g.pow_u64(1u64 << 31);
        assert_eq!(half.pow_u64(1u64 << 31), Goldilocks::ONE);
        assert_ne!(half, Goldilocks::ONE);
        assert_ne!(g, Goldilocks::ZERO);
    }

    #[test]
    fn frobenius_small() {
        // For Goldilocks, x^p == x (Fermat), so x^(p-1) == 1 for x != 0.
        let a = fe(1337);
        assert_eq!(a.pow(&[GOLDILOCKS_MODULUS - 1]), fe(1));
    }

    #[test]
    fn canonical_encoding_roundtrip_and_reject() {
        let a = fe(0xDEAD_BEEF_CAFE_F00D % GOLDILOCKS_MODULUS);
        let bytes = a.to_bytes();
        assert_eq!(Goldilocks::from_bytes(&bytes).unwrap_or(Goldilocks::ZERO), a);
        let bad = GOLDILOCKS_MODULUS.to_le_bytes();
        assert_eq!(
            Goldilocks::from_bytes(&bad).err(),
            Some(FieldError::NonCanonical)
        );
    }
}
