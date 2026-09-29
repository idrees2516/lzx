//! Quadratic extension field `F_{q^2}` over Goldilocks (Wave 6 substrate,
//! `NEXT_STEPS.md` §2.3).
//!
//! Five papers communicate sumcheck rounds as single **F_{q^e}** elements
//! (subfield batching via θ_a / Φ_δ / CRT slots): Cyclo §3 (sumchecks over
//! R_{q^e}), PikkuFold RingSC §2.5, RoKoko Π^lin §7, SALSA Π^sum final
//! check, Symphony (K = F_{q²} tensor ring). This module provides the
//! field-theoretic substrate over the Goldilocks base field:
//!
//! * arithmetic in `F_{p²} = F_p[Y] / (Y² - 7)` — **7 is the smallest
//!   quadratic non-residue mod p** (`7^((p-1)/2) ≡ -1`, verified by test),
//!   which makes `Y² - 7` irreducible over F_p. (Note: `X² + 1` is NOT
//!   irreducible here — Goldilocks `p ≡ 1 (mod 4)` so `sqrt(-1) = 2^48`
//!   lives in the base field and `F_p[i]/(i²+1)` splits as F_p × F_p; the
//!   Frobenius test catches exactly this degeneracy),
//! * Karatsuba multiplication (3 base multiplications),
//! * inversion via the norm (`z⁻¹ = conj(z)/N(z)` — the norm is a base
//!   field element, so one base inversion),
//! * conjugation = the Frobenius automorphism `z ↦ z^p` (Y^p = -Y for the
//!   non-residue constant),
//! * canonical serialization and **transcript sampling** with unbiased
//!   rejection on both limbs.
//!
//! `lattice_ring::extension` provides the ring-level counterpart
//! (R_q[Y]/(Y² + 1) over a Modulus32 base) for the R_q-anchored protocols.

use crate::field::{FieldError, Goldilocks};
use crate::transcript::{Transcript, TranscriptError};

/// The quadratic non-residue extension constant: `Y² = 7` (7 is the
/// smallest QNR mod Goldilocks — `X² + 1` would split since p ≡ 1 mod 4).
pub const EXT_D: u64 = 7;

/// An element `c0 + c1·Y` of F_{p²} over Goldilocks with `Y² = EXT_D`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fq2 {
    pub c0: Goldilocks,
    pub c1: Goldilocks,
}

impl Fq2 {
    pub const ZERO: Fq2 = Fq2 {
        c0: Goldilocks::ZERO,
        c1: Goldilocks::ZERO,
    };
    pub const ONE: Fq2 = Fq2 {
        c0: Goldilocks::ONE,
        c1: Goldilocks::ZERO,
    };
    /// The generator `Y` with `Y² = EXT_D`.
    pub const I: Fq2 = Fq2 {
        c0: Goldilocks::ZERO,
        c1: Goldilocks::ONE,
    };

    /// The generator `Y` (alias of [`Fq2::I`]).
    pub const Y: Fq2 = Fq2::I;

    pub const fn new(c0: Goldilocks, c1: Goldilocks) -> Self {
        Fq2 { c0, c1 }
    }

    pub fn from_base(c0: Goldilocks) -> Self {
        Fq2 { c0, c1: Goldilocks::ZERO }
    }

    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        Fq2 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// Pointwise addition.
    pub fn add(&self, other: &Self) -> Self {
        Fq2 {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
        }
    }

    /// Pointwise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Fq2 {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
        }
    }

    /// Doubling.
    pub fn double(&self) -> Self {
        Fq2 {
            c0: self.c0.double(),
            c1: self.c1.double(),
        }
    }

    /// Karatsuba multiplication: `(a0 + a1 Y)(b0 + b1 Y)` with `Y² = D`:
    /// `c0 = a0b0 + D·a1b1`, `c1 = (a0+a1)(b0+b1) - a0b0 - a1b1`.
    pub fn mul(&self, other: &Self) -> Self {
        let a0b0 = self.c0.mul(&other.c0);
        let a1b1 = self.c1.mul(&other.c1);
        let cross = self.c0.add(&self.c1).mul(&other.c0.add(&other.c1));
        Fq2 {
            c0: a0b0.add(&Goldilocks::from_u64(EXT_D).mul(&a1b1)),
            c1: cross.sub(&a0b0).sub(&a1b1),
        }
    }

    /// Squaring via the same Karatsuba shape.
    pub fn square(&self) -> Self {
        self.mul(self)
    }

    /// Conjugation `c0 - c1·Y` — the Frobenius `z ↦ z^p` (Y^p = -Y since
    /// D is a non-residue).
    pub fn conjugate(&self) -> Self {
        Fq2 {
            c0: self.c0,
            c1: self.c1.neg(),
        }
    }

    /// Norm `N(z) = z·conj(z) = c0² - D·c1²` — a base-field element.
    pub fn norm(&self) -> Goldilocks {
        let c0sq = self.c0.square();
        let dc1sq = Goldilocks::from_u64(EXT_D).mul(&self.c1.square());
        c0sq.sub(&dc1sq)
    }

    /// Trace `Tr(z) = z + conj(z) = 2·c0`.
    pub fn trace(&self) -> Goldilocks {
        self.c0.double()
    }

    /// Multiplicative inverse via `z⁻¹ = conj(z) / N(z)`; `None` for zero.
    pub fn inverse(&self) -> Option<Self> {
        let n = self.norm();
        let n_inv = n.inverse()?;
        Some(Fq2 {
            c0: self.c0.mul(&n_inv),
            c1: self.c1.neg().mul(&n_inv),
        })
    }

    /// Exponentiation by a u64 exponent (square-and-multiply).
    pub fn pow_u64(&self, exp: u64) -> Self {
        let mut acc = Fq2::ONE;
        let mut base = *self;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }

    /// Canonical 16-byte serialization (two 8-byte base limbs).
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.c0.to_bytes());
        out[8..].copy_from_slice(&self.c1.to_bytes());
        out
    }

    /// Decode canonical bytes; rejects non-canonical limbs.
    pub fn from_bytes(bytes: &[u8; 16]) -> Result<Self, FieldError> {
        let mut c0_bytes = [0u8; 8];
        c0_bytes.copy_from_slice(&bytes[..8]);
        let mut c1_bytes = [0u8; 8];
        c1_bytes.copy_from_slice(&bytes[8..]);
        let c0 = Goldilocks::from_bytes(&c0_bytes)?;
        let c1 = Goldilocks::from_bytes(&c1_bytes)?;
        Ok(Fq2 { c0, c1 })
    }

    /// Frobenius consistency check used by tests: `z^p == conj(z)`
    /// (`(a+b)^p = a^p + b^p` in characteristic p; `Y^p = -Y` because the
    /// extension constant is a quadratic non-residue).
    pub fn frobenius(&self) -> Self {
        self.conjugate()
    }
}

impl Default for Fq2 {
    fn default() -> Self {
        Fq2::ZERO
    }
}

/// Sample one F_{p²} challenge from the transcript: both limbs drawn with
/// the same unbiased rejection as base-field challenges, one query.
pub fn challenge_fq2(
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<Fq2, TranscriptError> {
    let fields = transcript.challenge_fields(label, 2)?;
    let mut it = fields.into_iter();
    let c0 = it.next().unwrap_or(Goldilocks::ZERO);
    let c1 = it.next().unwrap_or(Goldilocks::ZERO);
    Ok(Fq2 { c0, c1 })
}

/// Sample `n` F_{p²} challenges (batched rounds for extension-field
/// sumchecks).
pub fn challenge_fq2_vec(
    transcript: &mut Transcript,
    label: &[u8],
    n: usize,
) -> Result<Vec<Fq2>, TranscriptError> {
    let fields = transcript.challenge_fields(label, 2 * n)?;
    Ok(fields
        .chunks(2)
        .map(|pair| {
            let c0 = pair.first().copied().unwrap_or(Goldilocks::ZERO);
            let c1 = pair.get(1).copied().unwrap_or(Goldilocks::ZERO);
            Fq2 { c0, c1 }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GOLDILOCKS_MODULUS;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn generator_squares_to_extension_constant() {
        // Y² = 7 in F_{p²}.
        assert_eq!(Fq2::Y.square(), Fq2::from_base(fe(EXT_D)));
        // 7 is a quadratic NON-residue mod p (the irreducibility witness):
        // 7^((p-1)/2) ≡ -1.
        let euler = fe(EXT_D).pow_u64((GOLDILOCKS_MODULUS - 1) / 2);
        assert_eq!(euler, fe(GOLDILOCKS_MODULUS - 1));
        // And X²+1 would split: sqrt(-1) = 2^48 in the base field.
        let i_sq = fe(1u64 << 48).square();
        assert_eq!(i_sq, fe(GOLDILOCKS_MODULUS - 1));
    }

    #[test]
    fn field_axioms() {
        let a = Fq2::new(fe(3), fe(5));
        let b = Fq2::new(fe(7), fe(11));
        let c = Fq2::new(fe(13), fe(17));
        // Commutativity / associativity of mul.
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        // Distributivity.
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        // Identity / zero.
        assert_eq!(a.mul(&Fq2::ONE), a);
        assert_eq!(a.add(&Fq2::ZERO), a);
        assert_eq!(a.mul(&Fq2::ZERO), Fq2::ZERO);
        // Subtraction is the inverse of addition.
        assert_eq!(a.add(&b).sub(&b), a);
        // Negation.
        assert_eq!(a.add(&a.neg()), Fq2::ZERO);
    }

    #[test]
    fn inverse_and_division() {
        for (c0, c1) in [(1u64, 0), (0, 1), (3, 5), (0xFFFF_FFFF_0000_0000 - 1, 42)] {
            let z = Fq2::new(fe(c0), fe(c1));
            if z.is_zero() {
                continue;
            }
            let inv = match z.inverse() {
                Some(v) => v,
                None => continue,
            };
            assert_eq!(z.mul(&inv), Fq2::ONE);
            assert_eq!(inv.mul(&z), Fq2::ONE);
        }
        assert!(Fq2::ZERO.inverse().is_none());
        // I⁻¹ = conj(I)/N(I) with N(I) = -D.
        let y_inv = Fq2::I.inverse();
        assert_eq!(y_inv.map(|v| Fq2::I.mul(&v)), Some(Fq2::ONE));
    }

    #[test]
    fn frobenius_is_conjugation() {
        // z^p == conj(z) for random z (the field is genuinely degree-2:
        // the Frobenius is the non-trivial automorphism because the
        // extension constant is a non-residue).
        for (c0, c1) in [(2u64, 3), (5, 7), (11, 13), (1, 1)] {
            let z = Fq2::new(fe(c0), fe(c1));
            assert_eq!(z.pow_u64(GOLDILOCKS_MODULUS), z.conjugate(), "z=({c0},{c1})");
            assert_ne!(z.pow_u64(GOLDILOCKS_MODULUS), z, "Frobenius is trivial — extension degenerate");
        }
    }

    #[test]
    fn norm_multiplicative_and_trace() {
        let a = Fq2::new(fe(3), fe(5));
        let b = Fq2::new(fe(7), fe(11));
        assert_eq!(
            a.mul(&b).norm(),
            a.norm().mul(&b.norm())
        );
        // Norm is a base-field element and multiplicative; norm of Y is
        // -D (Y·conj(Y) = -Y² = -D).
        assert_eq!(Fq2::Y.norm(), fe(GOLDILOCKS_MODULUS - EXT_D));
        // Trace is base-field valued: Tr(a·b) vs... just check Tr(a) = 2c0.
        assert_eq!(a.trace(), fe(6));
    }

    #[test]
    fn serialization_roundtrip_and_rejection() {
        let z = Fq2::new(fe(0x1234_5678_9ABC_DEF0 % GOLDILOCKS_MODULUS), fe(99));
        let bytes = z.to_bytes();
        assert_eq!(Fq2::from_bytes(&bytes).ok().unwrap(), z);
        // Non-canonical limb rejected (value ≥ p).
        let mut bad = bytes;
        for b in bad.iter_mut().take(8) {
            *b = 0xFF;
        }
        assert!(Fq2::from_bytes(&bad).is_err());
    }

    #[test]
    fn transcript_sampling_deterministic_and_canonical() {
        let mut t1 = Transcript::new_default(b"fq2-test");
        let mut t2 = Transcript::new_default(b"fq2-test");
        let z1 = challenge_fq2(&mut t1, b"chal").ok().unwrap();
        let z2 = challenge_fq2(&mut t2, b"chal").ok().unwrap();
        assert_eq!(z1, z2);
        assert!(!z1.is_zero());
        assert!(z1.c0.to_canonical_u64() < GOLDILOCKS_MODULUS);
        assert!(z1.c1.to_canonical_u64() < GOLDILOCKS_MODULUS);
        // Different labels diverge.
        let mut t3 = Transcript::new_default(b"fq2-test");
        let z3 = challenge_fq2(&mut t3, b"other").ok().unwrap();
        assert_ne!(z1, z3);

        // Vector sampling.
        let mut t4 = Transcript::new_default(b"fq2-vec");
        let v = challenge_fq2_vec(&mut t4, b"rounds", 16).ok().unwrap();
        assert_eq!(v.len(), 16);
        for z in &v {
            assert!(!z.is_zero());
        }
    }

    #[test]
    fn karatsuba_matches_schoolbook() {
        // Reference product with explicit Y² = D reduction.
        let a = Fq2::new(fe(1234567), fe(GOLDILOCKS_MODULUS - 7));
        let b = Fq2::new(fe(GOLDILOCKS_MODULUS - 9), fe(89012345));
        let d = fe(EXT_D);
        let c0 = a.c0.mul(&b.c0).add(&d.mul(&a.c1.mul(&b.c1)));
        let c1 = a.c0.mul(&b.c1).add(&a.c1.mul(&b.c0));
        let expected = Fq2::new(c0, c1);
        assert_eq!(a.mul(&b), expected);
    }
}
