//! Quadratic extension **ring** `R_{q,2} = R_q[Y] / (Y² + 1)` (Wave 6
//! substrate, `NEXT_STEPS.md` §2.3 — the R_q-anchored counterpart of
//! `lattice_core::extension::Fq2`).
//!
//! Cyclo §3 (sumchecks over R_{q^e}), PikkuFold's RingSC, RoKoko's Π^lin
//! and SALSA's Π^sum all batch their sumcheck rounds into single
//! extension-ring elements. This module provides the arithmetic over
//! coefficient pairs `(c0, c1)` with `Y² = -1`:
//!
//! * Karatsuba multiplication (3 R_q products instead of 4),
//! * conjugation `c0 - c1·Y` and the relative norm `c0² + c1²`,
//! * small-coefficient sampling from a lattice-core short-challenge spec
//!   (both halves from the same distribution family).
//!
//! **Irreducibility caveat, stated honestly**: whether `Y² + 1` is
//! irreducible over R_q depends on q and n (e.g. over F_q it needs
//! q ≡ 3 mod 4). The protocols in the papers choose their moduli so the
//! extension is a domain at their parameters; this module provides exact
//! *algebra* regardless, and invertibility is per-element (via the norm
//! when it is a unit in R_q). Soundness-critical uses must instantiate at
//! parameters where the paper's irreducibility conditions hold.

use crate::ring::{RingConfig, RingElement, RingError};
use lattice_core::short_challenge::{ShortChallenge, ShortChallengeError, ShortChallengeSpec};

/// An element `c0 + c1·Y` of `R_q[Y]/(Y² + 1)`.
#[derive(Clone)]
pub struct Rq2 {
    pub c0: RingElement,
    pub c1: RingElement,
}

impl PartialEq for Rq2 {
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}

impl Eq for Rq2 {}

impl std::fmt::Debug for Rq2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rq2")
            .field("c0", &self.c0)
            .field("c1", &self.c1)
            .finish()
    }
}

impl Rq2 {
    /// Construct from two ring elements (lengths must match the config).
    pub fn new(c0: RingElement, c1: RingElement) -> Result<Self, RingError> {
        if c0.config().modulus.q != c1.config().modulus.q
            || c0.config().log_n != c1.config().log_n
        {
            return Err(RingError::LengthMismatch {
                expected: c0.coeffs().len(),
                got: c1.coeffs().len(),
            });
        }
        Ok(Rq2 { c0, c1 })
    }

    /// Zero element.
    pub fn zero(ring: &RingConfig) -> Self {
        Rq2 {
            c0: ring.zero(),
            c1: ring.zero(),
        }
    }

    /// One element.
    pub fn one(ring: &RingConfig) -> Self {
        Rq2 {
            c0: ring.one(),
            c1: ring.zero(),
        }
    }

    /// The generator Y (with Y² = -1).
    pub fn y(ring: &RingConfig) -> Self {
        Rq2 {
            c0: ring.zero(),
            c1: ring.one(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    /// Pointwise addition.
    pub fn add(&self, other: &Self) -> Result<Self, RingError> {
        Ok(Rq2 {
            c0: self.c0.add(&other.c0)?,
            c1: self.c1.add(&other.c1)?,
        })
    }

    /// Pointwise subtraction.
    pub fn sub(&self, other: &Self) -> Result<Self, RingError> {
        Ok(Rq2 {
            c0: self.c0.sub(&other.c0)?,
            c1: self.c1.sub(&other.c1)?,
        })
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        Rq2 {
            c0: self.c0.neg(),
            c1: self.c1.neg(),
        }
    }

    /// Multiplication via Karatsuba:
    /// `(a0 + a1 Y)(b0 + b1 Y) = (a0b0 - a1b1) + (a0b1 + a1b0) Y`
    /// computed with three R_q products: `p0 = a0b0`, `p1 = a1b1`,
    /// `pc = (a0+a1)(b0+b1)`.
    pub fn mul(&self, other: &Self) -> Result<Self, RingError> {
        let p0 = self.c0.mul(&other.c0)?;
        let p1 = self.c1.mul(&other.c1)?;
        let pc = self
            .c0
            .add(&self.c1)?
            .mul(&other.c0.add(&other.c1)?)?;
        Ok(Rq2 {
            c0: p0.sub(&p1)?,
            c1: pc.sub(&p0)?.sub(&p1)?,
        })
    }

    /// Squaring.
    pub fn square(&self) -> Result<Self, RingError> {
        self.mul(self)
    }

    /// Conjugation `c0 - c1·Y` (an involutive ring automorphism whenever
    /// the extension is a domain).
    pub fn conjugate(&self) -> Self {
        Rq2 {
            c0: self.c0.clone(),
            c1: self.c1.neg(),
        }
    }

    /// Relative norm `N(z) = z·conj(z) = c0² + c1²` (an R_q element).
    pub fn norm(&self) -> Result<RingElement, RingError> {
        let c0sq = self.c0.mul(&self.c0)?;
        let c1sq = self.c1.mul(&self.c1)?;
        c0sq.add(&c1sq)
    }

    /// Inversion via `z⁻¹ = conj(z) · N(z)⁻¹` when the norm is invertible
    /// in R_q (checked by coefficient-wise inversion of all NTT slots).
    pub fn inverse(&self) -> Option<Self> {
        let n = self.norm().ok()?;
        // Invert the norm element slot-wise in the NTT domain.
        let evals = n.to_ntt().ok()?;
        let ring = self.c0.config();
        let mut inv_evals = Vec::with_capacity(evals.len());
        for e in &evals {
            inv_evals.push(ring.modulus.inv(*e)?);
        }
        let n_inv = RingElement::from_ntt(ring, &inv_evals).ok()?;
        Some(Rq2 {
            c0: self.c0.mul(&n_inv).ok()?,
            c1: self.c1.neg().mul(&n_inv).ok()?,
        })
    }

    /// Canonical byte serialization (c0 then c1).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.c0.to_bytes();
        out.extend_from_slice(&self.c1.to_bytes());
        out
    }
}

/// Sample an extension-ring challenge: both halves drawn from the same
/// short-challenge family (the papers' `c ∈ C ⊂ R_{q^e}`). The spec must
/// cover exactly `2·n` slots so each half receives `n` coefficients
/// (even-indexed stream positions → c0, odd → c1).
pub fn sample_rq2_challenge(
    ring: &RingConfig,
    spec: &ShortChallengeSpec,
    seed: &[u8],
) -> Result<(Rq2, u64, u64), ShortChallengeError> {
    if spec.n != 2 * ring.n() {
        return Err(ShortChallengeError::InvalidParameters);
    }
    let c: ShortChallenge = spec.sample(seed)?;
    // Split the coefficient stream: even indices -> c0, odd -> c1 (keeps
    // both halves at the family's distribution over n/2 slots each).
    let mut c0_signed = Vec::with_capacity(ring.n());
    let mut c1_signed = Vec::with_capacity(ring.n());
    for (i, v) in c.coefficients.iter().enumerate() {
        if i % 2 == 0 {
            c0_signed.push(*v);
        } else {
            c1_signed.push(*v);
        }
        if c0_signed.len() == ring.n() && c1_signed.len() == ring.n() {
            break;
        }
    }
    while c0_signed.len() < ring.n() {
        c0_signed.push(0);
    }
    while c1_signed.len() < ring.n() {
        c1_signed.push(0);
    }
    let c0 = RingElement::from_signed(ring, &c0_signed);
    let c1 = RingElement::from_signed(ring, &c1_signed);
    // Per-half certified Γ (each half carries at most half the mass).
    let l2sq = |v: &[i64]| -> u128 {
        v.iter().map(|c| (*c as i128 * *c as i128) as u128).sum()
    };
    let g0 = ceil_sqrt_u128(l2sq(&c0_signed));
    let g1 = ceil_sqrt_u128(l2sq(&c1_signed));
    Ok((Rq2 { c0, c1 }, g0, g1))
}

/// Ceiling square root over u128 (float seed + integer correction).
fn ceil_sqrt_u128(x: u128) -> u64 {
    if x == 0 {
        return 0;
    }
    if x <= u64::MAX as u128 {
        return lattice_core::norm_budget::ceil_sqrt(x as u64);
    }
    let mut g = (x as f64).sqrt() as u128 + 2;
    loop {
        let next = (g + x / g) / 2;
        if next >= g {
            break;
        }
        g = next;
    }
    while g.saturating_mul(g) > x {
        g -= 1;
    }
    while let Some(p) = (g + 1).checked_mul(g + 1) {
        if p <= x {
            g += 1;
        } else {
            break;
        }
    }
    let g64 = u64::try_from(g).unwrap_or(u64::MAX);
    if g.saturating_mul(g) < x {
        g64.saturating_add(1)
    } else {
        g64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulus::Modulus32;

    fn ring() -> RingConfig {
        RingConfig::new(Modulus32::Q_32, 6).ok().unwrap()
    }

    fn elem(ring: &RingConfig, tag: &[u8]) -> RingElement {
        ring.random(tag)
    }

    #[test]
    fn y_squared_is_minus_one() {
        let r = ring();
        let y2 = Rq2::y(&r).square().ok().unwrap();
        let minus_one_c0 = r.constant(r.modulus.q - 1);
        assert!(y2.c0 == minus_one_c0 && y2.c1.is_zero());
    }

    #[test]
    fn ring_axioms() {
        let r = ring();
        let a = Rq2::new(elem(&r, b"a0"), elem(&r, b"a1")).ok().unwrap();
        let b = Rq2::new(elem(&r, b"b0"), elem(&r, b"b1")).ok().unwrap();
        let c = Rq2::new(elem(&r, b"c0"), elem(&r, b"c1")).ok().unwrap();
        assert_eq!(a.mul(&b).ok().unwrap(), b.mul(&a).ok().unwrap());
        assert_eq!(
            a.mul(&b).ok().unwrap().mul(&c).ok().unwrap(),
            a.mul(&b.mul(&c).ok().unwrap()).ok().unwrap()
        );
        assert_eq!(
            a.mul(&b.add(&c).ok().unwrap()).ok().unwrap(),
            a.mul(&b).ok().unwrap().add(&a.mul(&c).ok().unwrap()).ok().unwrap()
        );
        assert_eq!(a.mul(&Rq2::one(&r)).ok().unwrap(), a);
        assert!(a.mul(&Rq2::zero(&r)).ok().unwrap().is_zero());
        assert!(a.add(&a.neg()).ok().unwrap().is_zero());
    }

    #[test]
    fn conjugation_involutive_and_norm() {
        let r = ring();
        let a = Rq2::new(elem(&r, b"n0"), elem(&r, b"n1")).ok().unwrap();
        assert_eq!(a.conjugate().conjugate().c0, a.c0);
        assert_eq!(a.conjugate().conjugate().c1, a.c1);
        // z·conj(z) == N(z) (a pure R_q element).
        let z_conj = a.mul(&a.conjugate()).ok().unwrap();
        let n = a.norm().ok().unwrap();
        assert!(z_conj.c1.is_zero());
        assert_eq!(z_conj.c0, n);
        // Norm is multiplicative.
        let b = Rq2::new(elem(&r, b"m0"), elem(&r, b"m1")).ok().unwrap();
        assert_eq!(
            a.mul(&b).ok().unwrap().norm().ok().unwrap(),
            a.norm().ok().unwrap().mul(&b.norm().ok().unwrap()).ok().unwrap()
        );
    }

    #[test]
    fn inverse_via_norm_when_invertible() {
        let r = ring();
        // Small coefficients keep the norm's NTT slots away from zero whp.
        let a = Rq2::new(
            lattice_commitment_test_elem(&r, b"i0"),
            lattice_commitment_test_elem(&r, b"i1"),
        )
        .ok()
        .unwrap();
        if let Some(inv) = a.inverse() {
            let prod = a.mul(&inv).ok().unwrap();
            assert!(prod.c0 == r.one() && prod.c1.is_zero());
        }
        // Zero has no inverse.
        assert!(Rq2::zero(&r).inverse().is_none());
    }

    #[test]
    fn karatsuba_matches_schoolbook() {
        let r = ring();
        let a = Rq2::new(elem(&r, b"k0"), elem(&r, b"k1")).ok().unwrap();
        let b = Rq2::new(elem(&r, b"l0"), elem(&r, b"l1")).ok().unwrap();
        // Schoolbook: c0 = a0b0 - a1b1, c1 = a0b1 + a1b0.
        let c0 = a.c0.mul(&b.c0).ok().unwrap().sub(&a.c1.mul(&b.c1).ok().unwrap()).ok().unwrap();
        let c1 = a.c0.mul(&b.c1).ok().unwrap().add(&a.c1.mul(&b.c0).ok().unwrap()).ok().unwrap();
        let expected = Rq2::new(c0, c1).ok().unwrap();
        assert_eq!(a.mul(&b).ok().unwrap(), expected);
    }

    #[test]
    fn short_challenge_sampling_and_norms() {
        let r = ring();
        // Spec must cover exactly 2·n = 128 slots (weight 23 spread over
        // the interleaved halves).
        let spec = ShortChallengeSpec {
            n: 2 * r.n(),
            family: lattice_core::short_challenge::ShortChallengeFamily::FixedWeight {
                weight: 23,
                amplitude: 1,
            },
        };
        let (z, g0, g1) = sample_rq2_challenge(&r, &spec, b"chal-seed").ok().unwrap();
        // Deterministic.
        let (z2, g0b, g1b) = sample_rq2_challenge(&r, &spec, b"chal-seed").ok().unwrap();
        assert_eq!(z.c0, z2.c0);
        assert_eq!(z.c1, z2.c1);
        assert_eq!((g0, g1), (g0b, g1b));
        // Total nonzeros across halves = 23 (the fixed weight).
        let total_nonzeros = z.c0.coeffs().iter().filter(|c| **c != 0).count()
            + z.c1.coeffs().iter().filter(|c| **c != 0).count();
        assert_eq!(total_nonzeros, 23);
        // Values are ternary (balanced representatives of ±1).
        for c in z.c0.coeffs().iter().chain(z.c1.coeffs().iter()) {
            let q = r.modulus.q;
            let balanced = if *c <= q / 2 { *c } else { q - *c };
            assert!(balanced <= 1, "coeff {c}");
        }
        // Certified Γ per half: ⌈√(Σ per-half)⌉ ≥ the true per-half value.
        assert!(g0 >= 1 && g1 >= 1);
        // Wrong spec size rejected.
        let bad_spec = ShortChallengeSpec {
            n: r.n(),
            family: lattice_core::short_challenge::ShortChallengeFamily::FixedWeight {
                weight: 8,
                amplitude: 1,
            },
        };
        assert!(matches!(
            sample_rq2_challenge(&r, &bad_spec, b"x"),
            Err(ShortChallengeError::InvalidParameters)
        ));
    }

    #[test]
    fn symphony_family_sampling() {
        let r = ring();
        let spec = ShortChallengeSpec {
            n: 2 * r.n(),
            family: lattice_core::short_challenge::ShortChallengeFamily::SmallSet {
                values: vec![0, 1, -1, 2, -2],
            },
        };
        let (z, _, _) = sample_rq2_challenge(&r, &spec, b"sym-seed").ok().unwrap();
        let q = r.modulus.q;
        for c in z.c0.coeffs().iter().chain(z.c1.coeffs().iter()) {
            let balanced = if *c <= q / 2 { *c as i64 } else { *c as i64 - q as i64 };
            assert!(balanced.abs() <= 2, "coeff {balanced}");
        }
    }

    // Small-element helper (norm ≤ 2) for the inverse test.
    fn lattice_commitment_test_elem(ring: &RingConfig, tag: &[u8]) -> RingElement {
        use lattice_core::transcript::Transcript;
        let bytes = Transcript::xof(b"rq2-test", tag, ring.n() * 2);
        let coeffs: Vec<u32> = bytes
            .chunks(2)
            .take(ring.n())
            .map(|c| {
                let v = u16::from_le_bytes([c[0], c[1]]) % 5;
                ring.modulus.reduce_i64(v as i64 - 2)
            })
            .collect();
        RingElement::from_coeffs(ring, coeffs)
    }
}
