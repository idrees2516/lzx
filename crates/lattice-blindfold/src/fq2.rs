//! The quadratic extension K = F[Y]/(Y² − ν) with ν = 2 (§3.3:
//! "K = F[Y]/(Y² − ν) for a quadratic non-residue ν").
//!
//! Every element of R_K decomposes uniquely as a + bY with a, b ∈ R_F
//! (§3.3.1, the rank-doubling embedding). This module carries the field
//! K itself; the ring R_K lives in `rk`.
//!
//! The norm form N_{K/F}(a + bY) = a² − νb² (its non-vanishing is what
//! makes K-valued challenge masking sound, §3.3.5: det M = f̃a² − ν f̃b²).

use crate::fp::{check_nonresidue, Fq, NU};

/// An element of K = F_q2 in the {1, Y} basis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct K(pub Fq, pub Fq);

impl K {
    pub const ZERO: K = K(Fq::ZERO, Fq::ZERO);
    pub const ONE: K = K(Fq::ONE, Fq::ZERO);

    #[inline]
    pub fn from_fp(a: Fq) -> K {
        K(a, Fq::ZERO)
    }

    #[inline]
    pub fn from_fp2(a: Fq, b: Fq) -> K {
        K(a, b)
    }

    #[inline]
    pub fn from_i64(a: i64) -> K {
        K(Fq::from_i64(a), Fq::ZERO)
    }

    #[inline]
    pub fn add(&self, o: &K) -> K {
        K(self.0.add(&o.0), self.1.add(&o.1))
    }

    #[inline]
    pub fn sub(&self, o: &K) -> K {
        K(self.0.sub(&o.0), self.1.sub(&o.1))
    }

    #[inline]
    pub fn neg(&self) -> K {
        K(self.0.neg(), self.1.neg())
    }

    /// (a1 + b1 Y)(a2 + b2 Y) = (a1a2 + ν b1b2) + (a1b2 + b1a2) Y.
    #[inline]
    pub fn mul(&self, o: &K) -> K {
        let a = self.0.mul(&o.0).add(&Fq::new(NU).mul(&self.1.mul(&o.1)));
        let b = self.0.mul(&o.1).add(&self.1.mul(&o.0));
        K(a, b)
    }

    #[inline]
    pub fn square(&self) -> K {
        self.mul(self)
    }

    /// Scalar multiplication by a base-field element.
    #[inline]
    pub fn scale_fp(&self, s: &Fq) -> K {
        K(self.0.mul(s), self.1.mul(s))
    }

    /// Inverse: (a² − νb²)⁻¹ · (a − bY).
    pub fn inverse(&self) -> Option<K> {
        let norm = self.0.square().sub(&Fq::new(NU).mul(&self.1.square()));
        let inv = norm.inverse()?;
        Some(K(inv.mul(&self.0), inv.mul(&self.1.neg())))
    }

    /// The norm N_{K/F}(x) = a² − νb² (nonzero for x ≠ 0 since ν is a
    /// non-residue — §3.3.5's determinant).
    pub fn norm(&self) -> Fq {
        self.0.square().sub(&Fq::new(NU).mul(&self.1.square()))
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }

    /// Conjugation a + bY ↦ a − bY.
    pub fn conj(&self) -> K {
        K(self.0, self.1.neg())
    }

    pub fn uniform(seed: &[u8], counter: &mut u64) -> K {
        K(Fq::uniform(seed, counter), Fq::uniform(seed, counter))
    }

    /// The multiplicative group is cyclic of order q²−1; sample a unit.
    pub fn uniform_unit(seed: &[u8], counter: &mut u64) -> K {
        loop {
            let c = K::uniform(seed, counter);
            if !c.is_zero() {
                if let Some(inv) = c.inverse() {
                    let _ = inv;
                    return c;
                }
            }
        }
    }

    /// eq(x, y) = Π_i (x_i y_i + (1−x_i)(1−y_i)) over K (§2.1) — the
    /// equality polynomial for multilinear extensions.
    pub fn eq(xs: &[K], ys: &[K]) -> K {
        let mut acc = K::ONE;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let one = K::ONE;
            let term = x.mul(y).add(&one.sub(x).mul(&one.sub(y)));
            acc = acc.mul(&term);
        }
        acc
    }
}

/// Runtime sanity check: ν = 2 is a non-residue and X² − ν is irreducible.
pub fn check_extension() -> Result<(), &'static str> {
    check_nonresidue()?;
    // X² − 2 has a root iff 2 is a square.
    if Fq::new(NU).legendre() != -1 {
        return Err("X^2 - 2 must be irreducible over F_q");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_axioms_k() {
        check_extension().unwrap();
        let a = K::from_fp2(Fq::new(1234567), Fq::new(7654321));
        let b = K::from_fp2(Fq::new(89), Fq::new(98));
        let c = K::from_fp2(Fq::new(31337), Fq::new(1));
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        let ai = a.inverse().unwrap();
        assert_eq!(a.mul(&ai), K::ONE);
        assert!(K::ZERO.inverse().is_none());
    }

    #[test]
    fn y_squared_is_nu() {
        let y = K(Fq::ZERO, Fq::ONE);
        assert_eq!(y.square(), K::from_fp(Fq::new(NU)));
    }

    #[test]
    fn norm_nonvanishing() {
        // For random non-zero x, N(x) ≠ 0 (ν non-residue).
        let mut ctr = 0u64;
        let seed = b"norm-test";
        for _ in 0..64 {
            let x = K::uniform(seed, &mut ctr);
            if !x.is_zero() {
                assert!(!x.norm().is_zero());
                assert_eq!(x.mul(&x.conj()), K::from_fp(x.norm()));
            }
        }
    }

    #[test]
    fn eq_polynomial_on_cube() {
        // eq(x, x) = 1 for BOOLEAN x; eq(x, y) = 0 for distinct boolean
        // points (the defining properties on {0,1}^ℓ).
        let x = vec![K::ZERO, K::ONE];
        assert_eq!(K::eq(&x, &x), K::ONE);
        let y = vec![K::ONE, K::ZERO];
        assert_eq!(K::eq(&x, &y), K::ZERO);
        // For non-boolean points eq(z, z) = Π(z_i² + (1−z_i)²) ≠ 1 in
        // general — the identity is a cube property only.
        let z = vec![K::from_fp2(Fq::new(5), Fq::new(7)), K::ONE];
        let expect = z
            .iter()
            .map(|zi| zi.mul(zi).add(&K::ONE.sub(zi).mul(&K::ONE.sub(zi))))
            .fold(K::ONE, |a, b| a.mul(&b));
        assert_eq!(K::eq(&z, &z), expect);
    }
}
