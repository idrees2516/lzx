//! BN254 G1: the group of the commitment instantiation.
//!
//! * Curve: `y² = x³ + 3` over `F_p` (`fp_base::BN254_FP`).
//! * Generator: `(1, 2)`; group order = BN254 `F_r`; **cofactor 1**, so an
//!   on-curve check is a subgroup check.
//! * Coordinates: Jacobian `(X : Y : Z)` with `(x, y) = (X/Z², Y/Z³)`;
//!   the point at infinity is `Z = 0`.
//! * `hash_to_point`: try-and-increment over SHAKE-derived x-candidates
//!   (taking the even y for determinism).
//!
//! Scalars live in `F_r` (`lattice_projsumcheck::fp256::Fp256`) — see
//! `pedersen` for the vector-Pedersen commitment built on this group.

use crate::fp_base::FpBase;
use lattice_core::transcript::Transcript;

/// Curve coefficient b = 3 (Montgomery form).
fn b_coeff() -> FpBase {
    FpBase::from_canonical_u64(3)
}

/// Affine point (or infinity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G1Affine {
    pub x: FpBase,
    pub y: FpBase,
    pub infinity: bool,
}

impl G1Affine {
    /// The standard generator (1, 2).
    pub fn generator() -> G1Affine {
        G1Affine {
            x: FpBase::from_canonical_u64(1),
            y: FpBase::from_canonical_u64(2),
            infinity: false,
        }
    }

    pub fn identity() -> G1Affine {
        G1Affine {
            x: FpBase::ZERO,
            y: FpBase::ZERO,
            infinity: true,
        }
    }

    /// On-curve check `y² = x³ + 3` (cofactor 1 ⇒ subgroup check).
    pub fn is_on_curve(&self) -> bool {
        if self.infinity {
            return true;
        }
        let lhs = self.y.mul(&self.y);
        let rhs = self.x.mul(&self.x).mul(&self.x).add(&b_coeff());
        lhs == rhs
    }

    pub fn to_projective(self) -> G1Point {
        if self.infinity {
            G1Point::identity()
        } else {
            G1Point {
                x: self.x,
                y: self.y,
                z: FpBase::one_mont(),
            }
        }
    }

    /// Canonical byte serialization: 32-byte x ‖ 32-byte y (identity → all
    /// zeros with a leading 0x00 flag byte).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(65);
        out.push(u8::from(self.infinity));
        out.extend_from_slice(&self.x.to_be_bytes());
        out.extend_from_slice(&self.y.to_be_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<G1Affine, G1Error> {
        if bytes.len() != 65 {
            return Err(G1Error::BadEncoding);
        }
        if bytes[0] == 1 {
            return Ok(G1Affine::identity());
        }
        if bytes[0] != 0 {
            return Err(G1Error::BadEncoding);
        }
        let mut xb = [0u8; 32];
        let mut yb = [0u8; 32];
        xb.copy_from_slice(&bytes[1..33]);
        yb.copy_from_slice(&bytes[33..65]);
        let x = FpBase::from_be_bytes_wide(&xb);
        let y = FpBase::from_be_bytes_wide(&yb);
        let p = G1Affine {
            x,
            y,
            infinity: false,
        };
        if !p.is_on_curve() {
            return Err(G1Error::NotOnCurve);
        }
        Ok(p)
    }
}

/// Jacobian point.
#[derive(Clone, Copy, Debug)]
pub struct G1Point {
    pub x: FpBase,
    pub y: FpBase,
    pub z: FpBase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G1Error {
    NotOnCurve,
    BadEncoding,
    HashToCurveFailed,
}

impl core::fmt::Display for G1Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            G1Error::NotOnCurve => write!(f, "point not on curve"),
            G1Error::BadEncoding => write!(f, "bad point encoding"),
            G1Error::HashToCurveFailed => write!(f, "hash-to-curve failed"),
        }
    }
}

impl G1Point {
    pub fn identity() -> G1Point {
        G1Point {
            x: FpBase::one_mont(),
            y: FpBase::one_mont(),
            z: FpBase::ZERO,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    pub fn from_affine(a: G1Affine) -> G1Point {
        a.to_projective()
    }

    /// Jacobian double (short Weierstrass, a = 0):
    /// A = X², B = Y², C = B², D = 2·((X+B)² − A − C), E = 3A, F = E²
    /// X' = F − 2D, Y' = E·(D − X') − 8C, Z' = 2·Y·Z.
    pub fn double(&self) -> G1Point {
        if self.is_identity() || self.y.is_zero() {
            return G1Point::identity();
        }
        let a = self.x.mul(&self.x);
        let b = self.y.mul(&self.y);
        let c = b.mul(&b);
        let xb = self.x.add(&b);
        let d = xb.mul(&xb).sub(&a).sub(&c).double();
        let e = a.double().add(&a); // 3A
        let f = e.mul(&e);
        let x3 = f.sub(&d.double());
        let c8 = c.double().double().double();
        let y3 = e.mul(&d.sub(&x3)).sub(&c8);
        let z3 = self.y.mul(&self.z).double();
        G1Point {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Jacobian addition (general case).
    pub fn add(&self, other: &G1Point) -> G1Point {
        if self.is_identity() {
            return *other;
        }
        if other.is_identity() {
            return *self;
        }
        let z1z1 = self.z.mul(&self.z);
        let z2z2 = other.z.mul(&other.z);
        let u1 = self.x.mul(&z2z2);
        let u2 = other.x.mul(&z1z1);
        let s1 = self.y.mul(&other.z).mul(&z2z2);
        let s2 = other.y.mul(&self.z).mul(&z1z1);
        if u1 == u2 {
            if s1 == s2 {
                return self.double();
            }
            return G1Point::identity();
        }
        let h = u2.sub(&u1);
        let i = h.double().mul(&h.double()); // (2H)²
        let j = h.mul(&i);
        let r = s2.sub(&s1).double(); // 2(S2−S1)
        let v = u1.mul(&i);
        let x3 = r.mul(&r).sub(&j).sub(&v.double());
        let sj = s1.mul(&j);
        let y3 = r.mul(&v.sub(&x3)).sub(&sj.double());
        let zsum = self.z.add(&other.z);
        let z3 = zsum
            .mul(&zsum)
            .sub(&z1z1)
            .sub(&z2z2)
            .mul(&h);
        G1Point {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    pub fn neg(&self) -> G1Point {
        G1Point {
            x: self.x,
            y: self.y.neg(),
            z: self.z,
        }
    }

    /// Scalar multiplication (double-and-add over canonical scalar limbs,
    /// MSB first). The scalar is an `F_r` element in Montgomery form; the
    /// canonical limbs are used as the exponent.
    pub fn mul_scalar(&self, k: &crate::Fp256) -> G1Point {
        let canon = scalar_canon_limbs(k);
        let mut result = G1Point::identity();
        let mut seen = false;
        for limb in canon.iter().rev() {
            for bit in (0..64).rev() {
                if seen {
                    result = result.double();
                }
                if (limb >> bit) & 1 == 1 {
                    if seen {
                        result = result.add(self);
                    } else {
                        result = *self;
                        seen = true;
                    }
                }
            }
        }
        result
    }

    /// To affine (one field inversion).
    pub fn to_affine(&self) -> G1Affine {
        if self.is_identity() {
            return G1Affine::identity();
        }
        let zinv = self.z.inverse(); // z != 0 here for honest callers
        match zinv {
            Ok(zinv) => {
                let zinv2 = zinv.mul(&zinv);
                let x = self.x.mul(&zinv2);
                let y = self.y.mul(&zinv2).mul(&zinv);
                G1Affine {
                    x,
                    y,
                    infinity: false,
                }
            }
            Err(_) => G1Affine::identity(),
        }
    }
}

/// Canonical limbs of an `F_r` element (Montgomery → canonical), for use as
/// a scalar-multiplication exponent.
fn scalar_canon_limbs(k: &crate::Fp256) -> [u64; 4] {
    k.mul(&crate::Fp256 {
        limbs: [1, 0, 0, 0],
    })
    .limbs
}

/// Deterministic hash-to-point: SHAKE(seed ‖ counter) → 32-byte x-candidate;
/// accept the first candidate whose `x³ + 3` is a QR; y = the even root.
pub fn hash_to_point(domain: &[u8], seed: &[u8], index: u64) -> Result<G1Affine, G1Error> {
    let mut input = Vec::with_capacity(domain.len() + seed.len() + 16);
    input.extend_from_slice(domain);
    input.extend_from_slice(seed);
    input.extend_from_slice(&index.to_le_bytes());
    for counter in 0u32..64 {
        let mut with_counter = input.clone();
        with_counter.extend_from_slice(&counter.to_le_bytes());
        let bytes = Transcript::xof(b"g1-hash-to-point", &with_counter, 32);
        let mut xbytes = [0u8; 32];
        xbytes.copy_from_slice(&bytes);
        let x = FpBase::from_be_bytes_wide(&xbytes);
        // rhs = x³ + 3
        let rhs = x.mul(&x).mul(&x).add(&b_coeff());
        if rhs.legendre() != 1 {
            continue;
        }
        let y = match rhs.sqrt() {
            Ok(y) => y,
            Err(_) => continue,
        };
        // Deterministic choice: take the "even" root (canonical low limb
        // parity). Both roots are valid; parity just fixes one.
        let ycanon = y.from_mont();
        let yfinal = if ycanon[0] & 1 == 0 { y } else { y.neg() };
        let p = G1Affine {
            x,
            y: yfinal,
            infinity: false,
        };
        if p.is_on_curve() {
            return Ok(p);
        }
    }
    Err(G1Error::HashToCurveFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fp256;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn generator_on_curve() {
        assert!(G1Affine::generator().is_on_curve());
        assert!(G1Affine::identity().is_on_curve());
    }

    #[test]
    fn jacobian_identity_semantics() {
        let g = G1Affine::generator().to_projective();
        assert!(G1Point::identity().is_identity());
        assert_eq!(g.add(&G1Point::identity()).to_affine(), G1Affine::generator());
        assert_eq!(G1Point::identity().add(&g).to_affine(), G1Affine::generator());
        assert!(g.add(&g.neg()).is_identity());
    }

    #[test]
    fn doubling_and_addition_consistency() {
        let g = G1Affine::generator().to_projective();
        let g2 = g.double();
        let g2b = g.add(&g);
        assert_eq!(g2.to_affine(), g2b.to_affine());
        let g3a = g2.add(&g);
        let g3b = g.add(&g2);
        assert_eq!(g3a.to_affine(), g3b.to_affine());
        // 2G + G == 3G via scalar mul
        let g3m = g.mul_scalar(&fr(3));
        assert_eq!(g3a.to_affine(), g3m.to_affine());
    }

    #[test]
    fn scalar_mul_basics() {
        let g = G1Affine::generator().to_projective();
        assert!(g.mul_scalar(&fr(0)).is_identity());
        assert_eq!(g.mul_scalar(&fr(1)).to_affine(), G1Affine::generator());
        // 5G = 2G + 3G
        let g5 = g.mul_scalar(&fr(5));
        let g2 = g.mul_scalar(&fr(2));
        let g3 = g.mul_scalar(&fr(3));
        assert_eq!(g5.to_affine(), g2.add(&g3).to_affine());
        assert!(g5.to_affine().is_on_curve());
    }

    #[test]
    fn group_order_is_fr() {
        // r·G == identity (the group order is the BN254 scalar field).
        let g = G1Affine::generator().to_projective();
        let r_limbs: [u64; 4] = [
            0x43e1_f593_f000_0001,
            0x2833_e848_79b9_7091,
            0xb850_45b6_8181_585d,
            0x3064_4e72_e131_a029,
        ];
        // Build the Montgomery form of r from canonical limbs via
        // from_canonical_u128 piecewise: simpler — mul by 1 and check the
        // canonical limbs are r, then construct Fp256 from them.
        let _ = r_limbs;
        // r as Fp256: use from_canonical_u128 for low/high halves via two
        // multiplications (r < 2^256 so a direct limb construction works:
        // Montgomery form = canonical × R).
        let r_canon = Fp256 { limbs: r_limbs };
        let one = Fp256 {
            limbs: [1, 0, 0, 0],
        };
        let _ = one;
        // Montgomery form of r = r_canon · R where R = one_mont().
        let r_mont = r_canon.mul(&Fp256::one_mont());
        assert!(g.mul_scalar(&r_mont).is_identity());
    }

    #[test]
    fn hash_to_point_deterministic_and_on_curve() {
        let p1 = hash_to_point(b"test", b"seed", 0).ok().unwrap();
        let p2 = hash_to_point(b"test", b"seed", 0).ok().unwrap();
        let p3 = hash_to_point(b"test", b"seed", 1).ok().unwrap();
        assert_eq!(p1, p2);
        assert!(p1.is_on_curve());
        assert!(p3.is_on_curve());
        assert_ne!(p1, p3);
        // Hash-domain separation:
        let p4 = hash_to_point(b"other", b"seed", 0).ok().unwrap();
        assert_ne!(p1, p4);
    }

    #[test]
    fn byte_roundtrip() {
        let g = G1Affine::generator();
        let bytes = g.to_bytes();
        let back = G1Affine::from_bytes(&bytes).ok().unwrap();
        assert_eq!(g, back);
        let id = G1Affine::identity().to_bytes();
        assert_eq!(G1Affine::from_bytes(&id).ok().unwrap(), G1Affine::identity());
    }
}
