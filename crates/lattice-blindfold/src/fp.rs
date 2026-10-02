//! The base prime field F_q with q = 2^64 − 59 (the paper's Solinas prime,
//! §4.3.4: "Throughout we take the Solinas prime q = 2^64 − 59").
//!
//! q ≡ 5 (mod 8) (2^64 ≡ 0 mod 8, −59 ≡ 5 mod 8), which makes **2 a
//! quadratic non-residue** by the supplement to quadratic reciprocity:
//! (2/p) = (−1)^((p²−1)/8) = −1 for p ≡ ±5 (mod 8). The extension field
//! K = F[Y]/(Y² − ν) therefore instantiates with ν = 2 (§3.3: "writing
//! K = F(√ν) for a fixed non-residue ν ∈ F").
//!
//! Multiplication uses the Solinas shape: 2^64 ≡ 59 (mod q), so for
//! x = hi·2^64 + lo one has x ≡ hi·59 + lo (mod q); two cascaded rounds
//! land below 2^64 + 354, finished by conditional subtractions.

/// The modulus q = 2^64 − 59.
pub const Q: u64 = u64::MAX - 58;
/// The quadratic non-residue ν = 2 (valid because q ≡ 5 mod 8).
pub const NU: u64 = 2;

/// A field element in F_q, stored canonically in [0, q).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Fq(pub u64);

impl Fq {
    pub const ZERO: Fq = Fq(0);
    pub const ONE: Fq = Fq(1);
    pub const TWO: Fq = Fq(2);

    #[inline]
    pub const fn new(x: u64) -> Self {
        Fq(x % Q)
    }

    /// From a signed integer, reduced into the canonical window.
    #[inline]
    pub fn from_i64(x: i64) -> Self {
        if x >= 0 {
            Fq(x as u64 % Q)
        } else {
            let m = ((-x) as u64) % Q;
            Fq(if m == 0 { 0 } else { Q - m })
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn add(&self, o: &Fq) -> Fq {
        // a + b < 2Q < 2^65 — computed in u128 to avoid u64 overflow.
        let s = self.0 as u128 + o.0 as u128;
        Fq(if s >= Q as u128 {
            (s - Q as u128) as u64
        } else {
            s as u64
        })
    }

    #[inline]
    pub fn sub(&self, o: &Fq) -> Fq {
        Fq(if self.0 >= o.0 {
            self.0 - o.0
        } else {
            Q - (o.0 - self.0)
        })
    }

    #[inline]
    pub fn neg(&self) -> Fq {
        Fq(if self.0 == 0 { 0 } else { Q - self.0 })
    }

    #[inline]
    pub fn double(&self) -> Fq {
        self.add(self)
    }

    /// Solinas reduction of a 128-bit product.
    #[inline]
    pub fn mul(&self, o: &Fq) -> Fq {
        let prod = (self.0 as u128) * (o.0 as u128);
        Fq(reduce_u128(prod))
    }

    #[inline]
    pub fn square(&self) -> Fq {
        self.mul(self)
    }

    pub fn pow_u64(&self, e: u64) -> Fq {
        let mut acc = Fq::ONE;
        let mut base = *self;
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse; `None` for zero.
    pub fn inverse(&self) -> Option<Fq> {
        if self.0 == 0 {
            return None;
        }
        // Fermat: a^(q-2).
        Some(self.pow_u64(Q - 2))
    }

    /// Legendre symbol as i8: 1 = residue, -1 = non-residue, 0 = zero.
    pub fn legendre(&self) -> i8 {
        let l = self.pow_u64((Q - 1) / 2);
        if l.0 == 0 {
            0
        } else if l.0 == 1 {
            1
        } else {
            -1
        }
    }

    /// Uniform element from 32 bytes of seed material (splitmix-style).
    pub fn uniform(seed: &[u8], counter: &mut u64) -> Fq {
        loop {
            let mut buf = [0u8; 8];
            let c = counter.wrapping_add(1);
            *counter = c;
            buf.copy_from_slice(&c.to_le_bytes());
            let mut h = [0u8; 32];
            // A cheap deterministic mixer (xxhash-style avalanche on 8 lanes).
            let mut z = u64::from_le_bytes(buf);
            for (i, slot) in h.iter_mut().enumerate().take(8) {
                z = z
                    .wrapping_add(0x9E3779B97F4A7C15)
                    .wrapping_mul(0xBF58476D1CE4E5B9);
                let mut w = z ^ (z >> 30);
                w = w.wrapping_mul(0xBF58476D1CE4E5B9);
                w ^= w >> 27;
                w = w.wrapping_mul(0x94D049BB133111EB);
                w ^= w >> 31;
                // Fold seed bytes in for domain separation.
                let sb = seed[(i * 4 + (c as usize % 4)) % seed.len().max(1)];
                w ^= (sb as u64).wrapping_mul(0x2545F4914F6CDD1D);
                *slot = w as u8;
            }
            // Interpret the first 8 bytes; reject ≥ q (negligible bias 2^-6).
            let mut v = [0u8; 8];
            v.copy_from_slice(&h[..8]);
            let cand = u64::from_le_bytes(v);
            if cand < Q {
                return Fq(cand);
            }
        }
    }

    /// The symmetric representative in [-(q-1)/2, (q-1)/2] (§2.1).
    #[inline]
    pub fn sym(&self) -> i64 {
        if self.0 <= (Q - 1) / 2 {
            self.0 as i64
        } else {
            self.0 as i64 - Q as i64
        }
    }
}

/// Solinas reduction for u128 values (used by mul).
#[inline]
pub fn reduce_u128(x: u128) -> u64 {
    // x < 2^128. x = hi*2^64 + lo, 2^64 ≡ 59 (mod q).
    let lo = (x & 0xFFFF_FFFF_FFFF_FFFF) as u64;
    let hi = (x >> 64) as u64;
    // t = hi*59 + lo < 60·2^64 — fits u128 with no wrap.
    let t = (hi as u128) * 59 + (lo as u128);
    // Second round: t = hi2*2^64 + lo2 with hi2 ≤ 59.
    let lo2 = (t & 0xFFFF_FFFF_FFFF_FFFF) as u64;
    let hi2 = (t >> 64) as u64;
    // r = hi2*59 + lo2 < 2^64 + 3481 — computed in u128 (a u64 wrapping
    // add here would silently corrupt the result).
    let r = (hi2 as u128) * 59 + (lo2 as u128);
    // r < 2^64 + 3481 < 2Q, so a single conditional subtraction lands in
    // [0, 3540) ⊂ [0, Q).
    if r >= Q as u128 {
        (r - Q as u128) as u64
    } else {
        r as u64
    }
}

/// Verify at runtime that 2 is a non-residue mod q (the ν of §3.3).
pub fn check_nonresidue() -> Result<(), &'static str> {
    if Fq::TWO.legendre() != -1 {
        return Err("2 must be a quadratic non-residue mod q");
    }
    // q ≡ 5 (mod 8):
    if Q % 8 != 5 {
        return Err("q must be 5 mod 8");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solinas_reduction_identities() {
        assert_eq!(reduce_u128(Q as u128), 0);
        assert_eq!(reduce_u128((Q - 1) as u128 * (Q - 1) as u128), 1);
        // 2^64 ≡ 59
        assert_eq!(reduce_u128(1u128 << 64), 59);
    }

    #[test]
    fn field_axioms() {
        let a = Fq::new(123456789012345678);
        let b = Fq::new(98765432109876543);
        let c = Fq::new(5555555555555555);
        // commutativity / associativity / distributivity
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        // inverse
        let ai = a.inverse().unwrap();
        assert_eq!(a.mul(&ai), Fq::ONE);
        assert!(Fq::ZERO.inverse().is_none());
        // neg
        assert_eq!(a.add(&a.neg()), Fq::ZERO);
    }

    #[test]
    fn two_is_nonresidue() {
        check_nonresidue().unwrap();
        assert_eq!(Fq::TWO.legendre(), -1);
    }

    #[test]
    fn symmetric_representative() {
        let x = Fq::new(3);
        assert_eq!(x.sym(), 3);
        let y = Fq(Q - 3);
        assert_eq!(y.sym(), -3);
        assert_eq!(y.add(&x), Fq::ZERO);
    }

    #[test]
    fn legendre_quadratic_residue_roundtrip() {
        let a = Fq::new(7);
        let sq = a.square();
        assert_eq!(sq.legendre(), 1);
    }
}
