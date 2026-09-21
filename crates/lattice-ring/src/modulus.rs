//! NTT-friendly 31/32-bit prime moduli with metadata for negacyclic
//! transforms and Module-SIS parameter bookkeeping.

/// A u32 modulus (prime) with precomputed root metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Modulus32 {
    /// The prime q itself (odd, fits u32).
    pub q: u32,
    /// The 2-adicity of q - 1: q - 1 = 2^adicity * odd.
    pub two_adicity: u32,
    /// A generator g of the multiplicative group; g^((q-1)/2^adicity) is a
    /// primitive 2^adicity-th root of unity.
    pub generator: u32,
}

impl Modulus32 {
    /// q = 3221225473 = 3 * 2^30 + 1 — 31.58 bits, supports NTT up to 2^30.
    /// Standard 32-bit-class NTT prime used by lattice libraries.
    pub const Q_32: Modulus32 = Modulus32 {
        q: 3221225473,
        two_adicity: 30,
        generator: 5,
    };

    /// q = 12289 = 3 * 2^12 + 1 — the classic 13.6-bit NTT prime (Kyber
    /// family), useful for small-norm tests.
    pub const Q_12289: Modulus32 = Modulus32 {
        q: 12289,
        two_adicity: 12,
        generator: 11,
    };

    /// q = 2013265921 = 15 * 2^27 + 1 — 30.9 bits.
    pub const Q_2013265921: Modulus32 = Modulus32 {
        q: 2013265921,
        two_adicity: 27,
        generator: 31,
    };

    pub const fn q_u64(&self) -> u64 {
        self.q as u64
    }

    /// Reduce a u64 into [0, q).
    #[inline]
    pub fn reduce_u64(&self, x: u64) -> u32 {
        (x % self.q as u64) as u32
    }

    /// Reduce a signed integer into [0, q) (balanced-coefficient ingestion).
    #[inline]
    pub fn reduce_i64(&self, x: i64) -> u32 {
        let r = x.rem_euclid(self.q as i64);
        r as u32
    }

    /// Modular multiplication in u64 intermediate.
    #[inline]
    pub fn mul(&self, a: u32, b: u32) -> u32 {
        self.reduce_u64(a as u64 * b as u64)
    }

    /// Modular addition.
    #[inline]
    pub fn add(&self, a: u32, b: u32) -> u32 {
        let s = a as u64 + b as u64;
        if s >= self.q as u64 {
            (s - self.q as u64) as u32
        } else {
            s as u32
        }
    }

    /// Modular subtraction.
    #[inline]
    pub fn sub(&self, a: u32, b: u32) -> u32 {
        if a >= b {
            a - b
        } else {
            // Parentheses are load-bearing: a + q would overflow u32 before
            // subtracting b; (q - b) + a is the final value, always < q.
            a + (self.q - b)
        }
    }

    /// Modular negation.
    #[inline]
    pub fn neg(&self, a: u32) -> u32 {
        if a == 0 {
            0
        } else {
            self.q - a
        }
    }

    /// Modular exponentiation.
    pub fn pow(&self, base: u32, exp: u64) -> u32 {
        let mut acc: u32 = 1;
        let mut b = self.reduce_u64(base as u64);
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul(acc, b);
            }
            b = self.mul(b, b);
            e >>= 1;
        }
        acc
    }

    /// Modular inverse via Fermat (q prime).
    pub fn inv(&self, a: u32) -> Option<u32> {
        if a == 0 {
            None
        } else {
            Some(self.pow(a, self.q as u64 - 2))
        }
    }

    /// Primitive 2^k-th root of unity (k <= two_adicity).
    pub fn root_of_unity(&self, k: u32) -> Option<u32> {
        if k > self.two_adicity {
            return None;
        }
        // g^((q-1)/2^k)
        let pow = (self.q as u64 - 1) >> k;
        Some(self.pow(self.generator, pow))
    }

    /// Primitive 2n-th root (psi) for negacyclic NTT of length n = 2^k.
    pub fn root_of_unity_order_2n(&self, k: u32) -> Option<u32> {
        self.root_of_unity(k + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_arithmetic_matches_u64_reference() {
        let m = Modulus32::Q_32;
        let (a, b) = (1234567891u32, 3221225470u32);
        assert_eq!(
            m.mul(a, b),
            ((a as u64 * b as u64) % m.q as u64) as u32
        );
        // add/sub wrap correctly at the modulus
        let x = m.q - 1;
        assert_eq!(m.add(x, 2), 1);
        assert_eq!(m.sub(1, 2), m.q - 1);
        assert_eq!(m.neg(x), 1);
    }

    #[test]
    fn inverse_and_negative_values() {
        let m = Modulus32::Q_32;
        for a in [1u32, 2, 12345, m.q - 1] {
            let inv = m.inv(a).unwrap();
            assert_eq!(m.mul(a, inv), 1);
        }
        assert!(m.inv(0).is_none());
        // signed reduction maps negatives to balanced euclidean residues
        assert_eq!(m.reduce_i64(-1), m.q - 1);
        assert_eq!(m.reduce_i64(-(m.q as i64)), 0);
        assert_eq!(m.reduce_i64(-(m.q as i64 - 5)), 5);
    }

    #[test]
    fn root_orders() {
        for m in [Modulus32::Q_32, Modulus32::Q_12289, Modulus32::Q_2013265921] {
            for k in 1..=m.two_adicity.min(12) {
                let w = m.root_of_unity(k).unwrap();
                // w^(2^k) == 1, w^(2^(k-1)) != 1
                assert_eq!(m.pow(w, 1u64 << k), 1);
                assert_ne!(m.pow(w, 1u64 << (k - 1)), 1);
            }
            assert!(m.root_of_unity(m.two_adicity + 1).is_none());
        }
    }
}
