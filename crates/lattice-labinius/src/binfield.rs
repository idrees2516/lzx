//! The binary fields `B128 = GF(2)[x]/(x^128 + x^7 + x^2 + x + 1)` (GHASH modulus) and
//! `F162 = GF(2)[x]/(x^162 + x^81 + 1)` (= `GF(2)[Z]/Phi_243(Z)`), plus the F162 input front
//! end (the lift of a stream of field elements into binary ring elements of `R_648`).
//!
//! Port of `labinius` `fields/scalar.rs` + `f162.rs`. Upstream multiplies with `PCLMULQDQ`;
//! this port is dependency-free and uses a software carry-less product (byte-wise schoolbook),
//! which is exact and portable.

use crate::params::N;

/// 64x64 carry-less multiplication (software, byte-table free).
#[inline]
pub fn clmul64(a: u64, b: u64) -> u128 {
    let mut acc: u128 = 0;
    let mut b = b;
    let mut i = 0u32;
    while b != 0 {
        if b & 1 != 0 {
            acc ^= (a as u128) << i;
        }
        b >>= 1;
        i += 1;
    }
    acc
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct B128(pub u128);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub struct F162(pub [u64; 3]);

const GHASH_MOD: u128 = (1 << 7) | (1 << 2) | (1 << 1) | 1;

impl B128 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    pub fn add(self, o: Self) -> Self {
        Self(self.0 ^ o.0)
    }    /// Product modulo `x^128 + x^7 + x^2 + x + 1`.
    pub fn mul(self, o: Self) -> Self {
        let (a0, a1) = (self.0 as u64, (self.0 >> 64) as u64);
        let (b0, b1) = (o.0 as u64, (o.0 >> 64) as u64);
        let lo = clmul64(a0, b0);
        let hi = clmul64(a1, b1);
        let mid = clmul64(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
        let p_lo = lo ^ (mid << 64);
        let mut acc = p_lo;
        let mut h = hi ^ (mid >> 64);
        // fold the 129-bit overflow twice with the modulus
        for _ in 0..2 {
            let c = clmul64(h as u64, GHASH_MOD as u64) ^ (clmul64((h >> 64) as u64, GHASH_MOD as u64) << 64);
            let carry = clmul64((h >> 64) as u64, GHASH_MOD as u64) >> 64;
            acc ^= c;
            h = carry;
        }
        Self(acc)
    }
    /// Uniform from 16 little-endian bytes.
    pub fn from_le_bytes(b: &[u8; 16]) -> Self {
        Self(u128::from_le_bytes(*b))
    }
}

const M34: u64 = (1 << 34) - 1;

impl F162 {
    pub const ZERO: Self = Self([0; 3]);
    pub const ONE: Self = Self([1, 0, 0]);
    /// Number of significant bits (limbs 0,1 full; limb 2 holds bits 128..161).
    pub const BITS: usize = 162;

    pub fn from_b128(x: B128) -> Self {
        Self([x.0 as u64, (x.0 >> 64) as u64, 0])
    }

    pub fn add(self, o: Self) -> Self {
        Self([
            self.0[0] ^ o.0[0],
            self.0[1] ^ o.0[1],
            self.0[2] ^ o.0[2],
        ])
    }

    pub fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }

    /// Product modulo `x^162 + x^81 + 1`. The 324-bit carry-less product is reduced by folding
    /// `h = p >> 162` back with `h ^ (h << 81)`, exactly as upstream's limb arithmetic does.
    pub fn mul(self, o: Self) -> Self {
        let a = self.0;
        let b = o.0;
        let mut p = [0u64; 6];
        for i in 0..3 {
            for j in 0..3 {
                let t = clmul64(a[i], b[j]);
                p[i + j] ^= t as u64;
                p[i + j + 1] ^= (t >> 64) as u64;
            }
        }
        let h00 = (p[2] >> 34) | (p[3] << 30);
        let h01 = (p[3] >> 34) & ((1 << 17) - 1);
        let h10 = (p[3] >> 51) | (p[4] << 13);
        let h11 = (p[4] >> 51) | (p[5] << 13);
        Self([
            p[0] ^ h00 ^ h10,
            p[1] ^ h01 ^ h11 ^ (h00 << 17),
            (p[2] & M34) ^ (h01 << 17) ^ (h00 >> 47),
        ])
    }

    /// Uniform 162-bit element from 24 little-endian bytes (top 30 bits of limb 2 zeroed).
    pub fn from_le24(bytes: &[u8]) -> Self {
        let limb = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap());
        F162([limb(0), limb(1), limb(2) & ((1u64 << 34) - 1)])
    }

    pub fn to_le24(self) -> [u8; 24] {
        let mut out = [0u8; 24];
        for (k, l) in self.0.iter().enumerate() {
            out[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }
        out
    }

    /// Bit `m` (m < 162).
    pub fn bit(&self, m: usize) -> u32 {
        ((self.0[m >> 6] >> (m & 63)) & 1) as u32
    }
}

// =============================================================================================
// the lift: F162 stream -> binary ring elements of R_648
// =============================================================================================

/// The lift semantics: coefficient `4m + k` of ring element r is bit `m` of F162 element
/// `4r + k` (plain interleaving; all coefficients 0/1).
pub fn lift4(q: &[F162; 4]) -> [u32; N] {
    let mut c = [0u32; N];
    for m in 0..F162::BITS {
        for k in 0..4 {
            c[4 * m + k] = q[k].bit(m);
        }
    }
    c
}

/// Ring element `r` of a stream (elements `4r..4r+3`).
pub fn lift_elem(elems: &[F162], r: usize) -> [u32; N] {
    let q: &[F162; 4] = elems[4 * r..4 * r + 4].try_into().unwrap();
    lift4(q)
}

/// The inverse of [`lift4`] for testing.
pub fn pack4(c: &[u32; N]) -> [F162; 4] {
    let mut q = [F162([0; 3]); 4];
    for m in 0..F162::BITS {
        for k in 0..4 {
            q[k].0[m >> 6] |= (c[4 * m + k] as u64) << (m & 63);
        }
    }
    q
}

/// xorshift64* — deterministic randomness (upstream `rng.rs`, kept bit-for-bit).
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next_u64() >> 32) * n as u64 >> 32) as u32
    }
}

/// A uniform 162-bit element.
pub fn random_f162(rng: &mut Rng) -> F162 {
    F162([
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64() & ((1u64 << 34) - 1),
    ])
}

pub fn random_elems(n: usize, seed: u64) -> Vec<F162> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| random_f162(&mut rng)).collect()
}

// Operator forms used by the eval layer.
impl std::ops::Add for B128 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self(self.0 ^ o.0)
    }
}
impl std::ops::Mul for B128 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        B128::mul(self, o)
    }
}
impl std::ops::Add for F162 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        F162::add(self, o)
    }
}
impl std::ops::Mul for F162 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        F162::mul(self, o)
    }
}
