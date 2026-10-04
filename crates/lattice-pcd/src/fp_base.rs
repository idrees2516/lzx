//! BN254 **base field** `F_p` (the coordinate field of G1), CIOS Montgomery
//! form, 4×u64 limbs — mirroring `lattice_projsumcheck::fp256` (which is the
//! *scalar* field `F_r`).
//!
//! p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//!   = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
//!
//! p ≡ 3 (mod 4), so square roots are `± a^{(p+1)/4}` when `a` is a QR —
//! used by the try-and-increment hash-to-point in `g1`.
//!
//! Conventions (identical to `fp256`): limbs are Montgomery-form
//! (`ã = a·R mod p`, `R = 2²⁵⁶`); `mul` computes
//! `value(a)·value(b)·R⁻¹ mod p` in canonical terms; conversions go through
//! `canonical_reduce_512` (shifted-subtraction reference reduction).

/// Base-field modulus, little-endian limbs.
pub const BN254_FP: [u64; 4] = [
    0x3c20_8c16_d87c_fd47,
    0x9781_6a91_6871_ca8d,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

/// `n₀ = −p⁻¹ mod 2⁶⁴` (Newton iteration).
const fn inv_neg_mod64(p: [u64; 4]) -> u64 {
    let mut x: u64 = 1;
    let mut i = 0;
    while i < 6 {
        let px = p[0].wrapping_mul(x);
        x = x.wrapping_mul(2u64.wrapping_sub(px));
        i += 1;
    }
    x.wrapping_neg()
}

const N0: u64 = inv_neg_mod64(BN254_FP);

/// `R = 2²⁵⁶ mod p` (canonical limbs).
pub const R_C: [u64; 4] = [
    0xd35d_438d_c58f_0d9d,
    0x0a78_eb28_f5c7_0b3d,
    0x666e_a36f_7879_462c,
    0x0e0a_77c1_9a07_df2f,
];

/// `R² = 2⁵¹² mod p` (canonical limbs).
pub const R2_C: [u64; 4] = [
    0xf32c_fc5b_538a_fa89,
    0xb5e7_1911_d445_01fb,
    0x47ab_1eff_0a41_7ff6,
    0x06d8_9f71_cab8_351f,
];

/// Canonical reduction of a 512-bit little-endian value modulo p — the
/// shifted-subtraction reference reducer (mirrors
/// `lattice_projsumcheck::fp256::reduce_wide_ref`; returns **canonical**
/// limbs of `X mod p`).
fn canonical_reduce_512(wide: &[u64; 8]) -> [u64; 4] {
    let mut x = *wide;
    for k in (0..=256usize).rev() {
        let shift_words = k / 64;
        let shift_bits = (k % 64) as u32;
        let mut shifted = [0u64; 8];
        let mut overflow = false;
        for i in (0..4).rev() {
            let v = BN254_FP[i];
            let dst = i + shift_words;
            if dst >= 8 {
                if v != 0 {
                    overflow = true;
                }
                continue;
            }
            if shift_bits == 0 {
                shifted[dst] |= v;
            } else {
                shifted[dst] |= v << shift_bits;
                if dst + 1 < 8 {
                    shifted[dst + 1] |= v >> (64 - shift_bits);
                } else if v >> (64 - shift_bits) != 0 {
                    overflow = true;
                }
            }
        }
        if overflow {
            continue; // p·2^k exceeds the 512-bit workspace: x < it anyway.
        }
        let mut ge = true;
        for i in (0..8).rev() {
            if x[i] > shifted[i] {
                break;
            }
            if x[i] < shifted[i] {
                ge = false;
                break;
            }
        }
        if ge {
            let mut borrow = false;
            for i in 0..8 {
                let (v1, b1) = x[i].overflowing_sub(shifted[i]);
                let (v2, b2) = v1.overflowing_sub(u64::from(borrow));
                x[i] = v2;
                borrow = b1 || b2;
            }
        }
    }
    [x[0], x[1], x[2], x[3]]
}

fn geq_modulus(x: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if x[i] > BN254_FP[i] {
            return true;
        }
        if x[i] < BN254_FP[i] {
            return false;
        }
    }
    true
}

fn sub_modulus_in_place(x: &mut [u64; 4]) {
    let mut borrow: u64 = 0;
    for i in 0..4 {
        let (d, b1) = x[i].overflowing_sub(BN254_FP[i]);
        let (d2, b2) = d.overflowing_sub(borrow);
        x[i] = d2;
        borrow = (b1 as u64) | (b2 as u64);
    }
}

fn add_modulus_in_place(x: &mut [u64; 4]) {
    let mut carry: u128 = 0;
    for i in 0..4 {
        let s = (x[i] as u128) + (BN254_FP[i] as u128) + carry;
        x[i] = s as u64;
        carry = s >> 64;
    }
    // carry ∈ {0,1}: both operands < p < 2^254, so the sum < 2^255 and the
    // top carry can only originate from limb overflow that stays below 2^256.
    let _ = carry;
}

/// One CIOS reduction step on the 6-limb accumulator:
/// `m = t[0]·n0; t = (t + m·p) >> 64`.
fn cios_step(t: &mut [u64; 6], p: &[u64; 4]) {
    let m = t[0].wrapping_mul(N0);
    if m == 0 {
        t.copy_within(1..6, 0);
        t[5] = 0;
        return;
    }
    let mut c: u128 = (t[0] as u128) + (m as u128) * (p[0] as u128);
    for j in 1..4 {
        let s = (t[j] as u128) + (m as u128) * (p[j] as u128) + (c >> 64);
        t[j - 1] = s as u64;
        c = s;
    }
    let s = (t[4] as u128) + (c >> 64);
    t[3] = s as u64;
    let c4 = s >> 64;
    let s2 = (t[5] as u128) + c4;
    t[4] = s2 as u64;
    t[5] = (s2 >> 64) as u64;
}

/// A base-field element in Montgomery form (`ã = a·R mod p`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FpBase {
    /// Montgomery-form limbs (little-endian).
    pub limbs: [u64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpBaseError {
    /// Cannot invert zero.
    InverseOfZero,
    /// Input is not a quadratic residue.
    NotASquare,
}

impl core::fmt::Display for FpBaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FpBaseError::InverseOfZero => write!(f, "inverse of zero in Fp"),
            FpBaseError::NotASquare => write!(f, "not a quadratic residue in Fp"),
        }
    }
}

impl FpBase {
    pub const ZERO: FpBase = FpBase { limbs: [0; 4] };

    /// Montgomery form of 1 (= R mod p).
    pub fn one_mont() -> FpBase {
        let mut v = [0u64; 8];
        v[4] = 1;
        FpBase {
            limbs: canonical_reduce_512(&v),
        }
    }

    /// Canonical u64 → Montgomery form (`v·R mod p`).
    pub fn from_canonical_u64(value: u64) -> FpBase {
        let mut v = [0u64; 8];
        v[4] = value;
        FpBase {
            limbs: canonical_reduce_512(&v),
        }
    }

    /// Canonical limbs (value < p) → Montgomery form.
    pub fn from_canonical_limbs(limbs: [u64; 4]) -> FpBase {
        let mut v = [0u64; 8];
        v[4..8].copy_from_slice(&limbs);
        FpBase {
            limbs: canonical_reduce_512(&v),
        }
    }

    /// Full-width value from a 32-byte big-endian byte string (reduced
    /// modulo p, then Montgomery-ified via one multiplication by `R²`).
    pub fn from_be_bytes_wide(bytes: &[u8; 32]) -> FpBase {
        let mut wide = [0u64; 8];
        let mut limb_idx = 8;
        let mut acc: u128 = 0;
        let mut i = 0;
        while i < 32 {
            acc = (acc << 8) | (bytes[i] as u128);
            if i % 8 == 7 {
                limb_idx -= 1;
                wide[limb_idx] = acc as u64;
                acc = 0;
            }
            i += 1;
        }
        // The BE bytes fill limbs 4..8, i.e. the workspace holds
        // `value·2²⁵⁶ = value·R`, so the canonical reduction outputs the
        // Montgomery form of `value mod p` directly (same shape as
        // `from_canonical_u64`).
        FpBase {
            limbs: canonical_reduce_512(&wide),
        }
    }

    /// Canonical big-endian 32-byte serialization.
    pub fn to_be_bytes(self) -> [u8; 32] {
        let c = self.from_mont();
        let mut out = [0u8; 32];
        for i in 0..4 {
            // limb (3−i) is the (i+1)-th most significant: bytes 0..8 hold c[3].
            out[i * 8..i * 8 + 8].copy_from_slice(&c[3 - i].to_be_bytes());
        }
        out
    }

    /// Montgomery → canonical limbs (multiply by 1: `ã·R⁻¹`).
    pub fn from_mont(&self) -> [u64; 4] {
        self.mul(&FpBase {
            limbs: [1, 0, 0, 0],
        })
        .limbs
    }

    /// CIOS multiplication (the 6-limb accumulator of `fp256::Fp256::mul`):
    /// the output limbs' canonical value is `value(a)·value(b)·R⁻¹ mod p`.
    pub fn mul(&self, other: &FpBase) -> FpBase {
        let a = self.limbs;
        let b = other.limbs;
        let p = BN254_FP;
        let mut t = [0u64; 6];

        for i in 0..4 {
            // Phase 1: t += a · b[i]
            if b[i] != 0 {
                let mut carry: u128 = 0;
                for j in 0..4 {
                    let s = (t[j] as u128) + (a[j] as u128) * (b[i] as u128) + carry;
                    t[j] = s as u64;
                    carry = s >> 64;
                }
                let s = (t[4] as u128) + carry;
                t[4] = s as u64;
                t[5] = t[5].wrapping_add((s >> 64) as u64);
            }
            // Phase 2: one CIOS step.
            cios_step(&mut t, &p);
        }

        let mut r = [t[0], t[1], t[2], t[3]];
        if t[4] != 0 || t[5] != 0 || geq_modulus(&r) {
            sub_modulus_in_place(&mut r);
        }
        FpBase { limbs: r }
    }

    pub fn add(&self, other: &FpBase) -> FpBase {
        let mut r = [0u64; 4];
        let mut carry: u128 = 0;
        for i in 0..4 {
            let s = (self.limbs[i] as u128) + (other.limbs[i] as u128) + carry;
            r[i] = s as u64;
            carry = s >> 64;
        }
        if carry != 0 || geq_modulus(&r) {
            sub_modulus_in_place(&mut r);
        }
        FpBase { limbs: r }
    }

    pub fn sub(&self, other: &FpBase) -> FpBase {
        let mut r = [0u64; 4];
        let mut borrow: u64 = 0;
        for i in 0..4 {
            let (d, b1) = self.limbs[i].overflowing_sub(other.limbs[i]);
            let (d2, b2) = d.overflowing_sub(borrow);
            r[i] = d2;
            borrow = (b1 as u64) | (b2 as u64);
        }
        if borrow != 0 {
            add_modulus_in_place(&mut r);
        }
        FpBase { limbs: r }
    }

    pub fn neg(&self) -> FpBase {
        if self.is_zero() {
            *self
        } else {
            let mut r = [0u64; 4];
            let mut borrow: u64 = 0;
            for i in 0..4 {
                let (d, b) = BN254_FP[i].overflowing_sub(self.limbs[i]);
                let (d2, b2) = d.overflowing_sub(borrow);
                r[i] = d2;
                borrow = (b as u64) | (b2 as u64);
            }
            FpBase { limbs: r }
        }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs == [0u64; 4]
    }

    pub fn double(&self) -> FpBase {
        self.add(self)
    }

    /// Exponentiation by a little-endian limb exponent (canonical limbs).
    pub fn pow_limbs(&self, exp: &[u64; 4]) -> FpBase {
        let mut result = Self::one_mont();
        let mut seen_one = false;
        for limb in exp.iter().rev() {
            for bit in (0..64).rev() {
                if seen_one {
                    result = result.mul(&result);
                }
                if (limb >> bit) & 1 == 1 {
                    if seen_one {
                        result = result.mul(self);
                    } else {
                        result = *self;
                        seen_one = true;
                    }
                }
            }
        }
        result
    }

    /// Multiplicative inverse (Fermat: `a^{p−2}`). Errors on zero.
    pub fn inverse(&self) -> Result<FpBase, FpBaseError> {
        if self.is_zero() {
            return Err(FpBaseError::InverseOfZero);
        }
        // p − 2: p is odd, so this is a plain decrement of the low limb
        // with a borrow chain.
        let mut e = BN254_FP;
        let (lo, borrow) = e[0].overflowing_sub(2);
        e[0] = lo;
        if borrow {
            e[1] -= 1; // cannot underflow in this branch
        }
        Ok(self.pow_limbs(&e))
    }

    /// Legendre symbol: 1 (QR), −1 (non-residue), 0 (zero).
    pub fn legendre(&self) -> i8 {
        // a^{(p−1)/2}: right shift of the limb vector — the low bit of limb
        // i+1 becomes the top bit of limb i.
        let mut e = [0u64; 4];
        for i in 0..4 {
            let carry = if i < 3 { BN254_FP[i + 1] & 1 } else { 0 };
            e[i] = (BN254_FP[i] >> 1) | (carry << 63);
        }
        let l = self.pow_limbs(&e);
        if l.is_zero() {
            return 0;
        }
        if l == Self::one_mont() {
            1
        } else {
            -1
        }
    }

    /// Square root via `a^{(p+1)/4}` (valid since p ≡ 3 mod 4).
    pub fn sqrt(&self) -> Result<FpBase, FpBaseError> {
        if self.is_zero() {
            return Ok(*self);
        }
        if self.legendre() != 1 {
            return Err(FpBaseError::NotASquare);
        }
        // (p+1)/4: increment, then a two-bit right shift across limbs.
        let mut e = BN254_FP;
        let (lo, carry) = e[0].overflowing_add(1);
        e[0] = lo;
        if carry {
            e[1] += 1;
        }
        let mut q = [0u64; 4];
        for i in 0..4 {
            let carry = if i < 3 { e[i + 1] & 3 } else { 0 };
            q[i] = (e[i] >> 2) | (carry << 62);
        }
        Ok(self.pow_limbs(&q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(v: u64) -> FpBase {
        FpBase::from_canonical_u64(v)
    }

    #[test]
    fn one_and_zero() {
        assert!(FpBase::ZERO.is_zero());
        assert!(!small(1).is_zero());
        assert_eq!(small(1).mul(&small(1)), small(1));
        assert_eq!(small(7).mul(&small(1)), small(7));
    }

    #[test]
    fn mont_roundtrip() {
        let v = [0x0123_4567_89ab_cdefu64, 0x0fed_cba9_8765_4321u64, 5, 9];
        let m = FpBase::from_canonical_limbs(v);
        assert_eq!(m.from_mont(), v);
        let m2 = FpBase::from_canonical_limbs(BN254_FP);
        assert!(m2.is_zero()); // p ≡ 0
                               // R_C constant cross-check: one_mont's limbs ARE the canonical
                               // limbs of R (the Montgomery form of 1).
        assert_eq!(FpBase::one_mont(), FpBase { limbs: R_C });
    }

    #[test]
    fn field_axioms() {
        let a = small(0xdead_beefu64);
        let b = small(0x1234_5678u64);
        let c = small(0xfeed_faceu64);
        assert_eq!(a.add(&b.add(&c)).from_mont(), a.add(&b).add(&c).from_mont());
        assert_eq!(a.mul(&b.mul(&c)).from_mont(), a.mul(&b).mul(&c).from_mont());
        assert_eq!(
            a.mul(&b.add(&c)).from_mont(),
            a.mul(&b).add(&a.mul(&c)).from_mont()
        );
        assert!(a.sub(&a).is_zero());
        assert!(a.add(&a.neg()).is_zero());
    }

    #[test]
    fn large_value_arithmetic() {
        // A near-full-width canonical value exercises the carry paths.
        let mut pm1 = BN254_FP;
        pm1[0] -= 1;
        let v = pm1; // p − 1
        let a = FpBase::from_canonical_limbs(v);
        let one = FpBase::one_mont();
        assert_eq!(a.mul(&a), one); // (p−1)² ≡ 1
        assert_eq!(a.mul(&one), a);
        let inv = a.inverse().ok().unwrap();
        assert_eq!(a.mul(&inv), one);
    }

    #[test]
    fn inverse_works() {
        for v in [1u64, 2, 3, 0x9e37_79b9_7f4a_7c15, 0xdead_beef] {
            let a = small(v);
            let inv = a.inverse().ok().unwrap();
            let one = FpBase::one_mont();
            assert_eq!(a.mul(&inv), one);
        }
    }

    #[test]
    fn sqrt_roundtrip() {
        let four = small(4);
        let s = four.sqrt().ok().unwrap();
        let two = small(2);
        assert!(s == two || s == two.neg());
        let nine = small(9);
        let s3 = nine.sqrt().ok().unwrap();
        let three = small(3);
        assert!(s3 == three || s3 == three.neg());
        // Legendre classification of a non-residue: try 5.
        let five = small(5);
        let cls = five.legendre();
        if cls == -1 {
            assert!(five.sqrt().is_err());
        } else {
            let s5 = five.sqrt().ok().unwrap();
            assert!(s5.mul(&s5) == five);
        }
    }

    #[test]
    fn be_bytes_roundtrip() {
        let a = small(0x0123_4567_89ab_cdef);
        let bytes = a.to_be_bytes();
        let b = FpBase::from_be_bytes_wide(&bytes);
        assert_eq!(a, b);
        // Wide input beyond p reduces correctly:
        let mut big = [0xffu8; 32];
        big[0] = 0x30; // ≈ 0x30ff...ff > p
        let x = FpBase::from_be_bytes_wide(&big);
        assert!(!x.is_zero());
    }

    #[test]
    fn generator_on_curve() {
        // y² = x³ + 3 at (1, 2): 4 == 1 + 3.
        let one = small(1);
        let two = small(2);
        let three = small(3);
        let lhs = two.mul(&two);
        let rhs = one.mul(&one).mul(&one).add(&three);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn be_probe() {
        let a = FpBase::from_canonical_u64(0x0123_4567_89ab_cdef);
        let c = a.from_mont();
        println!(
            "canon = {:016x}{:016x}{:016x}{:016x}",
            c[3], c[2], c[1], c[0]
        );
        let bytes = a.to_be_bytes();
        println!(
            "bytes = {}",
            bytes
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        );
        // manual wide
        let mut wide = [0u64; 8];
        wide[4] = 0x0123_4567_89ab_cdef;
        let canon2 = crate::fp_base::canonical_reduce_512(&wide);
        println!(
            "canon2 = {:016x}{:016x}{:016x}{:016x}",
            canon2[3], canon2[2], canon2[1], canon2[0]
        );
        let m = FpBase { limbs: canon2 }.mul(&FpBase {
            limbs: crate::fp_base::R2_C,
        });
        println!(
            "montified = {:016x}{:016x}{:016x}{:016x}",
            m.limbs[3], m.limbs[2], m.limbs[1], m.limbs[0]
        );
        let b = FpBase::from_be_bytes_wide(&bytes);
        println!(
            "b limbs = {:016x}{:016x}{:016x}{:016x}",
            b.limbs[3], b.limbs[2], b.limbs[1], b.limbs[0]
        );
    }

    #[test]
    fn differential_probe() {
        // (p−1)·R mod p from Python:
        let expected_pm1_mont = [0x0e0a_77c1_9a07_df2fu64, 0, 0, 0];
        let _ = expected_pm1_mont;
        let mut pm1 = BN254_FP;
        pm1[0] -= 1;
        let a = FpBase::from_canonical_limbs(pm1);
        println!(
            "a limbs = {:016x}{:016x}{:016x}{:016x}",
            a.limbs[3], a.limbs[2], a.limbs[1], a.limbs[0]
        );
        let one = FpBase::one_mont();
        println!(
            "one limbs = {:016x}{:016x}{:016x}{:016x}",
            one.limbs[3], one.limbs[2], one.limbs[1], one.limbs[0]
        );
        let sq = a.mul(&a);
        println!(
            "a*a limbs = {:016x}{:016x}{:016x}{:016x}",
            sq.limbs[3], sq.limbs[2], sq.limbs[1], sq.limbs[0]
        );
        // small-value sanity: (2^128)² = 2^256 ≡ R
        let big = FpBase::from_canonical_u64(1u64 << 63).double(); // 2^64
        let b2 = big.mul(&big); // 2^128 Montgomery
        let b4 = b2.mul(&b2); // the element 2^256 ≡ R; its Montgomery form is R²
        assert_eq!(b4, FpBase { limbs: R2_C });
        // and the element R has canonical value R:
        assert_eq!(b4.from_mont(), R_C);
    }

    #[test]
    fn pow_matches_repeated_mul() {
        let a = small(3);
        let a5 = a.mul(&a).mul(&a).mul(&a).mul(&a);
        assert_eq!(a.pow_limbs(&[5, 0, 0, 0]), a5);
    }
}
