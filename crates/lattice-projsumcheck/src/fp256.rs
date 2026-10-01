//! **Upper-limb Montgomery challenges over 256-bit prime fields** —
//! §5 of ePrint 2026/762 ("Smaller challenges for 256-bit fields").
//!
//! When targeting λ ≈ 128 bits of security, sum-check challenges need
//! only be sampled from a subset of size ≈ 2^λ rather than the full
//! ≈ 2^256 field (round-by-round soundness `d/|S|` via Schwartz–Zippel,
//! Proposition 5.2). The paper's insight: choose the subset so that, in
//! **Montgomery form**, the challenge's lower limbs are zero —
//!
//! ```text
//! S = { a ∈ F_p : ā mod 2^{w·ℓ} = 0 },   ℓ = 2 zeroed low limbs
//! ```
//!
//! i.e. the λ-bit Fiat–Shamir output is left-shifted into the *upper*
//! limbs and treated as already being in Montgomery form. A CIOS
//! multiplication whose right operand lies in `S` then short-circuits:
//!
//! * **Phase 1** (operand scanning): for the two low `i` steps `b[i] = 0`
//!   skips the whole inner product — 8 of the 16 limb products;
//! * **Phase 2** (reduction): during the first two reduction steps the
//!   intermediate's low limb is already zero, so `m_i = −p^{−1}·c_i` is
//!   zero and the `m·p` limb products are skipped — 10 more.
//!
//! In total 18 of the 36 native multiplications of a 4-limb CIOS product
//! are eliminated — a measured **1.92×** chained-multiplication speedup,
//! and **1.82×** for the full projective binding loop (Table 3 / Table 5;
//! the theoretical ceiling is 2×).
//!
//! The 12-bit gap to a 128-bit target at λ = 125 is closed by a standard
//! **grinding** proof-of-work step (§5.3), implemented below.
//!
//! The modulus is the BN254 scalar field (the paper's benchmark field);
//! BLS12-381's Fr would use λ = 127 with one overflow bit.

// Index-arithmetic loops (MSB-first bit extraction, limb walks,
// prefix/suffix products) read clearer with explicit indices.
#![allow(clippy::needless_range_loop)]
/// BN254 scalar field modulus, little-endian limbs:
/// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001.
///
/// (Limb 2 was previously mistyped `…58d2` for `…585d` — a digit
/// transposition that silently replaced the prime field with a
/// **composite** modulus; every local test still passed because the
/// reference reducer shared the same wrong constant. Surfaced — and
/// fixed — by the fast-prover port's cross-constants
/// (`R`, `R²` computed against the true BN254 Fr).)
pub const BN254_FR: [u64; 4] = [
    0x43e1_f593_f000_0001,
    0x2833_e848_79b9_7091,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

/// Security parameter for the challenge subset: |S| = 2^125 (the top 3
/// bits of the high limb are zeroed to keep CIOS intermediates below
/// overflow — §5.1's caveat).
pub const CHALLENGE_LAMBDA: u32 = 125;

/// `n0 = −p^{−1} mod 2^64` — Newton iteration at compile time (six
/// doubling steps give all 64 bits).
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

const N0: u64 = inv_neg_mod64(BN254_FR);

/// `R = 2^256 mod p` in canonical limbs — the Montgomery radix (the
/// canonical value of the Montgomery form of 1).
pub const R_C: [u64; 4] = [
    0xac96_341c_4fff_fffb,
    0x36fc_7695_9f60_cd29,
    0x666e_a36f_7879_462e,
    0x0e0a_77c1_9a07_df2f,
];

/// `R² = 2^512 mod p` in canonical limbs — the canonical→Montgomery
/// conversion constant.
pub const R2_C: [u64; 4] = [
    0x1bb8_e645_ae21_6da7,
    0x53fe_3ab1_e35c_59e3,
    0x8c49_833d_53bb_8085,
    0x0216_d0b1_7f4e_44a5,
];

/// A 256-bit prime field element in Montgomery form
/// (`ã = a·R mod p`, `R = 2^256`), 4×u64 little-endian limbs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fp256 {
    /// Montgomery-form limbs (little-endian).
    pub limbs: [u64; 4],
}

impl Fp256 {
    pub const ZERO: Fp256 = Fp256 { limbs: [0; 4] };

    /// The Montgomery representation of 1 (= R mod p), via the reference
    /// reduction of `2^256`.
    pub fn one_mont() -> Fp256 {
        let mut v = [0u64; 8];
        v[4] = 1;
        Fp256 { limbs: reduce_wide_ref(&v) }
    }

    /// Canonical `u64` value → Montgomery form (`v·R mod p`).
    pub fn from_canonical_u64(value: u64) -> Fp256 {
        let mut v = [0u64; 8];
        v[4] = value;
        Fp256 { limbs: reduce_wide_ref(&v) }
    }

    /// Full CIOS Montgomery multiplication — phase 1: 16 limb products;
    /// phase 2: 4 reduction steps at 4+1 products = 36 native
    /// multiplications total (§5.1's count).
    ///
    /// Zero-limb short-circuits (exact — the skipped products are zero):
    /// * phase 1 skips the whole inner loop when `b[i] = 0`;
    /// * phase 2 shifts only when `m = t[0]·n0 = 0`.
    ///
    /// This makes every multiplication whose second operand is a small
    /// **canonical** integer (or an upper-limb challenge) cost ~12–24
    /// native multiplications instead of 36 — the sb kernel of the
    /// window fast prover (`crate::fastprover`).
    ///
    /// Accumulator invariant: `t[0..5]` holds a value < `2^{64·5}·(small
    /// slack)`; the extra sixth word absorbs the phase-1 carry and is
    /// folded back by phase 2's right shift.
    pub fn mul(&self, other: &Fp256) -> Fp256 {
        let a = self.limbs;
        let b = other.limbs;
        let p = BN254_FR;
        let mut t = [0u64; 6];

        for i in 0..4 {
            // Phase 1: t += a · b[i] — skipped entirely when b[i] = 0.
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

            // Phase 2: m = t[0]·n0;  t = (t + m·p) >> 64 — a pure shift
            // when m = 0 (t[0] = 0, the previous step's invariant).
            let m = t[0].wrapping_mul(N0);
            if m == 0 {
                t[0] = t[1];
                t[1] = t[2];
                t[2] = t[3];
                t[3] = t[4];
                t[4] = t[5];
                t[5] = 0;
                continue;
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

        let mut r = [t[0], t[1], t[2], t[3]];
        if t[4] != 0 || t[5] != 0 || geq_p(&r) {
            r = sub_p(&r);
        }
        Fp256 { limbs: r }
    }

    /// **Short-circuit multiplication with an upper-limb challenge**
    /// (§5.2): `upper` must satisfy `limbs[0] == limbs[1] == 0` (the
    /// Montgomery image of the challenge set `S`). Skips 8 phase-1 limb
    /// products (the `b[i] = 0` steps) and 10 phase-2 products (`m = 0`
    /// steps shift only) — **18 of 36** native multiplications, the
    /// measured 1.92× path.
    pub fn mul_upper_limb(&self, upper: &Fp256) -> Fp256 {
        debug_assert!(upper.limbs[0] == 0 && upper.limbs[1] == 0);
        let a = self.limbs;
        let b = upper.limbs;
        let p = BN254_FR;
        let mut t = [0u64; 6];

        for i in 0..4 {
            // Phase 1: b[i] = 0 for i < 2 — skip the entire inner loop.
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

            // Phase 2: t[0] is zero exactly when phase 1 was skipped
            // (the accumulator's low limb survives only through products
            // with b's zero limbs) — m = 0, so the m·p products vanish
            // and the step is a pure right shift.
            let m = t[0].wrapping_mul(N0);
            if m == 0 {
                // Shift-only reduction: t[0] = 0 by construction.
                t[0] = t[1];
                t[1] = t[2];
                t[2] = t[3];
                t[3] = t[4];
                t[4] = t[5];
                t[5] = 0;
                continue;
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

        let mut r = [t[0], t[1], t[2], t[3]];
        if t[4] != 0 || t[5] != 0 || geq_p(&r) {
            r = sub_p(&r);
        }
        Fp256 { limbs: r }
    }

    /// Sample an upper-limb challenge from 32 transcript bytes (§5.1's
    /// Fiat–Shamir instantiation): truncate to λ = 125 bits, left-shift
    /// by `w·ℓ = 128` bits into the upper limbs, and treat the result as
    /// *already being in Montgomery form*.
    pub fn sample_upper_limb(hash32: &[u8; 32]) -> Fp256 {
        let lo = u64::from_le_bytes([
            hash32[0], hash32[1], hash32[2], hash32[3],
            hash32[4], hash32[5], hash32[6], hash32[7],
        ]);
        let hi = u64::from_le_bytes([
            hash32[8], hash32[9], hash32[10], hash32[11],
            hash32[12], hash32[13], hash32[14], hash32[15],
        ]) & 0x1FFF_FFFF_FFFF_FFFF; // 61 bits: |S| = 2^125.
        Fp256 { limbs: [0, 0, lo, hi] }
    }

    /// Whether this value lies in the upper-limb challenge set.
    pub fn is_upper_limb(&self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0
    }

    /// Montgomery-domain addition (limb-wise with carry, then one
    /// conditional subtraction — operands are < 2p).
    pub fn add(&self, other: &Fp256) -> Fp256 {
        let mut r = [0u64; 4];
        let mut c: u128 = 0;
        for (ri, (&a, &b)) in r.iter_mut().zip(self.limbs.iter().zip(other.limbs.iter())) {
            let s = (a as u128) + (b as u128) + c;
            *ri = s as u64;
            c = s >> 64;
        }
        if c != 0 || geq_p(&r) {
            r = sub_p(&r);
        }
        Fp256 { limbs: r }
    }

    /// Montgomery-domain subtraction (limb-wise with borrow, then a
    /// conditional `+p` when the borrow fired — operands < p).
    pub fn sub(&self, other: &Fp256) -> Fp256 {
        let mut r = [0u64; 4];
        let mut borrow = false;
        for i in 0..4 {
            let (v1, b1) = self.limbs[i].overflowing_sub(other.limbs[i]);
            let (v2, b2) = v1.overflowing_sub(u64::from(borrow));
            r[i] = v2;
            borrow = b1 || b2;
        }
        if borrow {
            // Add p back: (a − b) + p < p since a, b < p.
            let mut c: u128 = 0;
            for i in 0..4 {
                let s = (r[i] as u128) + (BN254_FR[i] as u128) + c;
                r[i] = s as u64;
                c = s >> 64;
            }
        }
        Fp256 { limbs: r }
    }

    /// Negation (`p − self`, zero maps to zero) — valid for either form.
    pub fn neg(&self) -> Fp256 {
        if self.limbs.iter().all(|&l| l == 0) {
            return *self;
        }
        let mut r = [0u64; 4];
        let mut borrow = false;
        for i in 0..4 {
            let (v1, b1) = BN254_FR[i].overflowing_sub(self.limbs[i]);
            let (v2, b2) = v1.overflowing_sub(u64::from(borrow));
            r[i] = v2;
            borrow = b1 || b2;
        }
        Fp256 { limbs: r }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    // -----------------------------------------------------------------
    // The canonical ↔ Montgomery bridge (the fast prover's field layer)
    // -----------------------------------------------------------------

    /// The canonical representative of 1 (limb 0 = 1).
    pub const ONE_CANON: Fp256 = Fp256 { limbs: [1, 0, 0, 0] };

    /// `self` (a **canonical** value, limbs < p) → Montgomery form.
    /// `CIOS(a, R²) = a·R²·R^{-1} = a·R`.
    pub fn to_mont(&self) -> Fp256 {
        self.mul(&Fp256 { limbs: R2_C })
    }

    /// `self` (a **Montgomery** value) → canonical.
    /// `CIOS(ã, 1) = a·R·R^{-1} = a`.
    pub fn from_mont(&self) -> Fp256 {
        self.mul(&Fp256::ONE_CANON)
    }

    /// The TRUE product `a·b mod p` of two **canonical** values, returned
    /// in canonical form (`CIOS(to_mont(a), b) = a·b`). Two CIOS calls —
    /// for the verifier's O(d²)-per-round interpolation work only; the
    /// prover's hot loops use the Montgomery-constant kernels.
    pub fn mul_canon(&self, other: &Fp256) -> Fp256 {
        self.to_mont().mul(other)
    }

    /// The canonical limbs of a signed small integer `|s| < 2^127 < p/2`
    /// (positive: plain limbs; negative: `p − |s|`).
    pub fn canon_i128(s: i128) -> Fp256 {
        if s >= 0 {
            let u = s as u128;
            Fp256 { limbs: [u as u64, (u >> 64) as u64, 0, 0] }
        } else {
            let u = s.unsigned_abs();
            let small = [u as u64, (u >> 64) as u64, 0, 0];
            let mut r = [0u64; 4];
            let mut borrow = false;
            for i in 0..4 {
                let (v1, b1) = BN254_FR[i].overflowing_sub(small[i]);
                let (v2, b2) = v1.overflowing_sub(u64::from(borrow));
                r[i] = v2;
                borrow = b1 || b2;
            }
            Fp256 { limbs: r }
        }
    }

    /// The signed small value back (valid when the canonical value's
    /// magnitude < 2^127); `None` otherwise.
    pub fn to_i128(&self) -> Option<i128> {
        let c = self.limbs;
        if c[2] == 0 && c[3] == 0 {
            let u = (c[0] as u128) | ((c[1] as u128) << 64);
            if u < (1u128 << 127) {
                return Some(u as i128);
            }
        }
        // Negative branch: p − |x| with |x| < 2^127 ⟺ self + 2^127 > p.
        let neg = self.neg();
        let n = neg.limbs;
        if n[2] == 0 && n[3] == 0 {
            let u = (n[0] as u128) | ((n[1] as u128) << 64);
            if u < (1u128 << 127) {
                return Some(-(u as i128));
            }
        }
        None
    }

    /// The **sb kernel**: TRUE product `x·s` in canonical form, where
    /// `self = x̄` is MONTGOMERY and `s` is a signed small integer
    /// (`CIOS(x̄, s_canonical) = x·s`). Phase-1/phase-2 skip the zero
    /// limbs of `|s|` — ~12 native multiplications for one-limb `s`.
    pub fn mul_small(&self, s: i128) -> Fp256 {
        let neg = s < 0;
        let u = s.unsigned_abs(); // < 2^127
        let b = Fp256 { limbs: [u as u64, (u >> 64) as u64, 0, 0] };
        let r = self.mul(&b);
        if neg {
            r.neg()
        } else {
            r
        }
    }

    /// A canonical challenge from 32 transcript bytes: mapped into the
    /// upper-limb challenge set (limbs 0–1 zero — always < p, no
    /// rejection, and every later multiplication by its Montgomery form
    /// takes the CIOS short-circuit; §5 of ePrint 2026/762).
    pub fn challenge_upper(hash32: &[u8; 32]) -> Fp256 {
        Self::sample_upper_limb(hash32)
    }

    /// Fixed-width big-endian canonical byte encoding (the transcript
    /// encoding for round messages; caller keeps values canonical).
    pub fn canon_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.limbs[3 - i].to_be_bytes());
        }
        out
    }

    /// Little-endian canonical limbs (unchecked reduction; callers that
    /// need canonical form must supply limbs < p).
    pub fn from_limbs(limbs: [u64; 4]) -> Fp256 {
        Fp256 { limbs }
    }
    /// Canonical `u128` value → Montgomery form (`v·R mod p`): the value
    /// occupies the high half of the 512-bit workspace (`v·2^256`).
    pub fn from_canonical_u128(value: u128) -> Fp256 {
        let wide = [0, 0, 0, 0, value as u64, (value >> 64) as u64, 0, 0];
        Fp256 { limbs: reduce_wide_ref(&wide) }
    }

    /// Montgomery exponentiation (square-and-multiply over the 256-bit
    /// exponent) — used by [`Fp256::inverse`].
    pub fn pow(&self, exp: &[u64; 4]) -> Fp256 {
        let mut result = Fp256::one_mont();
        let base = *self;
        for limb in exp.iter().rev() {
            for i in (0..64).rev() {
                result = result.mul(&result);
                if (limb >> i) & 1 == 1 {
                    result = result.mul(&base);
                }
            }
        }
        result
    }

    /// Multiplicative inverse via Fermat little theorem (`a^{p−2}`) —
    /// `BN254_FR` is prime. Returns `None` for zero.
    pub fn inverse(&self) -> Option<Fp256> {
        if self.is_zero() {
            return None;
        }
        let mut e = BN254_FR;
        let mut borrow = false;
        for i in 0..4 {
            let (v1, b1) = e[i].overflowing_sub(if i == 0 { 2 } else { 0 });
            let (v2, b2) = v1.overflowing_sub(u64::from(borrow));
            e[i] = v2;
            borrow = b1 || b2;
        }
        Some(self.pow(&e))
    }

}

fn geq_p(x: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if x[i] > BN254_FR[i] {
            return true;
        }
        if x[i] < BN254_FR[i] {
            return false;
        }
    }
    true
}

fn sub_p(x: &[u64; 4]) -> [u64; 4] {
    let mut r = [0u64; 4];
    let mut borrow = false;
    for i in 0..4 {
        let (v1, b1) = x[i].overflowing_sub(BN254_FR[i]);
        let (v2, b2) = v1.overflowing_sub(u64::from(borrow));
        r[i] = v2;
        borrow = b1 || b2;
    }
    r
}

/// Reference reduction of a 512-bit little-endian integer modulo
/// `BN254_FR` by shift-and-subtract — the slow, obviously-correct path
/// used to bootstrap and cross-validate CIOS in the tests.
pub fn reduce_wide_ref(wide: &[u64; 8]) -> [u64; 4] {
    let mut x = *wide;
    for k in (0..=256usize).rev() {
        let shift_words = k / 64;
        let shift_bits = (k % 64) as u32;
        let mut shifted = [0u64; 8];
        let mut overflow = false;
        for i in (0..4).rev() {
            let v = BN254_FR[i];
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

/// **Grinding** (§5.3): find a nonce `η` such that `H(seed ∥ η)` has at
/// least `gamma` leading zero bits. A cheating prover must re-grind for
/// every modified commitment, multiplying attack cost by `2^γ`; the
/// honest prover pays `2^γ` hashes once, outside the sum-check hot loop.
pub fn grind(seed: &[u8], gamma_bits: u32) -> Option<(u64, [u8; 32])> {
    let mut nonce: u64 = 0;
    loop {
        let mut msg = seed.to_vec();
        msg.extend_from_slice(&nonce.to_le_bytes());
        let h = lattice_core::transcript::Transcript::hash_domain(b"projsumcheck-grind", &msg);
        if leading_zero_bits(&h) >= gamma_bits {
            return Some((nonce, h));
        }
        nonce = nonce.wrapping_add(1);
        if nonce == 0 {
            return None;
        }
    }
}

/// Verify a grinding nonce.
pub fn verify_grind(seed: &[u8], nonce: u64, gamma_bits: u32) -> bool {
    let mut msg = seed.to_vec();
    msg.extend_from_slice(&nonce.to_le_bytes());
    let h = lattice_core::transcript::Transcript::hash_domain(b"projsumcheck-grind", &msg);
    leading_zero_bits(&h) >= gamma_bits
}

fn leading_zero_bits(h: &[u8; 32]) -> u32 {
    let mut count = 0u32;
    for &b in h.iter() {
        if b == 0 {
            count += 8;
        } else {
            count += b.leading_zeros();
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mul_u64(a: u64, b: u64) -> (u64, u64) {
        let p = (a as u128) * (b as u128);
        (p as u64, (p >> 64) as u64)
    }

    /// CIOS equals the reference: `mont(to_mont(v1), to_mont(v2)) ==
    /// to_mont(v1·v2 mod p)` — the full bootstrap identity.
    #[test]
    fn cios_matches_reference() {
        for (v1, v2) in [
            (1u64, 1u64),
            (0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
            (u64::MAX, u64::MAX),
            (0x3064_4e72_e131_a029u64, 0x43e1_f593_f000_0001),
            (1u64 << 63, 3),
        ] {
            let a = Fp256::from_canonical_u64(v1);
            let b = Fp256::from_canonical_u64(v2);
            let prod = a.mul(&b);
            // Reference: (v1·v2 mod p) << 256, reduced = to_mont(v1·v2 mod p).
            let (lo, hi) = mul_u64(v1, v2);
            let wide = [lo, hi, 0, 0, 0, 0, 0, 0];
            let modprod = reduce_wide_ref(&wide);
            let mut shifted = [0u64; 8];
            shifted[4] = modprod[0];
            shifted[5] = modprod[1];
            shifted[6] = modprod[2];
            shifted[7] = modprod[3];
            let want = reduce_wide_ref(&shifted);
            assert_eq!(prod.limbs, want, "CIOS mismatch for {v1}·{v2}");
        }
    }

    /// Multiplicative identity and commutativity in the Montgomery domain.
    #[test]
    fn cios_identity_and_commutativity() {
        let one = Fp256::one_mont();
        let a = Fp256::from_canonical_u64(777);
        assert_eq!(a.mul(&one).limbs, a.limbs);
        let b = Fp256::from_canonical_u64(123_456_789);
        assert_eq!(a.mul(&b).limbs, b.mul(&a).limbs);
    }

    /// The upper-limb short-circuit is bit-exact with full CIOS on the
    /// same values, including through chained multiplications.
    #[test]
    fn upper_limb_matches_full_mul() {
        let a = Fp256::from_canonical_u64(0xdead_beef_cafe_f00d);
        let mut hash = [7u8; 32];
        hash[0] = 0x42;
        let up = Fp256::sample_upper_limb(&hash);
        assert!(up.is_upper_limb());
        assert_eq!(a.mul_upper_limb(&up).limbs, a.mul(&up).limbs);
        let mut hash2 = [9u8; 32];
        hash2[3] = 0xff;
        let up2 = Fp256::sample_upper_limb(&hash2);
        let chained_upper = a.mul_upper_limb(&up).mul_upper_limb(&up2);
        let chained_full = a.mul(&up).mul(&up2);
        assert_eq!(chained_upper.limbs, chained_full.limbs);
    }

    /// Montgomery addition is distributive over mul (field sanity).
    #[test]
    fn addition_distributes() {
        let a = Fp256::from_canonical_u64(31);
        let b = Fp256::from_canonical_u64(47);
        let c = Fp256::from_canonical_u64(59);
        let lhs = a.mul(&b.add(&c));
        let rhs = a.mul(&b).add(&a.mul(&c));
        assert_eq!(lhs.limbs, rhs.limbs);
    }

    /// Challenge sampling: 125-bit values in the upper limbs.
    #[test]
    fn challenge_shape() {
        for t in 0..8u8 {
            let mut h = [0u8; 32];
            h[0] = t;
            h[14] = 0xff;
            h[15] = 0xff;
            let c = Fp256::sample_upper_limb(&h);
            assert_eq!(c.limbs[0], 0);
            assert_eq!(c.limbs[1], 0);
            assert!(c.limbs[3] >> 61 == 0, "top bits not cleared");
        }
    }

    /// Grinding roundtrip at a small bit count.
    #[test]
    fn grinding_roundtrip() {
        let seed = b"grind-seed";
        if let Some((nonce, _)) = grind(seed, 8) {
            assert!(verify_grind(seed, nonce, 8));
            assert!(!verify_grind(seed, nonce.wrapping_add(1), 8));
        }
    }
}
