//! The ~2^50 modulus class with incomplete-NTT **quadratic slots**
//! (Wave 6 substrate, `NEXT_STEPS.md` §2.2).
//!
//! The papers that LZX implements need a modulus with
//! `q ≡ 129 (mod 256)` — exactly two-adicity 7 — to host their
//! extension-field challenge spaces and CRT slot batching:
//! * Cyclo: κ_nu 2^-94 at paper parameters (vs 2^-24 on the current
//!   fully-splitting Q32),
//! * PikkuFold: the ε_C wrap-margin condition,
//! * RoKoko: its §9 kernel story (this is RoKoko's own modulus),
//! * SALSA: CRT slot batching for the norm sumcheck.
//!
//! **Modulus**: `q = 2^50 - 2687 = 1125899906839937` — prime,
//! `q ≡ 129 (mod 256)`, two-adicity 7 (verified by test), the first
//! modulus of the RoKoko reference implementation.
//!
//! **Incomplete NTT discipline** (the `ntt_quad.rs` lineage): with
//! two-adicity 7, F_q contains the 128th roots of unity but no
//! 256th roots, so a *full* negacyclic NTT of length n = 2^k exists only
//! for k ≤ 6. For k ≤ 7 we instead run the **incomplete DIF transform**
//! that factors
//!
//! `X^n + 1 = Π_{p=0}^{n/2-1} (X² - γ_p)`,  `γ_p = ζ^{(2p+1)·2^{7-k}}`,
//!
//! where ζ is a primitive 128th root of unity. Each DIF level halves the
//! block span using per-block twiddles `w = ζ^{(2i+1)·2^{5-d}}` (the
//! square root of the block's γ); after k-1 levels the representation is
//! `n/2` quadratic slots, and ring multiplication is slotwise Karatsuba
//! (`(u0+u1X)(v0+v1X) mod (X²-γ)` = 3 modular multiplications). Every
//! butterfly is exactly invertible (`lo' = lo + w·hi`, `hi' = lo - w·hi`),
//! so the inverse needs no global scaling factor.
//!
//! At k = 7 (n = 128) the leaves are the paper's "64 quadratic slots"
//! with constants `ζ^{odd}`; at smaller k the leaf constants are the odd
//! powers of the primitive 2^{k}·... root — the factorization of X^n+1
//! over F_q into quadratics, exact for every n = 2^k ≤ 128.
//!
//! **Out of scope, honestly stated**: n > 128 needs the odd-conductor
//! mixed-radix machinery or an RNS modulus stack (Wave 8, `lattice-rns`);
//! this module is the sound, tested substrate at the papers' slot scale.

use std::sync::Arc;

/// q = 2^50 - 2687 — prime, ≡ 129 mod 256, two-adicity 7.
pub const Q_50: u64 = 1_125_899_906_839_937;

/// The two-adicity of q - 1 (q - 1 = 2^7 · 8796093022187).
pub const Q_50_TWO_ADICITY: u32 = 7;

/// A generator of the multiplicative group mod q.
pub const Q_50_GENERATOR: u64 = 3;

/// A ~2^50 prime modulus with Barrett hot-path arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Modulus50 {
    /// The prime q (odd, < 2^50).
    pub q: u64,
    /// Barrett reciprocal `⌊2^64/q⌋`.
    recip: u64,
    /// Two-adicity of q - 1.
    pub two_adicity: u32,
    /// Multiplicative-group generator.
    pub generator: u64,
}

impl Modulus50 {
    /// The canonical Wave-6 modulus (`2^50 - 2687`).
    pub const Q_50: Modulus50 = Modulus50 {
        q: Q_50,
        recip: u64::MAX / Q_50,
        two_adicity: Q_50_TWO_ADICITY,
        generator: Q_50_GENERATOR,
    };

    /// Construct with a custom ~2^50 prime (reciprocal derived).
    pub const fn new(q: u64, two_adicity: u32, generator: u64) -> Self {
        Modulus50 {
            q,
            recip: u64::MAX / q,
            two_adicity,
            generator,
        }
    }

    /// Reduce a u128 into [0, q). Products of two residues are < 2^100, so
    /// first fold the high bits via `2^50 ≡ 2^50 - q (mod q)`, then one
    /// Barrett pass over a value < 2^62 (u128 intermediate, ≤ 2
    /// conditional corrections).
    #[inline]
    pub fn reduce_u128(&self, x: u128) -> u64 {
        let q = self.q as u128;
        if x < q {
            return x as u64;
        }
        // Fold: x = hi·2^50 + lo  →  hi·(2^50 mod q) + lo  (< 2^50·2^12 = 2^62
        // when hi < 2^50 and 2^50 mod q < 2^12).
        let fold = (1u128 << 50) % q; // 2^50 - q for q ∈ (2^49, 2^50)
        let hi = x >> 50;
        let lo = x & ((1u128 << 50) - 1);
        let mut y = hi.wrapping_mul(fold).wrapping_add(lo);
        // y can still exceed 2^62 slightly when hi is near 2^50; re-fold once.
        if y >> 62 != 0 {
            let hi2 = y >> 50;
            let lo2 = y & ((1u128 << 50) - 1);
            y = hi2.wrapping_mul(fold).wrapping_add(lo2);
        }
        // Barrett over y < 2^63: q̂ = ⌊y·m/2^64⌋ ∈ {⌊y/q⌋-1, ⌊y/q⌋}.
        let qhat = (y * self.recip as u128) >> 64;
        let mut r = y.wrapping_sub(qhat.wrapping_mul(q));
        if r >= q {
            r -= q;
        }
        if r >= q {
            r -= q;
        }
        r as u64
    }

    /// Reduce a u64 into [0, q).
    #[inline]
    pub fn reduce_u64(&self, x: u64) -> u64 {
        self.reduce_u128(u128::from(x))
    }

    /// Modular multiplication of two residues.
    #[inline]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        self.reduce_u128(u128::from(a) * u128::from(b))
    }

    /// Modular addition.
    #[inline]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        if s >= self.q {
            s - self.q
        } else {
            s
        }
    }

    /// Modular subtraction.
    #[inline]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a + (self.q - b)
        }
    }

    /// Modular negation.
    #[inline]
    pub fn neg(&self, a: u64) -> u64 {
        if a == 0 {
            0
        } else {
            self.q - a
        }
    }

    /// Reduce a signed integer into [0, q).
    pub fn reduce_i64(&self, x: i64) -> u64 {
        let r = x.rem_euclid(self.q as i64);
        r as u64
    }

    /// Modular exponentiation.
    pub fn pow(&self, base: u64, exp: u64) -> u64 {
        let mut acc: u64 = 1;
        let mut b = base % self.q;
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
    pub fn inv(&self, a: u64) -> Option<u64> {
        if a % self.q == 0 {
            None
        } else {
            Some(self.pow(a, self.q - 2))
        }
    }

    /// Primitive 2^k-th root of unity for k ≤ two_adicity.
    pub fn root_of_unity(&self, k: u32) -> Option<u64> {
        if k > self.two_adicity {
            return None;
        }
        let pow = (self.q - 1) >> k;
        Some(self.pow(self.generator, pow))
    }
}

/// Precomputed incomplete-NTT tables for one (modulus, log n) pair.
///
/// Block→γ mapping (derived from the DIF recursion): block `i` at depth
/// `d` carries modulus `X^{span} - γ` with
/// `γ = ζ^{2^{6-d}·(2·rev_d(i) + 1)}` where `rev_d` is the d-bit
/// **bit-reversal** — the level-order (breadth-first) block layout puts
/// the naive `(2i+1)` exponents in bit-reversed order. The DIF split of
/// block `i` uses twiddle `w = √γ = ζ^{2^{5-d}·(2·rev_d(i)+1)}`.
pub struct QuadSlotTables {
    pub modulus: Modulus50,
    pub log_n: u32,
    /// The primitive 128th root of unity ζ (two-adicity 7).
    pub zeta: u64,
    /// Per-level per-block twiddles: `level_twiddles[d][i] =
    /// ζ^{2^{5-d}·(2·rev_d(i)+1)}` for DIF level d (0-based, blocks 2^d).
    level_twiddles: Vec<Vec<u64>>,
    /// Inverse twiddles.
    level_twiddles_inv: Vec<Vec<u64>>,
    /// Leaf slot constants `γ_p = ζ^{2^{7-k}·(2·rev_{k-1}(p)+1)}`
    /// (the quadratic-slot constants `X² ≡ γ_p`).
    pub leaf_gamma: Vec<u64>,
}

/// d-bit reversal of `x`.
fn rev_bits(x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for b in 0..bits {
        r |= ((x >> b) & 1) << (bits - 1 - b);
    }
    r
}

impl QuadSlotTables {
    /// Build tables for n = 2^log_n with log_n ∈ [1, 7] (the incomplete
    /// NTT regime for two-adicity 7).
    pub fn new(modulus: Modulus50, log_n: u32) -> Result<Self, Mod50Error> {
        if log_n == 0 || log_n > 7 {
            return Err(Mod50Error::LogNOutOfRange { got: log_n });
        }
        if modulus.two_adicity < 7 {
            return Err(Mod50Error::AdicityTooSmall {
                got: modulus.two_adicity,
            });
        }
        let zeta = modulus
            .root_of_unity(7)
            .ok_or(Mod50Error::AdicityTooSmall {
                got: modulus.two_adicity,
            })?;
        // Verify ζ is genuinely primitive (order exactly 128).
        if modulus.pow(zeta, 64) != modulus.q - 1 {
            return Err(Mod50Error::GeneratorInvalid);
        }
        let levels = (log_n - 1) as usize;
        let mut level_twiddles = Vec::with_capacity(levels);
        let mut level_twiddles_inv = Vec::with_capacity(levels);
        for d in 0..levels {
            // Twiddle exponent 2^{5-d}·(2·rev_d(i)+1); requires d ≤ 5
            // (log_n ≤ 7).
            let shift = 5u32
                .checked_sub(d as u32)
                .ok_or(Mod50Error::LogNOutOfRange { got: log_n })?;
            let blocks = 1usize << d;
            let mut fwd = Vec::with_capacity(blocks);
            let mut inv = Vec::with_capacity(blocks);
            for i in 0..blocks {
                let exp = ((2 * rev_bits(i, d as u32) as u64 + 1) << shift) % 128;
                let w = modulus.pow(zeta, exp);
                fwd.push(w);
                inv.push(modulus.inv(w).ok_or(Mod50Error::GeneratorInvalid)?);
            }
            level_twiddles.push(fwd);
            level_twiddles_inv.push(inv);
        }
        // Leaf constants: γ_p = ζ^{2^{7-k}·(2·rev_{k-1}(p)+1)}.
        let slots = 1usize << (log_n - 1);
        let leaf_shift = 7 - log_n;
        let mut leaf_gamma = Vec::with_capacity(slots);
        for p in 0..slots {
            let exp = ((2 * rev_bits(p, log_n - 1) as u64 + 1) << leaf_shift) % 128;
            leaf_gamma.push(modulus.pow(zeta, exp));
        }
        Ok(QuadSlotTables {
            modulus,
            log_n,
            zeta,
            level_twiddles,
            level_twiddles_inv,
            leaf_gamma,
        })
    }

    pub fn n(&self) -> usize {
        1usize << self.log_n
    }

    pub fn slots(&self) -> usize {
        self.n() / 2
    }

    /// Forward incomplete DIF: coefficients → quadratic-slot representation
    /// (in place). Block i at level d splits with twiddle w: lo' = lo + w·hi,
    /// hi' = lo - w·hi.
    pub fn forward(&self, a: &mut [u64]) -> Result<(), Mod50Error> {
        if a.len() != self.n() {
            return Err(Mod50Error::LengthMismatch {
                expected: self.n(),
                got: a.len(),
            });
        }
        let q = self.modulus;
        for d in 0..self.level_twiddles.len() {
            let span = self.n() >> d;
            let half = span / 2;
            for (i, w) in self.level_twiddles[d].iter().enumerate() {
                let start = i * span;
                for j in 0..half {
                    let lo = a[start + j];
                    let hi = a[start + j + half];
                    a[start + j] = q.add(lo, q.mul(*w, hi));
                    a[start + j + half] = q.sub(lo, q.mul(*w, hi));
                }
            }
        }
        Ok(())
    }

    /// Inverse of [`QuadSlotTables::forward`] (in place; every butterfly is
    /// exactly inverted, no global scaling).
    pub fn inverse(&self, a: &mut [u64]) -> Result<(), Mod50Error> {
        if a.len() != self.n() {
            return Err(Mod50Error::LengthMismatch {
                expected: self.n(),
                got: a.len(),
            });
        }
        let q = self.modulus;
        let inv2 = q.inv(2).ok_or(Mod50Error::GeneratorInvalid)?;
        for d in (0..self.level_twiddles.len()).rev() {
            let span = self.n() >> d;
            let half = span / 2;
            for (i, w_inv) in self.level_twiddles_inv[d].iter().enumerate() {
                let start = i * span;
                for j in 0..half {
                    let lo = a[start + j];
                    let hi = a[start + j + half];
                    // Forward: lo' = lo + w·hi, hi' = lo - w·hi.
                    // lo = (lo'+hi')/2, hi = (lo'-hi')/(2w).
                    let sum = q.add(lo, hi);
                    let diff = q.sub(lo, hi);
                    let half_diff = q.mul(diff, inv2);
                    a[start + j] = q.mul(sum, inv2);
                    a[start + j + half] = q.mul(half_diff, *w_inv);
                }
            }
        }
        Ok(())
    }

    /// Slotwise multiplication in the quadratic-slot representation:
    /// `(u0 + u1 X)(v0 + v1 X) mod (X² - γ)` per slot (Karatsuba, 3 muls).
    /// Inputs must already be in slot form (after [`Self::forward`]).
    pub fn slotwise_mul(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> Result<(), Mod50Error> {
        if a.len() != self.n() || b.len() != self.n() || out.len() != self.n() {
            return Err(Mod50Error::LengthMismatch {
                expected: self.n(),
                got: a.len(),
            });
        }
        let q = self.modulus;
        for p in 0..self.slots() {
            let (u0, u1) = (a[2 * p], a[2 * p + 1]);
            let (v0, v1) = (b[2 * p], b[2 * p + 1]);
            let gamma = self.leaf_gamma[p];
            // p0 = u0v0, p1 = u1v1, cross = (u0+u1)(v0+v1).
            let p0 = q.mul(u0, v0);
            let p1 = q.mul(u1, v1);
            let cross = q.mul(q.add(u0, u1), q.add(v0, v1));
            // c0 = u0v0 + γ·u1v1; c1 = cross - p0 - p1.
            out[2 * p] = q.add(p0, q.mul(gamma, p1));
            out[2 * p + 1] = q.sub(cross, q.add(p0, p1));
        }
        Ok(())
    }

    /// Full ring product in `Z_q[X]/(X^n + 1)` (forward both, slotwise
    /// Karatsuba, inverse).
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Mod50Error> {
        if a.len() != self.n() || b.len() != self.n() {
            return Err(Mod50Error::LengthMismatch {
                expected: self.n(),
                got: a.len(),
            });
        }
        let mut ta = a.to_vec();
        let mut tb = b.to_vec();
        self.forward(&mut ta)?;
        self.forward(&mut tb)?;
        let mut out = vec![0u64; self.n()];
        self.slotwise_mul(&ta, &tb, &mut out)?;
        self.inverse(&mut out)?;
        Ok(out)
    }
}

/// A ring configuration over a ~2^50 modulus with the incomplete-NTT
/// quadratic-slot arithmetic.
#[derive(Clone)]
pub struct RingConfig50 {
    pub modulus: Modulus50,
    pub log_n: u32,
    tables: Arc<QuadSlotTables>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mod50Error {
    LogNOutOfRange { got: u32 },
    AdicityTooSmall { got: u32 },
    GeneratorInvalid,
    LengthMismatch { expected: usize, got: usize },
}

impl RingConfig50 {
    /// Create with the canonical modulus and n = 2^log_n, log_n ∈ [1, 7].
    pub fn new(log_n: u32) -> Result<Self, Mod50Error> {
        Self::with_modulus(Modulus50::Q_50, log_n)
    }

    /// Create with a custom two-adicity-7 modulus.
    pub fn with_modulus(modulus: Modulus50, log_n: u32) -> Result<Self, Mod50Error> {
        let tables = QuadSlotTables::new(modulus, log_n)?;
        Ok(RingConfig50 {
            modulus,
            log_n,
            tables: Arc::new(tables),
        })
    }

    pub fn n(&self) -> usize {
        1usize << self.log_n
    }

    /// The quadratic-slot leaf constants (paper-facing slot view).
    pub fn slot_constants(&self) -> &[u64] {
        &self.tables.leaf_gamma
    }

    /// Ring product.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Mod50Error> {
        self.tables.mul(a, b)
    }

    /// Pointwise addition.
    pub fn add(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Mod50Error> {
        if a.len() != self.n() || b.len() != self.n() {
            return Err(Mod50Error::LengthMismatch {
                expected: self.n(),
                got: a.len(),
            });
        }
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| self.modulus.add(*x, *y))
            .collect())
    }

    /// Pointwise subtraction.
    pub fn sub(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Mod50Error> {
        if a.len() != self.n() || b.len() != self.n() {
            return Err(Mod50Error::LengthMismatch {
                expected: self.n(),
                got: a.len(),
            });
        }
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| self.modulus.sub(*x, *y))
            .collect())
    }

    /// From signed (balanced) coefficients.
    pub fn from_signed(&self, signed: &[i64]) -> Vec<u64> {
        let mut out = vec![0u64; self.n()];
        for (i, s) in signed.iter().take(self.n()).enumerate() {
            out[i] = self.modulus.reduce_i64(*s);
        }
        out
    }

    /// Infinity norm of the balanced representative.
    pub fn infinity_norm(&self, a: &[u64]) -> u64 {
        let half = self.modulus.q / 2;
        a.iter()
            .map(|c| if *c <= half { *c } else { self.modulus.q - *c })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Schoolbook negacyclic convolution mod q (the reference oracle).
    #[allow(clippy::needless_range_loop)] // index-mapped convolution by definition
    fn schoolbook_negacyclic(q: u64, a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len();
        let mut acc = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let prod = (a[i] as i128) * (b[j] as i128);
                let k = i + j;
                if k < n {
                    acc[k] += prod;
                } else {
                    acc[k - n] -= prod; // X^n ≡ -1
                }
            }
        }
        acc.iter().map(|c| c.rem_euclid(q as i128) as u64).collect()
    }

    fn random_vec(q: u64, n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            out.push((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 20) % q);
        }
        out
    }

    #[test]
    fn modulus_is_the_paper_prime() {
        let m = Modulus50::Q_50;
        assert_eq!(m.q, Q_50);
        assert_eq!(m.q % 256, 129, "q ≡ 129 mod 256 (the e=2 slot condition)");
        assert_eq!(m.two_adicity, 7);
        // 2-adicity check: (q-1)/2^7 is odd.
        assert_eq!(((m.q - 1) >> 7) & 1, 1);
        // ζ = root_of_unity(7) has order exactly 128.
        let zeta = match m.root_of_unity(7) {
            Some(z) => z,
            None => return,
        };
        assert_eq!(m.pow(zeta, 128), 1);
        assert_eq!(m.pow(zeta, 64), m.q - 1);
        // Primality (deterministic Miller-Rabin for u64 ranges).
        assert!(is_prime_u64(m.q), "q must be prime");
    }

    fn is_prime_u64(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if n % p == 0 {
                return n == p;
            }
        }
        let mut d = n - 1;
        let mut r = 0u32;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }
        for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            let mut x = 1u64;
            let mut e = d;
            let mut base = a % n;
            while e > 0 {
                if e & 1 == 1 {
                    x = (x as u128 * base as u128 % n as u128) as u64;
                }
                base = (base as u128 * base as u128 % n as u128) as u64;
                e >>= 1;
            }
            if x == 1 || x == n - 1 {
                continue;
            }
            let mut composite = true;
            for _ in 0..r.saturating_sub(1) {
                x = (x as u128 * x as u128 % n as u128) as u64;
                if x == n - 1 {
                    composite = false;
                    break;
                }
            }
            if composite {
                return false;
            }
        }
        true
    }

    #[test]
    fn barrett_matches_division() {
        let m = Modulus50::Q_50;
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        // Full-product range: u128 < 2^100 (products of residues).
        for _ in 0..200_000 {
            let a = next() % m.q;
            let b = next() % m.q;
            let x = u128::from(a) * u128::from(b);
            let expected = (x % m.q as u128) as u64;
            assert_eq!(m.reduce_u128(x), expected, "x = {x}");
            assert_eq!(m.mul(a, b), expected);
        }
        // Boundary values.
        for x in [
            0u128,
            1,
            u128::from(m.q - 1),
            u128::from(m.q),
            u128::from(m.q + 1),
            (1u128 << 50) - 1,
            1u128 << 50,
            (1u128 << 62) - 1,
            1u128 << 62,
            (1u128 << 99) - 1,
            u128::from(m.q - 1) * u128::from(m.q - 1),
        ] {
            let expected = (x % m.q as u128) as u64;
            assert_eq!(m.reduce_u128(x), expected, "x = {x}");
        }
    }

    #[test]
    fn roundtrip_all_log_n() {
        for log_n in 1..=7u32 {
            let ring = RingConfig50::new(log_n).ok().unwrap();
            let a = random_vec(ring.modulus.q, ring.n(), 0xABCD + log_n as u64);
            let mut t = a.clone();
            ring.tables.forward(&mut t).ok().unwrap();
            ring.tables.inverse(&mut t).ok().unwrap();
            assert_eq!(t, a, "roundtrip failed at log_n = {log_n}");
        }
    }

    #[test]
    fn mul_matches_schoolbook_all_log_n() {
        for log_n in [1u32, 2, 3, 5, 7] {
            let ring = RingConfig50::new(log_n).ok().unwrap();
            for trial in 0..4 {
                let a = random_vec(ring.modulus.q, ring.n(), 0x1111 + trial * 7 + log_n as u64);
                let b = random_vec(ring.modulus.q, ring.n(), 0x2222 + trial * 13 + log_n as u64);
                let got = ring.mul(&a, &b).ok().unwrap();
                let want = schoolbook_negacyclic(ring.modulus.q, &a, &b);
                assert_eq!(got, want, "log_n={log_n} trial={trial}");
            }
        }
    }

    #[test]
    fn mul_ring_axioms() {
        let ring = RingConfig50::new(7).ok().unwrap();
        let q = ring.modulus;
        let a = random_vec(q.q, ring.n(), 1);
        let b = random_vec(q.q, ring.n(), 2);
        let c = random_vec(q.q, ring.n(), 3);
        // Commutative.
        assert_eq!(
            ring.mul(&a, &b).ok().unwrap(),
            ring.mul(&b, &a).ok().unwrap()
        );
        // Associative.
        assert_eq!(
            ring.mul(&ring.mul(&a, &b).ok().unwrap(), &c).ok().unwrap(),
            ring.mul(&a, &ring.mul(&b, &c).ok().unwrap()).ok().unwrap()
        );
        // Distributive.
        let a_plus_b = ring.add(&a, &b).ok().unwrap();
        assert_eq!(
            ring.mul(&a_plus_b, &c).ok().unwrap(),
            ring.add(
                &ring.mul(&a, &c).ok().unwrap(),
                &ring.mul(&b, &c).ok().unwrap()
            )
            .ok()
            .unwrap()
        );
        // Identity and annihilator.
        let one = {
            let mut v = vec![0u64; ring.n()];
            v[0] = 1;
            v
        };
        assert_eq!(ring.mul(&a, &one).ok().unwrap(), a);
        assert!(ring
            .mul(&a, &vec![0u64; ring.n()])
            .ok()
            .unwrap()
            .iter()
            .all(|c| *c == 0));
        // X · X^{n-1} = X^n = -1.
        let mut x = vec![0u64; ring.n()];
        x[1] = 1;
        let mut x_nm1 = vec![0u64; ring.n()];
        x_nm1[ring.n() - 1] = 1;
        let prod = ring.mul(&x, &x_nm1).ok().unwrap();
        assert_eq!(prod[0], q.q - 1);
        assert!(prod[1..].iter().all(|c| *c == 0));
    }

    #[test]
    fn slot_constants_have_the_paper_structure() {
        // At log_n = 7 (n = 128): γ_p = ζ^{2p+1} — the 64 quadratic slot
        // constants with γ^64 = -1 (odd powers of the primitive 128th root).
        let ring = RingConfig50::new(7).ok().unwrap();
        let q = ring.modulus;
        assert_eq!(ring.slot_constants().len(), 64);
        for (p, gamma) in ring.slot_constants().iter().enumerate() {
            assert_ne!(*gamma, 0, "slot {p}");
            assert_eq!(q.pow(*gamma, 64), q.q - 1, "slot {p}: γ^64 must be -1");
        }
        // At log_n = 6 (n = 64): γ_p = (ζ²)^{2p+1} — odd powers of the
        // 64th root; γ^32 = -1.
        let ring6 = RingConfig50::new(6).ok().unwrap();
        let q6 = ring6.modulus;
        assert_eq!(ring6.slot_constants().len(), 32);
        for gamma in ring6.slot_constants() {
            assert_eq!(q6.pow(*gamma, 32), q6.q - 1);
        }
        // Leaf constants are distinct (a valid factorization).
        let mut sorted = ring.slot_constants().to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 64);
    }

    #[test]
    fn errors_rejected() {
        assert!(matches!(
            RingConfig50::new(0),
            Err(Mod50Error::LogNOutOfRange { got: 0 })
        ));
        assert!(matches!(
            RingConfig50::new(8),
            Err(Mod50Error::LogNOutOfRange { got: 8 })
        ));
        // A modulus with insufficient two-adicity cannot host the tree.
        let bad = Modulus50::new(Q_50, 5, Q_50_GENERATOR);
        assert!(matches!(
            RingConfig50::with_modulus(bad, 3),
            Err(Mod50Error::AdicityTooSmall { .. })
        ));
        // Length mismatch in mul.
        let ring = RingConfig50::new(3).ok().unwrap();
        assert!(matches!(
            ring.mul(&[1, 2, 3], &[1, 2, 3]),
            Err(Mod50Error::LengthMismatch { .. })
        ));
    }

    #[test]
    fn signed_ingestion_and_norm() {
        let ring = RingConfig50::new(4).ok().unwrap();
        let v = ring.from_signed(&[-3, -2, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5]);
        assert_eq!(ring.infinity_norm(&v), 5);
        // -3 maps to q-3.
        assert_eq!(v[0], ring.modulus.q - 3);
    }
}
