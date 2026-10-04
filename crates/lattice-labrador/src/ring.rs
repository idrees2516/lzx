//! Ring arithmetic over `Z_Q[X]/(X^64 + 1)`, `Q = 2^48 - 59`: exact centered i64 coefficients
//! with i128 accumulation, plus the sampling distributions and digit decomposition.

/// Ring degree.
pub const N: usize = 64;
/// The modulus `2^48 - 59`.
pub const Q: i128 = (1i128 << 48) - 59;
/// `Q` as i64 (fits: `2^48 - 59 < 2^63`).
pub const Q64: i64 = ((1u64 << 48) - 59) as i64;
/// `-1/Q mod 2^64` is unnecessary; we use i128 division for exactness.
pub const Q_INV: i128 = 1;

/// One ring element: 64 centered coefficients, each in `[-(Q-1)/2, (Q-1)/2]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Poly(pub [i64; N]);

impl Default for Poly {
    fn default() -> Self {
        Self([0; N])
    }
}

#[inline]
pub fn cmod(x: i128) -> i64 {
    // Barrett by construction of Q: 2^48 = Q + 59, so 2^48 ≡ 59 (mod Q) and every
    // `v = hi * 2^48 + lo` folds to `hi * 59 + lo` exactly. Four folds bring any i128
    // (the negacyclic accumulator's worst case is m products of |a b| <= (Q/2)^2 < 2^94)
    // below 2^48, and the arithmetic shifts make the fold exact for negative x as well
    // (`x = (x >> 48) * 2^48 + (x & MASK48)` in two's complement). Against `rem_euclid`
    // this trades a ~40-cycle i128 division for ~10 shift/multiply uops; the exhaustive
    // test in tests/ drives every reachable magnitude against the division form.
    const MASK48: i128 = (1i128 << 48) - 1;
    let mut v = (x >> 48) * 59 + (x & MASK48);
    v = (v >> 48) * 59 + (v & MASK48);
    v = (v >> 48) * 59 + (v & MASK48);
    v = (v >> 48) * 59 + (v & MASK48);
    // centre: at most a couple of conditional moves now
    let half = Q / 2;
    while v > half {
        v -= Q;
    }
    while v < -half {
        v += Q;
    }
    v as i64
}

/// The division-based reference the folded form is tested against.
#[inline]
pub fn cmod_div(x: i128) -> i64 {
    let r = x.rem_euclid(Q);
    if r > Q / 2 {
        (r - Q) as i64
    } else {
        r as i64
    }
}

impl Poly {
    pub fn zero() -> Self {
        Self([0; N])
    }
    pub fn one() -> Self {
        let mut p = Self::zero();
        p.0[0] = 1;
        p
    }
    /// The constant `v mod Q`, centered.
    pub fn constant(v: i64) -> Self {
        let mut p = Self::zero();
        p.0[0] = cmod(v as i128);
        p
    }
    /// The monomial `v X^k mod Q`, centered.
    pub fn monomial(v: i64, k: usize) -> Self {
        let mut p = Self::zero();
        p.0[k % N] = cmod(v as i128);
        p
    }
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&x| x == 0)
    }
    pub fn neg(&self) -> Self {
        Self(core::array::from_fn(|i| -self.0[i]))
    }
    pub fn add(&self, o: &Self) -> Self {
        Self(core::array::from_fn(|i| {
            cmod(self.0[i] as i128 + o.0[i] as i128)
        }))
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self(core::array::from_fn(|i| {
            cmod(self.0[i] as i128 - o.0[i] as i128)
        }))
    }
    pub fn add_assign(&mut self, o: &Self) {
        *self = self.add(o);
    }
    pub fn sub_assign(&mut self, o: &Self) {
        *self = self.sub(o);
    }
    /// Negacyclic product `a * b mod (X^64 + 1, Q)`, exact i128 accumulation — dispatched
    /// through the AVX-512 split-2^24 convolution ([`crate::conv::negacyclic_mul`]) when the
    /// CPU has the feature set; bit-identical either way.
    pub fn mul(&self, o: &Self) -> Self {
        crate::conv::negacyclic_mul(self, o)
    }
    /// The scalar i128 schoolbook reference (the specification the vectorized convolution
    /// is verified against).
    pub fn mul_schoolbook(&self, o: &Self) -> Self {
        let mut acc = [0i128; N];
        for i in 0..N {
            if self.0[i] == 0 {
                continue;
            }
            let a = self.0[i] as i128;
            for j in 0..N {
                let t = a * o.0[j] as i128;
                let k = i + j;
                if k < N {
                    acc[k] += t;
                } else {
                    // X^64 = -1
                    acc[k - N] -= t;
                }
            }
        }
        Self(core::array::from_fn(|i| cmod(acc[i])))
    }
    /// `self + v * o`.
    pub fn mul_add(&self, v: i64, o: &Self) -> Self {
        self.add(&o.scale(v))
    }
    pub fn scale(&self, v: i64) -> Self {
        Self(core::array::from_fn(|i| {
            cmod(self.0[i] as i128 * v as i128)
        }))
    }
    /// `sigma_{-1}`: the ring automorphism `X -> -X` (coefficient i negated for odd i).
    pub fn sigma_m1(&self) -> Self {
        Self(core::array::from_fn(|i| {
            if i.is_multiple_of(2) {
                self.0[i]
            } else {
                -self.0[i]
            }
        }))
    }
    /// The automorphism `X -> X^5` (5 is coprime to 128).
    pub fn sigma5(&self) -> Self {
        let mut out = [0i64; N];
        for i in 0..N {
            // X^i -> X^{5i mod 128}, negated when 5i >= 64 mod 128 in the negacyclic wrap
            let e = (5 * i) % 128;
            if e < N {
                out[e] = self.0[i];
            } else {
                out[e - N] = -self.0[i];
            }
        }
        Self(out)
    }
    /// The conjugate `X -> X^{-1}` = `X^{127}` map ("flip"): coefficient i of the image is the
    /// coefficient of `X^{128 - i}`... upstream `polx_flip` maps `a(X) -> a(X^{-1})`.
    pub fn flip(&self) -> Self {
        let mut out = [0i64; N];
        out[0] = self.0[0];
        for i in 1..N {
            // X^{-i} = X^{128-i} = -X^{64-i} for i in 1..64
            out[N - i] = -self.0[i];
        }
        Self(out)
    }
    /// The inner product `sum_i a_i b_i` (negacyclic product then constant term shortcut) —
    /// dispatched through the AVX-512 split convolution when available; bit-identical.
    pub fn sprod(a: &[Poly], b: &[Poly]) -> Poly {
        crate::conv::negacyclic_sprod(a, b)
    }

    /// The scalar i128 schoolbook sprod reference.
    pub fn sprod_schoolbook(a: &[Poly], b: &[Poly]) -> Poly {
        let mut acc = [0i128; N];
        for k in 0..a.len() {
            for i in 0..N {
                if a[k].0[i] == 0 {
                    continue;
                }
                let av = a[k].0[i] as i128;
                for j in 0..N {
                    let t = av * b[k].0[j] as i128;
                    let e = i + j;
                    if e < N {
                        acc[e] += t;
                    } else {
                        acc[e - N] -= t;
                    }
                }
            }
        }
        Self(core::array::from_fn(|i| cmod(acc[i])))
    }
    /// From centered i16 coefficient slices (one ring element per 64).
    pub fn from_i16(c: &[i16]) -> Self {
        let mut p = [0i64; N];
        for i in 0..N {
            p[i] = c[i] as i64;
        }
        Self(p)
    }
    /// 64 coefficients x 6 bytes, little-endian.
    pub fn to_le_bytes(self) -> Vec<u8> {
        let mut out = vec![0u8; 6 * N];
        for (i, &x) in self.0.iter().enumerate() {
            out[6 * i..6 * i + 6]
                .copy_from_slice(&(x as u64 & 0xFFFF_FFFF_FFFF).to_le_bytes()[..6]);
        }
        out
    }
}

/// `2^t` signed-digit decomposition of a centered mod-Q element into `t` digit ring elements:
/// digit `j < t-1` balanced in `[-2^{d-1}, 2^{d-1})`, the top digit the exact remainder.
/// Reconstruct: `x = sum_j 2^{j d} digit_j` (upstream `polz_decompose`).
pub fn decompose(a: &Poly, t: usize, d: u32) -> Vec<Poly> {
    let mut digits = vec![[0i64; N]; t];
    let dd = 1i128 << d;
    for i in 0..N {
        let mut r = a.0[i] as i128;
        for j in 0..t.saturating_sub(1) {
            let mut dig = r % dd;
            if dig > dd / 2 - 1 {
                dig -= dd;
            } else if dig < -(dd / 2) {
                dig += dd;
            }
            digits[j][i] = dig as i64;
            r = (r - dig) / dd;
        }
        if t > 0 {
            digits[t - 1][i] = r as i64;
        }
    }
    digits.into_iter().map(Poly).collect()
}

/// Inverse of [`decompose`]: `sum_j 2^{j d} digit_j`, centered mod Q.
pub fn reconstruct(digits: &[Poly], d: u32) -> Poly {
    let mut out = Poly::zero();
    for (j, dig) in digits.iter().enumerate() {
        let w = 1i64 << (d * j as u32);
        for i in 0..N {
            out.0[i] = cmod(out.0[i] as i128 + dig.0[i] as i128 * w as i128);
        }
    }
    out
}

/// Uniform mod-Q ring elements from 48-bit chunks of a SHAKE stream ("almostuniform").
pub fn uniform(len: usize, seed: &[u8; 32], nonce: u64) -> Vec<Poly> {
    let mut h = lattice_core::keccak::KeccakSponge::new_shake256();
    h.update(b"labrador/uniform/v1");
    h.update(seed);
    h.update(&nonce.to_le_bytes());
    h.finalize_in_place();
    let need = len * 48 / 8 + 8;
    let mut bytes = vec![0u8; need];
    h.squeeze(&mut bytes);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut limbs = [0u8; 6];
        limbs.copy_from_slice(&bytes[6 * i..6 * i + 6]);
        let v = u64::from_le_bytes([
            limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5], 0, 0,
        ]);
        out.push(Poly::constant(v as i64));
    }
    out
}

/// Quarternary coefficients in `{-1, 0, 1, ...}`? Upstream: 2 bits per coefficient giving
/// `{-2, -1, 0, 1}`-style small values; we sample `{-1, 0, 1}` ternary from 2 bits: values
/// `00 -> 0, 01 -> 1, 10 -> -1, 11 -> 0`.
pub fn quarternary(len: usize, seed: &[u8; 32], nonce: u64) -> Vec<Poly> {
    small_poly(len, seed, nonce, b"labrador/quarternary/v1")
}

/// The challenge distribution: ternary with fixed weight (upstream `polyvec_challenge`):
/// 32 nonzero `+-1` positions out of 64 per element.
pub fn challenge(len: usize, seed: &[u8; 32], nonce: u64) -> Vec<Poly> {
    small_challenge(len, seed, nonce)
}

fn small_poly(len: usize, seed: &[u8; 32], nonce: u64, label: &[u8]) -> Vec<Poly> {
    let mut h = lattice_core::keccak::KeccakSponge::new_shake256();
    h.update(label);
    h.update(seed);
    h.update(&nonce.to_le_bytes());
    h.finalize_in_place();
    let mut bytes = vec![0u8; len * 64 * 2];
    h.squeeze(&mut bytes);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut p = [0i64; N];
        for j in 0..N {
            let b = bytes[(i * 64 + j) * 2];
            p[j] = match b & 3 {
                1 => 1,
                2 => -1,
                _ => 0,
            };
        }
        out.push(Poly(p));
    }
    out
}

fn small_challenge(len: usize, seed: &[u8; 32], nonce: u64) -> Vec<Poly> {
    let mut h = lattice_core::keccak::KeccakSponge::new_shake256();
    h.update(b"labrador/challenge/v1");
    h.update(seed);
    h.update(&nonce.to_le_bytes());
    h.finalize_in_place();
    let mut bytes = vec![0u8; len * 256];
    h.squeeze(&mut bytes);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut p = [0i64; N];
        // 32 positions from the stream, Fisher-Yates over 64
        let mut perm: [u8; 64] = core::array::from_fn(|k| k as u8);
        for k in 0..32 {
            let b = bytes[i * 256 + k * 2] as usize % (64 - k);
            perm.swap(k, k + b);
        }
        for k in 0..32 {
            let pos = perm[k] as usize % 64;
            let sign = if bytes[i * 256 + 64 + k] & 1 == 0 {
                1
            } else {
                -1
            };
            p[pos] = sign;
        }
        out.push(Poly(p));
    }
    out
}

/// The JL projection sign matrix row k for polynomial p: 64 sign bits from the stream.
pub fn jl_signs(bytes: &[u8], k: usize) -> u64 {
    let mut v = 0u64;
    for b in 0..8 {
        v |= (bytes[k * 8 + b] as u64) << (8 * b);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmod_fold_matches_division() {
        // every reachable magnitude class: sums of m products of centered coefficients
        let mut mag = 1i128;
        while mag <= (1i128 << 120) {
            for &sign in &[1i128, -1] {
                for &k in &[1i128, 59, mag - 1, mag / 2, mag / 2 + 1, mag - 59, mag - 60] {
                    let v = sign * k;
                    assert_eq!(cmod(v), cmod_div(v), "v={v}");
                    assert_eq!(cmod(-v), cmod_div(-v), "v=-{v}");
                }
            }
            if mag > (1i128 << 111) {
                break; // the next <<= 8 would overflow i128
            }
            mag <<= 8;
        }
        // pseudo-random sweep across the reachable range
        let mut r = 0x9E37_79B9_7F4A_7C15i128;
        for _ in 0..100_000 {
            r = r
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (r >> 7) % (1i128 << 100);
            assert_eq!(cmod(v), cmod_div(v), "v={v}");
            assert_eq!(cmod(-v), cmod_div(-v), "v=-{v}");
        }
        // straddles around multiples of Q
        let q = Q;
        for k in 1..2000i128 {
            for d in [0i128, 1, 29, 30, 59, 60, q / 2, q / 2 + 1, q - 1] {
                let v = k * q + d;
                assert_eq!(cmod(v), cmod_div(v), "v={v}");
                assert_eq!(cmod(-v), cmod_div(-v), "v=-{v}");
            }
        }
        // extremes
        assert_eq!(cmod(i128::MAX), cmod_div(i128::MAX));
        assert_eq!(cmod(i128::MIN), cmod_div(i128::MIN));
        assert_eq!(cmod(0), 0);
    }
}
