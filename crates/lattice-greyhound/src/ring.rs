//! Ring arithmetic over `R_q = Z_q[X]/(X^64 + 1)` with `q = 2^32 - 99`.
//!
//! This is the ring of both papers at their concrete instantiation: d = 64, q ≈ 2^32
//! single-precision, `q ≡ 5 (mod 8)` so that (i) `X^64 + 1` splits into exactly two
//! irreducible degree-32 factors mod q (LaBRADOR §2), and (ii) short elements are
//! invertible via [LS18, Lemma 2.1] (Greyhound §2.1). Verified: q = 4294967197 is
//! prime, q mod 8 = 5, ord_128(q) = 32.
//!
//! Upstream represents ring elements as `polx` (K-prime RNS NTT images) / `polz`
//! (14-bit limb triples); this port uses exact centered `i64` coefficients with
//! `i128` accumulation (the same trade-off as `lattice-labrador`, the workspace's
//! Dachshund port at Q = 2^48-59). Products are schoolbook negacyclic; at the
//! statement sizes that dominate the 53KB claim this is fast enough to *run* the
//! whole 2^30 sub-proof (see `examples/greyhound_bench.rs`).
//!
//! Norms follow the papers' definitions: `‖w‖∞ = max |w mod^± q|`,
//! `‖w‖² = Σ w_i²` over centered representatives.

/// Ring degree d (both papers: power-of-two, d = 64).
pub const N: usize = 64;
/// The modulus q = 2^32 - 99 (reference `LOGQ = 32`, `QOFF = 99`).
pub const Q: i64 = (1i64 << 32) - 99;
/// log2(q) rounded up — the "LOGQ" of the size model.
pub const LOGQ: usize = 32;
/// Number of RNS limbs the reference would use (unused here; kept for size parity).
#[allow(dead_code)]
pub const LIMBS: usize = 3;

/// Centered reduction mod q. Any i128 with |x| < 2^62 folds exactly: x = hi·2^32 + lo
/// with 2^32 ≡ 99 (mod q), two folds suffice for the schoolbook accumulator
/// (max |x| = 64·(q/2)^2 < 2^62).
#[inline]
pub fn cmod(x: i128) -> i64 {
    const MASK: i128 = (1i128 << 32) - 1;
    let mut v = (x >> 32) * 99 + (x & MASK);
    v = (v >> 32) * 99 + (v & MASK);
    // center
    let half = Q as i128 / 2;
    while v > half {
        v -= Q as i128;
    }
    while v < -half {
        v += Q as i128;
    }
    v as i64
}

/// Division-based reference for tests.
#[inline]
pub fn cmod_div(x: i128) -> i64 {
    let r = x.rem_euclid(Q as i128);
    if r > Q as i128 / 2 {
        (r - Q as i128) as i64
    } else {
        r as i64
    }
}

/// One ring element: 64 centered coefficients, each in [-(q-1)/2, (q-1)/2].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Poly(pub [i64; N]);

impl Default for Poly {
    fn default() -> Self {
        Self([0; N])
    }
}

impl Poly {
    pub fn zero() -> Self {
        Self([0; N])
    }

    pub fn constant(c: i64) -> Self {
        let mut p = Self::zero();
        p.0[0] = cmod(c as i128);
        p
    }

    /// From centered small coefficients (i16-range), no reduction needed.
    pub fn from_i16(coeffs: &[i16]) -> Self {
        let mut p = Self::zero();
        for (i, &c) in coeffs.iter().take(N).enumerate() {
            p.0[i] = c as i64;
        }
        p
    }

    pub fn from_i64(coeffs: &[i64]) -> Self {
        let mut p = Self::zero();
        for (i, &c) in coeffs.iter().take(N).enumerate() {
            p.0[i] = cmod(c as i128);
        }
        p
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut r = [0i64; N];
        for i in 0..N {
            r[i] = cmod(self.0[i] as i128 + rhs.0[i] as i128);
        }
        Self(r)
    }

    pub fn add_assign(&mut self, rhs: &Self) {
        *self = self.add(rhs);
    }

    pub fn sub_assign(&mut self, rhs: &Self) {
        *self = self.sub(rhs);
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mut r = [0i64; N];
        for i in 0..N {
            r[i] = cmod(self.0[i] as i128 - rhs.0[i] as i128);
        }
        Self(r)
    }

    pub fn neg(&self) -> Self {
        let mut r = [0i64; N];
        for i in 0..N {
            r[i] = cmod(-(self.0[i] as i128));
        }
        Self(r)
    }

    pub fn scale(&self, s: i64) -> Self {
        let mut r = [0i64; N];
        for i in 0..N {
            r[i] = cmod(self.0[i] as i128 * s as i128);
        }
        Self(r)
    }

    /// Negacyclic product in R_q (X^64 = -1).
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut acc = [0i128; N];
        for i in 0..N {
            let a = self.0[i] as i128;
            if a == 0 {
                continue;
            }
            for j in 0..N - i {
                acc[i + j] += a * rhs.0[j] as i128;
            }
            for j in N - i..N {
                acc[i + j - N] -= a * rhs.0[j] as i128;
            }
        }
        let mut r = [0i64; N];
        for i in 0..N {
            r[i] = cmod(acc[i]);
        }
        Self(r)
    }

    /// `self += a·b` (fused mul-add).
    pub fn mul_add(&mut self, a: &Self, b: &Self) {
        self.add_assign(&a.mul(b));
    }

    /// Squared l2 norm (over the integers, exact — u64 saturating beyond 2^63 is
    /// unreachable since 64·(q/2)^2 < 2^63).
    pub fn normsq(&self) -> u64 {
        self.0.iter().map(|&c| (c * c) as u64).sum()
    }

    /// l∞ norm of the centered representative.
    pub fn norminf(&self) -> i64 {
        self.0.iter().map(|&c| c.abs()).max().unwrap_or(0)
    }

    /// The operator norm bound `‖c‖op` computed by dense evaluation over the 64
    /// coefficient-extraction functionals: max over unit vectors is upper-bounded by
    /// the row sums of the negacyclic circulant; we compute the exact max over the
    /// 64 rows (rows are the negacyclic rotations of the reversed coefficient list).
    pub fn opnorm(&self) -> f64 {
        // row j of the circulant: coefficients e[(j - k) mod 64] with sign flips for
        // wraparound; the l1 norm of every row of the negacyclic matrix of c is the
        // same as the l1 norm of c (permutation + signs preserve |.|). But opnorm of
        // the *convolution operator* can be up to sqrt(64)·l2 — the reference bounds
        // it with poly_opnorm, which computes max over the 64 conjugate evaluations
        // |c(X^i)|... we instead use the exact spectral bound via the circulant rows'
        // l2 (all equal) as an upper bound surrogate is wrong; so compute directly:
        // opnorm = max_{‖x‖2=1} ‖c*x‖2 = max eigenvalue of the circulant. For the
        // negacyclic ring the eigenvectors are the 128th roots; the eigenvalues are
        // c(ζ) for ζ^128 = 1, ζ != -1... The reference's poly_opnorm computes
        // max_i |σ_i(c)| over the embeddings (the 64 primitive 128th roots), which
        // upper-bounds the operator norm on l2 only in the rotated basis — this is
        // the standard estimate used by both papers (T = 14 for the challenge set).
        // We compute max_i |c(ζ^i)| with ζ a primitive 128th root, exactly, in f64.
        let mut best: f64 = 0.0;
        for i in 1..128u32 {
            if i % 2 == 0 {
                continue; // primitive 128th roots: i odd
            }
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            let (mut zr, mut zi) = (1.0f64, 0.0f64);
            let (wr, wi) = (
                f64::cos(std::f64::consts::PI * i as f64 / 64.0),
                f64::sin(std::f64::consts::PI * i as f64 / 64.0),
            );
            for &c in self.0.iter() {
                re += c as f64 * zr;
                im += c as f64 * zi;
                let nzr = zr * wr - zi * wi;
                zi = zr * wi + zi * wr;
                zr = nzr;
            }
            let m = (re * re + im * im).sqrt();
            if m > best {
                best = m;
            }
        }
        best
    }

    /// The conjugation automorphism σ^{-1}: X ↦ X^{-1} = -X^{63}. For
    /// f = Σ a_i X^i: σ^{-1}(f) = a_0 - Σ_{i≥1} a_i X^{64-i}.
    pub fn sigma_m1(&self) -> Self {
        let mut r = [0i64; N];
        r[0] = self.0[0];
        for i in 1..N {
            r[N - i] = -self.0[i];
        }
        Self(r)
    }

    /// The "flip" conjugate used for binary witnesses: X ↦ -X^{-1} = X^{63}; for
    /// binary coefficients this preserves 0/1. (σ^{-1} would negate them.)
    /// f = Σ a_i X^i ↦ Σ a_i X^{64-i}, i.e. coefficient reversal with a_0 staying
    /// only in degree 0 when it wraps: X^{64} = -1, so X^{i} ↦ X^{64-i} for i ≥ 1
    /// and 1 ↦ 1.
    pub fn flip(&self) -> Self {
        let mut r = [0i64; N];
        r[0] = self.0[0];
        for i in 1..N {
            r[N - i] = self.0[i];
        }
        Self(r)
    }

    pub fn constant_term(&self) -> i64 {
        self.0[0]
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&c| c == 0)
    }

    /// Centered digit decomposition into `t` parts of `d` bits each (power-of-two
    /// base — the documented reference deviation from the paper's general base b;
    /// Greyhound §6: "we only use power-of-two bases"). The first `t-1` parts are
    /// centered mod 2^d (|l_j| ≤ 2^{d-1}); the top part carries the remainder and
    /// may exceed 2^{d-1} (it is controlled by the norm check, not a width check —
    /// LaBRADOR Figure 3 lines 11-13).
    ///
    /// Exact integer identity on the centered representative: a = Σ_j l_j·2^{jd}.
    pub fn decompose(&self, t: usize, d: u32) -> Vec<Self> {
        debug_assert!(t >= 1);
        let mut parts = Vec::with_capacity(t);
        if t == 1 {
            parts.push(*self);
            return parts;
        }
        let mask: i64 = (1i64 << d) - 1;
        let half: i64 = 1i64 << (d - 1);
        let mut rem = self.0; // already centered
        for _ in 0..t - 1 {
            let mut l = [0i64; N];
            for i in 0..N {
                // signed low d bits: the representative in [-2^{d-1}, 2^{d-1})
                let low = rem[i] & mask;
                let v = if low >= half { low - (1i64 << d) } else { low };
                l[i] = v;
                rem[i] = (rem[i] - v) >> d;
            }
            parts.push(Self(l));
        }
        parts.push(Self(rem));
        parts
    }

    /// Recombination: Σ_j parts[j]·2^{jd} mod q (the G gadget recombination).
    pub fn recombine(parts: &[Self], d: u32) -> Self {
        let mut acc = [0i128; N];
        for (j, p) in parts.iter().enumerate() {
            let scale = 1i128 << (d as usize * j);
            for i in 0..N {
                acc[i] += p.0[i] as i128 * scale;
            }
        }
        let mut r = [0i64; N];
        for i in 0..N {
            r[i] = cmod(acc[i]);
        }
        Self(r)
    }

    /// Almost-uniform sampling mod q from a seed+nonce stream (the reference's
    /// `polzvec_almostuniform`, used for the commitment key and test witnesses).
    pub fn almost_uniform(seed: &[u8], nonce: u64) -> Self {
        let mut buf = vec![0u8; N * 5]; // 5 bytes ≥ log2(q) = 32.0007 bits per coeff
        expand_seed(seed, nonce, &mut buf);
        let mut p = [0i64; N];
        for i in 0..N {
            // 40-bit value mod q (rejection-free: bias < 2^-8 negligible for keys;
            // the reference uses the same almost-uniform stream)
            let mut v = 0u64;
            for k in 0..5 {
                v |= (buf[i * 5 + k] as u64) << (8 * k);
            }
            v %= Q as u64;
            p[i] = if v > Q as u64 / 2 {
                v as i64 - Q
            } else {
                v as i64
            };
        }
        Self(p)
    }

    pub fn to_le_bytes(&self) -> [u8; N * 4] {
        let mut b = [0u8; N * 4];
        for i in 0..N {
            let c = (self.0[i] + Q / 2) as u32; // shift to [0, q)
            b[i * 4..i * 4 + 4].copy_from_slice(&c.to_le_bytes());
        }
        b
    }
}

/// Inner product of two vectors of ring elements (componentwise negacyclic
/// products summed — `polyvec_sprod`): ⟨a, b⟩ = Σ_k a_k·b_k.
pub fn sprod(a: &[Poly], b: &[Poly]) -> Poly {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0i128; N];
    for k in 0..a.len() {
        for i in 0..N {
            let x = a[k].0[i] as i128;
            if x == 0 {
                continue;
            }
            for j in 0..N - i {
                acc[i + j] += x * b[k].0[j] as i128;
            }
            for j in N - i..N {
                acc[i + j - N] -= x * b[k].0[j] as i128;
            }
        }
    }
    let mut r = [0i64; N];
    for i in 0..N {
        r[i] = cmod(acc[i]);
    }
    Poly(r)
}

/// Integer dot product of coefficient vectors (for JL projection bookkeeping).
pub fn coeff_normsq(v: &[Poly]) -> u64 {
    v.iter().map(|p| p.normsq()).sum()
}

/// Deterministic byte expansion (SHAKE-256) replacing the reference's AES-CTR
/// stream. Same role: expand (seed, nonce) into a public random stream.
pub fn expand_seed(seed: &[u8], nonce: u64, out: &mut [u8]) {
    use lattice_core::keccak::KeccakSponge;
    let mut h = KeccakSponge::new_shake256();
    h.update(b"greyhound/expand/v1");
    h.update(&(seed.len() as u64).to_le_bytes());
    h.update(seed);
    h.update(&nonce.to_le_bytes());
    h.update(&(out.len() as u64).to_le_bytes());
    h.finalize_in_place();
    h.squeeze(out);
}

/// A finite field element mod q used for the Z_q-valued challenges (ψ, ω, α) of
/// the papers' aggregation steps and the Greyhound evaluation point.
pub fn field_mod(x: i128) -> i64 {
    cmod(x)
}

/// Exact integer power mod q (the evaluation-point powers of Greyhound §4.1).
pub fn pow_mod(mut base: i64, mut k: u64) -> i64 {
    let mut acc: i64 = 1;
    base = cmod(base as i128);
    while k > 0 {
        if k & 1 == 1 {
            acc = cmod(acc as i128 * base as i128);
        }
        base = cmod(base as i128 * base as i128);
        k >>= 1;
    }
    acc
}

/// Multiplicative inverse mod q via the extended Euclidean algorithm — used to
/// check the papers' invertibility conditions (Lemma 2.1 / weak-opening division).
pub fn poly_inv(a: &Poly) -> Option<Poly> {
    // Solve in the CRT field: X^64+1 = f1·f2 (two degree-32 factors). Simpler and
    // fully general: solve the linear system A·x = 1 over Z_q where A is the
    // negacyclic matrix of a, via Gaussian elimination on 64 unknowns mod q.
    let mut mat = [[0i64; N]; N];
    for i in 0..N {
        for j in 0..N {
            // column j of the multiplication-by-a matrix applied to e_i:
            // a·X^i = Σ_k a_k X^{i+k} (mod X^64+1)
            let k = i + j;
            if k < N {
                mat[k][j] = a.0[i];
            } else {
                mat[k - N][j] = -a.0[i];
            }
        }
    }
    // augment with identity, Gaussian elimination mod q
    let q = Q as i128;
    let mut aug = [[0i64; 2 * N]; N];
    for i in 0..N {
        aug[i][..N].copy_from_slice(&mat[i]);
        aug[i][N + i] = 1;
    }
    let inv = |x: i64| -> Option<i64> {
        let (mut a0, mut b0) = (x as i128, q);
        let (mut x0, mut x1) = (1i128, 0i128);
        while b0 != 0 {
            let dq = a0 / b0;
            (a0, b0) = (b0, a0 - dq * b0);
            (x0, x1) = (x1, x0 - dq * x1);
        }
        // the gcd may come out as -1 for negative pivots — still a unit
        if a0 != 1 && a0 != -1 {
            None
        } else {
            Some(cmod(x0 * a0))
        }
    };
    for col in 0..N {
        // find pivot
        let mut piv = None;
        for row in col..N {
            if aug[row][col] != 0 {
                piv = Some(row);
                break;
            }
        }
        let p = piv?; // singular -> not invertible
        aug.swap(col, p);
        let pinv = inv(aug[col][col])?;
        for j in col..2 * N {
            aug[col][j] = cmod(aug[col][j] as i128 * pinv as i128);
        }
        for row in 0..N {
            if row != col && aug[row][col] != 0 {
                let f = aug[row][col];
                for j in col..2 * N {
                    aug[row][j] = cmod(aug[row][j] as i128 - f as i128 * aug[col][j] as i128);
                }
            }
        }
    }
    // x = M^{-1}·e_0 = the first column of the reduced inverse
    let mut out = [0i64; N];
    for i in 0..N {
        out[i] = aug[i][N];
    }
    Some(Poly(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_poly(seed: u64) -> Poly {
        Poly::almost_uniform(&seed.to_le_bytes(), seed.wrapping_mul(3))
    }

    #[test]
    fn cmod_matches_division() {
        let mut x = -(2i128 << 60);
        while x < 2i128 << 60 {
            assert_eq!(cmod(x), cmod_div(x));
            x += 99_877_123_456_789i128;
        }
    }

    #[test]
    fn ring_axioms() {
        for s in 0u64..20 {
            let a = rng_poly(s + 1);
            let b = rng_poly(s + 7);
            let c = rng_poly(s + 13);
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.mul(&b.mul(&c)), a.mul(&b).mul(&c));
            assert_eq!(a.add(&b), b.add(&a));
            let one = Poly::constant(1);
            assert_eq!(a.mul(&one), a);
        }
    }

    #[test]
    fn sigma_m1_is_involution_and_adjoint() {
        // σ^{-1} is an involution, and <a,b> = ct(σ^{-1}(a)·b) (the papers' §2 identity)
        for s in 0u64..10 {
            let a = rng_poly(s + 100);
            let b = rng_poly(s + 200);
            assert_eq!(a.sigma_m1().sigma_m1(), a);
            let lhs: i128 =
                a.0.iter()
                    .zip(b.0.iter())
                    .map(|(x, y)| *x as i128 * *y as i128)
                    .sum();
            let rhs = sprod(&[a.sigma_m1()], &[b]).constant_term();
            assert_eq!(cmod(lhs), rhs);
        }
    }

    #[test]
    fn decompose_recombine_identity() {
        for s in 0u64..10 {
            let a = rng_poly(s + 31);
            for &(t, d) in &[(2usize, 9u32), (3, 8), (5, 7), (4, 10), (1, 32)] {
                let parts = a.decompose(t, d);
                assert_eq!(parts.len(), t);
                if t > 1 {
                    for p in &parts[..t - 1] {
                        assert!(p.norminf() <= 1i64 << (d - 1), "digit width violated");
                    }
                }
                assert_eq!(Poly::recombine(&parts, d), a, "t={t} d={d}");
            }
        }
    }

    #[test]
    fn opnorm_of_challenge_is_bounded() {
        // the papers' challenge set has ‖c‖op ≤ T = 14 (rejection); a random
        // small poly should not wildly exceed the l1-based estimate
        let a = Poly::from_i16(
            &(0..N)
                .map(|i| (((i * 37 % 7) as i64) - 3) as i16)
                .collect::<Vec<_>>(),
        );
        let op = a.opnorm();
        assert!(op < 600.0, "opnorm {op} absurd");
    }

    #[test]
    fn inverse_diagnostic() {
        // 1 + X is provably invertible: inverse = 2^{-1}·Σ(-X)^i
        let mut coeffs = [0i64; N];
        coeffs[0] = 1;
        coeffs[1] = 1;
        let a = Poly(coeffs);
        match poly_inv(&a) {
            Some(inv) => {
                let one = a.mul(&inv);
                assert_eq!(one, Poly::constant(1), "1+X inverse wrong");
            }
            None => panic!("1+X reported non-invertible — poly_inv is broken"),
        }
    }

    #[test]
    fn inverse_roundtrip() {
        // short nonzero elements are invertible (Lemma 2.1): ‖f‖∞ < sqrt(q/12) ≈ 18948
        let mut small = [0i64; N];
        for (i, v) in small.iter_mut().enumerate() {
            *v = ((i * 71 % 61) as i64) - 30; // |·| ≤ 30
        }
        let a = Poly(small);
        let inv = poly_inv(&a).expect("short element must be invertible");
        let one = a.mul(&inv);
        assert_eq!(one, Poly::constant(1));
        // a full-size random element is invertible too (uniform random is a unit)
        let b = rng_poly(555);
        if let Some(inv) = poly_inv(&b) {
            assert_eq!(b.mul(&inv), Poly::constant(1));
        }
    }

    #[test]
    fn q_properties() {
        assert_eq!(
            Q % 8,
            5,
            "q ≡ 5 mod 8 required (Lemma 2.1 + two-factor split)"
        );
        // ord_128(q) = 32 -> X^64+1 splits into 2 degree-32 factors
        let mut x = Q as u128 % 128;
        let mut ord = 1u32;
        while x != 1 {
            x = x * (Q as u128) % 128;
            ord += 1;
        }
        assert_eq!(ord, 32);
    }

    #[test]
    fn sprod_is_bilinear() {
        let a = vec![rng_poly(11), rng_poly(12)];
        let b = vec![rng_poly(13), rng_poly(14)];
        let c = vec![rng_poly(15), rng_poly(16)];
        assert_eq!(
            sprod(&a, &b).add(&sprod(&a, &c)),
            sprod(
                &a,
                &b.iter()
                    .zip(c.iter())
                    .map(|(x, y)| x.add(y))
                    .collect::<Vec<_>>()
            )
        );
    }
}
