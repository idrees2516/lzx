//! The CauchyFold field layer: the base field `F_q` with the paper's
//! `q = 2^48 − 59 = 281474976710597` and the degree-4 computation field
//! `K = F_q[u]/(u^4 − 4u^2 + 2)` with its public θ-basis
//! `θ = (1, u, u^2 − 2, u^3 − 3u)` (Appendix B.1).
//!
//! `q` coincides with the LaBRADOR ring modulus (`lattice_labrador::ring::Q`)
//! — the paper's linear chain runs over `R_{q,64}`, which this crate reuses
//! directly. All field arithmetic is exact `u64`/`i128` integer arithmetic
//! (products `≤ (q−1)^2 < 2^96`).

/// The base field `F_q`, `q = 2^48 − 59` (prime, `q ≡ 5 (mod 8)` — the
/// paper's unit-criterion modulus).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fq48(pub u64);

pub const Q48: u64 = 281_474_976_710_597; // 2^48 - 59

impl Fq48 {
    pub const ZERO: Fq48 = Fq48(0);
    pub const ONE: Fq48 = Fq48(1);

    #[inline]
    pub fn from_u64(x: u64) -> Self {
        Fq48(x % Q48)
    }

    #[inline]
    pub fn from_i64(x: i64) -> Self {
        let r = (x as i128).rem_euclid(Q48 as i128) as u64;
        Fq48(r)
    }

    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        Fq48((self.0 + o.0) % Q48)
    }

    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        Fq48((self.0 + Q48 - o.0) % Q48)
    }

    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        Fq48(((self.0 as u128 * o.0 as u128) % Q48 as u128) as u64)
    }

    #[inline]
    pub fn neg(&self) -> Self {
        if self.0 == 0 {
            Self::ZERO
        } else {
            Fq48(Q48 - self.0)
        }
    }

    pub fn pow(&self, e: u64) -> Self {
        let mut acc = Fq48::ONE;
        let mut base = *self;
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse (Fermat).
    pub fn inv(&self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        Some(self.pow(Q48 - 2))
    }

    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Centered representative in `[-(q-1)/2, (q-1)/2]`.
    pub fn centered(&self) -> i64 {
        if self.0 > Q48 / 2 {
            self.0 as i64 - Q48 as i64
        } else {
            self.0 as i64
        }
    }

    /// Uniform draw from transcript bytes (6 bytes = 48 bits, rejection
    /// rate `59/2^48`).
    pub fn challenge(
        transcript: &mut lattice_core::transcript::Transcript,
    ) -> Result<Self, String> {
        let bytes = transcript
            .challenge_bytes(b"cauchyfold-fq", 8)
            .map_err(|e| e.to_string())?;
        let mut arr = [0u8; 8];
        arr[..6].copy_from_slice(&bytes[..6]);
        let v = u64::from_le_bytes(arr) & ((1u64 << 48) - 1);
        if v < Q48 {
            Ok(Fq48(v))
        } else {
            Ok(Fq48(v - Q48)) // v < 2^48 < 2q: single fold is exact
        }
    }

    pub fn to_bytes(self) -> [u8; 6] {
        let b = self.0.to_le_bytes();
        [b[0], b[1], b[2], b[3], b[4], b[5]]
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let mut arr = [0u8; 8];
        arr[..6].copy_from_slice(&b[..6]);
        Fq48(u64::from_le_bytes(arr))
    }
}

/// An element of `K = F_q[u]/(u^4 − 4u^2 + 2)` in the power basis
/// `(1, u, u^2, u^3)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct K4(pub [Fq48; 4]);

impl K4 {
    pub const ZERO: K4 = K4([Fq48::ZERO; 4]);
    pub const ONE: K4 = K4([Fq48::ONE, Fq48::ZERO, Fq48::ZERO, Fq48::ZERO]);

    pub fn from_coeffs(c: [u64; 4]) -> Self {
        K4([Fq48(c[0]), Fq48(c[1]), Fq48(c[2]), Fq48(c[3])])
    }

    pub fn add(&self, o: &Self) -> Self {
        let mut r = [Fq48::ZERO; 4];
        for i in 0..4 {
            r[i] = self.0[i].add(&o.0[i]);
        }
        K4(r)
    }

    pub fn sub(&self, o: &Self) -> Self {
        let mut r = [Fq48::ZERO; 4];
        for i in 0..4 {
            r[i] = self.0[i].sub(&o.0[i]);
        }
        K4(r)
    }

    pub fn neg(&self) -> Self {
        K4(self.0.map(|x| x.neg()))
    }

    /// Scale by a full `K` scalar (equivalent to `mul`, kept for
    /// readability at the call sites).
    pub fn scale(&self, c: &K4) -> Self {
        self.mul(c)
    }

    /// Multiplication in the power basis: `u^4 = 4u^2 − 2`.
    pub fn mul(&self, o: &Self) -> Self {
        // Linear convolution of length 4 → 7, then reduce u^4, u^5, u^6.
        let mut conv = [Fq48::ZERO; 7];
        for i in 0..4 {
            for j in 0..4 {
                let t = self.0[i].mul(&o.0[j]);
                conv[i + j] = conv[i + j].add(&t);
            }
        }
        // u^4 = 4u^2 - 2;  u^5 = 4u^3 - 2u;  u^6 = 4u^4 - 2u^2 = 14u^2 - 8.
        let mut r = [Fq48::ZERO; 4];
        for (i, c) in conv.iter().enumerate().take(4) {
            r[i] = *c;
        }
        // u^4 = 4u^2 - 2:
        r[2] = r[2].add(&conv[4].mul(&Fq48(4)));
        r[0] = r[0].sub(&conv[4].mul(&Fq48(2)));
        // u^5 term:
        r[3] = r[3].add(&conv[5].mul(&Fq48(4)));
        r[1] = r[1].sub(&conv[5].mul(&Fq48(2)));
        // u^6 term: u^6 = 4u^4 - 2u^2 = 4(4u^2-2) - 2u^2 = 14u^2 - 8.
        r[2] = r[2].add(&conv[6].mul(&Fq48(14)));
        r[0] = r[0].sub(&conv[6].mul(&Fq48(8)));
        K4(r)
    }

    /// Exponentiation (for the γ-power ladders).
    pub fn pow(&self, exp: u64) -> Self {
        let mut acc = K4::ONE;
        let mut base = *self;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        acc
    }

    /// Inverse via the 4×4 linear system (the conjugate structure of the
    /// quartic; solvable by elimination — the norm is `a^4 − 4a^2b... `,
    /// we just solve exactly).
    pub fn inv(&self) -> Option<Self> {
        // Solve x * self = 1: linear in x's coefficients.
        // Multiplication by self as a 4x4 matrix over Fq (columns = images
        // of 1, u, u^2, u^3).
        // 1 * self = (a0, a1, a2, a3)
        // u * self = u(a0 + a1u + a2u^2 + a3u^3) = a0u + a1u^2 + a2u^3 + a3u^4
        //          = (−2a3) + a0u + a1u^2 + a2u^3 + ... wait u^4 = 4u^2−2:
        //          = −2a3 + a0 u + (a1 + 4a3)u^2 + a2 u^3
        // u^2 * self = a0u^2 + a1u^3 + a2u^4 + a3u^5
        //          = (−2a2 − 2a3·... ) — compute via mul.
        let one = K4::ONE;
        let u = K4([Fq48::ZERO, Fq48::ONE, Fq48::ZERO, Fq48::ZERO]);
        let u2 = u.mul(&u);
        let u3 = u2.mul(&u);
        let cols = [one.mul(self), u.mul(self), u2.mul(self), u3.mul(self)];
        // Solve the 4x4 system [cols] x = e0 (1,0,0,0).
        let mut m = [[Fq48::ZERO; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = cols[j].0[i];
            }
        }
        let rhs = [Fq48::ONE, Fq48::ZERO, Fq48::ZERO, Fq48::ZERO];
        let x = solve_linear4(&m, &rhs)?;
        Some(K4(x))
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|c| c.is_zero())
    }

    /// The θ-basis conversion (Appendix B.1): power basis →
    /// `(1, u, u^2 − 2, u^3 − 3u)` coordinates.
    pub fn to_theta(&self) -> Self {
        // (b0,b1,b2,b3)_power = (a0 − 2a2, a1 − 3a3, a2, a3)_θ means
        // a = (b0 + 2b2, b1 + 3b3, b2, b3)_θ in power basis... per B.1:
        // (a0,a1,a2,a3)_θ → (a0−2a2, a1−3a3, a2, a3)_power. Invert:
        let b0 = self.0[0].add(&self.0[2].mul(&Fq48(2)));
        let b1 = self.0[1].add(&self.0[3].mul(&Fq48(3)));
        K4([b0, b1, self.0[2], self.0[3]])
    }

    /// θ-basis → power basis: `(a0,a1,a2,a3)_θ → (a0−2a2, a1−3a3, a2, a3)`.
    pub fn from_theta(&self) -> Self {
        let p0 = self.0[0].sub(&self.0[2].mul(&Fq48(2)));
        let p1 = self.0[1].sub(&self.0[3].mul(&Fq48(3)));
        K4([p0, p1, self.0[2], self.0[3]])
    }

    pub fn to_bytes(self) -> [u8; 24] {
        let mut out = [0u8; 24];
        for i in 0..4 {
            out[i * 6..i * 6 + 6].copy_from_slice(&self.0[i].to_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let mut c = [Fq48::ZERO; 4];
        for i in 0..4 {
            c[i] = Fq48::from_bytes(&b[i * 6..i * 6 + 6]);
        }
        K4(c)
    }

    /// Uniform draw (4 coefficients).
    pub fn challenge(
        transcript: &mut lattice_core::transcript::Transcript,
    ) -> Result<Self, String> {
        let mut c = [Fq48::ZERO; 4];
        for i in 0..4 {
            c[i] = Fq48::challenge(transcript)?;
        }
        Ok(K4(c))
    }
}

/// Exact 4×4 linear solve over `F_q` (Gaussian elimination).
fn solve_linear4(m: &[[Fq48; 4]; 4], rhs: &[Fq48; 4]) -> Option<[Fq48; 4]> {
    let mut a = *m;
    let mut b = *rhs;
    for col in 0..4 {
        // Pivot.
        let mut piv = None;
        for r in col..4 {
            if !a[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        let p = piv?;
        a.swap(col, p);
        b.swap(col, p);
        let inv = a[col][col].inv()?;
        for j in 0..4 {
            a[col][j] = a[col][j].mul(&inv);
        }
        b[col] = b[col].mul(&inv);
        for r in 0..4 {
            if r != col && !a[r][col].is_zero() {
                let f = a[r][col];
                for j in 0..4 {
                    a[r][j] = a[r][j].sub(&a[col][j].mul(&f));
                }
                b[r] = b[r].sub(&b[col].mul(&f));
            }
        }
    }
    Some(b)
}

/// A polynomial over `K` (coefficients in K), for the Cauchy carrier
/// algebra (`D(T)`, `P_i`, `H(T)`, the discrepancy `F(T)`).
#[derive(Clone, Debug, PartialEq)]
pub struct KPoly {
    pub coeffs: Vec<K4>,
}

impl KPoly {
    pub fn zero() -> Self {
        KPoly { coeffs: vec![] }
    }

    pub fn constant(c: K4) -> Self {
        KPoly { coeffs: vec![c] }
    }

    pub fn from_coeffs(c: Vec<K4>) -> Self {
        let mut p = KPoly { coeffs: c };
        p.trim();
        p
    }

    fn trim(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().unwrap().is_zero() {
            self.coeffs.pop();
        }
        if self.coeffs.is_empty() {
            self.coeffs.push(K4::ZERO);
        }
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn add(&self, o: &Self) -> Self {
        let n = self.coeffs.len().max(o.coeffs.len());
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).copied().unwrap_or(K4::ZERO);
            let b = o.coeffs.get(i).copied().unwrap_or(K4::ZERO);
            c.push(a.add(&b));
        }
        KPoly::from_coeffs(c)
    }

    pub fn sub(&self, o: &Self) -> Self {
        let n = self.coeffs.len().max(o.coeffs.len());
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.coeffs.get(i).copied().unwrap_or(K4::ZERO);
            let b = o.coeffs.get(i).copied().unwrap_or(K4::ZERO);
            c.push(a.sub(&b));
        }
        KPoly::from_coeffs(c)
    }

    pub fn scale(&self, s: &K4) -> Self {
        KPoly::from_coeffs(self.coeffs.iter().map(|c| c.scale(s)).collect())
    }

    pub fn mul(&self, o: &Self) -> Self {
        let n = self.coeffs.len() + o.coeffs.len() - 1;
        let mut c = vec![K4::ZERO; n];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in o.coeffs.iter().enumerate() {
                let t = a.mul(b);
                c[i + j] = c[i + j].add(&t);
            }
        }
        KPoly::from_coeffs(c)
    }

    pub fn eval(&self, x: &K4) -> K4 {
        // Horner.
        let mut acc = K4::ZERO;
        for c in self.coeffs.iter().rev() {
            acc = acc.mul(x).add(c);
        }
        acc
    }

    /// Formal derivative.
    pub fn deriv(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return KPoly::zero();
        }
        let mut c = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, a) in self.coeffs.iter().enumerate().skip(1) {
            c.push(a.mul(&K4([Fq48(i as u64), Fq48::ZERO, Fq48::ZERO, Fq48::ZERO])));
        }
        KPoly::from_coeffs(c)
    }

    /// The interpolation polynomial through `points` (Lagrange).
    pub fn interpolate(points: &[(K4, K4)]) -> Self {
        let mut acc = KPoly::zero();
        for (i, (xi, yi)) in points.iter().enumerate() {
            let mut term = KPoly::constant(*yi);
            for (j, (xj, _)) in points.iter().enumerate() {
                if i != j {
                    let den = xi.sub(xj);
                    let inv = den.inv().expect("distinct interpolation points");
                    // (T - xj) / (xi - xj)
                    let lin = KPoly::from_coeffs(vec![xj.neg(), K4::ONE]);
                    term = term.mul(&lin.scale(&inv));
                }
            }
            acc = acc.add(&term);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fq48_arithmetic() {
        assert_eq!(Q48, 281_474_976_710_597);
        let a = Fq48(Q48 - 3);
        let b = Fq48(5);
        assert_eq!(a.add(&b), Fq48(2));
        let c = a.mul(&b);
        assert_eq!(c.mul(&b.inv().unwrap()), a);
        assert_eq!(Fq48::from_i64(-1), Fq48(Q48 - 1));
        assert_eq!(Fq48(Q48 - 1).centered(), -1);
        assert_eq!(Fq48(17).centered(), 17);
    }

    #[test]
    fn k4_ring_axioms() {
        let a = K4::from_coeffs([1, 2, 3, 4]);
        let b = K4::from_coeffs([5, 6, 7, 8]);
        let c = K4::from_coeffs([9, 1, 2, 3]);
        // Commutativity, associativity, distributivity on random elements.
        assert_eq!(a.mul(&b), b.mul(&a));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
        // u^4 = 4u^2 - 2.
        let u = K4([Fq48::ZERO, Fq48::ONE, Fq48::ZERO, Fq48::ZERO]);
        let u4 = u.mul(&u).mul(&u).mul(&u);
        let expect = K4([Fq48(Q48 - 2), Fq48::ZERO, Fq48(4), Fq48::ZERO]);
        assert_eq!(u4, expect);
    }

    #[test]
    fn k4_inverse() {
        for seed in 0..8u64 {
            let a = K4::from_coeffs([
                seed * 7919 + 1,
                seed * 104729 + 2,
                seed * 1299709 + 3,
                seed * 15485863 + 4,
            ]);
            let inv = a.inv().expect("random element invertible");
            assert_eq!(a.mul(&inv), K4::ONE);
        }
        assert!(K4::ZERO.inv().is_none());
    }

    #[test]
    fn theta_basis_roundtrip() {
        let a = K4::from_coeffs([11, 22, 33, 44]);
        let theta = a.to_theta();
        assert_eq!(theta.from_theta(), a);
        // The B.1 formula: power = (a0−2a2, a1−3a3, a2, a3) from θ.
        let t = K4::from_coeffs([1, 2, 3, 4]);
        let p = t.from_theta();
        assert_eq!(p.0[0], Fq48(1).sub(&Fq48(6)));
        assert_eq!(p.0[1], Fq48(2).sub(&Fq48(12)));
        assert_eq!(p.0[2], Fq48(3));
        assert_eq!(p.0[3], Fq48(4));
    }

    #[test]
    fn kpoly_arithmetic() {
        let p = KPoly::from_coeffs(vec![
            K4::ONE,
            K4::from_coeffs([0, 1, 0, 0]),
            K4::from_coeffs([0, 0, 1, 0]),
        ]);
        let q = p.mul(&p);
        let x = K4::from_coeffs([3, 1, 4, 1]);
        assert_eq!(q.eval(&x), p.eval(&x).mul(&p.eval(&x)));
        // Derivative: (1 + u·T + u²·T²)' = u + 2u²·T.
        let d = p.deriv();
        let expect_deriv =
            K4::from_coeffs([0, 1, 0, 0]).add(&K4::from_coeffs([0, 0, 2, 0]).mul(&x));
        assert_eq!(d.eval(&x), expect_deriv);
    }

    #[test]
    fn kpoly_interpolation() {
        // Genuinely quadratic data (degree-4 interpolant through 5
        // points; linear data would legitimately interpolate lower).
        let pts: Vec<(K4, K4)> = (0..5u64)
            .map(|i| {
                (
                    K4::from_coeffs([i * 13 + 1, i * 7, i * 3, i]),
                    K4::from_coeffs([
                        i * i * 5,
                        i * i * 11 + i * 2,
                        i * i * 17,
                        i * i * 19 + i * 3,
                    ]),
                )
            })
            .collect();
        let p = KPoly::interpolate(&pts);
        // The data is quadratic in the parameter, so the interpolant may
        // legitimately have degree 2 < 4; it must reproduce the points.
        assert!(p.degree() < pts.len());
        for (x, y) in &pts {
            assert_eq!(p.eval(x), *y);
        }
    }

    #[test]
    fn byte_roundtrips() {
        let a = Fq48(281_474_976_710_596);
        assert_eq!(Fq48::from_bytes(&a.to_bytes()), a);
        let k = K4::from_coeffs([1, Q48 - 1, 42, 281_474_976_710_596]);
        assert_eq!(K4::from_bytes(&k.to_bytes()), k);
    }
}
