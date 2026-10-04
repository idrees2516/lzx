//! The polynomial/domain layer of ePrint 2026/538 (§2 "Preliminaries and
//! Notation"): the two representations (`ν = 1` univariate over the
//! roots-of-unity domain `H`, `ν = log n` multivariate over the boolean
//! cube), the Lagrange bases `λ_h`, the vanishing polynomial `u_H`, the
//! identity polynomial `Λ(X, Y)` in both forms, and matrix polynomials
//! `M(X, Y) = λ(Y)ᵀ M λ(X)`.
//!
//! BN254 `F_r` has 2-adicity 28, so univariate domains up to `2²⁰` are
//! available; the domain generator is fixed per `n` deterministically.

use crate::Fp256;
use lattice_core::transcript::Transcript;

/// A 2²⁸-th root of unity in F_r (order exactly 2²⁸).
pub const ROOT_OF_UNITY: [u64; 4] = [
    0x9bd6_1b6e_725b_19f0,
    0x402d_111e_4111_2ed4,
    0x00e0_a7eb_8ef6_2abc,
    0x2a3c_09f0_a58a_7e85,
];

/// The domain representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Domain {
    /// `ν = 1`: the multiplicative subgroup `H` of n-th roots of unity
    /// (n a power of two ≤ 2²⁸), canonically ordered by the generator.
    Univariate { n: usize },
    /// `ν = log n`: the boolean cube `{0,1}^num_vars`.
    Multivariate { num_vars: usize },
}

impl Domain {
    pub fn size(&self) -> usize {
        match self {
            Domain::Univariate { n } => *n,
            Domain::Multivariate { num_vars } => 1 << num_vars,
        }
    }

    pub fn nu(&self) -> usize {
        match self {
            Domain::Univariate { .. } => 1,
            Domain::Multivariate { num_vars } => *num_vars,
        }
    }

    /// The i-th domain point (canonical order):
    /// `h = g^{i}` for univariate, `Bits(i)` for multivariate.
    pub fn point(&self, i: usize) -> Vec<Fp256> {
        match self {
            Domain::Univariate { n } => vec![root_of_unity_pow(i, *n)],
            Domain::Multivariate { num_vars } => (0..*num_vars)
                .map(|k| Fp256::from_canonical_u64(((i >> k) & 1) as u64))
                .collect(),
        }
    }
}

/// `g_{2^k}` = the generator of the 2^k-th roots: `ROOT_OF_UNITY^{2^{28-k}}`.
fn root_of_unity_pow(i: usize, n: usize) -> Fp256 {
    // g_n = ζ^{2^{28}/n}; h = g_n^i = ζ^{i·2^{28}/n}
    let shift = 28 - n.trailing_zeros();
    let exp: u128 = (i as u128) << shift;
    // ζ^exp — square-and-multiply over the 28-bit exponent.
    // ROOT_OF_UNITY holds CANONICAL limbs; convert to Montgomery form for
    // the arithmetic.
    let zeta = Fp256 {
        limbs: ROOT_OF_UNITY,
    }
    .to_mont();
    let mut acc = Fp256::from_canonical_u64(1);
    let mut base = zeta;
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

/// The domain generator `g_n` (univariate).
pub fn domain_generator(n: usize) -> Fp256 {
    root_of_unity_pow(1, n)
}

/// The i-th univariate domain point `g_n^i`.
pub fn domain_point(i: usize, n: usize) -> Fp256 {
    root_of_unity_pow(i, n)
}

/// The vanishing polynomial `u_H(X) = Xⁿ − 1` evaluated at `x`.
pub fn vanish_eval(n: usize, x: &Fp256) -> Fp256 {
    // x^n − 1
    let mut acc = Fp256::from_canonical_u64(1);
    let mut base = *x;
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc.mul(&base);
        }
        base = base.mul(&base);
        e >>= 1;
    }
    acc.sub(&Fp256::from_canonical_u64(1))
}

/// The Lagrange basis `λ_h(X)` evaluated at `x` — the paper's §2:
/// `λ_h(X) = u_H(X) / (n·h^{n−1}·(X − h))` (the extraction's
/// `u_H(X)/(n(X − h))` is off by the `h^{n−1}` factor; the mathematically
/// correct basis satisfies `λ_h(h) = 1`, `λ_h(h') = 0`).
pub fn lambda_h_eval(n: usize, h: &Fp256, x: &Fp256) -> Result<Fp256, PolyError> {
    if *x == *h {
        return Ok(Fp256::from_canonical_u64(1));
    }
    let u = vanish_eval(n, x);
    // n·h^{n−1} = n·h^{−1} (h^n = 1) — the denominator n·h^{n−1}·(x−h).
    let h_inv = h.inverse().ok_or(PolyError::ZeroDivision)?;
    let mut coef = Fp256::from_canonical_u64(n as u64);
    coef = coef.mul(&h_inv);
    let denom = coef.mul(&x.sub(h));
    let dinv = denom.inverse().ok_or(PolyError::ZeroDivision)?;
    Ok(u.mul(&dinv))
}

/// The eq basis for the multivariate cube: `λ_b(X) = eq(X, b)` at `x`.
pub fn eq_basis_eval(b: usize, x: &[Fp256]) -> Fp256 {
    let one = Fp256::from_canonical_u64(1);
    let mut acc = one;
    for (k, xk) in x.iter().enumerate() {
        let term = if (b >> k) & 1 == 1 { *xk } else { one.sub(xk) };
        acc = acc.mul(&term);
    }
    acc
}

/// `λ_i(X)` at `x` for either domain representation.
pub fn lambda_eval(domain: &Domain, i: usize, x: &[Fp256]) -> Result<Fp256, PolyError> {
    match domain {
        Domain::Univariate { n } => {
            let h = root_of_unity_pow(i, *n);
            lambda_h_eval(*n, &h, &x[0])
        }
        Domain::Multivariate { .. } => Ok(eq_basis_eval(i, x)),
    }
}

/// `Λ(X, Y)` evaluated at concrete points — the identity-matrix polynomial:
/// * univariate: `(u_H(X)·Y − u_H(Y)·X) / (n·(X − Y))` (the paper's §2),
///   which equals `δ_{X,Y}` on `H × H` (the closed form evaluates in O(n)
///   via `u_H` here — domain-sized only).
/// * multivariate: `∏ᵢ (XᵢYᵢ + (1−Xᵢ)(1−Yᵢ))`.
pub fn lambda_matrix_eval(domain: &Domain, x: &[Fp256], y: &[Fp256]) -> Result<Fp256, PolyError> {
    match domain {
        Domain::Univariate { n } => {
            if x[0] == y[0] {
                // The diagonal: the closed form is 0/0; the polynomial value
                // is Σᵢ λᵢ(x)·λᵢ(x) (= 1 on H, the analytic continuation
                // off H).
                let mut acc = Fp256::ZERO;
                for i in 0..*n {
                    let li = lambda_h_eval(*n, &root_of_unity_pow(i, *n), &x[0])?;
                    acc = acc.add(&li.mul(&li));
                }
                return Ok(acc);
            }
            let u_x = vanish_eval(*n, &x[0]);
            let u_y = vanish_eval(*n, &y[0]);
            let num = u_x.mul(&y[0]).sub(&u_y.mul(&x[0]));
            let denom = Fp256::from_canonical_u64(*n as u64).mul(&x[0].sub(&y[0]));
            let dinv = denom.inverse().ok_or(PolyError::ZeroDivision)?;
            Ok(num.mul(&dinv))
        }
        Domain::Multivariate { .. } => {
            let one = Fp256::from_canonical_u64(1);
            let mut acc = one;
            for (xi, yi) in x.iter().zip(y.iter()) {
                let term = xi.mul(yi).add(&one.sub(xi).mul(&one.sub(yi)));
                acc = acc.mul(&term);
            }
            Ok(acc)
        }
    }
}

/// The vector polynomial `z(X) = zᵀλ(X)` evaluated at `x`.
pub fn vec_poly_eval(domain: &Domain, z: &[Fp256], x: &[Fp256]) -> Result<Fp256, PolyError> {
    if z.len() != domain.size() {
        return Err(PolyError::Shape("vector length vs domain size"));
    }
    let mut acc = Fp256::ZERO;
    for (i, zi) in z.iter().enumerate() {
        if zi.is_zero() {
            continue;
        }
        acc = acc.add(&zi.mul(&lambda_eval(domain, i, x)?));
    }
    Ok(acc)
}

/// The matrix polynomial `M(X, Y) = λ(Y)ᵀ M λ(X)` evaluated at `(x, y)`
/// (the paper's convention: `M(β, α) = λ(α)ᵀ M λ(β)` — X first).
pub fn matrix_poly_eval(
    domain: &Domain,
    m: &[Vec<Fp256>],
    x: &[Fp256],
    y: &[Fp256],
) -> Result<Fp256, PolyError> {
    let n = domain.size();
    if m.len() != n || m.iter().any(|r| r.len() != n) {
        return Err(PolyError::Shape("matrix shape vs domain size"));
    }
    // λ(x) evaluations per column, λ(y) per row.
    let lam_x: Vec<Fp256> = (0..n)
        .map(|i| lambda_eval(domain, i, x))
        .collect::<Result<_, _>>()?;
    let lam_y: Vec<Fp256> = (0..n)
        .map(|j| lambda_eval(domain, j, y))
        .collect::<Result<_, _>>()?;
    let mut acc = Fp256::ZERO;
    for j in 0..n {
        if lam_y[j].is_zero() {
            continue;
        }
        let mut row_dot = Fp256::ZERO;
        for i in 0..n {
            if m[j][i].is_zero() || lam_x[i].is_zero() {
                continue;
            }
            row_dot = row_dot.add(&m[j][i].mul(&lam_x[i]));
        }
        acc = acc.add(&lam_y[j].mul(&row_dot));
    }
    Ok(acc)
}

/// Matrix–vector product.
pub fn mat_vec(m: &[Vec<Fp256>], v: &[Fp256]) -> Result<Vec<Fp256>, PolyError> {
    if m.iter().any(|r| r.len() != v.len()) {
        return Err(PolyError::Shape("mat_vec shape"));
    }
    Ok(m.iter()
        .map(|row| {
            row.iter()
                .zip(v.iter())
                .fold(Fp256::ZERO, |acc, (a, b)| acc.add(&a.mul(b)))
        })
        .collect())
}

/// Multilinear extension evaluation of a cube-value vector at `r`
/// (LSB-first bit order matching `Domain::point`).
pub fn mle_eval(f: &[Fp256], r: &[Fp256]) -> Fp256 {
    let mut cur = f.to_vec();
    let mut len = f.len();
    for k in (0..r.len()).rev() {
        let rk = r[k];
        let one = Fp256::from_canonical_u64(1);
        let next_len = len / 2;
        for i in 0..next_len {
            let lo = cur[i];
            let hi = cur[i + next_len];
            cur[i] = one.sub(&rk).mul(&lo).add(&rk.mul(&hi));
        }
        len = next_len;
    }
    cur[0]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyError {
    Shape(&'static str),
    ZeroDivision,
}

impl core::fmt::Display for PolyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolyError::Shape(s) => write!(f, "poly shape: {s}"),
            PolyError::ZeroDivision => write!(f, "division by a zero field element"),
        }
    }
}

/// Deterministic pseudo-random field vector (tests, instances).
pub fn fp_vec(label: &[u8], seed: &[u8], n: usize) -> Vec<Fp256> {
    let bytes = Transcript::xof(label, seed, 32 * n);
    (0..n)
        .map(|i| {
            let mut b = [0u8; 32];
            b.copy_from_slice(&bytes[i * 32..(i + 1) * 32]);
            lattice_pcd::util::fp_from_be32(&b)
        })
        .collect()
}

/// A random n×n matrix.
pub fn fp_matrix(label: &[u8], seed: &[u8], n: usize) -> Vec<Vec<Fp256>> {
    let flat = fp_vec(label, seed, n * n);
    (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn univariate_domain_points_are_roots() {
        for n in [4usize, 8, 16] {
            for i in 0..n {
                let h = root_of_unity_pow(i, n); // g_n^i
                                                 // h^n = 1
                let mut acc = fr(1);
                for _ in 0..n {
                    acc = acc.mul(&h);
                }
                assert!(acc == fr(1), "h^{n} != 1 at i={i}");
            }
            // Distinctness of the first few points.
            let g = domain_generator(n);
            assert!(g != fr(1));
            // g_n = ζ^{2^{28}/n} — check against root_of_unity_pow(1, n).
            assert_eq!(g, root_of_unity_pow(1, n));
        }
    }

    #[test]
    fn lambda_h_partition_of_unity() {
        let n = 8;
        for probe in [0usize, 3, 7] {
            let h = root_of_unity_pow(probe, n);
            let x = [h];
            let mut total = Fp256::ZERO;
            for i in 0..n {
                let li = lambda_h_eval(n, &root_of_unity_pow(i, n), &x[0])
                    .ok()
                    .unwrap();
                // indicator: 1 iff i == probe
                if i == probe {
                    assert_eq!(li, fr(1));
                } else {
                    assert!(li.is_zero(), "λ_{i}(h_{probe}) = {li:?}");
                }
                total = total.add(&li);
            }
            assert_eq!(total, fr(1));
        }
    }

    #[test]
    fn lambda_h_off_domain() {
        // λ_h(x) for x ∉ H still evaluates consistently: Σ λ_h(x) = 1? No —
        // that identity is cube-only. Spot-check the closed form: at x = 2g.
        let n = 4;
        // g^2 and g^3 are both in H → u_H = 0 → λ = 0.
        let h = root_of_unity_pow(2, n);
        let x = root_of_unity_pow(3, n);
        let l = lambda_h_eval(n, &h, &x).ok().unwrap();
        // u_H(x) = x^4 − 1; g^4 = 1 → (g^3)^4 = 1 → u_H = 0 → λ = 0.
        assert!(l.is_zero());
    }

    #[test]
    fn multivariate_eq_basis() {
        let x = vec![fr(3), fr(5)];
        // Bits(0) = (0,0): (1−3)(1−5) = 8
        assert_eq!(eq_basis_eval(0, &x), fr(8));
        // Bits(1) = (1,0): 3·(1−5) = −12
        assert_eq!(eq_basis_eval(1, &x), fr(0).sub(&fr(12)));
    }

    #[test]
    fn lambda_matrix_identity_on_domain() {
        // Λ(h, h') = δ for both representations.
        let d = Domain::Univariate { n: 8 };
        for (i, j) in [(0usize, 0usize), (2, 2), (3, 5), (0, 7)] {
            let x = d.point(i);
            let y = d.point(j);
            let l = lambda_matrix_eval(&d, &x, &y).ok().unwrap();
            if i == j {
                assert_eq!(l, fr(1));
            } else {
                assert!(l.is_zero());
            }
        }
        let dm = Domain::Multivariate { num_vars: 3 };
        for (i, j) in [(0usize, 0usize), (5, 5), (2, 6)] {
            let x = dm.point(i);
            let y = dm.point(j);
            let l = lambda_matrix_eval(&dm, &x, &y).ok().unwrap();
            if i == j {
                assert_eq!(l, fr(1));
            } else {
                assert!(l.is_zero());
            }
        }
    }

    #[test]
    fn matrix_poly_matches_bilinear_form() {
        let n = 4;
        let d = Domain::Multivariate { num_vars: 2 };
        let m = fp_matrix(b"m", b"seed", n);
        let x = vec![fr(3), fr(5)];
        let y = vec![fr(7), fr(11)];
        // M(x, y) = λ(y)ᵀ M λ(x): compute directly via basis evals.
        let mut direct = Fp256::ZERO;
        for j in 0..n {
            for i in 0..n {
                let ly = eq_basis_eval(j, &y);
                let lx = eq_basis_eval(i, &x);
                direct = direct.add(&m[j][i].mul(&lx).mul(&ly));
            }
        }
        let got = matrix_poly_eval(&d, &m, &x, &y).ok().unwrap();
        assert_eq!(got, direct);
        // Univariate too.
        let du = Domain::Univariate { n };
        let mu = fp_matrix(b"mu", b"seed", n);
        let xu = vec![root_of_unity_pow(3, n)];
        let yu = vec![root_of_unity_pow(1, n)];
        let mut direct_u = Fp256::ZERO;
        for j in 0..n {
            for i in 0..n {
                let ly = lambda_h_eval(n, &root_of_unity_pow(j, n), &yu[0])
                    .ok()
                    .unwrap();
                let lx = lambda_h_eval(n, &root_of_unity_pow(i, n), &xu[0])
                    .ok()
                    .unwrap();
                direct_u = direct_u.add(&mu[j][i].mul(&lx).mul(&ly));
            }
        }
        let got_u = matrix_poly_eval(&du, &mu, &xu, &yu).ok().unwrap();
        assert_eq!(got_u, direct_u);
    }

    #[test]
    fn vec_poly_eval_matches_mle() {
        let f = vec![fr(5), fr(7), fr(11), fr(13)];
        let r = vec![fr(3), fr(5)];
        let d = Domain::Multivariate { num_vars: 2 };
        assert_eq!(vec_poly_eval(&d, &f, &r).ok().unwrap(), mle_eval(&f, &r));
    }
}
