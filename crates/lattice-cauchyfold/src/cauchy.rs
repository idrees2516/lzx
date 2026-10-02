//! The Cauchy carrier algebra (the paper's §4 and Appendix A.3):
//! poles/scales, the scaled-Cauchy challenge family `a_i(c) = λ_i/(c−ξ_i)`,
//! the denominator-cleared fold `W(T) = D(T)z_0 + Σ λ_i P_i(T) z_i`, the
//! carrier `H_src(T)` with its `k` output-valued coefficients, the carrier
//! identity (Proposition 4.4), the fixed-carrier discrepancy `F(T)`
//! (Lemma 4.5), and the fast carrier construction (product trees +
//! multipoint evaluation + interpolation).

use crate::field_k::{K4, KPoly, Q48};

/// The public Cauchy parameters: distinct poles `ξ_1..ξ_k` and nonzero
/// scales `λ_1..λ_k`.
#[derive(Clone, Debug)]
pub struct CauchyParams {
    /// Distinct poles (the paper's k = 16 profile uses `ξ_i = i`).
    pub poles: Vec<K4>,
    /// Nonzero scales (the paper uses `λ_i = 1`).
    pub scales: Vec<K4>,
}

impl CauchyParams {
    /// The paper's profile: `ξ_i = i`, `λ_i = 1`, `i = 1..=k`.
    pub fn paper(k: usize) -> Self {
        CauchyParams {
            poles: (1..=k).map(|i| K4::from_coeffs([i as u64, 0, 0, 0])).collect(),
            scales: (0..k).map(|_| K4::ONE).collect(),
        }
    }

    pub fn k(&self) -> usize {
        self.poles.len()
    }

    /// `D(T) = Π_i (T − ξ_i)`.
    pub fn d_poly(&self) -> KPoly {
        let mut d = KPoly::constant(K4::ONE);
        for xi in &self.poles {
            d = d.mul(&KPoly::from_coeffs(vec![xi.neg(), K4::ONE]));
        }
        d
    }

    /// `P_i(T) = D(T)/(T − ξ_i)` — degree `k−1`.
    pub fn p_poly(&self, i: usize) -> KPoly {
        let mut p = KPoly::constant(K4::ONE);
        for (j, xj) in self.poles.iter().enumerate() {
            if j != i {
                p = p.mul(&KPoly::from_coeffs(vec![xj.neg(), K4::ONE]));
            }
        }
        p
    }

    /// `P_ij(T) = D(T)/((T−ξ_i)(T−ξ_j))` — degree `k−2`.
    pub fn p_ij_poly(&self, i: usize, j: usize) -> KPoly {
        let mut p = KPoly::constant(K4::ONE);
        for (l, xl) in self.poles.iter().enumerate() {
            if l != i && l != j {
                p = p.mul(&KPoly::from_coeffs(vec![xl.neg(), K4::ONE]));
            }
        }
        p
    }

    /// The folding coefficient `a_i(c) = λ_i/(c − ξ_i)` — requires `c` off
    /// the poles.
    pub fn a_i(&self, i: usize, c: &K4) -> Option<K4> {
        let den = c.sub(&self.poles[i]);
        let inv = den.inv()?;
        Some(self.scales[i].mul(&inv))
    }

    /// `D(c)` (the common denominator at the challenge).
    pub fn d_eval(&self, c: &K4) -> K4 {
        self.d_poly().eval(c)
    }

    /// The folded record `z* = z_0 + Σ a_i(c) z_i`.
    pub fn fold_z(&self, sources: &[Vec<K4>], c: &K4) -> Option<Vec<K4>> {
        assert_eq!(sources.len(), self.k() + 1);
        let s = sources[0].len();
        let mut acc = sources[0].clone();
        for i in 1..=self.k() {
            let a = self.a_i(i - 1, c)?;
            for l in 0..s {
                acc[l] = acc[l].add(&sources[i][l].scale(&a));
            }
        }
        Some(acc)
    }

    /// The folded residual `E* = E_0 + Σ a_i(c)^2 E_i + H_src(c)/D(c)`.
    pub fn fold_e(&self, residuals: &[Vec<K4>], carrier: &Carrier, c: &K4) -> Option<Vec<K4>> {
        assert_eq!(residuals.len(), self.k() + 1);
        let y = residuals[0].len();
        let mut acc = residuals[0].clone();
        for i in 1..=self.k() {
            let a = self.a_i(i - 1, c)?;
            let a2 = a.mul(&a);
            for j in 0..y {
                acc[j] = acc[j].add(&residuals[i][j].scale(&a2));
            }
        }
        let hc = carrier.eval(c);
        let dinv = self.d_eval(c).inv()?;
        for j in 0..y {
            acc[j] = acc[j].add(&hc[j].mul(&dinv));
        }
        Some(acc)
    }
}

/// A homogeneous quadratic map `Q : K^s → K^y` with its polarization
/// `B(x,y) = Q(x+y) − Q(x) − Q(y)`.
///
/// The paper's carrier algebra (Prop 4.4) requires homogeneity
/// (`Q(αz) = α²Q(z)`); its Appendix A.2 embeds relaxed R1CS by the
/// extended-vector convention (the linear `−uCz` term rides a leading
/// unit coordinate, whose fold drift IS the relaxation). This scaled
/// profile uses the purely homogeneous form `Q(z) = Az ⊙ Bz` with the
/// relaxation carried by the residuals `E_i` — the `c`/`u` fields record
/// the paper's embedding for the note but do not enter `eval`.
#[derive(Clone, Debug)]
pub struct QuadraticMap {
    pub s: usize,
    pub y: usize,
    /// `A ∈ K^{y×s}`.
    pub a: Vec<Vec<K4>>,
    /// `B ∈ K^{y×s}`.
    pub b: Vec<Vec<K4>>,
    /// `C ∈ K^{y×s}`.
    pub c: Vec<Vec<K4>>,
    /// The relaxation scalar `u`.
    pub u: K4,
}

impl QuadraticMap {
    /// A random diagonal-ish benchmark relation (the paper's
    /// "fixed-size, nonconstant diagonal relaxed R1CS").
    pub fn benchmark(s: usize, y: usize, seed: u64) -> Self {
        let mut nxt = seed;
        let mut rnd = move || {
            nxt = nxt
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            nxt >> 33
        };
        let mut mat = |rows: usize| -> Vec<Vec<K4>> {
            (0..rows)
                .map(|r| {
                    (0..s)
                        .map(|l| {
                            if (r + l) % 3 == 0 || r == l % y {
                                // nonconstant diagonal band
                                K4::from_coeffs([
                                    rnd() % Q48,
                                    rnd() % Q48,
                                    rnd() % Q48,
                                    rnd() % Q48,
                                ])
                            } else {
                                K4::ZERO
                            }
                        })
                        .collect()
                })
                .collect()
        };
        QuadraticMap {
            s,
            y,
            a: mat(y),
            b: mat(y),
            c: mat(y),
            u: K4::from_coeffs([rnd() % Q48, rnd() % Q48, rnd() % Q48, rnd() % Q48]),
        }
    }

    /// `Q(z) = Az ⊙ Bz` (the homogeneous form; see the struct docs).
    pub fn eval(&self, z: &[K4]) -> Vec<K4> {
        let mut out = vec![K4::ZERO; self.y];
        for j in 0..self.y {
            let mut az = K4::ZERO;
            let mut bz = K4::ZERO;
            for l in 0..self.s {
                az = az.add(&self.a[j][l].mul(&z[l]));
                bz = bz.add(&self.b[j][l].mul(&z[l]));
            }
            out[j] = az.mul(&bz);
        }
        out
    }

    /// The polarization `B(x, y) = Ax⊙By + Ay⊙Bx` (the `−uCz` part is
    /// linear and cancels in the polarization).
    pub fn polarize(&self, x: &[K4], y: &[K4]) -> Vec<K4> {
        let mut out = vec![K4::ZERO; self.y];
        for j in 0..self.y {
            let mut ax = K4::ZERO;
            let mut bx = K4::ZERO;
            let mut ay = K4::ZERO;
            let mut by = K4::ZERO;
            for l in 0..self.s {
                ax = ax.add(&self.a[j][l].mul(&x[l]));
                bx = bx.add(&self.b[j][l].mul(&x[l]));
                ay = ay.add(&self.a[j][l].mul(&y[l]));
                by = by.add(&self.b[j][l].mul(&y[l]));
            }
            out[j] = ax.mul(&by).add(&ay.mul(&bx));
        }
        out
    }
}

/// The source-determined carrier (§4.4, eq. (21)):
/// `H_src(T) = Σ λ_i P_i(T) B(z_0, z_i) + Σ_{i<j} λ_i λ_j P_ij(T) B(z_i, z_j)`
/// — a `K^y`-valued polynomial of degree `< k`, stored coefficient-wise.
#[derive(Clone, Debug)]
pub struct Carrier {
    /// `k` coefficient vectors, each in `K^y` (the carrier's coefficients).
    pub coeffs: Vec<Vec<K4>>,
}

impl Carrier {
    /// The direct pair-processing construction (§4.5's reference form).
    pub fn direct(
        params: &CauchyParams,
        q_map: &QuadraticMap,
        sources: &[Vec<K4>],
    ) -> Carrier {
        let k = params.k();
        let y = q_map.y;
        let mut coeffs = vec![vec![K4::ZERO; y]; k];
        // Add `value ⊗ poly` into the coefficient table.
        fn add_term(coeffs: &mut [Vec<K4>], poly: &KPoly, val: &[K4], y: usize) {
            for (t, pc) in poly.coeffs.iter().enumerate() {
                if t >= coeffs.len() {
                    break;
                }
                for j in 0..y {
                    coeffs[t][j] = coeffs[t][j].add(&val[j].mul(pc));
                }
            }
        }
        // Accumulator–source terms: λ_i P_i(T)·B(z_0, z_i).
        for i in 1..=k {
            let poly = params.p_poly(i - 1).scale(&params.scales[i - 1]);
            let val = q_map.polarize(&sources[0], &sources[i]);
            add_term(&mut coeffs, &poly, &val, y);
        }
        // Source–source terms: λ_i λ_j P_ij(T)·B(z_i, z_j).
        for i in 1..=k {
            for j in (i + 1)..=k {
                let scale = params.scales[i - 1].mul(&params.scales[j - 1]);
                let poly = params.p_ij_poly(i - 1, j - 1).scale(&scale);
                let val = q_map.polarize(&sources[i], &sources[j]);
                add_term(&mut coeffs, &poly, &val, y);
            }
        }
        // Trim to degree < k (P_ij has degree k−2, P_i degree k−1).
        coeffs.truncate(k);
        Carrier { coeffs }
    }

    /// The fast construction (Proposition 4.6 / Appendix A.3): product
    /// trees for `D` and the numerators, multipoint evaluation at `k`
    /// non-pole interpolation points, then interpolation. Differential
    /// tests pin it against [`Carrier::direct`].
    pub fn fast(
        params: &CauchyParams,
        q_map: &QuadraticMap,
        sources: &[Vec<K4>],
        interp_points: &[K4],
    ) -> Carrier {
        let k = params.k();
        let y = q_map.y;
        assert_eq!(interp_points.len(), k);
        // N(T) = Σ λ_i z_i P_i(T)  (K^s-valued),  M_Q(T) = Σ λ_i^2 Q(z_i) P_i(T).
        let mut n_num = vec![KPoly::zero(); q_map.s];
        let mut mq_num = vec![KPoly::zero(); y];
        for i in 0..k {
            let pi = params.p_poly(i);
            let lam = params.scales[i];
            let scaled = pi.scale(&lam);
            for l in 0..q_map.s {
                n_num[l] = n_num[l].add(&scaled.scale(&sources[i + 1][l]));
            }
            let qzi = q_map.eval(&sources[i + 1]);
            let lam2 = lam.mul(&lam);
            let scaled2 = pi.scale(&lam2);
            for j in 0..y {
                mq_num[j] = mq_num[j].add(&scaled2.scale(&qzi[j]));
            }
        }
        let d = params.d_poly();
        let d_prime = d.deriv();
        // Values at the interpolation points: h_t = D(τ)[Q(s(τ)) − Q(z_0) − e(τ)]
        // with s(τ) = z_0 + N(τ)/D(τ) and e(τ) = (M_Q D' − M_Q' D)/D²
        // (the squared-denominator term, eq. (45)).
        let mut pts: Vec<(K4, Vec<K4>)> = Vec::with_capacity(k);
        let z0q = q_map.eval(&sources[0]);
        for tau in interp_points {
            let dtau = d.eval(tau);
            let dinv = dtau.inv().expect("τ avoids the poles");
            // s(τ) = z_0 + N(τ)/D(τ)
            let mut s_tau = sources[0].clone();
            for l in 0..q_map.s {
                let nl = n_num[l].eval(tau);
                s_tau[l] = s_tau[l].add(&nl.scale(&dinv));
            }
            // e(τ) = (M_Q D' − M_Q' D)/D²  — evaluate per K^y coordinate:
            // M_Q is y separate K-polys; the formula combines them per
            // coordinate with the shared D, D'.
            let mut e_tau = vec![K4::ZERO; y];
            for j in 0..y {
                let mq = mq_num[j].eval(tau);
                let mqp = mq_num[j].deriv().eval(tau);
                let num = mq.mul(&d_prime.eval(tau)).sub(&mqp.mul(&dtau));
                let den = dtau.mul(&dtau);
                e_tau[j] = num.scale(&den.inv().expect("τ off poles"));
            }
            let q_s = q_map.eval(&s_tau);
            let mut h = vec![K4::ZERO; y];
            for j in 0..y {
                let inner = q_s[j].sub(&z0q[j]).sub(&e_tau[j]);
                h[j] = inner.scale(&dtau);
            }
            pts.push((*tau, h));
        }
        // Interpolate per K^y coordinate (degree < k).
        let mut coeffs = vec![vec![K4::ZERO; y]; k];
        for j in 0..y {
            let knot_pts: Vec<(K4, K4)> =
                pts.iter().map(|(t, h)| (*t, h[j])).collect();
            let poly = KPoly::interpolate(&knot_pts);
            for (t, c) in poly.coeffs.iter().enumerate().take(k) {
                coeffs[t][j] = *c;
            }
        }
        Carrier { coeffs }
    }

    /// The carrier as a `K^y`-valued polynomial (coefficient list).
    pub fn to_poly_rows(&self) -> Vec<KPoly> {
        let y = self.coeffs.first().map(|c| c.len()).unwrap_or(0);
        (0..y)
            .map(|j| {
                KPoly::from_coeffs(self.coeffs.iter().map(|c| c[j]).collect())
            })
            .collect()
    }

    /// `H(c)` — direct evaluation `Σ_t c^t · coeffs[t]`.
    pub fn eval(&self, c: &K4) -> Vec<K4> {
        let y = self.coeffs.first().map(|c| c.len()).unwrap_or(0);
        let mut out = vec![K4::ZERO; y];
        let mut cpow = K4::ONE;
        for coeff in &self.coeffs {
            for j in 0..y {
                out[j] = out[j].add(&coeff[j].scale(&cpow));
            }
            cpow = cpow.mul(c);
        }
        out
    }
}

/// The denominator-cleared fold `W(T) = D(T) z_0 + Σ λ_i P_i(T) z_i`
/// (§4.4) — a `K^s`-valued polynomial of degree `k`.
pub fn w_poly(params: &CauchyParams, sources: &[Vec<K4>]) -> Vec<KPoly> {
    let k = params.k();
    let s = sources[0].len();
    let d = params.d_poly();
    let mut rows = Vec::with_capacity(s);
    for l in 0..s {
        let mut row = d.scale(&sources[0][l]);
        for i in 0..k {
            let pi = params.p_poly(i).scale(&params.scales[i]);
            row = row.add(&pi.scale(&sources[i + 1][l]));
        }
        rows.push(row);
    }
    rows
}

/// The carrier identity (Proposition 4.4):
/// `Q(W(T)) = D(T)^2 Q(z_0) + Σ λ_i^2 P_i(T)^2 Q(z_i) + D(T)·H_src(T)`
/// — verified coefficient-wise.
pub fn carrier_identity_holds(
    params: &CauchyParams,
    q_map: &QuadraticMap,
    sources: &[Vec<K4>],
    carrier: &Carrier,
) -> bool {
    let k = params.k();
    let d = params.d_poly();
    let d2 = d.mul(&d);
    let w = w_poly(params, sources);
    // Q(W(T)) per K^y coordinate as polynomials: evaluate W's rows at
    // enough points and interpolate? Simpler: evaluate at k+1 points off
    // the poles and compare (both sides degree ≤ 2k + ... ≤ 2k + k = 3k
    // → use 3k+1 test points; a polynomial identity of degree ≤ 3k over K
    // vanishing at 3k+1 points is identically zero).
    let deg = 3 * k + 2;
    let mut test_pts = Vec::with_capacity(deg + 1);
    let mut t = 1u64;
    for _ in 0..deg + 1 {
        // Points away from the poles (poles are 1..k).
        t += 1;
        while (1..=k as u64).contains(&t) {
            t += 1;
        }
        test_pts.push(K4::from_coeffs([t * 1_000_003 % Q48, t * 7, t * 13, t]));
    }
    let h_rows = carrier.to_poly_rows();
    for c in &test_pts {
        // LHS: Q(W(c)).
        let wc: Vec<K4> = w.iter().map(|row| row.eval(c)).collect();
        let lhs = q_map.eval(&wc);
        // RHS.
        let qz0 = q_map.eval(&sources[0]);
        let mut rhs = vec![K4::ZERO; q_map.y];
        for j in 0..q_map.y {
            rhs[j] = rhs[j].add(&qz0[j].scale(&d2.eval(c)));
        }
        for i in 0..k {
            let qzi = q_map.eval(&sources[i + 1]);
            let pi_c = params.p_poly(i).eval(c);
            let lam = params.scales[i];
            let weight = lam.mul(&lam).mul(&pi_c).mul(&pi_c);
            for j in 0..q_map.y {
                rhs[j] = rhs[j].add(&qzi[j].scale(&weight));
            }
        }
        let d_c = d.eval(c);
        for j in 0..q_map.y {
            rhs[j] = rhs[j].add(&h_rows[j].eval(c).scale(&d_c));
        }
        if lhs != rhs {
            return false;
        }
    }
    true
}

/// The fixed-carrier discrepancy (Lemma 4.5, eq. (23)):
/// `F(T) = D²δ_0 + Σ λ_i² P_i² δ_i + D·(H_src − H)` with
/// `δ_i = Q(z_i) − E_i`.
pub fn discrepancy_poly(
    params: &CauchyParams,
    q_map: &QuadraticMap,
    sources: &[Vec<K4>],
    residuals: &[Vec<K4>],
    claimed_carrier: &Carrier,
) -> Vec<KPoly> {
    let k = params.k();
    let d = params.d_poly();
    let d2 = d.mul(&d);
    let src_carrier = Carrier::direct(params, q_map, sources);
    let mut rows = vec![KPoly::zero(); q_map.y];
    for j in 0..q_map.y {
        let delta0 = q_map.eval(&sources[0])[j].sub(&residuals[0][j]);
        rows[j] = rows[j].add(&d2.scale(&delta0));
        for i in 0..k {
            let deltai = q_map.eval(&sources[i + 1])[j].sub(&residuals[i + 1][j]);
            let pi = params.p_poly(i);
            let pi2 = pi.mul(&pi);
            let lam = params.scales[i];
            let weight = lam.mul(&lam);
            rows[j] = rows[j].add(&pi2.scale(&deltai).scale(&weight));
        }
        // D·(H_src − H) per coordinate.
        let src_row = KPoly::from_coeffs(src_carrier.coeffs.iter().map(|c| c[j]).collect());
        let claimed_row =
            KPoly::from_coeffs(claimed_carrier.coeffs.iter().map(|c| c[j]).collect());
        rows[j] = rows[j].add(&d.mul(&src_row.sub(&claimed_row)));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k4s(v: Vec<u64>) -> K4 {
        K4::from_coeffs(vec![v[0], v[1], v[2], v[3]].try_into().unwrap())
    }

    fn sources(k: usize, s: usize, seed: u64) -> Vec<Vec<K4>> {
        let mut nxt = seed;
        let mut rnd = move || {
            nxt = nxt.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            nxt >> 33
        };
        (0..k + 1)
            .map(|_| (0..s).map(|_| k4s(vec![rnd() % Q48, rnd() % Q48, rnd() % Q48, rnd() % Q48])).collect())
            .collect()
    }

    #[test]
    fn cauchy_partial_fraction_identity() {
        // a_i(c)·a_j(c) = (λ_j a_i − λ_i a_j)/(ξ_i − ξ_j)  (eq. (18)).
        let params = CauchyParams::paper(4);
        let c = k4s(vec![991, 17, 3, 5]);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                let ai = params.a_i(i, &c).unwrap();
                let aj = params.a_i(j, &c).unwrap();
                let lhs = ai.mul(&aj);
                // (λ_j a_i − λ_i a_j) / (ξ_i − ξ_j)
                let num = params
                    .scales[j]
                    .mul(&ai)
                    .sub(&params.scales[i].mul(&aj));
                let xi = params.poles[i];
                let xj = params.poles[j];
                let den = xi.sub(&xj);
                let rhs = num.scale(&den.inv().unwrap());
                assert_eq!(lhs, rhs, "partial-fraction identity at ({i},{j})");
            }
        }
    }

    #[test]
    fn carrier_identity_prop_4_4() {
        let k = 3;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(4, 2, 42);
        let srcs = sources(k, 4, 7);
        let carrier = Carrier::direct(&params, &q_map, &srcs);
        assert!(carrier_identity_holds(&params, &q_map, &srcs, &carrier));
    }

    #[test]
    fn carrier_identity_at_paper_arity() {
        let k = 8;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(3, 2, 123);
        let srcs = sources(k, 3, 99);
        let carrier = Carrier::direct(&params, &q_map, &srcs);
        assert!(carrier_identity_holds(&params, &q_map, &srcs, &carrier));
    }

    #[test]
    fn fast_carrier_matches_direct() {
        let k = 4;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(5, 3, 55);
        let srcs = sources(k, 5, 11);
        let direct = Carrier::direct(&params, &q_map, &srcs);
        // Interpolation points avoiding the poles 1..k.
        let interp: Vec<K4> = (0..k)
            .map(|i| k4s(vec![(i as u64 * 997 + 500) % Q48, i as u64 * 31 + 3, 7, 11]))
            .collect();
        let fast = Carrier::fast(&params, &q_map, &srcs, &interp);
        assert_eq!(direct.coeffs.len(), fast.coeffs.len());
        for (t, (a, b)) in direct.coeffs.iter().zip(fast.coeffs.iter()).enumerate() {
            for j in 0..q_map.y {
                assert_eq!(a[j], b[j], "carrier coefficient ({t},{j}) differs");
            }
        }
    }

    #[test]
    fn folded_record_satisfies_quadratic() {
        // Q(z*) = E* for honest sources and carrier (the §4.4 fold).
        let k = 4;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(4, 2, 9);
        let srcs = sources(k, 4, 21);
        let residuals: Vec<Vec<K4>> = srcs.iter().map(|z| q_map.eval(z)).collect();
        let carrier = Carrier::direct(&params, &q_map, &srcs);
        let c = k4s(vec![701, 1, 2, 3]);
        let z_star = params.fold_z(&srcs, &c).unwrap();
        let e_star = params.fold_e(&residuals, &carrier, &c).unwrap();
        let q_star = q_map.eval(&z_star);
        for j in 0..q_map.y {
            assert_eq!(q_star[j], e_star[j], "folded constraint {j}");
        }
    }

    #[test]
    fn lemma_4_5_discrepancy_soundness() {
        // An inconsistent carrier or a wrong residual gives F(T) ≠ 0 of
        // degree ≤ 2k, vanishing at ≤ 2k challenge points.
        let k = 3;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(4, 2, 4);
        let srcs = sources(k, 4, 13);
        let residuals: Vec<Vec<K4>> = srcs.iter().map(|z| q_map.eval(z)).collect();
        let carrier = Carrier::direct(&params, &q_map, &srcs);
        // Consistent case: F ≡ 0.
        let f_ok = discrepancy_poly(&params, &q_map, &srcs, &residuals, &carrier);
        for row in &f_ok {
            assert!(row.coeffs.iter().all(|c| c.is_zero()) || row.degree() == 0 && row.eval(&K4::ZERO).is_zero());
        }
        // Tamper the carrier: F ≠ 0, degree ≤ 2k.
        let mut bad_carrier = carrier.clone();
        bad_carrier.coeffs[0][0] = bad_carrier.coeffs[0][0].add(&K4::ONE);
        let f_bad = discrepancy_poly(&params, &q_map, &srcs, &residuals, &bad_carrier);
        let nonzero = f_bad
            .iter()
            .any(|row| row.coeffs.iter().any(|c| !c.is_zero()));
        assert!(nonzero, "tampered carrier must give nonzero F");
        for row in &f_bad {
            assert!(row.degree() <= 2 * k, "deg F ≤ 2k");
        }
        // Count roots over a window of challenge values: ≤ 2k.
        let mut roots = 0;
        for t in (k + 1)..=(k + 1 + 4 * k + 8) {
            let c = K4::from_coeffs([t as u64, 2, 3, 5]);
            if f_bad.iter().all(|row| row.eval(&c).is_zero()) {
                roots += 1;
            }
        }
        assert!(roots <= 2 * k, "F vanishes at ≤ 2k points");
    }

    #[test]
    fn wrong_residual_detected() {
        let k = 3;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(4, 2, 6);
        let srcs = sources(k, 4, 31);
        let mut residuals: Vec<Vec<K4>> = srcs.iter().map(|z| q_map.eval(z)).collect();
        let carrier = Carrier::direct(&params, &q_map, &srcs);
        residuals[2][0] = residuals[2][0].add(&K4::ONE);
        let f = discrepancy_poly(&params, &q_map, &srcs, &residuals, &carrier);
        assert!(f.iter().any(|row| row.coeffs.iter().any(|c| !c.is_zero())));
    }
}

