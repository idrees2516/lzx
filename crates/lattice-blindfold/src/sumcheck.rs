//! The Libra-style masked Sum-Check (§3.4, Protocol 5) instantiated over
//! K with the paper's exact composition (Protocol 6, Steps 3–8):
//!
//! ```text
//! Q(X⃗) = eq(X⃗,α)·(F(X⃗) + δ₀·NC(X⃗)) + δ₁·Eval(X⃗)   ∈ K[X⃗]
//! ```
//!
//! * **F(X⃗) = Σ_{i≤K} γ_i·[ct(U_{i,2})·ct(U_{i,3}) − ct(U_{i,1})]** —
//!   the R1CS term over the constant coefficients of the ring-valued
//!   multilinear extensions U_{i,j} = (M̄_j z_i)(X⃗) (§2.4.1: the field MLE
//!   (M_j z)^(X⃗) is exactly ct(U_{i,j}));
//! * **NC(X⃗) = Σ_{i≤K+k} γ_i·Π_{j=−(b−1)}^{b−1}(ẑ_i(X⃗) − j)** — the
//!   norm-check over the field MLE ẑ_i of the witness (degree 2b−1);
//! * **Eval(X⃗) = eq(X⃗, r)·Σ_{i>K,j,ℓ} γγγ·cf(U_{i,j})_ℓ(X⃗)** — the
//!   carried-over evaluation claims through the coefficient-slot MLEs.
//!
//! The mask **p(X⃗) = a₀ + Σ_i p̃_i(X_i)** (degree-Dmax univariates with
//! uniform coefficients) blinds every round message; the claimed sum
//! σ = ζP + T perfectly hides T (Remark 4.2.(7): P = 2^ℓa₀ +
//! 2^{ℓ−1}Σa_{i,j} is uniform, so σ is uniform regardless of T). The
//! final value h = (ζp + Q)(r′) is public; p(r′) and Q(r′) never are.
//!
//! Round polys have degree Dmax = max{u+1, 2b, 2} per variable (§2.1);
//! the prover evaluates the combined polynomial at the Dmax+1 points
//! 0..Dmax and interpolates (Lagrange over K).

use crate::embed::{EqArray, RingMle};
use crate::fp::Fq;
use crate::fq2::K;

/// The combined Sum-Check statement pieces (Protocol 6, Step 3).
pub struct SumcheckInputs {
    /// Ring-valued MLEs U_{i,j} = M̄_j z_i for i ∈ [K+k], j ∈ [t]
    /// (kept as cube arrays; bound progressively per round).
    pub ring_mles: Vec<Vec<RingMle>>, // [instance][matrix]
    /// Field MLEs ẑ_i for i ∈ [K+k].
    pub field_mles: Vec<Vec<K>>, // [instance][cube entry]
    /// The eq(·, α) full array.
    pub eq_alpha: Vec<K>,
    /// The eq(·, r) full array.
    pub eq_r: Vec<K>,
    /// Batching challenges γ^(1) ∈ K^{K+k}.
    pub gamma1: Vec<K>,
    /// δ₀, δ₁ ∈ K.
    pub delta0: K,
    pub delta1: K,
    /// Dmax (per-variable degree of Q).
    pub d_max: usize,
    /// The norm bound b (for the NC product range).
    pub b: i64,
    /// K (number of fresh instances) — F uses i ≤ K only.
    pub capital_k: usize,
    /// Pre-weighted coefficient-slot challenges γ^(2)·γ^(3) per
    /// (instance > K, matrix): the packaged-rotation weight vectors for
    /// the Eval term (set via `set_eval_weights`).
    pub eval_weights: Option<Vec<Vec<Vec<K>>>>,
}

/// The mask p = a₀ + Σ p̃_i(X_i) with uniform coefficients.
#[derive(Clone, Debug)]
pub struct LibraMask {
    pub a0: K,
    /// p̃_i coefficients [j=1..Dmax] of X^j.
    pub coef: Vec<Vec<K>>, // [var][j-1]
    pub d_max: usize,
}

impl LibraMask {
    pub fn sample(log_len: usize, d_max: usize, rng: &mut crate::gauss::Rng) -> LibraMask {
        let mut coef = Vec::with_capacity(log_len);
        for _ in 0..log_len {
            let mut c = Vec::with_capacity(d_max);
            for _ in 0..d_max {
                c.push(K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                ));
            }
            coef.push(c);
        }
        LibraMask {
            a0: K(
                Fq(rng.next_u64() % crate::fp::Q),
                Fq(rng.next_u64() % crate::fp::Q),
            ),
            coef,
            d_max,
        }
    }

    /// P = Σ_{x∈cube} p(x) = 2^ℓ·a₀ + 2^{ℓ−1}·Σ_{i,j} a_{i,j}
    /// (§4.1.1.1).
    pub fn cube_sum(&self) -> K {
        let ell = self.coef.len();
        let two_ell = K::from_fp(Fq::new(1u64 << ell.min(62)));
        let two_ellm1 = K::from_fp(Fq::new(1u64 << (ell - 1).min(62)));
        let mut sum_ij = K::ZERO;
        for c in &self.coef {
            for a in c {
                sum_ij = sum_ij.add(a);
            }
        }
        two_ell.mul(&self.a0).add(&two_ellm1.mul(&sum_ij))
    }

    /// p̃_i evaluated at X_i = v — with the FULL K-scalar coefficients
    /// (both {1, Y} components).
    pub fn univariate_eval(&self, var: usize, v: &K) -> K {
        let mut acc = K::ZERO;
        let mut vp = K::ONE;
        for c in &self.coef[var] {
            vp = vp.mul(v);
            acc = acc.add(&vp.mul(c));
        }
        acc
    }

    /// p(r⃗) = a₀ + Σ_i p̃_i(r_i).
    pub fn eval(&self, point: &[K]) -> K {
        let mut acc = self.a0;
        for (i, r) in point.iter().enumerate() {
            acc = acc.add(&self.univariate_eval(i, r));
        }
        acc
    }
}

/// The masked Sum-Check transcript.
#[derive(Clone, Debug)]
pub struct MaskedSumcheckTranscript {
    /// σ = ζP + T (the claimed sum, Step 6).
    pub sigma: K,
    /// Per-round coefficient vectors H_i(X) = Σ_{j=0..Dmax} h_j X^j
    /// (mask included), Dmax+1 coefficients each.
    pub rounds: Vec<Vec<K>>,
    /// The verifier's challenges r_i.
    pub challenges: Vec<K>,
    /// The final value h = H_ℓ(r_ℓ) = (ζp + Q)(r′).
    pub final_h: K,
    /// ζ ∈ K^× (the blinding challenge, Step 5).
    pub zeta: K,
}

/// The masked Sum-Check engine. `t_claim` is the Eval part of the claimed
/// sum (T of Protocol 6, Step 3's definition).
pub struct MaskedSumcheck {
    pub log_len: usize,
}

impl MaskedSumcheck {
    /// Prove with an internally-sampled mask (Protocol 5's flow).
    pub fn prove(
        inputs: &SumcheckInputs,
        t_claim: K,
        rng: &mut crate::gauss::Rng,
    ) -> Result<(MaskedSumcheckTranscript, Vec<K>), String> {
        let log_len = inputs.field_mles[0].len().trailing_zeros() as usize;
        let mask = LibraMask::sample(log_len, inputs.d_max, rng);
        let zeta = loop {
            let z = K(
                Fq(rng.next_u64() % (crate::fp::Q - 1) + 1),
                Fq(rng.next_u64() % crate::fp::Q),
            );
            if z.inverse().is_some() {
                break z;
            }
        };
        Self::prove_with_mask(inputs, t_claim, &mask, zeta, rng)
    }

    /// Prove with a pre-sampled mask and ζ (the Protocol-6 flow: the mask
    /// is committed BEFORE ζ is drawn — Step 4 before Step 5).
    pub fn prove_with_mask(
        inputs: &SumcheckInputs,
        t_claim: K,
        mask: &LibraMask,
        zeta: K,
        rng: &mut crate::gauss::Rng,
    ) -> Result<(MaskedSumcheckTranscript, Vec<K>), String> {
        let log_len = inputs.field_mles[0].len().trailing_zeros() as usize;
        let d_max = inputs.d_max;
        // σ = ζP + T.
        let sigma = zeta.mul(&mask.cube_sum()).add(&t_claim);
        // Working state: progressively bound arrays.
        let mut ring: Vec<Vec<Vec<crate::rk::PolyK>>> = inputs
            .ring_mles
            .iter()
            .map(|per_inst| per_inst.iter().map(|m| m.cube.clone()).collect())
            .collect();
        let mut field: Vec<Vec<K>> = inputs.field_mles.clone();
        let mut eqa = inputs.eq_alpha.clone();
        let mut eqr = inputs.eq_r.clone();
        let mut claim = sigma;
        let mut rounds: Vec<Vec<K>> = Vec::with_capacity(log_len);
        let mut challenges = Vec::with_capacity(log_len);
        let gamma1 = &inputs.gamma1;
        let nk = inputs.capital_k + gamma1.len().min(inputs.field_mles.len());
        let _ = nk;
        for round in 0..log_len {
            // Evaluate the combined (Q + ζp)-round polynomial at
            // X_round ∈ {0, 1, 2, 3, 4} (Dmax+1 points; Dmax ≤ 4 here —
            // the paper's parameter regime).
            let npts = d_max + 1;
            let mut pts: Vec<K> = Vec::with_capacity(npts);
            for v in 0..npts {
                let xv = K::from_fp(Fq::new(v as u64));
                // Bind all arrays at xv.
                let bound_ring: Vec<Vec<Vec<crate::rk::PolyK>>> = ring
                    .iter()
                    .map(|per_inst| {
                        per_inst
                            .iter()
                            .map(|cube| bind_polyk_cube(cube, &xv))
                            .collect()
                    })
                    .collect();
                let bound_field: Vec<Vec<K>> = field.iter().map(|f| bind_k_cube(f, &xv)).collect();
                let beqa = EqArray::bind(&eqa, &xv);
                let beqr = EqArray::bind(&eqr, &xv);
                // Sum over the suffix cube:
                let total = sum_q_suffix(inputs, &bound_ring, &bound_field, &beqa, &beqr, gamma1);
                pts.push(total);
            }
            // Interpolate the degree-Dmax poly through the points.
            let coeffs = lagrange_coeffs(&pts, d_max)?;
            // Add the mask's round contribution. The round-i message is
            // H_i(X) = Σ_{suffix} (ζp + Q)(prefix, X, suffix), and the
            // mask's suffix-sum decomposes as:
            //   suffix_len·(a0 + Σ_{i'<i} p̃_{i'}(r_{i'}))   — the constant
            //       and the already-bound rounds' values, once per suffix
            //       point;
            //   + Σ_{i'>i} (suffix_len/2)·Σ_j a_{i',j}        — each FUTURE
            //       round's univariate cube-sum;
            //   + p̃_i(X)                                        — this round.
            let mut h = coeffs;
            let suffix_len = 1usize << (log_len - round - 1);
            let mut const_term = mask.a0.scale_fp(&Fq::new(suffix_len as u64));
            for past in 0..round {
                let pv = mask.univariate_eval(past, &challenges[past]);
                const_term = const_term.add(&pv.scale_fp(&Fq::new(suffix_len as u64)));
            }
            for future in (round + 1)..log_len {
                let per: u64 = if suffix_len >= 2 {
                    (suffix_len / 2) as u64
                } else {
                    1
                };
                let sum_ij: K = mask.coef[future].iter().fold(K::ZERO, |a, c| a.add(c));
                const_term = const_term.add(&sum_ij.scale_fp(&Fq::new(per)));
            }
            h[0] = h[0].add(&zeta.mul(&const_term));
            // This round's univariate carries the same suffix
            // multiplicity (§4.1.1.1: "the coefficients of p̃_i acting on
            // the non-constant coefficients of h_i by 2^{ℓ−i}").
            let sl = Fq::new(suffix_len as u64);
            for (j, c) in mask.coef[round].iter().enumerate() {
                h[j + 1] = h[j + 1].add(&zeta.mul(&c.scale_fp(&sl)));
            }
            // Verifier-side consistency (checked by the caller's verify):
            // H(0) + H(1) = claim.
            let h0 = eval_poly(&h, &K::ZERO);
            let h1 = eval_poly(&h, &K::ONE);
            let got = h0.add(&h1);
            if got != claim {
                return Err(format!(
                    "round {round}: H(0)+H(1) = {got:?} ≠ claim {claim:?}"
                ));
            }
            rounds.push(h.clone());
            // The verifier's challenge r_round ← K.
            let r = K(
                Fq(rng.next_u64() % crate::fp::Q),
                Fq(rng.next_u64() % crate::fp::Q),
            );
            challenges.push(r);
            claim = eval_poly(&h, &r);
            // Bind the persistent state.
            for per_inst in ring.iter_mut() {
                for cube in per_inst.iter_mut() {
                    *cube = bind_polyk_cube(cube, &r);
                }
            }
            for f in field.iter_mut() {
                *f = bind_k_cube(f, &r);
            }
            eqa = EqArray::bind(&eqa, &r);
            eqr = EqArray::bind(&eqr, &r);
        }
        // Final: the last round's H_ℓ evaluated at r_ℓ is the running
        // claim; the final value h = (ζp + Q)(r′).
        let final_h = claim;
        let tr = MaskedSumcheckTranscript {
            sigma,
            rounds,
            challenges: challenges.clone(),
            final_h,
            zeta,
        };
        let _ = mask;
        Ok((tr, challenges))
    }

    /// Verify: per-round H_i(0) + H_i(1) = claim; returns the final h.
    pub fn verify(tr: &MaskedSumcheckTranscript) -> Result<K, String> {
        let mut claim = tr.sigma;
        for (i, h) in tr.rounds.iter().enumerate() {
            if h.len() < 2 {
                return Err("round too short".into());
            }
            let h0 = eval_poly(h, &K::ZERO);
            let h1 = eval_poly(h, &K::ONE);
            if h0.add(&h1) != claim {
                return Err(format!("round {i}: sum mismatch"));
            }
            claim = eval_poly(h, &tr.challenges[i]);
        }
        if claim != tr.final_h {
            return Err("final value mismatch".into());
        }
        Ok(tr.final_h)
    }
}

fn bind_polyk_cube(cube: &[crate::rk::PolyK], r: &K) -> Vec<crate::rk::PolyK> {
    let half = cube.len() / 2;
    (0..half)
        .map(|i| {
            // (1−r)·lo + r·hi — the K-scaled ring combination.
            let lo = &cube[i];
            let hi = &cube[half + i];
            lo.scale_k(&K::ONE.sub(r)).add(&hi.scale_k(r))
        })
        .collect()
}

fn bind_k_cube(arr: &[K], r: &K) -> Vec<K> {
    EqArray::bind(arr, r)
}

/// Σ_suffix eqα·(F + δ₀NC) + δ₁·eqr·Eval over the bound arrays.
fn sum_q_suffix(
    inputs: &SumcheckInputs,
    ring: &[Vec<Vec<crate::rk::PolyK>>],
    field: &[Vec<K>],
    eqa: &[K],
    eqr: &[K],
    gamma1: &[K],
) -> K {
    let d = inputs.b;
    let mut total = K::ZERO;
    let suffix_len = eqa.len();
    let t = ring.first().map(|v| v.len()).unwrap_or(0);
    let capital_k = inputs.capital_k;
    for s in 0..suffix_len {
        if eqa[s].is_zero() && eqr[s].is_zero() {
            continue;
        }
        // F-part (i ≤ K): γ_i·[ct(U_{i,2})·ct(U_{i,3}) − ct(U_{i,1})]
        let mut f_part = K::ZERO;
        for (i, gi) in gamma1.iter().enumerate().take(capital_k) {
            if gi.is_zero() || i >= ring.len() || t < 3 {
                continue;
            }
            let u1 = ring[i][0][s].ct();
            let u2 = ring[i][1.min(t - 1)][s].ct();
            let u3 = ring[i][2.min(t - 1)][s].ct();
            f_part = f_part.add(&gi.mul(&u2.mul(&u3).sub(&u1)));
        }
        // NC-part (i ≤ K+k): γ_i·Π_{j=−(b−1)}^{b−1}(ẑ_i − j)
        let mut nc_part = K::ZERO;
        for (i, gi) in gamma1.iter().enumerate() {
            if gi.is_zero() || i >= field.len() {
                continue;
            }
            let z = field[i][s];
            let mut prod = K::ONE;
            for j in -(d - 1)..=d - 1 {
                prod = prod.mul(&z.sub(&K::from_i64(j)));
            }
            nc_part = nc_part.add(&gi.mul(&prod));
        }
        // Eval-part: eqr·Σ_{i>K,j,ℓ} γγγ·cf(U_{i,j})_ℓ — coefficient-slot
        // sums per (i, j). The γ^(2)/γ^(3) challenges enter through the
        // caller's ring_mles pre-weighting: the engine folds them in via
        // eval_weights (below) to keep this loop allocation-free.
        let mut eval_core = K::ZERO;
        if let Some(ew) = inputs.eval_weights.as_ref() {
            for (inst_idx, per_inst) in ew.iter().enumerate() {
                if inst_idx >= ring.len() {
                    continue;
                }
                for (mat_idx, w) in per_inst.iter().enumerate() {
                    if mat_idx >= ring[inst_idx].len() || w.iter().all(|x| x.is_zero()) {
                        continue;
                    }
                    let val = ring[inst_idx][mat_idx][s].ct();
                    // Σ_ℓ γℓ·cf(U)_ℓ — the full coefficient sum; the
                    // ring value's coefficient-wise weighted sum is
                    // realized by the packaged rotation ρ = Σγℓτℓ:
                    let rho = crate::rk::PolyK::packaged_rotation(w);
                    let rot = crate::rk::PolyK::rotated_ct(&rho, &ring[inst_idx][mat_idx][s]);
                    let _ = val;
                    eval_core = eval_core.add(&rot);
                }
            }
        }
        let q_s = eqa[s]
            .mul(&f_part.add(&inputs.delta0.mul(&nc_part)))
            .add(&inputs.delta1.mul(&eqr[s]).mul(&eval_core));
        total = total.add(&q_s);
    }
    total
}

impl SumcheckInputs {
    /// Pre-weight the coefficient slots: γ^(2)_j·γ^(3)_ℓ per (j, ℓ),
    /// assembled as the packaged-rotation weight vectors for the Eval
    /// term (Eq: cf(M̄_jz_i)_ℓ MLEs with the tensored challenges).
    pub fn set_eval_weights(&mut self, gamma2: &[K], gamma3: &[K]) {
        let d_ring = self
            .ring_mles
            .first()
            .and_then(|v| v.first())
            .and_then(|m| m.cube.first())
            .map(|c| c.d())
            .unwrap_or(0);
        // eval_weights[instance][matrix] = the d K-weights (0 for
        // instances ≤ K — the Eval term sums over i > K only).
        let n_inst = self.ring_mles.len();
        let n_mat = self.ring_mles.first().map(|v| v.len()).unwrap_or(0);
        let mut ew = vec![vec![vec![K::ZERO; d_ring]; n_mat]; n_inst];
        for i in self.capital_k..n_inst {
            let gi = self.gamma1.get(i).copied().unwrap_or(K::ZERO);
            if gi.is_zero() {
                continue;
            }
            for j in 0..n_mat {
                let gj = gamma2.get(j).copied().unwrap_or(K::ZERO);
                if gj.is_zero() {
                    continue;
                }
                for l in 0..d_ring {
                    let gl = gamma3.get(l).copied().unwrap_or(K::ZERO);
                    // The full tensored challenge γ_i^(1)·γ_j^(2)·γ_ℓ^(3).
                    ew[i][j][l] = gi.mul(&gj).mul(&gl);
                }
            }
        }
        self.eval_weights = Some(ew);
    }
}

/// Lagrange interpolation through points (v, Q(v)) for v = 0..npts−1,
/// returning coefficients of a degree ≤ npts−1 polynomial.
fn lagrange_coeffs(pts: &[K], d_max: usize) -> Result<Vec<K>, String> {
    let n = pts.len();
    if n != d_max + 1 {
        return Err("point count mismatch".into());
    }
    // Solve the Vandermonde system by elimination (n ≤ 5).
    let mut mat: Vec<Vec<K>> = Vec::with_capacity(n);
    for r in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        let mut pw = K::ONE;
        for _ in 0..n {
            row.push(pw);
            pw = pw.mul(&K::from_fp(Fq::new(r as u64)));
        }
        row.push(pts[r]);
        mat.push(row);
    }
    // Gaussian elimination.
    for col in 0..n {
        // pivot
        let mut piv = None;
        for r in col..n {
            if !mat[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        let p = piv.ok_or("singular interpolation")?;
        mat.swap(col, p);
        let inv = mat[col][col].inverse().ok_or("singular interpolation")?;
        for c in col..=n {
            mat[col][c] = mat[col][c].mul(&inv);
        }
        for r in 0..n {
            if r != col && !mat[r][col].is_zero() {
                let f = mat[r][col];
                for c in col..=n {
                    mat[r][c] = mat[r][c].sub(&f.mul(&mat[col][c]));
                }
            }
        }
    }
    Ok((0..n).map(|r| mat[r][n]).collect())
}

fn eval_poly(coeffs: &[K], x: &K) -> K {
    let mut acc = K::ZERO;
    let mut pw = K::ONE;
    for c in coeffs {
        acc = acc.add(&pw.mul(c));
        pw = pw.mul(x);
    }
    acc
}

// The eval_weights field lives on SumcheckInputs (declared with the
// struct above); set_eval_weights assembles the tensored challenges.
impl SumcheckInputs {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::FieldVec;
    use crate::gauss::Rng;

    fn build_inputs(seed: &[u8], log_len: usize) -> (SumcheckInputs, K, Vec<FieldVec>) {
        let mut rng = Rng::new(seed);
        let m = 1usize << log_len;
        let d = 4usize;
        let capital_k = 1usize;
        let k_acc = 1usize;
        let nk = capital_k + k_acc;
        let t = 3usize;
        // Random small witnesses.
        let zs: Vec<FieldVec> = (0..nk)
            .map(|i| FieldVec::small_b(m, 2, format!("z{i}").as_bytes(), &mut gct()))
            .collect();
        // Ring MLEs: fake M̄_j z_i as random ring vectors (the algebra is
        // exercised end-to-end in protocol tests; here the engine's
        // arithmetic is under test).
        let mut ring_mles: Vec<Vec<RingMle>> = Vec::with_capacity(nk);
        for i in 0..nk {
            let mut per = Vec::with_capacity(t);
            for j in 0..t {
                let cube: Vec<crate::rk::PolyK> = (0..m)
                    .map(|x| {
                        crate::rk::PolyK::from_poly(crate::ring::Poly::small_b(
                            d,
                            2,
                            format!("u{i}{j}{x}").as_bytes(),
                            &mut gct(),
                        ))
                    })
                    .collect();
                per.push(RingMle { cube, log_len });
            }
            ring_mles.push(per);
        }
        let field_mles: Vec<Vec<K>> = zs
            .iter()
            .map(|z| z.0.iter().map(|&c| K::from_fp(c)).collect())
            .collect();
        let alpha: Vec<K> = (0..log_len)
            .map(|_| {
                K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                )
            })
            .collect();
        let r: Vec<K> = (0..log_len)
            .map(|_| {
                K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                )
            })
            .collect();
        let eq_alpha = EqArray::full(log_len, &alpha);
        let eq_r = EqArray::full(log_len, &r);
        let gamma1: Vec<K> = (0..nk)
            .map(|_| {
                K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                )
            })
            .collect();
        let mut inputs = SumcheckInputs {
            ring_mles,
            field_mles,
            eq_alpha,
            eq_r,
            gamma1,
            delta0: K::from_fp(Fq::new(5)),
            delta1: K::from_fp(Fq::new(7)),
            d_max: 4,
            b: 2,
            capital_k,
            eval_weights: None,
        };
        let gamma2: Vec<K> = (0..t)
            .map(|_| {
                K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                )
            })
            .collect();
        let gamma3: Vec<K> = (0..d)
            .map(|_| {
                K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                )
            })
            .collect();
        inputs.set_eval_weights(&gamma2, &gamma3);
        // T: the Eval part — Σ_{i>K} γγγ·cf(U_{i,j})_ℓ at the point r
        // is what an honest prover computes; for the engine test we use
        // a random T and only check protocol consistency (the R1CS/Nc
        // parts vanish only for consistent inputs — protocol tests cover
        // the honest case; here the sumcheck must still verify).
        let t_claim = K::from_fp(Fq(rng.next_u64() % crate::fp::Q));
        (inputs, t_claim, zs)
    }

    fn gct() -> u64 {
        use std::cell::Cell;
        thread_local! {
            static C: Cell<u64> = const { Cell::new(0) };
        }
        C.with(|c| {
            let v = c.get() + 1;
            c.set(v);
            v
        })
    }

    #[test]
    fn masked_sumcheck_prove_verify() {
        for log_len in [2usize, 3usize] {
            let (inputs, _t_claim, _) = build_inputs(b"sc", log_len);
            let mut rng = Rng::new(b"sc-prove");
            // Note: with random ring MLEs the F/NC parts do NOT vanish on
            // the cube, so T must include them: compute the true total.
            let true_total = compute_total(&inputs);
            let (tr, _) = MaskedSumcheck::prove(&inputs, true_total, &mut rng).unwrap();
            let h = MaskedSumcheck::verify(&tr).unwrap();
            assert_eq!(h, tr.final_h);
        }
    }

    fn compute_total(inputs: &SumcheckInputs) -> K {
        // Σ_x Q(x) directly.
        let n = inputs.eq_alpha.len();
        let mut acc = K::ZERO;
        for x in 0..n {
            let ea = inputs.eq_alpha[x];
            let er = inputs.eq_r[x];
            // F/NC/Eval at cube point:
            let mut f = K::ZERO;
            for (i, gi) in inputs.gamma1.iter().enumerate().take(inputs.capital_k) {
                let u1 = inputs.ring_mles[i][0].cube[x].ct();
                let u2 = inputs.ring_mles[i][1].cube[x].ct();
                let u3 = inputs.ring_mles[i][2].cube[x].ct();
                f = f.add(&gi.mul(&u2.mul(&u3).sub(&u1)));
            }
            let mut nc = K::ZERO;
            for (i, gi) in inputs.gamma1.iter().enumerate() {
                let z = inputs.field_mles[i][x];
                let mut prod = K::ONE;
                for j in -(inputs.b - 1)..=inputs.b - 1 {
                    prod = prod.mul(&z.sub(&K::from_i64(j)));
                }
                nc = nc.add(&gi.mul(&prod));
            }
            let mut ec = K::ZERO;
            if let Some(ew) = inputs.eval_weights.as_ref() {
                for (i, per) in ew.iter().enumerate() {
                    for (j, w) in per.iter().enumerate() {
                        if w.iter().all(|w| w.is_zero()) {
                            continue;
                        }
                        let rho = crate::rk::PolyK::packaged_rotation(w);
                        ec = ec.add(&crate::rk::PolyK::rotated_ct(
                            &rho,
                            &inputs.ring_mles[i][j].cube[x],
                        ));
                    }
                }
            }
            acc = acc.add(
                &ea.mul(&f.add(&inputs.delta0.mul(&nc)))
                    .add(&inputs.delta1.mul(&er).mul(&ec)),
            );
        }
        acc
    }

    #[test]
    fn libra_mask_cube_sum_formula() {
        // P = 2^ℓ·a₀ + 2^{ℓ−1}·Σ a_{i,j} — verify against direct summation.
        let mut rng = Rng::new(b"mask");
        let mask = LibraMask::sample(3, 4, &mut rng);
        let mut direct = K::ZERO;
        for x in 0..8 {
            let pt = vec![
                K::from_fp(Fq::new((x & 1) as u64)),
                K::from_fp(Fq::new(((x >> 1) & 1) as u64)),
                K::from_fp(Fq::new(((x >> 2) & 1) as u64)),
            ];
            direct = direct.add(&mask.eval(&pt));
        }
        assert_eq!(direct, mask.cube_sum());
    }

    #[test]
    fn tampered_sumcheck_rejected() {
        let (inputs, _, _) = build_inputs(b"sc-t", 2);
        let mut rng = Rng::new(b"sc-t-prove");
        let true_total = compute_total(&inputs);
        let (mut tr, _) = MaskedSumcheck::prove(&inputs, true_total, &mut rng).unwrap();
        // Tamper a round coefficient.
        tr.rounds[0][0] = tr.rounds[0][0].add(&K::ONE);
        assert!(MaskedSumcheck::verify(&tr).is_err());
    }
}
