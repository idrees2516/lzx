//! The module-valued sumcheck — the paper's Lemma 3.1 ("sumcheck over G")
//! instantiated with `G = M = (R_q)^{rows}` an `F_q`-vector space.
//!
//! The summed polynomial has **coefficients in the module** rather than the
//! field; both accordion protocols run it with individual degree `d = 2`:
//!
//! * `reduce`: `A(X) = Ŵ(X)·Ĝ(X) + T(X)·Ŵ(X)·P'` with target
//!   `C = cm + v·P'` (the layered-cube form of the paper's
//!   `f̂(X)G(X) + eq(X,z)f̂(X)P'`), and
//! * `accumulate`: `A(X) = Ĝ(X)·e(X)` with target `C = Σᵢ γⁱ Cᵢ`.
//!
//! Both have the shape `S(X)·Ĝ(X) + T(X)·Ŵ(X)·P'` (accumulate takes
//! `S = e`, `T = 0`, `W` unused), so one engine serves both: per round over
//! the first remaining variable `X_i`,
//!
//! ```text
//! c₀ = Σ_tail W_lo·G_lo + (Σ_tail T_lo·W_lo)·P'
//! c₁ = Σ_tail [W_lo·(G_hi−G_lo) + (W_hi−W_lo)·G_lo]
//!      + (Σ_tail [T_lo·(W_hi−W_lo) + (T_hi−T_lo)·W_lo])·P'
//! c₂ = Σ_tail (W_hi−W_lo)·(G_hi−G_lo) + (Σ_tail (T_hi−T_lo)·(W_hi−W_lo))·P'
//! ```
//!
//! the round message being the three module points `[c₀, c₁, c₂]` — the
//! paper's "3k G-elements". The verifier checks degree ≤ 2 plus the
//! recurrence `A₁(0)+A₁(1) = C`, `Aᵢ₊₁(0)+Aᵢ₊₁(1) = Aᵢ(rᵢ)`, and outputs
//! the terminal `(r, V = A_k(r_k))`.
//!
//! Soundness (Lemma 3.1): over a module that is an `F_q`-vector space, the
//! standard round-by-round argument gives error `d·m/q` — here
//! `2m/q ≤ 2^{-45}` for the cube sizes used (m ≤ 16).
//!
//! Restriction/folding after each challenge: every table (scalar `W`, `T`
//! and module `G`) folds as `lo + r·(hi − lo)` — the multilinear
//! restriction — so prover work telescopes to `O(N)` module-scalar
//! operations across the whole protocol, matching the paper's prover cost.

use crate::module::{eq_eval_index, Fq, ModulePoint, Srs};

/// Per-round message: the three coefficients of the degree-≤2 module-valued
/// univariate `A_i(X) = c₀ + c₁ X + c₂ X²`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundMessage(pub [ModulePoint; 3]);

impl RoundMessage {
    /// Evaluate `A_i(x)`.
    pub fn eval(&self, x: &Fq) -> ModulePoint {
        let x2 = x.mul(x);
        let mut out = self.0[0].clone();
        out = out.axpy(x, &self.0[1]);
        out = out.axpy(&x2, &self.0[2]);
        out
    }

    /// `A_i(0) + A_i(1)` — the sum the verifier's recurrence checks. For
    /// `A_i(X) = c₀ + c₁X + c₂X²` this is `2c₀ + c₁ + c₂` (evaluating at
    /// `X ∈ {0, 1}` and summing over the two-point cube).
    pub fn cube_sum(&self) -> ModulePoint {
        let mut out = self.0[0].clone();
        out = out.add(&self.0[0]);
        out = out.add(&self.0[1]);
        out = out.add(&self.0[2]);
        out
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for c in &self.0 {
            out.extend_from_slice(&c.to_bytes());
        }
        out
    }
}

/// The summand structure `A(X) = Ŵ(X)·Ĝ(X) + T(X)·Ŵ(X)·P'` over the current
/// restriction of the layered cube.
///
/// The prover maintains three tables restricted to the partial challenge
/// point: the scalar witness table `W` (length = remaining cube size), the
/// module generator table `G` (one `ModulePoint` per remaining index), and
/// the scalar factor table `T`. `accumulate` runs with `W` empty and `T`
/// zero (its summand is just `e·Ĝ`).
#[derive(Clone)]
pub struct SummandTables {
    pub w: Vec<Fq>,
    pub g: Vec<ModulePoint>,
    pub t: Vec<Fq>,
    /// The fixed module element `P' = α·P` (None for accumulate).
    pub p_prime: Option<ModulePoint>,
}

impl SummandTables {
    /// Build the reduce-side tables: witness layers, generators, T-factor.
    pub fn for_reduce(witness: &[Fq], srs: &Srs, t_table: &[Fq], p_prime: ModulePoint) -> Self {
        let g: Vec<ModulePoint> = (0..witness.len())
            .map(|b| srs.generator(b).clone())
            .collect();
        SummandTables {
            w: witness.to_vec(),
            g,
            t: t_table.to_vec(),
            p_prime: Some(p_prime),
        }
    }

    /// Build the accumulate-side tables: generators and the `e` factor.
    pub fn for_accumulate(srs: &Srs, e_table: &[Fq]) -> Self {
        let g: Vec<ModulePoint> = (0..e_table.len())
            .map(|b| srs.generator(b).clone())
            .collect();
        SummandTables {
            w: Vec::new(),
            g,
            t: e_table.to_vec(),
            p_prime: None,
        }
    }

    /// Compute the round message for the first remaining variable.
    ///
    /// `lo` = entries with the variable set to 0 (the table's first half),
    /// `hi` = the second half — MSB-first bit layout.
    pub fn round_message(&self) -> RoundMessage {
        let half = self.g.len() / 2;
        let dim = self.g.first().map(|g| g.dim()).unwrap_or(0);
        let mut c0 = ModulePoint::zero(dim);
        let mut c1 = ModulePoint::zero(dim);
        let mut c2 = ModulePoint::zero(dim);
        let mut s0 = Fq::ZERO; // Σ T_lo·W_lo
        let mut s1 = Fq::ZERO; // Σ [T_lo·(W_hi−W_lo) + (T_hi−T_lo)·W_lo]
        let mut s2 = Fq::ZERO; // Σ (T_hi−T_lo)·(W_hi−W_lo)
        let use_w = !self.w.is_empty();
        for t in 0..half {
            let g_lo = &self.g[t];
            let g_hi = &self.g[t + half];
            // ΔG = G_hi − G_lo
            let mut dg = vec![Fq::ZERO; dim];
            for (d, (a, b)) in dg.iter_mut().zip(g_lo.0.iter().zip(g_hi.0.iter())) {
                *d = b.sub(a);
            }
            if use_w {
                let w_lo = self.w[t];
                let w_hi = self.w[t + half];
                let dw = w_hi.sub(&w_lo);
                let t_lo = self.t[t];
                let t_hi = self.t[t + half];
                let dt = t_hi.sub(&t_lo);
                for (c, gl) in c0.0.iter_mut().zip(g_lo.0.iter()) {
                    *c = c.add(&gl.mul(&w_lo));
                }
                // c1 += W_lo·ΔG + ΔW·G_lo
                for (c, (dgc, gl)) in c1.0.iter_mut().zip(dg.iter().zip(g_lo.0.iter())) {
                    *c = c.add(&dgc.mul(&w_lo));
                    *c = c.add(&gl.mul(&dw));
                }
                // c2 += ΔW·ΔG
                for (c, dgc) in c2.0.iter_mut().zip(dg.iter()) {
                    *c = c.add(&dgc.mul(&dw));
                }
                s0 = s0.add(&t_lo.mul(&w_lo));
                s1 = s1.add(&t_lo.mul(&dw).add(&dt.mul(&w_lo)));
                s2 = s2.add(&dt.mul(&dw));
            } else {
                // accumulate: A = e·Ĝ only; the e-table lives in `t`.
                let e_lo = self.t[t];
                let e_hi = self.t[t + half];
                let de = e_hi.sub(&e_lo);
                for (c, gl) in c0.0.iter_mut().zip(g_lo.0.iter()) {
                    *c = c.add(&gl.mul(&e_lo));
                }
                for (c, (dgc, gl)) in c1.0.iter_mut().zip(dg.iter().zip(g_lo.0.iter())) {
                    *c = c.add(&dgc.mul(&e_lo));
                    *c = c.add(&gl.mul(&de));
                }
                for (c, dgc) in c2.0.iter_mut().zip(dg.iter()) {
                    *c = c.add(&dgc.mul(&de));
                }
            }
        }
        if let Some(pp) = &self.p_prime {
            c0 = c0.axpy(&s0, pp);
            c1 = c1.axpy(&s1, pp);
            c2 = c2.axpy(&s2, pp);
        }
        RoundMessage([c0, c1, c2])
    }

    /// Restrict every table to `X₁ = r` (fold halves), halving the cube.
    pub fn restrict(&mut self, r: &Fq) {
        let half = self.g.len() / 2;
        let mut g_next = Vec::with_capacity(half);
        for t in 0..half {
            let lo = &self.g[t];
            let hi = &self.g[t + half];
            // lo + r·(hi − lo)
            let mut restricted = lo.clone();
            for (a, b) in restricted.0.iter_mut().zip(hi.0.iter()) {
                *a = a.add(&b.mul(r));
            }
            for (a, b) in restricted.0.iter_mut().zip(lo.0.iter()) {
                *a = a.sub(&b.mul(r));
            }
            g_next.push(restricted);
        }
        self.g = g_next;
        if !self.w.is_empty() {
            let mut w_next = Vec::with_capacity(half);
            let mut t_next = Vec::with_capacity(half);
            for t in 0..half {
                let w_lo = self.w[t];
                let w_hi = self.w[t + half];
                w_next.push(w_lo.add(&w_hi.sub(&w_lo).mul(r)));
                let t_lo = self.t[t];
                let t_hi = self.t[t + half];
                t_next.push(t_lo.add(&t_hi.sub(&t_lo).mul(r)));
            }
            self.w = w_next;
            self.t = t_next;
        } else {
            let mut t_next = Vec::with_capacity(half);
            for t in 0..half {
                let lo = self.t[t];
                let hi = self.t[t + half];
                t_next.push(lo.add(&hi.sub(&lo).mul(r)));
            }
            self.t = t_next;
        }
    }

    /// After all rounds: the restricted witness value `Ŵ(r)` and factor
    /// `T(r)` (single entries).
    pub fn terminal_values(&self) -> (Fq, Fq) {
        if !self.w.is_empty() {
            (self.w[0], self.t[0])
        } else {
            (Fq::ZERO, self.t[0])
        }
    }
}

/// The verifier-side driver: consumes round messages, checks the
/// recurrences, emits challenges, and produces the terminal `(r, V)`.
pub struct ModuleSumcheckVerifier {
    pub num_vars: usize,
    pub target: ModulePoint,
    /// The running claimed value of `Aᵢ(r₁..rᵢ)`.
    current: ModulePoint,
    challenges: Vec<Fq>,
    dim: usize,
}

/// Errors of the module sumcheck verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumcheckError {
    WrongDegree,
    RecurrenceFailure(usize),
    DimensionMismatch,
}

impl ModuleSumcheckVerifier {
    pub fn new(num_vars: usize, target: ModulePoint) -> Self {
        let dim = target.dim();
        ModuleSumcheckVerifier {
            current: target.clone(),
            target,
            num_vars,
            challenges: Vec::new(),
            dim,
        }
    }

    /// Absorb one round message, check the recurrence, draw the challenge.
    pub fn round(&mut self, msg: &RoundMessage, challenge: Fq) -> Result<(), SumcheckError> {
        let round = self.challenges.len() + 1;
        if msg.0.iter().any(|c| c.dim() != self.dim) {
            return Err(SumcheckError::DimensionMismatch);
        }
        // Degree ≤ 2 is structural (three coefficients); the recurrence:
        let sum = msg.cube_sum();
        if sum != self.current {
            return Err(SumcheckError::RecurrenceFailure(round));
        }
        self.current = msg.eval(&challenge);
        self.challenges.push(challenge);
        Ok(())
    }

    /// The full challenge point after all rounds.
    pub fn point(&self) -> &[Fq] {
        &self.challenges
    }

    /// The terminal value `V = A_m(r_m)` — the module claim handed to the
    /// protocol layer (reduce: divided by `a`; accumulate: by `e(r)`).
    pub fn terminal_value(&self) -> &ModulePoint {
        &self.current
    }
}

/// Reference evaluation of the full summand on the cube — the differential
/// test oracle: `Σ_b [W(b)G(b) + T(b)W(b)P']`.
pub fn reference_cube_sum(tables: &SummandTables) -> ModulePoint {
    let n = tables.g.len();
    let dim = tables.g.first().map(|g| g.dim()).unwrap_or(0);
    let mut acc = ModulePoint::zero(dim);
    let use_w = !tables.w.is_empty();
    for b in 0..n {
        let mut term = tables.g[b].clone();
        if use_w {
            term = term.scale(&tables.w[b]);
            if let Some(pp) = &tables.p_prime {
                let s = tables.t[b].mul(&tables.w[b]);
                term = term.axpy(&s, pp);
            }
        } else {
            term = term.scale(&tables.t[b]);
        }
        acc = acc.add(&term);
    }
    acc
}

/// `eq(b, r)` on a cube of `2^k` entries — used by tests to cross-check
/// fold-based generator evaluation.
pub fn eq_index(b: usize, r: &[Fq], k: usize) -> Fq {
    eq_eval_index(b, r, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::LayeredCube;
    use lattice_core::transcript::Transcript;

    fn srs_small(n: usize) -> Srs {
        Srs::from_seed(1, 16, n, b"sumcheck-test")
    }

    #[test]
    fn round_message_matches_reference() {
        // reduce-shaped summand on a 2^4 cube.
        let n = 16;
        let srs = srs_small(n);
        let cube = LayeredCube::new(2, 4);
        let w: Vec<Fq> = (0..n)
            .map(|i| Fq::from_u64((i as u64 * 4099) % 0xFFFF))
            .collect();
        let u = vec![Fq::from_u64(31), Fq::from_u64(37)];
        let t = cube.t_table(&u);
        let pp = srs.value_column().scale(&Fq::from_u64(77));
        let tables = SummandTables::for_reduce(&w, &srs, &t, pp);
        let msg = tables.round_message();
        let reference = reference_cube_sum(&tables);
        assert_eq!(msg.cube_sum(), reference);
    }

    #[test]
    fn full_protocol_recurrence() {
        let n = 32;
        let srs = srs_small(n);
        let cube = LayeredCube::new(3, 4);
        let w: Vec<Fq> = (0..n)
            .map(|i| Fq::from_u64((i as u64 * 2053) % 0xFFFF))
            .collect();
        let u = vec![Fq::from_u64(11), Fq::from_u64(13), Fq::from_u64(17)];
        let t = cube.t_table(&u);
        let alpha = Fq::from_u64(991);
        let pp = srs.value_column().scale(&alpha);
        let mut tables = SummandTables::for_reduce(&w, &srs, &t, pp.clone());
        let target = reference_cube_sum(&tables);
        let mut verifier = ModuleSumcheckVerifier::new(cube.num_vars(), target.clone());
        let mut transcript = Transcript::new_default(b"accordion-sc-test");
        for _ in 0..cube.num_vars() {
            let msg = tables.round_message();
            let r = Fq::challenge(&mut transcript, b"r").expect("challenge");
            verifier.round(&msg, r).expect("recurrence holds");
            tables.restrict(&r);
        }
        // Terminal values: W(r) and T(r).
        let (a, b) = tables.terminal_values();
        // The verifier's terminal V must equal a·Ĝ(r) + b·a·P'.
        let r = verifier.point().to_vec();
        let g_r = srs.eval_generator_mle(&r);
        let expected = g_r.scale(&a).axpy(&a.mul(&b), &pp);
        assert_eq!(verifier.terminal_value(), &expected);
    }

    #[test]
    fn accumulate_shaped_recurrence() {
        let n = 16;
        let srs = srs_small(n);
        let cube = LayeredCube::new(2, 4);
        let r1 = vec![
            Fq::from_u64(3),
            Fq::from_u64(5),
            Fq::from_u64(7),
            Fq::from_u64(9),
        ];
        let r2 = vec![
            Fq::from_u64(11),
            Fq::from_u64(13),
            Fq::from_u64(15),
            Fq::from_u64(17),
        ];
        let g1 = Fq::from_u64(2);
        let g2 = Fq::from_u64(3);
        let e = cube.eq_batch_table(&[r1.clone(), r2.clone()], &[g1, g2]);
        let mut tables = SummandTables::for_accumulate(&srs, &e);
        let target = reference_cube_sum(&tables);
        // The target must equal γ¹·Ĝ(r₁) + γ²·Ĝ(r₂).
        let expect = srs
            .eval_generator_mle(&r1)
            .scale(&g1)
            .axpy(&g2, &srs.eval_generator_mle(&r2));
        assert_eq!(target, expect);
        let mut verifier = ModuleSumcheckVerifier::new(cube.num_vars(), target);
        let mut transcript = Transcript::new_default(b"acc-test");
        for _ in 0..cube.num_vars() {
            let msg = tables.round_message();
            let r = Fq::challenge(&mut transcript, b"r").expect("challenge");
            verifier.round(&msg, r).expect("recurrence");
            tables.restrict(&r);
        }
        let r = verifier.point().to_vec();
        let e_r = cube.eval_eq_batch(&r, &[r1, r2], &[g1, g2]);
        let g_r = srs.eval_generator_mle(&r);
        assert_eq!(verifier.terminal_value(), &g_r.scale(&e_r));
    }

    #[test]
    fn tampered_message_rejected() {
        let n = 16;
        let srs = srs_small(n);
        let cube = LayeredCube::new(2, 4);
        let w: Vec<Fq> = (0..n).map(|i| Fq::from_u64(i as u64 * 31)).collect();
        let u = vec![Fq::from_u64(41), Fq::from_u64(43)];
        let t = cube.t_table(&u);
        let pp = srs.value_column().scale(&Fq::ONE);
        let tables = SummandTables::for_reduce(&w, &srs, &t, pp);
        let target = reference_cube_sum(&tables);
        let mut verifier = ModuleSumcheckVerifier::new(cube.num_vars(), target);
        let msg = tables.round_message();
        let mut bad = msg.clone();
        bad.0[1] = bad.0[1].axpy(&Fq::ONE, srs.value_column());
        let r = Fq::from_u64(123);
        assert_eq!(
            verifier.round(&bad, r),
            Err(SumcheckError::RecurrenceFailure(1))
        );
        assert_eq!(verifier.round(&msg, r), Ok(()));
    }
}
