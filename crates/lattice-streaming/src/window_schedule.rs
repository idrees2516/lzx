//! **The streaming window schedule** — `EvalProductStream_{k},SC`
//! (ePrint 2026/587 §5.2 + Appendix C.4.2, Figure 2): the round-batched
//! window prover applied *repeatedly* under an `O(M^{1/k})` space
//! budget, with no materialized bound tables.
//!
//! # The schedule (Figure 2)
//!
//! `Ω = (ω_1, …, ω_T)` with `δ = log₂(d+1)`, `α = δ/(δ−1)`:
//!
//! * **time-constrained phase** — geometrically growing windows
//!   `ω_t = min(max(1, ⌊α^{t−1}/(δ−1)⌋), ℓ − S_t)` while
//!   `α^{t−1}/(δ−1) ≤ ⌊ℓ/(kδ)⌋`;
//! * **space-constrained phase** — the fixed window
//!   `ω_t = min(max(1, ⌊ℓ/(kδ)⌋), ℓ − S_t)`;
//! * **final phase** — plain linear-time sum-check once
//!   `ℓ − S_{T+1} ≤ ℓ/k`.
//!
//! # The space discipline (why `O(M^{1/k})` holds)
//!
//! The bound tables `p_k(r_{<S_t}, ·)` are **never materialized**:
//! every access is *emulated* by the eq-fold
//! `p_k(r_{<S}, x) = Σ_{j∈{0,1}^S} eq(r_{<S}, j)·p_k(j‖x)` over the
//! original oracle (Figure 2's "emulate streaming access … via on-the-fly
//! Lagrange weights"). Per window the prover touches
//! `d·2^ω·2^{ℓ−S−ω}·2^S = d·M` oracle evaluations — one pass-equivalent
//! of prefix adaptation, exactly the paper's `Θ(d·M)` per-pass cost —
//! while storing only the current window's per-suffix scratch
//! (`d·2^ω`) and the accumulated grid (`(d+2)^ω ≤ (d+2)^{⌊ℓ/(kδ)⌋}
//! ≈ 2^{ℓ/k} = M^{1/k}`). The total pass count is
//! `Θ(log_α ℓ + kδ) = Θ(log d·(log log M + k))`.
//!
//! The final phase materializes the bound tables at
//! `2^{ℓ−S_{T+1}} ≤ 2^{ℓ/k}` per factor and finishes in memory.
//!
//! Round messages are **bit-identical** to `lattice-sumcheck`'s
//! in-memory engine (the same transcript labels and round shapes) —
//! only the prover's memory strategy changes.

use crate::oracle::IndexOracle;
use crate::small_space::EqWalk;
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;
use lattice_sumcheck::multiproduct::multi_product_eval;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowScheduleError {
    Transcript(TranscriptError),
    BadShape { expected: usize, got: usize },
    RoundCheckFailed { round: usize },
    ClaimMismatch,
    FinalCheckFailed,
    EmptyInstance,
}

impl core::fmt::Display for WindowScheduleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WindowScheduleError::Transcript(e) => write!(f, "transcript error: {e:?}"),
            WindowScheduleError::BadShape { expected, got } => {
                write!(f, "window-schedule shape {got} != {expected}")
            }
            WindowScheduleError::RoundCheckFailed { round } => {
                write!(f, "round identity failed at round {round}")
            }
            WindowScheduleError::ClaimMismatch => write!(f, "claim mismatch"),
            WindowScheduleError::FinalCheckFailed => write!(f, "terminal identity failed"),
            WindowScheduleError::EmptyInstance => write!(f, "empty instance"),
        }
    }
}

/// The Figure-2 window schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowSchedule {
    /// Window sizes `ω_1..ω_T`.
    pub omegas: Vec<usize>,
    /// The remaining rounds after the last window (the final phase,
    /// `ℓ − S_{T+1} ≤ ℓ/k`).
    pub final_rounds: usize,
}

impl WindowSchedule {
    /// Plan the schedule for `ℓ` variables, degree-`d` products, and the
    /// space parameter `k ≥ 2` (`O(M^{1/k})` target).
    pub fn plan(ell: usize, d: usize, k: usize) -> WindowSchedule {
        let k = k.max(2);
        let delta = ((d + 1) as f64).log2();
        let alpha = delta / (delta - 1.0);
        let cap = ((ell as f64) / (k as f64 * delta)).floor().max(1.0) as usize;
        let final_target = ell / k;
        let mut omegas = Vec::new();
        let mut s = 0usize;
        // Time-constrained phase: windows grow geometrically until the
        // growth formula exceeds the space cap.
        let mut t = 1usize;
        while s < ell && ell - s > final_target {
            let growth = if t == 1 {
                1.0
            } else {
                alpha.powi(t as i32 - 1) / (delta - 1.0)
            };
            if t > 1 && growth > cap as f64 {
                break;
            }
            let w = (growth.floor().max(1.0) as usize).min(ell - s).max(1);
            omegas.push(w);
            s += w;
            t += 1;
        }
        // Space-constrained phase: the fixed window.
        while s < ell && ell - s > final_target {
            let w = cap.min(ell - s).max(1);
            omegas.push(w);
            s += w;
        }
        WindowSchedule {
            omegas,
            final_rounds: ell - s,
        }
    }

    /// The cumulative window-start positions `S_1..S_{T+1}`.
    pub fn starts(&self) -> Vec<usize> {
        let mut out = vec![0usize];
        for &w in &self.omegas {
            out.push(out.last().copied().unwrap_or(0) + w);
        }
        out
    }
}

/// Prover output — the Boolean engine's shape (bit-identical proofs).
#[derive(Clone, Debug)]
pub struct StreamWindowOutput {
    pub rounds: Vec<Vec<Goldilocks>>,
    pub challenges: Vec<Goldilocks>,
    pub final_claim: Goldilocks,
    pub factor_claims: Vec<Goldilocks>,
    /// Instrumentation: oracle evaluations consumed.
    pub oracle_evals: u64,
    /// Instrumentation: the peak element count stored (grid + scratch +
    /// final bound tables).
    pub peak_elems: usize,
    /// Instrumentation: the number of window passes (each one
    /// pass-equivalent of `Θ(d·M)` prefix adaptation).
    pub passes: usize,
}

/// The emulated bound-table access (Figure 2 step 1a's "on-the-fly
/// Lagrange weights"): `p_k(r_{<S}, x) = Σ_j eq(r_{<S}, j)·p_k(j‖x)`
/// via the Gray-coded eq walk. `x` indexes the remaining `ℓ−S` bits.
fn folded_eval(
    oracle: &mut dyn IndexOracle,
    walk_r: &[Goldilocks],
    x: u64,
    rem_bits: usize,
    stats: &mut u64,
) -> Goldilocks {
    let s = walk_r.len();
    let mut acc = Goldilocks::ZERO;
    let mut walk = EqWalk::new(walk_r);
    for j in 0..(1u64 << s) {
        let w = walk.weight();
        if !w.is_zero() {
            let idx = (j << rem_bits) | x;
            acc = acc.add(&w.mul(&oracle.eval(idx)));
            *stats += 1;
        }
        if j + 1 < (1u64 << s) {
            walk.advance();
        }
    }
    acc
}

/// The finite-node Lagrange weights over `{0..d}` at `r`, padded to the
/// grid's axis side `d+2` (index 0 = ∞ at weight zero) — the same
/// convention as the in-memory window prover.
fn lagrange_weights_u(r: &Goldilocks, d: usize) -> Vec<Goldilocks> {
    let side = d + 2;
    let mut w = vec![Goldilocks::ZERO; side];
    for j in 0..=d {
        let mut num = Goldilocks::ONE;
        let mut den = Goldilocks::ONE;
        for k2 in 0..=d {
            if k2 == j {
                continue;
            }
            let xk = Goldilocks::from_u64(k2 as u64);
            num = num.mul(&r.sub(&xk));
            let delta = j as i64 - k2 as i64;
            let dsign = if delta >= 0 {
                Goldilocks::from_u64(delta.unsigned_abs())
            } else {
                // (−u) mod p for the Goldilocks modulus.
                Goldilocks(0xFFFF_FFFF_0000_0001u64.wrapping_sub(delta.unsigned_abs()))
            };
            den = den.mul(&dsign);
        }
        let inv = den.inverse().unwrap_or(Goldilocks::ZERO);
        w[j + 1] = num.mul(&inv);
    }
    w
}

/// Interpolate the degree-`d` round polynomial through `g(0..=d)` at `r`.
fn interpolate_at(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
    let n = evals.len();
    let mut acc = Goldilocks::ZERO;
    for (i, &ei) in evals.iter().enumerate() {
        let xi = Goldilocks::from_u64(i as u64);
        let mut weight = Goldilocks::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            let inv = den.inverse().unwrap_or(Goldilocks::ZERO);
            weight = weight.mul(&num.mul(&inv));
        }
        acc = acc.add(&ei.mul(&weight));
    }
    acc
}

/// Prove `Σ_{x∈{0,1}^ℓ} P(x) = claim` with the streaming window
/// schedule at space parameter `k` — `O(M^{1/k})`-class storage, never
/// materializing a bound table before the final phase.
#[allow(clippy::too_many_lines)]
pub fn prove_stream_windowed(
    num_vars: usize,
    factors: &mut [&mut dyn IndexOracle],
    terms: &[(Goldilocks, Vec<usize>)],
    claim: Goldilocks,
    transcript: &mut Transcript,
    k: usize,
) -> Result<StreamWindowOutput, WindowScheduleError> {
    prove_with_schedule(
        num_vars,
        factors,
        terms,
        claim,
        transcript,
        &WindowSchedule::plan(
            num_vars,
            terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1),
            k,
        ),
    )
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub fn prove_with_schedule(
    num_vars: usize,
    factors: &mut [&mut dyn IndexOracle],
    terms: &[(Goldilocks, Vec<usize>)],
    claim: Goldilocks,
    transcript: &mut Transcript,
    schedule: &WindowSchedule,
) -> Result<StreamWindowOutput, WindowScheduleError> {
    if terms.is_empty() || factors.is_empty() {
        return Err(WindowScheduleError::EmptyInstance);
    }
    let ell = num_vars;
    let d = terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1);
    for f in factors.iter() {
        if f.len() as usize != 1 << ell {
            return Err(WindowScheduleError::BadShape {
                expected: 1 << ell,
                got: f.len() as usize,
            });
        }
    }
    let side = d + 2;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(ell);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(ell);
    let mut current_claim = claim;
    let mut oracle_evals: u64 = 0;
    let mut peak_elems: usize = 0;

    let starts = schedule.starts();
    for (t, &omega) in schedule.omegas.iter().enumerate() {
        let s_t = starts[t];
        let suffix_bits = ell - s_t - omega;
        let suffixes = 1usize << suffix_bits;
        // The accumulated per-term grids of q_t = Σ_{x'} Σ_j c·Π p_k(…):
        // grid construction is linear, so per-suffix grids sum.
        let mut q_grids: Vec<Vec<Goldilocks>> = terms
            .iter()
            .map(|_| vec![Goldilocks::ZERO; side.pow(omega as u32)])
            .collect();
        // Per-suffix scratch: the d window tables (2^ω each).
        let mut scratch: Vec<Vec<Goldilocks>> = factors
            .iter()
            .map(|_| vec![Goldilocks::ZERO; 1 << omega])
            .collect();
        peak_elems = peak_elems
            .max(q_grids.iter().map(|g| g.len()).sum::<usize>() + scratch.len() * (1 << omega));

        // Per suffix: gather the window tables via the emulated fold,
        // build the term grids, accumulate into q.
        for x2 in 0..suffixes {
            for (k, oracle) in factors.iter_mut().enumerate() {
                for b in 0..(1usize << omega) {
                    let x = (b << suffix_bits) | x2;
                    scratch[k][b] = folded_eval(
                        *oracle,
                        &challenges[..s_t],
                        x as u64,
                        ell - s_t,
                        &mut oracle_evals,
                    );
                }
            }
            for (ti, (_c, ids)) in terms.iter().enumerate() {
                let mut tables: Vec<Vec<Goldilocks>> =
                    ids.iter().map(|&fi| scratch[fi].clone()).collect();
                while tables.len() < d {
                    tables.push(vec![Goldilocks::ONE; 1 << omega]);
                }
                let (grid, _stats) = multi_product_eval(&tables, omega);
                for (gq, gv) in q_grids[ti].iter_mut().zip(grid.iter()) {
                    *gq = gq.add(gv);
                }
            }
        }

        // Emulate rounds [S_t+1, S_t+ω] on the accumulated grids — the
        // in-memory window prover's extraction (challenge-weighted
        // grid reads with composed finite-node Lagrange weights).
        let mut wj: Vec<Goldilocks> = vec![Goldilocks::ONE];
        for j in 1..=omega {
            let t_axis = j - 1;
            let tail_axes = omega - j;
            let strides: Vec<usize> = (0..omega)
                .map(|a| side.pow((omega - a - 1) as u32))
                .collect();
            let w_offsets: Vec<usize> = {
                let mut offs = Vec::with_capacity(1usize << tail_axes);
                for w in 0..1usize << tail_axes {
                    let mut off = 0usize;
                    for b in 0..tail_axes {
                        let bit = (w >> (tail_axes - 1 - b)) & 1;
                        off += (bit + 1) * strides[j + b];
                    }
                    offs.push(off);
                }
                offs
            };
            let stride_t = strides[t_axis];
            let u_count = side.pow((j - 1) as u32);
            let u_offsets: Vec<usize> = {
                let mut offs = Vec::with_capacity(u_count);
                for u in 0..u_count {
                    let mut u_off = 0usize;
                    let mut rem = u;
                    for a in (0..(j - 1)).rev() {
                        u_off += (rem % side) * strides[a];
                        rem /= side;
                    }
                    offs.push(u_off);
                }
                offs
            };
            let mut evals_at = vec![Goldilocks::ZERO; d + 1];
            for ((coeff, _ids), grid) in terms.iter().zip(q_grids.iter()) {
                for (u, &u_off) in u_offsets.iter().enumerate() {
                    let wgt = wj[u];
                    if wgt.is_zero() {
                        continue;
                    }
                    for t in 0..=d {
                        let base = u_off + (t + 1) * stride_t;
                        let mut acc = Goldilocks::ZERO;
                        for &off in &w_offsets {
                            acc = acc.add(&grid[base + off]);
                        }
                        evals_at[t] = evals_at[t].add(&acc.mul(&wgt).mul(coeff));
                    }
                }
            }
            let sum01 = evals_at[0].add(&evals_at[1]);
            if sum01 != current_claim {
                return Err(WindowScheduleError::ClaimMismatch);
            }
            transcript
                .append_field_slice(b"sumcheck-round", &evals_at)
                .map_err(WindowScheduleError::Transcript)?;
            let r = transcript
                .challenge_field(b"sumcheck-challenge")
                .map_err(WindowScheduleError::Transcript)?;
            challenges.push(r);
            current_claim = interpolate_at(&evals_at, &r);
            rounds.push(evals_at);
            if j < omega {
                let l = lagrange_weights_u(&r, d);
                let mut next = Vec::with_capacity(wj.len() * side);
                for wv in &wj {
                    for lv in &l {
                        next.push(wv.mul(lv));
                    }
                }
                wj = next;
            }
        }
    }

    // ---- Final phase: materialize the bound tables (≤ 2^{ℓ/k}) and
    // finish with the in-memory linear-time engine. ----
    let s_final = starts[schedule.omegas.len()];
    let rem_bits = ell - s_final;
    let rem = 1usize << rem_bits;
    let mut bound: Vec<Vec<Goldilocks>> = factors
        .iter()
        .map(|_| vec![Goldilocks::ZERO; rem])
        .collect();
    peak_elems = peak_elems.max(bound.len() * rem);
    {
        let mut walk = EqWalk::new(&challenges[..s_final]);
        for j in 0..(1u64 << s_final) {
            let w = walk.weight();
            if !w.is_zero() {
                for (k, oracle) in factors.iter_mut().enumerate() {
                    let base = j << rem_bits;
                    for x in 0..rem {
                        bound[k][x] = bound[k][x].add(&w.mul(&oracle.eval(base | x as u64)));
                        oracle_evals += 1;
                    }
                }
            }
            if j + 1 < (1u64 << s_final) {
                walk.advance();
            }
        }
    }

    // The linear-time tail: bind once per round, per-t affine values.
    let t_bars: Vec<Goldilocks> = (0..=d as u64).map(Goldilocks::from_u64).collect();
    for _round in 0..rem_bits {
        let half = bound[0].len() / 2;
        let deltas: Vec<Vec<Goldilocks>> = bound
            .iter()
            .map(|b| (0..half).map(|e| b[e + half].sub(&b[e])).collect())
            .collect();
        let mut evals_at = Vec::with_capacity(d + 1);
        for &tf in t_bars.iter() {
            let mut acc = Goldilocks::ZERO;
            for (c, ids) in terms {
                let mut term = Goldilocks::ZERO;
                for e in 0..half {
                    let mut prod = *c;
                    for &fi in ids {
                        let v = bound[fi][e].add(&deltas[fi][e].mul(&tf));
                        prod = prod.mul(&v);
                    }
                    term = term.add(&prod);
                }
                acc = acc.add(&term);
            }
            evals_at.push(acc);
        }
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(WindowScheduleError::RoundCheckFailed {
                round: rounds.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(WindowScheduleError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(WindowScheduleError::Transcript)?;
        challenges.push(r);
        current_claim = interpolate_at(&evals_at, &r);
        rounds.push(evals_at);
        for b in bound.iter_mut() {
            let lo: Vec<Goldilocks> = b[..half].to_vec();
            let hi: Vec<Goldilocks> = b[half..].to_vec();
            let mut next = Vec::with_capacity(half);
            for e in 0..half {
                next.push(lo[e].add(&hi[e].sub(&lo[e]).mul(&r)));
            }
            *b = next;
        }
    }

    // Terminal claims.
    let factor_claims: Vec<Goldilocks> = bound.iter().map(|b| b[0]).collect();
    let mut final_claim = Goldilocks::ZERO;
    for (c, ids) in terms {
        let mut prod = *c;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(WindowScheduleError::FinalCheckFailed);
    }

    Ok(StreamWindowOutput {
        rounds,
        challenges,
        final_claim,
        factor_claims,
        oracle_evals,
        peak_elems,
        passes: schedule.omegas.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use lattice_core::{DenseMle, Goldilocks};
    use lattice_sumcheck::sumcheck;
    use lattice_sumcheck::virtual_poly::VirtualPolynomial;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn random_dense(log_vars: usize, seed: u64) -> DenseMle {
        DenseMle::random(log_vars, &seed.to_le_bytes())
    }

    /// Bit-identity vs the in-memory Boolean engine, across shapes and
    /// space parameters.
    #[test]
    fn byte_identity_vs_in_memory() {
        for &(log_vars, d, k) in &[
            (6usize, 2usize, 2usize),
            (8, 2, 2),
            (7, 3, 2),
            (9, 2, 3),
            (10, 3, 3),
            (8, 2, 4),
        ] {
            let dense: Vec<DenseMle> = (0..d)
                .map(|i| random_dense(log_vars, 100 + i as u64))
                .collect();
            let mut vp = VirtualPolynomial::new(log_vars);
            let mut ids = Vec::new();
            for t in dense.iter() {
                ids.push(vp.add_factor(t.clone()).unwrap());
            }
            vp.add_term(Goldilocks::ONE, ids).unwrap();
            // Claim from the truth tables.
            let mut claim = Goldilocks::ZERO;
            for e in 0..(1usize << log_vars) {
                let mut prod = Goldilocks::ONE;
                for t in dense.iter() {
                    prod = prod.mul(&t.evaluations[e]);
                }
                claim = claim.add(&prod);
            }
            // In-memory reference.
            let mut ts0 = Transcript::new_default(b"stream-win");
            let base = sumcheck::prove(&vp, claim, &mut ts0).unwrap();

            // Streaming windowed prover.
            let mut oracles: Vec<OwnedOracle> = dense
                .iter()
                .map(|t| OwnedOracle::new(t.evaluations.clone()))
                .collect();
            let mut oracle_refs: Vec<&mut dyn IndexOracle> = oracles
                .iter_mut()
                .map(|o| o as &mut dyn IndexOracle)
                .collect();
            let terms = vec![(Goldilocks::ONE, (0..d).collect::<Vec<_>>())];
            let mut ts1 = Transcript::new_default(b"stream-win");
            let out = prove_stream_windowed(log_vars, &mut oracle_refs, &terms, claim, &mut ts1, k)
                .unwrap();
            assert_eq!(
                base.proof.rounds, out.rounds,
                "rounds: log={log_vars} d={d} k={k}"
            );
            assert_eq!(
                base.challenges, out.challenges,
                "challenges: log={log_vars} d={d} k={k}"
            );
            assert_eq!(base.final_claim, out.final_claim);
            assert_eq!(base.factor_claims, out.factor_claims);
        }
    }

    /// The schedule's shape: geometric growth, the space cap, and the
    /// final-phase cutoff at ℓ/k.
    #[test]
    fn schedule_shape() {
        // d = 2 (δ = log2 3 ≈ 1.585), ℓ = 20, k = 2:
        // cap = ⌊20/(2·1.585)⌋ = 6.
        let sch = WindowSchedule::plan(20, 2, 2);
        assert!(sch.omegas.iter().all(|&w| (1..=6 + 2).contains(&w)));
        // Geometric prefix (1, then α/(δ−1) ≈ 4), then the space cap.
        assert_eq!(
            &sch.omegas[..2],
            &[1usize, 4usize],
            "geometric start: {:?}",
            sch.omegas
        );
        assert!(
            sch.omegas.windows(2).all(|w| w[1] >= w[0]),
            "non-decreasing: {:?}",
            sch.omegas
        );
        // Every window after the growth phase sits at the cap.
        for &w in sch.omegas.iter().skip(2) {
            assert_eq!(w, 6, "space-capped window: {:?}", sch.omegas);
        }
        assert!(sch.final_rounds <= 10, "final ≤ ℓ/k");
        let starts = sch.starts();
        assert_eq!(starts.last().copied().unwrap_or(0) + sch.final_rounds, 20);
    }

    /// The space discipline: the peak stays grid+scratch sized, far
    /// below the in-memory engine's O(ℓ·M) tables.
    #[test]
    fn space_profile() {
        let log_vars = 12usize;
        let dense: Vec<DenseMle> = (0..2)
            .map(|i| random_dense(log_vars, 200 + i as u64))
            .collect();
        let mut claim = Goldilocks::ZERO;
        for e in 0..(1usize << log_vars) {
            claim = claim.add(&dense[0].evaluations[e].mul(&dense[1].evaluations[e]));
        }
        let mut oracles: Vec<OwnedOracle> = dense
            .iter()
            .map(|t| OwnedOracle::new(t.evaluations.clone()))
            .collect();
        let mut oracle_refs: Vec<&mut dyn IndexOracle> = oracles
            .iter_mut()
            .map(|o| o as &mut dyn IndexOracle)
            .collect();
        let terms = vec![(Goldilocks::ONE, vec![0usize, 1])];
        let mut ts = Transcript::new_default(b"space");
        let out =
            prove_stream_windowed(log_vars, &mut oracle_refs, &terms, claim, &mut ts, 2).unwrap();
        let m = 1usize << log_vars;
        // The in-memory engine holds ℓ·M factor entries; the streaming
        // windowed prover's peak must be a small fraction of M.
        assert!(
            out.peak_elems < m / 8,
            "peak {} not sublinear vs M={m}",
            out.peak_elems
        );
        // Passes: the schedule's window count.
        assert!(out.passes >= 2);
        assert!(out.oracle_evals > 0);
    }

    /// Multi-term instance with distinct coefficients roundtrips.
    #[test]
    fn multi_term_identity() {
        let log_vars = 7usize;
        let f0 = random_dense(log_vars, 7);
        let f1 = random_dense(log_vars, 8);
        let f2 = random_dense(log_vars, 9);
        let mut vp = VirtualPolynomial::new(log_vars);
        let i0 = vp.add_factor(f0.clone()).unwrap();
        let i1 = vp.add_factor(f1.clone()).unwrap();
        let i2 = vp.add_factor(f2.clone()).unwrap();
        vp.add_term(fe(3), vec![i0, i1, i2]).unwrap();
        vp.add_term(fe(5), vec![i0]).unwrap();
        // Claim over the truth tables.
        let mut claim = Goldilocks::ZERO;
        for e in 0..(1usize << log_vars) {
            claim = claim
                .add(
                    &fe(3)
                        .mul(&f0.evaluations[e])
                        .mul(&f1.evaluations[e])
                        .mul(&f2.evaluations[e]),
                )
                .add(&fe(5).mul(&f0.evaluations[e]));
        }
        let mut ts0 = Transcript::new_default(b"multi");
        let base = sumcheck::prove(&vp, claim, &mut ts0).unwrap();

        let tables = [f0, f1, f2];
        let mut oracles: Vec<OwnedOracle> = tables
            .iter()
            .map(|t| OwnedOracle::new(t.evaluations.clone()))
            .collect();
        let mut oracle_refs: Vec<&mut dyn IndexOracle> = oracles
            .iter_mut()
            .map(|o| o as &mut dyn IndexOracle)
            .collect();
        let terms = vec![(fe(3), vec![0usize, 1, 2]), (fe(5), vec![0])];
        let mut ts1 = Transcript::new_default(b"multi");
        let out = prove_stream_windowed(7, &mut oracle_refs, &terms, claim, &mut ts1, 2).unwrap();
        assert_eq!(base.proof.rounds, out.rounds);
        assert_eq!(base.challenges, out.challenges);
        assert_eq!(base.final_claim, out.final_claim);
        assert_eq!(base.factor_claims, out.factor_claims);
    }

    /// Wrong claim rejected at the first window round.
    #[test]
    fn wrong_claim_rejected() {
        let log_vars = 6usize;
        let f0 = random_dense(log_vars, 11);
        let f1 = random_dense(log_vars, 12);
        let mut claim = Goldilocks::ZERO;
        for e in 0..(1usize << log_vars) {
            claim = claim.add(&f0.evaluations[e].mul(&f1.evaluations[e]));
        }
        let tables = [f0, f1];
        let mut oracles: Vec<OwnedOracle> = tables
            .iter()
            .map(|t| OwnedOracle::new(t.evaluations.clone()))
            .collect();
        let mut oracle_refs: Vec<&mut dyn IndexOracle> = oracles
            .iter_mut()
            .map(|o| o as &mut dyn IndexOracle)
            .collect();
        let terms = vec![(Goldilocks::ONE, vec![0usize, 1])];
        let mut ts = Transcript::new_default(b"wrong");
        let bad = claim.add(&Goldilocks::ONE);
        assert!(matches!(
            prove_stream_windowed(log_vars, &mut oracle_refs, &terms, bad, &mut ts, 2),
            Err(WindowScheduleError::ClaimMismatch)
        ));
    }
}
