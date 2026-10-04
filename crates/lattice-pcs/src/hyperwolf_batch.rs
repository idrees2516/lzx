//! HyperWolf H7 — the three Appendix-B batching modes (Wave 7 item
//! 7.12's second half): multiple evaluation claims batched into ONE
//! evaluation proof.
//!
//! * **Mode 1 — multiple polynomials at a single point**: the verifier
//!   samples `α ∈ Z_q^n`; the prover forms `f = Σ α_i f_i` and both
//!   parties compute `y = Σ α_i v_i`; ONE evaluation protocol runs on
//!   `(f, y)` at the shared point.
//! * **Mode 2 — one polynomial at multiple points (multilinear)**: the
//!   prover builds `g(x) = Σ_i α_i·f̃(x)·eq(x, u_i)` and the parties
//!   run a degree-2 sumcheck over the coefficient hypercube
//!   `Σ_b g(b) = Σ_i α_i v_i`; the terminal binds `f̃(r)·E(r)` with
//!   `E(r) = Σ_i α_i eq(r, u_i)` VERIFIER-computed, reducing to ONE
//!   evaluation claim `f(r) = final/E(r)` handled by the Mode-1 path.
//! * **Mode 3 — multiple polynomials at multiple points**: the same
//!   sumcheck with `g(x) = Σ_i α_i·f_i(x)·eq(x, u_i)`; the terminal
//!   takes the prover's per-polynomial claims `w_i = f_i(r)` checked
//!   by `Σ_i α_i·w_i·eq(r, u_i) = final`, then Mode 1 batches the
//!   `f_i(r) = w_i` claims at the single point `r`.
//!
//! The Mode-2/3 sumcheck runs over the scalar field `Z_q` of the
//! HyperWolf ring (a bespoke degree-2 multilinear sumcheck with
//! per-round round-messages `(q(0), q(1), q(2))` and `Z_q` challenges —
//! the paper's own setting; the ring's negacyclic structure plays no
//! role at this layer).

use crate::hyperwolf::{HwCommitState, HwError, HyperWolfFull};
use lattice_core::transcript::Transcript;

// ---------------------------------------------------- scalar Z_q helpers #

fn sq_mul(a: u64, b: u64, q: u64) -> u64 {
    (((a as u128) * (b as u128)) % (q as u128)) as u64
}

fn sq_add(a: u64, b: u64, q: u64) -> u64 {
    let s = a + b;
    if s >= q {
        s - q
    } else {
        s
    }
}

fn sq_sub(a: u64, b: u64, q: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        q - (b - a)
    }
}

/// One degree-2 sumcheck round message: `q(t)` at `t ∈ {0, 1, 2}`.
type SqRound = [u64; 3];

/// Interpolate the quadratic through `(q(0), q(1), q(2))` at `r`.
fn sq_interp(qs: &[u64; 3], r: u64, q: u64) -> u64 {
    // L0 = (t²−3t+2)/2, L1 = 2t−t², L2 = (t²−t)/2 — with 2 invertible
    // (q odd).
    let inv2 = mod_pow(2, q - 2, q);
    let t = r;
    let t2 = sq_mul(t, t, q);
    let l0 = {
        let num = sq_sub(sq_add(t2, 2, q), sq_mul(3, t, q), q); // t²−3t+2
        sq_mul(num, inv2, q)
    };
    let l1 = sq_sub(sq_mul(2, t, q), t2, q);
    let l2 = sq_mul(sq_sub(t2, t, q), inv2, q);
    let mut acc = sq_mul(qs[0], l0, q);
    acc = sq_add(acc, sq_mul(qs[1], l1, q), q);
    sq_add(acc, sq_mul(qs[2], l2, q), q)
}

fn mod_pow(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut result = 1u64;
    base %= q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = sq_mul(result, base, q);
        }
        base = sq_mul(base, base, q);
        exp >>= 1;
    }
    result
}

/// The per-round binding of one multilinear table (in place): the
/// standard `new[i] = lo[i] + r·(hi[i] − lo[i])` over `Z_q`.
fn sq_bind(table: &mut Vec<u64>, r: u64, q: u64) {
    let half = table.len() / 2;
    for i in 0..half {
        let lo = table[i];
        let hi = table[i + half];
        let diff = sq_sub(hi, lo, q);
        table[i] = sq_add(lo, sq_mul(r, diff, q), q);
    }
    table.truncate(half);
}

// --------------------------------------------------------- Mode 1 #

/// Mode 1's public batch: the RLC challenges are transcript-derived;
/// the proof is ONE HyperWolf evaluation proof of `f = Σ α_i f_i`.
pub struct BatchSinglePoint {
    pub alphas: Vec<u64>,
    pub y_combined: u64,
    /// The combined witness commitment state (kernel scale: the
    /// combined polynomial's coefficients are recomputed by the
    /// verifier through the per-polynomial commitments' linearity —
    /// documented: at kernel scale the combined STATE is carried).
    pub combined_state: HwCommitState,
    /// The single evaluation proof.
    pub proof: crate::hyperwolf::HwProof,
}

impl HyperWolfFull {
    /// Mode 1: batch `f_i(u) = v_i` at ONE point into a single
    /// evaluation proof of `f = Σ α_i f_i` with `y = Σ α_i v_i`.
    pub fn batch_single_point_prove(
        &self,
        states: &[HwCommitState],
        point: &[u64],
        v_claims: &[u64],
        transcript: &mut Transcript,
    ) -> Result<BatchSinglePoint, HwError> {
        if states.len() != v_claims.len() || states.is_empty() {
            return Err(HwError::Shape {
                expected: v_claims.len(),
                got: states.len(),
            });
        }
        let q = self.ring.q;
        let alphas: Vec<u64> = (0..states.len())
            .map(|_| {
                let bytes = transcript
                    .challenge_bytes(b"hw-batch-alpha", 8)
                    .map_err(HwError::Transcript)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[..8]);
                Ok(u64::from_le_bytes(arr) % q)
            })
            .collect::<Result<Vec<_>, HwError>>()?;
        // f = Σ α_i f_i (coefficient-wise, over Z_q), y = Σ α_i v_i.
        let n = states[0].f_ints.len();
        let mut f_combined = vec![0u64; n];
        for (i, st) in states.iter().enumerate() {
            if st.f_ints.len() != n {
                return Err(HwError::Shape {
                    expected: n,
                    got: st.f_ints.len(),
                });
            }
            for (fc, &fc_i) in f_combined.iter_mut().zip(st.f_ints.iter()) {
                let v = fc_i.rem_euclid(q as i64) as u64;
                *fc = sq_add(*fc, sq_mul(alphas[i], v, q), q);
            }
        }
        let mut y_combined = 0u64;
        for (i, &v) in v_claims.iter().enumerate() {
            y_combined = sq_add(y_combined, sq_mul(alphas[i], v % q, q), q);
        }
        let f_ints: Vec<i64> = f_combined
            .iter()
            .map(|&c| {
                if c > q / 2 {
                    c as i64 - q as i64
                } else {
                    c as i64
                }
            })
            .collect();
        let (cm, combined_state) = self.commit(&f_ints)?;
        transcript
            .append_message(
                b"hw-batch-cm",
                &cm.iter()
                    .flat_map(|e| self.ring.to_bytes(e))
                    .collect::<Vec<u8>>(),
            )
            .map_err(HwError::Transcript)?;
        let (a0, a_list) = crate::hyperwolf::build_a_multilinear(
            &self.ring,
            point,
            self.params.k,
            self.params.b,
            self.params.d,
        )?;
        let proof = self.eval_prove(&combined_state, &a0, &a_list, transcript)?;
        Ok(BatchSinglePoint {
            alphas,
            y_combined,
            combined_state,
            proof,
        })
    }

    /// Mode 1's verifier: replay α, recompute the combined commitment,
    /// verify the single proof.
    pub fn batch_single_point_verify(
        &self,
        states: &[HwCommitState],
        point: &[u64],
        v_claims: &[u64],
        batch: &BatchSinglePoint,
        transcript: &mut Transcript,
    ) -> Result<bool, HwError> {
        let q = self.ring.q;
        let alphas: Vec<u64> = (0..states.len())
            .map(|_| {
                let bytes = transcript
                    .challenge_bytes(b"hw-batch-alpha", 8)
                    .map_err(HwError::Transcript)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[..8]);
                Ok(u64::from_le_bytes(arr) % q)
            })
            .collect::<Result<Vec<_>, HwError>>()?;
        if alphas != batch.alphas {
            return Ok(false);
        }
        let n = states[0].f_ints.len();
        let mut f_combined = vec![0u64; n];
        for (i, st) in states.iter().enumerate() {
            for (fc, &fc_i) in f_combined.iter_mut().zip(st.f_ints.iter()) {
                let v = fc_i.rem_euclid(q as i64) as u64;
                *fc = sq_add(*fc, sq_mul(alphas[i], v, q), q);
            }
        }
        let mut y_combined = 0u64;
        for (i, &v) in v_claims.iter().enumerate() {
            y_combined = sq_add(y_combined, sq_mul(alphas[i], v % q, q), q);
        }
        if y_combined != batch.y_combined {
            return Ok(false);
        }
        let f_ints: Vec<i64> = f_combined
            .iter()
            .map(|&c| {
                if c > q / 2 {
                    c as i64 - q as i64
                } else {
                    c as i64
                }
            })
            .collect();
        let (cm, _) = self.commit(&f_ints)?;
        transcript
            .append_message(
                b"hw-batch-cm",
                &cm.iter()
                    .flat_map(|e| self.ring.to_bytes(e))
                    .collect::<Vec<u8>>(),
            )
            .map_err(HwError::Transcript)?;
        let (a0, a_list) = crate::hyperwolf::build_a_multilinear(
            &self.ring,
            point,
            self.params.k,
            self.params.b,
            self.params.d,
        )?;
        self.eval_verify(&cm, &a0, &a_list, y_combined, &batch.proof, transcript)
    }
}

// --------------------------------------------------------- Mode 2/3 #

/// The Mode-2/3 batch: the degree-2 sumcheck over the coefficient
/// hypercube + the terminal claims.
pub struct BatchMultiPoint {
    pub alphas: Vec<u64>,
    /// The sumcheck rounds: one per coefficient axis.
    pub rounds: Vec<SqRound>,
    /// The final point `r` (the verifier's challenges).
    pub point: Vec<u64>,
    /// Mode 3: the per-polynomial claims `w_i = f_i(r)`.
    pub w_claims: Option<Vec<u64>>,
    /// The single evaluation proof for the reduced claim (Mode 2: one
    /// polynomial at `r`; Mode 3: the Mode-1 batch at `r`).
    pub single: BatchSinglePoint,
}

impl HyperWolfFull {
    /// The bespoke degree-2 `Z_q` sumcheck core shared by Modes 2/3:
    /// `Σ_b f̃(b)·E(b) = claim` with `E = Σ_i α_i·eq(u_i, ·)` — returns
    /// (rounds, point, final_claim) with the verifier-side replay
    /// identically derived.
    #[allow(clippy::too_many_arguments)]
    fn multi_point_sumcheck(
        &self,
        f_table: &[u64],
        e_table: &[u64],
        claim: u64,
        transcript: &mut Transcript,
    ) -> Result<(Vec<SqRound>, Vec<u64>, u64), HwError> {
        let q = self.ring.q;
        let mut f = f_table.to_vec();
        let mut e = e_table.to_vec();
        if f.len() != e.len() || !f.len().is_power_of_two() {
            return Err(HwError::Shape {
                expected: e.len(),
                got: f.len(),
            });
        }
        let num_vars = f.len().trailing_zeros() as usize;
        let mut rounds: Vec<SqRound> = Vec::with_capacity(num_vars);
        let mut point = Vec::with_capacity(num_vars);
        let mut cur = claim;
        for round in 0..num_vars {
            let half = f.len() / 2;
            // q(t) = Σ_w [(1−t)f⁰ + t f¹]·[(1−t)e⁰ + t e¹] — values at
            // t ∈ {0,1,2}.
            let q_at = |t: u64| -> u64 {
                let c_lo = sq_sub(1, t, q);
                let c_hi = t % q;
                let mut acc = 0u64;
                for i in 0..half {
                    let ft = sq_add(sq_mul(c_lo, f[i], q), sq_mul(c_hi, f[i + half], q), q);
                    let et = sq_add(sq_mul(c_lo, e[i], q), sq_mul(c_hi, e[i + half], q), q);
                    acc = sq_add(acc, sq_mul(ft, et, q), q);
                }
                acc
            };
            let qs = [q_at(0), q_at(1), q_at(2)];
            if sq_add(qs[0], qs[1], q) != cur {
                let _ = round;
                return Err(HwError::VerificationFailed);
            }
            rounds.push(qs);
            let bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
            transcript
                .append_message(b"hw-batch-sq-round", &bytes)
                .map_err(HwError::Transcript)?;
            let cb = transcript
                .challenge_bytes(b"hw-batch-sq-chal", 8)
                .map_err(HwError::Transcript)?;
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&cb[..8]);
            let r = u64::from_le_bytes(arr) % q;
            cur = sq_interp(&qs, r, q);
            sq_bind(&mut f, r, q);
            sq_bind(&mut e, r, q);
            point.push(r);
        }
        // The binding order is leading-variable-first (the index's MSB);
        // reverse to the module's LSB-first point convention for the
        // evaluation layer.
        point.reverse();
        Ok((rounds, point, cur))
    }

    /// Build `E(x) = Σ_i α_i·eq(u_i, x)`'s table over the hypercube.
    fn build_e_table(&self, u_points: &[Vec<u64>], alphas: &[u64]) -> Result<Vec<u64>, HwError> {
        let q = self.ring.q;
        let num_vars = u_points.first().map(|u| u.len()).unwrap_or(0);
        let n = 1usize << num_vars;
        let mut table = vec![0u64; n];
        for (i, u) in u_points.iter().enumerate() {
            if u.len() != num_vars {
                return Err(HwError::Shape {
                    expected: num_vars,
                    got: u.len(),
                });
            }
            // eq(u, x) over the cube: Π_j (x_j u_j + (1−x_j)(1−u_j)),
            // with x_j the index's bit j (the module's LSB-first
            // convention, matching `evaluate_direct`).
            for x in 0..n {
                let mut eqv = 1u64;
                for (j, &uj) in u.iter().enumerate() {
                    let xj = ((x >> j) & 1) as u64;
                    let term = sq_add(sq_mul(xj, uj, q), sq_mul(1 - xj, sq_sub(1, uj, q), q), q);
                    eqv = sq_mul(eqv, term, q);
                }
                table[x] = sq_add(table[x], sq_mul(alphas[i], eqv, q), q);
            }
        }
        Ok(table)
    }

    /// The f̃ coefficient table over Z_q (centered lift).
    fn f_table(&self, f_ints: &[i64]) -> Vec<u64> {
        let q = self.ring.q;
        f_ints
            .iter()
            .map(|&c| c.rem_euclid(q as i64) as u64)
            .collect()
    }

    /// Mode 2: ONE polynomial at multiple points — the sumcheck reduces
    /// to a single evaluation claim at `r`, proved by ONE evaluation
    /// proof (the Mode-1 machinery at `r`).
    #[allow(clippy::too_many_lines)]
    pub fn batch_multi_point_prove(
        &self,
        state: &HwCommitState,
        u_points: &[Vec<u64>],
        v_claims: &[u64],
        transcript: &mut Transcript,
    ) -> Result<BatchMultiPoint, HwError> {
        if u_points.len() != v_claims.len() || u_points.is_empty() {
            return Err(HwError::Shape {
                expected: v_claims.len(),
                got: u_points.len(),
            });
        }
        let q = self.ring.q;
        let alphas: Vec<u64> = (0..u_points.len())
            .map(|_| {
                let bytes = transcript
                    .challenge_bytes(b"hw-batch-alpha", 8)
                    .map_err(HwError::Transcript)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[..8]);
                Ok(u64::from_le_bytes(arr) % q)
            })
            .collect::<Result<Vec<_>, HwError>>()?;
        let e_table = self.build_e_table(u_points, &alphas)?;
        let f_table = self.f_table(&state.f_ints);
        // claim = Σ_i α_i v_i.
        let mut claim = 0u64;
        for (i, &v) in v_claims.iter().enumerate() {
            claim = sq_add(claim, sq_mul(alphas[i], v % q, q), q);
        }
        let (rounds, point, final_claim) =
            self.multi_point_sumcheck(&f_table, &e_table, claim, transcript)?;
        // Terminal: final = f̃(r)·E(r); E(r) verifier-computable from
        // the u-points, so the output claim is f̃(r) = final / E(r).
        let mut er = 0u64;
        for (i, u) in u_points.iter().enumerate() {
            let mut eqv = 1u64;
            for (j, &uj) in u.iter().enumerate() {
                let rj = point[j];
                let term = sq_add(
                    sq_mul(rj, uj, q),
                    sq_mul(sq_sub(1, rj, q), sq_sub(1, uj, q), q),
                    q,
                );
                eqv = sq_mul(eqv, term, q);
            }
            er = sq_add(er, sq_mul(alphas[i], eqv, q), q);
        }
        if er == 0 {
            return Err(HwError::Feature(
                crate::PcsFeatureError::BatchingUnsupported,
            ));
        }
        let w = sq_mul(final_claim, mod_pow(er, q - 2, q), q);
        // ONE evaluation proof at r for f(r) = w (Mode 1 with n=1).
        let single =
            self.batch_single_point_prove(std::slice::from_ref(state), &point, &[w], transcript)?;
        Ok(BatchMultiPoint {
            alphas,
            rounds,
            point,
            w_claims: None,
            single,
        })
    }

    /// Mode 2's verifier.
    pub fn batch_multi_point_verify(
        &self,
        state: &HwCommitState,
        u_points: &[Vec<u64>],
        v_claims: &[u64],
        batch: &BatchMultiPoint,
        transcript: &mut Transcript,
    ) -> Result<bool, HwError> {
        let q = self.ring.q;
        let alphas: Vec<u64> = (0..u_points.len())
            .map(|_| {
                let bytes = transcript
                    .challenge_bytes(b"hw-batch-alpha", 8)
                    .map_err(HwError::Transcript)?;
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes[..8]);
                Ok(u64::from_le_bytes(arr) % q)
            })
            .collect::<Result<Vec<_>, HwError>>()?;
        if alphas != batch.alphas {
            return Ok(false);
        }
        let f_table = self.f_table(&state.f_ints);
        let mut claim = 0u64;
        for (i, &v) in v_claims.iter().enumerate() {
            claim = sq_add(claim, sq_mul(alphas[i], v % q, q), q);
        }
        // Replay the sumcheck rounds.
        let num_vars = f_table.len().trailing_zeros() as usize;
        if batch.rounds.len() != num_vars || batch.point.len() != num_vars {
            return Ok(false);
        }
        let mut cur = claim;
        for (round, qs) in batch.rounds.iter().enumerate() {
            if sq_add(qs[0], qs[1], q) != cur {
                return Ok(false);
            }
            let bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
            transcript
                .append_message(b"hw-batch-sq-round", &bytes)
                .map_err(HwError::Transcript)?;
            let cb = transcript
                .challenge_bytes(b"hw-batch-sq-chal", 8)
                .map_err(HwError::Transcript)?;
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&cb[..8]);
            let r = u64::from_le_bytes(arr) % q;
            // batch.point is the REVERSED (LSB-first) view: round 0
            // bound the leading variable = the LAST entry.
            if r != batch.point[num_vars - 1 - round] {
                return Ok(false);
            }
            cur = sq_interp(qs, r, q);
        }
        // Terminal: E(r) verifier-computed; the reduced claim w must
        // reproduce the final value; then verify the single proof.
        let mut er = 0u64;
        for (i, u) in u_points.iter().enumerate() {
            let mut eqv = 1u64;
            for (j, &uj) in u.iter().enumerate() {
                let rj = batch.point[j];
                let term = sq_add(
                    sq_mul(rj, uj, q),
                    sq_mul(sq_sub(1, rj, q), sq_sub(1, uj, q), q),
                    q,
                );
                eqv = sq_mul(eqv, term, q);
            }
            er = sq_add(er, sq_mul(alphas[i], eqv, q), q);
        }
        if er == 0 {
            return Ok(false);
        }
        let w = sq_mul(cur, mod_pow(er, q - 2, q), q);
        // The single proof carries the ONE claim under its (random)
        // Mode-1 challenge α₀: recover w = y/α₀ and pin it.
        if batch.single.alphas.len() != 1 {
            return Ok(false);
        }
        let alpha0 = batch.single.alphas[0];
        if alpha0 == 0 {
            return Ok(false);
        }
        let w_single = sq_mul(batch.single.y_combined, mod_pow(alpha0, q - 2, q), q);
        if w_single != w {
            return Ok(false);
        }
        self.batch_single_point_verify(
            std::slice::from_ref(state),
            &batch.point,
            &[w],
            &batch.single,
            transcript,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperwolf::HwParams;

    fn hw() -> HyperWolfFull {
        let params = HwParams::new(8, 2, 3, 32);
        HyperWolfFull::new(params, b"hw-batch-seed")
    }

    fn states(hw: &HyperWolfFull, count: usize) -> Vec<(Vec<i64>, u64, Vec<u64>, HwCommitState)> {
        let params = &hw.params;
        let n = params.n_coeffs();
        let log_b = params.b.trailing_zeros() as usize;
        let log_d = params.d.trailing_zeros() as usize;
        let need = log_b + log_d + (params.k.saturating_sub(1)) * log_b;
        // One SHARED point (Mode 1's premise: a single evaluation
        // point for all polynomials).
        let point: Vec<u64> = (0..need).map(|j| 311 + j as u64 * 57).collect();
        let mut out = Vec::new();
        for i in 0..count {
            let f_ints: Vec<i64> = (0..n)
                .map(|j| (((j + i * 13) * 37) % 197) as i64 - 98)
                .collect();
            let y = hw.evaluate_direct(&f_ints, &point, true);
            let (cm, st) = hw.commit(&f_ints).unwrap();
            let _ = cm;
            out.push((f_ints, y, point.clone(), st));
        }
        out
    }

    #[test]
    fn mode1_batches_and_rejects_wrong_claims() {
        let hw = hw();
        let sts = states(&hw, 3);
        let states_ref: Vec<HwCommitState> = sts.iter().map(|(_, _, _, st)| st.clone()).collect();
        let point = sts[0].2.clone();
        let vs: Vec<u64> = sts.iter().map(|(_, y, _, _)| *y).collect();
        let mut t = Transcript::new_default(b"hw-batch");
        let batch = hw
            .batch_single_point_prove(&states_ref, &point, &vs, &mut t)
            .unwrap();
        let mut tv = Transcript::new_default(b"hw-batch");
        assert!(hw
            .batch_single_point_verify(&states_ref, &point, &vs, &batch, &mut tv)
            .unwrap());
        // A wrong v_i is rejected.
        let mut bad_vs = vs.clone();
        bad_vs[1] = bad_vs[1].wrapping_add(1);
        let mut tv2 = Transcript::new_default(b"hw-batch");
        assert!(!hw
            .batch_single_point_verify(&states_ref, &point, &bad_vs, &batch, &mut tv2)
            .unwrap());
    }

    #[test]
    fn eq_table_matches_direct_evaluation() {
        // The E-table route must reproduce evaluate_direct for a single
        // point: Σ_b eq(u, b)·f[b] == f(u).
        let hw = hw();
        let params = &hw.params;
        let n = params.n_coeffs();
        let f_ints: Vec<i64> = (0..n).map(|j| ((j * 41) % 131) as i64 - 65).collect();
        let num_vars = n.trailing_zeros() as usize;
        let u: Vec<u64> = (0..num_vars).map(|j| 97 + j as u64 * 11).collect();
        let q = hw.ring.q;
        let direct = hw.evaluate_direct(&f_ints, &u, true);
        // eq with the MSB-first bit pairing.
        let mut sum = 0u64;
        for x in 0..n {
            let mut eqv = 1u64;
            for (j, &uj) in u.iter().enumerate() {
                let xj = ((x >> j) & 1) as u64;
                let term = sq_add(sq_mul(xj, uj, q), sq_mul(1 - xj, sq_sub(1, uj, q), q), q);
                eqv = sq_mul(eqv, term, q);
            }
            sum = sq_add(
                sum,
                sq_mul(eqv, f_ints[x].rem_euclid(q as i64) as u64, q),
                q,
            );
        }
        assert_eq!(sum, direct % q);
    }

    #[test]
    fn mode2_multi_point_roundtrip() {
        let hw = hw();
        let sts = states(&hw, 1);
        let (f_ints, _, _, st) = &sts[0];
        let params = &hw.params;
        let log_b = params.b.trailing_zeros() as usize;
        let log_d = params.d.trailing_zeros() as usize;
        let need = log_b + log_d + (params.k.saturating_sub(1)) * log_b;
        // Two distinct points on the same polynomial.
        let u1: Vec<u64> = (0..need).map(|j| 101 + j as u64 * 37).collect();
        let u2: Vec<u64> = (0..need).map(|j| 509 + j as u64 * 71).collect();
        let v1 = hw.evaluate_direct(f_ints, &u1, true);
        let v2 = hw.evaluate_direct(f_ints, &u2, true);
        let mut t = Transcript::new_default(b"hw-batch2");
        let batch = hw
            .batch_multi_point_prove(st, &[u1.clone(), u2.clone()], &[v1, v2], &mut t)
            .unwrap();
        let mut tv = Transcript::new_default(b"hw-batch2");
        assert!(hw
            .batch_multi_point_verify(st, &[u1.clone(), u2.clone()], &[v1, v2], &batch, &mut tv)
            .unwrap());
        // A wrong v is rejected.
        let mut tv2 = Transcript::new_default(b"hw-batch2");
        assert!(!hw
            .batch_multi_point_verify(
                st,
                &[u1.clone(), u2],
                &[v1, v2.wrapping_add(1)],
                &batch,
                &mut tv2
            )
            .unwrap());
    }
}
