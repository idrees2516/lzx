//! HyperWolf H6 — the projection-compaction layer (Wave 7 item 7.12's
//! remaining half): replaces the per-round **clear transmission of the
//! JL projection vectors** (`projections: Vec<Vec<HwElt>>`, the proof's
//! dominant payload — 256 ring elements per slice per round) with
//! per-round Ajtai commitments + ONE terminal reveal.
//!
//! Mechanism (the commitment-linearity discipline):
//! * **Per-round projection commitments** — each round's per-slice
//!   projection vectors `p_i^(r)` are committed under the HyperWolf
//!   keys (`proj_commit`) BEFORE that round's fold challenges are drawn
//!   (the paper's ordering: the compared data is fixed before the
//!   comparison coefficients — the same §4 discipline Akita's fold
//!   follows). The transmitted per-round payload becomes
//!   `(fold, c_mins, proj_commit)` — `κ` ring elements instead of
//!   `256·b`.
//! * **The consistency check on commitments** (check 4 of the clear
//!   protocol, `Σ_i p_i^(r) = Σ_j C_j·p_j^(r−1)`) rides the Ajtai
//!   linearity: `commit(Σ_j C_j p_j) = Σ_j C_j·commit(p_j)` (mod q,
//!   exact), so the verifier checks
//!   `Σ_i c_p,i^(r) == Σ_j C_j^(r−1)·c_p,j^(r−1)` — small ring-scalar
//!   algebra on the commitments, no projection data transmitted.
//! * **The terminal reveal** (LaBRADOR §5.6's tail discipline): the
//!   LAST round's projections travel once (revealed and directly
//!   checked: the norm bound + the commitment opening + the final
//!   `σ⁻¹(Π)·s^(1)` tie), exactly like `s_final` does in the clear
//!   protocol.
//!
//! Honest residual (documented, not hidden): the INTERMEDIATE rounds'
//! per-coefficient norm certificates (check 2's exact
//! `Σ ct(p_j)² ≤ 128·β²` per round) are amortized away with the clear
//! vectors — the compact mode certifies the norms of the revealed
//! terminal projections exactly and carries the intermediate
//! witnesses' smallness through the `c_min` commitment chain's MSIS
//! binding. The full-fidelity route — ONE amortized Dachshund opening
//! over all rounds' projection vectors with per-round ℓ2 statements
//! and the fold-consistency dot-products — requires the LaBRADOR
//! engine re-parameterized to the HyperWolf ring (`Q = 2^48−59` vs
//! `q ≈ 2^61`: the constraint coefficients do not embed); that
//! re-parameterization is the recorded follow-up (NEXT_STEPS H6-resid).
//!
//! Wire format delta (the measurable compaction, pinned by the
//! `hw_compact_size` example): per round, `256·b` ring elements →
//! `b·κ_proj` elements + the one-time terminal reveal of `256·b`.

use crate::hyperwolf::{HwCommitState, HwElt, HwError, HwParams, HyperWolfFull};
use lattice_core::transcript::Transcript;

/// The compact proof: rounds without clear projections + the terminal
/// reveal.
#[derive(Clone, Debug)]
pub struct CompactHwProof {
    /// Per-round clear parts: `(fold, c_mins, proj_commit)`.
    pub rounds: Vec<CompactRound>,
    /// The final witness `s^(1)` (revealed, as in the clear protocol).
    pub s_final: Vec<HwElt>,
    /// The LAST round's per-slice projections (the terminal reveal).
    pub terminal_projections: Vec<Vec<HwElt>>,
}

/// One compact round message.
#[derive(Clone, Debug)]
pub struct CompactRound {
    pub fold: Vec<HwElt>,
    pub c_mins: Vec<Vec<HwElt>>,
    /// The per-slice projection commitments `c_p,i^(r)`.
    pub proj_commit: Vec<Vec<HwElt>>,
}

impl HyperWolfFull {
    /// Commit one slice's projection vector under the sliced keys (the
    /// same discipline as `commit_slices` — the Ajtai binding).
    fn commit_projections(&self, projs: &[Vec<HwElt>]) -> Result<Vec<Vec<HwElt>>, HwError> {
        let mut out = Vec::with_capacity(projs.len());
        for pr in projs {
            out.push(self.keys.commit_slices(&self.ring, pr));
        }
        Ok(out)
    }

    /// H6: the compact evaluation proof — the round loop of
    /// `eval_prove` with the projections committed instead of
    /// transmitted, and the terminal reveal carrying the last round's
    /// projections.
    pub fn eval_prove_compact(
        &self,
        state: &HwCommitState,
        a0_ints: &[u64],
        a_list: &[Vec<u64>],
        transcript: &mut Transcript,
    ) -> Result<CompactHwProof, HwError> {
        let p = &self.params;
        let mut s = state.s.clone();
        let mut c_mins = state.c_mins.clone();
        let a0c = self.a0_ext_conj(a0_ints);
        let jl = self.jl(transcript)?;
        let mut rounds: Vec<CompactRound> = Vec::with_capacity(p.k - 1);
        let mut level = p.k;
        let mut last_projs: Vec<Vec<HwElt>> = Vec::new();
        while level > 1 {
            let fold = crate::hyperwolf::fold_engine(&self.ring, &s, &a0c, &a_list[..level - 2])?;
            let slices = s.slices();
            let block = p.b * p.iota();
            let projs: Vec<Vec<HwElt>> = slices
                .iter()
                .map(|sl| jl.project(sl, block))
                .collect::<Result<Vec<_>, _>>()?;
            // Commit the projections BEFORE the challenges (§4's
            // ordering — the compared data fixed first).
            let proj_commit = self.commit_projections(&projs)?;
            // Absorb the compact round (the commitments stand in for
            // the clear vectors in the transcript).
            self.absorb_compact_round(transcript, &fold, &c_mins, &proj_commit)?;
            rounds.push(CompactRound {
                fold,
                c_mins: c_mins.clone(),
                proj_commit,
            });
            let c = self.draw_challenges(transcript, level)?;
            last_projs = projs;
            s = s.fold_outer(&self.ring, &c);
            let new_slices = s.slices();
            c_mins = new_slices
                .iter()
                .map(|sl| self.keys.commit_slices(&self.ring, sl))
                .collect();
            level -= 1;
        }
        Ok(CompactHwProof {
            rounds,
            s_final: s.flat.clone(),
            terminal_projections: last_projs,
        })
    }

    fn absorb_compact_round(
        &self,
        transcript: &mut Transcript,
        fold: &[HwElt],
        c_mins: &[Vec<HwElt>],
        proj_commit: &[Vec<HwElt>],
    ) -> Result<(), HwError> {
        let mut bytes = Vec::new();
        for f in fold {
            bytes.extend_from_slice(&self.ring.to_bytes(f));
        }
        for cm in c_mins.iter().flatten() {
            bytes.extend_from_slice(&self.ring.to_bytes(cm));
        }
        for pc in proj_commit.iter().flatten() {
            bytes.extend_from_slice(&self.ring.to_bytes(pc));
        }
        transcript
            .append_message(b"hw-compact-round", &bytes)
            .map_err(HwError::Transcript)
    }

    /// H6: the compact verifier — the clear checks (1: fold inner
    /// products, 3: the `c_min` binding chain, the final checks) plus
    /// the commitment-linearity consistency and the terminal reveal's
    /// direct checks.
    #[allow(clippy::too_many_lines)]
    pub fn eval_verify_compact(
        &self,
        cm: &[HwElt],
        a0_ints: &[u64],
        a_list: &[Vec<u64>],
        y_claim: u64,
        proof: &CompactHwProof,
        transcript: &mut Transcript,
    ) -> Result<bool, HwError> {
        let p = &self.params;
        let ring = &self.ring;
        let mut y = ring.zero();
        y.0[0] = y_claim % ring.q;
        let a0c = self.a0_ext_conj(a0_ints);
        let jl = self.jl(transcript)?;
        let mut cm_out = cm.to_vec();
        let mut level = p.k;
        let mut c_hist: Vec<Vec<HwElt>> = Vec::new();
        let mut pc_hist: Vec<Vec<Vec<HwElt>>> = Vec::new();
        let mut c_min_hist: Vec<Vec<Vec<HwElt>>> = Vec::new();
        let n_rounds = proof.rounds.len();
        for (r, msg) in proof.rounds.iter().enumerate() {
            if msg.fold.len() != p.b || msg.c_mins.len() != p.b || msg.proj_commit.len() != p.b {
                return Ok(false);
            }
            // ---- check 1 (clear): <fold, a_{level-1}> == y.
            let a_out = &a_list[level - 2];
            let mut ip = ring.zero();
            for (i, fr) in msg.fold.iter().enumerate() {
                let term = ring.scale(fr, i128::from(a_out[i]));
                ip = ring.add(&ip, &term);
            }
            if r == 0 {
                if ip.0[0] != y.0[0] {
                    return Ok(false);
                }
            } else if ip != y {
                return Ok(false);
            }
            // ---- check 3 (clear): the outer commitment binding chain.
            if r == 0 {
                let stack: Vec<HwElt> = msg.c_mins.iter().flatten().cloned().collect();
                if self
                    .keys
                    .outer_commit(ring, &stack, p.delta_t, p.iota_p())?
                    != cm_out
                {
                    return Ok(false);
                }
            } else if self.keys.outer_commit_from_fold(
                ring,
                &c_min_hist[c_min_hist.len() - 1],
                &c_hist[c_hist.len() - 1],
                p.delta_t,
                p.iota_p(),
            )? != cm_out
            {
                return Ok(false);
            }
            // ---- check 4 (H6): the consistency on COMMITMENTS —
            // Σ_i c_p,i^(r) == Σ_j C_j^(r−1)·c_p,j^(r−1) (Ajtai
            // linearity mod q).
            if r > 0 {
                let prev = &pc_hist[pc_hist.len() - 1];
                let cprev = &c_hist[c_hist.len() - 1];
                let mut lhs = ring.zero();
                for pc in &msg.proj_commit {
                    for x in pc {
                        lhs = ring.add(&lhs, x);
                    }
                }
                let mut rhs = ring.zero();
                for (j, pj) in prev.iter().enumerate() {
                    for x in pj {
                        let term = ring.mul(x, &cprev[j]);
                        rhs = ring.add(&rhs, &term);
                    }
                }
                if lhs != rhs {
                    return Ok(false);
                }
            }
            // ---- the compact round absorption + challenges.
            self.absorb_compact_round(transcript, &msg.fold, &msg.c_mins, &msg.proj_commit)?;
            let c = self.draw_challenges(transcript, level)?;
            y = ring.zero();
            for (i, fr) in msg.fold.iter().enumerate() {
                let term = ring.mul(fr, &c[i]);
                y = ring.add(&y, &term);
            }
            cm_out =
                self.keys
                    .outer_commit_from_fold(ring, &msg.c_mins, &c, p.delta_t, p.iota_p())?;
            c_hist.push(c);
            pc_hist.push(msg.proj_commit.clone());
            c_min_hist.push(msg.c_mins.clone());
            level -= 1;
        }
        // ---------------- The terminal reveal ----------------
        let s1 = &proof.s_final;
        if s1.len() != p.b * p.iota() {
            return Ok(false);
        }
        // (a) <conj(a0_ext), s^(1)> == y (the clear final check).
        let mut ip = ring.zero();
        for j in 0..s1.len() {
            let term = ring.mul(&a0c[j], &s1[j]);
            ip = ring.add(&ip, &term);
        }
        if ip != y {
            return Ok(false);
        }
        // (a') The final witness norm bound (clear protocol's check).
        let mut norm_sq: f64 = 0.0;
        for elt in s1 {
            for &c in &elt.0 {
                let v = ring.center(c) as f64;
                norm_sq += v * v;
            }
        }
        if norm_sq > p.beta(0).powi(2) {
            return Ok(false);
        }
        // (b) The revealed terminal projections: the per-slice norm
        // bound (check 2, exact on the revealed data) + the commitment
        // opening (the last round's proj_commit must BE the commitment
        // of these vectors) + the σ⁻¹(Π)s^(1) tie.
        if n_rounds == 0 || proof.terminal_projections.len() != p.b {
            return Ok(false);
        }
        let bound = (p.jl_rows as f64 / 2.0) * p.beta(1).powi(2);
        for pr in &proof.terminal_projections {
            if pr.len() != p.jl_rows {
                return Ok(false);
            }
            let mut s_sq: f64 = 0.0;
            for x in pr {
                let c0 = ring.center(x.0[0]) as f64;
                s_sq += c0 * c0;
            }
            if s_sq > bound {
                return Ok(false);
            }
        }
        // The commitment opening: recommit the revealed vectors.
        let recomputed = self.commit_projections(&proof.terminal_projections)?;
        if recomputed != proof.rounds[n_rounds - 1].proj_commit {
            return Ok(false);
        }
        // The σ⁻¹(Π)s^(1) tie (the clear protocol's final check (b)):
        // the JL of the revealed flat s^(1) equals the C-fold of the
        // last round's projections (the SAME matrix as the rounds).
        let block = p.b * p.iota();
        let c_last = &c_hist[c_hist.len() - 1];
        let mut lhs: Vec<HwElt> = vec![ring.zero(); p.jl_rows];
        for (j, pj) in proof.terminal_projections.iter().enumerate() {
            for (idx, x) in pj.iter().enumerate() {
                let term = ring.mul(x, &c_last[j]);
                lhs[idx] = ring.add(&lhs[idx], &term);
            }
        }
        let rhs = jl.project(s1, block)?;
        if lhs != rhs {
            return Ok(false);
        }
        // (c) A·s^(1) == Σ_i C_i·c_min,i (the clear final check).
        let lhs_c = self.keys.commit_slices(ring, s1);
        let last_mins = &c_min_hist[c_min_hist.len() - 1];
        let last_c = &c_hist[c_hist.len() - 1];
        let mut rhs_c = vec![ring.zero(); p.kappa()];
        for (i, c_min) in last_mins.iter().enumerate() {
            for (r, x) in c_min.iter().enumerate() {
                let term = ring.mul(x, &last_c[i]);
                rhs_c[r] = ring.add(&rhs_c[r], &term);
            }
        }
        if lhs_c != rhs_c {
            return Ok(false);
        }
        Ok(true)
    }
}

/// The measured wire-format delta: clear vs compact per-round payload
/// (ring elements), at the given shape.
pub fn compact_size_delta(params: &HwParams) -> (usize, usize) {
    let clear_per_round = params.b * (1 + params.jl_rows) + params.b * params.kappa();
    let compact_per_round = params.b * (1 + params.kappa()) + params.b * params.kappa();
    (clear_per_round, compact_per_round)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperwolf::paper_params;

    fn small_params() -> HwParams {
        HwParams::new(8, 2, 3, 32)
    }

    fn setup(seed: &[u8]) -> (HyperWolfFull, Vec<i64>, Vec<u64>, Vec<Vec<u64>>, u64) {
        let params = small_params();
        let hw = HyperWolfFull::new(params.clone(), seed);
        let n = params.n_coeffs();
        let f_ints: Vec<i64> = (0..n).map(|i| ((i * 37) % 251) as i64 - 125).collect();
        let log_b = params.b.trailing_zeros() as usize;
        let log_d = params.d.trailing_zeros() as usize;
        let need = log_b + log_d + (params.k.saturating_sub(1)) * log_b;
        let point: Vec<u64> = (0..need).map(|i| 1234 + i as u64 * 997).collect();
        let (a0_ints, a_list) =
            crate::hyperwolf::build_a_multilinear(&hw.ring, &point, params.k, params.b, params.d)
                .unwrap();
        let y = hw.evaluate_direct(&f_ints, &point, true);
        (hw, f_ints, a0_ints, a_list, y)
    }

    #[test]
    fn compact_prove_verify_roundtrip() {
        let (hw, f_ints, a0_ints, a_list, y) = setup(b"hw-c-1");
        let (cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-compact");
        let proof = hw
            .eval_prove_compact(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        let mut tv = Transcript::new_default(b"hw-compact");
        assert!(hw
            .eval_verify_compact(&cm, &a0_ints, &a_list, y, &proof, &mut tv)
            .unwrap());
        // A wrong y is rejected.
        let mut tv2 = Transcript::new_default(b"hw-compact");
        assert!(!hw
            .eval_verify_compact(&cm, &a0_ints, &a_list, y.wrapping_add(1), &proof, &mut tv2)
            .unwrap());
    }

    #[test]
    fn compact_tampered_proj_commit_rejected() {
        let (hw, f_ints, a0_ints, a_list, y) = setup(b"hw-c-2");
        let (cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-compact");
        let mut proof = hw
            .eval_prove_compact(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        // Tamper a projection commitment: the commitment-linearity
        // consistency (or the terminal opening) fails.
        let mut bytes = proof.rounds[0].proj_commit[0][0].0.clone();
        bytes[0] = bytes[0].wrapping_add(1);
        let elt = HwElt(bytes);
        proof.rounds[0].proj_commit[0][0] = elt;
        let mut tv = Transcript::new_default(b"hw-compact");
        assert!(!hw
            .eval_verify_compact(&cm, &a0_ints, &a_list, y, &proof, &mut tv)
            .unwrap());
    }

    #[test]
    fn compact_tampered_terminal_projection_rejected() {
        let (hw, f_ints, a0_ints, a_list, y) = setup(b"hw-c-3");
        let (cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-compact");
        let mut proof = hw
            .eval_prove_compact(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        // Tamper the revealed terminal projection: the recomputed
        // commitment no longer matches (or the σ⁻¹(Π)s tie fails).
        let mut bytes = proof.terminal_projections[0][0].0.clone();
        bytes[0] = bytes[0].wrapping_add(1);
        proof.terminal_projections[0][0] = HwElt(bytes);
        let mut tv = Transcript::new_default(b"hw-compact");
        assert!(!hw
            .eval_verify_compact(&cm, &a0_ints, &a_list, y, &proof, &mut tv)
            .unwrap());
    }

    #[test]
    fn compact_tampered_fold_rejected() {
        let (hw, f_ints, a0_ints, a_list, y) = setup(b"hw-c-4");
        let (cm, state) = hw.commit(&f_ints).unwrap();
        let mut t = Transcript::new_default(b"hw-compact");
        let mut proof = hw
            .eval_prove_compact(&state, &a0_ints, &a_list, &mut t)
            .unwrap();
        let mut bytes = proof.rounds[0].fold[0].0.clone();
        bytes[1] = bytes[1].wrapping_add(1);
        proof.rounds[0].fold[0] = HwElt(bytes);
        let mut tv = Transcript::new_default(b"hw-compact");
        assert!(!hw
            .eval_verify_compact(&cm, &a0_ints, &a_list, y, &proof, &mut tv)
            .unwrap());
    }

    #[test]
    fn compact_size_delta_measures_the_win() {
        // The per-round payload shrinks by the full projection vectors.
        let params = paper_params(1 << 15);
        let (clear, compact) = compact_size_delta(&params);
        assert!(compact < clear);
        // The projections dominate: clear ≈ b·(1 + 256) + b·κ.
        assert!(clear > 256 * params.b);
        // At the paper's b=2: the per-round delta ≈ 2·256 elements.
        assert!(clear - compact >= 256 * params.b.saturating_sub(1));
    }
}
