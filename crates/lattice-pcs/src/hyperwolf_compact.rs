//! H6 (Wave 7.12): LaBRADOR-style compaction for the HyperWolf full
//! protocol — a proof-size-reducing wrapper layer over
//! [`HyperWolfFull`](crate::hyperwolf::HyperWolfFull).
//!
//! # What the paper asks for and what lands here
//!
//! HyperWolf's core proof is O(log N) ring elements: per folding round the
//! prover transmits `fold ∈ R_q^b`, `b` JL projections in
//! `R_q^{jl_rows}`, and `b` slice commitments `c_min ∈ R_q^κ`, plus the
//! final `s^(1) ∈ R_q^{b·ι}`. The lab backlog (NEXT_STEPS §3.7 H6 /
//! §8.10) asks for **LaBRADOR compaction** of that transcript — the
//! amortization pattern from LaBRADOR/Dachshund (Beullens–Seiler): commit
//! to each instance, fold everything with random challenges, open ONE
//! amortized vector.
//!
//! This module implements the honest kernel-scale subset of that pattern:
//!
//! * **Round messages are committed, not revealed.** For every round `r`
//!   the prover transmits `cm_r = A·blocksum(pad(m_r)) ∈ R_q^κ` — a
//!   *linear* Ajtai commitment (the same transparent `A` key) to the
//!   bulky message `m_r = (projections, c_mins)` — instead of the
//!   `b·(jl_rows + κ)` elements themselves. The per-round fold vectors
//!   (b elements — the y-chain carriers), round-0's `c_mins` (the public
//!   commitment binding), the **last** round's full message (the final
//!   checks' input), and `s^(1)` stay in the clear.
//! * **One amortized RLC opening.** After the transcript absorbs
//!   everything, the verifier draws small scalars `λ_r ∈ [1, 256]` and
//!   the prover sends the single vector `z = Σ_r λ_r·m_r` over the
//!   hidden (middle) rounds — one round-message-shaped object regardless
//!   of how many rounds were compacted.
//! * **The amortized binding check** (LaBRADOR's commitment fold):
//!   `A·blocksum(pad(z)) == Σ_r λ_r·cm_r` — exact linear algebra; by
//!   linearity the honest prover passes, and under the MSIS binding of
//!   every `cm_r` the equation pins `z` to the committed messages.
//! * **The aggregate certified norm check** (LaBRADOR's norm-slack): the
//!   projection block of `z` satisfies `‖z_proj‖² ≤ (SLACK·√(b·jl_rows/2)
//!   · Σ_r λ_r·β_r)²` — the Cauchy–Schwarz/triangle aggregate of the
//!   per-round JL bounds, with a documented completeness slack (the same
//!   trade LaBRADOR makes explicit with its 128/30 norm slack).
//!
//! # Soundness analysis (honest)
//!
//! Compared to the full verifier
//! ([`HyperWolfFull::eval_verify`](crate::hyperwolf::HyperWolfFull::eval_verify))
//! the compact verifier **keeps** every load-bearing check:
//! round-0 outer-commitment binding (`B·G^{-1}(c_mins^{(0)}) == cm`),
//! the per-round evaluation identity and y-chain (check 1, folds are
//! clear), the last round's JL norms (check 2) and the final pinning
//! `(a) ⟨s^(1), σ^{-1}(a0_ext)⟩ == y`, `(b) σ^{-1}(Π)s^(1) ==
//! Σ C_j p_j^{(last)}`, `(c) A s^(1) == Σ C_j c_min,j^{(last)}`, and the
//! `‖s^(1)‖ ≤ β^{(0)}` bound. It **adds** the per-round MSIS commitments
//! `cm_r` (absorbed into the transcript *before* the round's challenges
//! are drawn — LaBRADOR's commit-then-challenge structure: the prover is
//! computationally bound to `m_r` at challenge time) and the amortized
//! fold/norm checks above, which is *strictly more* binding than the
//! statement-chain check it replaces (that check was already vacuous in
//! the ported full verifier — module deviation 1).
//!
//! It **drops** for the hidden middle rounds: the per-round JL norm
//! checks (check 2), the cross-round projection chain (check 4), and the
//! per-round statement chain (check 3, vacuous in the port anyway). The
//! soundness consequence is documented precisely: the compact layer is a
//! *size-reduced, weaker-soundness profile* of the full protocol. What
//! survives end-to-end is the standard binding spine (round-0 MSIS
//! binding → final pinning through the clear last round and `s^(1)`);
//! what is lost is the middle rounds' individual norm discipline and
//! chain consistency, which the paper's round-by-round extraction
//! argument (Theorem 5, (2T)^{k-1} slack machinery) consumes. A cheater
//! who stuffs garbage into hidden-round messages is caught by the
//! amortized fold check (MSIS) unless the *RLC* of the garbage is
//! norm-bounded — i.e. the aggregate check bounds the hidden material
//! only in combination, with the slack factor below.
//!
//! # Why full LaBRADOR integration is infeasible here (documented gap)
//!
//! `lattice-labrador` (this workspace's native Dachshund port) exposes
//! exactly the right engine — a simple-statement language of norm-bounded
//! witness vectors plus verifier-known linear dot-product constraints,
//! with the full inner/outer garbage-commitment amortization — but its
//! substrate is `Z_Q[X]/(X^64+1)` at `Q = 2^48−59` with **i16 witness
//! lanes**, while HyperWolf's messages live mod `q = 2^61−259` in
//! dimension 8. Carrying a 61-bit coefficient through i16 lanes needs a
//! limb split; the modular (mod-q) linear identities then need per-bit
//! quotient witnesses with range constraints that LaBRADOR's
//! language does not express (its constraints are exact linear forms mod
//! Q, and its norm specs are l2/binary only). That re-parameterization is
//! the "L, after re-parameterization" blocker NEXT_STEPS §3.7 already
//! flags. The paper-scale O(log log log N) target additionally needs
//! LaBRADOR's quadratic-garbage inner commitments re-instantiated over
//! the HyperWolf keys — future work, tracked as the honest remainder.
//!
//! # Size accounting (ring elements)
//!
//! | regime | full | compact | ratio |
//! |---|---|---|---|
//! | lab (d=8, b=2, k=6, jl=64, κ=34) | 1024 | 478 | 2.14× |
//! | paper model (k=24, jl=256, κ=34) | 13 400 | 1 510 | 8.9× |
//!
//! The compact proof is still O(k) = O(log N) — the *constant* of the
//! per-round term drops from `b·(jl_rows+κ)` to `κ` (≈17× at paper
//! scale) plus a one-time `b·(jl_rows+κ)` amortized opening. "Toward"
//! O(log log log N), not there: reaching sub-logarithmic size is exactly
//! the full-LaBRADOR-integration item above.

use crate::hyperwolf::{
    fold_engine, HwCommitState, HwElt, HwError, HwParams, HwProof, HyperWolfFull, JlMatrix,
};
use lattice_core::transcript::Transcript;

/// Completeness slack of the aggregate norm check over the Cauchy–Schwarz
/// tight bound (LaBRADOR's norm-slack analogue; documented in the module
/// docs — soundness of the aggregate check is correspondingly slack).
pub const COMPACT_NORM_SLACK: f64 = 4.0;

/// The compacted HyperWolf evaluation proof.
#[derive(Clone, Debug)]
pub struct CompactHwProof {
    /// Per-round fold vectors (b elements each) — the y-chain carriers,
    /// kept in the clear.
    pub folds: Vec<Vec<HwElt>>,
    /// Round 0's slice commitments `c_min,i^{(0)}` (κ each) — the public
    /// commitment binding (round-0 check 3), in the clear.
    pub c_mins_r0: Vec<Vec<HwElt>>,
    /// Per-round linear Ajtai commitments `cm_r = A·blocksum(pad(m_r))`
    /// (κ each) to every round's bulky message.
    pub cm_rounds: Vec<Vec<HwElt>>,
    /// The last round's per-slice JL projections (jl_rows each), in the
    /// clear — final check (b) input.
    pub last_projections: Vec<Vec<HwElt>>,
    /// The last round's slice commitments (κ each), in the clear — final
    /// check (c) input.
    pub last_c_mins: Vec<Vec<HwElt>>,
    /// The amortized RLC opening `z = Σ_r λ_r·m_r` over the hidden
    /// middle rounds (empty when none exist).
    pub z: Vec<HwElt>,
    /// The final witness `s^(1)` (b·ι elements), in the clear.
    pub s_final: Vec<HwElt>,
}

impl CompactHwProof {
    /// Transmitted ring elements (honest size accounting: the verifier
    /// reads every element of every field).
    pub fn size_elements(&self) -> usize {
        let sum = |vs: &[Vec<HwElt>]| -> usize {
            vs.iter().map(|v| v.len()).sum::<usize>()
        };
        sum(&self.folds)
            + sum(&self.c_mins_r0)
            + sum(&self.cm_rounds)
            + sum(&self.last_projections)
            + sum(&self.last_c_mins)
            + self.z.len()
            + self.s_final.len()
    }
}

/// Ring-element count of a full (uncompacted) `HwProof`.
pub fn full_proof_size_elements(proof: &HwProof) -> usize {
    proof
        .rounds
        .iter()
        .map(|r| {
            r.fold.len()
                + r.projections.iter().map(|p| p.len()).sum::<usize>()
                + r.c_mins.iter().map(|c| c.len()).sum::<usize>()
        })
        .sum::<usize>()
        + proof.s_final.len()
}

/// Round indices whose bulky messages are hidden behind `cm_r` and the
/// amortized opening: rounds `1..=k−3` (round 0 and the last round
/// `k−2` stay in the clear; rounds are `0..=k−2`).
fn hidden_rounds(params: &HwParams) -> Vec<usize> {
    (1..params.k.saturating_sub(2)).collect()
}

/// The per-round bulky message layout: `[projections (b·jl_rows),
/// c_mins (b·κ)]`, slice-major within each part.
fn stack_message(projections: &[Vec<HwElt>], c_mins: &[Vec<HwElt>]) -> Vec<HwElt> {
    let mut out = Vec::with_capacity(
        projections.iter().map(|p| p.len()).sum::<usize>()
            + c_mins.iter().map(|c| c.len()).sum::<usize>(),
    );
    for p in projections {
        out.extend(p.iter().cloned());
    }
    for c in c_mins {
        out.extend(c.iter().cloned());
    }
    out
}

/// Zero-pad a flat vector to a multiple of `block` (the `A`-key's column
/// tiling: `commit_slices` block-sums `a_cols`-sized blocks and would
/// silently drop a ragged tail).
fn pad_to(ring: &crate::hyperwolf::HwRing, flat: &[HwElt], block: usize) -> Vec<HwElt> {
    let mut out = flat.to_vec();
    let rem = out.len() % block;
    if rem != 0 {
        out.resize(out.len() + (block - rem), ring.zero());
    }
    out
}

/// A small positive RLC scalar `λ ∈ [1, 256]`: invertible (q is prime and
/// λ ≠ 0) and norm-preserving for the aggregate bound (|λ| ≤ 256).
fn sample_lambda(transcript: &mut Transcript, idx: usize) -> Result<u64, HwError> {
    let label = format!("hw:compact:lambda:{}", idx);
    let bytes = transcript.challenge_bytes(label.as_bytes(), 8)?;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[..8]);
    Ok(u64::from_le_bytes(arr) % 256 + 1)
}

impl HyperWolfFull {
    fn absorb_elt(transcript: &mut Transcript, label: &[u8], x: &HwElt) -> Result<(), HwError> {
        transcript.append_bytes(label, &self_ring_bytes(x))?;
        Ok(())
    }

    /// Compact prover: runs the honest protocol over the committed
    /// hypercube (identical witness folding), but transmits per round only
    /// the fold vector and the linear Ajtai commitment `cm_r` to the bulky
    /// message; everything else rides the amortized opening `z`.
    ///
    /// The transcript layout is the compact protocol's own: the public
    /// statement (cm, a-vectors, y) is absorbed first (a Fiat–Shamir
    /// binding improvement over the original protocol, which derives its
    /// challenges statement-independently), then per round
    /// `[round marker, folds, (round 0: c_mins), (last round:
    /// projections + c_mins), cm_r]`, then the challenges; `s^(1)` and
    /// the λ's close the stream.
    pub fn eval_prove_compact(
        &self,
        cm: &[HwElt],
        state: &HwCommitState,
        a0_ints: &[u64],
        a_list: &[Vec<u64>],
        y_claim: u64,
        transcript: &mut Transcript,
    ) -> Result<CompactHwProof, HwError> {
        let p = &self.params;
        if p.k < 2 {
            return Err(HwError::Shape { expected: 2, got: p.k });
        }
        let ring = &self.ring;
        // -- 0. bind the public statement
        for x in cm {
            Self::absorb_elt(transcript, b"hw:cp:cm", x)?;
        }
        let mut a0_bytes = Vec::with_capacity(8 * a0_ints.len());
        for &a in a0_ints {
            a0_bytes.extend_from_slice(&a.to_le_bytes());
        }
        transcript.append_bytes(b"hw:cp:a0", &a0_bytes)?;
        let mut ai_bytes = Vec::new();
        for ai in a_list {
            for &a in ai {
                ai_bytes.extend_from_slice(&a.to_le_bytes());
            }
        }
        transcript.append_bytes(b"hw:cp:ai", &ai_bytes)?;
        transcript.append_bytes(b"hw:cp:y", &y_claim.to_le_bytes())?;
        // -- 1. same JL derivation as the original protocol
        let jl = self.jl(transcript)?;
        let a0c = self.a0_ext_conj(a0_ints);
        // -- 2. the folding rounds
        let last = p.k - 2;
        let hidden = hidden_rounds(p);
        let mut s = state.s.clone();
        let mut c_mins = state.c_mins.clone();
        let mut folds: Vec<Vec<HwElt>> = Vec::with_capacity(p.k - 1);
        let mut cm_rounds: Vec<Vec<HwElt>> = Vec::with_capacity(p.k - 1);
        let mut all_projs: Vec<Vec<Vec<HwElt>>> = Vec::with_capacity(p.k - 1);
        let mut all_cmins: Vec<Vec<Vec<HwElt>>> = Vec::with_capacity(p.k - 1);
        let mut messages: Vec<Vec<HwElt>> = Vec::with_capacity(p.k - 1);
        let mut level = p.k;
        while level > 1 {
            let r = p.k - level;
            let fold = fold_engine(ring, &s, &a0c, &a_list[..level - 2])?;
            let slices = s.slices();
            let block = p.b * p.iota();
            let projs: Vec<Vec<HwElt>> = slices
                .iter()
                .map(|sl| jl.project(sl, block))
                .collect::<Result<Vec<_>, _>>()?;
            // -- compact round absorption
            transcript.append_bytes(b"hw:cp:round", b"")?;
            for fr in &fold {
                Self::absorb_elt(transcript, b"hw:cp:fold", fr)?;
            }
            if r == 0 {
                for cmn in &c_mins {
                    for x in cmn {
                        Self::absorb_elt(transcript, b"hw:cp:cmin0", x)?;
                    }
                }
            }
            if r == last {
                for pr in &projs {
                    for x in pr {
                        Self::absorb_elt(transcript, b"hw:cp:proj", x)?;
                    }
                }
                for cmn in &c_mins {
                    for x in cmn {
                        Self::absorb_elt(transcript, b"hw:cp:cmin", x)?;
                    }
                }
            }
            // -- per-round linear Ajtai commitment to the bulky message
            let m_r = stack_message(&projs, &c_mins);
            let padded = pad_to(ring, &m_r, self.keys.a_cols);
            let cm_r = self.keys.commit_slices(ring, &padded);
            for x in &cm_r {
                Self::absorb_elt(transcript, b"hw:cp:cmr", x)?;
            }
            // -- challenges (identical sampler + labels as the original)
            let c = self.draw_challenges(transcript, level)?;
            folds.push(fold);
            cm_rounds.push(cm_r);
            all_projs.push(projs);
            all_cmins.push(c_mins.clone());
            messages.push(m_r);
            // -- witness fold (identical to the original driver)
            s = s.fold_outer(ring, &c);
            let new_slices = s.slices();
            c_mins = new_slices
                .iter()
                .map(|sl| self.keys.commit_slices(ring, sl))
                .collect();
            level -= 1;
        }
        // -- 3. absorb s^(1), then draw the RLC scalars for hidden rounds
        for x in &s.flat {
            Self::absorb_elt(transcript, b"hw:cp:sfin", x)?;
        }
        let mut z: Vec<HwElt> = Vec::new();
        if !hidden.is_empty() {
            let lambdas: Vec<u64> = (0..hidden.len())
                .map(|i| sample_lambda(transcript, i))
                .collect::<Result<Vec<_>, _>>()?;
            let block_len = messages[0].len();
            z = vec![ring.zero(); block_len];
            for (i, &r) in hidden.iter().enumerate() {
                for j in 0..block_len {
                    let term = ring.scale(&messages[r][j], i128::from(lambdas[i]));
                    z[j] = ring.add(&z[j], &term);
                }
            }
        }
        Ok(CompactHwProof {
            folds,
            c_mins_r0: all_cmins[0].clone(),
            cm_rounds,
            last_projections: all_projs[last].clone(),
            last_c_mins: all_cmins[last].clone(),
            z,
            s_final: s.flat.clone(),
        })
    }

    /// Compact verifier: the load-bearing checks of the full protocol
    /// over the clear material, plus the amortized fold/norm checks over
    /// the hidden rounds. See the module docs for the honest soundness
    /// profile.
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
        if p.k < 2 {
            return Err(HwError::Shape { expected: 2, got: p.k });
        }
        let ring = &self.ring;
        let last = p.k - 2;
        let hidden = hidden_rounds(p);
        let bulky = p.b * (p.jl_rows + p.kappa());
        let z_expected = if hidden.is_empty() { 0 } else { bulky };
        // ---- shape checks (reject malformed proofs outright)
        if proof.folds.len() != p.k - 1
            || proof.cm_rounds.len() != p.k - 1
            || proof.s_final.len() != p.b * p.iota()
            || proof.c_mins_r0.len() != p.b
            || proof.last_projections.len() != p.b
            || proof.last_c_mins.len() != p.b
            || proof.z.len() != z_expected
        {
            return Ok(false);
        }
        for pr in &proof.last_projections {
            if pr.len() != p.jl_rows {
                return Ok(false);
            }
        }
        for cmn in proof.last_c_mins.iter().chain(proof.c_mins_r0.iter()) {
            if cmn.len() != p.kappa() {
                return Ok(false);
            }
        }
        for cmr in &proof.cm_rounds {
            if cmr.len() != p.kappa() {
                return Ok(false);
            }
        }
        for f in &proof.folds {
            if f.len() != p.b {
                return Ok(false);
            }
        }
        // ---- 0. public statement absorption (symmetric with the prover)
        for x in cm {
            Self::absorb_elt(transcript, b"hw:cp:cm", x)?;
        }
        let mut a0_bytes = Vec::with_capacity(8 * a0_ints.len());
        for &a in a0_ints {
            a0_bytes.extend_from_slice(&a.to_le_bytes());
        }
        transcript.append_bytes(b"hw:cp:a0", &a0_bytes)?;
        let mut ai_bytes = Vec::new();
        for ai in a_list {
            for &a in ai {
                ai_bytes.extend_from_slice(&a.to_le_bytes());
            }
        }
        transcript.append_bytes(b"hw:cp:ai", &ai_bytes)?;
        transcript.append_bytes(b"hw:cp:y", &y_claim.to_le_bytes())?;
        // ---- 1. JL matrix + conjugated a0 expansion
        let jl = self.jl(transcript)?;
        let a0c = self.a0_ext_conj(a0_ints);
        // ---- 2. the rounds
        let mut y = ring.zero();
        y.0[0] = y_claim % ring.q;
        let mut c_last: Vec<HwElt> = Vec::new();
        for (r, fold) in proof.folds.iter().enumerate() {
            let level = p.k - r;
            // -- absorb the round (symmetric)
            transcript.append_bytes(b"hw:cp:round", b"")?;
            for fr in fold {
                Self::absorb_elt(transcript, b"hw:cp:fold", fr)?;
            }
            if r == 0 {
                for cmn in &proof.c_mins_r0 {
                    for x in cmn {
                        Self::absorb_elt(transcript, b"hw:cp:cmin0", x)?;
                    }
                }
            }
            if r == last {
                for pr in &proof.last_projections {
                    for x in pr {
                        Self::absorb_elt(transcript, b"hw:cp:proj", x)?;
                    }
                }
                for cmn in &proof.last_c_mins {
                    for x in cmn {
                        Self::absorb_elt(transcript, b"hw:cp:cmin", x)?;
                    }
                }
            }
            for x in &proof.cm_rounds[r] {
                Self::absorb_elt(transcript, b"hw:cp:cmr", x)?;
            }
            // -- check 1 (evaluation identity / y-chain)
            let a_out = &a_list[level - 2];
            let mut ip = ring.zero();
            for (i, fr) in fold.iter().enumerate() {
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
            // -- round-0 check 3 (public commitment binding)
            if r == 0 {
                let stack: Vec<HwElt> =
                    proof.c_mins_r0.iter().flatten().cloned().collect();
                if self
                    .keys
                    .outer_commit(ring, &stack, p.delta_t, p.iota_p())?
                    != cm
                {
                    return Ok(false);
                }
            }
            // -- last-round check 2 (JL norms of the clear projections)
            if r == last {
                let bound = (p.jl_rows as f64 / 2.0) * p.beta(level - 1).powi(2);
                for pr in &proof.last_projections {
                    let mut s_sq: f64 = 0.0;
                    for x in pr {
                        let c0 = ring.center(x.0[0]) as f64;
                        s_sq += c0 * c0;
                    }
                    if s_sq > bound {
                        return Ok(false);
                    }
                }
            }
            // -- challenges + y update
            let c = self.draw_challenges(transcript, level)?;
            y = ring.zero();
            for (i, fr) in fold.iter().enumerate() {
                let term = ring.mul(fr, &c[i]);
                y = ring.add(&y, &term);
            }
            c_last = c;
            // NOTE (documented deviation): the per-round statement-chain
            // check 3 and cross-round projection chain check 4 of the full
            // protocol are NOT run for hidden rounds — replaced by the
            // amortized fold + aggregate norm checks below. See module
            // docs for the soundness profile.
        }
        // ---- 3. s^(1) absorption + RLC scalars (symmetric)
        for x in &proof.s_final {
            Self::absorb_elt(transcript, b"hw:cp:sfin", x)?;
        }
        let lambdas: Vec<u64> = (0..hidden.len())
            .map(|i| sample_lambda(transcript, i))
            .collect::<Result<Vec<_>, _>>()?;
        // ---- 4. amortized checks over the hidden rounds
        if !hidden.is_empty() {
            // (i) LaBRADOR commitment fold: A·blocksum(pad(z)) == Σ λ_r cm_r.
            let padded_z = pad_to(ring, &proof.z, self.keys.a_cols);
            let lhs = self.keys.commit_slices(ring, &padded_z);
            let mut rhs = vec![ring.zero(); p.kappa()];
            for (i, &r) in hidden.iter().enumerate() {
                for (row, x) in proof.cm_rounds[r].iter().enumerate() {
                    let term = ring.scale(x, i128::from(lambdas[i]));
                    rhs[row] = ring.add(&rhs[row], &term);
                }
            }
            if lhs != rhs {
                return Ok(false);
            }
            // (ii) aggregate certified norm over the projection block of z:
            // ‖z_proj‖² ≤ (SLACK·√(b·jl/2)·Σ λ_r·β_r)² — the triangle-
            // inequality aggregate of the hidden rounds' JL bounds.
            let mut s_sq: f64 = 0.0;
            for x in &proof.z[..p.b * p.jl_rows] {
                for &coef in &x.0 {
                    let v = ring.center(coef) as f64;
                    s_sq += v * v;
                }
            }
            let mut agg = 0.0f64;
            for (i, &r) in hidden.iter().enumerate() {
                agg += f64::from(lambdas[i]) * p.beta(p.k - r - 1);
            }
            let bound = COMPACT_NORM_SLACK * (p.b as f64 * p.jl_rows as f64 / 2.0).sqrt() * agg;
            if s_sq > bound.powi(2) {
                return Ok(false);
            }
        }
        // ---- 5. final checks (identical to the full verifier)
        let s1 = &proof.s_final;
        // (a) <conj(a0_ext), s^(1)> == y
        let mut ip = ring.zero();
        for j in 0..s1.len() {
            let term = ring.mul(&a0c[j], &s1[j]);
            ip = ring.add(&ip, &term);
        }
        if ip != y {
            return Ok(false);
        }
        // norm bound ||s^(1)||² <= beta^(0)^2
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
        // (b) sigma_{-1}(Pi) s^(1) == Σ_i C_i p_i^(last)
        let mut lhs = vec![ring.zero(); p.jl_rows];
        for (j, pj) in proof.last_projections.iter().enumerate() {
            for (idx, x) in pj.iter().enumerate() {
                let term = ring.mul(x, &c_last[j]);
                lhs[idx] = ring.add(&lhs[idx], &term);
            }
        }
        let rhs = jl.project(s1, p.b * p.iota())?;
        if lhs != rhs {
            return Ok(false);
        }
        // (c) A s^(1) == Σ_i C_i t_i (per commitment row)
        let lhs_c = self.keys.commit_slices(ring, s1);
        let mut rhs_c = vec![ring.zero(); p.kappa()];
        for (i, c_min) in proof.last_c_mins.iter().enumerate() {
            for (row, x) in c_min.iter().enumerate() {
                let term = ring.mul(x, &c_last[i]);
                rhs_c[row] = ring.add(&rhs_c[row], &term);
            }
        }
        if lhs_c != rhs_c {
            return Ok(false);
        }
        Ok(true)
    }
}

fn self_ring_bytes(x: &HwElt) -> Vec<u8> {
    x.0.iter().flat_map(|&c| c.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperwolf::{build_a_univariate, HwRing, HW_Q61};

    fn small_ints(n: usize, tag: &[u8], span: i64) -> Vec<i64> {
        (0..n)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"hw-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    4,
                );
                let mut arr = [0u8; 4];
                arr.copy_from_slice(&bytes[..4]);
                (u32::from_le_bytes(arr) as i64 % (2 * span + 1)) - span
            })
            .collect()
    }

    fn run_compact_e2e(k: usize) {
        let params = HwParams::new(8, 2, k, 64);
        let hw = HyperWolfFull::new(params.clone(), b"hw-compact-seed");
        let f_ints = small_ints(params.n_coeffs(), b"fc", 8);
        let (cm, state) = hw.commit(&f_ints).ok().unwrap();
        let u: u64 = 7;
        let y_direct = hw.evaluate_direct(&f_ints, &[u], false);
        let (a0, a_list) = build_a_univariate(&hw.ring, u, k, 2, 8);
        // ---- full proof (for the size comparison)
        let mut t_full = Transcript::new_default(b"lzx-hw-compact");
        let full = hw
            .eval_prove(&state, &a0, &a_list, &mut t_full)
            .ok()
            .unwrap();
        // ---- compact proof
        let mut t = Transcript::new_default(b"lzx-hw-compact");
        let proof = hw
            .eval_prove_compact(&cm, &state, &a0, &a_list, y_direct, &mut t)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"lzx-hw-compact");
        let ok = hw
            .eval_verify_compact(&cm, &a0, &a_list, y_direct, &proof, &mut vt)
            .ok()
            .unwrap();
        assert!(ok, "k={} compact roundtrip", k);
        // ---- size: strictly smaller than the full proof at k >= 5
        let full_size = full_proof_size_elements(&full);
        let compact_size = proof.size_elements();
        assert!(
            compact_size < full_size,
            "k={}: compact {} !< full {}",
            k,
            compact_size,
            full_size
        );
        // ---- wrong y rejected
        let mut vt2 = Transcript::new_default(b"lzx-hw-compact");
        let bad = hw
            .eval_verify_compact(&cm, &a0, &a_list, (y_direct + 1) % HW_Q61, &proof, &mut vt2)
            .ok()
            .unwrap();
        assert!(!bad, "k={} wrong-y accepted", k);
        // ---- tampered amortized opening z rejected (fold check)
        if !proof.z.is_empty() {
            let mut tp = proof.clone();
            let z0 = tp.z[0].clone();
            tp.z[0] = hw.ring.add(&z0, &hw.ring.one());
            let mut vt3 = Transcript::new_default(b"lzx-hw-compact");
            let bad2 = hw
                .eval_verify_compact(&cm, &a0, &a_list, y_direct, &tp, &mut vt3)
                .ok()
                .unwrap();
            assert!(!bad2, "k={} tampered-z accepted", k);
        }
        // ---- tampered per-round commitment rejected (transcript + fold)
        {
            let mut tp = proof.clone();
            let c0 = tp.cm_rounds[1][0].clone();
            tp.cm_rounds[1][0] = hw.ring.add(&c0, &hw.ring.one());
            let mut vt4 = Transcript::new_default(b"lzx-hw-compact");
            let bad3 = hw
                .eval_verify_compact(&cm, &a0, &a_list, y_direct, &tp, &mut vt4)
                .ok()
                .unwrap();
            assert!(!bad3, "k={} tampered-cm-round accepted", k);
        }
        // ---- tampered fold rejected (check 1)
        {
            let mut tp = proof.clone();
            let f0 = tp.folds[0][0].clone();
            tp.folds[0][0] = hw.ring.add(&f0, &hw.ring.one());
            let mut vt5 = Transcript::new_default(b"lzx-hw-compact");
            let bad4 = hw
                .eval_verify_compact(&cm, &a0, &a_list, y_direct, &tp, &mut vt5)
                .ok()
                .unwrap();
            assert!(!bad4, "k={} tampered-fold accepted", k);
        }
        // ---- tampered s^(1) rejected (final checks)
        {
            let mut tp = proof.clone();
            let s0 = tp.s_final[0].clone();
            tp.s_final[0] = hw.ring.add(&s0, &hw.ring.one());
            let mut vt6 = Transcript::new_default(b"lzx-hw-compact");
            let bad5 = hw
                .eval_verify_compact(&cm, &a0, &a_list, y_direct, &tp, &mut vt6)
                .ok()
                .unwrap();
            assert!(!bad5, "k={} tampered-s-final accepted", k);
        }
        // ---- tampered last-round projection rejected (final check b)
        {
            let mut tp = proof.clone();
            let p0 = tp.last_projections[0][0].clone();
            tp.last_projections[0][0] = hw.ring.add(&p0, &hw.ring.one());
            let mut vt7 = Transcript::new_default(b"lzx-hw-compact");
            let bad6 = hw
                .eval_verify_compact(&cm, &a0, &a_list, y_direct, &tp, &mut vt7)
                .ok()
                .unwrap();
            assert!(!bad6, "k={} tampered-last-proj accepted", k);
        }
        // ---- tampered round-0 c_mins rejected (public commitment binding)
        {
            let mut tp = proof.clone();
            let c0 = tp.c_mins_r0[0][0].clone();
            tp.c_mins_r0[0][0] = hw.ring.add(&c0, &hw.ring.one());
            let mut vt8 = Transcript::new_default(b"lzx-hw-compact");
            let bad7 = hw
                .eval_verify_compact(&cm, &a0, &a_list, y_direct, &tp, &mut vt8)
                .ok()
                .unwrap();
            assert!(!bad7, "k={} tampered-cmin0 accepted", k);
        }
        // ---- tampered public commitment rejected
        {
            let mut cm_bad = cm.clone();
            let c0 = cm_bad[0].clone();
            cm_bad[0] = hw.ring.add(&c0, &hw.ring.one());
            let mut vt9 = Transcript::new_default(b"lzx-hw-compact");
            let bad8 = hw
                .eval_verify_compact(&cm_bad, &a0, &a_list, y_direct, &proof, &mut vt9)
                .ok()
                .unwrap();
            assert!(!bad8, "k={} tampered-cm accepted", k);
        }
    }

    #[test]
    fn compact_roundtrip_and_tamper_k5_k6() {
        run_compact_e2e(5);
        run_compact_e2e(6);
    }

    #[test]
    fn compact_small_k_degenerates_honestly() {
        // k = 3: no hidden rounds; the compact proof equals the clear
        // transcript material and still verifies.
        let params = HwParams::new(8, 2, 3, 64);
        let hw = HyperWolfFull::new(params, b"hw-compact-seed");
        let ring = HwRing::new(HW_Q61, 8);
        let f_ints = small_ints(hw.params.n_coeffs(), b"fc3", 8);
        let (cm, state) = hw.commit(&f_ints).ok().unwrap();
        let u: u64 = 5;
        let y = hw.evaluate_direct(&f_ints, &[u], false);
        let (a0, a_list) = build_a_univariate(&ring, u, 3, 2, 8);
        let mut t = Transcript::new_default(b"lzx-hw-compact-small");
        let proof = hw
            .eval_prove_compact(&cm, &state, &a0, &a_list, y, &mut t)
            .ok()
            .unwrap();
        assert!(proof.z.is_empty());
        assert_eq!(hidden_rounds(&hw.params), Vec::<usize>::new());
        let mut vt = Transcript::new_default(b"lzx-hw-compact-small");
        let ok = hw
            .eval_verify_compact(&cm, &a0, &a_list, y, &proof, &mut vt)
            .ok()
            .unwrap();
        assert!(ok);
    }

    #[test]
    fn compact_paper_scale_size_model() {
        // Pure size arithmetic at the paper's 2^30 point (no execution):
        // full per round b·(1 + jl + κ) + final b·ι; compact =
        // folds b·(k−1) + round-0 c_mins b·κ + per-round cm κ·(k−1) +
        // amortized z b·(jl+κ) + s^(1) b·ι. Still O(log N) — the constant
        // drops ~κ/(b·(jl+κ)) ≈ 17× — the honest "toward" claim.
        let params = crate::hyperwolf::paper_params(1 << 30);
        let k = params.k;
        let b = params.b;
        let jl = params.jl_rows;
        let kappa = params.kappa();
        let iota = params.iota();
        let full = (k - 1) * b * (1 + jl + kappa) + b * iota;
        let compact = (k - 1) * b + b * kappa + (k - 1) * kappa + b * (jl + kappa) + b * iota;
        let ratio = full as f64 / compact as f64;
        // ~8.9× at k = 24 — the module-doc table.
        assert!(ratio > 8.0, "ratio {}", ratio);
        assert!(ratio < 9.5, "ratio {}", ratio);
        // Asymptotic constant: the per-round term shrinks by the
        // κ / (b·(jl+κ)) factor (≈ 1/17 at paper scale).
        let per_round_shrink = (kappa as f64) / (b * (jl + kappa)) as f64;
        assert!(per_round_shrink < 1.0 / 16.0);
        // Monotone in k: bigger polynomials compact better (the amortized
        // opening is one-time).
        let p20 = crate::hyperwolf::paper_params(1 << 20);
        let full20 =
            (p20.k - 1) * p20.b * (1 + p20.jl_rows + p20.kappa()) + p20.b * p20.iota();
        let compact20 = (p20.k - 1) * p20.b
            + p20.b * p20.kappa()
            + (p20.k - 1) * p20.kappa()
            + p20.b * (p20.jl_rows + p20.kappa())
            + p20.b * p20.iota();
        assert!(full as f64 / compact as f64 > full20 as f64 / compact20 as f64);
    }

    #[test]
    fn compact_lab_size_beats_full() {
        // Executable lab-scale accounting (k = 6): 478 vs 1024 elements.
        let k = 6usize;
        let params = HwParams::new(8, 2, k, 64);
        let b = params.b;
        let jl = params.jl_rows;
        let kappa = params.kappa();
        let iota = params.iota();
        let full = (k - 1) * b * (1 + jl + kappa) + b * iota;
        let compact = (k - 1) * b + b * kappa + (k - 1) * kappa + b * (jl + kappa) + b * iota;
        assert_eq!(full, 1024);
        assert_eq!(compact, 478);
        // And the executed proofs match the model.
        let hw = HyperWolfFull::new(params.clone(), b"hw-compact-seed");
        let f_ints = small_ints(params.n_coeffs(), b"sz", 8);
        let (cm, state) = hw.commit(&f_ints).ok().unwrap();
        let u: u64 = 7;
        let y = hw.evaluate_direct(&f_ints, &[u], false);
        let (a0, a_list) = build_a_univariate(&hw.ring, u, k, 2, 8);
        let mut t_full = Transcript::new_default(b"lzx-hw-size");
        let full_proof = hw.eval_prove(&state, &a0, &a_list, &mut t_full).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-hw-size");
        let compact_proof = hw
            .eval_prove_compact(&cm, &state, &a0, &a_list, y, &mut t)
            .ok()
            .unwrap();
        assert_eq!(full_proof_size_elements(&full_proof), full);
        assert_eq!(compact_proof.size_elements(), compact);
    }
}
