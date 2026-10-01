//! **Stage 4 — the leg batching** (DESIGN_50KB.md's specified final cut;
//! the T&S §4.2.1 random-power RLC machinery over the staged legs).
//!
//! The per-instance protocol (`memory.rs`) proves each of the 9 memory
//! instances with 13 (RW) / 3 (fetch) sum-check legs — ~117 sumchecks
//! whose round messages dominate the proof's non-opening bulk
//! (55–70 KB measured). This module re-structures the SAME legs into
//! **staged, dependency-ordered batches** via
//! [`prove_batch`](lattice_sumcheck::batch::prove_batch): each batch is
//! ONE sum-check whose message count is `max_rounds × max_degree`
//! regardless of the batch's size — the projected legs communication
//! drops 55–70 KB → ~8–12 KB.
//!
//! ## The staged protocol (dependency-safe order)
//!
//! Instances group into cube classes by `(log_rows, log_ts)` — the
//! register file (4), RAM (4), and the fetch instance (its `log_ts` is
//! `log_t`, not `2 + log_t`, so it is always its own class).
//!
//! | stage | legs | scope | cube |
//! |---|---|---|---|
//! | A | `B, R` per instance | per class | `log_rows + log_ts` |
//! | B | `C (,+ W, + T)` per instance | per class | `log_k + log_ts` |
//! | fetch-Ma | `Ma` of the fetch instance | standalone | `log_t` |
//! | 2 (read) | `Ma, V0` × 8 RW | **global** | `log_ts` |
//! | 3 | `Mu0` × 8 | global | `log_ts` |
//! | 4 (write) | `Mb, Mc, V1` × 8 | global | `log_ts` |
//! | 5 | `Mu1` × 8 | global | `log_ts` |
//! | 6 | `Md` × 8 | global | `log_ts` |
//!
//! The dependency order is preserved exactly: stage B's shared terminal
//! `r̄_CWT` feeds the read/write/tel groups' evaluation points; stage 2's
//! terminal `r̄_read` feeds stage 3's `Mu0` points (the `u`-factor
//! claims); stage 4's `r̄_write` feeds stage 5. `T` and `Md` stay
//! separate legs in different stages (different cubes), as specified.
//!
//! **Transcript stability**: every leg keeps its `absorb(inst, name)`
//! discipline and its pre-sumcheck challenge derivations, drawn per
//! stage BEFORE the batch — the leg names and claim semantics are
//! unchanged from the per-instance protocol (a protocol revision at the
//! envelope level, not a claim-semantics break).
//!
//! **Ledger discipline**: the prover records and the verifier pops the
//! base claims in the IDENTICAL sequence (stage by stage, instance by
//! instance, dedup'd per `(factor, point)` — e.g. `B` and `R` share the
//! stage-A terminal's `DigitBits` claim). Shared terminals concentrate
//! the ledger claims (one `DigitBits@r̄` serves both binds), which is
//! also what keeps the values-only claim list small.
//!
//! **Claim transmission**: the matrix/Val evaluations that the
//! per-instance protocol carried per-leg (`ra_at`, `val_at`, `u_at`,
//! `wa_at`, `inc_at`, …) become the batches' claimed-sum vectors —
//! pinned by the terminal identities of their own stages (each computed
//! from ledger claims at the batch terminals), exactly the pinning
//! structure the per-leg `LegProof::claim` had: every transmitted value
//! terminates in the ledger, whose values-only claim list is
//! authenticated by the bundle openings.

use crate::ledger::{idx_point, Factor, Ledger};
use crate::memory::{
    absorb, activity_factor, addr_factor, booleanity_vp_at, digit_affine, inc_factor, matrix_claim,
    matrix_vp, raf_vp, read_vp, rv_factor, tel_claim, val_vp, wv_factor, write_vp, MatrixKind,
    MemoryError, MemoryInstance, INC_OFFSET,
};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::batch::BatchClaim;
use lattice_sumcheck::sumcheck::{self, SumcheckProof};
use lattice_sumcheck::sumcheck::SumcheckOutput;
use lattice_sumcheck::VirtualPolynomial;

/// The batched legs' proof: 12 sumchecks total (3 stage-A + 3 stage-B +
/// fetch-Ma + 5 global) replacing the ~117 per-instance legs.
#[derive(Clone, Debug)]
pub struct BatchedLegs {
    /// Stage A (B+R) per cube class, in class order.
    pub br: Vec<SumcheckProof>,
    /// Stage B (C/W/T) per cube class, in class order.
    pub cwt: Vec<SumcheckProof>,
    /// The fetch instance's standalone `Ma` (the `log_t` cube).
    pub fetch_ma: SumcheckProof,
    /// Stage 2: the global read group `{Ma, V0} × 8 RW`.
    pub read: SumcheckProof,
    /// Stage 3: the global `{Mu0} × 8`.
    pub mu0: SumcheckProof,
    /// Stage 4: the global write group `{Mb, Mc, V1} × 8`.
    pub write: SumcheckProof,
    /// Stage 5: the global `{Mu1} × 8`.
    pub mu1: SumcheckProof,
    /// Stage 6: the global `{Md} × 8`.
    pub md: SumcheckProof,
    // ---- the transmitted evaluation claims (pinned by their stages'
    // terminal identities) ----
    /// `ra_at` per RW instance + the fetch instance's (last).
    pub ra_claims: Vec<Goldilocks>,
    /// `val_at` per RW instance at the read points (the `V0` claims).
    pub val_read_claims: Vec<Goldilocks>,
    /// `u` at `r̄_read` per RW instance (the `Mu0` claims).
    pub u_read_claims: Vec<Goldilocks>,
    /// `wa_at` per RW instance (the `Mb` claims).
    pub wa_claims: Vec<Goldilocks>,
    /// `inc_at_w` per RW instance (the `Mc` claims).
    pub inc_w_claims: Vec<Goldilocks>,
    /// `val_at` per RW instance at the write points (the `V1` claims).
    pub val_write_claims: Vec<Goldilocks>,
    /// `u` at `r̄_write` per RW instance (the `Mu1` claims).
    pub u_write_claims: Vec<Goldilocks>,
    /// `inc_at_tel` per RW instance (the `Md` claims).
    pub inc_tel_claims: Vec<Goldilocks>,
}

impl BatchedLegs {
    /// Iterate all 12 batched sumcheck proofs (size accounting and
    /// serialization).
    pub fn sumchecks(&self) -> Vec<&SumcheckProof> {
        let mut out: Vec<&SumcheckProof> = Vec::with_capacity(12);
        out.extend(self.br.iter());
        out.extend(self.cwt.iter());
        out.push(&self.fetch_ma);
        out.push(&self.read);
        out.push(&self.mu0);
        out.push(&self.write);
        out.push(&self.mu1);
        out.push(&self.md);
        out
    }
}

/// The cube classes: instances grouped by `(log_rows, log_ts)` — the
/// register file, RAM, and the fetch instance (always distinct: its
/// `log_ts` is `log_t`).
fn cube_classes(instances: &[MemoryInstance]) -> Vec<Vec<usize>> {
    let mut classes: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (i, m) in instances.iter().enumerate() {
        let key = (m.log_rows(), m.log_ts);
        if let Some(c) = classes.iter_mut().find(|(r, t, _)| *r == key.0 && *t == key.1) {
            c.2.push(i);
        } else {
            classes.push((key.0, key.1, vec![i]));
        }
    }
    classes.into_iter().map(|(_, _, idxs)| idxs).collect()
}

/// The RW instances in index order (excluding the read-only fetch).
fn rw_instances(instances: &[MemoryInstance]) -> Vec<usize> {
    (0..instances.len())
        .filter(|&i| !instances[i].read_only())
        .collect()
}

/// A staged batch under construction: the claims' virtual polynomials +
/// their claimed sums + the GLOBAL factor-offset bookkeeping (each
/// claim's factors are appended to the combined pool in order).
struct StageBatch {
    vps: Vec<VirtualPolynomial>,
    claims: Vec<Goldilocks>,
    /// The global factor index where each claim's factors begin.
    offsets: Vec<usize>,
    num_vars: usize,
    max_degree: usize,
}

impl StageBatch {
    fn new(num_vars: usize) -> Self {
        StageBatch {
            vps: Vec::new(),
            claims: Vec::new(),
            offsets: Vec::new(),
            num_vars,
            max_degree: 1,
        }
    }

    /// Push one leg: returns its leg index.
    fn push(&mut self, vp: VirtualPolynomial, claim: Goldilocks) -> usize {
        let offs = self.vps.iter().map(|v| v.factors.len()).sum();
        self.offsets.push(offs);
        self.max_degree = self.max_degree.max(vp.max_degree());
        self.num_vars = vp.num_vars;
        self.vps.push(vp);
        self.claims.push(claim);
        self.claims.len() - 1
    }

    /// The global factor indices of leg `leg`'s factors.
    fn leg_slice(&self, leg: usize) -> std::ops::Range<usize> {
        let base = self.offsets[leg];
        let len = self.vps[leg].factors.len();
        base..base + len
    }

    fn batch_claims(&self) -> Vec<BatchClaim<'_>> {
        self.vps
            .iter()
            .zip(self.claims.iter())
            .map(|(v, c)| BatchClaim { poly: v, claimed_sum: *c })
            .collect()
    }
}

/// Prove one staged batch.
fn prove_stage(
    stage: &StageBatch,
    transcript: &mut Transcript,
) -> Result<(SumcheckOutput, Vec<Goldilocks>), MemoryError> {
    let refs = stage.batch_claims();
    let (out, rhos) =
        lattice_sumcheck::batch::prove_batch(&refs, transcript).map_err(MemoryError::Batch)?;
    Ok((out, rhos))
}

/// Replay one staged batch on the verifier: draws the same `batch-rho`
/// challenges, forms the combined claimed sum, verifies the round chain,
/// and returns (rhos, terminal point, final claim).
fn replay_stage(
    sc: &SumcheckProof,
    num_vars: usize,
    max_degree: usize,
    claimed: &[Goldilocks],
    transcript: &mut Transcript,
) -> Result<(Vec<Goldilocks>, Vec<Goldilocks>, Goldilocks), MemoryError> {
    let rhos = transcript
        .challenge_fields(b"batch-rho", claimed.len())
        .map_err(MemoryError::Transcript)?;
    let mut combined = Goldilocks::ZERO;
    for (rho, c) in rhos.iter().zip(claimed.iter()) {
        combined = combined.add(&rho.mul(c));
    }
    let verdict = sc
        .verify(num_vars, max_degree, combined, transcript, None)
        .map_err(MemoryError::Sumcheck)?;
    Ok((rhos, verdict.point, verdict.final_claim))
}

/// Prover-side fail-closed binds of one matrix-eval leg's factor claims
/// (the `bind_matrix_factors` of the per-instance protocol, at the
/// batch's shared terminal). `fc` is the leg's factor-claim slice in the
/// batch's global indexing.
#[allow(clippy::too_many_arguments)]
fn bind_matrix_leg(
    inst: usize,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    rho_k: &[Goldilocks],
    rho_j: &[Goldilocks],
    terminal: &[Goldilocks],
    fc: &[Goldilocks],
    kind: MatrixKind,
) -> Result<(), MemoryError> {
    // Factor order: [eq, activity, digit-affine_b..., (inc-part)?].
    let eq_claim = fc[0];
    if eq_claim != DenseMle::eq_eval(rho_j, terminal).map_err(MemoryError::Mle)? {
        return Err(MemoryError::ClaimMismatch);
    }
    let act_claim = fc[1];
    let derived = ledger.tensor_claim(activity_factor(inst, kind != MatrixKind::Ra), terminal)?;
    if derived != act_claim {
        return Err(MemoryError::ClaimMismatch);
    }
    for b in 0..m.log_k {
        let mut point = idx_point(m.log_rows(), b);
        point.extend_from_slice(terminal);
        let row = ledger.tensor_claim(Factor::DigitBits { inst }, &point)?;
        if digit_affine(row, rho_k[b]) != fc[2 + b] {
            return Err(MemoryError::ClaimMismatch);
        }
    }
    if kind == MatrixKind::Inc {
        let inc_claim = *fc.last().ok_or(MemoryError::Shape)?;
        let derived = ledger.tensor_claim(inc_factor(inst), terminal)?;
        if derived.sub(&Goldilocks::from_u64(INC_OFFSET)) != inc_claim {
            return Err(MemoryError::ClaimMismatch);
        }
    }
    Ok(())
}

/// The matrix-eval terminal identity from ledger claims (the verifier's
/// `check_matrix_terminal` at a batch terminal).
#[allow(clippy::too_many_arguments)]
fn matrix_terminal_expect(
    inst: usize,
    m: &MemoryInstance,
    ledger: &mut Ledger<'_>,
    rho_k: &[Goldilocks],
    rho_j: &[Goldilocks],
    point: &[Goldilocks],
    kind: MatrixKind,
) -> Result<Goldilocks, MemoryError> {
    let mut expect = DenseMle::eq_eval(rho_j, point).map_err(MemoryError::Mle)?;
    let act = ledger.tensor_claim(activity_factor(inst, kind != MatrixKind::Ra), point)?;
    expect = expect.mul(&act);
    for b in 0..m.log_k {
        let mut pt = idx_point(m.log_rows(), b);
        pt.extend_from_slice(point);
        let row = ledger.tensor_claim(Factor::DigitBits { inst }, &pt)?;
        expect = expect.mul(&digit_affine(row, rho_k[b]));
    }
    if kind == MatrixKind::Inc {
        let inc = ledger.tensor_claim(inc_factor(inst), point)?;
        expect = expect.mul(&inc.sub(&Goldilocks::from_u64(INC_OFFSET)));
    }
    Ok(expect)
}

/// Prove the memory argument's legs in the batched staged protocol.
#[allow(clippy::too_many_lines)]
pub fn prove_legs_batched(
    instances: &[MemoryInstance],
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<BatchedLegs, MemoryError> {
    if instances.len() != 9 {
        return Err(MemoryError::Shape);
    }
    let classes = cube_classes(instances);
    let rw = rw_instances(instances);
    let fetch_idx = instances.len() - 1;

    // ================= Stage A: B+R per class =================
    let mut br_proofs = Vec::with_capacity(classes.len());
    let mut r_bool: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut r_prime: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    for cls in &classes {
        for &i in cls {
            let m = &instances[i];
            absorb(transcript, i, "B", m.log_k, m.log_ts)?;
            r_bool[i] = transcript
                .challenge_fields(b"mem-bool-r", m.log_rows() + m.log_ts)
                .map_err(MemoryError::Transcript)?;
            absorb(transcript, i, "R", m.log_k, m.log_ts)?;
            r_prime[i] = transcript
                .challenge_fields(b"mem-raf-r", m.log_ts)
                .map_err(MemoryError::Transcript)?;
        }
        let mut stage = StageBatch::new(instances[cls[0]].log_rows() + instances[cls[0]].log_ts);
        for &i in cls {
            let m = &instances[i];
            stage.push(booleanity_vp_at(m, &r_bool[i])?, Goldilocks::ZERO);
            let addr_claim = ledger.tensor_claim(addr_factor(i), &r_prime[i])?;
            stage.push(raf_vp(m, &r_prime[i])?, addr_claim);
        }
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        let r_br = out.challenges.clone();
        // Fail-closed binds at the shared terminal: ONE DigitBits claim
        // per instance serves B and R (the dedup'd ledger call).
        for (leg, &i) in cls.iter().enumerate() {
            let m = &instances[i];
            let log_rows = m.log_rows();
            let d_at = ledger.tensor_claim(Factor::DigitBits { inst: i }, &r_br)?;
            let bfc = &out.factor_claims[stage.leg_slice(2 * leg)];
            if bfc[0] != d_at
                || bfc[1] != d_at.sub(&Goldilocks::ONE)
                || bfc[2] != DenseMle::eq_eval(&r_bool[i], &r_br).map_err(MemoryError::Mle)?
            {
                return Err(MemoryError::FinalCheck("batched booleanity binding"));
            }
            let rfc = &out.factor_claims[stage.leg_slice(2 * leg + 1)];
            let eq_at =
                DenseMle::eq_eval(&r_prime[i], &r_br[log_rows..]).map_err(MemoryError::Mle)?;
            let w_at = m.raf_weights().evaluate(&r_br[..log_rows]).map_err(MemoryError::Mle)?;
            if rfc[2] != d_at || rfc[0] != eq_at || rfc[1] != w_at {
                return Err(MemoryError::FinalCheck("batched raf binding"));
            }
        }
        br_proofs.push(out.proof);
    }

    // ================= Stage B: C (+W +T) per class =================
    let mut cwt_proofs = Vec::with_capacity(classes.len());
    let mut r_c: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut r_w: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut r_t: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut cwt_term: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    for cls in &classes {
        for &i in cls {
            let m = &instances[i];
            absorb(transcript, i, "C", m.log_k, m.log_ts)?;
            r_c[i] = transcript
                .challenge_fields(b"mem-read-r", m.log_ts)
                .map_err(MemoryError::Transcript)?;
            if !m.read_only() {
                absorb(transcript, i, "W", m.log_k, m.log_ts)?;
                r_w[i] = transcript
                    .challenge_fields(b"mem-write-r", m.log_k + m.log_ts)
                    .map_err(MemoryError::Transcript)?;
                absorb(transcript, i, "T", m.log_k, m.log_ts)?;
                r_t[i] = transcript
                    .challenge_fields(b"mem-tel-r", m.log_k)
                    .map_err(MemoryError::Transcript)?;
            }
        }
        let mut stage = StageBatch::new(instances[cls[0]].log_k + instances[cls[0]].log_ts);
        // (leg index, instance) bookkeeping for the binds.
        let mut w_legs: Vec<(usize, usize)> = Vec::new();
        for &i in cls {
            let m = &instances[i];
            let rv_claim = ledger.tensor_claim(rv_factor(i), &r_c[i])?;
            stage.push(read_vp(m, &r_c[i])?, rv_claim);
            if !m.read_only() {
                let wl = stage.push(write_vp(m, &r_w[i])?, Goldilocks::ZERO);
                w_legs.push((wl, i));
                stage.push(
                    crate::memory::telescoping_vp(m, &r_t[i])?,
                    tel_claim(m, &r_t[i])?,
                );
            }
        }
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        let r_cwt = out.challenges.clone();
        // Fail-closed binds: W's wv column claim at the terminal's j-part.
        for (wl, i) in &w_legs {
            let m = &instances[*i];
            let wv_at = ledger.tensor_claim(wv_factor(*i), &r_cwt[m.log_k..])?;
            let wfc = &out.factor_claims[stage.leg_slice(*wl)];
            // W's factors: [eq, inc, wa, wv-lift, val].
            if wfc[3] != wv_at {
                return Err(MemoryError::FinalCheck("batched write wv binding"));
            }
            let eq_at = DenseMle::eq_eval(&r_w[*i], &r_cwt).map_err(MemoryError::Mle)?;
            if wfc[0] != eq_at {
                return Err(MemoryError::FinalCheck("batched write eq binding"));
            }
        }
        for &i in cls {
            cwt_term[i] = r_cwt.clone();
        }
        cwt_proofs.push(out.proof);
    }

    // ================= The fetch instance's standalone Ma =================
    let fetch_ma_proof;
    let fetch_ra_at;
    {
        let m = &instances[fetch_idx];
        absorb(transcript, fetch_idx, "Ma", m.log_k, m.log_ts)?;
        let point = cwt_term[fetch_idx].clone();
        let (rho_k, rho_j) = point.split_at(m.log_k);
        let claim = matrix_claim(m, &point, MatrixKind::Ra)?;
        fetch_ra_at = claim;
        let vp = matrix_vp(m, rho_k, rho_j, MatrixKind::Ra)?;
        let out = sumcheck::prove(&vp, claim, transcript).map_err(MemoryError::Sumcheck)?;
        let nf = 2 + m.log_k;
        bind_matrix_leg(
            fetch_idx,
            m,
            ledger,
            rho_k,
            rho_j,
            &out.challenges,
            &out.factor_claims[..nf],
            MatrixKind::Ra,
        )?;
        fetch_ma_proof = out.proof;
    }

    // ================= Stage 2: the global read group =================
    let mut ra_claims = Vec::with_capacity(rw.len() + 1);
    let mut val_read_claims = Vec::with_capacity(rw.len());
    let read_out;
    {
        let mut stage = StageBatch::new(instances[rw[0]].log_ts);
        for &i in &rw {
            let m = &instances[i];
            let point = cwt_term[i].clone();
            let (rho_k, rho_j) = point.split_at(m.log_k);
            let ra = matrix_claim(m, &point, MatrixKind::Ra)?;
            ra_claims.push(ra);
            stage.push(matrix_vp(m, rho_k, rho_j, MatrixKind::Ra)?, ra);
            let (vp, val, sum_claim) = val_vp(m, &point)?;
            val_read_claims.push(val);
            stage.push(vp, sum_claim);
        }
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        read_out = out;
    }
    let r_read = read_out.challenges.clone();
    // The u-factor claims at r̄_read (stage 3's input claims) + the
    // fail-closed binds.
    let mut u_read_claims = Vec::with_capacity(rw.len());
    for (vi, &i) in rw.iter().enumerate() {
        let m = &instances[i];
        let point = &cwt_term[i];
        let (rho_k, rho_j) = point.split_at(m.log_k);
        // V0's factors [u, lt]: the lt target is the evaluation point's
        // j-part (the CWT terminal's j-part — `val_vp`'s construction).
        let vfc = &read_out.factor_claims[read_out_local_slice(&rw, instances, vi, 1, 2)];
        u_read_claims.push(vfc[0]);
        let lt_at = DenseMle::lt_extension(&r_read, rho_j).map_err(MemoryError::Mle)?;
        if vfc[1] != lt_at {
            return Err(MemoryError::FinalCheck("batched read lt binding"));
        }
        // Ma's binds.
        let mfc = &read_out.factor_claims[read_out_local_slice(&rw, instances, vi, 0, 2 + m.log_k)];
        bind_matrix_leg(
            i, m, ledger, rho_k, rho_j, &r_read, mfc, MatrixKind::Ra,
        )?;
    }

    // ================= Stage 3: the global Mu0 group =================
    let mu0_out;
    {
        let mut stage = StageBatch::new(instances[rw[0]].log_ts);
        for &i in &rw {
            let m = &instances[i];
            let point = &cwt_term[i];
            let r_a = &point[..m.log_k];
            stage.push(matrix_vp(m, r_a, &r_read, MatrixKind::Inc)?, Goldilocks::ZERO);
        }
        // Replace the placeholder claims with the u-factor claims.
        stage.claims = u_read_claims.clone();
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        mu0_out = out;
    }
    let r_mu0 = mu0_out.challenges.clone();
    for (vi, &i) in rw.iter().enumerate() {
        let m = &instances[i];
        let point = &cwt_term[i];
        let r_a = &point[..m.log_k];
        let nf = 3 + m.log_k;
        let mfc = &mu0_out.factor_claims[stage_slice(&rw, instances, vi, nf)];
        bind_matrix_leg(
            i, m, ledger, r_a, &r_read, &r_mu0, mfc, MatrixKind::Inc,
        )?;
    }

    // ================= Stage 4: the global write group =================
    let mut wa_claims = Vec::with_capacity(rw.len());
    let mut inc_w_claims = Vec::with_capacity(rw.len());
    let mut val_write_claims = Vec::with_capacity(rw.len());
    let write_out;
    {
        let mut stage = StageBatch::new(instances[rw[0]].log_ts);
        for &i in &rw {
            let m = &instances[i];
            let point = cwt_term[i].clone();
            let (rho_k, rho_j) = point.split_at(m.log_k);
            let wa = matrix_claim(m, &point, MatrixKind::Wa)?;
            wa_claims.push(wa);
            stage.push(matrix_vp(m, rho_k, rho_j, MatrixKind::Wa)?, wa);
            let inc = matrix_claim(m, &point, MatrixKind::Inc)?;
            inc_w_claims.push(inc);
            stage.push(matrix_vp(m, rho_k, rho_j, MatrixKind::Inc)?, inc);
            let (vp, val, sum_claim) = val_vp(m, &point)?;
            val_write_claims.push(val);
            stage.push(vp, sum_claim);
        }
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        write_out = out;
    }
    let r_write = write_out.challenges.clone();
    // The u-factor claims at r̄_write (stage 5's input claims) + binds.
    let mut u_write_claims = Vec::with_capacity(rw.len());
    for (vi, &i) in rw.iter().enumerate() {
        let m = &instances[i];
        let point = &cwt_term[i];
        let (rho_k, rho_j) = point.split_at(m.log_k);
        let mb_nf = 2 + m.log_k;
        let mc_nf = 3 + m.log_k;
        // The per-instance triple (Mb, Mc, V1) — accumulate the actual
        // per-instance factor counts.
        let mut base = 0usize;
        for v in 0..vi {
            let mv = &instances[rw[v]];
            base += (2 + mv.log_k) + (3 + mv.log_k) + 2;
        }
        let vfc = &write_out.factor_claims[base + mb_nf + mc_nf..base + mb_nf + mc_nf + 2];
        u_write_claims.push(vfc[0]);
        let lt_at = DenseMle::lt_extension(&r_write, rho_j).map_err(MemoryError::Mle)?;
        if vfc[1] != lt_at {
            return Err(MemoryError::FinalCheck("batched write lt binding"));
        }
        // ONE combined Mb+Mc bind: the activity and digit-row ledger
        // claims are SHARED at the same terminal (a single record each —
        // the verifier's second pass hits the seen cache without
        // popping, so the queue stays in sync).
        {
            let mfc_b = &write_out.factor_claims[base..base + mb_nf];
            let mfc_c = &write_out.factor_claims[base + mb_nf..base + mb_nf + mc_nf];
            let eq_at = DenseMle::eq_eval(rho_j, &r_write).map_err(MemoryError::Mle)?;
            let act = ledger.tensor_claim(activity_factor(i, true), &r_write)?;
            let mut rows = Vec::with_capacity(m.log_k);
            for b in 0..m.log_k {
                let mut pt = idx_point(m.log_rows(), b);
                pt.extend_from_slice(&r_write);
                rows.push(ledger.tensor_claim(Factor::DigitBits { inst: i }, &pt)?);
            }
            let inc = ledger.tensor_claim(inc_factor(i), &r_write)?;
            // Mb (Wa-kind): [eq, activity, digit-affines...]
            if mfc_b[0] != eq_at || mfc_b[1] != act {
                return Err(MemoryError::ClaimMismatch);
            }
            for b in 0..m.log_k {
                if mfc_b[2 + b] != digit_affine(rows[b], rho_k[b]) {
                    return Err(MemoryError::ClaimMismatch);
                }
            }
            // Mc (Inc-kind): [eq, activity, digit-affines..., inc-part]
            if mfc_c[0] != eq_at || mfc_c[1] != act {
                return Err(MemoryError::ClaimMismatch);
            }
            for b in 0..m.log_k {
                if mfc_c[2 + b] != digit_affine(rows[b], rho_k[b]) {
                    return Err(MemoryError::ClaimMismatch);
                }
            }
            if *mfc_c.last().ok_or(MemoryError::Shape)?
                != inc.sub(&Goldilocks::from_u64(INC_OFFSET))
            {
                return Err(MemoryError::ClaimMismatch);
            }
        }
    }

    // ================= Stage 5: the global Mu1 group =================
    let mu1_out;
    {
        let mut stage = StageBatch::new(instances[rw[0]].log_ts);
        for &i in &rw {
            let m = &instances[i];
            let point = &cwt_term[i];
            let r_a = &point[..m.log_k];
            stage.push(matrix_vp(m, r_a, &r_write, MatrixKind::Inc)?, Goldilocks::ZERO);
        }
        stage.claims = u_write_claims.clone();
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        mu1_out = out;
    }
    let r_mu1 = mu1_out.challenges.clone();
    for (vi, &i) in rw.iter().enumerate() {
        let m = &instances[i];
        let point = &cwt_term[i];
        let r_a = &point[..m.log_k];
        let nf = 3 + m.log_k;
        let mfc = &mu1_out.factor_claims[stage_slice(&rw, instances, vi, nf)];
        bind_matrix_leg(
            i, m, ledger, r_a, &r_write, &r_mu1, mfc, MatrixKind::Inc,
        )?;
    }

    // ================= Stage 6: the global Md group =================
    let mut inc_tel_claims = Vec::with_capacity(rw.len());
    let md_out;
    {
        let mut stage = StageBatch::new(instances[rw[0]].log_ts);
        for &i in &rw {
            let m = &instances[i];
            let point = cwt_term[i].clone();
            let (rho_k, rho_j) = point.split_at(m.log_k);
            let inc = matrix_claim(m, &point, MatrixKind::Inc)?;
            inc_tel_claims.push(inc);
            stage.push(matrix_vp(m, rho_k, rho_j, MatrixKind::Inc)?, inc);
        }
        let (out, _rhos) = prove_stage(&stage, transcript)?;
        md_out = out;
    }
    let r_md = md_out.challenges.clone();
    for (vi, &i) in rw.iter().enumerate() {
        let m = &instances[i];
        let point = &cwt_term[i];
        let (rho_k, rho_j) = point.split_at(m.log_k);
        let nf = 3 + m.log_k;
        let mfc = &md_out.factor_claims[stage_slice(&rw, instances, vi, nf)];
        bind_matrix_leg(
            i, m, ledger, rho_k, rho_j, &r_md, mfc, MatrixKind::Inc,
        )?;
    }

    ra_claims.push(fetch_ra_at);

    Ok(BatchedLegs {
        br: br_proofs,
        cwt: cwt_proofs,
        fetch_ma: fetch_ma_proof,
        read: read_out.proof,
        mu0: mu0_out.proof,
        write: write_out.proof,
        mu1: mu1_out.proof,
        md: md_out.proof,
        ra_claims,
        val_read_claims,
        u_read_claims,
        wa_claims,
        inc_w_claims,
        val_write_claims,
        u_write_claims,
        inc_tel_claims,
    })
}

/// The global factor range of RW instance `vi`'s `leg_in_triple`-th leg
/// where each triple is `(Ma, V0)` with factor counts
/// `(2 + log_k, 2)` (stage 2's shape).
fn read_out_local_slice(
    rw: &[usize],
    instances: &[MemoryInstance],
    vi: usize,
    leg_in_pair: usize,
    _nf: usize,
) -> std::ops::Range<usize> {
    let mut base = 0usize;
    for v in 0..vi {
        let m = &instances[rw[v]];
        base += (2 + m.log_k) + 2;
    }
    let m = &instances[rw[vi]];
    let offs = if leg_in_pair == 0 { 0 } else { 2 + m.log_k };
    base + offs..base + offs + if leg_in_pair == 0 { 2 + m.log_k } else { 2 }
}

/// The global factor range of RW instance `vi`'s leg in a uniform
/// factor-count stage (stages 3/5/6: `nf = 3 + log_k` — still
/// instance-specific, so accumulate).
fn stage_slice(
    rw: &[usize],
    instances: &[MemoryInstance],
    vi: usize,
    _nf: usize,
) -> std::ops::Range<usize> {
    let mut base = 0usize;
    for v in 0..vi {
        let m = &instances[rw[v]];
        base += 3 + m.log_k;
    }
    let m = &instances[rw[vi]];
    base..base + 3 + m.log_k
}

// ---------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------

/// Verify the batched staged protocol (mirrors `prove_legs_batched`'s
/// transcript and ledger sequence exactly).
#[allow(clippy::too_many_lines)]
pub fn verify_legs_batched(
    instances: &[MemoryInstance],
    proof: &BatchedLegs,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), MemoryError> {
    if instances.len() != 9 {
        return Err(MemoryError::Shape);
    }
    let classes = cube_classes(instances);
    let rw = rw_instances(instances);
    let fetch_idx = instances.len() - 1;

    // ================= Stage A =================
    for (ci, cls) in classes.iter().enumerate() {
        let mut r_bool: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
        let mut r_prime: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
        for &i in cls {
            let m = &instances[i];
            absorb(transcript, i, "B", m.log_k, m.log_ts)?;
            r_bool[i] = transcript
                .challenge_fields(b"mem-bool-r", m.log_rows() + m.log_ts)
                .map_err(MemoryError::Transcript)?;
            absorb(transcript, i, "R", m.log_k, m.log_ts)?;
            r_prime[i] = transcript
                .challenge_fields(b"mem-raf-r", m.log_ts)
                .map_err(MemoryError::Transcript)?;
        }
        let mut claimed: Vec<Goldilocks> = Vec::with_capacity(cls.len() * 2);
        let mut num_vars = 0usize;
        for &i in cls {
            let m = &instances[i];
            num_vars = m.log_rows() + m.log_ts;
            claimed.push(Goldilocks::ZERO);
            let addr_claim = ledger.tensor_claim(addr_factor(i), &r_prime[i])?;
            claimed.push(addr_claim);
        }
        let (rhos, r_br, final_claim) =
            replay_stage(&proof.br[ci], num_vars, 3, &claimed, transcript)?;
        let mut expect = Goldilocks::ZERO;
        for (leg, &i) in cls.iter().enumerate() {
            let m = &instances[i];
            let log_rows = m.log_rows();
            let d_at = ledger.tensor_claim(Factor::DigitBits { inst: i }, &r_br)?;
            let e_b = DenseMle::eq_eval(&r_bool[i], &r_br)
                .map_err(MemoryError::Mle)?
                .mul(&d_at.square().sub(&d_at));
            let eq_v =
                DenseMle::eq_eval(&r_prime[i], &r_br[log_rows..]).map_err(MemoryError::Mle)?;
            let w_v = m.raf_weights().evaluate(&r_br[..log_rows]).map_err(MemoryError::Mle)?;
            let e_r = eq_v.mul(&w_v).mul(&d_at);
            expect = expect
                .add(&rhos[2 * leg].mul(&e_b))
                .add(&rhos[2 * leg + 1].mul(&e_r));
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage A"));
        }
    }

    // ================= Stage B =================
    let mut r_c: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut r_w: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut r_t: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    let mut cwt_term: Vec<Vec<Goldilocks>> = vec![Vec::new(); instances.len()];
    for (ci, cls) in classes.iter().enumerate() {
        for &i in cls {
            let m = &instances[i];
            absorb(transcript, i, "C", m.log_k, m.log_ts)?;
            r_c[i] = transcript
                .challenge_fields(b"mem-read-r", m.log_ts)
                .map_err(MemoryError::Transcript)?;
            if !m.read_only() {
                absorb(transcript, i, "W", m.log_k, m.log_ts)?;
                r_w[i] = transcript
                    .challenge_fields(b"mem-write-r", m.log_k + m.log_ts)
                    .map_err(MemoryError::Transcript)?;
                absorb(transcript, i, "T", m.log_k, m.log_ts)?;
                r_t[i] = transcript
                    .challenge_fields(b"mem-tel-r", m.log_k)
                    .map_err(MemoryError::Transcript)?;
            }
        }
        let mut claimed: Vec<Goldilocks> = Vec::new();
        let mut num_vars = 0usize;
        for &i in cls {
            let m = &instances[i];
            num_vars = m.log_k + m.log_ts;
            let rv_claim = ledger.tensor_claim(rv_factor(i), &r_c[i])?;
            claimed.push(rv_claim);
            if !m.read_only() {
                claimed.push(Goldilocks::ZERO);
                claimed.push(tel_claim(m, &r_t[i])?);
            }
        }
        let (rhos, r_cwt, final_claim) =
            replay_stage(&proof.cwt[ci], num_vars, 4, &claimed, transcript)?;
        let mut expect = Goldilocks::ZERO;
        let mut leg = 0usize;
        for &i in cls {
            let m = &instances[i];
            let is_rw = !m.read_only();
            let (rho_k, rho_j) = r_cwt.split_at(m.log_k);
            let vi = rw.iter().position(|&x| x == i);
            {
                let eq_v = DenseMle::eq_eval(&r_c[i], rho_j).map_err(MemoryError::Mle)?;
                let val_at = if is_rw {
                    proof.val_read_claims[vi.unwrap_or(0)]
                } else {
                    let table = m.table.as_ref().ok_or(MemoryError::Shape)?;
                    DenseMle::new(table.clone())
                        .map_err(MemoryError::Mle)?
                        .evaluate(rho_k)
                        .map_err(MemoryError::Mle)?
                };
                let ra_at = if is_rw {
                    proof.ra_claims[vi.unwrap_or(0)]
                } else {
                    proof.ra_claims[rw.len()]
                };
                expect = expect.add(&rhos[leg].mul(&eq_v.mul(&ra_at).mul(&val_at)));
            }
            leg += 1;
            if is_rw {
                let vi = vi.unwrap_or(0);
                {
                    let eq_v = DenseMle::eq_eval(&r_w[i], &r_cwt).map_err(MemoryError::Mle)?;
                    let wv_at = ledger.tensor_claim(wv_factor(i), rho_j)?;
                    let inner = proof.inc_w_claims[vi]
                        .sub(&proof.wa_claims[vi].mul(&wv_at.sub(&proof.val_write_claims[vi])));
                    expect = expect.add(&rhos[leg].mul(&eq_v.mul(&inner)));
                }
                leg += 1;
                {
                    let eq_v = DenseMle::eq_eval(&r_t[i], rho_k).map_err(MemoryError::Mle)?;
                    expect = expect.add(&rhos[leg].mul(&eq_v.mul(&proof.inc_tel_claims[vi])));
                }
                leg += 1;
            }
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage B"));
        }
        for &i in cls {
            cwt_term[i] = r_cwt.clone();
        }
    }

    // ================= The fetch Ma =================
    {
        let m = &instances[fetch_idx];
        absorb(transcript, fetch_idx, "Ma", m.log_k, m.log_ts)?;
        let point = cwt_term[fetch_idx].clone();
        let (rho_k, rho_j) = point.split_at(m.log_k);
        let deg = 2 + m.log_k;
        let verdict = proof
            .fetch_ma
            .verify(m.log_ts, deg, proof.ra_claims[rw.len()], transcript, None)
            .map_err(MemoryError::Sumcheck)?;
        let e = matrix_terminal_expect(
            fetch_idx, m, ledger, rho_k, rho_j, &verdict.point, MatrixKind::Ra,
        )?;
        if e != verdict.final_claim {
            return Err(MemoryError::FinalCheck("batched fetch Ma"));
        }
    }

    // ================= Stage 2 =================
    let r_read;
    {
        let mut claimed: Vec<Goldilocks> = Vec::with_capacity(rw.len() * 2);
        let mut num_vars = 0usize;
        let mut max_deg = 2usize;
        for &i in &rw {
            let m = &instances[i];
            num_vars = m.log_ts;
            max_deg = max_deg.max(2 + m.log_k);
        }
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let init_mle = DenseMle::new(m.init.clone()).map_err(MemoryError::Mle)?;
            let init_at = init_mle.evaluate(&point[..m.log_k]).map_err(MemoryError::Mle)?;
            claimed.push(proof.ra_claims[vi]);
            claimed.push(proof.val_read_claims[vi].sub(&init_at));
        }
        let (rhos, r_read_, final_claim) =
            replay_stage(&proof.read, num_vars, max_deg, &claimed, transcript)?;
        r_read = r_read_;
        let mut expect = Goldilocks::ZERO;
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let (rho_k, rho_j) = point.split_at(m.log_k);
            let e_ma =
                matrix_terminal_expect(i, m, ledger, rho_k, rho_j, &r_read, MatrixKind::Ra)?;
            let lt_at = DenseMle::lt_extension(&r_read, rho_j).map_err(MemoryError::Mle)?;
            let e_v = proof.u_read_claims[vi].mul(&lt_at);
            expect = expect
                .add(&rhos[2 * vi].mul(&e_ma))
                .add(&rhos[2 * vi + 1].mul(&e_v));
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage 2"));
        }
    }

    // ================= Stage 3 =================
    let r_mu0;
    {
        let claimed = proof.u_read_claims.clone();
        let mut num_vars = 0usize;
        let mut max_deg = 3usize;
        for &i in &rw {
            let m = &instances[i];
            num_vars = m.log_ts;
            max_deg = max_deg.max(3 + m.log_k);
        }
        let (rhos, r_mu0_, final_claim) =
            replay_stage(&proof.mu0, num_vars, max_deg, &claimed, transcript)?;
        r_mu0 = r_mu0_;
        let mut expect = Goldilocks::ZERO;
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let r_a = &point[..m.log_k];
            let e = matrix_terminal_expect(i, m, ledger, r_a, &r_read, &r_mu0, MatrixKind::Inc)?;
            expect = expect.add(&rhos[vi].mul(&e));
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage 3"));
        }
    }

    // ================= Stage 4 =================
    let r_write;
    {
        let mut claimed: Vec<Goldilocks> = Vec::with_capacity(rw.len() * 3);
        let mut num_vars = 0usize;
        let mut max_deg = 3usize;
        for &i in &rw {
            let m = &instances[i];
            num_vars = m.log_ts;
            max_deg = max_deg.max(3 + m.log_k);
        }
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let init_mle = DenseMle::new(m.init.clone()).map_err(MemoryError::Mle)?;
            let init_at = init_mle.evaluate(&point[..m.log_k]).map_err(MemoryError::Mle)?;
            claimed.push(proof.wa_claims[vi]);
            claimed.push(proof.inc_w_claims[vi]);
            claimed.push(proof.val_write_claims[vi].sub(&init_at));
        }
        let (rhos, r_write_, final_claim) =
            replay_stage(&proof.write, num_vars, max_deg, &claimed, transcript)?;
        r_write = r_write_;
        let mut expect = Goldilocks::ZERO;
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let (rho_k, rho_j) = point.split_at(m.log_k);
            let e_mb =
                matrix_terminal_expect(i, m, ledger, rho_k, rho_j, &r_write, MatrixKind::Wa)?;
            let e_mc =
                matrix_terminal_expect(i, m, ledger, rho_k, rho_j, &r_write, MatrixKind::Inc)?;
            let lt_at = DenseMle::lt_extension(&r_write, rho_j).map_err(MemoryError::Mle)?;
            let e_v = proof.u_write_claims[vi].mul(&lt_at);
            expect = expect
                .add(&rhos[3 * vi].mul(&e_mb))
                .add(&rhos[3 * vi + 1].mul(&e_mc))
                .add(&rhos[3 * vi + 2].mul(&e_v));
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage 4"));
        }
    }

    // ================= Stage 5 =================
    let r_mu1;
    {
        let claimed = proof.u_write_claims.clone();
        let mut num_vars = 0usize;
        let mut max_deg = 3usize;
        for &i in &rw {
            let m = &instances[i];
            num_vars = m.log_ts;
            max_deg = max_deg.max(3 + m.log_k);
        }
        let (rhos, r_mu1_, final_claim) =
            replay_stage(&proof.mu1, num_vars, max_deg, &claimed, transcript)?;
        r_mu1 = r_mu1_;
        let mut expect = Goldilocks::ZERO;
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let r_a = &point[..m.log_k];
            let e =
                matrix_terminal_expect(i, m, ledger, r_a, &r_write, &r_mu1, MatrixKind::Inc)?;
            expect = expect.add(&rhos[vi].mul(&e));
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage 5"));
        }
    }

    // ================= Stage 6 =================
    {
        let claimed = proof.inc_tel_claims.clone();
        let mut num_vars = 0usize;
        let mut max_deg = 3usize;
        for &i in &rw {
            let m = &instances[i];
            num_vars = m.log_ts;
            max_deg = max_deg.max(3 + m.log_k);
        }
        let (rhos, r_md, final_claim) =
            replay_stage(&proof.md, num_vars, max_deg, &claimed, transcript)?;
        let mut expect = Goldilocks::ZERO;
        for (vi, &i) in rw.iter().enumerate() {
            let m = &instances[i];
            let point = &cwt_term[i];
            let (rho_k, rho_j) = point.split_at(m.log_k);
            let e = matrix_terminal_expect(i, m, ledger, rho_k, rho_j, &r_md, MatrixKind::Inc)?;
            expect = expect.add(&rhos[vi].mul(&e));
        }
        if final_claim != expect {
            return Err(MemoryError::FinalCheck("batched stage 6"));
        }
    }
    let _ = (r_mu1,);
    Ok(())
}
