//! §5.2's zero-knowledge accumulation scheme for special-sound NARKs
//! ("zk-Protogalaxy") — ePrint 2026/289's second contribution.
//!
//! # Construction (the paper's §5.2, with two documented consistency
//! resolutions)
//!
//! **Accumulator**: `acc.x = (x, [Cᵢ]ᵢ∈[µ], [rᵢ]ᵢ∈[µ−1], E, β, v_g)`,
//! `acc.w = ([mᵢ], G(X), [blinds], r_E)` with `E = Com(e; r_E)` and the
//! kernel claim `v_g = MLE(G|_cube)(β)`.
//!
//! **Prover `ACC.P`** on `m` predicate pairs and `m` old accumulators, over
//! `L = ⌈log₂(2m+1)⌉` interpolation variables with party layout
//! `[0 = dummy | 1..m = predicates | m+1..2m = accumulators | zero-pads]`:
//! 1. challenge-consistency of the inputs;
//! 2. sample the **masking vector** — a random dummy pair `(qx₀, qw₀)`
//!    (the paper's ZK contribution #1) — and commit its error
//!    `C^e₀ = Com(V_sps(x₀, m₀, r₀))`;
//! 3. interpolate `x(X), mᵢ(X), rᵢ(X)` (eq bases, Corollary 1's
//!    no-cross-term trick) and form `F(X) := V_sps(x(X), m(X), r(X))` and
//!    the error interpolation `e(X)`; `F̃ := F − e` vanishes on the cube;
//! 4. `α, γ ← ρ_ACC` and run the **masked batched sum-check**
//!    (`zk_sumcheck`) — the paper's ZK contribution #2 — whose update
//!    statements re-randomize the old masks' kernel claims to the fresh
//!    point β ([KS24], stopping accumulator growth);
//! 5. extract the fresh error `ẽ = (v − γ·G'(β))·eq(β,α)⁻¹` (Eq. (5)),
//!    commit `E := eq⁰(β)·C^e₀ + Σⱼ eq^{j+m−1}(β)·Eⱼ + Com_pub(ẽ)`, and
//!    fold the tuple `([rᵢ], [Cᵢ], v_g)` and the witness `([mᵢ], G)` by
//!    the eq weights at β.
//!
//! **Resolution 1 (the fresh error's commitment)**: the paper's verifier
//! listing checks `E = Σⱼ eq·Eⱼ` with a `⊥` at the dummy's slot, which
//! would force the dummy's error to zero and contradict the "random dummy"
//! sampling. The consistent reading — and what this implementation does —
//! is that the *public* fresh error `ẽ` is committed unblinded
//! (`Com_pub(ẽ) = Σ ẽ_c·G_c`, binding, verifier-computable) and the
//! *dummy's* error commitment `C^e₀` is transmitted in `pf` (the "one more
//! proof-instance pair" overhead the paper itself cites). Completeness,
//! the decider, and the extractor's binding argument all close under this
//! reading.
//!
//! **Resolution 2 (kernel claims)**: with individual-degree-`D` masks the
//! update identity `Σ_b eq(β,b)·G(b) = G(β)` holds only for the
//! multilinearization, so `v_g` is pinned as the **kernel value**
//! `MLE(G|_cube)(β)` throughout (prover, verifier, decider).
//!
//! **Verifier `ACC.V`**: challenge consistency (all parties incl. the
//! dummy), the sum-check replay, the tuple fold `([rᵢ],[Cᵢ],v_g) =
//! Σ_p eq_p(β)·v^p`, and the E-check above.
//!
//! **Decider `ACC.D`**: `Cᵢ = Com(mᵢ)`, `e = V_sps(x, m, r)` with
//! `E = Com(e; r_E)`, and `kernel(G, β) = v_g`.

use crate::pedersen::{PedersenCommitment, PedersenKey};
use crate::sps::{derive_challenges, SpsInstance, SpsRelation, SpsWitness};
use crate::util::{challenge_fp, eq_basis_eval, eq_eval};
use crate::zk_sumcheck::{prove_masked_batched, verify_masked_batched, MaskPoly, ZkSumcheckProof};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccumError {
    Shape(&'static str),
    Sps(crate::sps::SpsError),
    Pedersen(crate::pedersen::PedersenError),
    Transcript(lattice_core::transcript::TranscriptError),
    ZkSc(crate::zk_sumcheck::ZkScError),
    ChallengeInconsistency { party: usize },
    TupleMismatch,
    ErrorCommitmentMismatch,
    DeciderRejected(&'static str),
}

impl core::fmt::Display for AccumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AccumError::Shape(s) => write!(f, "accum shape: {s}"),
            AccumError::Sps(e) => write!(f, "sps: {e}"),
            AccumError::Pedersen(e) => write!(f, "pedersen: {e}"),
            AccumError::Transcript(e) => write!(f, "transcript: {e}"),
            AccumError::ZkSc(e) => write!(f, "zk sumcheck: {e}"),
            AccumError::ChallengeInconsistency { party } => {
                write!(f, "challenge inconsistency at party {party}")
            }
            AccumError::TupleMismatch => write!(f, "tuple fold mismatch"),
            AccumError::ErrorCommitmentMismatch => write!(f, "error commitment mismatch"),
            AccumError::DeciderRejected(s) => write!(f, "decider rejected: {s}"),
        }
    }
}

impl From<crate::sps::SpsError> for AccumError {
    fn from(e: crate::sps::SpsError) -> Self {
        AccumError::Sps(e)
    }
}

impl From<crate::pedersen::PedersenError> for AccumError {
    fn from(e: crate::pedersen::PedersenError) -> Self {
        AccumError::Pedersen(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for AccumError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        AccumError::Transcript(e)
    }
}

impl From<crate::zk_sumcheck::ZkScError> for AccumError {
    fn from(e: crate::zk_sumcheck::ZkScError) -> Self {
        AccumError::ZkSc(e)
    }
}

/// The number of interpolation variables for `p` predicate pairs + `a`
/// accumulators + the dummy: `L = ⌈log₂(1 + p + a)⌉` (the paper's `2m+1`
/// is the equal-count case; the PCD construction accumulates 1 predicate
/// pair against `m` incoming accumulators).
pub fn num_vars_for(parties: usize) -> usize {
    let mut l = 0;
    while (1usize << l) < parties {
        l += 1;
    }
    l
}

/// `acc.x`: the public accumulator instance.
#[derive(Clone, Debug)]
pub struct AccumulatorInstance {
    /// The accumulated NARK instance vector `x`.
    pub x: Vec<Fp256>,
    /// Accumulated message commitments `[Cᵢ]ᵢ∈[µ]`.
    pub commitments: Vec<PedersenCommitment>,
    /// Accumulated challenges `[rᵢ]ᵢ∈[µ−1]`.
    pub challenges: Vec<Fp256>,
    /// The error commitment `E = Com(e; r_E)`.
    pub error_commitment: PedersenCommitment,
    /// The claim point β (L variables).
    pub beta: Vec<Fp256>,
    /// The kernel claims `v_g = MLE(G|_cube)(β)` (n coordinates).
    pub v_g: Vec<Fp256>,
}

/// `acc.w`: the accumulator witness.
#[derive(Clone, Debug)]
pub struct AccumulatorWitness {
    /// Accumulated messages `[mᵢ]ᵢ∈[µ]`.
    pub messages: Vec<Vec<Fp256>>,
    /// The accumulated mask `G(X)` (the KS24 no-growth representation).
    pub mask: MaskPoly,
    /// Commitment blinds for `[mᵢ]`.
    pub msg_blinds: Vec<Fp256>,
    /// The blind of `E`.
    pub error_blind: Fp256,
}

/// A full accumulator (instance + witness).
#[derive(Clone, Debug)]
pub struct Accumulator {
    pub instance: AccumulatorInstance,
    pub witness: AccumulatorWitness,
}

/// The accumulation proof `pf = (qx₀, tr, v'_g, C^e₀)`.
#[derive(Clone, Debug)]
pub struct AccumProof {
    /// The dummy predicate instance (its messages are witness-side).
    pub dummy: SpsInstance,
    /// The masked batched sum-check transcript.
    pub sumcheck: ZkSumcheckProof,
    /// The dummy's error commitment `C^e₀` (Resolution 1).
    pub dummy_error_commitment: PedersenCommitment,
}

/// Accumulation keys: `apk = (rel, key)`, `avk = (rel,)`, `adk = key`.
pub struct AccumKeys<'a> {
    pub relation: &'a dyn SpsRelation,
    pub key: &'a PedersenKey,
}

/// Base accumulator at a fixed variable count `L` (the chain's arity).
pub fn create_base_accumulator_at<R: SpsRelation>(
    rel: &R,
    key: &PedersenKey,
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<Accumulator, AccumError> {
    let n = rel.num_outputs();
    let mu = rel.num_rounds();
    // Zero error (constant-free maps vanish at zero inputs).
    let zero_msg_len: Vec<usize> = (0..mu).map(|i| rel.msg_len(i)).collect();
    let mut zero_msgs = Vec::with_capacity(mu);
    for t in &zero_msg_len {
        zero_msgs.push(vec![Fp256::ZERO; *t]);
    }
    // e₀ for the zero witness: the map at zero inputs. A nonzero constant
    // component (e.g. the R1CS z₀ − 1 coordinate's −1) is *tracked* as the
    // base's relaxed error — exactly the Nova-u drift semantics — not
    // rejected; the zero-pads of every accumulation carry the same value.
    let zero_challenges = vec![Fp256::ZERO; mu.saturating_sub(1)];
    let e0 = rel.eval_map(
        &vec![Fp256::ZERO; rel.inst_len()],
        &zero_msgs,
        &zero_challenges,
    )?;
    // Commit the zero messages (identity commitments with fresh blinds for
    // hiding) and the base error.
    let mut commitments = Vec::with_capacity(mu);
    let mut msg_blinds = Vec::with_capacity(mu);
    for m in &zero_msgs {
        let blind = challenge_fp(transcript, b"base-msg-blind")?;
        let c = key.commit(m, &blind)?;
        commitments.push(c);
        msg_blinds.push(blind);
    }
    let error_blind = challenge_fp(transcript, b"base-error-blind")?;
    let error_commitment = key.commit(&e0, &error_blind)?;
    let mask = MaskPoly::sample(n, num_vars, rel.degree() + 1, transcript)?;
    // β ←$ F^L and the kernel claim.
    let beta = crate::util::challenge_fp_vec(transcript, b"base-beta", num_vars)?;
    let v_g = mask.kernel_eval(&beta)?;
    Ok(Accumulator {
        instance: AccumulatorInstance {
            x: vec![Fp256::ZERO; rel.inst_len()],
            commitments,
            challenges: vec![Fp256::ZERO; mu.saturating_sub(1)],
            error_commitment,
            beta,
            v_g,
        },
        witness: AccumulatorWitness {
            messages: zero_msgs,
            mask,
            msg_blinds,
            error_blind,
        },
    })
}

/// The accumulation prover of §5.2.
#[allow(clippy::too_many_arguments)]
pub fn accumulate<R: SpsRelation>(
    rel: &R,
    key: &PedersenKey,
    predicate_instances: &[SpsInstance],
    predicate_witnesses: &[SpsWitness],
    old_accumulators: &[Accumulator],
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(Accumulator, AccumProof), AccumError> {
    let m = predicate_instances.len();
    if predicate_witnesses.len() != m {
        return Err(AccumError::Shape("predicate witness count"));
    }
    let a = old_accumulators.len();
    let mu = rel.num_rounds();
    let n = rel.num_outputs();
    let d = rel.degree();
    // The chain-fixed variable count (≥ the minimal one); extra positions
    // are dummy-copy pads so every accumulation in a chain shares one L —
    // required because the masks' update statements live in a common
    // variable space.
    let l = num_vars;
    if (1usize << l) < 1 + m + a {
        return Err(AccumError::Shape("num_vars too small for the party count"));
    }
    let parties = 1usize << l;

    // 1. Challenge consistency of the predicate inputs.
    for (j, inst) in predicate_instances.iter().enumerate() {
        if derive_challenges(inst)? != inst.challenges {
            return Err(AccumError::ChallengeInconsistency { party: j + 1 });
        }
    }

    // 2. (placeholder — the dummy is sampled after α, γ below so the
    // verifier can replay the transcript deterministically.)

    // 5. α, γ ← ρ_ACC (absorb the public inputs first).
    for inst in predicate_instances {
        for c in &inst.commitments {
            transcript.append_message(b"acc-px-com", &c.to_bytes())?;
        }
        crate::util::absorb_fp_slice(transcript, b"acc-px-x", &inst.x)?;
    }
    for accj in old_accumulators {
        for c in &accj.instance.commitments {
            transcript.append_message(b"acc-acc-com", &c.to_bytes())?;
        }
        transcript.append_message(b"acc-acc-e", &accj.instance.error_commitment.to_bytes())?;
        crate::util::absorb_fp_slice(transcript, b"acc-acc-beta", &accj.instance.beta)?;
        crate::util::absorb_fp_slice(transcript, b"acc-acc-vg", &accj.instance.v_g)?;
    }
    let alpha = crate::util::challenge_fp_vec(transcript, b"acc-alpha", l)?;
    let gamma = challenge_fp(transcript, b"acc-gamma")?;

    // The masking vector: a random dummy pair, sampled AFTER α, γ so the
    // verifier replays the identical challenge stream (draw-and-discard).
    let dummy_x = crate::util::challenge_fp_vec(transcript, b"dummy-x", rel.inst_len())?;
    let mut dummy_msgs = Vec::with_capacity(mu);
    let mut dummy_coms = Vec::with_capacity(mu);
    let mut dummy_blinds = Vec::with_capacity(mu);
    for i in 0..mu {
        let msg = crate::util::challenge_fp_vec(transcript, b"dummy-msg", rel.msg_len(i))?;
        let blind = challenge_fp(transcript, b"dummy-blind")?;
        let c = key.commit(&msg, &blind)?;
        dummy_msgs.push(msg);
        dummy_coms.push(c);
        dummy_blinds.push(blind);
    }
    let mut dummy_inst = SpsInstance {
        x: dummy_x.clone(),
        commitments: dummy_coms,
        challenges: Vec::new(),
    };
    dummy_inst.challenges = derive_challenges(&dummy_inst)?;
    let dummy_error = rel.eval_map(&dummy_inst.x, &dummy_msgs, &dummy_inst.challenges)?;
    let dummy_error_blind = challenge_fp(transcript, b"dummy-error-blind")?;
    let dummy_error_com = key.commit(&dummy_error, &dummy_error_blind)?;

    // 3. Assemble the party bundle (after the dummy exists).
    let mut xs: Vec<Vec<Fp256>> = Vec::with_capacity(parties);
    let mut msgs: Vec<Vec<Vec<Fp256>>> = Vec::with_capacity(parties);
    let mut challenges: Vec<Vec<Fp256>> = Vec::with_capacity(parties);
    let mut coms: Vec<Vec<PedersenCommitment>> = Vec::with_capacity(parties);
    let mut errors: Vec<Vec<Fp256>> = Vec::with_capacity(parties);
    let mut masks: Vec<MaskPoly> = Vec::with_capacity(parties);
    let mut msg_blinds: Vec<Vec<Fp256>> = Vec::with_capacity(parties);
    let mut error_blinds: Vec<Fp256> = Vec::with_capacity(parties);
    let mut error_coms: Vec<PedersenCommitment> = Vec::with_capacity(parties);
    let mut old_claims: Vec<Option<(Vec<Fp256>, Vec<Fp256>)>> = Vec::with_capacity(parties);

    // Party 0: the dummy.
    xs.push(dummy_inst.x.clone());
    msgs.push(dummy_msgs.clone());
    challenges.push(dummy_inst.challenges.clone());
    coms.push(dummy_inst.commitments.clone());
    errors.push(dummy_error.clone());
    masks.push(MaskPoly::zero(n, l, d + 1));
    msg_blinds.push(dummy_blinds.clone());
    error_blinds.push(dummy_error_blind);
    error_coms.push(dummy_error_com);
    old_claims.push(None);

    // Parties 1..m: predicate instances.
    for (inst, wit) in predicate_instances.iter().zip(predicate_witnesses.iter()) {
        xs.push(inst.x.clone());
        msgs.push(wit.messages.clone());
        challenges.push(inst.challenges.clone());
        coms.push(inst.commitments.clone());
        // The predicate's map value — zero for satisfying pairs (checked at
        // accumulation time only through the sum-check, but recorded for
        // the error interpolation).
        let e = rel.eval_map(&inst.x, &wit.messages, &inst.challenges)?;
        errors.push(e);
        masks.push(MaskPoly::zero(n, l, d + 1));
        msg_blinds.push(wit.blinds.clone());
        error_blinds.push(Fp256::ZERO);
        error_coms.push(PedersenCommitment::identity());
        old_claims.push(None);
    }

    // Parties m+1..m+a: the old accumulators.
    for accj in old_accumulators {
        xs.push(accj.instance.x.clone());
        msgs.push(accj.witness.messages.clone());
        challenges.push(accj.instance.challenges.clone());
        coms.push(accj.instance.commitments.clone());
        // The accumulator's error: its map value (the relaxed error).
        let e = rel.eval_map(
            &accj.instance.x,
            &accj.witness.messages,
            &accj.instance.challenges,
        )?;
        errors.push(e);
        masks.push(accj.witness.mask.clone());
        msg_blinds.push(accj.witness.msg_blinds.clone());
        error_blinds.push(accj.witness.error_blind);
        error_coms.push(accj.instance.error_commitment);
        old_claims.push(Some((
            accj.instance.beta.clone(),
            accj.instance.v_g.clone(),
        )));
    }

    // Pad positions carry COPIES of the dummy: their map value equals the
    // dummy's error e₀, so F̃ vanishes on the whole cube and the E-fold
    // closes with the public weight w₀ := eq⁰(β) + Σ_pad eq_pad(β) on the
    // single transmitted C^e₀ (the paper's party count 2m+1 < 2^L leaves
    // these positions; glossed there, made explicit here).
    for _p in 1 + m + a..parties {
        xs.push(dummy_inst.x.clone());
        msgs.push(dummy_msgs.clone());
        challenges.push(dummy_inst.challenges.clone());
        coms.push(dummy_inst.commitments.clone());
        errors.push(dummy_error.clone());
        masks.push(MaskPoly::zero(n, l, d + 1));
        msg_blinds.push(dummy_blinds.clone());
        error_blinds.push(dummy_error_blind);
        error_coms.push(dummy_error_com);
        old_claims.push(None);
    }

    // 4. The F̃ evaluation closure (eq-interpolated inputs → the map − the
    // error interpolation).
    let f_tilde = |pt: &[Fp256]| -> Vec<Fp256> {
        // eq basis values at pt for all parties.
        let eqs: Vec<Fp256> = (0..parties).map(|p| eq_basis_eval(p, pt)).collect();
        // Interpolate the inputs.
        let mut x_pt = vec![Fp256::ZERO; rel.inst_len()];
        for (p, e) in eqs.iter().enumerate() {
            if e.is_zero() {
                continue;
            }
            for (i, xv) in xs[p].iter().enumerate() {
                x_pt[i] = x_pt[i].add(&e.mul(xv));
            }
        }
        let mut msg_pt: Vec<Vec<Fp256>> = Vec::with_capacity(mu);
        for i in 0..mu {
            let mut mi = vec![Fp256::ZERO; rel.msg_len(i)];
            for (p, e) in eqs.iter().enumerate() {
                if e.is_zero() {
                    continue;
                }
                for (k, mv) in msgs[p][i].iter().enumerate() {
                    mi[k] = mi[k].add(&e.mul(mv));
                }
            }
            msg_pt.push(mi);
        }
        let mut ch_pt = vec![Fp256::ZERO; mu.saturating_sub(1)];
        for (p, e) in eqs.iter().enumerate() {
            if e.is_zero() {
                continue;
            }
            for (k, cv) in challenges[p].iter().enumerate() {
                ch_pt[k] = ch_pt[k].add(&e.mul(cv));
            }
        }
        let f = rel.eval_map(&x_pt, &msg_pt, &ch_pt).unwrap_or_default();
        let mut e_pt = vec![Fp256::ZERO; n];
        for (p, e) in eqs.iter().enumerate() {
            if e.is_zero() {
                continue;
            }
            for (c, ev) in errors[p].iter().enumerate() {
                e_pt[c] = e_pt[c].add(&e.mul(ev));
            }
        }
        (0..n).map(|c| f[c].sub(&e_pt[c])).collect()
    };

    // 6. The masked batched sum-check with the old masks' kernel claims.
    let mut old_masks: Vec<(MaskPoly, Vec<Fp256>, Vec<Fp256>)> = Vec::with_capacity(m);
    for accj in old_accumulators {
        old_masks.push((
            accj.witness.mask.clone(),
            accj.instance.beta.clone(),
            accj.instance.v_g.clone(),
        ));
    }
    let st = crate::zk_sumcheck::MaskedBatchStatement {
        num_vars: l,
        map_degree: d,
        n,
        alpha: alpha.clone(),
        gamma,
        f_tilde: &f_tilde,
        old_masks,
    };
    let (sc_proof, sc_out) = prove_masked_batched(&st, transcript)?;

    // 7. The fresh error and the new accumulator.
    let beta = &sc_out.beta;
    let eq_beta: Vec<Fp256> = (0..parties).map(|p| eq_basis_eval(p, beta)).collect();
    // ẽ = (first_final − γ·G'(β))·eq(β,α)⁻¹  (Eq. (5)).
    let eq_ba = eq_eval(beta, &alpha);
    let eq_ba_inv = eq_ba
        .inverse()
        .ok_or(AccumError::Shape("degenerate eq(β, α)"))?;
    let gamma_vg: Vec<Fp256> = (0..n)
        .map(|c| gamma.mul(&sc_proof.fresh_mask_eval[c]))
        .collect();
    let e_tilde: Vec<Fp256> = (0..n)
        .map(|c| sc_out.first_final[c].sub(&gamma_vg[c]).mul(&eq_ba_inv))
        .collect();

    // The new error: e_new = F(β) = ẽ + e(β) — compute directly via the
    // closure pieces.
    let f_beta = {
        // e(β) via the error interpolation.
        let mut e_b = vec![Fp256::ZERO; n];
        for (p, w) in eq_beta.iter().enumerate() {
            for (c, ev) in errors[p].iter().enumerate() {
                e_b[c] = e_b[c].add(&w.mul(ev));
            }
        }
        e_b
    };
    let _e_new: Vec<Fp256> = (0..n).map(|c| e_tilde[c].add(&f_beta[c])).collect();

    // Folded values.
    let fold_vec = |per_party: &[Vec<Fp256>]| -> Vec<Fp256> {
        let len = per_party.first().map(|v| v.len()).unwrap_or(0);
        let mut out = vec![Fp256::ZERO; len];
        for (p, w) in eq_beta.iter().enumerate() {
            for (i, v) in per_party[p].iter().enumerate() {
                out[i] = out[i].add(&w.mul(v));
            }
        }
        out
    };
    let x_new = fold_vec(&xs);
    let ch_new: Vec<Fp256> = if mu > 1 {
        fold_vec(&challenges)
    } else {
        Vec::new()
    };
    let mut msg_new: Vec<Vec<Fp256>> = Vec::with_capacity(mu);
    let mut msg_blind_new: Vec<Fp256> = Vec::with_capacity(mu);
    for i in 0..mu {
        let per_party: Vec<Vec<Fp256>> = msgs.iter().map(|mm| mm[i].clone()).collect();
        msg_new.push(fold_vec(&per_party));
        let per_blind: Vec<Vec<Fp256>> = msg_blinds.iter().map(|b| vec![b[i]]).collect();
        msg_blind_new.push(fold_vec(&per_blind)[0]);
    }
    // Commitment folds (homomorphic).
    let mut com_new: Vec<PedersenCommitment> = Vec::with_capacity(mu);
    for i in 0..mu {
        let items: Vec<(Fp256, PedersenCommitment)> =
            (0..parties).map(|p| (eq_beta[p], coms[p][i])).collect();
        com_new.push(PedersenCommitment::linear_combine(&items));
    }
    // The error commitment: E = w₀·C^e₀ + Σ_acc eq·Eⱼ + Com_pub(ẽ), where
    // w₀ aggregates the dummy and its pad copies.
    let mut w0 = eq_beta[0];
    for p in 1 + m + a..parties {
        w0 = w0.add(&eq_beta[p]);
    }
    let mut e_items: Vec<(Fp256, PedersenCommitment)> = vec![(w0, dummy_error_com)];
    for (j, accj) in old_accumulators.iter().enumerate() {
        e_items.push((eq_beta[m + 1 + j], accj.instance.error_commitment));
    }
    let pub_e = key.commit(&e_tilde, &Fp256::ZERO)?;
    let mut error_commitment = PedersenCommitment::linear_combine(&e_items);
    error_commitment = error_commitment.add(&pub_e);
    // (fold_vec over all parties already includes the pads' dummy blinds —
    // exactly w₀·b₀ + Σ_acc eq·bⱼ.)
    let error_blind_new = fold_vec(&error_blinds.iter().map(|b| vec![*b]).collect::<Vec<_>>())[0];

    // v_g fold: eq⁰·κ' + Σ_acc eq·κ_j (kernel values).
    let mut v_g_new: Vec<Fp256> = (0..n)
        .map(|c| eq_beta[0].mul(&sc_proof.fresh_mask_kernel[c]))
        .collect();
    for (j, kappa) in sc_out.old_kernel_evals.iter().enumerate() {
        for c in 0..n {
            v_g_new[c] = v_g_new[c].add(&eq_beta[m + 1 + j].mul(&kappa[c]));
        }
    }

    // The new mask: G = eq⁰·G' + Σ_acc eq·G_j.
    let mut mask_new = MaskPoly::zero(n, l, d + 1);
    mask_new.add_scaled(&eq_beta[0], &sc_out.fresh_mask)?;
    for (j, accj) in old_accumulators.iter().enumerate() {
        mask_new.add_scaled(&eq_beta[m + 1 + j], &accj.witness.mask)?;
    }

    let acc = Accumulator {
        instance: AccumulatorInstance {
            x: x_new,
            commitments: com_new,
            challenges: ch_new,
            error_commitment,
            beta: beta.clone(),
            v_g: v_g_new,
        },
        witness: AccumulatorWitness {
            messages: msg_new,
            mask: mask_new,
            msg_blinds: msg_blind_new,
            error_blind: error_blind_new,
        },
    };
    let pf = AccumProof {
        dummy: dummy_inst,
        sumcheck: sc_proof,
        dummy_error_commitment: dummy_error_com,
    };
    Ok((acc, pf))
}

/// The accumulation verifier of §5.2.
#[allow(clippy::too_many_arguments)]
pub fn verify_accumulation<R: SpsRelation>(
    rel: &R,
    key: &PedersenKey,
    predicate_instances: &[SpsInstance],
    old_accumulators: &[AccumulatorInstance],
    new_instance: &AccumulatorInstance,
    pf: &AccumProof,
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(), AccumError> {
    let m = predicate_instances.len();
    let a = old_accumulators.len();
    let mu = rel.num_rounds();
    let n = rel.num_outputs();
    let d = rel.degree();
    // The chain-fixed variable count (≥ the minimal one); extra positions
    // are dummy-copy pads so every accumulation in a chain shares one L —
    // required because the masks' update statements live in a common
    // variable space.
    let l = num_vars;
    if (1usize << l) < 1 + m + a {
        return Err(AccumError::Shape("num_vars too small for the party count"));
    }
    let parties = 1usize << l;

    // 1. Challenge consistency of the predicate inputs and the dummy.
    for (j, inst) in predicate_instances.iter().enumerate() {
        if derive_challenges(inst)? != inst.challenges {
            return Err(AccumError::ChallengeInconsistency { party: j + 1 });
        }
    }
    if derive_challenges(&pf.dummy)? != pf.dummy.challenges {
        return Err(AccumError::ChallengeInconsistency { party: 0 });
    }

    // 2. Replay the challenge derivation transcript.
    for inst in predicate_instances {
        for c in &inst.commitments {
            transcript.append_message(b"acc-px-com", &c.to_bytes())?;
        }
        crate::util::absorb_fp_slice(transcript, b"acc-px-x", &inst.x)?;
    }
    for accj in old_accumulators {
        for c in &accj.commitments {
            transcript.append_message(b"acc-acc-com", &c.to_bytes())?;
        }
        transcript.append_message(b"acc-acc-e", &accj.error_commitment.to_bytes())?;
        crate::util::absorb_fp_slice(transcript, b"acc-acc-beta", &accj.beta)?;
        crate::util::absorb_fp_slice(transcript, b"acc-acc-vg", &accj.v_g)?;
    }
    let alpha = crate::util::challenge_fp_vec(transcript, b"acc-alpha", l)?;
    let gamma = challenge_fp(transcript, b"acc-gamma")?;

    // Replay the prover's dummy sampling: draw and discard the identical
    // word sequence (x, per-round msg + blind, error blind).
    let _ = crate::util::challenge_fp_vec(transcript, b"dummy-x", rel.inst_len())?;
    for i in 0..mu {
        let _ = crate::util::challenge_fp_vec(transcript, b"dummy-msg", rel.msg_len(i))?;
        let _ = challenge_fp(transcript, b"dummy-blind")?;
    }
    let _ = challenge_fp(transcript, b"dummy-error-blind")?;

    // 3. Verify the sum-check.
    let old_claims: Vec<(Vec<Fp256>, Vec<Fp256>)> = old_accumulators
        .iter()
        .map(|a| (a.beta.clone(), a.v_g.clone()))
        .collect();
    let sc_out = verify_masked_batched(
        l,
        d,
        n,
        &alpha,
        &gamma,
        &old_claims,
        &pf.sumcheck,
        transcript,
    )?;
    let beta = &sc_out.beta;
    let eq_beta: Vec<Fp256> = (0..parties).map(|p| eq_basis_eval(p, beta)).collect();

    // 4. The tuple fold check.
    // [r_i]: dummy + predicates + accumulators.
    {
        let mut ok = true;
        if mu > 1 {
            let expect_ch: Vec<Fp256> = {
                let mut acc_v = vec![Fp256::ZERO; mu - 1];
                for p in 0..parties {
                    let vals: Vec<Fp256> = match p {
                        0 => pf.dummy.challenges.clone(),
                        j if (1..=m).contains(&j) => predicate_instances[j - 1].challenges.clone(),
                        j if (m + 1..=m + a).contains(&j) => {
                            old_accumulators[j - m - 1].challenges.clone()
                        }
                        _ => pf.dummy.challenges.clone(),
                    };
                    for (k, v) in vals.iter().enumerate() {
                        acc_v[k] = acc_v[k].add(&eq_beta[p].mul(v));
                    }
                }
                acc_v
            };
            ok &= expect_ch == new_instance.challenges;
        }
        // x fold (pads are dummy copies).
        let mut expect_x = vec![Fp256::ZERO; rel.inst_len()];
        for p in 0..parties {
            let vals: Vec<Fp256> = match p {
                0 => pf.dummy.x.clone(),
                j if (1..=m).contains(&j) => predicate_instances[j - 1].x.clone(),
                j if (m + 1..=m + a).contains(&j) => old_accumulators[j - m - 1].x.clone(),
                _ => pf.dummy.x.clone(),
            };
            for (i, v) in vals.iter().enumerate() {
                expect_x[i] = expect_x[i].add(&eq_beta[p].mul(v));
            }
        }
        ok &= expect_x == new_instance.x;
        // v_g fold: eq⁰·κ' + Σ_acc eq·κ_j.
        let mut expect_vg: Vec<Fp256> = (0..n)
            .map(|c| eq_beta[0].mul(&pf.sumcheck.fresh_mask_kernel[c]))
            .collect();
        for (j, kappa) in sc_out.old_kernel_evals.iter().enumerate() {
            for c in 0..n {
                expect_vg[c] = expect_vg[c].add(&eq_beta[m + 1 + j].mul(&kappa[c]));
            }
        }
        ok &= expect_vg == new_instance.v_g;
        // Commitment folds.
        for i in 0..mu {
            let mut expect_c = PedersenCommitment::identity();
            for p in 0..parties {
                let c = match p {
                    0 => pf.dummy.commitments[i],
                    j if (1..=m).contains(&j) => predicate_instances[j - 1].commitments[i],
                    j if (m + 1..=m + a).contains(&j) => old_accumulators[j - m - 1].commitments[i],
                    _ => pf.dummy.commitments[i],
                };
                expect_c = expect_c.add(&c.scale(&eq_beta[p]));
            }
            ok &= expect_c == new_instance.commitments[i];
        }
        if !ok {
            return Err(AccumError::TupleMismatch);
        }
    }

    // 5. The E-check: E = eq⁰·C^e₀ + Σ_acc eq·Eⱼ + Com_pub(ẽ).
    {
        let eq_ba = eq_eval(beta, &alpha);
        let eq_ba_inv = eq_ba
            .inverse()
            .ok_or(AccumError::Shape("degenerate eq(β, α)"))?;
        let e_tilde: Vec<Fp256> = (0..n)
            .map(|c| {
                sc_out.first_final[c]
                    .sub(&gamma.mul(&pf.sumcheck.fresh_mask_eval[c]))
                    .mul(&eq_ba_inv)
            })
            .collect();
        let pub_e = key.commit(&e_tilde, &Fp256::ZERO)?;
        let mut w0 = eq_beta[0];
        for p in 1 + m + a..parties {
            w0 = w0.add(&eq_beta[p]);
        }
        let mut expect_e = pf.dummy_error_commitment.scale(&w0);
        for (j, accj) in old_accumulators.iter().enumerate() {
            expect_e = expect_e.add(&accj.error_commitment.scale(&eq_beta[m + 1 + j]));
        }
        expect_e = expect_e.add(&pub_e);
        if expect_e != new_instance.error_commitment {
            return Err(AccumError::ErrorCommitmentMismatch);
        }
    }

    Ok(())
}

/// The decider of §5.2.
pub fn decide<R: SpsRelation>(
    rel: &R,
    key: &PedersenKey,
    acc: &Accumulator,
) -> Result<(), AccumError> {
    let mu = rel.num_rounds();
    if acc.witness.messages.len() != mu
        || acc.instance.commitments.len() != mu
        || acc.witness.msg_blinds.len() != mu
    {
        return Err(AccumError::DeciderRejected("witness shape"));
    }
    // 1. Cᵢ = Com(mᵢ).
    for i in 0..mu {
        if !key.verify_opening(
            &acc.instance.commitments[i],
            &acc.witness.messages[i],
            &acc.witness.msg_blinds[i],
        )? {
            return Err(AccumError::DeciderRejected("message commitment"));
        }
    }
    // 2+3. e = V_sps(x, m, r) and E = Com(e; r_E).
    let e = rel.eval_map(
        &acc.instance.x,
        &acc.witness.messages,
        &acc.instance.challenges,
    )?;
    if !key.verify_opening(&acc.instance.error_commitment, &e, &acc.witness.error_blind)? {
        return Err(AccumError::DeciderRejected("error commitment"));
    }
    // 4. kernel(G, β) = v_g.
    if acc.witness.mask.kernel_eval(&acc.instance.beta)? != acc.instance.v_g {
        return Err(AccumError::DeciderRejected("mask kernel claim"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::{R1csSps, SpsWitness};

    fn key() -> PedersenKey {
        PedersenKey::derive(&[11u8; 32], 64).ok().unwrap()
    }

    fn make_satisfying_pair(
        _rel: &R1csSps,
        key: &PedersenKey,
        x: &[Fp256],
        w: &[Fp256],
    ) -> (SpsInstance, SpsWitness) {
        let mut t = Transcript::new_default(b"pair");
        let mut z = vec![Fp256::from_canonical_u64(1)];
        z.extend(x.iter().cloned());
        z.extend(w.iter().cloned());
        let msgs = vec![z];
        let (mut inst, mut wit) = crate::sps::commit_messages(&msgs, key, &mut t)
            .ok()
            .unwrap();
        inst.x = x.to_vec();
        inst.challenges = derive_challenges(&inst).ok().unwrap();
        // Fill the witness messages correctly (commit_messages stored them).
        wit.messages = msgs;
        (inst, wit)
    }

    #[test]
    fn accumulate_r1cs_two_instances_roundtrip() {
        // m = 1 predicate pair + 1 old accumulator, L = ⌈log2(3)⌉ = 2.
        let mut rel = R1csSps::random(2, 6, 3, b"acc-seed");
        let (x1, w1) = rel.sample_satisfying(b"w1").ok().unwrap();
        let key = key();
        let (inst, wit) = make_satisfying_pair(&rel, &key, &x1, &w1);
        // The predicate witness blinds: commit_messages samples them via the
        // transcript; recover by re-committing? The SpsWitness from
        // commit_messages carries them.
        let _ = &wit.blinds;

        let mut t = Transcript::new_default(b"acc-test");
        // Base accumulator for the chain's L = 2.
        let base = create_base_accumulator_at(&rel, &key, 2, &mut t)
            .ok()
            .unwrap();

        let mut tprov = Transcript::new_default(b"acc-test2");
        let l2 = num_vars_for(3);
        let (acc, pf) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            std::slice::from_ref(&wit),
            std::slice::from_ref(&base),
            l2,
            &mut tprov,
        )
        .ok()
        .unwrap();

        // Verifier (fresh transcript, same label).
        let mut tver = Transcript::new_default(b"acc-test2");
        let res = verify_accumulation(
            &rel,
            &key,
            &[inst],
            std::slice::from_ref(&base.instance),
            &acc.instance,
            &pf,
            l2,
            &mut tver,
        );
        assert!(res.is_ok(), "verify failed: {res:?}");

        // Decider accepts the relaxed accumulator.
        let dec = decide(&rel, &key, &acc);
        assert!(dec.is_ok(), "decider failed: {dec:?}");
    }

    #[test]
    fn probe_chain_step1() {
        let rel = R1csSps::many_solutions(1, 5, 3, b"probe-seed");
        let key = key();
        let mut t = Transcript::new_default(b"probe");
        let base = create_base_accumulator_at(&rel, &key, 2, &mut t)
            .ok()
            .unwrap();
        // Step 0.
        let (x0, w0) = rel.draw_solution(b"pw0");
        let (inst0, wit0) = make_satisfying_pair(&rel, &key, &x0, &w0);
        let mut tp0 = Transcript::new_default(b"probe-step");
        let (acc0, _pf0) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst0),
            &[wit0],
            std::slice::from_ref(&base),
            2,
            &mut tp0,
        )
        .ok()
        .unwrap();
        assert!(decide(&rel, &key, &acc0).is_ok());
        // Step 1.
        let (x1, w1) = rel.draw_solution(b"pw1");
        let (inst1, wit1) = make_satisfying_pair(&rel, &key, &x1, &w1);
        let mut tp1 = Transcript::new_default(b"probe-step");
        let (acc1, _pf1) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst1),
            &[wit1],
            std::slice::from_ref(&acc0),
            2,
            &mut tp1,
        )
        .ok()
        .unwrap();
        // Compare the decider's map value with a direct evaluation of the
        // interpolated map at beta.
        let e_dec = rel
            .eval_map(
                &acc1.instance.x,
                &acc1.witness.messages,
                &acc1.instance.challenges,
            )
            .ok()
            .unwrap();
        println!(
            "decider e  = {:?}",
            e_dec.iter().map(|v| v.to_i128()).collect::<Vec<_>>()
        );
        // The expected committed error: rebuild via the fold of per-party
        // errors + e_tilde. e_tilde is not stored; instead check the opening
        // directly: E == Com(e_dec, blind)?
        let ok = key
            .verify_opening(
                &acc1.instance.error_commitment,
                &e_dec,
                &acc1.witness.error_blind,
            )
            .ok()
            .unwrap();
        println!("opening check: {}", ok);
        assert!(ok);
    }

    #[test]
    fn chain_of_accumulations() {
        // A 3-step chain: base → acc1 → acc2 → acc3, verifying each step.
        let rel = R1csSps::many_solutions(1, 5, 3, b"chain-seed");
        let key = key();
        let mut t = Transcript::new_default(b"chain");
        let mut current = create_base_accumulator_at(&rel, &key, 2, &mut t)
            .ok()
            .unwrap();
        for step in 0..3u8 {
            let seed: Vec<u8> = [b"chain-w".as_ref(), &[step]].concat();
            let (xs, ws) = rel.draw_solution(&seed);
            let (inst, wit) = make_satisfying_pair(&rel, &key, &xs, &ws);
            let mut tp = Transcript::new_default(b"chain-step");
            let (acc, pf) = accumulate(
                &rel,
                &key,
                std::slice::from_ref(&inst),
                &[wit],
                &[current.clone()],
                2,
                &mut tp,
            )
            .ok()
            .unwrap();
            let mut tv = Transcript::new_default(b"chain-step");
            let res = verify_accumulation(
                &rel,
                &key,
                &[inst],
                &[current.instance.clone()],
                &acc.instance,
                &pf,
                2,
                &mut tv,
            );
            assert!(res.is_ok(), "step {step} verify failed: {res:?}");
            let dres = decide(&rel, &key, &acc);
            assert!(dres.is_ok(), "step {step} decider failed: {dres:?}");
            current = acc;
        }
        assert!(decide(&rel, &key, &current).is_ok());
    }

    #[test]
    fn tampered_accumulation_rejected() {
        let mut rel = R1csSps::random(2, 5, 3, b"tamper-seed");
        let (xt, wt) = rel.sample_satisfying(b"tw").ok().unwrap();
        let key = key();
        let (inst, wit) = make_satisfying_pair(&rel, &key, &xt, &wt);
        let mut t = Transcript::new_default(b"tamper");
        let base = create_base_accumulator_at(&rel, &key, 2, &mut t)
            .ok()
            .unwrap();
        let mut tp = Transcript::new_default(b"tamper2");
        let (acc, pf) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            &[wit],
            std::slice::from_ref(&base),
            num_vars_for(3),
            &mut tp,
        )
        .ok()
        .unwrap();

        // (a) Tampered new instance x → tuple mismatch.
        let mut bad_inst = acc.instance.clone();
        bad_inst.x[0] = bad_inst.x[0].add(&Fp256::from_canonical_u64(1));
        let mut tv = Transcript::new_default(b"tamper2");
        assert!(verify_accumulation(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            std::slice::from_ref(&base.instance),
            &bad_inst,
            &pf,
            num_vars_for(3),
            &mut tv
        )
        .is_err());

        // (b) Tampered E → E-check mismatch.
        let mut bad_e = acc.instance.clone();
        bad_e.error_commitment = bad_e.error_commitment.add(
            &key.commit(&[Fp256::from_canonical_u64(7)], &Fp256::ZERO)
                .ok()
                .unwrap(),
        );
        let mut tv2 = Transcript::new_default(b"tamper2");
        assert!(verify_accumulation(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            std::slice::from_ref(&base.instance),
            &bad_e,
            &pf,
            num_vars_for(3),
            &mut tv2
        )
        .is_err());

        // (c) Tampered sumcheck (round coefficient) → sumcheck rejection.
        let mut bad_pf = pf.clone();
        if let Some(first) = bad_pf.sumcheck.rounds.first_mut() {
            if let Some(poly) = first.first_mut() {
                if !poly.is_empty() {
                    poly[0] = poly[0].add(&Fp256::from_canonical_u64(1));
                }
            }
        }
        let mut tv3 = Transcript::new_default(b"tamper2");
        assert!(verify_accumulation(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            std::slice::from_ref(&base.instance),
            &acc.instance,
            &bad_pf,
            num_vars_for(3),
            &mut tv3
        )
        .is_err());

        // (d) Tampered v_g → tuple mismatch.
        let mut bad_vg = acc.instance.clone();
        bad_vg.v_g[0] = bad_vg.v_g[0].add(&Fp256::from_canonical_u64(1));
        let mut tv4 = Transcript::new_default(b"tamper2");
        assert!(verify_accumulation(
            &rel,
            &key,
            &[inst],
            std::slice::from_ref(&base.instance),
            &bad_vg,
            &pf,
            num_vars_for(3),
            &mut tv4
        )
        .is_err());

        // (e) Decider tamper: a wrong witness message → reject.
        let mut bad_acc = acc.clone();
        if !bad_acc.witness.messages[0].is_empty() {
            bad_acc.witness.messages[0][0] =
                bad_acc.witness.messages[0][0].add(&Fp256::from_canonical_u64(1));
        }
        assert!(decide(&rel, &key, &bad_acc).is_err());
        // (f) Decider tamper: wrong mask claim.
        let mut bad_acc2 = acc.clone();
        bad_acc2.instance.v_g[0] = bad_acc2.instance.v_g[0].add(&Fp256::from_canonical_u64(1));
        assert!(decide(&rel, &key, &bad_acc2).is_err());
    }
}
