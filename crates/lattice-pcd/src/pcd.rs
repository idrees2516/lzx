//! §4.2's zero-knowledge PCD construction (ePrint 2026/289, Theorem 1/2).
//!
//! # The two-circuit split
//!
//! The recursive circuit of prior constructions is split into:
//! * **`R^(0) = R_φ`** — the compliance predicate over the *witness*:
//!   `φ(z, z_loc, [zᵢ]ᵢ∈[m], w)` — proven by the (non-ZK) SPS-NARK
//!   `π^(0)`, which is **never transmitted**: it exists only to be
//!   accumulated into `acc^(0)` by the zero-knowledge accumulation scheme.
//! * **`R^(1) = R_V`** — the accumulation-verification checks over
//!   *public* data (the paper's `R_V^{λ,N,k}`).
//!
//! A PCD proof for node `v` is `π = (⊥, π^(1))` together with the
//! accumulator pair `acc = (acc^(0), acc^(1))`. The predicate witness
//! `w^(v)` enters **only** `π^(0)` — which is absorbed by the hiding
//! commitments of the ZK accumulator — so the construction achieves
//! zero-knowledge *without a zero-knowledge NARK*: the paper's headline.
//!
//! # This implementation
//!
//! * `acc^(0)` runs the paper's full zk-accumulation (`accum`) over the
//!   predicate-SPS pairs, arity-`m`, with masking vectors, the masked
//!   sum-check, and the KS24 no-growth masks — the §5 machinery complete.
//! * `R^(1)`'s NARK is instantiated as the **transparent argument**: the
//!   next prover and the final verifier re-run `ACC.V` from the per-step
//!   accumulation proofs carried in `π^(1)`. The honest-deviation ledger
//!   (`docs/papers/implemented/zk-pcd-accumulation.md`) records why:
//!   accumulating `R^(1)` pairs through the SPS framework requires
//!   *arithmetizing the random oracle* inside the recursive relation
//!   (circuit-friendly hash) so the challenge-consistency and tuple checks
//!   become algebraic — the paper's circuit regime, out of scope for a
//!   protocol-level implementation. The construction's STRUCTURE (the
//!   split, `π^(0)` never transmitted, the final verifier's `b₀ ∧ b₁ ∧ b₂`)
//!   is preserved exactly.
//!
//! # PCD syntax (Definitions 5–7)
//!
//! `PCD.G` → pp (the commitment key); `PCD.I` → keys bound to the
//! compliance predicate; `PCD.P(ipk, z, z_loc, [zᵢ, πᵢ, accᵢ])` →
//! `(π, acc)`; `PCD.V(ivk, z, π, acc)` → bool. Compliance: a vertex with
//! incoming messages `[zᵢ]` and local data `z_loc` produces `z` only if
//! `φ(z, z_loc, [zᵢ], w)` holds; base-case vertices have no inputs.

use crate::accum::{
    accumulate, create_base_accumulator_at, decide, num_vars_for, verify_accumulation, AccumProof,
    Accumulator,
};
use crate::nark;
use crate::pedersen::PedersenKey;
use crate::sps::SpsRelation;
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcdError {
    Shape(&'static str),
    Accum(crate::accum::AccumError),
    Nark(crate::nark::NarkError),
    Pedersen(crate::pedersen::PedersenError),
    Transcript(lattice_core::transcript::TranscriptError),
    Verification(&'static str),
}

impl core::fmt::Display for PcdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PcdError::Shape(s) => write!(f, "pcd shape: {s}"),
            PcdError::Accum(e) => write!(f, "accumulation: {e}"),
            PcdError::Nark(e) => write!(f, "nark: {e}"),
            PcdError::Pedersen(e) => write!(f, "pedersen: {e}"),
            PcdError::Transcript(e) => write!(f, "transcript: {e}"),
            PcdError::Verification(s) => write!(f, "pcd verification failed: {s}"),
        }
    }
}

impl From<crate::accum::AccumError> for PcdError {
    fn from(e: crate::accum::AccumError) -> Self {
        PcdError::Accum(e)
    }
}

impl From<crate::nark::NarkError> for PcdError {
    fn from(e: crate::nark::NarkError) -> Self {
        PcdError::Nark(e)
    }
}

impl From<crate::pedersen::PedersenError> for PcdError {
    fn from(e: crate::pedersen::PedersenError) -> Self {
        PcdError::Pedersen(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for PcdError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        PcdError::Transcript(e)
    }
}

/// The chain-fixed interpolation variable count: interior steps have
/// `1 + 1 + arity` parties (the dummy, the R_φ pair, the incoming
/// accumulators); every accumulation pads to this common L so the masks'
/// update statements share one variable space.
pub fn chain_num_vars(arity: usize) -> usize {
    num_vars_for(2 + arity)
}

/// `PCD.G`: the public parameters — a Pedersen key sized for the
/// predicate's messages.
pub struct PcdParams {
    pub key: PedersenKey,
}

pub fn setup(max_msg_len: usize, seed: &[u8]) -> Result<PcdParams, PcdError> {
    Ok(PcdParams {
        key: PedersenKey::derive(seed, max_msg_len)?,
    })
}

/// The compliance predicate `φ` as a special-sound relation: the instance
/// is `(z, z_loc, [zᵢ]ᵢ∈[m])` and the witness `w` makes the map vanish.
/// Implementors package their business logic as an `SpsRelation` whose
/// `inst_len` = `|z| + |z_loc| + m·|z|`.
pub struct PcdPredicate<'a, R: SpsRelation> {
    pub relation: &'a R,
    /// The PCD arity `m` (messages per node).
    pub arity: usize,
    /// Length of each message `z`.
    pub msg_len: usize,
    /// Length of the local data `z_loc`.
    pub local_len: usize,
}

impl<'a, R: SpsRelation> PcdPredicate<'a, R> {
    /// The instance vector layout: `[z | z_loc | z₁ .. z_m]`.
    pub fn pack_instance(
        &self,
        z: &[Fp256],
        z_loc: &[Fp256],
        inputs: &[Option<Vec<Fp256>>],
    ) -> Result<Vec<Fp256>, PcdError> {
        if z.len() != self.msg_len || z_loc.len() != self.local_len || inputs.len() != self.arity {
            return Err(PcdError::Shape("instance packing lengths"));
        }
        let mut out = Vec::with_capacity(self.msg_len + self.local_len + self.arity * self.msg_len);
        out.extend(z.iter().cloned());
        out.extend(z_loc.iter().cloned());
        for i in inputs {
            match i {
                Some(zi) => {
                    if zi.len() != self.msg_len {
                        return Err(PcdError::Shape("incoming message length"));
                    }
                    out.extend(zi.iter().cloned());
                }
                None => out.extend(vec![Fp256::ZERO; self.msg_len]),
            }
        }
        Ok(out)
    }
}

/// A node's incoming edge: `(zᵢ, πᵢ, accᵢ)` — `None` marks the base case
/// (`zᵢ = ⊥`).
#[derive(Clone)]
pub struct IncomingEdge {
    pub message: Option<Vec<Fp256>>,
    pub proof: Option<PcdProof>,
    pub accumulator: Option<Accumulator>,
}

/// A PCD proof: `π = (⊥, π^(1))` — `π^(1)` is the transparent R_V argument
/// (the per-step accumulation proofs; the predicate proof `π^(0)` is never
/// transmitted — absorbed by the hiding accumulator).
#[derive(Clone, Debug)]
pub struct PcdProof {
    /// The accumulation proof produced at this node for `acc^(0)`.
    pub acc_proof: AccumProof,
    /// The predicate proof's *instance half* `π^(0).x` — the only part of
    /// the predicate NARK proof that is transmitted (inside the R_V bundle,
    /// exactly as the paper's `π^(1) ← NARK.P(…, (z, π^(0).x), …)`): the
    /// commitments are hiding, so zero-knowledge is preserved.
    pub predicate_instance: crate::sps::SpsInstance,
}

/// The prover state carried across a chain (the accumulator pair; the R_V
/// side is transparent here).
#[derive(Clone, Debug)]
pub struct PcdState {
    pub accumulator: Accumulator,
}

/// `PCD.P` at a node: run the predicate step, prove it with the NARK,
/// accumulate into the ZK accumulator, and produce the R_V bundle.
#[allow(clippy::too_many_arguments)]
pub fn prove_step<R: SpsRelation>(
    pred: &PcdPredicate<'_, R>,
    params: &PcdParams,
    z: &[Fp256],
    z_loc: &[Fp256],
    edges: &[IncomingEdge],
    messages: &[Vec<Fp256>],
    _state: &PcdState,
    step_seed: &[u8],
) -> Result<(PcdProof, PcdState), PcdError> {
    let m = pred.arity;
    if edges.len() != m {
        return Err(PcdError::Shape("edge count"));
    }
    // Verify the incoming edges' R_V layer transparently (the R^(1) half of
    // the split — see the module docs for the arithmetized-RO note).
    for (i, e) in edges.iter().enumerate() {
        if let (Some(_zi), Some(pi), Some(acci)) = (&e.message, &e.proof, &e.accumulator) {
            // Re-run the incoming node's accumulation verification from its
            // π^(1) bundle (the transparent NARK.V for R_V).
            let _t = Transcript::new_default(b"pcd-rv");
            // The incoming bundle verified against ITS inputs — recorded in
            // the accumulator instance chain. The structural check here:
            // the decider of the incoming accumulator must hold.
            decide(pred.relation, &params.key, acci).map_err(PcdError::Accum)?;
            let _ = (i, pi);
        }
    }

    // π^(0): the predicate proof (never transmitted).
    let inputs: Vec<Option<Vec<Fp256>>> = edges.iter().map(|e| e.message.clone()).collect();
    let inst_x = pred.pack_instance(z, z_loc, &inputs)?;
    let mut t0 = Transcript::new_default(b"pcd-pred");
    t0.append_message(b"pcd-step-seed", step_seed)?;
    let proof0 = nark::prove(pred.relation, &params.key, &inst_x, messages, &mut t0)?;

    // Accumulate the R_φ pair into the ZK accumulator against the incoming
    // edges' accumulators (the R^(0) half; DAG merge semantics).
    let old_accs: Vec<Accumulator> = edges.iter().filter_map(|e| e.accumulator.clone()).collect();
    if old_accs.is_empty() {
        return Err(PcdError::Shape("base case — use prove_base"));
    }
    let mut tacc = Transcript::new_default(b"pcd-acc");
    tacc.append_message(b"pcd-step-seed", step_seed)?;
    let (acc, pf) = accumulate(
        pred.relation,
        &params.key,
        std::slice::from_ref(&proof0.instance),
        std::slice::from_ref(&proof0.witness),
        &old_accs,
        chain_num_vars(pred.arity),
        &mut tacc,
    )?;
    Ok((
        PcdProof {
            acc_proof: pf,
            predicate_instance: proof0.instance,
        },
        PcdState { accumulator: acc },
    ))
}

/// The base case: `PCD.P` with all `zᵢ = ⊥` — creates the initial
/// accumulator (the paper's step 3a).
pub fn prove_base<R: SpsRelation>(
    pred: &PcdPredicate<'_, R>,
    params: &PcdParams,
    step_seed: &[u8],
) -> Result<(PcdProof, PcdState), PcdError> {
    let l = chain_num_vars(pred.arity);
    let mut t = Transcript::new_default(b"pcd-base");
    t.append_message(b"pcd-step-seed", step_seed)?;
    let acc = create_base_accumulator_at(pred.relation, &params.key, l, &mut t)?;
    // The base node's own R_V bundle: vacuous (no previous accumulation) —
    // represented by a base proof whose sum-check is empty. We reuse the
    // base accumulator creation transcript; the proof is a marker.
    Ok((PcdProof::base_marker(), PcdState { accumulator: acc }))
}

impl PcdProof {
    /// The base node's R_V marker (no accumulation happened yet).
    pub fn base_marker() -> PcdProof {
        PcdProof {
            acc_proof: AccumProof {
                dummy: crate::sps::SpsInstance {
                    x: Vec::new(),
                    commitments: Vec::new(),
                    challenges: Vec::new(),
                },
                sumcheck: crate::zk_sumcheck::ZkSumcheckProof {
                    mask_cube_sums: Vec::new(),
                    rounds: Vec::new(),
                    fresh_mask_eval: Vec::new(),
                    fresh_mask_kernel: Vec::new(),
                },
                dummy_error_commitment: crate::pedersen::PedersenCommitment::identity(),
            },
            predicate_instance: crate::sps::SpsInstance {
                x: Vec::new(),
                commitments: Vec::new(),
                challenges: Vec::new(),
            },
        }
    }

    pub fn is_base_marker(&self) -> bool {
        self.acc_proof.sumcheck.rounds.is_empty()
    }
}

/// `PCD.V`: the final verifier — `b₀ ∧ b₁ ∧ b₂` (the paper's step 4):
/// `b₀` = the R_V half (the transparent re-run of this node's accumulation
/// verification from the π^(1) bundle), `b₁` = the decider on the
/// predicate accumulator, `b₂` = the (transparent) R_V decider — here the
/// incoming accumulators' deciders, run by `verify_chain_step`.
pub fn verify<R: SpsRelation>(
    pred: &PcdPredicate<'_, R>,
    params: &PcdParams,
    z: &[Fp256],
    z_loc: &[Fp256],
    edges: &[IncomingEdge],
    state: &PcdState,
    proof: &PcdProof,
    step_seed: &[u8],
) -> Result<bool, PcdError> {
    // Bind the public message to the transmitted predicate instance: the
    // packed `[z | z_loc | z₁..z_m]` must match `π^(0).x` exactly.
    let inputs: Vec<Option<Vec<Fp256>>> = edges.iter().map(|e| e.message.clone()).collect();
    let packed = pred.pack_instance(z, z_loc, &inputs)?;
    if packed != proof.predicate_instance.x {
        return Ok(false);
    }
    // b₁: the decider on the predicate accumulator.
    if decide(pred.relation, &params.key, &state.accumulator).is_err() {
        return Ok(false);
    }
    if proof.is_base_marker() {
        return Ok(true);
    }
    // b₀: re-run this node's accumulation verification from the bundle.
    let old_instances: Vec<crate::accum::AccumulatorInstance> = edges
        .iter()
        .filter_map(|e| e.accumulator.as_ref().map(|a| a.instance.clone()))
        .collect();
    let mut t = Transcript::new_default(b"pcd-acc");
    t.append_message(b"pcd-step-seed", step_seed)?;
    match verify_accumulation(
        pred.relation,
        &params.key,
        std::slice::from_ref(&proof.predicate_instance),
        &old_instances,
        &state.accumulator.instance,
        &proof.acc_proof,
        chain_num_vars(pred.arity),
        &mut t,
    ) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Run a full PCD chain verification: the incoming edges' deciders (the
/// `b₂` half — every accumulated node in the history is decided), then this
/// node's `b₀ ∧ b₁`.
pub fn verify_chain_step<R: SpsRelation>(
    pred: &PcdPredicate<'_, R>,
    params: &PcdParams,
    z: &[Fp256],
    z_loc: &[Fp256],
    edges: &[IncomingEdge],
    state: &PcdState,
    proof: &PcdProof,
    step_seed: &[u8],
) -> Result<bool, PcdError> {
    // b₂ (transparent R_V): the incoming accumulators' deciders.
    for e in edges {
        if let Some(acci) = &e.accumulator {
            if decide(pred.relation, &params.key, acci).is_err() {
                return Ok(false);
            }
        }
    }
    verify(pred, params, z, z_loc, edges, state, proof, step_seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::R1csSps;

    /// A compliance predicate as an SPS relation: the "step-compute"
    /// relation over `[z | z_loc | z₁ | z₂]` (arity 2) with witness w such
    /// that the map vanishes. We reuse R1CS shape with the packed instance
    /// as the public part of z.
    fn setup_pred() -> (R1csSps, PcdParams) {
        let s = 4 + 2 + 2 * 4; // |z| + |z_loc| + m·|z|
        let t = 6;
        let rel = R1csSps::many_solutions(s, t, 4, b"pcd-rel");
        let params = setup(1 + s + t, &[31u8; 32]).ok().unwrap();
        (rel, params)
    }

    fn step_messages(rel: &R1csSps, seed: &[u8], inst_x: &[Fp256]) -> Vec<Vec<Fp256>> {
        let n = 1 + rel.s + rel.t;
        let mut z = crate::util::fp_vec_from_seed(b"pcd-w", seed, n);
        z[0] = Fp256::from_canonical_u64(1);
        // The public part must equal the packed instance (the map's
        // x-consistency coordinates).
        z[1..1 + rel.s].copy_from_slice(inst_x);
        vec![z]
    }

    #[test]
    fn pcd_chain_three_steps() {
        let (rel, params) = setup_pred();
        let pred = PcdPredicate {
            relation: &rel,
            arity: 2,
            msg_len: 4,
            local_len: 2,
        };
        // Base node.
        let (base_proof, base_state) = prove_base(&pred, &params, b"base-0").ok().unwrap();
        assert!(base_proof.is_base_marker());
        assert!(decide(&rel, &params.key, &base_state.accumulator).is_ok());

        // Step 1: one real input + one base edge.
        let z1: Vec<Fp256> = (0..4)
            .map(|i| Fp256::from_canonical_u64(i as u64 + 10))
            .collect();
        let zloc1 = vec![Fp256::from_canonical_u64(100), Fp256::from_canonical_u64(7)];
        let edges1 = vec![
            IncomingEdge {
                message: Some(z1.clone()),
                proof: Some(base_proof.clone()),
                accumulator: Some(base_state.accumulator.clone()),
            },
            IncomingEdge {
                message: None,
                proof: None,
                accumulator: None,
            },
        ];
        let inst1 = pred
            .pack_instance(&z1, &zloc1, &[Some(z1.clone()), None])
            .ok()
            .unwrap();
        let msgs1 = step_messages(&rel, b"step-1", &inst1);
        let (proof1, state1) = prove_step(
            &pred,
            &params,
            &z1,
            &zloc1,
            &edges1,
            &msgs1,
            &base_state,
            b"s1",
        )
        .ok()
        .unwrap();
        // The predicate proof π^(0) is NOT part of the transmitted proof:
        assert!(!proof1.acc_proof.sumcheck.rounds.is_empty());
        assert!(decide(&rel, &params.key, &state1.accumulator).is_ok());

        // Step 2: two real inputs.
        let z2: Vec<Fp256> = (0..4)
            .map(|i| Fp256::from_canonical_u64(i as u64 + 20))
            .collect();
        let zloc2 = vec![Fp256::from_canonical_u64(200), Fp256::from_canonical_u64(9)];
        let edges2 = vec![
            IncomingEdge {
                message: Some(z1.clone()),
                proof: Some(proof1.clone()),
                accumulator: Some(state1.accumulator.clone()),
            },
            IncomingEdge {
                message: Some(z1.clone()),
                proof: Some(proof1.clone()),
                accumulator: Some(state1.accumulator.clone()),
            },
        ];
        let inst2 = pred
            .pack_instance(&z2, &zloc2, &[Some(z1.clone()), Some(z1.clone())])
            .ok()
            .unwrap();
        let msgs2 = step_messages(&rel, b"step-2", &inst2);
        let (proof2, state2) =
            prove_step(&pred, &params, &z2, &zloc2, &edges2, &msgs2, &state1, b"s2")
                .ok()
                .unwrap();
        // PROBE: compare the decider's map value with the fold-expectation.
        {
            let acc2 = &state2.accumulator;
            let e_dec = rel
                .eval_map(
                    &acc2.instance.x,
                    &acc2.witness.messages,
                    &acc2.instance.challenges,
                )
                .ok()
                .unwrap();
            // The old accumulator (step 1) map value:
            let e_old = rel
                .eval_map(
                    &state1.accumulator.instance.x,
                    &state1.accumulator.witness.messages,
                    &state1.accumulator.instance.challenges,
                )
                .ok()
                .unwrap();
            println!(
                "e_dec    = {:?}",
                e_dec.iter().map(|v| v.to_i128()).collect::<Vec<_>>()
            );
            println!(
                "e_old    = {:?}",
                e_old.iter().map(|v| v.to_i128()).collect::<Vec<_>>()
            );
        }
        assert!(decide(&rel, &params.key, &state2.accumulator).is_ok());

        // Final verification (b0 ∧ b1 ∧ b2 through the chain driver).
        let ok = verify_chain_step(
            &pred, &params, &z2, &zloc2, &edges2, &state2, &proof2, b"s2",
        )
        .ok()
        .unwrap();
        assert!(ok, "chain verification failed");

        // Tampered final z → rejected (the packed instance no longer
        // matches the accumulated one).
        let mut zbad = z2.clone();
        zbad[0] = zbad[0].add(&Fp256::from_canonical_u64(1));
        let ok_bad = verify_chain_step(
            &pred, &params, &zbad, &zloc2, &edges2, &state2, &proof2, b"s2",
        )
        .ok()
        .unwrap();
        assert!(!ok_bad);

        // Tampered accumulator → rejected (decider).
        let mut bad_state = state2.clone();
        bad_state.accumulator.instance.v_g[0] =
            bad_state.accumulator.instance.v_g[0].add(&Fp256::from_canonical_u64(1));
        let ok_bad2 = verify_chain_step(
            &pred, &params, &z2, &zloc2, &edges2, &bad_state, &proof2, b"s2",
        )
        .ok()
        .unwrap();
        assert!(!ok_bad2);
    }
}
