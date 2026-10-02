//! The multilinear PCS with accumulation (the paper's Definition 4.3 /
//! Theorem 4.4) over the Ajtai module: `gen` (= `Srs::from_seed`), `com`,
//! `reduce`, `accumulate`, `decide`.
//!
//! * **com** — `cm = Σ_b w_b G_b` over the digit-layered witness `w` of the
//!   value vector `f` (short, `‖w‖∞ ≤ 2^16 − 1`); the evaluation claim is
//!   `Σ_b T(b)·w_b = v` with the layered equality factor
//!   `T(X) = eq(X_D, u)·E(X_L)` — exactly the paper's `f̂(u) = v` after the
//!   digit-layer regrouping.
//! * **reduce** (§5) — `α ←`, `P' = αP`, module sumcheck over
//!   `A(X) = Ŵ(X)Ĝ(X) + T(X)Ŵ(X)P'` with target `cm + vP'`, terminal
//!   `(r, V)`, prover's single field element `a = Ŵ(r)`, and the deferred
//!   instance `φ = (r, C = (V − baP')/a) ∈ L_G`.
//! * **accumulate** (§6) — `γ ←`, `C = Σᵢ γⁱ Cᵢ`, `e(X) = Σᵢ γⁱ eq(X, rᵢ)`,
//!   module sumcheck over `A(X) = Ĝ(X)e(X)`, output `(r, V/e(r))`.
//! * **decide** (§7, lattice route) — the amortized decider: `C ≟ Ĝ(r)`
//!   by direct public evaluation (`O(N)` ring-scalar operations, once per
//!   batch). The paper's group-BaseFold decider is hash-based (Merkle
//!   queries over folded RS layers) and does not port to Ajtai bindings —
//!   see the crate docs and the deviation ledger for the obstruction and
//!   the quantitative comparison.
//!
//! All verifier work per reduce/accumulate round is one module
//! evaluation of a degree-2 univariate (3 module-scalar mults) plus field
//! arithmetic — `O(m)` total, `m = k + κ` — matching the paper's
//! `O(log n)` verifier cost. Communication is `3m` module points and one
//! field element per reduce (the paper's "3k G-elements and one
//! F-element").

use crate::module::{eq_eval_index, Fq, LayeredCube, ModulePoint, Srs};
use crate::sumcheck::{ModuleSumcheckVerifier, RoundMessage, SummandTables};
use lattice_core::transcript::Transcript;

/// Structural parameters of the lattice ml-PCS.
#[derive(Clone, Debug)]
pub struct AccordionPcsParams {
    pub num_data_vars: usize,
    pub num_layers: usize,
    pub rows: usize,
    pub ring_degree: usize,
}

impl AccordionPcsParams {
    pub fn cube(&self) -> LayeredCube {
        LayeredCube::new(self.num_data_vars, self.num_layers)
    }

    pub fn srs_from_seed(&self, seed: &[u8]) -> Srs {
        Srs::from_seed(self.rows, self.ring_degree, self.cube().size(), seed)
    }
}

/// Errors of the PCS protocols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcsError {
    Shape(&'static str),
    /// The terminal scalar `a = Ŵ(r)` was zero — the division
    /// `(V − baP')/a` is undefined. Completeness error ≤ `2m/q` for honest
    /// provers (a nonzero multilinear vanishes at ≤ `m·deg` points), the
    /// paper's implicit nonzero assumption.
    TerminalZero,
    /// `e(r) = 0` in accumulate — completeness error ≤ `t·m/q`.
    EqFactorZero,
    Sumcheck(crate::sumcheck::SumcheckError),
    Transcript(String),
}

impl From<crate::sumcheck::SumcheckError> for PcsError {
    fn from(e: crate::sumcheck::SumcheckError) -> Self {
        PcsError::Sumcheck(e)
    }
}

/// A reduce proof: `m` round messages plus the single terminal field
/// element `a` (the challenge point is stored for API convenience and is
/// re-derived by the verifier from the transcript).
#[derive(Clone, Debug)]
pub struct ReduceProof {
    pub msgs: Vec<RoundMessage>,
    pub terminal_a: Fq,
    pub point: Vec<Fq>,
}

impl ReduceProof {
    /// Serialized proof size in bytes (communication accounting).
    pub fn size_bytes(&self) -> usize {
        self.msgs
            .iter()
            .map(|m| m.to_bytes().len())
            .sum::<usize>()
            + 8
    }
}

/// Prover side of `reduce` (§5). `w` is the layered digit witness of the
/// committed value vector; `v = f̂(u)` must be the true evaluation (the
/// protocol derives it here when `None`).
#[allow(clippy::too_many_arguments)]
pub fn reduce(
    srs: &Srs,
    cube: &LayeredCube,
    cm: &ModulePoint,
    u: &[Fq],
    v: &Fq,
    w: &[Fq],
    transcript: &mut Transcript,
) -> Result<ReduceProof, PcsError> {
    if w.len() != cube.size() || cm.dim() != srs.module_dim() {
        return Err(PcsError::Shape("reduce witness/commitment shape"));
    }
    // Bind the statement, then draw α and set P' = α·P.
    absorb_statement(transcript, cm, u, v)?;
    let alpha = Fq::challenge(transcript, b"acc-reduce-alpha").map_err(PcsError::Transcript)?;
    let p_prime = srs.value_column().scale(&alpha);
    let t_table = cube.t_table(u);
    let mut tables = SummandTables::for_reduce(w, srs, &t_table, p_prime);
    // Sumcheck rounds.
    let mut msgs = Vec::with_capacity(cube.num_vars());
    let mut point = Vec::with_capacity(cube.num_vars());
    for _ in 0..cube.num_vars() {
        let msg = tables.round_message();
        transcript
            .append_message(b"acc-round", &msg.to_bytes())
            .map_err(|e| PcsError::Transcript(e.to_string()))?;
        msgs.push(msg);
        let r = Fq::challenge(transcript, b"acc-round-chal").map_err(PcsError::Transcript)?;
        tables.restrict(&r);
        point.push(r);
    }
    let (a, _) = tables.terminal_values();
    if a.is_zero() {
        return Err(PcsError::TerminalZero);
    }
    transcript
        .append_message(b"acc-terminal", &a.to_bytes())
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    Ok(ReduceProof {
        msgs,
        terminal_a: a,
        point,
    })
}

/// Verifier side of `reduce`: replays the transcript, checks every
/// recurrence, and outputs the accumulated instance `(r, C)` — the claim
/// `C = Ĝ(r)` to be settled later by `decide` (or folded by `accumulate`).
pub fn reduce_verify(
    srs: &Srs,
    cube: &LayeredCube,
    cm: &ModulePoint,
    u: &[Fq],
    v: &Fq,
    proof: &ReduceProof,
    transcript: &mut Transcript,
) -> Result<(Vec<Fq>, ModulePoint), PcsError> {
    if proof.msgs.len() != cube.num_vars() || cm.dim() != srs.module_dim() {
        return Err(PcsError::Shape("reduce proof shape"));
    }
    absorb_statement(transcript, cm, u, v)?;
    let alpha = Fq::challenge(transcript, b"acc-reduce-alpha").map_err(PcsError::Transcript)?;
    let p_prime = srs.value_column().scale(&alpha);
    // Target: C := cm + v·P'.
    let target = cm.axpy(v, &p_prime);
    let mut verifier = ModuleSumcheckVerifier::new(cube.num_vars(), target);
    let mut point = Vec::with_capacity(cube.num_vars());
    for msg in &proof.msgs {
        transcript
            .append_message(b"acc-round", &msg.to_bytes())
            .map_err(|e| PcsError::Transcript(e.to_string()))?;
        let r = Fq::challenge(transcript, b"acc-round-chal").map_err(PcsError::Transcript)?;
        verifier.round(msg, r)?;
        point.push(r);
    }
    transcript
        .append_message(b"acc-terminal", &proof.terminal_a.to_bytes())
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    if point != proof.point {
        return Err(PcsError::Shape("point mismatch vs transcript"));
    }
    // b = T(r) — the verifier computes the public factor itself.
    let b = cube.eval_t(&point, u);
    let a = proof.terminal_a;
    if a.is_zero() {
        return Err(PcsError::TerminalZero);
    }
    // C = (V − b·a·P') / a  — the deferred evaluation of Ĝ at r.
    let v_term = verifier.terminal_value();
    let ba = b.mul(&a);
    let numerator = v_term.sub(&p_prime.scale(&ba));
    let a_inv = a.inv().ok_or(PcsError::TerminalZero)?;
    let c = numerator.scale(&a_inv);
    Ok((point, c))
}

/// Absorb the reduce statement `(cm, u, v)` — both sides must absorb the
/// same bytes before the `α` challenge so the protocol is bound to the
/// claim (the paper's public-coin statement binding).
fn absorb_statement(
    transcript: &mut Transcript,
    cm: &ModulePoint,
    u: &[Fq],
    v: &Fq,
) -> Result<(), PcsError> {
    transcript
        .append_message(b"acc-stmt-cm", &cm.to_bytes())
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    let mut u_bytes = Vec::new();
    for x in u {
        u_bytes.extend_from_slice(&x.to_bytes());
    }
    transcript
        .append_message(b"acc-stmt-u", &u_bytes)
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    transcript
        .append_message(b"acc-stmt-v", &v.to_bytes())
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    Ok(())
}

/// An accumulate proof (§6): `m` round messages; the challenge point is
/// re-derived from the transcript.
#[derive(Clone, Debug)]
pub struct AccumulateProof {
    pub msgs: Vec<RoundMessage>,
    pub point: Vec<Fq>,
}

impl AccumulateProof {
    pub fn size_bytes(&self) -> usize {
        self.msgs.iter().map(|m| m.to_bytes().len()).sum::<usize>()
    }
}

/// One accumulated instance: `(r, C)` with the claim `Ĝ(r) = C`.
pub type Instance = (Vec<Fq>, ModulePoint);

/// Absorb an instance into a transcript.
fn absorb_instance(
    transcript: &mut Transcript,
    inst: &Instance,
) -> Result<(), PcsError> {
    transcript
        .append_message(b"acc-inst-r", &{
            let mut b = Vec::new();
            for x in &inst.0 {
                b.extend_from_slice(&x.to_bytes());
            }
            b
        })
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    transcript
        .append_message(b"acc-inst-c", &inst.1.to_bytes())
        .map_err(|e| PcsError::Transcript(e.to_string()))?;
    Ok(())
}

/// Prover side of `accumulate` (§6): folds `t ≥ 1` instances into one.
pub fn accumulate(
    srs: &Srs,
    cube: &LayeredCube,
    instances: &[Instance],
    transcript: &mut Transcript,
) -> Result<AccumulateProof, PcsError> {
    if instances.is_empty() || instances.len() > cube.size() {
        return Err(PcsError::Shape("accumulate instance count"));
    }
    for (r, c) in instances {
        if r.len() != cube.num_vars() || c.dim() != srs.module_dim() {
            return Err(PcsError::Shape("accumulate instance shape"));
        }
        absorb_instance(transcript, &(r.clone(), c.clone()))?;
    }
    let gammas = derive_gammas(instances.len(), transcript)?;
    // C = Σ γⁱ Cᵢ  and  e(X) = Σ γⁱ eq(X, rᵢ).
    let mut target = ModulePoint::zero(srs.module_dim());
    for ((_, c), g) in instances.iter().zip(gammas.iter()) {
        target = target.axpy(g, c);
    }
    let e_table = cube.eq_batch_table(
        &instances.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>(),
        &gammas,
    );
    let mut tables = SummandTables::for_accumulate(srs, &e_table);
    let mut msgs = Vec::with_capacity(cube.num_vars());
    let mut point = Vec::with_capacity(cube.num_vars());
    for _ in 0..cube.num_vars() {
        let msg = tables.round_message();
        transcript
            .append_message(b"acc-acc-round", &msg.to_bytes())
            .map_err(|e| PcsError::Transcript(e.to_string()))?;
        msgs.push(msg);
        let r = Fq::challenge(transcript, b"acc-acc-chal").map_err(PcsError::Transcript)?;
        tables.restrict(&r);
        point.push(r);
    }
    Ok(AccumulateProof { msgs, point })
}

/// Verifier side of `accumulate`: replays and outputs the folded instance
/// `(r, V/e(r))`.
pub fn accumulate_verify(
    srs: &Srs,
    cube: &LayeredCube,
    instances: &[Instance],
    proof: &AccumulateProof,
    transcript: &mut Transcript,
) -> Result<Instance, PcsError> {
    if instances.is_empty() || proof.msgs.len() != cube.num_vars() {
        return Err(PcsError::Shape("accumulate proof shape"));
    }
    for (r, c) in instances {
        absorb_instance(transcript, &(r.clone(), c.clone()))?;
    }
    let gammas = derive_gammas(instances.len(), transcript)?;
    let mut target = ModulePoint::zero(srs.module_dim());
    for ((_, c), g) in instances.iter().zip(gammas.iter()) {
        target = target.axpy(g, c);
    }
    let mut verifier = ModuleSumcheckVerifier::new(cube.num_vars(), target);
    let mut point = Vec::with_capacity(cube.num_vars());
    for msg in &proof.msgs {
        transcript
            .append_message(b"acc-acc-round", &msg.to_bytes())
            .map_err(|e| PcsError::Transcript(e.to_string()))?;
        let r = Fq::challenge(transcript, b"acc-acc-chal").map_err(PcsError::Transcript)?;
        verifier.round(msg, r)?;
        point.push(r);
    }
    if point != proof.point {
        return Err(PcsError::Shape("point mismatch vs transcript"));
    }
    // e(r) — public.
    let e_r = cube.eval_eq_batch(
        &point,
        &instances.iter().map(|(r, _)| r.clone()).collect::<Vec<_>>(),
        &gammas,
    );
    let e_inv = e_r.inv().ok_or(PcsError::EqFactorZero)?;
    let c = verifier.terminal_value().scale(&e_inv);
    Ok((point, c))
}

/// `decide` (§7, the lattice route): check `(r, C) ∈ L_G` by direct public
/// evaluation — `O(N)` ring-scalar operations, executed once per decided
/// batch (the Halo-style amortization; see the deviation ledger for why the
/// paper's group-BaseFold decider does not port).
pub fn decide(srs: &Srs, instance: &Instance) -> bool {
    let (r, c) = instance;
    srs.eval_generator_mle(r) == *c
}

/// The amortized decider: fold `t` instances into one via `accumulate`,
/// then decide once — the prover and the verifier run on identically-seeded
/// transcripts. Returns `(folded_instance, accepted)`.
pub fn decide_batched(
    srs: &Srs,
    cube: &LayeredCube,
    instances: &[Instance],
) -> Result<(Instance, bool), PcsError> {
    let mut pt = Transcript::new_default(b"accordion-batched");
    let proof = accumulate(srs, cube, instances, &mut pt)?;
    let mut vt = Transcript::new_default(b"accordion-batched");
    let folded = accumulate_verify(srs, cube, instances, &proof, &mut vt)?;
    Ok((folded.clone(), decide(srs, &folded)))
}

/// The `γ` powers: `γⁱ` for `i ∈ [t]` drawn once from the transcript.
fn derive_gammas(t: usize, transcript: &mut Transcript) -> Result<Vec<Fq>, PcsError> {
    let gamma = Fq::challenge(transcript, b"acc-gamma").map_err(PcsError::Transcript)?;
    Ok((0..t).map(|i| gamma.pow(i as u64)).collect())
}

/// Compute the true evaluation `f̂(u)` from the layered witness — the
/// honest prover's claimed value.
pub fn eval_claim(cube: &LayeredCube, w: &[Fq], u: &[Fq]) -> Fq {
    let t = cube.t_table(u);
    let mut acc = Fq::ZERO;
    for (wi, ti) in w.iter().zip(t.iter()) {
        acc = acc.add(&ti.mul(wi));
    }
    acc
}

/// Cross-check helper: `f̂(u)` computed from the value vector directly.
pub fn eval_direct(f: &[Fq], u: &[Fq]) -> Fq {
    let k = f.len().trailing_zeros() as usize;
    let mut acc = Fq::ZERO;
    for (i, &fi) in f.iter().enumerate() {
        acc = acc.add(&eq_eval_index(i, u, k).mul(&fi));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(k: usize) -> AccordionPcsParams {
        AccordionPcsParams {
            num_data_vars: k,
            num_layers: 4,
            rows: 1,
            ring_degree: 16,
        }
    }

    fn values(k: usize, seed: u64) -> Vec<Fq> {
        let n = 1usize << k;
        (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
                Fq::from_u64(x)
            })
            .collect()
    }

    #[test]
    fn reduce_accumulate_decide_roundtrip() {
        let p = params(4);
        let cube = p.cube();
        let srs = p.srs_from_seed(b"pcs-seed-1");
        let f = values(4, 42);
        let w = cube.digit_layers(&f);
        let cm = srs.commit_scalars(&w);
        let u: Vec<Fq> = (0..4).map(|i| Fq::from_u64(1000 + i as u64 * 137)).collect();
        let v = eval_claim(&cube, &w, &u);
        assert_eq!(v, eval_direct(&f, &u));

        let mut pt = Transcript::new_default(b"t-reduce");
        let proof = reduce(&srs, &cube, &cm, &u, &v, &w, &mut pt).expect("reduce");
        let mut vt = Transcript::new_default(b"t-reduce");
        let instance = reduce_verify(&srs, &cube, &cm, &u, &v, &proof, &mut vt).expect("verify");
        // The instance is in L_G.
        assert!(decide(&srs, &instance));
    }

    #[test]
    fn accumulate_two_instances_and_decide() {
        let p = params(3);
        let cube = p.cube();
        let srs = p.srs_from_seed(b"pcs-seed-2");
        let mut instances = Vec::new();
        for s in 0..2u64 {
            let f = values(3, 7 + s);
            let w = cube.digit_layers(&f);
            let cm = srs.commit_scalars(&w);
            let u: Vec<Fq> = (0..3).map(|i| Fq::from_u64(50 + i as u64 * (s + 1))).collect();
            let v = eval_claim(&cube, &w, &u);
            let mut pt = Transcript::new_default(b"t-acc2");
            let proof = reduce(&srs, &cube, &cm, &u, &v, &w, &mut pt).expect("reduce");
            let mut vt = Transcript::new_default(b"t-acc2");
            instances.push(reduce_verify(&srs, &cube, &cm, &u, &v, &proof, &mut vt).unwrap());
        }
        let (folded, ok) = decide_batched(&srs, &cube, &instances).expect("batched");
        assert!(ok, "folded instance must be in L_G");
        let _ = folded;
    }

    #[test]
    fn tampered_value_rejected_by_sumcheck() {
        let p = params(3);
        let cube = p.cube();
        let srs = p.srs_from_seed(b"pcs-seed-3");
        let f = values(3, 99);
        let w = cube.digit_layers(&f);
        let cm = srs.commit_scalars(&w);
        let u: Vec<Fq> = (0..3).map(|i| Fq::from_u64(17 + i as u64 * 3)).collect();
        let v_true = eval_claim(&cube, &w, &u);
        let v_bad = v_true.add(&Fq::ONE);
        // The prover cannot produce an accepting reduce for a wrong value:
        // round 1's recurrence against cm + v_bad·P' fails.
        let mut pt = Transcript::new_default(b"t-bad");
        let proof = reduce(&srs, &cube, &cm, &u, &v_bad, &w, &mut pt).expect("reduce runs");
        let mut vt = Transcript::new_default(b"t-bad");
        let err = reduce_verify(&srs, &cube, &cm, &u, &v_bad, &proof, &mut vt);
        assert!(matches!(err, Err(PcsError::Sumcheck(_))));
    }

    #[test]
    fn tampered_round_message_rejected() {
        let p = params(3);
        let cube = p.cube();
        let srs = p.srs_from_seed(b"pcs-seed-4");
        let f = values(3, 5);
        let w = cube.digit_layers(&f);
        let cm = srs.commit_scalars(&w);
        let u: Vec<Fq> = (0..3).map(|i| Fq::from_u64(23 + i)).collect();
        let v = eval_claim(&cube, &w, &u);
        let mut pt = Transcript::new_default(b"t-tamper");
        let mut proof = reduce(&srs, &cube, &cm, &u, &v, &w, &mut pt).expect("reduce");
        // Tamper the second round's middle coefficient.
        proof.msgs[1].0[1] = proof.msgs[1].0[1].axpy(&Fq::ONE, srs.value_column());
        let mut vt = Transcript::new_default(b"t-tamper");
        let err = reduce_verify(&srs, &cube, &cm, &u, &v, &proof, &mut vt);
        assert!(matches!(err, Err(PcsError::Sumcheck(_))));
    }

    #[test]
    fn tampered_terminal_rejected_by_decide() {
        // A consistent proof with a wrong terminal `a` yields an instance
        // OUT of L_G — decide catches it (the knowledge-soundness backstop).
        let p = params(3);
        let cube = p.cube();
        let srs = p.srs_from_seed(b"pcs-seed-5");
        let f = values(3, 8);
        let w = cube.digit_layers(&f);
        let cm = srs.commit_scalars(&w);
        let u: Vec<Fq> = (0..3).map(|i| Fq::from_u64(61 + i * 2)).collect();
        let v = eval_claim(&cube, &w, &u);
        let mut pt = Transcript::new_default(b"t-term");
        let mut proof = reduce(&srs, &cube, &cm, &u, &v, &w, &mut pt).expect("reduce");
        proof.terminal_a = proof.terminal_a.add(&Fq::ONE);
        let mut vt = Transcript::new_default(b"t-term");
        let instance = reduce_verify(&srs, &cube, &cm, &u, &v, &proof, &mut vt).expect("verify");
        assert!(!decide(&srs, &instance), "tampered terminal must fail decide");
    }

    #[test]
    fn accumulate_tampered_instance_rejected() {
        let p = params(3);
        let cube = p.cube();
        let srs = p.srs_from_seed(b"pcs-seed-6");
        let f = values(3, 31);
        let w = cube.digit_layers(&f);
        let cm = srs.commit_scalars(&w);
        let u: Vec<Fq> = (0..3).map(|i| Fq::from_u64(71 + i)).collect();
        let v = eval_claim(&cube, &w, &u);
        let mut pt = Transcript::new_default(b"t-acc-bad");
        let proof = reduce(&srs, &cube, &cm, &u, &v, &w, &mut pt).expect("reduce");
        let mut vt = Transcript::new_default(b"t-acc-bad");
        let (r, mut c) = reduce_verify(&srs, &cube, &cm, &u, &v, &proof, &mut vt).unwrap();
        // Corrupt the instance's C: accumulate over it must fail or decide
        // must reject.
        c.0[0] = c.0[0].add(&Fq::ONE);
        let instances: Vec<Instance> = vec![(r, c)];
        let result = decide_batched(&srs, &cube, &instances);
        // The prover-side accumulate over the corrupted instance produces
        // proofs against the corrupted target — decide then fails.
        if let Ok((_, ok)) = result {
            assert!(!ok, "corrupted instance must not decide");
        }
    }
}
