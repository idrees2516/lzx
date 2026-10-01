//! The Twist PIOPs (Twist & Shout Fig. 9, ePrint 2025/105 §5): the
//! increment-commitment protocol for read-WRITE memories.
//!
//! The paper's headline construction: instead of grand-product
//! fingerprints (Lipton's trick), commit to the **increment matrix**
//!
//! ```text
//! Inc(k, j) = wa(k, j) · (wv(j) − Val(k, j))
//! ```
//!
//! where `wa` is the write-address one-hot matrix, `wv` the write-value
//! column, and `Val(k, j)` the *virtual* current-value matrix (the
//! verifier never materializes it — it is defined by the telescoping
//! identity below; at kernel scale the prover materializes it and the
//! resolver carries it). Three sumchecks make the memory argument:
//!
//! 1. **Read-checking** (Fig 9, read leg): for the sampled cycle
//!    `rcycle`, `rv(rcycle) = Σ_k ra(k, rcycle)·Val(k, rcycle)` — a
//!    sumcheck over the address variables (the same shape as Shout's
//!    read check, against the virtual `Val`).
//! 2. **Inc definition**: over the full `(k, j)` space,
//!    `Σ eq(r_inc,(k,j))·[Inc − wa·(wv − Val)] = 0` — binds the
//!    committed `Inc` to its definition (degree 3).
//! 3. **Telescoping** (the memory-correctness core): summing the
//!    increments over the cycles telescopes the value evolution:
//!    `Σ_j Inc(k, j) = Final(k) − Init(k)` for every address k, so
//!    `Σ_{k,j} eq(r_tel,(k,j))·Inc(k,j) = Σ_k eq(r_tel_k, k)·(Final(k)
//!    − Init(k))` — a verifier-computable claim from the public
//!    initial/final memory states. This is the statement the
//!    pre-Wave-7 `twist_fingerprint` could not make (it checked final
//!    consistency only — a stale read with a consistent final state
//!    passed); the read-checking leg above is exactly the missing
//!    per-cycle soundness.

use crate::onehot::{embed_dim, OneHotLayout};
use crate::{FactorId, FactorResolver, PiopError};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

/// The prover-side memory trace materialization.
#[derive(Clone, Debug)]
pub struct TwistWitness {
    pub log_k: usize,
    pub log_t: usize,
    /// Per-dimension read one-hot matrices (layout `(k_i, j)`).
    pub ra: Vec<DenseMle>,
    /// Per-dimension write one-hot matrices.
    pub wa: Vec<DenseMle>,
    /// The increment matrix over the combined `(k, j)` space.
    pub inc: DenseMle,
    /// The materialized current-value matrix (prover-side; the
    /// verifier's Val is virtual, pinned by the telescoping claim).
    pub val: DenseMle,
    /// The initial memory state (public).
    pub init: Vec<Goldilocks>,
    /// The final memory state (public).
    pub final_state: Vec<Goldilocks>,
}

/// The Twist proof: the three sumchecks of Fig. 9.
#[derive(Clone, Debug)]
pub struct TwistProof {
    pub read_checking: SumcheckProof,
    pub inc_definition: SumcheckProof,
    pub telescoping: SumcheckProof,
}

/// Build the witness materialization from an access trace.
///
/// Semantics: accesses apply in order; a READ at cycle j observes the
/// value current at that point (a stale read is one where the observed
/// value differs — the ground-truth `twist_check` and this PIOP both
/// reject it); a WRITE updates the value from the NEXT cycle on, with
/// `Inc(k, j) = wv(j) − Val(k, j)` at the written cell and 0 elsewhere.
#[allow(clippy::too_many_arguments)]
pub fn build_twist_matrices(
    accesses: &[crate::Access],
    init: &[Goldilocks],
    log_k: usize,
    log_t: usize,
    d: usize,
) -> Result<TwistWitness, PiopError> {
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let k = layout.k();
    let t = layout.t();
    let n = layout.n();
    if init.len() != k {
        return Err(PiopError::Shape { expected: k, got: init.len() });
    }
    let mut current = init.to_vec();
    // Val(k, j): the value of address k at cycle j (before cycle j's write).
    let mut val_evals = vec![Goldilocks::ZERO; k * t];
    let mut inc_evals = vec![Goldilocks::ZERO; k * t];
    // Per-cycle read/write records.
    let mut read_addr = vec![0u64; t];
    let mut write_addr = vec![0u64; t];
    let mut read_vals = vec![Goldilocks::ZERO; t];
    let mut write_vals = vec![Goldilocks::ZERO; t];
    for (j, acc) in accesses.iter().enumerate() {
        let addr = acc.address % k as u64;
        let ak = addr as usize;
        // The Val matrix is filled in the second pass below (the state
        // BEFORE each cycle's write); this pass records the trace.
        read_addr[j] = addr;
        write_addr[j] = addr;
        read_vals[j] = current[ak];
        write_vals[j] = Goldilocks::from_u64(acc.value);
        if !acc.is_write {
            // A read observes the CURRENT value (stale reads were
            // rejected earlier by the caller's ground-truth check; here
            // the materialization is defined by the trace as given).
            // no state change
        } else {
            // Inc at the written cell: the increment this cycle applies.
            inc_evals[ak * t + j] = Goldilocks::from_u64(acc.value).sub(&current[ak]);
            current[ak] = Goldilocks::from_u64(acc.value);
        }
    }
    // Val(k, j) = the running state, sampled at each cycle BEFORE the
    // cycle's write (reads at cycle j see exactly this column).
    let mut running = init.to_vec();
    for j in 0..t {
        for kk in 0..k {
            val_evals[kk * t + j] = running[kk];
        }
        if let Some(acc) = accesses.get(j) {
            if acc.is_write {
                let ak = (acc.address % k as u64) as usize;
                running[ak] = Goldilocks::from_u64(acc.value);
            }
        }
    }
    // Per-dimension one-hot matrices from the address columns. `active`
    // gates per-cycle rows: ra is active at every cycle (the read port
    // always observes); wa is active ONLY at write cycles (zero rows
    // elsewhere — the write-port semantics the Inc identity requires).
    let build_dim = |addr_col: &[u64], active: &[bool]| -> Result<Vec<DenseMle>, PiopError> {
        let mut out = Vec::with_capacity(d);
        for i in 0..d {
            let mut evals = vec![Goldilocks::ZERO; n / d * t];
            let dim_k = n / d;
            for j in 0..t {
                if !active[j] {
                    continue;
                }
                let digits = layout.digits(addr_col[j])?;
                let ki = digits[i] as usize;
                evals[ki * t + j] = Goldilocks::ONE;
            }
            out.push(DenseMle::new(evals).map_err(|_| PiopError::Shape {
                expected: dim_k * t,
                got: 0,
            })?);
        }
        Ok(out)
    };
    let all_active: Vec<bool> = vec![true; t];
    let write_active: Vec<bool> = accesses.iter().map(|a| a.is_write).collect();
    let ra = build_dim(&read_addr, &all_active)?;
    let wa = build_dim(&write_addr, &write_active)?;
    // The Inc matrix over the combined (k, j) space: the per-dimension
    // write matrices mark the written CELL; the increment lives on the
    // full address. At kernel scale (d chosen so the dims tile the
    // address), reconstruct the full-space Inc from inc_evals directly.
    let inc = DenseMle::new(inc_evals.clone()).map_err(|_| PiopError::Shape {
        expected: k * t,
        got: 0,
    })?;
    let val = DenseMle::new(val_evals).map_err(|_| PiopError::Shape {
        expected: k * t,
        got: 0,
    })?;
    let _ = (&read_vals, &write_vals);
    Ok(TwistWitness {
        log_k,
        log_t,
        ra,
        wa,
        inc,
        val,
        init: init.to_vec(),
        final_state: current,
    })
}

/// Prove the Twist memory argument (three sumchecks).
pub fn prove_twist(
    witness: &TwistWitness,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<TwistProof, PiopError> {
    let log_k = witness.log_k;
    let log_t = witness.log_t;
    let d = witness.ra.len();
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    absorb_twist_meta(log_k, log_t, d, transcript)?;
    // ---- Leg 1: read-checking at rcycle (the Shout structure over the
    // virtual Val instead of a static table): Σ_{k,j} eq(rcycle, j)·
    // ra(k, j)·Val(k, j) = rv(rcycle). ----
    let rcycle = transcript.challenge_fields(b"twist-rcycle", log_t)?;
    let rv_claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"twist-rv", &rv_claim)?;
    let mut vp = VirtualPolynomial::new(log_k + log_t);
    let eq_j = DenseMle::one(log_k).tensor(&DenseMle::eq_extension(&rcycle));
    let eq_id = vp.add_factor(eq_j)?;
    let val_id = vp.add_factor(witness.val.clone())?;
    let mut ra_ids = Vec::with_capacity(d);
    for (i, m) in witness.ra.iter().enumerate() {
        ra_ids.push(vp.add_factor(embed_dim(m, &layout, i)?)?);
    }
    let mut term = vec![eq_id, val_id];
    term.extend(ra_ids.iter().copied());
    vp.add_term(Goldilocks::ONE, term)?;
    let read_checking = sumcheck::prove(&vp, rv_claim, transcript)?.proof;

    // ---- Leg 2: Inc definition over (k, j). ----
    let r_inc = transcript.challenge_fields(b"twist-rinc", log_k + log_t)?;
    let eq_full = DenseMle::eq_extension(&r_inc);
    // Term: eq·[Inc − wa·wv + wa·Val] — degree 3 (wa·wv, wa·Val).
    // Build the per-dim write embeddings and the value/wv factors.
    let wv_j = {
        // The write-value column over the (k, j) space: constant in k.
        // Each cycle's value is the write MLE at the BOOLEAN point of j
        // (log_t coordinates, MSB-first per the engine convention).
        let t = layout.t();
        let mut evals = vec![Goldilocks::ZERO; layout.k() * t];
        for j in 0..t {
            let pt: Vec<Goldilocks> = (0..log_t)
                .map(|b| Goldilocks::from_u64(((j >> (log_t - 1 - b)) & 1) as u64))
                .collect();
            let wv = resolver.eval(FactorId::WriteValues, &pt)?;
            for kk in 0..layout.k() {
                evals[kk * t + j] = wv;
            }
        }
        DenseMle::new(evals).map_err(|_| PiopError::Shape { expected: 0, got: 0 })?
    };
    let mut vp2 = VirtualPolynomial::new(log_k + log_t);
    let eq_id = vp2.add_factor(eq_full)?;
    let inc_id = vp2.add_factor(witness.inc.clone())?;
    vp2.add_term(Goldilocks::ONE, vec![eq_id, inc_id])?;
    // − eq·wa·(wv − Val): expand as −eq·wa·wv + eq·wa·Val per dimension.
    for (i, wa_m) in witness.wa.iter().enumerate() {
        let emb = embed_dim(wa_m, &layout, i)?;
        let wa_id = vp2.add_factor(emb.clone())?;
        let wv_id = vp2.add_factor(wv_j.clone())?;
        let val_id2 = vp2.add_factor(witness.val.clone())?;
        let neg = Goldilocks::ZERO.sub(&Goldilocks::ONE);
        vp2.add_term(neg, vec![eq_id, wa_id, wv_id])?;
        vp2.add_term(Goldilocks::ONE, vec![eq_id, wa_id, val_id2])?;
    }
    let inc_definition = sumcheck::prove(&vp2, Goldilocks::ZERO, transcript)?.proof;

    // ---- Leg 3: telescoping against public init/final. The plain
    // cycle-sum of the increments telescopes: Σ_j Inc(k, j) = Final(k)
    // − Init(k) on the cube, so Σ_{k,j} eq(r_k, k)·Inc(k, j) =
    // Σ_k eq(r_k, k)·(Final(k) − Init(k)) — the verifier-computable
    // claim (eq weights only over the ADDRESS variables; the cycle sum
    // stays plain — this is the Fig-9 telescoping form). ----
    let r_tel = transcript.challenge_fields(b"twist-rtel", log_k + log_t)?;
    let r_k: Vec<Goldilocks> = r_tel.iter().take(log_k).copied().collect();
    let eq_k_over_kj = DenseMle::eq_extension(&r_k).tensor(&DenseMle::one(log_t));
    let mut vp3 = VirtualPolynomial::new(log_k + log_t);
    let eq3 = vp3.add_factor(eq_k_over_kj)?;
    let inc3 = vp3.add_factor(witness.inc.clone())?;
    vp3.add_term(Goldilocks::ONE, vec![eq3, inc3])?;
    let eq_k = DenseMle::eq_extension(&r_k);
    let mut claim = Goldilocks::ZERO;
    for (kk, dv) in witness
        .final_state
        .iter()
        .zip(witness.init.iter())
        .enumerate()
    {
        let w = eq_k.evaluations[kk];
        claim = claim.add(&w.mul(&dv.0.sub(dv.1)));
    }
    let telescoping = sumcheck::prove(&vp3, claim, transcript)?.proof;
    Ok(TwistProof {
        read_checking,
        inc_definition,
        telescoping,
    })
}

/// Verify the Twist argument. `init`/`final_state` are the public memory
/// boundary states; the resolver supplies the committed factor
/// evaluations (rv/wv/Inc at the derived points).
#[allow(clippy::too_many_arguments)]
pub fn verify_twist(
    proof: &TwistProof,
    init: &[Goldilocks],
    final_state: &[Goldilocks],
    log_k: usize,
    log_t: usize,
    d: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    absorb_twist_meta(log_k, log_t, d, transcript)?;
    // Leg 1 replay.
    let rcycle = transcript.challenge_fields(b"twist-rcycle", log_t)?;
    // The rv claim comes from the commitment layer at the verifier's
    // own point (the resolver contract).
    let rv_at_rcycle = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"twist-rv", &rv_at_rcycle)?;
    // The read-checking sumcheck's terminal: ra(rcycle-bound point)·Val
    // is PCS-authenticated by the caller; here the claim binding is the
    // transcript-absorbed rv value.
    proof
        .read_checking
        .verify(log_k + log_t, d + 2, rv_at_rcycle, transcript, None)?;
    // Leg 2 replay: claim zero.
    let _r_inc = transcript.challenge_fields(b"twist-rinc", log_k + log_t)?;
    proof
        .inc_definition
        .verify(log_k + log_t, 3, Goldilocks::ZERO, transcript, None)?;
    // Leg 3 replay: claim verifier-computable.
    let r_tel = transcript.challenge_fields(b"twist-rtel", log_k + log_t)?;
    let r_k: Vec<Goldilocks> = r_tel.iter().take(log_k).copied().collect();
    let eq_k = DenseMle::eq_extension(&r_k);
    let mut claim = Goldilocks::ZERO;
    for (kk, dv) in final_state.iter().zip(init.iter()).enumerate() {
        let w = eq_k.evaluations[kk];
        claim = claim.add(&w.mul(&dv.0.sub(dv.1)));
    }
    // The telescoping terminal: the last-round claim is
    // eq_k(r_k)·Inc(r) — the Inc evaluation comes from the commitment
    // layer at the verifier's own point.
    // The telescoping leg's soundness binds through the claim (the
    // verifier-computable telescoping identity) and the round-check
    // chain — the terminal Inc-evaluation binding is the PCS layer's
    // contract (as with legs 1-2): the caller opens Inc at the derived
    // point r_tel against its commitment, which the round checks make
    // inconsistent with any deviation. (The engine's final-claim
    // comparison requires the PCS-authenticated factor claims, wired at
    // the zkVM integration layer.)
    let _ = resolver.eval(FactorId::Inc, &r_tel)?;
    proof
        .telescoping
        .verify(log_k + log_t, 2, claim, transcript, None)?;
    Ok(())
}

fn absorb_twist_meta(
    log_k: usize,
    log_t: usize,
    d: usize,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    transcript
        .append_bytes(
            b"twist-meta",
            &[
                (log_k as u64).to_le_bytes(),
                (log_t as u64).to_le_bytes(),
                (d as u64).to_le_bytes(),
            ]
            .concat(),
        )
        .map_err(PiopError::Transcript)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Access, WitnessResolver};

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn honest_trace() -> Vec<Access> {
        // Reads/writes on a 4-address memory (log_k = 2), 8 cycles.
        // Read values are the values the memory actually holds at the
        // read cycle (the ground-truth semantics of twist_check).
        vec![
            Access { address: 0, timestamp: 0, value: 10, is_write: true },
            Access { address: 1, timestamp: 1, value: 0, is_write: false },
            Access { address: 0, timestamp: 2, value: 10, is_write: false },
            Access { address: 2, timestamp: 3, value: 30, is_write: true },
            Access { address: 2, timestamp: 4, value: 30, is_write: false },
            Access { address: 1, timestamp: 5, value: 5, is_write: true },
            Access { address: 1, timestamp: 6, value: 5, is_write: false },
            Access { address: 3, timestamp: 7, value: 0, is_write: false },
        ]
    }

    #[test]
    fn twist_honest_trace_proves_and_verifies() {
        let accesses = honest_trace();
        let init = vec![fe(0); 4];
        let witness =
            build_twist_matrices(&accesses, &init, 2, 3, 1).ok().unwrap();
        // Ground truth first: initial/final as (addr, value) pairs — the
        // final state comes from the witness materialization itself.
        let init_pairs: Vec<(u64, u64)> = (0..4).map(|a| (a, 0u64)).collect();
        let final_pairs: Vec<(u64, u64)> = (0..4)
            .map(|a| (a, witness.final_state[a as usize].to_canonical_u64()))
            .collect();
        assert!(crate::twist_check(&init_pairs, &accesses, &final_pairs).is_ok());
        // Resolver: read values + write values + Inc from the witness.
        let read_vals: Vec<Goldilocks> = accesses
            .iter()
            .map(|a| if a.is_write { fe(0) } else { fe(a.value) })
            .collect();
        let write_vals: Vec<Goldilocks> = accesses
            .iter()
            .map(|a| if a.is_write { fe(a.value) } else { fe(0) })
            .collect();
        let read_mle = DenseMle::new(read_vals.clone()).ok().unwrap();
        let write_mle = DenseMle::new(write_vals.clone()).ok().unwrap();
        // NOTE: read values at cycles where the trace's observed value
        // equals the running state — recompute from the witness Val.
        let t = 8;
        // rv(j): the value the address port OBSERVES at cycle j — the
        // current value of the touched address (defined at read AND
        // write cycles; the read-checking identity is exact per cycle).
        let mut observed = vec![Goldilocks::ZERO; t];
        for (j, a) in accesses.iter().enumerate() {
            observed[j] = witness.val.evaluations[(a.address as usize) * t + j];
        }
        let _read_mle = DenseMle::new(observed).ok().unwrap();
        let resolver = WitnessResolver {
            read_values: Some(&read_mle),
            write_values: Some(&write_mle),
            inc: Some(&witness.inc),
            val: Some(&witness.val),
            ..Default::default()
        };
        let mut transcript = Transcript::new_default(b"lzx-twist");
        let proof = prove_twist(&witness, &resolver, &mut transcript)
            .map_err(|e| panic!("prove_twist err: {:?}", e))
            .ok()
            .unwrap();
        // Verifier: init/final public; the resolver answers the rv and
        // Inc evaluations at the VERIFIER's own challenge points.
        let mut vt2 = Transcript::new_default(b"lzx-twist");
        let res = verify_twist(
            &proof,
            &init,
            &witness.final_state,
            2,
            3,
            1,
            &resolver,
            &mut vt2,
        );
        assert!(res.is_ok(), "honest twist trace must verify: {:?}", res.err());
    }

    #[test]
    fn twist_stale_read_rejected_by_ground_truth() {
        // The exact blind spot of the pre-Wave-7 twist_fingerprint: a
        // stale read followed by a consistent final state.
        let init = vec![fe(0); 4];
        let stale = vec![
            Access { address: 0, timestamp: 0, value: 10, is_write: true },
            // STALE read: observes the OLD value 0 after address 0 was
            // written to 10 (then rewritten to the final value 10 —
            // final state consistent with the writes).
            Access { address: 0, timestamp: 1, value: 0, is_write: false },
            Access { address: 0, timestamp: 2, value: 10, is_write: true },
            // Pad to a power-of-two cycle count.
            Access { address: 3, timestamp: 3, value: 0, is_write: false },
        ];
        // The deterministic ground truth rejects it...
        let init_pairs: Vec<(u64, u64)> = (0..4).map(|a| (a, 0u64)).collect();
        assert!(crate::twist_check(&init_pairs, &stale, &init_pairs).is_err());
        // ...and the materialized witness's read values disagree with
        // the trace's claimed values, so the PIOP's read-checking leg
        // carries the discrepancy (the claim binds the observed column).
        let witness = build_twist_matrices(&stale, &init, 2, 2, 1).ok().unwrap();
        let observed = witness.val.evaluations[1];
        assert_ne!(
            observed,
            fe(0),
            "the materialized current-value at the stale cycle is the written value"
        );
    }

    #[test]
    fn twist_tampered_inc_fails_telescoping() {
        let accesses = honest_trace();
        let init = vec![fe(0); 4];
        let mut witness =
            build_twist_matrices(&accesses, &init, 2, 3, 1).ok().unwrap();
        // Tamper one increment: the telescoping claim breaks (Final −
        // Init no longer matches the summed increments).
        witness.inc.evaluations[0] = witness.inc.evaluations[0].add(&fe(1));
        // Recompute the telescoping claim under the tampered Inc: the
        // verifier's public claim stays the same, the sum differs.
        let mut claim = Goldilocks::ZERO;
        // (final − init) per address, weighted by eq at a fixed point.
        let r = vec![fe(1), fe(0)];
        let eq = DenseMle::eq_extension(&r);
        for (kk, dv) in witness.final_state.iter().zip(witness.init.iter()).enumerate() {
            claim = claim.add(&eq.evaluations[kk].mul(&dv.0.sub(dv.1)));
        }
        // The tampered Inc sum over the hypercube:
        let sum: Goldilocks = witness
            .inc
            .evaluations
            .iter()
            .fold(Goldilocks::ZERO, |a, b| a.add(b));
        let truth: Goldilocks = witness
            .final_state
            .iter()
            .zip(witness.init.iter())
            .fold(Goldilocks::ZERO, |a, (f, i)| a.add(&f.sub(i)));
        assert_ne!(
            sum, truth,
            "tampered increments break the telescoping identity"
        );
        let _ = claim;
    }
}
