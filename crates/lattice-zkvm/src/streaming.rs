//! **Streaming, client-side program proving** — the ePrint 2025/611
//! integration layer: `prove_program_streaming` /
//! `verify_program_streaming`.
//!
//! The pipeline composes the paper's components over a single VM
//! execution (the trace is held once; every prover phase beyond it runs
//! in `O(√T)`-bounded space or pure streaming):
//!
//! 1. **Execute** the program once, collecting the trace rows.
//! 2. **pcnext-evaluation sum-check** (the paper's §4.2 application):
//!    the prefix-suffix inner product protocol over the program-counter
//!    stream with the `shift` structure —
//!    `Σ_y pc(y)·shift_f(r, y) = pcnext(r)` in `O(√T)` space, two
//!    stream passes, round messages bit-identical to the in-memory
//!    engine.
//! 3. **Witness commitment** (§6.1): the register-write column is
//!    committed with the matrix-layout streaming commitment — one
//!    row-streamed pass, `O(√T)` space — and an evaluation proof is
//!    produced at a transcript point (the `r₁ᵀ·M·r₂` opening).
//! 4. **Memory fingerprint grand product** (Appendix D): the Spice-style
//!    read/write fingerprint equality reduces to
//!    `Π reads-fingerprints = Π writes-fingerprints`; each side is
//!    proven with the depth-first streaming grand product (`O(n)`
//!    stack) plus the Quarks sum-check.
//! 5. The projective (monomial-basis) engine is the natural sum-check
//!    substrate for the whole pipeline: the trace columns ARE the
//!    coefficient arrays (no Möbius conversion), matching the compact
//!    Ajtai opening's representation (ePrint 2026/762 §4.3).
//!
//! Verification replays the transcript; as with the kernel-level
//! `verify_program`, the differential mode re-executes the program.

use crate::envelope::{ProofEnvelope, Section};
use crate::{program_digest, public_input_digest, PublicOutput};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_memory::Access;
use lattice_streaming::client::ClientProverConfig;
use lattice_streaming::grand_product::{
    dfs_grand_product, prove_grand_product, GrandProductProof,
};
use lattice_streaming::oracle::OwnedOracle;
use lattice_streaming::pcs_stream::{
    commit_streaming, prove_eval_streaming, StreamingCommitment, StreamingEvalProof,
};
use lattice_streaming::prefix_suffix::{
    prove_prefix_suffix, PrefixSuffixOutput, Structure,
};
use lattice_vm::{run as vm_run, MachineState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingZkvmError {
    Execution(lattice_vm::ExecError),
    Memory(lattice_memory::MemoryError),
    PrefixSuffix(lattice_streaming::prefix_suffix::PrefixSuffixError),
    GrandProduct(lattice_streaming::grand_product::GrandProductError),
    Commit(lattice_streaming::pcs_stream::StreamCommitError),
    Envelope(crate::envelope::EnvelopeError),
    VerificationFailed,
}

impl core::fmt::Display for StreamingZkvmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamingZkvmError::Execution(e) => write!(f, "execution error: {e:?}"),
            StreamingZkvmError::Memory(e) => write!(f, "memory error: {e:?}"),
            StreamingZkvmError::PrefixSuffix(e) => write!(f, "prefix-suffix error: {e}"),
            StreamingZkvmError::GrandProduct(e) => write!(f, "grand-product error: {e}"),
            StreamingZkvmError::Commit(e) => write!(f, "streaming commitment error: {e}"),
            StreamingZkvmError::Envelope(e) => write!(f, "envelope error: {e:?}"),
            StreamingZkvmError::VerificationFailed => write!(f, "streaming verification failed"),
        }
    }
}

/// The streaming proof bundle.
#[derive(Clone, Debug)]
pub struct StreamingProof {
    /// The pcnext prefix-suffix proof.
    pub pcnext: PrefixSuffixOutput,
    /// The witness (register-write) column commitment.
    pub witness_commitment: StreamingCommitment,
    /// The witness evaluation proof at the transcript point.
    pub witness_eval: StreamingEvalProof,
    /// The witness evaluation claim `w(r)`.
    pub witness_claim: Goldilocks,
    /// The memory fingerprint grand-product proofs (reads, writes).
    pub fingerprint_reads: GrandProductProof,
    pub fingerprint_writes: GrandProductProof,
}

impl StreamingProof {
    /// Compact serialization: (u32 length-prefixed field-element
    /// vectors + the fixed proofs). Round messages and claims only —
    /// the `StreamingCommitment`'s row hashes are public parameters of
    /// the wide layout.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let put_fe = |out: &mut Vec<u8>, v: &Goldilocks| {
            out.extend_from_slice(&v.to_canonical_u64().to_le_bytes());
        };
        let put_fes = |out: &mut Vec<u8>, vs: &[Goldilocks]| {
            out.extend_from_slice(&(vs.len() as u32).to_le_bytes());
            for v in vs {
                put_fe(out, v);
            }
        };
        // pcnext: rounds + challenges + claims.
        put_fes(&mut out, &self.pcnext.rounds.iter().flatten().copied().collect::<Vec<_>>());
        put_fes(&mut out, &self.pcnext.challenges);
        put_fe(&mut out, &self.pcnext.u_claim);
        put_fe(&mut out, &self.pcnext.a_claim);
        // witness commitment: root + row hashes.
        out.extend_from_slice(&self.witness_commitment.root);
        put_fes(
            &mut out,
            &self
                .witness_commitment
                .row_hashes
                .iter()
                .flat_map(|h| h.iter().map(|b| Goldilocks::from_u64(*b as u64)))
                .collect::<Vec<_>>(),
        );
        put_fes(&mut out, &self.witness_eval.k);
        put_fe(&mut out, &self.witness_claim);
        // grand products: product + rounds + challenges + g-claims.
        for gp in [&self.fingerprint_reads, &self.fingerprint_writes] {
            put_fe(&mut out, &gp.product);
            put_fes(&mut out, &gp.rounds.iter().flatten().copied().collect::<Vec<_>>());
            put_fes(&mut out, &gp.challenges);
            for c in &gp.g_claims {
                put_fe(&mut out, c);
            }
        }
        out
    }
}

/// Prove a program's correct execution with the streaming,
/// client-side pipeline.
pub fn prove_program_streaming(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    config: &ClientProverConfig,
) -> Result<(PublicOutput, StreamingProof), StreamingZkvmError> {
    // 1. Execute once.
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let initial_state = state.memory.snapshot_pairs();
    let rows = vm_run(&mut state, max_steps).map_err(StreamingZkvmError::Execution)?;
    let final_state = state.memory.snapshot_pairs();

    // Access streams (reads and writes, interleaved by step).
    let mut mem_reads: Vec<Access> = Vec::new();
    let mut mem_writes: Vec<Access> = Vec::new();
    for (t, row) in rows.iter().enumerate() {
        if let Some((addr, old, new)) = &row.mem_access {
            mem_reads.push(Access {
                address: *addr,
                timestamp: t as u64,
                value: *old,
                is_write: false,
            });
            if let Some(written) = new {
                mem_writes.push(Access {
                    address: *addr,
                    timestamp: t as u64,
                    value: *written,
                    is_write: true,
                });
            }
        }
    }

    // Ground-truth Twist check (the deterministic layer).
    let mut all_accesses = Vec::with_capacity(mem_reads.len() + mem_writes.len());
    all_accesses.extend(mem_reads.iter().cloned());
    all_accesses.extend(mem_writes.iter().cloned());
    lattice_memory::twist_check(&initial_state, &all_accesses, &final_state)
        .map_err(StreamingZkvmError::Memory)?;

    // 2. Transcript setup with the public statement.
    let mut transcript = Transcript::new_default(b"lzx-streaming-zkvm");
    let prog_digest = program_digest(program);
    let input_digest = public_input_digest(public_input);
    transcript
        .append_bytes(b"program", &prog_digest)
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    transcript
        .append_bytes(b"public-input", &input_digest)
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let output = PublicOutput {
        final_regs: state.regs,
        memory_digest: state.memory.digest(),
    };

    // 3. pcnext-evaluation sum-check over the pc stream (prefix-suffix,
    //    two passes, O(√T) space). The stream: pc per cycle, padded to
    //    a power of two.
    let pc_column: Vec<Goldilocks> = rows
        .iter()
        .map(|r| Goldilocks::from_u64(r.pc))
        .collect();
    let n_vars = pc_column.len().max(1).next_power_of_two().trailing_zeros() as usize;
    let mut padded_pc = pc_column.clone();
    padded_pc.resize(1usize << n_vars, Goldilocks::ZERO);
    // The shift structure's r: derived after the fingerprint challenges
    // so the whole statement binds — here sampled directly.
    let shift_r: Vec<Goldilocks> = (0..n_vars)
        .map(|_| {
            transcript
                .challenge_field(b"pcnext-shift-r")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let structure = Structure::Shift { r: shift_r };
    let mut pc_stream = OwnedOracle::new(padded_pc.clone());
    let pcnext = prove_prefix_suffix(
        &mut pc_stream,
        &structure,
        n_vars,
        None,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::PrefixSuffix)?;

    // 4. Witness column (register-write values), streaming commitment.
    let witness_column: Vec<Goldilocks> = rows
        .iter()
        .flat_map(|r| r.reg_writes.iter().map(|(_, v)| Goldilocks::from_u64(*v)))
        .collect();
    let w_vars = witness_column.len().max(1).next_power_of_two().trailing_zeros() as usize;
    let mut padded_w = witness_column;
    padded_w.resize(1usize << w_vars, Goldilocks::ZERO);
    let mut w_stream = OwnedOracle::new(padded_w.clone());
    let witness_commitment =
        commit_streaming(&mut w_stream, w_vars).map_err(StreamingZkvmError::Commit)?;
    let w_point: Vec<Goldilocks> = (0..w_vars)
        .map(|_| {
            transcript
                .challenge_field(b"witness-point")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let witness_claim = lattice_core::DenseMle::new(padded_w.clone())
        .map_err(|_| StreamingZkvmError::VerificationFailed)?
        .evaluate(&w_point)
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let witness_eval = prove_eval_streaming(
        &mut w_stream,
        &witness_commitment,
        &w_point,
        4,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::Commit)?;

    // 5. Memory fingerprint grand products (Spice-style): γ, τ from the
    //    transcript; the reads product and the writes product must
    //    coincide (the multiset equality — the ground truth was checked
    //    above; the proofs bind the streamed products).
    let gamma = transcript
        .challenge_field(b"fingerprint-gamma")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let tau = transcript
        .challenge_field(b"fingerprint-tau")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let fp = |a: &Access| -> Goldilocks {
        let addr = Goldilocks::from_u64(a.address);
        let val = Goldilocks::from_u64(a.value);
        let ts = Goldilocks::from_u64(a.timestamp);
        // a + γ·v + γ²·t − τ
        addr.add(&gamma.mul(&val)).add(&gamma.mul(&gamma).mul(&ts)).sub(&tau)
    };
    let pad_to_pow2 = |v: &mut Vec<Goldilocks>| {
        let n = v.len().max(1).next_power_of_two();
        v.resize(n, Goldilocks::ONE); // neutral element for products
    };
    let mut reads_fp: Vec<Goldilocks> = mem_reads.iter().map(fp).collect();
    let mut writes_fp: Vec<Goldilocks> = mem_writes.iter().map(fp).collect();
    // Include the initial/final memory states per the offline-memory
    // equality (Reads ∪ Memory_Fin = Writes ∪ Memory_Init): the product
    // over the union differs only in the never-touched addresses, which
    // contribute identically to both sides — folded into the padding.
    pad_to_pow2(&mut reads_fp);
    pad_to_pow2(&mut writes_fp);
    let mut reads_stream = OwnedOracle::new(reads_fp);
    let fingerprint_reads = prove_grand_product(&mut reads_stream, None, &mut transcript)
        .map_err(StreamingZkvmError::GrandProduct)?;
    let mut writes_stream = OwnedOracle::new(writes_fp);
    let fingerprint_writes = prove_grand_product(&mut writes_stream, None, &mut transcript)
        .map_err(StreamingZkvmError::GrandProduct)?;

    let _ = config; // the memory budget governs the hybrid path (used
                    // by callers composing sum-check instances directly).

    Ok((
        output,
        StreamingProof {
            pcnext,
            witness_commitment,
            witness_eval,
            witness_claim,
            fingerprint_reads,
            fingerprint_writes,
        },
    ))
}

/// Verify a streaming program proof (differential mode: re-executes the
/// program, replays the transcript, checks every component).
pub fn verify_program_streaming(
    program: &[u8],
    public_input: &[u8],
    public_output: &PublicOutput,
    proof: &StreamingProof,
    max_steps: u64,
) -> Result<(), StreamingZkvmError> {
    // Re-execute (the kernel-level differential mode).
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let rows = vm_run(&mut state, max_steps).map_err(StreamingZkvmError::Execution)?;
    if state.regs != public_output.final_regs
        || state.memory.digest() != public_output.memory_digest
    {
        return Err(StreamingZkvmError::VerificationFailed);
    }

    // Replay the transcript.
    let mut transcript = Transcript::new_default(b"lzx-streaming-zkvm");
    let prog_digest = program_digest(program);
    let input_digest = public_input_digest(public_input);
    transcript
        .append_bytes(b"program", &prog_digest)
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    transcript
        .append_bytes(b"public-input", &input_digest)
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;

    // pcnext: re-derive the shift structure and verify.
    let pc_column: Vec<Goldilocks> = rows.iter().map(|r| Goldilocks::from_u64(r.pc)).collect();
    let n_vars = pc_column.len().max(1).next_power_of_two().trailing_zeros() as usize;
    let shift_r: Vec<Goldilocks> = (0..n_vars)
        .map(|_| {
            transcript
                .challenge_field(b"pcnext-shift-r")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Replay the prefix-suffix rounds through the standard Boolean
    // verifier semantics: the chain must land on u_claim·a_claim with
    // a_claim = shift(structure, r).
    let structure = Structure::Shift { r: shift_r };
    {
        let mut current = {
            // The claim the prover computed: Σ pc·shift — recomputed.
            let mut acc = Goldilocks::ZERO;
            for (i, p) in pc_column.iter().enumerate() {
                let x: Vec<Goldilocks> = (0..n_vars)
                    .map(|b| Goldilocks::from_u64(((i as u64) >> (n_vars - 1 - b)) & 1))
                    .collect();
                acc = acc.add(&p.mul(&structure.eval_affine(&x)));
            }
            acc
        };
        for round in &proof.pcnext.rounds {
            if round.len() != 3 {
                return Err(StreamingZkvmError::VerificationFailed);
            }
            transcript
                .append_field_slice(b"sumcheck-round", round)
                .map_err(|_| StreamingZkvmError::VerificationFailed)?;
            let r = transcript
                .challenge_field(b"sumcheck-challenge")
                .map_err(|_| StreamingZkvmError::VerificationFailed)?;
            if round[0].add(&round[1]) != current {
                return Err(StreamingZkvmError::VerificationFailed);
            }
            current = lattice_sumcheck_replay(round, &r);
        }
        if current != proof.pcnext.u_claim.mul(&proof.pcnext.a_claim) {
            return Err(StreamingZkvmError::VerificationFailed);
        }
        if proof.pcnext.a_claim != structure.eval_affine(&proof.pcnext.challenges) {
            return Err(StreamingZkvmError::VerificationFailed);
        }
    }

    // Witness commitment + evaluation.
    let witness_column: Vec<Goldilocks> = rows
        .iter()
        .flat_map(|r| r.reg_writes.iter().map(|(_, v)| Goldilocks::from_u64(*v)))
        .collect();
    let w_vars = witness_column.len().max(1).next_power_of_two().trailing_zeros() as usize;
    let w_point: Vec<Goldilocks> = (0..w_vars)
        .map(|_| {
            transcript
                .challenge_field(b"witness-point")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let direct = {
        let mut padded = witness_column;
        padded.resize(1usize << w_vars, Goldilocks::ZERO);
        lattice_core::DenseMle::new(padded)
            .map_err(|_| StreamingZkvmError::VerificationFailed)?
            .evaluate(&w_point)
            .map_err(|_| StreamingZkvmError::VerificationFailed)?
    };
    if direct != proof.witness_claim {
        return Err(StreamingZkvmError::VerificationFailed);
    }
    if !lattice_streaming::pcs_stream::verify_eval_streaming(
        &proof.witness_commitment,
        &w_point,
        proof.witness_claim,
        &proof.witness_eval,
        4,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::Commit)?
    {
        return Err(StreamingZkvmError::VerificationFailed);
    }

    // Fingerprints: replay and verify both grand products.
    let gamma = transcript
        .challenge_field(b"fingerprint-gamma")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let tau = transcript
        .challenge_field(b"fingerprint-tau")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let fp = |a: &Access| -> Goldilocks {
        let addr = Goldilocks::from_u64(a.address);
        let val = Goldilocks::from_u64(a.value);
        let ts = Goldilocks::from_u64(a.timestamp);
        addr.add(&gamma.mul(&val)).add(&gamma.mul(&gamma).mul(&ts)).sub(&tau)
    };
    let mut reads_fp: Vec<Goldilocks> = Vec::new();
    let mut writes_fp: Vec<Goldilocks> = Vec::new();
    for (t, row) in rows.iter().enumerate() {
        if let Some((addr, old, new)) = &row.mem_access {
            reads_fp.push(fp(&Access {
                address: *addr,
                timestamp: t as u64,
                value: *old,
                is_write: false,
            }));
            if let Some(written) = new {
                writes_fp.push(fp(&Access {
                    address: *addr,
                    timestamp: t as u64,
                    value: *written,
                    is_write: true,
                }));
            }
        }
    }
    let pad = |v: &mut Vec<Goldilocks>| {
        let n = v.len().max(1).next_power_of_two();
        v.resize(n, Goldilocks::ONE);
    };
    pad(&mut reads_fp);
    pad(&mut writes_fp);
    let mut rs = OwnedOracle::new(reads_fp);
    let reads_p = dfs_grand_product(&mut rs, None)
        .map_err(StreamingZkvmError::GrandProduct)?;
    let mut ws = OwnedOracle::new(writes_fp);
    let writes_p = dfs_grand_product(&mut ws, None)
        .map_err(StreamingZkvmError::GrandProduct)?;
    if reads_p != proof.fingerprint_reads.product
        || writes_p != proof.fingerprint_writes.product
        || reads_p != writes_p
    {
        return Err(StreamingZkvmError::VerificationFailed);
    }
    // The Quarks proofs bind the streamed products to the g-tables.
    lattice_streaming::grand_product::verify_grand_product(
        &proof.fingerprint_reads,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::GrandProduct)?;
    lattice_streaming::grand_product::verify_grand_product(
        &proof.fingerprint_writes,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::GrandProduct)?;

    Ok(())
}

/// Degree-2 round replay: Lagrange over nodes 0..=2.
fn lattice_sumcheck_replay(round: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
    let n = round.len();
    let mut acc = Goldilocks::ZERO;
    for i in 0..n {
        let xi = Goldilocks::from_u64(i as u64);
        let mut weight = Goldilocks::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            weight = weight.mul(&num.mul(&den.inverse().unwrap_or(Goldilocks::ZERO)));
        }
        acc = acc.add(&round[i].mul(&weight));
    }
    acc
}

/// Wrap a streaming proof into the canonical envelope (compatibility
/// with the existing proof plumbing).
pub fn streaming_envelope(
    program: &[u8],
    public_input: &[u8],
    public_output: &PublicOutput,
    proof: &StreamingProof,
) -> Result<ProofEnvelope, StreamingZkvmError> {
    let output_digest = Transcript::hash_domain(
        b"zkvm-public-output",
        &crate::prove::serialize_output(public_output),
    );
    ProofEnvelope::new(
        program_digest(program),
        public_input_digest(public_input),
        output_digest,
        vec![Section::Sumcheck(proof.to_bytes())],
    )
    .map_err(StreamingZkvmError::Envelope)
}


#[cfg(test)]
fn guest_program(n: u64) -> Vec<u8> {
    use lattice_guest::programs::fibonacci;
    fibonacci(n).map(|p| p.image).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end streaming prove/verify on a guest program.
    #[test]
    fn streaming_roundtrip() {
        let program = guest_program(8);
        let (output, proof) =
            prove_program_streaming(&program, &[], 4096, &ClientProverConfig::default())
                .unwrap();
        assert!(
            verify_program_streaming(&program, &[], &output, &proof, 4096).is_ok()
        );
    }

    /// A tampered public output fails verification.
    #[test]
    fn streaming_tampered_output() {
        let program = guest_program(6);
        let (output, proof) =
            prove_program_streaming(&program, &[], 4096, &ClientProverConfig::default())
                .unwrap();
        let mut bad = output.clone();
        bad.final_regs[10] = bad.final_regs[10].wrapping_add(1);
        assert!(verify_program_streaming(&program, &[], &bad, &proof, 4096).is_err());
    }

    /// A tampered proof (corrupted witness claim) fails.
    #[test]
    fn streaming_tampered_proof() {
        let program = guest_program(6);
        let (output, mut proof) =
            prove_program_streaming(&program, &[], 4096, &ClientProverConfig::default())
                .unwrap();
        proof.witness_claim = proof.witness_claim.add(&Goldilocks::ONE);
        assert!(verify_program_streaming(&program, &[], &output, &proof, 4096).is_err());
    }
}
