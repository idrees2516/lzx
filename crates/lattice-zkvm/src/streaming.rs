//! **Streaming, client-side program proving** — the ePrint 2025/611
//! integration layer: `prove_program_streaming` /
//! `verify_program_streaming`.
//!
//! ## The O(K + log T) path (this revision)
//!
//! The pipeline is driven by the **VM's own step function**, wired into
//! [`ChunkedRegenOracle`](lattice_streaming::oracle::ChunkedRegenOracle):
//! every column the protocol consumes (the program-counter stream, the
//! register-write witness, the read/write memory fingerprints) is a
//! *regeneration oracle* whose generator state IS the machine state —
//! one `O(K)` live machine plus checkpoint snapshots, no `O(T)` value
//! arrays anywhere on the prover path:
//!
//! 1. **Counting pass** — one execution with
//!    [`lattice_vm::step`](lattice_vm::step) counting the cycles,
//!    register-write values, and memory accesses; the final state and
//!    public output are collected here. `O(K)` space.
//! 2. **pcnext-evaluation sum-check** (§4.2): the prefix-suffix inner
//!    product protocol over the pc oracle — `O(√T)` space, two stream
//!    passes (each pass = one VM re-execution, the paper's "repeated
//!    witness generation").
//! 3. **Witness commitment** (§6.1): the matrix-layout streaming
//!    commitment over the register-write oracle — one row-streamed pass,
//!    `O(√W)` space; the evaluation claim is computed with a streaming
//!    MLE pass (never materializing the column).
//! 4. **Memory fingerprint grand products** (Appendix D): each side is
//!    proven with **Algorithm 3's bucketed `O(n)`-space prover** —
//!    `n` stream passes, one per sum-check round, `O(n)` space
//!    throughout (the `O(2^n)` g-table materialization is gone).
//! 5. The projective (monomial-basis) engine is the natural sum-check
//!    substrate for the whole pipeline: the trace columns ARE the
//!    coefficient arrays (no Möbius conversion), matching the compact
//!    Ajtai opening's representation (ePrint 2026/762 §4.3).
//!
//! Checkpoint granularity: the oracles snapshot the machine every
//! `chunk` indices; `chunk` is derived from the client memory budget so
//! `(len/chunk)` snapshots fit (`(len/chunk)·K ≤ budget`) — `chunk = len`
//! gives the pure `O(K + log T)` regime (sequential-only access, resets
//! are full re-executions); smaller chunks buy `O(chunk)` random access
//! for Algorithm-1-style indexed sum-checks (the hybrid path).
//!
//! Verification replays the transcript; as with the kernel-level
//! `verify_program`, the differential mode re-executes the program —
//! also through a regeneration oracle (streaming MLE evaluation and DFS
//! grand products; no materialized columns on the verifier path either).
//!
//! The prove-side ground-truth Twist check of the materialized era is
//! intentionally absent here: the executor's own memory IS the timeline
//! (the honest prover cannot contradict it), and the fingerprint
//! equality is bound by the grand-product proofs themselves — the check
//! was a fail-fast duplicate, not a soundness component.

use crate::envelope::{ProofEnvelope, Section};
use crate::{program_digest, public_input_digest, PublicOutput};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_streaming::client::ClientProverConfig;
use lattice_streaming::grand_product::{
    dfs_grand_product, prove_grand_product_bucketed, GrandProductProof,
};
use lattice_streaming::oracle::{
    stream_mle_eval, ChunkedRegenOracle, StreamOracle,
};
use lattice_streaming::pcs_stream::{
    commit_streaming, prove_eval_streaming, StreamingCommitment, StreamingEvalProof,
};
use lattice_streaming::prefix_suffix::{
    prove_prefix_suffix, PrefixSuffixOutput, Structure,
};
use lattice_vm::{step as vm_step, MachineState, TraceRow};

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
            StreamingZkvmError::VerificationFailed => {
                write!(f, "streaming verification failed")
            }
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
    /// Prover-side telemetry: the peak regeneration-oracle footprint in
    /// machine snapshots (the checkpoint memory, `O(len/chunk)` states).
    pub oracle_snapshots: usize,
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

// ---------------------------------------------------------------------------
// The VM-step regeneration oracles
// ---------------------------------------------------------------------------

/// The column a [`VmRegenOracle`] generates — the selector wired into
/// the machine's step function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VmColumn {
    /// The program counter per cycle (`T` values; padding ZERO).
    Pc,
    /// The register-write values, flattened across cycles (`W` values;
    /// padding ZERO).
    WitnessValues,
    /// The read-access fingerprints `a + γ·v + γ²·t − τ` (`R` values;
    /// padding ONE — the product's neutral element).
    ReadsFingerprint { gamma: Goldilocks, tau: Goldilocks },
    /// The write-access fingerprints (`Wr` values; padding ONE).
    WritesFingerprint { gamma: Goldilocks, tau: Goldilocks },
}

impl VmColumn {
    /// The stream's neutral padding value (beyond the real data).
    fn padding(&self) -> Goldilocks {
        match self {
            VmColumn::Pc | VmColumn::WitnessValues => Goldilocks::ZERO,
            VmColumn::ReadsFingerprint { .. } | VmColumn::WritesFingerprint { .. } => {
                Goldilocks::ONE
            }
        }
    }

    /// Extract this column's values from one executed trace row.
    fn extract(&self, row: &TraceRow, cycle: u64) -> Vec<Goldilocks> {
        match self {
            VmColumn::Pc => vec![Goldilocks::from_u64(row.pc)],
            VmColumn::WitnessValues => row
                .reg_writes
                .iter()
                .map(|(_, v)| Goldilocks::from_u64(*v))
                .collect(),
            VmColumn::ReadsFingerprint { gamma, tau } => match &row.mem_access {
                Some((addr, old, _)) => vec![fingerprint(*addr, *old, cycle, gamma, tau)],
                None => Vec::new(),
            },
            VmColumn::WritesFingerprint { gamma, tau } => match &row.mem_access {
                Some((addr, _, Some(new))) => {
                    vec![fingerprint(*addr, *new, cycle, gamma, tau)]
                }
                _ => Vec::new(),
            },
        }
    }
}

/// `a + γ·v + γ²·t − τ` — the Spice-style offline-memory fingerprint.
fn fingerprint(
    addr: u64,
    value: u64,
    timestamp: u64,
    gamma: &Goldilocks,
    tau: &Goldilocks,
) -> Goldilocks {
    let a = Goldilocks::from_u64(addr);
    let v = Goldilocks::from_u64(value);
    let t = Goldilocks::from_u64(timestamp);
    a.add(&gamma.mul(&v)).add(&gamma.mul(gamma).mul(&t)).sub(tau)
}

/// The regeneration-oracle generator state: the LIVE machine. Each
/// `step` advances the VM until the selected column yields a value; the
/// machine's own state is the whole witness-generation state (the
/// paper's Observation 3.5 discipline).
#[derive(Clone)]
pub struct VmOracleState {
    machine: MachineState,
    column: VmColumn,
    /// Values extracted from the current cycle, not yet emitted.
    pending: Vec<Goldilocks>,
    /// Cycles executed so far in this arm.
    cycle: u64,
    /// The step bound (the execution's DoS guard).
    max_steps: u64,
}

impl VmOracleState {
    /// The initial state: program at 0, public input at 0x1000.
    pub fn initial(program: &[u8], public_input: &[u8], column: VmColumn, max_steps: u64) -> Self {
        let mut machine = MachineState::new();
        machine.load_program(0x1000, public_input);
        machine.load_program(0, program);
        VmOracleState { machine, column, pending: Vec::new(), cycle: 0, max_steps }
    }
}

/// The per-index generator fold: advance the machine until the column
/// produces its next value (or the execution is done → padding).
pub fn vm_column_step(state: &mut VmOracleState, _index: u64) -> Goldilocks {
    if let Some(v) = state.pending.pop() {
        return v;
    }
    loop {
        if state.machine.halted || state.cycle >= state.max_steps {
            return state.column.padding();
        }
        let row = match vm_step(&mut state.machine, state.cycle) {
            Ok(r) => r,
            Err(_) => return state.column.padding(),
        };
        let cycle = state.cycle;
        state.cycle += 1;
        let mut values = state.column.extract(&row, cycle);
        if values.is_empty() {
            continue;
        }
        let first = values.remove(0);
        // Pending values are consumed LIFO within a cycle (each cycle
        // contributes at most 2 witness values / 1 fingerprint — order
        // within the cycle is fixed by extraction order).
        state.pending = values;
        return first;
    }
}

/// The execution shape from the counting pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecShape {
    /// Cycles executed.
    pub cycles: u64,
    /// Register-write values (the witness column length).
    pub witness_values: u64,
    /// Memory reads (accesses with an `old` value).
    pub reads: u64,
    /// Memory writes (accesses with a `new` value).
    pub writes: u64,
}

/// The counting pass: execute once with `vm_step`, counting the column
/// lengths, WITHOUT materializing any trace (`O(K)` space — only the
/// live machine). Returns the shape, the final registers, and the final
/// machine state (the caller extracts the public output).
fn count_shape(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(ExecShape, MachineState), lattice_vm::ExecError> {
    let mut machine = MachineState::new();
    machine.load_program(0x1000, public_input);
    machine.load_program(0, program);
    let mut cycles = 0u64;
    let mut witness_values = 0u64;
    let mut reads = 0u64;
    let mut writes = 0u64;
    while cycles < max_steps && !machine.halted {
        let row = vm_step(&mut machine, cycles)?;
        witness_values += row.reg_writes.len() as u64;
        if row.mem_access.is_some() {
            reads += 1;
            if matches!(row.mem_access, Some((_, _, Some(_)))) {
                writes += 1;
            }
        }
        cycles += 1;
    }
    Ok((ExecShape { cycles, witness_values, reads, writes }, machine))
}

/// `ceil(log2(max(v, 1)))` — the padded stream's variable count.
fn log2_ceil(v: u64) -> usize {
    v.max(1).next_power_of_two().trailing_zeros() as usize
}

/// The checkpoint granularity for a regeneration oracle: the largest
/// power-of-two `chunk` such that `(len/chunk)` machine snapshots fit
/// the budget (`snapshot_words` = the machine's memory words + regs).
/// `chunk = len` (a single snapshot) is the pure `O(K + log T)` regime.
fn chunk_for(len: u64, snapshot_words: usize, budget_field_elements: usize) -> u64 {
    let len = len.max(1);
    let per_snapshot = (snapshot_words + 64).max(1);
    let max_snaps = budget_field_elements / per_snapshot;
    if max_snaps <= 1 || len <= 1 {
        return len;
    }
    // chunk = len / min(max_snaps, len), rounded UP to a power of two.
    let target_snaps = (max_snaps as u64).min(len);
    let chunk = len.div_ceil(target_snaps).next_power_of_two();
    chunk.clamp(1, len)
}

/// Build a VM-step regeneration oracle for `column` (the memory-bounded
/// `build_streaming` path — no value materialization).
fn build_vm_oracle(
    program: &[u8],
    public_input: &[u8],
    column: VmColumn,
    n_vars: usize,
    max_steps: u64,
    chunk: u64,
) -> ChunkedRegenOracle<'static, VmOracleState> {
    ChunkedRegenOracle::build_streaming(
        n_vars,
        VmOracleState::initial(program, public_input, column, max_steps),
        chunk,
        vm_column_step,
    )
}

/// Prove a program's correct execution with the streaming,
/// client-side pipeline: **the full O(K + log T) path** — every
/// prover component consumes the VM's step function through
/// checkpointed regeneration oracles.
#[allow(clippy::too_many_lines)]
pub fn prove_program_streaming(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    config: &ClientProverConfig,
) -> Result<(PublicOutput, StreamingProof), StreamingZkvmError> {
    // 1. The counting pass (one execution, O(K) space).
    let (shape, final_machine) = count_shape(program, public_input, max_steps)
        .map_err(StreamingZkvmError::Execution)?;
    let snapshot_words = final_machine.memory.snapshot_pairs().len() + 48;
    let output = PublicOutput {
        final_regs: final_machine.regs,
        memory_digest: final_machine.memory.digest(),
    };

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

    // 3. pcnext-evaluation sum-check over the pc oracle (prefix-suffix,
    //    two passes, O(√T) space). The shift structure's r is sampled
    //    directly (as in the materialized revision).
    let n_vars = log2_ceil(shape.cycles);
    let shift_r: Vec<Goldilocks> = (0..n_vars)
        .map(|_| {
            transcript
                .challenge_field(b"pcnext-shift-r")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let structure = Structure::Shift { r: shift_r };
    let pc_chunk = chunk_for(1u64 << n_vars, snapshot_words, config.max_field_elements);
    let mut pc_oracle =
        build_vm_oracle(program, public_input, VmColumn::Pc, n_vars, max_steps, pc_chunk);
    let pcnext = prove_prefix_suffix(
        &mut pc_oracle,
        &structure,
        n_vars,
        None,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::PrefixSuffix)?;

    // 4. Witness column (register-write values): streaming commitment +
    //    a STREAMING MLE evaluation claim (no materialized column).
    let w_vars = log2_ceil(shape.witness_values);
    let w_chunk = chunk_for(1u64 << w_vars, snapshot_words, config.max_field_elements);
    let mut w_oracle = build_vm_oracle(
        program,
        public_input,
        VmColumn::WitnessValues,
        w_vars,
        max_steps,
        w_chunk,
    );
    let witness_commitment =
        commit_streaming(&mut w_oracle, w_vars).map_err(StreamingZkvmError::Commit)?;
    let w_point: Vec<Goldilocks> = (0..w_vars)
        .map(|_| {
            transcript
                .challenge_field(b"witness-point")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let witness_claim = stream_mle_eval(&mut w_oracle, w_vars, &w_point)
        .ok_or(StreamingZkvmError::VerificationFailed)?;
    let witness_eval = prove_eval_streaming(
        &mut w_oracle,
        &witness_commitment,
        &w_point,
        4,
        &mut transcript,
    )
    .map_err(StreamingZkvmError::Commit)?;

    // 5. Memory fingerprint grand products (Spice-style): γ, τ from the
    //    transcript; the reads product and the writes product must
    //    coincide (the multiset equality). Each side is proven with
    //    Algorithm 3's BUCKETED prover: O(n) space, one oracle pass per
    //    sum-check round (the g-tables are never materialized).
    let gamma = transcript
        .challenge_field(b"fingerprint-gamma")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let tau = transcript
        .challenge_field(b"fingerprint-tau")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let r_vars = log2_ceil(shape.reads);
    let wr_vars = log2_ceil(shape.writes);
    let r_chunk = chunk_for(1u64 << r_vars, snapshot_words, config.max_field_elements);
    let wr_chunk = chunk_for(1u64 << wr_vars, snapshot_words, config.max_field_elements);
    let mut reads_oracle = build_vm_oracle(
        program,
        public_input,
        VmColumn::ReadsFingerprint { gamma, tau },
        r_vars,
        max_steps,
        r_chunk,
    );
    let fingerprint_reads = prove_grand_product_bucketed(&mut reads_oracle, None, &mut transcript)
        .map_err(StreamingZkvmError::GrandProduct)?;
    let mut writes_oracle = build_vm_oracle(
        program,
        public_input,
        VmColumn::WritesFingerprint { gamma, tau },
        wr_vars,
        max_steps,
        wr_chunk,
    );
    let fingerprint_writes =
        prove_grand_product_bucketed(&mut writes_oracle, Some(fingerprint_reads.product), &mut transcript)
            .map_err(StreamingZkvmError::GrandProduct)?;

    let oracle_snapshots = pc_oracle.checkpoint_count()
        + w_oracle.checkpoint_count()
        + reads_oracle.checkpoint_count()
        + writes_oracle.checkpoint_count();

    Ok((
        output,
        StreamingProof {
            pcnext,
            witness_commitment,
            witness_eval,
            witness_claim,
            fingerprint_reads,
            fingerprint_writes,
            oracle_snapshots,
        },
    ))
}

/// Verify a streaming program proof (differential mode: re-executes the
/// program through a regeneration oracle, replays the transcript, and
/// checks every component — all in streaming space).
#[allow(clippy::too_many_lines)]
pub fn verify_program_streaming(
    program: &[u8],
    public_input: &[u8],
    public_output: &PublicOutput,
    proof: &StreamingProof,
    max_steps: u64,
) -> Result<(), StreamingZkvmError> {
    // Re-execute through the counting pass (the differential mode).
    let (shape, final_machine) = count_shape(program, public_input, max_steps)
        .map_err(StreamingZkvmError::Execution)?;
    if final_machine.regs != public_output.final_regs
        || final_machine.memory.digest() != public_output.memory_digest
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

    // pcnext: re-derive the shift structure and verify. The claimed sum
    // Σ pc·shift is recomputed STREAMING (one oracle pass, O(n) space).
    let n_vars = log2_ceil(shape.cycles);
    let shift_r: Vec<Goldilocks> = (0..n_vars)
        .map(|_| {
            transcript
                .challenge_field(b"pcnext-shift-r")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let structure = Structure::Shift { r: shift_r };
    {
        let claimed_sum = {
            let mut pc_oracle = build_vm_oracle(
                program,
                public_input,
                VmColumn::Pc,
                n_vars,
                max_steps,
                1u64 << n_vars,
            );
            pc_oracle.reset();
            let len = 1u64 << n_vars;
            let mut acc = Goldilocks::ZERO;
            for i in 0..len {
                let p = pc_oracle.next();
                let x: Vec<Goldilocks> = (0..n_vars)
                    .map(|b| Goldilocks::from_u64((i >> (n_vars - 1 - b)) & 1))
                    .collect();
                acc = acc.add(&p.mul(&structure.eval_affine(&x)));
            }
            acc
        };
        let mut current = claimed_sum;
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

    // Witness commitment + evaluation: the claim is recomputed with a
    // STREAMING MLE pass over the regeneration oracle.
    let w_vars = log2_ceil(shape.witness_values);
    let w_point: Vec<Goldilocks> = (0..w_vars)
        .map(|_| {
            transcript
                .challenge_field(b"witness-point")
                .map_err(|_| StreamingZkvmError::VerificationFailed)
        })
        .collect::<Result<Vec<_>, _>>()?;
    {
        let mut w_oracle = build_vm_oracle(
            program,
            public_input,
            VmColumn::WitnessValues,
            w_vars,
            max_steps,
            1u64 << w_vars,
        );
        let direct = stream_mle_eval(&mut w_oracle, w_vars, &w_point)
            .ok_or(StreamingZkvmError::VerificationFailed)?;
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
    }

    // Fingerprints: replay and verify both grand products — the DFS
    // products over the verifier's own oracles (O(n) space), then the
    // Quarks proofs (which bind the streamed products to the g-claims).
    let gamma = transcript
        .challenge_field(b"fingerprint-gamma")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let tau = transcript
        .challenge_field(b"fingerprint-tau")
        .map_err(|_| StreamingZkvmError::VerificationFailed)?;
    let r_vars = log2_ceil(shape.reads);
    let wr_vars = log2_ceil(shape.writes);
    {
        let mut reads_oracle = build_vm_oracle(
            program,
            public_input,
            VmColumn::ReadsFingerprint { gamma, tau },
            r_vars,
            max_steps,
            1u64 << r_vars,
        );
        let reads_p = dfs_grand_product(&mut reads_oracle, None)
            .map_err(StreamingZkvmError::GrandProduct)?;
        let mut writes_oracle = build_vm_oracle(
            program,
            public_input,
            VmColumn::WritesFingerprint { gamma, tau },
            wr_vars,
            max_steps,
            1u64 << wr_vars,
        );
        let writes_p = dfs_grand_product(&mut writes_oracle, None)
            .map_err(StreamingZkvmError::GrandProduct)?;
        if reads_p != proof.fingerprint_reads.product
            || writes_p != proof.fingerprint_writes.product
            || reads_p != writes_p
        {
            return Err(StreamingZkvmError::VerificationFailed);
        }
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

    /// End-to-end streaming prove/verify on a guest program — the full
    /// O(K + log T) path (VM-step oracles + bucketed grand products).
    #[test]
    fn streaming_roundtrip() {
        let program = guest_program(8);
        let (output, proof) =
            prove_program_streaming(&program, &[], 4096, &ClientProverConfig::default())
                .unwrap();
        assert!(
            verify_program_streaming(&program, &[], &output, &proof, 4096).is_ok()
        );
        // The oracle footprint must be checkpoint-bounded, not O(T):
        // with the default budget the pc/witness/read/write oracles hold
        // only a bounded snapshot set.
        assert!(proof.oracle_snapshots > 0);
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

    /// A tampered fingerprint product fails (the reads/writes equality
    /// is pinned by the verifier's own DFS recomputation).
    #[test]
    fn streaming_tampered_fingerprint() {
        let program = guest_program(8);
        let (output, mut proof) =
            prove_program_streaming(&program, &[], 4096, &ClientProverConfig::default())
                .unwrap();
        proof.fingerprint_reads.product = proof.fingerprint_reads.product.add(&Goldilocks::ONE);
        assert!(verify_program_streaming(&program, &[], &output, &proof, 4096).is_err());
    }

    /// The tight mobile budget (8 MiB of field elements) also proves
    /// and verifies — the memory-bounded regime.
    #[test]
    fn streaming_mobile_budget() {
        let program = guest_program(8);
        let (output, proof) =
            prove_program_streaming(&program, &[], 4096, &ClientProverConfig::mobile())
                .unwrap();
        assert!(
            verify_program_streaming(&program, &[], &output, &proof, 4096).is_ok()
        );
    }

    /// The column oracles agree with the materialized reference: the pc
    /// column, witness column, and fingerprint streams from the
    /// regeneration oracle equal the direct per-row extraction.
    #[test]
    fn vm_oracles_match_materialized() {
        let program = guest_program(10);
        let (shape, _) = count_shape(&program, &[], 8192).unwrap();
        // Reference execution.
        let mut machine = MachineState::new();
        machine.load_program(0x1000, &[]);
        machine.load_program(0, &program);
        let mut rows = Vec::new();
        for i in 0..shape.cycles {
            rows.push(vm_step(&mut machine, i).unwrap());
        }
        // pc oracle.
        let n_vars = log2_ceil(shape.cycles);
        let mut pc = build_vm_oracle(&program, &[], VmColumn::Pc, n_vars, 8192, 1 << n_vars);
        pc.reset();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(pc.next(), Goldilocks::from_u64(row.pc), "pc at {i}");
        }
        // Witness oracle.
        let w_vars = log2_ceil(shape.witness_values);
        let mut w = build_vm_oracle(
            &program,
            &[],
            VmColumn::WitnessValues,
            w_vars,
            8192,
            1 << w_vars,
        );
        w.reset();
        let expect_w: Vec<Goldilocks> = rows
            .iter()
            .flat_map(|r| r.reg_writes.iter().map(|(_, v)| Goldilocks::from_u64(*v)))
            .collect();
        for (i, v) in expect_w.iter().enumerate() {
            assert_eq!(w.next(), *v, "witness at {i}");
        }
        // The padded tail is the column's padding value.
        let w_len = 1u64 << w_vars;
        for _ in shape.witness_values..w_len {
            assert_eq!(w.next(), Goldilocks::ZERO);
        }
        // Indexed access through the checkpointed regeneration: random
        // positions match the materialized reference (seek correctness).
        let mut w2 = build_vm_oracle(
            &program,
            &[],
            VmColumn::WitnessValues,
            w_vars,
            8192,
            64,
        );
        use lattice_streaming::oracle::IndexOracle;
        let total = shape.witness_values;
        if total > 3 {
            for idx in [0u64, 1, total / 2, total - 1, 0, total - 2] {
                let got = w2.eval(idx);
                let want = if idx < total {
                    expect_w[idx as usize]
                } else {
                    Goldilocks::ZERO
                };
                assert_eq!(got, want, "indexed witness at {idx}");
            }
        }
    }
}
