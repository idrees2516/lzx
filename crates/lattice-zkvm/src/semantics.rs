//! The instruction-semantics layer wired into the commitment stack:
//! the T2 constraint families proven over the trace's `CycleWitness`,
//! every factor claim bound through the bits/values bundles (the same
//! grouped-carrier discipline `memproof.rs` uses for the memory
//! argument).
//!
//! This is the pipeline integration of `constraints.rs` — the layer the
//! v2 pipeline's soundness scope documented as "the instruction-
//! semantics AIR is the next layer". With it, the zkVM's prove path
//! covers decode, ALU (arith + shifts + MUL + DIV), comparisons,
//! control flow, memory routing, and termination with the verifier
//! recomputing only public material, and every auxiliary column
//! range-linked to boolean bits so the limb recurrences carry integer
//! semantics.
//!
//! Statement discipline: the statement (program digest, input digest,
//! log_t, the RAM/fetch windows, the final registers) is absorbed
//! BEFORE any challenge; the bundle seeds derive from it; the coverage
//! gate is enforced by the DECODE leg's partition identities (the
//! verifier never sees the executed instruction list).

use crate::columns::{build_cycle_witness, CycleWitness, FetchWindow, RamWindow};
use crate::constraints::{
    aux_shape, prove_constraints, verify_constraints, AuxCols, ConstraintError, ConstraintLeg,
};
use crate::ledger::{
    bits_bundle_commit, values_bundle_commit, verify_bundle_opening, BaseClaim, BundleOpening,
    BundleProver, Factor, Ledger,
};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_vm::run as vm_run;
use lattice_vm::MachineState;

#[derive(Debug)]
pub enum SemanticsError {
    Execution(lattice_vm::ExecError),
    Witness(crate::columns::WitnessError),
    Constraints(ConstraintError),
    Ledger(crate::ledger::LedgerError),
}

impl From<lattice_vm::ExecError> for SemanticsError {
    fn from(e: lattice_vm::ExecError) -> Self {
        SemanticsError::Execution(e)
    }
}
impl From<crate::columns::WitnessError> for SemanticsError {
    fn from(e: crate::columns::WitnessError) -> Self {
        SemanticsError::Witness(e)
    }
}
impl From<ConstraintError> for SemanticsError {
    fn from(e: ConstraintError) -> Self {
        SemanticsError::Constraints(e)
    }
}
impl From<crate::ledger::LedgerError> for SemanticsError {
    fn from(e: crate::ledger::LedgerError) -> Self {
        SemanticsError::Ledger(e)
    }
}

/// The public semantics statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticsStatement {
    pub program_digest: [u8; 32],
    pub input_digest: [u8; 32],
    pub log_t: usize,
    pub ram_log_k: usize,
    pub fetch_log_k: usize,
    pub final_regs: [u64; 32],
}

/// The instruction-semantics proof: constraint legs + the claim list +
/// the two bundle openings.
#[derive(Clone, Debug)]
pub struct SemanticsProof {
    pub statement: SemanticsStatement,
    pub claims: Vec<BaseClaim>,
    pub legs: Vec<ConstraintLeg>,
    pub bits_commitment: Vec<u8>,
    pub values_commitment: Vec<u8>,
    pub bits_opening: BundleOpening,
    pub values_opening: BundleOpening,
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

fn absorb_statement(
    s: &SemanticsStatement,
    transcript: &mut Transcript,
) -> Result<(), crate::ledger::LedgerError> {
    let mut meta = vec![s.log_t as u64, s.ram_log_k as u64, s.fetch_log_k as u64];
    meta.extend(s.final_regs.iter().copied());
    let fields: Vec<Goldilocks> = meta.iter().map(|v| fe(*v)).collect();
    transcript
        .append_bytes(b"sem-prog", &s.program_digest)
        .map_err(crate::ledger::LedgerError::Transcript)?;
    transcript
        .append_bytes(b"sem-in", &s.input_digest)
        .map_err(crate::ledger::LedgerError::Transcript)?;
    transcript
        .append_field_slice(b"sem-meta", &fields)
        .map_err(crate::ledger::LedgerError::Transcript)?;
    Ok(())
}

fn derive_seed(s: &SemanticsStatement) -> [u8; 32] {
    let mut t = Transcript::new_default(b"lzx-semantics-seed");
    let _ = t.append_bytes(b"prog", &s.program_digest);
    let _ = t.append_bytes(b"in", &s.input_digest);
    let _ = t.append_bytes(
        b"meta",
        &[s.log_t as u64, s.ram_log_k as u64, s.fetch_log_k as u64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let mut seed = [0u8; 32];
    if let Ok(b) = t.challenge_bytes(b"seed", 32) {
        seed.copy_from_slice(&b);
    }
    seed
}

/// The bundle entries of a witness + aux (the deterministic factor
/// order: the six value tensors, the instruction tensor, every bit
/// column; the value columns separately).
/// A bundle factor table (factor -> its MLE) — the bits/values pairs.
type FactorBundle = Vec<(Factor, DenseMle)>;

fn bundle_entries(w: &CycleWitness, aux: &AuxCols) -> (FactorBundle, FactorBundle) {
    let log_t = w.log_t;
    let mut bits: Vec<(Factor, DenseMle)> = Vec::new();
    for slot in 0..crate::columns::VALUE_TENSORS {
        bits.push((Factor::ValueBits { slot }, w.values[slot].clone()));
    }
    bits.push((Factor::InstrBits, w.instr_bits.clone()));
    for (id, col) in aux.bits.iter().enumerate() {
        bits.push((
            Factor::BitCol { id },
            DenseMle {
                num_vars: log_t,
                evaluations: col.iter().map(|v| fe(*v as u64)).collect(),
            },
        ));
    }
    let mut vals: Vec<(Factor, DenseMle)> = Vec::new();
    for (id, col) in aux.vals.iter().enumerate() {
        vals.push((
            Factor::ValCol { id },
            DenseMle {
                num_vars: log_t,
                evaluations: col.clone(),
            },
        ));
    }
    (bits, vals)
}

/// Prove the instruction semantics of a program execution: executes,
/// builds the witness + aux, runs every constraint family, commits the
/// factor table through the bits/values bundles, and opens the recorded
/// claims.
pub fn prove_instruction_semantics(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    ram_log_k: usize,
    fetch_log_k: usize,
) -> Result<(SemanticsProof, [u64; 32]), SemanticsError> {
    // 1. Execute (the prover side; the verifier never does this).
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let rows = vm_run(&mut state, max_steps)?;
    let final_regs = state.regs;

    // 2. The witness + aux columns.
    let (w, _final_words) = build_cycle_witness(
        &rows,
        program,
        public_input,
        RamWindow { log_k: ram_log_k },
        FetchWindow { log_k: fetch_log_k },
    )?;
    let instrs: Vec<lattice_vm::decode::Instr> = rows.iter().map(|r| r.instr).collect();
    let aux = crate::constraints::build_aux(&w, &instrs)?;

    // 3. The statement + seeds + transcript.
    let statement = SemanticsStatement {
        program_digest: Transcript::hash_domain(b"zkvm-program", program),
        input_digest: Transcript::hash_domain(b"zkvm-public-input", public_input),
        log_t: w.log_t,
        ram_log_k,
        fetch_log_k,
        final_regs,
    };
    let seed = derive_seed(&statement);
    let mut transcript = Transcript::new_default(b"lzx-zkvm-semantics");
    absorb_statement(&statement, &mut transcript)?;

    // 4. The bundles over the factor table.
    let (bits_entries, values_entries) = bundle_entries(&w, &aux);
    let bits_prover: BundleProver = bits_bundle_commit(&bits_entries, seed)?;
    let values_prover: BundleProver = values_bundle_commit(&values_entries, seed)?;
    transcript
        .append_bytes(b"sem-bits-commitment", &bits_prover.commitment.to_bytes())
        .map_err(crate::ledger::LedgerError::Transcript)?;
    transcript
        .append_bytes(
            b"sem-values-commitment",
            &values_prover.commitment.to_bytes(),
        )
        .map_err(crate::ledger::LedgerError::Transcript)?;

    // 5. The constraint families over the ledger.
    let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
    for (f, m) in bits_entries.iter() {
        table.push((*f, m));
    }
    for (f, m) in values_entries.iter() {
        table.push((*f, m));
    }
    let mut ledger = Ledger::prover(table);
    let mut legs = Vec::new();
    prove_constraints(&w, &aux, &instrs, &mut ledger, &mut legs, &mut transcript)?;

    // 6. The bundle openings over the recorded claims.
    let claims: Vec<BaseClaim> = ledger.claims().to_vec();
    let bits_claims: Vec<BaseClaim> = claims
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = claims
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_opening = bits_prover.prove_opening(&bits_claims, &mut transcript)?;
    let values_opening = values_prover.prove_opening(&values_claims, &mut transcript)?;

    Ok((
        SemanticsProof {
            statement,
            claims,
            legs,
            bits_commitment: bits_prover.commitment.to_bytes(),
            values_commitment: values_prover.commitment.to_bytes(),
            bits_opening,
            values_opening,
        },
        final_regs,
    ))
}

/// Verify the instruction-semantics proof WITHOUT re-execution: replays
/// the statement transcript, walks every constraint family against the
/// claim-fed ledger, and checks both bundle openings.
pub fn verify_instruction_semantics(
    proof: &SemanticsProof,
    program: &[u8],
    public_input: &[u8],
) -> Result<(), SemanticsError> {
    // 1. The statement check (the digests + the public final regs).
    let expect = SemanticsStatement {
        program_digest: Transcript::hash_domain(b"zkvm-program", program),
        input_digest: Transcript::hash_domain(b"zkvm-public-input", public_input),
        log_t: proof.statement.log_t,
        ram_log_k: proof.statement.ram_log_k,
        fetch_log_k: proof.statement.fetch_log_k,
        final_regs: proof.statement.final_regs,
    };
    if expect != proof.statement {
        return Err(SemanticsError::Constraints(ConstraintError::Shape));
    }
    let log_t = proof.statement.log_t;

    // 2. The layout reconstruction (deterministic from log_t).
    let shape = aux_shape(log_t)?;
    let (bits_layout, values_layout) = layout_entries(&shape, log_t);

    // 3. The transcript replay.
    let seed = derive_seed(&proof.statement);
    let mut transcript = Transcript::new_default(b"lzx-zkvm-semantics");
    absorb_statement(&proof.statement, &mut transcript)?;
    transcript
        .append_bytes(b"sem-bits-commitment", &proof.bits_commitment)
        .map_err(crate::ledger::LedgerError::Transcript)?;
    transcript
        .append_bytes(b"sem-values-commitment", &proof.values_commitment)
        .map_err(crate::ledger::LedgerError::Transcript)?;

    // 4. The constraint families over the claim-fed ledger.
    let claims = proof.claims.clone();
    let mut ledger = Ledger::verifier(claims.clone());
    verify_constraints(&shape, &proof.legs, &mut ledger, &mut transcript)?;

    // 5. The bundle openings.
    let ring = crate::ledger::bundle_ring()?;
    let bits_claims: Vec<BaseClaim> = claims
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = claims
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_pk = bundle_pk(&ring, &bits_layout, true, seed)?;
    let values_pk = bundle_pk(&ring, &values_layout, false, seed)?;
    let bits_comm = lattice_commitment::ajtai::AjtaiCommitment::from_bytes(
        &ring,
        bits_pk.params.k,
        &proof.bits_commitment,
    )
    .map_err(crate::ledger::LedgerError::Ajtai)?;
    let values_comm = lattice_commitment::ajtai::AjtaiCommitment::from_bytes(
        &ring,
        values_pk.params.k,
        &proof.values_commitment,
    )
    .map_err(crate::ledger::LedgerError::Ajtai)?;
    verify_bundle_opening(
        &bits_pk,
        &bits_comm,
        &bits_layout,
        &bits_claims,
        &proof.bits_opening,
        true,
        &mut transcript,
    )?;
    verify_bundle_opening(
        &values_pk,
        &values_comm,
        &values_layout,
        &values_claims,
        &proof.values_opening,
        false,
        &mut transcript,
    )?;
    Ok(())
}

/// The layout entries (factor, num_vars, offset) the verifier
/// reconstructs from the deterministic factor order.
fn layout_entries(
    shape: &AuxCols,
    log_t: usize,
) -> (
    Vec<crate::ledger::BundleLayoutEntry>,
    Vec<crate::ledger::BundleLayoutEntry>,
) {
    let mut bits: Vec<crate::ledger::BundleLayoutEntry> = Vec::new();
    let mut off = 0usize;
    for slot in 0..crate::columns::VALUE_TENSORS {
        let n = 64usize << log_t;
        bits.push(crate::ledger::BundleLayoutEntry {
            factor: Factor::ValueBits { slot },
            num_vars: 6 + log_t,
            offset: off,
        });
        off += n;
    }
    bits.push(crate::ledger::BundleLayoutEntry {
        factor: Factor::InstrBits,
        num_vars: 5 + log_t,
        offset: off,
    });
    off += 32usize << log_t;
    for id in 0..shape.bits.len() {
        bits.push(crate::ledger::BundleLayoutEntry {
            factor: Factor::BitCol { id },
            num_vars: log_t,
            offset: off,
        });
        off += 1usize << log_t;
    }
    let mut vals: Vec<crate::ledger::BundleLayoutEntry> = Vec::new();
    let mut voff = 0usize;
    for id in 0..shape.vals.len() {
        vals.push(crate::ledger::BundleLayoutEntry {
            factor: Factor::ValCol { id },
            num_vars: log_t,
            offset: voff,
        });
        voff += 1usize << log_t;
    }
    (bits, vals)
}

/// Rebuild the verifier-side bundle public key from the layout (the
/// packed length matches the prover's construction).
fn bundle_pk(
    ring: &lattice_ring::RingConfig,
    layout: &[crate::ledger::BundleLayoutEntry],
    is_bits: bool,
    seed: [u8; 32],
) -> Result<lattice_commitment::ajtai::AjtaiPublicKey, crate::ledger::LedgerError> {
    // The packed stream length must match the prover's:
    // bits — 31 bits per coefficient, 64 coefficients per element;
    // values — three 22-bit limbs per field element.
    let flat_len: usize = layout
        .iter()
        .map(|e| 1usize << e.num_vars)
        .sum::<usize>()
        .next_power_of_two()
        .max(1);
    let ring_n = ring.n();
    let m = if is_bits {
        flat_len.div_ceil(31 * ring_n).max(1)
    } else {
        (flat_len * 3).div_ceil(ring_n).max(1)
    };
    let bound = if is_bits {
        crate::ledger::BITS_NORM_BOUND
    } else {
        crate::ledger::VALUES_NORM_BOUND
    };
    let params = lattice_commitment::ajtai::AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m,
        norm_bound: bound,
    };
    lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, seed)
        .map_err(crate::ledger::LedgerError::Ajtai)
}
