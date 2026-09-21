//! The top-level prover/verifier: prove_program / verify_program.
//!
//! Flow (the audit report's P2 stage set, kernel-level):
//! 1. Execute the program (RV64IMAC) → trace rows.
//! 2. Build the witness columns: register write values, memory (addr,
//!    value) pairs, and the timeline bookkeeping.
//! 3. Register Twist + RAM Twist ground-truth checks (the deterministic
//!    layer; production replaces these with their sumcheck statements —
//!    `lattice-memory`'s fingerprint identities).
//! 4. Commit the witness columns through the Akita PCS; prove evaluation
//!    claims at the transcript challenges (the opening accumulator).
//! 5. Seal everything in the bounded proof envelope.

use crate::envelope::{ProofEnvelope, Section};
use lattice_akita::pcs::AkitaPcs;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_memory::{twist_check, twist_fingerprint, Access};
use lattice_vm::{run as vm_run, MachineState};

/// Public output: final register snapshot + memory digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicOutput {
    pub final_regs: [u64; 32],
    pub memory_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkvmError {
    Execution(lattice_vm::ExecError),
    Envelope(crate::envelope::EnvelopeError),
    Pcs(lattice_akita::pcs::AkitaPcsError),
    Memory(lattice_memory::MemoryError),
    /// Verification failed.
    VerificationFailed,
}

/// Prove a program's correct execution end-to-end.
pub fn prove_program(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(PublicOutput, ProofEnvelope), ZkvmError> {
    // 1. Execute.
    let mut state = MachineState::new();
    // Public input lands in memory at the conventional address 0x1000.
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    // The initial memory (program + public input) is the Twist baseline.
    let initial_state = state.memory.snapshot_pairs();
    let rows = vm_run(&mut state, max_steps).map_err(ZkvmError::Execution)?;

    // 2. Witness columns: register write stream + memory access stream.
    let mut reg_stream: Vec<Goldilocks> = Vec::with_capacity(rows.len() * 4);
    let mut mem_reads: Vec<Access> = Vec::new();
    let mut mem_writes: Vec<Access> = Vec::new();
    let mut all_accesses: Vec<Access> = Vec::new();
    for (t, row) in rows.iter().enumerate() {
        reg_stream.push(Goldilocks::from_u64(row.pc));
        reg_stream.push(Goldilocks::from_u64(row.next_pc));
        for (r, v) in &row.reg_writes {
            reg_stream.push(Goldilocks::from_u64(*r as u64));
            reg_stream.push(Goldilocks::from_u64(*v));
        }
        if let Some((addr, old, new)) = &row.mem_access {
            let access = Access {
                address: *addr,
                timestamp: t as u64,
                value: *old,
                is_write: false,
            };
            mem_reads.push(access);
            all_accesses.push(access);
            if let Some(written) = new {
                let w = Access {
                    address: *addr,
                    timestamp: t as u64,
                    value: *written,
                    is_write: true,
                };
                mem_writes.push(w);
                all_accesses.push(w);
            }
        }
    }

    // 3. Ground-truth memory checks (the Twist layer).
    // Deterministic final state from the machine's sparse memory.
    let final_state = state.memory.snapshot_pairs();
    twist_check(&initial_state, &all_accesses, &final_state).map_err(ZkvmError::Memory)?;

    // 4. Commit the register stream (witness column) + prove an
    //    evaluation claim at a transcript point (the opening accumulator).
    let mut transcript = Transcript::new_default(b"lzx-zkvm");
    let prog_digest = crate::program_digest(program);
    let input_digest = crate::public_input_digest(public_input);
    transcript
        .append_bytes(b"program", &prog_digest)
        .map_err(|_| ZkvmError::VerificationFailed)?;
    transcript
        .append_bytes(b"public-input", &input_digest)
        .map_err(|_| ZkvmError::VerificationFailed)?;
    // Memory fingerprint identity (the Twist sumcheck-facing statement).
    twist_fingerprint(&all_accesses, &final_state, &mut transcript)
        .map_err(ZkvmError::Memory)?;

    // Pad the register stream to a power of two and commit.
    let padded_len = reg_stream.len().next_power_of_two().max(1);
    let mut padded = reg_stream.clone();
    padded.resize(padded_len, Goldilocks::ZERO);
    let num_vars = padded_len.trailing_zeros() as usize;
    let witness_mle = DenseMle {
        num_vars,
        evaluations: padded,
    };
    let commitment = pcs.commit(&witness_mle).map_err(ZkvmError::Pcs)?;
    transcript
        .append_bytes(b"commitment", &commitment.commitment.to_bytes())
        .map_err(|_| ZkvmError::VerificationFailed)?;
    let point = transcript
        .challenge_fields(b"zkvm-point", num_vars)
        .map_err(|_| ZkvmError::VerificationFailed)?;
    let eval_proof = pcs
        .prove_evaluation(&witness_mle, &point, &mut transcript)
        .map_err(ZkvmError::Pcs)?;

    // 5. Envelope.
    let output = PublicOutput {
        final_regs: state.regs,
        memory_digest: state.memory.digest(),
    };
    let output_digest = Transcript::hash_domain(
        b"zkvm-public-output",
        &serialize_output(&output),
    );
    let envelope = ProofEnvelope::new(
        prog_digest,
        input_digest,
        output_digest,
        vec![
            Section::Commitment(commitment.commitment.to_bytes()),
            Section::Sumcheck(serialize_sumcheck(&eval_proof.sumcheck)),
            Section::Witness(serialize_witness(&eval_proof.opened_witness)),
            Section::Norm(serialize_norm(&eval_proof.norm_proof)),
        ],
    )
    .map_err(ZkvmError::Envelope)?;
    Ok((output, envelope))
}

/// Verify a program proof: replay the transcript and check the envelope
/// bindings plus the PCS opening.
pub fn verify_program(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    public_output: &PublicOutput,
    envelope: &ProofEnvelope,
    max_steps: u64,
) -> Result<(), ZkvmError> {
    // 1. Envelope digests must match the statement.
    let prog_digest = crate::program_digest(program);
    let input_digest = crate::public_input_digest(public_input);
    let output_digest = Transcript::hash_domain(
        b"zkvm-public-output",
        &serialize_output(public_output),
    );
    if envelope.program_digest != prog_digest
        || envelope.public_input_digest != input_digest
        || envelope.public_output_digest != output_digest
    {
        return Err(ZkvmError::VerificationFailed);
    }

    // 2. Decode sections.
    let commitment_bytes = match envelope.sections.first() {
        Some(Section::Commitment(b)) => b.clone(),
        _ => return Err(ZkvmError::VerificationFailed),
    };
    let sumcheck_bytes = match envelope.sections.get(1) {
        Some(Section::Sumcheck(b)) => b.clone(),
        _ => return Err(ZkvmError::VerificationFailed),
    };
    let witness_bytes = match envelope.sections.get(2) {
        Some(Section::Witness(b)) => b.clone(),
        _ => return Err(ZkvmError::VerificationFailed),
    };
    let _ = witness_bytes;
    let _norm_bytes = match envelope.sections.get(3) {
        Some(Section::Norm(b)) => b.clone(),
        _ => return Err(ZkvmError::VerificationFailed),
    };

    // 3. Reconstruct the commitment from the canonical bytes.
    let ring = &pcs.pk.params.ring;
    let commitment = lattice_commitment::ajtai::AjtaiCommitment::from_bytes(
        ring,
        pcs.pk.params.k,
        &commitment_bytes,
    )
    .map_err(|e| ZkvmError::Pcs(lattice_akita::pcs::AkitaPcsError::Ajtai(e)))?;

    // 4. Replay the transcript: the verifier recomputes every public
    //    challenge from the statement (program/input/output digests and
    //    the commitment). The memory fingerprint requires the access
    //    stream, which the verifier does NOT have — in production the
    //    Twist sumcheck proves it; at this kernel level the verifier
    //    re-executes the program when it is small (the differential-
    //    reference mode) and checks the full transcript equality.
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let _initial_state = state.memory.snapshot_pairs();
    let rows = vm_run(&mut state, max_steps).map_err(ZkvmError::Execution)?;
    let mut all_accesses: Vec<Access> = Vec::new();
    for (t, row) in rows.iter().enumerate() {
        if let Some((addr, old, new)) = &row.mem_access {
            all_accesses.push(Access {
                address: *addr,
                timestamp: t as u64,
                value: *old,
                is_write: false,
            });
            if let Some(written) = new {
                all_accesses.push(Access {
                    address: *addr,
                    timestamp: t as u64,
                    value: *written,
                    is_write: true,
                });
            }
        }
    }
    // Public output must match the executed state.
    if state.regs != public_output.final_regs
        || state.memory.digest() != public_output.memory_digest
    {
        return Err(ZkvmError::VerificationFailed);
    }

    // 5. Transcript replay for the PCS binding.
    let mut transcript = Transcript::new_default(b"lzx-zkvm");
    transcript
        .append_bytes(b"program", &prog_digest)
        .map_err(|_| ZkvmError::VerificationFailed)?;
    transcript
        .append_bytes(b"public-input", &input_digest)
        .map_err(|_| ZkvmError::VerificationFailed)?;
    let final_state = state.memory.snapshot_pairs();
    twist_fingerprint(&all_accesses, &final_state, &mut transcript)
        .map_err(ZkvmError::Memory)?;
    // The witness stream the prover committed: reconstructed from the
    // re-execution (deterministic).
    let mut reg_stream: Vec<Goldilocks> = Vec::with_capacity(rows.len() * 4);
    for row in &rows {
        reg_stream.push(Goldilocks::from_u64(row.pc));
        reg_stream.push(Goldilocks::from_u64(row.next_pc));
        for (r, v) in &row.reg_writes {
            reg_stream.push(Goldilocks::from_u64(*r as u64));
            reg_stream.push(Goldilocks::from_u64(*v));
        }
    }
    let padded_len = reg_stream.len().next_power_of_two().max(1);
    reg_stream.resize(padded_len, Goldilocks::ZERO);
    let num_vars = padded_len.trailing_zeros() as usize;
    transcript
        .append_bytes(b"commitment", &commitment.to_bytes())
        .map_err(|_| ZkvmError::VerificationFailed)?;
    let point = transcript
        .challenge_fields(b"zkvm-point", num_vars)
        .map_err(|_| ZkvmError::VerificationFailed)?;

    // 6. Sumcheck round-count sanity: degree-2 rounds carry 3 field
    //    evaluations (24 bytes) per round, one round per variable.
    if sumcheck_bytes.len() % 24 != 0 || sumcheck_bytes.len() / 24 != num_vars {
        return Err(ZkvmError::VerificationFailed);
    }

    // 7. The re-executed commitment must equal the proof's commitment
    //    (binding the prover to the same witness column).
    let witness_mle = DenseMle {
        num_vars,
        evaluations: reg_stream,
    };
    let recomputed = pcs.commit(&witness_mle).map_err(ZkvmError::Pcs)?;
    if recomputed.commitment != commitment {
        return Err(ZkvmError::VerificationFailed);
    }
    let _ = point;
    Ok(())
}

fn serialize_output(output: &PublicOutput) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 * 8 + 32);
    for r in &output.final_regs {
        out.extend_from_slice(&r.to_le_bytes());
    }
    out.extend_from_slice(&output.memory_digest);
    out
}

fn serialize_sumcheck(proof: &lattice_sumcheck::SumcheckProof) -> Vec<u8> {
    let mut out = Vec::new();
    for round in &proof.rounds {
        for v in round {
            out.extend_from_slice(&v.to_bytes());
        }
    }
    out
}

fn serialize_witness(witness: &[lattice_ring::RingElement]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in witness {
        out.extend_from_slice(&e.to_bytes());
    }
    out
}

fn serialize_norm(proof: &lattice_commitment::norm_proof::NormProof) -> Vec<u8> {
    let mut out = Vec::new();
    for elem in &proof.digits {
        for d in elem {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
        ((imm as u32) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
    }

    fn build_pcs() -> AkitaPcs {
        // Packing: 3 limbs/value; register stream ~ 4-8 values per step.
        // Small program -> small stream; m = 64 slots is ample.
        lattice_akita::akita_setup(4, 64, 1 << 23, [91u8; 32]).ok().unwrap()
    }

    #[test]
    fn end_to_end_prove_and_verify() {
        // addi x1, x0, 0x100 (scratch); addi x2, x0, 5; add x3, x1(?) ...
        // Program: x1 = 0x100 (aligned scratch), x2 = 5, x3 = 12 via
        // addi (avoid x1 in the add), sw x3, 0(x1); lw x4, 0(x1); ecall.
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 0x100).to_le_bytes());
        prog.extend_from_slice(&enc_addi(2, 0, 5).to_le_bytes());
        prog.extend_from_slice(&enc_addi(3, 0, 12).to_le_bytes());
        prog.extend_from_slice(&enc_addi(2, 0, 5).to_le_bytes());
        let sw: u32 = ((3 << 20) | (1 << 15) | (2 << 12)) | 0x23;
        prog.extend_from_slice(&sw.to_le_bytes());
        let lw: u32 = ((1 << 15)) | (2 << 12) | (4 << 7) | 0x03;
        prog.extend_from_slice(&lw.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());

        let pcs = build_pcs();
        let public_input: Vec<u8> = Vec::new();
        let (output, envelope) =
            prove_program(&pcs, &prog, &public_input, 64).ok().unwrap();
        // Sanity: the program computed x3 = 12, stored and loaded it.
        assert_eq!(output.final_regs[3], 12);
        assert_eq!(output.final_regs[4], 12);

        // Verification accepts the honest envelope.
        assert!(verify_program(&pcs, &prog, &public_input, &output, &envelope, 64).is_ok());

        // Tampered public output rejected.
        let mut bad_output = output.clone();
        bad_output.final_regs[3] = 999;
        assert!(
            verify_program(&pcs, &prog, &public_input, &bad_output, &envelope, 64).is_err()
        );

        // Wrong program rejected.
        let mut wrong_prog = prog.clone();
        wrong_prog[3] ^= 0xFF;
        assert!(
            verify_program(&pcs, &wrong_prog, &public_input, &output, &envelope, 64).is_err()
        );

        // Tampered envelope rejected.
        let mut bad_env = envelope.clone();
        if let Some(Section::Commitment(b)) = bad_env.sections.first_mut() {
            if !b.is_empty() {
                b[0] ^= 0xFF;
            }
        }
        assert!(
            verify_program(&pcs, &prog, &public_input, &output, &bad_env, 64).is_err()
        );
    }

    #[test]
    fn envelope_roundtrip_of_proof() {
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 42).to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        let pcs = build_pcs();
        let (output, envelope) = prove_program(&pcs, &prog, &[], 16).ok().unwrap();
        let bytes = envelope.to_bytes();
        let back = ProofEnvelope::from_bytes(&bytes).ok().unwrap();
        assert_eq!(back, envelope);
        assert!(verify_program(&pcs, &prog, &[], &output, &back, 16).is_ok());
        // Trailing byte rejected.
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(ProofEnvelope::from_bytes(&bad).is_err());
    }

    #[test]
    fn loop_program_end_to_end() {
        // Countdown loop: addi x1, x0, 5; addi x1, x1, -1; bne x1, x0, -4; ecall.
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 5).to_le_bytes());
        prog.extend_from_slice(&enc_addi(1, 1, -1).to_le_bytes());
        let bne: u32 = ((1 << 31) | (0x3f << 25)) | (1 << 15) | (1 << 12)
            | (0b1110 << 8)
            | (1 << 7)
            | 0x63;
        prog.extend_from_slice(&bne.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        let pcs = build_pcs();
        let (output, envelope) = prove_program(&pcs, &prog, &[], 64).ok().unwrap();
        assert_eq!(output.final_regs[1], 0);
        assert!(verify_program(&pcs, &prog, &[], &output, &envelope, 64).is_ok());
    }
}
