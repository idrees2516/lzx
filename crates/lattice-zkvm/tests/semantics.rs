//! End-to-end tests for the instruction-semantics layer: prove over a
//! real execution, verify with NO re-execution, and tamper rejection
//! (corrupted claim / leg / commitment / final registers).

use lattice_core::Goldilocks;
use lattice_zkvm::semantics::{
    prove_instruction_semantics, verify_instruction_semantics, SemanticsProof,
};

fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
}

fn enc_r(f7: u32, rs2: u8, rs1: u8, f3: u32, rd: u8, op: u32) -> u32 {
    (f7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (f3 << 12) | ((rd as u32) << 7) | op
}

fn enc_shift_imm(f6: u32, shamt: u8, rs1: u8, f3: u32, rd: u8, op: u32) -> u32 {
    (f6 << 26)
        | ((shamt as u32) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | ((rd as u32) << 7)
        | op
}

/// addi/sw/lw/ecall — the classic memory program.
fn mem_program() -> Vec<u8> {
    let words = [
        0x0080_0093u32,
        0x0070_0113,
        0x0020_81b3,
        0x0030_b023,
        0x0000_b203,
        0x0000_0073,
    ];
    let mut v = Vec::new();
    for w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

/// A full-semantics program: arithmetic, shifts, MUL, DIV, branches,
/// memory, bitwise — every family in one trace.
fn full_program() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&enc_addi(1, 0, -7).to_le_bytes());
    p.extend_from_slice(&enc_addi(2, 0, 3).to_le_bytes());
    p.extend_from_slice(&enc_addi(10, 0, 40).to_le_bytes());
    p.extend_from_slice(&enc_addi(11, 0, 0x7FFF).to_le_bytes());
    p.extend_from_slice(&enc_r(1, 2, 1, 0, 3, 0x33).to_le_bytes()); // mul
    p.extend_from_slice(&enc_r(1, 2, 1, 1, 4, 0x33).to_le_bytes()); // mulh
    p.extend_from_slice(&enc_r(1, 2, 1, 3, 5, 0x33).to_le_bytes()); // mulhu
    p.extend_from_slice(&enc_r(1, 2, 1, 4, 6, 0x33).to_le_bytes()); // div
    p.extend_from_slice(&enc_r(1, 2, 1, 6, 7, 0x33).to_le_bytes()); // rem
    p.extend_from_slice(&enc_r(1, 2, 1, 5, 8, 0x33).to_le_bytes()); // divu
    p.extend_from_slice(&enc_r(0, 10, 1, 1, 12, 0x33).to_le_bytes()); // sll
    p.extend_from_slice(&enc_r(0, 10, 1, 5, 13, 0x33).to_le_bytes()); // srl
    p.extend_from_slice(&enc_r(0x20, 10, 1, 5, 14, 0x33).to_le_bytes()); // sra
    p.extend_from_slice(&enc_shift_imm(0, 5, 1, 1, 15, 0x13).to_le_bytes()); // slli
    p.extend_from_slice(&enc_shift_imm(0x10, 3, 1, 5, 16, 0x13).to_le_bytes()); // srai
    p.extend_from_slice(&enc_r(0, 11, 3, 7, 17, 0x33).to_le_bytes()); // and
    p.extend_from_slice(&enc_r(0x20, 2, 1, 0, 18, 0x33).to_le_bytes()); // sub
                                                                        // Store + load through memory.
    p.extend_from_slice(&enc_addi(20, 0, 64).to_le_bytes());
    let sd: u32 = (18u32 << 20) | (20 << 15) | (3 << 12) | 0x23;
    p.extend_from_slice(&sd.to_le_bytes());
    let ld: u32 = (20 << 15) | (3 << 12) | (21 << 7) | 0x03;
    p.extend_from_slice(&ld.to_le_bytes());
    p.extend_from_slice(&0x73u32.to_le_bytes());
    p
}

#[test]
fn semantics_mem_program_roundtrip() {
    let prog = mem_program();
    let input = 42u64.to_le_bytes().to_vec();
    let (proof, regs) = prove_instruction_semantics(&prog, &input, 64, 6, 4)
        .ok()
        .unwrap();
    assert_eq!(regs[4], 15);
    for (i, l) in proof.legs.iter().enumerate() {
        println!(
            "leg {i}: {} round0 len = {}",
            l.name,
            l.sc.rounds.first().map(|r| r.len()).unwrap_or(0)
        );
    }
    println!("claims: {}", proof.claims.len());
    for (i, c) in proof.claims.iter().enumerate().take(420).skip(378) {
        println!(
            "claim {i}: {:?} pt={:?}",
            c.factor,
            c.point.iter().map(|g| g.0).collect::<Vec<_>>()
        );
    }
    match verify_instruction_semantics(&proof, &prog, &input) {
        Ok(_) => {}
        Err(e) => panic!("verify: {e:?}"),
    }
}

#[test]
fn semantics_full_program_roundtrip() {
    let prog = full_program();
    let input = [];
    let (proof, _regs) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    match verify_instruction_semantics(&proof, &prog, &input) {
        Ok(_) => {}
        Err(e) => panic!("verify: {e:?}"),
    }
}

#[test]
fn semantics_tampered_claim_rejected() {
    let prog = full_program();
    let input = [];
    let (mut proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    if let Some(c) = proof.claims.first_mut() {
        c.value = c.value.add(&lattice_core::Goldilocks::ONE);
    }
    assert!(verify_instruction_semantics(&proof, &prog, &input).is_err());
}

#[test]
fn semantics_tampered_leg_rejected() {
    let prog = full_program();
    let input = [];
    let (mut proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    if let Some(leg) = proof.legs.first_mut() {
        if let Some(r0) = leg.sc.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&lattice_core::Goldilocks::ONE);
            }
        }
    }
    assert!(verify_instruction_semantics(&proof, &prog, &input).is_err());
}

#[test]
fn semantics_tampered_commitment_rejected() {
    let prog = full_program();
    let input = [];
    let (mut proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    if let Some(b) = proof.bits_commitment.first_mut() {
        *b ^= 0x01;
    }
    assert!(verify_instruction_semantics(&proof, &prog, &input).is_err());
}

#[test]
fn semantics_wrong_final_regs_rejected() {
    let prog = full_program();
    let input = [];
    let (mut proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    // The statement's final registers are public — tampering them
    // breaks the halt/ctrl family identities against the trace.
    proof.statement.final_regs[3] = proof.statement.final_regs[3].wrapping_add(1);
    let res = verify_instruction_semantics(&proof, &prog, &input);
    // Either the statement comparison or the family identities reject.
    assert!(res.is_err());
}

#[test]
fn semantics_wrong_program_rejected() {
    let prog = full_program();
    let input = [];
    let (proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    let mut other = prog.clone();
    other[0] ^= 0x01;
    assert!(verify_instruction_semantics(&proof, &other, &input).is_err());
}

#[test]
fn semantics_dropped_leg_rejected() {
    let prog = full_program();
    let input = [];
    let (mut proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    proof.legs.truncate(proof.legs.len() - 1);
    assert!(verify_instruction_semantics(&proof, &prog, &input).is_err());
}

#[test]
fn semantics_proof_shape() {
    let prog = full_program();
    let input = [];
    let (proof, _) = prove_instruction_semantics(&prog, &input, 128, 6, 5)
        .ok()
        .unwrap();
    let SemanticsProof { legs, claims, .. } = &proof;
    assert_eq!(legs.len(), 15);
    assert!(!claims.is_empty());
    // Every claim's factor is one of the committed families.
    for c in claims {
        assert!(
            c.factor.in_bits_bundle()
                || matches!(c.factor, lattice_zkvm::ledger::Factor::ValCol { .. })
        );
    }
}

// ---------------------------------------------------------------------------
// The P1 sub-word constraint polynomials: LB/LBU/LH/LHU/SB/SH through the
// full semantics layer (the byte-guest ISA's statement-path coverage).
// ---------------------------------------------------------------------------

fn enc_load(f3: u32, rd: u8, rs1: u8, imm: i64) -> u32 {
    ((imm as u32 & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | ((rd as u32) << 7)
        | 0x03
}

fn enc_store(f3: u32, rs1: u8, rs2: u8, imm: i64) -> u32 {
    // S-type: imm[11:5] at bits 25..31, imm[4:0] at bits 7..11.
    let i = imm as u32 & 0xFFF;
    ((i >> 5) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | ((i & 0x1F) << 7)
        | 0x23
}

/// The sub-word program: byte stores at even AND odd offsets (the mux
/// over all 8 byte positions), sign/zero extensions, half stores at two
/// alignments, and a sign-extended half load (0x8000 -> -32768).
fn subword_program() -> Vec<u8> {
    let words = [
        enc_addi(1, 0, 64),        // x1 = 64 (base)
        enc_addi(2, 0, -2),        // x2 = 0xFFFE
        enc_store(0, 1, 2, 0),     // sb x2, 0(x1)   -> 0xFE at 64
        enc_store(0, 1, 2, 5),     // sb x2, 5(x1)   -> 0xFE at 69 (odd)
        enc_load(4, 3, 1, 0),      // lbu x3, 0(x1)  -> 254
        enc_load(0, 4, 1, 0),      // lb  x4, 0(x1)  -> -2 (sext)
        enc_load(4, 5, 1, 5),      // lbu x5, 5(x1)  -> 254
        enc_load(0, 6, 1, 5),      // lb  x6, 5(x1)  -> -2 (sext)
        enc_addi(7, 0, 1),         // x7 = 1
        enc_shift_imm(0, 15, 7, 1, 7, 0x13), // slli x7, x7, 15 -> 0x8000
        enc_store(1, 1, 7, 2),     // sh x7, 2(x1)   -> half 0x8000 at 66
        enc_load(5, 8, 1, 2),      // lhu x8, 2(x1)  -> 32768
        enc_load(1, 9, 1, 2),      // lh  x9, 2(x1)  -> -32768 (sext)
        enc_store(1, 1, 7, 4),     // sh x7, 4(x1)   -> half at 68
        enc_load(1, 10, 1, 4),     // lh  x10, 4(x1) -> -32768
        0x73u32,                   // ecall
    ];
    let mut v = Vec::new();
    for w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

#[test]
fn semantics_subword_program_roundtrip() {
    let prog = subword_program();
    let input: Vec<u8> = Vec::new();
    let (proof, regs) = prove_instruction_semantics(&prog, &input, 64, 6, 4)
        .ok()
        .unwrap();
    // The executed semantics: lbu 254, lb -2, lhu 32768, lh -32768 —
    // the registers carry the two's-complement lifts as u64.
    assert_eq!(regs[3], 254);
    assert_eq!(regs[4], (-2i64) as u64);
    assert_eq!(regs[5], 254);
    assert_eq!(regs[6], (-2i64) as u64);
    assert_eq!(regs[8], 32768);
    assert_eq!(regs[9], (-32768i64) as u64);
    assert_eq!(regs[10], (-32768i64) as u64);
    // Verify with NO re-execution.
    assert!(verify_instruction_semantics(&proof, &prog, &input).is_ok());
}

#[test]
fn semantics_subword_tampered_claim_rejected() {
    let prog = subword_program();
    let input: Vec<u8> = Vec::new();
    let (mut proof, _) = prove_instruction_semantics(&prog, &input, 64, 6, 4)
        .ok()
        .unwrap();
    // Corrupt the first claim's value — the route family's P1 muxes
    // (among others) bind the committed columns; the tamper must fail.
    proof.claims[0].value = proof.claims[0].value.add(&Goldilocks::ONE);
    assert!(verify_instruction_semantics(&proof, &prog, &input).is_err());
}

#[test]
fn semantics_subword_wrong_program_rejected() {
    let prog = subword_program();
    let input: Vec<u8> = Vec::new();
    let (proof, _) = prove_instruction_semantics(&prog, &input, 64, 6, 4)
        .ok()
        .unwrap();
    // Change the first SB (instr 2, even offset 0) into an SH: the
    // store-merge polynomials differ (the half muxes vs the byte mux)
    // and the merged-word claims no longer match -> rejected.
    let mut other = prog.clone();
    other[(2 * 4) + 1] |= 0x10; // funct3 0 -> 1 at instr 2 (sb -> sh)
    assert!(verify_instruction_semantics(&proof, &other, &input).is_err());
}
