//! End-to-end memory-argument proof tests.

use lattice_zkvm::memproof::{prove_memory_argument, verify_memory_argument};

fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
    ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
}

/// addi x1, x0, 12; addi x2, x0, 8; sw x1, 0(x2); lw x3, 0(x2); ecall
fn store_load_program() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&enc_addi(1, 0, 12).to_le_bytes());
    p.extend_from_slice(&enc_addi(2, 0, 8).to_le_bytes());
    let sw: u32 = (1u32 << 20) | (2 << 15) | (2 << 12) | 0x23;
    p.extend_from_slice(&sw.to_le_bytes());
    let lw: u32 = (2 << 15) | (2 << 12) | (3 << 7) | 0x03;
    p.extend_from_slice(&lw.to_le_bytes());
    p.extend_from_slice(&0x73u32.to_le_bytes());
    p
}

#[test]
fn end_to_end_memory_argument() {
    let prog = store_load_program();
    let (proof, final_regs) = prove_memory_argument(&prog, &[], 64, 4, 3).ok().unwrap();
    // The program computed x1 = 12, stored and loaded it into x3.
    assert_eq!(final_regs[3], 12);
    assert!(verify_memory_argument(&proof, &prog, &[]).is_ok());
    // Tampered final registers rejected (the register telescoping is
    // bound to the public statement).
    let mut bad = proof.clone();
    bad.statement.final_regs[3] = 999;
    assert!(verify_memory_argument(&bad, &prog, &[]).is_err());
    // Tampered final memory rejected.
    let mut bad2 = proof.clone();
    if let Some(w) = bad2.statement.final_memory.get_mut(1) {
        *w ^= 1;
    }
    assert!(verify_memory_argument(&bad2, &prog, &[]).is_err());
    // Tampered claim rejected (the bundle openings bind them).
    let mut bad3 = proof.clone();
    if let Some(c) = bad3.claims.first_mut() {
        c.value = c.value.add(&lattice_core::Goldilocks::from_u64(1));
    }
    assert!(verify_memory_argument(&bad3, &prog, &[]).is_err());
    // Wrong program rejected.
    let mut wrong = prog.clone();
    wrong[3] ^= 0xFF;
    assert!(verify_memory_argument(&proof, &wrong, &[]).is_err());
}

#[test]
fn loop_program_memory_argument() {
    // addi x1, x0, 5; addi x1, x1, -1; bne x1, x0, -4; ecall
    let mut prog = Vec::new();
    prog.extend_from_slice(&enc_addi(1, 0, 5).to_le_bytes());
    prog.extend_from_slice(&enc_addi(1, 1, -1).to_le_bytes());
    let bne: u32 = ((1u32 << 31) | (0x3f << 25))
        | (1 << 15)
        | (1 << 12)
        | (0b1110 << 8)
        | (1 << 7)
        | 0x63;
    prog.extend_from_slice(&bne.to_le_bytes());
    prog.extend_from_slice(&0x73u32.to_le_bytes());
    let (proof, final_regs) =
        prove_memory_argument(&prog, &[], 64, 3, 3).ok().unwrap();
    assert_eq!(final_regs[1], 0);
    assert!(verify_memory_argument(&proof, &prog, &[]).is_ok());
}
