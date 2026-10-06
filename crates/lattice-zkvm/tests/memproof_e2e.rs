//! End-to-end memory-argument proof tests.
//!
//! The instruction encodings below keep zero bitfields (e.g. `(0 << 12)`)
//! spelled out to document the RISC-V field layout.
#![allow(clippy::identity_op)]

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
    let bne: u32 =
        ((1u32 << 31) | (0x3f << 25)) | (1 << 15) | (1 << 12) | (0b1110 << 8) | (1 << 7) | 0x63;
    prog.extend_from_slice(&bne.to_le_bytes());
    prog.extend_from_slice(&0x73u32.to_le_bytes());
    let (proof, final_regs) = prove_memory_argument(&prog, &[], 64, 3, 3).ok().unwrap();
    assert_eq!(final_regs[1], 0);
    assert!(verify_memory_argument(&proof, &prog, &[]).is_ok());
}

#[test]
fn subword_byte_guest_memory_argument() {
    // The byte-guest ISA through the full memory argument: a sub-word
    // store/load pipeline (SB at an odd lane, SH straddling-checked,
    // LB/LBU/LH/LHU extractions) proven and verified WITHOUT
    // re-execution. The word-granular RAM argument is unchanged — the
    // sub-word accesses are read-modify-write splices on the containing
    // words, carried by the mem_old/mem_new tensors.
    //   addi x1, x0, 0x40        ; the base
    //   addi x2, x0, -1           ; 0xFFFF..FF
    //   sb  x2, 3(x1)             ; splice 0xFF at lane 3
    //   lb  x3, 3(x1)             ; = 0xFF sign-extended
    //   lbu x4, 3(x1)             ; = 255
    //   addi x5, x0, 0x234
    //   sh  x5, 10(x1)            ; splice halfword at lane 2
    //   lhu x6, 10(x1)            ; = 0x234
    //   lh  x7, 10(x1)            ; = 0x234 (positive)
    //   ld  x8, 8(x1)             ; the containing word (bytes 8..15)
    //   ecall
    let mut p = Vec::new();
    p.extend_from_slice(&enc_addi(1, 0, 0x40).to_le_bytes());
    p.extend_from_slice(&enc_addi(2, 0, -1).to_le_bytes());
    let sb: u32 = (2 << 20) | (1 << 15) | (0 << 12) | (3 << 7) | 0x23;
    p.extend_from_slice(&sb.to_le_bytes());
    let lb: u32 = (3 << 20) | (1 << 15) | (0 << 12) | (3 << 7) | 0x03;
    p.extend_from_slice(&lb.to_le_bytes());
    let lbu: u32 = (3 << 20) | (1 << 15) | (4 << 12) | (4 << 7) | 0x03;
    p.extend_from_slice(&lbu.to_le_bytes());
    p.extend_from_slice(&enc_addi(5, 0, 0x234).to_le_bytes());
    let sh: u32 = (5 << 20) | (1 << 15) | (1 << 12) | (10 << 7) | 0x23;
    p.extend_from_slice(&sh.to_le_bytes());
    let lhu: u32 = (10 << 20) | (1 << 15) | (5 << 12) | (6 << 7) | 0x03;
    p.extend_from_slice(&lhu.to_le_bytes());
    let lh: u32 = (10 << 20) | (1 << 15) | (1 << 12) | (7 << 7) | 0x03;
    p.extend_from_slice(&lh.to_le_bytes());
    let ld: u32 = (8 << 20) | (1 << 15) | (3 << 12) | (8 << 7) | 0x03;
    p.extend_from_slice(&ld.to_le_bytes());
    p.extend_from_slice(&0x73u32.to_le_bytes());
    let (proof, final_regs) = prove_memory_argument(&p, &[], 64, 5, 5).ok().unwrap();
    // The extraction semantics (the register writes).
    assert_eq!(final_regs[3], 0xFFFF_FFFF_FFFF_FFFF); // LB sign-extends
    assert_eq!(final_regs[4], 0xFF); // LBU zero-extends
    assert_eq!(final_regs[6], 0x234); // LHU
    assert_eq!(final_regs[7], 0x234); // LH (positive)
    // The containing word after the splices: the SB at 0x43 sets word
    // 8's lane 3; the SH at 0x4A splices 0x234 into word 9's lane 2; the
    // LD at 0x48 reads word 9 = 0x234 << 16.
    assert_eq!(final_regs[8], 0x0000_0000_0234_0000);
    // Verify (no re-execution) + tamper rejections.
    assert!(verify_memory_argument(&proof, &p, &[]).is_ok());
    let mut bad = proof.clone();
    bad.statement.final_regs[4] = 7;
    assert!(verify_memory_argument(&bad, &p, &[]).is_err());
    let mut bad2 = proof.clone();
    if let Some(w) = bad2.statement.final_memory.get_mut(9) {
        *w ^= 0x100;
    }
    assert!(verify_memory_argument(&bad2, &p, &[]).is_err());
}

#[test]
fn byte_workload_block_and_compact_proof() {
    // The byte-guest workload through BOTH next-generation proof modes:
    // the block-commit opening (ONE packed block commitment per bundle)
    // and the compact opening (per-column commitments) — the sub-word
    // ISA flows through the whole stack.
    use lattice_zkvm::memproof::{
        prove_memory_argument_block, prove_memory_argument_compact,
        verify_memory_argument_block, verify_memory_argument_compact,
    };
    // A byte-swap loop: for i in 0..8: lb t0, i(0x40); sb t0, 15-i(0x40).
    let mut p = Vec::new();
    p.extend_from_slice(&enc_addi(1, 0, 0x40).to_le_bytes());
    p.extend_from_slice(&enc_addi(3, 0, 0).to_le_bytes()); // i
    // Loop head at instruction index 2.
    let loop_head = 2usize;
    // addi x4, x3, -16 ; offset = i - 16 (byte address = 0x40 + i - 16)
    p.extend_from_slice(&enc_addi(4, 3, -16).to_le_bytes());
    // add x5, x1, x4
    let add5: u32 = (3 << 20) | (1 << 15) | (4 << 7) | 0x33;
    p.extend_from_slice(&add5.to_le_bytes());
    // lb x6, 0(x5)
    let lb: u32 = (5 << 15) | (0 << 12) | (6 << 7) | 0x03;
    p.extend_from_slice(&lb.to_le_bytes());
    // sb x6, 0(x5) — same address (in-place read-then-overwrite)
    let sb: u32 = (6 << 20) | (5 << 15) | (0 << 12) | (0 << 7) | 0x23;
    p.extend_from_slice(&sb.to_le_bytes());
    // addi x3, x3, 1
    p.extend_from_slice(&enc_addi(3, 3, 1).to_le_bytes());
    // addi x7, x0, 8; bge x3, x7, +2 (exit); j loop
    p.extend_from_slice(&enc_addi(7, 0, 8).to_le_bytes());
    // bge x3, x7, imm=+8 (skip the j)
    let bge: u32 = (7 << 20) | (3 << 15) | (5 << 12) | (4 << 8) | 0x63;
    p.extend_from_slice(&bge.to_le_bytes());
    // j loop_head (offset = (loop_head - cur) * 4)
    let cur = p.len() / 4;
    let off = (loop_head as i64 - cur as i64) * 4;
    let j: u32 = (((off as u32 >> 12) & 0xFF) << 12)
        | (((off as u32 >> 11) & 1) << 20)
        | (((off as u32 >> 1) & 0x3FF) << 21)
        | (((off as u32 >> 20) & 1) << 31)
        | 0x6F;
    p.extend_from_slice(&j.to_le_bytes());
    p.extend_from_slice(&0x73u32.to_le_bytes());
    // Block mode.
    let (bp, bf) = prove_memory_argument_block(&p, &[], 128, 5, 5)
        .unwrap_or_else(|e| panic!("block prove err: {:?}", e));
    assert!(verify_memory_argument_block(&bp, &p, &[]).is_ok());
    let _ = bf;
    // Compact mode (the byte workload through the per-column path too).
    let (cp, cf) = prove_memory_argument_compact(&p, &[], 128, 5, 5).ok().unwrap();
    assert!(verify_memory_argument_compact(&cp, &p, &[]).is_ok());
    let _ = cf;
    // Tampered block commitment rejected.
    let mut bad = bp.clone();
    if bad.bits_commitment.len() > 12 {
        bad.bits_commitment[12] ^= 0xFF;
    }
    assert!(verify_memory_argument_block(&bad, &p, &[]).is_err());
}
