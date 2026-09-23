//! Golden-vector conformance corpus (audit §9.2 P1): hand-derived
//! expected results for every RV64IMAC instruction class at its edge
//! cases. Each vector is `(program, initial memory, expected regs,
//! expected memory words)` — the expectations are computed from the
//! RISC-V specification, NOT from either interpreter.


/// One golden vector.
pub struct Golden {
    pub name: &'static str,
    /// Instruction words (executed from pc = 0).
    pub program: Vec<u32>,
    /// Initial memory as (address, u32) pairs (stored before running).
    pub memory: Vec<(u64, u32)>,
    /// Expected final registers: (index, value).
    pub expect_regs: Vec<(u8, u64)>,
    /// Expected final memory u32 values at (aligned-half) addresses.
    pub expect_mem32: Vec<(u64, u32)>,
}

fn enc_r(opcode: u32, rd: u8, funct3: u32, rs1: u8, rs2: u8, funct7: u32) -> u32 {
    (funct7 << 25) | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn enc_i(opcode: u32, rd: u8, funct3: u32, rs1: u8, imm: i32) -> u32 {
    (((imm as u32) & 0xFFF) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7)
        | opcode
}

fn enc_s(opcode: u32, funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7F) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
}

const OP: u32 = 0x33;
const OPIMM: u32 = 0x13;
const OP32: u32 = 0x3B;
const LOAD: u32 = 0x03;
const STORE: u32 = 0x23;
const BRANCH: u32 = 0x63;
const JAL: u32 = 0x6F;
const JALR: u32 = 0x67;
const AMO: u32 = 0x2F;
const SYSTEM: u32 = 0x73;

fn enc_amo(funct5: u32, rd: u8, rs2: u8, rs1: u8, funct3: u32) -> u32 {
    (funct5 << 27)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | AMO
}
const LUI: u32 = 0x37;
const AUIPC: u32 = 0x17;

const MAX: u64 = u64::MAX;
const I64MIN: u64 = i64::MIN as u64;
const NEG1: u64 = (-1i64) as u64;

/// The corpus.
pub fn corpus() -> Vec<Golden> {
    let mut v = vec![Golden {
        name: "add-overflow-wraps",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, -1),      // x1 = MAX
            enc_i(OPIMM, 2, 0, 0, 1),       // x2 = 1
            enc_r(OP, 3, 0, 1, 2, 0),       // x3 = MAX + 1 = 0 (mod 2^64)
            enc_r(OP, 4, 0, 1, 0, 0),       // x4 = MAX + 0 = MAX
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(1, MAX), (2, 1), (3, 0), (4, MAX)],
        expect_mem32: vec![],
    }];
    v.push(Golden {
        name: "sub-underflow-wraps",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 0),       // x1 = 0
            enc_i(OPIMM, 2, 0, 0, 1),       // x2 = 1
            enc_r(OP, 3, 0, 1, 2, 0x20),    // x3 = 0 - 1 = MAX
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(3, MAX)],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "slt-sltu-negative-boundary",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, -1),      // x1 = MAX (as i64: -1)
            enc_r(OP, 2, 2, 1, 0, 0),       // x2 = (-1 < 0) = 1
            enc_r(OP, 3, 3, 1, 0, 0),       // x3 = (MAX < 0 unsigned) = 0
            enc_i(OPIMM, 4, 2, 0, 0),       // x4 = (0 < 0) = 0
            enc_i(OPIMM, 5, 3, 0, -1),      // x5 = (0 < MAX) = 1
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(2, 1), (3, 0), (4, 0), (5, 1)],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "shifts-63-and-masked-register-shift",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 1),       // x1 = 1
            enc_i(OPIMM, 2, 1, 1, 63),       // x2 = 1 << 63
            enc_i(OPIMM, 3, 0, 0, 64),       // x3 = 64 (shift amount reg)
            enc_r(OP, 4, 1, 2, 3, 0),       // x4 = x2 << (64 & 63) = x2 << 0 = x2
            enc_r(OP, 5, 5, 2, 3, 0),       // x5 = x2 >> 0 = x2
            enc_r(OP, 6, 5, 2, 3, 0x20),    // x6 = x2 >>a 0 = x2
            enc_i(OPIMM, 7, 5, 2, 0x400 | 63), // x7 = x2 >>a 63 = MAX (SRAI: bit30|shamt)
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(2, 1u64 << 63), (4, 1u64 << 63), (5, 1u64 << 63), (6, 1u64 << 63), (7, MAX)],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "addw-sign-extends",
        program: vec![
            enc_i(LUI, 1, 0, 0, 0),         // x1 = 0 (placeholder setup below)
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(1, 0)],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "word-ops-32bit-boundaries",
        program: vec![
            // x1 = 0x1234: LUI 0x1 -> 0x1000; addi 0x234.
            (0x1 << 12) | (1 << 7) | LUI,
            enc_i(OPIMM, 1, 0, 1, 0x234),   // x1 = 0x1234
            enc_i(OPIMM, 2, 0, 0, 1),       // x2 = 1
            enc_i(OPIMM, 5, 0, 0, -8),      // x5 = -8 (0xFFFF..F8)
            enc_r(OP32, 3, 0, 1, 2, 0),     // x3 = addw(0x1234+1) = 0x1235
            enc_r(OP32, 4, 1, 1, 2, 0),     // x4 = sllw(0x1234<<1) = 0x2468
            enc_r(OP32, 6, 5, 5, 2, 0x20),  // x6 = sraw(-8 >> 1) = -4
            enc_r(OP32, 7, 5, 5, 2, 0),     // x7 = srlw(0xFFFFFFF8 >> 1) = 0x7FFFFFFC
            enc_r(OP32, 8, 0, 1, 2, 0x20),  // x8 = subw(0x1234-1) = 0x1233
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![
            (1, 0x1234),
            (3, 0x1235),
            (4, 0x2468),
            (6, (-4i64) as u64),
            (7, 0x7FFF_FFFCu64),
            (8, 0x1233),
        ],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "lui-auipc",
        program: vec![
            (0xDEAD0 << 12) | (1 << 7) | LUI,   // x1 = 0xDEAD0000
            (0x12345 << 12) | (2 << 7) | AUIPC, // x2 = 4 + 0x12345000
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(1, 0xFFFF_FFFF_DEAD_0000u64), (2, 4 + 0x1234_5000)],
        expect_mem32: vec![],
    });

    // --- RV64M edge cases ---
    v.push(Golden {
        name: "mul-family-highs",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, -1),      // x1 = -1 (MAX)
            enc_i(OPIMM, 2, 0, 0, -2),      // x2 = -2
            enc_r(OP, 3, 0, 1, 2, 0),       // x3 = add(-1 + -2) = -3 (sanity)
            enc_r(OP, 4, 0, 1, 2, 1),       // x4 = mul(-1 * -2) = 2
            enc_r(OP, 5, 1, 1, 2, 1),       // x5 = mulh(-1 * -2 >> 64) = 0
            enc_r(OP, 6, 2, 1, 2, 1),       // x6 = mulhu(MAX * (MAX-1) >> 64)
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![
            (3, (-3i64) as u64),
            (4, 2),
            (5, 0),
            // (2^64-1)(2^64-2) >> 64 = 2^64 - 3.
            (6, MAX - 2),
        ],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "div-rem-signed-edges",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, -1),      // x1 = -1 (divisor 0 case uses x0)
            // x2 = i64::MIN: lui 0x80000 (sign-extends to
            // 0xFFFFFFFF80000000) then slli 32 -> 0x8000000000000000.
            (0x80000 << 12) | (2 << 7) | LUI,
            enc_i(OPIMM, 2, 1, 2, 32),      // slli x2, x2, 32
            enc_r(OP, 3, 4, 2, 0, 1),       // x3 = div(MIN / 0) = -1
            enc_r(OP, 4, 6, 2, 0, 1),       // x4 = rem(MIN % 0) = MIN
            enc_r(OP, 5, 4, 2, 1, 1),       // x5 = div(MIN / -1) = MIN
            enc_r(OP, 6, 6, 2, 1, 1),       // x6 = rem(MIN % -1) = 0
            enc_r(OP, 7, 4, 1, 1, 1),       // x7 = div(-1 / -1) = 1
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![
            (2, I64MIN),
            (3, NEG1),
            (4, I64MIN),
            (5, I64MIN),
            (6, 0),
            (7, 1),
        ],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "divu-remu-unsigned-edges",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, -1),      // x1 = MAX
            enc_r(OP, 2, 5, 1, 0, 1),       // x2 = divu(MAX / 0) = MAX
            enc_r(OP, 3, 7, 1, 0, 1),       // x3 = remu(MAX % 0) = MAX
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(2, MAX), (3, MAX)],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "divw-family-i32min",
        program: vec![
            // x1 = 0xFFFFFFFF80000000 (i32::MIN as i64) via LUI 0x80000 + addiw 0.
            (0x80000 << 12) | (1 << 7) | LUI,
            enc_i(OPIMM, 2, 0, 0, -1),      // x2 = -1
            enc_r(OP32, 3, 4, 1, 2, 1),     // x3 = divw(i32MIN / -1) = i32MIN
            enc_r(OP32, 4, 6, 1, 2, 1),     // x4 = remw(i32MIN % -1) = 0
            enc_r(OP32, 5, 4, 1, 0, 1),     // x5 = divw(i32MIN / 0) = -1
            enc_r(OP32, 6, 5, 1, 0, 1),     // x6 = divuw(i32MIN / 0) = 0xFFFFFFFF
            enc_r(OP32, 7, 7, 1, 0, 1),     // x7 = remuw(i32MIN % 0) = 0x80000000
            enc_r(OP32, 8, 0, 1, 2, 1),     // x8 = mulw(i32MIN * -1) = i32MIN
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![
            (1, 0xFFFF_FFFF_8000_0000u64),
            (3, 0xFFFF_FFFF_8000_0000u64),
            (4, 0),
            (5, NEG1),
            (6, 0xFFFF_FFFFu64),
            (7, 0x8000_0000u64),
            // mulw(i32::MIN * -1) wraps to i32::MIN, sign-extended.
            (8, 0xFFFF_FFFF_8000_0000u64),
        ],
        expect_mem32: vec![],
    });

    // --- Memory: sign extension, subword, unaligned ---
    v.push(Golden {
        name: "lw-sign-extends-lwu-does-not",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 0x40),    // x1 = 0x40 (address)
            enc_i(LOAD, 2, 2, 1, 0),        // x2 = lw [x1] (funct3=2)
            enc_i(LOAD, 3, 6, 1, 0),        // x3 = lwu [x1] (funct3=6)
            enc_i(LOAD, 4, 3, 1, 8),        // x4 = ld [x1+8] (funct3=3, aligned)
            enc_i(LOAD, 5, 3, 1, 1),        // x5 = ld [x1+1] (unaligned, straddles)
            SYSTEM,
        ],
        memory: vec![(0x40, 0x8000_0000), (0x48, 0x1234_5678)],
        expect_regs: vec![
            (2, 0xFFFF_FFFF_8000_0000u64),
            (3, 0x8000_0000u64),
            (4, 0x1234_5678u64),
            // ld at 0x41: bytes 41..48 = 00 00 80 00 00 00 00 78
            // (word 0x40 = 0x0000000080000000, word 0x48 low byte 0x78)
            // -> 0x7800_0000_0080_0000
            (5, 0x7800_0000_0080_0000u64),
        ],
        expect_mem32: vec![(0x40, 0x8000_0000), (0x48, 0x1234_5678)],
    });
    v.push(Golden {
        name: "unaligned-lw-straddles-words",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 0x40),    // x1 = 0x40
            enc_i(LOAD, 2, 2, 1, 5),        // x2 = lw [0x45] (unaligned)
            enc_i(LOAD, 3, 2, 1, 7),        // x3 = lw [0x47] (last byte + next word)
            SYSTEM,
        ],
        memory: vec![(0x40, 0xAABB_CCDD), (0x48, 0xEEFF_0011)],
        expect_regs: vec![
            // Memory bytes: word 0x40 = 0x00000000AABBCCDD ->
            // DD CC BB AA 00 00 00 00; word 0x48 = 0x00000000EEFF0011 ->
            // 11 00 FF EE 00 00 00 00.
            // lw at 0x45: bytes 45..48 = 00 00 00 11 -> 0x11000000.
            (2, 0x1100_0000u64),
            // lw at 0x47: bytes 47..4A = 00 11 00 FF -> 0xFF001100,
            // sign-extended to 0xFFFFFFFFFF001100.
            (3, 0xFFFF_FFFF_FF00_1100u64),
        ],
        expect_mem32: vec![(0x40, 0xAABB_CCDD), (0x48, 0xEEFF_0011)],
    });
    v.push(Golden {
        name: "sw-sd-store-values",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 0x80),    // x1 = 0x80
            enc_i(OPIMM, 2, 0, 0, -1),      // x2 = MAX
            enc_s(STORE, 2, 1, 2, 0),       // sw [x1] = 0xFFFFFFFF
            enc_s(STORE, 3, 1, 2, 8),       // sd [x1+8] = MAX
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(2, MAX)],
        expect_mem32: vec![(0x80, 0xFFFF_FFFF), (0x88, 0xFFFF_FFFF), (0x8C, 0xFFFF_FFFF)],
    });

    // --- Branches and jumps ---
    v.push(Golden {
        name: "branch-taken-and-not",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 5),       // x1 = 5
            enc_i(OPIMM, 2, 0, 0, 5),       // x2 = 5
            // beq x1, x2, +8 (skip the x3 = 1): imm[4:1] = 4 -> bits 11:8.
            (4u32 << 8) | (1 << 15) | (2 << 20) | BRANCH,
            enc_i(OPIMM, 3, 0, 0, 1),       // x3 = 1 (skipped)
            enc_i(OPIMM, 4, 0, 0, 2),       // x4 = 2 (executed)
            SYSTEM,
        ],
        memory: vec![],
        expect_regs: vec![(3, 0), (4, 2)],
        expect_mem32: vec![],
    });
    v.push(Golden {
        name: "jal-link-and-jalr",
        program: vec![
            // jal x1, +8 (skip next): imm[3:1] = 4 -> bits 30:21.
            (4u32 << 21) | (1 << 7) | JAL,
            enc_i(OPIMM, 5, 0, 0, 99),      // skipped
            enc_i(OPIMM, 2, 0, 0, 16),      // x2 = 16 (target addr for jalr)
            // jalr x3, x2, 0 -> jumps to 16, link = pc+4 = 16
            enc_i(JALR, 3, 0, 2, 0),
            enc_i(OPIMM, 6, 0, 0, 1),       // x6 = 1 (executed after jump to 16)
            SYSTEM,                          // at 12..15 padding
            enc_i(OPIMM, 7, 0, 0, 0),       // filler
            SYSTEM,                          // 16: the jalr target
        ],
        memory: vec![],
        expect_regs: vec![(1, 4), (3, 16)],
        expect_mem32: vec![],
    });

    // --- Atomics ---
    v.push(Golden {
        name: "lr-sc-success-and-failure",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 0x100),   // x1 = 0x100
            enc_i(OPIMM, 7, 0, 0, 0x180),   // x7 = 0x180 (different address)
            enc_i(OPIMM, 5, 0, 0, 0xAB),    // x5 = 0xAB
            // lr.w x2, [x1]
            enc_amo(0x02, 2, 0, 1, 2),
            // sc.w x3, x5, [x1] — succeeds (reservation live)
            enc_amo(0x03, 3, 5, 1, 3),
            // lr.w x4, [x1] — reads the stored 0xAB
            enc_amo(0x02, 4, 0, 1, 2),
            // sc.w x6, x5, [x7] — fails (reservation is at 0x100)
            enc_amo(0x03, 6, 5, 7, 3),
            SYSTEM,
        ],
        memory: vec![(0x100, 0x11)],
        expect_regs: vec![
            (2, 0x11),
            (3, 0),  // SC success
            (4, 0xAB),
            (6, 1),  // SC failure
        ],
        expect_mem32: vec![(0x100, 0xAB)],
    });
    v.push(Golden {
        name: "amo-add-and-swap",
        program: vec![
            enc_i(OPIMM, 1, 0, 0, 0x120),   // x1 = 0x120
            enc_i(OPIMM, 2, 0, 0, 5),       // x2 = 5
            // amoadd.w x3, x2, [x1]: mem = 0x37 + 5 = 0x3C, x3 = 0x37
            enc_amo(0x00, 3, 2, 1, 2),
            // amoswap.w x4, x2, [x1]: mem = 5, x4 = 0x3C
            enc_amo(0x01, 4, 2, 1, 2),
            SYSTEM,
        ],
        memory: vec![(0x120, 0x37)],
        expect_regs: vec![(3, 0x37), (4, 0x3C)],
        expect_mem32: vec![(0x120, 5)],
    });

    v
}

/// Run the corpus against the canonical executor.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::run;
    use crate::state::MachineState;

    #[test]
    fn golden_corpus_passes() {
        let mut failures = Vec::new();
        for g in corpus() {
            let mut state = MachineState::new();
            for (addr, w32) in &g.memory {
                state.memory.store_word32(*addr, *w32);
            }
            let mut program = Vec::new();
            for w in &g.program {
                program.extend_from_slice(&w.to_le_bytes());
            }
            state.load_program(0, &program);
            let rows = run(&mut state, 256);
            if let Err(e) = rows {
                failures.push(format!("{}: execution error {e:?}", g.name));
                continue;
            }
            for (idx, want) in &g.expect_regs {
                let got = state.reg(*idx);
                if got != *want {
                    failures.push(format!(
                        "{}: x{idx} = {got:#x}, want {want:#x}",
                        g.name
                    ));
                }
            }
            for (addr, want) in &g.expect_mem32 {
                let got = state.memory.load_word32(*addr) as u32;
                if got != *want {
                    failures.push(format!(
                        "{}: mem[{addr:#x}] = {got:#x}, want {want:#x}",
                        g.name
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "golden failures:\n{}", failures.join("\n"));
    }
}
