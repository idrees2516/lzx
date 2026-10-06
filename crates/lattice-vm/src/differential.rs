//! Randomized differential testing: canonical executor vs the
//! independent reference interpreter (audit §9.2 P1).
//!
//! Random programs over the full supported instruction surface (with
//! sandboxed memory addressing), random initial registers, and random
//! initial memory — both interpreters must agree on the final (regs,
//! memory bytes, pc, halted) state. Any divergence is a semantics bug
//! in one of the two implementations; the golden corpus decides which.

#![cfg(test)]

use crate::exec::run;
use crate::reference::RefMachine;
use crate::state::MachineState;

/// Deterministic xorshift PRNG.
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Prng { state: seed | 1 }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn range(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn enc_r(funct7: u32, rs2: u8, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    (funct7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

fn enc_i(imm: u32, rs1: u8, funct3: u32, rd: u8, opcode: u32) -> u32 {
    ((imm & 0xFFF) << 20) | ((rs1 as u32) << 15) | (funct3 << 12) | ((rd as u32) << 7) | opcode
}

fn enc_s(imm: u32, rs2: u8, rs1: u8, funct3: u32, opcode: u32) -> u32 {
    (((imm >> 5) & 0x7F) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
}

/// Generate a random straight-line program over the supported surface.
/// Memory accesses are sandboxed: base registers x1..x4 hold sandbox
/// addresses; offsets are small.
fn gen_program(prng: &mut Prng, n: usize) -> Vec<u32> {
    const OP: u32 = 0x33;
    const OPIMM: u32 = 0x13;
    const OP32: u32 = 0x3B;
    const OPIMM32: u32 = 0x1B;
    const LOAD: u32 = 0x03;
    const STORE: u32 = 0x23;
    const LUI: u32 = 0x37;
    const AUIPC: u32 = 0x17;
    const AMO: u32 = 0x2F;

    let mut words = Vec::with_capacity(n + 6);
    // Sandbox setup: x1 = 0x200, x2 = 0x240, x3 = 0x280, x4 = 0x2C0.
    for (i, base) in [0x200u32, 0x240, 0x280, 0x2C0].iter().enumerate() {
        words.push(enc_i(*base, 0, 0, (1 + i) as u8, OPIMM));
    }
    // Random immediate values into x5..x7.
    for rd in 5u8..8 {
        let imm = (prng.next() & 0xFFF) as u32;
        words.push(enc_i(imm, 0, 0, rd, OPIMM));
    }
    for _ in 0..n {
        let r8 = |p: &mut Prng| (p.range(32) as u8) & 0x1f;
        let choice = prng.range(14);
        let word = match choice {
            0 => {
                // OP register-register (incl. M).
                let funct3 = (prng.range(8) as u32) & 7;
                let funct7 = match funct3 {
                    0 => [0u32, 0x20, 1][prng.range(3) as usize],
                    5 => [0u32, 0x20, 1][prng.range(3) as usize],
                    1 | 2 | 3 | 4 | 6 | 7 => [0u32, 1][prng.range(2) as usize],
                    _ => 0,
                };
                enc_r(funct7, r8(prng), r8(prng), funct3, r8(prng), OP)
            }
            1 => {
                // OP-IMM.
                let funct3 = (prng.range(7) as u32) & 7;
                if funct3 == 1 || funct3 == 5 {
                    // Shifts: 6-bit shamt, correct funct6.
                    let shamt = (prng.range(64) as u32) & 0x3f;
                    let funct6 = if funct3 == 5 && prng.range(2) == 1 {
                        0x10u32
                    } else {
                        0
                    };
                    (funct6 << 26)
                        | (shamt << 20)
                        | ((r8(prng) as u32) << 15)
                        | (funct3 << 12)
                        | ((r8(prng) as u32) << 7)
                        | OPIMM
                } else {
                    enc_i(prng.range(0x1000) as u32, r8(prng), funct3, r8(prng), OPIMM)
                }
            }
            2 => {
                // OP-32.
                let funct3 = (prng.range(8) as u32) & 7;
                let funct7 = match funct3 {
                    0 => [0u32, 0x20, 1][prng.range(3) as usize],
                    5 => [0u32, 0x20, 1][prng.range(3) as usize],
                    _ => [0u32, 1][prng.range(2) as usize],
                };
                enc_r(funct7, r8(prng), r8(prng), funct3, r8(prng), OP32)
            }
            3 => {
                // OP-IMM-32: ADDIW or W-shift-immediates.
                let pick = prng.range(3);
                match pick {
                    0 => enc_i(prng.range(0x1000) as u32, r8(prng), 0, r8(prng), OPIMM32),
                    1 => {
                        let shamt = (prng.range(32) as u32) & 0x1f;
                        (shamt << 20)
                            | ((r8(prng) as u32) << 15)
                            | (1 << 12)
                            | ((r8(prng) as u32) << 7)
                            | OPIMM32
                    }
                    _ => {
                        let shamt = (prng.range(32) as u32) & 0x1f;
                        let funct7 = if prng.range(2) == 1 { 0x20u32 } else { 0 };
                        (funct7 << 25)
                            | (shamt << 20)
                            | ((r8(prng) as u32) << 15)
                            | (5 << 12)
                            | ((r8(prng) as u32) << 7)
                            | OPIMM32
                    }
                }
            }
            4 => {
                // LOAD: the full byte-guest surface — LB/LH/LW/LD/LBU/LHU/
                // LWU from sandbox bases, at every legal alignment
                // (sub-word loads are legal at ANY byte address, including
                // straddling halfwords).
                let funct3 = [0u32, 1, 2, 3, 4, 5, 6][prng.range(7) as usize];
                let rs1 = (1u8 + prng.range(4) as u8) & 0x1f;
                let imm = match funct3 {
                    2 | 6 => prng.range(0x60) as u32 & !3, // LW/LWU: 4-aligned
                    3 => prng.range(0x60) as u32 & !7,     // LD: 8-aligned
                    _ => prng.range(0x60) as u32,          // sub-word: any
                };
                enc_i(imm, rs1, funct3, r8(prng), LOAD)
            }
            5 => {
                // STORE: SB/SH/SW/SD to sandbox bases at every legal
                // alignment (SB/SH splice at any byte offset).
                let funct3 = [0u32, 1, 2, 3][prng.range(4) as usize];
                let rs1 = (1u8 + prng.range(4) as u8) & 0x1f;
                let rs2 = r8(prng);
                let imm = match funct3 {
                    2 => prng.range(0x60) as u32 & !3, // SW: 4-aligned
                    3 => prng.range(0x60) as u32 & !7, // SD: 8-aligned
                    _ => prng.range(0x60) as u32,      // SB/SH: any
                };
                enc_s(imm, rs2, rs1, funct3, STORE)
            }
            6 => {
                // LUI.
                ((prng.next() as u32) & 0xFFFF_F000) | ((r8(prng) as u32) << 7) | LUI
            }
            7 => {
                // AUIPC.
                ((prng.next() as u32) & 0xFFFF_F000) | ((r8(prng) as u32) << 7) | AUIPC
            }
            8 | 9 => {
                // AMO word-granular family (sandbox base x1..x4).
                let funct3 =
                    [2u32, 3, 1, 0, 4, 0xC, 8, 0x10, 0x14, 0x18, 0x1c][prng.range(11) as usize];
                let rs1 = (1u8 + prng.range(4) as u8) & 0x1f;
                enc_r(0x02, r8(prng), rs1, funct3, r8(prng), AMO)
            }
            _ => {
                // Random 32-bit word (may decode or not — both sides
                // must agree on the outcome).
                prng.next() as u32
            }
        };
        words.push(word);
    }
    // ECALL halt.
    words.push(0x00000073);
    words
}

/// Effective byte view of the canonical memory.
fn canonical_bytes(state: &MachineState) -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    for (addr, word) in state.memory.snapshot_pairs() {
        for i in 0..8 {
            out.push((addr + i, (word >> (8 * i)) as u8));
        }
    }
    out
}

fn run_differential(seed: u64, n_instr: usize) -> Result<(), String> {
    let mut prng = Prng::new(seed);
    let program = gen_program(&mut prng, n_instr);

    // Random initial state (identical for both machines).
    let init_regs: [u64; 32] = {
        let mut r = [0u64; 32];
        for v in r.iter_mut() {
            *v = prng.next();
        }
        r[0] = 0;
        r
    };
    let init_mem: Vec<(u64, u64)> = (0..8)
        .map(|_| (0x200 + prng.range(0x100) * 8, prng.next()))
        .collect();

    // Canonical machine.
    let mut canon = MachineState::new();
    canon.regs = init_regs;
    for (a, w) in &init_mem {
        canon.memory.store(*a, *w);
    }
    let mut image = Vec::new();
    for w in &program {
        image.extend_from_slice(&w.to_le_bytes());
    }
    canon.load_program(0, &image);
    let canon_result = run(&mut canon, 512);

    // Reference machine.
    let mut refr = RefMachine::new();
    refr.regs = init_regs;
    for (a, w) in &init_mem {
        refr.mem.write_u64(*a, *w);
    }
    refr.mem.load_image(0, &image);
    let refr_result = refr.run(512);

    // Error behavior must match (both error or both ok). The one
    // asymmetry: the canonical run() truncates silently at the step
    // limit while the reference reports StepLimitExceeded — treat that
    // as "both completed-truncated" and fall through to the state
    // comparison (both machines are deterministic and have executed
    // exactly max_steps steps).
    let refr_truncated = matches!(
        &refr_result,
        Err(crate::reference::RefError::StepLimitExceeded)
    );
    match (&canon_result, &refr_result) {
        (Err(_), Err(_)) => return Ok(()),
        (Err(c), Ok(_)) => {
            return Err(format!(
                "seed {seed}: canonical errored {c:?}, reference did not"
            ));
        }
        (Ok(_), Err(r)) if refr_truncated => {}
        (Ok(_), Err(r)) => {
            return Err(format!(
                "seed {seed}: reference errored {r:?}, canonical did not"
            ));
        }
        (Ok(_), Ok(_)) => {}
    }

    // Register state.
    for i in 0..32 {
        if canon.regs[i] != refr.regs[i] {
            return Err(format!(
                "seed {seed}: x{i} canonical {:#x} != reference {:#x}",
                canon.regs[i], refr.regs[i]
            ));
        }
    }
    // PC + halted.
    if canon.pc != refr.pc {
        return Err(format!(
            "seed {seed}: pc canonical {:#x} != reference {:#x}",
            canon.pc, refr.pc
        ));
    }
    if canon.halted != refr.halted {
        return Err(format!(
            "seed {seed}: halted canonical {} != reference {}",
            canon.halted, refr.halted
        ));
    }
    // Memory: compare the effective byte maps (union of addresses,
    // default zero).
    let cb = canonical_bytes(&canon);
    let rb = refr.mem.snapshot();
    let mut addr_set = std::collections::BTreeMap::new();
    for (a, b) in &cb {
        addr_set.insert(*a, *b);
    }
    for (a, b) in &rb {
        addr_set.insert(*a, *b);
    }
    // The program image region is byte-identical by construction; the
    // interesting region is the sandbox.
    for a in addr_set.keys() {
        let cbyte = cb
            .iter()
            .find(|(x, _)| x == a)
            .map(|(_, b)| *b)
            .unwrap_or(0);
        let rbyte = rb
            .iter()
            .find(|(x, _)| x == a)
            .map(|(_, b)| *b)
            .unwrap_or(0);
        if cbyte != rbyte {
            return Err(format!(
                "seed {seed}: mem[{a:#x}] canonical {cbyte:#x} != reference {rbyte:#x}"
            ));
        }
    }
    Ok(())
}

#[test]
fn differential_random_programs_agree() {
    let mut failures = Vec::new();
    for seed in 1..=120u64 {
        if let Err(msg) = run_differential(seed, 40) {
            failures.push(msg);
        }
    }
    assert!(
        failures.is_empty(),
        "differential failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn differential_long_programs_agree() {
    for seed in 1000..=1010u64 {
        if let Err(msg) = run_differential(seed, 150) {
            panic!("long-program differential failure: {msg}");
        }
    }
}
