//! Deterministic instruction execution producing canonical trace rows —
//! the witness source for the claim DAG.

use crate::decode::{decode, decode_compressed, Instr};
use crate::state::MachineState;
use lattice_core::Goldilocks;

/// One execution step's trace row: the exact operands/flags/addresses the
/// claim-DAG stages consume (registers, RAM, bytecode).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRow {
    pub pc: u64,
    pub instr: Instr,
    /// Instruction width in bytes (4 uncompressed, 2 compressed).
    pub width: u32,
    /// (register index, read value) pairs this step consumed.
    pub reg_reads: Vec<(u8, u64)>,
    /// (register index, written value) pairs produced.
    pub reg_writes: Vec<(u8, u64)>,
    /// Memory access, if any: (address, read value, written value).
    pub mem_access: Option<(u64, u64, Option<u64>)>,
    /// Next pc.
    pub next_pc: u64,
    /// Step index.
    pub step_index: u64,
    /// Halted after this step.
    pub halted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecError {
    Decode(crate::decode::DecodeError),
    MisalignedJump { pc: u64, target: u64 },
}

/// Execute one instruction at the machine's pc, mutating state and
/// returning the canonical trace row.
pub fn step(state: &mut MachineState, step_index: u64) -> Result<TraceRow, ExecError> {
    let pc = state.pc;
    // Fetch: 32-bit granular reads (subword-correct at any alignment);
    // compressed if the low two bits differ from 0b11.
    let half = state.memory.load_word32(pc) as u16;
    let (instr, width, next_pc_default) = if half & 0x3 != 0x3 {
        let (i, npc) = decode_compressed(pc, half).map_err(ExecError::Decode)?;
        (i, 2u32, npc)
    } else {
        let word = state.memory.load_word32(pc) as u32;
        let i = decode(pc, word).map_err(ExecError::Decode)?;
        (i, 4u32, pc + 4)
    };

    let mut next_pc = next_pc_default;
    let mut reg_reads = Vec::new();
    let mut reg_writes = Vec::new();
    let mut mem_access = None;
    let mut halted = false;
    let rd_write = |writes: &mut Vec<(u8, u64)>, rd: u8, v: u64| writes.push((rd, v));

    macro_rules! rr {
        ($r:expr) => {{
            let v = state.reg($r);
            reg_reads.push(($r, v));
            v
        }};
    }

    match instr {
        Instr::Addi { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, a.wrapping_add(imm as u64));
        }
        Instr::Addiw { rd, rs1, imm } => {
            let a = rr!(rs1);
            let w = (a as i32).wrapping_add(imm as i32);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Slti { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, ((a as i64) < imm) as u64);
        }
        Instr::Sltiu { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, (a < imm) as u64);
        }
        Instr::Xori { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, a ^ (imm as u64));
        }
        Instr::Ori { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, a | (imm as u64));
        }
        Instr::Andi { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, a & (imm as u64));
        }
        Instr::Slli { rd, rs1, shamt } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, a.wrapping_shl(shamt as u32));
        }
        Instr::Srli { rd, rs1, shamt } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, a.wrapping_shr(shamt as u32));
        }
        Instr::Srai { rd, rs1, shamt } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, ((a as i64).wrapping_shr(shamt as u32)) as u64);
        }
        Instr::Slliw { rd, rs1, shamt } => {
            let a = rr!(rs1);
            let w = (a as u32).wrapping_shl(shamt as u32);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Srliw { rd, rs1, shamt } => {
            let a = rr!(rs1);
            let w = (a as u32).wrapping_shr(shamt as u32);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Sraiw { rd, rs1, shamt } => {
            let a = rr!(rs1);
            let w = (a as i32).wrapping_shr(shamt as u32);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Lui { rd, imm } => {
            rd_write(&mut reg_writes, rd, imm as u64);
        }
        Instr::Auipc { rd, imm } => {
            rd_write(&mut reg_writes, rd, pc.wrapping_add(imm as u64));
        }
        Instr::Add { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a.wrapping_add(b));
        }
        Instr::Sub { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a.wrapping_sub(b));
        }
        Instr::Sll { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a.wrapping_shl(b as u32 & 0x3f));
        }
        Instr::Slt { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, ((a as i64) < (b as i64)) as u64);
        }
        Instr::Sltu { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, (a < b) as u64);
        }
        Instr::Xor { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a ^ b);
        }
        Instr::Srl { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a.wrapping_shr(b as u32 & 0x3f));
        }
        Instr::Sra { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, ((a as i64).wrapping_shr(b as u32 & 0x3f)) as u64);
        }
        Instr::Or { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a | b);
        }
        Instr::And { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a & b);
        }
        Instr::Addw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let w = (a as i32).wrapping_add(b as i32);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Subw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let w = (a as i32).wrapping_sub(b as i32);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Sllw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let w = ((a as u32).wrapping_shl(b as u32 & 0x1f)) as i32;
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Srlw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let w = (a as u32).wrapping_shr(b as u32 & 0x1f);
            rd_write(&mut reg_writes, rd, w as u64);
        }
        Instr::Sraw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let w = (a as i32).wrapping_shr(b as u32 & 0x1f);
            rd_write(&mut reg_writes, rd, w as i64 as u64);
        }
        Instr::Lw { rd, rs1, imm } => {
            let a = rr!(rs1);
            let addr = a.wrapping_add(imm as u64);
            let val = state.memory.load_word32(addr);
            mem_access = Some((addr, val, None));
            rd_write(&mut reg_writes, rd, val as i32 as i64 as u64);
        }
        Instr::Lwu { rd, rs1, imm } => {
            let a = rr!(rs1);
            let addr = a.wrapping_add(imm as u64);
            let val = state.memory.load_word32(addr);
            mem_access = Some((addr, val, None));
            rd_write(&mut reg_writes, rd, val);
        }
        Instr::Ld { rd, rs1, imm } => {
            let a = rr!(rs1);
            let addr = a.wrapping_add(imm as u64);
            // Spec-correct unaligned doubleword read.
            let val = state.memory.load_u64(addr);
            mem_access = Some((addr, val, None));
            rd_write(&mut reg_writes, rd, val);
        }
        Instr::Sw { rs1, rs2, imm } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let addr = a.wrapping_add(imm as u64);
            let old = state.memory.load_word32(addr);
            state.memory.store_word32(addr, v as u32);
            mem_access = Some((addr, old, Some(v & 0xFFFF_FFFF)));
        }
        Instr::Sd { rs1, rs2, imm } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let addr = a.wrapping_add(imm as u64);
            // Spec-correct unaligned doubleword write (RMW across words).
            let old = state.memory.load_u64(addr);
            state.memory.store_u64(addr, v);
            mem_access = Some((addr, old, Some(v)));
        }
        Instr::Beq { rs1, rs2, imm } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            if a == b {
                next_pc = pc.wrapping_add(imm as u64);
            }
        }
        Instr::Bne { rs1, rs2, imm } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            if a != b {
                next_pc = pc.wrapping_add(imm as u64);
            }
        }
        Instr::Blt { rs1, rs2, imm } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            if (a as i64) < (b as i64) {
                next_pc = pc.wrapping_add(imm as u64);
            }
        }
        Instr::Bge { rs1, rs2, imm } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            if (a as i64) >= (b as i64) {
                next_pc = pc.wrapping_add(imm as u64);
            }
        }
        Instr::Bltu { rs1, rs2, imm } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            if a < b {
                next_pc = pc.wrapping_add(imm as u64);
            }
        }
        Instr::Bgeu { rs1, rs2, imm } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            if a >= b {
                next_pc = pc.wrapping_add(imm as u64);
            }
        }
        Instr::Jal { rd, imm } => {
            rd_write(&mut reg_writes, rd, pc.wrapping_add(width as u64));
            next_pc = pc.wrapping_add(imm as u64);
        }
        Instr::Jalr { rd, rs1, imm } => {
            let a = rr!(rs1);
            rd_write(&mut reg_writes, rd, pc.wrapping_add(width as u64));
            next_pc = a.wrapping_add(imm as u64) & !1;
        }
        Instr::Ecall | Instr::Ebreak => {
            halted = true;
        }
        Instr::Mul { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, a.wrapping_mul(b));
        }
        Instr::Mulh { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, ((((a as i64) as i128) * ((b as i64) as i128)) >> 64) as i64 as u64);
        }
        Instr::Mulhu { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            rd_write(&mut reg_writes, rd, ((a as u128 * b as u128) >> 64) as u64);
        }
        Instr::Div { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = if b == 0 {
                u64::MAX
            } else {
                (a as i64).wrapping_div(b as i64) as u64
            };
            rd_write(&mut reg_writes, rd, v);
        }
        Instr::Divu { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = a.checked_div(b).unwrap_or(u64::MAX);
            rd_write(&mut reg_writes, rd, v);
        }
        Instr::Rem { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = if b == 0 { a } else { (a as i64).wrapping_rem(b as i64) as u64 };
            rd_write(&mut reg_writes, rd, v);
        }
        Instr::Remu { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = if b == 0 { a } else { a % b };
            rd_write(&mut reg_writes, rd, v);
        }
        Instr::Divw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = if (b as u32) == 0 {
                u32::MAX
            } else {
                (a as i32).wrapping_div(b as i32) as u32
            };
            rd_write(&mut reg_writes, rd, v as i32 as i64 as u64);
        }
        Instr::Divuw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = (a as u32).checked_div(b as u32).unwrap_or(u32::MAX);
            rd_write(&mut reg_writes, rd, v as u64);
        }
        Instr::Remw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = if (b as u32) == 0 {
                a as u32
            } else {
                (a as i32).wrapping_rem(b as i32) as u32
            };
            rd_write(&mut reg_writes, rd, v as i32 as i64 as u64);
        }
        Instr::Remuw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let v = if (b as u32) == 0 {
                a as u32
            } else {
                (a as u32) % (b as u32)
            };
            rd_write(&mut reg_writes, rd, v as u64);
        }
        Instr::Mulw { rd, rs1, rs2 } => {
            let (a, b) = (rr!(rs1), rr!(rs2));
            let w = (a as u32).wrapping_mul(b as u32);
            rd_write(&mut reg_writes, rd, w as i32 as i64 as u64);
        }
        Instr::LrW { rd, rs1, .. } => {
            let a = rr!(rs1);
            let val = state.memory.load_word32(a);
            state.reservation = Some(a);
            mem_access = Some((a, val, None));
            rd_write(&mut reg_writes, rd, val as i32 as i64 as u64);
        }
        Instr::ScW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let success = state.reservation == Some(a);
            if success {
                state.memory.store_word32(a, v as u32);
                mem_access = Some((a, v, Some(v & 0xFFFF_FFFF)));
                state.reservation = None;
            } else {
                let cur = state.memory.load_word32(a);
                mem_access = Some((a, cur, None));
            }
            rd_write(&mut reg_writes, rd, (!success) as u64);
        }
        Instr::AmoSwapW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            state.memory.store_word32(a, v as u32);
            mem_access = Some((a, old, Some(v & 0xFFFF_FFFF)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoAddW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = (old as u32).wrapping_add(v as u32);
            state.memory.store_word32(a, new);
            mem_access = Some((a, old, Some(new as u64)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoXorW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = (old as u32) ^ (v as u32);
            state.memory.store_word32(a, new);
            mem_access = Some((a, old, Some(new as u64)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoAndW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = (old as u32) & (v as u32);
            state.memory.store_word32(a, new);
            mem_access = Some((a, old, Some(new as u64)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoOrW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = (old as u32) | (v as u32);
            state.memory.store_word32(a, new);
            mem_access = Some((a, old, Some(new as u64)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoMinW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = if (old as i32) < (v as i32) { old } else { v };
            state.memory.store_word32(a, new as u32);
            mem_access = Some((a, old, Some(new)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoMaxW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = if (old as i32) > (v as i32) { old } else { v };
            state.memory.store_word32(a, new as u32);
            mem_access = Some((a, old, Some(new)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoMinuW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = if old < v { old } else { v };
            state.memory.store_word32(a, new as u32);
            mem_access = Some((a, old, Some(new)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
        Instr::AmoMaxuW { rd, rs1, rs2, .. } => {
            let (a, v) = (rr!(rs1), rr!(rs2));
            let old = state.memory.load_word32(a);
            let new = if old > v { old } else { v };
            state.memory.store_word32(a, new as u32);
            mem_access = Some((a, old, Some(new)));
            rd_write(&mut reg_writes, rd, old as i32 as i64 as u64);
        }
    }

    // Commit register writes (x0 filtering in set_reg).
    for (r, v) in &reg_writes {
        state.set_reg(*r, *v);
    }
    state.pc = next_pc;
    state.halted = state.halted || halted;

    Ok(TraceRow {
        pc,
        instr,
        width,
        reg_reads,
        reg_writes,
        mem_access,
        next_pc,
        step_index,
        halted,
    })
}

/// Run until halt or a step bound (DoS guard).
pub fn run(
    state: &mut MachineState,
    max_steps: u64,
) -> Result<Vec<TraceRow>, ExecError> {
    let mut rows = Vec::new();
    for i in 0..max_steps {
        if state.halted {
            break;
        }
        rows.push(step(state, i)?);
    }
    Ok(rows)
}

/// TraceRow -> field elements (witness column encoding).
impl TraceRow {
    pub fn to_fields(&self) -> Vec<Goldilocks> {
        let mut out = Vec::with_capacity(16);
        out.push(Goldilocks::from_u64(self.pc));
        out.push(Goldilocks::from_u64(self.width as u64));
        out.push(Goldilocks::from_u64(self.next_pc));
        for (r, v) in &self.reg_reads {
            out.push(Goldilocks::from_u64(*r as u64));
            out.push(Goldilocks::from_u64(*v));
        }
        for (r, v) in &self.reg_writes {
            out.push(Goldilocks::from_u64(*r as u64));
            out.push(Goldilocks::from_u64(*v));
        }
        if let Some((addr, old, new)) = &self.mem_access {
            out.push(Goldilocks::from_u64(*addr));
            out.push(Goldilocks::from_u64(*old));
            if let Some(n) = new {
                out.push(Goldilocks::from_u64(*n));
                out.push(Goldilocks::ONE);
            } else {
                out.push(Goldilocks::ZERO);
                out.push(Goldilocks::ZERO);
            }
        }
        out.push(Goldilocks::from_u64(self.halted as u64));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
        (((imm as u32) << 20) | ((rs1 as u32) << 15)) | ((rd as u32) << 7) | 0x13
    }

    #[test]
    fn addi_executes() {
        // addi x5, x0, 100
        let mut s = MachineState::new();
        s.load_program(0, &enc_addi(5, 0, 100).to_le_bytes());
        let row = step(&mut s, 0).ok().unwrap();
        assert_eq!(s.reg(5), 100);
        assert_eq!(row.instr, Instr::Addi { rd: 5, rs1: 0, imm: 100 });
        assert_eq!(row.reg_writes, vec![(5, 100)]);
        assert_eq!(s.pc, 4);
    }

    #[test]
    fn arithmetic_program() {
        // addi x1, x0, 7; addi x2, x0, 5; add x3, x1, x2; ecall
        let mut s = MachineState::new();
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 7).to_le_bytes());
        prog.extend_from_slice(&enc_addi(2, 0, 5).to_le_bytes());
        let add_x3 = (2u32 << 20) | (1 << 15) | (3 << 7) | 0x33; // add x3, x1, x2
        prog.extend_from_slice(&add_x3.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes()); // ecall
        s.load_program(0, &prog);
        let rows = run(&mut s, 16).ok().unwrap();
        assert_eq!(s.reg(3), 12);
        assert_eq!(rows.len(), 4);
        assert!(s.halted);
        assert_eq!(rows[3].instr, Instr::Ecall);
    }

    #[test]
    fn branch_and_loop() {
        // addi x1, x0, 3; addi x1, x1, -1; bne x1, x0, -4; ecall
        let mut s = MachineState::new();
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 3).to_le_bytes());
        prog.extend_from_slice(&enc_addi(1, 1, -1).to_le_bytes());
        // bne x1, x0, -4: imm[12]=1, imm[10:5]=0x3f, rs2=0, rs1=1, f3=1,
        // imm[4:1]=0xe, imm[11]=1
        let bne: u32 = ((1 << 31) | (0x3f << 25)) | (1 << 15) | (1 << 12)
            | (0b1110 << 8) | (1 << 7) | 0x63;
        prog.extend_from_slice(&bne.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        s.load_program(0, &prog);
        let rows = run(&mut s, 64).ok().unwrap();
        // 3 decrements: 1 + 3*(addi+bne) + ecall = 8 steps.
        assert_eq!(rows.len(), 8);
        assert_eq!(s.reg(1), 0);
        assert!(s.halted);
    }

    #[test]
    fn memory_load_store() {
        // sw x2, 0(x1); lw x3, 0(x1); ecall — with x1 = 0x100, x2 = 0xAB.
        let mut s = MachineState::new();
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 0x100).to_le_bytes());
        prog.extend_from_slice(&enc_addi(2, 0, 0xAB).to_le_bytes());
        // sw x2, 0(x1): f3=2, opcode 0x23.
        let sw: u32 = ((2 << 20) | (1 << 15) | (2 << 12)) | 0x23; // sw x2, 0(x1)
        prog.extend_from_slice(&sw.to_le_bytes());
        // lw x3, 0(x1): f3=2, opcode 0x03.
        let lw: u32 = ((1 << 15)) | (2 << 12) | (3 << 7) | 0x03;
        prog.extend_from_slice(&lw.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        s.load_program(0, &prog);
        let rows = run(&mut s, 16).ok().unwrap();
        assert_eq!(s.reg(3), 0xAB);
        // The store row records the memory access.
        let store_row = &rows[2];
        assert!(store_row.mem_access.is_some());
        let (addr, _old, written) = store_row.mem_access.ok_or(()).ok().unwrap();
        assert_eq!(addr, 0x100);
        assert_eq!(written, Some(0xAB));
    }

    #[test]
    fn mul_div_semantics() {
        // mulh, div-by-zero, rem.
        let mut s = MachineState::new();
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, -1).to_le_bytes()); // x1 = u64::MAX
        prog.extend_from_slice(&enc_addi(2, 0, 2).to_le_bytes());
        // mulh x3, x1, x2: f3=1, f7=1.
        let mulh: u32 = (1 << 25) | (2 << 20) | (1 << 15) | (1 << 12) | (3 << 7) | 0x33;
        prog.extend_from_slice(&mulh.to_le_bytes());
        // div x4, x1, x0 (divide by zero): f3=4, f7=1.
        let divz: u32 = ((1 << 25)) | (1 << 15) | (4 << 12) | (4 << 7) | 0x33;
        prog.extend_from_slice(&divz.to_le_bytes());
        // rem x5, x2, x0.
        let remz: u32 = ((1 << 25)) | (2 << 15) | (6 << 12) | (5 << 7) | 0x33;
        prog.extend_from_slice(&remz.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        s.load_program(0, &prog);
        let _rows = run(&mut s, 16).ok().unwrap();
        // mulh(-1, 2) = -1 >> 64... ( -1 * 2 ) >> 64 = -1 (since -2 >> 64 = -1).
        assert_eq!(s.reg(3), (-1i64) as u64);
        assert_eq!(s.reg(4), u64::MAX); // div by zero -> -1
        assert_eq!(s.reg(5), 2); // rem by zero -> dividend
    }

    #[test]
    fn jal_link_value() {
        // jal x1, +8: skip one instruction.
        let mut s = MachineState::new();
        let mut prog = Vec::new();
        // jal x1, 8: imm[20]=0, imm[10:1]=0000000100 (8>>1=4), imm[11]=0, imm[19:12]=0.
        let jal: u32 = (4 << 21) | (1 << 7) | 0x6f;
        prog.extend_from_slice(&jal.to_le_bytes());
        prog.extend_from_slice(&enc_addi(2, 0, 99).to_le_bytes()); // skipped
        prog.extend_from_slice(&0x73u32.to_le_bytes()); // target
        s.load_program(0, &prog);
        let rows = run(&mut s, 8).ok().unwrap();
        assert_eq!(s.pc, 12);
        assert_eq!(s.reg(1), 4); // link = pc + 4
        assert_eq!(rows.len(), 2); // jal + ecall
    }

    #[test]
    fn amo_and_lrsc() {
        // LR/SC round trip on a fresh address succeeds.
        let mut s = MachineState::new();
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(1, 0, 0x200).to_le_bytes());
        // lr.w x2, (x1): f5=0x2, f3=2, opcode 0x2f.
        let lr: u32 = (2 << 27) | (1 << 15) | (2 << 12) | (2 << 7) | 0x2f;
        prog.extend_from_slice(&lr.to_le_bytes());
        // sc.w x3, x0, (x1): f5=0x3, rs2=x0.
        let sc: u32 = ((3 << 27)) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2f;
        prog.extend_from_slice(&sc.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        s.load_program(0, &prog);
        let _rows = run(&mut s, 16).ok().unwrap();
        assert_eq!(s.reg(3), 0); // SC succeeded.
        // AMO add.
        let mut s2 = MachineState::new();
        let mut prog2 = Vec::new();
        prog2.extend_from_slice(&enc_addi(1, 0, 0x300).to_le_bytes());
        prog2.extend_from_slice(&enc_addi(2, 0, 5).to_le_bytes());
        // amoadd.w x3, x2, (x1): f5=0x00.
        let amo: u32 = ((2 << 20)) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2f;
        prog2.extend_from_slice(&amo.to_le_bytes());
        prog2.extend_from_slice(&0x73u32.to_le_bytes());
        s2.load_program(0, &prog2);
        let _rows2 = run(&mut s2, 16).ok().unwrap();
        assert_eq!(s2.memory.load_word32(0x300), 5);
        assert_eq!(s2.reg(3), 0); // old value.
    }

    #[test]
    fn trace_row_field_encoding() {
        let mut s = MachineState::new();
        s.load_program(0, &enc_addi(5, 0, 42).to_le_bytes());
        let row = step(&mut s, 0).ok().unwrap();
        let fields = row.to_fields();
        assert!(!fields.is_empty());
        assert_eq!(fields[0], Goldilocks::from_u64(0)); // pc
        assert_eq!(fields[2], Goldilocks::from_u64(4)); // next_pc
    }
}
