//! The independent RV64IMAC reference interpreter for differential
//! testing (audit §9.2 P1 "Conformance").
//!
//! Structural independence from the canonical `decode.rs`/`exec.rs`:
//! * decoding extracts bitfields inline per major opcode (no shared
//!   `Instr` enum);
//! * memory is a **byte-level** map (the canonical machine uses
//!   word-granular sparse memory with subword expansion — a completely
//!   different code path);
//! * register writes apply directly with x0 hardwiring at write time.
//!
//! Coverage note: mirrors the canonical instruction surface exactly —
//! RV64I integer subset (LW/LWU/LD/SW/SD, no byte/half loads), the full
//! RV64M family, the RV64A word-granular atomics, and the codebase's
//! compressed-expansion contract (see `decode_compressed` for its
//! funct3 mapping). Uncompressed encodings follow the RISC-V spec.

#![allow(clippy::manual_checked_ops)]

use std::collections::BTreeMap;

/// Byte-level reference memory (default zero).
#[derive(Clone, Debug, Default)]
pub struct RefMemory {
    bytes: BTreeMap<u64, u8>,
}

impl RefMemory {
    pub fn new() -> Self {
        RefMemory::default()
    }

    fn read_byte(&self, addr: u64) -> u8 {
        self.bytes.get(&addr).copied().unwrap_or(0)
    }

    fn write_byte(&mut self, addr: u64, v: u8) {
        self.bytes.insert(addr, v);
    }

    pub fn read_u32(&self, addr: u64) -> u32 {
        let mut v = 0u32;
        for i in 0..4 {
            v |= (self.read_byte(addr + i) as u32) << (8 * i);
        }
        v
    }

    pub fn read_u64(&self, addr: u64) -> u64 {
        let mut v = 0u64;
        for i in 0..8 {
            v |= (self.read_byte(addr + i) as u64) << (8 * i);
        }
        v
    }

    pub fn write_u32(&mut self, addr: u64, v: u32) {
        for i in 0..4 {
            self.write_byte(addr + i, (v >> (8 * i)) as u8);
        }
    }

    pub fn write_u64(&mut self, addr: u64, v: u64) {
        for i in 0..8 {
            self.write_byte(addr + i, (v >> (8 * i)) as u8);
        }
    }

    /// Snapshot as (address, byte) pairs for state comparison.
    pub fn snapshot(&self) -> Vec<(u64, u8)> {
        self.bytes.iter().map(|(a, v)| (*a, *v)).collect()
    }

    /// Load a program image.
    pub fn load_image(&mut self, base: u64, image: &[u8]) {
        for (i, b) in image.iter().enumerate() {
            self.write_byte(base + i as u64, *b);
        }
    }
}

/// The reference machine.
#[derive(Clone, Debug)]
pub struct RefMachine {
    pub pc: u64,
    pub regs: [u64; 32],
    pub mem: RefMemory,
    pub reservation: Option<u64>,
    pub halted: bool,
}

impl Default for RefMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefError {
    IllegalInstruction { pc: u64, word: u32 },
    IllegalCompressed { pc: u64, half: u16 },
    StepLimitExceeded,
}

fn sign_extend(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

impl RefMachine {
    pub fn new() -> Self {
        RefMachine {
            pc: 0,
            regs: [0u64; 32],
            mem: RefMemory::new(),
            reservation: None,
            halted: false,
        }
    }

    fn reg(&self, i: u8) -> u64 {
        if i == 0 {
            0
        } else {
            self.regs[i as usize]
        }
    }

    fn set_reg(&mut self, i: u8, v: u64) {
        if i != 0 {
            self.regs[i as usize] = v;
        }
    }

    /// Run until halt or the step limit.
    pub fn run(&mut self, max_steps: u64) -> Result<(), RefError> {
        let mut steps = 0u64;
        while !self.halted {
            if steps >= max_steps {
                return Err(RefError::StepLimitExceeded);
            }
            self.step()?;
            steps += 1;
        }
        Ok(())
    }

    /// One instruction step.
    pub fn step(&mut self) -> Result<(), RefError> {
        let pc = self.pc;
        let half = self.mem.read_u32(pc) as u16;
        if half & 0x3 != 0x3 {
            self.exec_compressed(pc, half)
        } else {
            let word = self.mem.read_u32(pc);
            self.exec_word(pc, word)
        }
    }

    // ---- 32-bit semantics (spec-derived major opcodes) ----

    #[allow(clippy::too_many_lines)]
    fn exec_word(&mut self, pc: u64, word: u32) -> Result<(), RefError> {
        // opcode = word[6:2].
        let opcode = (word >> 2) & 0x1f;
        let rd = ((word >> 7) & 0x1f) as u8;
        let rs1 = ((word >> 15) & 0x1f) as u8;
        let rs2 = ((word >> 20) & 0x1f) as u8;
        let funct3 = (word >> 12) & 0x7;
        let funct7 = word >> 25;
        let next = pc.wrapping_add(4);

        let imm_i = sign_extend((word >> 20) as u64, 12);
        let imm_s = sign_extend(
            ((((word >> 25) & 0x7f) << 5) | ((word >> 7) & 0x1f)) as u64,
            12,
        );
        let imm_b = sign_extend(
            ((((word >> 31) & 0x1) << 12)
                | (((word >> 7) & 0x1) << 11)
                | (((word >> 25) & 0x3f) << 5)
                | (((word >> 8) & 0xf) << 1)) as u64,
            13,
        );
        let imm_u = sign_extend((word & 0xFFFF_F000) as u64, 32) as u64;
        let imm_j = sign_extend(
            ((((word >> 31) & 0x1) << 20)
                | (((word >> 12) & 0xff) << 12)
                | (((word >> 20) & 0x1) << 11)
                | (((word >> 21) & 0x3ff) << 1)) as u64,
            21,
        );

        match opcode {
            // OP (0110011): register-register.
            0x0c => {
                let a = self.reg(rs1);
                let b = self.reg(rs2);
                let v = match (funct3, funct7) {
                    (0, 0x00) => a.wrapping_add(b),
                    (0, 0x20) => a.wrapping_sub(b),
                    (1, 0x00) => a.wrapping_shl(b as u32 & 63),
                    (2, 0x00) => ((a as i64) < (b as i64)) as u64,
                    (3, 0x00) => (a < b) as u64,
                    (4, 0x00) => a ^ b,
                    (5, 0x00) => a.wrapping_shr(b as u32 & 63),
                    (5, 0x20) => ((a as i64).wrapping_shr(b as u32 & 63)) as u64,
                    (6, 0x00) => a | b,
                    (7, 0x00) => a & b,
                    (0, 0x01) => a.wrapping_mul(b),
                    (1, 0x01) => (((a as i128) * (b as i128)) >> 64) as u64,
                    (2, 0x01) => (((a as u128) * (b as u128)) >> 64) as u64,
                    (4, 0x01) => {
                        if b == 0 {
                            u64::MAX // -1
                        } else if a == i64::MIN as u64 && b == (-1i64) as u64 {
                            i64::MIN as u64
                        } else {
                            ((a as i64) / (b as i64)) as u64
                        }
                    }
                    (5, 0x01) => {
                        if b == 0 {
                            u64::MAX
                        } else {
                            a / b
                        }
                    }
                    (6, 0x01) => {
                        if b == 0 {
                            a
                        } else if a == i64::MIN as u64 && b == (-1i64) as u64 {
                            0
                        } else {
                            ((a as i64) % (b as i64)) as u64
                        }
                    }
                    (7, 0x01) => {
                        if b == 0 {
                            a
                        } else {
                            a % b
                        }
                    }
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                };
                self.set_reg(rd, v);
                self.pc = next;
            }
            // OP-IMM (0010011).
            0x04 => {
                let a = self.reg(rs1);
                let v = match funct3 {
                    0 => a.wrapping_add(imm_i as u64),
                    2 => ((a as i64) < imm_i) as u64,
                    3 => (a < imm_i as u64) as u64,
                    4 => a ^ (imm_i as u64),
                    6 => a | (imm_i as u64),
                    7 => a & (imm_i as u64),
                    1 => a.wrapping_shl((word >> 20) & 0x3f),
                    _ => {
                        // funct3 = 5: SRLI / SRAI (RV64: 6-bit shamt,
                        // funct6 at bits[31:26]: 0 = SRLI, 0x10 = SRAI).
                        let shamt = (word >> 20) & 0x3f;
                        if (word >> 26) & 0x3f == 0x10 {
                            ((a as i64).wrapping_shr(shamt)) as u64
                        } else {
                            a.wrapping_shr(shamt)
                        }
                    }
                };
                self.set_reg(rd, v);
                self.pc = next;
            }
            // OP-IMM-32 (0011011): ADDIW (and the shift-immediates the
            // canonical decoder folds into the same family).
            0x06 => {
                let a = self.reg(rs1);
                let v = match funct3 {
                    0 => ((a as i32).wrapping_add(imm_i as i32)) as i64 as u64,
                    1 => {
                        // SLLIW: 5-bit shamt, funct7 = 0.
                        if funct7 != 0 {
                            return Err(RefError::IllegalInstruction { pc, word });
                        }
                        (a as u32).wrapping_shl((word >> 20) & 0x1f) as i64 as u64
                    }
                    5 => {
                        // SRLIW (funct7 = 0) / SRAW (funct7 = 0x20).
                        let shamt = (word >> 20) & 0x1f;
                        match funct7 {
                            0 => (a as u32).wrapping_shr(shamt) as i64 as u64,
                            0x20 => ((a as i32).wrapping_shr(shamt)) as i64 as u64,
                            _ => return Err(RefError::IllegalInstruction { pc, word }),
                        }
                    }
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                };
                self.set_reg(rd, v);
                self.pc = next;
            }
            // OP-32 (0111011).
            0x0e => {
                let a = self.reg(rs1);
                let b = self.reg(rs2);
                let v = match (funct3, funct7) {
                    (0, 0x00) => ((a as i32).wrapping_add(b as i32)) as i64 as u64,
                    (0, 0x20) => ((a as i32).wrapping_sub(b as i32)) as i64 as u64,
                    (1, 0x00) => (a as u32).wrapping_shl(b as u32 & 31) as i64 as u64,
                    (5, 0x00) => (a as u32).wrapping_shr(b as u32 & 31) as i64 as u64,
                    (5, 0x20) => ((a as i32).wrapping_shr(b as u32 & 31)) as i64 as u64,
                    (0, 0x01) => ((a as i32).wrapping_mul(b as i32)) as i64 as u64,
                    (4, 0x01) => {
                        let (x, y) = (a as i32, b as i32);
                        if y == 0 {
                            (-1i32) as i64 as u64
                        } else if x == i32::MIN && y == -1 {
                            i32::MIN as i64 as u64
                        } else {
                            (x / y) as i64 as u64
                        }
                    }
                    (5, 0x01) => {
                        let (x, y) = (a as u32, b as u32);
                        if y == 0 {
                            u32::MAX as i64 as u64
                        } else {
                            (x / y) as i64 as u64
                        }
                    }
                    (6, 0x01) => {
                        let (x, y) = (a as i32, b as i32);
                        if y == 0 {
                            x as i64 as u64
                        } else if x == i32::MIN && y == -1 {
                            0
                        } else {
                            (x % y) as i64 as u64
                        }
                    }
                    (7, 0x01) => {
                        let (x, y) = (a as u32, b as u32);
                        if y == 0 {
                            x as i64 as u64
                        } else {
                            (x % y) as i64 as u64
                        }
                    }
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                };
                self.set_reg(rd, v);
                self.pc = next;
            }
            // LUI (0110111).
            0x0d => {
                self.set_reg(rd, imm_u);
                self.pc = next;
            }
            // AUIPC (0010111).
            0x05 => {
                self.set_reg(rd, pc.wrapping_add(imm_u));
                self.pc = next;
            }
            // JAL (1101111).
            0x1b => {
                self.set_reg(rd, next);
                self.pc = pc.wrapping_add(imm_j as u64);
            }
            // JALR (1100111).
            0x19 => {
                let target = self.reg(rs1).wrapping_add(imm_i as u64) & !1u64;
                self.set_reg(rd, next);
                self.pc = target;
            }
            // BRANCH (1100011).
            0x18 => {
                let a = self.reg(rs1);
                let b = self.reg(rs2);
                let taken = match funct3 {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i64) < (b as i64),
                    5 => (a as i64) >= (b as i64),
                    6 => a < b,
                    7 => a >= b,
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                };
                self.pc = if taken {
                    pc.wrapping_add(imm_b as u64)
                } else {
                    next
                };
            }
            // LOAD (0000011): LW / LWU / LD.
            0x00 => {
                let addr = self.reg(rs1).wrapping_add(imm_i as u64);
                let v = match funct3 {
                    2 => self.mem.read_u32(addr) as i32 as i64 as u64,
                    3 => self.mem.read_u32(addr) as u64,
                    4 => self.mem.read_u64(addr),
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                };
                self.set_reg(rd, v);
                self.pc = next;
            }
            // STORE (0100011): SW / SD.
            0x08 => {
                let addr = self.reg(rs1).wrapping_add(imm_s as u64);
                match funct3 {
                    2 => self.mem.write_u32(addr, self.reg(rs2) as u32),
                    3 => self.mem.write_u64(addr, self.reg(rs2)),
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                }
                self.pc = next;
            }
            // SYSTEM (1110011): ECALL / EBREAK halt.
            0x1c => {
                self.halted = true;
                self.pc = next;
            }
            // MISC-MEM (0001111): FENCE — no-op.
            0x03 => {
                self.pc = next;
            }
            // AMO (0101111): word-granular atomics.
            0x0b => {
                let addr = self.reg(rs1);
                let rs2v = self.reg(rs2);
                match funct3 {
                    2 => {
                        // LR.W
                        self.set_reg(rd, self.mem.read_u32(addr) as i32 as i64 as u64);
                        self.reservation = Some(addr);
                    }
                    3 => {
                        // SC.W
                        let ok = self.reservation == Some(addr);
                        self.set_reg(rd, if ok { 0 } else { 1 });
                        if ok {
                            self.mem.write_u32(addr, rs2v as u32);
                        }
                        self.reservation = None;
                    }
                    1 => {
                        // AMOSWAP.W
                        let old = self.mem.read_u32(addr) as i32 as i64 as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, rs2v as u32);
                    }
                    0 => {
                        // AMOADD.W
                        let old = self.mem.read_u32(addr) as i32 as i64 as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, old.wrapping_add(rs2v) as u32);
                    }
                    4 => {
                        // AMOXOR.W
                        let old = self.mem.read_u32(addr) as i32 as i64 as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, (old ^ rs2v) as u32);
                    }
                    0xc => {
                        // AMOAND.W
                        let old = self.mem.read_u32(addr) as i32 as i64 as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, (old & rs2v) as u32);
                    }
                    8 => {
                        // AMOOR.W
                        let old = self.mem.read_u32(addr) as i32 as i64 as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, (old | rs2v) as u32);
                    }
                    0x10 => {
                        // AMOMIN.W
                        let old = self.mem.read_u32(addr) as i32 as i64;
                        self.set_reg(rd, old as u64);
                        self.mem.write_u32(addr, old.min(rs2v as i32 as i64) as u32);
                    }
                    0x14 => {
                        // AMOMAX.W
                        let old = self.mem.read_u32(addr) as i32 as i64;
                        self.set_reg(rd, old as u64);
                        self.mem.write_u32(addr, old.max(rs2v as i32 as i64) as u32);
                    }
                    0x18 => {
                        // AMOMINU.W
                        let old = self.mem.read_u32(addr) as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, old.min(rs2v & 0xFFFF_FFFF) as u32);
                    }
                    0x1c => {
                        // AMOMAXU.W
                        let old = self.mem.read_u32(addr) as u64;
                        self.set_reg(rd, old);
                        self.mem.write_u32(addr, old.max(rs2v & 0xFFFF_FFFF) as u32);
                    }
                    _ => return Err(RefError::IllegalInstruction { pc, word }),
                }
                self.pc = next;
            }
            _ => return Err(RefError::IllegalInstruction { pc, word }),
        }
        Ok(())
    }

    // ---- Compressed semantics (the codebase expansion contract) ----

    fn exec_compressed(&mut self, pc: u64, half: u16) -> Result<(), RefError> {
        let quadrant = half & 0x3;
        let funct3 = (half >> 13) & 0x7;
        let rs1c = ((half >> 7) & 0x7) as u8;
        let expand = |r: u8| r + 8;
        let next = pc + 2;

        match (quadrant, funct3) {
            (0, 0) => {
                // C.ADDI (or EBREAK hint).
                let imm = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
                let rd = ((half >> 7) & 0x1f) as u8;
                if rd == 0 && imm == 0 {
                    self.halted = true;
                } else {
                    let v = self.reg(rd).wrapping_add(sign_extend(imm, 6) as u64);
                    self.set_reg(rd, v);
                }
                self.pc = next;
            }
            (0, 2) => {
                // C.ADDI4SPN.
                let imm = ((((half >> 12) & 0x1) as u64) << 3)
                    | ((((half >> 5) & 0x3) as u64) << 6)
                    | ((((half >> 2) & 0x7) as u64) << 4)
                    | ((((half >> 3) & 0x3) as u64) << 2);
                let rd = expand(rs1c);
                let v = self.reg(2).wrapping_add(sign_extend(imm, 7) as u64);
                self.set_reg(rd, v);
                self.pc = next;
            }
            (0, 5) => {
                // C.LW.
                let imm = ((((half >> 10) & 0x7) as u64) << 3)
                    | ((((half >> 5) & 0x1) as u64) << 2)
                    | ((((half >> 6) & 0x1) as u64) << 1);
                let rd = expand(rs1c);
                let base = expand(((half >> 10) & 0x7) as u8);
                let addr = self.reg(base).wrapping_add(imm);
                let v = self.mem.read_u32(addr) as i32 as i64 as u64;
                self.set_reg(rd, v);
                self.pc = next;
            }
            (1, 2) => {
                // C.LI.
                let imm = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
                let rd = ((half >> 7) & 0x1f) as u8;
                self.set_reg(rd, sign_extend(imm, 6) as u64);
                self.pc = next;
            }
            (1, 1) => {
                // C.ADDI16SP / C.LUI.
                let rd = ((half >> 7) & 0x1f) as u8;
                if rd == 2 {
                    let imm = ((((half >> 12) & 0x1) as u64) << 9)
                        | ((((half >> 5) & 0x1) as u64) << 4)
                        | ((((half >> 2) & 0x3) as u64) << 7)
                        | ((((half >> 6) & 0x1) as u64) << 6)
                        | ((((half >> 3) & 0x1) as u64) << 5);
                    let v = self.reg(2).wrapping_add(sign_extend(imm, 10) as u64);
                    self.set_reg(2, v);
                } else {
                    let imm = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
                    self.set_reg(rd, (sign_extend(imm, 6) as u64) << 12);
                }
                self.pc = next;
            }
            (1, 4) if (half >> 12) & 0x1 == 1 => {
                // C.SUB / C.XOR / C.OR / C.AND.
                let rd = expand(rs1c);
                let rs2 = expand(((half >> 2) & 0x7) as u8);
                let a = self.reg(rd);
                let b = self.reg(rs2);
                let v = match (half >> 5) & 0x3 {
                    0 => a.wrapping_sub(b),
                    1 => a ^ b,
                    2 => a | b,
                    _ => a & b,
                };
                self.set_reg(rd, v);
                self.pc = next;
            }
            (1, 4) => {
                // C.SRLI.
                let rd = expand(rs1c);
                let shamt = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
                let v = self.reg(rd).wrapping_shr(shamt as u32);
                self.set_reg(rd, v);
                self.pc = next;
            }
            (1, 5) => {
                // C.SRAI.
                let rd = expand(rs1c);
                let shamt = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
                let v = ((self.reg(rd) as i64).wrapping_shr(shamt as u32)) as u64;
                self.set_reg(rd, v);
                self.pc = next;
            }
            (1, 6) | (1, 7) => {
                // C.BEQZ / C.BNEZ.
                let rs1 = expand(rs1c);
                let imm = ((((half >> 12) & 0x1) as u64) << 8)
                    | ((((half >> 5) & 0x3) as u64) << 3)
                    | ((((half >> 2) & 0x1) as u64) << 2)
                    | ((((half >> 10) & 0x3) as u64) << 6)
                    | ((((half >> 3) & 0x3) as u64) << 1);
                let taken = if funct3 == 6 {
                    self.reg(rs1) == 0
                } else {
                    self.reg(rs1) != 0
                };
                self.pc = if taken {
                    pc.wrapping_add(sign_extend(imm, 9) as u64)
                } else {
                    next
                };
            }
            (2, 6) => {
                // C.SW.
                let rs1 = expand(rs1c);
                let rs2 = expand(((half >> 2) & 0x7) as u8);
                let imm =
                    ((((half >> 10) & 0x7) as u64) << 3) | ((((half >> 5) & 0x3) as u64) << 1);
                let addr = self.reg(rs1).wrapping_add(imm);
                self.mem.write_u32(addr, self.reg(rs2) as u32);
                self.pc = next;
            }
            (2, 3) => {
                // C.SD.
                let rs1 = expand(rs1c);
                let rs2 = expand(((half >> 2) & 0x7) as u8);
                let imm =
                    ((((half >> 10) & 0x7) as u64) << 3) | ((((half >> 5) & 0x3) as u64) << 1);
                let addr = self.reg(rs1).wrapping_add(imm);
                self.mem.write_u64(addr, self.reg(rs2));
                self.pc = next;
            }
            (3, 4) if (half >> 12) & 0x1 == 0 => {
                // C.JR: jalr x0, rs1, 0.
                let rs1 = ((half >> 7) & 0x1f) as u8;
                self.pc = self.reg(rs1);
            }
            (3, 4) => {
                let rd = ((half >> 7) & 0x1f) as u8;
                let rs2 = ((half >> 2) & 0x1f) as u8;
                if rd == 0 && rs2 == 0 {
                    // C.EBREAK.
                    self.halted = true;
                } else {
                    // C.ADD.
                    let v = self.reg(rd).wrapping_add(self.reg(rs2));
                    self.set_reg(rd, v);
                }
                self.pc = next;
            }
            _ => return Err(RefError::IllegalCompressed { pc, half }),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_addi_and_halt() {
        let mut m = RefMachine::new();
        // addi x1, x0, 5; ecall.
        m.mem.write_u32(0, 0x00500093);
        m.mem.write_u32(4, 0x00000073);
        m.run(16).ok().unwrap();
        assert_eq!(m.reg(1), 5);
        assert!(m.halted);
    }

    #[test]
    fn div_edge_cases() {
        let mut m = RefMachine::new();
        // addi x1, x0, -1  => x1 = u64::MAX
        m.mem.write_u32(0, 0xFFF00093);
        // div x2, x1, x0 => -1 (divide BY zero): funct7=1, rs2=0, rs1=1,
        // funct3=4, rd=2, opcode=0x33.
        m.mem
            .write_u32(4, (1u32 << 25) | (1 << 15) | (4 << 12) | (2 << 7) | 0x33);
        // ecall
        m.mem.write_u32(8, 0x00000073);
        m.run(16).ok().unwrap();
        assert_eq!(m.reg(1), u64::MAX);
        assert_eq!(m.reg(2), u64::MAX);
    }
}
