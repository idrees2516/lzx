//! RV64IMAC decoder: canonical instruction enum for every supported
//! encoding, with compressed (C) instructions expanding to their base
//! micro-ops. One enum feeds execution, witness generation, and the
//! bytecode claim layer.

/// The canonical instruction set (subset exactly matching the trace
/// proving layer's supported semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    // RV64I integer register-immediate.
    Addi { rd: u8, rs1: u8, imm: i64 },
    Slti { rd: u8, rs1: u8, imm: i64 },
    Sltiu { rd: u8, rs1: u8, imm: u64 },
    Xori { rd: u8, rs1: u8, imm: i64 },
    Ori { rd: u8, rs1: u8, imm: i64 },
    Andi { rd: u8, rs1: u8, imm: i64 },
    Slli { rd: u8, rs1: u8, shamt: u8 },
    Srli { rd: u8, rs1: u8, shamt: u8 },
    Srai { rd: u8, rs1: u8, shamt: u8 },
    // RV64I register-register.
    Addiw { rd: u8, rs1: u8, imm: i64 },
    Add { rd: u8, rs1: u8, rs2: u8 },
    Sub { rd: u8, rs1: u8, rs2: u8 },
    Sll { rd: u8, rs1: u8, rs2: u8 },
    Slt { rd: u8, rs1: u8, rs2: u8 },
    Sltu { rd: u8, rs1: u8, rs2: u8 },
    Xor { rd: u8, rs1: u8, rs2: u8 },
    Srl { rd: u8, rs1: u8, rs2: u8 },
    Sra { rd: u8, rs1: u8, rs2: u8 },
    Or { rd: u8, rs1: u8, rs2: u8 },
    And { rd: u8, rs1: u8, rs2: u8 },
    Addw { rd: u8, rs1: u8, rs2: u8 },
    Subw { rd: u8, rs1: u8, rs2: u8 },
    Sllw { rd: u8, rs1: u8, rs2: u8 },
    Srlw { rd: u8, rs1: u8, rs2: u8 },
    Sraw { rd: u8, rs1: u8, rs2: u8 },
    // RV64I upper-immediate / PC-relative.
    Lui { rd: u8, imm: i64 },
    Auipc { rd: u8, imm: i64 },
    // RV64I loads/stores (word-granular kernel; the memory layer owns
    // subword expansion semantics).
    Lw { rd: u8, rs1: u8, imm: i64 },
    Lwu { rd: u8, rs1: u8, imm: i64 },
    Ld { rd: u8, rs1: u8, imm: i64 },
    Sw { rs1: u8, rs2: u8, imm: i64 },
    Sd { rs1: u8, rs2: u8, imm: i64 },
    // Branches.
    Beq { rs1: u8, rs2: u8, imm: i64 },
    Bne { rs1: u8, rs2: u8, imm: i64 },
    Blt { rs1: u8, rs2: u8, imm: i64 },
    Bge { rs1: u8, rs2: u8, imm: i64 },
    Bltu { rs1: u8, rs2: u8, imm: i64 },
    Bgeu { rs1: u8, rs2: u8, imm: i64 },
    // Jump and link.
    Jal { rd: u8, imm: i64 },
    Jalr { rd: u8, rs1: u8, imm: i64 },
    // System.
    Ecall,
    Ebreak,
    // RV64M multiply-divide.
    Mul { rd: u8, rs1: u8, rs2: u8 },
    Mulh { rd: u8, rs1: u8, rs2: u8 },
    Mulhu { rd: u8, rs1: u8, rs2: u8 },
    Div { rd: u8, rs1: u8, rs2: u8 },
    Divu { rd: u8, rs1: u8, rs2: u8 },
    Rem { rd: u8, rs1: u8, rs2: u8 },
    Remu { rd: u8, rs1: u8, rs2: u8 },
    Divw { rd: u8, rs1: u8, rs2: u8 },
    Divuw { rd: u8, rs1: u8, rs2: u8 },
    Remw { rd: u8, rs1: u8, rs2: u8 },
    Remuw { rd: u8, rs1: u8, rs2: u8 },
    Mulw { rd: u8, rs1: u8, rs2: u8 },
    // RV64A atomics (word-granular kernel).
    LrW { rd: u8, rs1: u8, aq: bool, rl: bool },
    ScW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoSwapW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoAddW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoXorW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoAndW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoOrW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoMinW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoMaxW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoMinuW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
    AmoMaxuW { rd: u8, rs1: u8, rs2: u8, aq: bool, rl: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrFormat {
    R,
    I,
    S,
    B,
    U,
    J,
    R4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    UnsupportedOpcode { pc: u64, word: u32 },
    UnsupportedFunct { pc: u64, word: u32 },
    IllegalCompressed { pc: u64, half: u16 },
}

#[inline]
fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// Decode one 32-bit instruction word at `pc`.
pub fn decode(pc: u64, word: u32) -> Result<Instr, DecodeError> {
    let opcode = word & 0x7f;
    let rd = ((word >> 7) & 0x1f) as u8;
    let funct3 = (word >> 12) & 0x7;
    let rs1 = ((word >> 15) & 0x1f) as u8;
    let rs2 = ((word >> 20) & 0x1f) as u8;
    let funct7 = (word >> 25) & 0x7f;
    match opcode {
        0x37 => Ok(Instr::Lui {
            rd,
            imm: sign_extend(word as u64 >> 12, 32) << 12,
        }),
        0x17 => Ok(Instr::Auipc {
            rd,
            imm: sign_extend(word as u64 >> 12, 32) << 12,
        }),
        0x6f => {
            let imm = ((word >> 31) << 20)
                | (((word >> 12) & 0xff) << 12)
                | (((word >> 20) & 0x1) << 11)
                | (((word >> 21) & 0x3ff) << 1);
            Ok(Instr::Jal {
                rd,
                imm: sign_extend(imm as u64, 21),
            })
        }
        0x67 => {
            let imm = (word as u64 >> 20) & 0xfff;
            Ok(Instr::Jalr {
                rd,
                rs1,
                imm: sign_extend(imm, 12),
            })
        }
        0x63 => {
            let imm = (((word >> 31) & 0x1) << 12)
                | (((word >> 7) & 0x1) << 11)
                | (((word >> 25) & 0x3f) << 5)
                | (((word >> 8) & 0xf) << 1);
            let imm = sign_extend(imm as u64, 13);
            Ok(match funct3 {
                0 => Instr::Beq { rs1, rs2, imm },
                1 => Instr::Bne { rs1, rs2, imm },
                4 => Instr::Blt { rs1, rs2, imm },
                5 => Instr::Bge { rs1, rs2, imm },
                6 => Instr::Bltu { rs1, rs2, imm },
                7 => Instr::Bgeu { rs1, rs2, imm },
                _ => {
                    return Err(DecodeError::UnsupportedFunct { pc, word });
                }
            })
        }
        0x03 => {
            let imm = sign_extend((word as u64 >> 20) & 0xfff, 12);
            Ok(match funct3 {
                2 => Instr::Lw { rd, rs1, imm },
                6 => Instr::Lwu { rd, rs1, imm },
                3 => Instr::Ld { rd, rs1, imm },
                _ => {
                    return Err(DecodeError::UnsupportedFunct { pc, word });
                }
            })
        }
        0x23 => {
            let imm = (((word >> 25) & 0x7f) << 5) | ((word >> 7) & 0x1f);
            let imm = sign_extend(imm as u64, 12);
            Ok(match funct3 {
                2 => Instr::Sw { rs1, rs2, imm },
                3 => Instr::Sd { rs1, rs2, imm },
                _ => {
                    return Err(DecodeError::UnsupportedFunct { pc, word });
                }
            })
        }
        0x13 => {
            let imm = sign_extend((word as u64 >> 20) & 0xfff, 12);
            Ok(match funct3 {
                0 => Instr::Addi { rd, rs1, imm },
                2 => Instr::Slti { rd, rs1, imm },
                3 => Instr::Sltiu { rd, rs1, imm: imm as u64 },
                4 => Instr::Xori { rd, rs1, imm },
                6 => Instr::Ori { rd, rs1, imm },
                7 => Instr::Andi { rd, rs1, imm },
                1 => Instr::Slli {
                    rd,
                    rs1,
                    shamt: ((word >> 20) & 0x3f) as u8,
                },
                5 => {
                    let shamt = ((word >> 20) & 0x3f) as u8;
                    match funct7 {
                        0 => Instr::Srli { rd, rs1, shamt },
                        0x20 => Instr::Srai { rd, rs1, shamt },
                        _ => {
                            return Err(DecodeError::UnsupportedFunct { pc, word });
                        }
                    }
                }
                _ => {
                    return Err(DecodeError::UnsupportedFunct { pc, word });
                }
            })
        }
        0x1b => {
            let imm = sign_extend((word as u64 >> 20) & 0xfff, 12);
            match funct3 {
                0 => Ok(Instr::Addiw { rd, rs1, imm }),
                1 => Ok(Instr::Slli {
                    rd,
                    rs1,
                    shamt: ((word >> 20) & 0x3f) as u8,
                }),
                5 => {
                    let shamt = ((word >> 20) & 0x3f) as u8;
                    match funct7 {
                        0 => Ok(Instr::Srli { rd, rs1, shamt }),
                        0x20 => Ok(Instr::Srai { rd, rs1, shamt }),
                        _ => Err(DecodeError::UnsupportedFunct { pc, word }),
                    }
                }
                _ => Err(DecodeError::UnsupportedFunct { pc, word }),
            }
        }
        0x33 => Ok(match (funct3, funct7) {
            (0, 0x00) => Instr::Add { rd, rs1, rs2 },
            (0, 0x20) => Instr::Sub { rd, rs1, rs2 },
            (1, 0x00) => Instr::Sll { rd, rs1, rs2 },
            (2, 0x00) => Instr::Slt { rd, rs1, rs2 },
            (3, 0x00) => Instr::Sltu { rd, rs1, rs2 },
            (4, 0x00) => Instr::Xor { rd, rs1, rs2 },
            (5, 0x00) => Instr::Srl { rd, rs1, rs2 },
            (5, 0x20) => Instr::Sra { rd, rs1, rs2 },
            (6, 0x00) => Instr::Or { rd, rs1, rs2 },
            (7, 0x00) => Instr::And { rd, rs1, rs2 },
            (0, 0x01) => Instr::Mul { rd, rs1, rs2 },
            (1, 0x01) => Instr::Mulh { rd, rs1, rs2 },
            (2, 0x01) => Instr::Mulhu { rd, rs1, rs2 },
            (4, 0x01) => Instr::Div { rd, rs1, rs2 },
            (5, 0x01) => Instr::Divu { rd, rs1, rs2 },
            (6, 0x01) => Instr::Rem { rd, rs1, rs2 },
            (7, 0x01) => Instr::Remu { rd, rs1, rs2 },
            _ => {
                return Err(DecodeError::UnsupportedFunct { pc, word });
            }
        }),
        0x3b => Ok(match (funct3, funct7) {
            (0, 0x00) => Instr::Addw { rd, rs1, rs2 },
            (0, 0x20) => Instr::Subw { rd, rs1, rs2 },
            (1, 0x00) => Instr::Sllw { rd, rs1, rs2 },
            (5, 0x00) => Instr::Srlw { rd, rs1, rs2 },
            (5, 0x20) => Instr::Sraw { rd, rs1, rs2 },
            (4, 0x01) => Instr::Divw { rd, rs1, rs2 },
            (5, 0x01) => Instr::Divuw { rd, rs1, rs2 },
            (6, 0x01) => Instr::Remw { rd, rs1, rs2 },
            (7, 0x01) => Instr::Remuw { rd, rs1, rs2 },
            (0, 0x01) => Instr::Mulw { rd, rs1, rs2 },
            _ => {
                return Err(DecodeError::UnsupportedFunct { pc, word });
            }
        }),
        0x0f => Err(DecodeError::UnsupportedFunct { pc, word }),
        0x2f => {
            // RV64A: funct5 in bits 27..32.
            let funct5 = (word >> 27) & 0x1f;
            let aq = (word >> 26) & 0x1 == 1;
            let rl = (word >> 25) & 0x1 == 1;
            match funct5 {
                0x02 => Ok(Instr::LrW { rd, rs1, aq, rl }),
                0x03 => Ok(Instr::ScW { rd, rs1, rs2, aq, rl }),
                0x01 => Ok(Instr::AmoSwapW { rd, rs1, rs2, aq, rl }),
                0x00 => Ok(Instr::AmoAddW { rd, rs1, rs2, aq, rl }),
                0x04 => Ok(Instr::AmoXorW { rd, rs1, rs2, aq, rl }),
                0x0c => Ok(Instr::AmoAndW { rd, rs1, rs2, aq, rl }),
                0x08 => Ok(Instr::AmoOrW { rd, rs1, rs2, aq, rl }),
                0x10 => Ok(Instr::AmoMinW { rd, rs1, rs2, aq, rl }),
                0x14 => Ok(Instr::AmoMaxW { rd, rs1, rs2, aq, rl }),
                0x18 => Ok(Instr::AmoMinuW { rd, rs1, rs2, aq, rl }),
                0x1c => Ok(Instr::AmoMaxuW { rd, rs1, rs2, aq, rl }),
                _ => Err(DecodeError::UnsupportedFunct { pc, word }),
            }
        }
        0x73 => match (funct3, word >> 20) {
            (0, 0) => Ok(Instr::Ecall),
            (0, 1) => Ok(Instr::Ebreak),
            _ => Err(DecodeError::UnsupportedFunct { pc, word }),
        },
        _ => Err(DecodeError::UnsupportedOpcode { pc, word }),
    }
}

/// Decode a compressed (16-bit) halfword into its canonical expansion
/// plus the NEXT pc (pc + 2 for compressed).
pub fn decode_compressed(pc: u64, half: u16) -> Result<(Instr, u64), DecodeError> {
    let quadrant = half & 0x3;
    let funct3 = (half >> 13) & 0x7;
    let rs1c = ((half >> 7) & 0x7) as u8; // x8..x15
    let expand = |r: u8| r + 8;
    match (quadrant, funct3) {
        // C.ADDI: addi rd, rd, nzimm.
        (0, 0) => {
            let imm = ((((half >> 12) & 0x1) as u64) << 5)
                | (((half >> 2) & 0x1f) as u64);
            let rd = ((half >> 7) & 0x1f) as u8;
            if rd == 0 && imm == 0 {
                Ok((Instr::Ebreak, pc + 2))
            } else {
                Ok((Instr::Addi {
                    rd,
                    rs1: rd,
                    imm: sign_extend(imm, 6),
                }, pc + 2))
            }
        }
        // C.ADDI4SPN.
        (0, 2) => {
            let imm = ((((half >> 12) & 0x1) as u64) << 3)
                | ((((half >> 5) & 0x3) as u64) << 6)
                | ((((half >> 2) & 0x7) as u64) << 4)
                | ((((half >> 3) & 0x3) as u64) << 2);
            let rd = expand(rs1c);
            Ok((
                Instr::Addi {
                    rd,
                    rs1: 2,
                    imm: sign_extend(imm, 7),
                },
                pc + 2,
            ))
        }
        // C.LW.
        (0, 5) => {
            let imm = ((((half >> 10) & 0x7) as u64) << 3)
                | ((((half >> 5) & 0x1) as u64) << 2)
                | ((((half >> 6) & 0x1) as u64) << 1);
            Ok((
                Instr::Lw {
                    rd: expand(rs1c),
                    rs1: expand(((half >> 10) & 0x7) as u8),
                    imm: imm as i64,
                },
                pc + 2,
            ))
        }
        // C.LI.
        (1, 2) => {
            let imm = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
            Ok((
                Instr::Addi {
                    rd: ((half >> 7) & 0x1f) as u8,
                    rs1: 0,
                    imm: sign_extend(imm, 6),
                },
                pc + 2,
            ))
        }
        // C.ADDI16SP / C.LUI share funct3=2; distinguish by rd.
        (1, 1) => {
            let rd = ((half >> 7) & 0x1f) as u8;
            if rd == 2 {
                // C.ADDI16SP.
                let imm = ((((half >> 12) & 0x1) as u64) << 9)
                    | ((((half >> 5) & 0x1) as u64) << 4)
                    | ((((half >> 2) & 0x3) as u64) << 7)
                    | ((((half >> 6) & 0x1) as u64) << 6)
                    | ((((half >> 3) & 0x1) as u64) << 5);
                Ok((
                    Instr::Addi {
                        rd: 2,
                        rs1: 2,
                        imm: sign_extend(imm, 10),
                    },
                    pc + 2,
                ))
            } else {
                // C.LUI.
                let imm = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
                Ok((
                    Instr::Lui {
                        rd,
                        imm: sign_extend(imm, 6) << 12,
                    },
                    pc + 2,
                ))
            }
        }
        // C.SUB / C.XOR / C.OR / C.AND (funct3=4 with bit12 set).
        (1, 4) if (half >> 12) & 0x1 == 1 => {
            let rd = expand(rs1c);
            let rs2 = expand(((half >> 2) & 0x7) as u8);
            match (half >> 5) & 0x3 {
                0 => Ok((Instr::Sub { rd, rs1: rd, rs2 }, pc + 2)),
                1 => Ok((Instr::Xor { rd, rs1: rd, rs2 }, pc + 2)),
                2 => Ok((Instr::Or { rd, rs1: rd, rs2 }, pc + 2)),
                _ => Ok((Instr::And { rd, rs1: rd, rs2 }, pc + 2)),
            }
        }
        // C.SRLI / C.SRAI / C.ANDI.
        (1, 4) => {
            let rd = expand(rs1c);
            let shamt = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
            Ok((
                Instr::Srli {
                    rd,
                    rs1: rd,
                    shamt: shamt as u8,
                },
                pc + 2,
            ))
        }
        (1, 5) => {
            let rd = expand(rs1c);
            let shamt = ((((half >> 12) & 0x1) as u64) << 5) | (((half >> 2) & 0x1f) as u64);
            Ok((
                Instr::Srai {
                    rd,
                    rs1: rd,
                    shamt: shamt as u8,
                },
                pc + 2,
            ))
        }
        // C.SW / C.SD.
        (2, 6) => {
            let rs1 = expand(rs1c);
            let rs2 = expand(((half >> 2) & 0x7) as u8);
            let imm = ((((half >> 10) & 0x7) as u64) << 3) | ((((half >> 5) & 0x3) as u64) << 1);
            Ok((Instr::Sw { rs1, rs2, imm: imm as i64 }, pc + 2))
        }
        (2, 3) => {
            let rs1 = expand(rs1c);
            let rs2 = expand(((half >> 2) & 0x7) as u8);
            let imm = ((((half >> 10) & 0x7) as u64) << 3) | ((((half >> 5) & 0x3) as u64) << 1);
            Ok((Instr::Sd { rs1, rs2, imm: imm as i64 }, pc + 2))
        }
        // C.BEQZ.
        (1, 6) => {
            // funct3 6 = C.BEQZ? (Actually C.BEQZ is funct3 6 in quadrant 1.)
            let rs1 = expand(rs1c);
            let imm = ((((half >> 12) & 0x1) as u64) << 8)
                | ((((half >> 5) & 0x3) as u64) << 3)
                | ((((half >> 2) & 0x1) as u64) << 2)
                | ((((half >> 10) & 0x3) as u64) << 6)
                | ((((half >> 3) & 0x3) as u64) << 1);
            Ok((
                Instr::Beq {
                    rs1,
                    rs2: 0,
                    imm: sign_extend(imm, 9),
                },
                pc + 2,
            ))
        }
        (1, 7) => {
            let rs1 = expand(rs1c);
            let imm = ((((half >> 12) & 0x1) as u64) << 8)
                | ((((half >> 5) & 0x3) as u64) << 3)
                | ((((half >> 2) & 0x1) as u64) << 2)
                | ((((half >> 10) & 0x3) as u64) << 6)
                | ((((half >> 3) & 0x3) as u64) << 1);
            Ok((
                Instr::Bne {
                    rs1,
                    rs2: 0,
                    imm: sign_extend(imm, 9),
                },
                pc + 2,
            ))
        }
        // C.J / C.JR / C.JALR / C.EBREAK handled via quadrant 3.
        (3, 4) if (half >> 12) & 0x1 == 0 => {
            // C.JR: jalr x0, rs1, 0.
            Ok((
                Instr::Jalr {
                    rd: 0,
                    rs1: ((half >> 7) & 0x1f) as u8,
                    imm: 0,
                },
                pc + 2,
            ))
        }
        (3, 4) => {
            let rd = ((half >> 7) & 0x1f) as u8;
            let rs2 = ((half >> 2) & 0x1f) as u8;
            if rd == 0 && rs2 == 0 {
                Ok((Instr::Ebreak, pc + 2))
            } else {
                // C.ADD: add rd, rd, rs2.
                Ok((Instr::Add { rd, rs1: rd, rs2 }, pc + 2))
            }
        }
        _ => Err(DecodeError::IllegalCompressed { pc, half }),
    }
}

// Fence placeholder mapping (we keep a minimal surface for Fence/Ecall).
impl Instr {
    /// Instruction byte length.
    pub fn width(&self) -> u32 {
        match self {
            Instr::Ecall | Instr::Ebreak => 4,
            _ => 4,
        }
    }
}
