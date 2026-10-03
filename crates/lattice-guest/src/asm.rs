//! Minimal but complete RV64IM assembler targeting the `lattice-vm`
//! guest semantics (RV64IMAC subset, uncompressed 4-byte instructions
//! only, word-granular sparse memory).
//!
//! Two front-ends share one encoding core:
//!
//! * a builder API (`Assembler`) with labels, two-pass fixups, sections,
//!   and one method per supported instruction;
//! * a small text assembler (`assemble_str`) with GAS-ish syntax
//!   (`li x1, 5`, `lw t0, 8(sp)`, `beq a0, zero, loop`, `.word`,
//!   `.byte`, `.data`, `.text`, `.org`) so guest programs stay readable.
//!
//! Encodings follow the RISC-V unprivileged spec and were checked
//! instruction-by-instruction against `lattice_vm::decode::decode`
//! (see the `decode_table` test — every emitted word round-trips
//! through the VM's own decoder).
//!
//! # Section / placement model
//!
//! Code is emitted first (byte address 0 = first instruction). Data may
//! be *inline* (appended after the code at `data_base = align8(len)`,
//! the default) or *placed* at an explicit address via
//! [`Assembler::place_data_at`] (text directive `.org`). `la` is
//! pc-relative (`auipc`+`addi`) and works in both modes; `la_abs`
//! materializes the absolute address and requires explicit placement.

use lattice_vm::{exec, MachineState};

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Assembler error: structured, with a `Display` that explains the
/// constraint that was violated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsmError {
    /// Register index out of 0..=31.
    RegisterOutOfRange(u8),
    /// Immediate outside the representable range for its format.
    ImmOutOfRange {
        what: &'static str,
        value: i64,
        lo: i64,
        hi: i64,
    },
    /// Shift amount outside 0..=63 (or 0..=31 for the W variants).
    ShiftOutOfRange {
        what: &'static str,
        value: i64,
        hi: i64,
    },
    /// Branch/jump offset not a multiple of 2.
    OddOffset(i64),
    /// Reference to a label that was never bound.
    UndefinedLabel(String),
    /// Label bound more than once.
    DuplicateLabel(String),
    /// Instruction emitted while the data section is active.
    InstrInData,
    /// `la_abs` on a label whose address is not known at emission time.
    LabelNotResolvable(String),
    /// Data placement changed after data bytes were emitted.
    PlacementAfterData,
    /// Text-assembler parse error, with 1-based line number.
    Parse {
        line: usize,
        msg: String,
    },
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsmError::RegisterOutOfRange(r) => {
                write!(f, "register index {r} out of range 0..=31")
            }
            AsmError::ImmOutOfRange { what, value, lo, hi } => {
                write!(f, "immediate for {what} out of range: {value} not in [{lo}, {hi}]")
            }
            AsmError::ShiftOutOfRange { what, value, hi } => {
                write!(f, "shift amount for {what} out of range: {value} not in [0, {hi}]")
            }
            AsmError::OddOffset(off) => {
                write!(f, "branch/jump offset {off} is not a multiple of 2")
            }
            AsmError::UndefinedLabel(name) => write!(f, "undefined label '{name}'"),
            AsmError::DuplicateLabel(name) => write!(f, "label '{name}' bound twice"),
            AsmError::InstrInData => {
                write!(f, "instruction emitted while the data section is active")
            }
            AsmError::LabelNotResolvable(name) => write!(
                f,
                "la_abs: address of '{name}' is not fixed at emission time \
                 (bind it in a data section placed with place_data_at/.org)"
            ),
            AsmError::PlacementAfterData => {
                write!(f, "data placement changed after data bytes were emitted")
            }
            AsmError::Parse { line, msg } => write!(f, "line {line}: {msg}"),
        }
    }
}

impl std::error::Error for AsmError {}

// ---------------------------------------------------------------------------
// Labels / targets / program
// ---------------------------------------------------------------------------

/// Opaque label id returned by [`Assembler::label`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(pub(crate) usize);

/// Branch/jump target: either a raw byte offset from the instruction, or a
/// label id resolved at `finish()` time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Byte offset relative to the branch/jump instruction.
    Rel(i64),
    /// Label id.
    Label(LabelId),
}

impl From<i64> for Target {
    fn from(v: i64) -> Self {
        Target::Rel(v)
    }
}

impl From<LabelId> for Target {
    fn from(l: LabelId) -> Self {
        Target::Label(l)
    }
}

/// Result of `finish()`: everything needed to load and run the program.
#[derive(Debug, Clone, Default)]
pub struct AssembledProgram {
    /// Byte image loaded at address 0 (instructions, plus inline data
    /// when the data section is in inline mode).
    pub code: Vec<u8>,
    /// Separate data image (non-empty only for explicit `place_data_at`
    /// placement; loaded at `data_base`).
    pub data: Vec<u8>,
    /// Byte addresses of every bound label (code and data).
    pub labels: HashMap<String, u64>,
    /// Address the data section was placed at (inline mode: first byte
    /// after the aligned code image).
    pub data_base: u64,
}

impl AssembledProgram {
    /// Load segments: `(address, bytes)` pairs covering code and data.
    pub fn segments(&self) -> Vec<(u64, &[u8])> {
        let mut segs = vec![(0u64, self.code.as_slice())];
        if !self.data.is_empty() {
            segs.push((self.data_base, self.data.as_slice()));
        }
        segs
    }
}

// ---------------------------------------------------------------------------
// Encoding primitives (checked against lattice-vm's decoder)
// ---------------------------------------------------------------------------

const OPC_LUI: u32 = 0x37;
const OPC_AUIPC: u32 = 0x17;
const OPC_JAL: u32 = 0x6f;
const OPC_JALR: u32 = 0x67;
const OPC_BRANCH: u32 = 0x63;
const OPC_LOAD: u32 = 0x03;
const OPC_STORE: u32 = 0x23;
const OPC_OPIMM: u32 = 0x13;
const OPC_OPIMM32: u32 = 0x1b;
const OPC_OP: u32 = 0x33;
const OPC_OP32: u32 = 0x3b;
const OPC_SYSTEM: u32 = 0x73;
const OPC_AMO: u32 = 0x2f;

#[inline]
fn check_reg(r: u8) -> Result<(), AsmError> {
    if r > 31 {
        Err(AsmError::RegisterOutOfRange(r))
    } else {
        Ok(())
    }
}

#[inline]
fn check_imm(what: &'static str, v: i64, lo: i64, hi: i64) -> Result<(), AsmError> {
    if !(lo..=hi).contains(&v) {
        Err(AsmError::ImmOutOfRange { what, value: v, lo, hi })
    } else {
        Ok(())
    }
}

#[inline]
fn check_shift(what: &'static str, v: u8, hi: u8) -> Result<(), AsmError> {
    if v > hi {
        Err(AsmError::ShiftOutOfRange { what, value: v as i64, hi: hi as i64 })
    } else {
        Ok(())
    }
}

#[inline]
fn enc_r(opcode: u32, rd: u8, f3: u32, rs1: u8, rs2: u8, f7: u32) -> u32 {
    (f7 << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

#[inline]
fn enc_i(opcode: u32, rd: u8, f3: u32, rs1: u8, imm: i64) -> u32 {
    (((imm as u32) & 0xFFF) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | ((rd as u32) << 7)
        | opcode
}

#[inline]
fn enc_s(opcode: u32, f3: u32, rs1: u8, rs2: u8, imm: i64) -> u32 {
    let imm = imm as u32;
    (((imm >> 5) & 0x7F) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
}

#[inline]
fn enc_b(opcode: u32, f3: u32, rs1: u8, rs2: u8, imm: i64) -> u32 {
    let imm = imm as u32;
    (((imm >> 12) & 0x1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (f3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 0x1) << 7)
        | opcode
}

#[inline]
fn enc_j(opcode: u32, rd: u8, imm: i64) -> u32 {
    let imm = imm as u32;
    (((imm >> 20) & 0x1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 0x1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | ((rd as u32) << 7)
        | opcode
}

#[inline]
fn enc_u(opcode: u32, rd: u8, imm20: i64) -> u32 {
    (((imm20 as u32) & 0xFFFFF) << 12) | ((rd as u32) << 7) | opcode
}

#[inline]
fn enc_amo(f5: u32, rd: u8, rs2: u8, rs1: u8, aq: bool, rl: bool) -> u32 {
    (f5 << 27)
        | ((aq as u32) << 26)
        | ((rl as u32) << 25)
        | ((rs2 as u32) << 20)
        | ((rs1 as u32) << 15)
        | (2 << 12)
        | ((rd as u32) << 7)
        | OPC_AMO
}

// ---------------------------------------------------------------------------
// Assembler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Text,
    Data,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataMode {
    /// Data appended after code (base = align8(code len)).
    Inline,
    /// Data placed at a fixed address.
    At(u64),
}

#[derive(Debug, Clone)]
enum Fixup {
    /// B-type immediate at instruction index `at`, from label delta.
    Branch { at: usize, label: LabelId },
    /// J-type immediate at instruction index `at`.
    Jal { at: usize, label: LabelId },
    /// `auipc` U-immediate at `at` for the pc-relative pair; the matching
    /// addi (PcrelLo) carries `anchor` = this instruction's index.
    PcrelHi { at: usize, label: LabelId },
    /// `addi` I-immediate at `at`, low half of the delta from the `auipc`
    /// at instruction index `anchor`.
    PcrelLo { at: usize, anchor: usize, label: LabelId },
}

/// The RV64IM assembler.
#[derive(Debug, Clone)]
pub struct Assembler {
    section: Section,
    /// Text section: 4-byte words, little-endian in the final image.
    code: Vec<u32>,
    /// Data section bytes.
    data: Vec<u8>,
    data_mode: DataMode,
    /// Bound offset (bytes) per label, plus the section it was bound in.
    bound: Vec<Option<(u64, Section)>>,
    names: HashMap<String, LabelId>,
    fixups: Vec<Fixup>,
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Assembler {
    /// New assembler emitting into the text section, inline data mode.
    pub fn new() -> Self {
        Assembler {
            section: Section::Text,
            code: Vec::new(),
            data: Vec::new(),
            data_mode: DataMode::Inline,
            bound: Vec::new(),
            names: HashMap::new(),
            fixups: Vec::new(),
        }
    }

    // -- sections / labels --------------------------------------------------

    /// Switch to the text (code) section.
    pub fn text(&mut self) -> &mut Self {
        self.section = Section::Text;
        self
    }

    /// Switch to the data section.
    pub fn data(&mut self) -> &mut Self {
        self.section = Section::Data;
        self
    }

    /// Place the data section at an explicit address (directive `.org`).
    /// Must be called before any data bytes are emitted.
    pub fn place_data_at(&mut self, addr: u64) -> Result<&mut Self, AsmError> {
        if !self.data.is_empty() {
            return Err(AsmError::PlacementAfterData);
        }
        self.data_mode = DataMode::At(addr);
        self.section = Section::Data;
        Ok(self)
    }

    /// Create (or look up) a label by name; returns its id for forward
    /// references in branches/jumps/`la`/`la_abs`.
    pub fn label(&mut self, name: &str) -> LabelId {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        let id = LabelId(self.bound.len());
        self.bound.push(None);
        self.names.insert(name.to_string(), id);
        id
    }

    /// Bind a label at the current position of the current section.
    pub fn bind(&mut self, label: LabelId) -> Result<&mut Self, AsmError> {
        if self.bound[label.0].is_some() {
            let name = self
                .names
                .iter()
                .find(|(_, &id)| id == label)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| format!("label#{}", label.0));
            return Err(AsmError::DuplicateLabel(name));
        }
        let off = match self.section {
            Section::Text => (self.code.len() as u64) * 4,
            Section::Data => self.data.len() as u64,
        };
        self.bound[label.0] = Some((off, self.section));
        Ok(self)
    }

    /// Create and bind a named label at the current position.
    pub fn bind_name(&mut self, name: &str) -> Result<&mut Self, AsmError> {
        let id = self.label(name);
        self.bind(id)
    }

    /// Current text-section byte offset (the pc the next instruction
    /// would get).
    pub fn current_pc(&self) -> u64 {
        (self.code.len() as u64) * 4
    }

    /// Bytes emitted into the data section so far.
    pub fn data_len(&self) -> u64 {
        self.data.len() as u64
    }

    // -- raw / data emission -------------------------------------------------

    /// Emit a raw 32-bit word into the text section. (Caller beware: a
    /// word whose low two bits are not `0b11` would be fetched as a
    /// compressed instruction by the VM.)
    pub fn raw(&mut self, word: u32) -> Result<&mut Self, AsmError> {
        self.require_text()?;
        self.code.push(word);
        Ok(self)
    }

    /// `.word`: emit a 32-bit little-endian word into the current section.
    pub fn word(&mut self, w: u32) -> Result<&mut Self, AsmError> {
        match self.section {
            Section::Text => {
                self.code.push(w);
            }
            Section::Data => {
                self.data.extend_from_slice(&w.to_le_bytes());
            }
        }
        Ok(self)
    }

    /// `.dword`: emit a 64-bit little-endian word into the current section.
    pub fn word64(&mut self, w: u64) -> Result<&mut Self, AsmError> {
        match self.section {
            Section::Text => {
                self.code.push(w as u32);
                self.code.push((w >> 32) as u32);
            }
            Section::Data => {
                self.data.extend_from_slice(&w.to_le_bytes());
            }
        }
        Ok(self)
    }

    /// `.byte`: emit one byte into the data section (text-section bytes
    /// are not supported; instructions are 4-byte aligned).
    pub fn byte(&mut self, b: u8) -> Result<&mut Self, AsmError> {
        match self.section {
            Section::Text => {
                // Pad into the code stream is unsafe for alignment; refuse.
                Err(AsmError::Parse {
                    line: 0,
                    msg: "byte emission requires the data section".to_string(),
                })
            }
            Section::Data => {
                self.data.push(b);
                Ok(self)
            }
        }
    }

    /// Emit a byte slice into the data section.
    pub fn bytes(&mut self, bs: &[u8]) -> Result<&mut Self, AsmError> {
        for &b in bs {
            self.byte(b)?;
        }
        Ok(self)
    }

    // -- pseudo-instructions --------------------------------------------------

    /// `li rd, imm` — materialize any i64 constant.
    ///
    /// Expansion: `addi` (12-bit), `lui`+`addi` (signed 32-bit, with a
    /// `lui`+`addiw` wrap-around special case for the top 2048 positive
    /// values, where the rounded `imm20` would hit +2^19 — outside the
    /// signed 20-bit field — but the *same bit pattern* as -2^19 loads
    /// -2^31 and `addiw` wraps the low bits back into range, exactly the
    /// GAS expansion), or a build-up of signed 12-bit chunks separated by
    /// `slli 12` for full 64-bit constants (≤ 11 instructions, exact
    /// under mod-2^64).
    pub fn li(&mut self, rd: u8, value: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        if (-2048..=2047).contains(&value) {
            return self.addi(rd, 0, value);
        }
        // Signed 32-bit fast path: value = hi20*4096 + lo12.
        if (i32::MIN as i64..=i32::MAX as i64).contains(&value) {
            let hi = (value + 0x800) >> 12;
            let lo = value - (hi << 12);
            if hi == (1 << 19) {
                // value in [2^31 - 2048, 2^31 - 1]: imm20 = 0x80000 (bit 19
                // set) decodes as -2^19 -> rd = -2^31 (lattice-vm sign-
                // extends the 32-bit LUI value), and addiw's 32-bit wrap
                // brings rd back into [2^31 - 2048, 2^31 - 1].
                self.lui(rd, hi - (1 << 20))?;
                self.addiw(rd, rd, lo)?;
            } else {
                self.lui(rd, hi)?;
                if lo != 0 {
                    self.addi(rd, rd, lo)?;
                }
            }
            return Ok(self);
        }
        // Full 64-bit: signed 12-bit chunk decomposition v = sum d_i * 4096^i.
        let v = value as u64;
        let mut d = [0i64; 6];
        let mut rem = v;
        for i in 0..6 {
            let low = (rem & 0xFFF) as i64;
            let di = if low >= 2048 { low - 4096 } else { low };
            d[i] = di;
            rem = (rem.wrapping_sub(di as u64)) >> 12;
        }
        let top = (0..6).rev().find(|&i| d[i] != 0).unwrap_or(0);
        // Horner from the top chunk: every position below `top` must be
        // shifted in, even when its chunk is zero (the low zero chunks
        // carry the factor 4096^i).
        self.addi(rd, 0, d[top])?;
        for j in (0..top).rev() {
            self.slli(rd, rd, 12)?;
            if d[j] != 0 {
                self.addi(rd, rd, d[j])?;
            }
        }
        Ok(self)
    }

    /// `mv rd, rs` = `addi rd, rs, 0`.
    pub fn mv(&mut self, rd: u8, rs: u8) -> Result<&mut Self, AsmError> {
        self.addi(rd, rs, 0)
    }

    /// `nop` = `addi x0, x0, 0`.
    pub fn nop(&mut self) -> Result<&mut Self, AsmError> {
        self.addi(0, 0, 0)
    }

    /// `j target` = `jal x0, target`.
    pub fn j(&mut self, target: Target) -> Result<&mut Self, AsmError> {
        self.jal(0, target)
    }

    /// `call target` = `jal x1, target` (link in ra).
    pub fn call(&mut self, target: Target) -> Result<&mut Self, AsmError> {
        self.jal(1, target)
    }

    /// `ret` = `jalr x0, x1, 0`.
    pub fn ret(&mut self) -> Result<&mut Self, AsmError> {
        self.jalr(0, 1, 0)
    }

    /// `jr rs` = `jalr x0, rs, 0`.
    pub fn jr(&mut self, rs: u8) -> Result<&mut Self, AsmError> {
        self.jalr(0, rs, 0)
    }

    /// `beqz rs, target` = `beq rs, x0, target`.
    pub fn beqz(&mut self, rs: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.beq(rs, 0, target)
    }

    /// `bnez rs, target` = `bne rs, x0, target`.
    pub fn bnez(&mut self, rs: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.bne(rs, 0, target)
    }

    /// `la rd, label` — pc-relative `auipc`+`addi` pair. Works for code
    /// and data labels, inline or placed data.
    pub fn la(&mut self, rd: u8, label: LabelId) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        self.require_text()?;
        let at = self.code.len();
        // Placeholder auipc (patched in finish()).
        self.code.push(enc_u(OPC_AUIPC, rd, 0));
        let fix_at = self.code.len();
        self.code.push(enc_i(OPC_OPIMM, rd, 0, rd, 0));
        self.fixups.push(Fixup::PcrelHi { at, label });
        self.fixups.push(Fixup::PcrelLo { at: fix_at, anchor: at, label });
        Ok(self)
    }

    /// `la_abs rd, label` — absolute address via `li`. Only valid when the
    /// label lives in a data section placed at a fixed address (the
    /// address is baked in at emission time).
    pub fn la_abs(&mut self, rd: u8, label: LabelId) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        let DataMode::At(base) = self.data_mode else {
            return Err(AsmError::LabelNotResolvable(self.label_name(label)));
        };
        let Some((off, Section::Data)) = self.bound[label.0] else {
            return Err(AsmError::LabelNotResolvable(self.label_name(label)));
        };
        let addr = base.wrapping_add(off);
        self.li(rd, addr as i64)
    }

    fn label_name(&self, label: LabelId) -> String {
        self.names
            .iter()
            .find(|(_, &id)| id == label)
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| format!("label#{}", label.0))
    }

    // -- branches / jumps ------------------------------------------------------

    fn emit_branch(
        &mut self,
        f3: u32,
        rs1: u8,
        rs2: u8,
        target: Target,
    ) -> Result<(), AsmError> {
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        match target {
            Target::Rel(off) => {
                check_imm("branch offset", off, -4096, 4094)?;
                if off & 1 != 0 {
                    return Err(AsmError::OddOffset(off));
                }
                self.code.push(enc_b(OPC_BRANCH, f3, rs1, rs2, off));
            }
            Target::Label(l) => {
                let at = self.code.len();
                self.code
                    .push(enc_b(OPC_BRANCH, f3, rs1, rs2, 0));
                self.fixups.push(Fixup::Branch { at, label: l });
            }
        }
        Ok(())
    }

    /// `beq rs1, rs2, target`
    pub fn beq(&mut self, rs1: u8, rs2: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.emit_branch(0, rs1, rs2, target)?;
        Ok(self)
    }

    /// `bne rs1, rs2, target`
    pub fn bne(&mut self, rs1: u8, rs2: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.emit_branch(1, rs1, rs2, target)?;
        Ok(self)
    }

    /// `blt rs1, rs2, target`
    pub fn blt(&mut self, rs1: u8, rs2: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.emit_branch(4, rs1, rs2, target)?;
        Ok(self)
    }

    /// `bge rs1, rs2, target`
    pub fn bge(&mut self, rs1: u8, rs2: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.emit_branch(5, rs1, rs2, target)?;
        Ok(self)
    }

    /// `bltu rs1, rs2, target`
    pub fn bltu(&mut self, rs1: u8, rs2: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.emit_branch(6, rs1, rs2, target)?;
        Ok(self)
    }

    /// `bgeu rs1, rs2, target`
    pub fn bgeu(&mut self, rs1: u8, rs2: u8, target: Target) -> Result<&mut Self, AsmError> {
        self.emit_branch(7, rs1, rs2, target)?;
        Ok(self)
    }

    /// `jal rd, target` (J-type, ±1 MiB).
    pub fn jal(&mut self, rd: u8, target: Target) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        self.require_text()?;
        match target {
            Target::Rel(off) => {
                check_imm("jal offset", off, -(1 << 20), (1 << 20) - 2)?;
                if off & 1 != 0 {
                    return Err(AsmError::OddOffset(off));
                }
                self.code.push(enc_j(OPC_JAL, rd, off));
            }
            Target::Label(l) => {
                let at = self.code.len();
                self.code.push(enc_j(OPC_JAL, rd, 0));
                self.fixups.push(Fixup::Jal { at, label: l });
            }
        }
        Ok(self)
    }

    /// `jalr rd, rs1, imm` (target = (rs1 + imm) & !1).
    pub fn jalr(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_imm("jalr imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_JALR, rd, 0, rs1, imm));
        Ok(self)
    }

    // -- upper-immediate --------------------------------------------------------

    /// `lui rd, imm20`: imm20 is the signed 20-bit *field*; the loaded
    /// value is the sign-extension of `imm20 << 12` as a 32-bit value
    /// (matching `lattice-vm`).
    pub fn lui(&mut self, rd: u8, imm20: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_imm("lui imm20", imm20, -(1 << 19), (1 << 19) - 1)?;
        self.require_text()?;
        self.code.push(enc_u(OPC_LUI, rd, imm20));
        Ok(self)
    }

    /// `auipc rd, imm20` (same immediate convention as `lui`).
    pub fn auipc(&mut self, rd: u8, imm20: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_imm("auipc imm20", imm20, -(1 << 19), (1 << 19) - 1)?;
        self.require_text()?;
        self.code.push(enc_u(OPC_AUIPC, rd, imm20));
        Ok(self)
    }

    // -- register-immediate ------------------------------------------------------

    fn emit_i(
        &mut self,
        f3: u32,
        rd: u8,
        rs1: u8,
        imm: i64,
    ) -> Result<(), AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_imm("i-type imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_OPIMM, rd, f3, rs1, imm));
        Ok(())
    }

    /// `addi rd, rs1, imm`
    pub fn addi(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        self.emit_i(0, rd, rs1, imm)?;
        Ok(self)
    }

    /// `slti rd, rs1, imm`
    pub fn slti(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        self.emit_i(2, rd, rs1, imm)?;
        Ok(self)
    }

    /// `sltiu rd, rs1, imm` (imm treated as the sign-extended 12-bit
    /// unsigned comparand, per spec).
    pub fn sltiu(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        self.emit_i(3, rd, rs1, imm)?;
        Ok(self)
    }

    /// `xori rd, rs1, imm`
    pub fn xori(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        self.emit_i(4, rd, rs1, imm)?;
        Ok(self)
    }

    /// `ori rd, rs1, imm`
    pub fn ori(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        self.emit_i(6, rd, rs1, imm)?;
        Ok(self)
    }

    /// `andi rd, rs1, imm`
    pub fn andi(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        self.emit_i(7, rd, rs1, imm)?;
        Ok(self)
    }

    /// `slli rd, rs1, shamt` (6-bit shamt).
    pub fn slli(&mut self, rd: u8, rs1: u8, shamt: u8) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_shift("slli", shamt, 63)?;
        self.require_text()?;
        // funct6 = 0.
        self.code.push(enc_i(OPC_OPIMM, rd, 1, rs1, shamt as i64));
        Ok(self)
    }

    /// `srli rd, rs1, shamt` (6-bit shamt).
    pub fn srli(&mut self, rd: u8, rs1: u8, shamt: u8) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_shift("srli", shamt, 63)?;
        self.require_text()?;
        // funct6 = 0.
        self.code.push(enc_i(OPC_OPIMM, rd, 5, rs1, shamt as i64));
        Ok(self)
    }

    /// `srai rd, rs1, shamt` (6-bit shamt).
    pub fn srai(&mut self, rd: u8, rs1: u8, shamt: u8) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_shift("srai", shamt, 63)?;
        self.require_text()?;
        // funct6 = 0b010000 at bits 31:26; shamt at bits 25:20.
        let word = (0x10u32 << 26)
            | ((shamt as u32) << 20)
            | ((rs1 as u32) << 15)
            | (5 << 12)
            | ((rd as u32) << 7)
            | OPC_OPIMM;
        self.code.push(word);
        Ok(self)
    }

    // -- register-register (RV64I + M) ------------------------------------------

    fn emit_r(
        &mut self,
        opcode: u32,
        f3: u32,
        f7: u32,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) -> Result<(), AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_r(opcode, rd, f3, rs1, rs2, f7));
        Ok(())
    }

    /// `add rd, rs1, rs2`
    pub fn add(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 0, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `sub rd, rs1, rs2`
    pub fn sub(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 0, 0x20, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `sll rd, rs1, rs2`
    pub fn sll(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 1, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `slt rd, rs1, rs2`
    pub fn slt(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 2, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `sltu rd, rs1, rs2`
    pub fn sltu(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 3, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `xor rd, rs1, rs2`
    pub fn xor(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 4, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `srl rd, rs1, rs2`
    pub fn srl(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 5, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `sra rd, rs1, rs2`
    pub fn sra(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 5, 0x20, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `or rd, rs1, rs2`
    pub fn or(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 6, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `and rd, rs1, rs2`
    pub fn and(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 7, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `mul rd, rs1, rs2`
    pub fn mul(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 0, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `mulh rd, rs1, rs2` (signed x signed high)
    pub fn mulh(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 1, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `mulhu rd, rs1, rs2` (unsigned x unsigned high) — funct3 = 3
    /// per the M-spec (the previous f3 = 2 was MULHSU's slot, paired
    /// with the old decoder's swapped mapping).
    pub fn mulhu(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 3, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `div rd, rs1, rs2`
    pub fn div(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 4, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `divu rd, rs1, rs2`
    pub fn divu(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 5, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `rem rd, rs1, rs2`
    pub fn rem(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 6, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `remu rd, rs1, rs2`
    pub fn remu(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP, 7, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    // -- W-suffixed (32-bit) operations ------------------------------------------

    /// `addiw rd, rs1, imm`
    pub fn addiw(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_imm("addiw imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_OPIMM32, rd, 0, rs1, imm));
        Ok(self)
    }

    /// `slliw rd, rs1, shamt` (5-bit shamt).
    pub fn slliw(&mut self, rd: u8, rs1: u8, shamt: u8) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_shift("slliw", shamt, 31)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_OPIMM32, rd, 1, rs1, shamt as i64));
        Ok(self)
    }

    /// `srliw rd, rs1, shamt` (5-bit shamt).
    pub fn srliw(&mut self, rd: u8, rs1: u8, shamt: u8) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_shift("srliw", shamt, 31)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_OPIMM32, rd, 5, rs1, shamt as i64));
        Ok(self)
    }

    /// `sraiw rd, rs1, shamt` (5-bit shamt).
    pub fn sraiw(&mut self, rd: u8, rs1: u8, shamt: u8) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_shift("sraiw", shamt, 31)?;
        self.require_text()?;
        let word = (0x20u32 << 25)
            | ((shamt as u32) << 20)
            | ((rs1 as u32) << 15)
            | (5 << 12)
            | ((rd as u32) << 7)
            | OPC_OPIMM32;
        self.code.push(word);
        Ok(self)
    }

    /// `addw rd, rs1, rs2`
    pub fn addw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 0, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `subw rd, rs1, rs2`
    pub fn subw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 0, 0x20, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `sllw rd, rs1, rs2`
    pub fn sllw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 1, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `srlw rd, rs1, rs2`
    pub fn srlw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 5, 0x00, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `sraw rd, rs1, rs2`
    pub fn sraw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 5, 0x20, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `mulw rd, rs1, rs2`
    pub fn mulw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 0, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `divw rd, rs1, rs2`
    pub fn divw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 4, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `divuw rd, rs1, rs2`
    pub fn divuw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 5, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `remw rd, rs1, rs2`
    pub fn remw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 6, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    /// `remuw rd, rs1, rs2`
    pub fn remuw(&mut self, rd: u8, rs1: u8, rs2: u8) -> Result<&mut Self, AsmError> {
        self.emit_r(OPC_OP32, 7, 0x01, rd, rs1, rs2)?;
        Ok(self)
    }

    // -- loads / stores ------------------------------------------------------------

    /// `lw rd, imm(rs1)` (sign-extended 32-bit load).
    pub fn lw(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_imm("lw imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_LOAD, rd, 2, rs1, imm));
        Ok(self)
    }

    /// `lwu rd, imm(rs1)` (zero-extended 32-bit load).
    pub fn lwu(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_imm("lwu imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_LOAD, rd, 6, rs1, imm));
        Ok(self)
    }

    /// `ld rd, imm(rs1)`.
    pub fn ld(&mut self, rd: u8, rs1: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_imm("ld imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_i(OPC_LOAD, rd, 3, rs1, imm));
        Ok(self)
    }

    /// `sw rs2, imm(rs1)`.
    pub fn sw(&mut self, rs1: u8, rs2: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rs1)?;
        check_reg(rs2)?;
        check_imm("sw imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_s(OPC_STORE, 2, rs1, rs2, imm));
        Ok(self)
    }

    /// `sd rs2, imm(rs1)`.
    pub fn sd(&mut self, rs1: u8, rs2: u8, imm: i64) -> Result<&mut Self, AsmError> {
        check_reg(rs1)?;
        check_reg(rs2)?;
        check_imm("sd imm", imm, -2048, 2047)?;
        self.require_text()?;
        self.code.push(enc_s(OPC_STORE, 3, rs1, rs2, imm));
        Ok(self)
    }

    // -- system --------------------------------------------------------------------

    /// `ecall` (halts the VM).
    pub fn ecall(&mut self) -> Result<&mut Self, AsmError> {
        self.require_text()?;
        self.code.push(OPC_SYSTEM);
        Ok(self)
    }

    /// `ebreak`.
    pub fn ebreak(&mut self) -> Result<&mut Self, AsmError> {
        self.require_text()?;
        self.code.push(0x0010_0073);
        Ok(self)
    }

    // -- atomics (RV64A word) --------------------------------------------------------

    /// `lr.w rd, (rs1)`.
    pub fn lr_w(&mut self, rd: u8, rs1: u8, aq: bool, rl: bool) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        self.require_text()?;
        let word = (0x02u32 << 27)
            | ((aq as u32) << 26)
            | ((rl as u32) << 25)
            | ((rs1 as u32) << 15)
            | (2 << 12)
            | ((rd as u32) << 7)
            | OPC_AMO;
        self.code.push(word);
        Ok(self)
    }

    /// `sc.w rd, rs2, (rs1)`.
    pub fn sc_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x03, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amoswap.w rd, rs2, (rs1)`.
    pub fn amoswap_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x01, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amoadd.w rd, rs2, (rs1)`.
    pub fn amoadd_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x00, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amoxor.w rd, rs2, (rs1)`.
    pub fn amoxor_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x04, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amoand.w rd, rs2, (rs1)`.
    pub fn amoand_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x0c, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amoor.w rd, rs2, (rs1)`.
    pub fn amoor_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x08, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amomin.w rd, rs2, (rs1)`.
    pub fn amomin_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x10, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amomax.w rd, rs2, (rs1)`.
    pub fn amomax_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x14, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amominu.w rd, rs2, (rs1)`.
    pub fn amominu_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x18, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    /// `amomaxu.w rd, rs2, (rs1)`.
    pub fn amomaxu_w(
        &mut self,
        rd: u8,
        rs1: u8,
        rs2: u8,
        aq: bool,
        rl: bool,
    ) -> Result<&mut Self, AsmError> {
        check_reg(rd)?;
        check_reg(rs1)?;
        check_reg(rs2)?;
        self.require_text()?;
        self.code.push(enc_amo(0x1c, rd, rs2, rs1, aq, rl));
        Ok(self)
    }

    // -- finish ---------------------------------------------------------------------

    fn require_text(&self) -> Result<(), AsmError> {
        if self.section != Section::Text {
            Err(AsmError::InstrInData)
        } else {
            Ok(())
        }
    }

    fn label_addr(&self, label: LabelId, data_base: u64) -> Result<u64, AsmError> {
        match self.bound[label.0] {
            Some((off, Section::Text)) => Ok(off),
            Some((off, Section::Data)) => Ok(data_base.wrapping_add(off)),
            None => Err(AsmError::UndefinedLabel(self.label_name(label))),
        }
    }

    /// Resolve all fixups and produce the final program.
    pub fn finish(&self) -> Result<AssembledProgram, AsmError> {
        // Inline data base: first 8-aligned byte after the code image.
        let code_len = (self.code.len() as u64) * 4;
        let data_base = match self.data_mode {
            DataMode::Inline => (code_len + 7) & !7,
            DataMode::At(a) => a,
        };

        let mut code = self.code.clone();
        for fixup in &self.fixups {
            match *fixup {
                Fixup::Branch { at, label } => {
                    let target = self.label_addr(label, data_base)?;
                    let here = (at as u64) * 4;
                    let off = target as i64 - here as i64;
                    check_imm("branch offset", off, -4096, 4094).map_err(|_| {
                        AsmError::Parse {
                            line: 0,
                            msg: format!(
                                "branch to '{}' out of range: offset {off}",
                                self.label_name(label)
                            ),
                        }
                    })?;
                    if off & 1 != 0 {
                        return Err(AsmError::OddOffset(off));
                    }
                    // Re-encode preserving register/funct fields.
                    let old = code[at];
                    let rs1 = ((old >> 15) & 0x1f) as u8;
                    let rs2 = ((old >> 20) & 0x1f) as u8;
                    let f3 = (old >> 12) & 0x7;
                    code[at] = enc_b(OPC_BRANCH, f3, rs1, rs2, off);
                }
                Fixup::Jal { at, label } => {
                    let target = self.label_addr(label, data_base)?;
                    let here = (at as u64) * 4;
                    let off = target as i64 - here as i64;
                    if off & 1 != 0 {
                        return Err(AsmError::OddOffset(off));
                    }
                    if !(-(1 << 20)..=(1 << 20) - 2).contains(&off) {
                        return Err(AsmError::Parse {
                            line: 0,
                            msg: format!(
                                "jal to '{}' out of range: offset {off}",
                                self.label_name(label)
                            ),
                        });
                    }
                    let old = code[at];
                    let rd = ((old >> 7) & 0x1f) as u8;
                    code[at] = enc_j(OPC_JAL, rd, off);
                }
                Fixup::PcrelHi { at, label } => {
                    let target = self.label_addr(label, data_base)?;
                    let here = (at as u64) * 4;
                    let delta = target as i64 - here as i64;
                    let hi = (delta + 0x800) >> 12;
                    check_imm("auipc imm20", hi, -(1 << 19), (1 << 19) - 1).map_err(|_| {
                        AsmError::Parse {
                            line: 0,
                            msg: format!(
                                "la target '{}' out of pc-relative range",
                                self.label_name(label)
                            ),
                        }
                    })?;
                    let old = code[at];
                    let rd = ((old >> 7) & 0x1f) as u8;
                    code[at] = enc_u(OPC_AUIPC, rd, hi);
                }
                Fixup::PcrelLo { at, anchor, label } => {
                    let target = self.label_addr(label, data_base)?;
                    let here = (anchor as u64) * 4;
                    let delta = target as i64 - here as i64;
                    let hi = (delta + 0x800) >> 12;
                    let lo = delta - (hi << 12);
                    let old = code[at];
                    let rd = ((old >> 7) & 0x1f) as u8;
                    let rs1 = ((old >> 15) & 0x1f) as u8;
                    code[at] = enc_i(OPC_OPIMM, rd, 0, rs1, lo);
                }
            }
        }

        // Labels map (byte addresses).
        let mut labels = HashMap::new();
        for (name, &id) in &self.names {
            labels.insert(name.clone(), self.label_addr(id, data_base)?);
        }

        // Final image.
        let mut code_bytes = Vec::with_capacity(code.len() * 4);
        for w in &code {
            code_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let mut data = self.data.clone();
        match self.data_mode {
            DataMode::Inline => {
                // Append data after the (aligned) code image.
                while code_bytes.len() % 8 != 0 {
                    code_bytes.push(0);
                }
                code_bytes.extend_from_slice(&data);
                while code_bytes.len() % 8 != 0 {
                    code_bytes.push(0);
                }
                data = Vec::new();
            }
            DataMode::At(_) => {
                while data.len() % 8 != 0 {
                    data.push(0);
                }
            }
        }

        Ok(AssembledProgram {
            code: code_bytes,
            data,
            labels,
            data_base,
        })
    }
}

// ---------------------------------------------------------------------------
// Text assembler
// ---------------------------------------------------------------------------

/// Register name table: x-names and ABI names -> index.
pub fn parse_register(name: &str) -> Option<u8> {
    let n = name.trim();
    if let Some(rest) = n.strip_prefix('x') {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            let idx: u32 = rest.parse().ok()?;
            if idx < 32 {
                return Some(idx as u8);
            }
            return None;
        }
    }
    Some(match n {
        "zero" => 0,
        "ra" => 1,
        "sp" => 2,
        "gp" => 3,
        "tp" => 4,
        "t0" => 5,
        "t1" => 6,
        "t2" => 7,
        "s0" | "fp" => 8,
        "s1" => 9,
        "a0" => 10,
        "a1" => 11,
        "a2" => 12,
        "a3" => 13,
        "a4" => 14,
        "a5" => 15,
        "a6" => 16,
        "a7" => 17,
        "s2" => 18,
        "s3" => 19,
        "s4" => 20,
        "s5" => 21,
        "s6" => 22,
        "s7" => 23,
        "s8" => 24,
        "s9" => 25,
        "s10" => 26,
        "s11" => 27,
        "t3" => 28,
        "t4" => 29,
        "t5" => 30,
        "t6" => 31,
        _ => return None,
    })
}

fn parse_num(s: &str, line: usize) -> Result<i64, AsmError> {
    let t = s.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let v = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(AsmError::Parse { line, msg: format!("bad hex literal '{t}'") });
        }
        u64::from_str_radix(hex, 16).map_err(|_| {
            AsmError::Parse { line, msg: format!("hex literal '{t}' out of range") }
        })?
    } else if let Some(bin) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        if bin.is_empty() || !bin.chars().all(|c| c == '0' || c == '1') {
            return Err(AsmError::Parse { line, msg: format!("bad binary literal '{t}'") });
        }
        u64::from_str_radix(bin, 2).map_err(|_| {
            AsmError::Parse { line, msg: format!("binary literal '{t}' out of range") }
        })?
    } else {
        if body.is_empty() || !body.chars().all(|c| c.is_ascii_digit()) {
            return Err(AsmError::Parse { line, msg: format!("bad integer literal '{t}'") });
        }
        body.parse::<u64>().map_err(|_| {
            AsmError::Parse { line, msg: format!("integer literal '{t}' out of range") }
        })?
    };
    if neg {
        if v > (1u64 << 63) {
            return Err(AsmError::Parse { line, msg: format!("literal '{t}' out of range") });
        }
        Ok((v as i64).wrapping_neg())
    } else if v <= i64::MAX as u64 {
        Ok(v as i64)
    } else {
        // Allow unsigned forms for 64-bit constants (bit pattern).
        Ok(v as i64)
    }
}

/// Parse `imm(reg)` -> (imm, reg).
fn parse_mem(s: &str, line: usize) -> Result<(i64, u8), AsmError> {
    let t = s.trim();
    let Some(open) = t.find('(') else {
        return Err(AsmError::Parse {
            line,
            msg: format!("expected memory operand 'imm(reg)', got '{t}'"),
        });
    };
    if !t.ends_with(')') {
        return Err(AsmError::Parse {
            line,
            msg: format!("malformed memory operand '{t}'"),
        });
    }
    let imm_part = t[..open].trim();
    let reg_part = t[open + 1..t.len() - 1].trim();
    let imm = if imm_part.is_empty() {
        0
    } else {
        parse_num(imm_part, line)?
    };
    let reg = parse_register(reg_part).ok_or_else(|| {
        AsmError::Parse { line, msg: format!("unknown register '{reg_part}'") }
    })?;
    Ok((imm, reg))
}

fn reg_operand(s: &str, line: usize) -> Result<u8, AsmError> {
    parse_register(s).ok_or_else(|| {
        AsmError::Parse { line, msg: format!("unknown register '{s}'") }
    })
}

fn target_operand(
    asm: &mut Assembler,
    s: &str,
    line: usize,
) -> Result<Target, AsmError> {
    let t = s.trim();
    // Numeric offset?
    if t.starts_with('-')
        || t.starts_with('+')
        || t.as_bytes().first().is_some_and(|b| b.is_ascii_digit())
    {
        // hex/binary also start with digits
        if t.chars().next().is_some_and(|c| {
            c.is_ascii_digit() || c == '-' || c == '+'
        }) && !t.contains('(')
        {
            if let Ok(v) = parse_num(t, line) {
                return Ok(Target::Rel(v));
            }
        }
    }
    if is_ident(t) {
        return Ok(Target::Label(asm.label(t)));
    }
    Err(AsmError::Parse { line, msg: format!("bad branch target '{t}'") })
}

fn is_ident(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap_or('_');
    if !(first.is_ascii_alphabetic() || first == '_' || first == '.' || first == '$') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$')
}

fn strip_comment(line: &str) -> &str {
    if let Some(pos) = line.find('#') {
        &line[..pos]
    } else {
        line
    }
}

/// Assemble a text program.
///
/// Syntax (one statement per line; `#` starts a comment):
///
/// ```text
/// .text / .data / .org 0x8000 / .word 1, 2, 3 / .dword 7 /
/// .byte 1, 2 / label: / li x1, 0x1234 / lw t0, 8(sp) /
/// beq a0, zero, loop / j done / call func / ret / la a0, TABLE /
/// ecall
/// ```
///
/// Labels may share a line with a statement. Registers accept `xN` and
/// ABI names. Immediates accept decimal, `0x` hex, `0b` binary, and
/// signs. Branch/jump targets are labels or raw byte offsets.
pub fn assemble_str(src: &str) -> Result<AssembledProgram, AsmError> {
    let mut asm = Assembler::new();
    let mut last_line = 0usize;
    for (idx, raw_line) in src.lines().enumerate() {
        let line_no = idx + 1;
        last_line = line_no;
        let mut rest = strip_comment(raw_line).trim();
        if rest.is_empty() {
            continue;
        }
        // Leading labels (possibly several: `a: b: stmt`).
        while let Some(colon) = rest.find(':') {
            let (name, after) = rest.split_at(colon);
            let name = name.trim();
            if !is_ident(name) {
                return Err(AsmError::Parse { line: line_no, msg: format!("bad label '{name}'") });
            }
            asm.bind_name(name).map_err(|e| relabel(e, line_no))?;
            rest = after[1..].trim_start();
        }
        if rest.is_empty() {
            continue;
        }
        parse_statement(&mut asm, rest, line_no)?;
    }
    asm.finish().map_err(|e| relabel(e, last_line))
}

fn relabel(e: AsmError, line: usize) -> AsmError {
    // Only wrap errors that lack line context (builder-level ones).
    match e {
        AsmError::Parse { line: 0, msg } => AsmError::Parse { line, msg },
        other => other,
    }
}

fn parse_statement(
    asm: &mut Assembler,
    stmt: &str,
    line: usize,
) -> Result<(), AsmError> {
    let stmt = stmt.trim();
    // Directive?
    if let Some(directive) = stmt.strip_prefix('.') {
        let mut parts = directive.split_whitespace();
        let name = parts.next().unwrap_or("");
        let args_str = &directive[name.len()..];
        return parse_directive(asm, name, args_str.trim(), line);
    }
    // Instruction: mnemonic, then comma-separated operands.
    let split = stmt.find(|c: char| c.is_whitespace()).unwrap_or(stmt.len());
    let (mn, operand_str) = stmt.split_at(split);
    let operands: Vec<String> = operand_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    parse_instruction(asm, mn, &operands, line)
}

fn parse_directive(
    asm: &mut Assembler,
    name: &str,
    args: &str,
    line: usize,
) -> Result<(), AsmError> {
    match name {
        "text" => {
            asm.text();
            Ok(())
        }
        "data" => {
            asm.data();
            Ok(())
        }
        "org" => {
            let addr = parse_num(args, line)?;
            if addr < 0 {
                return Err(AsmError::Parse { line, msg: ".org address must be >= 0".to_string() });
            }
            asm.place_data_at(addr as u64).map_err(|e| relabel(e, line))?;
            Ok(())
        }
        "word" | "dword" | "byte" => {
            if args.is_empty() {
                return Err(AsmError::Parse { line, msg: format!(".{name} needs values") });
            }
            for tok in args.split(',') {
                let tok = tok.trim();
                if tok.is_empty() {
                    return Err(AsmError::Parse { line, msg: "empty directive value".into() });
                }
                let v = parse_num(tok, line)?;
                match name {
                    "word" => {
                        let u = if v < 0 { (v as i32) as u32 } else { v as u32 };
                        asm.word(u).map_err(|e| relabel(e, line))?;
                    }
                    "dword" => {
                        asm.word64(v as u64).map_err(|e| relabel(e, line))?;
                    }
                    "byte" => {
                        let u = if v < 0 { (v as i8) as u8 } else if v <= 255 { v as u8 } else {
                            return Err(AsmError::Parse {
                                line,
                                msg: format!("byte value {v} out of range"),
                            });
                        };
                        asm.byte(u).map_err(|e| relabel(e, line))?;
                    }
                    _ => unreachable!(),
                }
            }
            Ok(())
        }
        other => Err(AsmError::Parse { line, msg: format!("unknown directive '.{other}'") }),
    }
}

fn parse_instruction(
    asm: &mut Assembler,
    mn: &str,
    ops: &[String],
    line: usize,
) -> Result<(), AsmError> {
    let n = ops.len();
    let bad = |msg: String| AsmError::Parse { line, msg };
    let arity = |want: usize| -> Result<(), AsmError> {
        if n != want {
            Err(bad(format!("'{mn}' expects {want} operand(s), got {n}")))
        } else {
            Ok(())
        }
    };
    // Convenience closures (re-borrowing keeps the borrow checker happy).
    macro_rules! r {
        ($i:expr) => {
            reg_operand(&ops[$i], line)?
        };
    }
    macro_rules! imm {
        ($i:expr) => {
            parse_num(&ops[$i], line)?
        };
    }
    macro_rules! tgt {
        ($i:expr) => {
            target_operand(asm, &ops[$i], line)?
        };
    }
    macro_rules! lab {
        ($i:expr) => {{
            let t = ops[$i].trim();
            if !is_ident(t) {
                return Err(bad(format!("bad label operand '{t}'")));
            }
            asm.label(t)
        }};
    }
    macro_rules! mem {
        ($i:expr) => {
            parse_mem(&ops[$i], line)?
        };
    }

    match mn {
        // ---- pseudo ----
        "li" => {
            arity(2)?;
            asm.li(r!(0), imm!(1)).map_err(|e| relabel(e, line))?;
        }
        "mv" => {
            arity(2)?;
            asm.mv(r!(0), r!(1)).map_err(|e| relabel(e, line))?;
        }
        "nop" => {
            arity(0)?;
            asm.nop().map_err(|e| relabel(e, line))?;
        }
        "j" => {
            arity(1)?;
            let t = tgt!(0);
            asm.j(t).map_err(|e| relabel(e, line))?;
        }
        "jal" if n == 1 => {
            let t = tgt!(0);
            asm.jal(1, t).map_err(|e| relabel(e, line))?;
        }
        "call" => {
            arity(1)?;
            let t = tgt!(0);
            asm.call(t).map_err(|e| relabel(e, line))?;
        }
        "ret" => {
            arity(0)?;
            asm.ret().map_err(|e| relabel(e, line))?;
        }
        "jr" => {
            arity(1)?;
            asm.jr(r!(0)).map_err(|e| relabel(e, line))?;
        }
        "la" => {
            arity(2)?;
            let l = lab!(1);
            asm.la(r!(0), l).map_err(|e| relabel(e, line))?;
        }
        "la_abs" => {
            arity(2)?;
            let l = lab!(1);
            asm.la_abs(r!(0), l).map_err(|e| relabel(e, line))?;
        }
        "beqz" => {
            arity(2)?;
            let t = tgt!(1);
            asm.beqz(r!(0), t).map_err(|e| relabel(e, line))?;
        }
        "bnez" => {
            arity(2)?;
            let t = tgt!(1);
            asm.bnez(r!(0), t).map_err(|e| relabel(e, line))?;
        }
        "ecall" => {
            arity(0)?;
            asm.ecall().map_err(|e| relabel(e, line))?;
        }
        "ebreak" => {
            arity(0)?;
            asm.ebreak().map_err(|e| relabel(e, line))?;
        }
        // ---- register-immediate ----
        "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" | "addiw" => {
            arity(3)?;
            let rd = r!(0);
            let rs1 = r!(1);
            let v = imm!(2);
            match mn {
                "addi" => asm.addi(rd, rs1, v),
                "slti" => asm.slti(rd, rs1, v),
                "sltiu" => asm.sltiu(rd, rs1, v),
                "xori" => asm.xori(rd, rs1, v),
                "ori" => asm.ori(rd, rs1, v),
                "andi" => asm.andi(rd, rs1, v),
                "addiw" => asm.addiw(rd, rs1, v),
                _ => unreachable!(),
            }
            .map_err(|e| relabel(e, line))?;
        }
        "slli" | "srli" | "srai" | "slliw" | "srliw" | "sraiw" => {
            arity(3)?;
            let rd = r!(0);
            let rs1 = r!(1);
            let v = imm!(2);
            if !(0..=63).contains(&v) {
                return Err(bad(format!("'{mn}' shift amount {v} out of range 0..=63")));
            }
            let sh = v as u8;
            match mn {
                "slli" => asm.slli(rd, rs1, sh),
                "srli" => asm.srli(rd, rs1, sh),
                "srai" => asm.srai(rd, rs1, sh),
                "slliw" => asm.slliw(rd, rs1, sh),
                "srliw" => asm.srliw(rd, rs1, sh),
                "sraiw" => asm.sraiw(rd, rs1, sh),
                _ => unreachable!(),
            }
            .map_err(|e| relabel(e, line))?;
        }
        "lui" | "auipc" => {
            arity(2)?;
            let rd = r!(0);
            let v = imm!(1);
            match mn {
                "lui" => asm.lui(rd, v),
                _ => asm.auipc(rd, v),
            }
            .map_err(|e| relabel(e, line))?;
        }
        // ---- register-register ----
        "add" | "sub" | "sll" | "slt" | "sltu" | "xor" | "srl" | "sra" | "or" | "and" | "mul"
        | "mulh" | "mulhu" | "div" | "divu" | "rem" | "remu" | "addw" | "subw" | "sllw"
        | "srlw" | "sraw" | "mulw" | "divw" | "divuw" | "remw" | "remuw" => {
            arity(3)?;
            let rd = r!(0);
            let rs1 = r!(1);
            let rs2 = r!(2);
            let m = match mn {
                "add" => asm.add(rd, rs1, rs2),
                "sub" => asm.sub(rd, rs1, rs2),
                "sll" => asm.sll(rd, rs1, rs2),
                "slt" => asm.slt(rd, rs1, rs2),
                "sltu" => asm.sltu(rd, rs1, rs2),
                "xor" => asm.xor(rd, rs1, rs2),
                "srl" => asm.srl(rd, rs1, rs2),
                "sra" => asm.sra(rd, rs1, rs2),
                "or" => asm.or(rd, rs1, rs2),
                "and" => asm.and(rd, rs1, rs2),
                "mul" => asm.mul(rd, rs1, rs2),
                "mulh" => asm.mulh(rd, rs1, rs2),
                "mulhu" => asm.mulhu(rd, rs1, rs2),
                "div" => asm.div(rd, rs1, rs2),
                "divu" => asm.divu(rd, rs1, rs2),
                "rem" => asm.rem(rd, rs1, rs2),
                "remu" => asm.remu(rd, rs1, rs2),
                "addw" => asm.addw(rd, rs1, rs2),
                "subw" => asm.subw(rd, rs1, rs2),
                "sllw" => asm.sllw(rd, rs1, rs2),
                "srlw" => asm.srlw(rd, rs1, rs2),
                "sraw" => asm.sraw(rd, rs1, rs2),
                "mulw" => asm.mulw(rd, rs1, rs2),
                "divw" => asm.divw(rd, rs1, rs2),
                "divuw" => asm.divuw(rd, rs1, rs2),
                "remw" => asm.remw(rd, rs1, rs2),
                _ => asm.remuw(rd, rs1, rs2),
            };
            m.map_err(|e| relabel(e, line))?;
        }
        // ---- branches ----
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
            arity(3)?;
            let rs1 = r!(0);
            let rs2 = r!(1);
            let t = target_operand(asm, &ops[2], line)?;
            match mn {
                "beq" => asm.beq(rs1, rs2, t),
                "bne" => asm.bne(rs1, rs2, t),
                "blt" => asm.blt(rs1, rs2, t),
                "bge" => asm.bge(rs1, rs2, t),
                "bltu" => asm.bltu(rs1, rs2, t),
                _ => asm.bgeu(rs1, rs2, t),
            }
            .map_err(|e| relabel(e, line))?;
        }
        // ---- jumps ----
        "jal" => {
            arity(2)?;
            let t = tgt!(1);
            asm.jal(r!(0), t).map_err(|e| relabel(e, line))?;
        }
        "jalr" => {
            if n == 2 {
                // jalr rd, rs
                asm.jalr(r!(0), r!(1), 0).map_err(|e| relabel(e, line))?;
            } else {
                arity(3)?;
                asm.jalr(r!(0), r!(1), imm!(2)).map_err(|e| relabel(e, line))?;
            }
        }
        // ---- loads ----
        "lw" | "lwu" | "ld" => {
            arity(2)?;
            let rd = r!(0);
            let (imm, rs1) = mem!(1);
            match mn {
                "lw" => asm.lw(rd, rs1, imm),
                "lwu" => asm.lwu(rd, rs1, imm),
                _ => asm.ld(rd, rs1, imm),
            }
            .map_err(|e| relabel(e, line))?;
        }
        // ---- stores ----
        "sw" | "sd" => {
            arity(2)?;
            let rs2 = r!(0);
            let (imm, rs1) = mem!(1);
            match mn {
                "sw" => asm.sw(rs1, rs2, imm),
                _ => asm.sd(rs1, rs2, imm),
            }
            .map_err(|e| relabel(e, line))?;
        }
        // ---- atomics ----
        "lr.w" => {
            arity(2)?;
            let rd = r!(0);
            let (imm, rs1) = mem!(1);
            if imm != 0 {
                return Err(bad("lr.w offset must be 0".into()));
            }
            asm.lr_w(rd, rs1, false, false).map_err(|e| relabel(e, line))?;
        }
        "sc.w" | "amoswap.w" | "amoadd.w" | "amoxor.w" | "amoand.w" | "amoor.w" | "amomin.w"
        | "amomax.w" | "amominu.w" | "amomaxu.w" => {
            arity(3)?;
            let rd = r!(0);
            let rs2 = r!(1);
            let (imm, rs1) = mem!(2);
            if imm != 0 {
                return Err(bad(format!("'{mn}' offset must be 0")));
            }
            let m = match mn {
                "sc.w" => asm.sc_w(rd, rs1, rs2, false, false),
                "amoswap.w" => asm.amoswap_w(rd, rs1, rs2, false, false),
                "amoadd.w" => asm.amoadd_w(rd, rs1, rs2, false, false),
                "amoxor.w" => asm.amoxor_w(rd, rs1, rs2, false, false),
                "amoand.w" => asm.amoand_w(rd, rs1, rs2, false, false),
                "amoor.w" => asm.amoor_w(rd, rs1, rs2, false, false),
                "amomin.w" => asm.amomin_w(rd, rs1, rs2, false, false),
                "amomax.w" => asm.amomax_w(rd, rs1, rs2, false, false),
                "amominu.w" => asm.amominu_w(rd, rs1, rs2, false, false),
                _ => asm.amomaxu_w(rd, rs1, rs2, false, false),
            };
            m.map_err(|e| relabel(e, line))?;
        }
        other => {
            return Err(AsmError::Parse {
                line,
                msg: format!("unknown mnemonic '{other}'"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// VM convenience runner (test-only: lattice-vm is a dev-dependency)
// ---------------------------------------------------------------------------

/// Result of running a program on the VM: final state + executed step
/// count (the honest cycle count).
#[derive(Debug, Clone)]
pub struct VmRun {
    pub state: MachineState,
    pub steps: u64,
}

/// Load an assembled program (all segments) at their addresses, place the
/// public input at [`crate::PUBLIC_INPUT_BASE`], and run to ECALL or
/// `max_steps`.
pub fn run_on_vm(
    program: &AssembledProgram,
    public_input: &[u8],
    max_steps: u64,
) -> Result<VmRun, exec::ExecError> {
    let mut state = MachineState::new();
    for (addr, bytes) in program.segments() {
        state.load_program(addr, bytes);
    }
    if !public_input.is_empty() {
        state.load_program(crate::PUBLIC_INPUT_BASE, public_input);
    }
    let rows = exec::run(&mut state, max_steps)?;
    Ok(VmRun { state, steps: rows.len() as u64 })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_vm::decode::{decode, Instr};

    fn first_instr(asm: &Assembler) -> Instr {
        let prog = asm.clone().finish().expect("finish");
        let word = u32::from_le_bytes(prog.code[0..4].try_into().expect("4 bytes"));
        decode(0, word).expect("decodes")
    }

    /// Every instruction method round-trips through the VM decoder.
    #[test]
    fn decode_table() {
        // (builder action, expected decode)
        struct Case {
            name: &'static str,
            emit: fn(&mut Assembler) -> Result<(), AsmError>,
            want: Instr,
        }
        let cases: Vec<Case> = vec![
            Case {
                name: "addi",
                emit: |a| a.addi(5, 6, -2048).map(|_| ()),
                want: Instr::Addi { rd: 5, rs1: 6, imm: -2048 },
            },
            Case {
                name: "slti",
                emit: |a| a.slti(7, 8, 2047).map(|_| ()),
                want: Instr::Slti { rd: 7, rs1: 8, imm: 2047 },
            },
            Case {
                name: "sltiu",
                emit: |a| a.sltiu(9, 10, -1).map(|_| ()),
                want: Instr::Sltiu { rd: 9, rs1: 10, imm: (-1i64) as u64 },
            },
            Case {
                name: "xori",
                emit: |a| a.xori(1, 2, -5).map(|_| ()),
                want: Instr::Xori { rd: 1, rs1: 2, imm: -5 },
            },
            Case {
                name: "ori",
                emit: |a| a.ori(3, 4, 0x7ff).map(|_| ()),
                want: Instr::Ori { rd: 3, rs1: 4, imm: 0x7ff },
            },
            Case {
                name: "andi",
                emit: |a| a.andi(11, 12, -2048).map(|_| ()),
                want: Instr::Andi { rd: 11, rs1: 12, imm: -2048 },
            },
            Case {
                name: "slli",
                emit: |a| a.slli(13, 14, 63).map(|_| ()),
                want: Instr::Slli { rd: 13, rs1: 14, shamt: 63 },
            },
            Case {
                name: "srli",
                emit: |a| a.srli(15, 16, 32).map(|_| ()),
                want: Instr::Srli { rd: 15, rs1: 16, shamt: 32 },
            },
            Case {
                name: "srai",
                emit: |a| a.srai(17, 18, 31).map(|_| ()),
                want: Instr::Srai { rd: 17, rs1: 18, shamt: 31 },
            },
            Case {
                name: "addiw",
                emit: |a| a.addiw(19, 20, -3).map(|_| ()),
                want: Instr::Addiw { rd: 19, rs1: 20, imm: -3 },
            },
            Case {
                name: "slliw",
                emit: |a| a.slliw(21, 22, 31).map(|_| ()),
                want: Instr::Slliw { rd: 21, rs1: 22, shamt: 31 },
            },
            Case {
                name: "srliw",
                emit: |a| a.srliw(23, 24, 0).map(|_| ()),
                want: Instr::Srliw { rd: 23, rs1: 24, shamt: 0 },
            },
            Case {
                name: "sraiw",
                emit: |a| a.sraiw(25, 26, 15).map(|_| ()),
                want: Instr::Sraiw { rd: 25, rs1: 26, shamt: 15 },
            },
            Case {
                name: "add",
                emit: |a| a.add(1, 2, 3).map(|_| ()),
                want: Instr::Add { rd: 1, rs1: 2, rs2: 3 },
            },
            Case {
                name: "sub",
                emit: |a| a.sub(4, 5, 6).map(|_| ()),
                want: Instr::Sub { rd: 4, rs1: 5, rs2: 6 },
            },
            Case {
                name: "sll",
                emit: |a| a.sll(7, 8, 9).map(|_| ()),
                want: Instr::Sll { rd: 7, rs1: 8, rs2: 9 },
            },
            Case {
                name: "slt",
                emit: |a| a.slt(10, 11, 12).map(|_| ()),
                want: Instr::Slt { rd: 10, rs1: 11, rs2: 12 },
            },
            Case {
                name: "sltu",
                emit: |a| a.sltu(13, 14, 15).map(|_| ()),
                want: Instr::Sltu { rd: 13, rs1: 14, rs2: 15 },
            },
            Case {
                name: "xor",
                emit: |a| a.xor(16, 17, 18).map(|_| ()),
                want: Instr::Xor { rd: 16, rs1: 17, rs2: 18 },
            },
            Case {
                name: "srl",
                emit: |a| a.srl(19, 20, 21).map(|_| ()),
                want: Instr::Srl { rd: 19, rs1: 20, rs2: 21 },
            },
            Case {
                name: "sra",
                emit: |a| a.sra(22, 23, 24).map(|_| ()),
                want: Instr::Sra { rd: 22, rs1: 23, rs2: 24 },
            },
            Case {
                name: "or",
                emit: |a| a.or(25, 26, 27).map(|_| ()),
                want: Instr::Or { rd: 25, rs1: 26, rs2: 27 },
            },
            Case {
                name: "and",
                emit: |a| a.and(28, 29, 30).map(|_| ()),
                want: Instr::And { rd: 28, rs1: 29, rs2: 30 },
            },
            Case {
                name: "mul",
                emit: |a| a.mul(1, 3, 5).map(|_| ()),
                want: Instr::Mul { rd: 1, rs1: 3, rs2: 5 },
            },
            Case {
                name: "mulh",
                emit: |a| a.mulh(2, 4, 6).map(|_| ()),
                want: Instr::Mulh { rd: 2, rs1: 4, rs2: 6 },
            },
            Case {
                name: "mulhu",
                emit: |a| a.mulhu(3, 5, 7).map(|_| ()),
                want: Instr::Mulhu { rd: 3, rs1: 5, rs2: 7 },
            },
            Case {
                name: "div",
                emit: |a| a.div(4, 6, 8).map(|_| ()),
                want: Instr::Div { rd: 4, rs1: 6, rs2: 8 },
            },
            Case {
                name: "divu",
                emit: |a| a.divu(5, 7, 9).map(|_| ()),
                want: Instr::Divu { rd: 5, rs1: 7, rs2: 9 },
            },
            Case {
                name: "rem",
                emit: |a| a.rem(6, 8, 10).map(|_| ()),
                want: Instr::Rem { rd: 6, rs1: 8, rs2: 10 },
            },
            Case {
                name: "remu",
                emit: |a| a.remu(7, 9, 11).map(|_| ()),
                want: Instr::Remu { rd: 7, rs1: 9, rs2: 11 },
            },
            Case {
                name: "addw",
                emit: |a| a.addw(8, 10, 12).map(|_| ()),
                want: Instr::Addw { rd: 8, rs1: 10, rs2: 12 },
            },
            Case {
                name: "subw",
                emit: |a| a.subw(9, 11, 13).map(|_| ()),
                want: Instr::Subw { rd: 9, rs1: 11, rs2: 13 },
            },
            Case {
                name: "sllw",
                emit: |a| a.sllw(12, 14, 16).map(|_| ()),
                want: Instr::Sllw { rd: 12, rs1: 14, rs2: 16 },
            },
            Case {
                name: "srlw",
                emit: |a| a.srlw(13, 15, 17).map(|_| ()),
                want: Instr::Srlw { rd: 13, rs1: 15, rs2: 17 },
            },
            Case {
                name: "sraw",
                emit: |a| a.sraw(14, 16, 18).map(|_| ()),
                want: Instr::Sraw { rd: 14, rs1: 16, rs2: 18 },
            },
            Case {
                name: "mulw",
                emit: |a| a.mulw(15, 17, 19).map(|_| ()),
                want: Instr::Mulw { rd: 15, rs1: 17, rs2: 19 },
            },
            Case {
                name: "divw",
                emit: |a| a.divw(16, 18, 20).map(|_| ()),
                want: Instr::Divw { rd: 16, rs1: 18, rs2: 20 },
            },
            Case {
                name: "divuw",
                emit: |a| a.divuw(17, 19, 21).map(|_| ()),
                want: Instr::Divuw { rd: 17, rs1: 19, rs2: 21 },
            },
            Case {
                name: "remw",
                emit: |a| a.remw(18, 20, 22).map(|_| ()),
                want: Instr::Remw { rd: 18, rs1: 20, rs2: 22 },
            },
            Case {
                name: "remuw",
                emit: |a| a.remuw(19, 21, 23).map(|_| ()),
                want: Instr::Remuw { rd: 19, rs1: 21, rs2: 23 },
            },
            Case {
                name: "lui",
                emit: |a| a.lui(6, -1).map(|_| ()),
                want: Instr::Lui { rd: 6, imm: (-1i64 << 12) },
            },
            Case {
                name: "lui_pos",
                emit: |a| a.lui(7, 0x7f).map(|_| ()),
                want: Instr::Lui { rd: 7, imm: 0x7f << 12 },
            },
            Case {
                name: "auipc",
                emit: |a| a.auipc(8, -0x80000).map(|_| ()),
                want: Instr::Auipc { rd: 8, imm: (-0x80000i64) << 12 },
            },
            Case {
                name: "jal",
                emit: |a| a.jal(1, Target::Rel(8)).map(|_| ()),
                want: Instr::Jal { rd: 1, imm: 8 },
            },
            Case {
                name: "jal_backward",
                emit: |a| a.jal(0, Target::Rel(-2048)).map(|_| ()),
                want: Instr::Jal { rd: 0, imm: -2048 },
            },
            Case {
                name: "jalr",
                emit: |a| a.jalr(2, 3, -2048).map(|_| ()),
                want: Instr::Jalr { rd: 2, rs1: 3, imm: -2048 },
            },
            Case {
                name: "beq",
                emit: |a| a.beq(1, 2, Target::Rel(-4096)).map(|_| ()),
                want: Instr::Beq { rs1: 1, rs2: 2, imm: -4096 },
            },
            Case {
                name: "bne",
                emit: |a| a.bne(3, 4, Target::Rel(4094)).map(|_| ()),
                want: Instr::Bne { rs1: 3, rs2: 4, imm: 4094 },
            },
            Case {
                name: "blt",
                emit: |a| a.blt(5, 6, Target::Rel(2)).map(|_| ()),
                want: Instr::Blt { rs1: 5, rs2: 6, imm: 2 },
            },
            Case {
                name: "bge",
                emit: |a| a.bge(7, 8, Target::Rel(-6)).map(|_| ()),
                want: Instr::Bge { rs1: 7, rs2: 8, imm: -6 },
            },
            Case {
                name: "bltu",
                emit: |a| a.bltu(9, 10, Target::Rel(16)).map(|_| ()),
                want: Instr::Bltu { rs1: 9, rs2: 10, imm: 16 },
            },
            Case {
                name: "bgeu",
                emit: |a| a.bgeu(11, 12, Target::Rel(-1024)).map(|_| ()),
                want: Instr::Bgeu { rs1: 11, rs2: 12, imm: -1024 },
            },
            Case {
                name: "lw",
                emit: |a| a.lw(13, 14, -2048).map(|_| ()),
                want: Instr::Lw { rd: 13, rs1: 14, imm: -2048 },
            },
            Case {
                name: "lwu",
                emit: |a| a.lwu(15, 16, 2047).map(|_| ()),
                want: Instr::Lwu { rd: 15, rs1: 16, imm: 2047 },
            },
            Case {
                name: "ld",
                emit: |a| a.ld(17, 18, 8).map(|_| ()),
                want: Instr::Ld { rd: 17, rs1: 18, imm: 8 },
            },
            Case {
                name: "sw",
                emit: |a| a.sw(19, 20, -4).map(|_| ()),
                want: Instr::Sw { rs1: 19, rs2: 20, imm: -4 },
            },
            Case {
                name: "sd",
                emit: |a| a.sd(21, 22, 16).map(|_| ()),
                want: Instr::Sd { rs1: 21, rs2: 22, imm: 16 },
            },
            Case {
                name: "ecall",
                emit: |a| a.ecall().map(|_| ()),
                want: Instr::Ecall,
            },
            Case {
                name: "ebreak",
                emit: |a| a.ebreak().map(|_| ()),
                want: Instr::Ebreak,
            },
            Case {
                name: "lr_w",
                emit: |a| a.lr_w(2, 3, true, false).map(|_| ()),
                want: Instr::LrW { rd: 2, rs1: 3, aq: true, rl: false },
            },
            Case {
                name: "sc_w",
                emit: |a| a.sc_w(4, 5, 6, false, true).map(|_| ()),
                want: Instr::ScW { rd: 4, rs1: 5, rs2: 6, aq: false, rl: true },
            },
            Case {
                name: "amoswap_w",
                emit: |a| a.amoswap_w(7, 8, 9, false, false).map(|_| ()),
                want: Instr::AmoSwapW { rd: 7, rs1: 8, rs2: 9, aq: false, rl: false },
            },
            Case {
                name: "amoadd_w",
                emit: |a| a.amoadd_w(10, 11, 12, true, true).map(|_| ()),
                want: Instr::AmoAddW { rd: 10, rs1: 11, rs2: 12, aq: true, rl: true },
            },
            Case {
                name: "amoxor_w",
                emit: |a| a.amoxor_w(13, 14, 15, false, false).map(|_| ()),
                want: Instr::AmoXorW { rd: 13, rs1: 14, rs2: 15, aq: false, rl: false },
            },
            Case {
                name: "amoand_w",
                emit: |a| a.amoand_w(16, 17, 18, false, false).map(|_| ()),
                want: Instr::AmoAndW { rd: 16, rs1: 17, rs2: 18, aq: false, rl: false },
            },
            Case {
                name: "amoor_w",
                emit: |a| a.amoor_w(19, 20, 21, false, false).map(|_| ()),
                want: Instr::AmoOrW { rd: 19, rs1: 20, rs2: 21, aq: false, rl: false },
            },
            Case {
                name: "amomin_w",
                emit: |a| a.amomin_w(22, 23, 24, false, false).map(|_| ()),
                want: Instr::AmoMinW { rd: 22, rs1: 23, rs2: 24, aq: false, rl: false },
            },
            Case {
                name: "amomax_w",
                emit: |a| a.amomax_w(25, 26, 27, false, false).map(|_| ()),
                want: Instr::AmoMaxW { rd: 25, rs1: 26, rs2: 27, aq: false, rl: false },
            },
            Case {
                name: "amominu_w",
                emit: |a| a.amominu_w(28, 29, 30, false, false).map(|_| ()),
                want: Instr::AmoMinuW { rd: 28, rs1: 29, rs2: 30, aq: false, rl: false },
            },
            Case {
                name: "amomaxu_w",
                emit: |a| a.amomaxu_w(31, 1, 2, false, false).map(|_| ()),
                want: Instr::AmoMaxuW { rd: 31, rs1: 1, rs2: 2, aq: false, rl: false },
            },
        ];
        for case in cases {
            let mut asm = Assembler::new();
            (case.emit)(&mut asm).unwrap_or_else(|e| panic!("{}: emit: {e}", case.name));
            let got = first_instr(&asm);
            assert_eq!(got, case.want, "case '{}'", case.name);
        }
    }

    #[test]
    fn imm_range_checks() {
        let mut asm = Assembler::new();
        assert!(matches!(
            asm.addi(1, 0, 2048),
            Err(AsmError::ImmOutOfRange { what: "i-type imm", .. })
        ));
        assert!(matches!(
            asm.lui(1, 1 << 19),
            Err(AsmError::ImmOutOfRange { what: "lui imm20", .. })
        ));
        assert!(matches!(
            asm.slli(1, 2, 64),
            Err(AsmError::ShiftOutOfRange { what: "slli", .. })
        ));
        assert!(matches!(
            asm.slliw(1, 2, 32),
            Err(AsmError::ShiftOutOfRange { what: "slliw", .. })
        ));
        assert!(matches!(
            asm.beq(1, 2, Target::Rel(4096)),
            Err(AsmError::ImmOutOfRange { what: "branch offset", .. })
        ));
        assert!(matches!(asm.beq(1, 2, Target::Rel(3)), Err(AsmError::OddOffset(3))));
        assert!(matches!(asm.add(32, 1, 2), Err(AsmError::RegisterOutOfRange(32))));
        assert!(matches!(
            asm.lw(1, 2, 3000),
            Err(AsmError::ImmOutOfRange { what: "lw imm", .. })
        ));
    }

    /// `li` materializes every class of constant exactly (run on the VM).
    #[test]
    fn li_constants() {
        let values: Vec<i64> = vec![
            0,
            1,
            -1,
            2047,
            -2048,
            2048,
            -2049,
            4095,
            4096,
            0x12345,
            -0x12345,
            0x7fff_ffff,
            -0x8000_0000,
            0x1234_5678,
            i64::MIN,
            i64::MAX,
            0x0123_4567_89ab_cdef,
            -0x0123_4567_89ab_cdef,
            (0x8000_0000_0000_0000u64) as i64,
            (0xfedc_ba98_7654_3210u64) as i64,
            (u64::MAX) as i64,
            (1u64 << 63) as i64,
            0xdead_beef_cafe_f00du64 as i64,
            0xdead_beef_cafe_f00du64.wrapping_neg() as i64,
        ];
        for &v in &values {
            let mut asm = Assembler::new();
            asm.li(10, v).expect("li emits");
            asm.ecall().expect("ecall");
            let prog = asm.finish().expect("finish");
            let run = run_on_vm(&prog, &[], 64).expect("runs");
            assert_eq!(run.state.reg(10), v as u64, "li {v} (#0x{:x})", v as u64);
            assert!(run.state.halted);
        }
    }

    #[test]
    fn labels_branches_and_loops() {
        // Count 10 -> 0 with a backward branch to a label bound later
        // (forward reference), then jump over data.
        let mut asm = Assembler::new();
        let loop_lbl = asm.label("loop");
        let done = asm.label("done");
        asm.li(5, 10).expect("li");
        asm.bind(loop_lbl).expect("bind");
        asm.beqz(5, done.into()).expect("beqz");
        asm.addi(5, 5, -1).expect("addi");
        asm.j(loop_lbl.into()).expect("j");
        asm.bind(done).expect("bind");
        asm.li(6, 77).expect("li");
        asm.ecall().expect("ecall");
        let prog = asm.finish().expect("finish");
        let run = run_on_vm(&prog, &[], 256).expect("runs");
        assert_eq!(run.state.reg(5), 0);
        assert_eq!(run.state.reg(6), 77);
        // li + 11x beqz (10 not-taken + 1 taken) + 10x (addi + j) + li + ecall.
        assert_eq!(run.steps, 1 + 11 + 2 * 10 + 1 + 1);
    }

    #[test]
    fn inline_data_and_la() {
        let mut asm = Assembler::new();
        asm.li(5, 0).expect("li");
        let tbl = asm.label("TABLE");
        asm.la(6, tbl).expect("la");
        asm.ld(7, 6, 0).expect("ld");
        asm.ld(8, 6, 8).expect("ld");
        asm.add(5, 7, 8).expect("add");
        asm.ecall().expect("ecall");
        asm.data();
        asm.bind(tbl).expect("bind"); // bind at data offset 0
        asm.word64(100).expect("word64");
        asm.word64(23).expect("word64");
        let prog = asm.finish().expect("finish");
        // TABLE label points at the data appended after code.
        let base = prog.labels["TABLE"];
        assert_eq!(base, prog.data_base);
        let run = run_on_vm(&prog, &[], 64).expect("runs");
        assert_eq!(run.state.reg(5), 123);
    }

    #[test]
    fn placed_data_and_la_abs() {
        let mut asm = Assembler::new();
        asm.li(5, 0).expect("li");
        asm.data();
        asm.place_data_at(0x8000).expect("place");
        asm.bind_name("K").expect("bind");
        asm.word(0x428a2f98).expect("word");
        asm.word(0x71374491).expect("word");
        asm.text();
        let k = asm.label("K");
        asm.la_abs(6, k).expect("la_abs");
        asm.lwu(7, 6, 0).expect("lwu");
        asm.lw(8, 6, 4).expect("lw");
        asm.add(5, 7, 8).expect("add");
        asm.ecall().expect("ecall");
        let prog = asm.finish().expect("finish");
        assert_eq!(prog.data_base, 0x8000);
        assert!(!prog.data.is_empty());
        let run = run_on_vm(&prog, &[], 64).expect("runs");
        assert_eq!(run.state.reg(5), 0x428a2f98 + 0x71374491);
    }

    #[test]
    fn call_ret_subroutine() {
        let mut asm = Assembler::new();
        asm.li(5, 0).expect("li");
        asm.li(6, 5).expect("li");
        let sub = asm.label("sub");
        asm.call(sub.into()).expect("call");
        asm.add(5, 5, 10).expect("add"); // after return: a0 = 7+10? no: add 5,5,10
        asm.ecall().expect("ecall");
        asm.bind(sub).expect("bind");
        asm.addi(10, 6, 2).expect("addi"); // a0 = 6+2 = 7
        asm.ret().expect("ret");
        let prog = asm.finish().expect("finish");
        let run = run_on_vm(&prog, &[], 64).expect("runs");
        assert_eq!(run.state.reg(10), 7);
        assert_eq!(run.state.reg(5), 7);
    }

    #[test]
    fn undefined_and_duplicate_labels() {
        let mut asm = Assembler::new();
        let l = asm.label("missing");
        asm.j(l.into()).expect("j");
        assert!(matches!(
            asm.finish(),
            Err(AsmError::UndefinedLabel(name)) if name == "missing"
        ));

        let mut asm = Assembler::new();
        let l = asm.label("twice");
        asm.bind(l).expect("bind");
        assert!(matches!(
            asm.bind(l),
            Err(AsmError::DuplicateLabel(name)) if name == "twice"
        ));
    }

    #[test]
    fn instr_in_data_rejected() {
        let mut asm = Assembler::new();
        asm.data();
        assert!(matches!(asm.addi(1, 2, 3), Err(AsmError::InstrInData)));
    }

    #[test]
    fn placement_after_data_rejected() {
        let mut asm = Assembler::new();
        asm.data();
        asm.word(1).expect("word");
        assert!(matches!(
            asm.place_data_at(0x8000),
            Err(AsmError::PlacementAfterData)
        ));
    }

    // ---- text assembler --------------------------------------------------

    #[test]
    fn text_assembler_basic() {
        let src = r#"
            # count down 3 -> 0, then read a table
            li   t0, 3
        loop:
            beqz t0, done
            addi t0, t0, -1
            j    loop
        done:
            la   t1, TABLE
            ld   t2, 0(t1)
            add  a0, t2, t0
            ecall
            .data
        TABLE:
            .dword 55
        "#;
        let prog = assemble_str(src).expect("assembles");
        assert_eq!(prog.labels["TABLE"], prog.data_base);
        let run = run_on_vm(&prog, &[], 128).expect("runs");
        assert_eq!(run.state.reg(10), 55);
        assert_eq!(run.state.reg(5), 0);
    }

    #[test]
    fn text_assembler_all_operand_forms() {
        let src = r#"
            .text
            lw   a0, 8(sp)
            sw   a0, 12(sp)
            sd   a0, (sp)
            ld   a1, -8(sp)
            lui  t0, 0x12345
            auipc t1, -1
            jal  ra, sub1
            beq  a0, a1, +8
            nop
            ecall
        sub1:
            ret
            .data
            .word 1, 2, 3
            .byte 4, 5
        "#;
        let prog = assemble_str(src).expect("assembles");
        // 11 instructions (44 bytes) before .data: data follows inline at
        // align8(44) = 48; the final image is code+data padded to 8.
        assert_eq!(prog.code.len() % 4, 0);
        assert_eq!(prog.data_base, 48);
        assert_eq!(prog.labels["sub1"], 40);
        let run = run_on_vm(&prog, &[], 128).expect("runs");
        assert!(run.state.halted);
    }

    #[test]
    fn text_assembler_mem_ops_execute() {
        let src = r#"
            li   t0, 0x2000
            li   t1, -1
            sd   t1, 0(t0)
            ld   t2, 0(t0)
            sw   t1, 8(t0)
            lwu  t3, 8(t0)
            lw   t4, 8(t0)
            sub  a0, t2, t3
            sub  a1, t2, t4
            ecall
        "#;
        let prog = assemble_str(src).expect("assembles");
        let run = run_on_vm(&prog, &[], 64).expect("runs");
        // lwu zero-extends the stored 0xFFFFFFFF; lw sign-extends it back
        // to -1, so t2 - t3 spans the zero-extension and t2 - t4 is 0.
        assert_eq!(run.state.reg(10), u64::MAX - 0xFFFF_FFFF);
        assert_eq!(run.state.reg(11), 0);
    }

    #[test]
    fn text_assembler_errors() {
        type ErrCheck = fn(&AsmError) -> bool;
        let cases: Vec<(&str, ErrCheck)> = vec![
            ("foo x1, x2\n", |e| matches!(e, AsmError::Parse { msg, .. } if msg.contains("unknown mnemonic"))),
            ("add x1, y2, x3\n", |e| matches!(e, AsmError::Parse { msg, .. } if msg.contains("unknown register"))),
            ("addi x1, x2, 5000\n", |e| matches!(e, AsmError::ImmOutOfRange { .. })),
            ("j nowhere\n", |e| matches!(e, AsmError::UndefinedLabel(_))),
            (".frobnicate 1\n", |e| matches!(e, AsmError::Parse { msg, .. } if msg.contains("unknown directive"))),
            ("li x1, 12ab\n", |e| matches!(e, AsmError::Parse { msg, .. } if msg.contains("bad"))),
            ("add x1, x2\n", |e| matches!(e, AsmError::Parse { msg, .. } if msg.contains("operand"))),
            ("bad label:\n  nop\n", |e| matches!(e, AsmError::Parse { msg, .. } if msg.contains("bad label"))),
        ];
        for (src, check) in cases {
            let err = assemble_str(src).expect_err("must fail");
            assert!(check(&err), "source {src:?} gave {err:?}");
        }
    }

    #[test]
    fn text_assembler_data_and_org() {
        // `la_abs` bakes the absolute address at emission time, so the
        // placed data section comes first; `la` (pc-relative) is checked
        // against the same placed label from the text section.
        let src = r#"
            .org 0x8000
        FAR:
            .word 0x1abcdef
            .text
            la_abs t0, FAR
            lwu    a0, 0(t0)
            la     t1, FAR
            lw     a1, 0(t1)
            ecall
        "#;
        let prog = assemble_str(src).expect("assembles");
        assert_eq!(prog.data_base, 0x8000);
        assert_eq!(prog.labels["FAR"], 0x8000);
        let run = run_on_vm(&prog, &[], 64).expect("runs");
        assert_eq!(run.state.reg(10), 0x1abcdef);
        assert_eq!(run.state.reg(11), 0x1abcdef);
    }

    #[test]
    fn text_assembler_atomics_parse() {
        let src = r#"
            li     t0, 0x2000
            lr.w   t1, (t0)
            sc.w   t2, t1, (t0)
            amoadd.w t3, t1, (t0)
            ecall
        "#;
        let prog = assemble_str(src).expect("assembles");
        let run = run_on_vm(&prog, &[], 64).expect("runs");
        assert_eq!(run.state.reg(12), 0); // SC succeeded.
    }

    #[test]
    fn raw_word_roundtrip() {
        let mut asm = Assembler::new();
        asm.raw(0x00b5_0513).expect("raw"); // addi a0, a0, 11 (from VM tests)
        asm.ecall().expect("ecall");
        let prog = asm.finish().expect("finish");
        let run = run_on_vm(&prog, &[], 16).expect("runs");
        // addi a0, a0, 11 with a0 = 0.
        assert_eq!(run.state.reg(10), 11);
        // Decode directly.
        let word = u32::from_le_bytes(prog.code[0..4].try_into().expect("4"));
        assert_eq!(decode(0, word).expect("dec"), Instr::Addi { rd: 10, rs1: 10, imm: 11 });
    }

    /// The text front-end: every mnemonic (real + pseudo) round-trips
    /// through the VM decoder — complements `decode_table`, which covers
    /// the builder API.
    #[test]
    fn text_assembler_full_decode_roundtrip() {
        let src = "
            addi x1, x2, -2048
            slti x3, x4, 2047
            sltiu x5, x6, -1
            xori x7, x8, -5
            ori x9, x10, 0x7ff
            andi x11, x12, -2048
            slli x13, x14, 63
            srli x15, x16, 32
            srai x17, x18, 31
            addiw x19, x20, -3
            slliw x21, x22, 31
            srliw x23, x24, 0
            sraiw x25, x26, 15
            lui x27, -1
            auipc x28, -0x80000
            add x1, x2, x3
            sub x4, x5, x6
            sll x7, x8, x9
            slt x10, x11, x12
            sltu x13, x14, x15
            xor x16, x17, x18
            srl x19, x20, x21
            sra x22, x23, x24
            or x25, x26, x27
            and x28, x29, x30
            mul x1, x3, x5
            mulh x2, x4, x6
            mulhu x3, x5, x7
            div x4, x6, x8
            divu x5, x7, x9
            rem x6, x8, x10
            remu x7, x9, x11
            addw x8, x10, x12
            subw x9, x11, x13
            sllw x12, x14, x16
            srlw x13, x15, x17
            sraw x14, x16, x18
            mulw x15, x17, x19
            divw x16, x18, x20
            divuw x17, x19, x21
            remw x18, x20, x22
            remuw x19, x21, x23
            lw x20, -2048(x21)
            lwu x22, 2047(x23)
            ld x24, 8(x25)
            sw x26, -4(x27)
            sd x28, 16(x29)
            beq x1, x2, 8
            bne x3, x4, -4096
            blt x5, x6, 2
            bge x7, x8, -6
            bltu x9, x10, 16
            bgeu x11, x12, -1024
            jal x1, 8
            jalr x2, x3, -2048
            ecall
            ebreak
            lr.w x2, (x3)
            sc.w x4, x5, (x6)
            amoswap.w x7, x8, (x9)
            amoadd.w x10, x11, (x12)
            amoxor.w x13, x14, (x15)
            amoand.w x16, x17, (x18)
            amoor.w x19, x20, (x21)
            amomin.w x22, x23, (x24)
            amomax.w x25, x26, (x27)
            amominu.w x28, x29, (x30)
            amomaxu.w x31, x1, (x2)
            li x1, 5
            mv x2, x3
            nop
            j 8
            jal 8
            ret
            beqz x5, 8
            bnez x6, -4
        ";
        let want: Vec<Instr> = vec![
            Instr::Addi { rd: 1, rs1: 2, imm: -2048 },
            Instr::Slti { rd: 3, rs1: 4, imm: 2047 },
            Instr::Sltiu { rd: 5, rs1: 6, imm: (-1i64) as u64 },
            Instr::Xori { rd: 7, rs1: 8, imm: -5 },
            Instr::Ori { rd: 9, rs1: 10, imm: 0x7ff },
            Instr::Andi { rd: 11, rs1: 12, imm: -2048 },
            Instr::Slli { rd: 13, rs1: 14, shamt: 63 },
            Instr::Srli { rd: 15, rs1: 16, shamt: 32 },
            Instr::Srai { rd: 17, rs1: 18, shamt: 31 },
            Instr::Addiw { rd: 19, rs1: 20, imm: -3 },
            Instr::Slliw { rd: 21, rs1: 22, shamt: 31 },
            Instr::Srliw { rd: 23, rs1: 24, shamt: 0 },
            Instr::Sraiw { rd: 25, rs1: 26, shamt: 15 },
            Instr::Lui { rd: 27, imm: -4096 },
            Instr::Auipc { rd: 28, imm: (-0x80000i64) << 12 },
            Instr::Add { rd: 1, rs1: 2, rs2: 3 },
            Instr::Sub { rd: 4, rs1: 5, rs2: 6 },
            Instr::Sll { rd: 7, rs1: 8, rs2: 9 },
            Instr::Slt { rd: 10, rs1: 11, rs2: 12 },
            Instr::Sltu { rd: 13, rs1: 14, rs2: 15 },
            Instr::Xor { rd: 16, rs1: 17, rs2: 18 },
            Instr::Srl { rd: 19, rs1: 20, rs2: 21 },
            Instr::Sra { rd: 22, rs1: 23, rs2: 24 },
            Instr::Or { rd: 25, rs1: 26, rs2: 27 },
            Instr::And { rd: 28, rs1: 29, rs2: 30 },
            Instr::Mul { rd: 1, rs1: 3, rs2: 5 },
            Instr::Mulh { rd: 2, rs1: 4, rs2: 6 },
            Instr::Mulhu { rd: 3, rs1: 5, rs2: 7 },
            Instr::Div { rd: 4, rs1: 6, rs2: 8 },
            Instr::Divu { rd: 5, rs1: 7, rs2: 9 },
            Instr::Rem { rd: 6, rs1: 8, rs2: 10 },
            Instr::Remu { rd: 7, rs1: 9, rs2: 11 },
            Instr::Addw { rd: 8, rs1: 10, rs2: 12 },
            Instr::Subw { rd: 9, rs1: 11, rs2: 13 },
            Instr::Sllw { rd: 12, rs1: 14, rs2: 16 },
            Instr::Srlw { rd: 13, rs1: 15, rs2: 17 },
            Instr::Sraw { rd: 14, rs1: 16, rs2: 18 },
            Instr::Mulw { rd: 15, rs1: 17, rs2: 19 },
            Instr::Divw { rd: 16, rs1: 18, rs2: 20 },
            Instr::Divuw { rd: 17, rs1: 19, rs2: 21 },
            Instr::Remw { rd: 18, rs1: 20, rs2: 22 },
            Instr::Remuw { rd: 19, rs1: 21, rs2: 23 },
            Instr::Lw { rd: 20, rs1: 21, imm: -2048 },
            Instr::Lwu { rd: 22, rs1: 23, imm: 2047 },
            Instr::Ld { rd: 24, rs1: 25, imm: 8 },
            Instr::Sw { rs1: 27, rs2: 26, imm: -4 },
            Instr::Sd { rs1: 29, rs2: 28, imm: 16 },
            Instr::Beq { rs1: 1, rs2: 2, imm: 8 },
            Instr::Bne { rs1: 3, rs2: 4, imm: -4096 },
            Instr::Blt { rs1: 5, rs2: 6, imm: 2 },
            Instr::Bge { rs1: 7, rs2: 8, imm: -6 },
            Instr::Bltu { rs1: 9, rs2: 10, imm: 16 },
            Instr::Bgeu { rs1: 11, rs2: 12, imm: -1024 },
            Instr::Jal { rd: 1, imm: 8 },
            Instr::Jalr { rd: 2, rs1: 3, imm: -2048 },
            Instr::Ecall,
            Instr::Ebreak,
            Instr::LrW { rd: 2, rs1: 3, aq: false, rl: false },
            Instr::ScW { rd: 4, rs1: 6, rs2: 5, aq: false, rl: false },
            Instr::AmoSwapW { rd: 7, rs1: 9, rs2: 8, aq: false, rl: false },
            Instr::AmoAddW { rd: 10, rs1: 12, rs2: 11, aq: false, rl: false },
            Instr::AmoXorW { rd: 13, rs1: 15, rs2: 14, aq: false, rl: false },
            Instr::AmoAndW { rd: 16, rs1: 18, rs2: 17, aq: false, rl: false },
            Instr::AmoOrW { rd: 19, rs1: 21, rs2: 20, aq: false, rl: false },
            Instr::AmoMinW { rd: 22, rs1: 24, rs2: 23, aq: false, rl: false },
            Instr::AmoMaxW { rd: 25, rs1: 27, rs2: 26, aq: false, rl: false },
            Instr::AmoMinuW { rd: 28, rs1: 30, rs2: 29, aq: false, rl: false },
            Instr::AmoMaxuW { rd: 31, rs1: 2, rs2: 1, aq: false, rl: false },
            // Pseudo expansions.
            Instr::Addi { rd: 1, rs1: 0, imm: 5 },   // li x1, 5
            Instr::Addi { rd: 2, rs1: 3, imm: 0 },   // mv x2, x3
            Instr::Addi { rd: 0, rs1: 0, imm: 0 },   // nop
            Instr::Jal { rd: 0, imm: 8 },            // j 8
            Instr::Jal { rd: 1, imm: 8 },            // jal 8 (link in ra)
            Instr::Jalr { rd: 0, rs1: 1, imm: 0 },   // ret
            Instr::Beq { rs1: 5, rs2: 0, imm: 8 },   // beqz x5, 8
            Instr::Bne { rs1: 6, rs2: 0, imm: -4 },  // bnez x6, -4
        ];
        let prog = assemble_str(src).expect("assembles");
        assert_eq!(prog.code.len(), want.len() * 4, "one word per statement");
        for (i, w) in want.iter().enumerate() {
            let off = i * 4;
            let word = u32::from_le_bytes(prog.code[off..off + 4].try_into().expect("4 bytes"));
            let got = decode(off as u64, word).expect("decodes");
            assert_eq!(&got, w, "statement {i}");
        }
    }
}
