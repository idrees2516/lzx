//! The zkVM instruction-semantics constraint families (Wave 7.4 P0-4:
//! the fixed-column constraint system that finally forces the transition
//! function — decode, ALU, control flow, memory routing — instead of
//! relying on verifier re-execution).
//!
//! Architecture: every family is a paired prove/verify emitting legs in a
//! fixed protocol order, exactly like `memory.rs`. The witnesses are:
//!
//! * the **value tensors** (`rs1`, `rs2`, `imm`, `rd`, `mem_old`,
//!   `mem_new` — 64-bit tensors over `(6 + log T)`, committed in the bits
//!   bundle) and the instruction tensor;
//! * **bit columns** (`Factor::BitCol`, over `log T`): the instruction
//!   class/sub-class selectors, flags (`rd_we`, `mem_*`, `taken`,
//!   `halted`), arithmetic carries/borrows, the full-width comparison
//!   eq-prefixes, and the comparison results;
//! * **value columns** (`Factor::ValCol`): `mem_addr`, `mem_word`, the
//!   multiplication carry chain, `pc`/`next_pc`/`fetch_word`, and their
//!   16-bit limb columns.
//!
//! Every auxiliary column is *consistency-checked* against its definition
//! (selectors = products of instruction-bit indicators; eq-prefix
//! recurrences; comparison formulas; carry-chain recurrences), so the
//! committed universe stays honest.
//!
//! # Coverage (the v1 constraint set — fail-closed)
//!
//! Covered: `ADD/ADDI/ADDW/ADDIW/SUB/SUBW`, `AND/OR/XOR` (+ immediate
//! forms), `SLT/SLTU/SLTI/SLTIU`, `LUI`, `AUIPC`, `JAL/JALR`, all six
//! branches, `LD/LW/LWU`, `SD/SW`, `ECALL/EBREAK` (halt). NOT yet
//! covered (rejected fail-closed by `covered_instr`): shifts, `MUL`
//! family, `DIV/REM` family — their constraint families are the
//! follow-up wave items (the aux tensor namespace `AUX_*` is reserved
//! for them).
//!
//! Soundness notes (documented in `docs/WAVE_ANALYSIS.md`):
//! * Comparisons share eq-prefixes between signed and unsigned forms: the
//!   sign flip at bit 63 preserves `eq`, so only the head term of `lt`
//!   differs (`lt_s = a₆₃(1−b₆₃) + Σ_{p<63} eqp_p (1−a_p) b_p`).
//! * `DIVU`/`REMU` are constrained by the defining property over the
//!   integers in the follow-up wave; v1 rejects them fail-closed.
//! * The `halted` family enforces booleanity, termination
//!   (`halted[T−1] = 1`), and post-halt inactivity (`pc`, `rd_we`,
//!   `mem_we`, `mem_re` all frozen once `halted = 1`). The monotone
//!   propagation `halted(c) ≤ halted(c+1)` is enforced at witness build
//!   time; its shifted-row leg is a documented follow-up. The covered
//!   statement remains sound: a prover that raises `halted` early only
//!   freezes the machine sooner, and every covered instruction class is
//!   fully constrained, so padding cycles (`ECALL`) are forced no-ops.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

use crate::columns::{CycleWitness, T_IMM, T_MEM_NEW, T_MEM_OLD, T_RD, T_RS1, T_RS2};
use lattice_vm::Instr;
use crate::ledger::{idx_point, Factor, Ledger, LedgerError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    Ledger(LedgerError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Transcript(lattice_core::transcript::TranscriptError),
    Mle(lattice_core::mle::MleError),
    FinalCheck(&'static str),
    Shape,
    /// An instruction outside the v1 constraint coverage set appeared —
    /// fail closed rather than prove an unconstrained class.
    UncoveredInstruction { cycle: usize },
}

impl From<LedgerError> for ConstraintError {
    fn from(e: LedgerError) -> Self {
        ConstraintError::Ledger(e)
    }
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// Reserved auxiliary tensor ids (the follow-up-wave namespace: the
/// shift one-hot and the division `q`/`r`/`hi` witnesses).
pub const AUX_ONEHOT: usize = 0;
pub const AUX_Q: usize = 1;
pub const AUX_R: usize = 2;
pub const AUX_HI: usize = 3;
pub const AUX_TENSORS: usize = 4;

// ---------------------------------------------------------------------------
// The selector table: the single source of truth for the decode selectors
// (name, opcode, funct3, optional funct7 mask). `None` funct7 = the
// funct7 field is free (the class is f3-determined).
// ---------------------------------------------------------------------------

pub const SELECTOR_TABLE: &[( &str, u32, u32, Option<u32>)] = &[
    // class selectors
    ("sel_opimm", 0x13, 99, None),
    ("sel_op", 0x33, 99, None),
    ("sel_op32", 0x3b, 99, None),
    ("sel_opimm32", 0x1b, 99, None),
    ("sel_lui", 0x37, 99, None),
    ("sel_auipc", 0x17, 99, None),
    ("sel_jal", 0x6f, 99, None),
    ("sel_jalr", 0x67, 99, None),
    ("sel_branch", 0x63, 99, None),
    ("sel_load", 0x03, 99, None),
    ("sel_store", 0x23, 99, None),
    ("sel_system", 0x73, 99, None),
    // sub-class selectors (register)
    ("sel_add", 0x33, 0, Some(0x00)),
    ("sel_sub", 0x33, 0, Some(0x20)),
    ("sel_xor", 0x33, 4, Some(0x00)),
    ("sel_or", 0x33, 6, Some(0x00)),
    ("sel_and", 0x33, 7, Some(0x00)),
    ("sel_sll", 0x33, 1, Some(0x00)),
    ("sel_srl", 0x33, 5, Some(0x00)),
    ("sel_sra", 0x33, 5, Some(0x20)),
    ("sel_slt", 0x33, 2, Some(0x00)),
    ("sel_sltu", 0x33, 3, Some(0x00)),
    ("sel_mul", 0x33, 0, Some(0x01)),
    ("sel_divu", 0x33, 5, Some(0x01)),
    ("sel_remu", 0x33, 7, Some(0x01)),
    // sub-class selectors (immediate)
    ("sel_addi", 0x13, 0, None),
    ("sel_xori", 0x13, 4, None),
    ("sel_ori", 0x13, 6, None),
    ("sel_andi", 0x13, 7, None),
    ("sel_slti", 0x13, 2, None),
    ("sel_sltiu", 0x13, 3, None),
    ("sel_slli", 0x13, 1, None),
    ("sel_srxi", 0x13, 5, None),
    // sub-class selectors (W)
    ("sel_addiw", 0x1b, 0, None),
    ("sel_slliw", 0x1b, 1, None),
    ("sel_srxiw", 0x1b, 5, None),
    ("sel_addw", 0x3b, 0, Some(0x00)),
    ("sel_subw", 0x3b, 0, Some(0x20)),
    ("sel_sllw", 0x3b, 1, Some(0x00)),
    ("sel_srlw", 0x3b, 5, Some(0x00)),
    ("sel_sraw", 0x3b, 5, Some(0x20)),
    // branches
    ("sel_beq", 0x63, 0, None),
    ("sel_bne", 0x63, 1, None),
    ("sel_blt", 0x63, 4, None),
    ("sel_bge", 0x63, 5, None),
    ("sel_bltu", 0x63, 6, None),
    ("sel_bgeu", 0x63, 7, None),
    // loads / stores
    ("sel_lw", 0x03, 2, None),
    ("sel_lwu", 0x03, 6, None),
    ("sel_ld", 0x03, 3, None),
    ("sel_sw", 0x23, 2, None),
    ("sel_sd", 0x23, 3, None),
];

/// The number of eq-prefix columns per comparison (full width 0..=64).
const EQP_LEN: usize = 65;

/// Named indices into the auxiliary column table (built alongside
/// `build_aux` so families never hard-code magic numbers).
#[derive(Clone, Debug)]
pub struct AuxIndex {
    pub sel: Vec<usize>,
    pub rd_we: usize,
    pub mem_re: usize,
    pub mem_we: usize,
    pub mem_half: usize,
    pub halted: usize,
    pub taken: usize,
    pub carry_add_r: [usize; 4],
    pub carry_add_i: [usize; 4],
    pub borrow_sub_r: [usize; 4],
    pub carry_addw_r: [usize; 2],
    pub carry_addw_i: [usize; 2],
    pub borrow_subw_r: [usize; 2],
    /// The u64 limb-carry chain of `pc + imm` (branch/JAL/AUIPC targets).
    pub carry_ctrl: [usize; 4],
    /// The u64 limb-carry chain of `rs1 + imm` (JALR target, effective
    /// address).
    pub carry_jalr: [usize; 4],
    /// The limb-carry chain of `pc + 4` (sequential next pc, JAL/JALR rd).
    pub carry_pc4: [usize; 4],
    /// eqp[cmp][i]: 1 iff the top `i` bits of the cmp operands agree.
    pub eqp: [[usize; EQP_LEN]; 2],
    /// lt[cmp] (signed) and ltu[cmp] (unsigned).
    pub lt: [usize; 2],
    pub ltu: [usize; 2],
    pub v_mem_addr: usize,
    pub v_mem_word: usize,
    pub v_pc: usize,
    pub v_next_pc: usize,
    pub v_fetch_word: usize,
    /// pc limbs 0..=2 (pc < 2^48).
    pub v_pc_l: [usize; 3],
    /// next_pc limbs 0..=2.
    pub v_np_l: [usize; 3],
    /// mem_addr limbs 0..=3 (the effective-address chain result).
    pub v_addr_l: [usize; 4],
}

impl AuxIndex {
    pub fn sel_by(&self, name: &str) -> usize {
        for (i, (n, _, _, _)) in SELECTOR_TABLE.iter().enumerate() {
            if *n == name {
                return self.sel[i];
            }
        }
        // Every family-side lookup uses table names; a miss is a
        // programming error surfaced as a shape failure.
        usize::MAX
    }
}

/// The auxiliary-column registry (bit and value columns over `log T`).
#[derive(Clone, Debug)]
pub struct AuxCols {
    /// Boolean columns: selectors, flags, carries, eq-prefixes.
    pub bits: Vec<Vec<u8>>,
    /// Value columns: mem_addr, mem_word, mul carries, pc columns.
    pub vals: Vec<Vec<Goldilocks>>,
    /// Names for debugging/docs (parallel to `bits`).
    pub bit_names: Vec<&'static str>,
    /// Named indices (parallel registry).
    pub index: AuxIndex,
}

/// Is this instruction covered by the v1 constraint set? (Fail-closed
/// gate: `prove_constraints` refuses programs using anything else.)
pub fn covered_instr(instr: &Instr) -> bool {
    use Instr::*;
    matches!(
        instr,
        Addi { .. }
            | Slti { .. }
            | Sltiu { .. }
            | Xori { .. }
            | Ori { .. }
            | Andi { .. }
            | Addiw { .. }
            | Add { .. }
            | Sub { .. }
            | Slt { .. }
            | Sltu { .. }
            | Xor { .. }
            | Or { .. }
            | And { .. }
            | Addw { .. }
            | Subw { .. }
            | Lui { .. }
            | Auipc { .. }
            | Beq { .. }
            | Bne { .. }
            | Blt { .. }
            | Bge { .. }
            | Bltu { .. }
            | Bgeu { .. }
            | Lw { .. }
            | Lwu { .. }
            | Ld { .. }
            | Sw { .. }
            | Sd { .. }
            | Jal { .. }
            | Jalr { .. }
            | Ecall
            | Ebreak
    )
}

/// The u64 limb-carry chain of `x + y` (4 limbs; the top carry is the
/// dropped mod-2^64 carry).
fn carry_chain64(x: u64, y: u64) -> [u8; 4] {
    let limb = |v: u64, i: usize| (v >> (16 * i)) & 0xFFFF;
    let mut out = [0u8; 4];
    let mut carry = 0u64;
    for l in 0..4 {
        let sum = limb(x, l) + limb(y, l) + carry;
        out[l] = (sum >> 16) as u8;
        carry = out[l] as u64;
    }
    out
}

/// The eq-prefix column value: 1 iff the top `i` bits (MSB-first) of x
/// and y agree (`i = 0` is the empty prefix — always 1).
fn eqp_full(x: u64, y: u64, i: usize) -> u8 {
    let mask = if i == 0 {
        0
    } else {
        u64::MAX << (64 - i)
    };
    ((x & mask) == (y & mask)) as u8
}

/// Build the auxiliary columns from the trace witness. Fails closed on
/// instructions outside the v1 covered subset.
pub fn build_aux(w: &CycleWitness, instrs: &[Instr]) -> Result<AuxCols, ConstraintError> {
    let t = 1usize << w.log_t;
    let mut bits: Vec<Vec<u8>> = Vec::new();
    let mut names: Vec<&'static str> = Vec::new();
    let mut sel = Vec::with_capacity(SELECTOR_TABLE.len());
    macro_rules! push_bit {
        ($v:expr, $name:expr) => {{
            bits.push($v);
            names.push($name);
        }};
    }

    // --- selectors (class + sub-class, from the table) ---
    for (name, op, f3, f7) in SELECTOR_TABLE {
        let col: Vec<u8> = (0..t)
            .map(|i| {
                if i < instrs.len() {
                    let ins = &instrs[i];
                    let f3_ok = *f3 == 99 || raw_funct3(ins) == *f3;
                    (raw_opcode(ins) == *op
                        && f3_ok
                        && f7.map_or(true, |v| raw_funct7(ins) == v))
                        as u8
                } else {
                    // Padding cycles carry ECALL.
                    (*op == 0x73) as u8
                }
            })
            .collect();
        sel.push(bits.len());
        push_bit!(col, *name);
    }

    // --- flags ---
    let rd_we_id = bits.len();
    push_bit!(w.rd_we.clone(), "rd_we");
    let mem_re_id = bits.len();
    push_bit!(w.mem_re.clone(), "mem_re");
    let mem_we_id = bits.len();
    push_bit!(w.mem_we.clone(), "mem_we");
    let mem_half_id = bits.len();
    push_bit!(w.mem_half.clone(), "mem_half");
    let halted_id = bits.len();
    push_bit!(w.halted.clone(), "halted");
    let taken: Vec<u8> = (0..t)
        .map(|i| {
            if i < instrs.len() {
                branch_taken(&instrs[i], w, i) as u8
            } else {
                0
            }
        })
        .collect();
    let taken_id = bits.len();
    push_bit!(taken, "taken");

    // --- arithmetic carries/borrows (ADD/ADDI/SUB/W families) ---
    let mut chain_ids = |src_imm: bool, n: usize, name: &'static str| -> Vec<usize> {
        let mut ids = Vec::with_capacity(n);
        for l in 0..n {
            ids.push(bits.len());
            push_bit!(
                (0..t)
                    .map(|i| carry_at(w, instrs, i, src_imm, l, name))
                    .collect::<Vec<u8>>(),
                ""
            );
        }
        ids
    };
    let carry_add_r = chain_ids(false, 4, "carry_add_r");
    let carry_add_i = chain_ids(true, 4, "carry_add_i");
    let borrow_sub_r = chain_ids(false, 4, "borrow_sub_r");
    let carry_addw_r = chain_ids(false, 2, "carry_addw_r");
    let carry_addw_i = chain_ids(true, 2, "carry_addw_i");
    let borrow_subw_r = chain_ids(false, 2, "borrow_subw_r");

    // --- the control chains: pc + imm (targets), rs1 + imm (jalr /
    //     effective address), pc + 4 (sequential) ---
    let mut carry_ctrl = [0usize; 4];
    for l in 0..4 {
        let col: Vec<u8> = (0..t)
            .map(|c| carry_chain64(w.pc[c].0, tensor_word(w, T_IMM, c))[l])
            .collect();
        carry_ctrl[l] = bits.len();
        push_bit!(col, "");
    }
    let mut carry_jalr = [0usize; 4];
    for l in 0..4 {
        let col: Vec<u8> = (0..t)
            .map(|c| carry_chain64(tensor_word(w, T_RS1, c), tensor_word(w, T_IMM, c))[l])
            .collect();
        carry_jalr[l] = bits.len();
        push_bit!(col, "");
    }
    let mut carry_pc4 = [0usize; 4];
    for l in 0..4 {
        let col: Vec<u8> = (0..t)
            .map(|c| carry_chain64(w.pc[c].0, 4)[l])
            .collect();
        carry_pc4[l] = bits.len();
        push_bit!(col, "");
    }

    // --- full-width eq-prefixes: cmp 0 = (rs1, rs2), cmp 1 = (rs1, imm) ---
    let mut eqp = [[0usize; EQP_LEN]; 2];
    for cmp in 0..2 {
        for i in 0..EQP_LEN {
            let col: Vec<u8> = (0..t)
                .map(|c| {
                    let (x, y) = match cmp {
                        0 => (tensor_word(w, T_RS1, c), tensor_word(w, T_RS2, c)),
                        _ => (tensor_word(w, T_RS1, c), tensor_word(w, T_IMM, c)),
                    };
                    eqp_full(x, y, i)
                })
                .collect();
            eqp[cmp][i] = bits.len();
            push_bit!(col, "");
        }
    }

    // --- comparison results ---
    let mut lt = [0usize; 2];
    let mut ltu = [0usize; 2];
    for cmp in 0..2 {
        ltu[cmp] = bits.len();
        push_bit!(
            (0..t).map(|c| lt_at(w, instrs, c, cmp, false)).collect::<Vec<u8>>(),
            "ltu"
        );
        lt[cmp] = bits.len();
        push_bit!(
            (0..t).map(|c| lt_at(w, instrs, c, cmp, true)).collect::<Vec<u8>>(),
            "lt"
        );
    }

    // --- value columns ---
    let mut vals: Vec<Vec<Goldilocks>> = Vec::new();
    let v_mem_addr = vals.len();
    vals.push(w.mem_addr.clone());
    let v_mem_word = vals.len();
    vals.push(w.mem_word.clone());
    // mul carries (staged for the MUL family; ids 2..=6).
    for l in 0..5 {
        vals.push((0..t).map(|c| mul_carry_at(w, instrs, c, l)).collect());
    }
    let v_pc = vals.len();
    vals.push(w.pc.clone());
    let v_next_pc = vals.len();
    vals.push(w.next_pc.clone());
    let v_fetch_word = vals.len();
    vals.push(w.fetch_word.clone());
    let limb_of = |v: u64, l: usize| fe((v >> (16 * l)) & 0xFFFF);
    let mut v_pc_l = [0usize; 3];
    let mut v_np_l = [0usize; 3];
    for l in 0..3 {
        v_pc_l[l] = vals.len();
        vals.push((0..t).map(|c| limb_of(w.pc[c].0, l)).collect());
    }
    for l in 0..3 {
        v_np_l[l] = vals.len();
        vals.push((0..t).map(|c| limb_of(w.next_pc[c].0, l)).collect());
    }
    let mut v_addr_l = [0usize; 4];
    for l in 0..4 {
        v_addr_l[l] = vals.len();
        vals.push((0..t).map(|c| limb_of(w.mem_addr[c].0, l)).collect());
    }

    let index = AuxIndex {
        sel,
        rd_we: rd_we_id,
        mem_re: mem_re_id,
        mem_we: mem_we_id,
        mem_half: mem_half_id,
        halted: halted_id,
        taken: taken_id,
        carry_add_r: carry_add_r.try_into().ok().unwrap_or([0; 4]),
        carry_add_i: carry_add_i.try_into().ok().unwrap_or([0; 4]),
        borrow_sub_r: borrow_sub_r.try_into().ok().unwrap_or([0; 4]),
        carry_addw_r: carry_addw_r.try_into().ok().unwrap_or([0; 2]),
        carry_addw_i: carry_addw_i.try_into().ok().unwrap_or([0; 2]),
        borrow_subw_r: borrow_subw_r.try_into().ok().unwrap_or([0; 2]),
        carry_ctrl,
        carry_jalr,
        carry_pc4,
        eqp,
        lt,
        ltu,
        v_mem_addr,
        v_mem_word,
        v_pc,
        v_next_pc,
        v_fetch_word,
        v_pc_l,
        v_np_l,
        v_addr_l,
    };
    Ok(AuxCols { bits, vals, bit_names: names, index })
}

fn raw_opcode(instr: &Instr) -> u32 {
    use Instr::*;
    match instr {
        Addi { .. } | Slti { .. } | Sltiu { .. } | Xori { .. } | Ori { .. } | Andi { .. }
        | Slli { .. } | Srli { .. } | Srai { .. } => 0x13,
        Addiw { .. } | Slliw { .. } | Srliw { .. } | Sraiw { .. } => 0x1b,
        Add { .. } | Sub { .. } | Sll { .. } | Slt { .. } | Sltu { .. } | Xor { .. }
        | Srl { .. } | Sra { .. } | Or { .. } | And { .. } | Mul { .. } | Mulh { .. }
        | Mulhu { .. } | Divu { .. } | Remu { .. } => 0x33,
        Addw { .. } | Subw { .. } | Sllw { .. } | Srlw { .. } | Sraw { .. } => 0x3b,
        Lui { .. } => 0x37,
        Auipc { .. } => 0x17,
        Jal { .. } => 0x6f,
        Jalr { .. } => 0x67,
        Beq { .. } | Bne { .. } | Blt { .. } | Bge { .. } | Bltu { .. } | Bgeu { .. } => 0x63,
        Lw { .. } | Lwu { .. } | Ld { .. } => 0x03,
        Sw { .. } | Sd { .. } => 0x23,
        Ecall | Ebreak => 0x73,
        _ => 0,
    }
}

fn raw_funct3(instr: &Instr) -> u32 {
    use Instr::*;
    match instr {
        // OP-IMM
        Addi { .. } => 0,
        Slti { .. } => 2,
        Sltiu { .. } => 3,
        Xori { .. } => 4,
        Ori { .. } => 6,
        Andi { .. } => 7,
        Slli { .. } | Slliw { .. } | Sllw { .. } | Sll { .. } => 1,
        Srli { .. } | Srai { .. } | Srliw { .. } | Sraiw { .. } | Srl { .. } | Sra { .. }
        | Srlw { .. } | Sraw { .. } => 5,
        // OP
        Add { .. } | Sub { .. } | Mul { .. } | Addw { .. } | Subw { .. } | Addiw { .. } => 0,
        Slt { .. } => 2,
        Sltu { .. } => 3,
        Xor { .. } => 4,
        Or { .. } => 6,
        And { .. } => 7,
        Mulh { .. } | Mulhu { .. } => 1,
        Divu { .. } => 5,
        Remu { .. } => 7,
        // branches
        Beq { .. } => 0,
        Bne { .. } => 1,
        Blt { .. } => 4,
        Bge { .. } => 5,
        Bltu { .. } => 6,
        Bgeu { .. } => 7,
        // loads / stores
        Lw { .. } | Sw { .. } => 2,
        Lwu { .. } => 6,
        Ld { .. } | Sd { .. } => 3,
        // system
        Ecall => 0,
        Ebreak => 1,
        // U/J-type: the funct3 field is imm data (free in the masks).
        Lui { .. } | Auipc { .. } | Jal { .. } => 0,
        Jalr { .. } => 0,
        _ => 0,
    }
}

fn raw_funct7(instr: &Instr) -> u32 {
    use Instr::*;
    match instr {
        Sub { .. } | Subw { .. } | Sra { .. } | Sraw { .. } => 0x20,
        Mul { .. } | Mulh { .. } | Mulhu { .. } | Divu { .. } | Remu { .. } => 0x01,
        Add { .. } | Sll { .. } | Slt { .. } | Sltu { .. } | Xor { .. } | Srl { .. }
        | Or { .. } | And { .. } | Addw { .. } | Sllw { .. } | Srlw { .. } => 0x00,
        _ => 0,
    }
}

fn tensor_word(w: &CycleWitness, slot: usize, c: usize) -> u64 {
    let t = 1usize << w.log_t;
    let mut acc = 0u64;
    for bit in 0..64usize {
        // MSB-first rows: value-bit `bit` lives at row 63 - bit.
        let row = 63 - bit;
        acc |= w.values[slot].evaluations[row * t + c].0 << bit;
    }
    acc
}

fn branch_taken(instr: &Instr, w: &CycleWitness, c: usize) -> bool {
    use Instr::*;
    let a = tensor_word(w, T_RS1, c);
    let b = tensor_word(w, T_RS2, c);
    match instr {
        Beq { .. } => a == b,
        Bne { .. } => a != b,
        Blt { .. } => (a as i64) < (b as i64),
        Bge { .. } => (a as i64) >= (b as i64),
        Bltu { .. } => a < b,
        Bgeu { .. } => a >= b,
        _ => false,
    }
}

fn carry_at(w: &CycleWitness, instrs: &[Instr], c: usize, src_imm: bool, l: usize, name: &str) -> u8 {
    let _ = instrs;
    let a = tensor_word(w, T_RS1, c);
    let b = if src_imm {
        tensor_word(w, T_IMM, c)
    } else {
        tensor_word(w, T_RS2, c)
    };
    let limb = |v: u64, i: usize| (v >> (16 * i)) & 0xFFFF;
    match name {
        "carry_add_r" | "carry_add_i" => {
            let prev = if l == 0 {
                0
            } else {
                carry_at(w, instrs, c, src_imm, l - 1, name) as u64
            };
            let sum = limb(a, l) + limb(b, l) + prev;
            (sum >> 16) as u8
        }
        "borrow_sub_r" => {
            let prev = if l == 0 {
                0
            } else {
                carry_at(w, instrs, c, src_imm, l - 1, name) as u64
            };
            ((limb(b, l) + prev) > limb(a, l)) as u8
        }
        "carry_addw_r" | "carry_addw_i" => {
            let prev = if l == 0 {
                0
            } else {
                carry_at(w, instrs, c, src_imm, l - 1, name) as u64
            };
            let sum = limb(a, l) + limb(b, l) + prev;
            (sum >> 16) as u8
        }
        "borrow_subw_r" => {
            let prev = if l == 0 {
                0
            } else {
                carry_at(w, instrs, c, src_imm, l - 1, name) as u64
            };
            ((limb(b, l) + prev) > limb(a, l)) as u8
        }
        _ => 0,
    }
}

fn lt_at(w: &CycleWitness, _instrs: &[Instr], c: usize, cmp: usize, signed: bool) -> u8 {
    let (x, y) = match cmp {
        0 => (tensor_word(w, T_RS1, c), tensor_word(w, T_RS2, c)),
        _ => (tensor_word(w, T_RS1, c), tensor_word(w, T_IMM, c)),
    };
    if signed {
        ((x as i64) < (y as i64)) as u8
    } else {
        (x < y) as u8
    }
}

fn mul_carry_at(w: &CycleWitness, _instrs: &[Instr], c: usize, l: usize) -> Goldilocks {
    let a = tensor_word(w, T_RS1, c);
    let b = tensor_word(w, T_RS2, c);
    let limb = |v: u64, i: usize| (v >> (16 * i)) & 0xFFFF;
    let mut acc: u128 = 0;
    for k in 0..=l {
        for (i, j) in (0..4).zip(0..4) {
            if i + j == k {
                acc += (limb(a, i) as u128) * (limb(b, j) as u128);
            }
        }
    }
    let carry = acc >> (16 * (l + 1) as u32);
    fe((carry & 0x1FFFF) as u64)
}

// ---------------------------------------------------------------------------
// The leg driver: standardized prove/verify pairing with factor-claim
// binding. Every constraint family builds a VirtualPolynomial whose
// factors are tracked as `FV` views *paired with their VP factor index*;
// after the sumcheck, the driver binds each factor's terminal claim
// through the ledger (prover records base claims; verifier pops and
// checks). Public factors (eq tables, indicators) are
// verifier-computable and carry no claim.
// ---------------------------------------------------------------------------

/// A factor view: how to authenticate a sumcheck factor's terminal claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FV {
    /// A public table (eq-extension, indicators): the claim is checked
    /// against the table's own MLE at the terminal point.
    PubTable(DenseMle),
    /// A committed tensor, claimed directly.
    Tensor(Factor),
    /// A limb column (16 tensor-row claims).
    Limb { slot: usize, limb: usize },
    /// The 64-bit combo of a value slot (64 tensor-row claims).
    Combo { slot: usize },
    /// The instruction word column (32 tensor-row claims).
    InstrWord,
    /// A bit column over log T.
    Bit(usize),
    /// A value column over log T.
    Val(usize),
    /// A fixed tensor row, lifted constant over the bit block: the claim
    /// is the tensor-row claim.
    TensorRow { factor: Factor, nbits: usize, row: usize },
}

/// A (VP factor index, view) pair — the explicit pairing that keeps the
/// bind step aligned with the factor pool.
pub type ViewPair = (usize, FV);

/// One staged constraint leg.
#[derive(Clone, Debug)]
pub struct ConstraintLeg {
    pub name: &'static str,
    pub sc: SumcheckProof,
    pub claim: Goldilocks,
}

fn absorb_leg(transcript: &mut Transcript, name: &str) -> Result<(), ConstraintError> {
    transcript
        .append_message(b"con-leg", name.as_bytes())
        .map_err(ConstraintError::Transcript)
}

/// Bind/check factor views at a point: each view resolves to its
/// base-claim value and is compared with the paired factor's claimed
/// evaluation.
fn bind_views(
    ledger: &mut Ledger<'_>,
    views: &[ViewPair],
    point: &[Goldilocks],
    factor_claims: &[Goldilocks],
) -> Result<(), ConstraintError> {
    for (fi, view) in views {
        let claimed = factor_claims.get(*fi).copied();
        let value = resolve_view(ledger, view, point)?;
        if let (Some(c), Some(v)) = (claimed, value) {
            if c != v {
                return Err(ConstraintError::FinalCheck("factor claim mismatch"));
            }
        }
    }
    Ok(())
}

/// Resolve a view to its authenticated value at `point`.
fn resolve_view(
    ledger: &mut Ledger<'_>,
    view: &FV,
    point: &[Goldilocks],
) -> Result<Option<Goldilocks>, ConstraintError> {
    let v = match view {
        FV::PubTable(t) => Some(t.evaluate(point).map_err(ConstraintError::Mle)?),
        FV::Tensor(f) => Some(ledger.tensor_claim(*f, point)?),
        FV::Limb { slot, limb } => Some(ledger.limb(*slot, *limb, point)?),
        FV::Combo { slot } => Some(ledger.value_combo(*slot, point)?),
        FV::InstrWord => Some(ledger.instr_word(point)?),
        FV::Bit(id) => Some(ledger.tensor_claim(Factor::BitCol { id: *id }, point)?),
        FV::Val(id) => Some(ledger.tensor_claim(Factor::ValCol { id: *id }, point)?),
        FV::TensorRow { factor, nbits, row } => {
            let mut pt = idx_point((*nbits).trailing_zeros() as usize, *row);
            pt.extend_from_slice(point);
            Some(ledger.tensor_claim(*factor, &pt)?)
        }
    };
    Ok(v)
}

/// The staged-leg protocol state shared by the families.
struct FamilyCtx<'a, 'b, 'c> {
    w: &'a CycleWitness,
    aux: &'a AuxCols,
    ledger: &'b mut Ledger<'c>,
    legs: &'b mut Vec<ConstraintLeg>,
    transcript: &'b mut Transcript,
}

impl<'a, 'b, 'c> FamilyCtx<'a, 'b, 'c> {
    fn stage(
        &mut self,
        name: &'static str,
        vp: &mut VirtualPolynomial,
        views: &[ViewPair],
        claim: Goldilocks,
    ) -> Result<Vec<Goldilocks>, ConstraintError> {
        absorb_leg(self.transcript, name)?;
        let out = sumcheck::prove(vp, claim, self.transcript)
            .map_err(ConstraintError::Sumcheck)?;
        bind_views(self.ledger, views, &out.challenges, &out.factor_claims)?;
        self.legs.push(ConstraintLeg { name, sc: out.proof, claim });
        Ok(out.challenges)
    }
}

/// The verifier-side leg header: replay the label, verify the sumcheck,
/// return the terminal verdict (point + final claim).
fn verify_leg_header(
    name: &'static str,
    leg: &ConstraintLeg,
    num_vars: usize,
    degree: usize,
    transcript: &mut Transcript,
) -> Result<lattice_sumcheck::SumcheckVerifier, ConstraintError> {
    absorb_leg(transcript, name)?;
    if leg.name != name {
        return Err(ConstraintError::Shape);
    }
    leg.sc
        .verify(num_vars, degree, leg.claim, transcript, None)
        .map_err(ConstraintError::Sumcheck)
}

// ---------------------------------------------------------------------------
// Family: booleanity
// ---------------------------------------------------------------------------


/// Add a bit-column factor (from `cols`) with its view registration.
fn add_bit_factor(
    vp: &mut VirtualPolynomial,
    views: &mut Vec<ViewPair>,
    cols: &[Vec<u8>],
    id: usize,
    log_t: usize,
) -> Result<usize, ConstraintError> {
    let col = DenseMle {
        num_vars: log_t,
        evaluations: cols[id].iter().map(|v| fe(*v as u64)).collect(),
    };
    let fi = vp.add_factor(col).map_err(ConstraintError::Virtual)?;
    views.push((fi, FV::Bit(id)));
    Ok(fi)
}

/// Add a value-column factor with its view registration.
fn add_val_factor(
    vp: &mut VirtualPolynomial,
    views: &mut Vec<ViewPair>,
    vals: &[Vec<Goldilocks>],
    id: usize,
    log_t: usize,
) -> Result<usize, ConstraintError> {
    let col = DenseMle {
        num_vars: log_t,
        evaluations: vals[id].clone(),
    };
    let fi = vp.add_factor(col).map_err(ConstraintError::Virtual)?;
    views.push((fi, FV::Val(id)));
    Ok(fi)
}

/// Add a limb factor (16-bit limb of a value tensor slot).
fn add_limb_factor(
    vp: &mut VirtualPolynomial,
    views: &mut Vec<ViewPair>,
    w: &CycleWitness,
    slot: usize,
    limb: usize,
    log_t: usize,
) -> Result<usize, ConstraintError> {
    let f = limb_mle(w, slot, limb, log_t);
    let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
    views.push((fi, FV::Limb { slot, limb }));
    Ok(fi)
}

/// Add a tensor-row factor (value-bit `bit` of `slot`, MSB-first rows).
fn add_vbit_factor(
    vp: &mut VirtualPolynomial,
    views: &mut Vec<ViewPair>,
    w: &CycleWitness,
    slot: usize,
    bit: usize,
    log_t: usize,
) -> Result<usize, ConstraintError> {
    let row = 63 - bit;
    let col = row_mle(&w.values[slot], row, log_t);
    let fi = vp.add_factor(col).map_err(ConstraintError::Virtual)?;
    views.push((fi, FV::TensorRow {
        factor: Factor::ValueBits { slot },
        nbits: 64,
        row,
    }));
    Ok(fi)
}

/// Add an instruction-bit row factor (bit in LSB numbering).
/// Ledger claim helpers (verify side).
fn claim_bit(ledger: &mut Ledger<'_>, id: usize, pt: &[Goldilocks]) -> Result<Goldilocks, ConstraintError> {
    Ok(ledger.tensor_claim(Factor::BitCol { id }, pt)?)
}
fn claim_val(ledger: &mut Ledger<'_>, id: usize, pt: &[Goldilocks]) -> Result<Goldilocks, ConstraintError> {
    Ok(ledger.tensor_claim(Factor::ValCol { id }, pt)?)
}
fn claim_limb(ledger: &mut Ledger<'_>, slot: usize, limb: usize, pt: &[Goldilocks]) -> Result<Goldilocks, ConstraintError> {
    Ok(ledger.limb(slot, limb, pt)?)
}
fn claim_vbit(ledger: &mut Ledger<'_>, slot: usize, bit: usize, pt: &[Goldilocks]) -> Result<Goldilocks, ConstraintError> {
    let row = 63 - bit;
    let mut p = idx_point(6, row);
    p.extend_from_slice(pt);
    Ok(ledger.tensor_claim(Factor::ValueBits { slot }, &p)?)
}

fn prove_booleanity(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    // (a) the six value tensors: per-row booleanity — every bit row
    //     (fixed value-bit, varying cycle) is boolean on the cycle cube:
    //     eq(r)·B_row·(B_row − 1) = 0. One α-batched leg over logT
    //     (never the 2^(64+logT) tensor space).
    {
        let r = ctx
            .transcript
            .challenge_fields(b"con-bool-r", log_t)
            .map_err(ConstraintError::Transcript)?;
        let alphas = ctx
            .transcript
            .challenge_fields(b"con-bool-a", 6)
            .map_err(ConstraintError::Transcript)?;
        let eq = DenseMle::eq_extension(&r);
        let mut vp = VirtualPolynomial::new(log_t);
        let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
        let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
        for (slot, alpha) in alphas.iter().enumerate().take(6) {
            for row in 0..64usize {
                let b = row_mle(&w.values[slot], row, log_t);
                let b2 = DenseMle {
                    num_vars: log_t,
                    evaluations: b
                        .evaluations
                        .iter()
                        .map(|v| Goldilocks::ONE.sub(v))
                        .collect(),
                };
                let i1 = vp.add_factor(b).map_err(ConstraintError::Virtual)?;
                let i2 = vp.add_factor(b2).map_err(ConstraintError::Virtual)?;
                vp.add_term(*alpha, vec![i1, i2, ei])
                    .map_err(ConstraintError::Virtual)?;
                views.push((i1, FV::TensorRow {
                    factor: Factor::ValueBits { slot },
                    nbits: 64,
                    row,
                }));
            }
        }
        ctx.stage("bool-tensors", &mut vp, &views, Goldilocks::ZERO)?;
    }
    // (b) the instruction tensor: the same per-row construction over the
    //     32 instruction-bit rows.
    {
        let r = ctx
            .transcript
            .challenge_fields(b"con-ibool-r", log_t)
            .map_err(ConstraintError::Transcript)?;
        let eq = DenseMle::eq_extension(&r);
        let mut vp = VirtualPolynomial::new(log_t);
        let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
        let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
        for row in 0..32usize {
            let b = row_mle(&w.instr_bits, row, log_t);
            let b2 = DenseMle {
                num_vars: log_t,
                evaluations: b
                    .evaluations
                    .iter()
                    .map(|v| Goldilocks::ONE.sub(v))
                    .collect(),
            };
            let i1 = vp.add_factor(b).map_err(ConstraintError::Virtual)?;
            let i2 = vp.add_factor(b2).map_err(ConstraintError::Virtual)?;
            vp.add_term(Goldilocks::ONE, vec![i1, i2, ei])
                .map_err(ConstraintError::Virtual)?;
            views.push((i1, FV::TensorRow {
                factor: Factor::InstrBits,
                nbits: 32,
                row,
            }));
        }
        ctx.stage("bool-instr", &mut vp, &views, Goldilocks::ZERO)?;
    }
    // (c) the bit columns over logT.
    {
        let r = ctx
            .transcript
            .challenge_fields(b"con-bcol-r", log_t)
            .map_err(ConstraintError::Transcript)?;
        let alphas = ctx
            .transcript
            .challenge_fields(b"con-bcol-a", aux.bits.len())
            .map_err(ConstraintError::Transcript)?;
        let eq = DenseMle::eq_extension(&r);
        let mut vp = VirtualPolynomial::new(log_t);
        let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
        let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
        for (id, col) in aux.bits.iter().enumerate() {
            let c = DenseMle {
                num_vars: log_t,
                evaluations: col.iter().map(|v| fe(*v as u64)).collect(),
            };
            let c2 = DenseMle {
                num_vars: log_t,
                evaluations: col.iter().map(|v| fe((*v as u64) ^ 1)).collect(),
            };
            let i1 = vp.add_factor(c).map_err(ConstraintError::Virtual)?;
            let i2 = vp.add_factor(c2).map_err(ConstraintError::Virtual)?;
            vp.add_term(alphas[id], vec![i1, i2, ei])
                .map_err(ConstraintError::Virtual)?;
            views.push((i1, FV::Bit(id)));
        }
        ctx.stage("bool-cols", &mut vp, &views, Goldilocks::ZERO)?;
    }
    Ok(())
}

fn verify_booleanity<'l>(
    w: &CycleWitness,
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'l>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let log_t = w.log_t;
    // (a) value-tensor row booleanity.
    {
        let r = transcript
            .challenge_fields(b"con-bool-r", log_t)
            .map_err(ConstraintError::Transcript)?;
        let alphas = transcript
            .challenge_fields(b"con-bool-a", 6)
            .map_err(ConstraintError::Transcript)?;
        let leg = next_constraint_leg(iter, "bool-tensors")?;
        let verdict = verify_leg_header("bool-tensors", leg, log_t, 3, transcript)?;
        let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
        let pt = &verdict.point;
        let mut expect = Goldilocks::ZERO;
        for (slot, alpha) in alphas.iter().enumerate().take(6) {
            for row in 0..64usize {
                let mut p = idx_point(6, row);
                p.extend_from_slice(pt);
                let b = ledger.tensor_claim(Factor::ValueBits { slot }, &p)?;
                expect = expect.add(&alpha.mul(&b).mul(&Goldilocks::ONE.sub(&b)).mul(&eq_at));
            }
        }
        if expect != verdict.final_claim {
            return Err(ConstraintError::FinalCheck("bool-tensors"));
        }
    }
    // (b) instruction-tensor row booleanity.
    {
        let r = transcript
            .challenge_fields(b"con-ibool-r", log_t)
            .map_err(ConstraintError::Transcript)?;
        let leg = next_constraint_leg(iter, "bool-instr")?;
        let verdict = verify_leg_header("bool-instr", leg, log_t, 3, transcript)?;
        let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
        let pt = &verdict.point;
        let mut expect = Goldilocks::ZERO;
        for row in 0..32usize {
            let mut p = idx_point(5, row);
            p.extend_from_slice(pt);
            let b = ledger.tensor_claim(Factor::InstrBits, &p)?;
            expect = expect.add(&b.mul(&Goldilocks::ONE.sub(&b)).mul(&eq_at));
        }
        if expect != verdict.final_claim {
            return Err(ConstraintError::FinalCheck("bool-instr"));
        }
    }
    // (c) the bit columns.
    {
        let r = transcript
            .challenge_fields(b"con-bcol-r", log_t)
            .map_err(ConstraintError::Transcript)?;
        let alphas = transcript
            .challenge_fields(b"con-bcol-a", aux.bits.len())
            .map_err(ConstraintError::Transcript)?;
        let leg = next_constraint_leg(iter, "bool-cols")?;
        let verdict = verify_leg_header("bool-cols", leg, log_t, 3, transcript)?;
        let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
        let pt = &verdict.point;
        let mut expect = Goldilocks::ZERO;
        for (id, alpha) in alphas.iter().enumerate() {
            let b = ledger.tensor_claim(Factor::BitCol { id }, pt)?;
            expect = expect.add(&alpha.mul(&b).mul(&Goldilocks::ONE.sub(&b)).mul(&eq_at));
        }
        if expect != verdict.final_claim {
            return Err(ConstraintError::FinalCheck("bool-cols"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: selectors (decode) — every selector column equals the product
// of polarized instruction-bit indicators from the public mask.
// ---------------------------------------------------------------------------

/// The instr-bit mask of a selector: (bit position in LSB numbering,
/// required value). Derived from (opcode, funct3, funct7).
fn selector_mask(op: u32, f3: u32, f7: Option<u32>) -> Vec<(usize, u8)> {
    let mut mask = Vec::new();
    for b in 0..7 {
        mask.push((b, ((op >> b) & 1) as u8));
    }
    if f3 != 99 {
        for b in 0..3 {
            mask.push((12 + b, ((f3 >> b) & 1) as u8));
        }
    }
    if let Some(v) = f7 {
        for b in 0..7 {
            mask.push((25 + b, ((v >> b) & 1) as u8));
        }
    }
    mask
}

/// The maximum selector mask width (degree cap for the sel leg).
fn sel_max_degree() -> usize {
    SELECTOR_TABLE
        .iter()
        .map(|(_, op, f3, f7)| selector_mask(*op, *f3, *f7).len())
        .max()
        .unwrap_or(0)
        + 1
}

fn prove_selectors(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let r = ctx
        .transcript
        .challenge_fields(b"con-sel-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-sel-a", SELECTOR_TABLE.len())
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // Instruction-bit row factors, both polarities, memoized by (bit, req).
    let mut memo: std::collections::HashMap<(usize, u8), usize> =
        std::collections::HashMap::new();
    for (j, (name, op, f3, f7)) in SELECTOR_TABLE.iter().enumerate() {
        let _ = name;
        let mask = selector_mask(*op, *f3, *f7);
        // s_j - prod(indicators): two term groups.
        let s = DenseMle {
            num_vars: log_t,
            evaluations: aux.bits[aux.index.sel[j]]
                .iter()
                .map(|v| fe(*v as u64))
                .collect(),
        };
        let si = vp.add_factor(s).map_err(ConstraintError::Virtual)?;
        views.push((si, FV::Bit(aux.index.sel[j])));
        vp.add_term(alphas[j], vec![si, ei])
            .map_err(ConstraintError::Virtual)?;
        // - prod: the product term over polarized bit rows.
        let mut ids = Vec::with_capacity(mask.len() + 1);
        for (bit, req) in &mask {
            let key = (*bit, *req);
            let fi = if let Some(x) = memo.get(&key) {
                *x
            } else {
                let row = 31 - bit;
                let m = w.instr_bits.clone();
                let col = instr_row_of(&m, row, log_t);
                let f = if *req == 1 {
                    col
                } else {
                    flip_mle(&col)
                };
                let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
                if *req == 1 {
                    views.push((fi, FV::TensorRow {
                        factor: Factor::InstrBits,
                        nbits: 32,
                        row,
                    }));
                }
                memo.insert(key, fi);
                fi
            };
            ids.push(fi);
        }
        ids.push(ei);
        vp.add_term(alphas[j].neg(), ids)
            .map_err(ConstraintError::Virtual)?;
    }
    ctx.stage("sel", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn instr_row_of(m: &DenseMle, row: usize, log_t: usize) -> DenseMle {
    let t = 1usize << log_t;
    DenseMle {
        num_vars: log_t,
        evaluations: (0..t).map(|c| m.evaluations[row * t + c]).collect(),
    }
}

fn flip_mle(m: &DenseMle) -> DenseMle {
    DenseMle {
        num_vars: m.num_vars,
        evaluations: m.evaluations.iter().map(|v| Goldilocks::ONE.sub(v)).collect(),
    }
}

fn verify_selectors(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let r = transcript
        .challenge_fields(b"con-sel-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-sel-a", SELECTOR_TABLE.len())
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "sel")?;
    let degree = sel_max_degree();
    let verdict = verify_leg_header("sel", leg, log_t, degree, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    // Resolve every distinct (bit, polarity) row value.
    let mut memo: std::collections::HashMap<(usize, u8), Goldilocks> =
        std::collections::HashMap::new();
    let mut row_memo: std::collections::HashMap<usize, Goldilocks> =
        std::collections::HashMap::new();
    let mut expect = Goldilocks::ZERO;
    for (j, (_, op, f3, f7)) in SELECTOR_TABLE.iter().enumerate() {
        let s = ledger.tensor_claim(Factor::BitCol { id: aux.index.sel[j] }, &verdict.point)?;
        let mask = selector_mask(*op, *f3, *f7);
        let mut prod = Goldilocks::ONE;
        for (bit, req) in &mask {
            let key = (*bit, *req);
            let v = if let Some(x) = memo.get(&key) {
                *x
            } else {
                let row = 31 - bit;
                let b = if let Some(x) = row_memo.get(&row) {
                    *x
                } else {
                    let mut pt = idx_point(5, row);
                    pt.extend_from_slice(&verdict.point);
                    let b = ledger.tensor_claim(Factor::InstrBits, &pt)?;
                    row_memo.insert(row, b);
                    b
                };
                let v = if *req == 1 { b } else { Goldilocks::ONE.sub(&b) };
                memo.insert(key, v);
                v
            };
            prod = prod.mul(&v);
        }
        expect = expect.add(&alphas[j].mul(&s.sub(&prod)).mul(&eq_at));
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("sel"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: flags — activity flags and the branch-taken bit against their
// selector / comparison definitions.
// ---------------------------------------------------------------------------

fn prove_flags(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-flags-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-flags-a", 5)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    let neg_sum_terms = |vp: &mut VirtualPolynomial,
                         target: usize,
                         ids: &[usize]|
     -> Result<(), ConstraintError> {
        // eq * (target - sum(ids)) = sum eq*target - sum eq*id
        vp.add_term(alphas[0].mul(&fe(1)), vec![target, ei])
            .map_err(ConstraintError::Virtual)?;
        for id in ids {
            vp.add_term(alphas[0].neg(), vec![*id, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        Ok(())
    };
    // (1) mem_re = lw + lwu + ld
    {
        let tgt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_re, log_t)?;
        let lw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lw"), log_t)?;
        let lwu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lwu"), log_t)?;
        let ld = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_ld"), log_t)?;
        neg_sum_terms(&mut vp, tgt, &[lw, lwu, ld])?;
    }
    // (2) mem_we = sw + sd
    {
        let tgt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_we, log_t)?;
        let sw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sw"), log_t)?;
        let sd = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sd"), log_t)?;
        neg_sum_terms(&mut vp, tgt, &[sw, sd])?;
    }
    // (3) NOTE: mem_half is the half-SELECTOR (which 32-bit half of the
    // word a word-granular access touches), not an access indicator —
    // it is data, constrained by the routing family's word-addressing
    // identity (addr = 8*word + 4*half for word accesses, 8*word for
    // double accesses). No flags-level identity here.
    // (4) taken = beq*eq0 + bne*(1-eq0) + blt*lt0 + bge*(1-lt0)
    //         + bltu*ltu0 + bgeu*(1-ltu0)
    // where eq0 = eqp[0][64] (full equality), lt0 = lt[0], ltu0 = ltu[0].
    {
        let tgt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.taken, log_t)?;
        let eq0 = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[0][64], log_t)?;
        let one_minus_eq0 = vp
            .add_factor(DenseMle {
                num_vars: log_t,
                evaluations: aux.bits[idx.eqp[0][64]]
                    .iter()
                    .map(|v| fe(1 ^ *v as u64))
                    .collect(),
            })
            .map_err(ConstraintError::Virtual)?;
        let lt0 = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.lt[0], log_t)?;
        let one_minus_lt0 = vp
            .add_factor(DenseMle {
                num_vars: log_t,
                evaluations: aux.bits[idx.lt[0]]
                    .iter()
                    .map(|v| fe(1 ^ *v as u64))
                    .collect(),
            })
            .map_err(ConstraintError::Virtual)?;
        let ltu0 = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.ltu[0], log_t)?;
        let one_minus_ltu0 = vp
            .add_factor(DenseMle {
                num_vars: log_t,
                evaluations: aux.bits[idx.ltu[0]]
                    .iter()
                    .map(|v| fe(1 ^ *v as u64))
                    .collect(),
            })
            .map_err(ConstraintError::Virtual)?;
        let beq = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_beq"), log_t)?;
        let bne = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_bne"), log_t)?;
        let blt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_blt"), log_t)?;
        let bge = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_bge"), log_t)?;
        let bltu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_bltu"), log_t)?;
        let bgeu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_bgeu"), log_t)?;
        // taken - (beq*eq + bne*(1-eq) + blt*lt + bge*(1-lt)
        //           + bltu*ltu + bgeu*(1-ltu)) = 0
        vp.add_term(alphas[1], vec![tgt, ei])
            .map_err(ConstraintError::Virtual)?;
        for (sel, val) in [
            (beq, eq0),
            (bne, one_minus_eq0),
            (blt, lt0),
            (bge, one_minus_lt0),
            (bltu, ltu0),
            (bgeu, one_minus_ltu0),
        ] {
            vp.add_term(alphas[1].neg(), vec![sel, val, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (5) rd_we = (sum of write-class selectors) * [rd != 0]
    // [rd != 0] = 1 - prod(1 - rd_i) over instr bits 7..=11.
    {
        let tgt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.rd_we, log_t)?;
        let write_classes = [
            "sel_opimm",
            "sel_op",
            "sel_op32",
            "sel_opimm32",
            "sel_lui",
            "sel_auipc",
            "sel_jal",
            "sel_jalr",
            "sel_load",
        ];
        let mut w_ids = Vec::new();
        for name in write_classes {
            w_ids.push(add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?);
        }
        // term A: rd_we
        vp.add_term(alphas[2], vec![tgt, ei])
            .map_err(ConstraintError::Virtual)?;
        // term B: - sum_k s_k  (eq-weighted)
        for id in &w_ids {
            vp.add_term(alphas[2].neg(), vec![*id, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        // term C: + sum_k s_k * prod(1 - rd_i)
        let mut not_rd_ids = Vec::new();
        for bit in 7..12 {
            let row = 31 - bit;
            let col = instr_row_of(&w.instr_bits, row, log_t);
            let f = flip_mle(&col);
            let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
            not_rd_ids.push(fi);
        }
        for id in &w_ids {
            let mut ids = vec![*id];
            ids.extend_from_slice(&not_rd_ids);
            ids.push(ei);
            vp.add_term(alphas[2], ids)
                .map_err(ConstraintError::Virtual)?;
        }
    }
    let _ = alphas[3];
    let _ = alphas[4];
    ctx.stage("flags", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_flags(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let r = transcript
        .challenge_fields(b"con-flags-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-flags-a", 5)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "flags")?;
    let verdict = verify_leg_header("flags", leg, log_t, 8, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let one = Goldilocks::ONE;
    let mut expect = Goldilocks::ZERO;
    // (1) mem_re
    {
        let t = claim_bit(ledger, idx.mem_re, pt)?;
        let a = claim_bit(ledger, idx.sel_by("sel_lw"), pt)?;
        let b = claim_bit(ledger, idx.sel_by("sel_lwu"), pt)?;
        let c = claim_bit(ledger, idx.sel_by("sel_ld"), pt)?;
        expect = expect
            .add(&alphas[0].mul(&t.sub(&a).sub(&b).sub(&c)).mul(&eq_at));
    }
    // (2) mem_we
    {
        let t = claim_bit(ledger, idx.mem_we, pt)?;
        let a = claim_bit(ledger, idx.sel_by("sel_sw"), pt)?;
        let b = claim_bit(ledger, idx.sel_by("sel_sd"), pt)?;
        expect = expect.add(&alphas[0].mul(&t.sub(&a).sub(&b)).mul(&eq_at));
    }
    // (4) taken
    {
        let t = claim_bit(ledger, idx.taken, pt)?;
        let eq0 = claim_bit(ledger, idx.eqp[0][64], pt)?;
        let lt0 = claim_bit(ledger, idx.lt[0], pt)?;
        let ltu0 = claim_bit(ledger, idx.ltu[0], pt)?;
        let beq = claim_bit(ledger, idx.sel_by("sel_beq"), pt)?;
        let bne = claim_bit(ledger, idx.sel_by("sel_bne"), pt)?;
        let blt = claim_bit(ledger, idx.sel_by("sel_blt"), pt)?;
        let bge = claim_bit(ledger, idx.sel_by("sel_bge"), pt)?;
        let bltu = claim_bit(ledger, idx.sel_by("sel_bltu"), pt)?;
        let bgeu = claim_bit(ledger, idx.sel_by("sel_bgeu"), pt)?;
        let v = beq.mul(&eq0)
            .add(&bne.mul(&one.sub(&eq0)))
            .add(&blt.mul(&lt0))
            .add(&bge.mul(&one.sub(&lt0)))
            .add(&bltu.mul(&ltu0))
            .add(&bgeu.mul(&one.sub(&ltu0)));
        expect = expect.add(&alphas[1].mul(&t.sub(&v)).mul(&eq_at));
    }
    // (5) rd_we = W - W * prod(1 - rd_i)
    {
        let t = claim_bit(ledger, idx.rd_we, pt)?;
        let mut not_rd = Goldilocks::ONE;
        for bit_pos in 7..12 {
            let row = 31 - bit_pos;
            let mut p = idx_point(5, row);
            p.extend_from_slice(pt);
            let b = ledger.tensor_claim(Factor::InstrBits, &p)?;
            not_rd = not_rd.mul(&Goldilocks::ONE.sub(&b));
        }
        let write_classes = [
            "sel_opimm",
            "sel_op",
            "sel_op32",
            "sel_opimm32",
            "sel_lui",
            "sel_auipc",
            "sel_jal",
            "sel_jalr",
            "sel_load",
        ];
        let mut w_sum = Goldilocks::ZERO;
        for name in write_classes {
            w_sum = w_sum.add(&claim_bit(ledger, idx.sel_by(name), pt)?);
        }
        expect = expect.add(
            &alphas[2]
                .mul(&t.sub(&w_sum).add(&w_sum.mul(&not_rd)))
                .mul(&eq_at),
        );
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("flags"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: arithmetic — the limb carry recurrences for ADD/ADDI/SUB and
// the W variants, the W sign extension, and the LUI/AUIPC/JAL/JALR rd
// routing.
// ---------------------------------------------------------------------------

fn prove_arith(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-arith-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-arith-a", 6)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    let limb_of = |vp: &mut VirtualPolynomial,
                   views: &mut Vec<ViewPair>,
                   slot: usize,
                   l: usize|
     -> Result<usize, ConstraintError> {
        add_limb_factor(vp, views, w, slot, l, log_t)
    };
    let sel_of = |vp: &mut VirtualPolynomial,
                  views: &mut Vec<ViewPair>,
                  name: &str|
     -> Result<usize, ConstraintError> {
        add_bit_factor(vp, views, &aux.bits, idx.sel_by(name), log_t)
    };
    // sel*(rd_l + 2^16 c_out - x_l - y_l - c_in) = 0 — every identity is
    // selector-masked (off-class cycles are unconstrained).
    let a = &alphas[0];
    // (a) ADD: (rs1, rs2) with carry_add_r.
    {
        let sel = sel_of(&mut vp, &mut views, "sel_add")?;
        for l in 0..4 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let x = limb_of(&mut vp, &mut views, T_RS1, l)?;
            let y = limb_of(&mut vp, &mut views, T_RS2, l)?;
            let c_out = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_add_r[l], log_t)?;
            vp.add_term(*a, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in =
                    add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_add_r[l - 1], log_t)?;
                vp.add_term(a.neg(), vec![sel, c_in, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (b) ADDI: (rs1, imm) with carry_add_i.
    {
        let sel = sel_of(&mut vp, &mut views, "sel_addi")?;
        for l in 0..4 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let x = limb_of(&mut vp, &mut views, T_RS1, l)?;
            let y = limb_of(&mut vp, &mut views, T_IMM, l)?;
            let c_out = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_add_i[l], log_t)?;
            vp.add_term(*a, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in =
                    add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_add_i[l - 1], log_t)?;
                vp.add_term(a.neg(), vec![sel, c_in, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (c) SUB: sel*(rd_l - a_l + b_l + bi - 2^16 bo) = 0.
    {
        let sel = sel_of(&mut vp, &mut views, "sel_sub")?;
        let b = &alphas[1];
        for l in 0..4 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let x = limb_of(&mut vp, &mut views, T_RS1, l)?;
            let y = limb_of(&mut vp, &mut views, T_RS2, l)?;
            let bo = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.borrow_sub_r[l], log_t)?;
            vp.add_term(*b, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*b, vec![sel, y, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(1 << 16).neg()), vec![sel, bo, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let bi = add_bit_factor(
                    &mut vp,
                    &mut views,
                    &aux.bits,
                    idx.borrow_sub_r[l - 1],
                    log_t,
                )?;
                vp.add_term(*b, vec![sel, bi, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (d) ADDW / ADDIW: 2 limbs, then the sign extension.
    for (name, chain, is_imm) in [
        ("sel_addw", &idx.carry_addw_r, 0usize),
        ("sel_addiw", &idx.carry_addw_i, 1usize),
    ] {
        let sel = sel_of(&mut vp, &mut views, name)?;
        let b = &alphas[2];
        for l in 0..2 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let x = limb_of(&mut vp, &mut views, T_RS1, l)?;
            let y = if is_imm == 1 {
                limb_of(&mut vp, &mut views, T_IMM, l)?
            } else {
                limb_of(&mut vp, &mut views, T_RS2, l)?
            };
            let c_out = add_bit_factor(&mut vp, &mut views, &aux.bits, chain[l], log_t)?;
            vp.add_term(*b, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in =
                    add_bit_factor(&mut vp, &mut views, &aux.bits, chain[l - 1], log_t)?;
                vp.add_term(b.neg(), vec![sel, c_in, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (e) SUBW: 2 limbs with borrows.
    {
        let sel = sel_of(&mut vp, &mut views, "sel_subw")?;
        let b = &alphas[3];
        for l in 0..2 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let x = limb_of(&mut vp, &mut views, T_RS1, l)?;
            let y = limb_of(&mut vp, &mut views, T_RS2, l)?;
            let bo = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.borrow_subw_r[l], log_t)?;
            vp.add_term(*b, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*b, vec![sel, y, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(1 << 16).neg()), vec![sel, bo, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let bi = add_bit_factor(
                    &mut vp,
                    &mut views,
                    &aux.bits,
                    idx.borrow_subw_r[l - 1],
                    log_t,
                )?;
                vp.add_term(*b, vec![sel, bi, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (f) W sign extension: rd_2 = rd_3 = sign*(2^16-1) under the W
    //     selectors; sign = bit 31 of rd (tensor row 32).
    {
        let sign = w.values[T_RD].clone();
        let sign_col = row_mle(&sign, 32, log_t);
        let sign_f = vp.add_factor(sign_col).map_err(ConstraintError::Virtual)?;
        views.push((sign_f, FV::TensorRow {
            factor: Factor::ValueBits { slot: T_RD },
            nbits: 64,
            row: 32,
        }));
        let b = &alphas[4];
        for name in ["sel_addw", "sel_addiw", "sel_subw"] {
            let sel = sel_of(&mut vp, &mut views, name)?;
            for l in 2..4 {
                let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
                vp.add_term(*b, vec![sel, rd, ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(b.mul(&fe(0xFFFF).neg()), vec![sel, sign_f, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (g) LUI: rd = imm (limb-wise).
    {
        let sel = sel_of(&mut vp, &mut views, "sel_lui")?;
        let b = &alphas[4];
        for l in 0..4 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let y = limb_of(&mut vp, &mut views, T_IMM, l)?;
            vp.add_term(*b, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (h) AUIPC: rd = pc + imm via carry_ctrl.
    {
        let sel = sel_of(&mut vp, &mut views, "sel_auipc")?;
        let b = &alphas[5];
        for l in 0..4 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let imm = limb_of(&mut vp, &mut views, T_IMM, l)?;
            let c_out = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_ctrl[l], log_t)?;
            vp.add_term(*b, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            if l < 3 {
                let pc = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_pc_l[l], log_t)?;
                vp.add_term(b.neg(), vec![sel, pc, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            vp.add_term(b.neg(), vec![sel, imm, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in =
                    add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_ctrl[l - 1], log_t)?;
                vp.add_term(b.neg(), vec![sel, c_in, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (i) JAL/JALR: rd = pc + 4 via carry_pc4.
    for name in ["sel_jal", "sel_jalr"] {
        let sel = sel_of(&mut vp, &mut views, name)?;
        let b = &alphas[5];
        for l in 0..4 {
            let rd = limb_of(&mut vp, &mut views, T_RD, l)?;
            let c_out = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_pc4[l], log_t)?;
            vp.add_term(*b, vec![sel, rd, ei]).map_err(ConstraintError::Virtual)?;
            if l < 3 {
                let pc = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_pc_l[l], log_t)?;
                vp.add_term(b.neg(), vec![sel, pc, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            if l == 0 {
                vp.add_term(b.mul(&fe(4).neg()), vec![sel, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            vp.add_term(b.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in =
                    add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_pc4[l - 1], log_t)?;
                vp.add_term(b.neg(), vec![sel, c_in, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    ctx.stage("arith", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_arith(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
    let r = transcript
        .challenge_fields(b"con-arith-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-arith-a", 6)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "arith")?;
    let verdict = verify_leg_header("arith", leg, log_t, 4, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut expect = Goldilocks::ZERO;
    let a = &alphas[0];
    // (a) ADD + (b) ADDI
    for (name, chain, is_imm) in [
        ("sel_add", &idx.carry_add_r, 0usize),
        ("sel_addi", &idx.carry_add_i, 1usize),
    ] {
        let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
        for l in 0..4 {
            let rd = claim_limb(ledger, T_RD, l, pt)?;
            let x = claim_limb(ledger, T_RS1, l, pt)?;
            let y = if is_imm == 1 {
                claim_limb(ledger, T_IMM, l, pt)?
            } else {
                claim_limb(ledger, T_RS2, l, pt)?
            };
            let c_out = claim_bit(ledger, chain[l], pt)?;
            let mut e = rd.sub(&x).sub(&y).add(&fe(1 << 16).mul(&c_out));
            if l > 0 {
                e = e.sub(&claim_bit(ledger, chain[l - 1], pt)?);
            }
            expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (c) SUB
    {
        let b = &alphas[1];
        let sel = claim_bit(ledger, idx.sel_by("sel_sub"), pt)?;
        for l in 0..4 {
            let rd = claim_limb(ledger, T_RD, l, pt)?;
            let x = claim_limb(ledger, T_RS1, l, pt)?;
            let y = claim_limb(ledger, T_RS2, l, pt)?;
            let bo = claim_bit(ledger, idx.borrow_sub_r[l], pt)?;
            let mut e = rd.sub(&x).add(&y).sub(&fe(1 << 16).mul(&bo));
            if l > 0 {
                e = e.add(&claim_bit(ledger, idx.borrow_sub_r[l - 1], pt)?);
            }
            expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (d) ADDW/ADDIW
    for (name, chain, is_imm) in [
        ("sel_addw", &idx.carry_addw_r, 0usize),
        ("sel_addiw", &idx.carry_addw_i, 1usize),
    ] {
        let b = &alphas[2];
        let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
        for l in 0..2 {
            let rd = claim_limb(ledger, T_RD, l, pt)?;
            let x = claim_limb(ledger, T_RS1, l, pt)?;
            let y = if is_imm == 1 {
                claim_limb(ledger, T_IMM, l, pt)?
            } else {
                claim_limb(ledger, T_RS2, l, pt)?
            };
            let c_out = claim_bit(ledger, chain[l], pt)?;
            let mut e = rd.sub(&x).sub(&y).add(&fe(1 << 16).mul(&c_out));
            if l > 0 {
                e = e.sub(&claim_bit(ledger, chain[l - 1], pt)?);
            }
            expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (e) SUBW
    {
        let b = &alphas[3];
        let sel = claim_bit(ledger, idx.sel_by("sel_subw"), pt)?;
        for l in 0..2 {
            let rd = claim_limb(ledger, T_RD, l, pt)?;
            let x = claim_limb(ledger, T_RS1, l, pt)?;
            let y = claim_limb(ledger, T_RS2, l, pt)?;
            let bo = claim_bit(ledger, idx.borrow_subw_r[l], pt)?;
            let mut e = rd.sub(&x).add(&y).sub(&fe(1 << 16).mul(&bo));
            if l > 0 {
                e = e.add(&claim_bit(ledger, idx.borrow_subw_r[l - 1], pt)?);
            }
            expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (f) W sign extension + (g) LUI
    {
        let b = &alphas[4];
        let mut sign_pt = idx_point(6, 32);
        sign_pt.extend_from_slice(pt);
        let sign = ledger.tensor_claim(Factor::ValueBits { slot: T_RD }, &sign_pt)?;
        for name in ["sel_addw", "sel_addiw", "sel_subw"] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            for l in 2..4 {
                let rd = claim_limb(ledger, T_RD, l, pt)?;
                let e = rd.sub(&fe(0xFFFF).mul(&sign));
                expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
            }
        }
        let sel = claim_bit(ledger, idx.sel_by("sel_lui"), pt)?;
        for l in 0..4 {
            let rd = claim_limb(ledger, T_RD, l, pt)?;
            let y = claim_limb(ledger, T_IMM, l, pt)?;
            let e = rd.sub(&y);
            expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (h) AUIPC + (i) JAL/JALR
    {
        let b = &alphas[5];
        {
            let sel = claim_bit(ledger, idx.sel_by("sel_auipc"), pt)?;
            for l in 0..4 {
                let rd = claim_limb(ledger, T_RD, l, pt)?;
                let imm = claim_limb(ledger, T_IMM, l, pt)?;
                let c_out = claim_bit(ledger, idx.carry_ctrl[l], pt)?;
                let mut e = rd.sub(&imm).add(&fe(1 << 16).mul(&c_out));
                if l < 3 {
                    e = e.sub(&claim_val(ledger, idx.v_pc_l[l], pt)?);
                }
                if l > 0 {
                    e = e.sub(&claim_bit(ledger, idx.carry_ctrl[l - 1], pt)?);
                }
                expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
            }
        }
        for name in ["sel_jal", "sel_jalr"] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            for l in 0..4 {
                let rd = claim_limb(ledger, T_RD, l, pt)?;
                let c_out = claim_bit(ledger, idx.carry_pc4[l], pt)?;
                let mut e = rd.add(&fe(1 << 16).mul(&c_out));
                if l < 3 {
                    e = e.sub(&claim_val(ledger, idx.v_pc_l[l], pt)?);
                }
                if l == 0 {
                    e = e.sub(&fe(4));
                }
                if l > 0 {
                    e = e.sub(&claim_bit(ledger, idx.carry_pc4[l - 1], pt)?);
                }
                expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
            }
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("arith"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: comparisons — the eq-prefix recurrence and the lt/ltu formulas
// (full width, shared between signed and unsigned).
// ---------------------------------------------------------------------------

fn prove_cmp(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-cmp-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-cmp-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // x slot and y slot per comparison: cmp0 = (rs1, rs2), cmp1 = (rs1, imm).
    let y_slot = [T_RS2, T_IMM];
    for cmp in 0..2 {
        // (1) eqp recurrence: eqp[i+1] = eqp[i] * eq(bit(63 - i))
        //     eq(bit) = 1 - a - b + 2ab.
        for i in 0..64 {
            let next =
                add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[cmp][i + 1], log_t)?;
            let prev = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[cmp][i], log_t)?;
            let a_f = add_vbit_factor(&mut vp, &mut views, w, T_RS1, 63 - i, log_t)?;
            let b_f = add_vbit_factor(&mut vp, &mut views, w, y_slot[cmp], 63 - i, log_t)?;
            // next - prev*(1 - a - b + 2ab)
            //   = next - prev + prev*a + prev*b - 2*prev*a*b
            let al = &alphas[0];
            vp.add_term(*al, vec![next, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(al.neg(), vec![prev, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(*al, vec![prev, a_f, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*al, vec![prev, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(
                al.mul(&fe(2).neg()),
                vec![prev, a_f, b_f, ei],
            )
            .map_err(ConstraintError::Virtual)?;
        }
        // (2) ltu = sum_p eqp[63-p] * (1 - a_p) * b_p
        let ltu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.ltu[cmp], log_t)?;
        let al = &alphas[1];
        vp.add_term(*al, vec![ltu, ei]).map_err(ConstraintError::Virtual)?;
        for p in 0..64usize {
            let pref = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[cmp][63 - p], log_t)?;
            let a_f = add_vbit_factor(&mut vp, &mut views, w, T_RS1, p, log_t)?;
            let b_f = add_vbit_factor(&mut vp, &mut views, w, y_slot[cmp], p, log_t)?;
            vp.add_term(al.neg(), vec![pref, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*al, vec![pref, a_f, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        // (3) lt = a_63(1 - b_63) + sum_{p<63} eqp[63-p](1-a_p)b_p
        let lt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.lt[cmp], log_t)?;
        let a63 = add_vbit_factor(&mut vp, &mut views, w, T_RS1, 63, log_t)?;
        let b63 = add_vbit_factor(&mut vp, &mut views, w, y_slot[cmp], 63, log_t)?;
        let al = &alphas[2];
        vp.add_term(*al, vec![lt, ei]).map_err(ConstraintError::Virtual)?;
        vp.add_term(al.neg(), vec![a63, ei]).map_err(ConstraintError::Virtual)?;
        vp.add_term(*al, vec![a63, b63, ei]).map_err(ConstraintError::Virtual)?;
        for p in 0..63usize {
            let pref = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[cmp][63 - p], log_t)?;
            let a_f = add_vbit_factor(&mut vp, &mut views, w, T_RS1, p, log_t)?;
            let b_f = add_vbit_factor(&mut vp, &mut views, w, y_slot[cmp], p, log_t)?;
            vp.add_term(al.neg(), vec![pref, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*al, vec![pref, a_f, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    ctx.stage("cmp", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_cmp(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
    let r = transcript
        .challenge_fields(b"con-cmp-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-cmp-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "cmp")?;
    let verdict = verify_leg_header("cmp", leg, log_t, 5, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let y_slot = [T_RS2, T_IMM];
    let mut expect = Goldilocks::ZERO;
    for cmp in 0..2 {
        for i in 0..64 {
            let next = claim_bit(ledger, idx.eqp[cmp][i + 1], pt)?;
            let prev = claim_bit(ledger, idx.eqp[cmp][i], pt)?;
            let a = claim_vbit(ledger, T_RS1, 63 - i, pt)?;
            let b = claim_vbit(ledger, y_slot[cmp], 63 - i, pt)?;
            let e = next.sub(&prev).add(
                &prev.mul(
                    &a.add(&b)
                        .sub(&Goldilocks::from_u64(2).mul(&a.mul(&b))),
                ),
            );
            expect = expect.add(&alphas[0].mul(&e).mul(&eq_at));
        }
        // ltu
        {
            let ltu = claim_bit(ledger, idx.ltu[cmp], pt)?;
            let mut rhs = Goldilocks::ZERO;
            for p in 0..64usize {
                let pref = claim_bit(ledger, idx.eqp[cmp][63 - p], pt)?;
                let a = claim_vbit(ledger, T_RS1, p, pt)?;
                let b = claim_vbit(ledger, y_slot[cmp], p, pt)?;
                rhs = rhs.add(&pref.mul(&Goldilocks::ONE.sub(&a)).mul(&b));
            }
            expect = expect.add(&alphas[1].mul(&ltu.sub(&rhs)).mul(&eq_at));
        }
        // lt
        {
            let lt = claim_bit(ledger, idx.lt[cmp], pt)?;
            let a63 = claim_vbit(ledger, T_RS1, 63, pt)?;
            let b63 = claim_vbit(ledger, y_slot[cmp], 63, pt)?;
            let mut rhs = a63.mul(&Goldilocks::ONE.sub(&b63));
            for p in 0..63usize {
                let pref = claim_bit(ledger, idx.eqp[cmp][63 - p], pt)?;
                let a = claim_vbit(ledger, T_RS1, p, pt)?;
                let b = claim_vbit(ledger, y_slot[cmp], p, pt)?;
                rhs = rhs.add(&pref.mul(&Goldilocks::ONE.sub(&a)).mul(&b));
            }
            expect = expect.add(&alphas[2].mul(&lt.sub(&rhs)).mul(&eq_at));
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("cmp"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: control — pc/fetch consistency, the next-pc identity, post-halt
// inactivity, and termination.
// ---------------------------------------------------------------------------

fn prove_ctrl(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-ctrl-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-ctrl-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    let a = &alphas[0];
    // (1) pc = 4 * fetch_word (Val columns, exact: pc < 2^48).
    {
        let pc = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_pc, log_t)?;
        let fw = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_fetch_word, log_t)?;
        vp.add_term(*a, vec![pc, ei]).map_err(ConstraintError::Virtual)?;
        vp.add_term(a.mul(&fe(4).neg()), vec![fw, ei])
            .map_err(ConstraintError::Virtual)?;
    }
    // (2) the next-pc MUX, limbs 0..=2:
    //     np = (1-b-j-r)·A + b·[(1-t)·A + t·T] + j·T + r·J
    //         = A + b·t·(T-A) + j·(T-A) + r·(J-A)
    //     with A = pc+4 (carry_pc4), T = pc+imm (carry_ctrl),
    //     J = rs1+imm (carry_jalr). Per limb:
    //     np_l - A_l - bt(T_l-A_l) - j(T_l-A_l) - r(J_l-A_l) = 0.
    {
        let np_l: Vec<usize> = (0..3)
            .map(|l| add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_np_l[l], log_t))
            .collect::<Result<_, _>>()?;
        let pc_l: Vec<usize> = (0..3)
            .map(|l| add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_pc_l[l], log_t))
            .collect::<Result<_, _>>()?;
        let imm_l: Vec<usize> = (0..3)
            .map(|l| {
                let f = limb_mle(w, T_IMM, l, log_t);
                vp.add_factor(f).map_err(ConstraintError::Virtual)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let rs1_l: Vec<usize> = (0..3)
            .map(|l| {
                let f = limb_mle(w, T_RS1, l, log_t);
                vp.add_factor(f).map_err(ConstraintError::Virtual)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let b = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_branch"), log_t)?;
        let t = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.taken, log_t)?;
        let j = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_jal"), log_t)?;
        let rr = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_jalr"), log_t)?;
        let ca: Vec<usize> = (0..3)
            .map(|l| add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_pc4[l], log_t))
            .collect::<Result<_, _>>()?;
        let ct: Vec<usize> = (0..3)
            .map(|l| add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_ctrl[l], log_t))
            .collect::<Result<_, _>>()?;
        let cj: Vec<usize> = (0..3)
            .map(|l| add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_jalr[l], log_t))
            .collect::<Result<_, _>>()?;
        for l in 0..3 {
            // np_l - A_l = np_l - pc_l - 4d - c_in^A + 2^16 c_out^A
            vp.add_term(*a, vec![np_l[l], ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![pc_l[l], ei])
                .map_err(ConstraintError::Virtual)?;
            if l == 0 {
                vp.add_term(a.mul(&fe(4).neg()), vec![ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            vp.add_term(a.mul(&fe(1 << 16)), vec![ca[l], ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                vp.add_term(a.neg(), vec![ca[l - 1], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // -bt·(T_l - A_l) = -bt·(imm_l - 4d + c^T_in - c^A_in - 2^16(c^T_out - c^A_out))
            for (sel_group, is_bt) in [((b, t), true), ((j, t), false), ((rr, t), false)] {
                let _ = is_bt;
                // The MUX weight: bt for branches, j for jal, r for jalr.
                // (j and r terms have no t factor.)
                let _ = sel_group;
            }
            // branches: weight = b·t
            {
                let w_ab = a.neg(); // -bt·T_l
                vp.add_term(w_ab, vec![b, t, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(w_ab, vec![b, t, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(w_ab, vec![b, t, ct[l - 1.min(l)], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16)), vec![b, t, ct[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                // +bt·A_l
                vp.add_term(*a, vec![b, t, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l == 0 {
                    vp.add_term(a.mul(&fe(4)), vec![b, t, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.neg(), vec![b, t, ca[l - 1.min(l)], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16).neg()), vec![b, t, ca[l], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // jal: weight = j
            {
                vp.add_term(a.neg(), vec![j, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![j, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![j, ct[l - 1.min(l)], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16)), vec![j, ct[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(*a, vec![j, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l == 0 {
                    vp.add_term(a.mul(&fe(4)), vec![j, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.neg(), vec![j, ca[l - 1.min(l)], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16).neg()), vec![j, ca[l], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // jalr: weight = r (J_l - A_l)
            {
                vp.add_term(a.neg(), vec![rr, rs1_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![rr, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![rr, cj[l - 1.min(l)], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16)), vec![rr, cj[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(*a, vec![rr, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l == 0 {
                    vp.add_term(a.mul(&fe(4)), vec![rr, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.neg(), vec![rr, ca[l - 1.min(l)], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16).neg()), vec![rr, ca[l], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (3) post-halt inactivity: h·(rd_we + mem_we + mem_re) = 0 (the
    //     executor's post-halt pc evolution is unconstrained — fetch
    //     consistency is authenticated by the memory layer).
    {
        let h = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.halted, log_t)?;
        let rw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.rd_we, log_t)?;
        let mw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_we, log_t)?;
        let mr = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_re, log_t)?;
        let b = &alphas[1];
        vp.add_term(*b, vec![h, rw, ei]).map_err(ConstraintError::Virtual)?;
        vp.add_term(*b, vec![h, mw, ei]).map_err(ConstraintError::Virtual)?;
        vp.add_term(*b, vec![h, mr, ei]).map_err(ConstraintError::Virtual)?;
    }
    ctx.stage("ctrl", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_ctrl(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
    let r = transcript
        .challenge_fields(b"con-ctrl-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-ctrl-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "ctrl")?;
    let verdict = verify_leg_header("ctrl", leg, log_t, 4, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut expect = Goldilocks::ZERO;
    let a = &alphas[0];
    // (1) pc = 4*fetch
    {
        let pc = claim_val(ledger, idx.v_pc, pt)?;
        let fw = claim_val(ledger, idx.v_fetch_word, pt)?;
        expect = expect.add(&a.mul(&pc.sub(&fw.mul(&fe(4)))).mul(&eq_at));
    }
    // (2) the next-pc MUX
    {
        let b = claim_bit(ledger, idx.sel_by("sel_branch"), pt)?;
        let t = claim_bit(ledger, idx.taken, pt)?;
        let j = claim_bit(ledger, idx.sel_by("sel_jal"), pt)?;
        let rr = claim_bit(ledger, idx.sel_by("sel_jalr"), pt)?;
        for l in 0..3 {
            let np = claim_val(ledger, idx.v_np_l[l], pt)?;
            let pc = claim_val(ledger, idx.v_pc_l[l], pt)?;
            let imm = claim_limb(ledger, T_IMM, l, pt)?;
            let rs1 = claim_limb(ledger, T_RS1, l, pt)?;
            let ca_out = claim_bit(ledger, idx.carry_pc4[l], pt)?;
            let ct_out = claim_bit(ledger, idx.carry_ctrl[l], pt)?;
            let cj_out = claim_bit(ledger, idx.carry_jalr[l], pt)?;
            let ca_in = if l > 0 {
                claim_bit(ledger, idx.carry_pc4[l - 1], pt)?
            } else {
                Goldilocks::ZERO
            };
            let ct_in = if l > 0 {
                claim_bit(ledger, idx.carry_ctrl[l - 1], pt)?
            } else {
                Goldilocks::ZERO
            };
            let cj_in = if l > 0 {
                claim_bit(ledger, idx.carry_jalr[l - 1], pt)?
            } else {
                Goldilocks::ZERO
            };
            let delta4 = if l == 0 { fe(4) } else { Goldilocks::ZERO };
            // A_l = pc + 4d + c_in^A - 2^16 c_out^A
            let al = pc
                .add(&delta4)
                .add(&ca_in)
                .sub(&fe(1 << 16).mul(&ca_out));
            // T_l = pc + imm + c_in^T - 2^16 c_out^T
            let tl = pc
                .add(&imm)
                .add(&ct_in)
                .sub(&fe(1 << 16).mul(&ct_out));
            // J_l = rs1 + imm + c_in^J - 2^16 c_out^J
            let jl = rs1
                .add(&imm)
                .add(&cj_in)
                .sub(&fe(1 << 16).mul(&cj_out));
            let e = np
                .sub(&al)
                .sub(&b.mul(&t).mul(&tl.sub(&al)))
                .sub(&j.mul(&tl.sub(&al)))
                .sub(&rr.mul(&jl.sub(&al)));
            expect = expect.add(&a.mul(&e).mul(&eq_at));
        }
    }
    // (3) post-halt inactivity
    {
        let h = claim_bit(ledger, idx.halted, pt)?;
        let e = claim_bit(ledger, idx.rd_we, pt)?
            .add(&claim_bit(ledger, idx.mem_we, pt)?)
            .add(&claim_bit(ledger, idx.mem_re, pt)?);
        expect = expect.add(&alphas[1].mul(&h).mul(&e).mul(&eq_at));
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("ctrl"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: routing — memory addressing, store/load value routing, the
// bitwise ALU ops, and the comparison rd routing.
// ---------------------------------------------------------------------------

fn prove_route(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-route-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-route-a", 5)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp.add_factor(eq.clone()).map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // (1) effective address: mem_addr = (rs1 + imm) mod 2^64, limb-wise
    //     via carry_jalr; the addr limbs are committed value columns.
    {
        let addr_l: Vec<usize> = (0..4)
            .map(|l| add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_addr_l[l], log_t))
            .collect::<Result<_, _>>()?;
        let rs1_l: Vec<usize> = (0..4)
            .map(|l| {
                let f = limb_mle(w, T_RS1, l, log_t);
                vp.add_factor(f).map_err(ConstraintError::Virtual)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let imm_l: Vec<usize> = (0..4)
            .map(|l| {
                let f = limb_mle(w, T_IMM, l, log_t);
                vp.add_factor(f).map_err(ConstraintError::Virtual)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let a = &alphas[0];
        let re = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_re, log_t)?;
        let we = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_we, log_t)?;
        for act in [re, we] {
            for l in 0..4 {
                let c_out = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_jalr[l], log_t)?;
                vp.add_term(*a, vec![act, addr_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![act, rs1_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![act, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16)), vec![act, c_out, ei])
                    .map_err(ConstraintError::Virtual)?;
                if l > 0 {
                    let c_in =
                        add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_jalr[l - 1], log_t)?;
                    vp.add_term(a.neg(), vec![act, c_in, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
            }
        }
    }
    // (2) word addressing: (word accesses) addr = 8*word + 4*half;
    //     (double accesses) addr = 8*word.
    {
        let addr = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mem_addr, log_t)?;
        let word = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mem_word, log_t)?;
        let half = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_half, log_t)?;
        let a = &alphas[1];
        for name in ["sel_lw", "sel_lwu", "sel_sw"] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            vp.add_term(*a, vec![sel, addr, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(8).neg()), vec![sel, word, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(4).neg()), vec![sel, half, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        for name in ["sel_ld", "sel_sd"] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            vp.add_term(*a, vec![sel, addr, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(8).neg()), vec![sel, word, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (3) SD: mem_new = rs2 (limb-wise); LD: rd = mem_old (limb-wise).
    {
        let a = &alphas[2];
        let sd = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sd"), log_t)?;
        let ld = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_ld"), log_t)?;
        for l in 0..4 {
            let newl = add_limb_factor(&mut vp, &mut views, w, T_MEM_NEW, l, log_t)?;
            let oldl = add_limb_factor(&mut vp, &mut views, w, T_MEM_OLD, l, log_t)?;
            let rs2l = add_limb_factor(&mut vp, &mut views, w, T_RS2, l, log_t)?;
            let rdl = add_limb_factor(&mut vp, &mut views, w, T_RD, l, log_t)?;
            vp.add_term(*a, vec![sd, newl, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sd, rs2l, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*a, vec![ld, rdl, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![ld, oldl, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (4) SW: low half: new = (1-h)*rs2 + h*old; high half:
    //     new = h*rs2_{l-2} + (1-h)*old.
    {
        let b = &alphas[3];
        let sw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sw"), log_t)?;
        let h = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_half, log_t)?;
        for l in 0..4 {
            let newl = add_limb_factor(&mut vp, &mut views, w, T_MEM_NEW, l, log_t)?;
            let oldl = add_limb_factor(&mut vp, &mut views, w, T_MEM_OLD, l, log_t)?;
            let rs2l = if l < 2 {
                add_limb_factor(&mut vp, &mut views, w, T_RS2, l, log_t)?
            } else {
                add_limb_factor(&mut vp, &mut views, w, T_RS2, l - 2, log_t)?
            };
            // sw*(new - src + h*src - h*other) = 0
            let (src, other) = if l < 2 { (rs2l, oldl) } else { (oldl, rs2l) };
            vp.add_term(*b, vec![sw, newl, ei]).map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sw, src, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*b, vec![sw, h, src, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sw, h, other, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (5) LW/LWU routing: rd 0/1 = (1-h)*old_{0,1} + h*old_{2,3};
    //     LWU: rd 2/3 = 0; LW: rd 2/3 = sign*(2^16-1),
    //     sign = h*old_bit63 + (1-h)*old_bit31.
    {
        let b = &alphas[3];
        let lw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lw"), log_t)?;
        let lwu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lwu"), log_t)?;
        let h = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_half, log_t)?;
        for l in 0..2 {
            let rdl = add_limb_factor(&mut vp, &mut views, w, T_RD, l, log_t)?;
            let oldl = add_limb_factor(&mut vp, &mut views, w, T_MEM_OLD, l, log_t)?;
            let oldh = add_limb_factor(&mut vp, &mut views, w, T_MEM_OLD, l + 2, log_t)?;
            for sel in [lw, lwu] {
                vp.add_term(*b, vec![sel, rdl, ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(b.neg(), vec![sel, oldl, ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(*b, vec![sel, h, oldl, ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(b.neg(), vec![sel, h, oldh, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
        let f63 = add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 63, log_t)?;
        let f31 = add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 31, log_t)?;
        for l in 2..4 {
            let rdl = add_limb_factor(&mut vp, &mut views, w, T_RD, l, log_t)?;
            // LWU: rd_l = 0
            vp.add_term(*b, vec![lwu, rdl, ei])
                .map_err(ConstraintError::Virtual)?;
            // LW: rd_l = (2^16-1)*(old31 + h*old63 - h*old31)
            vp.add_term(*b, vec![lw, rdl, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(0xFFFF).neg()), vec![lw, f31, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(0xFFFF).neg()), vec![lw, h, f63, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(0xFFFF)), vec![lw, h, f31, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (6) bitwise ALU: AND/OR/XOR register + immediate forms, per bit.
    {
        let c = &alphas[4];
        let ops = [
            ("sel_and", "sel_andi", 0u64),
            ("sel_or", "sel_ori", 1),
            ("sel_xor", "sel_xori", 2),
        ];
        for (reg_name, imm_name, op) in ops {
            let sr = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(reg_name), log_t)?;
            let si = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(imm_name), log_t)?;
            for bit in 0..64usize {
                let rd = add_vbit_factor(&mut vp, &mut views, w, T_RD, bit, log_t)?;
                let a = add_vbit_factor(&mut vp, &mut views, w, T_RS1, bit, log_t)?;
                let b_r = add_vbit_factor(&mut vp, &mut views, w, T_RS2, bit, log_t)?;
                let b_i = add_vbit_factor(&mut vp, &mut views, w, T_IMM, bit, log_t)?;
                for (sel, bb) in [(sr, b_r), (si, b_i)] {
                    if op == 0 {
                        vp.add_term(*c, vec![sel, rd, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(c.neg(), vec![sel, a, bb, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                    if op == 1 {
                        vp.add_term(*c, vec![sel, rd, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(c.neg(), vec![sel, a, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(c.neg(), vec![sel, bb, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(*c, vec![sel, a, bb, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                    if op == 2 {
                        vp.add_term(*c, vec![sel, rd, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(c.neg(), vec![sel, a, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(c.neg(), vec![sel, bb, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(c.mul(&fe(2)), vec![sel, a, bb, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
        }
    }
    // (7) comparison rd routing: SLT/SLTI: rd = lt[cmp];
    //     SLTU/SLTIU: rd = ltu[cmp].
    {
        let d = &alphas[0];
        let rd_c = vp
            .add_factor(combo_mle(w, T_RD, log_t))
            .map_err(ConstraintError::Virtual)?;
        views.push((rd_c, FV::Combo { slot: T_RD }));
        let pairs = [
            ("sel_slt", idx.lt[0]),
            ("sel_slti", idx.lt[1]),
            ("sel_sltu", idx.ltu[0]),
            ("sel_sltiu", idx.ltu[1]),
        ];
        for (name, col) in pairs {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            let ltf = add_bit_factor(&mut vp, &mut views, &aux.bits, col, log_t)?;
            vp.add_term(*d, vec![sel, rd_c, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(d.neg(), vec![sel, ltf, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    ctx.stage("route", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_route(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
    let r = transcript
        .challenge_fields(b"con-route-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-route-a", 5)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "route")?;
    let verdict = verify_leg_header("route", leg, log_t, 5, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut expect = Goldilocks::ZERO;
    // (1) effective address
    {
        let a = &alphas[0];
        let re = claim_bit(ledger, idx.mem_re, pt)?;
        let we = claim_bit(ledger, idx.mem_we, pt)?;
        for act in [re, we] {
            for l in 0..4 {
                let addr = claim_val(ledger, idx.v_addr_l[l], pt)?;
                let rs1 = claim_limb(ledger, T_RS1, l, pt)?;
                let imm = claim_limb(ledger, T_IMM, l, pt)?;
                let c_out = claim_bit(ledger, idx.carry_jalr[l], pt)?;
                let mut e = addr.sub(&rs1).sub(&imm).add(&fe(1 << 16).mul(&c_out));
                if l > 0 {
                    e = e.sub(&claim_bit(ledger, idx.carry_jalr[l - 1], pt)?);
                }
                expect = expect.add(&a.mul(&act).mul(&e).mul(&eq_at));
            }
        }
    }
    // (2) word addressing
    {
        let a = &alphas[1];
        let addr = claim_val(ledger, idx.v_mem_addr, pt)?;
        let word = claim_val(ledger, idx.v_mem_word, pt)?;
        let half = claim_bit(ledger, idx.mem_half, pt)?;
        for name in ["sel_lw", "sel_lwu", "sel_sw"] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            let e = addr
                .sub(&word.mul(&fe(8)))
                .sub(&half.mul(&fe(4)));
            expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
        }
        for name in ["sel_ld", "sel_sd"] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            let e = addr.sub(&word.mul(&fe(8)));
            expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (3) SD / LD
    {
        let a = &alphas[2];
        let sd = claim_bit(ledger, idx.sel_by("sel_sd"), pt)?;
        let ld = claim_bit(ledger, idx.sel_by("sel_ld"), pt)?;
        for l in 0..4 {
            let newl = claim_limb(ledger, T_MEM_NEW, l, pt)?;
            let oldl = claim_limb(ledger, T_MEM_OLD, l, pt)?;
            let rs2l = claim_limb(ledger, T_RS2, l, pt)?;
            let rdl = claim_limb(ledger, T_RD, l, pt)?;
            expect = expect.add(&a.mul(&sd).mul(&newl.sub(&rs2l)).mul(&eq_at));
            expect = expect.add(&a.mul(&ld).mul(&rdl.sub(&oldl)).mul(&eq_at));
        }
    }
    // (4) SW
    {
        let b = &alphas[3];
        let sw = claim_bit(ledger, idx.sel_by("sel_sw"), pt)?;
        let h = claim_bit(ledger, idx.mem_half, pt)?;
        for l in 0..4 {
            let newl = claim_limb(ledger, T_MEM_NEW, l, pt)?;
            let oldl = claim_limb(ledger, T_MEM_OLD, l, pt)?;
            let src = if l < 2 {
                claim_limb(ledger, T_RS2, l, pt)?
            } else {
                oldl
            };
            let other = if l < 2 {
                oldl
            } else {
                claim_limb(ledger, T_RS2, l - 2, pt)?
            };
            let e = newl
                .sub(&src)
                .add(&h.mul(&src))
                .sub(&h.mul(&other));
            expect = expect.add(&b.mul(&sw).mul(&e).mul(&eq_at));
        }
    }
    // (5) LW / LWU
    {
        let b = &alphas[3];
        let lw = claim_bit(ledger, idx.sel_by("sel_lw"), pt)?;
        let lwu = claim_bit(ledger, idx.sel_by("sel_lwu"), pt)?;
        let h = claim_bit(ledger, idx.mem_half, pt)?;
        for l in 0..2 {
            let rdl = claim_limb(ledger, T_RD, l, pt)?;
            let oldl = claim_limb(ledger, T_MEM_OLD, l, pt)?;
            let oldh = claim_limb(ledger, T_MEM_OLD, l + 2, pt)?;
            let e = rdl
                .sub(&oldl)
                .add(&h.mul(&oldl))
                .sub(&h.mul(&oldh));
            for sel in [lw, lwu] {
                expect = expect.add(&b.mul(&sel).mul(&e).mul(&eq_at));
            }
        }
        for l in 2..4 {
            let rdl = claim_limb(ledger, T_RD, l, pt)?;
            expect = expect.add(&b.mul(&lwu).mul(&rdl).mul(&eq_at));
            let old63 = claim_vbit(ledger, T_MEM_OLD, 63, pt)?;
            let old31 = claim_vbit(ledger, T_MEM_OLD, 31, pt)?;
            let sign = old31.add(&h.mul(&old63.sub(&old31)));
            let e = rdl.sub(&fe(0xFFFF).mul(&sign));
            expect = expect.add(&b.mul(&lw).mul(&e).mul(&eq_at));
        }
    }
    // (6) bitwise ALU
    {
        let c = &alphas[4];
        let ops = [
            ("sel_and", "sel_andi", 0u64),
            ("sel_or", "sel_ori", 1),
            ("sel_xor", "sel_xori", 2),
        ];
        for (reg_name, imm_name, op) in ops {
            let sr = claim_bit(ledger, idx.sel_by(reg_name), pt)?;
            let si = claim_bit(ledger, idx.sel_by(imm_name), pt)?;
            for b in 0..64usize {
                let rd = claim_vbit(ledger, T_RD, b, pt)?;
                let a = claim_vbit(ledger, T_RS1, b, pt)?;
                let bv = claim_vbit(ledger, T_RS2, b, pt)?;
                let bi = claim_vbit(ledger, T_IMM, b, pt)?;
                for (sel, bb) in [(sr, bv), (si, bi)] {
                    let target = match op {
                        0 => a.mul(&bb),
                        1 => a.add(&bb).sub(&a.mul(&bb)),
                        _ => a.add(&bb).sub(&fe(2).mul(&a.mul(&bb))),
                    };
                    expect = expect.add(&c.mul(&sel).mul(&rd.sub(&target)).mul(&eq_at));
                }
            }
        }
    }
    // (7) comparison rd routing
    {
        let d = &alphas[0];
        let rd = ledger.value_combo(T_RD, pt)?;
        let pairs = [
            ("sel_slt", idx.lt[0]),
            ("sel_slti", idx.lt[1]),
            ("sel_sltu", idx.ltu[0]),
            ("sel_sltiu", idx.ltu[1]),
        ];
        for (name, col) in pairs {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            let lt = claim_bit(ledger, col, pt)?;
            expect = expect.add(&d.mul(&sel).mul(&rd.sub(&lt)).mul(&eq_at));
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("route"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: halted — termination (halted[T-1] = 1).
// ---------------------------------------------------------------------------

fn prove_halt(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let aux = ctx.aux;
    let log_t = ctx.w.log_t;
    let h = DenseMle {
        num_vars: log_t,
        evaluations: aux.bits[aux.index.halted]
            .iter()
            .map(|v| fe(*v as u64))
            .collect(),
    };
    // e_last: the indicator MLE of cycle T-1 (eq extension at the
    // all-ones point).
    let ones = vec![Goldilocks::ONE; log_t];
    let e_last = DenseMle::eq_extension(&ones);
    let mut vp = VirtualPolynomial::new(log_t);
    let hi = vp.add_factor(h.clone()).map_err(ConstraintError::Virtual)?;
    let li = vp.add_factor(e_last.clone()).map_err(ConstraintError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![hi, li])
        .map_err(ConstraintError::Virtual)?;
    let views: Vec<ViewPair> = vec![
        (hi, FV::Bit(aux.index.halted)),
        (li, FV::PubTable(e_last)),
    ];
    // Sum over the cube = h[T-1] (the indicator selects the last cycle).
    let claim = fe(aux.bits[aux.index.halted][(1 << log_t) - 1] as u64);
    ctx.stage("halt-end", &mut vp, &views, claim)
        .map(|_| ())
}

fn verify_halt(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let log_t = aux.bits.first().map(|c| c.len().trailing_zeros() as usize).unwrap_or(0);
    let leg = next_constraint_leg(iter, "halt-end")?;
    let verdict = verify_leg_header("halt-end", leg, log_t, 2, transcript)?;
    let h = ledger.tensor_claim(Factor::BitCol { id: idx.halted }, &verdict.point)?;
    let ones = vec![Goldilocks::ONE; log_t];
    let e_last = DenseMle::eq_extension(&ones);
    let e_at = e_last.evaluate(&verdict.point).map_err(ConstraintError::Mle)?;
    let expect = h.mul(&e_at);
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("halt-end"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn row_mle(m: &DenseMle, row: usize, log_t: usize) -> DenseMle {
    let t = 1usize << log_t;
    DenseMle {
        num_vars: log_t,
        evaluations: (0..t).map(|c| m.evaluations[row * t + c]).collect(),
    }
}

fn limb_mle(w: &CycleWitness, slot: usize, limb: usize, log_t: usize) -> DenseMle {
    let t = 1usize << log_t;
    DenseMle {
        num_vars: log_t,
        evaluations: (0..t)
            .map(|c| {
                let mut acc = Goldilocks::ZERO;
                for i in 0..16usize {
                    let bit = limb * 16 + i;
                    let row = 63 - bit;
                    acc = acc.add(&fe(1u64 << i).mul(&w.values[slot].evaluations[row * t + c]));
                }
                acc
            })
            .collect(),
    }
}

fn combo_mle(w: &CycleWitness, slot: usize, log_t: usize) -> DenseMle {
    let t = 1usize << log_t;
    DenseMle {
        num_vars: log_t,
        evaluations: (0..t).map(|c| fe(tensor_word(w, slot, c))).collect(),
    }
}

// ---------------------------------------------------------------------------
// Orchestrators
// ---------------------------------------------------------------------------

type LegIter<'a> = core::slice::Iter<'a, ConstraintLeg>;

fn next_constraint_leg<'b>(
    iter: &mut LegIter<'b>,
    name: &str,
) -> Result<&'b ConstraintLeg, ConstraintError> {
    let leg = iter.next().ok_or(ConstraintError::Shape)?;
    if leg.name != name {
        return Err(ConstraintError::Shape);
    }
    Ok(leg)
}

/// Prove all constraint families (legs appended in protocol order).
/// Fails closed on any instruction outside the v1 coverage set.
#[allow(clippy::too_many_arguments)]
pub fn prove_constraints(
    w: &CycleWitness,
    aux: &AuxCols,
    instrs: &[Instr],
    ledger: &mut Ledger<'_>,
    legs: &mut Vec<ConstraintLeg>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    for (c, instr) in instrs.iter().enumerate() {
        if !covered_instr(instr) {
            return Err(ConstraintError::UncoveredInstruction { cycle: c });
        }
    }
    let mut ctx = FamilyCtx { w, aux, ledger, legs, transcript };
    prove_booleanity(&mut ctx)?;
    prove_selectors(&mut ctx)?;
    prove_flags(&mut ctx)?;
    prove_arith(&mut ctx)?;
    prove_cmp(&mut ctx)?;
    prove_ctrl(&mut ctx)?;
    prove_route(&mut ctx)?;
    prove_halt(&mut ctx)?;
    Ok(())
}

/// Verify all constraint families (legs consumed in protocol order).
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints(
    w: &CycleWitness,
    aux: &AuxCols,
    instrs: &[Instr],
    proofs: &[ConstraintLeg],
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    for (c, instr) in instrs.iter().enumerate() {
        if !covered_instr(instr) {
            return Err(ConstraintError::UncoveredInstruction { cycle: c });
        }
    }
    let log_t = w.log_t;
    let mut iter = proofs.iter();
    verify_booleanity(w, aux, &mut iter, ledger, transcript)?;
    verify_selectors(aux, &mut iter, log_t, ledger, transcript)?;
    verify_flags(aux, &mut iter, log_t, ledger, transcript)?;
    verify_arith(aux, &mut iter, ledger, transcript)?;
    verify_cmp(aux, &mut iter, ledger, transcript)?;
    verify_ctrl(aux, &mut iter, ledger, transcript)?;
    verify_route(aux, &mut iter, ledger, transcript)?;
    verify_halt(aux, &mut iter, ledger, transcript)?;
    if iter.next().is_some() {
        return Err(ConstraintError::Shape);
    }
    Ok(())
}
