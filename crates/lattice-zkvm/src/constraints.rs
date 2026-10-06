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
use crate::ledger::{idx_point, Factor, Ledger, LedgerError};
use lattice_vm::Instr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    Ledger(LedgerError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Transcript(lattice_core::transcript::TranscriptError),
    Mle(lattice_core::mle::MleError),
    FinalCheck(&'static str),
    Shape,
    /// A sparse-engine (constraint-family) prover failure.
    Sparse(String),
    /// An instruction outside the v1 constraint coverage set appeared —
    /// fail closed rather than prove an unconstrained class.
    UncoveredInstruction {
        cycle: usize,
    },
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

/// One selector-table row: (name, opcode, funct3, funct7?, funct6?).
pub type SelectorRow = (&'static str, u32, u32, Option<u32>, Option<u32>);

pub const SELECTOR_TABLE: &[SelectorRow] = &[
    // class selectors
    ("sel_opimm", 0x13, 99, None, None),
    ("sel_op", 0x33, 99, None, None),
    ("sel_op32", 0x3b, 99, None, None),
    ("sel_opimm32", 0x1b, 99, None, None),
    ("sel_lui", 0x37, 99, None, None),
    ("sel_auipc", 0x17, 99, None, None),
    ("sel_jal", 0x6f, 99, None, None),
    ("sel_jalr", 0x67, 99, None, None),
    ("sel_branch", 0x63, 99, None, None),
    ("sel_load", 0x03, 99, None, None),
    ("sel_store", 0x23, 99, None, None),
    ("sel_system", 0x73, 99, None, None),
    // sub-class selectors (register)
    ("sel_add", 0x33, 0, Some(0x00), None),
    ("sel_sub", 0x33, 0, Some(0x20), None),
    ("sel_xor", 0x33, 4, Some(0x00), None),
    ("sel_or", 0x33, 6, Some(0x00), None),
    ("sel_and", 0x33, 7, Some(0x00), None),
    ("sel_sll", 0x33, 1, Some(0x00), None),
    ("sel_srl", 0x33, 5, Some(0x00), None),
    ("sel_sra", 0x33, 5, Some(0x20), None),
    ("sel_slt", 0x33, 2, Some(0x00), None),
    ("sel_sltu", 0x33, 3, Some(0x00), None),
    ("sel_mul", 0x33, 0, Some(0x01), None),
    ("sel_mulh", 0x33, 1, Some(0x01), None),
    ("sel_mulhu", 0x33, 3, Some(0x01), None),
    ("sel_div", 0x33, 4, Some(0x01), None),
    ("sel_divu", 0x33, 5, Some(0x01), None),
    ("sel_rem", 0x33, 6, Some(0x01), None),
    ("sel_remu", 0x33, 7, Some(0x01), None),
    // sub-class selectors (immediate) — the 64-bit shift immediates
    // carry a 6-bit shamt, so their discriminant is funct6 (bits
    // 26..31), NOT funct7: bit 25 is shamt data.
    ("sel_addi", 0x13, 0, None, None),
    ("sel_xori", 0x13, 4, None, None),
    ("sel_ori", 0x13, 6, None, None),
    ("sel_andi", 0x13, 7, None, None),
    ("sel_slti", 0x13, 2, None, None),
    ("sel_sltiu", 0x13, 3, None, None),
    ("sel_slli", 0x13, 1, None, Some(0x00)),
    ("sel_srli", 0x13, 5, None, Some(0x00)),
    ("sel_srai", 0x13, 5, None, Some(0x10)),
    ("sel_srxi", 0x13, 5, None, None),
    // sub-class selectors (W)
    ("sel_addiw", 0x1b, 0, None, None),
    ("sel_slliw", 0x1b, 1, Some(0x00), None),
    ("sel_srxiw", 0x1b, 5, None, None),
    ("sel_srliw", 0x1b, 5, Some(0x00), None),
    ("sel_sraiw", 0x1b, 5, Some(0x20), None),
    ("sel_addw", 0x3b, 0, Some(0x00), None),
    ("sel_subw", 0x3b, 0, Some(0x20), None),
    ("sel_sllw", 0x3b, 1, Some(0x00), None),
    ("sel_srlw", 0x3b, 5, Some(0x00), None),
    ("sel_sraw", 0x3b, 5, Some(0x20), None),
    ("sel_mulw", 0x3b, 0, Some(0x01), None),
    ("sel_divw", 0x3b, 4, Some(0x01), None),
    ("sel_divuw", 0x3b, 5, Some(0x01), None),
    ("sel_remw", 0x3b, 6, Some(0x01), None),
    ("sel_remuw", 0x3b, 7, Some(0x01), None),
    // branches
    ("sel_beq", 0x63, 0, None, None),
    ("sel_bne", 0x63, 1, None, None),
    ("sel_blt", 0x63, 4, None, None),
    ("sel_bge", 0x63, 5, None, None),
    ("sel_bltu", 0x63, 6, None, None),
    ("sel_bgeu", 0x63, 7, None, None),
    // loads / stores
    ("sel_lw", 0x03, 2, None, None),
    ("sel_lwu", 0x03, 6, None, None),
    ("sel_ld", 0x03, 3, None, None),
    // the byte-guest ISA: the sub-word surface (P1)
    ("sel_lb", 0x03, 0, None, None),
    ("sel_lh", 0x03, 1, None, None),
    ("sel_lbu", 0x03, 4, None, None),
    ("sel_lhu", 0x03, 5, None, None),
    ("sel_sb", 0x23, 0, None, None),
    ("sel_sh", 0x23, 1, None, None),
    ("sel_sw", 0x23, 2, None, None),
    ("sel_sd", 0x23, 3, None, None),
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
    /// The address's low three bits (the byte-granular sub-word
    /// offsets — the P1 layer's position selectors).
    pub mem_off: [usize; 3],
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
    /// The fetched instruction word column (the instr-tensor binding).
    pub v_instr: usize,
    /// pc limbs 0..=2 (pc < 2^48).
    pub v_pc_l: [usize; 3],
    /// next_pc limbs 0..=2.
    pub v_np_l: [usize; 3],
    /// mem_addr limbs 0..=3 (the effective-address chain result).
    pub v_addr_l: [usize; 4],
    // --- the shift family (the AUX_ONEHOT namespace) ---
    /// Register-shift one-hot over the 6-bit shamt (rs2 & 63).
    pub shoh6_r: [usize; 64],
    /// Immediate-shift one-hot over the 6-bit shamt (instr bits 20..25).
    pub shoh6_i: [usize; 64],
    /// Register W-shift one-hot over the 5-bit shamt (rs2 & 31).
    pub shoh5_r: [usize; 32],
    /// Immediate W-shift one-hot over the 5-bit shamt (bits 20..24).
    pub shoh5_i: [usize; 32],
    // --- the mul family (limbs linked from boolean bit columns) ---
    /// The low 64 bits of the unsigned product (limb value columns).
    pub v_mul_lo: [usize; 4],
    /// The high 64 bits of the unsigned product.
    pub v_mul_hi: [usize; 4],
    /// The multiplication carries c_1..=c_7 (17-bit value columns).
    pub v_mul_c: [usize; 7],
    /// MULH's rd-composition borrows (2-bit value columns).
    pub v_mulh_bor: [usize; 4],
    // --- the div family ---
    /// The quotient limbs (rd-facing for DIV/DIVU forms).
    pub v_div_q: [usize; 4],
    /// The remainder limbs (rd-facing for REM/REMU forms).
    pub v_div_r: [usize; 4],
    /// The division recurrence carries d_1..=d_4 (17-bit; d_4 = 0).
    pub v_div_d: [usize; 4],
    /// |a| limbs (the signed-division magnitudes; raw values for the
    /// unsigned classes — the sign source is class-dependent).
    pub v_mag_a: [usize; 4],
    /// |b| limbs.
    pub v_mag_b: [usize; 4],
    /// |q| limbs.
    pub v_mag_q: [usize; 4],
    /// |r| limbs.
    pub v_mag_r: [usize; 4],
    /// The magnitude recurrence carries (17-bit value columns).
    pub v_mag_d: [usize; 4],
    /// The (mag_r - mag_b) borrow chain: bor_1..=bor_4, with bor_4 the
    /// [mag_r < mag_b] indicator (boolean bit columns).
    pub v_rbor: [usize; 4],
    /// The out limbs of (mag_r - mag_b) mod 2^64 (range-linked).
    pub v_rlt_out: [usize; 4],
    /// The eq-to-zero prefix over rs2's 64 bits (bz = eqz[64]).
    pub eqz: [usize; 65],
    /// The eq-to-zero prefix over rs2's low 32 bits (bzw = eqzw[32]).
    pub eqzw: [usize; 33],
    /// The boolean bit decompositions of every mul/div limb and carry
    /// column (flat, group-major): the range enforcement that makes the
    /// limb-level recurrences carry INTEGER semantics.
    ///
    /// Layout: [mul_lo(4*16), mul_hi(4*16), mul_c(7*17), mulh_bor(4*2),
    /// div_q(4*16), div_r(4*16), div_d(4*17), mag_a(4*16), mag_b(4*16),
    /// mag_q(4*16), mag_r(4*16), mag_d(4*17)].
    pub range_bits: Vec<usize>,
}

impl AuxIndex {
    pub fn sel_by(&self, name: &str) -> usize {
        for (i, (n, _, _, _, _)) in SELECTOR_TABLE.iter().enumerate() {
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
            | Slli { .. }
            | Srli { .. }
            | Srai { .. }
            | Addiw { .. }
            | Slliw { .. }
            | Srliw { .. }
            | Sraiw { .. }
            | Add { .. }
            | Sub { .. }
            | Sll { .. }
            | Srl { .. }
            | Sra { .. }
            | Slt { .. }
            | Sltu { .. }
            | Xor { .. }
            | Or { .. }
            | And { .. }
            | Addw { .. }
            | Subw { .. }
            | Sllw { .. }
            | Srlw { .. }
            | Sraw { .. }
            | Mul { .. }
            | Mulh { .. }
            | Mulhu { .. }
            | Mulw { .. }
            | Div { .. }
            | Divu { .. }
            | Rem { .. }
            | Remu { .. }
            | Divw { .. }
            | Divuw { .. }
            | Remw { .. }
            | Remuw { .. }
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
            | Lb { .. }
            | Lh { .. }
            | Lbu { .. }
            | Lhu { .. }
            | Sw { .. }
            | Sd { .. }
            | Sb { .. }
            | Sh { .. }
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
    let mask = if i == 0 { 0 } else { u64::MAX << (64 - i) };
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
    for (name, op, f3, f7, f6) in SELECTOR_TABLE {
        let col: Vec<u8> = (0..t)
            .map(|i| {
                if i < instrs.len() {
                    let ins = &instrs[i];
                    let f3_ok = *f3 == 99 || raw_funct3(ins) == *f3;
                    let f7_ok = f7.map_or(true, |v| raw_funct7(ins) == v);
                    let f6_ok = f6.map_or(true, |v| raw_funct6(ins) == v);
                    (raw_opcode(ins) == *op && f3_ok && f7_ok && f6_ok) as u8
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
    // The P1 sub-word offsets: the address's low three bits, derived
    // from the committed mem_addr value column (booleanity rides the
    // flags family's (c) section — every aux bit column is covered).
    let mem_off_id = [bits.len(), bits.len() + 1, bits.len() + 2];
    for i in 0..3 {
        push_bit!(
            (0..t)
                .map(|c| ((w.mem_addr[c].to_canonical_u64() >> i) & 1) as u8)
                .collect::<Vec<u8>>(),
            "mem_off"
        );
    }
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
        let col: Vec<u8> = (0..t).map(|c| carry_chain64(w.pc[c].0, 4)[l]).collect();
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
            (0..t)
                .map(|c| lt_at(w, instrs, c, cmp, false))
                .collect::<Vec<u8>>(),
            "ltu"
        );
        lt[cmp] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| lt_at(w, instrs, c, cmp, true))
                .collect::<Vec<u8>>(),
            "lt"
        );
    }

    // --- value columns ---
    let mut vals: Vec<Vec<Goldilocks>> = Vec::new();
    let v_mem_addr = vals.len();
    vals.push(w.mem_addr.clone());
    let v_mem_word = vals.len();
    vals.push(w.mem_word.clone());
    let v_pc = vals.len();
    vals.push(w.pc.clone());
    let v_next_pc = vals.len();
    vals.push(w.next_pc.clone());
    let v_fetch_word = vals.len();
    vals.push(w.fetch_word.clone());
    let v_instr = vals.len();
    vals.push(w.instr.clone());
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

    // --- the shift one-hots (the AUX_ONEHOT namespace) ---
    let rs2w = |c: usize| tensor_word(w, T_RS2, c);
    let iword = |c: usize| {
        let mut acc = 0u64;
        for bit in 0..32usize {
            let row = 31 - bit;
            acc |= w.instr_bits.evaluations[row * t + c].0 << bit;
        }
        acc
    };
    let mut shoh6_r = [0usize; 64];
    for s in 0..64usize {
        shoh6_r[s] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| ((rs2w(c) & 63) as usize == s) as u8)
                .collect::<Vec<u8>>(),
            ""
        );
    }
    let mut shoh6_i = [0usize; 64];
    for s in 0..64usize {
        shoh6_i[s] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| (((iword(c) >> 20) & 0x3f) as usize == s) as u8)
                .collect::<Vec<u8>>(),
            ""
        );
    }
    let mut shoh5_r = [0usize; 32];
    for s in 0..32usize {
        shoh5_r[s] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| ((rs2w(c) & 31) as usize == s) as u8)
                .collect::<Vec<u8>>(),
            ""
        );
    }
    let mut shoh5_i = [0usize; 32];
    for s in 0..32usize {
        shoh5_i[s] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| (((iword(c) >> 20) & 0x1f) as usize == s) as u8)
                .collect::<Vec<u8>>(),
            ""
        );
    }

    // --- the eq-to-zero prefixes over rs2 (bz for the 64- and 32-bit
    //     divisor-zero checks) ---
    let mut eqz = [0usize; 65];
    for i in 0..65usize {
        eqz[i] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| {
                    let b = rs2w(c);
                    let mask = if i == 0 { 0 } else { u64::MAX << (64 - i) };
                    ((b & mask) == 0) as u8
                })
                .collect::<Vec<u8>>(),
            ""
        );
    }
    let mut eqzw = [0usize; 33];
    for i in 0..33usize {
        eqzw[i] = bits.len();
        push_bit!(
            (0..t)
                .map(|c| {
                    let b = rs2w(c) & 0xFFFF_FFFF;
                    let mask = if i == 0 { 0 } else { (1u64 << i) - 1 };
                    ((b & mask) == 0) as u8
                })
                .collect::<Vec<u8>>(),
            ""
        );
    }

    // --- the mul/div value columns (limbs + carries; each range-linked
    //     to boolean bit columns below) ---
    let limbs_of = |v: u64| -> [u64; 4] {
        [
            v & 0xFFFF,
            (v >> 16) & 0xFFFF,
            (v >> 32) & 0xFFFF,
            (v >> 48) & 0xFFFF,
        ]
    };
    let rdw = |c: usize| tensor_word(w, T_RD, c);
    let rs1w = |c: usize| tensor_word(w, T_RS1, c);
    let immw = |c: usize| tensor_word(w, T_IMM, c);
    // The per-cycle mul witness: (lo, hi, carries) of |rs1|*|rs2|.
    let mul_wit = |c: usize| -> ([u64; 4], [u64; 4], [u64; 7]) {
        let a = rs1w(c);
        let b = rs2w(c);
        let (lo, hi, car) = mul_recurrence(a, b);
        (lo, hi, car)
    };
    // The per-cycle div witness per class.
    let div_wit = |c: usize| -> DivWitness {
        let a = rs1w(c);
        let b = rs2w(c);
        let rd = rdw(c);
        let ins = instrs.get(c).copied().unwrap_or(Instr::Ecall);
        div_witness_of(a, b, rd, &ins)
    };
    let mut v_mul_lo = [0usize; 4];
    for l in 0..4 {
        v_mul_lo[l] = vals.len();
        vals.push((0..t).map(|c| fe(mul_wit(c).0[l])).collect());
    }
    let mut v_mul_hi = [0usize; 4];
    for l in 0..4 {
        v_mul_hi[l] = vals.len();
        vals.push((0..t).map(|c| fe(mul_wit(c).1[l])).collect());
    }
    let mut v_mul_c = [0usize; 7];
    for l in 0..7 {
        v_mul_c[l] = vals.len();
        vals.push((0..t).map(|c| fe(mul_wit(c).2[l])).collect());
    }
    let mut v_mulh_bor = [0usize; 4];
    {
        let bor: Vec<[u64; 4]> = (0..t)
            .map(|c| {
                let a = rs1w(c);
                let b = rs2w(c);
                let rd = rdw(c);
                let hi = limbs_of(((a as u128 * b as u128) >> 64) as u64);
                mulh_borrows(
                    hi,
                    (a >> 63) & 1,
                    (b >> 63) & 1,
                    limbs_of(a),
                    limbs_of(b),
                    limbs_of(rd),
                )
            })
            .collect();
        for l in 0..4 {
            v_mulh_bor[l] = vals.len();
            vals.push((0..t).map(|c| fe(bor[c][l])).collect());
        }
    }
    let mut v_div_q = [0usize; 4];
    for l in 0..4 {
        v_div_q[l] = vals.len();
        vals.push((0..t).map(|c| fe(div_wit(c).q[l])).collect());
    }
    let mut v_div_r = [0usize; 4];
    for l in 0..4 {
        v_div_r[l] = vals.len();
        vals.push((0..t).map(|c| fe(div_wit(c).r[l])).collect());
    }
    let mut v_div_d = [0usize; 4];
    {
        let ds: Vec<[u64; 4]> = (0..t)
            .map(|c| {
                let d = div_wit(c);
                div_recurrence(d.mag_a, d.mag_q, d.mag_b, d.mag_r)
            })
            .collect();
        for l in 0..4 {
            v_div_d[l] = vals.len();
            vals.push((0..t).map(|c| fe(ds[c][l])).collect());
        }
    }
    let mut v_mag_a = [0usize; 4];
    for l in 0..4 {
        v_mag_a[l] = vals.len();
        vals.push((0..t).map(|c| fe(div_wit(c).mag_a[l])).collect());
    }
    let mut v_mag_b = [0usize; 4];
    for l in 0..4 {
        v_mag_b[l] = vals.len();
        vals.push((0..t).map(|c| fe(div_wit(c).mag_b[l])).collect());
    }
    let mut v_mag_q = [0usize; 4];
    for l in 0..4 {
        v_mag_q[l] = vals.len();
        vals.push((0..t).map(|c| fe(div_wit(c).mag_q[l])).collect());
    }
    let mut v_mag_r = [0usize; 4];
    for l in 0..4 {
        v_mag_r[l] = vals.len();
        vals.push((0..t).map(|c| fe(div_wit(c).mag_r[l])).collect());
    }
    let mut v_mag_d = [0usize; 4];
    {
        let ds: Vec<[u64; 4]> = (0..t)
            .map(|c| {
                let d = div_wit(c);
                div_recurrence(d.mag_a, d.mag_q, d.mag_b, d.mag_r)
            })
            .collect();
        for l in 0..4 {
            v_mag_d[l] = vals.len();
            vals.push((0..t).map(|c| fe(ds[c][l])).collect());
        }
    }
    let _ = immw;
    // The (mag_r - mag_b) subtraction: the out limbs (range-linked) and
    // the borrow bits (bor_4 = [mag_r < mag_b]).
    let mut v_rlt_out = [0usize; 4];
    let mut v_rbor = [0usize; 4];
    {
        let chain: Vec<([u64; 4], [u8; 4])> = (0..t)
            .map(|c| {
                let d = div_wit(c);
                borrow_chain(d.mag_r, d.mag_b)
            })
            .collect();
        for l in 0..4 {
            v_rlt_out[l] = vals.len();
            vals.push((0..t).map(|c| fe(chain[c].0[l])).collect());
        }
        for l in 0..4 {
            v_rbor[l] = bits.len();
            push_bit!((0..t).map(|c| chain[c].1[l]).collect::<Vec<u8>>(), "");
        }
    }

    // --- the range bit decompositions (the integer-semantics anchor) ---
    // Layout (group-major, limb-major, bit-minor):
    //   [0..64)    mul_lo  (4 x 16)
    //   [64..128)  mul_hi  (4 x 16)
    //   [128..261) mul_c   (7 x 19)
    //   [261..269) mulh_bor(4 x 2)
    //   [269..333) div_q   (4 x 16)
    //   [333..397) div_r   (4 x 16)
    //   [397..473) div_d   (4 x 19)
    //   [473..537) mag_a   (4 x 16)
    //   [537..601) mag_b   (4 x 16)
    //   [601..665) mag_q   (4 x 16)
    //   [665..729) mag_r   (4 x 16)
    //   [729..805) mag_d   (4 x 19)
    //   [805..869) rlt_out (4 x 16)
    let mut range_bits: Vec<usize> = Vec::with_capacity(RANGE_BITS_LEN);
    macro_rules! push_range_group {
        ($cols:expr, $width:expr) => {{
            let mut ids: Vec<usize> = Vec::with_capacity($cols.len() * $width);
            for &col in $cols.iter() {
                for j in 0..$width {
                    ids.push(bits.len());
                    push_bit!(
                        (0..t)
                            .map(|c| ((vals[col][c].0 >> j) & 1) as u8)
                            .collect::<Vec<u8>>(),
                        ""
                    );
                }
            }
            ids
        }};
    }
    let rb_mul_lo = push_range_group!(v_mul_lo, 16);
    let rb_mul_hi = push_range_group!(v_mul_hi, 16);
    let rb_mul_c = push_range_group!(v_mul_c, 19);
    let rb_mulh_bor = push_range_group!(v_mulh_bor, 2);
    let rb_div_q = push_range_group!(v_div_q, 16);
    let rb_div_r = push_range_group!(v_div_r, 16);
    let rb_div_d = push_range_group!(v_div_d, 19);
    let rb_mag_a = push_range_group!(v_mag_a, 16);
    let rb_mag_b = push_range_group!(v_mag_b, 16);
    let rb_mag_q = push_range_group!(v_mag_q, 16);
    let rb_mag_r = push_range_group!(v_mag_r, 16);
    let rb_mag_d = push_range_group!(v_mag_d, 19);
    let rb_rlt_out = push_range_group!(v_rlt_out, 16);
    range_bits.extend_from_slice(&rb_mul_lo);
    range_bits.extend_from_slice(&rb_mul_hi);
    range_bits.extend_from_slice(&rb_mul_c);
    range_bits.extend_from_slice(&rb_mulh_bor);
    range_bits.extend_from_slice(&rb_div_q);
    range_bits.extend_from_slice(&rb_div_r);
    range_bits.extend_from_slice(&rb_div_d);
    range_bits.extend_from_slice(&rb_mag_a);
    range_bits.extend_from_slice(&rb_mag_b);
    range_bits.extend_from_slice(&rb_mag_q);
    range_bits.extend_from_slice(&rb_mag_r);
    range_bits.extend_from_slice(&rb_mag_d);
    range_bits.extend_from_slice(&rb_rlt_out);
    debug_assert_eq!(range_bits.len(), RANGE_BITS_LEN);

    let index = AuxIndex {
        sel,
        rd_we: rd_we_id,
        mem_re: mem_re_id,
        mem_we: mem_we_id,
        mem_half: mem_half_id,
        mem_off: mem_off_id,
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
        v_instr,
        v_pc_l,
        v_np_l,
        v_addr_l,
        shoh6_r,
        shoh6_i,
        shoh5_r,
        shoh5_i,
        v_mul_lo,
        v_mul_hi,
        v_mul_c,
        v_mulh_bor,
        v_div_q,
        v_div_r,
        v_div_d,
        v_mag_a,
        v_mag_b,
        v_mag_q,
        v_mag_r,
        v_mag_d,
        v_rbor,
        v_rlt_out,
        eqz,
        eqzw,
        range_bits,
    };
    Ok(AuxCols {
        bits,
        vals,
        bit_names: names,
        index,
    })
}

fn raw_opcode(instr: &Instr) -> u32 {
    use Instr::*;
    match instr {
        Addi { .. }
        | Slti { .. }
        | Sltiu { .. }
        | Xori { .. }
        | Ori { .. }
        | Andi { .. }
        | Slli { .. }
        | Srli { .. }
        | Srai { .. } => 0x13,
        Addiw { .. } | Slliw { .. } | Srliw { .. } | Sraiw { .. } => 0x1b,
        Add { .. }
        | Sub { .. }
        | Sll { .. }
        | Slt { .. }
        | Sltu { .. }
        | Xor { .. }
        | Srl { .. }
        | Sra { .. }
        | Or { .. }
        | And { .. }
        | Mul { .. }
        | Mulh { .. }
        | Mulhu { .. }
        | Div { .. }
        | Divu { .. }
        | Rem { .. }
        | Remu { .. } => 0x33,
        Addw { .. }
        | Subw { .. }
        | Sllw { .. }
        | Srlw { .. }
        | Sraw { .. }
        | Mulw { .. }
        | Divw { .. }
        | Divuw { .. }
        | Remw { .. }
        | Remuw { .. } => 0x3b,
        Lui { .. } => 0x37,
        Auipc { .. } => 0x17,
        Jal { .. } => 0x6f,
        Jalr { .. } => 0x67,
        Beq { .. } | Bne { .. } | Blt { .. } | Bge { .. } | Bltu { .. } | Bgeu { .. } => 0x63,
        Lw { .. } | Lwu { .. } | Ld { .. } => 0x03,
        Lb { .. } | Lh { .. } | Lbu { .. } | Lhu { .. } => 0x03,
        Sw { .. } | Sd { .. } => 0x23,
        Sb { .. } | Sh { .. } => 0x23,
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
        Srli { .. }
        | Srai { .. }
        | Srliw { .. }
        | Sraiw { .. }
        | Srl { .. }
        | Sra { .. }
        | Srlw { .. }
        | Sraw { .. } => 5,
        // OP
        Add { .. } | Sub { .. } | Mul { .. } | Addw { .. } | Subw { .. } | Addiw { .. } => 0,
        Slt { .. } => 2,
        Sltu { .. } => 3,
        Xor { .. } => 4,
        Or { .. } => 6,
        And { .. } => 7,
        Mulh { .. } => 1,
        Mulhu { .. } => 3,
        Div { .. } | Divw { .. } => 4,
        Divu { .. } | Divuw { .. } => 5,
        Rem { .. } | Remw { .. } => 6,
        Remu { .. } | Remuw { .. } => 7,
        Mulw { .. } => 0,
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
        Lb { .. } | Sb { .. } => 0,
        Lh { .. } | Sh { .. } => 1,
        Lbu { .. } => 4,
        Lhu { .. } => 5,
        // system
        Ecall => 0,
        Ebreak => 1,
        // U/J-type: the funct3 field is imm data (free in the masks).
        Lui { .. } | Auipc { .. } | Jal { .. } => 0,
        Jalr { .. } => 0,
        _ => 0,
    }
}

/// The funct6 field (bits 26..31): the RV64 6-bit-shamt discriminant
/// for the 64-bit shift immediates.
fn raw_funct6(instr: &Instr) -> u32 {
    use Instr::*;
    match instr {
        Slli { .. } | Srli { .. } => 0,
        Srai { .. } => 0x10,
        _ => 0,
    }
}

fn raw_funct7(instr: &Instr) -> u32 {
    use Instr::*;
    match instr {
        Sub { .. } | Subw { .. } | Sra { .. } | Sraw { .. } | Sraiw { .. } => 0x20,
        Mul { .. }
        | Mulh { .. }
        | Mulhu { .. }
        | Mulw { .. }
        | Div { .. }
        | Divu { .. }
        | Divw { .. }
        | Divuw { .. }
        | Rem { .. }
        | Remu { .. }
        | Remw { .. }
        | Remuw { .. } => 0x01,
        Add { .. }
        | Sll { .. }
        | Slt { .. }
        | Sltu { .. }
        | Xor { .. }
        | Srl { .. }
        | Or { .. }
        | And { .. }
        | Addw { .. }
        | Sllw { .. }
        | Srlw { .. }
        | Slliw { .. }
        | Srliw { .. } => 0x00,
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

fn carry_at(
    w: &CycleWitness,
    instrs: &[Instr],
    c: usize,
    src_imm: bool,
    l: usize,
    name: &str,
) -> u8 {
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

// ---------------------------------------------------------------------------
// The mul/div recurrence helpers (the limb-level integer discipline).
// Every column they produce is range-linked to boolean bit columns, so
// the limb recurrences carry INTEGER (not merely field) semantics.
// ---------------------------------------------------------------------------

/// The total number of range bit columns (see the layout table in
/// `build_aux`).
pub const RANGE_BITS_LEN: usize = 869;

/// Range-group offsets into `AuxIndex::range_bits` (the carry groups
/// carry 19 bits: positions with up to four 16-bit limb products plus
/// the incoming carry reach ~2^18).
pub const RG_MUL_LO: usize = 0;
pub const RG_MUL_HI: usize = 64;
pub const RG_MUL_C: usize = 128;
pub const RG_MULH_BOR: usize = 261;
pub const RG_DIV_Q: usize = 269;
pub const RG_DIV_R: usize = 333;
pub const RG_DIV_D: usize = 397;
pub const RG_MAG_A: usize = 473;
pub const RG_MAG_B: usize = 537;
pub const RG_MAG_Q: usize = 601;
pub const RG_MAG_R: usize = 665;
pub const RG_MAG_D: usize = 729;
pub const RG_RLT_OUT: usize = 805;

/// The 4-limb decomposition of a u64.
fn limbs64(v: u64) -> [u64; 4] {
    [
        v & 0xFFFF,
        (v >> 16) & 0xFFFF,
        (v >> 32) & 0xFFFF,
        (v >> 48) & 0xFFFF,
    ]
}

/// The multiplication recurrence over 16-bit limbs:
/// `t_k = S_k(a,b) + c_k`, `p_k = t_k mod 2^16`, `c_{k+1} = t_k >> 16`.
/// Returns (lo limbs p_0..3, hi limbs p_4..7, carries c_1..=c_7);
/// `c_8 = 0` by the 128-bit closure (the k = 7 identity omits the
/// carry-out term).
fn mul_recurrence(a: u64, b: u64) -> ([u64; 4], [u64; 4], [u64; 7]) {
    let al = limbs64(a);
    let bl = limbs64(b);
    let mut out = [0u64; 8];
    let mut carries = [0u64; 7];
    let mut c = 0u64; // c_k (c_0 = 0)
    for k in 0..8usize {
        let mut s = c;
        for i in 0..4usize {
            for j in 0..4usize {
                if i + j == k {
                    s += al[i] * bl[j];
                }
            }
        }
        out[k] = s & 0xFFFF;
        c = s >> 16;
        if k < 7 {
            carries[k] = c; // carries[k] = c_{k+1}
        }
    }
    (
        [out[0], out[1], out[2], out[3]],
        [out[4], out[5], out[6], out[7]],
        carries,
    )
}

/// The division recurrence `a = q·b + r` over 16-bit limbs: at position
/// k, `S_k(q,b) + d_k + r_k - a_k - 2^16·d_{k+1} = 0`. Returns
/// `d_1..=d_4` (d_4 = 0 when the identity closes over 64 bits).
fn div_recurrence(_a: [u64; 4], q: [u64; 4], b: [u64; 4], r: [u64; 4]) -> [u64; 4] {
    let mut d = [0u64; 4];
    let mut carry = 0u64; // d_k
    for k in 0..4usize {
        let mut s = carry;
        for i in 0..4usize {
            for j in 0..4usize {
                if i + j == k {
                    s += q[i] * b[j];
                }
            }
        }
        s += r[k];
        let next = s >> 16;
        if k < 3 {
            d[k] = next; // d_{k+1}
        } else {
            d[3] = next; // d_4
        }
        carry = next;
    }
    d
}

/// The subtraction borrow chain `r - b = out - 2^64·bor_4`: returns
/// (the out limbs of `(r - b) mod 2^64`, the borrows bor_1..=bor_4 with
/// bor_4 = [r < b]).
fn borrow_chain(r: [u64; 4], b: [u64; 4]) -> ([u64; 4], [u8; 4]) {
    let mut out = [0u64; 4];
    let mut bor = [0u8; 4];
    let mut bin = 0i128;
    for l in 0..4usize {
        let v = r[l] as i128 - b[l] as i128 - bin;
        if v < 0 {
            out[l] = (v + (1i128 << 16)) as u64;
            bor[l] = 1;
            bin = 1;
        } else {
            out[l] = v as u64;
            bor[l] = 0;
            bin = 0;
        }
    }
    (out, bor)
}

/// The MULH rd-composition borrows: `rd_l = hi_l - sa·b_l - sb·a_l -
/// bor_l + 2^16·bor_{l+1}` (bor_0 = 0; bor values in {0, 1, 2}).
fn mulh_borrows(
    hi: [u64; 4],
    sa: u64,
    sb: u64,
    a: [u64; 4],
    b: [u64; 4],
    _rd: [u64; 4],
) -> [u64; 4] {
    let mut bor = [0u64; 4];
    let mut bin = 0i128;
    for l in 0..4usize {
        let x = sa * b[l] + sb * a[l];
        let v = hi[l] as i128 - x as i128 - bin;
        let bout = if v < 0 {
            ((-v + (1i128 << 16) - 1) / (1i128 << 16)) as u64
        } else {
            0
        };
        bor[l] = bout;
        bin = bout as i128;
    }
    bor
}

/// The per-cycle division witness.
#[derive(Clone, Copy, Debug)]
struct DivWitness {
    /// The quotient (the rd value for the DIV forms).
    q: [u64; 4],
    /// The remainder (the rd value for the REM forms).
    r: [u64; 4],
    /// The operand magnitude |a| (raw for the unsigned classes).
    mag_a: [u64; 4],
    /// |b| (raw for unsigned).
    mag_b: [u64; 4],
    /// |q| (raw for unsigned).
    mag_q: [u64; 4],
    /// |r| (raw for unsigned).
    mag_r: [u64; 4],
}

/// The division witness per instruction class (matching the executor's
/// semantics exactly, including the divide-by-zero specials).
fn div_witness_of(a: u64, b: u64, _rd: u64, ins: &Instr) -> DivWitness {
    use Instr::*;
    let limbs = limbs64;
    let (q, r, ma, mb, mq, mr) = match ins {
        Divu { .. } => {
            let (q, r) = match (b, a.checked_div(b), a.checked_rem(b)) {
                (0, _, _) => (u64::MAX, a),
                (_, Some(q), Some(r)) => (q, r),
                _ => (u64::MAX, a),
            };
            (q, r, a, b, q, r)
        }
        Remu { .. } => {
            let (q, r) = match (b, a.checked_div(b), a.checked_rem(b)) {
                (0, _, _) => (u64::MAX, a),
                (_, Some(q), Some(r)) => (q, r),
                _ => (u64::MAX, a),
            };
            (q, r, a, b, q, r)
        }
        Div { .. } => {
            let (q, r) = if b == 0 {
                (u64::MAX, a)
            } else {
                (
                    (a as i64).wrapping_div(b as i64) as u64,
                    (a as i64).wrapping_rem(b as i64) as u64,
                )
            };
            let ma = (a as i64).unsigned_abs();
            let mb = (b as i64).unsigned_abs();
            let mq = (q as i64).unsigned_abs();
            let mr = (r as i64).unsigned_abs();
            (q, r, ma, mb, mq, mr)
        }
        Rem { .. } => {
            let (q, r) = if b == 0 {
                (u64::MAX, a)
            } else {
                (
                    (a as i64).wrapping_div(b as i64) as u64,
                    (a as i64).wrapping_rem(b as i64) as u64,
                )
            };
            let ma = (a as i64).unsigned_abs();
            let mb = (b as i64).unsigned_abs();
            let mq = (q as i64).unsigned_abs();
            let mr = (r as i64).unsigned_abs();
            (q, r, ma, mb, mq, mr)
        }
        Divw { .. } => {
            let a32 = a as u32 as i32 as i64;
            let b32 = b as u32 as i32 as i64;
            let (q, r) = if (b as u32) == 0 {
                (-1i64, a32)
            } else {
                (a32.wrapping_div(b32), a32.wrapping_rem(b32))
            };
            let q64 = q as u64;
            let r64 = r as u64;
            (
                q64,
                r64,
                a32.unsigned_abs(),
                b32.unsigned_abs(),
                q.unsigned_abs(),
                r.unsigned_abs(),
            )
        }
        Remw { .. } => {
            let a32 = a as u32 as i32 as i64;
            let b32 = b as u32 as i32 as i64;
            let (q, r) = if (b as u32) == 0 {
                (-1i64, a32)
            } else {
                (a32.wrapping_div(b32), a32.wrapping_rem(b32))
            };
            let q64 = q as u64;
            let r64 = r as u64;
            (
                q64,
                r64,
                a32.unsigned_abs(),
                b32.unsigned_abs(),
                q.unsigned_abs(),
                r.unsigned_abs(),
            )
        }
        Divuw { .. } => {
            let au = a as u32 as u64;
            let bu = b as u32 as u64;
            let (q, r) = if (b as u32) == 0 {
                (u32::MAX as u64, au)
            } else {
                (au / bu, au % bu)
            };
            // The executor's se32 composition happens at rd-routing;
            // the raw 32-bit witness here.
            let q32 = q as u32 as u64;
            let r32 = r as u32 as u64;
            (q32, r32, au, bu, q32, r32)
        }
        Remuw { .. } => {
            let au = a as u32 as u64;
            let bu = b as u32 as u64;
            let (q, r) = if (b as u32) == 0 {
                (u32::MAX as u64, au)
            } else {
                (au / bu, au % bu)
            };
            let q32 = q as u32 as u64;
            let r32 = r as u32 as u64;
            (q32, r32, au, bu, q32, r32)
        }
        _ => (0, 0, 0, 0, 0, 0),
    };
    DivWitness {
        q: limbs(q),
        r: limbs(r),
        mag_a: limbs(ma),
        mag_b: limbs(mb),
        mag_q: limbs(mq),
        mag_r: limbs(mr),
    }
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
    /// The complement of a bit column (1 − the bit): the P1 sub-word
    /// position products' zero legs.
    FlipBit(usize),
    /// A value column over log T.
    Val(usize),
    /// A fixed tensor row, lifted constant over the bit block: the claim
    /// is the tensor-row claim.
    TensorRow {
        factor: Factor,
        nbits: usize,
        row: usize,
    },
    /// The complement of a fixed tensor row (the flipped-polarity
    /// factors of the selector decode): the claim is 1 - the row.
    FlipTensorRow {
        factor: Factor,
        nbits: usize,
        row: usize,
    },
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
    #[cfg(debug_assertions)]
    {
        let seq: Vec<(u8, usize)> = views
            .iter()
            .filter_map(|(_, v)| match v {
                FV::PubTable(_) => None,
                FV::Tensor(f) => Some((f.discriminant(), f.payload())),
                FV::Limb { slot, .. } => Some((0, *slot)),
                FV::Combo { slot } => Some((0, *slot)),
                FV::InstrWord => Some((1, 0)),
                FV::Bit(id) => Some((3, *id)),
                FV::FlipBit(id) => Some((3, *id)),
                FV::Val(id) => Some((4, *id)),
                FV::TensorRow { factor, .. } => Some((factor.discriminant(), factor.payload())),
                FV::FlipTensorRow { factor, .. } => Some((factor.discriminant(), factor.payload())),
            })
            .collect();
        let _ = seq;
    }
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
        FV::FlipBit(id) => Some(
            Goldilocks::ONE.sub(&ledger.tensor_claim(Factor::BitCol { id: *id }, point)?),
        ),
        FV::Val(id) => Some(ledger.tensor_claim(Factor::ValCol { id: *id }, point)?),
        FV::TensorRow { factor, nbits, row } => {
            let mut pt = idx_point((*nbits).trailing_zeros() as usize, *row);
            pt.extend_from_slice(point);
            Some(ledger.tensor_claim(*factor, &pt)?)
        }
        FV::FlipTensorRow { factor, nbits, row } => {
            let mut pt = idx_point((*nbits).trailing_zeros() as usize, *row);
            pt.extend_from_slice(point);
            Some(Goldilocks::ONE.sub(&ledger.tensor_claim(*factor, &pt)?))
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

/// Env-gated per-family prover timing (the semantics benchmark's
/// attribution output; zero cost when `LZX_SEM_TIMING` is unset).
struct FamilyTimer(std::time::Instant, &'static str);

#[inline]
fn family_timer(name: &'static str) -> Option<FamilyTimer> {
    if std::env::var_os("LZX_SEM_TIMING").is_some() {
        Some(FamilyTimer(std::time::Instant::now(), name))
    } else {
        None
    }
}

impl Drop for FamilyTimer {
    fn drop(&mut self) {
        eprintln!(
            "[sem-family] {:<14} {:>10.1} ms",
            self.1,
            self.0.elapsed().as_secs_f64() * 1e3
        );
    }
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
        let _timer = family_timer(name);
        let out = sumcheck::prove(vp, claim, self.transcript).map_err(ConstraintError::Sumcheck)?;
        bind_views(self.ledger, views, &out.challenges, &out.factor_claims)?;
        self.legs.push(ConstraintLeg {
            name,
            sc: out.proof,
            claim,
        });
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
    views.push((
        fi,
        FV::TensorRow {
            factor: Factor::ValueBits { slot },
            nbits: 64,
            row,
        },
    ));
    Ok(fi)
}

/// Add an instruction-bit row factor (bit in LSB numbering).
/// Ledger claim helpers (verify side).
fn claim_bit(
    ledger: &mut Ledger<'_>,
    id: usize,
    pt: &[Goldilocks],
) -> Result<Goldilocks, ConstraintError> {
    Ok(ledger.tensor_claim(Factor::BitCol { id }, pt)?)
}
fn claim_val(
    ledger: &mut Ledger<'_>,
    id: usize,
    pt: &[Goldilocks],
) -> Result<Goldilocks, ConstraintError> {
    Ok(ledger.tensor_claim(Factor::ValCol { id }, pt)?)
}
fn claim_limb(
    ledger: &mut Ledger<'_>,
    slot: usize,
    limb: usize,
    pt: &[Goldilocks],
) -> Result<Goldilocks, ConstraintError> {
    Ok(ledger.limb(slot, limb, pt)?)
}
/// Resolve a value-tensor bit (LSB numbering) at the verdict point.
fn claim_tensor_bit(
    ledger: &mut Ledger<'_>,
    slot: usize,
    bit: usize,
    pt: &[Goldilocks],
) -> Result<Goldilocks, ConstraintError> {
    let mut p = idx_point(6, 63 - bit);
    p.extend_from_slice(pt);
    Ok(ledger.tensor_claim(Factor::ValueBits { slot }, &p)?)
}

/// Resolve a carry value c_k (k = 0 -> ZERO) from its column family.
fn claim_carry(
    ledger: &mut Ledger<'_>,
    cols: &[usize],
    k: usize,
    pt: &[Goldilocks],
) -> Result<Goldilocks, ConstraintError> {
    if k == 0 {
        return Ok(Goldilocks::ZERO);
    }
    claim_val(ledger, cols[k - 1], pt)
}

fn claim_vbit(
    ledger: &mut Ledger<'_>,
    slot: usize,
    bit: usize,
    pt: &[Goldilocks],
) -> Result<Goldilocks, ConstraintError> {
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
        let ei = vp
            .add_factor(eq.clone())
            .map_err(ConstraintError::Virtual)?;
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
                views.push((
                    i1,
                    FV::TensorRow {
                        factor: Factor::ValueBits { slot },
                        nbits: 64,
                        row,
                    },
                ));
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
        let ei = vp
            .add_factor(eq.clone())
            .map_err(ConstraintError::Virtual)?;
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
            views.push((
                i1,
                FV::TensorRow {
                    factor: Factor::InstrBits,
                    nbits: 32,
                    row,
                },
            ));
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
        let ei = vp
            .add_factor(eq.clone())
            .map_err(ConstraintError::Virtual)?;
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

fn verify_booleanity_shape<'l>(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    ledger: &mut Ledger<'l>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
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
/// required value). Derived from (opcode, funct3, funct7, funct6). The
/// `funct6` field (bits 26..31) is the RV64 6-bit-shamt discriminant.
fn selector_mask(op: u32, f3: u32, f7: Option<u32>, f6: Option<u32>) -> Vec<(usize, u8)> {
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
    if let Some(v) = f6 {
        for b in 0..6 {
            mask.push((26 + b, ((v >> b) & 1) as u8));
        }
    }
    mask
}

/// The maximum selector mask width (degree cap for the sel leg).
fn sel_max_degree() -> usize {
    SELECTOR_TABLE
        .iter()
        .map(|(_, op, f3, f7, f6)| selector_mask(*op, *f3, *f7, *f6).len())
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
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // Instruction-bit row factors, both polarities, memoized by (bit, req).
    let mut memo: std::collections::HashMap<(usize, u8), usize> = std::collections::HashMap::new();
    for (j, (name, op, f3, f7, f6)) in SELECTOR_TABLE.iter().enumerate() {
        let _ = name;
        let mask = selector_mask(*op, *f3, *f7, *f6);
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
                let f = if *req == 1 { col } else { flip_mle(&col) };
                let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
                // BOTH polarities claim their row (the queue stays in
                // lockstep with the verifier's per-row resolution; the
                // flipped factor's view resolves to 1 - the row).
                if *req == 1 {
                    views.push((
                        fi,
                        FV::TensorRow {
                            factor: Factor::InstrBits,
                            nbits: 32,
                            row,
                        },
                    ));
                } else {
                    views.push((
                        fi,
                        FV::FlipTensorRow {
                            factor: Factor::InstrBits,
                            nbits: 32,
                            row,
                        },
                    ));
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
        evaluations: m
            .evaluations
            .iter()
            .map(|v| Goldilocks::ONE.sub(v))
            .collect(),
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
    for (j, (_, op, f3, f7, f6)) in SELECTOR_TABLE.iter().enumerate() {
        let s = ledger.tensor_claim(
            Factor::BitCol {
                id: aux.index.sel[j],
            },
            &verdict.point,
        )?;
        let mask = selector_mask(*op, *f3, *f7, *f6);
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
                let v = if *req == 1 {
                    b
                } else {
                    Goldilocks::ONE.sub(&b)
                };
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
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    let neg_sum_terms =
        |vp: &mut VirtualPolynomial, target: usize, ids: &[usize]| -> Result<(), ConstraintError> {
            // eq * (target - sum(ids)) = sum eq*target - sum eq*id
            vp.add_term(alphas[0].mul(&fe(1)), vec![target, ei])
                .map_err(ConstraintError::Virtual)?;
            for id in ids {
                vp.add_term(alphas[0].neg(), vec![*id, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            Ok(())
        };
    // (1) mem_re = lw + lwu + ld + lb + lbu + lh + lhu
    {
        let tgt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_re, log_t)?;
        let lw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lw"), log_t)?;
        let lwu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lwu"), log_t)?;
        let ld = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_ld"), log_t)?;
        let lb = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lb"), log_t)?;
        let lbu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lbu"), log_t)?;
        let lh = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lh"), log_t)?;
        let lhu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_lhu"), log_t)?;
        neg_sum_terms(&mut vp, tgt, &[lw, lwu, ld, lb, lbu, lh, lhu])?;
    }
    // (2) mem_we = sw + sd + sb + sh
    {
        let tgt = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_we, log_t)?;
        let sw = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sw"), log_t)?;
        let sd = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sd"), log_t)?;
        let sb = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sb"), log_t)?;
        let sh = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sh"), log_t)?;
        neg_sum_terms(&mut vp, tgt, &[sw, sd, sb, sh])?;
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
        let bltu = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_bltu"),
            log_t,
        )?;
        let bgeu = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_bgeu"),
            log_t,
        )?;
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
            w_ids.push(add_bit_factor(
                &mut vp,
                &mut views,
                &aux.bits,
                idx.sel_by(name),
                log_t,
            )?);
        }
        // term A: rd_we
        vp.add_term(alphas[2], vec![tgt, ei])
            .map_err(ConstraintError::Virtual)?;
        // term B: - sum_k s_k  (eq-weighted)
        for id in &w_ids {
            vp.add_term(alphas[2].neg(), vec![*id, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        // term C: + sum_k s_k * prod(1 - rd_i) — the flipped rows carry
        // FlipTensorRow views so their claims ride the queue (the
        // verifier resolves each rd bit row and derives the flip).
        let mut not_rd_ids = Vec::new();
        for bit in 7..12 {
            let row = 31 - bit;
            let col = instr_row_of(&w.instr_bits, row, log_t);
            let f = flip_mle(&col);
            let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
            views.push((
                fi,
                FV::FlipTensorRow {
                    factor: Factor::InstrBits,
                    nbits: 32,
                    row,
                },
            ));
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
        let d = claim_bit(ledger, idx.sel_by("sel_lb"), pt)?;
        let e = claim_bit(ledger, idx.sel_by("sel_lbu"), pt)?;
        let f = claim_bit(ledger, idx.sel_by("sel_lh"), pt)?;
        let g = claim_bit(ledger, idx.sel_by("sel_lhu"), pt)?;
        expect = expect
            .add(&alphas[0]
                .mul(&t.sub(&a).sub(&b).sub(&c).sub(&d).sub(&e).sub(&f).sub(&g))
                .mul(&eq_at));
    }
    // (2) mem_we
    {
        let t = claim_bit(ledger, idx.mem_we, pt)?;
        let a = claim_bit(ledger, idx.sel_by("sel_sw"), pt)?;
        let b = claim_bit(ledger, idx.sel_by("sel_sd"), pt)?;
        let c = claim_bit(ledger, idx.sel_by("sel_sb"), pt)?;
        let d = claim_bit(ledger, idx.sel_by("sel_sh"), pt)?;
        expect = expect.add(&alphas[0].mul(&t.sub(&a).sub(&b).sub(&c).sub(&d)).mul(&eq_at));
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
        let v = beq
            .mul(&eq0)
            .add(&bne.mul(&one.sub(&eq0)))
            .add(&blt.mul(&lt0))
            .add(&bge.mul(&one.sub(&lt0)))
            .add(&bltu.mul(&ltu0))
            .add(&bgeu.mul(&one.sub(&ltu0)));
        expect = expect.add(&alphas[1].mul(&t.sub(&v)).mul(&eq_at));
    }
    // (5) rd_we = W - W * prod(1 - rd_i) — claim order follows the
    // prover's view order (rd_we, the write-class selectors, then the
    // rd-bit rows).
    {
        let t = claim_bit(ledger, idx.rd_we, pt)?;
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
        let mut not_rd = Goldilocks::ONE;
        for bit_pos in 7..12 {
            let row = 31 - bit_pos;
            let mut p = idx_point(5, row);
            p.extend_from_slice(pt);
            let b = ledger.tensor_claim(Factor::InstrBits, &p)?;
            not_rd = not_rd.mul(&Goldilocks::ONE.sub(&b));
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
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
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
            vp.add_term(*a, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in = add_bit_factor(
                    &mut vp,
                    &mut views,
                    &aux.bits,
                    idx.carry_add_r[l - 1],
                    log_t,
                )?;
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
            vp.add_term(*a, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in = add_bit_factor(
                    &mut vp,
                    &mut views,
                    &aux.bits,
                    idx.carry_add_i[l - 1],
                    log_t,
                )?;
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
            vp.add_term(*b, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*b, vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
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
            vp.add_term(*b, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.mul(&fe(1 << 16)), vec![sel, c_out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let c_in = add_bit_factor(&mut vp, &mut views, &aux.bits, chain[l - 1], log_t)?;
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
            vp.add_term(*b, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(b.neg(), vec![sel, x, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*b, vec![sel, y, ei])
                .map_err(ConstraintError::Virtual)?;
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
        views.push((
            sign_f,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RD },
                nbits: 64,
                row: 32,
            },
        ));
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
            vp.add_term(*b, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
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
            vp.add_term(*b, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
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
            vp.add_term(*b, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
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
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // x slot and y slot per comparison: cmp0 = (rs1, rs2), cmp1 = (rs1, imm).
    let y_slot = [T_RS2, T_IMM];
    for cmp in 0..2 {
        // (1) eqp recurrence: eqp[i+1] = eqp[i] * eq(bit(63 - i))
        //     eq(bit) = 1 - a - b + 2ab.
        for i in 0..64 {
            let next = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[cmp][i + 1], log_t)?;
            let prev = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqp[cmp][i], log_t)?;
            let a_f = add_vbit_factor(&mut vp, &mut views, w, T_RS1, 63 - i, log_t)?;
            let b_f = add_vbit_factor(&mut vp, &mut views, w, y_slot[cmp], 63 - i, log_t)?;
            // next - prev*(1 - a - b + 2ab)
            //   = next - prev + prev*a + prev*b - 2*prev*a*b
            let al = &alphas[0];
            vp.add_term(*al, vec![next, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(al.neg(), vec![prev, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*al, vec![prev, a_f, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*al, vec![prev, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(al.mul(&fe(2).neg()), vec![prev, a_f, b_f, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        // (2) ltu = sum_p eqp[63-p] * (1 - a_p) * b_p
        let ltu = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.ltu[cmp], log_t)?;
        let al = &alphas[1];
        vp.add_term(*al, vec![ltu, ei])
            .map_err(ConstraintError::Virtual)?;
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
        vp.add_term(*al, vec![lt, ei])
            .map_err(ConstraintError::Virtual)?;
        vp.add_term(al.neg(), vec![a63, ei])
            .map_err(ConstraintError::Virtual)?;
        vp.add_term(*al, vec![a63, b63, ei])
            .map_err(ConstraintError::Virtual)?;
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
            let e = next
                .sub(&prev)
                .add(&prev.mul(&a.add(&b).sub(&Goldilocks::from_u64(2).mul(&a.mul(&b)))));
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
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    let a = &alphas[0];
    // (1) pc = 4 * fetch_word (Val columns, exact: pc < 2^48).
    {
        let pc = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_pc, log_t)?;
        let fw = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_fetch_word, log_t)?;
        vp.add_term(*a, vec![pc, ei])
            .map_err(ConstraintError::Virtual)?;
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
                let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
                views.push((
                    fi,
                    FV::Limb {
                        slot: T_IMM,
                        limb: l,
                    },
                ));
                Ok(fi)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let rs1_l: Vec<usize> = (0..3)
            .map(|l| {
                let f = limb_mle(w, T_RS1, l, log_t);
                let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
                views.push((
                    fi,
                    FV::Limb {
                        slot: T_RS1,
                        limb: l,
                    },
                ));
                Ok(fi)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let b = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_branch"),
            log_t,
        )?;
        let t = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.taken, log_t)?;
        let j = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_jal"), log_t)?;
        let rr = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_jalr"),
            log_t,
        )?;
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
            vp.add_term(*a, vec![np_l[l], ei])
                .map_err(ConstraintError::Virtual)?;
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
            // -bt·(T_l - A_l) + j·(A_l - T_l) + r·(A_l - J_l) via the
            // per-limb recurrences; the in-carries are l>0 only (the
            // l=0 limb has no incoming carry — the pre-fix
            // `l - 1.min(l)` index leaked the limb-0 OUT carry into the
            // l=0 identity, which any taken branch with a target that
            // crosses the 16-bit boundary (negative offsets!) violated).
            // branches: weight = b·t
            {
                let w_ab = a.neg(); // -bt·T_l
                vp.add_term(w_ab, vec![b, t, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(w_ab, vec![b, t, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l > 0 {
                    vp.add_term(w_ab, vec![b, t, ct[l - 1], ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.mul(&fe(1 << 16)), vec![b, t, ct[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                // +bt·A_l
                vp.add_term(*a, vec![b, t, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l == 0 {
                    vp.add_term(a.mul(&fe(4)), vec![b, t, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                if l > 0 {
                    vp.add_term(a.neg(), vec![b, t, ca[l - 1], ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.mul(&fe(1 << 16).neg()), vec![b, t, ca[l], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // jal: weight = j
            {
                vp.add_term(a.neg(), vec![j, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![j, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l > 0 {
                    vp.add_term(a.neg(), vec![j, ct[l - 1], ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.mul(&fe(1 << 16)), vec![j, ct[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(*a, vec![j, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l == 0 {
                    vp.add_term(a.mul(&fe(4)), vec![j, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                if l > 0 {
                    vp.add_term(a.neg(), vec![j, ca[l - 1], ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.mul(&fe(1 << 16).neg()), vec![j, ca[l], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // jalr: weight = r (J_l - A_l)
            {
                vp.add_term(a.neg(), vec![rr, rs1_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![rr, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l > 0 {
                    vp.add_term(a.neg(), vec![rr, cj[l - 1], ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                vp.add_term(a.mul(&fe(1 << 16)), vec![rr, cj[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(*a, vec![rr, pc_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                if l == 0 {
                    vp.add_term(a.mul(&fe(4)), vec![rr, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                if l > 0 {
                    vp.add_term(a.neg(), vec![rr, ca[l - 1], ei])
                        .map_err(ConstraintError::Virtual)?;
                }
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
        vp.add_term(*b, vec![h, rw, ei])
            .map_err(ConstraintError::Virtual)?;
        vp.add_term(*b, vec![h, mw, ei])
            .map_err(ConstraintError::Virtual)?;
        vp.add_term(*b, vec![h, mr, ei])
            .map_err(ConstraintError::Virtual)?;
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
            let al = pc.add(&delta4).add(&ca_in).sub(&fe(1 << 16).mul(&ca_out));
            // T_l = pc + imm + c_in^T - 2^16 c_out^T
            let tl = pc.add(&imm).add(&ct_in).sub(&fe(1 << 16).mul(&ct_out));
            // J_l = rs1 + imm + c_in^J - 2^16 c_out^J
            let jl = rs1.add(&imm).add(&cj_in).sub(&fe(1 << 16).mul(&cj_out));
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
        .challenge_fields(b"con-route-a", 8)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
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
                let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
                views.push((
                    fi,
                    FV::Limb {
                        slot: T_RS1,
                        limb: l,
                    },
                ));
                Ok(fi)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let imm_l: Vec<usize> = (0..4)
            .map(|l| {
                let f = limb_mle(w, T_IMM, l, log_t);
                let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
                views.push((
                    fi,
                    FV::Limb {
                        slot: T_IMM,
                        limb: l,
                    },
                ));
                Ok(fi)
            })
            .collect::<Result<_, ConstraintError>>()?;
        let a = &alphas[0];
        let re = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_re, log_t)?;
        let we = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_we, log_t)?;
        for act in [re, we] {
            for l in 0..4 {
                let c_out =
                    add_bit_factor(&mut vp, &mut views, &aux.bits, idx.carry_jalr[l], log_t)?;
                vp.add_term(*a, vec![act, addr_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![act, rs1_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.neg(), vec![act, imm_l[l], ei])
                    .map_err(ConstraintError::Virtual)?;
                vp.add_term(a.mul(&fe(1 << 16)), vec![act, c_out, ei])
                    .map_err(ConstraintError::Virtual)?;
                if l > 0 {
                    let c_in = add_bit_factor(
                        &mut vp,
                        &mut views,
                        &aux.bits,
                        idx.carry_jalr[l - 1],
                        log_t,
                    )?;
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
            vp.add_term(*a, vec![sel, addr, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(8).neg()), vec![sel, word, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(4).neg()), vec![sel, half, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        for name in ["sel_ld", "sel_sd"] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            vp.add_term(*a, vec![sel, addr, ei])
                .map_err(ConstraintError::Virtual)?;
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
            vp.add_term(*a, vec![sd, newl, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![sd, rs2l, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*a, vec![ld, rdl, ei])
                .map_err(ConstraintError::Virtual)?;
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
            vp.add_term(*b, vec![sw, newl, ei])
                .map_err(ConstraintError::Virtual)?;
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
    // (8) THE P1 SUB-WORD ADDRESSING (the byte-guest ISA's routing):
    //     byte accesses: addr = 8*word + off0 + 2*off1 + 4*off2;
    //     half accesses: addr = 8*word + 2*off1 + 4*off2 AND off0 = 0
    //     (the alignment, fail-closed — a misaligned half is rejected).
    {
        let a = &alphas[5];
        let addr = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mem_addr, log_t)?;
        let word = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mem_word, log_t)?;
        let off: Vec<usize> = (0..3)
            .map(|i| add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_off[i], log_t))
            .collect::<Result<_, _>>()?;
        for name in ["sel_lb", "sel_lbu", "sel_sb"] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            vp.add_term(*a, vec![sel, addr, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(8).neg()), vec![sel, word, ei])
                .map_err(ConstraintError::Virtual)?;
            for (i, w2) in [(0usize, 1u64), (1, 2), (2, 4)] {
                vp.add_term(a.mul(&fe(w2).neg()), vec![sel, off[i], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
        for name in ["sel_lh", "sel_lhu", "sel_sh"] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            vp.add_term(*a, vec![sel, addr, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.mul(&fe(8).neg()), vec![sel, word, ei])
                .map_err(ConstraintError::Virtual)?;
            for (i, w2) in [(1usize, 2u64), (2, 4)] {
                vp.add_term(a.mul(&fe(w2).neg()), vec![sel, off[i], ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // sel*off0 = 0 — the half-alignment gate.
            vp.add_term(*a, vec![sel, off[0], ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (9) THE P1 SUB-WORD LOADS (bit-grain muxes over the off one-hot):
    //     LBU/LB: rd[b] = sum_p pos_p*old[8p+b] for b in 0..8, where
    //     pos_p = prod_i (off_i == p_i); LBU: rd[b] = 0 for b in 8..64;
    //     LB: rd[b] = sign = sum_p pos_p*old[8p+7] for b in 8..64.
    //     LHU/LH: rd[bit] = sum_h hpos_h*old[16h+bit] for bit in 0..16
    //     with hpos_h = (off1 == h1)*(off2 == h2); LHU zero / LH sign.
    {
        let a = &alphas[6];
        let off: Vec<usize> = (0..3)
            .map(|i| add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_off[i], log_t))
            .collect::<Result<_, _>>()?;
        // The complements (1 - off_i) — the position products' zero legs.
        let noff: Vec<usize> = (0..3)
            .map(|i| {
                let col = DenseMle {
                    num_vars: log_t,
                    evaluations: aux.bits[idx.mem_off[i]]
                        .iter()
                        .map(|v| fe(1 ^ (*v as u64)))
                        .collect(),
                };
                let fi = vp.add_factor(col).map_err(ConstraintError::Virtual)?;
                views.push((fi, FV::FlipBit(idx.mem_off[i])));
                Ok(fi)
            })
            .collect::<Result<Vec<usize>, ConstraintError>>()?;
        // The byte-position product factor list for position p.
        let byte_pos = |p: usize| -> Vec<usize> {
            (0..3)
                .map(|i| if (p >> i) & 1 == 1 { off[i] } else { noff[i] })
                .collect()
        };
        // The half-position product factor list for half h (bits 1..2).
        let half_pos = |h: usize| -> Vec<usize> {
            (0..2)
                .map(|i| if (h >> i) & 1 == 1 { off[i + 1] } else { noff[i + 1] })
                .collect()
        };
        for (name, signed) in [("sel_lbu", false), ("sel_lb", true)] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            for b in 0..8usize {
                let rd = add_vbit_factor(&mut vp, &mut views, w, T_RD, b, log_t)?;
                vp.add_term(*a, vec![sel, rd, ei])
                    .map_err(ConstraintError::Virtual)?;
                for p in 0..8usize {
                    let old = add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 8 * p + b, log_t)?;
                    let mut ids = vec![sel];
                    ids.extend(byte_pos(p));
                    ids.push(old);
                    ids.push(ei);
                    vp.add_term(a.neg(), ids).map_err(ConstraintError::Virtual)?;
                }
            }
            for b in 8..64usize {
                let rd = add_vbit_factor(&mut vp, &mut views, w, T_RD, b, log_t)?;
                vp.add_term(*a, vec![sel, rd, ei])
                    .map_err(ConstraintError::Virtual)?;
                if signed {
                    // LB: the sign fans from the muxed top bit (8p+7).
                    for p in 0..8usize {
                        let old =
                            add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 8 * p + 7, log_t)?;
                        let mut ids = vec![sel];
                        ids.extend(byte_pos(p));
                        ids.push(old);
                        ids.push(ei);
                        vp.add_term(a.neg(), ids).map_err(ConstraintError::Virtual)?;
                    }
                }
            }
        }
        for (name, signed) in [("sel_lhu", false), ("sel_lh", true)] {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            for b in 0..16usize {
                let rd = add_vbit_factor(&mut vp, &mut views, w, T_RD, b, log_t)?;
                vp.add_term(*a, vec![sel, rd, ei])
                    .map_err(ConstraintError::Virtual)?;
                for h in 0..4usize {
                    let old = add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 16 * h + b, log_t)?;
                    let mut ids = vec![sel];
                    ids.extend(half_pos(h));
                    ids.push(old);
                    ids.push(ei);
                    vp.add_term(a.neg(), ids).map_err(ConstraintError::Virtual)?;
                }
            }
            for b in 16..64usize {
                let rd = add_vbit_factor(&mut vp, &mut views, w, T_RD, b, log_t)?;
                vp.add_term(*a, vec![sel, rd, ei])
                    .map_err(ConstraintError::Virtual)?;
                if signed {
                    for h in 0..4usize {
                        let old = add_vbit_factor(
                            &mut vp,
                            &mut views,
                            w,
                            T_MEM_OLD,
                            16 * h + 15,
                            log_t,
                        )?;
                        let mut ids = vec![sel];
                        ids.extend(half_pos(h));
                        ids.push(old);
                        ids.push(ei);
                        vp.add_term(a.neg(), ids).map_err(ConstraintError::Virtual)?;
                    }
                }
            }
        }
    }
    // (10) THE P1 SUB-WORD STORE MERGES: SB replaces byte p of the word
    //      with rs2's low byte; SH replaces half h with rs2's low half:
    //      new[8p+b] = pos_p*rs2[b] + (1 - pos_p)*old[8p+b] (bit-grain).
    {
        let a = &alphas[7];
        let off: Vec<usize> = (0..3)
            .map(|i| add_bit_factor(&mut vp, &mut views, &aux.bits, idx.mem_off[i], log_t))
            .collect::<Result<_, _>>()?;
        let noff: Vec<usize> = (0..3)
            .map(|i| {
                let col = DenseMle {
                    num_vars: log_t,
                    evaluations: aux.bits[idx.mem_off[i]]
                        .iter()
                        .map(|v| fe(1 ^ (*v as u64)))
                        .collect(),
                };
                let fi = vp.add_factor(col).map_err(ConstraintError::Virtual)?;
                views.push((fi, FV::FlipBit(idx.mem_off[i])));
                Ok(fi)
            })
            .collect::<Result<Vec<usize>, ConstraintError>>()?;
        let byte_pos = |p: usize| -> Vec<usize> {
            (0..3)
                .map(|i| if (p >> i) & 1 == 1 { off[i] } else { noff[i] })
                .collect()
        };
        let half_pos = |h: usize| -> Vec<usize> {
            (0..2)
                .map(|i| if (h >> i) & 1 == 1 { off[i + 1] } else { noff[i + 1] })
                .collect()
        };
        {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sb"), log_t)?;
            for p in 0..8usize {
                for b in 0..8usize {
                    let new =
                        add_vbit_factor(&mut vp, &mut views, w, T_MEM_NEW, 8 * p + b, log_t)?;
                    let old =
                        add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 8 * p + b, log_t)?;
                    let rs2 = add_vbit_factor(&mut vp, &mut views, w, T_RS2, b, log_t)?;
                    // sel*(new - old - pos_p*(rs2 - old)) = 0
                    vp.add_term(*a, vec![sel, new, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a.neg(), vec![sel, old, ei])
                        .map_err(ConstraintError::Virtual)?;
                    let mut ids = vec![sel];
                    ids.extend(byte_pos(p));
                    ids.push(rs2);
                    ids.push(ei);
                    vp.add_term(a.neg(), ids).map_err(ConstraintError::Virtual)?;
                    let mut ids = vec![sel];
                    ids.extend(byte_pos(p));
                    ids.push(old);
                    ids.push(ei);
                    vp.add_term(*a, ids).map_err(ConstraintError::Virtual)?;
                }
            }
        }
        {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_sh"), log_t)?;
            for h in 0..4usize {
                for b in 0..16usize {
                    let new =
                        add_vbit_factor(&mut vp, &mut views, w, T_MEM_NEW, 16 * h + b, log_t)?;
                    let old =
                        add_vbit_factor(&mut vp, &mut views, w, T_MEM_OLD, 16 * h + b, log_t)?;
                    let rs2 = add_vbit_factor(&mut vp, &mut views, w, T_RS2, b, log_t)?;
                    vp.add_term(*a, vec![sel, new, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a.neg(), vec![sel, old, ei])
                        .map_err(ConstraintError::Virtual)?;
                    let mut ids = vec![sel];
                    ids.extend(half_pos(h));
                    ids.push(rs2);
                    ids.push(ei);
                    vp.add_term(a.neg(), ids).map_err(ConstraintError::Virtual)?;
                    let mut ids = vec![sel];
                    ids.extend(half_pos(h));
                    ids.push(old);
                    ids.push(ei);
                    vp.add_term(*a, ids).map_err(ConstraintError::Virtual)?;
                }
            }
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
        .challenge_fields(b"con-route-a", 8)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "route")?;
    // Degree cap 6: the P1 sub-word muxes carry 6-factor terms
    // (sel + 3 off-bit selectors + value-bit + eq).
    let verdict = verify_leg_header("route", leg, log_t, 6, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut expect = Goldilocks::ZERO;
    let one = Goldilocks::ONE;
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
            let e = addr.sub(&word.mul(&fe(8))).sub(&half.mul(&fe(4)));
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
            let e = newl.sub(&src).add(&h.mul(&src)).sub(&h.mul(&other));
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
            let e = rdl.sub(&oldl).add(&h.mul(&oldl)).sub(&h.mul(&oldh));
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
    // (8) THE P1 SUB-WORD ADDRESSING (mirror of the prover's (8)):
    //     byte: addr - 8*word - off0 - 2*off1 - 4*off2; half: the same
    //     without off0 plus the alignment sel*off0 = 0.
    {
        let a = &alphas[5];
        let addr = claim_val(ledger, idx.v_mem_addr, pt)?;
        let word = claim_val(ledger, idx.v_mem_word, pt)?;
        let off: Vec<Goldilocks> =
            (0..3).map(|i| claim_bit(ledger, idx.mem_off[i], pt)).collect::<Result<_, _>>()?;
        for name in ["sel_lb", "sel_lbu", "sel_sb"] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            let e = addr
                .sub(&word.mul(&fe(8)))
                .sub(&off[0])
                .sub(&off[1].mul(&fe(2)))
                .sub(&off[2].mul(&fe(4)));
            expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
        }
        for name in ["sel_lh", "sel_lhu", "sel_sh"] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            let e = addr
                .sub(&word.mul(&fe(8)))
                .sub(&off[1].mul(&fe(2)))
                .sub(&off[2].mul(&fe(4)));
            expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
            // the alignment gate: sel*off0 = 0
            expect = expect.add(&a.mul(&sel).mul(&off[0]).mul(&eq_at));
        }
    }
    // (9) THE P1 SUB-WORD LOADS (mirror): the bit-grain muxes evaluated
    //     at the leg point — pos_p is the product of (off_i or 1-off_i).
    {
        let a = &alphas[6];
        let off: Vec<Goldilocks> =
            (0..3).map(|i| claim_bit(ledger, idx.mem_off[i], pt)).collect::<Result<_, _>>()?;
        let noff: Vec<Goldilocks> = off.iter().map(|o| one.sub(o)).collect();
        let byte_pos = |p: usize| -> Goldilocks {
            (0..3)
                .map(|i| if (p >> i) & 1 == 1 { off[i] } else { noff[i] })
                .fold(one, |acc, v| acc.mul(&v))
        };
        let half_pos = |h: usize| -> Goldilocks {
            (0..2)
                .map(|i| if (h >> i) & 1 == 1 { off[i + 1] } else { noff[i + 1] })
                .fold(one, |acc, v| acc.mul(&v))
        };
        for (name, signed) in [("sel_lbu", false), ("sel_lb", true)] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            for b in 0..8usize {
                let rd = claim_vbit(ledger, T_RD, b, pt)?;
                let mut mux = Goldilocks::ZERO;
                for p in 0..8usize {
                    let old = claim_vbit(ledger, T_MEM_OLD, 8 * p + b, pt)?;
                    mux = mux.add(&byte_pos(p).mul(&old));
                }
                expect = expect.add(&a.mul(&sel).mul(&rd.sub(&mux)).mul(&eq_at));
            }
            for b in 8..64usize {
                let rd = claim_vbit(ledger, T_RD, b, pt)?;
                let mut target = Goldilocks::ZERO;
                if signed {
                    for p in 0..8usize {
                        let old = claim_vbit(ledger, T_MEM_OLD, 8 * p + 7, pt)?;
                        target = target.add(&byte_pos(p).mul(&old));
                    }
                }
                expect = expect.add(&a.mul(&sel).mul(&rd.sub(&target)).mul(&eq_at));
            }
        }
        for (name, signed) in [("sel_lhu", false), ("sel_lh", true)] {
            let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
            for b in 0..16usize {
                let rd = claim_vbit(ledger, T_RD, b, pt)?;
                let mut mux = Goldilocks::ZERO;
                for h in 0..4usize {
                    let old = claim_vbit(ledger, T_MEM_OLD, 16 * h + b, pt)?;
                    mux = mux.add(&half_pos(h).mul(&old));
                }
                expect = expect.add(&a.mul(&sel).mul(&rd.sub(&mux)).mul(&eq_at));
            }
            for b in 16..64usize {
                let rd = claim_vbit(ledger, T_RD, b, pt)?;
                let mut target = Goldilocks::ZERO;
                if signed {
                    for h in 0..4usize {
                        let old = claim_vbit(ledger, T_MEM_OLD, 16 * h + 15, pt)?;
                        target = target.add(&half_pos(h).mul(&old));
                    }
                }
                expect = expect.add(&a.mul(&sel).mul(&rd.sub(&target)).mul(&eq_at));
            }
        }
    }
    // (10) THE P1 SUB-WORD STORE MERGES (mirror): new = old + pos*(rs2 -
    //      old) at the byte/half grain, per bit.
    {
        let a = &alphas[7];
        let off: Vec<Goldilocks> =
            (0..3).map(|i| claim_bit(ledger, idx.mem_off[i], pt)).collect::<Result<_, _>>()?;
        let noff: Vec<Goldilocks> = off.iter().map(|o| one.sub(o)).collect();
        let byte_pos = |p: usize| -> Goldilocks {
            (0..3)
                .map(|i| if (p >> i) & 1 == 1 { off[i] } else { noff[i] })
                .fold(one, |acc, v| acc.mul(&v))
        };
        let half_pos = |h: usize| -> Goldilocks {
            (0..2)
                .map(|i| if (h >> i) & 1 == 1 { off[i + 1] } else { noff[i + 1] })
                .fold(one, |acc, v| acc.mul(&v))
        };
        {
            let sel = claim_bit(ledger, idx.sel_by("sel_sb"), pt)?;
            for p in 0..8usize {
                for b in 0..8usize {
                    let new = claim_vbit(ledger, T_MEM_NEW, 8 * p + b, pt)?;
                    let old = claim_vbit(ledger, T_MEM_OLD, 8 * p + b, pt)?;
                    let rs2 = claim_vbit(ledger, T_RS2, b, pt)?;
                    let e = new
                        .sub(&old)
                        .sub(&byte_pos(p).mul(&rs2.sub(&old)));
                    expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
                }
            }
        }
        {
            let sel = claim_bit(ledger, idx.sel_by("sel_sh"), pt)?;
            for h in 0..4usize {
                for b in 0..16usize {
                    let new = claim_vbit(ledger, T_MEM_NEW, 16 * h + b, pt)?;
                    let old = claim_vbit(ledger, T_MEM_OLD, 16 * h + b, pt)?;
                    let rs2 = claim_vbit(ledger, T_RS2, b, pt)?;
                    let e = new
                        .sub(&old)
                        .sub(&half_pos(h).mul(&rs2.sub(&old)));
                    expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
                }
            }
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
    let li = vp
        .add_factor(e_last.clone())
        .map_err(ConstraintError::Virtual)?;
    vp.add_term(Goldilocks::ONE, vec![hi, li])
        .map_err(ConstraintError::Virtual)?;
    let views: Vec<ViewPair> = vec![(hi, FV::Bit(aux.index.halted)), (li, FV::PubTable(e_last))];
    // Sum over the cube = h[T-1] (the indicator selects the last cycle).
    let claim = fe(aux.bits[aux.index.halted][(1 << log_t) - 1] as u64);
    ctx.stage("halt-end", &mut vp, &views, claim).map(|_| ())
}

fn verify_halt(
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
    let leg = next_constraint_leg(iter, "halt-end")?;
    let verdict = verify_leg_header("halt-end", leg, log_t, 2, transcript)?;
    let h = ledger.tensor_claim(Factor::BitCol { id: idx.halted }, &verdict.point)?;
    let ones = vec![Goldilocks::ONE; log_t];
    let e_last = DenseMle::eq_extension(&ones);
    let e_at = e_last
        .evaluate(&verdict.point)
        .map_err(ConstraintError::Mle)?;
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
// Family: range links — every mul/div limb and carry column is composed
// from boolean bit columns (the integer-semantics anchor: without the
// range, the limb recurrences would only constrain field combinations).
// ---------------------------------------------------------------------------

/// One range-decomposed group: (base offset, the limb columns, bits per
/// limb, limb count).
fn range_groups(aux: &AuxCols) -> Vec<(usize, &[usize], usize, usize)> {
    let i = &aux.index;
    vec![
        (RG_MUL_LO, &i.v_mul_lo, 16, 4),
        (RG_MUL_HI, &i.v_mul_hi, 16, 4),
        (RG_MUL_C, &i.v_mul_c, 19, 7),
        (RG_MULH_BOR, &i.v_mulh_bor, 2, 4),
        (RG_DIV_Q, &i.v_div_q, 16, 4),
        (RG_DIV_R, &i.v_div_r, 16, 4),
        (RG_DIV_D, &i.v_div_d, 19, 4),
        (RG_MAG_A, &i.v_mag_a, 16, 4),
        (RG_MAG_B, &i.v_mag_b, 16, 4),
        (RG_MAG_Q, &i.v_mag_q, 16, 4),
        (RG_MAG_R, &i.v_mag_r, 16, 4),
        (RG_MAG_D, &i.v_mag_d, 19, 4),
        (RG_RLT_OUT, &i.v_rlt_out, 16, 4),
    ]
}

fn prove_range_links(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let aux = ctx.aux;
    let log_t = ctx.w.log_t;
    let r = ctx
        .transcript
        .challenge_fields(b"con-rl-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let groups = range_groups(aux);
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-rl-a", groups.len())
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    for (g, (base, cols, width, n)) in groups.iter().enumerate() {
        for l in 0..*n {
            let limb = add_val_factor(&mut vp, &mut views, &aux.vals, cols[l], log_t)?;
            vp.add_term(alphas[g], vec![limb, ei])
                .map_err(ConstraintError::Virtual)?;
            for j in 0..*width {
                let bit = aux.index.range_bits[base + l * width + j];
                let bf = add_bit_factor(&mut vp, &mut views, &aux.bits, bit, log_t)?;
                vp.add_term(alphas[g].mul(&fe(1u64 << j).neg()), vec![bf, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    ctx.stage("rangelinks", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_range_links(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let groups = range_groups(aux);
    let r = transcript
        .challenge_fields(b"con-rl-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-rl-a", groups.len())
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "rangelinks")?;
    let verdict = verify_leg_header("rangelinks", leg, log_t, 2, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut expect = Goldilocks::ZERO;
    for (g, (base, cols, width, n)) in groups.iter().enumerate() {
        for l in 0..*n {
            let limb = claim_val(ledger, cols[l], pt)?;
            let mut e = limb;
            for j in 0..*width {
                let bit = aux.index.range_bits[base + l * width + j];
                let b = claim_bit(ledger, bit, pt)?;
                e = e.sub(&fe(1u64 << j).mul(&b));
            }
            expect = expect.add(&alphas[g].mul(&e).mul(&eq_at));
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("rangelinks"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: decode — the instruction tensor is bound to the fetched word
// (pc = 4·fetch already lives in ctrl; here the TENSOR is bound to the
// fetch column), and the coverage partitions force every fetched word
// into exactly one covered class/sub-class (the verifier-side coverage
// gate: the kernel tests' `instrs` parameter is NOT available to the
// pipeline verifier, so the decode itself must carry the gate).
// ---------------------------------------------------------------------------

/// The 12 opcode classes.
const CLASS_SELECTORS: [&str; 12] = [
    "sel_opimm",
    "sel_op",
    "sel_op32",
    "sel_opimm32",
    "sel_lui",
    "sel_auipc",
    "sel_jal",
    "sel_jalr",
    "sel_branch",
    "sel_load",
    "sel_store",
    "sel_system",
];

/// The sub-class partition of each multi-member class.
const SUBCLASS_PARTITIONS: [(&str, &[&str]); 7] = [
    (
        "sel_op",
        &[
            "sel_add",
            "sel_sub",
            "sel_xor",
            "sel_or",
            "sel_and",
            "sel_sll",
            "sel_srl",
            "sel_sra",
            "sel_slt",
            "sel_sltu",
            "sel_mul",
            "sel_mulh",
            "sel_mulhu",
            "sel_div",
            "sel_divu",
            "sel_rem",
            "sel_remu",
        ],
    ),
    (
        "sel_opimm",
        &[
            "sel_addi",
            "sel_slti",
            "sel_sltiu",
            "sel_xori",
            "sel_ori",
            "sel_andi",
            "sel_slli",
            "sel_srli",
            "sel_srai",
        ],
    ),
    (
        "sel_op32",
        &[
            "sel_addw",
            "sel_subw",
            "sel_sllw",
            "sel_srlw",
            "sel_sraw",
            "sel_mulw",
            "sel_divw",
            "sel_divuw",
            "sel_remw",
            "sel_remuw",
        ],
    ),
    (
        "sel_opimm32",
        &["sel_addiw", "sel_slliw", "sel_srliw", "sel_sraiw"],
    ),
    (
        "sel_branch",
        &[
            "sel_beq", "sel_bne", "sel_blt", "sel_bge", "sel_bltu", "sel_bgeu",
        ],
    ),
    (
        "sel_load",
        &["sel_lw", "sel_lwu", "sel_ld", "sel_lb", "sel_lh", "sel_lbu", "sel_lhu"],
    ),
    (
        "sel_store",
        &["sel_sw", "sel_sd", "sel_sb", "sel_sh"],
    ),
];

fn prove_decode(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-dec-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-dec-a", 4)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // (1) the fetch binding: sum_i 2^i * instr_bit_i = fetch_word.
    {
        let a = &alphas[0];
        for i in 0..32usize {
            let row = 31 - i;
            let f = instr_row_of(&w.instr_bits, row, log_t);
            let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
            views.push((
                fi,
                FV::TensorRow {
                    factor: Factor::InstrBits,
                    nbits: 32,
                    row,
                },
            ));
            vp.add_term(a.mul(&fe(1u64 << i)), vec![fi, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        let iw = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_instr, log_t)?;
        vp.add_term(a.neg(), vec![iw, ei])
            .map_err(ConstraintError::Virtual)?;
    }
    // (2) the class partition: sum(classes) - 1 = 0 (the constant rides
    //     the eq table whose cube-sum is 1).
    {
        let a = &alphas[1];
        for name in CLASS_SELECTORS {
            let s = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
            vp.add_term(*a, vec![s, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        vp.add_term(a.neg(), vec![ei])
            .map_err(ConstraintError::Virtual)?;
    }
    // (3) the sub-class partitions: sum(subs) - class = 0.
    {
        let a = &alphas[2];
        for (class, subs) in SUBCLASS_PARTITIONS {
            let cf = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(class), log_t)?;
            vp.add_term(a.neg(), vec![cf, ei])
                .map_err(ConstraintError::Virtual)?;
            for name in subs {
                let s = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
                vp.add_term(*a, vec![s, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (4) the system discipline: sel_system forces f3 = 0 and
    //     funct12 in {0, 1} (ECALL/EBREAK; CSR space excluded).
    {
        let a = &alphas[3];
        let sys = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_system"),
            log_t,
        )?;
        for bit in [12usize, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] {
            let row = 31 - bit;
            let f = instr_row_of(&w.instr_bits, row, log_t);
            let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
            views.push((
                fi,
                FV::TensorRow {
                    factor: Factor::InstrBits,
                    nbits: 32,
                    row,
                },
            ));
            vp.add_term(*a, vec![sys, fi, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    ctx.stage("decode", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_decode(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let r = transcript
        .challenge_fields(b"con-dec-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-dec-a", 4)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "decode")?;
    let verdict = verify_leg_header("decode", leg, log_t, 3, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut expect = Goldilocks::ZERO;
    // (1) fetch binding (the row claims resolve BEFORE the instr
    // column — the prover's view order).
    {
        let mut e = Goldilocks::ZERO;
        for i in 0..32usize {
            let row = 31 - i;
            let mut p = idx_point(5, row);
            p.extend_from_slice(pt);
            let b = ledger.tensor_claim(Factor::InstrBits, &p)?;
            e = e.add(&fe(1u64 << i).mul(&b));
        }
        let iw = claim_val(ledger, idx.v_instr, pt)?;
        e = e.sub(&iw);
        expect = expect.add(&alphas[0].mul(&e).mul(&eq_at));
    }
    // (2) class partition.
    {
        let mut s = Goldilocks::ZERO;
        for name in CLASS_SELECTORS {
            s = s.add(&claim_bit(ledger, idx.sel_by(name), pt)?);
        }
        expect = expect.add(&alphas[1].mul(&s.sub(&Goldilocks::ONE)).mul(&eq_at));
    }
    // (3) sub-class partitions.
    {
        for (class, subs) in SUBCLASS_PARTITIONS {
            let c = claim_bit(ledger, idx.sel_by(class), pt)?;
            let mut s = Goldilocks::ZERO;
            for name in subs {
                s = s.add(&claim_bit(ledger, idx.sel_by(name), pt)?);
            }
            expect = expect.add(&alphas[2].mul(&s.sub(&c)).mul(&eq_at));
        }
    }
    // (4) system discipline.
    {
        let sys = claim_bit(ledger, idx.sel_by("sel_system"), pt)?;
        let mut e = Goldilocks::ZERO;
        for bit in [12usize, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] {
            let row = 31 - bit;
            let mut p = idx_point(5, row);
            p.extend_from_slice(pt);
            let b = ledger.tensor_claim(Factor::InstrBits, &p)?;
            e = e.add(&b);
        }
        expect = expect.add(&alphas[3].mul(&sys).mul(&e).mul(&eq_at));
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("decode"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: shifts — the shamt one-hots (register 6-bit / W 5-bit, register
// vs immediate sources) and the per-bit shift MUXes.
// ---------------------------------------------------------------------------

/// The shift classes: (selector, one-hot group, kind, is_w).
/// Kind: 0 = left, 1 = right-logical, 2 = right-arithmetic.
#[derive(Clone, Copy)]
struct ShiftClass {
    sel: &'static str,
    /// 0: shoh6_r, 1: shoh6_i, 2: shoh5_r, 3: shoh5_i.
    oh: usize,
    kind: u8,
    is_w: bool,
}

const SHIFT_CLASSES: [ShiftClass; 12] = [
    ShiftClass {
        sel: "sel_sll",
        oh: 0,
        kind: 0,
        is_w: false,
    },
    ShiftClass {
        sel: "sel_srl",
        oh: 0,
        kind: 1,
        is_w: false,
    },
    ShiftClass {
        sel: "sel_sra",
        oh: 0,
        kind: 2,
        is_w: false,
    },
    ShiftClass {
        sel: "sel_slli",
        oh: 1,
        kind: 0,
        is_w: false,
    },
    ShiftClass {
        sel: "sel_srli",
        oh: 1,
        kind: 1,
        is_w: false,
    },
    ShiftClass {
        sel: "sel_srai",
        oh: 1,
        kind: 2,
        is_w: false,
    },
    ShiftClass {
        sel: "sel_sllw",
        oh: 2,
        kind: 0,
        is_w: true,
    },
    ShiftClass {
        sel: "sel_srlw",
        oh: 2,
        kind: 1,
        is_w: true,
    },
    ShiftClass {
        sel: "sel_sraw",
        oh: 2,
        kind: 2,
        is_w: true,
    },
    ShiftClass {
        sel: "sel_slliw",
        oh: 3,
        kind: 0,
        is_w: true,
    },
    ShiftClass {
        sel: "sel_srliw",
        oh: 3,
        kind: 1,
        is_w: true,
    },
    ShiftClass {
        sel: "sel_sraiw",
        oh: 3,
        kind: 2,
        is_w: true,
    },
];

fn prove_shift(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    prove_shift_sparse(ctx)
}

/// The pre-sparse reference path (the dense-engine construction) —
/// retained for the byte-identity differential test against the sparse
/// route (the round polynomials must be identical products, so the
/// transcripts and proofs must match exactly).
#[cfg(test)]
fn prove_shift_dense_cfg_test(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-sh-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-sh-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // Memoized tensor-row factors.
    let mut rs1_rows: Vec<Option<usize>> = vec![None; 64];
    let mut rs2_rows: Vec<Option<usize>> = vec![None; 64];
    let mut instr_rows: Vec<Option<usize>> = vec![None; 32];
    let mut add_rs1_row = |vp: &mut VirtualPolynomial,
                           views: &mut Vec<ViewPair>,
                           bit: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = rs1_rows[bit] {
            return Ok(f);
        }
        let row = 63 - bit;
        let f = row_mle(&w.values[T_RS1], row, log_t);
        let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
        views.push((
            fi,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RS1 },
                nbits: 64,
                row,
            },
        ));
        rs1_rows[bit] = Some(fi);
        Ok(fi)
    };
    let mut add_rs2_row = |vp: &mut VirtualPolynomial,
                           views: &mut Vec<ViewPair>,
                           bit: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = rs2_rows[bit] {
            return Ok(f);
        }
        let row = 63 - bit;
        let f = row_mle(&w.values[T_RS2], row, log_t);
        let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
        views.push((
            fi,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RS2 },
                nbits: 64,
                row,
            },
        ));
        rs2_rows[bit] = Some(fi);
        Ok(fi)
    };
    let mut add_instr_row = |vp: &mut VirtualPolynomial,
                             views: &mut Vec<ViewPair>,
                             bit: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = instr_rows[bit] {
            return Ok(f);
        }
        let row = 31 - bit;
        let f = instr_row_of(&w.instr_bits, row, log_t);
        let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
        views.push((
            fi,
            FV::TensorRow {
                factor: Factor::InstrBits,
                nbits: 32,
                row,
            },
        ));
        instr_rows[bit] = Some(fi);
        Ok(fi)
    };
    let add_rd_row = |vp: &mut VirtualPolynomial,
                      views: &mut Vec<ViewPair>,
                      bit: usize|
     -> Result<usize, ConstraintError> {
        let row = 63 - bit;
        let f = row_mle(&w.values[T_RD], row, log_t);
        let fi = vp.add_factor(f).map_err(ConstraintError::Virtual)?;
        views.push((
            fi,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RD },
                nbits: 64,
                row,
            },
        ));
        Ok(fi)
    };
    // (1) the one-hot decodes: shoh[s] = prod over the source bits
    //     (polarized). The sources: rs2's low bits (register forms) or
    //     the instruction's shamt field (immediate forms).
    {
        let a = &alphas[0];
        let groups: [(&[usize], usize, usize); 4] = [
            (&idx.shoh6_r, 6, 0), // (columns, nbits, source: rs2)
            (&idx.shoh6_i, 6, 1), // instr bits 20..25
            (&idx.shoh5_r, 5, 0),
            (&idx.shoh5_i, 5, 1),
        ];
        for (ohs, nbits, src) in groups {
            for s in 0..(1usize << nbits) {
                let oh = add_bit_factor(&mut vp, &mut views, &aux.bits, ohs[s], log_t)?;
                vp.add_term(*a, vec![oh, ei])
                    .map_err(ConstraintError::Virtual)?;
                // -prod(polarized source bits)
                let mut ids = Vec::with_capacity(nbits + 1);
                for b in 0..nbits {
                    let req = (s >> b) & 1;
                    let fi = if src == 0 {
                        add_rs2_row(&mut vp, &mut views, b)?
                    } else {
                        add_instr_row(&mut vp, &mut views, 20 + b)?
                    };
                    if req == 1 {
                        ids.push(fi);
                    } else {
                        // flipped polarity: a fresh factor over the same
                        // row (the view stays the unflipped row; the
                        // verifier derives the flip).
                        let row = if src == 0 { 63 - b } else { 31 - (20 + b) };
                        let f = if src == 0 {
                            row_mle(&w.values[T_RS2], row, log_t)
                        } else {
                            instr_row_of(&w.instr_bits, row, log_t)
                        };
                        let flipped = flip_mle(&f);
                        let ffi = vp.add_factor(flipped).map_err(ConstraintError::Virtual)?;
                        ids.push(ffi);
                    }
                }
                ids.push(ei);
                vp.add_term(a.neg(), ids)
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (2) the per-bit shift MUXes.
    {
        let a = &alphas[1];
        let b2 = &alphas[2];
        for class in SHIFT_CLASSES {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(class.sel), log_t)?;
            let ohs: &[usize] = match class.oh {
                0 => &idx.shoh6_r,
                1 => &idx.shoh6_i,
                2 => &idx.shoh5_r,
                _ => &idx.shoh5_i,
            };
            let width = if class.oh < 2 { 64usize } else { 32usize };
            let top = if class.is_w { 32usize } else { 64usize };
            for i in 0..64usize {
                let rd = add_rd_row(&mut vp, &mut views, i)?;
                if i >= top {
                    // The W sign extension: rd_bit[i] = rd_bit[31].
                    if class.is_w {
                        let rd31 = add_rd_row(&mut vp, &mut views, 31)?;
                        vp.add_term(*b2, vec![sel, rd, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(b2.neg(), vec![sel, rd31, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                    continue;
                }
                // The in-range source bit and the shift bound.
                let (lo_s, hi_s): (usize, usize) = match class.kind {
                    0 => (0, i.min(width - 1)),             // s <= i
                    _ => (0, (top - 1 - i).min(width - 1)), // i + s <= top-1
                };
                let alpha = if class.is_w { *b2 } else { *a };
                vp.add_term(alpha, vec![sel, rd, ei])
                    .map_err(ConstraintError::Virtual)?;
                let mut covered = Vec::with_capacity(hi_s + 1);
                for s in lo_s..=hi_s {
                    let src_bit = match class.kind {
                        0 => i - s, // SLL: rs1_bit[i - s]
                        _ => i + s, // SRL/SRA: rs1_bit[i + s]
                    };
                    let oh = add_bit_factor(&mut vp, &mut views, &aux.bits, ohs[s], log_t)?;
                    let sb = add_rs1_row(&mut vp, &mut views, src_bit)?;
                    vp.add_term(alpha.neg(), vec![sel, oh, sb, ei])
                        .map_err(ConstraintError::Virtual)?;
                    covered.push(s);
                }
                // The arithmetic fill: out-of-range s contribute the
                // sign bit (bit top-1) via the one-hot complement. The
                // identity is rd - (main + sg·(1 - covered)) = 0, so the
                // fill's constant term is SUBTRACTED and the covered
                // correction added.
                if class.kind == 2 {
                    let sign_bit = top - 1;
                    let sg = add_rs1_row(&mut vp, &mut views, sign_bit)?;
                    vp.add_term(alpha.neg(), vec![sel, sg, ei])
                        .map_err(ConstraintError::Virtual)?;
                    for s in covered {
                        let oh = add_bit_factor(&mut vp, &mut views, &aux.bits, ohs[s], log_t)?;
                        vp.add_term(alpha, vec![sel, sg, oh, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
        }
    }
    ctx.stage("shift", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

/// The sparse-engine route for the shift family (the "0s are free"
/// doctrine, `lattice-memory::sparse_engine`, applied to the constraint
/// layer): the per-(bit, shamt) MUX products — the O(64^2)-per-class
/// term expansion that dominated the semantics stage's prover time —
/// are gated by the selector and one-hot columns, which are nonzero on
/// a vanishing fraction of rows. The sparse sumcheck evaluates each
/// term only over its true support (the selector/one-hot intersection)
/// instead of the full cycle cube, emitting the byte-identical proof
/// the dense engine would produce over the same virtual polynomial
/// (same round polynomials — the products are the same — pinned by the
/// differential test below); the verifier is UNCHANGED.
fn prove_shift_sparse(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    use lattice_memory::sparse_engine::{
        prove_sparse_sumcheck, DenseFactor, SparseFactor, SparseInstance, SparseTerm,
    };
    use std::collections::HashMap;

    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-sh-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-sh-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let t_len = 1u64 << log_t;

    // ---- registries ----
    // Dense pool: index 0 is eq (the PubTable view); then the memoized
    // tensor-row MLEs and their flipped variants (identity var_map —
    // every factor spans the full log_t variables).
    let var_map: Vec<usize> = (0..log_t).collect();
    let mut dense: Vec<DenseFactor> = vec![DenseFactor::Table(eq.clone(), var_map.clone())];
    let mut views: Vec<(usize, FV)> = vec![(0usize, FV::PubTable(eq.clone()))];
    let mut rs1_memo: HashMap<usize, usize> = HashMap::new();
    let mut rs2_memo: HashMap<usize, usize> = HashMap::new();
    let mut instr_memo: HashMap<usize, usize> = HashMap::new();
    let mut rd_memo: HashMap<usize, usize> = HashMap::new();
    let mut flip_memo: HashMap<(u8, usize), usize> = HashMap::new(); // (which, bit)

    macro_rules! add_row {
        ($tensor:expr, $nbits:expr, $bit:expr, $memo:expr, $view:expr) => {{
            let bit: usize = $bit;
            if let Some(&di) = $memo.get(&bit) {
                di
            } else {
                let row = $nbits - 1 - bit;
                let f = row_mle(&$tensor, row, log_t);
                let di = dense.len();
                dense.push(DenseFactor::Table(f, var_map.clone()));
                views.push((di, $view(row)));
                $memo.insert(bit, di);
                di
            }
        }};
    }
    let mut sparse: Vec<SparseFactor> = Vec::new();
    let mut terms: Vec<SparseTerm> = Vec::new();
    // The full-cube position list shared by every all-dense term (the
    // decode section's polarized products).
    let full_cube: Vec<u64> = (0..t_len).collect();

    // A bit column's nonzero support (row, value), ascending.
    let col_support = |id: usize| -> Vec<(u64, Goldilocks)> {
        aux.bits[id]
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(row, v)| (row as u64, fe(*v as u64)))
            .collect()
    };

    // ---- (1) the one-hot decodes ----
    {
        let a = &alphas[0];
        let groups: [(&[usize], usize, usize); 4] = [
            (&idx.shoh6_r, 6, 0),
            (&idx.shoh6_i, 6, 1),
            (&idx.shoh5_r, 5, 0),
            (&idx.shoh5_i, 5, 1),
        ];
        for (ohs, nbits, src) in groups {
            for s in 0..(1usize << nbits) {
                // Term A: +alpha · oh[s] · eq — the one-hot's full
                // support as the single sparse factor.
                let support = col_support(ohs[s]);
                let fi = sparse.len();
                sparse.push(SparseFactor {
                    entries: support.clone(),
                    var_map: var_map.clone(),
                });
                views.push((usize::MAX, FV::Bit(ohs[s])));
                // (usize::MAX marks a view whose claim comes from the
                // sparse pool; bind_views ignores indices absent from
                // the (empty) factor-claims slice, and resolve_view
                // derives the value from the ledger's own column.)
                terms.push(SparseTerm {
                    coeff: *a,
                    positions: support.iter().map(|e| e.0).collect(),
                    sparse: vec![fi],
                    dense: vec![0],
                });
                // Term B: -alpha · prod(polarized source bits) · eq —
                // all-dense over the full cube.
                let mut dense_ids = Vec::with_capacity(nbits + 1);
                for b in 0..nbits {
                    let req = (s >> b) & 1;
                    if req == 1 {
                        let di = if src == 0 {
                            add_row!(w.values[T_RS2], 64, b, rs2_memo, |row: usize| {
                                FV::TensorRow {
                                    factor: Factor::ValueBits { slot: T_RS2 },
                                    nbits: 64,
                                    row,
                                }
                            })
                        } else {
                            add_row!(w.instr_bits, 32, 20 + b, instr_memo, |row: usize| {
                                FV::TensorRow {
                                    factor: Factor::InstrBits,
                                    nbits: 32,
                                    row,
                                }
                            })
                        };
                        dense_ids.push(di);
                    } else {
                        // flipped polarity — a dense factor, no view
                        // (the verifier derives the flip).
                        let key = (src as u8, b);
                        let di = if let Some(&d) = flip_memo.get(&key) {
                            d
                        } else {
                            let row = if src == 0 { 63 - b } else { 31 - (20 + b) };
                            let f = if src == 0 {
                                row_mle(&w.values[T_RS2], row, log_t)
                            } else {
                                instr_row_of(&w.instr_bits, row, log_t)
                            };
                            let d = dense.len();
                            dense.push(DenseFactor::Table(flip_mle(&f), var_map.clone()));
                            flip_memo.insert(key, d);
                            d
                        };
                        dense_ids.push(di);
                    }
                }
                dense_ids.push(0);
                terms.push(SparseTerm {
                    coeff: a.neg(),
                    positions: full_cube.clone(),
                    sparse: vec![],
                    dense: dense_ids,
                });
            }
        }
    }

    // ---- (2) the per-bit shift MUXes ----
    //
    // CORRECTNESS DISCIPLINE (pinned by the engine-level differential
    // test in `sparse_engine.rs::identity_differential`): a term may
    // carry AT MOST ONE sparse factor, whose entries are that factor's
    // OWN full nonzero support. Two sparse factors filtered to their
    // boolean intersection is WRONG: the multilinear products have
    // suffix-level cross terms outside the boolean intersection (the
    // round polynomials at t >= 2 sample the extensions), so the
    // intersection drops nonzero contributions. Selectors therefore
    // ride as DENSE factors in the one-hot-gated terms.
    {
        let a = &alphas[1];
        let b2 = &alphas[2];
        let mut sel_dense_memo: HashMap<&'static str, usize> = HashMap::new();
        let mut oh_support_cache: HashMap<usize, Vec<(u64, Goldilocks)>> = HashMap::new();
        for class in SHIFT_CLASSES {
            let sel_support = col_support(idx.sel_by(class.sel));
            let sel_fi_base = sparse.len();
            sparse.push(SparseFactor {
                entries: sel_support.clone(),
                var_map: var_map.clone(),
            });
            views.push((usize::MAX, FV::Bit(idx.sel_by(class.sel))));
            // The selector as a DENSE factor (for the one-hot-gated terms).
            let sel_di = match sel_dense_memo.get(class.sel) {
                Some(&d) => d,
                None => {
                    let col: Vec<Goldilocks> = aux.bits[idx.sel_by(class.sel)]
                        .iter()
                        .map(|v| fe(*v as u64))
                        .collect();
                    let d = dense.len();
                    dense.push(DenseFactor::Table(DenseMle::new(col) .map_err(|e| ConstraintError::Sparse(format!("{e:?}")))?, var_map.clone()));
                    sel_dense_memo.insert(class.sel, d);
                    d
                }
            };
            let ohs: &[usize] = match class.oh {
                0 => &idx.shoh6_r,
                1 => &idx.shoh6_i,
                2 => &idx.shoh5_r,
                _ => &idx.shoh5_i,
            };
            let width = if class.oh < 2 { 64usize } else { 32usize };
            let top = if class.is_w { 32usize } else { 64usize };
            for i in 0..64usize {
                // rd's row (memoized dense + view).
                let rd_di = add_row!(w.values[T_RD], 64, i, rd_memo, |row: usize| FV::TensorRow {
                    factor: Factor::ValueBits { slot: T_RD },
                    nbits: 64,
                    row,
                });
                if i >= top {
                    // The W sign extension: rd_bit[i] = rd_bit[31].
                    if class.is_w {
                        let rd31 = add_row!(w.values[T_RD], 64, 31, rd_memo, |row: usize| {
                            FV::TensorRow {
                                factor: Factor::ValueBits { slot: T_RD },
                                nbits: 64,
                                row,
                            }
                        });
                        terms.push(SparseTerm {
                            coeff: *b2,
                            positions: sel_support.iter().map(|e| e.0).collect(),
                            sparse: vec![sel_fi_base],
                            dense: vec![rd_di, 0],
                        });
                        terms.push(SparseTerm {
                            coeff: b2.neg(),
                            positions: sel_support.iter().map(|e| e.0).collect(),
                            sparse: vec![sel_fi_base],
                            dense: vec![rd31, 0],
                        });
                    }
                    continue;
                }
                let (lo_s, hi_s): (usize, usize) = match class.kind {
                    0 => (0, i.min(width - 1)),
                    _ => (0, (top - 1 - i).min(width - 1)),
                };
                let alpha = if class.is_w { *b2 } else { *a };
                terms.push(SparseTerm {
                    coeff: alpha,
                    positions: sel_support.iter().map(|e| e.0).collect(),
                    sparse: vec![sel_fi_base],
                    dense: vec![rd_di, 0],
                });
                let mut covered = Vec::with_capacity(hi_s + 1);
                for s in lo_s..=hi_s {
                    let src_bit = match class.kind {
                        0 => i - s,
                        _ => i + s,
                    };
                    // The one-hot's OWN support (single sparse factor —
                    // the correctness discipline above).
                    let oh_entries = match oh_support_cache.get(&ohs[s]) {
                        Some(e) => e.clone(),
                        None => {
                            let e = col_support(ohs[s]);
                            oh_support_cache.insert(ohs[s], e.clone());
                            e
                        }
                    };
                    let sb_di = add_row!(w.values[T_RS1], 64, src_bit, rs1_memo, |row: usize| {
                        FV::TensorRow {
                            factor: Factor::ValueBits { slot: T_RS1 },
                            nbits: 64,
                            row,
                        }
                    });
                    let oh_fi = sparse.len();
                    sparse.push(SparseFactor {
                        entries: oh_entries.clone(),
                        var_map: var_map.clone(),
                    });
                    terms.push(SparseTerm {
                        coeff: alpha.neg(),
                        positions: oh_entries.iter().map(|e| e.0).collect(),
                        sparse: vec![oh_fi],
                        dense: vec![sel_di, sb_di, 0],
                    });
                    covered.push(s);
                }
                // The arithmetic fill (SRA): out-of-range s contribute
                // the sign bit via the one-hot complement.
                if class.kind == 2 {
                    let sign_bit = top - 1;
                    let sg_di = add_row!(w.values[T_RS1], 64, sign_bit, rs1_memo, |row: usize| {
                        FV::TensorRow {
                            factor: Factor::ValueBits { slot: T_RS1 },
                            nbits: 64,
                            row,
                        }
                    });
                    terms.push(SparseTerm {
                        coeff: alpha.neg(),
                        positions: sel_support.iter().map(|e| e.0).collect(),
                        sparse: vec![sel_fi_base],
                        dense: vec![sg_di, 0],
                    });
                    for s in covered {
                        let oh_entries = match oh_support_cache.get(&ohs[s]) {
                            Some(e) => e.clone(),
                            None => {
                                let e = col_support(ohs[s]);
                                oh_support_cache.insert(ohs[s], e.clone());
                                e
                            }
                        };
                        let oh_fi = sparse.len();
                        sparse.push(SparseFactor {
                            entries: oh_entries.clone(),
                            var_map: var_map.clone(),
                        });
                        terms.push(SparseTerm {
                            coeff: alpha,
                            positions: oh_entries.iter().map(|e| e.0).collect(),
                            sparse: vec![oh_fi],
                            dense: vec![sel_di, sg_di, 0],
                        });
                    }
                }
            }
        }
    }
    // ---- run the sparse sumcheck (byte-identical rounds) ----
    let inst = SparseInstance {
        num_vars: log_t,
        sparse,
        dense,
        terms,
    };
    absorb_leg(ctx.transcript, "shift")?;
    let out = prove_sparse_sumcheck(&inst, Goldilocks::ZERO, ctx.transcript)
        .map_err(|e| ConstraintError::Sparse(format!("{e:?}")))?;
    // Record the view claims through the ledger's own resolution (the
    // verifier pops the identical keys); the engine's internal factor
    // claims differ for the filtered sparse copies by design, so the
    // dense-engine cross-check is not applicable here — the engine's
    // own final-claim guard plus the family's end-to-end verification
    // carry the correctness weight.
    bind_views(ctx.ledger, &views, &out.challenges, &[])?;
    ctx.legs.push(ConstraintLeg {
        name: "shift",
        sc: out.proof,
        claim: Goldilocks::ZERO,
    });
    Ok(())
}

fn verify_shift(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let r = transcript
        .challenge_fields(b"con-sh-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-sh-a", 3)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "shift")?;
    let verdict = verify_leg_header("shift", leg, log_t, 8, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    // LAZY claim caches: the prover claims exactly the rows its leg
    // references (the memoized factor pool), so the verifier must pop
    // in the same on-demand pattern — resolving every row upfront would
    // desync the claim queue.
    let mut rs1_memo: std::collections::HashMap<usize, Goldilocks> =
        std::collections::HashMap::new();
    let mut rs2_memo: std::collections::HashMap<usize, Goldilocks> =
        std::collections::HashMap::new();
    let mut instr_memo: std::collections::HashMap<usize, Goldilocks> =
        std::collections::HashMap::new();
    let mut rd_memo: std::collections::HashMap<usize, Goldilocks> =
        std::collections::HashMap::new();
    let rs1_bits = |ledger: &mut Ledger<'_>,
                    memo: &mut std::collections::HashMap<usize, Goldilocks>,
                    bit: usize|
     -> Result<Goldilocks, ConstraintError> {
        if let Some(v) = memo.get(&bit) {
            return Ok(*v);
        }
        let mut p = idx_point(6, 63 - bit);
        p.extend_from_slice(pt);
        let v = ledger.tensor_claim(Factor::ValueBits { slot: T_RS1 }, &p)?;
        memo.insert(bit, v);
        Ok(v)
    };
    let _ = &rs1_bits;
    let rs2_bits = |ledger: &mut Ledger<'_>,
                    memo: &mut std::collections::HashMap<usize, Goldilocks>,
                    bit: usize|
     -> Result<Goldilocks, ConstraintError> {
        if let Some(v) = memo.get(&bit) {
            return Ok(*v);
        }
        let mut p = idx_point(6, 63 - bit);
        p.extend_from_slice(pt);
        let v = ledger.tensor_claim(Factor::ValueBits { slot: T_RS2 }, &p)?;
        memo.insert(bit, v);
        Ok(v)
    };
    let instr_bits = |ledger: &mut Ledger<'_>,
                      memo: &mut std::collections::HashMap<usize, Goldilocks>,
                      bit: usize|
     -> Result<Goldilocks, ConstraintError> {
        if let Some(v) = memo.get(&bit) {
            return Ok(*v);
        }
        let mut p = idx_point(5, 31 - bit);
        p.extend_from_slice(pt);
        let v = ledger.tensor_claim(Factor::InstrBits, &p)?;
        memo.insert(bit, v);
        Ok(v)
    };
    let rd_bits = |ledger: &mut Ledger<'_>,
                   memo: &mut std::collections::HashMap<usize, Goldilocks>,
                   bit: usize|
     -> Result<Goldilocks, ConstraintError> {
        if let Some(v) = memo.get(&bit) {
            return Ok(*v);
        }
        let mut p = idx_point(6, 63 - bit);
        p.extend_from_slice(pt);
        let v = ledger.tensor_claim(Factor::ValueBits { slot: T_RD }, &p)?;
        memo.insert(bit, v);
        Ok(v)
    };
    let mut expect = Goldilocks::ZERO;
    // (1) the one-hot decodes.
    {
        let a = &alphas[0];
        let groups: [(&[usize], usize, usize); 4] = [
            (&idx.shoh6_r, 6, 0),
            (&idx.shoh6_i, 6, 1),
            (&idx.shoh5_r, 5, 0),
            (&idx.shoh5_i, 5, 1),
        ];
        for (ohs, nbits, src) in groups {
            for s in 0..(1usize << nbits) {
                let oh = claim_bit(ledger, ohs[s], pt)?;
                let mut prod = Goldilocks::ONE;
                for b in 0..nbits {
                    let v = if src == 0 {
                        rs2_bits(ledger, &mut rs2_memo, b)?
                    } else {
                        instr_bits(ledger, &mut instr_memo, 20 + b)?
                    };
                    let req = (s >> b) & 1;
                    let term = if req == 1 { v } else { Goldilocks::ONE.sub(&v) };
                    prod = prod.mul(&term);
                }
                expect = expect.add(&a.mul(&oh.sub(&prod)).mul(&eq_at));
            }
        }
    }
    // (2) the per-bit MUXes.
    {
        let a = &alphas[1];
        let b2 = &alphas[2];
        for class in SHIFT_CLASSES {
            let sel = claim_bit(ledger, idx.sel_by(class.sel), pt)?;
            let ohs: &[usize] = match class.oh {
                0 => &idx.shoh6_r,
                1 => &idx.shoh6_i,
                2 => &idx.shoh5_r,
                _ => &idx.shoh5_i,
            };
            let width = if class.oh < 2 { 64usize } else { 32usize };
            let top = if class.is_w { 32usize } else { 64usize };
            for i in 0..64usize {
                // Claim order per the prover's view order: rd's row
                // first, then the per-s one-hots and rs1 rows.
                let rd_i = rd_bits(ledger, &mut rd_memo, i)?;
                if i >= top {
                    if class.is_w {
                        let e = rd_i.sub(&rd_bits(ledger, &mut rd_memo, 31)?);
                        expect = expect.add(&b2.mul(&sel).mul(&e).mul(&eq_at));
                    }
                    continue;
                }
                let (lo_s, hi_s): (usize, usize) = match class.kind {
                    0 => (0, i.min(width - 1)),
                    _ => (0, (top - 1 - i).min(width - 1)),
                };
                let alpha = if class.is_w { *b2 } else { *a };
                let mut rhs = Goldilocks::ZERO;
                for s in lo_s..=hi_s {
                    let src_bit = match class.kind {
                        0 => i - s,
                        _ => i + s,
                    };
                    let oh = claim_bit(ledger, ohs[s], pt)?;
                    rhs = rhs.add(&oh.mul(&rs1_bits(ledger, &mut rs1_memo, src_bit)?));
                }
                if class.kind == 2 {
                    // + sign * (1 - sum of the covered one-hots)
                    let sign = rs1_bits(ledger, &mut rs1_memo, top - 1)?;
                    let mut covered = Goldilocks::ZERO;
                    for s in lo_s..=hi_s {
                        covered = covered.add(&claim_bit(ledger, ohs[s], pt)?);
                    }
                    rhs = rhs.add(&sign.mul(&Goldilocks::ONE.sub(&covered)));
                }
                expect = expect.add(&alpha.mul(&sel).mul(&rd_i.sub(&rhs)).mul(&eq_at));
            }
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("shift"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: mul — the limb recurrence of the unsigned product with
// range-linked limbs/carries (integer semantics), the MUL/MULW rd
// routing, and MULH's sign-corrected composition.
// ---------------------------------------------------------------------------

fn prove_mul(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-mul-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-mul-a", 5)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // Memoized operand-limb factors.
    let mut a_limbs: Vec<Option<usize>> = vec![None; 4];
    let mut b_limbs: Vec<Option<usize>> = vec![None; 4];
    let mut add_a = |vp: &mut VirtualPolynomial,
                     views: &mut Vec<ViewPair>,
                     l: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = a_limbs[l] {
            return Ok(f);
        }
        let fi = add_limb_factor(vp, views, w, T_RS1, l, log_t)?;
        a_limbs[l] = Some(fi);
        Ok(fi)
    };
    let mut add_b = |vp: &mut VirtualPolynomial,
                     views: &mut Vec<ViewPair>,
                     l: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = b_limbs[l] {
            return Ok(f);
        }
        let fi = add_limb_factor(vp, views, w, T_RS2, l, log_t)?;
        b_limbs[l] = Some(fi);
        Ok(fi)
    };
    // (1) MUL: k = 0..3 with p = rd limbs (c_4 free — the 2^64 wrap).
    {
        let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by("sel_mul"), log_t)?;
        let a0 = alphas[0];
        // Inline (the closure-over-closure borrow rules make a shared
        // emitter awkward for the rd case): emit directly.
        for k in 0..4usize {
            for i in 0..4usize {
                for j in 0..4usize {
                    if i + j == k {
                        let af = add_a(&mut vp, &mut views, i)?;
                        let bf = add_b(&mut vp, &mut views, j)?;
                        vp.add_term(a0, vec![sel, af, bf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
            if k > 0 {
                let cf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_c[k - 1], log_t)?;
                vp.add_term(a0, vec![sel, cf, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            let rd = add_limb_factor(&mut vp, &mut views, w, T_RD, k, log_t)?;
            vp.add_term(a0.neg(), vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            let cf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_c[k], log_t)?;
            vp.add_term(a0.mul(&fe(1u64 << 16).neg()), vec![sel, cf, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (2) MULH + MULHU: k = 0..7 with p = lo (k < 4) / hi (k >= 4);
    //     the k = 7 identity closes without carry-out (c_8 = 0).
    for name in ["sel_mulh", "sel_mulhu"] {
        let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(name), log_t)?;
        let a1 = alphas[1];
        for k in 0..8usize {
            for i in 0..4usize {
                for j in 0..4usize {
                    if i + j == k {
                        let af = add_a(&mut vp, &mut views, i)?;
                        let bf = add_b(&mut vp, &mut views, j)?;
                        vp.add_term(a1, vec![sel, af, bf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
            if k > 0 {
                let cf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_c[k - 1], log_t)?;
                vp.add_term(a1, vec![sel, cf, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            let pf = if k < 4 {
                add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_lo[k], log_t)?
            } else {
                add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_hi[k - 4], log_t)?
            };
            vp.add_term(a1.neg(), vec![sel, pf, ei])
                .map_err(ConstraintError::Virtual)?;
            if k < 7 {
                let cf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_c[k], log_t)?;
                vp.add_term(a1.mul(&fe(1u64 << 16).neg()), vec![sel, cf, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (3) MULW: k = 0..1 over the low limbs (c_2 free).
    {
        let sel = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_mulw"),
            log_t,
        )?;
        let a2 = alphas[2];
        for k in 0..2usize {
            for i in 0..2usize {
                for j in 0..2usize {
                    if i + j == k {
                        let af = add_a(&mut vp, &mut views, i)?;
                        let bf = add_b(&mut vp, &mut views, j)?;
                        vp.add_term(a2, vec![sel, af, bf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
            if k > 0 {
                let cf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_c[k - 1], log_t)?;
                vp.add_term(a2, vec![sel, cf, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            let rd = add_limb_factor(&mut vp, &mut views, w, T_RD, k, log_t)?;
            vp.add_term(a2.neg(), vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            let cf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_c[k], log_t)?;
            vp.add_term(a2.mul(&fe(1u64 << 16).neg()), vec![sel, cf, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        // The W sign extension: rd_2 = rd_3 = s31·0xFFFF.
        let sign = row_mle(&w.values[T_RD], 32, log_t);
        let sf = vp.add_factor(sign).map_err(ConstraintError::Virtual)?;
        views.push((
            sf,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RD },
                nbits: 64,
                row: 32,
            },
        ));
        for l in 2..4usize {
            let rd = add_limb_factor(&mut vp, &mut views, w, T_RD, l, log_t)?;
            vp.add_term(a2, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a2.mul(&fe(0xFFFF).neg()), vec![sel, sf, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (4) MULHU rd routing: rd_l = hi_l (limb copies).
    {
        let sel = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_mulhu"),
            log_t,
        )?;
        let a3 = alphas[3];
        for l in 0..4usize {
            let rd = add_limb_factor(&mut vp, &mut views, w, T_RD, l, log_t)?;
            let hi = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_hi[l], log_t)?;
            vp.add_term(a3, vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a3.neg(), vec![sel, hi, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (5) MULH rd composition: rd_l = hi_l - sa·b_l - sb·a_l - bor_l +
    //     2^16·bor_{l+1} (bor_0 = 0, bor_4 free — the mod-2^64 wrap).
    {
        let sel = add_bit_factor(
            &mut vp,
            &mut views,
            &aux.bits,
            idx.sel_by("sel_mulh"),
            log_t,
        )?;
        let a4 = alphas[4];
        // sa = rs1 bit 63 (row 0), sb = rs2 bit 63 (row 0).
        let sa = row_mle(&w.values[T_RS1], 0, log_t);
        let saf = vp.add_factor(sa).map_err(ConstraintError::Virtual)?;
        views.push((
            saf,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RS1 },
                nbits: 64,
                row: 0,
            },
        ));
        let sb = row_mle(&w.values[T_RS2], 0, log_t);
        let sbf = vp.add_factor(sb).map_err(ConstraintError::Virtual)?;
        views.push((
            sbf,
            FV::TensorRow {
                factor: Factor::ValueBits { slot: T_RS2 },
                nbits: 64,
                row: 0,
            },
        ));
        for l in 0..4usize {
            let hi = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mul_hi[l], log_t)?;
            let rd = add_limb_factor(&mut vp, &mut views, w, T_RD, l, log_t)?;
            let bf = add_b(&mut vp, &mut views, l)?;
            let af = add_a(&mut vp, &mut views, l)?;
            // + hi_l
            vp.add_term(a4, vec![sel, hi, ei])
                .map_err(ConstraintError::Virtual)?;
            // - sa·b_l - sb·a_l
            vp.add_term(a4.neg(), vec![sel, saf, bf, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a4.neg(), vec![sel, sbf, af, ei])
                .map_err(ConstraintError::Virtual)?;
            // - rd_l
            vp.add_term(a4.neg(), vec![sel, rd, ei])
                .map_err(ConstraintError::Virtual)?;
            // - bor_l (l > 0) + 2^16·bor_{l+1} (all l: bor_4 is the free
            // mod-2^64 wrap borrow — it MUST appear to close the chain).
            if l > 0 {
                let bor =
                    add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mulh_bor[l - 1], log_t)?;
                vp.add_term(a4.neg(), vec![sel, bor, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            {
                let bor = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mulh_bor[l], log_t)?;
                vp.add_term(a4.mul(&fe(1u64 << 16)), vec![sel, bor, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    ctx.stage("mul", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_mul(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let r = transcript
        .challenge_fields(b"con-mul-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-mul-a", 5)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "mul")?;
    let verdict = verify_leg_header("mul", leg, log_t, 4, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let mut a_limbs = [Goldilocks::ZERO; 4];
    let mut b_limbs = [Goldilocks::ZERO; 4];
    for l in 0..4usize {
        a_limbs[l] = claim_limb(ledger, T_RS1, l, pt)?;
        b_limbs[l] = claim_limb(ledger, T_RS2, l, pt)?;
    }
    let rd_limbs: Vec<Goldilocks> = (0..4)
        .map(|l| claim_limb(ledger, T_RD, l, pt))
        .collect::<Result<_, _>>()?;
    let mut expect = Goldilocks::ZERO;
    // (1) MUL.
    {
        let sel = claim_bit(ledger, idx.sel_by("sel_mul"), pt)?;
        for k in 0..4usize {
            let mut s = claim_carry(ledger, &idx.v_mul_c, k, pt)?;
            for i in 0..4usize {
                for j in 0..4usize {
                    if i + j == k {
                        s = s.add(&a_limbs[i].mul(&b_limbs[j]));
                    }
                }
            }
            let e = s.sub(&rd_limbs[k]).sub(&fe(1u64 << 16).mul(&claim_carry(
                ledger,
                &idx.v_mul_c,
                k + 1,
                pt,
            )?));
            expect = expect.add(&alphas[0].mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (2) MULH + MULHU.
    for name in ["sel_mulh", "sel_mulhu"] {
        let sel = claim_bit(ledger, idx.sel_by(name), pt)?;
        for k in 0..8usize {
            let mut s = claim_carry(ledger, &idx.v_mul_c, k, pt)?;
            for i in 0..4usize {
                for j in 0..4usize {
                    if i + j == k {
                        s = s.add(&a_limbs[i].mul(&b_limbs[j]));
                    }
                }
            }
            let p = if k < 4 {
                claim_val(ledger, idx.v_mul_lo[k], pt)?
            } else {
                claim_val(ledger, idx.v_mul_hi[k - 4], pt)?
            };
            let carry_out = if k < 7 {
                fe(1u64 << 16).mul(&claim_carry(ledger, &idx.v_mul_c, k + 1, pt)?)
            } else {
                Goldilocks::ZERO
            };
            let e = s.sub(&p).sub(&carry_out);
            expect = expect.add(&alphas[1].mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (3) MULW + sign.
    {
        let sel = claim_bit(ledger, idx.sel_by("sel_mulw"), pt)?;
        for k in 0..2usize {
            let mut s = claim_carry(ledger, &idx.v_mul_c, k, pt)?;
            for i in 0..2usize {
                for j in 0..2usize {
                    if i + j == k {
                        s = s.add(&a_limbs[i].mul(&b_limbs[j]));
                    }
                }
            }
            let e = s.sub(&rd_limbs[k]).sub(&fe(1u64 << 16).mul(&claim_carry(
                ledger,
                &idx.v_mul_c,
                k + 1,
                pt,
            )?));
            expect = expect.add(&alphas[2].mul(&sel).mul(&e).mul(&eq_at));
        }
        let mut sgn_pt = idx_point(6, 32);
        sgn_pt.extend_from_slice(pt);
        let s31 = ledger.tensor_claim(Factor::ValueBits { slot: T_RD }, &sgn_pt)?;
        for l in 2..4usize {
            let e = rd_limbs[l].sub(&fe(0xFFFF).mul(&s31));
            expect = expect.add(&alphas[2].mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (4) MULHU routing.
    {
        let sel = claim_bit(ledger, idx.sel_by("sel_mulhu"), pt)?;
        for l in 0..4usize {
            let hi = claim_val(ledger, idx.v_mul_hi[l], pt)?;
            let e = rd_limbs[l].sub(&hi);
            expect = expect.add(&alphas[3].mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    // (5) MULH composition.
    {
        let sel = claim_bit(ledger, idx.sel_by("sel_mulh"), pt)?;
        let mut sa_pt = idx_point(6, 0);
        sa_pt.extend_from_slice(pt);
        let sa = ledger.tensor_claim(Factor::ValueBits { slot: T_RS1 }, &sa_pt)?;
        let mut sb_pt = idx_point(6, 0);
        sb_pt.extend_from_slice(pt);
        let sb = ledger.tensor_claim(Factor::ValueBits { slot: T_RS2 }, &sb_pt)?;
        for l in 0..4usize {
            let hi = claim_val(ledger, idx.v_mul_hi[l], pt)?;
            let bor_in = if l > 0 {
                claim_val(ledger, idx.v_mulh_bor[l - 1], pt)?
            } else {
                Goldilocks::ZERO
            };
            let bor_out = claim_val(ledger, idx.v_mulh_bor[l], pt)?;
            let e = hi
                .sub(&sa.mul(&b_limbs[l]))
                .sub(&sb.mul(&a_limbs[l]))
                .sub(&bor_in)
                .sub(&rd_limbs[l])
                .add(&fe(1u64 << 16).mul(&bor_out));
            expect = expect.add(&alphas[4].mul(&sel).mul(&e).mul(&eq_at));
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("mul"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family: div — the divide-by-remainder discipline over range-linked
// magnitude limbs: |a| = |q|·|b| + |r| with 0 <= |r| < |b| (the Euclidean
// uniqueness pins (q, r)); the sign compositions route rd; the
// divide-by-zero specials follow the executor exactly.
// ---------------------------------------------------------------------------

/// The div classes: (selector, signed, is_w, is_rem).
#[derive(Clone, Copy)]
struct DivClass {
    sel: &'static str,
    signed: bool,
    is_w: bool,
    is_rem: bool,
}

const DIV_CLASSES: [DivClass; 8] = [
    DivClass {
        sel: "sel_divu",
        signed: false,
        is_w: false,
        is_rem: false,
    },
    DivClass {
        sel: "sel_remu",
        signed: false,
        is_w: false,
        is_rem: true,
    },
    DivClass {
        sel: "sel_div",
        signed: true,
        is_w: false,
        is_rem: false,
    },
    DivClass {
        sel: "sel_rem",
        signed: true,
        is_w: false,
        is_rem: true,
    },
    DivClass {
        sel: "sel_divuw",
        signed: false,
        is_w: true,
        is_rem: false,
    },
    DivClass {
        sel: "sel_remuw",
        signed: false,
        is_w: true,
        is_rem: true,
    },
    DivClass {
        sel: "sel_divw",
        signed: true,
        is_w: true,
        is_rem: false,
    },
    DivClass {
        sel: "sel_remw",
        signed: true,
        is_w: true,
        is_rem: true,
    },
];

fn prove_div(ctx: &mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError> {
    let w = ctx.w;
    let aux = ctx.aux;
    let log_t = w.log_t;
    let idx = &aux.index;
    let r = ctx
        .transcript
        .challenge_fields(b"con-div-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = ctx
        .transcript
        .challenge_fields(b"con-div-a", 6)
        .map_err(ConstraintError::Transcript)?;
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(log_t);
    let ei = vp
        .add_factor(eq.clone())
        .map_err(ConstraintError::Virtual)?;
    let mut views: Vec<ViewPair> = vec![(ei, FV::PubTable(eq.clone()))];
    // Memoized factors.
    let mut rs1_limbs: Vec<Option<usize>> = vec![None; 4];
    let mut rs2_limbs: Vec<Option<usize>> = vec![None; 4];
    let mut add_rs1 = |vp: &mut VirtualPolynomial,
                       views: &mut Vec<ViewPair>,
                       l: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = rs1_limbs[l] {
            return Ok(f);
        }
        let fi = add_limb_factor(vp, views, w, T_RS1, l, log_t)?;
        rs1_limbs[l] = Some(fi);
        Ok(fi)
    };
    let mut add_rs2 = |vp: &mut VirtualPolynomial,
                       views: &mut Vec<ViewPair>,
                       l: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = rs2_limbs[l] {
            return Ok(f);
        }
        let fi = add_limb_factor(vp, views, w, T_RS2, l, log_t)?;
        rs2_limbs[l] = Some(fi);
        Ok(fi)
    };
    let mut rd_limbs: Vec<Option<usize>> = vec![None; 4];
    let mut add_rd = |vp: &mut VirtualPolynomial,
                      views: &mut Vec<ViewPair>,
                      l: usize|
     -> Result<usize, ConstraintError> {
        if let Some(f) = rd_limbs[l] {
            return Ok(f);
        }
        let fi = add_limb_factor(vp, views, w, T_RD, l, log_t)?;
        rd_limbs[l] = Some(fi);
        Ok(fi)
    };
    // (1) the eqz/eqzw recurrences (unconditional).
    {
        let a = &alphas[0];
        for i in 0..64usize {
            let next = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqz[i + 1], log_t)?;
            let prev = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqz[i], log_t)?;
            let row = 63 - (63 - i); // bit (63 - i) at row (63 - (63-i)) = i
            let b = row_mle(&w.values[T_RS2], row, log_t);
            let bf = vp.add_factor(b).map_err(ConstraintError::Virtual)?;
            views.push((
                bf,
                FV::TensorRow {
                    factor: Factor::ValueBits { slot: T_RS2 },
                    nbits: 64,
                    row,
                },
            ));
            // next - prev + prev·b = 0
            vp.add_term(*a, vec![next, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![prev, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*a, vec![prev, bf, ei])
                .map_err(ConstraintError::Virtual)?;
        }
        for i in 0..32usize {
            let next = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqzw[i + 1], log_t)?;
            let prev = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.eqzw[i], log_t)?;
            let row = 63 - i; // bit i at row 63 - i
            let b = row_mle(&w.values[T_RS2], row, log_t);
            let bf = vp.add_factor(b).map_err(ConstraintError::Virtual)?;
            views.push((
                bf,
                FV::TensorRow {
                    factor: Factor::ValueBits { slot: T_RS2 },
                    nbits: 64,
                    row,
                },
            ));
            vp.add_term(*a, vec![next, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![prev, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(*a, vec![prev, bf, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // The per-class bz factor index (eqz[64] or eqzw[32]) and its flip.
    let bz_of = |vp: &mut VirtualPolynomial,
                 views: &mut Vec<ViewPair>,
                 is_w: bool|
     -> Result<(usize, usize), ConstraintError> {
        let bz = add_bit_factor(
            vp,
            views,
            &aux.bits,
            if is_w { idx.eqzw[32] } else { idx.eqz[64] },
            log_t,
        )?;
        let nbz_col = if is_w { idx.eqzw[32] } else { idx.eqz[64] };
        let flipped = DenseMle {
            num_vars: log_t,
            evaluations: aux.bits[nbz_col]
                .iter()
                .map(|v| fe(1 ^ *v as u64))
                .collect(),
        };
        let nbz = vp.add_factor(flipped).map_err(ConstraintError::Virtual)?;
        Ok((bz, nbz))
    };
    // (2) the magnitude definitions.
    {
        let a = &alphas[1];
        for class in DIV_CLASSES {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(class.sel), log_t)?;
            // The sign sources.
            let (sa_f, sb_f): (Option<usize>, Option<usize>) = if !class.signed {
                (None, None)
            } else {
                let row = if class.is_w { 32 } else { 0 };
                let fa = row_mle(&w.values[T_RS1], row, log_t);
                let saf = vp.add_factor(fa).map_err(ConstraintError::Virtual)?;
                views.push((
                    saf,
                    FV::TensorRow {
                        factor: Factor::ValueBits { slot: T_RS1 },
                        nbits: 64,
                        row,
                    },
                ));
                let fb = row_mle(&w.values[T_RS2], row, log_t);
                let sbf = vp.add_factor(fb).map_err(ConstraintError::Virtual)?;
                views.push((
                    sbf,
                    FV::TensorRow {
                        factor: Factor::ValueBits { slot: T_RS2 },
                        nbits: 64,
                        row,
                    },
                ));
                (Some(saf), Some(sbf))
            };
            let limb_hi = if class.is_w { 2 } else { 4 };
            for l in 0..limb_hi {
                let delta = if l == 0 { 1u64 } else { 0 };
                // mag_a vs rs1's limb l.
                {
                    let mag =
                        add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_a[l], log_t)?;
                    let x = add_rs1(&mut vp, &mut views, l)?;
                    vp.add_term(*a, vec![sel, mag, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a.neg(), vec![sel, x, ei])
                        .map_err(ConstraintError::Virtual)?;
                    if let Some(sf) = sa_f {
                        vp.add_term(a.mul(&fe(2)), vec![sel, sf, x, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(a.mul(&fe(0xFFFF + delta).neg()), vec![sel, sf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
                // mag_b vs rs2's limb l.
                {
                    let mag =
                        add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_b[l], log_t)?;
                    let x = add_rs2(&mut vp, &mut views, l)?;
                    vp.add_term(*a, vec![sel, mag, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a.neg(), vec![sel, x, ei])
                        .map_err(ConstraintError::Virtual)?;
                    if let Some(sf) = sb_f {
                        vp.add_term(a.mul(&fe(2)), vec![sel, sf, x, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(a.mul(&fe(0xFFFF + delta).neg()), vec![sel, sf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
            // The W top limbs are zero.
            if class.is_w {
                for l in 2..4usize {
                    for mag_cols in [&idx.v_mag_a, &idx.v_mag_b] {
                        let mag =
                            add_val_factor(&mut vp, &mut views, &aux.vals, mag_cols[l], log_t)?;
                        vp.add_term(*a, vec![sel, mag, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
        }
    }
    // (3) the division recurrence + closure, masked by (1 - bz).
    {
        let a = &alphas[2];
        let a2 = &alphas[3];
        for class in DIV_CLASSES {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(class.sel), log_t)?;
            let (_bz, nbz) = bz_of(&mut vp, &mut views, class.is_w)?;
            let k_hi = if class.is_w { 2 } else { 4 };
            for k in 0..k_hi {
                // + S_k(mag_q, mag_b)
                for i in 0..4usize {
                    for j in 0..4usize {
                        if i + j == k {
                            let qf = add_val_factor(
                                &mut vp,
                                &mut views,
                                &aux.vals,
                                idx.v_mag_q[i],
                                log_t,
                            )?;
                            let bf = add_val_factor(
                                &mut vp,
                                &mut views,
                                &aux.vals,
                                idx.v_mag_b[j],
                                log_t,
                            )?;
                            vp.add_term(*a, vec![sel, nbz, qf, bf, ei])
                                .map_err(ConstraintError::Virtual)?;
                        }
                    }
                }
                // + d_k (k > 0) + mag_r_k - mag_a_k
                if k > 0 {
                    let df =
                        add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_d[k - 1], log_t)?;
                    vp.add_term(*a, vec![sel, nbz, df, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
                let rf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_r[k], log_t)?;
                vp.add_term(*a, vec![sel, nbz, rf, ei])
                    .map_err(ConstraintError::Virtual)?;
                let af = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_a[k], log_t)?;
                vp.add_term(a.neg(), vec![sel, nbz, af, ei])
                    .map_err(ConstraintError::Virtual)?;
                // - 2^16·d_{k+1}
                let df = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_d[k], log_t)?;
                vp.add_term(a.mul(&fe(1u64 << 16).neg()), vec![sel, nbz, df, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            // The closure: d_{k_hi} = 0 under (1 - bz).
            {
                let df =
                    add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_d[k_hi - 1], log_t)?;
                vp.add_term(*a2, vec![sel, nbz, df, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
    }
    // (4) the (mag_r - mag_b) borrow chain (unconditional) + the bound.
    {
        let a = &alphas[3];
        let a4 = &alphas[4];
        for l in 0..4usize {
            let rf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_r[l], log_t)?;
            let bf = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_mag_b[l], log_t)?;
            let out = add_val_factor(&mut vp, &mut views, &aux.vals, idx.v_rlt_out[l], log_t)?;
            // mag_r_l - mag_b_l - bor_l + 2^16·bor_{l+1} - out_l = 0
            // (bor_4 = the [mag_r < mag_b] indicator appears at l = 3).
            vp.add_term(*a, vec![rf, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![bf, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a.neg(), vec![out, ei])
                .map_err(ConstraintError::Virtual)?;
            if l > 0 {
                let bor = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.v_rbor[l - 1], log_t)?;
                vp.add_term(a.neg(), vec![bor, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
            {
                let bor = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.v_rbor[l], log_t)?;
                vp.add_term(a.mul(&fe(1u64 << 16)), vec![bor, ei])
                    .map_err(ConstraintError::Virtual)?;
            }
        }
        // The bound: sel·(1 - bz)·(1 - bor_4) = 0 per class (selector-
        // masked: the bound only applies on division cycles).
        for class in DIV_CLASSES {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(class.sel), log_t)?;
            let (_bz, nbz) = bz_of(&mut vp, &mut views, class.is_w)?;
            let bor4 = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.v_rbor[3], log_t)?;
            vp.add_term(*a4, vec![sel, nbz, ei])
                .map_err(ConstraintError::Virtual)?;
            vp.add_term(a4.neg(), vec![sel, nbz, bor4, ei])
                .map_err(ConstraintError::Virtual)?;
        }
    }
    // (5) the rd routing.
    {
        let a5 = &alphas[5];
        for class in DIV_CLASSES {
            let sel = add_bit_factor(&mut vp, &mut views, &aux.bits, idx.sel_by(class.sel), log_t)?;
            let (bz, _nbz) = bz_of(&mut vp, &mut views, class.is_w)?;
            // The W sign extension: rd_2 = rd_3 = s31·0xFFFF (s31 =
            // rd's bit 31, row 32).
            if class.is_w {
                let sign = row_mle(&w.values[T_RD], 32, log_t);
                let sf = vp.add_factor(sign).map_err(ConstraintError::Virtual)?;
                views.push((
                    sf,
                    FV::TensorRow {
                        factor: Factor::ValueBits { slot: T_RD },
                        nbits: 64,
                        row: 32,
                    },
                ));
                for l in 2..4usize {
                    let rd = add_rd(&mut vp, &mut views, l)?;
                    vp.add_term(*a5, vec![sel, rd, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a5.mul(&fe(0xFFFF).neg()), vec![sel, sf, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
            }
            let n_limbs = if class.is_w { 2 } else { 4 };
            // The value routing.
            if !class.signed {
                // rd_l = mag_l (copies) — q for DIV, r for REM.
                let src = if class.is_rem {
                    &idx.v_mag_r
                } else {
                    &idx.v_mag_q
                };
                for l in 0..n_limbs {
                    let rd = add_rd(&mut vp, &mut views, l)?;
                    let mag = add_val_factor(&mut vp, &mut views, &aux.vals, src[l], log_t)?;
                    vp.add_term(*a5, vec![sel, rd, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a5.neg(), vec![sel, mag, ei])
                        .map_err(ConstraintError::Virtual)?;
                }
            } else {
                // The conditional negation: rd_l = (1-s)·mag_l + s·(0xFFFF
                // - mag_l) + s·delta_l with s the composed sign.
                let src = if class.is_rem {
                    &idx.v_mag_r
                } else {
                    &idx.v_mag_q
                };
                let row = if class.is_w { 32 } else { 0 };
                let sa = row_mle(&w.values[T_RS1], row, log_t);
                let saf = vp.add_factor(sa).map_err(ConstraintError::Virtual)?;
                views.push((
                    saf,
                    FV::TensorRow {
                        factor: Factor::ValueBits { slot: T_RS1 },
                        nbits: 64,
                        row,
                    },
                ));
                let sb = row_mle(&w.values[T_RS2], row, log_t);
                let sbf = vp.add_factor(sb).map_err(ConstraintError::Virtual)?;
                views.push((
                    sbf,
                    FV::TensorRow {
                        factor: Factor::ValueBits { slot: T_RS2 },
                        nbits: 64,
                        row,
                    },
                ));
                for l in 0..n_limbs {
                    let rd = add_rd(&mut vp, &mut views, l)?;
                    let mag = add_val_factor(&mut vp, &mut views, &aux.vals, src[l], log_t)?;
                    let delta = if l == 0 { 1u64 } else { 0 };
                    // rd - mag + 2·s·mag - s·(0xFFFF + delta):
                    // s = sa for REM; s = sa + sb - 2·sa·sb for DIV.
                    vp.add_term(*a5, vec![sel, rd, ei])
                        .map_err(ConstraintError::Virtual)?;
                    vp.add_term(a5.neg(), vec![sel, mag, ei])
                        .map_err(ConstraintError::Virtual)?;
                    if class.is_rem {
                        vp.add_term(a5.mul(&fe(2)), vec![sel, saf, mag, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(a5.mul(&fe(0xFFFF + delta).neg()), vec![sel, saf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    } else {
                        for sf in [saf, sbf] {
                            vp.add_term(a5.mul(&fe(2)), vec![sel, sf, mag, ei])
                                .map_err(ConstraintError::Virtual)?;
                            vp.add_term(a5.mul(&fe(0xFFFF + delta).neg()), vec![sel, sf, ei])
                                .map_err(ConstraintError::Virtual)?;
                        }
                        vp.add_term(a5.mul(&fe(4).neg()), vec![sel, saf, sbf, mag, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(a5.mul(&fe(2 * (0xFFFF + delta))), vec![sel, saf, sbf, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
            // (6) the divide-by-zero specials (selector-masked: only
            // division cycles carry them).
            {
                if !class.is_rem {
                    // rd = -1: all limbs 0xFFFF.
                    for l in 0..4usize {
                        let rd = add_rd(&mut vp, &mut views, l)?;
                        vp.add_term(*a5, vec![sel, bz, rd, ei])
                            .map_err(ConstraintError::Virtual)?;
                        // the constant -0xFFFF rides eq
                        vp.add_term(a5.mul(&fe(0xFFFF).neg()), vec![sel, bz, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                } else {
                    // rd = rs1 (low limbs; the W extension above covers
                    // the top).
                    for l in 0..n_limbs {
                        let rd = add_rd(&mut vp, &mut views, l)?;
                        let x = add_rs1(&mut vp, &mut views, l)?;
                        vp.add_term(*a5, vec![sel, bz, rd, ei])
                            .map_err(ConstraintError::Virtual)?;
                        vp.add_term(a5.neg(), vec![sel, bz, x, ei])
                            .map_err(ConstraintError::Virtual)?;
                    }
                }
            }
        }
    }
    ctx.stage("div", &mut vp, &views, Goldilocks::ZERO)
        .map(|_| ())
}

fn verify_div(
    aux: &AuxCols,
    iter: &mut LegIter<'_>,
    log_t: usize,
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let idx = &aux.index;
    let r = transcript
        .challenge_fields(b"con-div-r", log_t)
        .map_err(ConstraintError::Transcript)?;
    let alphas = transcript
        .challenge_fields(b"con-div-a", 6)
        .map_err(ConstraintError::Transcript)?;
    let leg = next_constraint_leg(iter, "div")?;
    let verdict = verify_leg_header("div", leg, log_t, 5, transcript)?;
    let eq_at = DenseMle::eq_eval(&r, &verdict.point).map_err(ConstraintError::Mle)?;
    let pt = &verdict.point;
    let rs1_limbs: Vec<Goldilocks> = (0..4)
        .map(|l| claim_limb(ledger, T_RS1, l, pt))
        .collect::<Result<_, _>>()?;
    let rs2_limbs: Vec<Goldilocks> = (0..4)
        .map(|l| claim_limb(ledger, T_RS2, l, pt))
        .collect::<Result<_, _>>()?;
    let rd_limbs: Vec<Goldilocks> = (0..4)
        .map(|l| claim_limb(ledger, T_RD, l, pt))
        .collect::<Result<_, _>>()?;
    let mut expect = Goldilocks::ZERO;
    // (1) the eqz/eqzw recurrences.
    {
        let a = &alphas[0];
        for i in 0..64usize {
            let next = claim_bit(ledger, idx.eqz[i + 1], pt)?;
            let prev = claim_bit(ledger, idx.eqz[i], pt)?;
            let b = claim_tensor_bit(ledger, T_RS2, 63 - i, pt)?;
            let e = next.sub(&prev).add(&prev.mul(&b));
            expect = expect.add(&a.mul(&e).mul(&eq_at));
        }
        for i in 0..32usize {
            let next = claim_bit(ledger, idx.eqzw[i + 1], pt)?;
            let prev = claim_bit(ledger, idx.eqzw[i], pt)?;
            let b = claim_tensor_bit(ledger, T_RS2, i, pt)?;
            let e = next.sub(&prev).add(&prev.mul(&b));
            expect = expect.add(&a.mul(&e).mul(&eq_at));
        }
    }
    // (2) + (3) + (4) + (5) + (6) per class.
    for class in DIV_CLASSES {
        let sel = claim_bit(ledger, idx.sel_by(class.sel), pt)?;
        let bz = claim_bit(
            ledger,
            if class.is_w {
                idx.eqzw[32]
            } else {
                idx.eqz[64]
            },
            pt,
        )?;
        let nbz = Goldilocks::ONE.sub(&bz);
        let limb_hi = if class.is_w { 2 } else { 4 };
        // (2) the magnitude definitions.
        {
            let a = &alphas[1];
            let sa = if class.signed {
                Some(claim_tensor_bit(
                    ledger,
                    T_RS1,
                    if class.is_w { 31 } else { 63 },
                    pt,
                )?)
            } else {
                None
            };
            let sb = if class.signed {
                Some(claim_tensor_bit(
                    ledger,
                    T_RS2,
                    if class.is_w { 31 } else { 63 },
                    pt,
                )?)
            } else {
                None
            };
            for l in 0..limb_hi {
                for (cols, x, s) in [
                    (&idx.v_mag_a, rs1_limbs[l], sa),
                    (&idx.v_mag_b, rs2_limbs[l], sb),
                ] {
                    let m = claim_val(ledger, cols[l], pt)?;
                    let mut e = m.sub(&x);
                    if let Some(s) = s {
                        let delta = if l == 0 { 1u64 } else { 0 };
                        e = e
                            .add(&fe(2).mul(&s).mul(&x))
                            .sub(&s.mul(&fe(0xFFFF + delta)));
                    }
                    expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
                }
            }
            if class.is_w {
                for l in 2..4usize {
                    for cols in [&idx.v_mag_a, &idx.v_mag_b] {
                        let e = claim_val(ledger, cols[l], pt)?;
                        expect = expect.add(&a.mul(&sel).mul(&e).mul(&eq_at));
                    }
                }
            }
        }
        // (3) the recurrence + closure.
        {
            let a = &alphas[2];
            let a2 = &alphas[3];
            let k_hi = if class.is_w { 2 } else { 4 };
            for k in 0..k_hi {
                let mut s = claim_carry(ledger, &idx.v_mag_d, k, pt)?;
                for i in 0..4usize {
                    for j in 0..4usize {
                        if i + j == k {
                            s = s.add(&claim_val(ledger, idx.v_mag_q[i], pt)?.mul(&claim_val(
                                ledger,
                                idx.v_mag_b[j],
                                pt,
                            )?));
                        }
                    }
                }
                s = s.add(&claim_val(ledger, idx.v_mag_r[k], pt)?);
                let e = s
                    .sub(&claim_val(ledger, idx.v_mag_a[k], pt)?)
                    .sub(&fe(1u64 << 16).mul(&claim_carry(ledger, &idx.v_mag_d, k + 1, pt)?));
                expect = expect.add(&a.mul(&sel).mul(&nbz).mul(&e).mul(&eq_at));
            }
            let e = claim_carry(ledger, &idx.v_mag_d, k_hi, pt)?;
            expect = expect.add(&a2.mul(&sel).mul(&nbz).mul(&e).mul(&eq_at));
        }
        // (4b) the bound.
        {
            let a4 = &alphas[4];
            let bor4 = claim_bit(ledger, idx.v_rbor[3], pt)?;
            let e = nbz.mul(&Goldilocks::ONE.sub(&bor4));
            expect = expect.add(&a4.mul(&sel).mul(&e).mul(&eq_at));
        }
        // (5) the rd routing.
        {
            let a5 = &alphas[5];
            if class.is_w {
                let s31 = claim_tensor_bit(ledger, T_RD, 31, pt)?;
                for l in 2..4usize {
                    let e = rd_limbs[l].sub(&fe(0xFFFF).mul(&s31));
                    expect = expect.add(&a5.mul(&sel).mul(&e).mul(&eq_at));
                }
            }
            let src = if class.is_rem {
                &idx.v_mag_r
            } else {
                &idx.v_mag_q
            };
            if !class.signed {
                for l in 0..limb_hi {
                    let e = rd_limbs[l].sub(&claim_val(ledger, src[l], pt)?);
                    expect = expect.add(&a5.mul(&sel).mul(&e).mul(&eq_at));
                }
            } else {
                let sa = claim_tensor_bit(ledger, T_RS1, if class.is_w { 31 } else { 63 }, pt)?;
                let sb = claim_tensor_bit(ledger, T_RS2, if class.is_w { 31 } else { 63 }, pt)?;
                let sq = if class.is_rem {
                    sa
                } else {
                    sa.add(&sb).sub(&fe(2).mul(&sa.mul(&sb)))
                };
                for l in 0..limb_hi {
                    let delta = if l == 0 { 1u64 } else { 0 };
                    let e = rd_limbs[l]
                        .sub(&claim_val(ledger, src[l], pt)?)
                        .add(&fe(2).mul(&sq).mul(&claim_val(ledger, src[l], pt)?))
                        .sub(&sq.mul(&fe(0xFFFF + delta)));
                    expect = expect.add(&a5.mul(&sel).mul(&e).mul(&eq_at));
                }
            }
            // (6) the bz specials.
            if !class.is_rem {
                for l in 0..4usize {
                    let e = rd_limbs[l].sub(&fe(0xFFFF));
                    expect = expect.add(&a5.mul(&sel).mul(&bz).mul(&e).mul(&eq_at));
                }
            } else {
                for l in 0..limb_hi {
                    let e = rd_limbs[l].sub(&rs1_limbs[l]);
                    expect = expect.add(&a5.mul(&sel).mul(&bz).mul(&e).mul(&eq_at));
                }
            }
        }
    }
    // (4a) the borrow chain (class-independent).
    {
        let a = &alphas[3];
        for l in 0..4usize {
            let bor_in = if l > 0 {
                claim_bit(ledger, idx.v_rbor[l - 1], pt)?
            } else {
                Goldilocks::ZERO
            };
            let bor_out = claim_bit(ledger, idx.v_rbor[l], pt)?;
            let out = claim_val(ledger, idx.v_rlt_out[l], pt)?;
            let e = claim_val(ledger, idx.v_mag_r[l], pt)?
                .sub(&claim_val(ledger, idx.v_mag_b[l], pt)?)
                .sub(&bor_in)
                .add(&fe(1u64 << 16).mul(&bor_out))
                .sub(&out);
            expect = expect.add(&a.mul(&e).mul(&eq_at));
        }
    }
    if expect != verdict.final_claim {
        return Err(ConstraintError::FinalCheck("div"));
    }
    Ok(())
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
    let mut ctx = FamilyCtx {
        w,
        aux,
        ledger,
        legs,
        transcript,
    };
    let dbg = std::env::var_os("LZX_SEM_TIMING").is_some();
    for (name, f) in [
        (
            "booleanity",
            prove_booleanity as fn(&mut FamilyCtx<'_, '_, '_>) -> Result<(), ConstraintError>,
        ),
        ("selectors", prove_selectors),
        ("decode", prove_decode),
        ("flags", prove_flags),
        ("arith", prove_arith),
        ("shift", prove_shift),
        ("mul", prove_mul),
        ("div", prove_div),
        ("rangelinks", prove_range_links),
        ("cmp", prove_cmp),
        ("ctrl", prove_ctrl),
        ("route", prove_route),
        ("halt", prove_halt),
    ] {
        if dbg {
            eprintln!("[sem-family] >>> enter {name}");
        }
        f(&mut ctx)?;
    }
    Ok(())
}

/// The verifier-side constraint shape: everything `verify_constraints`
/// needs that is layout-derived (the column registry, the counts, the
/// alpha arities) — NO witness data. The pipeline verifier builds this
/// from the public log_t; the kernel tests pass the prover's aux.
#[derive(Clone, Debug)]
pub struct ConstraintShape {
    pub log_t: usize,
    /// The aux column registry (selector positions, value-column ids).
    pub index: AuxIndex,
    /// The number of bit columns (the bool-cols alpha arity).
    pub num_bits: usize,
    /// The number of value columns.
    pub num_vals: usize,
}

/// The layout-only aux registry: identical column positions to
/// `build_aux` (construction order is instr-independent); the data
/// columns are blanked with their lengths preserved as the counts the
/// verifier's challenge derivations need.
pub fn aux_shape(log_t: usize) -> Result<AuxCols, ConstraintError> {
    let lt = log_t.max(1);
    let t = 1usize << lt;
    // A synthetic all-zero witness: the registry construction does not
    // read the content for positions.
    let zero_vals = vec![Goldilocks::ZERO; t];
    let mut values = Vec::new();
    for _ in 0..crate::columns::VALUE_TENSORS {
        values.push(DenseMle {
            num_vars: 6 + lt,
            evaluations: vec![Goldilocks::ZERO; 64 * t],
        });
    }
    let w = CycleWitness {
        log_t: lt,
        steps: 1,
        pc: zero_vals.clone(),
        next_pc: zero_vals.clone(),
        instr: zero_vals.clone(),
        instr_bits: DenseMle {
            num_vars: 5 + lt,
            evaluations: vec![Goldilocks::ZERO; 32 * t],
        },
        values: values.try_into().map_err(|_| ConstraintError::Shape)?,
        rs1_idx: vec![0; t],
        rs2_idx: vec![0; t],
        rd_idx: vec![0; t],
        rd_we: vec![0; t],
        mem_re: vec![0; t],
        mem_we: vec![0; t],
        mem_half: vec![0; t],
        mem_word: zero_vals.clone(),
        mem_addr: zero_vals.clone(),
        fetch_word: zero_vals.clone(),
        halted: vec![0; t],
    };
    let mut aux = build_aux(&w, &[Instr::Ecall])?;
    // Blank the data but PRESERVE the column lengths (the verifier's
    // log_t derivation and the alpha arities read them).
    for c in aux.bits.iter_mut() {
        c.clear();
        c.resize(t, 0);
    }
    for c in aux.vals.iter_mut() {
        c.clear();
        c.resize(t, Goldilocks::ZERO);
    }
    Ok(aux)
}

/// The shape of an aux registry (for `ConstraintShape`).
pub fn constraint_shape(aux: &AuxCols) -> ConstraintShape {
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
    ConstraintShape {
        log_t,
        index: aux.index.clone(),
        num_bits: aux.bits.len(),
        num_vals: aux.vals.len(),
    }
}

/// Verify all constraint families (legs consumed in protocol order).
/// The verifier never sees the witness: every factor value arrives as a
/// ledger claim, and the coverage gate is enforced by the decode leg's
/// partition identities (not by a prover-side instruction list).
pub fn verify_constraints(
    aux: &AuxCols,
    proofs: &[ConstraintLeg],
    ledger: &mut Ledger<'_>,
    transcript: &mut Transcript,
) -> Result<(), ConstraintError> {
    let log_t = aux
        .bits
        .first()
        .map(|c| c.len().trailing_zeros() as usize)
        .unwrap_or(0);
    let mut iter = proofs.iter();
    verify_booleanity_shape(aux, &mut iter, ledger, transcript)?;
    verify_selectors(aux, &mut iter, log_t, ledger, transcript)?;
    verify_decode(aux, &mut iter, log_t, ledger, transcript)?;
    verify_flags(aux, &mut iter, log_t, ledger, transcript)?;
    verify_arith(aux, &mut iter, ledger, transcript)?;
    verify_shift(aux, &mut iter, log_t, ledger, transcript)?;
    verify_mul(aux, &mut iter, log_t, ledger, transcript)?;
    verify_div(aux, &mut iter, log_t, ledger, transcript)?;
    verify_range_links(aux, &mut iter, log_t, ledger, transcript)?;
    verify_cmp(aux, &mut iter, ledger, transcript)?;
    verify_ctrl(aux, &mut iter, ledger, transcript)?;
    verify_route(aux, &mut iter, ledger, transcript)?;
    verify_halt(aux, &mut iter, ledger, transcript)?;
    if iter.next().is_some() {
        return Err(ConstraintError::Shape);
    }
    Ok(())
}

#[cfg(test)]
mod shift_sparse_tests {
    use super::*;
    use crate::columns::{build_cycle_witness, FetchWindow, RamWindow};
    use crate::ledger::BaseClaim;

    /// A program exercising every shift class (register/immediate ×
    /// 64/32-bit × logical/arithmetic) — the constraint-test corpus
    /// program (mixed addi + shifts of both shamt widths).
    fn shift_program() -> Vec<u8> {
        let r = |f7: u32, rs2: u8, rs1: u8, f3: u32, rd: u8, op: u32| {
            (f7 << 25)
                | ((rs2 as u32) << 20)
                | ((rs1 as u32) << 15)
                | (f3 << 12)
                | ((rd as u32) << 7)
                | op
        };
        let i = |f6: u32, shamt: u8, rs1: u8, f3: u32, rd: u8, op: u32| {
            (f6 << 26)
                | ((shamt as u32) << 20)
                | ((rs1 as u32) << 15)
                | (f3 << 12)
                | ((rd as u32) << 7)
                | op
        };
        let addi = |rd: u8, rs1: u8, imm: i64| {
            ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
        };
        let words = [
            addi(1, 0, -1),
            addi(2, 0, 0x123),
            addi(3, 0, 0x40000000),
            i(0x00, 5, 1, 1, 4, 0x13),  // slli shamt 5
            i(0x00, 37, 1, 5, 5, 0x13), // srli shamt 37
            i(0x10, 13, 1, 5, 6, 0x13), // srai shamt 13
            r(0, 3, 2, 1, 7, 0x1b),     // slliw
            r(0, 9, 2, 5, 8, 0x1b),     // srliw
            r(0x20, 7, 2, 5, 9, 0x1b),  // sraiw
            addi(10, 0, 40),
            r(0, 10, 1, 1, 11, 0x33),    // sll
            r(0, 10, 1, 5, 12, 0x33),    // srl
            r(0x20, 10, 1, 5, 13, 0x33), // sra
            r(0, 10, 1, 1, 14, 0x3b),    // sllw
            r(0, 10, 1, 5, 15, 0x3b),    // srlw
            r(0x20, 10, 1, 5, 16, 0x3b), // sraw
            0x73u32,
        ];
        let mut v = Vec::new();
        for w in words {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }

    /// Run one shift-family prover variant over the same witness and
    /// return (legs, claims).
    fn run_variant(
        w: &CycleWitness,
        aux: &AuxCols,
        sparse: bool,
    ) -> (Vec<ConstraintLeg>, Vec<BaseClaim>) {
        let bit_mles: Vec<DenseMle> = aux
            .bits
            .iter()
            .map(|c| DenseMle {
                num_vars: w.log_t,
                evaluations: c.iter().map(|v| fe(*v as u64)).collect(),
            })
            .collect();
        let val_mles: Vec<DenseMle> = aux
            .vals
            .iter()
            .map(|c| DenseMle {
                num_vars: w.log_t,
                evaluations: c.clone(),
            })
            .collect();
        let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
        for slot in 0..crate::columns::VALUE_TENSORS {
            table.push((Factor::ValueBits { slot }, &w.values[slot]));
        }
        table.push((Factor::InstrBits, &w.instr_bits));
        for (id, m) in bit_mles.iter().enumerate() {
            table.push((Factor::BitCol { id }, m));
        }
        for (id, m) in val_mles.iter().enumerate() {
            table.push((Factor::ValCol { id }, m));
        }
        let mut ledger = Ledger::prover(table);
        let mut legs = Vec::new();
        let mut tr = Transcript::new_default(b"con-test");
        let mut ctx = FamilyCtx {
            w,
            aux,
            ledger: &mut ledger,
            legs: &mut legs,
            transcript: &mut tr,
        };
        if sparse {
            super::prove_shift(&mut ctx).ok().unwrap();
        } else {
            super::prove_shift_dense_cfg_test(&mut ctx).ok().unwrap();
        }
        (legs, ledger.claims().to_vec())
    }

    /// The sparse-engine shift proof must be BYTE-IDENTICAL to the dense
    /// engine's over the same virtual polynomial: same round messages,
    /// same transcript flow, same claim values per key.
    #[test]
    fn sparse_shift_proof_is_byte_identical() {
        let prog = shift_program();
        let mut state = lattice_vm::MachineState::new();
        state.load_program(0, &prog);
        let rows = lattice_vm::run(&mut state, 256).ok().unwrap();
        let (w, _fw) = match build_cycle_witness(
            &rows,
            &prog,
            &[],
            RamWindow { log_k: 6 },
            FetchWindow { log_k: 5 },
        ) {
            Ok(v) => v,
            Err(e) => panic!("witness: {e:?}"),
        };
        let instrs: Vec<Instr> = rows.iter().map(|r| r.instr).collect();
        let aux = match build_aux(&w, &instrs) {
            Ok(v) => v,
            Err(e) => panic!("aux: {e:?}"),
        };

        let (legs_d, claims_d) = run_variant(&w, &aux, false);
        let (legs_s, claims_s) = run_variant(&w, &aux, true);
        assert_eq!(legs_d.len(), 1);
        assert_eq!(legs_s.len(), 1);
        assert_eq!(legs_d[0].name, legs_s[0].name);
        assert_eq!(legs_d[0].claim, legs_s[0].claim);
        assert_eq!(
            legs_d[0].sc.rounds.len(),
            legs_s[0].sc.rounds.len(),
            "round count"
        );
        for (ri, (rd, rs)) in legs_d[0]
            .sc
            .rounds
            .iter()
            .zip(legs_s[0].sc.rounds.iter())
            .enumerate()
        {
            if rd != rs {
                eprintln!("DIVERGING ROUND {ri}: dense={rd:?} sparse={rs:?}");
            }
            assert_eq!(rd, rs, "round {ri} messages must be byte-identical");
        }
        // Claim VALUES per key must agree (the sparse prover records each
        // key once; the dense prover may record duplicates — compare the
        // deduplicated key->value maps).
        let map_of = |claims: &[BaseClaim]| {
            let mut m = std::collections::HashMap::new();
            for c in claims {
                m.insert((c.factor, c.point.clone()), c.value);
            }
            m
        };
        let md = map_of(&claims_d);
        let ms = map_of(&claims_s);
        for (k, v) in &md {
            assert_eq!(ms.get(k), Some(v), "claim for {k:?}");
        }
        // The sparse prover must cover every key the verifier pops.
        assert!(ms.len() >= 300, "expected the full row/one-hot cover");
    }
}
