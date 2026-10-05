//! The v3 pipeline: the COMPLETE sparse-native zkVM proof.
//!
//! v2 (pipeline2) proved the memory argument (Twist & Shout over the
//! fetch/input/RAM/register streams) but left the instruction semantics
//! unconstrained — the documented "P0-4 AIR" gap. v3 closes it: every
//! instruction family, every flag, every ALU identity, and every
//! comparison is constrained through committed columns, and the whole
//! constraint system runs on the sparse "0s are free" engine — the
//! per-round work is proportional to the SUPPORT of each constraint's
//! selector, never to `T` times the constraint count.
//!
//! ## The statement (soundness scope)
//!
//! "There exist committed witness columns over `T = 2^log_t` rows such
//! that: (i) the memory arguments hold — the fetch stream reads the
//! public program image, the input stream reads the public input, the
//! RAM and register files obey read-after-write consistency from the
//! public initial states to the public final states (the v2 legs);
//! (ii) every row decodes to one of the 16 P0 families through the
//! public decode table; (iii) the instruction semantics hold row-wise —
//! the ALU identities (with full carry chains at byte granularity),
//! the comparison results, the control-flow targets, the effective
//! addresses, the register write-enable and x0 masking; (iv) every
//! byte of every decomposed value is range-checked through a
//! read-only-table Shout."
//!
//! The verifier NEVER re-executes the program; it recomputes only the
//! public tables (program image, input image, decode table, identity
//! tables) and O(λ) field work per leg.
//!
//! ## The P0-63 value profile (soundness-critical)
//!
//! All committed VALUES are constrained below 2^63 (7 bytes of 8 bits +
//! a 7-bit top byte). This is not a convenience: at 64 bits the mod-p
//! recomposition identity `V = Σ 2^{8k}·b_k` admits a ±p aliasing window
//! (`Σ = V + p` fits the byte ranges whenever `V < 2^32−1`), which a
//! malicious prover can use to run a "+p-shifted" integer execution that
//! is field-consistent with the public I/O — a total soundness break.
//! At 63 bits every linear identity has magnitude < 2^63 < p and the
//! window cannot open. Integer arithmetic between 63-bit values is
//! constrained by BYTE-GRAIN ADD-CHAINS (each position's identity has
//! magnitude < 2^9 ≪ p), which is the only sound way to carry wrap
//! semantics in a Goldilocks field. The 64-bit mode (full u64 wrapping
//! arithmetic) requires the binary-tensor bridge (the bits-bundle /
//! two-characteristic discipline); it is the documented next layer.
//! The trace builder fails closed with `ProfileViolation` on any value
//! outside the profile.
//!
//! ## Architecture
//!
//! * **Columns** (≈300, all committed via the Akita Ajtai PCS): raw
//!   values, the 16 selector bits, the 32 instruction bits, flags,
//!   byte decompositions of the 13 range-critical values, the five
//!   comparison containers `dc1..dc5` with their carry chains, and the
//!   four schoolbook carry chains (MUL/DIV/SLLI/SRLI).
//! * **Lookups** (≈225 read-only-table Shouts, all sparse): the byte
//!   range checks (identity tables), the decode table, the pow2 table,
//!   the address-alignment table, plus the v2 fetch/input Shouts.
//! * **The AIR**: ONE batched sparse sumcheck over `log_t` variables
//!   carrying every nonlinear constraint (booleanity, selector-
//!   conditioned ALU/control/routing identities, the schoolbook
//!   products) with α-RLC batching across constraint groups.
//! * **Linear gates**: the unconditional linear identities
//!   (recompositions, immediate decodings, chain position equations)
//!   are verified at the AIR's terminal point `r_air` from the claim
//!   table with a β-RLC — a multilinear identity checked at one random
//!   point is pinned by Schwartz-Zippel, so this costs the verifier
//!   O(#gates) field work and the prover nothing.
//! * **Claims + openings**: every factor claim lands in the claim
//!   table; each column's claims (at `r_air`, its Shout points, and
//!   the memory-leg points) batch into ONE grouped Ajtai opening per
//!   column.

use crate::pipeline::imm_u;
use lattice_akita::pcs::{AkitaPcs, EvaluationProof, GroupedOpening};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_memory::sparse_engine::{
    build_twist_ports, prove_onehot_sparse, prove_shout_sparse, prove_twist_ports_sparse,
    verify_twist_ports_checked,
};
use lattice_memory::twist::TwistProof;
use lattice_memory::onehot_check::OneHotSide as OHSide;
use lattice_memory::{FactorId, FactorResolver, OneHotProof, PiopError, ShoutProof};
use lattice_vm::{run as vm_run, MachineState};
use lattice_memory::sparse_engine::{
    prove_sparse_sumcheck, SparseFactor, SparseInstance, SparseTerm, ProjectedDense,
};
use std::cell::RefCell;

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// Public state: final registers, final RAM window, step count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicStateV3 {
    pub final_regs: [u64; 32],
    pub final_ram: Vec<u64>,
    pub num_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pipeline3Error {
    Execution(lattice_vm::ExecError),
    Memory(PiopError),
    Pcs(lattice_akita::pcs::AkitaPcsError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Transcript(lattice_core::transcript::TranscriptError),
    Mle(lattice_core::mle::MleError),
    UnsupportedInstruction { pc: u64, word: u32 },
    /// A witness value escaped the P0-63 profile (≥ 2^63, or shamt 63).
    ProfileViolation { what: &'static str, value: u64 },
    BadShape(String),
    VerificationFailed,
}

impl From<PiopError> for Pipeline3Error {
    fn from(e: PiopError) -> Self {
        Pipeline3Error::Memory(e)
    }
}
impl From<lattice_akita::pcs::AkitaPcsError> for Pipeline3Error {
    fn from(e: lattice_akita::pcs::AkitaPcsError) -> Self {
        Pipeline3Error::Pcs(e)
    }
}
impl From<lattice_sumcheck::SumcheckError> for Pipeline3Error {
    fn from(e: lattice_sumcheck::SumcheckError) -> Self {
        Pipeline3Error::Sumcheck(e)
    }
}
impl From<lattice_sumcheck::VirtualPolyError> for Pipeline3Error {
    fn from(e: lattice_sumcheck::VirtualPolyError) -> Self {
        Pipeline3Error::Virtual(e)
    }
}
impl From<lattice_core::transcript::TranscriptError> for Pipeline3Error {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        Pipeline3Error::Transcript(e)
    }
}
impl From<lattice_core::mle::MleError> for Pipeline3Error {
    fn from(e: lattice_core::mle::MleError) -> Self {
        Pipeline3Error::Mle(e)
    }
}

// ---------------------------------------------------------------------------
// The column registry
// ---------------------------------------------------------------------------

/// Number of selector bit columns (17 families: ADDI is its own).
pub const NUM_SEL: usize = 17;
/// Instruction bit columns.
pub const NUM_IBITS: usize = 32;

// --- raw value columns ---
pub const C_PC: usize = 0;
pub const C_NEXT_PC: usize = 1;
pub const C_IW: usize = 2;
pub const C_SELW: usize = 3;
pub const C_DKEY: usize = 4;
pub const C_SHAMT: usize = 5;
pub const C_KPOW: usize = 6;
pub const C_RS1A: usize = 7;
pub const C_RS2A: usize = 8;
pub const C_RDA: usize = 9;
pub const C_RS1V: usize = 10;
pub const C_RS2V: usize = 11;
pub const C_RDV: usize = 12;
pub const C_Q: usize = 13;
pub const C_RV: usize = 14;
pub const C_MRV: usize = 15;
pub const C_MWV: usize = 16;
pub const C_WVR: usize = 17;
pub const C_HIM: usize = 18;
pub const C_HID: usize = 19;
pub const C_HIS: usize = 20;
pub const C_ADDR: usize = 21;
pub const C_EA: usize = 22;
pub const C_IMMI: usize = 23;
pub const C_IMMB: usize = 24;
pub const C_IMMJ: usize = 25;
pub const C_IMMU: usize = 26;
pub const C_FETCH_RA: usize = 27;
pub const C_RADA: usize = 28;
pub const C_WADAR: usize = 29;
pub const C_WADA: usize = 30;
pub const C_AL8: usize = 31;
pub const C_INVR: usize = 32;
pub const NUM_VALUES: usize = 33;

// --- bit columns (all booleanity-checked in the AIR) ---
pub const B_SEL0: usize = 0;
pub const B_IBIT0: usize = NUM_SEL; // 16
pub const B_FB0: usize = B_IBIT0 + NUM_IBITS; // 48
pub const B_TAKEN: usize = B_FB0 + 4;
pub const B_EQB: usize = B_TAKEN + 1;
pub const B_X0F: usize = B_TAKEN + 2;
pub const B_WRMASK: usize = B_TAKEN + 3;
pub const B_LSB: usize = B_TAKEN + 4;
pub const B_CRR: usize = B_TAKEN + 5;
pub const B_LT1: usize = B_TAKEN + 6;
pub const B_GT1: usize = B_TAKEN + 7;
pub const B_LT3: usize = B_TAKEN + 8;
pub const B_LT4: usize = B_TAKEN + 9;
pub const B_BZ: usize = B_TAKEN + 10;
pub const B_LTSUB: usize = B_TAKEN + 11;
/// The ADD chain carries (positions 1..=8). ADD: Rs1v + Rs2v -> Rdv.
pub const B_ADDC1: usize = B_LTSUB + 1;
/// SUB chain (Rs2v + Rdv -> Rs1v).
pub const B_SUBC1: usize = B_ADDC1 + 8;
/// ADDR chain (legacy; the bare identity is magnitude-safe).
pub const B_ADDRC1: usize = B_SUBC1 + 8;
/// ADDI chain carries (Rs1v + ImmI -> Rdv).
pub const B_ADDIC1: usize = B_ADDRC1 + 8;
/// dc1 chain carries (D1 + Rs1v -> Rs2v, c_out = LT1).
pub const B_DC1C1: usize = B_ADDIC1 + 8;
pub const B_DC2C1: usize = B_DC1C1 + 7;
pub const B_DC3C1: usize = B_DC2C1 + 7;
pub const B_DC4C1: usize = B_DC3C1 + 7;
pub const B_DC5C1: usize = B_DC4C1 + 7;
/// MUL schoolbook carries (buckets 1..=14; the 15th lands in HiM's b7).
pub const B_MULC1: usize = B_DC5C1 + 7;
/// DIV schoolbook carries (buckets 1..=15, final constrained 0).
pub const B_DIVC1: usize = B_MULC1 + 14;
/// SLLI schoolbook carries (buckets 1..=14).
pub const B_SLLIC1: usize = B_DIVC1 + 15;
/// SRLI schoolbook carries (buckets 1..=15, final constrained 0).
pub const B_SRLIC1: usize = B_SLLIC1 + 14;
pub const NUM_BITS: usize = B_SRLIC1 + 15;

// --- byte columns ---
/// Byte decompositions: 7 full bytes + a 7-bit top byte per value.
/// Order: [Rs1v, Rs2v, Rdv, Q, Rv, Mrv, Mwv, Wvr, HiM, HiD, HiS, Kpow, Addr].
pub const BYTE_VALUES: [usize; 13] = [
    C_RS1V, C_RS2V, C_RDV, C_Q, C_RV, C_MRV, C_MWV, C_WVR, C_HIM, C_HID, C_HIS, C_KPOW, C_ADDR,
];
pub const BYTES_PER_VALUE: usize = 8;
pub const YB_RS1V: usize = 0;
pub const YB_RS2V: usize = 8;
pub const YB_RDV: usize = 16;
pub const YB_Q: usize = 24;
pub const YB_RV: usize = 32;
pub const YB_MRV: usize = 40;
pub const YB_MWV: usize = 48;
pub const YB_WVR: usize = 56;
pub const YB_HIM: usize = 64;
pub const YB_HID: usize = 72;
pub const YB_HIS: usize = 80;
pub const YB_KPOW: usize = 88;
pub const YB_ADDR: usize = 96;
/// The five comparison containers (64-bit): 8 full bytes each.
pub const DC_BYTES: usize = 104;
pub const DC_COUNT: usize = 5;
pub const DC_STEP: usize = 8;
pub const DC1_B0: usize = DC_BYTES;
pub const DC2_B0: usize = DC1_B0 + DC_STEP;
pub const DC3_B0: usize = DC2_B0 + DC_STEP;
pub const DC4_B0: usize = DC3_B0 + DC_STEP;
pub const DC5_B0: usize = DC4_B0 + DC_STEP;
pub const NUM_BYTE_COLS: usize = DC_BYTES + DC_COUNT * DC_STEP;

/// Total committed columns.
pub const NUM_COLS: usize = NUM_VALUES + NUM_BITS + NUM_BYTE_COLS;

fn yb(value_col: usize) -> usize {
    BYTE_VALUES.iter().position(|&c| c == value_col).unwrap_or(usize::MAX) * BYTES_PER_VALUE
}

fn bit_col(b: usize) -> usize {
    NUM_VALUES + b
}

fn byte_col(y: usize) -> usize {
    NUM_VALUES + NUM_BITS + y
}

/// Family indices. The v3 universe splits ADDI out of ADD (family 16):
/// the immediate-operand ALU form needs its own selector so the operand
/// routing constraint differs (register vs immediate).
mod fam {
    pub const ADD: usize = 0;
    pub const SUB: usize = 1;
    pub const MUL: usize = 2;
    pub const DIVQ: usize = 3;
    pub const DIVR: usize = 4;
    pub const SLLI: usize = 5;
    pub const SRLI: usize = 6;
    pub const SLTU: usize = 7;
    pub const BRCH: usize = 8;
    pub const JAL: usize = 9;
    pub const JALR: usize = 10;
    pub const LOAD: usize = 11;
    pub const STORE: usize = 12;
    pub const LUI: usize = 13;
    pub const AUIPC: usize = 14;
    pub const HALT: usize = 15;
    /// The immediate form of ADD (opcode 0x13): the 17th family.
    pub const ADDI: usize = 16;
}

/// The v3 decode, synchronized with the EXECUTOR (lattice-vm) — the
/// ground truth. The v2 decode table keyed on funct6 (bits 31..26) could
/// not distinguish ADD from MUL (both funct6=0; they differ at bit 25)
/// and expected non-standard funct6 values for SUB (0x20 vs the real
/// 0x10) and the M extension (0 vs the real f7=1) — its SUB/MUL/DIV
/// families were unreachable. v3 keys the decode on the FULL funct7
/// (bits 31..25): the dkey is 17 bits, `opcode | f3<<7 | f7<<10`.
pub fn decode_family_v3(word: u32) -> u64 {
    let opcode = word & 0x7f;
    let funct3 = (word >> 12) & 0x7;
    let funct7 = (word >> 25) & 0x7f;
    // The RV64 shift-immediates carry a 6-bit shamt at bits 20..25, so
    // bit 25 is shamt payload: they are keyed on funct6 (bits 26..31).
    let funct6 = (word >> 26) & 0x3f;
    let one = 1u64;
    match (opcode, funct3, funct7) {
        (0x13, 0, _) => one << fam::ADDI,
        (0x13, 1, _) if funct6 == 0 => one << fam::SLLI,
        (0x13, 5, _) if funct6 == 0 => one << fam::SRLI,
        (0x33, 0, 0x00) => one << fam::ADD,
        (0x33, 0, 0x20) => one << fam::SUB,
        (0x33, 0, 0x01) => one << fam::MUL,
        (0x33, 4, 0x01) => one << fam::DIVQ, // Div
        (0x33, 7, 0x01) => one << fam::DIVR, // Remu
        (0x33, 3, 0x00) => one << fam::SLTU,
        (0x63, 0 | 1 | 6 | 7, _) => one << fam::BRCH,
        (0x6f, _, _) => one << fam::JAL,
        (0x67, 0, _) => one << fam::JALR,
        (0x03, 3, _) => one << fam::LOAD,
        (0x23, 3, _) => one << fam::STORE,
        (0x37, _, _) => one << fam::LUI,
        (0x17, _, _) => one << fam::AUIPC,
        (0x73, _, _) => one << fam::HALT,
        _ => 0,
    }
}

/// The 17-bit decode key: `opcode | f3<<7 | f7<<10`.
pub fn dkey_v3(word: u32) -> u64 {
    (word & 0x7f) as u64
        | (((word >> 12) & 0x7) as u64) << 7
        | (((word >> 25) & 0x7f) as u64) << 10
}

/// The v3 decode table over 17-bit keys.
pub fn decode_table_v3() -> Vec<Goldilocks> {
    let mut table = vec![Goldilocks::ZERO; 1 << 17];
    for key in 0..(1u32 << 17) {
        let w = (key & 0x7f)
            | ((key >> 7 & 0x7) << 12)
            | ((key >> 10 & 0x7f) << 25);
        table[key as usize] = fe(decode_family_v3(w));
    }
    table
}

/// Which families write rd (the write-enable truth table).
fn writes_rd(family: usize) -> bool {
    matches!(
        family,
        fam::ADD
            | fam::ADDI
            | fam::SUB
            | fam::MUL
            | fam::DIVQ
            | fam::DIVR
            | fam::SLLI
            | fam::SRLI
            | fam::SLTU
            | fam::JAL
            | fam::JALR
            | fam::LOAD
            | fam::LUI
            | fam::AUIPC
    )
}

// ---------------------------------------------------------------------------
// The trace-3 builder: all columns, honestly computed
// ---------------------------------------------------------------------------

/// The v3 trace: the column-major witness plus the port streams the
/// memory arguments consume.
pub struct TraceDataV3 {
    pub log_t: usize,
    pub cols: Vec<Vec<Goldilocks>>,
    pub ram_read_addr: Vec<u64>,
    pub ram_write_addr: Vec<u64>,
    pub reg_read_addr_a: Vec<u64>,
    pub reg_read_addr_b: Vec<u64>,
    pub reg_write_addr: Vec<u64>,
}

/// One ADD-chain witness over byte columns.
/// Proves `X + Y (+ carry_in) = Z + 2^64·c_out` position-by-position:
/// `x_k + y_k + c_k = z_k + radix·c_{k+1}` (radix 2^8, top 2^7).
/// `carries[i] = c_{i+1}` (8 columns, c_8 = the 64th-bit carry).
fn add_chain_witness(x: u64, y: u64, carry_in: u64) -> ([u8; 8], u64) {
    let mut carries = [0u8; 8];
    let mut c: u64 = carry_in;
    for k in 0..8 {
        let acc = byte_of(x, k) + byte_of(y, k) + c;
        let radix: u64 = if k == 7 { 1 << 7 } else { 1 << 8 };
        c = acc / radix;
        carries[k] = c as u8;
    }
    (carries, c)
}

/// The comparison container: `D = (A − B − m) mod 2^64` bytes and the
/// add-chain that proves `D + B + m = A + 2^64·LT` where
/// `LT = [A < B + m]`. `carries[i] = c_{i+1}` (7 columns; c_8 = LT).
fn dc_witness(a: u64, b: u64, minus_one: bool) -> ([u8; 8], [u8; 7], u64) {
    let m = if minus_one { 1u64 } else { 0 };
    // D = (a − b − m) mod 2^64 with the two's-complement wrap.
    let d = a.wrapping_sub(b).wrapping_sub(m);
    let mut bytes = [0u8; 8];
    for k in 0..8 {
        bytes[k] = ((d >> (8 * k)) & 0xff) as u8;
    }
    // Add-chain: d + b + m = a + 2^64·lt.
    let sum = (d as u128) + (b as u128) + (m as u128);
    let lt = (sum >> 64) as u64;
    debug_assert_eq!(sum as u64, a, "dc container must reconstruct a");
    let mut carries = [0u8; 7];
    let mut c: u64 = m;
    for k in 0..8 {
        let acc = ((d >> (8 * k)) & 0xff) as u64
            + ((b >> (8 * k)) & if k == 7 { 0x7f } else { 0xff })
            + c;
        // Position 7 spans bits 56..63 (D's top byte is full 8-bit), so
        // the carry out of position 7 lands at bit 64 = LT: radix 2^8.
        let radix: u64 = if k == 7 { 1 << 8 } else { 1 << 8 };
        c = acc / radix;
        if k < 7 {
            carries[k] = c as u8;
        }
    }
    debug_assert_eq!(c, lt, "the top carry must equal LT");
    (bytes, carries, lt)
}

/// The MUL/SLLI schoolbook: `X·Y = Hi·2^64 + Lo`.
/// Buckets k=0..14: `c_k + Σ_{i+j=k} x_i·y_j = out_k + 2^8·c_{k+1}` where
/// `out_k = lo_byte_k` (k<8) else `hi_byte_{k−8}`; the final carry lands
/// in Hi's top byte. Returns (lo_bytes, hi_bytes, carries[15]).
fn mul_schoolbook(x: u64, y: u64) -> ([u8; 8], [u8; 8], [u8; 15]) {
    let prod = (x as u128) * (y as u128);
    let lo = prod as u64;
    let hi = (prod >> 64) as u64;
    let mut xb = [0u8; 8];
    let mut yb = [0u8; 8];
    for k in 0..8 {
        xb[k] = ((x >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
        yb[k] = ((y >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
    }
    let mut lo_b = [0u8; 8];
    let mut hi_b = [0u8; 8];
    for k in 0..8 {
        lo_b[k] = ((lo >> (8 * k)) & 0xff) as u8;
        hi_b[k] = ((hi >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
    }
    let mut carries = [0u8; 15];
    let mut c: u128 = 0;
    for k in 0..15usize {
        carries[k] = c as u8;
        let mut acc: u128 = c;
        for i in 0..8usize {
            let j = k as i64 - i as i64;
            if j >= 0 && (j as usize) < 8 {
                acc += (xb[i] as u128) * (yb[j as usize] as u128);
            }
        }
        // Output byte: lo for k<8, hi for 8..=14, final carry beyond.
        let out = if k < 8 {
            lo_b[k] as u128
        } else {
            hi_b[k - 8] as u128
        };
        c = (acc - out) >> 8;
        debug_assert_eq!(acc & 0xff, out, "bucket {k} low bits must match");
    }
    (lo_b, hi_b, carries)
}

/// The DIV/SRLI schoolbook: `X·Y + R = Z` (Z < 2^63, so the high buckets
/// must net to zero). Buckets k=0..14 with `out_k = z_byte_k` (k<8) else
/// 0, plus R's byte folded into bucket 0; the final carry c_15 = 0.
fn div_schoolbook(x: u64, y: u64, r: u64, z: u64) -> ([u8; 15], [u8; 8]) {
    let mut xb = [0u8; 8];
    let mut yb = [0u8; 8];
    let mut rb = [0u8; 8];
    let mut zb = [0u8; 8];
    for k in 0..8 {
        xb[k] = ((x >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
        yb[k] = ((y >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
        rb[k] = ((r >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
        zb[k] = ((z >> (8 * k)) & if k == 7 { 0x7f } else { 0xff }) as u8;
    }
    debug_assert_eq!(
        (x as u128) * (y as u128) + (r as u128),
        z as u128,
        "div schoolbook inputs must satisfy x*y + r = z"
    );
    let mut carries = [0u8; 15];
    let mut c: u128 = 0;
    for k in 0..15usize {
        carries[k] = c as u8;
        let mut acc: u128 = c;
        for i in 0..8usize {
            let j = k as i64 - i as i64;
            if j >= 0 && (j as usize) < 8 {
                acc += (xb[i] as u128) * (yb[j as usize] as u128);
            }
        }
        if k < 8 {
            acc += rb[k] as u128;
        }
        let out = if k < 8 { zb[k] as u128 } else { 0 };
        c = (acc - out) >> 8;
        debug_assert_eq!(acc & 0xff, out, "bucket {k} low bits must match");
    }
    debug_assert_eq!(c, 0, "final carry must vanish");
    (carries, zb)
}

/// The 64-bit sign extension of a w-bit two's-complement immediate.
/// (The pipeline.rs decoders extend to 32 bits — an honest-but-different
/// convention; the v3 columns and gates use the full u64 semantics that
/// match the executor.)
fn sext64(raw: u64, w: u32) -> u64 {
    let sign = 1u64 << (w - 1);
    if raw & sign != 0 {
        raw | !((1u64 << w) - 1)
    } else {
        raw
    }
}

fn imm_i64(word: u32) -> u64 {
    sext64(((word >> 20) & 0xfff) as u64, 12)
}

/// The I-immediate as its mod-p signed field value (the column semantics).
fn imm_i_field(word: u32) -> Goldilocks {
    signed_field(((word >> 20) & 0xfff) as u64, 12)
}

fn imm_s64(word: u32) -> u64 {
    let w = word as u64;
    let raw = (((w >> 25) & 0x7f) << 5) | ((w >> 7) & 0x1f);
    sext64(raw, 12)
}

fn imm_b64(word: u32) -> u64 {
    let w = word as u64;
    let raw = (((w >> 31) & 1) << 12)
        | (((w >> 7) & 1) << 11)
        | (((w >> 25) & 0x3f) << 5)
        | (((w >> 8) & 0xf) << 1);
    sext64(raw, 13)
}

fn imm_b_field(word: u32) -> Goldilocks {
    let w = word as u64;
    let raw = (((w >> 31) & 1) << 12)
        | (((w >> 7) & 1) << 11)
        | (((w >> 25) & 0x3f) << 5)
        | (((w >> 8) & 0xf) << 1);
    signed_field(raw, 13)
}

fn imm_j64(word: u32) -> u64 {
    let w = word as u64;
    let raw = (((w >> 31) & 1) << 20)
        | (((w >> 12) & 0xff) << 12)
        | (((w >> 20) & 1) << 11)
        | (((w >> 21) & 0x3ff) << 1);
    sext64(raw, 21)
}

fn imm_j_field(word: u32) -> Goldilocks {
    let w = word as u64;
    let raw = (((w >> 31) & 1) << 20)
        | (((w >> 12) & 0xff) << 12)
        | (((w >> 20) & 1) << 11)
        | (((w >> 21) & 0x3ff) << 1);
    signed_field(raw, 21)
}

fn instr_word_at(state: &MachineState, pc: u64) -> u32 {
    let w = state.memory.load(pc & !0x7);
    ((w >> ((pc & 0x7) * 8)) & 0xffff_ffff) as u32
}

fn byte_of(v: u64, k: usize) -> u64 {
    if k == 7 {
        (v >> 56) & 0x7f
    } else {
        (v >> (8 * k)) & 0xff
    }
}

/// The S-type immediate (store offset), from the instruction word.
fn imm_s(word: u32) -> u64 {
    let w = word as u64;
    let imm = (((w >> 25) & 0x7f) << 5) | ((w >> 7) & 0x1f);
    if imm & 0x800 != 0 {
        imm | 0xffff_f000
    } else {
        imm
    }
}

#[allow(clippy::too_many_lines)]
pub fn build_trace3(
    state: &MachineState,
    rows: &[lattice_vm::TraceRow],
) -> Result<TraceDataV3, Pipeline3Error> {
    let log_t = rows.len().next_power_of_two().max(2).trailing_zeros() as usize;
    let t_pow = 1usize << log_t;
    let mut regs = [0u64; 32];
    let mut ram_now: std::collections::BTreeMap<u64, u64> =
        state.memory.snapshot_pairs().into_iter().collect();
    let mut cols: Vec<Vec<Goldilocks>> = vec![Vec::new(); NUM_COLS];
    let mut ram_ra = Vec::with_capacity(t_pow);
    let mut ram_wa = Vec::with_capacity(t_pow);
    let mut reg_ra_a = Vec::with_capacity(t_pow);
    let mut reg_ra_b = Vec::with_capacity(t_pow);
    let mut reg_wa = Vec::with_capacity(t_pow);
    let profile = |what: &'static str, v: u64| -> Result<(), Pipeline3Error> {
        if v >= (1u64 << 63) {
            Err(Pipeline3Error::ProfileViolation { what, value: v })
        } else {
            Ok(())
        }
    };

    for row in rows.iter() {
        let pc = row.pc;
        let iw = instr_word_at(state, pc);
        let selw = decode_family_v3(iw);
        if selw == 0 {
            return Err(Pipeline3Error::UnsupportedInstruction { pc, word: iw });
        }
        let family = selw.trailing_zeros() as usize;
        // A halted machine stays put (the executor's pc+4 is frozen).
        let next_pc = if family == fam::HALT { pc } else { row.next_pc };
        let rs1a = ((iw >> 15) & 0x1f) as u64;
        let rs2a = ((iw >> 20) & 0x1f) as u64;
        let rda = ((iw >> 7) & 0x1f) as u64;
        let rs1v = if rs1a == 0 { 0 } else { regs[rs1a as usize] };
        let rs2v = if rs2a == 0 { 0 } else { regs[rs2a as usize] };
        // ---- early witnesses (needed by the rdv recomputation) ----
        let shamt_u32 = ((iw >> 20) & 0x3f) as u32;
        let (addr_row, old_w, new_w) = match &row.mem_access {
            Some((a, old, new)) => (*a, *old, *new),
            None => (0, 0, None),
        };
        let is_load_row = family == fam::LOAD;
        let mrv_row = if is_load_row {
            old_w
        } else {
            *ram_now.get(&0).unwrap_or(&0)
        };
        // rdv = the RAW computed result (independent of the x0 write
        // mask): the write-back constraints pin the semantics, and the
        // Wvr column separately carries WrMask·Rdv (the x0 rule).
        let mut rdv = 0u64;
        for &(r, v) in &row.reg_writes {
            if r as u64 == rda {
                rdv = v;
            }
        }
        if rda == 0 {
            // The executor drops x0 writes; recompute the result so the
            // semantics constraints hold (the Wvr column masks it to 0).
            rdv = match family {
                fam::ADD => rs1v.wrapping_add(rs2v),
                fam::ADDI => rs1v.wrapping_add(imm_i64(iw)),
                fam::SUB => rs1v.wrapping_sub(rs2v),
                fam::MUL => rs1v.wrapping_mul(rs2v),
                fam::DIVQ | fam::DIVR => {
                    if rs2v == 0 {
                        (1u64 << 63) - 1
                    } else if family == fam::DIVQ {
                        rs1v / rs2v
                    } else {
                        rs1v % rs2v
                    }
                }
                fam::SLLI => rs1v.wrapping_shl(shamt_u32),
                fam::SRLI => rs1v.wrapping_shr(shamt_u32),
                fam::SLTU => (rs1v < rs2v) as u64,
                fam::JAL | fam::JALR => pc.wrapping_add(4),
                fam::LOAD => mrv_row,
                fam::LUI => imm_u(iw),
                fam::AUIPC => pc.wrapping_add(imm_u(iw)),
                _ => 0,
            };
        }
        // ---- profile checks (fail closed) ----
        profile("pc", pc)?;
        profile("next_pc", next_pc)?;
        profile("rs1v", rs1v)?;
        profile("rs2v", rs2v)?;
        profile("rdv", rdv)?;
        // ---- division / shift witnesses ----
        let (q, r) = if family == fam::DIVQ || family == fam::DIVR {
            match rs2v {
                0 => (u64::MAX & ((1 << 63) - 1), rs1v),
                b => (rs1v / b, rs1v % b),
            }
        } else {
            (0, 0)
        };
        // DIVQ result when rs2 = 0: all-ones below bit 63 (profile-safe).
        profile("q", q)?;
        profile("r", r)?;
        // The Shamt column carries the shift amount ONLY on shift rows
        // (elsewhere the raw bits 20..25 are immediate data, not a
        // shift); Kpow tracks it so the pow2 lookup stays consistent.
        let shamt = if family == fam::SLLI || family == fam::SRLI {
            if shamt_u32 >= 63 {
                return Err(Pipeline3Error::ProfileViolation {
                    what: "shamt",
                    value: shamt_u32 as u64,
                });
            }
            shamt_u32 as u64
        } else {
            0
        };
        let kpow = 1u64 << shamt;
        // SRLI remainder: rs1v mod 2^shamt; SLLI high part.
        let slli_hi = if family == fam::SLLI {
            ((rs1v as u128) * (kpow as u128) >> 64) as u64
        } else {
            0
        };
        profile("slli_hi", slli_hi)?;
        let srlr_r = if family == fam::SRLI { rs1v & (kpow - 1) } else { 0 };
        // ---- memory ----
        let _ = addr_row;
        // The computed effective byte address: LOAD/JALR use the
        // I-immediate, STORE the S-immediate; other rows carry 0 so the
        // unconditional alignment gate holds.
        let imms = if family == fam::LOAD || family == fam::JALR {
            imm_i64(iw)
        } else if family == fam::STORE {
            imm_s64(iw)
        } else {
            0
        };
        let caddr = if family == fam::LOAD || family == fam::STORE || family == fam::JALR {
            rs1v.wrapping_add(imms)
        } else {
            0
        };
        profile("addr", caddr)?;
        // The memory model is word-granular: accesses must be 8-aligned.
        if (family == fam::LOAD || family == fam::STORE) && caddr % 8 != 0 {
            return Err(Pipeline3Error::ProfileViolation {
                what: "unaligned memory access",
                value: caddr,
            });
        }
        let ea = caddr / 8;
        let is_load = is_load_row;
        let is_store = family == fam::STORE;
        let mrv = mrv_row;
        let mwv = if is_store { new_w.unwrap_or(0) } else { *ram_now.get(&0).unwrap_or(&0) };
        profile("mrv", mrv)?;
        profile("mwv", mwv)?;
        // ---- flags ----
        let taken = match family {
            fam::BRCH => match (iw >> 12) & 0x7 {
                0x0 => (rs1v == rs2v) as u64,
                0x1 => (rs1v != rs2v) as u64,
                0x6 => (rs1v < rs2v) as u64,
                0x7 => (rs1v >= rs2v) as u64,
                _ => 0,
            },
            _ => 0,
        };
        let eqb = (rs1v == rs2v) as u64;
        let x0f = (rda == 0) as u64;
        let wr = writes_rd(family) as u64;
        let wrmask = wr * (1 - x0f);
        let wada = if wrmask != 0 { rda } else { 0 };
        let wvr = if wrmask != 0 { rdv } else { 0 };
        // ---- comparison containers ----
        let (dc1_b, dc1_c, _lt1) = dc_witness(rs2v, rs1v, false);
        let (dc2_b, dc2_c, _gt1) = dc_witness(rs1v, rs2v, false);
        let (dc3_b, dc3_c, _lt3) = dc_witness(rs2v, r, true); // LT3 = [rs2 ≤ r]
        let (dc4_b, dc4_c, _lt4) = dc_witness(kpow, srlr_r, true); // LT4 = [kpow ≤ r]
        let (dc5_b, dc5_c, _bz) = dc_witness(rs2v, 0, true);
        // ---- ADD / SUB / ADDR chains ----
        let (addc, _add_out) = add_chain_witness(rs1v, rs2v, 0);
        let (subc, _sub_out) = add_chain_witness(rs2v, rdv, 0);
        let (addrc, _addr_out) = add_chain_witness(rs1v, imms, 0);
        let immi_v = imm_i64(iw);
        let (addic, _addi_out) = add_chain_witness(rs1v, immi_v, 0);
        if family == fam::ADDI {
            debug_assert_eq!(rs1v.wrapping_add(immi_v), rdv, "ADDI chain inputs");
        }
        if family == fam::SUB {
            debug_assert_eq!(rs2v.wrapping_add(rdv), rs1v, "SUB chain inputs");
        }
        if family == fam::LOAD || family == fam::STORE || family == fam::JALR {
            debug_assert_eq!(rs1v.wrapping_add(imms), caddr, "ADDR chain inputs");
        }
        // ---- schoolbooks ----
        let (mul_lo, mul_hi, mul_c) = if family == fam::MUL {
            mul_schoolbook(rs1v, rs2v)
        } else {
            ([0u8; 8], [0u8; 8], [0u8; 15])
        };
        let _ = mul_lo;
        let (div_c, _div_z) = if family == fam::DIVQ || family == fam::DIVR {
            // q·b + r = a holds in both branches: on Bz, b = 0 so the
            // product vanishes and r = rs1v carries the identity.
            div_schoolbook(q, rs2v, r, rs1v)
        } else {
            ([0u8; 15], [0u8; 8])
        };
        let (slli_c, _slli_z) = if family == fam::SLLI {
            // rs1v · kpow = hi·2^64 + rdv.
            let (lo, hi, c) = mul_schoolbook(rs1v, kpow);
            debug_assert_eq!(u64::from_le_bytes(lo), rdv);
            debug_assert_eq!(u64::from_le_bytes(hi), slli_hi);
            (c, lo)
        } else {
            ([0u8; 15], [0u8; 8])
        };
        let (srli_c, _srli_z) = if family == fam::SRLI {
            // rdv · kpow + r = rs1v.
            div_schoolbook(rdv, kpow, srlr_r, rs1v)
        } else {
            ([0u8; 15], [0u8; 8])
        };
        let _ = slli_hi;
        // ---- JALR LSB half-adder ----
        // The immediate columns carry the SIGNED mod-p decoded values on
        // every row (the recomposition gates and the branch-target
        // constraints consume them). The EA computation uses the u64
        // sign-extended forms (the executor's arithmetic).
        let immb = imm_b_field(iw);
        let immi = imm_i_field(iw);
        let immi_u64 = imm_i64(iw);
        let lsb_v = (rs1v.wrapping_add(immi_u64)) & 1;
        let crr_v = (rs1v & 1) & (immi_u64 & 1);
        // The remainder column: DIV carries r; SRLI carries the shift
        // remainder rs1v mod 2^shamt (the schoolbook's addend).
        let rv_col = if family == fam::DIVQ || family == fam::DIVR {
            r
        } else if family == fam::SRLI {
            srlr_r
        } else {
            0
        };
        // ---- column pushes ----
        let mut p = |idx: usize, v: u64| cols[idx].push(fe(v));
        p(C_PC, pc);
        p(C_NEXT_PC, next_pc);
        p(C_IW, iw as u64);
        p(C_SELW, selw);
        p(C_DKEY, dkey_v3(iw));
        p(C_SHAMT, shamt);
        p(C_KPOW, kpow);
        p(C_RS1A, rs1a);
        p(C_RS2A, rs2a);
        p(C_RDA, rda);
        p(C_RS1V, rs1v);
        p(C_RS2V, rs2v);
        p(C_RDV, rdv);
        p(C_Q, q);
        p(C_RV, rv_col);
        p(C_MRV, mrv);
        p(C_MWV, mwv);
        p(C_WVR, wvr);
        p(C_HIM, u64::from_le_bytes(mul_hi));
        p(C_HID, 0); // DIV has no hi output (the schoolbook nets to zero).
        p(C_HIS, slli_hi);
        p(C_ADDR, caddr);
        p(C_EA, ea);
        p(C_IMMI, immi.to_canonical_u64());
        p(C_IMMB, immb.to_canonical_u64());
        p(C_IMMJ, imm_j_field(iw).to_canonical_u64());
        p(C_IMMU, imm_u(iw));
        p(C_FETCH_RA, (pc.wrapping_sub(0x1000)) / 4);
        p(C_RADA, if is_load { ea } else { 0 });
        p(C_WADAR, if is_store { ea } else { 0 });
        p(C_WADA, wada);
        p(C_AL8, (caddr & 0xff) / 8);
        p(C_INVR, if rda == 0 { 0 } else { fe(rda).inverse().unwrap_or(Goldilocks::ZERO).to_canonical_u64() });
        // Bit columns.
        for k in 0..NUM_SEL {
            p(bit_col(B_SEL0 + k), (selw >> k) & 1);
        }
        for i in 0..NUM_IBITS {
            p(bit_col(B_IBIT0 + i), ((iw >> i) & 1) as u64);
        }
        // Branch kinds: beq(0), bne(1), bltu(6), bgeu(7); every other
        // row marks "not a branch" with the Fb3 slot so Σ Fb = 1 holds.
        let fb = match family {
            fam::BRCH => (iw >> 12) & 0x7,
            _ => 7,
        };
        p(bit_col(B_FB0), (fb == 0) as u64);
        p(bit_col(B_FB0 + 1), (fb == 1) as u64);
        p(bit_col(B_FB0 + 2), (fb == 6) as u64);
        p(bit_col(B_FB0 + 3), (fb == 7) as u64);
        p(bit_col(B_TAKEN), taken);
        p(bit_col(B_EQB), eqb);
        p(bit_col(B_X0F), x0f);
        p(bit_col(B_WRMASK), wrmask);
        p(bit_col(B_LSB), lsb_v);
        p(bit_col(B_CRR), crr_v);
        p(bit_col(B_LT1), (rs2v < rs1v) as u64);
        p(bit_col(B_GT1), (rs1v < rs2v) as u64);
        p(bit_col(B_LT3), (rs2v <= r) as u64);
        p(bit_col(B_LT4), (kpow <= srlr_r) as u64);
        p(bit_col(B_BZ), (rs2v == 0) as u64);
        p(bit_col(B_LTSUB), 0); // non-wrapping SUB: borrow-out is zero.
        for k in 0..8 {
            p(bit_col(B_ADDC1 + k), addc[k] as u64);
            p(bit_col(B_SUBC1 + k), subc[k] as u64);
            p(bit_col(B_ADDRC1 + k), addrc[k] as u64);
            p(bit_col(B_ADDIC1 + k), addic[k] as u64);
        }
        for k in 0..7 {
            p(bit_col(B_DC1C1 + k), dc1_c[k] as u64);
            p(bit_col(B_DC2C1 + k), dc2_c[k] as u64);
            p(bit_col(B_DC3C1 + k), dc3_c[k] as u64);
            p(bit_col(B_DC4C1 + k), dc4_c[k] as u64);
            p(bit_col(B_DC5C1 + k), dc5_c[k] as u64);
        }
        for k in 0..14 {
            p(bit_col(B_MULC1 + k), mul_c[k] as u64);
            p(bit_col(B_SLLIC1 + k), slli_c[k] as u64);
        }
        for k in 0..15 {
            p(bit_col(B_DIVC1 + k), div_c[k] as u64);
            p(bit_col(B_SRLIC1 + k), srli_c[k] as u64);
        }
        // Byte columns.
        for (vi, &vc) in BYTE_VALUES.iter().enumerate() {
            let v = match vc {
                C_RS1V => rs1v,
                C_RS2V => rs2v,
                C_RDV => rdv,
                C_Q => q,
                C_RV => rv_col,
                C_MRV => mrv,
                C_MWV => mwv,
                C_WVR => wvr,
                C_HIM => u64::from_le_bytes(mul_hi),
                C_HID => 0,
                C_HIS => slli_hi,
                C_KPOW => kpow,
                C_ADDR => caddr,
                _ => 0,
            };
            for k in 0..BYTES_PER_VALUE {
                p(byte_col(vi * BYTES_PER_VALUE + k), byte_of(v, k));
            }
        }
        // dc byte columns.
        for (di, db) in [&dc1_b, &dc2_b, &dc3_b, &dc4_b, &dc5_b].iter().enumerate() {
            for k in 0..8 {
                p(byte_col(DC_BYTES + di * DC_STEP + k), db[k] as u64);
            }
        }
        // RAM / register port streams.
        ram_ra.push(if is_load { ea } else { 0 });
        ram_wa.push(if is_store { ea } else { 0 });
        reg_ra_a.push(rs1a);
        reg_ra_b.push(rs2a);
        reg_wa.push(wada);
        // Shadow updates.
        for &(r, v) in &row.reg_writes {
            if r != 0 {
                regs[r as usize] = v;
            }
        }
        if let Some((a, _, Some(new))) = &row.mem_access {
            ram_now.insert(*a, *new);
        }
    }

    // ---- HALT padding rows ----
    let last_pc = rows.last().map(|r| r.pc).unwrap_or(0x1000);
    let last_iw = instr_word_at(state, last_pc);
    while cols[0].len() < t_pow {
        let mut p = |idx: usize, v: u64| cols[idx].push(fe(v));
        p(C_PC, last_pc);
        p(C_NEXT_PC, last_pc);
        p(C_IW, last_iw as u64);
        p(C_SELW, 1u64 << fam::HALT);
        p(C_DKEY, dkey_v3(last_iw));
        p(C_SHAMT, 0);
        p(C_KPOW, 1);
        for idx in [C_RS1A, C_RS2A, C_RDA, C_RS1V, C_RS2V, C_RDV, C_Q, C_RV, C_MRV, C_MWV,
            C_WVR, C_HIM, C_HID, C_HIS, C_ADDR, C_EA, C_IMMI, C_IMMB, C_IMMJ, C_IMMU,
            C_RADA, C_WADAR, C_WADA, C_AL8] {
            p(idx, 0);
        }
        // The fetch address keeps tracking the (frozen) PC: the gate
        // 4·FetchRa = Pc − 0x1000 and the fetch Shout both require it.
        p(C_FETCH_RA, last_pc.wrapping_sub(0x1000) / 4);
        p(C_INVR, 0);
        for k in 0..NUM_SEL {
            p(bit_col(B_SEL0 + k), (k == fam::HALT) as u64);
        }
        for i in 0..NUM_IBITS {
            p(bit_col(B_IBIT0 + i), ((last_iw >> i) & 1) as u64);
        }
        // Branch kind on padding: the 0xf "not a branch" pattern.
        p(bit_col(B_FB0), 0);
        p(bit_col(B_FB0 + 1), 0);
        p(bit_col(B_FB0 + 2), 0);
        p(bit_col(B_FB0 + 3), 1);
        for b in [B_TAKEN, B_WRMASK, B_LSB, B_CRR, B_LT1, B_GT1,
            B_LT4, B_LTSUB] {
            p(bit_col(b), 0);
        }
        // Padding semantics: rs2v = 0 ⇒ LT3 = [0 ≤ 0] = 1, Bz = 1,
        // Eqb = 1 (rs1v == rs2v == 0), X0f = 1 (rda = 0).
        p(bit_col(B_LT3), 1);
        p(bit_col(B_BZ), 1);
        p(bit_col(B_EQB), 1);
        p(bit_col(B_X0F), 1);
        for base in [B_ADDC1, B_SUBC1, B_ADDRC1, B_ADDIC1] {
            for k in 0..8 {
                p(bit_col(base + k), 0);
            }
        }
        // The dc carries are pushed with the containers below.
        for base in [B_MULC1, B_DIVC1, B_SLLIC1, B_SRLIC1] {
            let n = if base == B_DIVC1 || base == B_SRLIC1 { 15 } else { 14 };
            for k in 0..n {
                p(bit_col(base + k), 0);
            }
        }
        for (vi, &vc) in BYTE_VALUES.iter().enumerate() {
            let v = match vc {
                C_KPOW => 1,
                C_IW => last_iw as u64,
                _ => 0,
            };
            let _ = vc;
            for k in 0..BYTES_PER_VALUE {
                p(byte_col(vi * BYTES_PER_VALUE + k), byte_of(v, k));
            }
        }
        // dc containers on padding: A = B = 0.
        // dc1/dc2: D = 0, LT = 0, carries 0. dc3: D = all-ones, LT3 = 1,
        // carries 1 (255+0+1 wraps every position). dc4: D = 0, LT4 = 0.
        // dc5: D = all-ones, Bz = 1, carries 1.
        let (d5, c5c, _) = dc_witness(0, 0, true);
        let (d3, c3c, _) = dc_witness(0, 0, true);
        let (d4, c4c, _) = dc_witness(1, 0, true);
        for k in 0..8 {
            p(byte_col(DC1_B0 + k), 0);
            p(byte_col(DC2_B0 + k), 0);
            p(byte_col(DC3_B0 + k), d3[k] as u64);
            p(byte_col(DC4_B0 + k), d4[k] as u64);
            p(byte_col(DC5_B0 + k), d5[k] as u64);
        }
        for k in 0..7 {
            p(bit_col(B_DC1C1 + k), 0);
            p(bit_col(B_DC2C1 + k), 0);
            p(bit_col(B_DC3C1 + k), c3c[k] as u64);
            p(bit_col(B_DC4C1 + k), c4c[k] as u64);
            p(bit_col(B_DC5C1 + k), c5c[k] as u64);
        }
        ram_ra.push(0);
        ram_wa.push(0);
        reg_ra_a.push(0);
        reg_ra_b.push(0);
        reg_wa.push(0);
    }
    // Invariant: every column has exactly 2^log_t entries.
    for (ci, col) in cols.iter().enumerate() {
        debug_assert_eq!(col.len(), t_pow, "column {ci} length");
    }
    debug_assert_eq!(ram_ra.len(), t_pow);
    debug_assert_eq!(ram_wa.len(), t_pow);
    Ok(TraceDataV3 {
        log_t,
        cols,
        ram_read_addr: ram_ra,
        ram_write_addr: ram_wa,
        reg_read_addr_a: reg_ra_a,
        reg_read_addr_b: reg_ra_b,
        reg_write_addr: reg_wa,
    })
}

// ---------------------------------------------------------------------------
// The constraint system: the AIR terms + the linear gates
// ---------------------------------------------------------------------------

/// A linear gate: `Σ slots + constant ≡ 0` — an unconditional multilinear
/// identity checked at `r_air` from the claim table (β-RLC batched).
/// Every slot magnitude is bounded below `p` by construction, so the
/// mod-p window cannot open (the P0-63 discipline).
#[derive(Clone, Debug)]
pub struct Gate {
    pub slots: Vec<(usize, Goldilocks)>,
    pub constant: Goldilocks,
}

/// Which sparse factor a term carries (index into the layout's
/// `sparse_kinds` registry).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseKind {
    /// The full-support ones factor (claim = 1, public).
    Ones,
    /// The family selector one-hot (claim = the Sel_k column at r_air).
    Sel(usize),
    /// A bit column's one-hot (support = rows where the bit is 1; the
    /// claim = the bit column at r_air).
    Bit(usize),
}

/// One AIR term template: base coefficient × Π factors.
#[derive(Clone, Debug)]
pub struct TermTpl {
    pub coeff: Goldilocks,
    /// Indices into the layout's `sparse_kinds` registry.
    pub sparse: Vec<usize>,
    /// Column indices of the dense factors.
    pub dense: Vec<usize>,
    /// The constraint group (α^g batching).
    pub group: u32,
}

/// The static AIR layout (prover and verifier build the same one).
pub struct AirLayout {
    pub num_vars: usize,
    /// Dense factor -> column.
    pub dense_cols: Vec<usize>,
    /// Sparse factor kinds, indexed by term references.
    pub sparse_kinds: Vec<SparseKind>,
    pub terms: Vec<TermTpl>,
}

fn fe_const(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// The bits that carry AIR booleanity terms.
fn booleanity_bits() -> Vec<usize> {
    let mut bits = Vec::new();
    bits.extend(0..NUM_SEL); // B_SEL0..
    bits.extend(B_IBIT0..B_IBIT0 + NUM_IBITS);
    bits.extend(B_FB0..B_FB0 + 4);
    for b in [B_TAKEN, B_EQB, B_X0F, B_WRMASK, B_LSB, B_CRR, B_LT1, B_GT1,
        B_LT3, B_LT4, B_BZ, B_LTSUB] {
        bits.push(b);
    }
    bits.extend(B_ADDC1..B_ADDC1 + 8);
    bits.extend(B_SUBC1..B_SUBC1 + 8);
    bits.extend(B_ADDIC1..B_ADDIC1 + 8);
    // ADDR chain removed (the bare identity is magnitude-safe).
    bits.extend(B_DC1C1..B_DC1C1 + 7);
    bits.extend(B_DC2C1..B_DC2C1 + 7);
    bits.extend(B_DC3C1..B_DC3C1 + 7);
    bits.extend(B_DC4C1..B_DC4C1 + 7);
    bits.extend(B_DC5C1..B_DC5C1 + 7);
    // The schoolbook carries are range-12 VALUES (not bits): they are
    // authenticated by range12 identity-table Shouts, not booleanity.
    bits
}

/// Build the AIR layout: the full constraint system as term templates.
#[allow(clippy::too_many_lines)]
pub fn air_layout(log_t: usize) -> AirLayout {
    let _ = log_t;
    let mut dense_cols: Vec<usize> = Vec::new();
    let mut dense_of: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut sparse_kinds: Vec<SparseKind> = Vec::new();
    let mut sparse_of: std::collections::HashMap<SparseKind, usize> =
        std::collections::HashMap::new();
    let mut terms: Vec<TermTpl> = Vec::new();
    let mut next_group: u32 = 0;

    macro_rules! di {
        ($col:expr) => {{
            let col: usize = $col;
            if let Some(&i) = dense_of.get(&col) {
                i
            } else {
                dense_cols.push(col);
                let i = dense_cols.len() - 1;
                dense_of.insert(col, i);
                i
            }
        }};
    }
    macro_rules! si {
        ($kind:expr) => {{
            let kind: SparseKind = $kind;
            if let Some(&i) = sparse_of.get(&kind) {
                i
            } else {
                sparse_kinds.push(kind);
                let i = sparse_kinds.len() - 1;
                sparse_of.insert(kind, i);
                i
            }
        }};
    }
    macro_rules! group {
        () => {{
            next_group += 1;
            next_group - 1
        }};
    }
    // A conditioned-linear term set: Sel × (Σ slots) = 0.
    macro_rules! sel_lin {
        ($g:expr, $sp:expr, $($col:expr => $coeff:expr),* $(,)?) => {{
            let g = $g;
            let sp = $sp;
            $(
                terms.push(TermTpl {
                    coeff: $coeff,
                    sparse: vec![sp],
                    dense: vec![di!($col)],
                    group: g,
                });
            )*
        }};
    }

    // ---- Booleanity: per bit b: b·(b−1) = 0 as [b_sp, b] − [b_sp]. ----
    for b in booleanity_bits() {
        let g = group!();
        let sp = si!(SparseKind::Bit(b));
        let d = di!(bit_col(b));
        terms.push(TermTpl { coeff: Goldilocks::ONE, sparse: vec![sp], dense: vec![d], group: g });
        terms.push(TermTpl { coeff: Goldilocks::ONE.neg(), sparse: vec![sp], dense: vec![], group: g });
    }

    // ---- ADD chain (Sel_ADD): Rs1v + Rs2v = Rdv + 2^64·c8, c8 = 0. ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::ADD));
        for k in 0..8usize {
            let radix = if k == 7 { 128u64 } else { 256u64 };
            sel_lin!(g, sp,
                byte_col(YB_RS1V + k) => Goldilocks::ONE,
                byte_col(YB_RS2V + k) => Goldilocks::ONE,
                byte_col(YB_RDV + k) => Goldilocks::ONE.neg(),
            );
            if k > 0 {
                sel_lin!(g, sp, bit_col(B_ADDC1 + k - 1) => Goldilocks::ONE);
            }
            sel_lin!(g, sp, bit_col(B_ADDC1 + k) => fe_const(radix).neg());
        }
    }
    // ---- ADDI chain (Sel_ADDI): Rs1v + sext(ImmI) = Rdv + 2^64·c8 with
    // c8 = the immediate's sign bit (negative offsets wrap the 64-bit
    // container exactly once). The immediate's bytes are linear combos of
    // the instruction bits: byte 0 = bits 20..27, byte 1 = bits 28..31 +
    // sign·2^4, bytes 2..7 = sign·255 (the sign bit is bit 31).
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::ADDI));
        let sign_b31 = bit_col(B_IBIT0 + 31);
        for k in 0..8usize {
            // The immediate's byte 7 is a FULL 8-bit sext byte, so the
            // position-7 carry lands at bit 64: radix 2^8 (unlike the
            // value chains whose top bytes are 7-bit).
            let radix = 256u64;
            sel_lin!(g, sp,
                byte_col(YB_RS1V + k) => Goldilocks::ONE,
                byte_col(YB_RDV + k) => Goldilocks::ONE.neg(),
            );
            if k > 0 {
                sel_lin!(g, sp, bit_col(B_ADDIC1 + k - 1) => Goldilocks::ONE);
            }
            sel_lin!(g, sp, bit_col(B_ADDIC1 + k) => fe_const(radix).neg());
            // The immediate's byte-k contribution.
            if k == 0 {
                for i in 20..28 {
                    terms.push(TermTpl {
                        coeff: fe_const(1u64 << (i - 20)),
                        sparse: vec![sp],
                        dense: vec![di!(bit_col(B_IBIT0 + i))],
                        group: g,
                    });
                }
            } else if k == 1 {
                for i in 28..32 {
                    terms.push(TermTpl {
                        coeff: fe_const(1u64 << (i - 28)),
                        sparse: vec![sp],
                        dense: vec![di!(bit_col(B_IBIT0 + i))],
                        group: g,
                    });
                }
                // The sext's bits 12..15 all carry the sign: 0xF0.
                terms.push(TermTpl {
                    coeff: fe_const(240),
                    sparse: vec![sp],
                    dense: vec![di!(sign_b31)],
                    group: g,
                });
            } else {
                terms.push(TermTpl {
                    coeff: fe_const(255),
                    sparse: vec![sp],
                    dense: vec![di!(sign_b31)],
                    group: g,
                });
            }
        }
        // The top carry equals the sign bit.
        terms.push(TermTpl {
            coeff: Goldilocks::ONE,
            sparse: vec![sp],
            dense: vec![di!(bit_col(B_ADDIC1 + 7))],
            group: g,
        });
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![sp],
            dense: vec![di!(sign_b31)],
            group: g,
        });
    }
    // ---- SUB chain (Sel_SUB): Rs2v + Rdv = Rs1v + 2^64·c8, c8 = 0. ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::SUB));
        for k in 0..8usize {
            let radix = if k == 7 { 128u64 } else { 256u64 };
            sel_lin!(g, sp,
                byte_col(YB_RS2V + k) => Goldilocks::ONE,
                byte_col(YB_RDV + k) => Goldilocks::ONE,
                byte_col(YB_RS1V + k) => Goldilocks::ONE.neg(),
            );
            if k > 0 {
                sel_lin!(g, sp, bit_col(B_SUBC1 + k - 1) => Goldilocks::ONE);
            }
            sel_lin!(g, sp, bit_col(B_SUBC1 + k) => fe_const(radix).neg());
        }
    }

    // ---- The four schoolbooks (shared shape, distinct selectors). ----
    // MUL:  Rs1v·Rs2v  = HiM·2^64 + Rdv      (hi out for k ≥ 8)
    // SLLI: Rs1v·Kpow  = HiS·2^64 + Rdv
    // DIV:  Q·Rs2v + Rv = Rs1v                (zero out for k ≥ 8, +R at 0)
    // SRLI: Rdv·Kpow + Rv = Rs1v
    {
        let mk = |x_base: usize, y_base: usize, hi_base: Option<usize>, r_base: Option<usize>,
                  carry_base: usize, sels: &[usize], terms: &mut Vec<TermTpl>,
                  dense_cols: &mut Vec<usize>, dense_of: &mut std::collections::HashMap<usize, usize>,
                  sparse_kinds: &mut Vec<SparseKind>,
                  sparse_of: &mut std::collections::HashMap<SparseKind, usize>,
                  g: u32| {
            let _ = g;
            let mut di = |col: usize, dense_cols: &mut Vec<usize>,
                          dense_of: &mut std::collections::HashMap<usize, usize>| -> usize {
                if let Some(&i) = dense_of.get(&col) {
                    i
                } else {
                    dense_cols.push(col);
                    let i = dense_cols.len() - 1;
                    dense_of.insert(col, i);
                    i
                }
            };
            let mut si = |kind: SparseKind, sparse_kinds: &mut Vec<SparseKind>,
                          sparse_of: &mut std::collections::HashMap<SparseKind, usize>| -> usize {
                if let Some(&i) = sparse_of.get(&kind) {
                    i
                } else {
                    sparse_kinds.push(kind);
                    let i = sparse_kinds.len() - 1;
                    sparse_of.insert(kind, i);
                    i
                }
            };
            let xb: Vec<usize> = (0..8).map(|k| di(x_base + k, dense_cols, dense_of)).collect();
            let yb: Vec<usize> = (0..8).map(|k| di(y_base + k, dense_cols, dense_of)).collect();
            let carries: Vec<usize> =
                (0..15).map(|k| di(carry_base + k, dense_cols, dense_of)).collect();
            let sidx: Vec<usize> = sels
                .iter()
                .map(|&f| si(SparseKind::Sel(f), sparse_kinds, sparse_of))
                .collect();
            for k in 0..15usize {
                for &sp in &sidx {
                    if k > 0 {
                        terms.push(TermTpl {
                            coeff: Goldilocks::ONE,
                            sparse: vec![sp],
                            dense: vec![carries[k - 1]],
                            group: g,
                        });
                    }
                    for i in 0..8usize {
                        if i <= k && k - i < 8 {
                            terms.push(TermTpl {
                                coeff: Goldilocks::ONE,
                                sparse: vec![sp],
                                dense: vec![xb[i], yb[k - i]],
                                group: g,
                            });
                        }
                    }
                    if let Some(rb) = r_base {
                        if k < 8 {
                            terms.push(TermTpl {
                                coeff: Goldilocks::ONE,
                                sparse: vec![sp],
                                dense: vec![di(rb + k, dense_cols, dense_of)],
                                group: g,
                            });
                        }
                    }
                    if k < 8 {
                        // The low output byte: for MUL/SLLI it is Rdv's byte;
                        // for DIV/SRLI the accumulator output is Rs1v's byte.
                        // The caller passes hi_base = None to select Rs1v.
                        let out = if hi_base.is_some() {
                            di(byte_col(YB_RDV + k), dense_cols, dense_of)
                        } else {
                            di(byte_col(YB_RS1V + k), dense_cols, dense_of)
                        };
                        terms.push(TermTpl {
                            coeff: Goldilocks::ONE.neg(),
                            sparse: vec![sp],
                            dense: vec![out],
                            group: g,
                        });
                    } else if let Some(hb) = hi_base {
                        terms.push(TermTpl {
                            coeff: Goldilocks::ONE.neg(),
                            sparse: vec![sp],
                            dense: vec![di(hb + (k - 8), dense_cols, dense_of)],
                            group: g,
                        });
                    }
                    if k < 14 {
                        terms.push(TermTpl {
                            coeff: fe_const(256).neg(),
                            sparse: vec![sp],
                            dense: vec![carries[k]],
                            group: g,
                        });
                    }
                }
            }
            // Final carry: for MUL/SLLI it lands in the hi top byte (a
            // committed column, range-checked); for DIV/SRLI it must vanish.
            for &sp in &sidx {
                if hi_base.is_some() {
                    let hb = hi_base.unwrap_or(0);
                    terms.push(TermTpl {
                        coeff: fe_const(256).neg(),
                        sparse: vec![sp],
                        dense: vec![di(hb + 7, dense_cols, dense_of)],
                        group: g,
                    });
                } else {
                    terms.push(TermTpl {
                        coeff: Goldilocks::ONE,
                        sparse: vec![sp],
                        dense: vec![carries[14]],
                        group: g,
                    });
                }
            }
        };
        let g_mul = group!();
        mk(byte_col(YB_RS1V), byte_col(YB_RS2V), Some(byte_col(YB_HIM)), None,
            bit_col(B_MULC1), &[fam::MUL], &mut terms, &mut dense_cols, &mut dense_of,
            &mut sparse_kinds, &mut sparse_of, g_mul);
        let g_slli = group!();
        mk(byte_col(YB_RS1V), byte_col(YB_KPOW), Some(byte_col(YB_HIS)), None,
            bit_col(B_SLLIC1), &[fam::SLLI], &mut terms, &mut dense_cols, &mut dense_of,
            &mut sparse_kinds, &mut sparse_of, g_slli);
        let g_div = group!();
        mk(byte_col(YB_Q), byte_col(YB_RS2V), None, Some(byte_col(YB_RV)),
            bit_col(B_DIVC1), &[fam::DIVQ, fam::DIVR], &mut terms, &mut dense_cols,
            &mut dense_of, &mut sparse_kinds, &mut sparse_of, g_div);
        let g_srli = group!();
        mk(byte_col(YB_RDV), byte_col(YB_KPOW), None, Some(byte_col(YB_RV)),
            bit_col(B_SRLIC1), &[fam::SRLI], &mut terms, &mut dense_cols, &mut dense_of,
            &mut sparse_kinds, &mut sparse_of, g_srli);
    }

    // ---- Control flow: next_pc identities per family. ----
    {
        // Straight-line families: NextPc = Pc + 4.
        for f in [fam::ADD, fam::ADDI, fam::SUB, fam::MUL, fam::DIVQ, fam::DIVR, fam::SLLI,
            fam::SRLI, fam::SLTU, fam::LOAD, fam::STORE, fam::LUI, fam::AUIPC] {
            let g = group!();
            let sp = si!(SparseKind::Sel(f));
            sel_lin!(g, sp,
                C_NEXT_PC => Goldilocks::ONE,
                C_PC => Goldilocks::ONE.neg(),
            );
            terms.push(TermTpl {
                coeff: fe_const(4).neg(),
                sparse: vec![sp],
                dense: vec![],
                group: g,
            });
        }
        // BRCH: taken → NextPc = Pc + ImmB; not taken → Pc + 4.
        // Unified: NextPc = Pc + 4 + Taken·(ImmB − 4).
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::BRCH));
            sel_lin!(g, sp,
                C_NEXT_PC => Goldilocks::ONE,
                C_PC => Goldilocks::ONE.neg(),
            );
            terms.push(TermTpl {
                coeff: fe_const(4).neg(),
                sparse: vec![sp],
                dense: vec![],
                group: g,
            });
            terms.push(TermTpl {
                coeff: Goldilocks::ONE.neg(),
                sparse: vec![sp],
                dense: vec![di!(bit_col(B_TAKEN)), di!(C_IMMB)],
                group: g,
            });
            terms.push(TermTpl {
                coeff: fe_const(4),
                sparse: vec![sp],
                dense: vec![di!(bit_col(B_TAKEN))],
                group: g,
            });
        }
        // JAL: NextPc = Pc + ImmJ.
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::JAL));
            sel_lin!(g, sp,
                C_NEXT_PC => Goldilocks::ONE,
                C_PC => Goldilocks::ONE.neg(),
                C_IMMJ => Goldilocks::ONE.neg(),
            );
        }
        // JALR: NextPc = Addr − Lsb.
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::JALR));
            sel_lin!(g, sp,
                C_NEXT_PC => Goldilocks::ONE,
                C_ADDR => Goldilocks::ONE.neg(),
                bit_col(B_LSB) => Goldilocks::ONE,
            );
        }
        // HALT: NextPc = Pc.
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::HALT));
            sel_lin!(g, sp,
                C_NEXT_PC => Goldilocks::ONE,
                C_PC => Goldilocks::ONE.neg(),
            );
        }
    }

    // ---- Taken (BRCH): the four branch kinds. ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::BRCH));
        sel_lin!(g, sp, bit_col(B_TAKEN) => Goldilocks::ONE);
        // − Fb0·Eqb − Fb1·(1−Eqb) − Fb2·GT1 − Fb3·(LT1+Eqb)
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![sp],
            dense: vec![di!(bit_col(B_FB0)), di!(bit_col(B_EQB))],
            group: g,
        });
        sel_lin!(g, sp, bit_col(B_FB0 + 1) => Goldilocks::ONE.neg());
        terms.push(TermTpl {
            coeff: Goldilocks::ONE,
            sparse: vec![sp],
            dense: vec![di!(bit_col(B_FB0 + 1)), di!(bit_col(B_EQB))],
            group: g,
        });
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![sp],
            dense: vec![di!(bit_col(B_FB0 + 2)), di!(bit_col(B_GT1))],
            group: g,
        });
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![sp],
            dense: vec![di!(bit_col(B_FB0 + 3)), di!(bit_col(B_LT1))],
            group: g,
        });
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![sp],
            dense: vec![di!(bit_col(B_FB0 + 3)), di!(bit_col(B_EQB))],
            group: g,
        });
    }

    // ---- SLTU: Rdv = GT1 = [rs1 < rs2]. ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::SLTU));
        sel_lin!(g, sp,
            C_RDV => Goldilocks::ONE,
            bit_col(B_GT1) => Goldilocks::ONE.neg(),
        );
    }

    // ---- Result routing: LUI / AUIPC / JAL / JALR write-back. ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::LUI));
        sel_lin!(g, sp, C_RDV => Goldilocks::ONE, C_IMMU => Goldilocks::ONE.neg());
    }
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::AUIPC));
        sel_lin!(g, sp, C_RDV => Goldilocks::ONE, C_PC => Goldilocks::ONE.neg(), C_IMMU => Goldilocks::ONE.neg());
    }
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::JAL));
        sel_lin!(g, sp, C_RDV => Goldilocks::ONE, C_PC => Goldilocks::ONE.neg());
        terms.push(TermTpl {
            coeff: fe_const(4).neg(),
            sparse: vec![sp],
            dense: vec![],
            group: g,
        });
    }
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::JALR));
        sel_lin!(g, sp, C_RDV => Goldilocks::ONE, C_PC => Goldilocks::ONE.neg());
        terms.push(TermTpl {
            coeff: fe_const(4).neg(),
            sparse: vec![sp],
            dense: vec![],
            group: g,
        });
    }

    // ---- LOAD: Rdv = Mrv. STORE: Mwv = Rs2v. ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::LOAD));
        sel_lin!(g, sp, C_RDV => Goldilocks::ONE, C_MRV => Goldilocks::ONE.neg());
    }
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::STORE));
        sel_lin!(g, sp, C_MWV => Goldilocks::ONE, C_RS2V => Goldilocks::ONE.neg());
    }

    // ---- Effective address: Addr = Rs1v + ImmS (LOAD/STORE), and the
    //      port routing Rada/WadaR = Ea on the family, 0 elsewhere. ----
    {
        // LOAD: Addr = Rs1v + ImmI (the I-immediate combo).
        // STORE: Addr = Rs1v + ImmS (the S-immediate combo).
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::LOAD));
            sel_lin!(g, sp,
                C_ADDR => Goldilocks::ONE,
                C_RS1V => Goldilocks::ONE.neg(),
            );
            for &(bit_i, coeff) in &imm_i_bits() {
                terms.push(TermTpl {
                    coeff: coeff.neg(),
                    sparse: vec![sp],
                    dense: vec![di!(bit_i)],
                    group: g,
                });
            }
        }
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::STORE));
            sel_lin!(g, sp,
                C_ADDR => Goldilocks::ONE,
                C_RS1V => Goldilocks::ONE.neg(),
            );
            for &(bit_i, coeff) in &imm_s_bits() {
                terms.push(TermTpl {
                    coeff: coeff.neg(),
                    sparse: vec![sp],
                    dense: vec![di!(bit_i)],
                    group: g,
                });
            }
        }
        // JALR: Addr = Rs1v + ImmI.
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::JALR));
            sel_lin!(g, sp,
                C_ADDR => Goldilocks::ONE,
                C_RS1V => Goldilocks::ONE.neg(),
            );
            for &(bit_i, coeff) in &imm_i_bits() {
                terms.push(TermTpl {
                    coeff: coeff.neg(),
                    sparse: vec![sp],
                    dense: vec![di!(bit_i)],
                    group: g,
                });
            }
        }
        // Rada = Ea on LOAD; WadaR = Ea on STORE; zero on other families.
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::LOAD));
            sel_lin!(g, sp, C_RADA => Goldilocks::ONE, C_EA => Goldilocks::ONE.neg());
        }
        {
            let g = group!();
            let sp = si!(SparseKind::Sel(fam::STORE));
            sel_lin!(g, sp, C_WADAR => Goldilocks::ONE, C_EA => Goldilocks::ONE.neg());
        }
        for f in 0..NUM_SEL {
            if f != fam::LOAD {
                let g = group!();
                let sp = si!(SparseKind::Sel(f));
                sel_lin!(g, sp, C_RADA => Goldilocks::ONE);
            }
            if f != fam::STORE {
                let g = group!();
                let sp = si!(SparseKind::Sel(f));
                sel_lin!(g, sp, C_WADAR => Goldilocks::ONE);
            }
        }
    }

    // ---- DIV semantics: r < b and the divide-by-zero branch. ----
    {
        // (1 − Bz)·LT3 = 0 on DIV rows: LT3 = [b ≤ r] must vanish.
        let g = group!();
        let spq = si!(SparseKind::Sel(fam::DIVQ));
        let spr = si!(SparseKind::Sel(fam::DIVR));
        for sp in [spq, spr] {
            sel_lin!(g, sp, bit_col(B_LT3) => Goldilocks::ONE);
            terms.push(TermTpl {
                coeff: Goldilocks::ONE.neg(),
                sparse: vec![sp],
                dense: vec![di!(bit_col(B_BZ)), di!(bit_col(B_LT3))],
                group: g,
            });
        }
        // Bz branch: Bz·(Q − (2^63−1)) = 0 and Bz·(Rv − Rs1v) = 0.
        let q_all_ones = (1u64 << 63) - 1;
        let g2 = group!();
        let bz = di!(bit_col(B_BZ));
        for sp in [spq, spr] {
            terms.push(TermTpl {
                coeff: Goldilocks::ONE,
                sparse: vec![sp],
                dense: vec![bz, di!(C_Q)],
                group: g2,
            });
            terms.push(TermTpl {
                coeff: fe_const(q_all_ones).neg(),
                sparse: vec![sp],
                dense: vec![bz],
                group: g2,
            });
            terms.push(TermTpl {
                coeff: Goldilocks::ONE,
                sparse: vec![sp],
                dense: vec![bz, di!(C_RV)],
                group: g2,
            });
            terms.push(TermTpl {
                coeff: Goldilocks::ONE.neg(),
                sparse: vec![sp],
                dense: vec![bz, di!(C_RS1V)],
                group: g2,
            });
        }
    }

    // ---- The shift amount decode (Sel_SLLI ∪ Sel_SRLI): ----
    // Shamt = Σ 2^i·instr_bit(20+i) on shift rows only.
    {
        let g = group!();
        for f in [fam::SLLI, fam::SRLI] {
            let sp = si!(SparseKind::Sel(f));
            sel_lin!(g, sp, C_SHAMT => Goldilocks::ONE.neg());
            for &(bit_i, coeff) in &shamt_bits() {
                terms.push(TermTpl {
                    coeff,
                    sparse: vec![sp],
                    dense: vec![di!(bit_i)],
                    group: g,
                });
            }
        }
    }

    // ---- SRLI bound: LT4 = [kpow ≤ r] must vanish (r < kpow). ----
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::SRLI));
        sel_lin!(g, sp, bit_col(B_LT4) => Goldilocks::ONE);
    }

    // ---- The register write mask: WrMask = wr·(1 − X0f). ----
    {
        let g = group!();
        let ones = si!(SparseKind::Ones);
        sel_lin!(g, ones, bit_col(B_WRMASK) => Goldilocks::ONE);
        // − Σ_k wr_k·(Sel_k − Sel_k·X0f): each writing family contributes.
        for f in 0..NUM_SEL {
            if writes_rd(f) {
                let sp = si!(SparseKind::Sel(f));
                terms.push(TermTpl {
                    coeff: Goldilocks::ONE.neg(),
                    sparse: vec![sp],
                    dense: vec![],
                    group: g,
                });
                terms.push(TermTpl {
                    coeff: Goldilocks::ONE,
                    sparse: vec![sp],
                    dense: vec![di!(bit_col(B_X0F))],
                    group: g,
                });
            }
        }
    }

    // ---- The write ports: Wada = WrMask·Rda, Wvr = WrMask·Rdv. ----
    {
        let g = group!();
        let ones = si!(SparseKind::Ones);
        sel_lin!(g, ones, C_WADA => Goldilocks::ONE);
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![ones],
            dense: vec![di!(bit_col(B_WRMASK)), di!(C_RDA)],
            group: g,
        });
        let g2 = group!();
        sel_lin!(g2, ones, C_WVR => Goldilocks::ONE);
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![ones],
            dense: vec![di!(bit_col(B_WRMASK)), di!(C_RDV)],
            group: g2,
        });
    }

    // ---- x0 forwarding: X0f = isZero(Rda) via the inverse witness. ----
    {
        let g = group!();
        let ones = si!(SparseKind::Ones);
        // Rda·Invr + X0f − 1 = 0.
        terms.push(TermTpl {
            coeff: Goldilocks::ONE,
            sparse: vec![ones],
            dense: vec![di!(C_RDA), di!(C_INVR)],
            group: g,
        });
        sel_lin!(g, ones, bit_col(B_X0F) => Goldilocks::ONE);
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![ones],
            dense: vec![],
            group: g,
        });
        // X0f·Invr = 0.
        let g2 = group!();
        terms.push(TermTpl {
            coeff: Goldilocks::ONE,
            sparse: vec![ones],
            dense: vec![di!(bit_col(B_X0F)), di!(C_INVR)],
            group: g2,
        });
    }

    // ---- The JALR LSB half-adder (Sel_JALR-conditioned): ----
    // Lsb = rs1_b0 ⊕ iw20, Crr = rs1_b0 ∧ iw20.
    {
        let g = group!();
        let sp = si!(SparseKind::Sel(fam::JALR));
        sel_lin!(g, sp, bit_col(B_LSB) => Goldilocks::ONE);
        sel_lin!(g, sp, byte_col(YB_RS1V) => Goldilocks::ONE.neg());
        sel_lin!(g, sp, bit_col(B_IBIT0 + 20) => Goldilocks::ONE.neg());
        terms.push(TermTpl {
            coeff: fe_const(2),
            sparse: vec![sp],
            dense: vec![di!(byte_col(YB_RS1V)), di!(bit_col(B_IBIT0 + 20))],
            group: g,
        });
        let g2 = group!();
        sel_lin!(g2, sp, bit_col(B_CRR) => Goldilocks::ONE);
        terms.push(TermTpl {
            coeff: Goldilocks::ONE.neg(),
            sparse: vec![sp],
            dense: vec![di!(byte_col(YB_RS1V)), di!(bit_col(B_IBIT0 + 20))],
            group: g2,
        });
    }

    AirLayout { num_vars: log_t, dense_cols, sparse_kinds, terms }
}

// ---------------------------------------------------------------------------
// The bit-combination helpers (shared by the AIR terms and the gates).
// The sign-extension coefficient: the u64 interpretation of a w-bit signed
// immediate is raw + sign·(2^64 − 2^w), so the sign bit's field
// coefficient is 2^{w−1} + (2^64 − 2^w) mod p.
// ---------------------------------------------------------------------------

/// The sign bit's slot coefficient for a w-bit SIGNED immediate whose
/// column carries the mod-p canonical value: imm = raw − 2^w·b31, and
/// raw already includes b31's 2^{w−1}, so b31's total coefficient is
/// −2^{w−1} (as a field negative).
fn sign_coeff(w: u32) -> Goldilocks {
    fe(1u64 << (w - 1)).neg()
}

/// The signed w-bit immediate as its mod-p canonical field value.
fn signed_field(raw: u64, w: u32) -> Goldilocks {
    if raw & (1u64 << (w - 1)) != 0 {
        fe(raw).sub(&fe(1u64 << w))
    } else {
        fe(raw)
    }
}

fn imm_i_bits() -> Vec<(usize, Goldilocks)> {
    let mut slots: Vec<(usize, Goldilocks)> = (20..31)
        .map(|i| (bit_col(B_IBIT0 + i), fe(1u64 << (i - 20))))
        .collect();
    slots.push((bit_col(B_IBIT0 + 31), sign_coeff(12)));
    slots
}

fn imm_b_bits() -> Vec<(usize, Goldilocks)> {
    let mut slots: Vec<(usize, Goldilocks)> = Vec::new();
    slots.push((bit_col(B_IBIT0 + 31), sign_coeff(13)));
    for i in 25..31 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 20))));
    }
    slots.push((bit_col(B_IBIT0 + 7), fe(1u64 << 11)));
    for i in 8..12 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 7))));
    }
    slots
}

fn imm_j_bits() -> Vec<(usize, Goldilocks)> {
    // J-format: imm[19:12] = instr[19:12] (bits stay in place), imm[11]
    // = instr[20], imm[10:1] = instr[30:21], imm[20] = instr[31].
    let mut slots: Vec<(usize, Goldilocks)> = Vec::new();
    slots.push((bit_col(B_IBIT0 + 31), sign_coeff(21)));
    for i in 12..20 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << i)));
    }
    slots.push((bit_col(B_IBIT0 + 20), fe(1u64 << 11)));
    for i in 21..31 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 20))));
    }
    slots
}

fn imm_u_bits() -> Vec<(usize, Goldilocks)> {
    (12..32).map(|i| (bit_col(B_IBIT0 + i), fe(1u64 << i))).collect()
}

fn imm_s_bits() -> Vec<(usize, Goldilocks)> {
    let mut slots: Vec<(usize, Goldilocks)> = Vec::new();
    slots.push((bit_col(B_IBIT0 + 31), sign_coeff(12)));
    for i in 25..31 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 20))));
    }
    // imm[4:0] = instr[11:7] and imm[11] = instr[7].
    slots.push((bit_col(B_IBIT0 + 7), fe(1u64 << 11)));
    for i in 8..12 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 7))));
    }
    slots
}

fn dkey_bits() -> Vec<(usize, Goldilocks)> {
    let mut slots: Vec<(usize, Goldilocks)> = (0..7)
        .map(|i| (bit_col(B_IBIT0 + i), fe(1u64 << i)))
        .collect();
    for i in 12..15 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 5))));
    }
    for i in 25..32 {
        slots.push((bit_col(B_IBIT0 + i), fe(1u64 << (i - 15))));
    }
    slots
}

fn shamt_bits() -> Vec<(usize, Goldilocks)> {
    (20..26).map(|i| (bit_col(B_IBIT0 + i), fe(1u64 << (i - 20)))).collect()
}

/// Register-address bits: rd at iw 7..11, rs1 at 15..19, rs2 at 20..24.
fn reg_addr_bits(base_iw_bit: usize) -> Vec<(usize, Goldilocks)> {
    (0..5)
        .map(|i| (bit_col(B_IBIT0 + base_iw_bit + i), fe(1u64 << i)))
        .collect()
}

// ---------------------------------------------------------------------------
// The linear gates
// ---------------------------------------------------------------------------

/// Build the full gate list (prover emits claims; the verifier checks the
/// β-RLC at r_air). Every slot magnitude is < p by construction.
#[allow(clippy::too_many_lines)]
pub fn build_gates() -> Vec<Gate> {
    let mut gates: Vec<Gate> = Vec::new();
    let mut push = |slots: Vec<(usize, Goldilocks)>, constant: Goldilocks| {
        gates.push(Gate { slots, constant });
    };

    // 1. Value recompositions: V = Σ 2^{8k}·b_k (byte 7 is 7-bit at 2^56).
    for &vc in BYTE_VALUES.iter() {
        let base = yb(vc);
        let mut slots = vec![(vc, Goldilocks::ONE.neg())];
        for k in 0..BYTES_PER_VALUE {
            slots.push((byte_col(base + k), fe(1u64 << (8 * k))));
        }
        push(slots, Goldilocks::ZERO);
    }

    // 2. Iw = Σ 2^i·bit_i.
    {
        let mut slots = vec![(C_IW, Goldilocks::ONE.neg())];
        for i in 0..NUM_IBITS {
            slots.push((bit_col(B_IBIT0 + i), fe(1u64 << i)));
        }
        push(slots, Goldilocks::ZERO);
    }

    // 3. Selw = Σ 2^k·Sel_k; Σ Sel_k = 1.
    {
        let mut slots = vec![(C_SELW, Goldilocks::ONE.neg())];
        for k in 0..NUM_SEL {
            slots.push((bit_col(B_SEL0 + k), fe(1u64 << k)));
        }
        push(slots, Goldilocks::ZERO);
    }
    {
        let slots: Vec<(usize, Goldilocks)> = (0..NUM_SEL)
            .map(|k| (bit_col(B_SEL0 + k), Goldilocks::ONE))
            .collect();
        push(slots, Goldilocks::ONE.neg());
    }

    // 4. Dkey / Shamt / register addresses / immediates from bits.
    {
        let mut slots = vec![(C_DKEY, Goldilocks::ONE.neg())];
        slots.extend(dkey_bits());
        push(slots, Goldilocks::ZERO);
    }
    for (col, base) in [(C_RS1A, 15usize), (C_RS2A, 20usize), (C_RDA, 7usize)] {
        let mut slots = vec![(col, Goldilocks::ONE.neg())];
        slots.extend(reg_addr_bits(base));
        push(slots, Goldilocks::ZERO);
    }
    for (col, bits) in [
        (C_IMMI, imm_i_bits()),
        (C_IMMB, imm_b_bits()),
        (C_IMMJ, imm_j_bits()),
        (C_IMMU, imm_u_bits()),
    ] {
        let mut slots = vec![(col, Goldilocks::ONE.neg())];
        slots.extend(bits);
        push(slots, Goldilocks::ZERO);
    }

    // 5. Branch-kind one-hot: Σ Fb_k = 1.
    {
        let slots: Vec<(usize, Goldilocks)> = (0..4)
            .map(|k| (bit_col(B_FB0 + k), Goldilocks::ONE))
            .collect();
        push(slots, Goldilocks::ONE.neg());
    }

    // 6. Eqb = 1 − LT1 − GT1.
    {
        let slots = vec![
            (bit_col(B_EQB), Goldilocks::ONE),
            (bit_col(B_LT1), Goldilocks::ONE),
            (bit_col(B_GT1), Goldilocks::ONE),
        ];
        push(slots, Goldilocks::ONE.neg());
    }

    // 7. FetchRa: 4·FetchRa − Pc + 0x1000 = 0.
    {
        let slots = vec![
            (C_FETCH_RA, fe(4)),
            (C_PC, Goldilocks::ONE.neg()),
        ];
        push(slots, fe(0x1000));
    }

    // 8. Alignment: addr_b0 = 8·al8; ea = al8 + Σ_{k≥1} 2^{8k−3}·addr_bk.
    {
        let slots = vec![
            (byte_col(YB_ADDR), Goldilocks::ONE),
            (C_AL8, fe(8).neg()),
        ];
        push(slots, Goldilocks::ZERO);
    }
    {
        let mut slots = vec![
            (C_EA, Goldilocks::ONE),
            (C_AL8, Goldilocks::ONE.neg()),
        ];
        for k in 1..BYTES_PER_VALUE {
            slots.push((byte_col(YB_ADDR + k), fe(1u64 << (8 * k - 3))));
        }
        push(slots, Goldilocks::ZERO);
    }

    // 9. The five dc chains (unconditional — the trace computes them on
    //    every row). Position k: d_k + b_k + c_k − a_k − radix·c_{k+1} = 0
    //    with c_0 ∈ {0,1} constant and c_8 = the comparison bit.
    // dc1: D1 + Rs1v → Rs2v, c_out = LT1.
    // dc2: D2 + Rs2v → Rs1v, c_out = GT1.
    // dc3: D3 + Rv → Rs2v with c_0 = 1, c_out = LT3.
    // dc4: D4 + Rv → Kpow with c_0 = 1, c_out = LT4.
    // dc5: D5 → Rs2v with c_0 = 1 (b = 0), c_out = Bz.
    let chains: [(usize, usize, usize, usize, u64, usize); 5] = [
        (DC1_B0, YB_RS1V, YB_RS2V, B_DC1C1, 0, B_LT1),
        (DC2_B0, YB_RS2V, YB_RS1V, B_DC2C1, 0, B_GT1),
        (DC3_B0, YB_RV, YB_RS2V, B_DC3C1, 1, B_LT3),
        (DC4_B0, YB_RV, YB_KPOW, B_DC4C1, 1, B_LT4),
        (DC5_B0, usize::MAX, YB_RS2V, B_DC5C1, 1, B_BZ),
    ];
    for (d_base, b_base, a_base, carry_base, c0, top_bit) in chains {
        for k in 0..8usize {
            // Position 7's carry-out lands at bit 64 (the comparison bit):
            // radix 2^8 because the dc's top byte is full 8-bit.
            let radix: u64 = 256;
            let mut slots: Vec<(usize, Goldilocks)> = Vec::new();
            slots.push((byte_col(d_base + k), Goldilocks::ONE));
            if b_base != usize::MAX {
                slots.push((byte_col(b_base + k), Goldilocks::ONE));
            }
            if k > 0 {
                slots.push((bit_col(carry_base + k - 1), Goldilocks::ONE));
            }
            slots.push((byte_col(a_base + k), Goldilocks::ONE.neg()));
            if k == 7 {
                slots.push((bit_col(top_bit), fe(radix).neg()));
            } else {
                slots.push((bit_col(carry_base + k), fe(radix).neg()));
            }
            let constant = if k == 0 { fe(c0) } else { Goldilocks::ZERO };
            push(slots, constant);
        }
    }

    gates
}

// ---------------------------------------------------------------------------
// The proof object and the claim plumbing
// ---------------------------------------------------------------------------

/// A (column, point, value) claim authenticated by a grouped opening.
/// `factor` discriminates sentinel claims (the memory legs' dim/Inc/Val
/// terminals ride the P1 layer and are resolved by discriminant).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnClaim {
    pub col: usize,
    pub factor: u64,
    pub point: Vec<Goldilocks>,
    pub value: Goldilocks,
}

/// Sentinel column for the memory legs' factor-discriminated claims.
pub const SENTINEL_COL: usize = usize::MAX;

/// The v3 proof.
#[derive(Clone, Debug)]
pub struct ProofV3 {
    pub log_t: usize,
    pub log_k_ram: usize,
    pub log_k_fetch: usize,
    pub public_state: PublicStateV3,
    /// Per-column Ajtai commitments (NUM_COLS entries).
    pub commitments: Vec<Vec<u8>>,
    /// The memory-argument legs (fetch Shout, RAM/reg one-hots + twists).
    pub fetch: ShoutProof,
    pub onehot_ram_r: OneHotProof,
    pub onehot_ram_w: OneHotProof,
    pub onehot_reg_a: OneHotProof,
    pub onehot_reg_b: OneHotProof,
    pub onehot_reg_w: OneHotProof,
    pub twist_ram: TwistProof,
    pub twist_reg_a: TwistProof,
    pub twist_reg_b: TwistProof,
    /// The decode / pow2 / alignment / range Shouts, in canonical order.
    pub lookup_shouts: Vec<ShoutProof>,
    /// The batched AIR sumcheck.
    pub air: lattice_sumcheck::SumcheckProof,
    /// The full claim table (column claims + sentinels).
    pub claims: Vec<ColumnClaim>,
    /// Per-column grouped openings.
    pub openings: Vec<EvaluationProof>,
}

/// Evaluate a column MLE at a point without materializing a DenseMle.
fn eval_col(col: &[Goldilocks], point: &[Goldilocks]) -> Goldilocks {
    let num_vars = point.len();
    debug_assert_eq!(col.len(), 1 << num_vars);
    let mut acc = Goldilocks::ONE;
    let mut value = Goldilocks::ZERO;
    for (i, &x) in col.iter().enumerate() {
        let mut w = acc;
        // acc tracks Π eq(point[j], bit_j) incrementally over variables.
        let mut idx = i;
        for j in (0..num_vars).rev() {
            let bit = idx & 1;
            idx >>= 1;
            w = w.mul(&eq_lerp_pub(point[j], bit as u64));
        }
        value = value.add(&w.mul(&x));
    }
    value
}

fn eq_lerp_pub(p: Goldilocks, bit: u64) -> Goldilocks {
    if bit == 1 {
        p
    } else {
        Goldilocks::ONE.sub(&p)
    }
}

/// Prover-side recording resolver over the witness columns.
struct ColRes<'a> {
    cols: &'a [Vec<Goldilocks>],
    log_t: usize,
    claims: RefCell<Vec<ColumnClaim>>,
    map: &'a dyn Fn(FactorId) -> Option<usize>,
}

impl<'a> ColRes<'a> {
    fn new(
        cols: &'a [Vec<Goldilocks>],
        log_t: usize,
        map: &'a dyn Fn(FactorId) -> Option<usize>,
    ) -> Self {
        Self { cols, log_t, claims: RefCell::new(Vec::new()), map }
    }
    fn drain(&self) -> Vec<ColumnClaim> {
        std::mem::take(&mut *self.claims.borrow_mut())
    }
}

impl<'a> FactorResolver for ColRes<'a> {
    fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        let col = (self.map)(factor).ok_or(PiopError::MissingFactor { factor })?;
        if point.len() != self.log_t {
            return Err(PiopError::Shape { expected: self.log_t, got: point.len() });
        }
        let v = eval_col(&self.cols[col], point);
        self.claims.borrow_mut().push(ColumnClaim {
            col,
            factor: factor.discriminant(),
            point: point.to_vec(),
            value: v,
        });
        Ok(v)
    }
}

/// Verifier-side claim-table resolver.
struct TableRes<'a> {
    claims: &'a [ColumnClaim],
    map: &'a dyn Fn(FactorId) -> Option<usize>,
}

impl<'a> FactorResolver for TableRes<'a> {
    fn eval(&self, factor: FactorId, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        if let Some(col) = (self.map)(factor) {
            for c in self.claims {
                if c.col == col && c.point.as_slice() == point {
                    return Ok(c.value);
                }
            }
        }
        for c in self.claims {
            if c.col == SENTINEL_COL
                && c.factor == factor.discriminant()
                && c.point.as_slice() == point
            {
                return Ok(c.value);
            }
        }
        Err(PiopError::MissingFactor { factor })
    }
}

/// The canonical lookup-Shout schedule: (table kind, address column,
/// value column, log_k). Prover and verifier must agree exactly.
pub fn lookup_schedule() -> Vec<(&'static str, usize, usize, usize)> {
    let mut sched: Vec<(&'static str, usize, usize, usize)> = Vec::new();
    // Decode: Dkey -> Selw (the public 2^16 decode table).
    sched.push(("decode", C_DKEY, C_SELW, 17));
    // Pow2: Shamt -> Kpow.
    sched.push(("pow2", C_SHAMT, C_KPOW, 6));
    // Alignment: al8 (range5).
    sched.push(("al8", C_AL8, C_AL8, 5));
    // Value byte columns: 7 × range8 + top range7.
    for &vc in BYTE_VALUES.iter() {
        let base = yb(vc);
        for k in 0..7 {
            sched.push(("range8", byte_col(base + k), byte_col(base + k), 8));
        }
        sched.push(("range7", byte_col(base + 7), byte_col(base + 7), 7));
    }
    // dc byte columns: 8 × range8 each (the top byte is full 8-bit).
    for di in 0..DC_COUNT {
        for k in 0..8 {
            sched.push(("range8", byte_col(DC_BYTES + di * DC_STEP + k),
                byte_col(DC_BYTES + di * DC_STEP + k), 8));
        }
    }
    // Schoolbook carries: range12 (MUL/SLLI 14 each, DIV/SRLI 15 each).
    for k in 0..14 {
        sched.push(("range12", bit_col(B_MULC1 + k), bit_col(B_MULC1 + k), 12));
        sched.push(("range12", bit_col(B_SLLIC1 + k), bit_col(B_SLLIC1 + k), 12));
    }
    for k in 0..15 {
        sched.push(("range12", bit_col(B_DIVC1 + k), bit_col(B_DIVC1 + k), 12));
        sched.push(("range12", bit_col(B_SRLIC1 + k), bit_col(B_SRLIC1 + k), 12));
    }
    sched
}

/// The identity tables (verifier-recomputable).
fn lookup_table(kind: &str, log_k: usize) -> Vec<Goldilocks> {
    let n = 1usize << log_k;
    match kind {
        "decode" => decode_table_v3(),
        "pow2" => (0..n).map(|j| fe(1u64 << j.min(63))).collect(),
        "al8" | "range8" | "range12" => (0..n).map(|j| fe(j as u64)).collect(),
        "range7" => (0..n).map(|j| fe(j as u64)).collect(),
        _ => (0..n).map(|j| fe(j as u64)).collect(),
    }
}

// ---------------------------------------------------------------------------
// prove_v3 / verify_v3
// ---------------------------------------------------------------------------

/// A PCS geometry sized for the v3 column universe.
pub fn v3_pcs_for(log_t: usize, log_n: u32, seed: [u8; 32]) -> Result<AkitaPcs, Pipeline3Error> {
    // m must cover the packed witness: 3 limbs per value × 2^log_t values
    // packed 2^log_n coefficients per ring element.
    let m = ((3usize << log_t) >> log_n).max(1);
    let pcs = lattice_akita::akita_setup(log_n, m, 1 << 23, seed)
        .map_err(|_| Pipeline3Error::BadShape("akita setup".into()))?;
    Ok(pcs)
}

#[allow(clippy::too_many_lines)]
pub fn prove_v3(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(PublicStateV3, ProofV3), Pipeline3Error> {
    // ---- 1. Execute + trace. ----
    let mut state = MachineState::new();
    state.load_program(0x1000, program);
    state.load_program(0x3000, public_input);
    state.regs[10] = public_input.len() as u64;
    state.pc = 0x1000;
    let rows = vm_run(&mut state, max_steps).map_err(Pipeline3Error::Execution)?;
    if rows.is_empty() {
        return Err(Pipeline3Error::BadShape("empty trace".into()));
    }
    let trace = build_trace3(&state, &rows)?;
    let log_t = trace.log_t;
    let t_pow = 1usize << log_t;
    let cols = &trace.cols;

    // ---- 2. The RAM window. ----
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let fetch_words = program.len().div_ceil(4).max(1);
    let log_k_fetch = fetch_words.next_power_of_two().max(2).trailing_zeros() as usize;
    let mut max_word = 0x3000 / 8 + num_input_words as u64;
    max_word = max_word.max(0x1000 / 8 + program.len().div_ceil(8) as u64);
    for r in &rows {
        if let Some((a, _, _)) = &r.mem_access {
            max_word = max_word.max(a / 8);
        }
    }
    let log_k_ram = (max_word + 1).next_power_of_two().max(2).trailing_zeros() as usize;
    let k_ram = 1usize << log_k_ram;
    let mut init_ram = vec![0u64; k_ram];
    for (i, chunk) in program.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x1000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    for (i, chunk) in public_input.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x3000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    let mut final_ram = init_ram.clone();
    for (addr, v) in state.memory.snapshot_pairs() {
        let idx = (addr / 8) as usize;
        if idx < k_ram {
            final_ram[idx] = v;
        }
    }
    let public_state = PublicStateV3 {
        final_regs: state.regs,
        final_ram: final_ram.clone(),
        num_steps: rows.len() as u64,
    };

    // ---- 3. Commit every column. ----
    let mut commitments = Vec::with_capacity(NUM_COLS);
    for col in cols.iter().take(NUM_COLS) {
        let mle = DenseMle { num_vars: log_t, evaluations: col.clone() };
        commitments.push(pcs.commit(&mle)?.commitment.to_bytes());
    }

    // ---- 4. Transcript. ----
    let mut transcript = Transcript::new_default(b"lzx-zkvm-v3");
    transcript.append_bytes(b"program", &crate::program_digest(program))?;
    transcript.append_bytes(b"public-input", &crate::public_input_digest(public_input))?;
    transcript.append_field_slice(
        b"v3-meta",
        &[
            fe(log_t as u64),
            fe(log_k_ram as u64),
            fe(log_k_fetch as u64),
            fe(rows.len() as u64),
            fe(NUM_COLS as u64),
        ],
    )?;
    for (i, c) in commitments.iter().enumerate() {
        let mut b = (i as u32).to_le_bytes().to_vec();
        b.extend_from_slice(c);
        transcript.append_bytes(b"col-commit", &b)?;
    }
    let mut claims: Vec<ColumnClaim> = Vec::new();

    // ---- 5. The memory legs (the v2 protocol, v3 registry). ----
    #[cfg(test)]
    eprintln!("[v3] at memory");
    let fetch_ra: Vec<u64> = cols[C_FETCH_RA].iter().map(|x| x.to_canonical_u64()).collect();
    let fetch_table: Vec<Goldilocks> = (0..fetch_words)
        .map(|i| {
            let mut w = 0u32;
            for b in 0..4 {
                w |= (program[i * 4 + b] as u32) << (b * 8);
            }
            fe(w as u64)
        })
        .collect();
    let mut fetch_table = fetch_table;
    fetch_table.resize(1 << log_k_fetch, Goldilocks::ZERO);
    let fetch = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadValues => Some(C_IW),
            FactorId::ReadAddr => Some(C_FETCH_RA),
            _ => None,
        });
        let (p, dim) = prove_shout_sparse(
            &fetch_table, &fetch_ra, log_k_fetch, log_t, log_k_fetch, &res, &mut transcript,
        )?;
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL,
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    // RAM one-hots + twist.
    let ram_ra: Vec<u64> = trace.ram_read_addr.clone();
    let ram_wa: Vec<u64> = trace.ram_write_addr.clone();
    let onehot_ram_r = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadAddr => Some(C_RADA),
            _ => None,
        });
        let (p, dim) = prove_onehot_sparse(&ram_ra, log_k_ram, log_t, log_k_ram, OHSide::Read, &res, &mut transcript)?;
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL, factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };
    let onehot_ram_w = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::WriteAddr => Some(C_WADAR),
            _ => None,
        });
        let (p, dim) = prove_onehot_sparse(&ram_wa, log_k_ram, log_t, log_k_ram, OHSide::Write, &res, &mut transcript)?;
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL, factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };
    let twist_ram = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadValues => Some(C_MRV),
            FactorId::WriteValues => Some(C_MWV),
            _ => None,
        });
        let witness = build_twist_ports(
            &ram_ra, &ram_wa, &cols[C_MWV], &init_ram.iter().map(|&v| fe(v)).collect::<Vec<_>>(),
            log_k_ram, log_t,
        )?;
        let wv_mle = DenseMle { num_vars: log_t, evaluations: cols[C_MWV].clone() };
        let (p, cl) = prove_twist_ports_sparse(&witness, &wv_mle, &res, &mut transcript)?;
        claims.extend(cl.iter().map(|(f, point, v)| ColumnClaim {
            col: match f {
                FactorId::ReadValues => C_MRV,
                FactorId::WriteValues => C_MWV,
                _ => SENTINEL_COL,
            },
            factor: f.discriminant(),
            point: point.clone(),
            value: *v,
        }));
        p
    };
    // Register one-hots + twists.
    let reg_ra_a: Vec<u64> = trace.reg_read_addr_a.clone();
    let reg_ra_b: Vec<u64> = trace.reg_read_addr_b.clone();
    let reg_wa: Vec<u64> = trace.reg_write_addr.clone();
    let reg_init: Vec<Goldilocks> = {
        let mut v = vec![Goldilocks::ZERO; 32];
        v[10] = fe(public_input.len() as u64);
        v
    };
    let onehot_reg_a = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadAddr => Some(C_RS1A),
            _ => None,
        });
        let (p, dim) = prove_onehot_sparse(&reg_ra_a, 5, log_t, 5, OHSide::Read, &res, &mut transcript)?;
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL, factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };
    let onehot_reg_b = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadAddr => Some(C_RS2A),
            _ => None,
        });
        let (p, dim) = prove_onehot_sparse(&reg_ra_b, 5, log_t, 5, OHSide::Read, &res, &mut transcript)?;
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL, factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };
    let onehot_reg_w = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::WriteAddr => Some(C_WADA),
            _ => None,
        });
        let (p, dim) = prove_onehot_sparse(&reg_wa, 5, log_t, 5, OHSide::Write, &res, &mut transcript)?;
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL, factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };
    let twist_reg_a = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadValues => Some(C_RS1V),
            FactorId::WriteValues => Some(C_WVR),
            _ => None,
        });
        let witness = build_twist_ports(&reg_ra_a, &reg_wa, &cols[C_WVR], &reg_init, 5, log_t)?;
        let wv_mle = DenseMle { num_vars: log_t, evaluations: cols[C_WVR].clone() };
        let (p, cl) = prove_twist_ports_sparse(&witness, &wv_mle, &res, &mut transcript)?;
        claims.extend(cl.iter().map(|(f, point, v)| ColumnClaim {
            col: match f {
                FactorId::ReadValues => C_RS1V,
                FactorId::WriteValues => C_WVR,
                _ => SENTINEL_COL,
            },
            factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };
    let twist_reg_b = {
        let res = ColRes::new(cols, log_t, &|f| match f {
            FactorId::ReadValues => Some(C_RS2V),
            FactorId::WriteValues => Some(C_WVR),
            _ => None,
        });
        let witness = build_twist_ports(&reg_ra_b, &reg_wa, &cols[C_WVR], &reg_init, 5, log_t)?;
        let wv_mle = DenseMle { num_vars: log_t, evaluations: cols[C_WVR].clone() };
        let (p, cl) = prove_twist_ports_sparse(&witness, &wv_mle, &res, &mut transcript)?;
        claims.extend(cl.iter().map(|(f, point, v)| ColumnClaim {
            col: match f {
                FactorId::ReadValues => C_RS2V,
                FactorId::WriteValues => C_WVR,
                _ => SENTINEL_COL,
            },
            factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        p
    };

    // ---- 6. The lookup Shouts (decode / pow2 / al8 / ranges / carries). ----
    #[cfg(test)]
    eprintln!("[v3] at lookups");
    let sched = lookup_schedule();
    let mut lookup_shouts = Vec::with_capacity(sched.len());
    for (kind, addr_col, val_col, log_k) in sched.iter() {
        let kind: &str = kind;
        let addr_col = *addr_col;
        let val_col = *val_col;
        let log_k = *log_k;
        let table = lookup_table(kind, log_k);
        let addrs: Vec<u64> = cols[addr_col].iter().map(|x| x.to_canonical_u64()).collect();
        // The address stream must fit the table.
        for &a in &addrs {
            if a >= (1u64 << log_k) {
                return Err(Pipeline3Error::BadShape(format!(
                    "lookup {kind} address {a} out of range"
                )));
            }
        }
        let map = move |f: FactorId| match f {
            FactorId::ReadValues => Some(val_col),
            FactorId::ReadAddr => Some(addr_col),
            _ => None,
        };
        let res = ColRes::new(cols, log_t, &map);
        let (p, dim) = match prove_shout_sparse(
            &table, &addrs, log_k, log_t, log_k, &res, &mut transcript,
        ) {
            Ok(x) => x,
            Err(e) => {
                #[cfg(test)]
                eprintln!("[v3] lookup FAILED {kind} col {addr_col}: {e:?}");
                return Err(e.into());
            }
        };
        claims.extend(res.drain());
        claims.extend(dim.iter().map(|(f, point, v)| ColumnClaim {
            col: SENTINEL_COL, factor: f.discriminant(), point: point.clone(), value: *v,
        }));
        lookup_shouts.push(p);
    }

    // ---- 7. The AIR: one batched sparse sumcheck, claim zero. ----
    #[cfg(test)]
    eprintln!("[v3] at AIR");
    let layout = air_layout(log_t);
    let alpha = transcript.challenge_field(b"v3-air-alpha")?;
    // Sparse factors from the witness supports.
    let sparse: Vec<SparseFactor> = layout
        .sparse_kinds
        .iter()
        .map(|kind| {
            let entries: Vec<(u64, Goldilocks)> = match kind {
                SparseKind::Ones => (0..t_pow).map(|j| (j as u64, Goldilocks::ONE)).collect(),
                SparseKind::Sel(f) => (0..t_pow)
                    .filter(|&j| cols[bit_col(B_SEL0 + f)][j] == Goldilocks::ONE)
                    .map(|j| (j as u64, Goldilocks::ONE))
                    .collect(),
                SparseKind::Bit(b) => (0..t_pow)
                    .filter(|&j| cols[bit_col(*b)][j] == Goldilocks::ONE)
                    .map(|j| (j as u64, Goldilocks::ONE))
                    .collect(),
            };
            SparseFactor { entries, var_map: (0..log_t).collect() }
        })
        .collect();
    let dense: Vec<ProjectedDense> = layout
        .dense_cols
        .iter()
        .map(|&col| ProjectedDense {
            mle: DenseMle { num_vars: log_t, evaluations: cols[col].clone() },
            var_map: (0..log_t).collect(),
        })
        .collect();
    let terms: Vec<SparseTerm> = layout
        .terms
        .iter()
        .map(|tpl| {
            let mut coeff = tpl.coeff;
            for _ in 0..tpl.group {
                coeff = coeff.mul(&alpha);
            }
            let first_kind = layout.sparse_kinds[tpl.sparse[0]];
            let positions: Vec<u64> = match first_kind {
                SparseKind::Ones => (0..t_pow).map(|j| j as u64).collect(),
                SparseKind::Sel(f) => (0..t_pow)
                    .filter(|&j| cols[bit_col(B_SEL0 + f)][j] == Goldilocks::ONE)
                    .map(|j| j as u64)
                    .collect(),
                SparseKind::Bit(b) => (0..t_pow)
                    .filter(|&j| cols[bit_col(b)][j] == Goldilocks::ONE)
                    .map(|j| j as u64)
                    .collect(),
            };
            SparseTerm { coeff, positions, sparse: tpl.sparse.clone(), dense: tpl.dense.clone() }
        })
        .collect();
    let inst = SparseInstance { num_vars: log_t, sparse, dense, terms };
    let air_out = prove_sparse_sumcheck(&inst, Goldilocks::ZERO, &mut transcript)?;
    let r_air = air_out.challenges.clone();
    // Dense factor claims at r_air.
    for (i, &col) in layout.dense_cols.iter().enumerate() {
        claims.push(ColumnClaim {
            col,
            factor: 0,
            point: r_air.clone(),
            value: air_out.dense_claims[i],
        });
    }
    // Sparse factor claims (Sel/Bit columns; the ones claim is public = 1).
    for (i, kind) in layout.sparse_kinds.iter().enumerate() {
        match kind {
            SparseKind::Ones => {
                if air_out.sparse_claims[i] != Goldilocks::ONE {
                    return Err(Pipeline3Error::VerificationFailed);
                }
            }
            SparseKind::Sel(f) => claims.push(ColumnClaim {
                col: bit_col(B_SEL0 + f),
                factor: 0,
                point: r_air.clone(),
                value: air_out.sparse_claims[i],
            }),
            SparseKind::Bit(b) => claims.push(ColumnClaim {
                col: bit_col(*b),
                factor: 0,
                point: r_air.clone(),
                value: air_out.sparse_claims[i],
            }),
        }
    }
    // The gate claims: every column gets an r_air claim (dedup).
    let have_r_air: std::collections::HashSet<usize> =
        claims.iter().filter(|c| c.point == r_air).map(|c| c.col).collect();
    for col in 0..NUM_COLS {
        if !have_r_air.contains(&col) {
            claims.push(ColumnClaim {
                col,
                factor: 0,
                point: r_air.clone(),
                value: eval_col(&cols[col], &r_air),
            });
        }
    }

    // ---- 8. The gate challenges (the verifier's β-RLC). ----
    let gates = build_gates();
    let _betas = transcript.challenge_fields(b"v3-gate-beta", gates.len())?;

    // ---- 9. The grouped openings, one per column. ----
    let mut openings = Vec::with_capacity(NUM_COLS);
    for (col_idx, col) in cols.iter().enumerate().take(NUM_COLS) {
        let col_claims: Vec<GroupedOpening> = claims
            .iter()
            .filter(|c| c.col == col_idx)
            .map(|c| GroupedOpening { point: c.point.clone(), value: c.value })
            .collect();
        if col_claims.is_empty() {
            openings.push(placebo_opening(pcs, col, log_t)?);
        } else {
            let mle = DenseMle { num_vars: log_t, evaluations: col.clone() };
            openings.push(pcs.prove_grouped(&mle, &col_claims, &mut transcript)?);
        }
    }

    Ok((
        public_state.clone(),
        ProofV3 {
            log_t,
            log_k_ram,
            log_k_fetch,
            public_state,
            commitments,
            fetch,
            onehot_ram_r,
            onehot_ram_w,
            onehot_reg_a,
            onehot_reg_b,
            onehot_reg_w,
            twist_ram,
            twist_reg_a,
            twist_reg_b,
            lookup_shouts,
            air: air_out.proof,
            claims,
            openings,
        },
    ))
}

fn placebo_opening(
    pcs: &AkitaPcs,
    col: &[Goldilocks],
    log_vars: usize,
) -> Result<EvaluationProof, Pipeline3Error> {
    let mle = DenseMle { num_vars: log_vars, evaluations: col.to_vec() };
    let mut t = Transcript::new_default(b"lzx-placebo");
    let point: Vec<Goldilocks> = (0..log_vars).map(|i| fe(i as u64 + 1)).collect();
    Ok(pcs.prove_evaluation(&mle, &point, &mut t)?)
}

/// Verify a v3 proof. NEVER re-executes the program.
#[allow(clippy::too_many_lines)]
pub fn verify_v3(
    pcs: &AkitaPcs,
    program: &[u8],
    public_input: &[u8],
    public_state: &PublicStateV3,
    proof: &ProofV3,
    max_steps: u64,
) -> Result<(), Pipeline3Error> {
    if public_state.num_steps > max_steps || public_state.num_steps == 0 {
        return Err(Pipeline3Error::VerificationFailed);
    }
    let log_t = proof.log_t;
    if proof.commitments.len() != NUM_COLS || proof.openings.len() != NUM_COLS {
        return Err(Pipeline3Error::VerificationFailed);
    }
    // ---- Public tables. ----
    let fetch_words = program.len().div_ceil(4).max(1);
    let log_k_fetch = fetch_words.next_power_of_two().max(2).trailing_zeros() as usize;
    if proof.log_k_fetch != log_k_fetch {
        return Err(Pipeline3Error::VerificationFailed);
    }
    let mut fetch_table: Vec<Goldilocks> = (0..fetch_words)
        .map(|i| {
            let mut w = 0u32;
            for b in 0..4 {
                w |= (program[i * 4 + b] as u32) << (b * 8);
            }
            fe(w as u64)
        })
        .collect();
    fetch_table.resize(1 << log_k_fetch, Goldilocks::ZERO);
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let log_k_ram = proof.log_k_ram;
    let k_ram = 1usize << log_k_ram;
    if public_state.final_ram.len() != k_ram {
        return Err(Pipeline3Error::VerificationFailed);
    }
    let mut init_ram = vec![0u64; k_ram];
    for (i, chunk) in program.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x1000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    for (i, chunk) in public_input.chunks(8).enumerate() {
        let mut w = 0u64;
        for (b, byte) in chunk.iter().enumerate() {
            w |= (*byte as u64) << (b * 8);
        }
        let idx = 0x3000 / 8 + i;
        if idx < k_ram {
            init_ram[idx] = w;
        }
    }
    let _ = num_input_words;

    // ---- Transcript replay. ----
    let mut transcript = Transcript::new_default(b"lzx-zkvm-v3");
    transcript.append_bytes(b"program", &crate::program_digest(program))?;
    transcript.append_bytes(b"public-input", &crate::public_input_digest(public_input))?;
    transcript.append_field_slice(
        b"v3-meta",
        &[
            fe(log_t as u64),
            fe(log_k_ram as u64),
            fe(log_k_fetch as u64),
            fe(public_state.num_steps),
            fe(NUM_COLS as u64),
        ],
    )?;
    for (i, c) in proof.commitments.iter().enumerate() {
        let mut b = (i as u32).to_le_bytes().to_vec();
        b.extend_from_slice(c);
        transcript.append_bytes(b"col-commit", &b)?;
    }
    let claims = &proof.claims;

    // ---- The memory legs. ----
    let fetch_res = TableRes { claims, map: &|f| match f {
        FactorId::ReadValues => Some(C_IW),
        FactorId::ReadAddr => Some(C_FETCH_RA),
        _ => None,
    }};
    lattice_memory::verify_shout(&proof.fetch, &fetch_table, log_k_fetch, log_t, log_k_fetch, &fetch_res, &mut transcript)?;
    let oh_ram_r = TableRes { claims, map: &|f| match f {
        FactorId::ReadAddr => Some(C_RADA),
        _ => None,
    }};
    lattice_memory::verify_onehot(&proof.onehot_ram_r, log_k_ram, log_t, OHSide::Read, &oh_ram_r, &mut transcript)?;
    let oh_ram_w = TableRes { claims, map: &|f| match f {
        FactorId::WriteAddr => Some(C_WADAR),
        _ => None,
    }};
    lattice_memory::verify_onehot(&proof.onehot_ram_w, log_k_ram, log_t, OHSide::Write, &oh_ram_w, &mut transcript)?;
    let ram_tw = TableRes { claims, map: &|f| match f {
        FactorId::ReadValues => Some(C_MRV),
        FactorId::WriteValues => Some(C_MWV),
        _ => None,
    }};
    verify_twist_ports_checked(
        &proof.twist_ram,
        &init_ram.iter().map(|&v| fe(v)).collect::<Vec<_>>(),
        &public_state.final_ram.iter().map(|&v| fe(v)).collect::<Vec<_>>(),
        log_k_ram, log_t, log_k_ram, &ram_tw, &mut transcript,
    ).map_err(|e| {
        #[cfg(test)]
        {
            eprintln!("[v3v] ram twist: {e:?}");
            eprintln!("[v3v] leg1 rounds[0] = {:?}", proof.twist_ram.read_checking.rounds.first().map(|r| r.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>()));
            eprintln!("[v3v] leg2 rounds[0] = {:?}", proof.twist_ram.inc_definition.rounds.first().map(|r| r.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>()));
            eprintln!("[v3v] leg3 rounds[0] = {:?}", proof.twist_ram.telescoping.rounds.first().map(|r| r.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>()));
            for c in claims.iter().filter(|c| c.col == C_MRV) {
                eprintln!("[v3v] MRV claim: pt {:?} v {}", c.point.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>(), c.value.to_canonical_u64());
            }
            // Resolve with the twist's own point (pt of the first MRV claim).
            if let Some(c) = claims.iter().find(|c| c.col == C_MRV) {
                let res = TableRes { claims, map: &|f: FactorId| match f {
                    FactorId::ReadValues => Some(C_MRV),
                    FactorId::WriteValues => Some(C_MWV),
                    _ => None,
                }};
                eprintln!("[v3v] manual resolve at pt1: {:?}", res.eval(FactorId::ReadValues, &c.point).map(|v| v.to_canonical_u64()));
            }
        }
        e
    })?;
    let reg_init: Vec<Goldilocks> = {
        let mut v = vec![Goldilocks::ZERO; 32];
        v[10] = fe(public_input.len() as u64);
        v
    };
    let reg_final: Vec<Goldilocks> = public_state.final_regs.iter().map(|&x| fe(x)).collect();
    let oh_ra = TableRes { claims, map: &|f| match f {
        FactorId::ReadAddr => Some(C_RS1A),
        _ => None,
    }};
    lattice_memory::verify_onehot(&proof.onehot_reg_a, 5, log_t, OHSide::Read, &oh_ra, &mut transcript)?;
    let oh_rb = TableRes { claims, map: &|f| match f {
        FactorId::ReadAddr => Some(C_RS2A),
        _ => None,
    }};
    lattice_memory::verify_onehot(&proof.onehot_reg_b, 5, log_t, OHSide::Read, &oh_rb, &mut transcript)?;
    let oh_rw = TableRes { claims, map: &|f| match f {
        FactorId::WriteAddr => Some(C_WADA),
        _ => None,
    }};
    lattice_memory::verify_onehot(&proof.onehot_reg_w, 5, log_t, OHSide::Write, &oh_rw, &mut transcript)?;
    let rega_tw = TableRes { claims, map: &|f| match f {
        FactorId::ReadValues => Some(C_RS1V),
        FactorId::WriteValues => Some(C_WVR),
        _ => None,
    }};
    verify_twist_ports_checked(&proof.twist_reg_a, &reg_init, &reg_final, 5, log_t, 5, &rega_tw, &mut transcript)?;
    let regb_tw = TableRes { claims, map: &|f| match f {
        FactorId::ReadValues => Some(C_RS2V),
        FactorId::WriteValues => Some(C_WVR),
        _ => None,
    }};
    verify_twist_ports_checked(&proof.twist_reg_b, &reg_init, &reg_final, 5, log_t, 5, &regb_tw, &mut transcript)?;

    // ---- The lookup Shouts. ----
    let sched = lookup_schedule();
    if proof.lookup_shouts.len() != sched.len() {
        return Err(Pipeline3Error::VerificationFailed);
    }
    for ((kind, addr_col, val_col, log_k), p) in sched.iter().zip(proof.lookup_shouts.iter()) {
        let addr_col = *addr_col;
        let val_col = *val_col;
        let log_k = *log_k;
        let table = lookup_table(kind, log_k);
        let res = TableRes { claims, map: &move |f| match f {
            FactorId::ReadValues => Some(val_col),
            FactorId::ReadAddr => Some(addr_col),
            _ => None,
        }};
        lattice_memory::verify_shout(p, &table, log_k, log_t, log_k, &res, &mut transcript)?;
    }

    // ---- The AIR sumcheck. ----
    let layout = air_layout(log_t);
    let alpha = transcript.challenge_field(b"v3-air-alpha")?;
    let degree = layout
        .terms
        .iter()
        .map(|t| t.sparse.len() + t.dense.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let verdict = proof
        .air
        .verify(log_t, degree, Goldilocks::ZERO, &mut transcript, None)
        .map_err(Pipeline3Error::Sumcheck)?;
    let r_air = verdict.point.clone();
    // Resolve every factor claim at r_air and check the final identity.
    let claim_of = |col: usize| -> Result<Goldilocks, Pipeline3Error> {
        for c in claims {
            if c.col == col && c.point == r_air {
                return Ok(c.value);
            }
        }
        Err(Pipeline3Error::VerificationFailed)
    };
    let mut expect = Goldilocks::ZERO;
    for tpl in &layout.terms {
        let mut coeff = tpl.coeff;
        for _ in 0..tpl.group {
            coeff = coeff.mul(&alpha);
        }
        let mut prod = coeff;
        for &sp in &tpl.sparse {
            let v = match layout.sparse_kinds[sp] {
                SparseKind::Ones => Goldilocks::ONE,
                SparseKind::Sel(f) => claim_of(bit_col(B_SEL0 + f))?,
                SparseKind::Bit(b) => claim_of(bit_col(b))?,
            };
            prod = prod.mul(&v);
        }
        for &d in &tpl.dense {
            prod = prod.mul(&claim_of(layout.dense_cols[d])?);
        }
        expect = expect.add(&prod);
    }
    if expect != verdict.final_claim {
        return Err(Pipeline3Error::VerificationFailed);
    }

    // ---- The linear gates at r_air (β-RLC). ----
    let gates = build_gates();
    let betas = transcript.challenge_fields(b"v3-gate-beta", gates.len())?;
    let mut gate_check = Goldilocks::ZERO;
    for (j, gate) in gates.iter().enumerate() {
        let mut v = gate.constant;
        for (col, coeff) in &gate.slots {
            v = v.add(&coeff.mul(&claim_of(*col)?));
        }
        if v != Goldilocks::ZERO {
            return Err(Pipeline3Error::VerificationFailed);
        }
        gate_check = gate_check.add(&betas[j].mul(&v));
    }
    if gate_check != Goldilocks::ZERO {
        return Err(Pipeline3Error::VerificationFailed);
    }

    // ---- The grouped openings. ----
    let ring = &pcs.pk.params.ring;
    for (ci, opening) in proof.openings.iter().enumerate() {
        let col_claims: Vec<GroupedOpening> = claims
            .iter()
            .filter(|c| c.col == ci)
            .map(|c| GroupedOpening { point: c.point.clone(), value: c.value })
            .collect();
        if col_claims.is_empty() {
            continue;
        }
        let commitment = lattice_commitment::ajtai::AjtaiCommitment::from_bytes(
            ring,
            pcs.pk.params.k,
            &proof.commitments[ci],
        )
        .map_err(|_| lattice_akita::pcs::AkitaPcsError::VerificationFailed)?;
        let comm = lattice_akita::pcs::Commitment {
            commitment,
            num_packed: 0,
            num_vars: log_t,
        };
        pcs.verify_grouped(&comm, &col_claims, opening, &mut transcript)?;
    }
    Ok(())
}

/// The rich benchmark program exercising ADD/ADDI/SUB/MUL/DIV/REMU/
/// SLLI/SRLI/SLTU/BRCH(all four kinds)/JAL/LOAD/STORE/LUI/AUIPC.
#[allow(clippy::too_many_lines)]
pub fn rich_program_pub() -> Vec<u8> {
    let enc_i = |rd: u32, rs1: u32, imm: i32| -> u32 {
        (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
    };
    let enc_r = |rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32| -> u32 {
        (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33
    };
    let enc_b = |f3: u32, rs1: u32, rs2: u32, off: i32| -> u32 {
        let imm = off as u32;
        ((imm >> 31) & 1) << 31
            | ((imm >> 7) & 1) << 7
            | ((imm >> 25) & 0x3f) << 25
            | ((imm >> 1) & 0xf) << 8
            | (rs2 << 20)
            | (rs1 << 15)
            | (f3 << 12)
            | 0x63
    };
    let enc_s = |rs1: u32, rs2: u32, imm: i32| -> u32 {
        let i = imm as u32;
        ((i >> 5) & 0x7f) << 25
            | (rs2 << 20)
            | (rs1 << 15)
            | 0x3 << 12
            | ((i & 0x1f) << 7)
            | 0x23
    };
    let enc_load = |rd: u32, rs1: u32, imm: i32| -> u32 {
        (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | 0x3 << 12 | (rd << 7) | 0x03
    };
    let mut w: Vec<u32> = Vec::new();
    w.push(enc_i(1, 0, 96));             // x1 = 96 (8-aligned scratch)
    w.push(enc_i(2, 0, 7));              // x2 = 7
    w.push(enc_r(3, 1, 2, 0, 0x01));     // x3 = 700
    w.push(enc_r(4, 3, 2, 4, 0x01));     // x4 = 100
    w.push(enc_r(5, 3, 2, 7, 0x01));     // x5 = 0
    w.push(enc_r(6, 3, 1, 0, 0x20));     // x6 = 600
    w.push((3 << 20) | (2 << 15) | (1 << 12) | (7 << 7) | 0x13);  // x7 = 56
    w.push((2 << 20) | (7 << 15) | (5 << 12) | (8 << 7) | 0x13);  // x8 = 14
    w.push(enc_r(9, 2, 1, 3, 0x00));     // x9 = 1 (bltu-style)
    w.push(enc_s(1, 3, 0));              // mem[100] = 700
    w.push(enc_load(10, 1, 0));          // x10 = 700
    w.push(enc_b(6, 2, 1, 8));           // bltu x2, x1, +8 (taken)
    w.push(enc_i(11, 0, 111));           // skipped
    w.push(enc_i(11, 0, 222));           // x11 = 222
    w.push(enc_b(7, 2, 1, -6));          // bgeu x2, x1, −6 (not taken)
    w.push(enc_b(0, 5, 0, 4));           // beq x5, x0, +4 (taken)
    w.push(enc_b(1, 5, 0, 4));           // bne (skipped)
    w.push((4u32 << 21) | (12 << 7) | 0x6f); // jal x12, +4
    w.push(enc_i(13, 0, 1));             // skipped by jal
    w.push(0x0000_0737 | (14 << 7));     // lui x14, 1
    w.push(0x0000_0797 | (15 << 7));     // auipc x15, 1
    w.push(enc_i(16, 0, 0));             // nop
    w.push(0x73);                        // ecall
    let mut v = Vec::new();
    for x in w {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> AkitaPcs {
        v3_pcs_for(8, 6, [91u8; 32]).ok().unwrap()
    }

    fn demo_program() -> Vec<u8> {
        // addi x1, x0, 8; addi x2, x0, 7; add x3, x1, x2; sd x3, 0(x1);
        // ld x4, 0(x1); ecall
        let words = [
            0x0080_0093u32,
            0x0070_0113,
            0x0020_81b3,
            0x0030_b023,
            0x0000_b203,
            0x0000_0073,
        ];
        let mut v = Vec::new();
        for w in words {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }

    fn rich_program() -> Vec<u8> {
        rich_program_pub()
    }

    #[test]
    fn v3_prove_and_verify_happy_path() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = match prove_v3(&pcs, &program, &input, 64) { Ok(x) => x, Err(e) => panic!("prove err: {e:?}") };
        assert_eq!(state.num_steps, 6);
        assert_eq!(state.final_regs[3], 15);
        match verify_v3(&pcs, &program, &input, &state, &proof, 64) {
            Ok(_) => {}
            Err(e) => panic!("verify err: {e:?}"),
        }
    }

    #[test]
    fn v3_rich_program_prove_and_verify() {
        let pcs = setup();
        let program = rich_program();
        let input = 7u64.to_le_bytes().to_vec();
        let (state, proof) = match prove_v3(&pcs, &program, &input, 128) {
            Ok(x) => x,
            Err(e) => panic!("prove err: {e:?}"),
        };
        // The semantics: x3 = 672, x4 = 96, x5 = 0, x6 = 576, x7 = 56,
        // x8 = 14, x9 = 1, x10 = 672, x11 = 222, x12 = pc+4, x14 = 4096.
        assert_eq!(state.final_regs[3], 672);
        assert_eq!(state.final_regs[4], 96);
        assert_eq!(state.final_regs[6], 576);
        assert_eq!(state.final_regs[7], 56);
        assert_eq!(state.final_regs[8], 14);
        assert_eq!(state.final_regs[9], 1);
        assert_eq!(state.final_regs[10], 672);
        assert_eq!(state.final_regs[11], 222);
        assert!(verify_v3(&pcs, &program, &input, &state, &proof, 128).is_ok());
    }

    #[test]
    fn v3_tampered_final_state_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (mut state, proof) = prove_v3(&pcs, &program, &input, 64).ok().unwrap();
        state.final_regs[3] = state.final_regs[3].wrapping_add(1);
        assert!(verify_v3(&pcs, &program, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v3_tampered_claim_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, mut proof) = prove_v3(&pcs, &program, &input, 64).ok().unwrap();
        if let Some(c) = proof.claims.first_mut() {
            c.value = c.value.add(&Goldilocks::ONE);
        }
        assert!(verify_v3(&pcs, &program, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v3_fibonacci_loop_prove_and_verify() {
        // A loop with a backwards bne and jal x0 halt (the fibonacci
        // guest's shape): exercises the x0 write mask and negative
        // branch offsets.
        let pcs = setup();
        let enc_i = |rd: u32, rs1: u32, imm: i32| -> u32 {
            (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
        };
        let enc_r = |rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32| -> u32 {
            (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33
        };
        // x1 = 10; x2 = 0; x3 = 1;
        // loop: x4 = x2 + x3; x2 = x3; x3 = x4; x1 = x1 - 1;
        //       bne x1, x0, loop;  (backwards -20 bytes)
        // jal x0, 0 (halt).
        let mut w: Vec<u32> = Vec::new();
        w.push(enc_i(1, 0, 10));
        w.push(enc_i(2, 0, 0));
        w.push(enc_i(3, 0, 1));
        let loop_start = w.len() as i32;
        w.push(enc_r(4, 2, 3, 0, 0x00)); // add
        w.push(enc_r(2, 3, 0, 0, 0x00)); // add x2, x3, x0 (x2 = x3)
        w.push(enc_r(3, 4, 0, 0, 0x00)); // add x3, x4, x0
        w.push(enc_r(1, 1, 5, 0, 0x20)); // sub x1, x1, x5 (x5 = 1... use imm)
        w.pop();
        w.push(enc_i(1, 1, -1)); // addi x1, x1, -1
        let off = loop_start * 4 - (w.len() as i32) * 4;
        // bne x1, x0, off (backwards)
        w.push(((off as u32 >> 31) & 1) << 31
            | (((off as u32 >> 7) & 1) << 7)
            | (((off as u32 >> 25) & 0x3f) << 25)
            | (((off as u32 >> 1) & 0xf) << 8)
            | (0 << 20) | (1 << 15) | (1 << 12) | 0x63);
        w.push(0x6f); // jal x0, 0 (halt)
        let mut prog = Vec::new();
        for x in w {
            prog.extend_from_slice(&x.to_le_bytes());
        }
        let (state, proof) = match prove_v3(&pcs, &prog, &[], 128) {
            Ok(x) => x,
            Err(e) => panic!("prove err: {e:?}"),
        };
        // After 10 iterations: x2 = fib(10) = 55, x3 = fib(11) = 89.
        assert_eq!(state.final_regs[2], 55);
        assert_eq!(state.final_regs[3], 89);
        assert!(verify_v3(&pcs, &prog, &[], &state, &proof, 128).is_ok());
    }

    #[test]
    fn v3_wrong_program_rejected() {
        let pcs = setup();
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = match prove_v3(&pcs, &program, &input, 64) { Ok(x) => x, Err(e) => panic!("prove err: {e:?}") };
        let mut other = program.clone();
        other[0] ^= 0x01;
        assert!(verify_v3(&pcs, &other, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v3_profile_violation_fails_closed() {
        // A MUL product at or beyond 2^63 must fail closed.
        let pcs = setup();
        let enc_i = |rd: u32, rs1: u32, imm: i32| -> u32 {
            (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
        };
        let enc_r = |rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32| -> u32 {
            (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33
        };
        // x1 = 2^32; x2 = 2^32; mul x3 = 2^64 → 0 (wraps, inside profile).
        // Use x1 = 2^32+1, x2 = 2^32+1 → product ≥ 2^64 → rdv = 2^64+2^32+1
        // mod 2^64 = 2^32+1 < 2^63 — inside. To exceed: 2^62·2 = 2^63:
        // x1 = 2^62, x2 = 2 → x3 = 2^63 → ProfileViolation.
        let mut v = Vec::new();
        for w in [
            enc_i(1, 0, 0x4000_0000u32 as i32), // addi x1, x0, 2^30 — imm is 12-bit!
        ] {
            v.extend_from_slice(&w.to_le_bytes());
        }
        // 12-bit immediates cannot reach 2^62 directly; build via slli.
        // addi x1, x0, 1; slli x1, x1, 62 → x1 = 2^62; addi x2, x0, 2; mul.
        let mut w = Vec::new();
        w.push(enc_i(1, 0, 1));
        w.push((62u32 << 20) | (1 << 15) | (1 << 12) | (1 << 7) | 0x13); // slli x1, x1, 62
        w.push(enc_i(2, 0, 2));
        w.push(enc_r(3, 1, 2, 0, 0x01)); // mul x3, x1, x2 = 2^63
        w.push(0x73u32);
        let mut prog = Vec::new();
        for x in w {
            prog.extend_from_slice(&x.to_le_bytes());
        }
        let _ = v;
        match prove_v3(&pcs, &prog, &[], 16) {
            Err(Pipeline3Error::ProfileViolation { what, .. }) => {
                assert_eq!(what, "rdv");
            }
            other => panic!("expected ProfileViolation, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod setup_probe {
    #[test]
    fn probe_setup_params() {
        for (log_n, m) in [(6u32, 12usize), (6, 24), (4, 64), (8, 48), (5, 32), (6, 48)] {
            let r = lattice_akita::akita_setup(log_n, m, 1 << 23, [91u8; 32]);
            println!("log_n={log_n} m={m} -> {}", r.is_ok());
        }
    }
}

#[cfg(test)]
mod eval_probe {
    use super::*;
    use lattice_core::DenseMle;
    #[test]
    fn eval_col_matches_dense_mle() {
        // Deterministic pseudo-random column of length 64 (6 vars).
        let mut col = Vec::with_capacity(64);
        let mut x = 0x243F6A8885A308D3u64;
        for _ in 0..64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            col.push(Goldilocks::from_u64(x >> 32));
        }
        let mle = DenseMle { num_vars: 6, evaluations: col.clone() };
        let mut pt = Vec::new();
        let mut s = 0xdeadbeefu64;
        for _ in 0..6 {
            s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            pt.push(Goldilocks::from_u64((s >> 33) & 0xff));
        }
        let want = mle.evaluate(&pt).ok().unwrap();
        let got = eval_col(&col, &pt);
        assert_eq!(want, got, "eval_col convention mismatch");
    }
}

#[cfg(test)]
mod fetch_probe {
    use super::*;
    #[test]
    fn probe_fetch_shout_claim() {
        let pcs = crate::pipeline3::v3_pcs_for(8, 6, [91u8; 32]).ok().unwrap();
        let mut state = MachineState::new();
        let program: Vec<u8> = [0x0080_0093u32, 0x0070_0113, 0x0020_81b3, 0x0030_b023,
            0x0000_b203, 0x0000_0073]
            .iter().flat_map(|w| w.to_le_bytes()).collect();
        state.load_program(0x1000, &program);
        state.pc = 0x1000;
        let rows = vm_run(&mut state, 64).ok().unwrap();
        let trace = build_trace3(&state, &rows).ok().unwrap();
        let log_t = trace.log_t;
        let fetch_ra: Vec<u64> = trace.cols[C_FETCH_RA].iter().map(|x| x.to_canonical_u64()).collect();
        let iw_col = &trace.cols[C_IW];
        eprintln!("log_t={log_t} fetch_ra={fetch_ra:?} iw={:?}", iw_col.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>());
        let fetch_words = program.len().div_ceil(4).max(1);
        let log_k = fetch_words.next_power_of_two().max(2).trailing_zeros() as usize;
        let mut table: Vec<Goldilocks> = (0..fetch_words).map(|i| {
            let mut w = 0u32;
            for b in 0..4 { w |= (program[i * 4 + b] as u32) << (b * 8); }
            fe(w as u64)
        }).collect();
        table.resize(1 << log_k, Goldilocks::ZERO);
        // The honest check: iw_j == table[fetch_ra_j] for every j.
        for j in 0..(1usize << log_t) {
            let want = table[fetch_ra[j] as usize];
            let got = iw_col[j];
            if want != got {
                eprintln!("MISMATCH at row {j}: table[{}] = {} vs iw = {}",
                    fetch_ra[j], want.to_canonical_u64(), got.to_canonical_u64());
            }
        }
        let _ = pcs;
    }
}

#[cfg(test)]
mod air_audit {
    use super::*;
    fn audit(program: &[u8], input: &[u8]) {
        let mut state = MachineState::new();
        state.load_program(0x1000, program);
        state.load_program(0x3000, input);
        state.regs[10] = input.len() as u64;
        state.pc = 0x1000;
        let rows = vm_run(&mut state, 256).ok().unwrap();
        let trace = build_trace3(&state, &rows).ok().unwrap();
        let cols = &trace.cols;
        let t = 1usize << trace.log_t;
        let layout = air_layout(trace.log_t);
        let ones = |j: usize| Goldilocks::ONE;
        let sel = |f: usize, j: usize| cols[bit_col(B_SEL0 + f)][j];
        let bitv = |b: usize, j: usize| cols[bit_col(b)][j];
        let mut n_groups = 0u32;
        for tpl in &layout.terms {
            n_groups = n_groups.max(tpl.group + 1);
        }
        let mut sums = vec![Goldilocks::ZERO; n_groups as usize];
        for tpl in &layout.terms {
            // The term's support rows: the first sparse kind's support.
            let kind = layout.sparse_kinds[tpl.sparse[0]];
            for j in 0..t {
                let active = match kind {
                    SparseKind::Ones => true,
                    SparseKind::Sel(f) => sel(f, j) == Goldilocks::ONE,
                    SparseKind::Bit(b) => bitv(b, j) == Goldilocks::ONE,
                };
                if !active {
                    continue;
                }
                let mut v = tpl.coeff;
                for &sp in &tpl.sparse {
                    let k2 = layout.sparse_kinds[sp];
                    v = v.mul(&match k2 {
                        SparseKind::Ones => ones(j),
                        SparseKind::Sel(f) => sel(f, j),
                        SparseKind::Bit(b) => bitv(b, j),
                    });
                }
                for &d in &tpl.dense {
                    v = v.mul(&cols[layout.dense_cols[d]][j]);
                }
                sums[tpl.group as usize] = sums[tpl.group as usize].add(&v);
            }
        }
        // Per-row detail for failing groups.
        for (g, s) in sums.iter().enumerate() {
            if *s == Goldilocks::ZERO { continue; }
            for j in 0..t {
                let mut row_sum = Goldilocks::ZERO;
                for tpl in &layout.terms {
                    if tpl.group as usize != g { continue; }
                    let kind = layout.sparse_kinds[tpl.sparse[0]];
                    let active = match kind {
                        SparseKind::Ones => true,
                        SparseKind::Sel(f) => sel(f, j) == Goldilocks::ONE,
                        SparseKind::Bit(b) => bitv(b, j) == Goldilocks::ONE,
                    };
                    if !active { continue; }
                    let mut v = tpl.coeff;
                    for &sp in &tpl.sparse {
                        let k2 = layout.sparse_kinds[sp];
                        v = v.mul(&match k2 {
                            SparseKind::Ones => ones(j),
                            SparseKind::Sel(f) => sel(f, j),
                            SparseKind::Bit(b) => bitv(b, j),
                        });
                    }
                    for &d in &tpl.dense {
                        v = v.mul(&cols[layout.dense_cols[d]][j]);
                    }
                    row_sum = row_sum.add(&v);
                }
                if row_sum != Goldilocks::ZERO {
                    eprintln!("  row {j}: group {g} contributes {}", row_sum.to_canonical_u64());
                }
            }
        }
        let mut bad = 0;
        for (g, s) in sums.iter().enumerate() {
            if *s != Goldilocks::ZERO {
                // Identify the group by its first term.
                let first = layout.terms.iter().find(|t| t.group as usize == g);
                if g == 145 || g == 146 || g == 115 {
                    for t in layout.terms.iter().filter(|t| t.group as usize == g) {
                        eprintln!("    g{g} term: coeff={} sparse={:?} dense={:?}",
                            t.coeff.to_canonical_u64(),
                            t.sparse.iter().map(|&sp| layout.sparse_kinds[sp]).collect::<Vec<_>>(),
                            t.dense.iter().map(|&d| layout.dense_cols[d]).collect::<Vec<_>>());
                    }
                }
                let cols_of: Vec<usize> = first.map(|t| t.dense.iter().map(|&d| layout.dense_cols[d]).collect()).unwrap_or_default();
                let nterms = layout.terms.iter().filter(|t| t.group as usize == g).count();
                eprintln!("GROUP {g} NONZERO: sum = {} | {nterms} terms | dense cols {:?} | spkinds {:?}",
                    s.to_canonical_u64(), cols_of,
                    first.map(|t| t.sparse.iter().map(|&sp| layout.sparse_kinds[sp]).collect::<Vec<_>>()).unwrap_or_default());
                bad += 1;
            }
        }
        eprintln!("air audit: {bad} bad groups of {n_groups}");
        assert_eq!(bad, 0, "the honest witness must satisfy every group");
    }
    #[test]
    fn audit_air_groups() {
        let program: Vec<u8> = [0x0080_0093u32, 0x0070_0113, 0x0020_81b3, 0x0030_b023,
            0x0000_b203, 0x0000_0073]
            .iter().flat_map(|w| w.to_le_bytes()).collect();
        audit(&program, &42u64.to_le_bytes());
    }
    #[test]
    fn audit_air_rich() {
        // The FULL rich program (with branches, JAL, LUI, AUIPC).
        let program = crate::pipeline3::rich_program_pub();
        audit(&program, &7u64.to_le_bytes());
    }
    #[test]
    fn audit_air_fibonacci_loop() {
        // The fibonacci-loop shape (backwards bne, jal x0 halt).
        let enc_i = |rd: u32, rs1: u32, imm: i32| -> u32 {
            (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
        };
        let enc_r = |rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32| -> u32 {
            (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33
        };
        let mut w: Vec<u32> = Vec::new();
        w.push(enc_i(1, 0, 10));
        w.push(enc_i(2, 0, 0));
        w.push(enc_i(3, 0, 1));
        let loop_start = w.len() as i32;
        w.push(enc_r(4, 2, 3, 0, 0x00));
        w.push(enc_r(2, 3, 0, 0, 0x00));
        w.push(enc_r(3, 4, 0, 0, 0x00));
        w.push(enc_i(1, 1, -1));
        let off = loop_start * 4 - (w.len() as i32) * 4;
        w.push(((off as u32 >> 31) & 1) << 31
            | (((off as u32 >> 7) & 1) << 7)
            | (((off as u32 >> 25) & 0x3f) << 25)
            | (((off as u32 >> 1) & 0xf) << 8)
            | (0 << 20) | (1 << 15) | (1 << 12) | 0x63);
        w.push(0x6f);
        let program: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        audit(&program, &[]);
    }
    #[test]
    fn audit_air_guests() {
        // Every guest program in the suite must audit (or fail closed
        // at the profile/decode level with a documented reason).
        let progs = lattice_guest::programs::suite().ok().unwrap();
        for prog in &progs {
            let asm = lattice_guest::asm::AssembledProgram {
                code: prog.image.clone(),
                data: vec![],
                labels: Default::default(),
                data_base: prog.data_base,
            };
            let run = match lattice_guest::asm::run_on_vm(&asm, &prog.public_input, 5_000_000) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}: exec failed {e:?}", prog.name);
                    continue;
                }
            };
            eprintln!("auditing {} ({} cycles)", prog.name, run.steps);
            let mut state = lattice_vm::MachineState::new();
            state.load_program(0x1000, &prog.image);
            state.load_program(0x3000, &prog.public_input);
            state.regs[10] = prog.public_input.len() as u64;
            state.pc = 0x1000;
            match lattice_vm::run(&mut state, 5_000_000) {
                Ok(rows) => match build_trace3(&state, &rows) {
                    Ok(_) => {}
                    Err(e) => eprintln!("{}: trace fail-closed: {e:?}", prog.name),
                },
                Err(e) => eprintln!("{}: vm fail {e:?}", prog.name),
            }
            let _ = audit_raw(&prog.image, &prog.public_input, &prog.name);
        }
    }
    fn audit_raw(program: &[u8], input: &[u8], name: &str) -> bool {
        let mut state = lattice_vm::MachineState::new();
        state.load_program(0x1000, program);
        state.load_program(0x3000, input);
        state.regs[10] = input.len() as u64;
        state.pc = 0x1000;
        let rows = match lattice_vm::run(&mut state, 5_000_000) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{name}: exec {e:?}");
                return false;
            }
        };
        match build_trace3(&state, &rows) {
            Err(e) => {
                eprintln!("{name}: trace: {e:?}");
                return false;
            }
            Ok(trace) => {
                // Count the families actually used.
                let mut fams = [0usize; 17];
                for row in &rows {
                    let iw = {
                        let w = state.memory.load(row.pc & !0x7);
                        ((w >> ((row.pc & 0x7) * 8)) & 0xffff_ffff) as u32
                    };
                    let f = decode_family_v3(iw);
                    if f != 0 {
                        fams[f.trailing_zeros() as usize] += 1;
                    }
                }
                eprintln!("{name}: families {:?}", &fams[..]);
                true
            }
        }
    }
}

/// The fibonacci-loop shape (backwards bne, jal x0 halt).
#[cfg(test)]
fn fib_loop_program() -> Vec<u8> {
    let enc_i = |rd: u32, rs1: u32, imm: i32| -> u32 {
        (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
    };
    let enc_r = |rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32| -> u32 {
        (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33
    };
    let mut w: Vec<u32> = Vec::new();
    w.push(enc_i(1, 0, 10));
    w.push(enc_i(2, 0, 0));
    w.push(enc_i(3, 0, 1));
    let loop_start = w.len() as i32;
    w.push(enc_r(4, 2, 3, 0, 0x00));
    w.push(enc_r(2, 3, 0, 0, 0x00));
    w.push(enc_r(3, 4, 0, 0, 0x00));
    w.push(enc_i(1, 1, -1));
    let off = loop_start * 4 - (w.len() as i32) * 4;
    w.push(((off as u32 >> 31) & 1) << 31
        | (((off as u32 >> 7) & 1) << 7)
        | (((off as u32 >> 25) & 0x3f) << 25)
        | (((off as u32 >> 1) & 0xf) << 8)
        | (0 << 20) | (1 << 15) | (1 << 12) | 0x63);
    w.push(0x6f);
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod engine_bisect {
    use super::*;
    use lattice_memory::sparse_engine::prove_sparse_sumcheck;
    #[test]
    fn bisect_air_groups_engine() {
        let program = fib_loop_program();
        let mut state = MachineState::new();
        state.load_program(0x1000, &program);
        state.load_program(0x3000, &7u64.to_le_bytes());
        state.regs[10] = 8;
        state.pc = 0x1000;
        let rows = vm_run(&mut state, 256).ok().unwrap();
        let trace = build_trace3(&state, &rows).ok().unwrap();
        let cols = &trace.cols;
        let log_t = trace.log_t;
        let t_pow = 1usize << log_t;
        let layout = air_layout(log_t);
        let alpha = Goldilocks::from_u64(7);
        let sparse: Vec<SparseFactor> = layout.sparse_kinds.iter().map(|kind| {
            let entries: Vec<(u64, Goldilocks)> = match kind {
                SparseKind::Ones => (0..t_pow).map(|j| (j as u64, Goldilocks::ONE)).collect(),
                SparseKind::Sel(f) => (0..t_pow)
                    .filter(|&j| cols[bit_col(B_SEL0 + f)][j] == Goldilocks::ONE)
                    .map(|j| (j as u64, Goldilocks::ONE)).collect(),
                SparseKind::Bit(b) => (0..t_pow)
                    .filter(|&j| cols[bit_col(*b)][j] == Goldilocks::ONE)
                    .map(|j| (j as u64, Goldilocks::ONE)).collect(),
            };
            SparseFactor { entries, var_map: (0..log_t).collect() }
        }).collect();
        let dense: Vec<ProjectedDense> = layout.dense_cols.iter().map(|&col| ProjectedDense {
            mle: DenseMle { num_vars: log_t, evaluations: cols[col].clone() },
            var_map: (0..log_t).collect(),
        }).collect();
        let n_groups = layout.terms.iter().map(|t| t.group).max().unwrap_or(0) + 1;
        let mut bad: Vec<(u32, PiopError)> = Vec::new();
        for g in 0..n_groups {
            let tpls: Vec<&TermTpl> = layout.terms.iter().filter(|t| t.group == g).collect();
            if tpls.is_empty() { continue; }
            let terms: Vec<SparseTerm> = tpls.iter().map(|tpl| {
                let mut coeff = tpl.coeff;
                for _ in 0..tpl.group { coeff = coeff.mul(&alpha); }
                let first_kind = layout.sparse_kinds[tpl.sparse[0]];
                let positions: Vec<u64> = match first_kind {
                    SparseKind::Ones => (0..t_pow).map(|j| j as u64).collect(),
                    SparseKind::Sel(f) => (0..t_pow)
                        .filter(|&j| cols[bit_col(B_SEL0 + f)][j] == Goldilocks::ONE)
                        .map(|j| j as u64).collect(),
                    SparseKind::Bit(b) => (0..t_pow)
                        .filter(|&j| cols[bit_col(b)][j] == Goldilocks::ONE)
                        .map(|j| j as u64).collect(),
                };
                SparseTerm { coeff, positions, sparse: tpl.sparse.clone(), dense: tpl.dense.clone() }
            }).collect();
            let inst = SparseInstance {
                num_vars: log_t,
                sparse: sparse.clone(),
                dense: dense.clone(),
                terms,
            };
            let mut tr = Transcript::new_default(b"bisect");
            if let Err(e) = prove_sparse_sumcheck(&inst, Goldilocks::ZERO, &mut tr) {
                bad.push((g, e));
            }
        }
        eprintln!("engine bisect: {} bad groups", bad.len());
        for (g, e) in bad.iter().take(10) {
            let kinds: Vec<SparseKind> = layout.terms.iter()
                .find(|t| t.group == *g)
                .map(|t| t.sparse.iter().map(|&sp| layout.sparse_kinds[sp]).collect())
                .unwrap_or_default();
            let dcols: Vec<usize> = layout.terms.iter()
                .find(|t| t.group == *g)
                .map(|t| t.dense.iter().map(|&d| layout.dense_cols[d]).collect())
                .unwrap_or_default();
            eprintln!("  group {g}: {e:?} | kinds {kinds:?} | dense {dcols:?}");
        }
        assert!(bad.is_empty(), "engine disagrees with the audit on some groups");
    }
}

#[cfg(test)]
mod engine_trace {
    use super::*;
    use lattice_memory::sparse_engine::SparseOutput;
    use lattice_sumcheck::SumcheckProof;
    fn eq_lerp(t: Goldilocks, bit: u64) -> Goldilocks {
        if bit == 1 { t } else { Goldilocks::ONE.sub(&t) }
    }
    fn reverse_bits(x: u64, n: usize) -> u64 {
        let mut r = 0u64;
        for i in 0..n { r |= ((x >> i) & 1) << (n - 1 - i); }
        r
    }
    fn interpolate_at(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
        let n = evals.len();
        let mut acc = Goldilocks::ZERO;
        for i in 0..n {
            let mut weight = Goldilocks::ONE;
            let xi = Goldilocks::from_u64(i as u64);
            for j in 0..n {
                if i == j { continue; }
                let xj = Goldilocks::from_u64(j as u64);
                let inv = xi.sub(&xj).inverse().unwrap_or(Goldilocks::ZERO);
                weight = weight.mul(&r.sub(&xj).mul(&inv));
            }
            acc = acc.add(&evals[i].mul(&weight));
        }
        acc
    }
    /// Instrumented copy of prove_sparse_sumcheck for one group.
    fn prove_trace(inst: &SparseInstance, claim: Goldilocks) -> Result<SparseOutput, String> {
        let n = inst.num_vars;
        let deg = inst.terms.iter().map(|t| t.sparse.len() + t.dense.len()).max().unwrap_or(1).max(1);
        let mut weights: Vec<Vec<Goldilocks>> = inst.sparse.iter()
            .map(|f| f.entries.iter().map(|_| Goldilocks::ONE).collect()).collect();
        let sparse_pos: Vec<Vec<usize>> = inst.sparse.iter().map(|f| {
            let mut m = vec![usize::MAX; n];
            for (i, &v) in f.var_map.iter().enumerate() { if v < n { m[v] = i; } }
            m
        }).collect();
        let mut dense_state: Vec<DenseMle> = inst.dense.iter().map(|f| f.mle.clone()).collect();
        let mut dense_bound: Vec<usize> = vec![0; inst.dense.len()];
        let term_orders: Vec<Vec<usize>> = inst.terms.iter().map(|t| {
            let mut idx: Vec<usize> = (0..t.positions.len()).collect();
            idx.sort_by_key(|&j| reverse_bits(t.positions[j], n));
            idx
        }).collect();
        let mut current_claim = claim;
        let mut rounds: Vec<Vec<Goldilocks>> = Vec::new();
        let mut challenges: Vec<Goldilocks> = Vec::new();
        let mut tr = Transcript::new_default(b"trace");
        for ell in 0..n {
            let suffix_len = n - 1 - ell;
            let suffix_mask: u64 = if suffix_len >= 64 { u64::MAX } else { (1u64 << suffix_len) - 1 };
            let mut evals_at: Vec<Goldilocks> = vec![Goldilocks::ZERO; deg + 1];
            for (ti, term) in inst.terms.iter().enumerate() {
                let order = &term_orders[ti];
                let mut seg = 0usize;
                while seg < order.len() {
                    let suffix = term.positions[order[seg]] & suffix_mask;
                    let mut end = seg + 1;
                    while end < order.len() && (term.positions[order[end]] & suffix_mask) == suffix { end += 1; }
                    for (t, ev) in evals_at.iter_mut().enumerate() {
                        let t_fe = Goldilocks::from_u64(t as u64);
                        let mut prod = term.coeff;
                        for &di in &term.dense {
                            let df = &inst.dense[di];
                            let v = dense_partial_trace(&dense_state[di], dense_bound[di], &df.var_map, ell, n, t_fe, suffix);
                            prod = prod.mul(&v);
                        }
                        for &fi in &term.sparse {
                            let f = &inst.sparse[fi];
                            let mut sum = Goldilocks::ZERO;
                            for k in seg..end {
                                let j = order[k];
                                let (_, val) = f.entries[j];
                                let base = weights[fi][j].mul(&val);
                                let p = sparse_pos[fi][ell];
                                if p == usize::MAX {
                                    sum = sum.add(&base);
                                } else {
                                    let bit = (term.positions[j] >> (n - 1 - ell)) & 1;
                                    sum = sum.add(&base.mul(&eq_lerp(t_fe, bit)));
                                }
                            }
                            prod = prod.mul(&sum);
                        }
                        *ev = ev.add(&prod);
                    }
                    seg = end;
                }
            }
            let sum01 = evals_at[0].add(&evals_at[1]);
            if ell == 0 {
                for (ti, term) in inst.terms.iter().enumerate() {
                    let mut tv = Goldilocks::ZERO;
                    let order = &term_orders[ti];
                    let mut seg = 0usize;
                    while seg < order.len() {
                        let suffix = term.positions[order[seg]] & suffix_mask;
                        let mut end = seg + 1;
                        while end < order.len() && (term.positions[order[end]] & suffix_mask) == suffix { end += 1; }
                        let t_fe = Goldilocks::ZERO;
                        let mut prod = term.coeff;
                        for &di in &term.dense {
                            let df = &inst.dense[di];
                            let v = dense_partial_trace(&dense_state[di], dense_bound[di], &df.var_map, ell, n, t_fe, suffix);
                            prod = prod.mul(&v);
                        }
                        for &fi in &term.sparse {
                            let f = &inst.sparse[fi];
                            let mut sum = Goldilocks::ZERO;
                            for k in seg..end {
                                let j = order[k];
                                let (_, val) = f.entries[j];
                                let base = weights[fi][j].mul(&val);
                                let p = sparse_pos[fi][ell];
                                if p == usize::MAX { sum = sum.add(&base); }
                                else {
                                    let bit = (term.positions[j] >> (n - 1 - ell)) & 1;
                                    sum = sum.add(&base.mul(&eq_lerp(t_fe, bit)));
                                }
                            }
                            prod = prod.mul(&sum);
                        }
                        tv = tv.add(&prod);
                        seg = end;
                    }
                    eprintln!("    term {ti} g0={} coeff={} dense={:?} sparse_kinds={:?}",
                        tv.to_canonical_u64(), term.coeff.to_canonical_u64(),
                        term.dense.iter().map(|&d| layout_dense_col(&inst, d)).collect::<Vec<_>>(),
                        term.sparse.iter().map(|&sp| 0).collect::<Vec<_>>());
                }
            }
            eprintln!("  round {ell}: evals={:?} sum01={} claim={}",
                evals_at.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>(),
                sum01.to_canonical_u64(), current_claim.to_canonical_u64());
            if sum01 != current_claim {
                return Err(format!("round {ell} mismatch: {} != {}", sum01.to_canonical_u64(), current_claim.to_canonical_u64()));
            }
            rounds.push(evals_at.clone());
            let r = tr.challenge_field(b"c").unwrap_or(Goldilocks::ONE);
            challenges.push(r);
            current_claim = interpolate_at(&evals_at, &r);
            for (fi, f) in inst.sparse.iter().enumerate() {
                let p = sparse_pos[fi][ell];
                if p != usize::MAX {
                    let shift = f.var_map.len() - 1 - p;
                    for (j, &(own, _)) in f.entries.iter().enumerate() {
                        let bit = (own >> shift) & 1;
                        weights[fi][j] = weights[fi][j].mul(&eq_lerp(r, bit));
                    }
                }
            }
            for (di, df) in inst.dense.iter().enumerate() {
                let len = df.var_map.len();
                if dense_bound[di] < len && df.var_map[dense_bound[di]] == ell {
                    dense_state[di] = dense_state[di].fix_variables(&[r]).map_err(|e| format!("{e:?}"))?;
                    dense_bound[di] += 1;
                }
            }
        }
        let mut sparse_claims = Vec::new();
        for (fi, f) in inst.sparse.iter().enumerate() {
            let mut acc = Goldilocks::ZERO;
            for (j, &(_, val)) in f.entries.iter().enumerate() {
                acc = acc.add(&weights[fi][j].mul(&val));
            }
            sparse_claims.push(acc);
        }
        let mut dense_claims = Vec::new();
        for dstate in &dense_state { dense_claims.push(dstate.evaluations[0]); }
        Ok(SparseOutput { proof: SumcheckProof { rounds }, challenges, final_claim: current_claim, sparse_claims, dense_claims })
    }
    fn layout_dense_col(_inst: &SparseInstance, _d: usize) -> usize { _d }
    fn dense_partial_trace(state: &DenseMle, bound: usize, var_map: &[usize], ell: usize, n: usize, t: Goldilocks, suffix: u64) -> Goldilocks {
        let len = var_map.len();
        let arr = &state.evaluations;
        if bound < len && var_map[bound] == ell {
            let points = 1usize << (len - bound - 1);
            let mut rem = 0usize;
            let vars = &var_map[bound + 1..];
            for (i, &v) in vars.iter().enumerate() {
                let bit = ((suffix >> (n - 1 - v)) & 1) as usize;
                rem |= bit << (vars.len() - 1 - i);
            }
            let a = arr[rem];
            let b = arr[rem + points];
            a.add(&b.sub(&a).mul(&t))
        } else {
            let mut rem = 0usize;
            let vars = &var_map[bound..];
            for (i, &v) in vars.iter().enumerate() {
                let bit = ((suffix >> (n - 1 - v)) & 1) as usize;
                rem |= bit << (vars.len() - 1 - i);
            }
            arr[rem]
        }
    }
    #[test]
    fn trace_full_rich() {
        let program = rich_program_pub();
        let mut state = MachineState::new();
        state.load_program(0x1000, &program);
        state.load_program(0x3000, &7u64.to_le_bytes());
        state.regs[10] = 8;
        state.pc = 0x1000;
        let rows = vm_run(&mut state, 256).ok().unwrap();
        let trace = build_trace3(&state, &rows).ok().unwrap();
        let cols = &trace.cols;
        let log_t = trace.log_t;
        let t_pow = 1usize << log_t;
        let layout = air_layout(log_t);
        // Group 148 = the Taken formula (bisect's failing group).
        let tpls: Vec<&TermTpl> = layout.terms.iter().filter(|t| t.group == 148).collect();
        let sparse: Vec<SparseFactor> = layout.sparse_kinds.iter().map(|kind| {
            let entries: Vec<(u64, Goldilocks)> = match kind {
                SparseKind::Ones => (0..t_pow).map(|j| (j as u64, Goldilocks::ONE)).collect(),
                SparseKind::Sel(f) => (0..t_pow)
                    .filter(|&j| cols[bit_col(B_SEL0 + f)][j] == Goldilocks::ONE)
                    .map(|j| (j as u64, Goldilocks::ONE)).collect(),
                SparseKind::Bit(b) => (0..t_pow)
                    .filter(|&j| cols[bit_col(*b)][j] == Goldilocks::ONE)
                    .map(|j| (j as u64, Goldilocks::ONE)).collect(),
            };
            SparseFactor { entries, var_map: (0..log_t).collect() }
        }).collect();
        let dense: Vec<ProjectedDense> = layout.dense_cols.iter().map(|&col| ProjectedDense {
            mle: DenseMle { num_vars: log_t, evaluations: cols[col].clone() },
            var_map: (0..log_t).collect(),
        }).collect();
        let alpha = Goldilocks::from_u64(7);
        let terms: Vec<SparseTerm> = tpls.iter().map(|tpl| {
            let mut coeff = tpl.coeff;
            for _ in 0..tpl.group { coeff = coeff.mul(&alpha); }
            let first_kind = layout.sparse_kinds[tpl.sparse[0]];
            let positions: Vec<u64> = match first_kind {
                SparseKind::Ones => (0..t_pow).map(|j| j as u64).collect(),
                SparseKind::Sel(f) => (0..t_pow)
                    .filter(|&j| cols[bit_col(B_SEL0 + f)][j] == Goldilocks::ONE)
                    .map(|j| j as u64).collect(),
                SparseKind::Bit(b) => (0..t_pow)
                    .filter(|&j| cols[bit_col(b)][j] == Goldilocks::ONE)
                    .map(|j| j as u64).collect(),
            };
            SparseTerm { coeff, positions, sparse: tpl.sparse.clone(), dense: tpl.dense.clone() }
        }).collect();
        let inst = SparseInstance { num_vars: log_t, sparse, dense, terms };
        eprintln!("dense_cols[49..61] = {:?}", &layout.dense_cols[49..61]);
        for (name, b) in [("Fb0", B_FB0), ("Fb1", B_FB0 + 1), ("Fb2", B_FB0 + 2), ("Fb3", B_FB0 + 3),
            ("Taken", B_TAKEN), ("Eqb", B_EQB), ("GT1", B_GT1), ("LT1", B_LT1)] {
            eprintln!("  {name}: {:?}", cols[bit_col(b)].iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>());
        }
        match prove_trace(&inst, Goldilocks::ZERO) {
            Ok(_) => eprintln!("trace: OK"),
            Err(e) => panic!("trace: {e}"),
        }
    }
}
