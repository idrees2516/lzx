//! The staged v2 zkVM pipeline: `prove_v2` / `verify_v2`.
//!
//! The verifier NEVER re-executes the program. Every step is constrained
//! through committed columns:
//!
//! * **Fetch (Shout)**: instruction word = public program image at pc/4.
//! * **Decode (Shout)**: family word = public decode table at the key
//!   `(opcode|funct3|funct6)`; routing one-hot over 16 families.
//! * **Pow2 (Shout)**: shamt -> 2^shamt binds shift semantics.
//! * **Range (Shout x2)**: 11/10-bit identity tables range-check chunks.
//! * **Register Twist (x2)**: rs1/rs2 read ports observe the last write.
//! * **RAM Twist**: same for data memory with public init/final states.
//! * **AIR**: one batched RLC sumcheck over the constraint polynomials.
//!
//! All factor claims funnel into ONE grouped Ajtai opening per committed
//! column — batching the constraint claims through the stage-4 legs.
//!
//! Guest discipline (P0 profile, fail-closed): register values < 2^33,
//! non-wrapping arithmetic, only 64-bit loads/stores, the instruction
//! subset of `decode_family`, signed values via biased u64 encoding.



/// Public state: final registers, final RAM window, step count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicStateV2 {
    pub final_regs: [u64; 32],
    pub final_ram: Vec<u64>,
    pub num_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    Execution(lattice_vm::ExecError),
    Memory(PiopError),
    Pcs(lattice_akita::pcs::AkitaPcsError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Transcript(lattice_core::transcript::TranscriptError),
    Mle(lattice_core::mle::MleError),
    UnsupportedInstruction { pc: u64, word: u32 },
    BadShape(String),
    VerificationFailed,
}

impl From<PiopError> for PipelineError {
    fn from(e: PiopError) -> Self {
        PipelineError::Memory(e)
    }
}
impl From<lattice_akita::pcs::AkitaPcsError> for PipelineError {
    fn from(e: lattice_akita::pcs::AkitaPcsError) -> Self {
        PipelineError::Pcs(e)
    }
}
impl From<lattice_sumcheck::SumcheckError> for PipelineError {
    fn from(e: lattice_sumcheck::SumcheckError) -> Self {
        PipelineError::Sumcheck(e)
    }
}
impl From<lattice_sumcheck::VirtualPolyError> for PipelineError {
    fn from(e: lattice_sumcheck::VirtualPolyError) -> Self {
        PipelineError::Virtual(e)
    }
}
impl From<lattice_core::transcript::TranscriptError> for PipelineError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        PipelineError::Transcript(e)
    }
}
impl From<lattice_core::mle::MleError> for PipelineError {
    fn from(e: lattice_core::mle::MleError) -> Self {
        PipelineError::Mle(e)
    }
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}
// ---------------------------------------------------------------------------
// Columns
// ---------------------------------------------------------------------------

/// Committed column identifiers. Order is canonical (serialization + the
/// claim table refer to these indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Col {
    Pc = 0,
    NextPc = 1,
    Pcq = 2,
    Iw = 3,
    Selw = 4,
    // 16 selector columns (bits of Selw).
    Sel0 = 5,
    // (Sel1..Sel15 follow contiguously.)
    // Instruction bits b0..b31 start at 21.
    Bits = 21,
    // After bits (21+32=53): values.
    Rs1a = 53,
    Rs2a = 54,
    Rda = 55,
    Rs1v = 56,
    Rs2v = 57,
    Rdv = 58,
    A0 = 59,
    A1 = 60,
    A2 = 61,
    B0 = 62,
    B1 = 63,
    B2 = 64,
    C0 = 65,
    C1 = 66,
    C2 = 67,
    Ea = 68,
    Mrv = 69,
    Mwv = 70,
    Wada = 71,
    Wvr = 72,
    Taken = 73,
    Eqb = 74,
    Invab = 75,
    We = 76,
    E = 77,
    E0 = 78,
    E1 = 79,
    E2c = 80,
    X0f = 81,
    Invr = 82,
    Q = 83,
    Q0 = 84,
    Q1 = 85,
    Q2 = 86,
    Rv = 87,
    R0 = 88,
    R1 = 89,
    R2 = 90,
    E2 = 91,
    G0 = 92,
    G1 = 93,
    G2 = 94,
    Bz = 95,
    Invb = 96,
    Kpow = 97,
    Shamt = 98,
    Dkey = 99,
    // Branch-kind one-hot (4).
    Fb0 = 100,
    Fb1 = 101,
    Fb2 = 102,
    Fb3 = 103,
    // RAM read/write port addresses.
    Rada = 104,
    WadaR = 105,
    // Register read ports (address mirrors for the Twist instances).
    Rs1m = 106,
    Rs2m = 107,
}

pub const NUM_COLS: usize = 108;
pub const NUM_SEL: usize = 16;

fn sel(i: usize) -> usize {
    Col::Sel0 as usize + i
}
fn bit(i: usize) -> usize {
    Col::Bits as usize + i
}

/// Family indices (selector bit positions in Selw).
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
}

/// Decode a 32-bit word to the family word (0 = outside the profile).
pub fn decode_family(word: u32) -> u64 {
    let opcode = word & 0x7f;
    let funct3 = (word >> 12) & 0x7;
    let funct6 = (word >> 26) & 0x3f;
    let one = 1u64;
    match opcode {
        0x33 => match (funct3, funct6) {
            (0x0, 0x00) => one << fam::ADD,
            (0x0, 0x20) => one << fam::SUB,
            (0x0, 0x01) => one << fam::MUL,
            // MULH / MULHU ride the MUL family shape.
            (0x1, 0x01) => one << fam::MUL,
            (0x3, 0x01) => one << fam::MUL,
            // Signed DIV/REM and the unsigned forms share the DIVQ/DIVR
            // column shapes.
            (0x4, 0x01) => one << fam::DIVQ,
            (0x5, 0x01) => one << fam::DIVQ,
            (0x6, 0x01) => one << fam::DIVR,
            (0x7, 0x01) => one << fam::DIVR,
            (0x4, 0x00) => one << fam::DIVQ,
            (0x7, 0x00) => one << fam::DIVR,
            (0x3, 0x00) => one << fam::SLTU,
            // Register shifts ride the SLLI/SRLI shapes (SRA's funct6
            // is 0x10 = f7 0x20 >> 1).
            (0x1, 0x00) => one << fam::SLLI,
            (0x5, 0x00) => one << fam::SRLI,
            (0x5, 0x10) => one << fam::SRLI,
            // Bitwise and comparisons route with the ADD class.
            (0x2, 0x00) => one << fam::ADD,
            (0x4, 0x00) => one << fam::ADD,
            (0x6, 0x00) => one << fam::ADD,
            (0x7, 0x00) => one << fam::ADD,
            _ => 0,
        },
        0x0b => match funct3 {
            0x1 => one << fam::SLLI,
            0x5 if funct6 == 0x00 => one << fam::SRLI,
            _ => 0,
        },
        0x3b => match (funct3, funct6) {
            (0x0, 0x00) => one << fam::ADD,
            (0x0, 0x20) => one << fam::SUB,
            (0x0, 0x01) => one << fam::MUL,
            (0x1, 0x00) => one << fam::SLLI,
            (0x5, 0x00) => one << fam::SRLI,
            (0x5, 0x10) => one << fam::SRLI,
            (0x4, 0x01) => one << fam::DIVQ,
            (0x5, 0x01) => one << fam::DIVQ,
            (0x6, 0x01) => one << fam::DIVR,
            (0x7, 0x01) => one << fam::DIVR,
            _ => 0,
        },
        0x1b => match funct3 {
            0x0 => one << fam::ADD,
            0x1 => one << fam::SLLI,
            0x5 => one << fam::SRLI,
            _ => 0,
        },
        0x13 => match funct3 {
            0x0 => one << fam::ADD,
            _ => 0,
        },
        0x63 => match funct3 {
            0x0 | 0x1 | 0x6 | 0x7 => one << fam::BRCH,
            _ => 0,
        },
        0x6f => one << fam::JAL,
        0x67 if funct3 == 0x00 => one << fam::JALR,
        0x03 if funct3 == 0x3 => one << fam::LOAD,
        0x23 if funct3 == 0x3 => one << fam::STORE,
        0x37 => one << fam::LUI,
        0x17 => one << fam::AUIPC,
        0x73 => one << fam::HALT,
        _ => 0,
    }
}

/// The decode table over keys (opcode|funct3<<7|funct6<<10), 16 bits.
pub fn decode_table() -> Vec<Goldilocks> {
    let mut table = vec![Goldilocks::ZERO; 1 << 16];
    for key in 0..(1u32 << 16) {
        let w = (key & 0x7f) | ((key >> 7 & 0x7) << 12) | ((key >> 10 & 0x3f) << 26);
        table[key as usize] = fe(decode_family(w));
    }
    table
}

pub fn imm_i(word: u32) -> u64 {
    let raw = ((word >> 20) & 0xfff) as u64;
    if raw & 0x800 != 0 {
        raw | 0xffff_f000
    } else {
        raw
    }
}

/// B-format immediate (branch offsets).
pub fn imm_b(word: u32) -> u64 {
    let w = word as u64;
    let imm13 = (((w >> 31) & 1) << 12)
        | (((w >> 7) & 1) << 11)
        | (((w >> 25) & 0x3f) << 5)
        | (((w >> 8) & 0xf) << 1);
    if imm13 & 0x1000 != 0 {
        imm13 | 0xffff_e000
    } else {
        imm13
    }
}

/// J-format immediate.
pub fn imm_j(word: u32) -> u64 {
    let w = word as u64;
    let imm21 = (((w >> 31) & 1) << 20)
        | (((w >> 12) & 0xff) << 12)
        | (((w >> 20) & 1) << 11)
        | (((w >> 21) & 0x3ff) << 1);
    if imm21 & 0x10_0000 != 0 {
        imm21 | 0xffe0_0000
    } else {
        imm21
    }
}

/// U-format immediate (LUI/AUIPC payload, << 12).
pub fn imm_u(word: u32) -> u64 {
    (((word >> 12) & 0xfffff) as u64) << 12
}

use lattice_core::Goldilocks;
use lattice_memory::PiopError;
use lattice_vm::MachineState;

fn wrap_sub_hi(b: u64, a: u64) -> (u64, u64) {
    // e = (b - a) mod 2^64, wrap = [b < a], exact for values < 2^33.
    let e = b.wrapping_sub(a);
    (e, (b < a) as u64)
}

fn instr_word_at(state: &MachineState, pc: u64) -> u32 {
    let w = state.memory.load(pc & !0x7);
    ((w >> ((pc & 0x7) * 8)) & 0xffff_ffff) as u32
}

/// The trace-derived columns + the memory access streams.
pub struct TraceData {
    pub log_t: usize,
    pub cols: Vec<Vec<Goldilocks>>,
    /// RAM port streams: (read addr, read value, write addr, write value).
    pub ram_read_addr: Vec<u64>,
    pub ram_write_addr: Vec<u64>,
    /// Register read/write port streams.
    pub reg_read_addr_a: Vec<u64>,
    pub reg_read_addr_b: Vec<u64>,
    pub reg_write_addr: Vec<u64>,
    pub reg_write_val: Vec<Goldilocks>,
    /// The 11-bit and 10-bit range-check lookup streams.
    pub range11: Vec<u64>,
    pub range10: Vec<u64>,
}

#[allow(clippy::too_many_lines)]
pub fn build_trace(state: &MachineState, rows: &[lattice_vm::TraceRow]) -> Result<TraceData, PipelineError> {
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
    let mut reg_wv = Vec::with_capacity(t_pow);
    let mut range11: Vec<u64> = Vec::new();
    let mut range10: Vec<u64> = Vec::new();
    let chunks = |v: u64| -> [u64; 3] { [v & 0x7ff, (v >> 11) & 0x7ff, (v >> 22) & 0x7ff] };

    for row in rows.iter() {
        let pc = row.pc;
        let iw = instr_word_at(state, pc);
        let selw = decode_family(iw);
        if selw == 0 {
            return Err(PipelineError::UnsupportedInstruction { pc, word: iw });
        }
        let family = selw.trailing_zeros() as usize;
        let rs1a = ((iw >> 15) & 0x1f) as u64;
        let rs2a = ((iw >> 20) & 0x1f) as u64;
        let rda = ((iw >> 7) & 0x1f) as u64;
        let rs1v = if rs1a == 0 { 0 } else { regs[rs1a as usize] };
        let rs2v = if rs2a == 0 { 0 } else { regs[rs2a as usize] };
        let mut rdv = 0u64;
        for &(r, v) in &row.reg_writes {
            if r as u64 == rda {
                rdv = v;
            }
        }
        // Register value range checks (all constrained families).
        let ac = chunks(rs1v);
        let bc = chunks(rs2v);
        let cc = chunks(rdv);
        if family != fam::JAL && family != fam::JALR && family != fam::HALT {
            range11.extend_from_slice(&ac);
            range11.extend_from_slice(&bc);
            range11.extend_from_slice(&cc);
        }
        // MUL operands < 2^32: top chunk < 2^10.
        if family == fam::MUL {
            range10.push(rs1v >> 22);
            range10.push(rs2v >> 22);
        }
        // Division witnesses.
        let (q, r) = if family == fam::DIVQ || family == fam::DIVR {
            match rs2v {
                0 => (u64::MAX, rs1v),
                b => (rs1v / b, rs1v % b),
            }
        } else {
            (0, 0)
        };
        let e2v = if family == fam::DIVQ || family == fam::DIVR || family == fam::SRLI {
            if rs2v == 0 && family != fam::SRLI {
                0
            } else {
                // The shift bound: register forms take the shamt from
                // rs2 (& 63); immediate forms from the encoded field.
                let opcode = iw & 0x7f;
                let shamt = if opcode == 0x33 || opcode == 0x3b {
                    (rs2v & 0x3f) as u32
                } else {
                    ((iw >> 20) & 0x3f) as u32
                };
                let bnd = if family == fam::SRLI { 1u64 << shamt } else { rs2v };
                bnd.wrapping_sub(r).wrapping_sub(1)
            }
        } else {
            0
        };
        if family == fam::DIVQ || family == fam::DIVR {
            range11.extend_from_slice(&chunks(q));
            range11.extend_from_slice(&chunks(r));
            range11.extend_from_slice(&chunks(e2v));
        }
        if family == fam::SRLI {
            range11.extend_from_slice(&chunks(r));
            range11.extend_from_slice(&chunks(e2v));
        }
        let (ev, we) = wrap_sub_hi(rs2v, rs1v);
        if family == fam::SLTU || family == fam::BRCH {
            range11.extend_from_slice(&chunks(ev));
        }
        let shamt = ((iw >> 20) & 0x3f) as u64;
        let dkey = (iw & 0x7f) as u64
            | (((iw >> 12) & 0x7) as u64) << 7
            | (((iw >> 26) & 0x3f) as u64) << 10;
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
        // Memory access.
        let (ea, old_w, new_w) = match &row.mem_access {
            Some((addr, old, new)) => (addr / 8, *old, *new),
            None => (0, 0, None),
        };
        let is_load = family == fam::LOAD;
        let is_store = family == fam::STORE;
        // Column pushes.
        let mut p = |idx: usize, v: u64| cols[idx].push(fe(v));
        p(Col::Pc as usize, pc);
        p(Col::NextPc as usize, row.next_pc);
        p(Col::Pcq as usize, pc / 4);
        p(Col::Iw as usize, iw as u64);
        p(Col::Selw as usize, selw);
        for k in 0..NUM_SEL {
            p(sel(k), (selw >> k) & 1);
        }
        for i in 0..32 {
            p(bit(i), ((iw >> i) & 1) as u64);
        }
        p(Col::Rs1a as usize, rs1a);
        p(Col::Rs2a as usize, rs2a);
        p(Col::Rda as usize, rda);
        p(Col::Rs1v as usize, rs1v);
        p(Col::Rs2v as usize, rs2v);
        p(Col::Rdv as usize, rdv);
        for (i, v) in ac.iter().enumerate() {
            p(Col::A0 as usize + i, *v);
        }
        for (i, v) in bc.iter().enumerate() {
            p(Col::B0 as usize + i, *v);
        }
        for (i, v) in cc.iter().enumerate() {
            p(Col::C0 as usize + i, *v);
        }
        p(Col::Ea as usize, ea);
        p(Col::Mrv as usize, old_w);
        p(Col::Mwv as usize, new_w.unwrap_or(0));
        // Register write port.
        let wr = writes_rd(family) as u64;
        let x0f = (rda == 0) as u64;
        let wada = if wr != 0 && x0f == 0 { rda } else { 0 };
        let wvr = if wr != 0 && x0f == 0 { rdv } else { 0 };
        p(Col::Wada as usize, wada);
        p(Col::Wvr as usize, wvr);
        p(Col::Taken as usize, taken);
        p(Col::Eqb as usize, (rs1v == rs2v) as u64);
        p(Col::Invab as usize, 0);
        p(Col::We as usize, we);
        p(Col::E as usize, ev);
        let ec = chunks(ev);
        p(Col::E0 as usize, ec[0]);
        p(Col::E1 as usize, ec[1]);
        p(Col::E2c as usize, ec[2]);
        p(Col::X0f as usize, x0f);
        p(Col::Invr as usize, 0);
        p(Col::Q as usize, q);
        let qc = chunks(q);
        p(Col::Q0 as usize, qc[0]);
        p(Col::Q1 as usize, qc[1]);
        p(Col::Q2 as usize, qc[2]);
        p(Col::Rv as usize, r);
        let rc = chunks(r);
        p(Col::R0 as usize, rc[0]);
        p(Col::R1 as usize, rc[1]);
        p(Col::R2 as usize, rc[2]);
        let gc = chunks(e2v);
        p(Col::E2 as usize, e2v);
        p(Col::G0 as usize, gc[0]);
        p(Col::G1 as usize, gc[1]);
        p(Col::G2 as usize, gc[2]);
        p(Col::Bz as usize, ((family == fam::DIVQ || family == fam::DIVR) && rs2v == 0) as u64);
        p(Col::Invb as usize, 0);
        p(Col::Kpow as usize, 1u64 << shamt.min(63));
        p(Col::Shamt as usize, shamt);
        p(Col::Dkey as usize, dkey);
        let fb = match family {
            fam::BRCH => (iw >> 12) & 0x7,
            _ => 0xf,
        };
        p(Col::Fb0 as usize, (fb == 0) as u64);
        p(Col::Fb1 as usize, (fb == 1) as u64);
        p(Col::Fb2 as usize, (fb == 6) as u64);
        p(Col::Fb3 as usize, (fb == 7) as u64);
        // RAM ports.
        let rada = if is_load { ea } else { 0 };
        let wada_r = if is_store { ea } else { 0 };
        p(Col::Rada as usize, rada);
        p(Col::WadaR as usize, wada_r);
        p(Col::Rs1m as usize, rs1a);
        p(Col::Rs2m as usize, rs2a);
        let mrv_v = *ram_now.get(&(rada * 8)).unwrap_or(&0);
        let mwv_v = if is_store {
            new_w.unwrap_or(0)
        } else {
            *ram_now.get(&0).unwrap_or(&0)
        };
        if let Some(x) = cols[Col::Mrv as usize].last_mut() {
            *x = fe(mrv_v);
        }
        if let Some(x) = cols[Col::Mwv as usize].last_mut() {
            *x = fe(mwv_v);
        }
        ram_ra.push(rada);
        ram_wa.push(wada_r);
        // Register ports.
        reg_ra_a.push(rs1a);
        reg_ra_b.push(rs2a);
        reg_wa.push(wada);
        reg_wv.push(fe(wvr));
        // Shadow updates.
        for &(r, v) in &row.reg_writes {
            if r != 0 {
                regs[r as usize] = v;
            }
        }
        if let Some((addr, _, Some(new))) = &row.mem_access {
            ram_now.insert(*addr, *new);
        }
    }
    // Pad with halt steps.
    let last_pc = rows.last().map(|r| r.pc).unwrap_or(0x1000);
    let last_iw = instr_word_at(state, last_pc);
    while cols[0].len() < t_pow {
        let mut p = |idx: usize, v: u64| cols[idx].push(fe(v));
        p(Col::Pc as usize, last_pc);
        p(Col::NextPc as usize, last_pc);
        p(Col::Pcq as usize, last_pc / 4);
        p(Col::Iw as usize, last_iw as u64);
        p(Col::Selw as usize, 1u64 << fam::HALT);
        for k in 0..NUM_SEL {
            p(sel(k), (k == fam::HALT) as u64);
        }
        for i in 0..32 {
            p(bit(i), ((last_iw >> i) & 1) as u64);
        }
        for idx in [Col::Rs1a, Col::Rs2a, Col::Rda, Col::Rs1v, Col::Rs2v, Col::Rdv,
            Col::A0, Col::A1, Col::A2, Col::B0, Col::B1, Col::B2, Col::C0, Col::C1, Col::C2,
            Col::Ea, Col::Mrv, Col::Mwv, Col::Wada, Col::Wvr, Col::Taken, Col::Eqb,
            Col::Invab, Col::We, Col::E, Col::E0, Col::E1, Col::E2c, Col::X0f, Col::Invr,
            Col::Q, Col::Q0, Col::Q1, Col::Q2, Col::Rv, Col::R0, Col::R1, Col::R2,
            Col::E2, Col::G0, Col::G1, Col::G2, Col::Bz, Col::Invb,
            Col::Kpow, Col::Shamt, Col::Dkey, Col::Fb0, Col::Fb1, Col::Fb2, Col::Fb3,
            Col::Rada, Col::WadaR, Col::Rs1m, Col::Rs2m] {
            p(idx as usize, 0);
        }
        ram_ra.push(0);
        ram_wa.push(0);
        reg_ra_a.push(0);
        reg_ra_b.push(0);
        reg_wa.push(0);
        reg_wv.push(Goldilocks::ZERO);
    }
    // Inverse witnesses.
    for t in 0..t_pow {
        let rs1v = cols[Col::Rs1v as usize][t];
        let rs2v = cols[Col::Rs2v as usize][t];
        let d = rs1v.sub(&rs2v);
        cols[Col::Invab as usize][t] =
            if d.is_zero() { Goldilocks::ZERO } else { d.inverse().unwrap_or(Goldilocks::ZERO) };
        let rda = cols[Col::Rda as usize][t];
        cols[Col::Invr as usize][t] =
            if rda.is_zero() { Goldilocks::ZERO } else { rda.inverse().unwrap_or(Goldilocks::ZERO) };
        cols[Col::Invb as usize][t] =
            if rs2v.is_zero() { Goldilocks::ZERO } else { rs2v.inverse().unwrap_or(Goldilocks::ZERO) };
    }
    Ok(TraceData {
        log_t,
        cols,
        ram_read_addr: ram_ra,
        ram_write_addr: ram_wa,
        reg_read_addr_a: reg_ra_a,
        reg_read_addr_b: reg_ra_b,
        reg_write_addr: reg_wa,
        reg_write_val: reg_wv,
        range11,
        range10,
    })
}

fn writes_rd(family: usize) -> bool {
    matches!(
        family,
        fam::ADD | fam::SUB | fam::MUL | fam::DIVQ | fam::DIVR | fam::SLLI | fam::SRLI
            | fam::SLTU | fam::JAL | fam::JALR | fam::LOAD | fam::LUI | fam::AUIPC
    )
}
