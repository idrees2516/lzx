//! The zkVM witness: fixed-width trace columns built from the executor's
//! rows (Wave 7.4 P0-4 — the column restructure).
//!
//! Representation (see `docs/WAVE_ANALYSIS.md`): every 64-bit machine
//! value is carried as its **64-bit tensor** — an MLE over
//! `(6 + log T)` variables whose Boolean-cube values are the bits of the
//! value at that cycle (bit block first, MSB-first; evaluation index
//! `bit * T + t`). The 16-bit limbs and field combos are *derived*
//! factors — affine functions of tensor evaluation claims — so the
//! committed universe stays `O(T)` bit-packed values. The Twist & Shout
//! one-hot matrices (`ra`/`wa`/`Inc`) are *virtual*: pure functions of
//! the committed digit-bit columns, authenticated through dedicated
//! matrix-evaluation sumchecks (`memory.rs`).
//!
//! Memory model: the RAM argument is word-granular (8-byte words keyed
//! by `addr >> 3`) and runs **per limb** — four Twist instances whose
//! read/write values are the word's 16-bit limbs (each `< 2^16 < p`, so
//! the field representation is canonical and the `v` vs `v − p` aliasing
//! of 64-bit combo encodings cannot occur). `mem_old`/`mem_new` tensors
//! carry the full 8-byte words before/after each access, reconstructed
//! with a shadow replay (including the half-word merges `LW`/`SW`
//! induce). The zkVM v1 proves the aligned subset — 4-aligned
//! `LW`/`LWU`/`SW`, 8-aligned `LD`/`SD`, 4-byte instructions only —
//! and fails closed otherwise.

use lattice_core::{DenseMle, Goldilocks};
use lattice_vm::{Instr, TraceRow};

/// Number of machine values carried as 64-bit tensors.
pub const VALUE_TENSORS: usize = 6;
/// Tensor slots (stable factor addressing).
pub const T_RS1: usize = 0;
pub const T_RS2: usize = 1;
pub const T_IMM: usize = 2;
pub const T_RD: usize = 3;
pub const T_MEM_OLD: usize = 4;
pub const T_MEM_NEW: usize = 5;

/// The word-granular memory state as a sorted map.
pub type WordMap = std::collections::BTreeMap<u64, u64>;

/// A fixed-width cycle witness.
#[derive(Clone, Debug)]
pub struct CycleWitness {
    pub log_t: usize,
    /// Executed steps before padding.
    pub steps: usize,
    /// pc / next_pc as plain field columns (`< 2^48`, no wrap).
    pub pc: Vec<Goldilocks>,
    pub next_pc: Vec<Goldilocks>,
    /// The fetched 32-bit instruction word column (`< 2^32`).
    pub instr: Vec<Goldilocks>,
    /// The instruction 32-bit tensor: MLE over `(5 + log_t)`.
    pub instr_bits: DenseMle,
    /// The six value tensors: MLEs over `(6 + log_t)`.
    pub values: [DenseMle; VALUE_TENSORS],
    /// Per-cycle register indices (5-bit values, `x0` = 0).
    pub rs1_idx: Vec<u8>,
    pub rs2_idx: Vec<u8>,
    pub rd_idx: Vec<u8>,
    /// Register write enable (instruction writes rd and rd != x0).
    pub rd_we: Vec<u8>,
    /// Memory access flags; `mem_half` selects the 32-bit half within the
    /// word for `LW`/`LWU`/`SW` accesses.
    pub mem_re: Vec<u8>,
    pub mem_we: Vec<u8>,
    pub mem_half: Vec<u8>,
    /// Word-level memory address column (`addr >> 3`).
    pub mem_word: Vec<Goldilocks>,
    /// Effective byte address column (`rs1 + imm` as integers, no wrap
    /// for in-window addresses).
    pub mem_addr: Vec<Goldilocks>,
    /// Fetch word index column (`pc >> 2`).
    pub fetch_word: Vec<Goldilocks>,
    /// halted flag per cycle (padding cycles are halted).
    pub halted: Vec<u8>,
}

/// Errors building the witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessError {
    /// Compressed instruction encountered (v1 proves the 4-byte subset).
    CompressedInstruction { cycle: usize },
    /// Alignment contract violated.
    UnalignedAccess { cycle: usize, addr: u64 },
    /// Memory address outside the declared RAM window.
    AddressOutOfRange { cycle: usize, addr: u64 },
    /// pc outside the declared program window.
    PcOutOfRange { cycle: usize, pc: u64 },
    /// No halt within max_steps (or empty trace).
    NoHalt,
    /// Program too short for the fetch of a reachable pc.
    ProgramTruncated { pc: u64 },
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// Bits of `v`, MSB-first, length `nbits`.
fn bits_of(v: u64, nbits: usize) -> Vec<u8> {
    (0..nbits)
        .map(|i| ((v >> (nbits - 1 - i)) & 1) as u8)
        .collect()
}

/// Build a bit tensor: MLE over `(log2(nbits) + log_t)` variables; the
/// evaluation table is the row-major `bit-row x cycle` grid (row block =
/// most significant variables, MSB first); evaluation index
/// `bit * T + t`. The bit-row count is `nbits` (2^log2(nbits)), so the
/// MLE is structurally consistent: 2^(log2(nbits) + log_t) evaluations.
fn bit_tensor(bits_per_cycle: &[Vec<u8>], log_t: usize, nbits: usize) -> DenseMle {
    let t = 1usize << log_t;
    let mut evals = Vec::with_capacity(nbits * t);
    for b in 0..nbits {
        for tt in 0..t {
            evals.push(fe(bits_per_cycle[tt][b] as u64));
        }
    }
    DenseMle {
        num_vars: nbits.trailing_zeros() as usize + log_t,
        evaluations: evals,
    }
}

/// The RAM window: `2^log_k` 8-byte words (word keys `addr >> 3`).
#[derive(Clone, Copy, Debug)]
pub struct RamWindow {
    pub log_k: usize,
}

/// The program window: `2^log_k` 32-bit instruction words.
#[derive(Clone, Copy, Debug)]
pub struct FetchWindow {
    pub log_k: usize,
}

/// The immediate value of an instruction, as a u64 (sign-extended).
fn imm_of(instr: &Instr) -> u64 {
    use Instr::*;
    match instr {
        Addi { imm, .. }
        | Slti { imm, .. }
        | Xori { imm, .. }
        | Ori { imm, .. }
        | Andi { imm, .. }
        | Addiw { imm, .. }
        | Lw { imm, .. }
        | Lwu { imm, .. }
        | Ld { imm, .. }
        | Sw { imm, .. }
        | Sd { imm, .. }
        | Beq { imm, .. }
        | Bne { imm, .. }
        | Blt { imm, .. }
        | Bge { imm, .. }
        | Bltu { imm, .. }
        | Bgeu { imm, .. }
        | Jal { imm, .. }
        | Jalr { imm, .. } => *imm as u64,
        Sltiu { imm, .. } => *imm,
        Slli { shamt, .. } | Srli { shamt, .. } | Srai { shamt, .. } => *shamt as u64,
        Slliw { shamt, .. } | Srliw { shamt, .. } | Sraiw { shamt, .. } => *shamt as u64,
        Lui { imm, .. } | Auipc { imm, .. } => *imm as u64,
        // Shifts by register carry the shift amount through rs2 (imm = 0).
        _ => 0,
    }
}

/// Whether the instruction writes rd (loads/jumps included).
fn instruction_writes(instr: &Instr) -> bool {
    use Instr::*;
    !matches!(
        instr,
        Sw { .. }
            | Sd { .. }
            | Beq { .. }
            | Bne { .. }
            | Blt { .. }
            | Bge { .. }
            | Bltu { .. }
            | Bgeu { .. }
            | Ecall
            | Ebreak
    )
}

/// Build the fixed-width witness from executed rows plus the program
/// image (for fetch words). Returns the witness and the word-level final
/// memory state (every word the program touched).
pub fn build_cycle_witness(
    rows: &[TraceRow],
    program: &[u8],
    public_input: &[u8],
    ram: RamWindow,
    fetch: FetchWindow,
) -> Result<(CycleWitness, WordMap), WitnessError> {
    let steps = rows.len();
    if steps == 0 || !rows.last().map(|r| r.halted).unwrap_or(false) {
        return Err(WitnessError::NoHalt);
    }
    let log_t = usize::max(1, steps.next_power_of_two().trailing_zeros() as usize);
    let t = 1usize << log_t;

    let mut pc = vec![Goldilocks::ZERO; t];
    let mut next_pc = vec![Goldilocks::ZERO; t];
    let mut instr = vec![Goldilocks::ZERO; t];
    let mut instr_bits_per = vec![vec![0u8; 32]; t];
    let mut val_bits: Vec<[Vec<u8>; VALUE_TENSORS]> = (0..t)
        .map(|_| std::array::from_fn(|_| vec![0u8; 64]))
        .collect();
    let mut rs1_idx = vec![0u8; t];
    let mut rs2_idx = vec![0u8; t];
    let mut rd_idx = vec![0u8; t];
    let mut rd_we = vec![0u8; t];
    let mut mem_re = vec![0u8; t];
    let mut mem_we = vec![0u8; t];
    let mut mem_half = vec![0u8; t];
    let mut mem_word = vec![Goldilocks::ZERO; t];
    let mut mem_addr = vec![Goldilocks::ZERO; t];
    let mut fetch_word = vec![Goldilocks::ZERO; t];
    let mut halted = vec![0u8; t];

    // Shadow word memory for full-word old/new reconstruction,
    // initialized from the program + public-input image (reads of
    // never-written words see the initial state, not zero).
    let init_image = |word_key: u64| -> u64 {
        let base = word_key * 8;
        let mut word = 0u64;
        for i in 0..8usize {
            let addr = base + i as u64;
            let byte = if (addr as usize) < program.len() {
                program[addr as usize]
            } else if addr >= 0x1000 && (addr as usize - 0x1000) < public_input.len() {
                public_input[addr as usize - 0x1000]
            } else {
                0
            };
            word |= (byte as u64) << (8 * i);
        }
        word
    };
    let mut shadow: WordMap = WordMap::new();

    for (cycle, row) in rows.iter().enumerate() {
        if row.width != 4 {
            return Err(WitnessError::CompressedInstruction { cycle });
        }
        let fetch_index = row.pc >> 2;
        if fetch_index >= (1u64 << fetch.log_k) || (row.pc & 3) != 0 {
            return Err(WitnessError::PcOutOfRange { cycle, pc: row.pc });
        }
        let byte_off = row.pc as usize;
        if byte_off + 4 > program.len() {
            return Err(WitnessError::ProgramTruncated { pc: row.pc });
        }
        let word = u32::from_le_bytes([
            program[byte_off],
            program[byte_off + 1],
            program[byte_off + 2],
            program[byte_off + 3],
        ]);

        pc[cycle] = fe(row.pc);
        next_pc[cycle] = fe(row.next_pc);
        fetch_word[cycle] = fe(fetch_index);
        instr[cycle] = fe(word as u64);
        instr_bits_per[cycle] = bits_of(word as u64, 32);

        // Operands: reg_reads are pushed in operand order (rs1, rs2).
        let reads = &row.reg_reads;
        let rs1v = reads.first().map(|(_, v)| *v).unwrap_or(0);
        let rs2v = reads.get(1).map(|(_, v)| *v).unwrap_or(0);
        let rdv = row.reg_writes.first().map(|(_, v)| *v).unwrap_or(0);
        let (i1, i2, id) = reg_indices_of(&row.instr);
        rs1_idx[cycle] = i1;
        rs2_idx[cycle] = i2;
        rd_idx[cycle] = id;
        rd_we[cycle] = (instruction_writes(&row.instr) && id != 0) as u8;
        val_bits[cycle][T_RS1] = bits_of(rs1v, 64);
        val_bits[cycle][T_RS2] = bits_of(rs2v, 64);
        val_bits[cycle][T_IMM] = bits_of(imm_of(&row.instr), 64);
        val_bits[cycle][T_RD] = bits_of(rdv, 64);

        // Memory access: word-granular shadow replay.
        if let Some((addr, _old, new)) = &row.mem_access {
            let is_64 = matches!(row.instr, Instr::Ld { .. } | Instr::Sd { .. });
            if (addr & 3) != 0 || (is_64 && (addr & 7) != 0) {
                return Err(WitnessError::UnalignedAccess { cycle, addr: *addr });
            }
            let word_key = addr >> 3;
            if word_key >= (1u64 << ram.log_k) {
                return Err(WitnessError::AddressOutOfRange { cycle, addr: *addr });
            }
            let is_write = new.is_some();
            mem_re[cycle] = (!is_write) as u8;
            mem_we[cycle] = is_write as u8;
            mem_half[cycle] = ((addr >> 2) & 1) as u8;
            mem_word[cycle] = fe(word_key);
            mem_addr[cycle] = fe(*addr);
            let before = match shadow.get(&word_key) {
                Some(v) => *v,
                None => init_image(word_key),
            };
            let access_new = new.unwrap_or(before);
            let after = merge_word(&row.instr, before, *addr, access_new);
            val_bits[cycle][T_MEM_OLD] = bits_of(before, 64);
            val_bits[cycle][T_MEM_NEW] = bits_of(after, 64);
            if new.is_some() {
                shadow.insert(word_key, after);
            }
        }

        halted[cycle] = row.halted as u8;
    }
    // Padding cycles: replicate the halt row's pc with an ECALL decode —
    // every constraint family (next_pc, fetch, halted propagation) then
    // stays well-formed across the padding.
    let halt_pc = rows[steps - 1].pc;
    for cycle in steps..t {
        pc[cycle] = fe(halt_pc);
        next_pc[cycle] = fe(halt_pc.wrapping_add(4));
        fetch_word[cycle] = fe(halt_pc >> 2);
        halted[cycle] = 1;
        instr[cycle] = fe(0x0000_0073);
        instr_bits_per[cycle] = bits_of(0x73, 32);
    }

    let instr_bits = bit_tensor(&instr_bits_per, log_t, 32);
    let mut values: [Option<DenseMle>; VALUE_TENSORS] = Default::default();
    for slot in 0..VALUE_TENSORS {
        let per: Vec<Vec<u8>> = (0..t).map(|tt| val_bits[tt][slot].clone()).collect();
        values[slot] = Some(bit_tensor(&per, log_t, 64));
    }
    let mut fixed: [DenseMle; VALUE_TENSORS] = std::array::from_fn(|_| DenseMle::zero(0));
    for slot in 0..VALUE_TENSORS {
        fixed[slot] = values[slot].take().unwrap_or_else(|| DenseMle::zero(0));
    }

    let witness = CycleWitness {
        log_t,
        steps,
        pc,
        next_pc,
        instr,
        instr_bits,
        values: fixed,
        rs1_idx,
        rs2_idx,
        rd_idx,
        rd_we,
        mem_re,
        mem_we,
        mem_half,
        mem_word,
        mem_addr,
        fetch_word,
        halted,
    };
    Ok((witness, shadow))
}

/// The full 8-byte word after an access, given the word before.
fn merge_word(instr: &Instr, before: u64, addr: u64, access_new: u64) -> u64 {
    match instr {
        // Reads leave the word unchanged.
        Instr::Lw { .. } | Instr::Lwu { .. } | Instr::Ld { .. } => before,
        Instr::Sd { .. } => access_new,
        Instr::Sw { .. } => {
            let half = (addr >> 2) & 1;
            if half == 0 {
                (before & 0xFFFF_FFFF_0000_0000) | (access_new & 0xFFFF_FFFF)
            } else {
                (before & 0x0000_0000_FFFF_FFFF) | ((access_new & 0xFFFF_FFFF) << 32)
            }
        }
        _ => before,
    }
}

/// (rs1, rs2, rd) indices of an instruction.
fn reg_indices_of(instr: &Instr) -> (u8, u8, u8) {
    use Instr::*;
    match instr {
        Addi { rd, rs1, .. }
        | Slti { rd, rs1, .. }
        | Sltiu { rd, rs1, .. }
        | Xori { rd, rs1, .. }
        | Ori { rd, rs1, .. }
        | Andi { rd, rs1, .. }
        | Slli { rd, rs1, .. }
        | Srli { rd, rs1, .. }
        | Srai { rd, rs1, .. }
        | Addiw { rd, rs1, .. }
        | Slliw { rd, rs1, .. }
        | Srliw { rd, rs1, .. }
        | Sraiw { rd, rs1, .. }
        | Lw { rd, rs1, .. }
        | Lwu { rd, rs1, .. }
        | Ld { rd, rs1, .. }
        | Jalr { rd, rs1, .. } => (*rs1, 0, *rd),
        Add { rd, rs1, rs2 }
        | Sub { rd, rs1, rs2 }
        | Sll { rd, rs1, rs2 }
        | Slt { rd, rs1, rs2 }
        | Sltu { rd, rs1, rs2 }
        | Xor { rd, rs1, rs2 }
        | Srl { rd, rs1, rs2 }
        | Sra { rd, rs1, rs2 }
        | Or { rd, rs1, rs2 }
        | And { rd, rs1, rs2 }
        | Addw { rd, rs1, rs2 }
        | Subw { rd, rs1, rs2 }
        | Sllw { rd, rs1, rs2 }
        | Srlw { rd, rs1, rs2 }
        | Sraw { rd, rs1, rs2 }
        | Mul { rd, rs1, rs2 }
        | Mulh { rd, rs1, rs2 }
        | Mulhu { rd, rs1, rs2 }
        | Div { rd, rs1, rs2 }
        | Divu { rd, rs1, rs2 }
        | Rem { rd, rs1, rs2 }
        | Remu { rd, rs1, rs2 }
        | Divw { rd, rs1, rs2 }
        | Divuw { rd, rs1, rs2 }
        | Remw { rd, rs1, rs2 }
        | Remuw { rd, rs1, rs2 }
        | Mulw { rd, rs1, rs2 } => (*rs1, *rs2, *rd),
        Sw { rs1, rs2, .. } | Sd { rs1, rs2, .. } => (*rs1, *rs2, 0),
        Beq { rs1, rs2, .. }
        | Bne { rs1, rs2, .. }
        | Blt { rs1, rs2, .. }
        | Bge { rs1, rs2, .. }
        | Bltu { rs1, rs2, .. }
        | Bgeu { rs1, rs2, .. } => (*rs1, *rs2, 0),
        Lui { rd, .. } | Auipc { rd, .. } | Jal { rd, .. } => (0, 0, *rd),
        LrW { rd, rs1, .. } => (*rs1, 0, *rd),
        ScW { rd, rs1, rs2, .. } => (*rs1, *rs2, *rd),
        AmoSwapW { rd, rs1, rs2, .. }
        | AmoAddW { rd, rs1, rs2, .. }
        | AmoXorW { rd, rs1, rs2, .. }
        | AmoAndW { rd, rs1, rs2, .. }
        | AmoOrW { rd, rs1, rs2, .. }
        | AmoMinW { rd, rs1, rs2, .. }
        | AmoMaxW { rd, rs1, rs2, .. }
        | AmoMinuW { rd, rs1, rs2, .. }
        | AmoMaxuW { rd, rs1, rs2, .. } => (*rs1, *rs2, *rd),
        Ecall | Ebreak => (0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_vm::{run as vm_run, MachineState};

    fn enc_addi(rd: u8, rs1: u8, imm: i64) -> u32 {
        ((imm as u32 & 0xFFF) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
    }

    /// addi x1, x0, 12; sw x1, 0(x2=8); lw x3, 0(x2); ecall
    fn program() -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&enc_addi(1, 0, 12).to_le_bytes());
        p.extend_from_slice(&enc_addi(2, 0, 8).to_le_bytes());
        let sw: u32 = ((1u32) << 20) | (2 << 15) | (2 << 12) | 0x23;
        p.extend_from_slice(&sw.to_le_bytes());
        let lw: u32 = (2 << 15) | (2 << 12) | (3 << 7) | 0x03;
        p.extend_from_slice(&lw.to_le_bytes());
        p.extend_from_slice(&0x73u32.to_le_bytes());
        p
    }

    fn run_trace(prog: &[u8]) -> Vec<TraceRow> {
        let mut state = MachineState::new();
        state.load_program(0, prog);
        vm_run(&mut state, 64).ok().unwrap()
    }

    #[test]
    fn builds_witness_and_tracks_words() {
        let prog = program();
        let rows = run_trace(&prog);
        let (w, final_words) = build_cycle_witness(
            &rows,
            &prog,
            &[],
            RamWindow { log_k: 4 },
            FetchWindow { log_k: 3 },
        )
        .ok()
        .unwrap();
        assert_eq!(w.steps, 5);
        assert_eq!(w.log_t, 3);
        // rs1 values follow reg_reads: x0=0 then x1=12 for the sw.
        assert_eq!(w.values[T_RS1].evaluations[8 + 2].0, 0);
        // The stored word: the sw writes 12 into the LOW half of word 1,
        // whose high half holds program bytes (the initial image).
        let lw_word = u32::from_le_bytes([prog[12], prog[13], prog[14], prog[15]]) as u64;
        let expect_word = (lw_word << 32) | 12;
        assert_eq!(final_words.get(&1).copied(), Some(expect_word));
        // mem_word column at the sw cycle (index 2) is key 1.
        assert_eq!(w.mem_word[2].0, 1);
        // mem_old/mem_new tensors: word 1 goes 0 -> 12 (bits 3,2 set; the
        // tensor's bit block is MSB-first, so value-bit b lives at row
        // 63-b).
        assert_eq!(w.values[T_MEM_NEW].evaluations[63 * 8 + 2].0, 0);
        assert_eq!(w.values[T_MEM_NEW].evaluations[62 * 8 + 2].0, 0);
        assert_eq!(w.values[T_MEM_NEW].evaluations[61 * 8 + 2].0, 1);
        assert_eq!(w.values[T_MEM_NEW].evaluations[60 * 8 + 2].0, 1);
        // mem_old (the word before the sw) is all zeros.
        assert_eq!(w.values[T_MEM_OLD].evaluations[61 * 8 + 2].0, 0);
        // instr column carries the raw words.
        assert_eq!(w.instr[0].0, enc_addi(1, 0, 12) as u64);
        // Padding cycles are halted with ECALL.
        assert_eq!(w.halted[5], 1);
        assert_eq!(w.instr[5].0, 0x73);
    }

    #[test]
    fn rejects_unaligned_access() {
        // addi x2, x0, 5; sw x0, 1(x2) — 4-aligned contract violated.
        let mut prog = Vec::new();
        prog.extend_from_slice(&enc_addi(2, 0, 5).to_le_bytes());
        let sw: u32 = (1u32 << 20) | (2 << 15) | (2 << 12) | 0x23;
        prog.extend_from_slice(&sw.to_le_bytes());
        prog.extend_from_slice(&0x73u32.to_le_bytes());
        let rows = run_trace(&prog);
        assert!(build_cycle_witness(
            &rows,
            &prog,
            &[],
            RamWindow { log_k: 4 },
            FetchWindow { log_k: 3 }
        )
        .is_err());
    }
}
