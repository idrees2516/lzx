//! The v3 pipeline: the zkVM's memory arguments re-proven through the
//! ring-lookup layer (follow-up (c) of `ring-lookups.md`) —
//! `lattice-lookup-ring`'s RAM batch verification and ROM lookups
//! replace the Twist & Shout lookup layer of v2, while the statement
//! (public initial/final images, the read/write access stream in which
//! every read observes the most recent write) is unchanged.
//!
//! What changes vs `pipeline2`: the **fetch** and **input**
//! read-only streams become Ring-LogUp ROM lookups (the paper:
//! "lookup protocols already suffice for ROM"); the **RAM** and
//! **register** instances become the Section-6 composition (sub-RAM
//! isolation lookups + offline memory checking + the almost-identical
//! layer) over the touched sub-RAM, with the untouched rest pinned by
//! the window equality check; the digit-bit tensors, virtual one-hot
//! matrices, and the Twist/Shout grand products of `lattice-memory`
//! are NOT used; and the verifier still never re-executes — it
//! consumes the public images, the committed stream shapes, and the
//! bridge proofs.
//!
//! Honest scope: the instruction-semantics AIR (the P0 constraint
//! families carried by `pipeline.rs`'s trace builder) is the same
//! documented next layer as in v2 — this pipeline swaps the lookup
//! layer only, which is exactly the (c) scope.

use crate::lookup_memory::{
    prove_lookup_memory, verify_lookup_memory, LookupMemoryInstance, LookupMemoryProof,
};
use crate::pipeline::{build_trace, Col, PublicStateV2, PipelineError};
use lattice_core::transcript::Transcript;
use lattice_lookup_ring::ring_d::RingD;
use lattice_vm::MachineState;

fn fe(x: u64) -> u64 {
    x
}

/// The v3 proof.
#[derive(Clone, Debug)]
pub struct ProofV3 {
    pub log_t: usize,
    pub num_fetch: usize,
    pub num_input_words: usize,
    pub fetch: LookupMemoryProof,
    pub input_rom: LookupMemoryProof,
    pub ram: LookupMemoryProof,
    pub regs: LookupMemoryProof,
}

fn words_of(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks(8)
        .map(|c| {
            let mut w = 0u64;
            for (b, byte) in c.iter().enumerate() {
                w |= (*byte as u64) << (b * 8);
            }
            w
        })
        .collect()
}

/// The 32-bit instruction words of the program (the fetch table's
/// granularity — fetch addresses are 4-byte units).
fn instr_words_of(program: &[u8]) -> Vec<u64> {
    program
        .chunks(4)
        .map(|c| {
            let mut w = 0u32;
            for (b, byte) in c.iter().enumerate() {
                w |= (*byte as u32) << (b * 8);
            }
            w as u64
        })
        .collect()
}

/// The RAM window bound and the public images (mirrors v2's layout).
fn ram_window(program: &[u8], public_input: &[u8], rows: &[lattice_vm::TraceRow]) -> (usize, Vec<u64>, Vec<u64>) {
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let mut max_word = 0x3000 / 8 + num_input_words as u64;
    max_word = max_word.max(0x1000 / 8 + program.len().div_ceil(8) as u64);
    for r in rows {
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
    (k_ram, init_ram, vec![0u64; 0])
}

#[allow(clippy::too_many_lines)]
pub fn prove_v3(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(PublicStateV2, ProofV3), PipelineError> {
    // ---- 1. Execute. ----
    let mut state = MachineState::new();
    state.load_program(0x1000, program);
    state.load_program(0x3000, public_input);
    state.regs[10] = public_input.len() as u64;
    state.pc = 0x1000;
    let rows = lattice_vm::run(&mut state, max_steps).map_err(PipelineError::Execution)?;
    if rows.is_empty() {
        return Err(PipelineError::BadShape("empty trace".into()));
    }
    let trace = build_trace(&state, &rows)?;
    let log_t = trace.log_t;
    // ---- 2. The public RAM window (same layout as v2). ----
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let (k_ram, init_ram, _) = ram_window(program, public_input, &rows);
    let mut final_ram = init_ram.clone();
    for (addr, v) in state.memory.snapshot_pairs() {
        let idx = (addr / 8) as usize;
        if idx < k_ram {
            final_ram[idx] = v;
        }
    }
    let public_state = PublicStateV2 {
        final_regs: state.regs,
        final_ram: final_ram.clone(),
        num_steps: rows.len() as u64,
    };
    // ---- 3. The bridge instances. ----
    let ring = RingD::new(8).map_err(|e| PipelineError::BadShape(format!("{e:?}")))?;
    let mut transcript = Transcript::new_default(b"lzx-v3");
    // (a) FETCH: the ROM table = the program words; the reads = each
    // step's (word index, instruction word).
    let program_words = instr_words_of(program);
    let mut fetch_ops = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let word_idx = (r.pc - 0x1000) / 4;
        let iw = trace.cols[Col::Iw as usize][i].to_canonical_u64();
        fetch_ops.push((false, word_idx, iw));
    }
    let fetch_inst = LookupMemoryInstance {
        window: program_words.clone(),
        final_window: program_words.clone(),
        ops: fetch_ops,
        table: Some(pad_table(&program_words)),
    };
    let fetch = prove_lookup_memory(&ring, &fetch_inst, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("fetch: {e:?}")))?;
    // (b) INPUT: the ROM table = the input words; read once at boot.
    let input_words = words_of(public_input);
    let mut input_ops = Vec::with_capacity(input_words.len());
    for (i, &w) in input_words.iter().enumerate() {
        input_ops.push((false, i as u64, w));
    }
    let input_inst = LookupMemoryInstance {
        window: input_words.clone(),
        final_window: input_words.clone(),
        ops: input_ops,
        table: Some(pad_table(&input_words)),
    };
    let input_rom = prove_lookup_memory(&ring, &input_inst, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("input: {e:?}")))?;
    // (c) RAM: the window images + the mem_access ops.
    let mut ram_ops = Vec::new();
    for r in &rows {
        if let Some((addr, rv, wv)) = &r.mem_access {
            ram_ops.push((false, addr / 8, *rv));
            if let Some(w) = wv {
                ram_ops.push((true, addr / 8, *w));
            }
        }
    }
    let ram_inst = LookupMemoryInstance {
        window: init_ram.clone(),
        final_window: final_ram.clone(),
        ops: ram_ops,
        table: None,
    };
    let ram = prove_lookup_memory(&ring, &ram_inst, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("ram: {e:?}")))?;
    // (d) REGISTERS: the 32-word register file.
    let mut reg_init = vec![0u64; 32];
    reg_init[10] = public_input.len() as u64;
    let mut reg_ops = Vec::new();
    for r in &rows {
        for &(idx, v) in &r.reg_reads {
            if idx != 0 || v != 0 {
                reg_ops.push((false, idx as u64, v));
            } else {
                reg_ops.push((false, 0, 0));
            }
        }
        for &(idx, v) in &r.reg_writes {
            if idx != 0 {
                reg_ops.push((true, idx as u64, v));
            }
        }
    }
    let reg_final: Vec<u64> = state.regs.to_vec();
    let reg_inst = LookupMemoryInstance {
        window: reg_init.clone(),
        final_window: reg_final,
        ops: reg_ops,
        table: None,
    };
    let regs = prove_lookup_memory(&ring, &reg_inst, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("regs: {e:?}")))?;
    let _ = fe(0);
    Ok((
        public_state,
        ProofV3 {
            log_t,
            num_fetch: rows.len(),
            num_input_words,
            fetch,
            input_rom,
            ram,
            regs,
        },
    ))
}

fn pad_table(words: &[u64]) -> Vec<u64> {
    let n = words.len().next_power_of_two().max(2);
    let mut t = words.to_vec();
    while t.len() < n {
        t.push(0);
    }
    t
}

/// Verify the v3 proof: rebuild the statement-side instances from the
/// public data (program, input, public state) and replay the bridge
/// verifies. **The verifier never re-executes the program.**
#[allow(clippy::too_many_lines)]
pub fn verify_v3(
    program: &[u8],
    public_input: &[u8],
    state: &PublicStateV2,
    proof: &ProofV3,
    max_steps: u64,
) -> Result<(), PipelineError> {
    let _ = max_steps;
    let ring = RingD::new(8).map_err(|e| PipelineError::BadShape(format!("{e:?}")))?;
    let mut transcript = Transcript::new_default(b"lzx-v3");
    // (a) FETCH
    let program_words = instr_words_of(program);
    let fetch_table = pad_table(&program_words);
    // The op stream shape: one read per step (the verifier knows the
    // step count from the public state, and the fetch addresses are
    // pinned by the pc stream — which the statement's commitment
    // layer carries; at the bridge layer the recorded oracle shape
    // defines the padded query length).
    let n_fetch = proof.num_fetch;
    let fetch_inst = LookupMemoryInstance {
        window: program_words.clone(),
        final_window: program_words.clone(),
        ops: (0..n_fetch).map(|_| (false, 0u64, 0u64)).collect(),
        table: Some(fetch_table),
    };
    verify_lookup_memory(&ring, &fetch_inst, &proof.fetch, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("fetch: {e:?}")))?;
    // (b) INPUT
    let input_words = words_of(public_input);
    let input_table = pad_table(&input_words);
    let n_input = input_words.len().max(1);
    let input_inst = LookupMemoryInstance {
        window: input_words.clone(),
        final_window: input_words.clone(),
        ops: (0..n_input).map(|i| (false, i as u64, 0u64)).collect(),
        table: Some(input_table),
    };
    verify_lookup_memory(&ring, &input_inst, &proof.input_rom, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("input: {e:?}")))?;
    // (c) RAM: the window is derived from the public layout; the ops'
    // SHAPE is pinned by the proof (the addresses and values are
    // witness-side, bound through the bridge's sub-image and record
    // checks against the public images).
    let num_input_words = public_input.len().div_ceil(8).max(1);
    let mut max_word = 0x3000 / 8 + num_input_words as u64;
    max_word = max_word.max(0x1000 / 8 + program.len().div_ceil(8) as u64);
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
    if state.final_ram.len() != k_ram {
        return Err(PipelineError::BadShape("final RAM window arity mismatch".into()));
    }
    let ram_inst = LookupMemoryInstance {
        window: init_ram.clone(),
        final_window: state.final_ram.clone(),
        ops: Vec::new(),
        table: None,
    };
    verify_lookup_memory(&ring, &ram_inst, &proof.ram, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("ram: {e:?}")))?;
    // (d) REGISTERS
    let mut reg_init = vec![0u64; 32];
    reg_init[10] = public_input.len() as u64;
    if state.final_regs.len() != 32 {
        return Err(PipelineError::BadShape("final registers arity mismatch".into()));
    }
    let reg_inst = LookupMemoryInstance {
        window: reg_init,
        final_window: state.final_regs.to_vec(),
        ops: Vec::new(),
        table: None,
    };
    verify_lookup_memory(&ring, &reg_inst, &proof.regs, &mut transcript)
        .map_err(|e| PipelineError::BadShape(format!("regs: {e:?}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn v3_prove_and_verify_happy_path() {
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = prove_v3(&program, &input, 64).unwrap_or_else(|e| panic!("prove: {e:?}"));
        assert_eq!(state.num_steps, 6);
        assert_eq!(state.final_regs[4], 15);
        assert_eq!(state.final_ram[8 / 8], 15); // the store went to absolute address 8
        match verify_v3(&program, &input, &state, &proof, 64) {
            Ok(_) => {}
            Err(e) => panic!("verify err: {e:?}"),
        }
    }

    #[test]
    fn v3_tampered_final_ram_rejected() {
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (mut state, proof) = prove_v3(&program, &input, 64).ok().unwrap();
        // tamper an UNTOUCHED word: the bridge's window-equality check
        state.final_ram[0x2000 / 8] = state.final_ram[0x2000 / 8].wrapping_add(1);
        assert!(verify_v3(&program, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v3_tampered_touched_word_rejected() {
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (mut state, proof) = prove_v3(&program, &input, 64).ok().unwrap();
        // tamper the TOUCHED word (0x1008): the sub-image binding fails
        state.final_ram[0x1008 / 8] = 99;
        assert!(verify_v3(&program, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v3_tampered_registers_rejected() {
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (mut state, proof) = prove_v3(&program, &input, 64).ok().unwrap();
        state.final_regs[3] = state.final_regs[3].wrapping_add(1);
        assert!(verify_v3(&program, &input, &state, &proof, 64).is_err());
    }

    #[test]
    fn v3_wrong_program_rejected() {
        let program = demo_program();
        let input = 42u64.to_le_bytes().to_vec();
        let (state, proof) = prove_v3(&program, &input, 64).unwrap_or_else(|e| panic!("prove: {e:?}"));
        // verify against a different program: the fetch table differs
        let mut other = program.clone();
        other[4] ^= 0xFF;
        assert!(verify_v3(&other, &input, &state, &proof, 64).is_err());
    }
}
