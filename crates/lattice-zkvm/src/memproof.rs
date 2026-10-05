//! The zkVM memory-argument proof: a complete, non-re-executing
//! prove/verify pipeline over real program executions (Wave 7.4 P0-5's
//! first landed stage).
//!
//! **What this proves**: for a program `P` with public input `I`, the
//! proof binds a claimed final register state and final memory image to a
//! *committed memory evolution* — four per-limb Twist instances for the
//! register file, four for RAM, and one read-only Shout instance for
//! instruction fetch — with virtual-`Val` Val-evaluation sumchecks
//! (Fig 9), matrix-evaluation sumchecks for every virtual one-hot /
//! increment claim, and two Ajtai bundles (bit-packed and value-packed)
//! authenticating every base claim through grouped openings with compact
//! norm proofs. The verifier never runs the program: it recomputes only
//! public tables (the program image, the initial memory image, the
//! challenge points) and O(λ) field work.
//!
//! **What it does not yet prove** (the honest ledger): the
//! instruction-semantics constraint families that pin the read/write
//! streams to the *semantics of the executed instructions* (ALU results,
//! decode, control flow). Those land in `constraints.rs` — the booleanity
//! family and the auxiliary-column substrate are staged there; the
//! arith/logic/comparison/control/routing/halted families are the next
//! wave. Until they land, `prove_memory_argument` must be read as a
//! proof of the memory-checking relation over committed streams, not yet
//! of full instruction semantics.
//!
//! Memory layout contract: program at address 0, public input at
//! `0x1000`, 8-byte word granularity, the RAM window `2^log_k` words,
//! the fetch window `2^log_kf` instruction words. Unaligned accesses,
//! compressed instructions, and out-of-window addresses fail closed at
//! witness build time.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_vm::{run as vm_run, MachineState};

use crate::claimsfold::{prove_claims_fold, verify_claims_fold, ClaimsFoldProof};
use crate::columns::{build_cycle_witness, CycleWitness, FetchWindow, RamWindow};
use crate::ledger::{
    bits_bundle_commit, values_bundle_commit, verify_bundle_opening, BaseClaim, BundleOpening,
    Factor, Ledger, SlotLedger,
};
use crate::memory::{
    activity_factor, addr_factor, inc_factor, prove_memory, rv_factor, verify_memory, wv_factor,
    MemoryInstance, MemoryProof, INC_OFFSET,
};

/// Public statement: the program/input digests + the claimed final state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryStatement {
    pub program_digest: [u8; 32],
    pub input_digest: [u8; 32],
    pub log_t: usize,
    pub ram_log_k: usize,
    pub fetch_log_k: usize,
    pub final_regs: [u64; 32],
    /// The final RAM image over the window (`2^ram_log_k` words), in the
    /// public output.
    pub final_memory: Vec<u64>,
}

/// The full proof.
#[derive(Clone, Debug)]
pub struct MemoryArgumentProof {
    pub statement: MemoryStatement,
    /// Base claims in protocol order.
    pub claims: Vec<BaseClaim>,
    /// Per-instance legs (instances 0..9 in the fixed order).
    pub legs: Vec<MemoryProof>,
    /// Bundle commitments (serialized).
    pub bits_commitment: Vec<u8>,
    pub values_commitment: Vec<u8>,
    pub bits_opening: BundleOpening,
    pub values_opening: BundleOpening,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemProofError {
    Witness(crate::columns::WitnessError),
    Memory(crate::memory::MemoryError),
    Ledger(crate::ledger::LedgerError),
    Execution(lattice_vm::ExecError),
    /// The Stage-5.2 claims fold's own error surface.
    Fold(crate::claimsfold::FoldError),
    Shape,
    VerificationFailed,
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// The 4-bit limb of a u64.
fn limb(v: u64, l: usize) -> u64 {
    (v >> (16 * l)) & 0xFFFF
}

/// Build the nine memory instances from the witness + machine state.
#[allow(clippy::too_many_arguments)]
fn build_instances(
    w: &CycleWitness,
    final_regs: &[u64; 32],
    final_memory: &std::collections::BTreeMap<u64, u64>,
    ram_log_k: usize,
    fetch_log_k: usize,
    program: &[u8],
    public_input: &[u8],
) -> Vec<MemoryInstance> {
    let t = 1usize << w.log_t;
    let log_ts = 2 + w.log_t;
    let t_s = 1usize << log_ts;
    let k_ram = 1usize << ram_log_k;
    let k_fetch = 1usize << fetch_log_k;
    let mut out = Vec::with_capacity(9);

    // The initial memory image: program words + input words at 0x1000.
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

    // Register instances (limbs 0..4).
    for l in 0..4usize {
        let mut addr = vec![0u64; t_s];
        let mut ractive = vec![0u8; t_s];
        let mut wactive = vec![0u8; t_s];
        let mut rv = vec![Goldilocks::ZERO; t_s];
        let mut wv = vec![Goldilocks::ZERO; t_s];
        let mut inc_off = vec![fe(INC_OFFSET); t_s];
        let mut reg_state = vec![0u64; 32];
        for c in 0..t {
            let rs1 = w.rs1_idx[c] as usize;
            let rs2 = w.rs2_idx[c] as usize;
            let rd = w.rd_idx[c] as usize;
            let writes = w.rd_we[c] != 0;
            // Slots: [rs1-read, rs2-read, rd-write, pad] — CYCLE-MAJOR
            // layout (j = 4c + s) so a cycle's reads precede its write in
            // the memory timeline.
            let slot = |c: usize, s: usize| c * 4 + s;
            let s0 = slot(c, 0);
            let s1 = slot(c, 1);
            let s2 = slot(c, 2);
            let s3 = slot(c, 3);
            let _ = s3;
            addr[s0] = rs1 as u64;
            addr[s1] = rs2 as u64;
            addr[s2] = rd as u64;
            addr[s3] = 0;
            ractive[s0] = 1;
            ractive[s1] = 1;
            ractive[s3] = 0;
            wactive[s2] = writes as u8;
            rv[s0] = fe(limb(reg_state[rs1], l));
            rv[s1] = fe(limb(reg_state[rs2], l));
            if writes {
                let new_limb = limb(final_of(w, T_RD, c), l);
                wv[s2] = fe(new_limb);
                inc_off[s2] = fe(new_limb)
                    .sub(&fe(limb(reg_state[rd], l)))
                    .add(&fe(INC_OFFSET));
                reg_state[rd] = final_of(w, T_RD, c);
            }
        }
        let init = vec![Goldilocks::ZERO; 32];
        let final_state: Vec<Goldilocks> = final_regs.iter().map(|r| fe(limb(*r, l))).collect();
        out.push(MemoryInstance {
            log_k: 5,
            log_ts,
            addr,
            ractive,
            wactive,
            rv,
            wv,
            inc_off,
            init,
            final_state,
            table: None,
        });
    }

    // RAM instances (limbs 0..4).
    for l in 0..4usize {
        let mut addr = vec![0u64; t_s];
        let mut ractive = vec![0u8; t_s];
        let mut wactive = vec![0u8; t_s];
        let mut rv = vec![Goldilocks::ZERO; t_s];
        let mut wv = vec![Goldilocks::ZERO; t_s];
        let mut inc_off = vec![fe(INC_OFFSET); t_s];
        // The running word state (from the witness's mem_old/mem_new).
        let mut state: std::collections::BTreeMap<u64, u64> =
            (0..k_ram as u64).map(|k| (k, init_image(k))).collect();
        for c in 0..t {
            let access = w.mem_re[c] != 0 || w.mem_we[c] != 0;
            let word = w.mem_word[c].to_canonical_u64();
            let s0 = c * 4;
            let s1 = c * 4 + 1;
            let s2 = c * 4 + 2;
            let s3 = c * 4 + 3;
            let _ = s3;
            addr[s0] = word;
            addr[s1] = word;
            addr[s2] = 0;
            addr[s3] = 0;
            ractive[s0] = access as u8;
            wactive[s1] = (w.mem_we[c] != 0) as u8;
            if access {
                let old = state.get(&word).copied().unwrap_or(0);
                rv[s0] = fe(limb(old, l));
                if w.mem_we[c] != 0 {
                    let new_word = final_of(w, T_MEM_NEW, c);
                    wv[s1] = fe(limb(new_word, l));
                    inc_off[s1] = fe(limb(new_word, l))
                        .sub(&fe(limb(old, l)))
                        .add(&fe(INC_OFFSET));
                    state.insert(word, new_word);
                }
            }
        }
        let init: Vec<Goldilocks> = (0..k_ram as u64)
            .map(|k| fe(limb(init_image(k), l)))
            .collect();
        let final_state: Vec<Goldilocks> = (0..k_ram as u64)
            .map(|k| {
                let v = final_memory
                    .get(&k)
                    .copied()
                    .unwrap_or_else(|| init_image(k));
                fe(limb(v, l))
            })
            .collect();
        out.push(MemoryInstance {
            log_k: ram_log_k,
            log_ts,
            addr,
            ractive,
            wactive,
            rv,
            wv,
            inc_off,
            init,
            final_state,
            table: None,
        });
    }

    // Fetch instance: one read slot per cycle against the program table.
    {
        let mut addr = vec![0u64; t];
        let ractive = vec![1u8; t];
        let wactive = vec![0u8; t];
        let mut rv = vec![Goldilocks::ZERO; t];
        let wv = vec![Goldilocks::ZERO; t];
        let inc_off = vec![fe(INC_OFFSET); t];
        for c in 0..t {
            addr[c] = w.fetch_word[c].to_canonical_u64();
            rv[c] = w.instr[c];
        }
        let table: Vec<Goldilocks> = (0..k_fetch)
            .map(|k| {
                let base = k * 4;
                let mut word = 0u32;
                for i in 0..4usize {
                    if base + i < program.len() {
                        word |= (program[base + i] as u32) << (8 * i);
                    }
                }
                fe(word as u64)
            })
            .collect();
        let init = table.clone();
        let final_state = table.clone();
        out.push(MemoryInstance {
            log_k: fetch_log_k,
            log_ts: w.log_t,
            addr,
            ractive,
            wactive,
            rv,
            wv,
            inc_off,
            init,
            final_state,
            table: Some(table),
        });
    }
    out
}

use crate::columns::{T_MEM_NEW, T_RD};

/// Reconstruct a cycle's tensor word (the committed value).
fn final_of(w: &CycleWitness, slot: usize, c: usize) -> u64 {
    let t = 1usize << w.log_t;
    let mut v = 0u64;
    for bit in 0..64usize {
        v |= (w.values[slot].evaluations[bit * t + c].to_canonical_u64() & 1) << (63 - bit);
    }
    v
}

/// Prove the memory argument end-to-end.
pub fn prove_memory_argument(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    ram_log_k: usize,
    fetch_log_k: usize,
) -> Result<(MemoryArgumentProof, [u64; 32]), MemProofError> {
    // 1. Execute.
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let rows = vm_run(&mut state, max_steps).map_err(MemProofError::Execution)?;
    let final_regs = state.regs;

    // 2. Witness.
    let (w, final_words) = build_cycle_witness(
        &rows,
        program,
        public_input,
        RamWindow { log_k: ram_log_k },
        FetchWindow { log_k: fetch_log_k },
    )
    .map_err(MemProofError::Witness)?;
    let t = 1usize << w.log_t;
    let k_ram = 1usize << ram_log_k;
    let _ = k_ram;
    // The final memory image over the window: touched words + init.
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
    let final_memory: Vec<u64> = (0..(1usize << ram_log_k))
        .map(|k| {
            final_words
                .get(&(k as u64))
                .copied()
                .unwrap_or_else(|| init_image(k as u64))
        })
        .collect();

    // 3. Instances.
    let instances = build_instances(
        &w,
        &final_regs,
        &final_words,
        ram_log_k,
        fetch_log_k,
        program,
        public_input,
    );

    // 4. The factor table + bundles.
    let mut digit_tensors: Vec<(Factor, DenseMle)> = Vec::new();
    let mut activity_cols: Vec<(Factor, DenseMle)> = Vec::new();
    let mut stream_cols: Vec<(Factor, DenseMle)> = Vec::new();
    for (i, m) in instances.iter().enumerate() {
        digit_tensors.push((Factor::DigitBits { inst: i }, m.digit_tensor()));
        activity_cols.push((activity_factor(i, false), m.activity_col(false)));
        activity_cols.push((activity_factor(i, true), m.activity_col(true)));
        stream_cols.push((addr_factor(i), m.addr_col()));
        stream_cols.push((rv_factor(i), m.rv_col()));
        if !m.read_only() {
            stream_cols.push((wv_factor(i), m.wv_col()));
            stream_cols.push((inc_factor(i), m.inc_col()));
        }
    }
    // 5. The statement (public) + the derived seed.
    let program_digest = Transcript::hash_domain(b"zkvm-program", program);
    let input_digest = Transcript::hash_domain(b"zkvm-public-input", public_input);
    let statement = MemoryStatement {
        program_digest,
        input_digest,
        log_t: w.log_t,
        ram_log_k,
        fetch_log_k,
        final_regs,
        final_memory: final_memory.clone(),
    };
    let seed = derive_seed(&statement);
    let mut bits_entries = digit_tensors.clone();
    bits_entries.extend(activity_cols.clone());
    let bits_prover = bits_bundle_commit(&bits_entries, seed).map_err(MemProofError::Ledger)?;
    let values_prover = values_bundle_commit(&stream_cols, seed).map_err(MemProofError::Ledger)?;
    let bits_commitment = bits_prover.commitment.clone();
    let values_commitment = values_prover.commitment.clone();
    let mut transcript = Transcript::new_default(b"lzx-zkvm-memarg");
    absorb_statement(&statement, &mut transcript)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"bits-commitment", &bits_commitment.to_bytes())
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"values-commitment", &values_commitment.to_bytes())
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;

    // 6. The ledger over the factor table.
    let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
    for (f, m) in digit_tensors.iter() {
        table.push((*f, m));
    }
    for (f, m) in activity_cols.iter() {
        table.push((*f, m));
    }
    for (f, m) in stream_cols.iter() {
        table.push((*f, m));
    }
    let mut ledger = Ledger::prover(table);
    let mut all_legs: Vec<MemoryProof> = Vec::with_capacity(instances.len());
    for (i, m) in instances.iter().enumerate() {
        let mut inst_legs: Vec<crate::memory::LegProof> = Vec::new();
        let r = prove_memory(i, m, &mut ledger, &mut inst_legs, &mut transcript);
        if let Err(e) = &r {
            println!(
                "instance {} (log_k={}, log_ts={}) failed: {:?} after {} legs (last: {:?})",
                i,
                m.log_k,
                m.log_ts,
                e,
                inst_legs.len(),
                inst_legs.last().map(|l| l.name)
            );
        }
        r.map_err(MemProofError::Memory)?;
        all_legs.push(MemoryProof { legs: inst_legs });
    }
    let _ = t;

    // 7. Bundle openings over the recorded claims.
    let claims: Vec<BaseClaim> = ledger.claims().to_vec();
    let bits_claims: Vec<BaseClaim> = claims
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = claims
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_opening = bits_prover
        .prove_opening(&bits_claims, &mut transcript)
        .map_err(MemProofError::Ledger)?;
    let values_opening = values_prover
        .prove_opening(&values_claims, &mut transcript)
        .map_err(MemProofError::Ledger)?;

    Ok((
        MemoryArgumentProof {
            statement,
            claims,
            legs: all_legs,
            bits_commitment: bits_commitment.to_bytes(),
            values_commitment: values_commitment.to_bytes(),
            bits_opening,
            values_opening,
        },
        final_regs,
    ))
}

fn absorb_statement(
    s: &MemoryStatement,
    transcript: &mut Transcript,
) -> Result<(), lattice_core::transcript::TranscriptError> {
    let mut meta = vec![
        s.log_t as u64,
        s.ram_log_k as u64,
        s.fetch_log_k as u64,
        s.final_memory.len() as u64,
    ];
    meta.extend(s.final_regs.iter().copied());
    meta.extend(s.final_memory.iter().copied());
    let fields: Vec<Goldilocks> = meta.iter().map(|v| Goldilocks::from_u64(*v)).collect();
    transcript.append_bytes(b"stmt-prog", &s.program_digest)?;
    transcript.append_bytes(b"stmt-in", &s.input_digest)?;
    transcript.append_field_slice(b"stmt-meta", &fields)?;
    Ok(())
}

/// Verify the memory argument with NO re-execution.
pub fn verify_memory_argument(
    proof: &MemoryArgumentProof,
    program: &[u8],
    public_input: &[u8],
) -> Result<(), MemProofError> {
    // 1. Statement digests.
    let program_digest = Transcript::hash_domain(b"zkvm-program", program);
    let input_digest = Transcript::hash_domain(b"zkvm-public-input", public_input);
    let s = &proof.statement;
    if s.program_digest != program_digest || s.input_digest != input_digest {
        return Err(MemProofError::VerificationFailed);
    }

    // 2. Rebuild the instances' public structure (init/final/table) — no
    //    execution, only public-table recomputation.
    let t = 1usize << s.log_t;
    let log_ts = 2 + s.log_t;
    let k_ram = 1usize << s.ram_log_k;
    let k_fetch = 1usize << s.fetch_log_k;
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
    let limb = |v: u64, l: usize| fe((v >> (16 * l)) & 0xFFFF);
    let mut instances: Vec<MemoryInstance> = Vec::with_capacity(9);
    for l in 0..4usize {
        instances.push(MemoryInstance {
            log_k: 5,
            log_ts,
            addr: vec![0; 1 << log_ts],
            ractive: vec![0; 1 << log_ts],
            wactive: vec![0; 1 << log_ts],
            rv: vec![Goldilocks::ZERO; 1 << log_ts],
            wv: vec![Goldilocks::ZERO; 1 << log_ts],
            inc_off: vec![fe(INC_OFFSET); 1 << log_ts],
            init: vec![Goldilocks::ZERO; 32],
            final_state: s.final_regs.iter().map(|r| limb(*r, l)).collect(),
            table: None,
        });
    }
    for l in 0..4usize {
        instances.push(MemoryInstance {
            log_k: s.ram_log_k,
            log_ts,
            addr: vec![0; 1 << log_ts],
            ractive: vec![0; 1 << log_ts],
            wactive: vec![0; 1 << log_ts],
            rv: vec![Goldilocks::ZERO; 1 << log_ts],
            wv: vec![Goldilocks::ZERO; 1 << log_ts],
            inc_off: vec![fe(INC_OFFSET); 1 << log_ts],
            init: (0..k_ram as u64).map(|k| limb(init_image(k), l)).collect(),
            final_state: (0..k_ram)
                .map(|k| limb(s.final_memory.get(k).copied().unwrap_or(0), l))
                .collect(),
            table: None,
        });
    }
    {
        let table: Vec<Goldilocks> = (0..k_fetch)
            .map(|k| {
                let base = k * 4;
                let mut word = 0u32;
                for i in 0..4usize {
                    if base + i < program.len() {
                        word |= (program[base + i] as u32) << (8 * i);
                    }
                }
                fe(word as u64)
            })
            .collect();
        instances.push(MemoryInstance {
            log_k: s.fetch_log_k,
            log_ts: s.log_t,
            addr: vec![0; t],
            ractive: vec![1; t],
            wactive: vec![0; t],
            rv: vec![Goldilocks::ZERO; t],
            wv: vec![Goldilocks::ZERO; t],
            inc_off: vec![fe(INC_OFFSET); t],
            init: table.clone(),
            final_state: table.clone(),
            table: Some(table),
        });
    }

    // 3. Transcript replay.
    let mut transcript = Transcript::new_default(b"lzx-zkvm-memarg");
    absorb_statement(s, &mut transcript)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    let ring = crate::ledger::bundle_ring().map_err(MemProofError::Ledger)?;
    let bits_t =
        lattice_commitment::ajtai::AjtaiCommitment::from_bytes(&ring, 2, &proof.bits_commitment)
            .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Ajtai(e)))?;
    let values_t =
        lattice_commitment::ajtai::AjtaiCommitment::from_bytes(&ring, 2, &proof.values_commitment)
            .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Ajtai(e)))?;
    transcript
        .append_bytes(b"bits-commitment", &bits_t.to_bytes())
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"values-commitment", &values_t.to_bytes())
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;

    // 4. Ledger (verifier mode) + per-instance verification.
    let mut ledger = Ledger::verifier(proof.claims.clone());
    if proof.legs.len() != 9 {
        return Err(MemProofError::Shape);
    }
    for (i, m) in instances.iter().enumerate() {
        verify_memory(i, m, &proof.legs[i], &mut ledger, &mut transcript)
            .map_err(MemProofError::Memory)?;
    }

    // 5. Bundle openings.
    let bits_claims: Vec<BaseClaim> = proof
        .claims
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = proof
        .claims
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    // Rebuild the bundle layouts (deterministic from the instance
    // geometry).
    let bits_layout = {
        // MUST match the prover's entry order: all digit tensors first,
        // then the activity columns (false/true per instance).
        let mut entries: Vec<(Factor, DenseMle)> = Vec::new();
        for (i, m) in instances.iter().enumerate() {
            entries.push((Factor::DigitBits { inst: i }, m.digit_tensor()));
        }
        for (i, m) in instances.iter().enumerate() {
            entries.push((activity_factor(i, false), m.activity_col(false)));
            entries.push((activity_factor(i, true), m.activity_col(true)));
        }
        let (_, layout) = crate::ledger::build_flat_mle(&entries).map_err(MemProofError::Ledger)?;
        layout
    };
    let values_layout = {
        let mut entries: Vec<(Factor, DenseMle)> = Vec::new();
        for (i, m) in instances.iter().enumerate() {
            entries.push((addr_factor(i), m.addr_col()));
            entries.push((rv_factor(i), m.rv_col()));
            if !m.read_only() {
                entries.push((wv_factor(i), m.wv_col()));
                entries.push((inc_factor(i), m.inc_col()));
            }
        }
        let (_, layout) = crate::ledger::build_flat_mle(&entries).map_err(MemProofError::Ledger)?;
        layout
    };
    // Recreate the public keys (deterministic from the seed — the seed is
    // derived from the statement digest).
    let seed = derive_seed(s);
    let bits_pk = {
        let m = bits_m_from_layout(&bits_layout);
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: crate::ledger::BITS_NORM_BOUND,
        };
        lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, seed)
            .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Ajtai(e)))?
    };
    let values_pk = {
        let m = values_m_from_layout(&values_layout);
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: crate::ledger::VALUES_NORM_BOUND,
        };
        lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, seed)
            .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Ajtai(e)))?
    };
    verify_bundle_opening(
        &bits_pk,
        &bits_t,
        &bits_layout,
        &bits_claims,
        &proof.bits_opening,
        true,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    verify_bundle_opening(
        &values_pk,
        &values_t,
        &values_layout,
        &values_claims,
        &proof.values_opening,
        false,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    Ok(())
}

fn derive_seed(s: &MemoryStatement) -> [u8; 32] {
    let mut buf = Vec::new();
    buf.extend_from_slice(&s.program_digest);
    buf.extend_from_slice(&s.input_digest);
    buf.extend_from_slice(&s.log_t.to_le_bytes());
    buf.extend_from_slice(&s.ram_log_k.to_le_bytes());
    buf.extend_from_slice(&s.fetch_log_k.to_le_bytes());
    let mut seed = [0u8; 32];
    let h = Transcript::hash_domain(b"zkvm-seed", &buf);
    seed.copy_from_slice(&h);
    seed
}

fn bits_m_from_layout(layout: &[crate::ledger::BundleLayoutEntry]) -> usize {
    let flat: usize = layout.iter().map(|e| 1usize << e.num_vars).sum();
    let padded = flat.next_power_of_two().max(1);
    padded.div_ceil(31 * 64)
}

fn values_m_from_layout(layout: &[crate::ledger::BundleLayoutEntry]) -> usize {
    let flat: usize = layout.iter().map(|e| 1usize << e.num_vars).sum();
    let padded = flat.next_power_of_two().max(1);
    (padded * 3).div_ceil(64)
}

// ---------------------------------------------------------------------------
// The compact proof mode (docs/DESIGN_50KB.md): the same legs + carrier,
// with the Θ(N) digit reveal replaced by the folded compact opening.
// ---------------------------------------------------------------------------

use crate::compact::{compact_bundle_commit, verify_compact_opening, CompactOpening};
use crate::ledger::{idx_point, LedgerError};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::VirtualPolynomial;

/// The compact memory-argument proof — **the Stage-5.2 folded form**: the
/// values-only claims list is REPLACED by the claims fold (`fold`), with
/// only the 18 pre-leg claimed-vector entries (the address / read-value
/// entries) still crossing the wire in the clear.
#[derive(Clone, Debug)]
pub struct CompactMemoryProof {
    pub statement: MemoryStatement,
    /// The claims fold: the layer sumchecks + the terminal claims + the
    /// per-bundle leaf sums (the values list never appears).
    pub fold: ClaimsFoldProof,
    /// The Stage-4 batched legs (12 sumchecks replacing the ~117
    /// per-instance legs; the DESIGN_50KB final cut).
    pub legs: crate::legbatch::BatchedLegs,
    /// The per-bundle column commitments (r × k ring elements each).
    pub bits_commitment: Vec<u8>,
    pub values_commitment: Vec<u8>,
    /// The Goldilocks grouped carriers (with the fold-derived weights).
    pub bits_carrier: sumcheck::SumcheckProof,
    pub values_carrier: sumcheck::SumcheckProof,
    /// f(r_sc) per bundle (the carrier terminal bound by the fold).
    pub bits_w: Goldilocks,
    pub values_w: Goldilocks,
    /// The compact openings.
    pub bits_opening: CompactOpening,
    pub values_opening: CompactOpening,
    /// Factor lengths per bundle (for the verifier's layout rebuild).
    pub bits_factor_lens: Vec<usize>,
    pub values_factor_lens: Vec<usize>,
}

/// The per-bundle fold parameters (r, k) chosen from the packed size: the
/// response stays near a fixed budget while the commitment scales with r.
fn fold_params_for(total_values: usize, max_value_bytes: usize) -> (usize, usize) {
    // Target ~8–16k response coefficients: r ≈ stream/2^13 where
    // stream ≈ total_values × avg bytes.
    let stream = total_values.saturating_mul(max_value_bytes.max(1));
    let mut r = 4usize;
    while r < 64 && stream / (r * 2) > 8192 {
        r *= 2;
    }
    // k = 4: the estimator-run interim hardening (SECURITY.md's MSIS
    // table — the rank-2 module is broken at every response length; the
    // full sound posture is the second-level fold, Stage 5.2).
    (r, 4usize)
}

/// Map claims to their flat-domain points (the ledger's flat_point
/// convention: offset bits then the factor's own variables). Full-Factor
/// matching (discriminant + payload) picks the entry.
fn flat_points_for_claims_full(
    entries: &[(Factor, &DenseMle)],
    claims: &[BaseClaim],
    log_flat: usize,
) -> Result<Vec<Vec<Goldilocks>>, MemProofError> {
    let le = |e: LedgerError| MemProofError::Ledger(e);
    let mut out = Vec::with_capacity(claims.len());
    for c in claims {
        let (num_vars, offset) = {
            let mut off = 0usize;
            let mut found = None;
            for (f, mle) in entries {
                if *f == c.factor {
                    found = Some((mle.num_vars, off));
                    break;
                }
                off += mle.evaluations.len();
            }
            found
        }
        .ok_or_else(|| le(LedgerError::Layout("claim factor not in bundle".into())))?;
        if c.point.len() != num_vars {
            return Err(le(LedgerError::PointArity {
                expected: num_vars,
                got: c.point.len(),
            }));
        }
        let head_bits = log_flat - num_vars;
        let slice = offset >> num_vars;
        let mut pt = idx_point(head_bits, slice);
        pt.extend_from_slice(&c.point);
        out.push(pt);
    }
    Ok(out)
}

/// Prove the memory argument with the compact (folded) openings.
#[allow(clippy::too_many_arguments)]
pub fn prove_memory_argument_compact(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    ram_log_k: usize,
    fetch_log_k: usize,
) -> Result<(CompactMemoryProof, [u64; 32]), MemProofError> {
    // 1–3. Execute, witness, instances (shared with the Clear mode).
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let rows = vm_run(&mut state, max_steps).map_err(MemProofError::Execution)?;
    let final_regs = state.regs;
    let (w, final_words) = build_cycle_witness(
        &rows,
        program,
        public_input,
        RamWindow { log_k: ram_log_k },
        FetchWindow { log_k: fetch_log_k },
    )
    .map_err(MemProofError::Witness)?;
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
    let final_memory: Vec<u64> = (0..(1usize << ram_log_k))
        .map(|k| {
            final_words
                .get(&(k as u64))
                .copied()
                .unwrap_or_else(|| init_image(k as u64))
        })
        .collect();
    let instances = build_instances(
        &w,
        &final_regs,
        &final_words,
        ram_log_k,
        fetch_log_k,
        program,
        public_input,
    );

    // 4. The factor table.
    let mut digit_tensors: Vec<(Factor, DenseMle)> = Vec::new();
    let mut activity_cols: Vec<(Factor, DenseMle)> = Vec::new();
    let mut stream_cols: Vec<(Factor, DenseMle)> = Vec::new();
    for (i, m) in instances.iter().enumerate() {
        digit_tensors.push((Factor::DigitBits { inst: i }, m.digit_tensor()));
        activity_cols.push((activity_factor(i, false), m.activity_col(false)));
        activity_cols.push((activity_factor(i, true), m.activity_col(true)));
        stream_cols.push((addr_factor(i), m.addr_col()));
        stream_cols.push((rv_factor(i), m.rv_col()));
        if !m.read_only() {
            stream_cols.push((wv_factor(i), m.wv_col()));
            stream_cols.push((inc_factor(i), m.inc_col()));
        }
    }

    // 5. The statement + seeds.
    let program_digest = Transcript::hash_domain(b"zkvm-program", program);
    let input_digest = Transcript::hash_domain(b"zkvm-public-input", public_input);
    let statement = MemoryStatement {
        program_digest,
        input_digest,
        log_t: w.log_t,
        ram_log_k,
        fetch_log_k,
        final_regs,
        final_memory: final_memory.clone(),
    };
    let seed = derive_seed(&statement);

    // 6. The compact bundle commitments.
    let bits_entries: Vec<(u32, &DenseMle)> = digit_tensors
        .iter()
        .chain(activity_cols.iter())
        .map(|(f, m)| (f.discriminant() as u32, m))
        .collect();
    let values_entries: Vec<(u32, &DenseMle)> = stream_cols
        .iter()
        .map(|(f, m)| (f.discriminant() as u32, m))
        .collect();
    let bits_total: usize = bits_entries.iter().map(|(_, m)| m.evaluations.len()).sum();
    let values_total: usize = values_entries
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .sum();
    let (r_bits, k_bits) = fold_params_for(bits_total, 1);
    let (r_vals, k_vals) = fold_params_for(values_total, 3);
    let bits_prover = compact_bundle_commit(&bits_entries, seed, r_bits, k_bits)
        .map_err(MemProofError::Ledger)?;
    let values_prover = compact_bundle_commit(&values_entries, seed, r_vals, k_vals)
        .map_err(MemProofError::Ledger)?;
    let bits_commitment = bits_prover.commitment_bytes();
    let values_commitment = values_prover.commitment_bytes();

    let mut transcript = Transcript::new_default(b"lzx-zkvm-memarg-compact");
    absorb_statement(&statement, &mut transcript)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"bits-commitment", &bits_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"values-commitment", &values_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;

    // 7. The legs + the fold context (the values never leave the prover).
    let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
    for (f, m) in digit_tensors.iter() {
        table.push((*f, m));
    }
    for (f, m) in activity_cols.iter() {
        table.push((*f, m));
    }
    for (f, m) in stream_cols.iter() {
        table.push((*f, m));
    }
    let mut ledger = Ledger::prover(table);
    let (all_legs, fold_ctx) =
        crate::legbatch::prove_legs_batched(&instances, &mut ledger, &mut transcript)
            .map_err(MemProofError::Memory)?;
    // The pre-leg claimed-vector entries are transmitted in the clear:
    // absorb them (FS hygiene) before any fold challenge.
    absorb_field_slice(&mut transcript, b"fold-addr", &fold_ctx.addr_claims)?;
    absorb_field_slice(&mut transcript, b"fold-rv", &fold_ctx.rv_claims)?;

    // 8. The claims fold (Stage 5.2): the deferred expects over the slot
    //    list, proven by the layer sumchecks — no values cross.
    let slot_claims: Vec<BaseClaim> = ledger.slot_claims();
    let slot_values: Vec<Goldilocks> = slot_claims.iter().map(|c| c.value).collect();
    let slot_bits: Vec<bool> = slot_claims
        .iter()
        .map(|c| c.factor.in_bits_bundle())
        .collect();
    let set = crate::legbatch::derive_claims_fold(
        &instances,
        &all_legs,
        &fold_ctx,
        |f, pt| {
            ledger
                .claim_index(f, pt)
                .ok_or(crate::memory::MemoryError::ClaimMismatch)
        },
        &mut transcript,
    )
    .map_err(MemProofError::Memory)?;
    let (fold, binding) = prove_claims_fold(
        &set,
        &slot_values,
        &slot_bits,
        fold_ctx.addr_claims.clone(),
        fold_ctx.rv_claims.clone(),
        &mut transcript,
    )
    .map_err(MemProofError::Fold)?;

    // 9. The carriers with the fold-derived weights + compact openings.
    let bits_flat = bits_prover.flat_mle();
    let values_flat = values_prover.flat_mle();
    let bits_refs: Vec<(Factor, &DenseMle)> = digit_tensors
        .iter()
        .chain(activity_cols.iter())
        .map(|(f, m)| (*f, m))
        .collect();
    let values_refs: Vec<(Factor, &DenseMle)> = stream_cols.iter().map(|(f, m)| (*f, m)).collect();
    let bits_claims: Vec<BaseClaim> = slot_claims
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = slot_claims
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_slot_of: Vec<usize> = (0..slot_claims.len())
        .filter(|&i| slot_claims[i].factor.in_bits_bundle())
        .collect();
    let values_slot_of: Vec<usize> = (0..slot_claims.len())
        .filter(|&i| !slot_claims[i].factor.in_bits_bundle())
        .collect();
    let bits_pts = flat_points_for_claims_full(&bits_refs, &bits_claims, bits_flat.num_vars)?;
    let values_pts =
        flat_points_for_claims_full(&values_refs, &values_claims, values_flat.num_vars)?;
    let bits_weights: Vec<Goldilocks> = bits_slot_of.iter().map(|&i| binding.weights[i]).collect();
    let values_weights: Vec<Goldilocks> =
        values_slot_of.iter().map(|&i| binding.weights[i]).collect();

    let (bits_carrier, bits_rsc, bits_w) = prove_carrier_folded(
        bits_flat,
        &bits_claims,
        &bits_pts,
        &bits_weights,
        fold.s_bits,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    let (values_carrier, values_rsc, values_w) = prove_carrier_folded(
        values_flat,
        &values_claims,
        &values_pts,
        &values_weights,
        fold.s_vals,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;

    let bits_opening = bits_prover
        .prove_compact_opening(&bits_rsc, &bits_w, &mut transcript)
        .map_err(MemProofError::Ledger)?;
    let values_opening = values_prover
        .prove_compact_opening(&values_rsc, &values_w, &mut transcript)
        .map_err(MemProofError::Ledger)?;

    Ok((
        CompactMemoryProof {
            statement,
            fold,
            legs: all_legs,
            bits_commitment,
            values_commitment,
            bits_carrier,
            values_carrier,
            bits_w,
            values_w,
            bits_opening,
            values_opening,
            bits_factor_lens: bits_entries
                .iter()
                .map(|(_, m)| m.evaluations.len())
                .collect(),
            values_factor_lens: values_entries
                .iter()
                .map(|(_, m)| m.evaluations.len())
                .collect(),
        },
        final_regs,
    ))
}

/// Absorb a field-element slice under a label.
fn absorb_field_slice(
    transcript: &mut Transcript,
    label: &[u8],
    values: &[Goldilocks],
) -> Result<(), MemProofError> {
    transcript
        .append_field_slice(label, values)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))
}

/// The folded carrier: the same grouped sumcheck as the clear-list
/// design with the fold-derived weights replacing the fresh ρ challenges
/// and the right-hand side the fold-bound per-bundle partial sum plus
/// the (verifier-computable) ρ′-weighted transmitted entries.
fn prove_carrier_folded(
    flat: &DenseMle,
    claims: &[BaseClaim],
    points: &[Vec<Goldilocks>],
    fold_weights: &[Goldilocks],
    s_bundle: Goldilocks,
    transcript: &mut Transcript,
) -> Result<(sumcheck::SumcheckProof, Vec<Goldilocks>, Goldilocks), LedgerError> {
    // The fresh ρ′ for the transmitted (addr/rv) slots of this bundle —
    // their values are public, so their carrier contribution is
    // verifier-computable.
    let trans: Vec<bool> = claims
        .iter()
        .map(|c| matches!(c.factor, Factor::AddrCol { .. } | Factor::RvCol { .. }))
        .collect();
    let n_trans = trans.iter().filter(|&&t| t).count();
    let rhos = transcript
        .challenge_fields(b"trans-rho", n_trans)
        .map_err(LedgerError::Transcript)?;
    let mut weights = Vec::with_capacity(claims.len());
    let mut rho_iter = rhos.iter();
    let mut rhs = s_bundle;
    for (i, c) in claims.iter().enumerate() {
        let w = fold_weights
            .get(i)
            .copied()
            .ok_or_else(|| LedgerError::Layout("fold weight arity".into()))?;
        if trans[i] {
            let rho = rho_iter.next().ok_or(LedgerError::Layout("rho".into()))?;
            weights.push(w.add(rho));
            rhs = rhs.add(&rho.mul(&c.value));
        } else {
            weights.push(w);
        }
    }
    let mut vp = VirtualPolynomial::new(flat.num_vars);
    let fi = vp.add_factor(flat.clone()).map_err(LedgerError::Virtual)?;
    for (i, pt) in points.iter().enumerate() {
        let eq = DenseMle::eq_extension(pt);
        let ei = vp.add_factor(eq).map_err(LedgerError::Virtual)?;
        vp.add_term(weights[i], vec![fi, ei])
            .map_err(LedgerError::Virtual)?;
    }
    let out = sumcheck::prove(&vp, rhs, transcript).map_err(LedgerError::Sumcheck)?;
    let w = out.factor_claims[fi];
    Ok((out.proof, out.challenges, w))
}

/// The folded carrier's verifier: recomputes the weights (the fold's W +
/// the fresh ρ′ for the transmitted slots) and the right-hand side.
#[allow(clippy::too_many_arguments)]
fn verify_carrier_folded(
    log_flat: usize,
    claims: &[BaseClaim],
    points: &[Vec<Goldilocks>],
    fold_weights: &[Goldilocks],
    s_bundle: Goldilocks,
    carrier: &sumcheck::SumcheckProof,
    w: &Goldilocks,
    transcript: &mut Transcript,
) -> Result<(Vec<Goldilocks>, Goldilocks), LedgerError> {
    let trans: Vec<bool> = claims
        .iter()
        .map(|c| matches!(c.factor, Factor::AddrCol { .. } | Factor::RvCol { .. }))
        .collect();
    let n_trans = trans.iter().filter(|&&t| t).count();
    let rhos = transcript
        .challenge_fields(b"trans-rho", n_trans)
        .map_err(LedgerError::Transcript)?;
    let mut weights = Vec::with_capacity(claims.len());
    let mut rho_iter = rhos.iter();
    let mut rhs = s_bundle;
    for (i, c) in claims.iter().enumerate() {
        let w = fold_weights
            .get(i)
            .copied()
            .ok_or_else(|| LedgerError::Layout("fold weight arity".into()))?;
        if trans[i] {
            let rho = rho_iter.next().ok_or(LedgerError::Layout("rho".into()))?;
            weights.push(w.add(rho));
            rhs = rhs.add(&rho.mul(&c.value));
        } else {
            weights.push(w);
        }
    }
    let verdict = carrier
        .verify(log_flat, 2, rhs, transcript, None)
        .map_err(LedgerError::Sumcheck)?;
    // E(r_sc) = Σ_i weight_i·eq(pt_i, r_sc); require E·w = final_claim.
    let mut e_r = Goldilocks::ZERO;
    for (i, pt) in points.iter().enumerate() {
        let eq_v = DenseMle::eq_eval(pt, &verdict.point).map_err(LedgerError::Mle)?;
        e_r = e_r.add(&weights[i].mul(&eq_v));
    }
    if e_r.mul(w) != verdict.final_claim {
        return Err(LedgerError::DerivedMismatch);
    }
    Ok((verdict.point, verdict.final_claim))
}

/// Verify the compact memory argument with NO re-execution: the same
/// public-table rebuild + leg replay as the Clear mode, with the carriers
/// and compact openings replacing the digit reveal.
pub fn verify_memory_argument_compact(
    proof: &CompactMemoryProof,
    program: &[u8],
    public_input: &[u8],
) -> Result<(), MemProofError> {
    // 1. Statement digests.
    let program_digest = Transcript::hash_domain(b"zkvm-program", program);
    let input_digest = Transcript::hash_domain(b"zkvm-public-input", public_input);
    let s = &proof.statement;
    if s.program_digest != program_digest || s.input_digest != input_digest {
        return Err(MemProofError::VerificationFailed);
    }

    // 2. Rebuild the instances' public structure (no execution).
    let t = 1usize << s.log_t;
    let log_ts = 2 + s.log_t;
    let k_ram = 1usize << s.ram_log_k;
    let k_fetch = 1usize << s.fetch_log_k;
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
    let limb = |v: u64, l: usize| fe((v >> (16 * l)) & 0xFFFF);
    let mut instances: Vec<MemoryInstance> = Vec::with_capacity(9);
    for l in 0..4usize {
        instances.push(MemoryInstance {
            log_k: 5,
            log_ts,
            addr: vec![0; 1 << log_ts],
            ractive: vec![0; 1 << log_ts],
            wactive: vec![0; 1 << log_ts],
            rv: vec![Goldilocks::ZERO; 1 << log_ts],
            wv: vec![Goldilocks::ZERO; 1 << log_ts],
            inc_off: vec![fe(INC_OFFSET); 1 << log_ts],
            init: vec![Goldilocks::ZERO; 32],
            final_state: s.final_regs.iter().map(|r| limb(*r, l)).collect(),
            table: None,
        });
    }
    for l in 0..4usize {
        instances.push(MemoryInstance {
            log_k: s.ram_log_k,
            log_ts,
            addr: vec![0; 1 << log_ts],
            ractive: vec![0; 1 << log_ts],
            wactive: vec![0; 1 << log_ts],
            rv: vec![Goldilocks::ZERO; 1 << log_ts],
            wv: vec![Goldilocks::ZERO; 1 << log_ts],
            inc_off: vec![fe(INC_OFFSET); 1 << log_ts],
            init: (0..k_ram as u64).map(|k| limb(init_image(k), l)).collect(),
            final_state: (0..k_ram)
                .map(|k| limb(s.final_memory.get(k).copied().unwrap_or(0), l))
                .collect(),
            table: None,
        });
    }
    {
        let table: Vec<Goldilocks> = (0..k_fetch)
            .map(|k| {
                let base = k * 4;
                let mut word = 0u32;
                for i in 0..4usize {
                    if base + i < program.len() {
                        word |= (program[base + i] as u32) << (8 * i);
                    }
                }
                fe(word as u64)
            })
            .collect();
        instances.push(MemoryInstance {
            log_k: s.fetch_log_k,
            log_ts: s.log_t,
            addr: vec![0; t],
            ractive: vec![1; t],
            wactive: vec![0; t],
            rv: vec![Goldilocks::ZERO; t],
            wv: vec![Goldilocks::ZERO; t],
            inc_off: vec![fe(INC_OFFSET); t],
            init: table.clone(),
            final_state: table.clone(),
            table: Some(table),
        });
    }

    // 3. Transcript replay (statement → compact commitments → legs).
    let mut transcript = Transcript::new_default(b"lzx-zkvm-memarg-compact");
    absorb_statement(s, &mut transcript)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"bits-commitment", &proof.bits_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"values-commitment", &proof.values_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;

    // 4. The deferred leg replay (NO values cross): the claimed vectors'
    let ctx = crate::legbatch::verify_legs_batched_folded(
        &instances,
        &proof.legs,
        &proof.fold.addr_claims,
        &proof.fold.rv_claims,
        &mut transcript,
    )
    .map_err(MemProofError::Memory)?;
    // The pre-leg values are absorbed AFTER the legs (the prover's
    // order: the legs' sumcheck totals pin them first).
    absorb_field_slice(&mut transcript, b"fold-addr", &proof.fold.addr_claims)?;
    absorb_field_slice(&mut transcript, b"fold-rv", &proof.fold.rv_claims)?;

    // 5. The claims fold: the derivation records the slots (the resolver)
    //    and the layer sumchecks verify the deferred expects.
    let mut slots = SlotLedger::new();
    let set = crate::legbatch::derive_claims_fold(
        &instances,
        &proof.legs,
        &ctx,
        |f, pt| slots.pop(f, pt).map_err(crate::memory::MemoryError::Ledger),
        &mut transcript,
    )
    .map_err(MemProofError::Memory)?;
    let num_slots = slots.slots().len();
    let binding = verify_claims_fold(&proof.fold, &set, num_slots, &mut transcript)
        .map_err(MemProofError::Fold)?;

    // The derived slot claims (factor + verifier-derived point, in slot
    // order) — the carriers' flat mapping.
    let mut derived: Vec<BaseClaim> = slots.to_claims();
    // The transmitted pre-leg values (addr/rv) fill their slots' values —
    // both in the derivation's pop order.
    {
        let mut addr_iter = proof.fold.addr_claims.iter();
        let mut rv_iter = proof.fold.rv_claims.iter();
        for c in derived.iter_mut() {
            match c.factor {
                Factor::AddrCol { .. } => {
                    if let Some(v) = addr_iter.next() {
                        c.value = *v;
                    }
                }
                Factor::RvCol { .. } => {
                    if let Some(v) = rv_iter.next() {
                        c.value = *v;
                    }
                }
                _ => {}
            }
        }
    }
    let bits_claims: Vec<BaseClaim> = derived
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = derived
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_slot_of: Vec<usize> = (0..derived.len())
        .filter(|&i| derived[i].factor.in_bits_bundle())
        .collect();
    let values_slot_of: Vec<usize> = (0..derived.len())
        .filter(|&i| !derived[i].factor.in_bits_bundle())
        .collect();
    // Rebuild the bundle geometries (deterministic from the instances).
    let bits_entries = bundle_entries(&instances, true);
    let values_entries = bundle_entries(&instances, false);
    let bits_factor_lens: Vec<usize> = bits_entries
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .collect();
    let values_factor_lens: Vec<usize> = values_entries
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .collect();
    if bits_factor_lens != proof.bits_factor_lens || values_factor_lens != proof.values_factor_lens
    {
        return Err(MemProofError::Shape);
    }
    let bits_flat_log = {
        let total: usize = bits_factor_lens.iter().sum();
        total.next_power_of_two().max(1).trailing_zeros() as usize
    };
    let values_flat_log = {
        let total: usize = values_factor_lens.iter().sum();
        total.next_power_of_two().max(1).trailing_zeros() as usize
    };

    // 6. The carriers with the fold-derived weights + the openings.
    let bits_refs: Vec<(Factor, &DenseMle)> = bits_entries.iter().map(|(f, m)| (*f, m)).collect();
    let values_refs: Vec<(Factor, &DenseMle)> =
        values_entries.iter().map(|(f, m)| (*f, m)).collect();
    let bits_pts = flat_points_for_claims_full(&bits_refs, &bits_claims, bits_flat_log)?;
    let values_pts = flat_points_for_claims_full(&values_refs, &values_claims, values_flat_log)?;
    let bits_weights: Vec<Goldilocks> = bits_slot_of.iter().map(|&i| binding.weights[i]).collect();
    let values_weights: Vec<Goldilocks> =
        values_slot_of.iter().map(|&i| binding.weights[i]).collect();
    let (bits_rsc, _) = verify_carrier_folded(
        bits_flat_log,
        &bits_claims,
        &bits_pts,
        &bits_weights,
        proof.fold.s_bits,
        &proof.bits_carrier,
        &proof.bits_w,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    let (values_rsc, _) = verify_carrier_folded(
        values_flat_log,
        &values_claims,
        &values_pts,
        &values_weights,
        proof.fold.s_vals,
        &proof.values_carrier,
        &proof.values_w,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;

    // 6. The compact openings.
    let seed = derive_seed(s);
    verify_compact_opening(
        seed,
        &proof.bits_commitment,
        &bits_factor_lens,
        bits_flat_log,
        &bits_rsc,
        &proof.bits_w,
        &proof.bits_opening,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    verify_compact_opening(
        seed,
        &proof.values_commitment,
        &values_factor_lens,
        values_flat_log,
        &values_rsc,
        &proof.values_w,
        &proof.values_opening,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The Sound memory argument (DESIGN_50KB Stage 5.2 — the LaBRADOR decider
// wired into the live memory-argument path): the same legs + carrier +
// compact-bundle structure, with the openings replaced by the Sound
// profile (the never-transmitted response + the width fold's [A₂ | −T]
// binding).
// ---------------------------------------------------------------------------

use crate::compact::{
    compact_bundle_commit_with_params, verify_sound_opening, FoldParams, SoundOpening,
};

/// The Sound memory-argument proof — **the Stage-5.2 folded form**: the
/// claims list is replaced by the fold exactly as in the compact mode.
#[derive(Clone, Debug)]
pub struct SoundMemoryProof {
    pub statement: MemoryStatement,
    /// The claims fold (the values list never appears).
    pub fold: ClaimsFoldProof,
    /// The Stage-4 batched legs (identical to the Clear/compact modes).
    pub legs: crate::legbatch::BatchedLegs,
    /// The per-bundle column commitments (r × k ring elements each).
    pub bits_commitment: Vec<u8>,
    pub values_commitment: Vec<u8>,
    /// The Goldilocks grouped carriers (the fold-derived weights).
    pub bits_carrier: sumcheck::SumcheckProof,
    pub values_carrier: sumcheck::SumcheckProof,
    /// f(r_sc) per bundle (the carrier terminal bound by the fold).
    pub bits_w: Goldilocks,
    pub values_w: Goldilocks,
    /// The Sound openings (the width fold replacing the response).
    pub bits_opening: SoundOpening,
    pub values_opening: SoundOpening,
    /// Factor lengths per bundle (for the verifier's layout rebuild).
    pub bits_factor_lens: Vec<usize>,
    pub values_factor_lens: Vec<usize>,
}

/// Prove the memory argument in the Sound profile: the binding is the
/// width fold's `[A₂ | −T]` instance (the estimator's sound row), not
/// the level-1 `[F̄ | −y]` (the Stage 5.1 broken regime at the compact
/// mode's response lengths).
///
/// Fail-closed when the bundle streams exceed the single-stage sound
/// regime's ceilings (the packing bound or the β₁ ceiling — the honest
/// boundary recorded in BENCHMARKS.md §2j; the follow-ups are the
/// recursive width-fold staging and the Modulus-50 class).
#[allow(clippy::too_many_arguments)]
pub fn prove_memory_argument_sound(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
    ram_log_k: usize,
    fetch_log_k: usize,
) -> Result<(SoundMemoryProof, [u64; 32]), MemProofError> {
    // 1–5. Execute, witness, instances, statement (the compact mode's
    //      construction, verbatim).
    let mut state = MachineState::new();
    state.load_program(0x1000, public_input);
    state.load_program(0, program);
    let rows = vm_run(&mut state, max_steps).map_err(MemProofError::Execution)?;
    let final_regs = state.regs;
    let (w, final_words) = build_cycle_witness(
        &rows,
        program,
        public_input,
        RamWindow { log_k: ram_log_k },
        FetchWindow { log_k: fetch_log_k },
    )
    .map_err(MemProofError::Witness)?;
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
    let final_memory: Vec<u64> = (0..(1usize << ram_log_k))
        .map(|k| {
            final_words
                .get(&(k as u64))
                .copied()
                .unwrap_or_else(|| init_image(k as u64))
        })
        .collect();
    let instances = build_instances(
        &w,
        &final_regs,
        &final_words,
        ram_log_k,
        fetch_log_k,
        program,
        public_input,
    );
    let program_digest = Transcript::hash_domain(b"zkvm-program", program);
    let input_digest = Transcript::hash_domain(b"zkvm-public-input", public_input);
    let statement = MemoryStatement {
        program_digest,
        input_digest,
        log_t: w.log_t,
        ram_log_k,
        fetch_log_k,
        final_regs,
        final_memory: final_memory.clone(),
    };
    let seed = derive_seed(&statement);
    let k_ram = 1usize << ram_log_k;
    let _ = k_ram;

    // 6. The compact bundle commitments under the SOUND profile's
    //    parameters (the amplitude the width fold's estimator ceiling
    //    allows; the column count that lands the response in the cheap
    //    sound row).
    let mut bits_factors_owned: Vec<(Factor, DenseMle)> = Vec::new();
    let mut values_factors_owned: Vec<(Factor, DenseMle)> = Vec::new();
    // The compact mode's GROUPED order (digits, then activities — the
    // digit tensors are `rows·t_s` while the activity cols are `t_s`, so
    // any interleaving breaks the flat-offset alignment the carrier's
    // point mapping relies on).
    for (i, m) in instances.iter().enumerate() {
        bits_factors_owned.push((Factor::DigitBits { inst: i }, m.digit_tensor()));
    }
    for (i, m) in instances.iter().enumerate() {
        bits_factors_owned.push((activity_factor(i, false), m.activity_col(false)));
        bits_factors_owned.push((activity_factor(i, true), m.activity_col(true)));
        values_factors_owned.push((addr_factor(i), m.addr_col()));
        values_factors_owned.push((rv_factor(i), m.rv_col()));
        if !m.read_only() {
            values_factors_owned.push((wv_factor(i), m.wv_col()));
            values_factors_owned.push((inc_factor(i), m.inc_col()));
        }
    }
    let bits_factors: Vec<(Factor, &DenseMle)> =
        bits_factors_owned.iter().map(|(f, m)| (*f, m)).collect();
    let values_factors: Vec<(Factor, &DenseMle)> =
        values_factors_owned.iter().map(|(f, m)| (*f, m)).collect();
    let bits_entries: Vec<(u32, &DenseMle)> = bits_factors
        .iter()
        .map(|(f, m)| (f.discriminant() as u32, *m))
        .collect();
    let values_entries: Vec<(u32, &DenseMle)> = values_factors
        .iter()
        .map(|(f, m)| (f.discriminant() as u32, *m))
        .collect();
    let bits_factor_lens: Vec<usize> = bits_factors
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .collect();
    let values_factor_lens: Vec<usize> = values_factors
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .collect();
    let bits_min_len = bits_factor_lens.iter().copied().min().unwrap_or(1);
    let values_min_len = values_factor_lens.iter().copied().min().unwrap_or(1);
    let bits_total: usize = bits_factor_lens.iter().sum();
    let values_total: usize = values_factor_lens.iter().sum();
    let (r_bits, k_bits) = crate::compact::sound_fold_params_for(bits_total, 1, bits_min_len)
        .map_err(|e| MemProofError::Ledger(LedgerError::Layout(format!("bits: {e}"))))?;
    let (r_vals, k_vals) =
        crate::compact::sound_fold_params_for(values_total, 3, values_min_len)
            .map_err(|e| MemProofError::Ledger(LedgerError::Layout(format!("values: {e}"))))?;
    let bits_prover = compact_bundle_commit_with_params(
        &bits_entries,
        seed,
        r_bits,
        k_bits,
        FoldParams::sound(r_bits, k_bits),
    )
    .map_err(MemProofError::Ledger)?;
    let values_prover = compact_bundle_commit_with_params(
        &values_entries,
        seed,
        r_vals,
        k_vals,
        FoldParams::sound(r_vals, k_vals),
    )
    .map_err(MemProofError::Ledger)?;
    let bits_commitment = bits_prover.commitment_bytes();
    let values_commitment = values_prover.commitment_bytes();

    let mut transcript = Transcript::new_default(b"lzx-zkvm-memarg-sound");
    absorb_statement(&statement, &mut transcript)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"bits-commitment", &bits_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"values-commitment", &values_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;

    // 7. The legs + the fold context (the values never leave the prover).
    let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
    table.extend(bits_factors.iter().copied());
    table.extend(values_factors.iter().copied());
    let mut ledger = Ledger::prover(table);
    let (all_legs, fold_ctx) =
        crate::legbatch::prove_legs_batched(&instances, &mut ledger, &mut transcript)
            .map_err(MemProofError::Memory)?;
    absorb_field_slice(&mut transcript, b"fold-addr", &fold_ctx.addr_claims)?;
    absorb_field_slice(&mut transcript, b"fold-rv", &fold_ctx.rv_claims)?;

    // 8. The claims fold (identical to the compact mode's).
    let slot_claims: Vec<BaseClaim> = ledger.slot_claims();
    let slot_values: Vec<Goldilocks> = slot_claims.iter().map(|c| c.value).collect();
    let slot_bits: Vec<bool> = slot_claims
        .iter()
        .map(|c| c.factor.in_bits_bundle())
        .collect();
    let set = crate::legbatch::derive_claims_fold(
        &instances,
        &all_legs,
        &fold_ctx,
        |f, pt| {
            ledger
                .claim_index(f, pt)
                .ok_or(crate::memory::MemoryError::ClaimMismatch)
        },
        &mut transcript,
    )
    .map_err(MemProofError::Memory)?;
    let (fold, binding) = prove_claims_fold(
        &set,
        &slot_values,
        &slot_bits,
        fold_ctx.addr_claims.clone(),
        fold_ctx.rv_claims.clone(),
        &mut transcript,
    )
    .map_err(MemProofError::Fold)?;

    // 9. The carriers with the fold-derived weights + the SOUND openings.
    let bits_flat = bits_prover.flat_mle();
    let values_flat = values_prover.flat_mle();
    let bits_refs: Vec<(Factor, &DenseMle)> = bits_factors.clone();
    let values_refs: Vec<(Factor, &DenseMle)> = values_factors.clone();
    let bits_claims: Vec<BaseClaim> = slot_claims
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = slot_claims
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_slot_of: Vec<usize> = (0..slot_claims.len())
        .filter(|&i| slot_claims[i].factor.in_bits_bundle())
        .collect();
    let values_slot_of: Vec<usize> = (0..slot_claims.len())
        .filter(|&i| !slot_claims[i].factor.in_bits_bundle())
        .collect();
    let bits_pts = flat_points_for_claims_full(&bits_refs, &bits_claims, bits_flat.num_vars)?;
    let values_pts =
        flat_points_for_claims_full(&values_refs, &values_claims, values_flat.num_vars)?;
    let bits_weights: Vec<Goldilocks> = bits_slot_of.iter().map(|&i| binding.weights[i]).collect();
    let values_weights: Vec<Goldilocks> =
        values_slot_of.iter().map(|&i| binding.weights[i]).collect();

    let (bits_carrier, bits_rsc, bits_w) = prove_carrier_folded(
        bits_flat,
        &bits_claims,
        &bits_pts,
        &bits_weights,
        fold.s_bits,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    let (values_carrier, values_rsc, values_w) = prove_carrier_folded(
        values_flat,
        &values_claims,
        &values_pts,
        &values_weights,
        fold.s_vals,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;

    let bits_opening = bits_prover
        .prove_sound_opening(&bits_rsc, &bits_w, &mut transcript)
        .map_err(MemProofError::Ledger)?;
    let values_opening = values_prover
        .prove_sound_opening(&values_rsc, &values_w, &mut transcript)
        .map_err(MemProofError::Ledger)?;

    Ok((
        SoundMemoryProof {
            statement,
            fold,
            legs: all_legs,
            bits_commitment,
            values_commitment,
            bits_carrier,
            values_carrier,
            bits_w,
            values_w,
            bits_opening,
            values_opening,
            bits_factor_lens: bits_entries
                .iter()
                .map(|(_, m)| m.evaluations.len())
                .collect(),
            values_factor_lens: values_entries
                .iter()
                .map(|(_, m)| m.evaluations.len())
                .collect(),
        },
        final_regs,
    ))
}

/// Verify the Sound memory argument: the same leg/carrier replay as the
/// compact mode, with the openings verified through the Sound profile
/// (the public-target computation + the width fold's `(W0)`–`(W4)` and
/// the estimator posture gate).
pub fn verify_memory_argument_sound(
    proof: &SoundMemoryProof,
    program: &[u8],
    public_input: &[u8],
) -> Result<(), MemProofError> {
    // The statement replay + instance reconstruction (the compact
    // verifier's construction, shared verbatim).
    let program_digest = Transcript::hash_domain(b"zkvm-program", program);
    let input_digest = Transcript::hash_domain(b"zkvm-public-input", public_input);
    let s = &proof.statement;
    if s.program_digest != program_digest || s.input_digest != input_digest {
        return Err(MemProofError::VerificationFailed);
    }
    let k_ram = 1usize << s.ram_log_k;
    let k_fetch = 1usize << s.fetch_log_k;
    let t = 1usize << s.log_t;
    let log_ts = 2 + s.log_t;
    let t_s = 1usize << log_ts;
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
    let limb = |v: u64, l: usize| fe((v >> (16 * l)) & 0xFFFF);
    let mut instances: Vec<MemoryInstance> = Vec::with_capacity(9);
    for l in 0..4usize {
        instances.push(MemoryInstance {
            log_k: 5,
            log_ts,
            addr: vec![0; t_s],
            ractive: vec![0; t_s],
            wactive: vec![0; t_s],
            rv: vec![Goldilocks::ZERO; t_s],
            wv: vec![Goldilocks::ZERO; t_s],
            inc_off: vec![fe(INC_OFFSET); t_s],
            init: vec![Goldilocks::ZERO; 32],
            final_state: s.final_regs.iter().map(|r| limb(*r, l)).collect(),
            table: None,
        });
    }
    for l in 0..4usize {
        instances.push(MemoryInstance {
            log_k: s.ram_log_k,
            log_ts,
            addr: vec![0; t_s],
            ractive: vec![0; t_s],
            wactive: vec![0; t_s],
            rv: vec![Goldilocks::ZERO; t_s],
            wv: vec![Goldilocks::ZERO; t_s],
            inc_off: vec![fe(INC_OFFSET); t_s],
            init: (0..k_ram as u64).map(|k| limb(init_image(k), l)).collect(),
            final_state: (0..k_ram)
                .map(|k| limb(s.final_memory.get(k).copied().unwrap_or(0), l))
                .collect(),
            table: None,
        });
    }
    {
        let table: Vec<Goldilocks> = (0..k_fetch)
            .map(|k| {
                let base = k * 4;
                let mut word = 0u32;
                for i in 0..4usize {
                    if base + i < program.len() {
                        word |= (program[base + i] as u32) << (8 * i);
                    }
                }
                fe(word as u64)
            })
            .collect();
        instances.push(MemoryInstance {
            log_k: s.fetch_log_k,
            log_ts: s.log_t,
            addr: vec![0; t],
            ractive: vec![1; t],
            wactive: vec![0; t],
            rv: vec![Goldilocks::ZERO; t],
            wv: vec![Goldilocks::ZERO; t],
            inc_off: vec![fe(INC_OFFSET); t],
            init: table.clone(),
            final_state: table.clone(),
            table: Some(table),
        });
    }

    // The transcript replay (the SOUND domain).
    let mut transcript = Transcript::new_default(b"lzx-zkvm-memarg-sound");
    absorb_statement(s, &mut transcript)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"bits-commitment", &proof.bits_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;
    transcript
        .append_bytes(b"values-commitment", &proof.values_commitment)
        .map_err(|e| MemProofError::Ledger(crate::ledger::LedgerError::Transcript(e)))?;

    // The deferred leg replay + the claims fold (the compact verifier's
    // flow, verbatim).
    let ctx = crate::legbatch::verify_legs_batched_folded(
        &instances,
        &proof.legs,
        &proof.fold.addr_claims,
        &proof.fold.rv_claims,
        &mut transcript,
    )
    .map_err(MemProofError::Memory)?;
    // The pre-leg values absorbed after the legs (the prover's order).
    absorb_field_slice(&mut transcript, b"fold-addr", &proof.fold.addr_claims)?;
    absorb_field_slice(&mut transcript, b"fold-rv", &proof.fold.rv_claims)?;
    let mut slots = SlotLedger::new();
    let set = crate::legbatch::derive_claims_fold(
        &instances,
        &proof.legs,
        &ctx,
        |f, pt| slots.pop(f, pt).map_err(crate::memory::MemoryError::Ledger),
        &mut transcript,
    )
    .map_err(MemProofError::Memory)?;
    let num_slots = slots.slots().len();
    let binding = verify_claims_fold(&proof.fold, &set, num_slots, &mut transcript)
        .map_err(MemProofError::Fold)?;
    let mut derived: Vec<BaseClaim> = slots.to_claims();
    // The transmitted pre-leg values (addr/rv) fill their slots' values —
    // both in the derivation's pop order.
    {
        let mut addr_iter = proof.fold.addr_claims.iter();
        let mut rv_iter = proof.fold.rv_claims.iter();
        for c in derived.iter_mut() {
            match c.factor {
                Factor::AddrCol { .. } => {
                    if let Some(v) = addr_iter.next() {
                        c.value = *v;
                    }
                }
                Factor::RvCol { .. } => {
                    if let Some(v) = rv_iter.next() {
                        c.value = *v;
                    }
                }
                _ => {}
            }
        }
    }
    let bits_claims: Vec<BaseClaim> = derived
        .iter()
        .filter(|c| c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let values_claims: Vec<BaseClaim> = derived
        .iter()
        .filter(|c| !c.factor.in_bits_bundle())
        .cloned()
        .collect();
    let bits_slot_of: Vec<usize> = (0..derived.len())
        .filter(|&i| derived[i].factor.in_bits_bundle())
        .collect();
    let values_slot_of: Vec<usize> = (0..derived.len())
        .filter(|&i| !derived[i].factor.in_bits_bundle())
        .collect();
    let bits_entries = bundle_entries(&instances, true);
    let values_entries = bundle_entries(&instances, false);
    let bits_factor_lens: Vec<usize> = bits_entries
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .collect();
    let values_factor_lens: Vec<usize> = values_entries
        .iter()
        .map(|(_, m)| m.evaluations.len())
        .collect();
    if bits_factor_lens != proof.bits_factor_lens || values_factor_lens != proof.values_factor_lens
    {
        return Err(MemProofError::Shape);
    }
    let bits_flat_log = {
        let total: usize = bits_factor_lens.iter().sum();
        total.next_power_of_two().max(1).trailing_zeros() as usize
    };
    let values_flat_log = {
        let total: usize = values_factor_lens.iter().sum();
        total.next_power_of_two().max(1).trailing_zeros() as usize
    };

    // The carriers (the fold-derived weights).
    let bits_refs: Vec<(Factor, &DenseMle)> = bits_entries.iter().map(|(f, m)| (*f, m)).collect();
    let values_refs: Vec<(Factor, &DenseMle)> =
        values_entries.iter().map(|(f, m)| (*f, m)).collect();
    let bits_pts = flat_points_for_claims_full(&bits_refs, &bits_claims, bits_flat_log)?;
    let values_pts = flat_points_for_claims_full(&values_refs, &values_claims, values_flat_log)?;
    let bits_weights: Vec<Goldilocks> = bits_slot_of.iter().map(|&i| binding.weights[i]).collect();
    let values_weights: Vec<Goldilocks> =
        values_slot_of.iter().map(|&i| binding.weights[i]).collect();
    let (bits_rsc, _) = verify_carrier_folded(
        bits_flat_log,
        &bits_claims,
        &bits_pts,
        &bits_weights,
        proof.fold.s_bits,
        &proof.bits_carrier,
        &proof.bits_w,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    let (values_rsc, _) = verify_carrier_folded(
        values_flat_log,
        &values_claims,
        &values_pts,
        &values_weights,
        proof.fold.s_vals,
        &proof.values_carrier,
        &proof.values_w,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;

    // The SOUND openings.
    let seed = derive_seed(s);
    verify_sound_opening(
        seed,
        &proof.bits_commitment,
        &bits_factor_lens,
        bits_flat_log,
        &bits_rsc,
        &proof.bits_w,
        &proof.bits_opening,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    verify_sound_opening(
        seed,
        &proof.values_commitment,
        &values_factor_lens,
        values_flat_log,
        &values_rsc,
        &proof.values_w,
        &proof.values_opening,
        &mut transcript,
    )
    .map_err(MemProofError::Ledger)?;
    Ok(())
}

/// The bundle entries (Factor, DenseMle) rebuilt on the verifier side —
/// the SAME order as the prover's (digit tensors, then activity columns;
/// stream columns for the values bundle). The MLE contents are
/// public-derivable shapes (the real values live in the committed
/// columns; only the geometry matters here).
fn bundle_entries(instances: &[MemoryInstance], bits: bool) -> Vec<(Factor, DenseMle)> {
    let mut out = Vec::new();
    if bits {
        for (i, m) in instances.iter().enumerate() {
            out.push((Factor::DigitBits { inst: i }, m.digit_tensor()));
        }
        for (i, m) in instances.iter().enumerate() {
            out.push((activity_factor(i, false), m.activity_col(false)));
            out.push((activity_factor(i, true), m.activity_col(true)));
        }
    } else {
        for (i, m) in instances.iter().enumerate() {
            out.push((addr_factor(i), m.addr_col()));
            out.push((rv_factor(i), m.rv_col()));
            if !m.read_only() {
                out.push((wv_factor(i), m.wv_col()));
                out.push((inc_factor(i), m.inc_col()));
            }
        }
    }
    out
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    fn fib_program() -> Vec<u8> {
        // A tiny hand-assembled straight-line arithmetic program (the
        // guest crate covers real programs; here 16 deterministic ops).
        let mut code = vec![
            0x93, 0x00, 0x10, 0x00, // addi x1, x0, 1
            0x13, 0x01, 0x20, 0x00, // addi x2, x0, 1
        ];
        for _ in 0..14 {
            code.extend_from_slice(&[0xB3, 0x01, 0x21, 0x00]); // add x3, x1, x2
        }
        code.extend_from_slice(&0x73u32.to_le_bytes()); // ecall (halt)
        code
    }

    #[test]
    fn compact_memproof_honest_and_tamper() {
        let program = fib_program();
        let input: Vec<u8> = vec![];
        // Prove.
        let (proof, final_regs) =
            prove_memory_argument_compact(&program, &input, 64, 4, 5).unwrap();
        // Verify (honest).
        verify_memory_argument_compact(&proof, &program, &input).unwrap();

        // Size accounting: the honest wire estimate.
        let mut bytes = 0usize;
        // The fold replaces the values-only claims list: the layers + the
        // claim pairs + the two S values + the 18 pre-leg entries.
        for layer in &proof.fold.layers {
            bytes += layer.rounds.len() * layer.rounds[0].len().max(1) * 8 + 16;
        }
        bytes += proof.fold.claims.len() * 2 * 8;
        bytes += 16; // s_bits + s_vals
        bytes += (proof.fold.addr_claims.len() + proof.fold.rv_claims.len()) * 8;
        for sc in proof.legs.sumchecks() {
            bytes += sc.rounds.len() * sc.rounds[0].len().max(1) * 8 + 16;
        }
        bytes += proof.bits_commitment.len();
        bytes += proof.values_commitment.len();
        for (carrier, _w) in [
            (&proof.bits_carrier, &proof.bits_w),
            (&proof.values_carrier, &proof.values_w),
        ] {
            bytes += carrier.rounds.iter().map(|r| r.len() * 8).sum::<usize>() + 16;
            bytes += 8; // w
        }
        for (op, lens) in [
            (&proof.bits_opening, &proof.bits_factor_lens),
            (&proof.values_opening, &proof.values_factor_lens),
        ] {
            bytes += op.u_tilde.len() * 8;
            bytes += op.response.hist.len() + op.response.payload.len() + op.response.raw.len();
            bytes += lens.len() + 16;
        }
        println!(
            "COMPACT PROOF SIZE: {} B = {:.1} KB",
            bytes,
            bytes as f64 / 1024.0
        );
        assert!(
            bytes < 300_000,
            "compact proof should be far under the Clear mode"
        );

        // ---- Tamper suite ----
        // Wrong final register: rejected.
        let mut bad = clone_proof(&proof);
        bad.statement.final_regs[1] = bad.statement.final_regs[1].wrapping_add(1);
        assert!(verify_memory_argument_compact(&bad, &program, &input).is_err());

        // Wrong program digest: rejected.
        let mut bad2 = clone_proof(&proof);
        bad2.statement.program_digest[0] ^= 0xFF;
        assert!(verify_memory_argument_compact(&bad2, &program, &input).is_err());

        // Tampered pre-leg value (the fold's transmitted addr entry):
        // rejected.
        let mut bad3 = clone_proof(&proof);
        if let Some(c) = bad3.fold.addr_claims.first_mut() {
            *c = c.add(&Goldilocks::from_u64(1));
        }
        assert!(verify_memory_argument_compact(&bad3, &program, &input).is_err());

        // Tampered fold layer round message: rejected.
        let mut bad3b = clone_proof(&proof);
        if let Some(r) = bad3b
            .fold
            .layers
            .first_mut()
            .and_then(|l| l.rounds.first_mut())
        {
            if let Some(x) = r.first_mut() {
                *x = x.add(&Goldilocks::from_u64(1));
            }
        }
        assert!(verify_memory_argument_compact(&bad3b, &program, &input).is_err());

        // Tampered fold leaf claim pair: rejected.
        let mut bad3c = clone_proof(&proof);
        if let Some(p) = bad3c.fold.claims.last_mut() {
            p[0] = p[0].add(&Goldilocks::from_u64(1));
        }
        assert!(verify_memory_argument_compact(&bad3c, &program, &input).is_err());

        // Tampered fold S value: rejected.
        let mut bad3d = clone_proof(&proof);
        bad3d.fold.s_bits = bad3d.fold.s_bits.add(&Goldilocks::from_u64(1));
        assert!(verify_memory_argument_compact(&bad3d, &program, &input).is_err());

        // Tampered carrier terminal w: rejected.
        let mut bad4 = clone_proof(&proof);
        bad4.bits_w = bad4.bits_w.add(&Goldilocks::from_u64(1));
        assert!(verify_memory_argument_compact(&bad4, &program, &input).is_err());

        // Tampered u_tilde: rejected.
        let mut bad5 = clone_proof(&proof);
        if let Some(u) = bad5.bits_opening.u_tilde.first_mut() {
            *u = u.add(&Goldilocks::from_u64(1));
        }
        assert!(verify_memory_argument_compact(&bad5, &program, &input).is_err());

        // Tampered response: rejected.
        let mut bad6 = clone_proof(&proof);
        if let Some(x) = bad6.values_opening.response.raw.first_mut() {
            *x ^= 0x40;
        }
        assert!(verify_memory_argument_compact(&bad6, &program, &input).is_err());

        // Tampered commitment: rejected.
        let mut bad7 = clone_proof(&proof);
        if bad7.bits_commitment.len() > 12 {
            bad7.bits_commitment[12] ^= 0xFF;
        }
        assert!(verify_memory_argument_compact(&bad7, &program, &input).is_err());

        // Tampered widths: rejected.
        let mut bad8 = clone_proof(&proof);
        if let Some(x) = bad8.values_opening.widths.first_mut() {
            *x = (*x + 1) % 9;
        }
        assert!(verify_memory_argument_compact(&bad8, &program, &input).is_err());

        let _ = final_regs;
    }

    fn clone_proof(p: &CompactMemoryProof) -> CompactMemoryProof {
        p.clone()
    }

    /// The Sound memory argument end-to-end: the LaBRADOR decider wired
    /// into the live path — the binding is the width fold's `[A₂ | −T]`
    /// instance, the level-1 response never transmitted.
    #[test]
    fn sound_memproof_honest_and_tamper() {
        let program = fib_program();
        let input: Vec<u8> = vec![];
        // Prove (fail-closed on the estimator floor at every bundle).
        let (proof, final_regs) = prove_memory_argument_sound(&program, &input, 64, 4, 5).unwrap();
        // Verify (honest).
        verify_memory_argument_sound(&proof, &program, &input).unwrap();

        // Size accounting: the Sound profile's honest price — the same
        // legs/carriers/commitments as the compact mode, with the
        // openings' response replaced by the width-fold artifact.
        let mut bytes = 0usize;
        for layer in &proof.fold.layers {
            bytes += layer.rounds.len() * layer.rounds[0].len().max(1) * 8 + 16;
        }
        bytes += proof.fold.claims.len() * 2 * 8 + 16;
        bytes += (proof.fold.addr_claims.len() + proof.fold.rv_claims.len()) * 8;
        for sc in proof.legs.sumchecks() {
            bytes += sc.rounds.len() * sc.rounds[0].len().max(1) * 8 + 16;
        }
        bytes += proof.bits_commitment.len();
        bytes += proof.values_commitment.len();
        for carrier in [&proof.bits_carrier, &proof.values_carrier] {
            bytes += carrier.rounds.iter().map(|r| r.len() * 8).sum::<usize>() + 16;
            bytes += 8;
        }
        for op in [&proof.bits_opening, &proof.values_opening] {
            bytes += op.u_tilde.len() * 8;
            for st in &op.width_proof.stages {
                bytes += st.p_images.len()
                    + st.garbage.len()
                    + st.t_inner.len()
                    + st.u_parts.len()
                    + st.g_func.len();
                bytes += st.response.hist.len() + st.response.payload.len() + st.response.raw.len();
                bytes += 64;
            }
        }
        println!(
            "SOUND PROOF SIZE: {} B = {:.1} KB",
            bytes,
            bytes as f64 / 1024.0
        );
        // The honest multiple of the compact mode's ~60 KB at this
        // scale (the quadratic garbage's price at small traces).
        assert!(bytes < 400_000);

        // ---- Tamper suite ----
        // Wrong final register: rejected.
        let mut bad = proof.clone();
        bad.statement.final_regs[1] = bad.statement.final_regs[1].wrapping_add(1);
        assert!(verify_memory_argument_sound(&bad, &program, &input).is_err());

        // Wrong program digest: rejected.
        let mut bad2 = proof.clone();
        bad2.statement.program_digest[0] ^= 0xFF;
        assert!(verify_memory_argument_sound(&bad2, &program, &input).is_err());

        // Tampered pre-leg value (the fold's transmitted rv entry):
        // rejected.
        let mut bad3 = proof.clone();
        if let Some(c) = bad3.fold.rv_claims.first_mut() {
            *c = c.add(&Goldilocks::from_u64(1));
        }
        assert!(verify_memory_argument_sound(&bad3, &program, &input).is_err());

        // Tampered fold S value: rejected.
        let mut bad3b = proof.clone();
        bad3b.fold.s_vals = bad3b.fold.s_vals.add(&Goldilocks::from_u64(1));
        assert!(verify_memory_argument_sound(&bad3b, &program, &input).is_err());

        // Tampered carrier terminal w: rejected.
        let mut bad4 = proof.clone();
        bad4.bits_w = bad4.bits_w.add(&Goldilocks::from_u64(1));
        assert!(verify_memory_argument_sound(&bad4, &program, &input).is_err());

        // Tampered u_tilde (the level-1 public layer): rejected.
        let mut bad5 = proof.clone();
        if let Some(u) = bad5.bits_opening.u_tilde.first_mut() {
            *u = u.add(&Goldilocks::from_u64(1));
        }
        assert!(verify_memory_argument_sound(&bad5, &program, &input).is_err());

        // Tampered width-chain response (z): W1/W2 reject.
        let mut bad6 = proof.clone();
        {
            use crate::compact::{decode_response, encode_response};
            let stages = &mut bad6.values_opening.width_proof.stages;
            let last = stages.len() - 1;
            let mut coeffs = decode_response(&stages[last].response).unwrap();
            assert!(!coeffs.is_empty());
            coeffs[0] = coeffs[0].wrapping_add(1);
            stages[last].response = encode_response(&coeffs).unwrap();
        }
        assert!(verify_memory_argument_sound(&bad6, &program, &input).is_err());

        // Tampered quadratic garbage: the exact fold identity rejects.
        let mut bad_g = proof.clone();
        assert!(!bad_g.bits_opening.width_proof.stages[0].garbage.is_empty());
        bad_g.bits_opening.width_proof.stages[0].garbage[3] ^= 0x20;
        assert!(verify_memory_argument_sound(&bad_g, &program, &input).is_err());

        // Tampered inner commitments: the MSIS binding rejects.
        let mut bad_t = proof.clone();
        assert!(!bad_t.values_opening.width_proof.stages[0]
            .t_inner
            .is_empty());
        bad_t.values_opening.width_proof.stages[0].t_inner[7] ^= 0x08;
        assert!(verify_memory_argument_sound(&bad_t, &program, &input).is_err());

        // Tampered commitment: the W0 target chain rejects.
        let mut bad7 = proof.clone();
        if bad7.bits_commitment.len() > 12 {
            bad7.bits_commitment[12] ^= 0xFF;
        }
        assert!(verify_memory_argument_sound(&bad7, &program, &input).is_err());

        let _ = final_regs;
    }
}
