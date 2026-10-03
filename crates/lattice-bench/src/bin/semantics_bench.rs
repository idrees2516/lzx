//! The instruction-semantics stage benchmark (BENCHMARKS.md §2d):
//! per-phase prover/verifier attribution at scale for the semantics
//! layer (`semantics.rs`), which the family-completion wave wired into
//! the live pipeline. Measures: execution, witness build, aux build,
//! bundle commits, the constraint families (per-family with
//! `LZX_SEM_TIMING=1`), the grouped-carrier openings; and on the verify
//! side: the constraint replay, the seeded key regeneration, and the
//! bundle-opening checks (the carrier factoring targets).
//!
//! Run: `cargo run --release -p lattice-bench --bin semantics_bench --
//! [log_t ...]` (default scales: 6 8 10 12).

use std::time::Instant;

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_guest::asm::Assembler;
use lattice_vm::{run as vm_run, MachineState};
use lattice_zkvm::columns::{build_cycle_witness, FetchWindow, RamWindow};
use lattice_zkvm::constraints::{aux_shape, build_aux, prove_constraints, verify_constraints};
use lattice_zkvm::ledger::{
    bits_bundle_commit, values_bundle_commit, verify_bundle_opening, BaseClaim,
    BundleLayoutEntry, BundleProver, Factor, Ledger, BITS_NORM_BOUND, VALUES_NORM_BOUND,
};
use lattice_zkvm::semantics::{
    prove_instruction_semantics, verify_instruction_semantics, SemanticsProof, SemanticsStatement,
};
use lattice_vm::decode::Instr;

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

/// A scalable mixed-instruction loop: every covered family appears in
/// the trace (ALU/shift/MUL/DIV/bitwise/branch/memory is exercised by
/// the route/decode legs regardless).
fn mixed_loop_program(iters: u64) -> Vec<u8> {
    let mut a = Assembler::new();
    let x1 = 1u8;
    let x2 = 2u8;
    let x3 = 3u8;
    let x4 = 4u8;
    let x5 = 5u8;
    let x6 = 6u8;
    let x7 = 7u8;
    a.li(x1, iters as i64).expect("li");
    a.li(x2, 1).expect("li");
    a.li(x3, 3).expect("li");
    let lp = a.label("lp");
    a.bind(lp).expect("bind");
    a.addi(x2, x2, 5).expect("addi");
    a.addi(x3, x3, 7).expect("addi");
    a.mul(x4, x2, x3).expect("mul");
    a.divu(x5, x4, x3).expect("divu");
    a.sll(x6, x2, x3).expect("sll");
    a.xor(x7, x6, x2).expect("xor");
    a.addi(x1, x1, -1).expect("addi");
    a.bne(x1, 0, lp.into()).expect("bne");
    a.ecall().expect("ecall");
    a.finish().expect("finish").code
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn kb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

fn proof_bytes(p: &SemanticsProof) -> usize {
    let mut sz = 0usize;
    for c in &p.claims {
        sz += 1 + 1 + c.point.len() * 8 + 8;
    }
    for l in &p.legs {
        sz += 8 + l.name.len();
        for r in &l.sc.rounds {
            sz += r.len() * 8;
        }
    }
    sz += p.bits_commitment.len() + p.values_commitment.len();
    for o in [&p.bits_opening, &p.values_opening] {
        for r in &o.carrier.rounds {
            sz += r.len() * 8;
        }
        sz += o.digits.len() * 2 + 16;
    }
    sz
}

/// The verifier-side layout reconstruction (mirrors semantics.rs).
fn layout_entries(
    shape: &lattice_zkvm::constraints::AuxCols,
    log_t: usize,
) -> (Vec<BundleLayoutEntry>, Vec<BundleLayoutEntry>) {
    let mut bits = Vec::new();
    let mut off = 0usize;
    for slot in 0..lattice_zkvm::columns::VALUE_TENSORS {
        let n = 64usize << log_t;
        bits.push(BundleLayoutEntry {
            factor: Factor::ValueBits { slot },
            num_vars: 6 + log_t,
            offset: off,
        });
        off += n;
    }
    bits.push(BundleLayoutEntry {
        factor: Factor::InstrBits,
        num_vars: 5 + log_t,
        offset: off,
    });
    off += 32usize << log_t;
    for id in 0..shape.bits.len() {
        bits.push(BundleLayoutEntry {
            factor: Factor::BitCol { id },
            num_vars: log_t,
            offset: off,
        });
        off += 1usize << log_t;
    }
    let mut vals = Vec::new();
    let mut voff = 0usize;
    for id in 0..shape.vals.len() {
        vals.push(BundleLayoutEntry {
            factor: Factor::ValCol { id },
            num_vars: log_t,
            offset: voff,
        });
        voff += 1usize << log_t;
    }
    (bits, vals)
}

fn bundle_pk(
    ring: &lattice_ring::RingConfig,
    layout: &[BundleLayoutEntry],
    is_bits: bool,
    seed: [u8; 32],
) -> Result<AjtaiPublicKey, String> {
    let flat_len: usize = layout
        .iter()
        .map(|e| 1usize << e.num_vars)
        .sum::<usize>()
        .next_power_of_two()
        .max(1);
    let ring_n = ring.n();
    let m = if is_bits {
        flat_len.div_ceil(31 * ring_n).max(1)
    } else {
        (flat_len * 3).div_ceil(ring_n).max(1)
    };
    let bound = if is_bits {
        BITS_NORM_BOUND
    } else {
        VALUES_NORM_BOUND
    };
    let params = AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m,
        norm_bound: bound,
    };
    AjtaiPublicKey::from_seed(params, seed).map_err(|e| format!("{e:?}"))
}

fn main() {
    let scales: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse::<usize>().ok())
        .collect();
    let scales = if scales.is_empty() {
        vec![6, 8, 10, 12]
    } else {
        scales
    };

    println!(
        "| log_t | cycles | exec | witness | aux | commit | families | openings | prove total | replay | pk | openings | verify total | ms/cycle | proof KB | claims |"
    );
    println!(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    );

    for log_t_target in scales {
        // 8 instructions per iteration + ~3 setup + ecall.
        let iters = ((1usize << log_t_target).saturating_sub(16) / 8).max(1) as u64;
        let prog = mixed_loop_program(iters);
        let input: Vec<u8> = vec![];

        // Warm: execute to learn the true cycle count (log_t is derived).
        let mut state = MachineState::new();
        state.load_program(0, &prog);
        let t_exec = Instant::now();
        let rows = vm_run(&mut state, 10_000_000).expect("exec");
        let exec_ms = ms(t_exec);
        let log_t = (rows.len().next_power_of_two().max(2)).trailing_zeros() as usize;
        let cycles = rows.len();

        let ram_log_k = 6usize.min(log_t);
        let fetch_log_k = 4usize.min(log_t);

        // ---- The real end-to-end prove (totals + proof object). ----
        let t0 = Instant::now();
        let (proof, _regs) =
            prove_instruction_semantics(&prog, &input, 10_000_000, ram_log_k, fetch_log_k)
                .expect("prove");
        let prove_total = ms(t0);

        // ---- Phase-attributed replication of the prover (same calls). ----
        let t_w = Instant::now();
        let (w, _fw) = build_cycle_witness(
            &rows,
            &prog,
            &input,
            RamWindow { log_k: ram_log_k },
            FetchWindow { log_k: fetch_log_k },
        )
        .expect("witness");
        let witness_ms = ms(t_w);
        let instrs: Vec<Instr> = rows.iter().map(|r| r.instr).collect();
        let t_aux = Instant::now();
        let aux = build_aux(&w, &instrs).expect("aux");
        let aux_ms = ms(t_aux);

        let statement = SemanticsStatement {
            program_digest: Transcript::hash_domain(b"zkvm-program", &prog),
            input_digest: Transcript::hash_domain(b"zkvm-public-input", &input),
            log_t: w.log_t,
            ram_log_k,
            fetch_log_k,
            final_regs: proof.statement.final_regs,
        };

        let t_commit = Instant::now();
        let (bits_entries, values_entries) = {
            let mut bits: Vec<(Factor, DenseMle)> = Vec::new();
            for slot in 0..lattice_zkvm::columns::VALUE_TENSORS {
                bits.push((Factor::ValueBits { slot }, w.values[slot].clone()));
            }
            bits.push((Factor::InstrBits, w.instr_bits.clone()));
            for (id, col) in aux.bits.iter().enumerate() {
                bits.push((
                    Factor::BitCol { id },
                    DenseMle {
                        num_vars: w.log_t,
                        evaluations: col.iter().map(|v| fe(*v as u64)).collect(),
                    },
                ));
            }
            let mut vals: Vec<(Factor, DenseMle)> = Vec::new();
            for (id, col) in aux.vals.iter().enumerate() {
                vals.push((
                    Factor::ValCol { id },
                    DenseMle {
                        num_vars: w.log_t,
                        evaluations: col.clone(),
                    },
                ));
            }
            (bits, vals)
        };
        // Re-derive the statement seed exactly as the prover does.
        let mut seed_t = Transcript::new_default(b"lzx-semantics-seed");
        let _ = seed_t.append_bytes(b"prog", &statement.program_digest);
        let _ = seed_t.append_bytes(b"in", &statement.input_digest);
        let meta_bytes: Vec<u8> = [
            statement.log_t as u64,
            statement.ram_log_k as u64,
            statement.fetch_log_k as u64,
        ]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
        let _ = seed_t.append_bytes(b"meta", &meta_bytes);
        let mut seed = [0u8; 32];
        if let Ok(b) = seed_t.challenge_bytes(b"seed", 32) {
            seed.copy_from_slice(&b);
        }
        let bits_prover: BundleProver = bits_bundle_commit(&bits_entries, seed).expect("bits");
        let values_prover: BundleProver =
            values_bundle_commit(&values_entries, seed).expect("values");
        let commit_ms = ms(t_commit);

        let t_fam = Instant::now();
        let mut table: Vec<(Factor, &DenseMle)> = Vec::new();
        for (f, m) in bits_entries.iter() {
            table.push((*f, m));
        }
        for (f, m) in values_entries.iter() {
            table.push((*f, m));
        }
        let mut ledger = Ledger::prover(table);
        let mut legs = Vec::new();
        let mut tr = Transcript::new_default(b"lzx-zkvm-semantics");
        // absorb the statement (same discipline as semantics.rs)
        {
            let mut meta = vec![
                statement.log_t as u64,
                statement.ram_log_k as u64,
                statement.fetch_log_k as u64,
            ];
            meta.extend(statement.final_regs.iter().copied());
            let fields: Vec<Goldilocks> = meta.iter().map(|v| fe(*v)).collect();
            tr.append_bytes(b"sem-prog", &statement.program_digest)
                .expect("absorb");
            tr.append_bytes(b"sem-in", &statement.input_digest)
                .expect("absorb");
            tr.append_field_slice(b"sem-meta", &fields).expect("absorb");
        }
        tr.append_bytes(b"sem-bits-commitment", &bits_prover.commitment.to_bytes())
            .expect("absorb");
        tr.append_bytes(
            b"sem-values-commitment",
            &values_prover.commitment.to_bytes(),
        )
        .expect("absorb");
        prove_constraints(&w, &aux, &instrs, &mut ledger, &mut legs, &mut tr).expect("families");
        let families_ms = ms(t_fam);

        let t_open = Instant::now();
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
        let _ = bits_prover.prove_opening(&bits_claims, &mut tr).expect("open");
        let _ = values_prover
            .prove_opening(&values_claims, &mut tr)
            .expect("open");
        let openings_ms = ms(t_open);

        // ---- The real end-to-end verify (total). ----
        let t1 = Instant::now();
        verify_instruction_semantics(&proof, &prog, &input).expect("verify");
        let verify_total = ms(t1);

        // ---- Phase-attributed replication of the verifier. ----
        let t_shape = Instant::now();
        let shape = aux_shape(proof.statement.log_t).expect("shape");
        let (bits_layout, values_layout) = layout_entries(&shape, proof.statement.log_t);
        let shape_ms = ms(t_shape);

        let t_replay = Instant::now();
        let vclaims = proof.claims.clone();
        let mut vtr = Transcript::new_default(b"lzx-zkvm-semantics");
        {
            let mut meta = vec![
                proof.statement.log_t as u64,
                proof.statement.ram_log_k as u64,
                proof.statement.fetch_log_k as u64,
            ];
            meta.extend(proof.statement.final_regs.iter().copied());
            let fields: Vec<Goldilocks> = meta.iter().map(|v| fe(*v)).collect();
            vtr.append_bytes(b"sem-prog", &proof.statement.program_digest)
                .expect("absorb");
            vtr.append_bytes(b"sem-in", &proof.statement.input_digest)
                .expect("absorb");
            vtr.append_field_slice(b"sem-meta", &fields).expect("absorb");
        }
        vtr.append_bytes(b"sem-bits-commitment", &proof.bits_commitment)
            .expect("absorb");
        vtr.append_bytes(b"sem-values-commitment", &proof.values_commitment)
            .expect("absorb");
        let mut vledger = Ledger::verifier(vclaims.clone());
        verify_constraints(&shape, &proof.legs, &mut vledger, &mut vtr).expect("replay");
        let replay_ms = ms(t_replay);

        let t_pk = Instant::now();
        let ring = lattice_zkvm::ledger::bundle_ring().expect("ring");
        let bits_pk = bundle_pk(&ring, &bits_layout, true, seed).expect("pk");
        let values_pk = bundle_pk(&ring, &values_layout, false, seed).expect("pk");
        let pk_ms = ms(t_pk);

        let t_chk = Instant::now();
        let bits_comm = AjtaiCommitment::from_bytes(
            &ring,
            bits_pk.params.k,
            &proof.bits_commitment,
        )
        .expect("comm");
        let values_comm = AjtaiCommitment::from_bytes(
            &ring,
            values_pk.params.k,
            &proof.values_commitment,
        )
        .expect("comm");
        let bcl: Vec<BaseClaim> = vclaims
            .iter()
            .filter(|c| c.factor.in_bits_bundle())
            .cloned()
            .collect();
        let vcl: Vec<BaseClaim> = vclaims
            .iter()
            .filter(|c| !c.factor.in_bits_bundle())
            .cloned()
            .collect();
        verify_bundle_opening(
            &bits_pk,
            &bits_comm,
            &bits_layout,
            &bcl,
            &proof.bits_opening,
            true,
            &mut vtr,
        )
        .expect("bits opening");
        verify_bundle_opening(
            &values_pk,
            &values_comm,
            &values_layout,
            &vcl,
            &proof.values_opening,
            false,
            &mut vtr,
        )
        .expect("values opening");
        let chk_ms = ms(t_chk);

        let n_claims = proof.claims.len();
        // Claim histogram by factor type.
        let mut hist: std::collections::BTreeMap<String, (usize, usize)> =
            std::collections::BTreeMap::new();
        for c in &proof.claims {
            let key = match c.factor {
                Factor::ValueBits { .. } => "ValueBits",
                Factor::InstrBits => "InstrBits",
                Factor::BitCol { .. } => "BitCol",
                Factor::ValCol { .. } => "ValCol",
                Factor::DigitBits { .. } => "DigitBits",
                Factor::RvCol { .. } => "RvCol",
                Factor::WvCol { .. } => "WvCol",
                Factor::IncCol { .. } => "IncCol",
                Factor::AddrCol { .. } => "AddrCol",
                Factor::ActiveCol { .. } => "ActiveCol",
            };
            let e = hist.entry(key.to_string()).or_insert((0, 0));
            e.0 += 1;
            e.1 += c.point.len();
        }
        eprintln!("  [claims] {n_claims} total: {hist:?}");
        let sz = kb(proof_bytes(&proof));
        let per_cycle = prove_total / cycles as f64;
        println!(
            "| {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.1} | {:.1} | {} |",
            log_t,
            cycles,
            exec_ms,
            witness_ms,
            aux_ms,
            commit_ms,
            families_ms,
            openings_ms,
            prove_total,
            replay_ms + shape_ms,
            pk_ms,
            chk_ms,
            verify_total,
            per_cycle,
            sz,
            n_claims
        );
        eprintln!(
            "  [log_t={log_t}] verify detail: shape {shape_ms:.0} replay {replay_ms:.0} pk-derive {pk_ms:.0} openings {chk_ms:.0} | m_bits={} m_vals={} flat_bits_log={} claims={}",
            bits_pk.params.m,
            values_pk.params.m,
            bits_layout
                .iter()
                .map(|e| 1usize << e.num_vars)
                .sum::<usize>()
                .next_power_of_two()
                .trailing_zeros(),
            n_claims
        );
    }
}
