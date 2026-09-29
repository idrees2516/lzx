//! The zkVM end-to-end benchmark: Jolt-style guest programs measured for
//! execution cycles, memory-argument proof generation, verification, and
//! proof size (see docs/BENCHMARKS.md for the SOTA comparison context).

use std::time::Instant;

use lattice_guest::asm::{run_on_vm, AssembledProgram};
use lattice_guest::programs::{suite, GuestProgram};
use lattice_zkvm::memproof::{prove_memory_argument, verify_memory_argument};

/// Programs above this cycle count skip the memory-argument proof (the
/// dense prover materializes K×T_s matrices; the sparse prover is the
/// documented next wave).
const PROVE_CYCLE_CAP: u64 = 4096;

fn main() {
    let programs = match suite() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("suite build failed: {e:?}");
            return;
        }
    };
    println!("| program | cycles | prove (ms) | verify (ms) | proof (KB) | proved |");
    println!("|---|---|---|---|---|---|");
    for prog in &programs {
        let asm_prog = AssembledProgram {
            code: prog.image.clone(),
            data: vec![],
            labels: Default::default(),
            data_base: prog.data_base,
        };
        let run = match run_on_vm(&asm_prog, &prog.public_input, 5_000_000) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: execution failed {e:?}", prog.name);
                continue;
            }
        };
        let cycles = run.steps;
        let (prove_ms, verify_ms, proof_kb, proved) = if cycles <= PROVE_CYCLE_CAP {
            match measure_memory_argument(prog) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("{}: memory-argument failed: {e:?}", prog.name);
                    (String::from("err"), String::from("err"), 0.0, false)
                }
            }
        } else {
            (String::from("-"), String::from("-"), 0.0, false)
        };
        println!(
            "| {} | {} | {} | {} | {:.1} | {} |",
            prog.name, cycles, prove_ms, verify_ms, proof_kb, proved
        );
    }
}

fn measure_memory_argument(
    prog: &GuestProgram,
) -> Result<(String, String, f64, bool), String> {
    // Adaptive RAM window: the smallest power-of-two word window whose
    // prove does not fail closed on an out-of-window address.
    let fetch_words = prog.image.len().div_ceil(4);
    let fetch_log_k = fetch_words.next_power_of_two().max(1).trailing_zeros() as usize;
    let t0 = Instant::now();
    let mut proof = None;
    // The dense prover materializes K x T_s matrices; cap the window at
    // 256 words so K*T_s stays tractable (the sparse prover is the next
    // wave — see docs/BENCHMARKS.md).
    for ram_log_k in 4..=8usize {
        if let Ok((p, _)) = prove_memory_argument(
            &prog.image,
            &prog.public_input,
            1 << 22,
            ram_log_k,
            fetch_log_k,
        ) {
            proof = Some(p);
            break;
        }
    }
    let proof = proof.ok_or_else(|| "no window fits".to_string())?;
    let prove_ms = t0.elapsed().as_millis();
    let t1 = Instant::now();
    verify_memory_argument(&proof, &prog.image, &prog.public_input)
        .map_err(|e| format!("verify: {e:?}"))?;
    let verify_ms = t1.elapsed().as_millis();
    let proof_kb = proof_size_kb(&proof) as f64;
    Ok((prove_ms.to_string(), verify_ms.to_string(), proof_kb, true))
}

fn proof_size_kb(proof: &lattice_zkvm::memproof::MemoryArgumentProof) -> usize {
    // The serialized size: claims + legs + commitments + openings.
    let mut bytes = 0usize;
    for c in &proof.claims {
        bytes += 1 + 8 + c.point.len() * 8 + 8;
    }
    for inst in &proof.legs {
        for leg in &inst.legs {
            bytes += leg.sc.rounds.len() * leg.sc.rounds[0].len().max(1) * 8 + 16;
        }
    }
    bytes += proof.bits_commitment.len() + proof.values_commitment.len();
    bytes += proof.bits_opening.digits.len() * 2 + 64;
    bytes += proof.values_opening.digits.len() * 2 + 64;
    bytes / 1024
}
