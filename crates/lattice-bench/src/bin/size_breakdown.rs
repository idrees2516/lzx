//! Size-breakdown analysis for the zkVM memory-argument proof: exact byte
//! accounting per component, plus the shape parameters the 50 KB pipeline
//! design (docs/DESIGN_50KB.md) needs.

use lattice_guest::asm::{run_on_vm, AssembledProgram};
use lattice_guest::programs::suite;
use lattice_zkvm::memproof::{prove_memory_argument_compact, CompactMemoryProof};

fn main() {
    let programs = match suite() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("suite build failed: {e:?}");
            return;
        }
    };
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
        if run.steps > 4096 {
            println!("== {}: {} cycles (skip: dense cap) ==\n", prog.name, run.steps);
            continue;
        }
        let fetch_words = prog.image.len().div_ceil(4);
        let fetch_log_k = fetch_words.next_power_of_two().max(1).trailing_zeros() as usize;
        let mut proof = None;
        let mut ram_used = 4usize;
        for ram_log_k in 4..=8usize {
            if let Ok((p, _used)) = prove_memory_argument_compact(
                &prog.image,
                &prog.public_input,
                1 << 22,
                ram_log_k,
                fetch_log_k,
            ) {
                proof = Some(p);
                ram_used = ram_log_k;
                break;
            }
        }
        let Some(proof) = proof else { continue };
        println!(
            "== {}: {} cycles, ram_log_k={}, fetch_log_k={} (COMPACT) ==",
            prog.name,
            run.steps,
            ram_used,
            fetch_log_k
        );
        breakdown_compact(&proof);
    }
}

fn breakdown_compact(proof: &CompactMemoryProof) {
    let mut claims_bytes = 0usize;
    let mut n_claims = 0usize;
    for _c in &proof.claims {
        claims_bytes += 1 + 1 + 8; // values-only
        n_claims += 1;
    }
    let mut legs_bytes = 0usize;
    let mut n_legs = 0usize;
    for sc in proof.legs.sumchecks() {
        legs_bytes += sc.rounds.len() * sc.rounds[0].len().max(1) * 8 + 16;
        n_legs += 1;
    }
    // The transmitted batched-claim vectors (the batches' input claims).
    legs_bytes += (proof.legs.ra_claims.len()
        + proof.legs.val_read_claims.len()
        + proof.legs.u_read_claims.len()
        + proof.legs.wa_claims.len()
        + proof.legs.inc_w_claims.len()
        + proof.legs.val_write_claims.len()
        + proof.legs.u_write_claims.len()
        + proof.legs.inc_tel_claims.len())
        * 8;
    let bits_c = proof.bits_commitment.len();
    let vals_c = proof.values_commitment.len();
    let mut carrier_bytes = 0usize;
    for carrier in [&proof.bits_carrier, &proof.values_carrier] {
        carrier_bytes += carrier.rounds.iter().map(|r| r.len() * 8).sum::<usize>() + 16;
    }
    let mut opening_bytes = 0usize;
    for op in [&proof.bits_opening, &proof.values_opening] {
        opening_bytes += op.u_tilde.len() * 8;
        opening_bytes += op.response.hist.len() + op.response.payload.len() + op.response.raw.len();
        opening_bytes += 16;
    }
    let stmt_bytes = 32 + 32 + 24 + proof.statement.final_regs.len() * 8
        + proof.statement.final_memory.len() * 8;
    let total = stmt_bytes + claims_bytes + legs_bytes + bits_c + vals_c + carrier_bytes
        + opening_bytes + 16 + 16;
    println!("   claims: {} claims, {} bytes", n_claims, claims_bytes);
    println!("   legs: {} legs, {} bytes", n_legs, legs_bytes);
    println!(
        "   bits commit {} B, values commit {} B, carriers {} B, openings {} B",
        bits_c, vals_c, carrier_bytes, opening_bytes
    );
    println!(
        "   statement {} B | TOTAL: {} B = {:.1} KB",
        stmt_bytes,
        total,
        total as f64 / 1024.0
    );
    println!();
}
