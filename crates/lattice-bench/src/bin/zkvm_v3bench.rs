//! The v3 zkVM benchmark: the FULL sparse-native pipeline (memory
//! arguments + instruction-semantics AIR + lookups) over the guest suite,
//! with NO cycle cap — the sparse engine and the per-column commitments
//! scale with T, and the P0-63 profile fail-closes on out-of-profile
//! programs (reported honestly).

use std::time::Instant;

use lattice_guest::asm::{run_on_vm, AssembledProgram};
use lattice_guest::programs::suite;
use lattice_zkvm::pipeline4::{prove_v3, verify_v3, v3_pcs_for};

fn main() {
    // A profile-compliant scale loop: x1 = n; repeat { x1 -= 1; bne x1, x0, loop }.
    // Values stay tiny (the P0-63 profile holds at any n).
    for n in [60u32] {
        let enc_i = |rd: u32, rs1: u32, imm: i32| -> u32 {
            (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
        };
        let mut w: Vec<u32> = Vec::new();
        w.push(enc_i(1, 0, n as i32));
        let loop_start = 1i32;
        w.push(enc_i(1, 1, -1));
        let off = loop_start * 4 - (w.len() as i32) * 4;
        w.push(((off as u32 >> 31) & 1) << 31
            | (((off as u32 >> 7) & 1) << 7)
            | (((off as u32 >> 25) & 0x3f) << 25)
            | (((off as u32 >> 1) & 0xf) << 8)
            | (0 << 20) | (1 << 15) | (1 << 12) | 0x63);
        w.push(0x6f); // jal x0, 0 (halt)
        let program: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let cycles = (n as usize) * 2 + 1;
        let t0 = Instant::now();
        let log_t = cycles.next_power_of_two().max(2).trailing_zeros() as usize;
        let pcs = match v3_pcs_for(log_t, 6, [91u8; 32]) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("scale n={n}: pcs error {e:?}");
                continue;
            }
        };
        match prove_v3(&pcs, &program, &[], 10_000_000) {
            Ok((state, proof)) => {
                let prove_ms = t0.elapsed().as_millis();
                let t1 = Instant::now();
                let ok = verify_v3(&pcs, &program, &[], &state, &proof, 10_000_000).is_ok();
                let verify_ms = t1.elapsed().as_millis();
                let kb = proof_size_kb(&proof);
                let thr = (cycles as f64) / (prove_ms as f64 / 1000.0);
                println!(
                    "| scale_loop(n={n}) | {} | {} | {} | {:.1} | {:.0} | {} |",
                    cycles, prove_ms, verify_ms, kb, thr,
                    if ok { "verified" } else { "VERIFY FAILED" }
                );
            }
            Err(e) => println!("| scale_loop(n={n}) | {cycles} | - | - | - | - | {e:?} |"),
        }
    }
    let programs = match suite() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("suite build failed: {e:?}");
            return;
        }
    };
    println!("| program | cycles | prove (ms) | verify (ms) | proof (KB) | cycles/s | status |");
    println!("|---|---|---|---|---|---|---|");
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
        let t0 = Instant::now();
        let log_t = (cycles as usize).next_power_of_two().max(2).trailing_zeros() as usize;
        // Geometry: ring degree 2^6, m covering 3 limbs per value.
        let pcs = match v3_pcs_for(log_t, 6, [91u8; 32]) {
            Ok(p) => p,
            Err(e) => {
                println!("| {} | {} | - | - | - | - | pcs error {e:?} |", prog.name, cycles);
                continue;
            }
        };
        let (state, proof) = match prove_v3(&pcs, &prog.image, &prog.public_input, 5_000_000) {
            Ok(x) => x,
            Err(e) => {
                println!(
                    "| {} | {} | - | - | - | - | {} |",
                    prog.name,
                    cycles,
                    status_of(&e)
                );
                continue;
            }
        };
        let prove_ms = t0.elapsed().as_millis();
        let t1 = Instant::now();
        let verdict = verify_v3(&pcs, &prog.image, &prog.public_input, &state, &proof, 5_000_000);
        let verify_ms = t1.elapsed().as_millis();
        let kb = proof_size_kb(&proof);
        let thr = (cycles as f64) / (prove_ms as f64 / 1000.0);
        let status = if verdict.is_ok() {
            "verified"
        } else {
            "VERIFY FAILED"
        };
        println!(
            "| {} | {} | {} | {} | {:.1} | {:.0} | {} |",
            prog.name, cycles, prove_ms, verify_ms, kb, thr, status
        );
    }
}

fn status_of(e: &lattice_zkvm::pipeline4::Pipeline3Error) -> String {
    use lattice_zkvm::pipeline4::Pipeline3Error::*;
    match e {
        ProfileViolation { what, .. } => format!("profile: {what}"),
        UnsupportedInstruction { .. } => "unsupported instruction".into(),
        Execution(_) => "execution".into(),
        other => format!("{other:?}"),
    }
}

fn proof_size_kb(proof: &lattice_zkvm::pipeline4::ProofV3) -> f64 {
    let mut bytes = 0usize;
    for c in &proof.commitments {
        bytes += c.len();
    }
    for sh in &proof.lookup_shouts {
        bytes += sumcheck_bytes(&sh.read_checking);
    }
    bytes += sumcheck_bytes(&proof.fetch.read_checking);
    for oh in [&proof.onehot_ram_r, &proof.onehot_ram_w, &proof.onehot_reg_a,
        &proof.onehot_reg_b, &proof.onehot_reg_w]
    {
        for leg in &oh.booleanity {
            bytes += sumcheck_bytes(leg);
        }
        bytes += sumcheck_bytes(&oh.raf);
    }
    for tw in [&proof.twist_ram, &proof.twist_reg_a, &proof.twist_reg_b] {
        bytes += sumcheck_bytes(&tw.read_checking);
        bytes += sumcheck_bytes(&tw.inc_definition);
        bytes += sumcheck_bytes(&tw.telescoping);
    }
    bytes += sumcheck_bytes(&proof.air);
    for c in &proof.claims {
        bytes += 8 + 8 + c.point.len() * 8 + 8;
    }
    for op in &proof.openings {
        bytes += op
            .opened_witness
            .iter()
            .map(|w| w.to_bytes().len())
            .sum::<usize>();
        bytes += sumcheck_bytes(&op.sumcheck);
    }
    (bytes / 1024) as f64
}

fn sumcheck_bytes(sc: &lattice_sumcheck::SumcheckProof) -> usize {
    sc.rounds.iter().map(|r| r.len() * 8).sum()
}
