//! The Stage-5.2 claims-fold evidence: the folded compact proof's size
//! breakdown at the benchmark shapes (the values-only claims list is
//! gone — replaced by the fold's layers + claim pairs + the two S
//! values + the 18 pre-leg entries).

use lattice_guest::programs;
use lattice_zkvm::memproof::{prove_memory_argument_compact, CompactMemoryProof};

fn fold_bytes(p: &CompactMemoryProof) -> usize {
    let mut bytes = 0usize;
    for layer in &p.fold.layers {
        bytes += layer.rounds.len() * layer.rounds[0].len().max(1) * 8 + 16;
    }
    bytes += p.fold.claims.len() * 2 * 8;
    bytes += 16; // s_bits + s_vals
    bytes += (p.fold.addr_claims.len() + p.fold.rv_claims.len()) * 8;
    bytes
}

fn total_bytes(p: &CompactMemoryProof) -> usize {
    let mut bytes = fold_bytes(p);
    for sc in p.legs.sumchecks() {
        bytes += sc.rounds.len() * sc.rounds[0].len().max(1) * 8 + 16;
        bytes += 8 * 8; // the transmitted leg claims (8 vectors)
    }
    bytes += p.bits_commitment.len();
    bytes += p.values_commitment.len();
    for carrier in [&p.bits_carrier, &p.values_carrier] {
        bytes += carrier.rounds.iter().map(|r| r.len() * 8).sum::<usize>() + 16 + 8;
    }
    for (op, lens) in [
        (&p.bits_opening, &p.bits_factor_lens),
        (&p.values_opening, &p.values_factor_lens),
    ] {
        bytes += op.u_tilde.len() * 8;
        bytes += op.response.hist.len() + op.response.payload.len() + op.response.raw.len();
        bytes += lens.len() + 16;
    }
    bytes += 64; // statement estimate
    bytes
}

fn probe(name: &str, program: &[u8], input: &[u8], max_steps: u64) {
    let fetch_words = program.len().div_ceil(4);
    let fetch_log_k = fetch_words.next_power_of_two().max(1).trailing_zeros() as usize;
    let mut proof: Option<CompactMemoryProof> = None;
    for ram_log_k in 4..=12usize {
        if let Ok((p, _)) =
            prove_memory_argument_compact(program, input, max_steps, ram_log_k, fetch_log_k)
        {
            proof = Some(p);
            break;
        }
    }
    let Some(p) = proof else {
        println!("{name}: no window fits");
        return;
    };
    let layers = p.fold.layers.len();
    let rounds: usize = p.fold.layers.iter().map(|l| l.rounds.len()).sum();
    println!(
        "| {name} | {layers} layers | {rounds} rounds | fold {} B | total {} B ({:.1} KB) |",
        fold_bytes(&p),
        total_bytes(&p),
        total_bytes(&p) as f64 / 1024.0
    );
}

fn fib_program() -> Vec<u8> {
    let mut code: Vec<u8> = Vec::new();
    code.extend_from_slice(&[0x93, 0x05, 0x10, 0x00]); // addi x1, x0, 1
    code.extend_from_slice(&[0x13, 0x01, 0x10, 0x00]); // addi x2, x0, 1
    for _ in 0..14 {
        code.extend_from_slice(&[0xB3, 0x01, 0x21, 0x00]); // add x3, x1, x2
    }
    code.extend_from_slice(&0x73u32.to_le_bytes()); // ecall (halt)
    code
}

fn main() {
    println!("| program | fold layers | fold rounds | fold bytes | total proof |");
    println!("|---|---|---|---|---|");
    probe("fib-test", &fib_program(), &[], 64);
    match programs::fibonacci(18) {
        Ok(p) => probe("fib-bench", &p.image, &p.public_input, 1 << 22),
        Err(e) => println!("guest build failed: {e:?}"),
    }
    match programs::fibonacci(40) {
        Ok(p) => probe("fib-40", &p.image, &p.public_input, 1 << 22),
        Err(e) => println!("guest build failed: {e:?}"),
    }
}
