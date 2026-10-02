//! CauchyFold benchmark: the full node protocol (prove → verify) at
//! several arities, plus the carrier algebra and the boundary analysis.
//!
//! Run with `cargo run --release -p lattice-cauchyfold --example
//! cauchyfold_bench`.

use lattice_cauchyfold::boundary::analyze;
use lattice_cauchyfold::cauchy::{Carrier, CauchyParams, QuadraticMap};
use lattice_cauchyfold::node::{honest_witness, prove, verify, NodeParams};
use std::time::Instant;

fn bench_node(k: usize) {
    let params = NodeParams::scaled(k);
    let witness = honest_witness(&params, 42 + k as u64);
    let t0 = Instant::now();
    let (instance, proof) = prove(&params, &witness, b"bench").expect("prove");
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    verify(&params, &instance, &proof).expect("verify");
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;
    // The wire: the field front end + the chain + the terminal codec.
    let chain_bytes: usize = proof
        .chain
        .layers
        .iter()
        .map(|l| l.t.iter().map(|tj| tj.len()).sum::<usize>())
        .sum::<usize>()
        * 384
        + proof.chain.terminal.commitment.len() * 384
        + proof
            .chain
            .terminal
            .blocks
            .iter()
            .map(|b| b.len())
            .sum::<usize>()
            * 384;
    println!(
        "k={k:2} | prove {prove_ms:8.1} ms | verify {verify_ms:8.1} ms | chain-wire ~{chain_bytes:6} B",
    );
}

fn bench_carrier(k: usize) {
    let params = CauchyParams::paper(k);
    let q = QuadraticMap::benchmark(4, 2, 7);
    let sources: Vec<Vec<lattice_cauchyfold::field_k::K4>> = (0..k + 1)
        .map(|i| {
            (0..4)
                .map(|l| {
                    lattice_cauchyfold::field_k::K4::from_coeffs([
                        (i * 31 + l * 7 + 1) as u64,
                        (i * 13 + l * 3) as u64,
                        i as u64,
                        l as u64,
                    ])
                })
                .collect()
        })
        .collect();
    let t0 = Instant::now();
    let carrier = Carrier::direct(&params, &q, &sources);
    let direct_ms = t0.elapsed().as_secs_f64() * 1e3;
    let interp: Vec<lattice_cauchyfold::field_k::K4> = (0..k)
        .map(|t| lattice_cauchyfold::field_k::K4::from_coeffs([(5000 + t * 997) as u64, 3, 7, 11]))
        .collect();
    let t1 = Instant::now();
    let fast = Carrier::fast(&params, &q, &sources, &interp);
    let fast_ms = t1.elapsed().as_secs_f64() * 1e3;
    let agree = fast.coeffs == carrier.coeffs;
    println!(
        "k={k:2} | carrier direct {direct_ms:6.2} ms | fast {fast_ms:6.2} ms | agree={agree}"
    );
}

fn bench_boundary(k: usize) {
    let params = CauchyParams::paper(k);
    let q = QuadraticMap::benchmark(4, 3, 17);
    let support: Vec<lattice_cauchyfold::field_k::K4> = (0..2 * k + 3)
        .map(|i| lattice_cauchyfold::field_k::K4::from_coeffs([(9000 + i * 997) as u64, 3, 7, 11]))
        .collect();
    let t0 = Instant::now();
    let a = analyze(&params, &support, &q);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "k={k:2} | boundary {ms:6.2} ms | dim Va={}/{k} (attained: {}) r={} m_min={}",
        a.dim_va,
        a.dim_va == k,
        a.r,
        a.m_min
    );
}

fn main() {
    println!("lattice-cauchyfold benchmark (q = 2^48-59, K = Fq4, R_{{q,64}})");
    for k in [2, 4, 8, 16] {
        bench_carrier(k);
    }
    for k in [2, 4, 8] {
        bench_boundary(k);
    }
    for k in [2, 4, 8, 16] {
        bench_node(k);
    }
}
