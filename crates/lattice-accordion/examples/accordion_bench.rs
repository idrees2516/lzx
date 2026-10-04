//! Accordion benchmark: reduce / accumulate / decide costs over the Ajtai
//! module at several layered-cube sizes, plus the extraction harness cost.
//!
//! Run with `cargo run --release -p lattice-accordion --example
//! accordion_bench`.

use lattice_accordion::module::Fq;
use lattice_accordion::pcs::{
    decide_batched, eval_claim, reduce, reduce_verify, AccordionPcsParams,
};
use lattice_core::transcript::Transcript;
use std::time::Instant;

fn bench(k: usize, rows: usize) {
    let params = AccordionPcsParams {
        num_data_vars: k,
        num_layers: 4,
        rows,
        ring_degree: 64,
    };
    let cube = params.cube();
    let t0 = Instant::now();
    let srs = params.srs_from_seed(b"bench-seed");
    let setup_ms = t0.elapsed().as_secs_f64() * 1e3;

    let n = 1usize << k;
    let f: Vec<Fq> = (0..n)
        .map(|i| Fq::from_u64((i as u64 * 6364136223846793005 + 11) & 0xFFFF_FFFF))
        .collect();
    let w = cube.digit_layers(&f);
    let t0 = Instant::now();
    let cm = srs.commit_scalars(&w);
    let commit_ms = t0.elapsed().as_secs_f64() * 1e3;

    let u: Vec<Fq> = (0..k)
        .map(|i| Fq::from_u64(31337 + i as u64 * 17))
        .collect();
    let v = eval_claim(&cube, &w, &u);

    let t0 = Instant::now();
    let mut pt = Transcript::new_default(b"bench-reduce");
    let proof = reduce(&srs, &cube, &cm, &u, &v, &w, &mut pt).expect("reduce");
    let reduce_ms = t0.elapsed().as_secs_f64() * 1e3;
    let proof_bytes = proof.size_bytes();

    let t0 = Instant::now();
    let mut vt = Transcript::new_default(b"bench-reduce");
    let instance = reduce_verify(&srs, &cube, &cm, &u, &v, &proof, &mut vt).expect("verify");
    let verify_ms = t0.elapsed().as_secs_f64() * 1e3;

    // The amortized decider: fold t = 4 claims, decide once.
    let mut instances = vec![instance];
    for s in 1..4u64 {
        let f2: Vec<Fq> = (0..n)
            .map(|i| Fq::from_u64((i as u64 * 998244353 + s) & 0xFFFF_FFFF))
            .collect();
        let w2 = cube.digit_layers(&f2);
        let cm2 = srs.commit_scalars(&w2);
        let u2: Vec<Fq> = (0..k)
            .map(|i| Fq::from_u64(5 + i as u64 * (s + 1)))
            .collect();
        let v2 = eval_claim(&cube, &w2, &u2);
        let mut pt2 = Transcript::new_default(b"bench-reduce");
        let proof2 = reduce(&srs, &cube, &cm2, &u2, &v2, &w2, &mut pt2).expect("reduce");
        let mut vt2 = Transcript::new_default(b"bench-reduce");
        instances.push(reduce_verify(&srs, &cube, &cm2, &u2, &v2, &proof2, &mut vt2).unwrap());
    }
    let t0 = Instant::now();
    let (_folded, ok) = decide_batched(&srs, &cube, &instances).expect("batched");
    let decide_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert!(ok);

    println!(
        "k={k:2} N={:5} rows={rows} | srs {setup_ms:8.1} ms | commit {commit_ms:7.2} ms | \
         reduce {reduce_ms:7.2} ms | verify {verify_ms:6.2} ms | decide(4-fold) {decide_ms:7.2} ms | \
         proof {proof_bytes:6} B",
        cube.size(),
    );
}

fn main() {
    println!("lattice-accordion benchmark (q = 2^50 - 2687, 16-bit digit layers)");
    for k in [4, 6, 8, 10] {
        bench(k, 1);
    }
    for rows in [2, 4] {
        bench(8, rows);
    }
}
