//! Streaming-prover benchmarks (ePrint 2025/611): time AND peak memory
//! for the in-memory engine vs the streamed / hybrid / prefix-suffix /
//! grand-product / commitment paths.
//!
//! Memory is measured two ways:
//! * the deterministic `MemMeter` field-element accounting (portable,
//!   WASM-safe);
//! * the process high-water mark (`VmHWM` on Linux) — the honest
//!   end-to-end RSS number.

use lattice_core::{DenseMle, Goldilocks, Transcript};
use lattice_streaming::client::{prove_client, ClientProverConfig, MemMeter};
use lattice_streaming::grand_product::{dfs_grand_product, prove_grand_product};
use lattice_streaming::oracle::OwnedOracle;
use lattice_streaming::pcs_stream::commit_streaming;
use lattice_streaming::prefix_suffix::{prove_prefix_suffix, Structure};
use lattice_streaming::small_space::{prove_small_space, SmallSpaceInstance};
use std::time::Instant;

fn vm_hwm_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn mb(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn main() {
    println!("=== Streaming prover benchmarks (ePrint 2025/611) ===\n");

    // ------------------------------------------------------------------
    // 1. Sum-check proving: in-memory vs hybrid at several budgets.
    // ------------------------------------------------------------------
    for n in [16usize, 20] {
        let df = DenseMle::random(n, b"sb-f");
        let dh = DenseMle::random(n, b"sb-h");
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| df.evaluations[i].mul(&dh.evaluations[i]))
            .fold(Goldilocks::ZERO, |a, v| a.add(&v));
        let terms = vec![(Goldilocks::ONE, vec![0usize, 1])];

        // In-memory reference.
        let hwm0 = vm_hwm_bytes();
        let t0 = Instant::now();
        {
            let mut o0 = OwnedOracle::new(df.evaluations.clone());
            let mut o1 = OwnedOracle::new(dh.evaluations.clone());
            let mut inst = SmallSpaceInstance {
                num_vars: n,
                factors: vec![&mut o0, &mut o1],
                terms: terms.clone(),
            };
            let mut ts = Transcript::new_default(b"sb-seed");
            let out = prove_small_space(&mut inst, claim, &mut ts).unwrap();
            let _ = out.final_claim.to_canonical_u64();
        }
        let t_ref = t0.elapsed().as_secs_f64();
        let hwm_ref = vm_hwm_bytes().unwrap_or(0).saturating_sub(hwm0.unwrap_or(0));

        println!("[sumcheck n={n}] fully-streamed (Algorithm 1): {t_ref:.3} s, ΔVmHWM {} (data held: {})", mb(hwm_ref), mb((1u64 << n) * 16));

        for budget_elems in [1 << 10, 1 << 14, 1 << 18] {
            let config = ClientProverConfig {
                max_field_elements: budget_elems,
                progress: None,
            };
            let meter = MemMeter::new();
            let c = config.switch_round(n, 2);
            meter.alloc(2 * (1usize << c));
            let t0 = Instant::now();
            {
                let mut o0 = OwnedOracle::new(df.evaluations.clone());
                let mut o1 = OwnedOracle::new(dh.evaluations.clone());
                let mut oracles: [&mut dyn lattice_streaming::oracle::IndexOracle; 2] =
                    [&mut o0, &mut o1];
                let mut ts = Transcript::new_default(b"sb-seed");
                let out = prove_client(n, &terms, &mut oracles, claim, &config, &mut ts).unwrap();
                let _ = out.final_claim.to_canonical_u64();
            }
            let t = t0.elapsed().as_secs_f64();
            println!(
                "  hybrid (budget {} = {}, c={c}): {t:.3} s, metered prover peak {}",
                budget_elems,
                mb(budget_elems as u64 * 8),
                mb(meter.peak_bytes() as u64)
            );
        }
    }

    // ------------------------------------------------------------------
    // 2. Prefix-suffix inner product (pcnext / M-evaluation shape).
    // ------------------------------------------------------------------
    for n in [16usize, 20] {
        let u = DenseMle::random(n, b"ps-u").evaluations;
        let r: Vec<Goldilocks> = (1..=n as u64)
            .map(|i| Goldilocks::from_u64(i.wrapping_mul(1_000_000_007)))
            .collect();
        let structure = Structure::Shift { r };
        let mut stream = OwnedOracle::new(u);
        let mut ts = Transcript::new_default(b"ps-seed");
        let t0 = Instant::now();
        let out = prove_prefix_suffix(&mut stream, &structure, n, None, &mut ts).unwrap();
        let t = t0.elapsed().as_secs_f64();
        println!(
            "[prefix-suffix n={n}] {t:.3} s, {} stream passes, prover tables ~{} (2·√N·k), final claim bound",
            3,
            mb(4u64 * (1u64 << (n / 2)) * 8)
        );
        let _ = out.final_claim().to_canonical_u64();
    }

    // ------------------------------------------------------------------
    // 3. Streaming grand product (the DFS product + the Quarks proof).
    // ------------------------------------------------------------------
    for n in [16usize, 20] {
        let data: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| Goldilocks::from_u64(i.wrapping_mul(6364136223846793005).wrapping_add(1)))
            .collect();
        let mut stream = OwnedOracle::new(data.clone());
        let t0 = Instant::now();
        let p = dfs_grand_product(&mut stream, None).unwrap();
        let t_dfs = t0.elapsed().as_secs_f64();
        let mut stream2 = OwnedOracle::new(data);
        let mut ts = Transcript::new_default(b"gp-seed");
        let t0 = Instant::now();
        let proof = prove_grand_product(&mut stream2, Some(p), &mut ts).unwrap();
        let t_proof = t0.elapsed().as_secs_f64();
        println!(
            "[grand product n={n}] DFS O(n)-space product: {t_dfs:.3} s; Quarks proof: {t_proof:.3} s; stack ≤ {} entries",
            n + 1
        );
        let _ = proof.product.to_canonical_u64();
    }

    // ------------------------------------------------------------------
    // 4. Matrix-layout streaming commitment.
    // ------------------------------------------------------------------
    for n in [16usize, 20] {
        let data = DenseMle::random(n, b"sc-bench").evaluations;
        let mut stream = OwnedOracle::new(data);
        let t0 = Instant::now();
        let commitment = commit_streaming(&mut stream, n).unwrap();
        let t = t0.elapsed().as_secs_f64();
        println!(
            "[streaming commitment n={n}] {t:.3} s, one pass, O(√N) = {} row buffer",
            mb((1u64 << (n / 2)) * 8)
        );
        let _ = commitment.root;
    }

    if let Some(hwm) = vm_hwm_bytes() {
        println!("\n(process VmHWM at exit: {})", mb(hwm));
    }
}
