//! SVSC benchmark: the ss-class weighting restructure vs the round-by-round
//! reference over `Fp256` — the measured κ (bb:ss cost ratio) and the
//! window speedup (ePrint 2025/1117 §5 / 2026/587 §9's methodology).
//!
//! Run: `cargo run -p lattice-projsumcheck --example svsc_bench --release`

use lattice_core::transcript::Transcript;
use lattice_projsumcheck::fp256::Fp256;
use lattice_projsumcheck::svsc::{
    prove_reference, prove_svsc, SmallFactor, SvscInstance, WindowSchedule,
};

fn std_time<F: FnMut()>(mut f: F) -> std::time::Duration {
    // Warm-up + best-of-3.
    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn claim_of(inst: &SvscInstance) -> Fp256 {
    let mut accs = vec![0u128; 1usize << inst.num_vars];
    for (c, ids) in &inst.terms {
        for idx in 0..(1usize << inst.num_vars) {
            let mut prod = *c as u128;
            for &fi in ids {
                prod *= inst.factors[fi].coeffs[idx] as u128;
            }
            accs[idx] += prod;
        }
    }
    let total: u128 = accs.iter().sum();
    Fp256::from_canonical_u128(total)
}

/// The measured κ: one bb (CIOS Fp256 mul) vs one ss (u64×u64→u128 small
/// multiply accumulated) — the paper's §9 cost model.
fn measure_kappa() -> f64 {
    let a = Fp256::from_canonical_u64(0x9E37_79B9_7F4A_7C15);
    let b = Fp256::from_canonical_u64(0x51ED_2701_2134_5B67);
    let data: Vec<u64> = (0..4096).map(|i| (i as u64 * 6364136223846793005) >> 32).collect();

    let bb = std_time(|| {
        let mut acc = a;
        for _ in 0..100 {
            for _ in 0..4096 {
                acc = acc.mul(&b);
            }
        }
        std::hint::black_box(acc);
    });
    let ss = std_time(|| {
        let mut acc: u128 = 0;
        for _ in 0..100 {
            for chunk in data.chunks(2) {
                acc = acc.wrapping_add((chunk[0] as u128) * (chunk[1] as u128));
            }
        }
        std::hint::black_box(acc);
    });
    let per_bb = bb.as_nanos() as f64 / (100.0 * 4096.0);
    let per_ss = ss.as_nanos() as f64 / (100.0 * 2048.0);
    per_bb / per_ss
}

fn main() {
    println!("=== κ measurement (bb:ss cost ratio, 4-limb CIOS vs u64 ss) ===");
    let kappa = measure_kappa();
    println!("measured κ ≈ {kappa:.1}  (the paper's model: 2N²+N = 36 at N=4)");
    println!();

    println!("=== The windowed prover vs the reference (d=2, 33-bit values) ===");
    println!("ℓ     v     reference      windowed       speedup");
    for &(ell, v) in &[(12usize, 2usize), (12, 3), (14, 3), (16, 3), (16, 4), (18, 3), (18, 4)] {
        let factors: Vec<SmallFactor> = (0..2)
            .map(|k| SmallFactor::random_small(ell, 33, format!("bench-{ell}-{k}").as_bytes()))
            .collect();
        let inst = SvscInstance::product(factors, 33).ok().unwrap();
        let claim = claim_of(&inst);

        let t_ref = std_time(|| {
            let mut t = Transcript::new_default(b"svsc-bench");
            let _ = prove_reference(&inst, &claim, &mut t).ok().unwrap();
        });
        let t_win = std_time(|| {
            let mut t = Transcript::new_default(b"svsc-bench");
            let _ = prove_svsc(&inst, &claim, &mut t, &WindowSchedule::Early { v })
                .ok()
                .unwrap();
        });
        println!(
            "{ell:<5} {v:<5} {:>12.1?}  {:>12.1?}  {:.2}×",
            t_ref,
            t_win,
            t_ref.as_secs_f64() / t_win.as_secs_f64()
        );
    }
    println!();
    println!("The paper's Lemma 5 optimum at d=2, κ≈33: v* = log_3(4κ) ≈ {}",
        (4.0 * kappa).log2() / 3.0_f64.log2());
}
