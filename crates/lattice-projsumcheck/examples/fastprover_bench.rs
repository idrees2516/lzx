//! Fp256 window fast-prover benchmark (ePrint 2025/1117 + 2026/587 §5,
//! the small-value setting at the CIOS 256-bit field).
//!
//! Reports:
//! 1. the **measured κ** — bb (full CIOS) vs ss (native i128) vs sb
//!    (zero-limb-skipped CIOS) wall-clock ratios, against the paper's
//!    `κ ≈ 2N² + N = 36` at N = 4 limbs;
//! 2. baseline (LinearTimeSC) vs window prover wall-clock on digit-table
//!    instances at the optimal window `v* = log_{d+1}(d²κ)` — the
//!    papers' 2.5–4× claim — plus the byte-identity check;
//! 3. the native-multiplication instrumentation (bb/sb/ss).

use lattice_core::transcript::Transcript;
use lattice_projsumcheck::fastprover::{
    kappa_limbs, optimal_window, prove_baseline, prove_fast, take_last_stats, FastFpOpts,
    FpFactor, FpVirtualPolynomial,
};
use lattice_projsumcheck::fp256::Fp256;

fn digit_table(log_vars: usize, digit_bound: i128, seed: u64) -> Vec<i128> {
    let n = 1usize << log_vars;
    (0..n)
        .map(|i| {
            let h = Transcript::hash_domain(
                b"fp-bench",
                &[seed.to_le_bytes(), (i as u64).to_le_bytes()].concat(),
            );
            (u64::from_le_bytes(h[..8].try_into().unwrap_or([0; 8]))
                % (digit_bound as u64 * 2 + 1)) as i128
                - digit_bound
        })
        .collect()
}

fn main() {
    println!("=== Fp256 window fast prover (2025/1117 + 2026/587, SV setting) ===\n");

    // ---- 1. The measured cost ratios (κ) ----
    let iters = 2_000_000u64;
    let a = Fp256::from_canonical_u64(0xdead_beef_cafe_f00d);
    let b = Fp256::from_canonical_u64(0x1234_5678_9abc_def0);

    let t0 = std::time::Instant::now();
    let mut acc_bb = a;
    for i in 0..iters {
        acc_bb = acc_bb.mul(if (i & 1) == 0 { &b } else { &a });
    }
    let bb_ns = t0.elapsed().as_secs_f64() * 1e9 / iters as f64;
    std::hint::black_box(&acc_bb);

    let t1 = std::time::Instant::now();
    let mut acc_sb = a;
    for i in 0..iters {
        acc_sb = acc_sb.mul_small(((i as i128) % 65536) - 32768);
    }
    let sb_ns = t1.elapsed().as_secs_f64() * 1e9 / iters as f64;
    std::hint::black_box(&acc_sb);

    let t2 = std::time::Instant::now();
    let mut acc_ss: i128 = 0x5eed_c0de_1234_5678u64 as i128;
    for _ in 0..iters {
        acc_ss = acc_ss.wrapping_mul(0x1234_5678_9abc_def0u64 as i128);
        acc_ss ^= acc_ss >> 7;
    }
    let ss_ns = t2.elapsed().as_secs_f64() * 1e9 / iters as f64;
    std::hint::black_box(acc_ss);

    let kappa_ss = bb_ns / ss_ns.max(0.001);
    let kappa_sb = bb_ns / sb_ns.max(0.001);
    println!("[kernels over {iters} iters]");
    println!("  bb (full CIOS)      : {bb_ns:7.1} ns");
    println!("  sb (zero-limb CIOS) : {sb_ns:7.1} ns   (x{kappa_sb:.2} vs bb)");
    println!("  ss (native i128)    : {ss_ns:7.1} ns   (x{kappa_ss:.2} vs bb)");
    println!(
        "  paper's kappa = 2N^2+N = {} (N=4 limbs); measured ss-ratio {:.1}\n",
        kappa_limbs(4),
        kappa_ss
    );

    // ---- 2. Baseline vs window on digit-table instances ----
    for &(log_vars, d, digit_bound) in
        &[(14usize, 2usize, 255i128), (13, 3, 255), (14, 3, 15)]
    {
        let factors: Vec<FpFactor> = (0..d)
            .map(|k| FpFactor::Small(digit_table(log_vars, digit_bound, 40 + k as u64)))
            .collect();
        let vp = FpVirtualPolynomial::product(factors).expect("instance");
        let claim = vp.total_sum().expect("claim");

        let kappa = kappa_limbs(4);
        let vstar = optimal_window(d, kappa, log_vars);
        println!(
            "[digit tables: log_vars={log_vars} (M=2^{log_vars}), d={d}, |digits|<={digit_bound}] v* = {vstar}"
        );

        // Baseline (the honest LinearTimeSC comparison point).
        let t0 = std::time::Instant::now();
        let mut ts0 = Transcript::new_default(b"fp-bench");
        let base = prove_baseline(&vp, claim, &mut ts0).expect("baseline");
        let base_elapsed = t0.elapsed().as_secs_f64() * 1e3;

        // Sweep the window — the paper's v* uses the ss-ratio κ; the
        // honest optimum on this machine uses the MEASURED sb ratio
        // (the weighting mults are sb-class, not ss), reported below.
        let mut best = (0usize, f64::INFINITY);
        for w in 1..=(vstar + 2).min(log_vars) {
            let t1 = std::time::Instant::now();
            let mut ts1 = Transcript::new_default(b"fp-bench");
            let fast = prove_fast(
                &vp,
                claim,
                &mut ts1,
                &FastFpOpts { window: w, collect_stats: true },
            )
            .expect("fast");
            let fast_ms = t1.elapsed().as_secs_f64() * 1e3;
            let identical =
                base.proof == fast.proof && base.challenges == fast.challenges;
            let stats = take_last_stats().unwrap_or_default();
            let speedup = base_elapsed / fast_ms.max(0.001);
            if fast_ms < best.1 {
                best = (w, fast_ms);
            }
            println!(
                "  w={w}: fast {fast_ms:8.1} ms | speedup {speedup:5.2}x | identity={identical} | bb={} sb={} ss={}",
                stats.bb_mults, stats.sb_mults, stats.ss_mults
            );
        }
        println!(
            "  => best w={}, {:.2}x vs baseline ({:.1} ms -> {:.1} ms); paper v*={vstar} (ss-kappa), measured sb-kappa {:.2}",
            best.0,
            base_elapsed / best.1.max(0.001),
            base_elapsed,
            best.1,
            kappa_sb
        );
        println!();
    }
}
