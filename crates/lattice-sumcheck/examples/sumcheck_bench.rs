//! Goldilocks AVX-512 vs scalar benchmark for the sumcheck hot loops
//! (audit-gate companion to `lattice_core::field_simd`).
//!
//! * End-to-end: full `sumcheck::prove` (and `verify`) on representative
//!   virtual polynomials over 2^16 and 2^18 evaluations (16 / 18 variables).
//! * Micro: `mul_slices`, the packed eq-table builder, and one full round
//!   computation (the `sum_products` shape: per-`t` half-binding of every
//!   factor + 8-lane lazy term-product accumulation).
//!
//! Timing doctrine matches the repo harness (`lattice-bench`): warmup run,
//! then N timed repetitions, MEDIAN reported, `std::hint::black_box` around
//! every result. Everything is field-exact, so the scalar and AVX-512 proofs
//! are compared bit-for-bit (proof digest) as a live exactness gate.
//!
//! The scalar column is measured in a child process of this same binary run
//! with `LZX_NO_SIMD=1` (the `lattice_core::field_simd` escape hatch), so one
//! plain invocation prints the whole comparison table:
//!
//! ```text
//! cargo run --release -p lattice-sumcheck --example sumcheck_bench
//! ```
//!
//! (Running the example directly with `LZX_NO_SIMD=1` benches the scalar
//! path only; on non-AVX-512 hosts both columns are scalar.)

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use lattice_core::field_simd::{
    bind_half_slices, eq_table, mul_slices, Sum8,
};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck::{self, SumcheckOutput};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

/// Timed repetitions per bench item (>= 5 per the bench protocol).
const REPS: usize = 5;

/// Warmup + `REPS` timed runs; returns the median in microseconds.
fn median_us<F: FnMut()>(mut f: F) -> f64 {
    f(); // warmup (cache warm, allocator warm)
    let mut samples: Vec<f64> = (0..REPS)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    samples[REPS / 2]
}

/// Deterministic checksum over a proof: rounds, challenges, final claim.
fn proof_digest(out: &SumcheckOutput) -> u64 {
    let mut acc: u64 = 0x243F_6A88_85A3_08D3;
    for round in &out.proof.rounds {
        for e in round {
            acc = acc
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(e.0);
        }
    }
    for c in &out.challenges {
        acc = acc.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(c.0);
    }
    acc ^ out.final_claim.0
}

/// The representative virtual polynomial: 3 shared factors, 3 product terms
/// (degrees 2, 3, 1 — the shape every staged relation compiles to).
fn build_vp(num_vars: usize) -> VirtualPolynomial {
    let mut vp = VirtualPolynomial::new(num_vars);
    let f = vp
        .add_factor(DenseMle::random(num_vars, b"bench-f"))
        .ok()
        .unwrap();
    let g = vp
        .add_factor(DenseMle::random(num_vars, b"bench-g"))
        .ok()
        .unwrap();
    let h = vp
        .add_factor(DenseMle::random(num_vars, b"bench-h"))
        .ok()
        .unwrap();
    vp.add_term(Goldilocks::from_u64(3), vec![f, g]).ok().unwrap();
    vp.add_term(Goldilocks::from_u64(5), vec![g, h, f])
        .ok()
        .unwrap();
    vp.add_term(Goldilocks::from_u64(11), vec![h]).ok().unwrap();
    vp
}

/// The `sum_products` round shape on public kernels: bind the first variable
/// of every factor to `t`, then accumulate all term products (this mirrors
/// the vectorized loop inside `sumcheck::prove`).
fn one_round_eval(
    factors: &[DenseMle],
    terms: &[(Goldilocks, Vec<usize>)],
    t: Goldilocks,
) -> Goldilocks {
    let points = factors[0].evaluations.len() / 2;
    let mut bound_vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(factors.len());
    for f in factors {
        let evs = &f.evaluations;
        let mut vals = vec![Goldilocks::ZERO; points];
        if t.is_zero() {
            vals.copy_from_slice(&evs[..points]);
        } else if t == Goldilocks::ONE {
            vals.copy_from_slice(&evs[points..]);
        } else {
            bind_half_slices(&evs[..points], &evs[points..], t, &mut vals);
        }
        bound_vals.push(vals);
    }
    let mut acc = Sum8::new();
    let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
    for (coeff, ids) in terms {
        fslices.clear();
        fslices.extend(ids.iter().map(|fi| bound_vals[*fi].as_slice()));
        acc.accumulate_term(*coeff, &fslices);
    }
    acc.finish()
}

/// One bench item result.
struct Row {
    id: &'static str,
    us: f64,
}

/// Everything one bench mode produces: table rows + proof digests.
struct BenchOutput {
    rows: Vec<Row>,
    digests: Vec<(&'static str, u64)>,
}

/// Run every bench in the *current* process mode (AVX-512 if the gate is on).
fn run_all() -> BenchOutput {
    let mut rows = Vec::new();
    let mut digests = Vec::new();

    // ---- micro: mul_slices ------------------------------------------------
    for log_n in [16usize, 18] {
        let n = 1usize << log_n;
        let a: Vec<Goldilocks> = (0..n)
            .map(|i| Goldilocks::from_u64((i as u64).wrapping_mul(2654435761)))
            .collect();
        let b: Vec<Goldilocks> = (0..n)
            .map(|i| Goldilocks::from_u64((i as u64).wrapping_mul(40503) ^ 0x5555))
            .collect();
        let mut out = vec![Goldilocks::ZERO; n];
        let us = median_us(|| {
            mul_slices(black_box(&a), black_box(&b), black_box(&mut out));
        });
        black_box(out[n / 2]);
        rows.push(Row {
            id: if log_n == 16 { "mul-slices-2^16" } else { "mul-slices-2^18" },
            us,
        });
    }

    // ---- micro: eq table ---------------------------------------------------
    for log_m in [16usize, 18] {
        let point: Vec<Goldilocks> = (0..log_m)
            .map(|i| Goldilocks::from_u64((i as u64 * 2654435761) % 0xFFFF_FFFF))
            .collect();
        let mut table = Vec::new();
        let us = median_us(|| {
            table = eq_table(black_box(&point));
        });
        black_box(table.as_ptr());
        rows.push(Row {
            id: if log_m == 16 { "eq-table-16vars" } else { "eq-table-18vars" },
            us,
        });
    }

    // ---- micro: one round computation (sum_products shape) -----------------
    for num_vars in [16usize, 18] {
        let vp = build_vp(num_vars);
        let terms = vp.terms.clone();
        let us = median_us(|| {
            // One full round = g(t) for t = 0..=max_degree.
            let mut sink = Goldilocks::ZERO;
            for t in 0..=vp.max_degree() {
                sink = one_round_eval(
                    black_box(&vp.factors),
                    black_box(&terms),
                    Goldilocks::from_u64(t as u64),
                );
            }
            black_box(sink);
        });
        rows.push(Row {
            id: if num_vars == 16 {
                "one-round-2^16"
            } else {
                "one-round-2^18"
            },
            us,
        });
    }

    // ---- end-to-end: prove + verify ----------------------------------------
    for num_vars in [16usize, 18] {
        let vp = build_vp(num_vars);
        let claim = vp.sum_over_hypercube();
        let mut prove_digest: u64 = 0;
        let us = median_us(|| {
            let mut t = Transcript::new_default(b"lzx-sumcheck-bench");
            let out = sumcheck::prove(black_box(&vp), claim, &mut t)
                .ok()
                .unwrap();
            prove_digest = proof_digest(&out);
            black_box(out.final_claim);
        });
        rows.push(Row {
            id: if num_vars == 16 { "prove-2^16 (full)" } else { "prove-2^18 (full)" },
            us,
        });
        digests.push((
            if num_vars == 16 { "prove-2^16 digest" } else { "prove-2^18 digest" },
            prove_digest,
        ));

        // Verify (untimed correctness gate + timed row): the digest mode also
        // checks the verifier accepts the proof.
        let mut pt = Transcript::new_default(b"lzx-sumcheck-bench");
        let out = sumcheck::prove(&vp, claim, &mut pt).ok().unwrap();
        let expected = vp.evaluate(&out.challenges).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-sumcheck-bench");
        let verdict = out
            .proof
            .verify(vp.num_vars, vp.max_degree(), claim, &mut vt, Some(expected));
        let ok = verdict.is_ok();
        let us_verify = median_us(|| {
            let mut vt2 = Transcript::new_default(b"lzx-sumcheck-bench");
            let v = out
                .proof
                .verify(vp.num_vars, vp.max_degree(), claim, &mut vt2, Some(expected));
            black_box(v.is_ok());
        });
        rows.push(Row {
            id: if num_vars == 16 {
                "verify-2^16 (full)"
            } else {
                "verify-2^18 (full)"
            },
            us: us_verify,
        });
        digests.push((
            if num_vars == 16 { "verify-2^16 ok" } else { "verify-2^18 ok" },
            u64::from(ok),
        ));
    }

    BenchOutput { rows, digests }
}

fn fmt_us(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:8.2} ms", us / 1000.0)
    } else {
        format!("{:8.2} us", us)
    }
}

fn main() {
    let simd_on = lattice_core::field_simd::avx512_field();
    let child_mode = std::env::var_os("LZX_BENCH_CHILD").is_some();

    if child_mode {
        // Machine-readable scalar run for the parent to parse.
        let out = run_all();
        for r in &out.rows {
            println!("BENCH {} {:.3}", r.id, r.us);
        }
        for (id, d) in &out.digests {
            println!("DIGEST {} {:016x}", id, d);
        }
        return;
    }

    println!("===================================================================");
    println!(" lzx sumcheck bench — Goldilocks AVX-512 vs scalar, median of {REPS}");
    println!("===================================================================");
    println!(
        " field kernels: {}",
        if simd_on {
            "AVX-512 ON (runtime-detected)"
        } else {
            "OFF (LZX_NO_SIMD=1 or no AVX-512) — both columns scalar"
        }
    );
    println!();

    // Scalar column: re-exec this binary with the SIMD escape hatch.
    let scalar: Option<BenchOutput> = std::env::current_exe()
        .ok()
        .and_then(|exe| {
            Command::new(exe)
                .env("LZX_NO_SIMD", "1")
                .env("LZX_BENCH_CHILD", "1")
                .output()
                .ok()
        })
        .and_then(|out| {
            if !out.status.success() {
                eprintln!("scalar child failed:\n{}", String::from_utf8_lossy(&out.stderr));
                return None;
            }
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            let mut rows = Vec::new();
            let mut digests = Vec::new();
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("BENCH ") {
                    if let Some((id, us)) = rest.rsplit_once(' ') {
                        if let Ok(v) = us.trim().parse::<f64>() {
                            // Leak the id: child ids are the parent's literals.
                            let id: &'static str = Box::leak(id.to_string().into_boxed_str());
                            rows.push(Row { id, us: v });
                        }
                    }
                } else if let Some(rest) = line.strip_prefix("DIGEST ") {
                    if let Some((id, d)) = rest.rsplit_once(' ') {
                        if let Ok(v) = u64::from_str_radix(d.trim(), 16) {
                            let id: &'static str = Box::leak(id.to_string().into_boxed_str());
                            digests.push((id, v));
                        }
                    }
                }
            }
            if rows.is_empty() {
                None
            } else {
                Some(BenchOutput { rows, digests })
            }
        });
    if scalar.is_none() {
        println!(" NOTE: could not launch the scalar child (LZX_NO_SIMD=1) run;");
        println!("       printing the ambient-mode column only.");
    }

    // SIMD column (ambient mode).
    let simd_out = run_all();

    // ---- the table ---------------------------------------------------------
    println!(" {:<22} {:>12} {:>12} {:>9}", "kernel", "scalar", "simd", "speedup");
    println!(" -----------------------------------------------------------------");
    for r in &simd_out.rows {
        let sc = scalar.as_ref().and_then(|s| {
            s.rows.iter().find(|x| x.id == r.id).map(|x| x.us)
        });
        match sc {
            Some(sc) => println!(
                " {:<22} {:>12} {:>12} {:>8.2}x",
                r.id,
                fmt_us(sc),
                fmt_us(r.us),
                sc / r.us
            ),
            None => println!(" {:<22} {:>12} {:>12} {:>9}", r.id, "-", fmt_us(r.us), "-"),
        }
    }

    // ---- cross-mode exactness ----------------------------------------------
    println!(" -----------------------------------------------------------------");
    if let Some(sbench) = &scalar {
        let mut all_match = true;
        for (id, d) in &sbench.digests {
            let simd_val = simd_out
                .digests
                .iter()
                .find(|(sid, _)| sid == id)
                .map(|(_, v)| *v);
            match simd_val {
                Some(v) => {
                    let m = v == *d;
                    all_match &= m;
                    if id.ends_with("digest") {
                        println!(" {id}: {:016x} vs {:016x} -> {}", d, v, if m { "MATCH" } else { "MISMATCH" });
                    } else {
                        println!(" {id}: {}", if *d == 1 { "verified OK" } else { "FAILED" });
                    }
                }
                None => all_match = false,
            }
        }
        println!(
            " scalar vs SIMD proofs bit-identical: {}",
            if all_match { "YES" } else { "NO" }
        );
    } else {
        println!(" (no scalar column: cross-mode digest check skipped)");
    }
    println!("===================================================================");
}
