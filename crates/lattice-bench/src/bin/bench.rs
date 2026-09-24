//! The reproducible benchmark matrix (audit gate G7 / §9.6 item 30).
//!
//! Pure-std timing harness (zero external dependencies — the workspace
//! rule): warmup + N repetitions, median and minimum reported. Output
//! is a deterministic markdown matrix + CSV written to the target
//! directory; the release profile is required for meaningful numbers
//! (`cargo run -p lattice-bench --release`).
//!
//! Matrix rows: field arithmetic, NTT (generic vs fast), ring
//! multiplication, Ajtai commit/open, sumcheck prove/verify, the ZK
//! sumcheck + ABDLOP proof, Akita PCS commit/prove/verify, the zkVM
//! end-to-end, plus proof/setup SIZE accounting.



use lattice_bench::{fmt_ns, measure, Timing};

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_field() -> Vec<Timing> {
    use lattice_core::Goldilocks;
    let a = Goldilocks::from_u64(0x1234_5678_9ABC_DEF0u64);
    let mut acc = Goldilocks::ONE;
    vec![
        measure("field-mul", || {
            for _ in 0..1000 {
                acc = acc.mul(&a);
            }
        }, 2, 10),
        measure("field-inverse", || {
            for _ in 0..100 {
                let _ = a.inverse();
            }
        }, 1, 5),
    ]
}

fn bench_ntt() -> Vec<Timing> {
    use lattice_ring::ntt::NttTables;
use lattice_ring::{Modulus32, RingConfig, RingElement};
    let mut out = Vec::new();
    for &log_n in &[6u32, 8, 10] {
        let cfg = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let tables = NttTables::new(Modulus32::Q_32, log_n).ok().unwrap();
        let base: Vec<u32> = (0..(1usize << log_n))
            .map(|i| (i as u64 * 2654435761 % 3221225473) as u32)
            .collect();
        let tables_ref = std::rc::Rc::new(tables);
        out.push(measure(
            &format!("ntt-forward-generic-n{}", 1usize << log_n),
            {
                let mut v = base.clone();
                let tables = std::rc::Rc::clone(&tables_ref);
                move || {
                    tables.forward(&mut v).ok().unwrap();
                }
            },
            2,
            15,
        ));
        out.push(measure(
            &format!("ntt-forward-fast-n{}", 1usize << log_n),
            {
                let mut v = base.clone();
                let tables = std::rc::Rc::clone(&tables_ref);
                move || {
                    tables.forward_fast(&mut v).ok().unwrap();
                }
            },
            2,
            15,
        ));
        out.push(measure(
            &format!("ring-mul-n{}", 1usize << log_n),
            {
                let a = RingElement::from_coeffs(&cfg, base.clone());
                let b = RingElement::from_coeffs(&cfg, base.clone());
                move || {
                    let _ = a.mul(&b);
                }
            },
            2,
            15,
        ));
    }
    out
}

fn bench_commitment() -> Vec<Timing> {
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};
    let params = AjtaiParams {
        ring: RingConfig::new(Modulus32::Q_32, 8).ok().unwrap(),
        k: 2,
        m: 16,
        norm_bound: 1 << 23,
    };
    let pk = AjtaiPublicKey::from_seed(params, [42u8; 32]).ok().unwrap();
    let s = lattice_commitment::ajtai::sample_small_secret(&pk.params.ring, pk.params.m, 32, b"bench");
    let t = pk.commit(&s).ok().unwrap();
    vec![
        measure("ajtai-commit-cached-ntt-m16-n256", || {
            let _ = pk.commit(&s);
        }, 1, 10),
        measure("ajtai-commit-naive-m16-n256", || {
            let _ = pk.commit_naive_reference(&s);
        }, 1, 10),
        measure("ajtai-verify-opening-m16", || {
            let _ = pk.verify_opening(&t, &s);
        }, 1, 10),
    ]
}

// ---------------------------------------------------------------------------
// Wave 6 benchmark families: folding, lookup, salsa, short challenges,
// the Modulus50 quadratic-slot kernel, and the SIS estimator.
// ---------------------------------------------------------------------------

fn bench_folding() -> Vec<Timing> {
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey, sample_small_secret};
    use lattice_folding::cyclo::{CycloAccumulator, MAX_FOLDS_BEFORE_REFRESH};
    use lattice_folding::symphony::{fold_many_degree2, SymRelation};
    use lattice_ring::{Modulus32, RingConfig};
    let mut out = Vec::new();

    // Cyclo fold + refresh at ring n = 256, m = 16.
    let ring = RingConfig::new(Modulus32::Q_32, 8).ok().unwrap();
    let params = AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m: 16,
        norm_bound: 1 << 26,
    };
    let pk = AjtaiPublicKey::from_seed(params.clone(), [61u8; 32]).ok().unwrap();
    let ext_params = AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m: 64,
        norm_bound: 1 << 26,
    };
    let pk_ext = AjtaiPublicKey::from_seed(ext_params, [62u8; 32]).ok().unwrap();
    let acc = CycloAccumulator::new(&pk, &sample_small_secret(&ring, 16, 256, b"b-acc"))
        .ok()
        .unwrap();
    let input = sample_small_secret(&ring, 16, 256, b"b-in");
    out.push(measure("cyclo-fold-m16-n256", || {
        let _ = acc.fold(&pk, &input);
    }, 1, 10));
    {
        let folded = acc.fold(&pk, &input).ok().unwrap();
        out.push(measure("cyclo-refresh-ext-commit", || {
            let _ = folded.accumulator.refresh(&pk_ext, 8);
        }, 1, 5));
    }
    // The paper-faithful ring-challenge fold (short-challenge + Γ gate).
    use lattice_core::short_challenge::{ShortChallengeFamily, ShortChallengeSpec};
    let spec = ShortChallengeSpec {
        n: ring.n(),
        family: ShortChallengeFamily::BiasedTernary {
            p_nonzero_permille: 500,
        },
    };
    out.push(measure("cyclo-fold-ring-challenge", || {
        let _ = acc.fold_ring_challenge(&pk, &input, &spec, 32);
    }, 1, 10));

    // Symphony μ-ary one-shot fold (μ = 4, degree 2).
    let sym_params = AjtaiParams {
        ring: ring.clone(),
        k: 2,
        m: 8,
        norm_bound: 1 << 26,
    };
    let sym_pk = AjtaiPublicKey::from_seed(sym_params.clone(), [63u8; 32]).ok().unwrap();
    let cross_pk = AjtaiPublicKey::from_seed(
        AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: 16,
            norm_bound: 1 << 26,
        },
        [64u8; 32],
    )
    .ok()
    .unwrap();
    let rel = SymRelation {
        num_slots: 3,
        terms: vec![(1, vec![0, 0]), (3, vec![1, 2])],
    };
    let witnesses: Vec<Vec<lattice_ring::RingElement>> = (0..4)
        .map(|i| sample_small_secret(&ring, 8, 64, format!("b-w{i}").as_bytes()))
        .collect();
    let commitments: Vec<lattice_commitment::ajtai::AjtaiCommitment> = witnesses
        .iter()
        .map(|w| sym_pk.commit(w).ok().unwrap())
        .collect();
    out.push(measure("symphony-fold-mu4-d2", || {
        let _ = fold_many_degree2(&sym_pk, &cross_pk, &rel, &witnesses, &commitments);
    }, 1, 5));
    let _ = MAX_FOLDS_BEFORE_REFRESH;
    out
}

fn bench_lookup() -> Vec<Timing> {
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_core::transcript::Transcript;
    use lattice_core::Goldilocks;
    use lattice_lookup::{prove_lookup_committed, verify_lookup_committed};
    use lattice_ring::{Modulus32, RingConfig};
    let ring = RingConfig::new(Modulus32::Q_32, 8).ok().unwrap();
    // Table of 128 values: 128·3 limbs = 384 coeffs → 1.5 elements at
    // n=256; m = 4 covers the padding.
    let params = AjtaiParams {
        ring,
        k: 2,
        m: 4,
        norm_bound: 1 << 23,
    };
    let pk = AjtaiPublicKey::from_seed(params, [71u8; 32]).ok().unwrap();
    let table: Vec<Goldilocks> = (0..128u64).map(|i| Goldilocks::from_u64(i * 7 + 3)).collect();
    // 16 distinct reads (indices 0, 7, 14, ... mod 128 are distinct).
    let reads: Vec<Goldilocks> = (0..16usize).map(|i| table[i * 7 % 128]).collect();
    let mut pt = Transcript::new_default(b"bench-lookup");
    let proof = prove_lookup_committed(&pk, &table, &reads, &mut pt).ok().unwrap();
    vec![
        measure("lookup-committed-prove-n128", || {
            let mut t = Transcript::new_default(b"bench-lookup");
            let _ = prove_lookup_committed(&pk, &table, &reads, &mut t);
        }, 1, 5),
        measure("lookup-committed-verify-n128", || {
            let mut t = Transcript::new_default(b"bench-lookup");
            let _ = verify_lookup_committed(&pk, &proof, &mut t);
        }, 1, 5),
    ]
}

fn bench_salsa() -> Vec<Timing> {
    use lattice_core::transcript::Transcript;
    use lattice_core::DenseMle;
    use lattice_salsa::{prove_norm, verify_norm};
    let z = DenseMle::random(10, b"bench-salsa");
    let mut pt = Transcript::new_default(b"bench-salsa");
    let (proof, claim, point) = prove_norm(&z, &mut pt).ok().unwrap();
    vec![
        measure("salsa-norm-prove-10vars", || {
            let mut t = Transcript::new_default(b"bench-salsa");
            let _ = prove_norm(&z, &mut t);
        }, 1, 5),
        measure("salsa-norm-verify-10vars", || {
            let mut t = Transcript::new_default(b"bench-salsa");
            let _ = verify_norm(&proof, 10, claim, point.first().copied(), &mut t);
        }, 1, 5),
    ]
}

fn bench_wave6_substrate() -> Vec<Timing> {
    use lattice_core::short_challenge::{pikkufold_spec, ShortChallengeFamily, ShortChallengeSpec};
    use lattice_ring::modulus50::RingConfig50;
    let mut out = Vec::new();
    // Short-challenge sampling + Γ certification (PikkuFold profile).
    let spec = pikkufold_spec();
    out.push(measure("short-challenge-sample-fw256-w23", || {
        let _ = spec.sample_with_gamma_cap(b"bench", 8, 16);
    }, 1, 10));
    // Fq2 extension-field multiplication.
    use lattice_core::extension::challenge_fq2;
    let mut t = lattice_core::transcript::Transcript::new_default(b"bench-fq2");
    let z = challenge_fq2(&mut t, b"z").ok().unwrap();
    out.push(measure("fq2-mul", || {
        let _ = z.mul(&z);
    }, 1, 10));
    // Modulus50 quadratic-slot ring product at n = 128 (the paper's
    // 64-quadratic-slot regime).
    let ring50 = RingConfig50::new(7).ok().unwrap();
    let a: Vec<u64> = (0..128u64).map(|i| (i * 2654435761) % ring50.modulus.q).collect();
    let b: Vec<u64> = (0..128u64).map(|i| (i * 40503) % ring50.modulus.q).collect();
    out.push(measure("modulus50-quad-slot-mul-n128", || {
        let _ = ring50.mul(&a, &b);
    }, 1, 10));
    // SIS estimator: one full infinity-norm estimate (offline gating cost).
    let params = lattice_sis_estimator::SisParameters {
        n: 64,
        q: (1u128 << 48) - 59,
        m: 512,
        length_bound: 2,
        norm: lattice_sis_estimator::SisNorm::Infinity,
    };
    out.push(measure("sis-estimate-infinity", || {
        let _ = lattice_sis_estimator::estimate(
            &params,
            lattice_sis_estimator::ReductionCostModel::Adps16 {
                mode: lattice_sis_estimator::Adps16Mode::Classical,
            },
        );
    }, 1, 3));
    let _ = ShortChallengeSpec {
        n: 8,
        family: ShortChallengeFamily::FixedWeight { weight: 4, amplitude: 1 },
    };
    out
}

fn bench_sumcheck() -> Vec<Timing> {
    use lattice_core::transcript::Transcript;
    use lattice_core::{DenseMle, Goldilocks};
    use lattice_sumcheck::virtual_poly::VirtualPolynomial;
    let mut out = Vec::new();
    for &num_vars in &[6usize, 8] {
        let z = DenseMle::random(num_vars, b"bench-z");
        let claim = z
            .evaluations
            .iter()
            .map(|v| v.mul(v))
            .fold(Goldilocks::ZERO, |a, b| a.add(&b));
        out.push(measure(
            &format!("sumcheck-norm-prove-{}vars", num_vars),
            || {
                let mut vp = VirtualPolynomial::new(num_vars);
                let i1 = vp.add_factor(z.clone()).ok().unwrap();
                let i2 = vp.add_factor(z.clone()).ok().unwrap();
                vp.add_term(Goldilocks::ONE, vec![i1, i2]).ok().unwrap();
                let mut t = Transcript::new_default(b"bench-sumcheck");
                let _ = lattice_sumcheck::sumcheck::prove(&vp, claim, &mut t);
            },
            1,
            5,
        ));
        out.push(measure(
            &format!("sumcheck-norm-verify-{}vars", num_vars),
            || {
                let mut vp = VirtualPolynomial::new(num_vars);
                let i1 = vp.add_factor(z.clone()).ok().unwrap();
                let i2 = vp.add_factor(z.clone()).ok().unwrap();
                vp.add_term(Goldilocks::ONE, vec![i1, i2]).ok().unwrap();
                let mut t = Transcript::new_default(b"bench-sumcheck");
                let proof = lattice_sumcheck::sumcheck::prove(&vp, claim, &mut t).ok().unwrap();
                let mut vt = Transcript::new_default(b"bench-sumcheck");
                let _ = proof.proof.verify(num_vars, 2, claim, &mut vt, None);
            },
            1,
            5,
        ));
    }
    out
}

fn bench_zk() -> Vec<Timing> {
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_core::transcript::Transcript;
    use lattice_core::{DenseMle, Goldilocks};
    use lattice_ring::{Modulus32, RingConfig};
    use lattice_zk::entropy::{SecretSeed, ShakeStream};
    use lattice_zk::zk_sumcheck::{zk_prove, zk_simulate, zk_verify, ZkSumcheckStatement};

    let num_vars = 3usize; // kernel-scale instance (10n + 2 slots)
    let n = 1usize << num_vars;
    let m = lattice_zk::zk_sumcheck::required_slots(n);
    let params = AjtaiParams {
        ring: RingConfig::new(Modulus32::Q_32, 4).ok().unwrap(),
        k: 2,
        m,
        norm_bound: 1 << 24,
    };
    let pk = AjtaiPublicKey::from_seed(params, [9u8; 32]).ok().unwrap();
    let z = DenseMle::random(num_vars, b"bench-zk");
    let f: Vec<Goldilocks> = z.evaluations.clone();
    let claim = f.iter().fold(Goldilocks::ZERO, |a, b| a.add(b));
    let statement = ZkSumcheckStatement { num_vars, claim };

    vec![
        measure("zk-sumcheck-prove-3vars", || {
            let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"bench-zk"), b"m");
            let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
            let _ = zk_prove(&pk, &f, claim, &mut stream, &mut t);
        }, 1, 5),
        measure("zk-sumcheck-simulate-3vars", || {
            let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"bench-zk-sim"), b"s");
            let _ = zk_simulate(&pk, &statement, &mut stream);
        }, 1, 5),
        measure("zk-sumcheck-verify-3vars", || {
            let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"bench-zk"), b"m");
            let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
            let (proof, _r, v) = zk_prove(&pk, &f, claim, &mut stream, &mut t).ok().unwrap();
            let mut vt = Transcript::new_default(b"lzx-zk-sumcheck");
            let _ = zk_verify(&pk, &statement, &v, &proof, &mut vt);
        }, 1, 5),
    ]
}

fn bench_akita() -> Vec<Timing> {
    use lattice_akita::akita_setup;
    use lattice_core::transcript::Transcript;
    use lattice_core::{DenseMle, Goldilocks};
    let num_vars = 5usize;
    let mle = DenseMle::random(num_vars, b"bench-akita");
    let point: Vec<Goldilocks> = (1..=num_vars)
        .map(|i| Goldilocks::from_u64((i * 9973) as u64))
        .collect();
    vec![
        measure("akita-commit-5vars", || {
            let pcs = akita_setup(5, 24, 1 << 23, [71u8; 32]).ok().unwrap();
            let _ = pcs.commit(&mle);
        }, 1, 5),
        measure("akita-prove-eval-5vars", || {
            let pcs = akita_setup(5, 24, 1 << 23, [71u8; 32]).ok().unwrap();
            let mut t = Transcript::new_default(b"bench-akita");
            let _ = pcs.prove_evaluation(&mle, &point, &mut t);
        }, 1, 5),
        measure("akita-verify-eval-5vars", || {
            let pcs = akita_setup(5, 24, 1 << 23, [71u8; 32]).ok().unwrap();
            let mut t = Transcript::new_default(b"bench-akita");
            let proof = pcs.prove_evaluation(&mle, &point, &mut t).ok().unwrap();
            let commitment = pcs.commit(&mle).ok().unwrap();
            let mut vt = Transcript::new_default(b"bench-akita");
            let _ = pcs.verify_evaluation(&commitment, &proof, &mut vt);
        }, 1, 5),
    ]
}

fn bench_zkvm() -> Vec<Timing> {
    use lattice_akita::akita_setup;
    use lattice_vm::state::MachineState;
    use lattice_zkvm::prove_program;

    // Program: arithmetic + memory + a short loop.
    let enc_addi = |rd: u8, rs1: u8, imm: i64| -> u32 {
        ((imm as u32) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
    };
    let mut prog = Vec::new();
    prog.extend_from_slice(&enc_addi(1, 0, 0x100).to_le_bytes());
    prog.extend_from_slice(&enc_addi(2, 0, 5).to_le_bytes());
    prog.extend_from_slice(&enc_addi(3, 0, 12).to_le_bytes());
    let sw: u32 = (3 << 20) | (1 << 15) | (2 << 12) | 0x23;
    prog.extend_from_slice(&sw.to_le_bytes());
    let lw: u32 = (1 << 15) | (2 << 12) | (4 << 7) | 0x03;
    prog.extend_from_slice(&lw.to_le_bytes());
    prog.extend_from_slice(&0x73u32.to_le_bytes());

    vec![
        measure("zkvm-execute-6instr", || {
            let mut state = MachineState::new();
            state.load_program(0, &prog);
            let _ = lattice_vm::run(&mut state, 64);
        }, 1, 20),
        measure("zkvm-prove-program", || {
            let pcs = akita_setup(4, 64, 1 << 23, [91u8; 32]).ok().unwrap();
            let _ = prove_program(&pcs, &prog, &[], 64);
        }, 1, 3),
        measure("zkvm-verify-program", || {
            let pcs = akita_setup(4, 64, 1 << 23, [91u8; 32]).ok().unwrap();
            let (output, envelope) = prove_program(&pcs, &prog, &[], 64).ok().unwrap();
            let _ = lattice_zkvm::verify_program(&pcs, &prog, &[], &output, &envelope, 64);
        }, 1, 3),
    ]
}

// ---------------------------------------------------------------------------
// Size accounting
// ---------------------------------------------------------------------------

fn size_accounting() -> Vec<(String, usize)> {
    use lattice_akita::akita_setup;
    use lattice_core::{DenseMle, Goldilocks};
    use lattice_zkvm::prove_program;
    let mut out = Vec::new();
    // Akita setup key material: the seed (32B) is the verifiable
    // representation; the expanded matrix is k*m*n*4 bytes.
    for &(log_n, m) in &[(4u32, 64usize), (6, 16)] {
        let expanded = 2 * m * (1usize << log_n) * 4;
        out.push((
            format!("akita-setup-expanded-logn{log_n}-m{m}"),
            expanded,
        ));
    }
    // Proof envelope size for the zkvm program.
    let enc_addi = |rd: u8, rs1: u8, imm: i64| -> u32 {
        ((imm as u32) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7) | 0x13
    };
    let mut prog = Vec::new();
    prog.extend_from_slice(&enc_addi(1, 0, 42).to_le_bytes());
    prog.extend_from_slice(&0x73u32.to_le_bytes());
    let pcs = akita_setup(4, 64, 1 << 23, [91u8; 32]).ok().unwrap();
    if let Ok((_out, envelope)) = prove_program(&pcs, &prog, &[], 16) {
        out.push(("zkvm-envelope-bytes".into(), envelope.to_bytes().len()));
    }
    // ZK sumcheck proof size (rounds + commitment + linear proof).
    {
        use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
        use lattice_core::transcript::Transcript;
        use lattice_ring::{Modulus32, RingConfig};
        use lattice_zk::entropy::{SecretSeed, ShakeStream};
        use lattice_zk::zk_sumcheck::{zk_prove, ZkSumcheckStatement};
        let num_vars = 3usize;
        let n = 1usize << num_vars;
        let m = lattice_zk::zk_sumcheck::required_slots(n);
        let params = AjtaiParams {
            ring: RingConfig::new(Modulus32::Q_32, 4).ok().unwrap(),
            k: 2,
            m,
            norm_bound: 1 << 24,
        };
        let pk = AjtaiPublicKey::from_seed(params, [7u8; 32]).ok().unwrap();
        let mle = DenseMle::random(num_vars, b"size-zk");
        let claim = mle.evaluations.iter().fold(Goldilocks::ZERO, |a, b| a.add(b));
        let mut stream = ShakeStream::new(SecretSeed::from_kat_label(b"size-zk"), b"m");
        let mut t = Transcript::new_default(b"lzx-zk-sumcheck");
        if let Ok((proof, _r, _v)) = zk_prove(&pk, &mle.evaluations, claim, &mut stream, &mut t) {
            let rounds_bytes = proof.rounds.len() * 16;
            let t1 = proof.commitment.to_bytes().len();
            let t2 = proof.carry_commitment.to_bytes().len();
            let mut linear = 0usize;
            for w in &proof.linear_proof.mask_commitments {
                for row in w {
                    linear += row.to_bytes().len();
                }
            }
            for z in &proof.linear_proof.responses {
                for e in z {
                    linear += e.to_bytes().len();
                }
            }
            out.push(("zk-sumcheck-rounds-bytes".into(), rounds_bytes));
            out.push(("zk-sumcheck-commitment-bytes".into(), t1 + t2));
            out.push(("zk-sumcheck-linear-proof-bytes".into(), linear));
        }
        let _ = ZkSumcheckStatement {
            num_vars,
            claim: Goldilocks::ZERO,
        };
    }
    out
}

// ---------------------------------------------------------------------------
// Main: emit the matrix
// ---------------------------------------------------------------------------

fn main() {
    let mut timings = Vec::new();
    timings.extend(bench_field());
    timings.extend(bench_ntt());
    timings.extend(bench_commitment());
    timings.extend(bench_sumcheck());
    timings.extend(bench_zk());
    timings.extend(bench_akita());
    timings.extend(bench_zkvm());
    // Wave 6 families.
    timings.extend(bench_folding());
    timings.extend(bench_lookup());
    timings.extend(bench_salsa());
    timings.extend(bench_wave6_substrate());

    println!("# LZX Benchmark Matrix\n");
    println!("| stage | median | min | reps |");
    println!("|---|---|---|---|");
    let mut csv = String::from("stage,median_ns,min_ns,reps\n");
    for t in &timings {
        println!(
            "| {} | {} | {} | {} |",
            t.name,
            fmt_ns(t.median_ns),
            fmt_ns(t.min_ns),
            t.reps
        );
        csv.push_str(&format!(
            "{},{},{},{}\n",
            t.name, t.median_ns, t.min_ns, t.reps
        ));
    }
    println!("\n## Size accounting\n");
    println!("| artifact | bytes |");
    println!("|---|---|");
    let mut csv_sizes = String::from("artifact,bytes\n");
    for (name, bytes) in size_accounting() {
        println!("| {name} | {bytes} |");
        csv_sizes.push_str(&format!("{name},{bytes}\n"));
    }
    // Write artifacts next to the binary's working directory.
    let _ = std::fs::create_dir_all("target/bench");
    let _ = std::fs::write("target/bench/lzx-benchmark-matrix.md", format!("{timings:#?}"));
    let _ = std::fs::write("target/bench/timings.csv", csv);
    let _ = std::fs::write("target/bench/sizes.csv", csv_sizes);
    println!("\nArtifacts: target/bench/timings.csv, target/bench/sizes.csv");
}
