//! Benchmarks for ePrint 2026/289 (ZK-PCD from Accumulation Schemes):
//! the zk-accumulation prover/verifier/decider costs across relation types
//! (R1CS d=2, CCS d=q, permutation d=n) and chain depths, plus the PCD
//! step cost.
//!
//! Run: `cargo run --release --example pcd_bench [-- [depths...]]`

use lattice_core::transcript::Transcript;
use lattice_pcd::accum::{
    accumulate, create_base_accumulator_at, decide, num_vars_for, verify_accumulation,
};
use lattice_pcd::pcd::{
    chain_num_vars, prove_base, prove_step, verify_chain_step, IncomingEdge, PcdParams,
    PcdPredicate,
};
use lattice_pcd::pedersen::PedersenKey;
use lattice_pcd::sps::{derive_challenges, CcsSps, PermutationSps, R1csSps, SpsRelation};
use lattice_pcd::Fp256;

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

fn bench_r1cs() {
    println!("== zk-accumulation: R1CS (d = 2, µ = 1) ==");
    for (s, t, rows) in [(2usize, 6usize, 4usize), (4, 12, 8), (8, 24, 16)] {
        let rel = R1csSps::many_solutions(s, t, rows, b"bench-r1cs");
        let key = PedersenKey::derive(&[41u8; 32], 1 + s + t).ok().unwrap();
        let l = num_vars_for(3);
        let mut t0 = Transcript::new_default(b"bench");
        let base = create_base_accumulator_at(&rel, &key, l, &mut t0)
            .ok()
            .unwrap();
        // A satisfying pair.
        let (x, w) = rel.draw_solution(b"bw");
        let mut z = vec![Fp256::from_canonical_u64(1)];
        z.extend(x.iter().cloned());
        z.extend(w.iter().cloned());
        let mut tt = Transcript::new_default(b"bench-pair");
        let (mut inst, mut wit) = lattice_pcd::sps::commit_messages(&[z.clone()], &key, &mut tt)
            .ok()
            .unwrap();
        inst.x = x.clone();
        inst.challenges = derive_challenges(&inst).ok().unwrap();
        wit.messages = vec![z];
        let t_start = now_ms();
        let mut tp = Transcript::new_default(b"bench-acc");
        let (acc, pf) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            &[wit],
            std::slice::from_ref(&base),
            l,
            &mut tp,
        )
        .ok()
        .unwrap();
        let t_prove = now_ms() - t_start;
        let t_start = now_ms();
        let mut tv = Transcript::new_default(b"bench-acc");
        let ok = verify_accumulation(
            &rel,
            &key,
            &[inst],
            &[base.instance],
            &acc.instance,
            &pf,
            l,
            &mut tv,
        )
        .is_ok();
        let t_verify = now_ms() - t_start;
        let t_start = now_ms();
        let okd = decide(&rel, &key, &acc).is_ok();
        let t_decide = now_ms() - t_start;
        println!(
            "  s={s:2} t={t:2} rows={rows:2} n={n:2}: prove {t_prove:8.1} ms | verify {t_verify:6.1} ms | decide {t_decide:6.1} ms | ok={ok}/{okd}",
            n = rel.num_outputs()
        );
    }
}

fn bench_ccs() {
    println!("== zk-accumulation: CCS (high degree d = q) ==");
    for (rows, t_m, q, d) in [(4usize, 3usize, 3usize, 3usize), (6, 4, 4, 4), (8, 5, 6, 6)] {
        let mut rel = CcsSps::random(2, 8, rows, t_m, q, d, b"bench-ccs");
        rel.sample_satisfying_mut(b"bw");
        let (x, w) = {
            // after mutation the drawn witness satisfies
            let n = rel.s + rel.t;
            let mut z = lattice_pcd::util::fp_vec_from_seed(b"bench-ccs-w", b"bw", n);
            if z.iter().all(|v| v.is_zero()) {
                z[0] = Fp256::from_canonical_u64(1);
            }
            (z[..rel.s].to_vec(), z[rel.s..].to_vec())
        };
        let key = PedersenKey::derive(&[42u8; 32], rel.s + rel.t)
            .ok()
            .unwrap();
        let l = num_vars_for(3);
        let mut t0 = Transcript::new_default(b"bench");
        let base = create_base_accumulator_at(&rel, &key, l, &mut t0)
            .ok()
            .unwrap();
        let mut z = x.clone();
        z.extend(w.iter().cloned());
        let mut tt = Transcript::new_default(b"bench-pair2");
        let (mut inst, mut wit) = lattice_pcd::sps::commit_messages(&[z.clone()], &key, &mut tt)
            .ok()
            .unwrap();
        inst.x = x.clone();
        inst.challenges = derive_challenges(&inst).ok().unwrap();
        wit.messages = vec![z];
        let t_start = now_ms();
        let mut tp = Transcript::new_default(b"bench-acc2");
        let (acc, pf) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            &[wit],
            std::slice::from_ref(&base),
            l,
            &mut tp,
        )
        .ok()
        .unwrap();
        let t_prove = now_ms() - t_start;
        let mut tv = Transcript::new_default(b"bench-acc2");
        let ok = verify_accumulation(
            &rel,
            &key,
            &[inst],
            &[base.instance],
            &acc.instance,
            &pf,
            l,
            &mut tv,
        )
        .is_ok();
        println!(
            "  rows={rows} t_M={t_m} q={q} d={d} n={}: prove {t_prove:8.1} ms | ok={ok}",
            rel.num_outputs()
        );
    }
}

fn bench_permutation() {
    println!("== zk-accumulation: permutation (d = n, µ = 2, challenge-dependent) ==");
    for n in [4usize, 8, 12] {
        let rel = PermutationSps::random(n, b"bench-perm");
        let key = PedersenKey::derive(&[43u8; 32], n).ok().unwrap();
        let l = num_vars_for(3);
        let mut t0 = Transcript::new_default(b"bench");
        let base = create_base_accumulator_at(&rel, &key, l, &mut t0)
            .ok()
            .unwrap();
        // A satisfying pair: b random, a = π(b).
        let b = lattice_pcd::util::fp_vec_from_seed(b"bench-b", b"bp", n);
        let a = rel.apply(&b).ok().unwrap();
        let mut tt = Transcript::new_default(b"bench-pair3");
        let (mut inst, mut wit) =
            lattice_pcd::sps::commit_messages(&[a.clone(), b.clone()], &key, &mut tt)
                .ok()
                .unwrap();
        inst.challenges = derive_challenges(&inst).ok().unwrap();
        wit.messages = vec![a, b];
        let t_start = now_ms();
        let mut tp = Transcript::new_default(b"bench-acc3");
        let (acc, pf) = accumulate(
            &rel,
            &key,
            std::slice::from_ref(&inst),
            &[wit],
            std::slice::from_ref(&base),
            l,
            &mut tp,
        )
        .ok()
        .unwrap();
        let t_prove = now_ms() - t_start;
        let mut tv = Transcript::new_default(b"bench-acc3");
        let ok = verify_accumulation(
            &rel,
            &key,
            &[inst],
            &[base.instance],
            &acc.instance,
            &pf,
            l,
            &mut tv,
        )
        .is_ok();
        println!("  n={n:2} (d={n}): prove {t_prove:8.1} ms | ok={ok}");
    }
}

fn bench_pcd_chain(depth: usize) {
    println!("== ZK-PCD chain (arity 2, depth {depth}) ==");
    let t = 8usize;
    let msg_len = 4usize;
    let local_len = 2usize;
    let arity = 2usize;
    let rel = R1csSps::many_solutions(msg_len + local_len + arity * msg_len, t, 4, b"bench-pcd");
    let params = PcdParams {
        key: PedersenKey::derive(&[44u8; 32], 1 + rel.s + rel.t)
            .ok()
            .unwrap(),
    };
    let pred = PcdPredicate {
        relation: &rel,
        arity,
        msg_len,
        local_len,
    };
    let t_start = now_ms();
    let (mut proof, mut state) = prove_base(&pred, &params, b"bench-base").ok().unwrap();
    let mut z_prev: Option<Vec<Fp256>> = None;
    // The message/state that the FINAL step took as its real-edge input.
    let mut z_edge_input: Option<Vec<Fp256>> = None;
    let mut acc_edge_input: Option<lattice_pcd::accum::Accumulator> = None;
    for step in 0..depth {
        // Step 0 chains off the base state (both edges base); later steps
        // take one real edge and one base edge so the accumulator merges.
        let z: Vec<Fp256> = (0..msg_len)
            .map(|i| Fp256::from_canonical_u64((step * 7 + i) as u64 + 1))
            .collect();
        let zloc = vec![
            Fp256::from_canonical_u64(step as u64),
            Fp256::from_canonical_u64(3),
        ];
        let has_real = step > 0 && z_prev.is_some();
        let edges: Vec<IncomingEdge> = (0..arity)
            .map(|i| {
                if has_real && i == 0 {
                    IncomingEdge {
                        message: z_prev.clone(),
                        proof: Some(proof.clone()),
                        accumulator: Some(state.accumulator.clone()),
                    }
                } else {
                    IncomingEdge {
                        message: None,
                        proof: None,
                        accumulator: None,
                    }
                }
            })
            .collect();
        if !has_real {
            // Base-extension step: accumulate against the base accumulator
            // directly (the base node's message).
            let inputs: Vec<Option<Vec<Fp256>>> = edges.iter().map(|e| e.message.clone()).collect();
            let inst_x = pred.pack_instance(&z, &zloc, &inputs).ok().unwrap();
            let mut zw = vec![Fp256::from_canonical_u64(1)];
            zw.extend(inst_x.iter().cloned());
            zw.extend(lattice_pcd::util::fp_vec_from_seed(
                b"bench-pcd-w",
                &[step as u8],
                t,
            ));
            // Accumulate against the base state's accumulator via a real
            // edge carrying the base message.
            let edges_real: Vec<IncomingEdge> = vec![
                IncomingEdge {
                    message: None,
                    proof: None,
                    accumulator: Some(state.accumulator.clone()),
                },
                IncomingEdge {
                    message: None,
                    proof: None,
                    accumulator: Some(state.accumulator.clone()),
                },
            ];
            let (p, snew) = prove_step(
                &pred,
                &params,
                &z,
                &zloc,
                &edges_real,
                &[zw],
                &state,
                &[step as u8],
            )
            .ok()
            .unwrap();
            proof = p;
            acc_edge_input = Some(state.accumulator.clone());
            state = snew;
            z_edge_input = None;
            z_prev = Some(z);
            continue;
        }
        let inputs: Vec<Option<Vec<Fp256>>> = edges.iter().map(|e| e.message.clone()).collect();
        let inst_x = pred.pack_instance(&z, &zloc, &inputs).ok().unwrap();
        let mut zw = vec![Fp256::from_canonical_u64(1)];
        zw.extend(inst_x.iter().cloned());
        zw.extend(lattice_pcd::util::fp_vec_from_seed(
            b"bench-pcd-w",
            &[step as u8],
            t,
        ));
        z_edge_input = z_prev.clone();
        acc_edge_input = Some(state.accumulator.clone());
        let (p, snew) = prove_step(
            &pred,
            &params,
            &z,
            &zloc,
            &edges,
            &[zw],
            &state,
            &[step as u8],
        )
        .ok()
        .unwrap();
        proof = p;
        state = snew;
        z_prev = Some(z);
    }
    let t_prove = now_ms() - t_start;
    // Verify the final message: mirror the LAST step's edge structure
    // (one real edge carrying the previous message, one base edge).
    let z_final0 = z_prev.clone().unwrap_or_default();
    let zloc = vec![
        Fp256::from_canonical_u64((depth.saturating_sub(1)) as u64),
        Fp256::from_canonical_u64(3),
    ];
    let prev_msg = z_edge_input.clone();
    let edges: Vec<IncomingEdge> = (0..arity)
        .map(|i| {
            if i == 0 && prev_msg.is_some() {
                IncomingEdge {
                    message: prev_msg.clone(),
                    proof: None, // not consulted by the verifier
                    accumulator: Some(match &acc_edge_input {
                        Some(a) => a.clone(),
                        None => state.accumulator.clone(),
                    }),
                }
            } else {
                IncomingEdge {
                    message: None,
                    proof: None,
                    accumulator: None,
                }
            }
        })
        .collect();
    let z_final = z_final0;
    let t_start = now_ms();
    let ok = verify_chain_step(
        &pred,
        &params,
        &z_final,
        &zloc,
        &edges,
        &state,
        &proof,
        &[(depth.saturating_sub(1)) as u8],
    )
    .ok()
    .unwrap();
    let t_verify = now_ms() - t_start;
    println!(
        "  depth {depth}: chain prove {t_prove:8.1} ms | final verify {t_verify:6.1} ms | ok={ok} | L={}",
        chain_num_vars(arity)
    );
}

fn main() {
    println!("lattice-pcd benchmarks (ePrint 2026/289) — BN254 Fr/G1, vector Pedersen");
    println!();
    bench_r1cs();
    println!();
    bench_ccs();
    println!();
    bench_permutation();
    println!();
    for depth in [2usize, 4, 8] {
        bench_pcd_chain(depth);
    }
}
