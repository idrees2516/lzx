//! Benchmarks for ePrint 2026/538 (PCD via Holography Accumulation):
//! Π_GBF1/Π_GBF2 in both representations, Barebones (the SuperSpartan /
//! SuperMarlin recovery), Π_Fold chains, and the decider.

use lattice_holo::barebones::{barebones_prove, barebones_verify};
use lattice_holo::decider::decide;
use lattice_holo::fold::{fold_verify, fold_with_matrices};
use lattice_holo::gbf1::{gbf1_prove, gbf1_verify, Gbf1Statement};
use lattice_holo::gbf2::{gbf2_prove, gbf2_verify, Gbf2Statement};
use lattice_holo::pc::PcKey;
use lattice_holo::pcd::{pcd_decide, pcd_prove_step, pcd_verify_step};
use lattice_holo::poly::Domain;
use lattice_holo::relations::{Ccs, GbfInstance, GbfLeft, GbfRight, GbfWitness};
use lattice_holo::Fp256;
use lattice_core::transcript::Transcript;

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

fn fr(v: u64) -> Fp256 {
    Fp256::from_canonical_u64(v)
}

/// One GBF statement: s = u₁ᵀ(M₁v₁ + M₂v₁).
struct GbfCase {
    matrices: Vec<Vec<Vec<Fp256>>>,
    instance: GbfInstance,
    witness: GbfWitness,
}

fn build_case(domain: &Domain, seed: &[u8]) -> GbfCase {
    let n = domain.size();
    let matrices = vec![
        lattice_holo::poly::fp_matrix(b"bm", seed, n),
        lattice_holo::poly::fp_matrix(b"bm2", seed, n),
    ];
    let u1 = lattice_holo::poly::fp_vec(b"bu", seed, n);
    let v1 = lattice_holo::poly::fp_vec(b"bv", seed, n);
    let m1v = lattice_holo::poly::mat_vec(&matrices[0], &v1).ok().unwrap();
    let m2v = lattice_holo::poly::mat_vec(&matrices[1], &v1).ok().unwrap();
    let mut s = Fp256::ZERO;
    for r in 0..n {
        s = s.add(&u1[r].mul(&m1v[r].add(&m2v[r])));
    }
    let instance = GbfInstance {
        left: GbfLeft {
            constants: vec![fr(1)],
            sets: vec![vec![0]],
        },
        right: GbfRight {
            constants: vec![fr(1), fr(1)],
            sets: vec![vec![(0, 0)], vec![(1, 0)]],
        },
        u_commitments: vec![lattice_holo::pc::PcCommitment::identity()],
        v_commitments: vec![lattice_holo::pc::PcCommitment::identity()],
        matrix_commitments: Vec::new(),
        s,
        alpha: None,
        beta: None,
    };
    GbfCase {
        matrices,
        instance,
        witness: GbfWitness {
            us: vec![u1],
            vs: vec![v1],
        },
    }
}

fn bench_gbf(domain: &Domain, label: &str) {
    let seed = format!("{label}-seed").into_bytes();
    let case = build_case(domain, &seed);
    let key = PcKey::new(domain.clone(), &[71u8; 32]).ok().unwrap();
    // Commit u/v.
    let mut t = Transcript::new_default(label.as_bytes());
    let (uc, uw) = key
        .commit_encoding(&case.witness.us[0], &mut t)
        .ok()
        .unwrap();
    let (vc, vw) = key
        .commit_encoding(&case.witness.vs[0], &mut t)
        .ok()
        .unwrap();
    let mut inst = case.instance.clone();
    inst.u_commitments = vec![uc];
    inst.v_commitments = vec![vc];
    let wits = vec![GbfWitness {
        us: vec![uw.encoding],
        vs: vec![vw.encoding],
    }];
    // GBF2.
    let st2 = Gbf2Statement {
        domain: domain.clone(),
        instances: &[inst.clone()],
        witnesses: &wits,
        matrices: &case.matrices,
    };
    let t0 = now_ms();
    let mut tp = Transcript::new_default(b"bench");
    let (p2, _o2) = gbf2_prove(&st2, &mut tp).ok().unwrap();
    let t_prov2 = now_ms() - t0;
    let mut tv = Transcript::new_default(b"bench");
    let ok2 = gbf2_verify(domain, &[inst.clone()], &p2, &mut tv).is_ok();
    // GBF1.
    let d1 = lattice_holo::poly::mat_vec(&case.matrices[0], &wits[0].vs[0]).ok().unwrap();
    let d2 = lattice_holo::poly::mat_vec(&case.matrices[1], &wits[0].vs[0]).ok().unwrap();
    let mut t2 = Transcript::new_default(b"bench1");
    let (dc1, dw1) = key.commit_encoding(&d1, &mut t2).ok().unwrap();
    let (dc2, _dw2) = key.commit_encoding(&d2, &mut t2).ok().unwrap();
    let _ = dw1;
    let dcs = vec![dc1, dc2];
    let st1 = Gbf1Statement {
        domain: domain.clone(),
        instances: &[inst.clone()],
        witnesses: &wits,
        matrices: &case.matrices,
        d_commitments: &dcs,
    };
    let t0 = now_ms();
    let mut tp1 = Transcript::new_default(b"bench1");
    let (p1, _o1) = gbf1_prove(&st1, &mut tp1).ok().unwrap();
    let t_prov1 = now_ms() - t0;
    let mut tv1 = Transcript::new_default(b"bench1");
    let ok1 = gbf1_verify(domain, &[inst.clone()], &dcs, &p1, &mut tv1).is_ok();
    println!(
        "  {label} (n={}): GBF2 prove {t_prov2:7.1} ms (ok={ok2}) | GBF1 prove {t_prov1:7.1} ms (ok={ok1})",
        domain.size()
    );
}

fn bench_barebones(domain: &Domain, label: &str) {
    let seed = format!("{label}-b").into_bytes();
    let (ccs, x, w) = Ccs::random_with_solution(domain, 2, 6, 3, 3, 3, &seed);
    let key = PcKey::new(domain.clone(), &[72u8; 32]).ok().unwrap();
    let coms: Vec<_> = ccs
        .matrices
        .iter()
        .map(|m| key.commit_matrix(domain, m).ok().unwrap())
        .collect();
    let t0 = now_ms();
    let mut t = Transcript::new_default(b"bbb");
    let (proof, acc) = barebones_prove(&ccs, &key, &coms, &x, &w, &mut t).ok().unwrap();
    let t_prov = now_ms() - t0;
    let t0 = now_ms();
    let mut tv = Transcript::new_default(b"bbb");
    let ok = barebones_verify(&ccs, &key, &coms, &x, &proof, &mut tv).ok().unwrap();
    let t_ver = now_ms() - t0;
    // Fold: two accumulators.
    let mut t3 = Transcript::new_default(b"bbb3");
    let (_p2, acc2) = barebones_prove(&ccs, &key, &coms, &x, &w, &mut t3).ok().unwrap();
    let t0 = now_ms();
    let mut tf = Transcript::new_default(b"bbf");
    let (fp, facc) = fold_with_matrices(&key, &ccs.matrices, &coms, &[acc.clone(), acc2.clone()], &mut tf)
        .ok()
        .unwrap();
    let t_fold = now_ms() - t0;
    let mut tfv = Transcript::new_default(b"bbf");
    let _ = fold_verify(&key, &coms, &[acc, acc2], &fp, &mut tfv);
    // Decider.
    let t0 = now_ms();
    let mut td = Transcript::new_default(b"bbd");
    let okd = decide(&key, &[facc], std::slice::from_ref(&ccs.matrices), &[coms], &mut td)
        .ok()
        .unwrap();
    let t_dec = now_ms() - t0;
    println!(
        "  {label}: barebones prove {t_prov:7.1} ms | verify {t_ver:6.1} ms (ok={ok}) | fold {t_fold:6.1} ms | decide {t_dec:6.1} ms (ok={okd})"
    );
}

fn bench_pcd_chain(domain: &Domain, depth: usize) {
    let (ccs, _x, w0) = Ccs::random_with_solution(domain, 2, 6, 3, 3, 3, b"pcdb");
    let key = PcKey::new(domain.clone(), &[73u8; 32]).ok().unwrap();
    let coms: Vec<_> = ccs
        .matrices
        .iter()
        .map(|m| key.commit_matrix(domain, m).ok().unwrap())
        .collect();
    let t0 = now_ms();
    let mut accs = Vec::new();
    let mut proofs = Vec::new();
    for step in 0..depth as u64 {
        let x = vec![fr(step), fr(step + 1)];
        let w = {
            let mut ww = w0.clone();
            ww[0] = ww[0].add(&fr(step));
            ww
        };
        let mut t = Transcript::new_default(b"pcdb");
        t.append_message(b"step", &step.to_le_bytes()).ok().unwrap();
        let (proof, acc) = pcd_prove_step(&ccs, &key, &coms, &x, &w, &accs, &mut t)
            .ok()
            .unwrap();
        accs.push(acc);
        proofs.push(proof);
    }
    let t_prov = now_ms() - t0;
    // Verify each step + the decider.
    let t0 = now_ms();
    let mut all_ok = true;
    for step in 0..depth {
        let incoming: Vec<_> = accs[..step].to_vec();
        let mut tv = Transcript::new_default(b"pcdb");
        tv.append_message(b"step", &(step as u64).to_le_bytes()).ok().unwrap();
        let ok = pcd_verify_step(&ccs, &key, &coms, &incoming, &proofs[step], &accs[step], &mut tv)
            .ok()
            .unwrap();
        all_ok &= ok;
    }
    let mut td = Transcript::new_default(b"pcdbd");
    let okd = pcd_decide(&key, &ccs, &coms, &accs[depth - 1], &mut td).ok().unwrap();
    let t_ver = now_ms() - t0;
    println!(
        "  depth {depth}: chain prove {t_prov:8.1} ms | verify+decide {t_ver:7.1} ms | ok={all_ok}/{okd}"
    );
}

fn main() {
    println!("lattice-holo benchmarks (ePrint 2026/538) — BN254 Fr/G1, vector Pedersen");
    println!();
    println!("== Π_GBF1 / Π_GBF2 ==");
    bench_gbf(&Domain::Multivariate { num_vars: 2 }, "mv n=4");
    bench_gbf(&Domain::Multivariate { num_vars: 3 }, "mv n=8");
    bench_gbf(&Domain::Univariate { n: 4 }, "uv n=4");
    bench_gbf(&Domain::Univariate { n: 8 }, "uv n=8");
    println!();
    println!("== Barebones + Fold + Decider ==");
    bench_barebones(&Domain::Multivariate { num_vars: 3 }, "mv n=8");
    bench_barebones(&Domain::Univariate { n: 8 }, "uv n=8");
    println!();
    println!("== PCD chains ==");
    for depth in [2usize, 4] {
        bench_pcd_chain(&Domain::Multivariate { num_vars: 3 }, depth);
    }
}
