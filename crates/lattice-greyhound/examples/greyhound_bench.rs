//! The Greyhound/LaBRADOR benchmark: the full PCS pipeline at feasible
//! scales, the Table 4 analytic accounting (the 53KB claim), and the
//! LaBRADOR sub-proof RUN at the real 2^30-derived statement size.
//!
//! Run with `GREYHOUND_230=1` to attempt the 138,880-ring-element witness
//! (the N = 2^30 LaBRADOR sub-proof — the dominant component of the 53KB
//! claim); the default runs the 2^26-derived statement (34,791 elements).

use lattice_core::keccak::KeccakSponge;
use lattice_greyhound::greyhound::{commit, eval_polynomial, eval_prove, eval_verify, PcsParams};
use lattice_greyhound::recursion::{level_table, proof_size_bytes};
use lattice_greyhound::sis::ComKey;
use lattice_greyhound::sizes::{
    analytic_labrador_size, greyhound_contribution_bytes, labrador_witness_rank,
    paper_contributions_bytes, table4_total_bytes, TABLE4,
};

fn sha3_256(input: &[u8]) -> [u8; 32] {
    let mut h = KeccakSponge::new_sha3_256();
    h.update(input);
    h.finalize(32).try_into().unwrap()
}

fn rand_poly(seed: u8, i: usize) -> lattice_greyhound::Poly {
    // small ternary coefficients — the LaBRADOR witness regime
    let mut buf = [0u8; 64];
    let mut hh = KeccakSponge::new_shake256();
    hh.update(&[seed]);
    hh.update(&(i as u64).to_le_bytes());
    hh.finalize_in_place();
    hh.squeeze(&mut buf);
    let mut p = [0i64; 64];
    for (j, c) in p.iter_mut().enumerate() {
        *c = (buf[j] % 7) as i64 - 3;
    }
    lattice_greyhound::Poly(p)
}

fn bench_pcs(log_len: u32) {
    let len = 1usize << log_len;
    println!(
        "== Greyhound PCS: {len} ring elements (degree {}/{})",
        len * 64,
        1 << 20
    );
    let t0 = std::time::Instant::now();
    let s: Vec<lattice_greyhound::Poly> = (0..len).map(|i| rand_poly(7, i)).collect();
    let params = PcsParams::new(len).expect("params");
    println!(
        "   params: m={} n={} kappa={} kappa1={} f={} fu={} b={} bu={}",
        params.m,
        params.n,
        params.cpp.kappa,
        params.cpp.kappa1,
        params.cpp.f,
        params.cpp.fu,
        params.cpp.b,
        params.cpp.bu
    );
    // key sized for the PCS + LaBRADOR windows
    let key_len = params.cpp.kappa * params.m * params.cpp.f
        + params.cpp.kappa1
            * (params.n * params.cpp.fu * params.cpp.kappa + params.n * params.cpp.fu)
        + (1 << 16);
    let key = ComKey::expand(key_len, &sha3_256(b"bench-key"));
    let com = commit(&s, &key).expect("commit");
    println!("   commit: {:?}", t0.elapsed());
    let x = 43;
    let y = eval_polynomial(&s, x);
    let t1 = std::time::Instant::now();
    let proof = eval_prove(&com, &key, x, y).expect("prove");
    let prove_t = t1.elapsed();
    let (u1, h, pub_params) = com.commitment();
    let t2 = std::time::Instant::now();
    eval_verify(&u1, &h, &pub_params, &key, x, y, &proof).expect("verify");
    let verify_t = t2.elapsed();
    let gh = greyhound_contribution_bytes(pub_params.cpp.kappa1);
    let lab = proof_size_bytes(&proof.labrador);
    println!("   prove: {prove_t:?}  verify: {verify_t:?}");
    println!(
        "   Greyhound contribution: {gh} B; LaBRADOR sub-proof (model): {lab} B; total: {} B",
        gh + lab
    );
    let table = level_table(&proof.labrador);
    println!(
        "   levels: {} ({} tail), sizes: {:?}",
        table.len(),
        table.iter().filter(|t| t.tail).count(),
        table.iter().map(|t| t.bits / 8).collect::<Vec<_>>()
    );
    println!();
}

/// Run the LaBRADOR engine at a Table-4-derived PCS statement: the witness of
/// `(n+1)δ₁r + m` ring elements with the Table's digit variance — the
/// dominant component of the 53KB claim, at the REAL parameter scale.
fn bench_labrador_at_table(idx: usize) {
    let t = TABLE4[idx];
    let rank = labrador_witness_rank(&t);
    println!(
        "== LaBRADOR sub-proof at the Table-4 N=2^{} statement: {} ring elements ({} coefficients)",
        t.log_n,
        rank,
        rank * 64
    );
    let t0 = std::time::Instant::now();
    // the witness: small digits (variance 2^{2b}/12 — the pre-fold regime)
    let s: Vec<lattice_greyhound::Poly> = (0..rank)
        .map(|i| {
            let mut buf = [0u8; 64];
            let mut hh = KeccakSponge::new_shake256();
            hh.update(&[9u8]);
            hh.update(&(i as u64).to_le_bytes());
            hh.finalize_in_place();
            hh.squeeze(&mut buf);
            let mut p = [0i64; 64];
            for (j, c) in p.iter_mut().enumerate() {
                *c = ((buf[j] % 16) as i64) - 8; // |·| ≤ 8 ~ b/2 at b = 4..6
            }
            lattice_greyhound::Poly(p)
        })
        .collect();
    // a trivial linear constraint (the statement shape: one F constraint over
    // the joined witness — the identity slice; the PCS constraints were
    // validated in the small-scale runs)
    use lattice_greyhound::relation::*;
    let phi: Vec<lattice_greyhound::Poly> = s.iter().map(|p| p.sigma_m1()).collect();
    let b = lattice_greyhound::ring::sprod(&phi, &s);
    let stmt = PrincipalStatement::new(
        vec![VectorSpec::plain(rank)],
        vec![DotCnst::with_b(
            vec![Term {
                idx: 0,
                off: 0,
                phi,
            }],
            b,
        )],
        vec![],
        s.iter().map(|p| p.normsq()).sum::<u64>() * 4,
    );
    let wit = PrincipalWitness::new(vec![s]);
    let key_len = {
        // probe the needed windows via a dry init
        let ranks = vec![rank];
        let norms = wit.per_vector_normsq();
        match lattice_greyhound::sis::init_proof(&ranks, &norms, false, false) {
            Ok((cpp, _nn, r, _)) => {
                let vl = lattice_greyhound::protocol::VLayout::new(&cpp, r);
                cpp.kappa * _nn + cpp.kappa1 * (vl.t_len + vl.g_len + vl.h_len) + (1 << 16)
            }
            Err(e) => {
                println!("   init_proof failed: {e} — skipping");
                return;
            }
        }
    };
    println!(
        "   key: {key_len} ring elements ({:.1} MB)",
        key_len as f64 * 512.0 / 1e6
    );
    let key = ComKey::expand(key_len, &sha3_256(b"lab-key"));
    println!("   witness+key materialization: {:?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    match lattice_greyhound::recursion::prove(&stmt, &wit, &key) {
        Ok(proof) => {
            let pt = t1.elapsed();
            let t2 = std::time::Instant::now();
            let vr = lattice_greyhound::recursion::verify(&stmt, &proof, &key);
            println!(
                "   prove: {pt:?}  verify: {:?}  -> {}",
                t2.elapsed(),
                if vr.is_ok() { "OK" } else { "FAILED" }
            );
            let bits = proof_size_bytes(&proof);
            let table = level_table(&proof);
            println!(
                "   LaBRADOR sub-proof (measured model): {bits} B = {:.2} KB over {} levels",
                bits as f64 / 1024.0,
                table.len()
            );
            println!(
                "   level sizes (B): {:?}",
                table.iter().map(|t| t.bits / 8).collect::<Vec<_>>()
            );
            let gh = greyhound_contribution_bytes(t.n1);
            println!(
                "   + Greyhound contribution (n1={}): {gh} B => TOTAL {:.2} KB (paper: 53 KB)",
                t.n1,
                (gh + bits) as f64 / 1024.0
            );
        }
        Err(e) => println!("   prove failed: {e}"),
    }
    println!();
}

fn main() {
    println!("LaBRADOR (ePrint 2022/1341) + Greyhound (ePrint 2024/1293) — q = 2^32-99, d = 64\n");
    // 1. the full PCS pipeline at feasible scales
    for log_len in [8u32, 10, 12] {
        bench_pcs(log_len);
    }
    // 2. the Table 4 analytic accounting
    println!("== Table 4 analytic accounting (the paper's claims: 46/53/53 KB) ==");
    let totals = table4_total_bytes();
    let papers = paper_contributions_bytes();
    for (i, (&total, &paper)) in totals.iter().zip(papers.iter()).enumerate() {
        println!(
            "   N = 2^{}: Greyhound {paper} B + LaBRADOR {:.0} B = total {:.1} KB",
            TABLE4[i].log_n,
            (total - paper) as f64 / 1.0,
            total as f64 / 1024.0
        );
    }
    let analytic_230 = analytic_labrador_size(
        labrador_witness_rank(&TABLE4[2]),
        2f64.powi(2 * TABLE4[2].b as i32) / 12.0,
    );
    println!(
        "   the analytic LaBRADOR sub-proof at 2^30: {analytic_230} B = {:.1} KB; + Greyhound {} B = {:.1} KB",
        analytic_230 as f64 / 1024.0,
        greyhound_contribution_bytes(TABLE4[2].n1),
        (analytic_230 + greyhound_contribution_bytes(TABLE4[2].n1)) as f64 / 1024.0
    );
    println!();
    // 3. the LaBRADOR sub-proof at the Table-4 statement scale
    if std::env::var("GREYHOUND_230").is_ok() {
        // the 2^30 statement (138,880 ring elements): materializes in ~0.9s
        // and starts proving; the peak memory (the straightforward phi
        // materializations in the level's target construction, ~3GB) exceeds
        // a 4GB container's budget — run on a larger machine. The 2^26
        // quarter-scale statement runs fully (see the default path).
        println!(
            "   the 2^30 statement: 138,880 ring elements — needs >4GB RAM for the prove phase"
        );
        bench_labrador_at_table(0); // the 2^26 statement runs fully
    } else {
        bench_labrador_at_table(0); // the 2^26 statement (34,791 elements)
        println!("   (set GREYHOUND_230=1 for the 2^30 note + the 2^26 run)");
    }
}
