//! Benchmarks for LatticeBlindFold: the end-to-end Π_LBF folding step at
//! the toy parameter sets, with per-stage timings and transcript-size
//! accounting (the communication rows of §4.3.3, measured at toy scale).

use lattice_blindfold::fp::Fq;
use lattice_blindfold::gauss::Rng;
use lattice_blindfold::params::{Params, SecurityBudget};
use lattice_blindfold::protocol::*;

fn build_setup(params: Params, seed: &[u8]) -> LbfSetup {
    let nf_circ = params.nf - params.nf_bl;
    let m2_circ: Vec<Vec<i64>> = (0..nf_circ)
        .map(|r| (0..nf_circ).map(|c| if r == c { 1 } else { 0 }).collect())
        .collect();
    let m3_circ = m2_circ.clone();
    let mut rng = Rng::new(seed);
    LbfSetup::setup(params, &m2_circ, &m3_circ, &mut rng)
}

fn bench_at(params: Params, label: &str) {
    let setup = build_setup(params.clone(), label.as_bytes());
    let mut rng = Rng::new(format!("{label}-run").as_bytes());
    // Witness: Boolean circuit part + the constant-one wire.
    let nf_circ = params.nf - params.nf_bl;
    let mut x = vec![Fq::ZERO; params.nf_in];
    x[0] = Fq::ONE;
    for slot in x.iter_mut().skip(1) {
        *slot = Fq::from_i64(rng.below(2) as i64);
    }
    let w: Vec<Fq> = (0..nf_circ - params.nf_in)
        .map(|_| Fq::from_i64(rng.below(2) as i64))
        .collect();

    let t0 = std::time::Instant::now();
    let (out, tr) = lbf_protocol(&setup, &x, &w, &mut rng).expect("protocol");
    let prove = t0.elapsed();

    let t1 = std::time::Instant::now();
    lbf_verify(&setup, &x, &out, &tr).expect("verify");
    let verify = t1.elapsed();

    // Communication accounting (the dominant rows of §4.3.3's table).
    let abdlop_coms: usize = 1 // coeffs
        + tr.r1cs.hint_coms.iter().map(|c| c.len()).sum::<usize>()
        + tr.r1cs.surr_coms.iter().map(|c| c.len()).sum::<usize>()
        + tr.r1cs.sq_coms.len()
        + tr.r1cs.cube_coms.len()
        + tr.r1cs.prod_coms.len()
        + 3 // the σ/degree-0/u⋆ wrapper garbage commitments (one each)
        + tr.r1cs.step15.as_ref().map(|t| t.garbage_com.len()).unwrap_or(0)
        + tr.rlc.cy0.len()
        + tr.rlc.cyj.len();
    let ajtai_coms = out.len() + 1;
    // The masked openings (z₁, z₂ vectors) dominate the proof size: one
    // entry per PoK invocation per block.
    let pok_blocks = tr.r1cs.step1.as_ref().map(|t| t.z1.len()).unwrap_or(0)
        + tr.r1cs.step7.inner_linear.as_ref().map(|t| t.z1.len()).unwrap_or(0)
        + tr.r1cs.step12.inner_linear.as_ref().map(|t| t.z1.len()).unwrap_or(0)
        + tr.r1cs.step15.as_ref().map(|t| t.inner.z1.len()).unwrap_or(0)
        + tr.r1cs.step18.inner_linear.as_ref().map(|t| t.z1.len()).unwrap_or(0)
        + tr.rlc.step1.as_ref().map(|t| t.z1.len()).unwrap_or(0)
        + tr.rlc.step17.as_ref().map(|t| t.z1.len()).unwrap_or(0)
        + tr.dec.step5.as_ref().map(|t| t.z1.len()).unwrap_or(0);
    let masked_openings_elems =
        pok_blocks * (params.m1 + params.m2) * params.d;

    println!("== {label} ==");
    println!(
        "  params: d={} nf=2^{} k={} (B=b^k={}) t={} K=1",
        params.d,
        params.nf.trailing_zeros(),
        params.k,
        params.b.pow(params.k as u32),
        params.t,
    );
    let bud = SecurityBudget::evaluate(&params);
    println!(
        "  budget: Wmax={} N_PoK={} interactive-bits={:.0} blinding-cap=2^{:.0}",
        bud.w_max, bud.n_pok, bud.bits_overall, bud.eps_blinding_cap
    );
    println!("  prove:  {prove:?}");
    println!("  verify: {verify:?}");
    println!(
        "  comms:  {abdlop_coms} ABDLOP commitments, {ajtai_coms} compact Ajtai, \
         ~{masked_openings_elems} ring elements of masked openings"
    );
    println!("  output: {} CE_com(b) instances at the fresh point", out.len());
}

fn main() {
    let only_medium = std::env::args().any(|a| a == "--medium-only");
    if !only_medium {
        bench_at(Params::toy(), "toy (d=4, nf=2^6)");
    }
    bench_at(Params::toy_medium(), "medium (d=8, nf=2^8)");
}
