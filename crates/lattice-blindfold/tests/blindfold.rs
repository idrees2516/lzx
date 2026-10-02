//! End-to-end integration tests for LatticeBlindFold (ePrint 2026/1857):
//! the full Π_LBF protocol (Protocol 12) over a blinded-layout R1CS
//! circuit, with adversarial tampering at every layer and the blinding
//! simulator.

use lattice_blindfold::embed::FieldVec;
use lattice_blindfold::fp::Fq;
use lattice_blindfold::gauss::Rng;
use lattice_blindfold::params::Params;
use lattice_blindfold::protocol::*;

/// Build a toy setup with a satisfiable circuit: rows read
/// z_i·z_i − z_i = 0 (Boolean witnesses) over the circuit columns.
fn toy_setup(seed: &[u8]) -> LbfSetup {
    let params = Params::toy();
    let nf_circ = params.nf - params.nf_bl;
    // M2° = M3° = identity over the circuit columns.
    let m2_circ: Vec<Vec<i64>> = (0..nf_circ)
        .map(|r| (0..nf_circ).map(|c| if r == c { 1 } else { 0 }).collect())
        .collect();
    let m3_circ = m2_circ.clone();
    let mut rng = Rng::new(seed);
    LbfSetup::setup(params, &m2_circ, &m3_circ, &mut rng)
}

/// A Boolean witness for the toy circuit with the constant-one wire.
fn toy_witness(setup: &LbfSetup, seed: &[u8]) -> (Vec<Fq>, Vec<Fq>) {
    let p = &setup.params;
    let nf_circ = p.nf - p.nf_bl;
    let mut rng = Rng::new(seed);
    let mut x = vec![Fq::ZERO; p.nf_in];
    x[0] = Fq::ONE; // the constant-one wire ι₁ = 0
    for slot in x.iter_mut().skip(1) {
        *slot = Fq::from_i64(rng.below(2) as i64);
    }
    let w: Vec<Fq> = (0..nf_circ - p.nf_in)
        .map(|_| Fq::from_i64(rng.below(2) as i64))
        .collect();
    (x, w)
}

#[test]
fn cecom_relation_roundtrip() {
    // A CEcom pair satisfies Definition 3.21 and tampering is detected.
    let setup = toy_setup(b"cecom");
    let mut rng = Rng::new(b"cecom-r");
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    for acc in &accs {
        cecom_check(&setup, acc, 2, 2).unwrap();
    }
    // Tamper the witness: the norm bound still holds for Gaussian pieces
    // (split_b pieces are b-bounded), but the Ajtai commitment breaks.
    let mut bad = accs[0].clone();
    bad.wit.z.0[0] = bad.wit.z.0[0].add(&Fq::ONE);
    assert!(cecom_check(&setup, &bad, 2, 2).is_err());
}

#[test]
fn r1cs_reduction_end_to_end() {
    // Protocol 6: fold 1 fresh + k accumulated into K+k pairs at r′.
    let setup = toy_setup(b"r1cs");
    let mut rng = Rng::new(b"r1cs-r");
    let (x, w) = toy_witness(&setup, b"r1cs-w");
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    // ι_bl: pad the witness with the blinding block.
    let e = setup.blinded.sample_blinding_block(2, &mut rng);
    let mut z1 = vec![Fq::ZERO; setup.params.nf];
    z1[..x.len()].copy_from_slice(&x);
    z1[x.len()..x.len() + w.len()].copy_from_slice(&w);
    let flat_e: Vec<Fq> = e.iter().flat_map(|pp| pp.0.iter().copied()).collect();
    z1[setup.params.nf - setup.params.nf_bl..].copy_from_slice(&flat_e);
    let z1 = FieldVec(z1);
    let fresh_c = vec![setup.ajtai.commit(&z1.to_ring(setup.params.d))];
    let (out, tr) = r1cs_reduction(&setup, &[z1], &accs, &mut rng).unwrap();
    // The output: K+k pairs at the new point r′, all CEcom-valid.
    assert_eq!(out.pairs.len(), 1 + accs.len());
    for pair in &out.pairs {
        cecom_check(&setup, pair, 2, 2).unwrap();
        assert_eq!(pair.inst.r, out.r_prime);
    }
    // Verification accepts.
    r1cs_verify(&setup, &fresh_c, &accs, &out, &tr).unwrap();
}

#[test]
fn rlc_reduction_end_to_end() {
    // Protocol 7: the rejection-sampled fold of K+k pairs into one.
    let setup = toy_setup(b"rlc");
    let mut rng = Rng::new(b"rlc-r");
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    let (out, tr) = rlc_reduction(&setup, &accs, &mut rng).unwrap();
    // The folded pair is CEcom-valid at the LARGER norm bound B = b^k
    // (the restricted relation of Definition 3.25: ∥z∥ ≤ τs < B).
    let bound = setup.params.b.pow(setup.params.k as u32);
    cecom_check(&setup, &out, bound, setup.params.b_fold()).unwrap();
    rlc_verify(&setup, &accs, &out, &tr).unwrap();
    // Tamper the folded commitment: the homomorphic check breaks.
    let mut bad_out = out.clone();
    bad_out.inst.coms[0].t_b[0] = bad_out.inst.coms[0].t_b[0].add(&lattice_blindfold::ring::Poly::one(setup.params.d));
    assert!(rlc_verify(&setup, &accs, &bad_out, &tr).is_err());
}

#[test]
fn dec_reduction_end_to_end() {
    // Protocol 8: split one norm-B claim into k norm-b claims with fresh
    // salts; c = Σ b^{i−1} c_i and the batched y_j identity hold.
    let setup = toy_setup(b"dec");
    let mut rng = Rng::new(b"dec-r");
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    let (folded, rlc_tr) = rlc_reduction(&setup, &accs, &mut rng).unwrap();
    let (pieces, tr) = dec_reduction(&setup, &folded, &mut rng).unwrap();
    assert_eq!(pieces.len(), setup.params.k);
    for p in &pieces {
        cecom_check(&setup, p, 2, 2).unwrap();
    }
    dec_verify(&setup, &folded, &pieces, &tr).unwrap();
    // Tamper one piece's commitment: the recomposition check breaks.
    let mut bad = pieces.clone();
    bad[0].inst.c[0] = bad[0].inst.c[0].add(&lattice_blindfold::ring::Poly::one(setup.params.d));
    assert!(dec_verify(&setup, &folded, &bad, &tr).is_err());
    let _ = rlc_tr;
}

#[test]
fn lbf_protocol_end_to_end() {
    // Protocol 12: the full Π_LBF = Π'_DEC ∘ Π'_RLC ∘ Π'_R1CS with the
    // ι_bl precomposition — one blinding folding step over a finished
    // circuit.
    let setup = toy_setup(b"lbf");
    let mut rng = Rng::new(b"lbf-r");
    let (x, w) = toy_witness(&setup, b"lbf-w");
    let (out, tr) = lbf_protocol(&setup, &x, &w, &mut rng).unwrap();
    assert_eq!(out.len(), setup.params.k);
    for pair in &out {
        cecom_check(&setup, pair, 2, 2).unwrap();
    }
    lbf_verify(&setup, &x, &out, &tr).unwrap();
}

#[test]
fn lbf_rejects_unsatisfied_circuit() {
    // A witness violating the R1CS rows must fail the reduction (the
    // F-term does not vanish on the cube; the anchoring breaks).
    let setup = toy_setup(b"lbf-bad");
    let mut rng = Rng::new(b"lbf-bad-r");
    let (mut x, mut w) = toy_witness(&setup, b"lbf-bad-w");
    // Break Booleanity: a coordinate of 2 violates z²−z = 0.
    if w.len() > 1 {
        w[1] = Fq::from_i64(2);
    } else {
        x[1] = Fq::from_i64(2);
    }
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    let e = setup.blinded.sample_blinding_block(2, &mut rng);
    let mut z1 = vec![Fq::ZERO; setup.params.nf];
    z1[..x.len()].copy_from_slice(&x);
    z1[x.len()..x.len() + w.len()].copy_from_slice(&w);
    let flat_e: Vec<Fq> = e.iter().flat_map(|pp| pp.0.iter().copied()).collect();
    z1[setup.params.nf - setup.params.nf_bl..].copy_from_slice(&flat_e);
    let z1 = FieldVec(z1);
    // The sumcheck's claimed sum T is computed from the honest relations;
    // with a violated constraint the anchoring target mismatches — the
    // reduction must fail (the round consistency or the σ-anchoring).
    let res = r1cs_reduction(&setup, &[z1], &accs, &mut rng);
    assert!(res.is_err(), "an unsatisfied circuit must not fold");
}

#[test]
fn lbf_rejects_norm_violation() {
    // A witness with ∥z∥∞ ≥ b must fail the CEcom relation downstream.
    let setup = toy_setup(b"lbf-norm");
    let mut rng = Rng::new(b"lbf-norm-r");
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    // Corrupt one accumulator witness coordinate beyond b.
    let mut bad = accs.clone();
    bad[0].wit.z.0[0] = Fq::from_i64(5);
    assert!(cecom_check(&setup, &bad[0], 2, 2).is_err());
    // The honest protocol still folds the ORIGINAL accumulators.
    let (out, tr) = rlc_reduction(&setup, &accs, &mut rng).unwrap();
    let bound = setup.params.b.pow(setup.params.k as u32);
    cecom_check(&setup, &out, bound, setup.params.b_fold()).unwrap();
    let _ = tr;
}

// ---------------------------------------------------------------------------
// The blinding simulator (Theorem 4.13's transcript-level check).
// ---------------------------------------------------------------------------

#[test]
fn simulator_transcripts_accept() {
    // The S_ABDLOP-style simulator produces accepting transcripts with
    // no access to the secrets (Protocols 1-3's hybrid chain endpoint).
    use lattice_blindfold::abdlop::AbdlopPp;
    use lattice_blindfold::pok::{simulate_linear_pok, RelRow, RelRows};
    use lattice_blindfold::rk::PolyK;

    let mut rng = Rng::new(b"sim");
    let pp = AbdlopPp::setup(3, 6, 4, 6, 4, &mut rng);
    let rows = RelRows {
        rows: vec![RelRow::const_coeff(
            vec![(0, 0, PolyK::one(4))],
            lattice_blindfold::fq2::K::ZERO,
            4,
        )],
    };
    let u = vec![PolyK::zero(4)];
    let (coms, tr) = simulate_linear_pok(&pp, 1, &rows, &u, (300.0, 300.0), 6.0, 1, &mut rng);
    assert!(lattice_blindfold::pok::verify_linear(&pp, &coms, &rows, &u, &tr).is_ok());
}

#[test]
fn lbf_protocol_medium_params() {
    // The medium toy (d = 8, nf = 2^8) — exercises the deeper ring and
    // the larger decomposition depth.
    let params = Params::toy_medium();
    let nf_circ = params.nf - params.nf_bl;
    let m2_circ: Vec<Vec<i64>> = (0..nf_circ)
        .map(|r| (0..nf_circ).map(|c| if r == c { 1 } else { 0 }).collect())
        .collect();
    let m3_circ = m2_circ.clone();
    let mut rng = Rng::new(b"med");
    let setup = LbfSetup::setup(params, &m2_circ, &m3_circ, &mut rng);
    let p = &setup.params;
    let mut x = vec![Fq::ZERO; p.nf_in];
    x[0] = Fq::ONE;
    for slot in x.iter_mut().skip(1) {
        *slot = Fq::from_i64(rng.below(2) as i64);
    }
    let w: Vec<Fq> = (0..nf_circ - p.nf_in)
        .map(|_| Fq::from_i64(rng.below(2) as i64))
        .collect();
    let (out, tr) = lbf_protocol(&setup, &x, &w, &mut rng).unwrap();
    for pair in &out {
        cecom_check(&setup, pair, 2, 2).unwrap();
    }
    lbf_verify(&setup, &x, &out, &tr).unwrap();
}

// debug the medium c-recomposition

#[test]
fn accumulator_free_variant() {
    // Corollary 4.24: Π°_LBF folds the single fresh pair with NO
    // prover-sampled accumulators — still complete, sound and blinding.
    let setup = toy_setup(b"af");
    let mut rng = Rng::new(b"af-r");
    let (x, w) = toy_witness(&setup, b"af-w");
    let (out, tr) = lbf_accumulator_free(&setup, &x, &w, &mut rng).unwrap();
    assert_eq!(out.len(), setup.params.k);
    for pair in &out {
        cecom_check(&setup, pair, 2, 2).unwrap();
    }
    lbf_verify(&setup, &x, &out, &tr).unwrap();
}

#[test]
fn fold_blueprint_optional_branch() {
    // Protocol 13: with supplied accumulators the blueprint runs the
    // same reductions on the carried pairs.
    let setup = toy_setup(b"bp");
    let mut rng = Rng::new(b"bp-r");
    let (x, w) = toy_witness(&setup, b"bp-w");
    let accs = sample_blind_cecom(&setup, &mut rng).unwrap();
    let (out, tr) = lbf_fold_blueprint(&setup, &x, &w, Some(accs), &mut rng).unwrap();
    assert_eq!(out.len(), setup.params.k);
    lbf_verify(&setup, &x, &out, &tr).unwrap();
    // The None branch reduces to Π_LBF.
    let (out2, tr2) = lbf_fold_blueprint(&setup, &x, &w, None, &mut rng).unwrap();
    assert_eq!(out2.len(), setup.params.k);
    lbf_verify(&setup, &x, &out2, &tr2).unwrap();
}
