//! End-to-end integration: the Greyhound PCS + the LaBRADOR engine + the R1CS
//! front ends, with adversarial cases.

use lattice_greyhound::greyhound::{commit, eval_polynomial, eval_prove, eval_verify, PcsParams};
use lattice_greyhound::r1cs::{binary_r1cs_reduce, r1cs_mod_reduce};
use lattice_greyhound::recursion::prove as labrador_prove;
use lattice_greyhound::recursion::verify as labrador_verify;
use lattice_greyhound::relation::{
    DotCnst, PrincipalStatement, PrincipalWitness, Term, VectorSpec,
};
use lattice_greyhound::ring::{sprod, Poly, N};
use lattice_greyhound::sis::ComKey;

fn sha3_256(input: &[u8]) -> [u8; 32] {
    let mut h = lattice_core::keccak::KeccakSponge::new_sha3_256();
    h.update(input);
    h.finalize(32).try_into().unwrap()
}

fn rand_small(seed: u8, i: usize) -> Poly {
    let mut buf = [0u8; 64];
    let mut hh = lattice_core::keccak::KeccakSponge::new_shake256();
    hh.update(&[seed]);
    hh.update(&(i as u64).to_le_bytes());
    hh.finalize_in_place();
    hh.squeeze(&mut buf);
    let mut p = [0i64; 64];
    for (j, c) in p.iter_mut().enumerate() {
        *c = (buf[j] % 7) as i64 - 3;
    }
    Poly(p)
}

#[test]
fn greyhound_pcs_end_to_end_tamper_matrix() {
    let len = 128;
    let s: Vec<Poly> = (0..len).map(|i| rand_small(7, i)).collect();
    let params = PcsParams::new(len).unwrap();
    let key_len = params.cpp.kappa * params.m * params.cpp.f
        + params.cpp.kappa1
            * (params.n * params.cpp.fu * params.cpp.kappa + params.n * params.cpp.fu)
        + (1 << 16);
    let key = ComKey::expand(key_len, &sha3_256(b"it-key"));
    let com = commit(&s, &key).unwrap();
    let (u1, h, pub_params) = com.commitment();
    let x = 43;
    let y = eval_polynomial(&s, x);
    let proof = eval_prove(&com, &key, x, y).unwrap();

    // the honest roundtrip
    eval_verify(&u1, &h, &pub_params, &key, x, y, &proof).unwrap();

    // 1. a wrong evaluation point
    assert!(eval_verify(&u1, &h, &pub_params, &key, x.wrapping_add(1), y, &proof).is_err());
    // 2. a wrong claimed value
    assert!(eval_verify(&u1, &h, &pub_params, &key, x, y.wrapping_add(1), &proof).is_err());
    // 3. a tampered u2 (the D-commitment to ŵ)
    let mut bad = proof.clone();
    bad.u2[0] = bad.u2[0].add(&Poly::constant(1));
    assert!(eval_verify(&u1, &h, &pub_params, &key, x, y, &bad).is_err());
    // 4. a tampered LaBRADOR final witness
    let mut bad2 = proof.clone();
    if let Some(v) = bad2.labrador.final_witness.s.first_mut() {
        if let Some(p) = v.first_mut() {
            *p = p.add(&Poly::constant(1));
        }
    }
    assert!(eval_verify(&u1, &h, &pub_params, &key, x, y, &bad2).is_err());
    // 5. a tampered tail piece (the tail's inner commitments — the E3 check)
    let mut bad3 = proof.clone();
    if let Some(p) = bad3.labrador.tail.u1.first_mut() {
        *p = p.add(&Poly::constant(1));
    }
    assert!(eval_verify(&u1, &h, &pub_params, &key, x, y, &bad3).is_err());
    // 6. a tampered JL projection (level or tail — wherever it lives)
    let mut bad4 = proof.clone();
    let mut tampered = false;
    if let Some(lp) = bad4.labrador.levels.first_mut() {
        lp.p[0] += 1;
        tampered = true;
    }
    if !tampered {
        bad4.labrador.tail.p[0] += 1;
    }
    assert!(eval_verify(&u1, &h, &pub_params, &key, x, y, &bad4).is_err());
    // 7. a tampered lift polynomial b'' (level or tail)
    let mut bad5 = proof.clone();
    if let Some(lp) = bad5.labrador.levels.first_mut() {
        if let Some(b) = lp.bb.first_mut() {
            *b = b.add(&Poly::constant(1));
        }
    } else if let Some(b) = bad5.labrador.tail.bb.first_mut() {
        *b = b.add(&Poly::constant(1));
    }
    assert!(eval_verify(&u1, &h, &pub_params, &key, x, y, &bad5).is_err());
    // 8. a wrong commitment key
    let key2 = ComKey::expand(key_len, &sha3_256(b"other-key"));
    assert!(eval_verify(&u1, &h, &pub_params, &key2, x, y, &proof).is_err());
}

#[test]
// The full-engine run is scale-sensitive (the quadratic joining's garbage
// (rr²+rr)/2 needs the padded part rank to dominate — the reference's
// dachshund front end avoids this via its 3-block structure). The reduction
// itself is always checked; the full run is exercised in greyhound_bench.
#[ignore = "scale-sensitive: run via `cargo test -- --ignored` or the bench"]
fn binary_r1cs_full_labrador_roundtrip() {
    // a binary R1CS padded to 128 constraints × 256 variables (the level
    // machinery needs the witness to dominate the garbage: m ≤ 1.1·nn)
    let (a, b, c, w) = {
        let k = 2048;
        let n = 2048;
        let _one = vec![1u8; n];
        let zero = vec![0u8; n];
        let mut am = vec![zero.clone(); k];
        let mut bm = vec![zero.clone(); k];
        let mut cm = vec![zero.clone(); k];
        // the real constraints: rows 0/1 (w0·w1 = w2, w2·w3 = w0); the rest
        // are tautologies (0·0 = 0)
        am[0][0] = 1;
        bm[0][1] = 1;
        cm[0][2] = 1;
        am[1][2] = 1;
        bm[1][3] = 1;
        cm[1][0] = 1;
        let mut wv = vec![1u8; n];
        wv[3] = 1;
        (am, bm, cm, wv)
    };
    let key = ComKey::expand(1 << 15, &sha3_256(b"r1cs-key"));
    let (stmt, wit, _t, gs) = binary_r1cs_reduce(&a, &b, &c, &w, &key, 0, 16, b"it-r1cs").unwrap();
    assert!(gs.iter().all(|g| g % 2 == 0));
    // the reduction statement is satisfiable (always checked)
    stmt.check_all(&wit.s).unwrap();
    // the full LaBRADOR proof of the R1CS-derived statement (the ignored path)
    let proof = labrador_prove(&stmt, &wit, &key).unwrap();
    labrador_verify(&stmt, &proof, &key).unwrap();
    // a tampered final witness is rejected
    let mut bad = proof.clone();
    if let Some(v) = bad.final_witness.s.first_mut() {
        if let Some(p) = v.first_mut() {
            *p = p.add(&Poly::constant(1));
        }
    }
    assert!(labrador_verify(&stmt, &bad, &key).is_err());
    // a wrong witness fails at the reduction
    let mut w2 = w.clone();
    w2[2] = 0;
    assert!(binary_r1cs_reduce(&a, &b, &c, &w2, &key, 0, 16, b"it-r1cs").is_err());
}

#[test]
#[ignore = "scale-sensitive: run via `cargo test -- --ignored` or the bench"]
fn r1cs_mod_2p64_full_labrador_roundtrip() {
    // w0·w1 = w2, w2·w3 = w5 (mod 2^64+1) padded to 128×256 (0·0 = 0 rows)
    let (a, b, c, w) = {
        let k = 2048;
        let n = 2048;
        let mut am = vec![vec![0u64; n]; k];
        let mut bm = vec![vec![0u64; n]; k];
        let mut cm = vec![vec![0u64; n]; k];
        am[0][0] = 1;
        bm[0][1] = 1;
        cm[0][2] = 1;
        am[1][2] = 1;
        bm[1][3] = 1;
        cm[1][5] = 1;
        let mut wv = vec![0u64; n];
        wv[0] = 3;
        wv[1] = 5;
        wv[2] = 15;
        wv[3] = 7;
        wv[5] = 105;
        (am, bm, cm, wv)
    };
    let key = ComKey::expand(1 << 19, &sha3_256(b"r1csmod-key"));
    let (stmt, wit, _t, gjs) = r1cs_mod_reduce(&a, &b, &c, &w, &key, 0, 2, b"it-r1csmod").unwrap();
    assert!(gjs.iter().all(|g| g.is_zero()));
    let proof = labrador_prove(&stmt, &wit, &key).unwrap();
    labrador_verify(&stmt, &proof, &key).unwrap();
}

#[test]
fn labrador_engine_standalone_roundtrip() {
    // a mixed F/F' statement with quadratic terms (the E4 path)
    let mk = |seed: u64| -> Vec<Poly> {
        (0..512)
            .map(|i| {
                let mut p = [0i64; N];
                for (j, cc) in p.iter_mut().enumerate() {
                    *cc = ((i * 37 + j * 17 + seed as usize * 13) % 7) as i64 - 3;
                }
                Poly(p)
            })
            .collect()
    };
    let s0 = mk(1);
    let s1 = mk(2);
    // f = 5·⟨s0, s1⟩ + ⟨φ, s0⟩ − b (quadratic)
    let phi = mk(3);
    // the off-diagonal a-entry is evaluated with the symmetric factor 2
    let quad = sprod(&s0, &s1).scale(10);
    let lin = sprod(&phi, &s0);
    let b = quad.add(&lin);
    // an F' constraint with the honest ct
    let phi2 = mk(4);
    let val2 = sprod(&phi2, &s1);
    let mut b2 = mk(9).pop().unwrap();
    b2.0[0] = val2.constant_term();
    let stmt = PrincipalStatement::new(
        vec![VectorSpec::plain(512), VectorSpec::plain(512)],
        vec![DotCnst {
            terms: vec![Term {
                idx: 0,
                off: 0,
                phi,
            }],
            a: vec![(0, 1, Poly::constant(5))],
            b: Some(b),
            ct_only: false,
        }],
        vec![DotCnst {
            terms: vec![Term {
                idx: 1,
                off: 0,
                phi: phi2,
            }],
            a: vec![],
            b: Some(b2),
            ct_only: true,
        }],
        u32::MAX as u64,
    );
    let wit = PrincipalWitness::new(vec![s0, s1]);
    let key = ComKey::expand(1 << 16, &sha3_256(b"quad-key"));
    let proof = labrador_prove(&stmt, &wit, &key).unwrap();
    labrador_verify(&stmt, &proof, &key).unwrap();
    // tamper the quadratic coefficient — rejected
    let stmt_bad = PrincipalStatement::new(
        stmt.vectors.clone(),
        vec![DotCnst {
            terms: stmt.cnst[0].terms.clone(),
            a: vec![(0, 1, Poly::constant(6))],
            b: stmt.cnst[0].b,
            ct_only: false,
        }],
        stmt.ct_cnst.clone(),
        stmt.betasq,
    );
    assert!(labrador_verify(&stmt_bad, &proof, &key).is_err());
}

#[test]
fn size_model_reproduces_the_paper_regime() {
    // the Table 4 analytic accounting: near-constant 34-53 KB across
    // 2^26..2^30 (the paper's 46/53/53 KB regime)
    let totals = lattice_greyhound::sizes::table4_total_bytes();
    let kbs: Vec<f64> = totals.iter().map(|&t| t as f64 / 1024.0).collect();
    assert!(
        kbs.iter().all(|&kb| (20.0..130.0).contains(&kb)),
        "totals out of regime: {kbs:?}"
    );
    assert!(kbs[2] / kbs[0] < 2.5, "not near-constant: {kbs:?}");
}
