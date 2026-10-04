//! Wave 7.5: the Recursive opening mode, wired end-to-end.

use lattice_labinius::params::N;
use lattice_labinius::scheme::{Opening, Prover, Verifier};
use lattice_labinius::*;

fn e2e(log_len: u32, cols: u32, extra: Vec<Modulus>, tag: [u8; 32]) {
    let params = Params::new(log_len, cols, extra, Opening::Recursive).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), tag);
    let (p, v) = (Prover::new(&pp), Verifier::new(&pp));
    let w = Witness::random(&params, tag);
    let (c, o) = p.commit(&w);
    let mut t = Transcript::new(b"labinius/recursive");
    let point = v.derive_evaluation_point(&mut t, &c);
    let row = w.row_evaluate(&point);
    let ch = v.derive_folding_challenges(&mut t, &row);
    // The recursive opening proof: v and its transforms never travel.
    let proof = match p.prove_recursive(&pp, &o, &row, &ch) {
        Ok(pr) => pr,
        Err(e) => panic!("prove_recursive: {e}"),
    };
    let claim = w.mle_evaluate(&point);
    assert!(
        v.verify_opening_recursive(&point, &claim, &row, &ch, &proof)
            .is_ok(),
        "the honest recursive round must verify"
    );
    // The claim identity is checked inside the recursive verify — a wrong
    // claim must fail.
    assert!(v
        .verify_opening_recursive(&point, &(claim + F162::ONE), &row, &ch, &proof)
        .is_err());
    // Tampered row → the digest desyncs → the LaBRADOR statement mismatch.
    let mut bad_row = row.clone();
    let vals = bad_row.values_mut();
    vals[0] = vals[0] + F162::ONE;
    assert!(v
        .verify_opening_recursive(&point, &claim, &bad_row, &ch, &proof)
        .is_err());
    let _ = c;
    let _ = N;
    let _ = &w;
}

#[test]
fn recursive_round_end_to_end_small() {
    e2e(10, 2, vec![Modulus::Q2917_Q_S], [7u8; 32]);
}

#[test]
fn recursive_round_end_to_end_default() {
    e2e(12, 3, vec![Modulus::Q9721_FS_S], [8u8; 32]);
}

#[test]
fn recursive_tampered_proof_rejected() {
    let params = Params::new(10, 2, vec![Modulus::Q2917_Q_S], Opening::Recursive).unwrap();
    let pp = PublicParameters::from_seed(params.clone(), [11u8; 32]);
    let (p, v) = (Prover::new(&pp), Verifier::new(&pp));
    let w = Witness::random(&params, [12u8; 32]);
    let (_c, o) = p.commit(&w);
    let mut t = Transcript::new(b"labinius/recursive-t");
    let point = v.derive_evaluation_point(&mut t, &_c);
    let row = w.row_evaluate(&point);
    let ch = v.derive_folding_challenges(&mut t, &row);
    let mut proof = p.prove_recursive(&pp, &o, &row, &ch).ok().unwrap();
    let claim = w.mle_evaluate(&point);
    // Tampered norm announcement → cap check fails.
    proof.norms[0] = u64::MAX;
    assert!(v
        .verify_opening_recursive(&point, &claim, &row, &ch, &proof)
        .is_err());
    // Tampered proof structure → the LaBRADOR verify rejects.
    let mut proof2 = p.prove_recursive(&pp, &o, &row, &ch).ok().unwrap();
    proof2.proof.digits.clear();
    assert!(v
        .verify_opening_recursive(&point, &claim, &row, &ch, &proof2)
        .is_err());
    // Tampered digit coefficient (out of the witness-coefficient range).
    let mut proof3 = p.prove_recursive(&pp, &o, &row, &ch).ok().unwrap();
    if let Some(d) = proof3.proof.digits.first_mut() {
        if let Some(c) = d.first_mut() {
            *c = i16::MAX;
        }
    }
    assert!(v
        .verify_opening_recursive(&point, &claim, &row, &ch, &proof3)
        .is_err());
}
