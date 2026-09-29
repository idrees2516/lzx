//! The reference round benchmark (port of labinius's `labinius` binary): commit -> derive the
//! point -> evaluate -> fold -> verify, per-stage medians over RUNS, plus proof-size floors.
//! `--suite xs|s|m|l` selects the witness size (2^14/2^16/2^18/2^20 F162).
use lattice_labinius::*;
use lattice_labinius::params::N;
use std::time::Instant;

fn median(v: &mut [u128]) -> f64 {
    v.sort();
    let n = v.len();
    if n % 2 == 1 { v[n / 2] as f64 } else { (v[n / 2 - 1] + v[n / 2]) as f64 / 2.0 }
}

fn main() {
    let suite = std::env::args().nth(1).unwrap_or_else(|| "m".into());
    let suite = Suite::from_flag(&suite).unwrap_or(&SUITES[2]);
    let runs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5).max(1);
    let params = Params::sized(suite, Opening::Clear);
    println!("labinius round bench — suite {} (2^{} F162, {} columns, moduli {:?})",
        suite.name, suite.witness_log_len, params.columns(),
        params.primes());
    let pp = PublicParameters::from_seed(params.clone(), [7u8; 32]);
    let witness = Witness::random(&params, [42u8; 32]);
    let (prover, verifier) = (Prover::new(&pp), Verifier::new(&pp));
    let mut t_commit = vec![]; let mut t_point = vec![]; let mut t_eval = vec![];
    let mut t_chal = vec![]; let mut t_fold = vec![]; let mut t_verify = vec![];
    let mut t_wire = vec![];
    let mut commit_bytes = 0usize; let mut opening_bytes = 0usize; let mut opening_raw = 0usize;
    let digest = wire_digest(&params);
    for _ in 0..runs {
        let t = Instant::now();
        let (commitment, opening) = prover.commit(&witness);
        t_commit.push(t.elapsed().as_nanos());
        commit_bytes = commitment.wire_bytes();
        let mut transcript = Transcript::new(b"labinius/reference");
        let t = Instant::now();
        let point = verifier.derive_evaluation_point(&mut transcript, &commitment);
        t_point.push(t.elapsed().as_nanos());
        let t = Instant::now();
        let claimed = witness.mle_evaluate(&point);
        let row = witness.row_evaluate(&point);
        t_eval.push(t.elapsed().as_nanos());
        let t = Instant::now();
        let challenges = verifier.derive_folding_challenges(&mut transcript, &row);
        t_chal.push(t.elapsed().as_nanos());
        let t = Instant::now();
        let folded = prover.fold(opening, &challenges);
        t_fold.push(t.elapsed().as_nanos());
        // the entropy-coded wire form of the folded opening (rANS, ~8-9 bits/coefficient)
        let t = Instant::now();
        let folded_wire = folded.to_wire(&digest).expect("encode");
        let _ = FoldedWitness::from_wire(&folded_wire, &digest).expect("decode");
        t_wire.push(t.elapsed().as_nanos());
        opening_bytes = folded_wire.len();
        opening_raw = folded.raw_wire_bytes();
        let t = Instant::now();
        verifier.verify_evaluation(&point, &claimed, &row).unwrap();
        let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
        let folded_row = verifier.fold_row_evaluation(&row, &challenges);
        verifier.verify_opening(&folded_commitment, &folded, &point, &folded_row).unwrap();
        t_verify.push(t.elapsed().as_nanos());
        let _ = &claimed;
    }
    println!("  commit    {:>10.3} ms", median(&mut t_commit) / 1e6);
    println!("  point     {:>10.3} ms", median(&mut t_point) / 1e6);
    println!("  evaluate  {:>10.3} ms", median(&mut t_eval) / 1e6);
    println!("  challenge {:>10.3} ms", median(&mut t_chal) / 1e6);
    println!("  fold      {:>10.3} ms", median(&mut t_fold) / 1e6);
    println!("  verify    {:>10.3} ms", median(&mut t_verify) / 1e6);
    println!("  wire      {:>10.3} ms  (rANS encode+decode of the folded opening)", median(&mut t_wire) / 1e6);
    println!("  sizes: commitment {commit_bytes} B, folded opening {opening_bytes} B \
        (raw i16 floor {opening_raw} B, {:.2} bits/coefficient), \
        row evaluation {} B, total proof {} B",
        8.0 * opening_bytes as f64 / (opening_raw as f64 / 2.0),
        24 * params.columns(), commit_bytes + opening_bytes + 24 * params.columns());
}

/// A deterministic digest binding the wire artifact to the round's parameters.
fn wire_digest(params: &Params) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, q) in params.primes().iter().enumerate() {
        d[2 * i..2 * i + 2].copy_from_slice(&q.to_le_bytes());
    }
    d[8..12].copy_from_slice(&(params.witness_len() as u32).to_le_bytes());
    d[12..16].copy_from_slice(&(params.columns() as u32).to_le_bytes());
    d
}

