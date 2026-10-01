//! TTRP benchmark: prove/verify timings, proof size, and the verifier's
//! tensor-structured evaluation vs the naive O(m̄r) row materialisation
//! (the paper's Table 1 comparison axis).

use lattice_core::transcript::Transcript;
use lattice_ring::{Modulus32, RingConfig, RingElement};
use lattice_ttrp::cores::{sample_cores, TtrpParams};
use lattice_ttrp::projection::{cf_vec, conj, ct, materialize_row_q, project_integer, project_ring};
use lattice_ttrp::protocol::{mle_eval_ring, prove, verify, TtrpStatement};
use std::time::Instant;

fn cen(c: i64, q: u32) -> i64 {
    let qi = q as i64;
    let c = c.rem_euclid(qi);
    if c > qi / 2 {
        c - qi
    } else {
        c
    }
}

fn bench(params: &TtrpParams, tag: &str) {
    let ring = RingConfig::new(Modulus32::Q_32, params.phi_log).ok().unwrap();
    // Deterministic small witness.
    let mut st = 0x1234_5678_u64 | 1;
    let mut coeffs_all = Vec::new();
    for _ in 0..(params.m_bar() * ring.n()) {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let v = (st % 9) as i64 - 4;
        coeffs_all.push(v.rem_euclid(ring.modulus.q as i64) as u32);
    }
    let v: Vec<RingElement> = coeffs_all
        .chunks(ring.n())
        .map(|c| RingElement::from_coeffs(&ring, c.to_vec()))
        .collect();
    let x = cf_vec(&v);
    let qi = ring.modulus.q as i64;
    let norm: u64 = (x
        .iter()
        .map(|&c| {
            let w = cen(c, ring.modulus.q);
            (w * w) as f64
        })
        .sum::<f64>()
        .sqrt()
        .ceil()) as u64;

    let stmt = TtrpStatement {
        params: params.clone(),
        ring: ring.clone(),
        bound_b: norm.max(1),
        statement_digest: Transcript::hash_domain(b"ttrp-bench", tag.as_bytes()),
    };

    let t0 = Instant::now();
    let mut tp = Transcript::new_default(b"ttrp-bench");
    let proof = match prove(&stmt, &v, &mut tp) {
        Ok(p) => p,
        Err(e) => {
            println!("{tag}: prove error {e:?}; b_hat_sq={}", stmt.b_hat_squared());
            return;
        }
    };
    let prove_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let mut tv = Transcript::new_default(b"ttrp-bench");
    let verified = verify(&stmt, &proof, &mut tv).ok().unwrap();
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

    // Outer claim check.
    let conj_r: Vec<_> = verified.challenges.iter().map(conj).collect();
    let want = mle_eval_ring(&v, &conj_r).ok().unwrap();
    assert_eq!(want, verified.w_r);

    // Proof size: y0 + y1 + rounds + w_r + challenges (transmitted parts).
    let bytes = proof.y0.len() * 4
        + proof.y1.len() * ring.n() * 4
        + proof.rounds.len() * 3 * ring.n() * 4
        + ring.n() * 4;

    // Verifier cost comparison: tensor-structured evaluation (inside
    // verify, reported as verify_ms) vs materialising the k TT rows
    // (the unstructured-JL verifier's matrix-processing cost).
    let t2 = Instant::now();
    let seed = [0u8; 32];
    let rows = sample_cores(params, &seed);
    let mut acc = 0u64;
    for cores in &rows {
        let row_q = materialize_row_q(params, cores, ring.modulus.q);
        for &c in &row_q {
            acc = acc.wrapping_add(c as u64);
        }
    }
    let naive_ms = t2.elapsed().as_secs_f64() * 1e3;

    // Projection self-consistency (ct identity spot check).
    let v_bar: Vec<_> = v.iter().map(conj).collect();
    let y0 = project_integer(params, &rows, ring.modulus.q, &x);
    let y = project_ring(params, &rows, &ring, &v_bar).ok().unwrap();
    let ident = (0..params.k).all(|j| ct(&y[j]) as i64 == y0[j].rem_euclid(qi));

    println!(
        "{tag}: m_bar=2^{} phi=2^{} d={} mu1={} mu2={} c={} k={} k'={} | prove {prove_ms:.1} ms, verify {verify_ms:.1} ms (tensor), naive-row materialisation {naive_ms:.1} ms, proof {bytes} B, ct-identity {ident}",
        params.nu, params.phi_log, params.d(), params.mu1, params.mu2, params.c, params.k, params.k1
    );
}

fn main() {
    // Small: phi=16, m_bar=2^8, d=4, mu=6.
    bench(
        &TtrpParams { phi_log: 4, nu: 8, ell: 2, mu1: 4, mu2: 2, c: 4, k: 16, k1: 2 },
        "small",
    );
    // Mid: phi=64, m_bar=2^12, d=4, mu=9.
    bench(
        &TtrpParams { phi_log: 6, nu: 12, ell: 2, mu1: 6, mu2: 3, c: 8, k: 32, k1: 4 },
        "mid",
    );
    // Large-ish: phi=64, m_bar=2^14, d=4, mu=11 (k' at 32-bit-q lambda/128->4).
    bench(
        &TtrpParams { phi_log: 6, nu: 14, ell: 2, mu1: 7, mu2: 3, c: 8, k: 48, k1: 4 },
        "large",
    );
}
