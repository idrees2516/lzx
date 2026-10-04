//! Ring-LogUp (Construction 5.11 of ePrint 2026/471): the
//! log-derivatives lookup PIOP over the split ring
//! `R ≅ F_{q^{d/2}} × F_{q^{d/2}}`.
//!
//! The relation (Lemma 5.8) — with tags from the challenge space `C`,
//! the bijection `g`, multiplicities `m ∈ Z_q^N`:
//!
//! ```text
//! Σ_{i∈[M]} 1/(a_i + c_i·x₁ − x₂)  =  Σ_{j∈[N]} m[b_j]/(b_j + g(j)·x₁ − x₂)
//! ```
//!
//! over the total ring of fractions. The tags close the zero-divisor
//! attacks of Section 4: the CRT-uniqueness argument in the ⇐=
//! direction needs each denominator's two slots to pin a single index
//! map.
//!
//! PIOP flow:
//! 1. the multiplicity oracle `m̂` is sent;
//! 2. the verifier samples `α, β ← C`;
//! 3. the prover sends `Â`, `B̂` with
//!    `Â(bin i) = 1/(a_i + c_i·α − β)` and
//!    `B̂(bin j) = m[b_j]/(b_j + g(j)·α − β)` — **inversions over the
//!    CRT slots** (an element is invertible iff both slots are
//!    nonzero; Remark 5.10's `Θ(1−2/q^{d/2})` success probability is
//!    handled by salted resampling on zero-divisor abort);
//! 4. `v = ΣÂ = ΣB̂` plus two sum-checks;
//! 5. two zero-checks
//!    `Σ EQ(x,γ)(Â(â+αĉ−β) − 1) = 0` and
//!    `Σ EQ(x,δ)(B̂(b̂+αĝ−β) − m̂) = 0`;
//! 6. the integer check on `m` (Construction B.15);
//! 7. the binary check on `c` (Construction B.19).

use crate::ring_d::{Elem, RingD};
use crate::ring_sumcheck::{
    prove_sumcheck, verify_sumcheck, RingFactor, RingSumcheckProof, RingTerm, RingVirtualPoly,
};
use crate::subprotocols::{
    prove_binary_check, prove_integer_check, verify_binary_check, verify_integer_check, SubError,
};
use lattice_core::transcript::Transcript;

/// The oracle view for the plain-PIOP verifier.
#[derive(Debug, Clone)]
pub struct LogupOracles {
    pub a: Vec<Elem>,
    pub b: Vec<Elem>,
    pub c: Vec<Elem>,
    pub g_vec: Vec<Elem>,
    pub m: Vec<Elem>,
    pub big_a: Vec<Elem>,
    pub big_b: Vec<Elem>,
}

impl LogupOracles {
    pub fn eval(&self, ring: &RingD, label: &str, point: &[Elem]) -> Result<Elem, String> {
        let v: &[Elem] = match label {
            "lu-a" => &self.a,
            "lu-b" => &self.b,
            "lu-c" => &self.c,
            "lu-gN" => &self.g_vec,
            "lu-m" => &self.m,
            "lu-A" => &self.big_a,
            "lu-B" => &self.big_b,
            _ => {
                if let Some(j) = label.strip_prefix("lu-bc-cf") {
                    let j: usize = j.parse().map_err(|_| "bad cf index")?;
                    let row: Vec<Elem> = self
                        .c
                        .iter()
                        .map(|e| ring.constant(e.coeffs()[j]))
                        .collect();
                    return ring.mle_eval(&row, point).map_err(|e| format!("{e:?}"));
                }
                return Err(format!("unknown oracle label {label}"));
            }
        };
        ring.mle_eval(v, point).map_err(|e| format!("{e:?}"))
    }
}

#[derive(Debug, Clone)]
pub struct RingLogupProof {
    pub alpha: Elem,
    pub beta: Elem,
    /// ΣÂ = ΣB̂.
    pub v: Elem,
    pub sc_a: RingSumcheckProof,
    pub sc_b: RingSumcheckProof,
    pub zc_a: RingSumcheckProof,
    pub zc_b: RingSumcheckProof,
    pub gamma: Vec<Elem>,
    pub delta: Vec<Elem>,
    pub int_check: crate::subprotocols::IntegerCheckProof,
    pub binary: crate::subprotocols::BinaryCheckProof,
    /// The number of (α,β) resamples before all denominators inverted.
    pub inv_retries: u32,
}

fn absorb_vec(ring: &RingD, label: &[u8], v: &[Elem], tr: &mut Transcript) {
    let mut buf = Vec::with_capacity(v.len() * ring.d * 8);
    for e in v {
        for &c in e.coeffs() {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    let _ = tr.append_bytes(label, &buf);
}

fn cube_sum(ring: &RingD, v: &[Elem]) -> Elem {
    let mut acc = ring.zero();
    for e in v {
        acc = ring.add(&acc, e);
    }
    acc
}

/// Derive the rational-term vectors: `A_i = 1/(a_i + c_i·α − β)` and
/// `B_j = m_j·(b_j + g(j)·α − β)^{-1}`. Returns `None` on a
/// zero-divisor denominator (Remark 5.10: probability `≈ 2/q^{d/2}`).
#[allow(clippy::too_many_arguments)]
pub fn derive_terms(
    ring: &RingD,
    a: &[Elem],
    c: &[Elem],
    b: &[Elem],
    g_vec: &[Elem],
    m: &[Elem],
    alpha: &Elem,
    beta: &Elem,
) -> Option<(Vec<Elem>, Vec<Elem>)> {
    let mut den_a = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        let t = ring.mul(alpha, &c[i]);
        let d = ring.sub(&ring.add(&a[i], &t), beta);
        den_a.push(d);
    }
    let inv_a = ring.batch_inv(&den_a)?;
    let mut den_b = Vec::with_capacity(b.len());
    for j in 0..b.len() {
        let t = ring.mul(alpha, &g_vec[j]);
        let d = ring.sub(&ring.add(&b[j], &t), beta);
        den_b.push(d);
    }
    let inv_b = ring.batch_inv(&den_b)?;
    let big_a = inv_a;
    let big_b: Vec<Elem> = (0..b.len()).map(|j| ring.mul(&m[j], &inv_b[j])).collect();
    Some((big_a, big_b))
}

/// Prove `(a, b, c) ∈ RILU` via Ring-LogUp.
#[allow(clippy::too_many_lines)]
pub fn prove_ring_logup(
    ring: &RingD,
    a: &[Elem],
    b: &[Elem],
    c: &[Elem],
    transcript: &mut Transcript,
) -> Result<(RingLogupProof, LogupOracles), SubError> {
    let m_len = a.len();
    let n = b.len();
    if c.len() != m_len {
        return Err(SubError::Shape("c length mismatch".into()));
    }
    if !n.is_power_of_two() || !m_len.is_power_of_two() {
        return Err(SubError::Shape("M, N must be powers of two".into()));
    }
    if m_len > (1usize << ring.d) || n > (1usize << ring.d) {
        return Err(SubError::Shape("M, N must be < |C| = 2^d".into()));
    }
    // fail-closed: valid indexed lookup — every (a_i, c_i) pair must
    // appear as some (b_j, g(j)) pair. Duplicate table VALUES are fine
    // (the g-map disambiguates); duplicate (value, tag) pairs cannot
    // occur since g is injective.
    for (i, ai) in a.iter().enumerate() {
        if !c[i].is_binary() {
            return Err(SubError::Verify("c_i not in C".into()));
        }
        let ok = b
            .iter()
            .enumerate()
            .any(|(j, bj)| ai == bj && c[i] == ring.g_map(j as u64));
        if !ok {
            return Err(SubError::Verify(format!("a_{i} unmatched")));
        }
    }
    // The multiplicity vector (positional): m[j] = #{i : (a_i, c_i) == (b_j, g(j))}.
    let mut m: Vec<Elem> = vec![ring.zero(); n];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            if ai == bj && c[i] == ring.g_map(j as u64) {
                let next = m[j].ct() + 1;
                m[j] = ring.constant(next);
            }
        }
    }
    let g_vec: Vec<Elem> = (0..n).map(|j| ring.g_map(j as u64)).collect();
    absorb_vec(ring, b"lu-m", &m, transcript);
    // Challenges (α, β) with salted resampling on zero-divisors.
    let mut inv_retries = 0u32;
    let (alpha, beta, big_a, big_b) = loop {
        let mut salted = Transcript::new_default(b"lu-chal");
        let _ = salted.append_bytes(b"retry", &inv_retries.to_le_bytes());
        let alpha = ring.sample_challenge(&mut salted, b"lu-alpha");
        let beta = ring.sample_challenge(&mut salted, b"lu-beta");
        if let Some((ba, bb)) = derive_terms(ring, a, c, b, &g_vec, &m, &alpha, &beta) {
            break (alpha, beta, ba, bb);
        }
        inv_retries += 1;
        if inv_retries > 64 {
            return Err(SubError::Verify(
                "denominator inversion retries exceeded".into(),
            ));
        }
    };
    // NOTE: the retry salt is derived from a fixed-label transcript;
    // the (alpha, beta, retries) triple travels in the proof so the
    // verifier replays the same derivation.
    absorb_vec(ring, b"lu-A", &big_a, transcript);
    absorb_vec(ring, b"lu-B", &big_b, transcript);
    let v = cube_sum(ring, &big_a);
    if v != cube_sum(ring, &big_b) {
        return Err(SubError::Verify("completeness: ΣA ≠ ΣB".into()));
    }
    let _ = transcript.append_bytes(b"lu-v", &elem_bytes(ring, &v));
    // The two sum-checks.
    let log_m = m_len.trailing_zeros() as usize;
    let log_n = n.trailing_zeros() as usize;
    let poly_a = RingVirtualPoly {
        num_vars: log_m,
        claimed_sum: v.clone(),
        terms: vec![RingTerm {
            coeff: ring.one(),
            factors: vec![RingFactor::Mle(big_a.clone())],
        }],
    };
    let sc_a = prove_sumcheck(ring, &poly_a, transcript)?;
    let poly_b = RingVirtualPoly {
        num_vars: log_n,
        claimed_sum: v.clone(),
        terms: vec![RingTerm {
            coeff: ring.one(),
            factors: vec![RingFactor::Mle(big_b.clone())],
        }],
    };
    let sc_b = prove_sumcheck(ring, &poly_b, transcript)?;
    // The zero-check challenges γ ∈ C^{logM}, δ ∈ C^{logN}.
    let gamma: Vec<Elem> = (0..log_m)
        .map(|i| ring.sample_challenge(transcript, format!("lu-gamma-{i}").as_bytes()))
        .collect();
    let delta: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("lu-delta-{i}").as_bytes()))
        .collect();
    // Zero-check 1: Σ EQ(x,γ)(Â(x)(â(x) + α·ĉ(x) − β) − 1) = 0.
    let eq_g = ring.eq_row(&gamma);
    let neg_beta = ring.neg(&beta);
    let zc_a_poly = RingVirtualPoly {
        num_vars: log_m,
        claimed_sum: ring.zero(),
        terms: vec![
            RingTerm {
                coeff: ring.one(),
                factors: vec![
                    RingFactor::Eq(eq_g.clone()),
                    RingFactor::Mle(big_a.clone()),
                    RingFactor::Mle(a.to_vec()),
                ],
            },
            RingTerm {
                coeff: alpha.clone(),
                factors: vec![
                    RingFactor::Eq(eq_g.clone()),
                    RingFactor::Mle(big_a.clone()),
                    RingFactor::Mle(c.to_vec()),
                ],
            },
            RingTerm {
                coeff: neg_beta.clone(),
                factors: vec![RingFactor::Eq(eq_g.clone()), RingFactor::Mle(big_a.clone())],
            },
            RingTerm {
                coeff: ring.neg(&ring.one()),
                factors: vec![RingFactor::Eq(eq_g)],
            },
        ],
    };
    let zc_a = prove_sumcheck(ring, &zc_a_poly, transcript)?;
    // Zero-check 2: Σ EQ(x,δ)(B̂(b̂ + α·ĝ − β) − m̂) = 0.
    let eq_d = ring.eq_row(&delta);
    let zc_b_poly = RingVirtualPoly {
        num_vars: log_n,
        claimed_sum: ring.zero(),
        terms: vec![
            RingTerm {
                coeff: ring.one(),
                factors: vec![
                    RingFactor::Eq(eq_d.clone()),
                    RingFactor::Mle(big_b.clone()),
                    RingFactor::Mle(b.to_vec()),
                ],
            },
            RingTerm {
                coeff: alpha.clone(),
                factors: vec![
                    RingFactor::Eq(eq_d.clone()),
                    RingFactor::Mle(big_b.clone()),
                    RingFactor::Mle(g_vec.clone()),
                ],
            },
            RingTerm {
                coeff: neg_beta,
                factors: vec![RingFactor::Eq(eq_d.clone()), RingFactor::Mle(big_b.clone())],
            },
            RingTerm {
                coeff: ring.neg(&ring.one()),
                factors: vec![RingFactor::Eq(eq_d), RingFactor::Mle(m.clone())],
            },
        ],
    };
    let zc_b = prove_sumcheck(ring, &zc_b_poly, transcript)?;
    // The integer check on m (amplified).
    let (int_check, _iq) = prove_integer_check(ring, &m, 4, transcript)?;
    // The binary check on c.
    let (binary, _bq) = prove_binary_check(ring, c, transcript)?;
    let oracles = LogupOracles {
        a: a.to_vec(),
        b: b.to_vec(),
        c: c.to_vec(),
        g_vec,
        m,
        big_a,
        big_b,
    };
    Ok((
        RingLogupProof {
            alpha,
            beta,
            v,
            sc_a,
            sc_b,
            zc_a,
            zc_b,
            gamma,
            delta,
            int_check,
            binary,
            inv_retries,
        },
        oracles,
    ))
}

fn elem_bytes(ring: &RingD, e: &Elem) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ring.d * 8);
    for &c in e.coeffs() {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    buf
}

/// Verify the Ring-LogUp proof against the oracle view.
#[allow(clippy::too_many_lines)]
pub fn verify_ring_logup(
    ring: &RingD,
    m_len: usize,
    n: usize,
    proof: &RingLogupProof,
    oracles: &LogupOracles,
    transcript: &mut Transcript,
) -> Result<(), SubError> {
    if !n.is_power_of_two() || !m_len.is_power_of_two() {
        return Err(SubError::Shape("M, N must be powers of two".into()));
    }
    let log_m = m_len.trailing_zeros() as usize;
    let log_n = n.trailing_zeros() as usize;
    absorb_vec(ring, b"lu-m", &oracles.m, transcript);
    // Replay the (α, β) derivation with the same retry salt.
    let mut salted = Transcript::new_default(b"lu-chal");
    let _ = salted.append_bytes(b"retry", &proof.inv_retries.to_le_bytes());
    let alpha = ring.sample_challenge(&mut salted, b"lu-alpha");
    let beta = ring.sample_challenge(&mut salted, b"lu-beta");
    if alpha != proof.alpha || beta != proof.beta {
        return Err(SubError::Verify("(α, β) replay mismatch".into()));
    }
    // (The A/B oracles are absorbed as the caller's commitments in the
    // compiled layer; the plain PIOP absorbs their bytes.)
    absorb_vec(ring, b"lu-A", &oracles.big_a, transcript);
    absorb_vec(ring, b"lu-B", &oracles.big_b, transcript);
    let _ = transcript.append_bytes(b"lu-v", &elem_bytes(ring, &proof.v));
    // Sum-check 1.
    let shape_a = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_m,
        terms: vec![crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 1,
        }],
    };
    verify_sumcheck(
        ring,
        &shape_a,
        &proof.v,
        &proof.sc_a,
        transcript,
        &mut |ti, fi, pt| {
            let _ = (ti, fi);
            oracles.eval(ring, "lu-A", pt)
        },
    )?;
    // Sum-check 2.
    let shape_b = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_n,
        terms: vec![crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 1,
        }],
    };
    verify_sumcheck(
        ring,
        &shape_b,
        &proof.v,
        &proof.sc_b,
        transcript,
        &mut |ti, fi, pt| {
            let _ = (ti, fi);
            oracles.eval(ring, "lu-B", pt)
        },
    )?;
    // Zero-check challenges.
    let gamma: Vec<Elem> = (0..log_m)
        .map(|i| ring.sample_challenge(transcript, format!("lu-gamma-{i}").as_bytes()))
        .collect();
    let delta: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(transcript, format!("lu-delta-{i}").as_bytes()))
        .collect();
    if gamma != proof.gamma || delta != proof.delta {
        return Err(SubError::Verify("γ/δ replay mismatch".into()));
    }
    let eq_g = ring.eq_row(&gamma);
    let eq_d = ring.eq_row(&delta);
    // Zero-check 1: terms [EQ·A·a] + α[EQ·A·c] − β[EQ·A] − [EQ].
    let terms_a = vec![
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 3,
        },
        crate::ring_sumcheck::RingTermShape {
            coeff: alpha.clone(),
            num_factors: 3,
        },
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&beta),
            num_factors: 2,
        },
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&ring.one()),
            num_factors: 1,
        },
    ];
    let shape_zc_a = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_m,
        terms: terms_a,
    };
    let eqg_ref = &eq_g;
    verify_sumcheck(
        ring,
        &shape_zc_a,
        &ring.zero(),
        &proof.zc_a,
        transcript,
        &mut |ti, fi, pt| match (ti, fi) {
            (0, 0) | (1, 0) | (2, 0) | (3, 0) => {
                ring.mle_eval(eqg_ref, pt).map_err(|e| format!("{e:?}"))
            }
            (0, 1) | (1, 1) | (2, 1) => oracles.eval(ring, "lu-A", pt),
            (0, 2) => oracles.eval(ring, "lu-a", pt),
            (1, 2) => oracles.eval(ring, "lu-c", pt),
            _ => Err("bad factor".into()),
        },
    )?;
    // Zero-check 2: terms [EQ·B·b] + α[EQ·B·g] − β[EQ·B] − [EQ·m].
    let terms_b = vec![
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 3,
        },
        crate::ring_sumcheck::RingTermShape {
            coeff: alpha.clone(),
            num_factors: 3,
        },
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&beta),
            num_factors: 2,
        },
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&ring.one()),
            num_factors: 2,
        },
    ];
    let shape_zc_b = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_n,
        terms: terms_b,
    };
    let eqd_ref = &eq_d;
    verify_sumcheck(
        ring,
        &shape_zc_b,
        &ring.zero(),
        &proof.zc_b,
        transcript,
        &mut |ti, fi, pt| match (ti, fi) {
            (0, 0) | (1, 0) | (2, 0) | (3, 0) => {
                ring.mle_eval(eqd_ref, pt).map_err(|e| format!("{e:?}"))
            }
            (0, 1) | (1, 1) | (2, 1) => oracles.eval(ring, "lu-B", pt),
            (0, 2) => oracles.eval(ring, "lu-b", pt),
            (1, 2) => oracles.eval(ring, "lu-gN", pt),
            (3, 1) => oracles.eval(ring, "lu-m", pt),
            _ => Err("bad factor".into()),
        },
    )?;
    // Integer check on m.
    verify_integer_check(ring, n, &proof.int_check, transcript, &|l, pt| match l {
        "ic-a" => oracles.eval(ring, "lu-m", pt),
        _ => Err("bad label".into()),
    })?;
    // Binary check on c.
    verify_binary_check(ring, m_len, &proof.binary, transcript, &|l, pt| match l {
        "bc-f" => oracles.eval(ring, "lu-c", pt),
        other => oracles.eval(ring, &format!("lu-{other}"), pt),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    fn build_case(r: &RingD, m: usize, n: usize, seed: &str) -> (Vec<Elem>, Vec<Elem>, Vec<Elem>) {
        let b: Vec<Elem> = (0..n)
            .map(|j| r.random(format!("{seed}-b{j}").as_bytes()))
            .collect();
        let mut a = Vec::with_capacity(m);
        let mut c = Vec::with_capacity(m);
        for i in 0..m {
            let j = (i * 5 + 3) % n;
            a.push(b[j].clone());
            c.push(r.g_map(j as u64));
        }
        (a, b, c)
    }

    #[test]
    fn logup_end_to_end() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "rt");
        let mut tr = Transcript::new_default(b"lu");
        let (proof, oracles) =
            prove_ring_logup(&r, &a, &b, &c, &mut tr).unwrap_or_else(|e| panic!("{e:?}"));
        let mut tr2 = Transcript::new_default(b"lu");
        verify_ring_logup(&r, 4, 4, &proof, &oracles, &mut tr2)
            .unwrap_or_else(|e| panic!("verify: {e:?}"));
    }

    #[test]
    fn logup_larger_instance() {
        let r = ring();
        let (a, b, c) = build_case(&r, 8, 8, "big");
        let mut tr = Transcript::new_default(b"lu2");
        let (proof, oracles) = prove_ring_logup(&r, &a, &b, &c, &mut tr).ok().unwrap();
        let mut tr2 = Transcript::new_default(b"lu2");
        verify_ring_logup(&r, 8, 8, &proof, &oracles, &mut tr2)
            .ok()
            .unwrap();
    }

    #[test]
    fn logup_tampered_oracles_rejected() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "tam");
        let (proof, mut oracles) = {
            let mut tr = Transcript::new_default(b"lu");
            let (p, o) = prove_ring_logup(&r, &a, &b, &c, &mut tr).ok().unwrap();
            (p, o)
        };
        // Tamper A: the first sum-check's final eval fails (or the
        // zero-check's well-formedness).
        oracles.big_a[2] = r.add(&oracles.big_a[2], &r.one());
        let mut tr2 = Transcript::new_default(b"lu");
        assert!(verify_ring_logup(&r, 4, 4, &proof, &oracles, &mut tr2).is_err());
        // Tamper m: the second sum-check / zero-check / integer check.
        let (proof3, mut oracles3) = {
            let mut tr = Transcript::new_default(b"lu");
            let (p, o) = prove_ring_logup(&r, &a, &b, &c, &mut tr).ok().unwrap();
            (p, o)
        };
        oracles3.m[1] = r.add(&oracles3.m[1], &r.one());
        let mut tr3 = Transcript::new_default(b"lu");
        assert!(verify_ring_logup(&r, 4, 4, &proof3, &oracles3, &mut tr3).is_err());
        // Tamper c to a non-binary tag.
        let (proof4, mut oracles4) = {
            let mut tr = Transcript::new_default(b"lu");
            let (p, o) = prove_ring_logup(&r, &a, &b, &c, &mut tr).ok().unwrap();
            (p, o)
        };
        oracles4.c[0] = r.constant(2);
        let mut tr4 = Transcript::new_default(b"lu");
        assert!(verify_ring_logup(&r, 4, 4, &proof4, &oracles4, &mut tr4).is_err());
    }

    #[test]
    fn logup_prover_rejects_invalid() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "bad");
        let mut a_bad = a.clone();
        a_bad[1] = r.add(&a_bad[1], &r.one());
        let mut tr = Transcript::new_default(b"lu3");
        assert!(prove_ring_logup(&r, &a_bad, &b, &c, &mut tr).is_err());
        let mut c_bad = c.clone();
        c_bad[0] = r.g_map(999);
        let mut tr2 = Transcript::new_default(b"lu4");
        assert!(prove_ring_logup(&r, &a, &b, &c_bad, &mut tr2).is_err());
    }

    #[test]
    fn denominators_invert_with_high_probability() {
        // Remark 5.10: random (α, β) make every denominator invertible
        // whp — the resampling loop terminates almost immediately.
        let r = ring();
        let (a, b, c) = build_case(&r, 4, 4, "inv");
        let mut tr = Transcript::new_default(b"lu5");
        let (proof, _o) = prove_ring_logup(&r, &a, &b, &c, &mut tr).ok().unwrap();
        assert!(
            proof.inv_retries <= 2,
            "expected no/low retries, got {}",
            proof.inv_retries
        );
    }
}
