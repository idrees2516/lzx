//! The Greyhound-style compile (follow-up (a) of `ring-lookups.md`):
//! the Ring-LogUp PIOP compiled onto the Ajtai/carrier stack.
//!
//! Compilation recipe (the paper §2.4's framework, instantiated on the
//! workspace's own carrier instead of Greyhound's response layer):
//!
//! * every oracle message of the PIOP (`m`, `Â`, `B̂` — plus the
//!   statement vectors `a`, `b`, `c`, `g([N])` at indexing time) is
//!   replaced by a **digit-windowed Ajtai commitment**
//!   ([`windowed.rs`]): the prover sends `t = A·s_digits`, never the
//!   raw vector;
//! * every polynomial-evaluation query the PIOP verifier makes is
//!   answered by a **binding-pass proof**
//!   ([`prove_windowed_eval`]): `v̂(point) = value` against the
//!   commitment, via the grid tensor `a^T·S·b` over the digit layers
//!   (Greyhound's `y = a^T·S·b` evaluation view, Definition 2.3's
//!   `REval` relation);
//! * Fiat–Shamir challenges are re-derived from
//!   **commitment-absorbing** transcripts (the wave-6.4
//!   statement-absorption discipline), so nothing in the interaction
//!   depends on the raw oracles.
//!
//! The result is a succinct-argument shape for the indexed lookup
//! relation: verification touches only commitments, challenge points,
//! and the binding-pass responses — never the oracle vectors.
//!
//! Honest scope: the response layer transmits `z` in the clear
//! (`O(N·K_w)` ring elements — the paper's PIOP proof-length bound);
//! Greyhound's `√N` split-and-check compression is the documented
//! follow-up (the Serval module carries the pattern for its own
//! relation). Parameters are demonstration-scale; production
//! instantiation runs the `lattice-sis-estimator` discipline from
//! `SECURITY.md`.


// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
use crate::carrier::{CarrierCommitment, CarrierKey};
use crate::ring_d::{Elem, RingD};
use crate::ring_logup::{derive_terms, LogupOracles};
use crate::ring_sumcheck::{prove_sumcheck, verify_sumcheck, RingFactor, RingSumcheckProof, RingTerm, RingVirtualPoly};
use crate::subprotocols::{
    prove_binary_check, prove_integer_check, verify_binary_check, verify_integer_check, SubError,
};
use crate::windowed::{
    prove_windowed_eval, verify_windowed_eval, windowed_slots, WindowedEvalProof,
};
use lattice_core::transcript::Transcript;
use std::collections::BTreeMap;

/// A committed oracle vector: digit slots + the Ajtai commitment.
#[derive(Clone, Debug)]
pub struct CommittedVector {
    pub label: String,
    pub slots: Vec<Elem>,
    pub commitment: CarrierCommitment,
}

impl CommittedVector {
    pub fn commit(key: &CarrierKey, label: &str, v: &[Elem]) -> Result<Self, SubError> {
        let slots = windowed_slots(&key.params.ring, v)
            .map_err(|e| SubError::Shape(format!("{e:?}")))?;
        let commitment = key
            .commit(&slots)
            .map_err(|e| SubError::Shape(format!("{e:?}")))?;
        Ok(CommittedVector { label: label.to_string(), slots, commitment })
    }
}

/// One settled evaluation query: `(label, point, value, proof)`.
#[derive(Clone, Debug)]
pub struct SettledQuery {
    pub label: String,
    pub point: Vec<Elem>,
    pub value: Elem,
    pub proof: WindowedEvalProof,
}

/// The compiled Ring-LogUp proof.
#[derive(Clone, Debug)]
pub struct CompiledLogupProof {
    /// The committed oracles, keyed by label.
    pub committed: Vec<CommittedVector>,
    /// Statement-side vectors that stay public (the table `b` is part
    /// of the indexed statement in the holographic setting; we commit
    /// it too and keep only the commitment here).
    pub alpha: Elem,
    pub beta: Elem,
    pub v: Elem,
    pub inv_retries: u32,
    pub sc_a: RingSumcheckProof,
    pub sc_b: RingSumcheckProof,
    pub zc_a: RingSumcheckProof,
    pub zc_b: RingSumcheckProof,
    pub int_check: crate::subprotocols::IntegerCheckProof,
    pub binary: crate::subprotocols::BinaryCheckProof,
    /// Every evaluation query, settled by a binding-pass proof.
    pub queries: Vec<SettledQuery>,
}

fn elem_bytes(ring: &RingD, e: &Elem) -> Vec<u8> {
    let mut buf = Vec::with_capacity(ring.d * 8);
    for &c in e.coeffs() {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    buf
}

fn absorb_commitment(tr: &mut Transcript, label: &[u8], c: &CarrierCommitment) {
    let _ = tr.append_bytes(label, &c.to_bytes());
}

/// The committed prover. `key` carries the carrier parameters for
/// `m` slots = `N·K_w`; all oracle vectors share it (they have equal
/// lengths `N` in the logup instance: pad `a`/`c` to `N` if `M < N` —
/// the caller's responsibility; here we require `M == N` for the
/// single-key demo and document the padding rule).
#[allow(clippy::too_many_lines)]
pub fn prove_logup_committed(
    ring: &RingD,
    key: &CarrierKey,
    a: &[Elem],
    b: &[Elem],
    c: &[Elem],
) -> Result<CompiledLogupProof, SubError> {
    let m_len = a.len();
    let n = b.len();
    if m_len != n {
        // The padding rule: duplicate-table lookups pad the query with
        // any valid (b_j, g(j)) pair to the next power of two; the
        // multiplicity bookkeeping is unchanged. Demanded here: equal.
        return Err(SubError::Shape("compiled demo requires M == N (see the padding rule)".into()));
    }
    // ---- the plain PIOP machinery on the true vectors (fail-closed) ----
    let mut plain_tr = Transcript::new_default(b"lu-compiled");
    let (_plain_proof, oracles) = crate::ring_logup::prove_ring_logup(ring, a, b, c, &mut plain_tr)?;
    // ---- the compiled driver: commitments drive the transcript ----
    let mut tr = Transcript::new_default(b"lu-compiled");
    // 1. commit m, a, b, c, gN — the statement/indexer side first —
    //    plus the d CF rows of the binary check (coefficient extraction
    //    does NOT commute with the ring product, so the rows cannot be
    //    derived from a single settled c evaluation; the binary check's
    //    own consistency block binds them back to c).
    let cv_m = CommittedVector::commit(key, "lu-m", &oracles.m)?;
    let cv_a = CommittedVector::commit(key, "lu-a", &oracles.a)?;
    let cv_b = CommittedVector::commit(key, "lu-b", &oracles.b)?;
    let cv_c = CommittedVector::commit(key, "lu-c", &oracles.c)?;
    let cv_g = CommittedVector::commit(key, "lu-gN", &oracles.g_vec)?;
    let cf_rows: Vec<Vec<Elem>> = (0..ring.d)
        .map(|j| c.iter().map(|e| ring.constant(e.coeffs()[j])).collect())
        .collect();
    let mut cv_cfs: Vec<CommittedVector> = Vec::with_capacity(ring.d);
    for (j, row) in cf_rows.iter().enumerate() {
        cv_cfs.push(CommittedVector::commit(key, &format!("lu-bc-cf{j}"), row)?);
    }
    absorb_commitment(&mut tr, b"cl-m", &cv_m.commitment);
    absorb_commitment(&mut tr, b"cl-a", &cv_a.commitment);
    absorb_commitment(&mut tr, b"cl-b", &cv_b.commitment);
    absorb_commitment(&mut tr, b"cl-c", &cv_c.commitment);
    absorb_commitment(&mut tr, b"cl-g", &cv_g.commitment);
    // 2. (α, β) with the retry loop (challenges from commitments).
    //    The statement hash is state-free — an XOF over the commitment
    //    bytes in absorption order — so retries never consume the live
    //    sponge and the prover/verifier replay identically regardless
    //    of the retry count.
    let stmt_bytes = {
        let mut all = Vec::new();
        for cv in [&cv_m, &cv_a, &cv_b, &cv_c, &cv_g] {
            all.extend_from_slice(&cv.commitment.to_bytes());
        }
        Transcript::xof(b"lu-stmt", &all, 32)
    };
    let mut inv_retries = 0u32;
    let (alpha, beta, big_a, big_b) = loop {
        let mut salted = Transcript::new_default(b"lu-chal");
        let _ = salted.append_bytes(b"retry", &inv_retries.to_le_bytes());
        let _ = salted.append_bytes(b"stmt", &stmt_bytes);
        let alpha = ring.sample_challenge(&mut salted, b"lu-alpha");
        let beta = ring.sample_challenge(&mut salted, b"lu-beta");
        if let Some((ba, bb)) =
            derive_terms(ring, a, c, b, &oracles.g_vec, &oracles.m, &alpha, &beta)
        {
            break (alpha, beta, ba, bb);
        }
        inv_retries += 1;
        if inv_retries > 64 {
            return Err(SubError::Verify("inversion retries exceeded".into()));
        }
    };
    let cv_big_a = CommittedVector::commit(key, "lu-A", &big_a)?;
    let cv_big_b = CommittedVector::commit(key, "lu-B", &big_b)?;
    absorb_commitment(&mut tr, b"cl-A", &cv_big_a.commitment);
    absorb_commitment(&mut tr, b"cl-B", &cv_big_b.commitment);
    let v = {
        let mut acc = ring.zero();
        for e in &big_a {
            acc = ring.add(&acc, e);
        }
        acc
    };
    let _ = tr.append_bytes(b"lu-v", &elem_bytes(ring, &v));
    // 3. the two sum-checks (engine challenges ride the same transcript).
    let log_n = n.trailing_zeros() as usize;
    let poly_a = RingVirtualPoly {
        num_vars: log_n,
        claimed_sum: v.clone(),
        terms: vec![RingTerm { coeff: ring.one(), factors: vec![RingFactor::Mle(big_a.clone())] }],
    };
    let sc_a = prove_sumcheck(ring, &poly_a, &mut tr)?;
    let poly_b = RingVirtualPoly {
        num_vars: log_n,
        claimed_sum: v.clone(),
        terms: vec![RingTerm { coeff: ring.one(), factors: vec![RingFactor::Mle(big_b.clone())] }],
    };
    let sc_b = prove_sumcheck(ring, &poly_b, &mut tr)?;
    // 4. the zero-check challenges and proofs.
    let gamma: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(&mut tr, format!("lu-gamma-{i}").as_bytes()))
        .collect();
    let delta: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(&mut tr, format!("lu-delta-{i}").as_bytes()))
        .collect();
    let eq_g = ring.eq_row(&gamma);
    let eq_d = ring.eq_row(&delta);
    let neg_beta = ring.neg(&beta);
    let zc_a_poly = RingVirtualPoly {
        num_vars: log_n,
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
    let zc_a = prove_sumcheck(ring, &zc_a_poly, &mut tr)?;
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
                    RingFactor::Mle(oracles.g_vec.clone()),
                ],
            },
            RingTerm {
                coeff: neg_beta,
                factors: vec![RingFactor::Eq(eq_d.clone()), RingFactor::Mle(big_b.clone())],
            },
            RingTerm {
                coeff: ring.neg(&ring.one()),
                factors: vec![RingFactor::Eq(eq_d), RingFactor::Mle(oracles.m.clone())],
            },
        ],
    };
    let zc_b = prove_sumcheck(ring, &zc_b_poly, &mut tr)?;
    // 5. integer check on m and binary check on c.
    let (int_check, _iq) = prove_integer_check(ring, &oracles.m, 2, &mut tr)?;
    let (binary, _bq) = prove_binary_check(ring, c, &mut tr)?;
    // 6. The structural query points (the authoritative set — the
    //    compiled verifier derives them from the same proofs and
    //    transcript replays below): sum-check finals, zero-check
    //    factor evals, integer-check scalar points, binary-check
    //    points and scalar-product finals.
    // The structural query points:
    let mut query_points: Vec<(String, Vec<Elem>)> = Vec::new();
    let pt_a = sc_a.point.clone();
    query_points.push(("lu-A".into(), pt_a.clone()));
    let pt_b = sc_b.point.clone();
    query_points.push(("lu-B".into(), pt_b.clone()));
    let pt_za = zc_a.point.clone();
    for l in ["lu-A", "lu-a", "lu-c"] {
        query_points.push((l.into(), pt_za.clone()));
    }
    let pt_zb = zc_b.point.clone();
    for l in ["lu-B", "lu-b", "lu-gN", "lu-m"] {
        query_points.push((l.into(), pt_zb.clone()));
    }
    // Integer-check points on m.
    for (pt, _val) in int_check.claims.iter() {
        query_points.push(("lu-m".into(), pt.clone()));
    }
    // Binary-check points: the CF rows are committed separately; their
    // query points are the binary sum-check's final point and each
    // scalar product's own final point. The f-check queries c itself.
    let pt_bin = binary.binary_sc.point.clone();
    for j in 0..ring.d {
        query_points.push((format!("lu-bc-cf{j}"), pt_bin.clone()));
    }
    for sp in binary.cf_sps.iter() {
        for j in 0..ring.d {
            query_points.push((format!("lu-bc-cf{j}"), sp.sc.point.clone()));
        }
    }
    query_points.push(("lu-c".into(), binary.f_sp.sc.point.clone()));
    query_points.dedup();
    // 7. Settle every query with a binding-pass proof.
    let mut queries = Vec::with_capacity(query_points.len());
    let mut by_label: BTreeMap<String, &CommittedVector> = [
        ("lu-m".to_string(), &cv_m),
        ("lu-a".to_string(), &cv_a),
        ("lu-b".to_string(), &cv_b),
        ("lu-c".to_string(), &cv_c),
        ("lu-gN".to_string(), &cv_g),
        ("lu-A".to_string(), &cv_big_a),
        ("lu-B".to_string(), &cv_big_b),
    ]
    .into_iter()
    .collect();
    for cv in cv_cfs.iter() {
        by_label.insert(cv.label.clone(), cv);
    }
    for (label, point) in query_points {
        let cv = by_label.get(&label).ok_or_else(|| {
            SubError::Shape(format!("uncommitted oracle {label}"))
        })?;
        let value = if label.starts_with("lu-bc-cf") {
            // the CF rows settle directly against their commitments
            let j: usize = label
                .strip_prefix("lu-bc-cf")
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| SubError::Shape("bad cf label".into()))?;
            ring.mle_eval(&cf_rows[j], &point)
                .map_err(|e| SubError::Shape(format!("{e:?}")))?
        } else {
            let view = LogupOracles {
                a: oracles.a.clone(),
                b: oracles.b.clone(),
                c: oracles.c.clone(),
                g_vec: oracles.g_vec.clone(),
                m: oracles.m.clone(),
                big_a: big_a.clone(),
                big_b: big_b.clone(),
            };
            view.eval(ring, &label, &point).map_err(SubError::Verify)?
        };
        let proof = prove_windowed_eval(
            ring,
            key,
            &cv.slots,
            &point,
            &value,
            &cv.commitment,
            label.as_bytes(),
        )
        .map_err(|e| SubError::Shape(format!("{e:?}")))?;
        queries.push(SettledQuery { label, point, value, proof });
    }
    let mut committed = vec![cv_m, cv_a, cv_b, cv_c, cv_g, cv_big_a, cv_big_b];
    committed.extend(cv_cfs);
    Ok(CompiledLogupProof {
        committed,
        alpha,
        beta,
        v,
        inv_retries,
        sc_a,
        sc_b,
        zc_a,
        zc_b,
        int_check,
        binary,
        queries,
    })
}

/// The compiled verifier: settles every query against its commitment,
/// then replays the PIOP verification with the settled values.
#[allow(clippy::too_many_lines)]
pub fn verify_logup_committed(
    ring: &RingD,
    key: &CarrierKey,
    m_len: usize,
    n: usize,
    proof: &CompiledLogupProof,
) -> Result<(), SubError> {
    if m_len != n {
        return Err(SubError::Shape("compiled demo requires M == N".into()));
    }
    let log_n = n.trailing_zeros() as usize;
    // 1. Settle every query: the binding pass pins each claimed value
    //    to its Ajtai commitment.
    let mut settled: BTreeMap<(String, Vec<Vec<u64>>), Elem> = BTreeMap::new();
    for q in &proof.queries {
        let cv = proof
            .committed
            .iter()
            .find(|c| c.label == q.label)
            .ok_or_else(|| SubError::Shape(format!("missing commitment {}", q.label)))?;
        verify_windowed_eval(ring, key, &cv.commitment, &q.point, &q.value, &q.proof)
            .map_err(|e| SubError::Verify(format!("binding pass: {e:?}")))?;
        let key = (q.label.clone(), q.point.iter().map(|e| e.coeffs().to_vec()).collect::<Vec<_>>());
        settled.insert(key, q.value.clone());
    }
    // 2. Replay the PIOP with a resolver answering from the settled
    //    map; the transcript absorbs the commitments in the prover's
    //    order (lu-m, lu-a, lu-b, lu-c, lu-gN, lu-A, lu-B under the
    //    cl-* labels).
    let mut tr = Transcript::new_default(b"lu-compiled");
    // (label, committed-vector label) pairs in the prover's absorption
    // order: cl-g absorbs the lu-gN vector.
    for (absorb_label, cv_label) in [
        ("cl-m", "lu-m"),
        ("cl-a", "lu-a"),
        ("cl-b", "lu-b"),
        ("cl-c", "lu-c"),
        ("cl-g", "lu-gN"),
        ("cl-A", "lu-A"),
        ("cl-B", "lu-B"),
    ] {
        let cv = proof
            .committed
            .iter()
            .find(|c| c.label == cv_label)
            .ok_or_else(|| SubError::Shape(format!("missing {cv_label}")))?;
        absorb_commitment(&mut tr, absorb_label.as_bytes(), &cv.commitment);
    }
    let lookup = |label: &str, point: &[Elem]| -> Result<Elem, String> {
        let key = (label.to_string(), point.iter().map(|e| e.coeffs().to_vec()).collect::<Vec<_>>());
        settled
            .get(&key)
            .cloned()
            .ok_or_else(|| format!("unsettled query {label} @{:?}", point.len()))
    };
    // (α, β) replay with the same state-free salt discipline.
    let stmt_bytes = {
        // the same five statement-side commitments, in absorption order
        let labels = ["lu-m", "lu-a", "lu-b", "lu-c", "lu-gN"];
        let mut all = Vec::new();
        for l in labels {
            let cv = proof
                .committed
                .iter()
                .find(|c| c.label == l)
                .ok_or_else(|| SubError::Shape(format!("missing {l}")))?;
            all.extend_from_slice(&cv.commitment.to_bytes());
        }
        Transcript::xof(b"lu-stmt", &all, 32)
    };
    let alpha;
    let beta;
    {
        let mut salted = Transcript::new_default(b"lu-chal");
        let _ = salted.append_bytes(b"retry", &proof.inv_retries.to_le_bytes());
        let _ = salted.append_bytes(b"stmt", &stmt_bytes);
        alpha = ring.sample_challenge(&mut salted, b"lu-alpha");
        beta = ring.sample_challenge(&mut salted, b"lu-beta");
    }
    if alpha != proof.alpha || beta != proof.beta {
        return Err(SubError::Verify("(α, β) replay mismatch".into()));
    }
    let _ = tr.append_bytes(b"lu-v", &elem_bytes(ring, &proof.v));
    // Sum-checks.
    let shape1 = crate::ring_sumcheck::RingSumcheckShape {
        num_vars: log_n,
        terms: vec![crate::ring_sumcheck::RingTermShape {
            coeff: ring.one(),
            num_factors: 1,
        }],
    };
    verify_sumcheck(ring, &shape1, &proof.v, &proof.sc_a, &mut tr, &mut |_ti, _fi, pt| {
        lookup("lu-A", pt)
    })
    .map_err(SubError::from)?;
    verify_sumcheck(ring, &shape1, &proof.v, &proof.sc_b, &mut tr, &mut |_ti, _fi, pt| {
        lookup("lu-B", pt)
    })
    .map_err(SubError::from)?;
    // Zero-check challenges.
    let gamma: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(&mut tr, format!("lu-gamma-{i}").as_bytes()))
        .collect();
    let delta: Vec<Elem> = (0..log_n)
        .map(|i| ring.sample_challenge(&mut tr, format!("lu-delta-{i}").as_bytes()))
        .collect();
    let eq_g = ring.eq_row(&gamma);
    let eq_d = ring.eq_row(&delta);
    // Zero-check 1.
    let terms_a = vec![
        crate::ring_sumcheck::RingTermShape { coeff: ring.one(), num_factors: 3 },
        crate::ring_sumcheck::RingTermShape { coeff: alpha.clone(), num_factors: 3 },
        crate::ring_sumcheck::RingTermShape { coeff: ring.neg(&beta), num_factors: 2 },
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&ring.one()),
            num_factors: 1,
        },
    ];
    let shape_zc_a = crate::ring_sumcheck::RingSumcheckShape { num_vars: log_n, terms: terms_a };
    let eqg_ref = &eq_g;
    verify_sumcheck(ring, &shape_zc_a, &ring.zero(), &proof.zc_a, &mut tr, &mut |ti, fi, pt| {
        match (ti, fi) {
            (0, 0) | (1, 0) | (2, 0) | (3, 0) => {
                ring.mle_eval(eqg_ref, pt).map_err(|e| format!("{e:?}"))
            }
            (0, 1) | (1, 1) | (2, 1) => lookup("lu-A", pt),
            (0, 2) => lookup("lu-a", pt),
            (1, 2) => lookup("lu-c", pt),
            _ => Err("bad factor".into()),
        }
    })
    .map_err(SubError::from)?;
    // Zero-check 2.
    let terms_b = vec![
        crate::ring_sumcheck::RingTermShape { coeff: ring.one(), num_factors: 3 },
        crate::ring_sumcheck::RingTermShape { coeff: alpha.clone(), num_factors: 3 },
        crate::ring_sumcheck::RingTermShape { coeff: ring.neg(&beta), num_factors: 2 },
        crate::ring_sumcheck::RingTermShape {
            coeff: ring.neg(&ring.one()),
            num_factors: 2,
        },
    ];
    let shape_zc_b = crate::ring_sumcheck::RingSumcheckShape { num_vars: log_n, terms: terms_b };
    let eqd_ref = &eq_d;
    verify_sumcheck(ring, &shape_zc_b, &ring.zero(), &proof.zc_b, &mut tr, &mut |ti, fi, pt| {
        match (ti, fi) {
            (0, 0) | (1, 0) | (2, 0) | (3, 0) => {
                ring.mle_eval(eqd_ref, pt).map_err(|e| format!("{e:?}"))
            }
            (0, 1) | (1, 1) | (2, 1) => lookup("lu-B", pt),
            (0, 2) => lookup("lu-b", pt),
            (1, 2) => lookup("lu-gN", pt),
            (3, 1) => lookup("lu-m", pt),
            _ => Err("bad factor".into()),
        }
    })
    .map_err(SubError::from)?;
    // Integer + binary checks.
    verify_integer_check(ring, n, &proof.int_check, &mut tr, &|l, pt| match l {
        "ic-a" => lookup("lu-m", pt),
        _ => Err("bad label".into()),
    })?;
    verify_binary_check(ring, m_len, &proof.binary, &mut tr, &|l, pt| match l {
        "bc-f" => lookup("lu-c", pt),
        other => lookup(&format!("lu-{other}"), pt),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windowed::carrier_params_for;

    fn ring() -> RingD {
        RingD::new(4).ok().unwrap()
    }

    fn build_case(r: &RingD, n: usize, seed: &str) -> (Vec<Elem>, Vec<Elem>, Vec<Elem>) {
        let b: Vec<Elem> = (0..n).map(|j| r.random(format!("{seed}-b{j}").as_bytes())).collect();
        let mut a = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i * 3 + 1) % n;
            a.push(b[j].clone());
            c.push(r.g_map(j as u64));
        }
        (a, b, c)
    }

    #[test]
    fn compiled_logup_end_to_end() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, "cmp");
        let params = carrier_params_for(&r, 4, 2, 1 << 16);
        let key = CarrierKey::from_seed(params, [21u8; 32]);
        let proof = prove_logup_committed(&r, &key, &a, &b, &c)
            .unwrap_or_else(|e| panic!("prove: {e:?}"));
        verify_logup_committed(&r, &key, 4, 4, &proof)
            .unwrap_or_else(|e| panic!("verify: {e:?}"));
    }

    #[test]
    fn compiled_logup_tamper_rejections() {
        let r = ring();
        let (a, b, c) = build_case(&r, 4, "cmp-t");
        let params = carrier_params_for(&r, 4, 2, 1 << 16);
        let key = CarrierKey::from_seed(params, [22u8; 32]);
        let proof = prove_logup_committed(&r, &key, &a, &b, &c).ok().unwrap();
        // Tamper a settled query value: the binding pass rejects it.
        let mut bad = proof.clone();
        if let Some(q) = bad.queries.first_mut() {
            q.value = r.add(&q.value, &r.one());
        }
        assert!(verify_logup_committed(&r, &key, 4, 4, &bad).is_err());
        // Tamper a commitment: every binding pass against it fails.
        let mut bad2 = proof.clone();
        if let Some(cv) = bad2.committed.first_mut() {
            cv.commitment.rows[0] = r.add(&cv.commitment.rows[0], &r.one());
        }
        assert!(verify_logup_committed(&r, &key, 4, 4, &bad2).is_err());
        // Tamper the claimed sum v: the sum-check rejects.
        let mut bad3 = proof.clone();
        bad3.v = r.add(&bad3.v, &r.one());
        assert!(verify_logup_committed(&r, &key, 4, 4, &bad3).is_err());
        // Tamper a zero-check round message.
        let mut bad4 = proof.clone();
        if let Some(round) = bad4.zc_a.rounds.first_mut() {
            round[0] = r.add(&round[0], &r.one());
        }
        assert!(verify_logup_committed(&r, &key, 4, 4, &bad4).is_err());
    }
}
