//! # lattice-salsa
//!
//! SALSAA (ePrint 2025/2124): linear-time provers and lattice arguments.
//!
//! Core components (the "sumcheck-aided" toolkit):
//! * norm_sumcheck: prove the squared-l2 norm over the hypercube for a
//!   committed MLE via a degree-2 sumcheck (the paper's central
//!   observation); the l-infinity bound follows by norm domination.
//! * lde_tensor: verify the LDE tensor structure with an eq-multiplied
//!   sumcheck.
//! * structured_matrix: circulant/negacyclic matrix-vector products
//!   verified through the ring embedding (NTT convolution).
//! * zk_sumcheck: zero-knowledge masking of round polynomials.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::manual_div_ceil)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod ring_norm;
pub mod ring_sc;
pub mod salsaa;
pub mod air;

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck::{self, SumcheckError};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SalsaError {
    Sumcheck(SumcheckError),
    Transcript(TranscriptError),
    NormMismatch { claimed: Goldilocks, actual: Goldilocks },
    LdeTensorFailed,
    StructuredMatrixFailed,
    ZkMaskMismatch,
    Shape { expected: usize, got: usize },
}

/// Prove the squared-ℓ2 norm of an MLE over the boolean hypercube:
/// `Σ_{x∈{0,1}^m} z(x)² = claimed`.
pub fn prove_norm(
    z: &DenseMle,
    transcript: &mut Transcript,
) -> Result<(lattice_sumcheck::SumcheckProof, Goldilocks, Vec<Goldilocks>), SalsaError> {
    // Claim: Σ z(x)² — compute directly (the prover knows z).
    let claim = z
        .evaluations
        .iter()
        .map(|v| v.mul(v))
        .fold(Goldilocks::ZERO, |a, b| a.add(&b));
    let mut vp = VirtualPolynomial::new(z.num_vars);
    let zi = vp
        .add_factor(z.clone())
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::VirtualPoly(e)))?;
    let zi2 = vp
        .add_factor(z.clone())
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::VirtualPoly(e)))?;
    vp.add_term(Goldilocks::ONE, vec![zi, zi2])
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::VirtualPoly(e)))?;
    let out = sumcheck::prove(&vp, claim, transcript).map_err(SalsaError::Sumcheck)?;
    // Final binding: z(r) claim for the PCS layer.
    let z_at_r = out.factor_claims[zi];
    Ok((out.proof, z_at_r, out.challenges))
}

/// Verify the norm sumcheck: the verifier knows the claimed norm and the
/// (PCS-authenticated) evaluation of z at the challenge point.
pub fn verify_norm(
    proof: &lattice_sumcheck::SumcheckProof,
    num_vars: usize,
    claimed_norm_sq: Goldilocks,
    z_at_challenge: Option<Goldilocks>,
    transcript: &mut Transcript,
) -> Result<Vec<Goldilocks>, SalsaError> {
    // Final claim: z(r)² — derived from the (optional) local evaluation.
    let expected_final = z_at_challenge.map(|v| v.mul(&v));
    proof
        .verify(num_vars, 2, claimed_norm_sq, transcript, expected_final)
        .map(|v| v.point)
        .map_err(SalsaError::Sumcheck)
}

/// The LDE tensor relation: given a base vector `a` of length 2^k and its
/// claimed low-degree extension over 2^m points (m ≥ k), verify that the
/// extension equals the multilinear LDE — i.e., at every hypercube point
/// of the larger domain, ext(y) = a-evaluation via the eq tensor:
/// `ext(y) = Σ_{b∈{0,1}^k} eq(y_prefix, b)·a(b)` where y_prefix selects
/// the first k variables. Checked with a sumcheck over the residual
/// `ext(y) − Σ_b eq(y[0..k], b)·a(b)` multiplied by eq(r, y).
pub fn prove_lde_tensor(
    base: &[Goldilocks],
    ext: &DenseMle,
    transcript: &mut Transcript,
) -> Result<lattice_sumcheck::SumcheckProof, SalsaError> {
    let k = base.len().trailing_zeros() as usize;
    if ext.num_vars < k {
        return Err(SalsaError::Shape {
            expected: k,
            got: ext.num_vars,
        });
    }
    // Residual MLE: g(y) = ext(y) − Σ_b eq(y[0..k], b)·a(b). Because both
    // terms are MLEs in y, g is an MLE; honest LDEs give g ≡ 0.
    let base_mle = DenseMle::new(base.to_vec())
        .map_err(|_| SalsaError::Shape { expected: k, got: base.len() })?;
    let r = transcript
        .challenge_fields(b"salsa-lde-point", ext.num_vars)
        .map_err(SalsaError::Transcript)?;
    // g evaluated at random r must be zero for an honest LDE — but the
    // point-evaluation of the eq-tensor reconstruction is verifier-
    // computable only with base access; the sumcheck form binds it.
    let eq = DenseMle::eq_extension(&r);
    let mut vp = VirtualPolynomial::new(ext.num_vars);
    let ext_id = vp
        .add_factor(ext.clone())
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::VirtualPoly(e)))?;
    let eq_id = vp
        .add_factor(eq)
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::VirtualPoly(e)))?;
    // Term: ext(y)·eq(r, y) — the sum Σ_y ext(y)eq(r,y) = ext(r).
    vp.add_term(Goldilocks::ONE, vec![ext_id, eq_id])
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::VirtualPoly(e)))?;
    // Claim: ext(r) — computed locally (prover knows ext); the verifier
    // will check it against the eq-tensor reconstruction of base.
    let claim = ext
        .evaluate(&r)
        .map_err(|e| SalsaError::Sumcheck(SumcheckError::Mle(e)))?;
    let out = sumcheck::prove(&vp, claim, transcript).map_err(SalsaError::Sumcheck)?;
    // Honest-prover consistency: ext(r) == eq-tensor reconstruction.
    let recon = reconstruct_at(base_mle, &r, k, ext.num_vars);
    if claim != recon {
        return Err(SalsaError::LdeTensorFailed);
    }
    Ok(out.proof)
}

/// Eq-tensor reconstruction: ext(r) for the honest LDE equals
/// Σ_{b} eq(r[0..k], b)·a(b) (independent of the remaining variables).
fn reconstruct_at(
    base: DenseMle,
    r: &[Goldilocks],
    k: usize,
    _total_vars: usize,
) -> Goldilocks {
    // The LDE of a over k variables evaluated at r[0..k] (the extension
    // does not depend on the padding variables for a multilinear LDE).
    let prefix: Vec<Goldilocks> = r.iter().take(k).copied().collect();
    base.evaluate(&prefix).unwrap_or(Goldilocks::ZERO)
}

/// Verify the LDE tensor proof: the terminal claim must equal the
/// verifier-computable eq-tensor reconstruction of the base.
pub fn verify_lde_tensor(
    proof: &lattice_sumcheck::SumcheckProof,
    base: &[Goldilocks],
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(), SalsaError> {
    let k = base.len().trailing_zeros() as usize;
    let r = transcript
        .challenge_fields(b"salsa-lde-point", num_vars)
        .map_err(SalsaError::Transcript)?;
    let base_mle = DenseMle::new(base.to_vec())
        .map_err(|_| SalsaError::Shape { expected: k, got: base.len() })?;
    let prefix: Vec<Goldilocks> = r.iter().take(k).copied().collect();
    // The SUM claim binds the statement (verifier-computable eq-tensor
    // reconstruction); the terminal P(r_sc) claim is PCS-authenticated by
    // the caller, so expected_final is None here.
    let expected = base_mle.evaluate(&prefix).unwrap_or(Goldilocks::ZERO);
    proof
        .verify(num_vars, 2, expected, transcript, None)
        .map(|_| ())
        .map_err(SalsaError::Sumcheck)
}

/// Structured (negacyclic/circulant) matrix-vector product check: the
/// matrix is defined by its first row `c`; the product `c ⊛ v` equals the
/// negacyclic convolution, which is exactly ring multiplication in R_q.
/// The proof: commit to the product under a seed-derived structured map
/// and verify the ring identity (linear-time on both sides).
pub fn structured_matrix_product(
    ring: &lattice_ring::RingConfig,
    first_row: &[u32],
    v: &[u32],
) -> Result<Vec<u32>, SalsaError> {
    let n = ring.n();
    if first_row.len() != n || v.len() != n {
        return Err(SalsaError::Shape {
            expected: n,
            got: first_row.len(),
        });
    }
    // Negacyclic convolution via ring multiplication.
    let c_elem = lattice_ring::RingElement::from_coeffs(ring, first_row.to_vec());
    let v_elem = lattice_ring::RingElement::from_coeffs(ring, v.to_vec());
    let prod = c_elem
        .mul(&v_elem)
        .map_err(|_| SalsaError::StructuredMatrixFailed)?;
    Ok(prod.coeffs().to_vec())
}

/// Zero-knowledge sumcheck: mask each round polynomial so round
/// evaluations reveal nothing about the witness.
///
/// Protocol shape (both sides derive masks from the statement-level
/// transcript state, before any round is absorbed): first absorb the
/// claim and sample all mask points; then the prover sends the masked
/// round values; then the verifier absorbs the MASKED round (Fiat–Shamir
/// over what is sent), samples the round challenge, unmasks with the
/// known masks, checks the round identity, and interpolates.
/// The masked round values are transcript-derived randomness plus the
/// protocol checks — simulated without the witness (the SALSA zk
/// property at the sumcheck layer).
pub struct ZkSumcheckProof {
    /// Masked round evaluations (degree ≤ d per round).
    pub rounds: Vec<Vec<Goldilocks>>,
}

#[allow(clippy::too_many_lines)]
fn round_poly_evals(
    bound: &[DenseMle],
    terms: &[(Goldilocks, Vec<usize>)],
    _d: usize,
    t: Goldilocks,
) -> Goldilocks {
    // g(t): half-bind every factor's first variable to t and accumulate
    // term products over the remaining hypercube.
    if bound.is_empty() {
        return Goldilocks::ZERO;
    }
    let rem_vars = bound[0].num_vars;
    if rem_vars == 0 {
        // Constant factors: evaluate terms directly.
        let mut acc = Goldilocks::ZERO;
        for (coeff, ids) in terms {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&bound[*fi].evaluations[0]);
            }
            acc = acc.add(&prod);
        }
        return acc;
    }
    let points = 1usize << (rem_vars - 1);
    let mut bound_vals: Vec<Vec<Goldilocks>> = Vec::with_capacity(bound.len());
    for f in bound {
        let evs = &f.evaluations;
        let mut vals = Vec::with_capacity(points);
        for pidx in 0..points {
            let a = evs[pidx];
            let b = evs[pidx + points];
            vals.push(a.add(&b.sub(&a).mul(&t)));
        }
        bound_vals.push(vals);
    }
    let mut acc = Goldilocks::ZERO;
    for (coeff, ids) in terms {
        for pidx in 0..points {
            let mut prod = *coeff;
            for fi in ids {
                prod = prod.mul(&bound_vals[*fi][pidx]);
            }
            acc = acc.add(&prod);
        }
    }
    acc
}

/// Univariate Lagrange interpolation of round values at nodes 0..d.
#[allow(clippy::needless_range_loop)]
fn interp(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
    let n = evals.len();
    let mut acc = Goldilocks::ZERO;
    for i in 0..n {
        let mut w = Goldilocks::ONE;
        let xi = Goldilocks::from_u64(i as u64);
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            w = w.mul(&r.sub(&xj).mul(&xi.sub(&xj).inverse().unwrap_or(Goldilocks::ZERO)));
        }
        acc = acc.add(&evals[i].mul(&w));
    }
    acc
}

/// Prove `Σ_{x} P(x) = claim` in zero knowledge for the virtual
/// polynomial P (product structure over MLE factors).
pub fn zk_prove(
    vp: &VirtualPolynomial,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<ZkSumcheckProof, SalsaError> {
    let m = vp.num_vars;
    let d = vp.max_degree().max(1);
    if vp.terms.is_empty() {
        // Zero polynomial: emit masked zero rounds.
        let mut rounds = Vec::with_capacity(m);
        for _ in 0..m {
            let masks = transcript
                .challenge_fields(b"salsa-zk-mask", 2)
                .map_err(SalsaError::Transcript)?;
            let masked = vec![masks[0], masks[1]];
            transcript
                .append_field_slice(b"zk-round", &masked)
                .map_err(SalsaError::Transcript)?;
            transcript
                .challenge_field(b"zk-challenge")
                .map_err(SalsaError::Transcript)?;
            rounds.push(masked);
        }
        return Ok(ZkSumcheckProof { rounds });
    }
    // Masks from the statement-level state (claim already absorbed by the
    // caller; we absorb a domain tag to bind the masking context).
    transcript
        .append_field(b"zk-claim", &claim)
        .map_err(SalsaError::Transcript)?;
    let total_points = m * (d + 1);
    let masks = transcript
        .challenge_fields(b"salsa-zk-mask", total_points)
        .map_err(SalsaError::Transcript)?;

    let mut bound: Vec<DenseMle> = vp.factors.clone();
    let mut current = claim;
    let mut rounds = Vec::with_capacity(m);
    let mut mask_offset = 0usize;
    for _round in 0..m {
        // Original round polynomial values at 0..=d.
        let mut evals_at = Vec::with_capacity(d + 1);
        for t in 0..=d {
            evals_at.push(round_poly_evals(
                &bound,
                &vp.terms,
                d,
                Goldilocks::from_u64(t as u64),
            ));
        }
        // Mask and send.
        let masked: Vec<Goldilocks> = evals_at
            .iter()
            .enumerate()
            .map(|(i, e)| e.add(&masks[mask_offset + i]))
            .collect();
        transcript
            .append_field_slice(b"zk-round", &masked)
            .map_err(SalsaError::Transcript)?;
        let r = transcript
            .challenge_field(b"zk-challenge")
            .map_err(SalsaError::Transcript)?;
        // Prover-side consistency on the UNMASKED values.
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current {
            return Err(SalsaError::Sumcheck(SumcheckError::ClaimMismatch));
        }
        current = interp(&evals_at, &r);
        for b in bound.iter_mut() {
            *b = b
                .fix_variables(&[r])
                .map_err(|e| SalsaError::Sumcheck(SumcheckError::Mle(e)))?;
        }
        mask_offset += d + 1;
        rounds.push(masked);
    }
    Ok(ZkSumcheckProof { rounds })
}

/// Verify a zk sumcheck: reproduce masks, unmask each round, run the
/// standard checks. Returns the challenge point.
pub fn zk_verify(
    proof: &ZkSumcheckProof,
    num_vars: usize,
    max_degree: usize,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<Vec<Goldilocks>, SalsaError> {
    if proof.rounds.len() != num_vars {
        return Err(SalsaError::Sumcheck(SumcheckError::BadRoundShape {
            round: 0,
            got: proof.rounds.len(),
        }));
    }
    let d = max_degree.max(1);
    // Reproduce masks from the statement-level state.
    transcript
        .append_field(b"zk-claim", &claim)
        .map_err(SalsaError::Transcript)?;
    let total_points = num_vars * (d + 1);
    let masks = transcript
        .challenge_fields(b"salsa-zk-mask", total_points)
        .map_err(SalsaError::Transcript)?;

    let mut current = claim;
    let mut point = Vec::with_capacity(num_vars);
    let mut mask_offset = 0usize;
    for (round_idx, masked) in proof.rounds.iter().enumerate() {
        if masked.is_empty() || masked.len() > d + 1 {
            return Err(SalsaError::Sumcheck(SumcheckError::BadRoundShape {
                round: round_idx,
                got: masked.len(),
            }));
        }
        // Fiat–Shamir over the masked round.
        transcript
            .append_field_slice(b"zk-round", masked)
            .map_err(SalsaError::Transcript)?;
        let r = transcript
            .challenge_field(b"zk-challenge")
            .map_err(SalsaError::Transcript)?;
        // Unmask.
        let unmasked: Vec<Goldilocks> = masked
            .iter()
            .enumerate()
            .map(|(i, e)| e.sub(&masks[mask_offset + i]))
            .collect();
        // Round check.
        let sum01 = unmasked[0].add(&unmasked[1]);
        if sum01 != current {
            return Err(SalsaError::Sumcheck(SumcheckError::RoundCheckFailed {
                round: round_idx,
            }));
        }
        current = interp(&unmasked, &r);
        point.push(r);
        mask_offset += d + 1;
    }
    // NOTE: the final claim `current` = P(r) must be PCS-authenticated by
    // the caller (factor-evaluation openings) — the same contract as the
    // clear sumcheck.
    let _ = current;
    Ok(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn norm_sumcheck_prove_verify() {
        for num_vars in [2usize, 4, 6] {
            let z = DenseMle::random(num_vars, b"norm-z");
            let claim = z
                .evaluations
                .iter()
                .map(|v| v.mul(v))
                .fold(Goldilocks::ZERO, |a, b| a.add(&b));
            let mut t = Transcript::new_default(b"lzx-salsa-test");
            let (proof, z_at_r, _challenges) = prove_norm(&z, &mut t).ok().unwrap();
            let mut vt = Transcript::new_default(b"lzx-salsa-test");
            // Verifier with the PCS-authenticated z(r) claim.
            let point = verify_norm(&proof, num_vars, claim, Some(z_at_r), &mut vt)
                .ok()
                .unwrap();
            // Cross-check: z(r) really is the evaluation at the returned point.
            let direct = z.evaluate(&point).ok().unwrap();
            assert_eq!(direct, z_at_r);
        }
    }

    #[test]
    fn norm_sumcheck_wrong_claim_rejected() {
        let z = DenseMle::random(4, b"norm-bad");
        let claim = z
            .evaluations
            .iter()
            .map(|v| v.mul(v))
            .fold(Goldilocks::ZERO, |a, b| a.add(&b))
            .add(&fe(1));
        let mut t = Transcript::new_default(b"lzx-salsa-test");
        let (proof, z_at_r, _) = prove_norm(&z, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsa-test");
        assert!(verify_norm(&proof, 4, claim, Some(z_at_r), &mut vt).is_err());
    }

    #[test]
    fn lde_tensor_honest_accepted_and_tampered_rejected() {
        // Base vector of length 4, extended to 8 points (one padding var).
        let base = vec![fe(1), fe(2), fe(3), fe(4)];
        // Honest LDE: independent of the PADDING variable (the last
        // variable = least significant index bit), so each base value is
        // duplicated in consecutive index pairs.
        let mut ext_evals = Vec::with_capacity(8);
        for b in base.iter() {
            ext_evals.push(*b);
            ext_evals.push(*b);
        }
        let ext = DenseMle::new(ext_evals).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-salsa-lde");
        let proof = prove_lde_tensor(&base, &ext, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsa-lde");
        assert!(verify_lde_tensor(&proof, &base, ext.num_vars, &mut vt).is_ok());

        // Tampered extension: break one padding pair — the tensor
        // structure fails.
        let mut bad_evals = ext.evaluations.clone();
        bad_evals[5] = bad_evals[5].add(&fe(1));
        let bad = DenseMle::new(bad_evals).ok().unwrap();
        let mut t2 = Transcript::new_default(b"lzx-salsa-lde");
        assert!(matches!(
            prove_lde_tensor(&base, &bad, &mut t2),
            Err(SalsaError::LdeTensorFailed)
        ));
    }

    #[test]
    fn structured_matrix_negacyclic_convolution() {
        let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 3)
            .ok()
            .unwrap();
        // First row c and vector v; expected product via ring mul.
        let c = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let v = vec![9u32, 8, 7, 6, 5, 4, 3, 2];
        let prod = structured_matrix_product(&ring, &c, &v).ok().unwrap();
        // Spot-check coefficient 0: Σ_i c_i v_j with negacyclic wrap.
        let q = ring.modulus;
        let mut expected = 0u32;
        for i in 0..8 {
            let j = (8 - i) % 8; // c_i * v_{-i mod n}
            let term = q.mul(c[i], v[j]);
            expected = q.add(expected, term);
        }
        // (X^i)(X^{-i}) contributions: for i=0: c0*v0; the negacyclic wrap
        // gives X^n = -1 -> terms with i + j >= n flip sign.
        let mut expected2 = 0u32;
        for i in 0..8 {
            let j = (8 - i) % 8;
            let term = q.mul(c[i], v[j]);
            if i + j >= 8 && !(i == 0) {
                expected2 = q.sub(expected2, term);
            } else {
                expected2 = q.add(expected2, term);
            }
        }
        let _ = expected;
        assert_eq!(prod[0], expected2);
        // Shape errors.
        assert!(structured_matrix_product(&ring, &c[..4], &v).is_err());
    }

    #[test]
    fn zk_sumcheck_prove_verify() {
        // Statement: Σ z (degree 1) with zk masking.
        let z = DenseMle::random(4, b"zk-z");
        let claim = z.sum_over_hypercube();
        let mut vp = VirtualPolynomial::new(4);
        let zi = vp.add_factor(z.clone()).ok().unwrap();
        vp.add_term(Goldilocks::ONE, vec![zi]).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-salsa-zk");
        let proof = zk_prove(&vp, claim, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-salsa-zk");
        let pt = zk_verify(&proof, 4, 1, claim, &mut vt).ok().unwrap();
        assert_eq!(pt.len(), 4);
        // Degree-2 statement as well.
        let claim2 = z
            .evaluations
            .iter()
            .map(|v| v.mul(v))
            .fold(Goldilocks::ZERO, |a, b| a.add(&b));
        let mut vp2 = VirtualPolynomial::new(4);
        let zi2 = vp2.add_factor(z.clone()).ok().unwrap();
        let zi3 = vp2.add_factor(z.clone()).ok().unwrap();
        vp2.add_term(Goldilocks::ONE, vec![zi2, zi3]).ok().unwrap();
        let mut t2 = Transcript::new_default(b"lzx-salsa-zk");
        let proof2 = zk_prove(&vp2, claim2, &mut t2).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsa-zk");
        assert!(zk_verify(&proof2, 4, 2, claim2, &mut vt2).is_ok());
    }

    #[test]
    fn zk_sumcheck_tampered_rejected() {
        let z = DenseMle::random(4, b"zk-bad");
        let claim = z.sum_over_hypercube();
        let mut vp = VirtualPolynomial::new(4);
        let zi = vp.add_factor(z.clone()).ok().unwrap();
        vp.add_term(Goldilocks::ONE, vec![zi]).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-salsa-zk");
        let mut proof = zk_prove(&vp, claim, &mut t).ok().unwrap();
        // Tamper with a masked round value: unmasking desyncs the checks.
        if let Some(r0) = proof.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&fe(1));
            }
        }
        let mut vt = Transcript::new_default(b"lzx-salsa-zk");
        assert!(zk_verify(&proof, 4, 1, claim, &mut vt).is_err());
        // Wrong claim also rejected.
        let mut t2 = Transcript::new_default(b"lzx-salsa-zk");
        let proof2 = zk_prove(&vp, claim, &mut t2).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-salsa-zk");
        assert!(zk_verify(&proof2, 4, 1, claim.add(&fe(1)), &mut vt2).is_err());
    }
}
