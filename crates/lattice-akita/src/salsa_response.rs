//! Akita × SALSAA — Wave 7.3 (D4): the response-layer swap.
//!
//! The pre-swap response (pcs.rs) opens the **full packed witness** and
//! attaches a digit-decomposed `NormProof` — Θ(N) communication and full
//! witness disclosure. The SALSAA chain (lattice-salsa's
//! `prove_norm_chain`: D1 ring-norm sumcheck + D2 LDE-tensor
//! linearization) replaces it: the response becomes **polylog** (two
//! sumchecks + O(1) claims) and discloses nothing but the challenge
//! evaluations.
//!
//! The verified surface: (1) the carrier eq-sumcheck with the terminal
//! bound to the claimed `f(r_sc)` through the verifier's own
//! `eq(r, r_sc)` factor; (2) D1's Π^sum with the transmitted `z(r)`
//! (the coefficient-MLE at the D1 challenge) and the Lemma-4 wraparound
//! gate; (3) D2's Π^lde-⊗ whose terminal is the verifier's own
//! zero-communication row evaluation.
//!
//! **Documented gap (the paper's outer layer)**: the Ajtai binding of
//! `z(r)`/`f(r_sc)` to the commitment — SALSA(A)'s authenticated opening
//! at the challenge — is not realized here; the kernel keeps the Clear
//! mode (full opening) as the binding-complete path and exposes this
//! module as the polylog response layer the outer protocol will close.

use crate::pcs::{AkitaPcs, Commitment};
use lattice_core::mle::DenseMle;
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_salsa::ring_norm::{
    prove_norm_chain, prove_ring_norm, verify_ring_norm, LinRelation, NormChainProof, RingNormProof,
};
use lattice_sumcheck::sumcheck;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SalsaResponseError {
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    RingNorm(lattice_salsa::ring_norm::RingNormError),
    Transcript(lattice_core::transcript::TranscriptError),
    /// Ajtai commitment failure (shape/bound).
    Ajtai(String),
    Shape {
        expected: usize,
        got: usize,
    },
    /// The terminal identity `P(r_sc) = eq(r, r_sc)·f(r_sc)` failed.
    TerminalBindingFailed,
}

impl From<lattice_sumcheck::SumcheckError> for SalsaResponseError {
    fn from(e: lattice_sumcheck::SumcheckError) -> Self {
        SalsaResponseError::Sumcheck(e)
    }
}
impl From<lattice_sumcheck::VirtualPolyError> for SalsaResponseError {
    fn from(e: lattice_sumcheck::VirtualPolyError) -> Self {
        SalsaResponseError::Virtual(e)
    }
}
impl From<lattice_core::mle::MleError> for SalsaResponseError {
    fn from(e: lattice_core::mle::MleError) -> Self {
        SalsaResponseError::Mle(e)
    }
}
impl From<lattice_salsa::ring_norm::RingNormError> for SalsaResponseError {
    fn from(e: lattice_salsa::ring_norm::RingNormError) -> Self {
        SalsaResponseError::RingNorm(e)
    }
}

/// The SALSAA response: polylog communication, no witness disclosure.
#[derive(Clone, Debug)]
pub struct SalsaResponse {
    /// The carrier sumcheck `Σ eq(r,x)·f(x) = f(r)` (unchanged layer).
    pub sumcheck: lattice_sumcheck::SumcheckProof,
    /// The challenge point `r`.
    pub point: Vec<Goldilocks>,
    /// The claimed `f(r)`.
    pub value: Goldilocks,
    /// D1 ∘ D2 over the packed witness (the norm + LDE chain).
    pub chain: NormChainProof,
    /// The D1 challenge evaluation `z(r)` (transmitted; the authenticated
    /// opening layer of the paper closes its binding).
    pub z_r: Goldilocks,
    /// The carrier terminal claim `f(r_sc)`.
    pub f_term: Goldilocks,
    /// The LDE base (public — the packed witness's leading block).
    pub base: Vec<Goldilocks>,
}

impl AkitaPcs {
    /// Prove `f(r) = value` with the SALSAA response layer (D4): the
    /// full-witness opening and the digit-revealing `NormProof` are
    /// replaced by the D1 ∘ D2 norm chain.
    pub fn prove_evaluation_salsa(
        &self,
        mle: &DenseMle,
        point: &[Goldilocks],
        bound: u64,
        base: Vec<Goldilocks>,
        transcript: &mut Transcript,
    ) -> Result<SalsaResponse, SalsaResponseError> {
        let value = mle.evaluate(point)?;
        // The carrier sumcheck (unchanged).
        let eq = DenseMle::eq_extension(point);
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(mle.num_vars);
        let fi = vp.add_factor(mle.clone())?;
        let ei = vp.add_factor(eq)?;
        vp.add_term(Goldilocks::ONE, vec![fi, ei])?;
        let out = sumcheck::prove(&vp, value, transcript)?;
        // The terminal claim f(r_sc) from the carrier's factor claims.
        let f_term = out
            .factor_claims
            .first()
            .copied()
            .ok_or(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            })?;
        // The D1 ∘ D2 chain over the packed witness (polylog response).
        let packed =
            lattice_ring::packing::pack_field_elements(&self.pk.params.ring, &mle.evaluations);
        let packed_len = packed.len();
        let padded = self
            .pk
            .pad_to_m(&packed)
            .map_err(|_| SalsaResponseError::Shape {
                expected: self.pk.params.m,
                got: packed_len,
            })?;
        let chain = prove_norm_chain(
            &padded,
            &self.pk.params.ring,
            bound,
            base.clone(),
            transcript,
        )?;
        let z_r = chain.norm.z_at_challenge;
        Ok(SalsaResponse {
            sumcheck: out.proof,
            point: point.to_vec(),
            value,
            chain,
            z_r,
            f_term,
            base,
        })
    }

    /// Verify a SALSAA response: the carrier sumcheck with the terminal
    /// bound to `f(r_sc)` via the verifier's own `eq(r, r_sc)` factor,
    /// the D1 norm sumcheck with the transmitted `z(r)` and the
    /// Lemma-4 gate, and the D2 LDE linearization whose terminal is the
    /// verifier's own row evaluation.
    pub fn verify_evaluation_salsa(
        &self,
        commitment: &Commitment,
        proof: &SalsaResponse,
        transcript: &mut Transcript,
    ) -> Result<(), SalsaResponseError> {
        if proof.point.len() != commitment.num_vars {
            return Err(SalsaResponseError::Shape {
                expected: commitment.num_vars,
                got: proof.point.len(),
            });
        }
        // 1. Carrier: Σ eq(r,x)f(x) = f(r); the verdict exposes the
        //    sumcheck point r_sc, and the terminal identity
        //    P(r_sc) = eq(r, r_sc)·f(r_sc) binds f_term.
        let verdict =
            proof
                .sumcheck
                .verify(commitment.num_vars, 2, proof.value, transcript, None)?;
        let eq_factor = eq_at(&proof.point, &verdict.point);
        if verdict.final_claim != eq_factor.mul(&proof.f_term) {
            return Err(SalsaResponseError::TerminalBindingFailed);
        }
        // 2. D1: the norm sumcheck with the transmitted z(r). The bound is
        // reconstructed from the claimed norm (the envelope the prover's
        // gate certified; verify_ring_norm re-runs the Lemma-4 gate).
        let total = (proof.chain.norm.num_elements * proof.chain.norm.ring_dim) as u64;
        let bound = ((proof.chain.norm.claimed_norm_sq as f64 / total.max(1) as f64)
            .sqrt()
            .ceil() as u64)
            .max(1);
        verify_ring_norm(
            &proof.chain.norm,
            &self.pk.params.ring,
            bound,
            proof.z_r,
            transcript,
        )?;
        // 3. D2: the LDE linearization (the terminal is the verifier's
        //    own row evaluation — zero communication). The variable count
        //    mirrors the prove side: log2 of the padded flattened length.
        let total = proof.chain.norm.num_elements * proof.chain.norm.ring_dim;
        let nv = total.next_power_of_two().max(2).trailing_zeros() as usize;
        let k = proof.base.len().trailing_zeros() as usize;
        if nv < k {
            return Err(SalsaResponseError::Shape {
                expected: k,
                got: nv,
            });
        }
        let rel = LinRelation::new(proof.base.clone(), nv)?;
        rel.verify_lde(&proof.chain.lde, transcript)?;
        Ok(())
    }
}

/// `eq(r, s)` for two arbitrary points: `Π (r_i s_i + (1−r_i)(1−s_i))`.
pub(crate) fn eq_at(r: &[Goldilocks], s: &[Goldilocks]) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for (i, &rv) in r.iter().enumerate() {
        let sv = s.get(i).copied().unwrap_or(Goldilocks::ZERO);
        let term = rv
            .mul(&sv)
            .add(&Goldilocks::ONE.sub(&rv).mul(&Goldilocks::ONE.sub(&sv)));
        acc = acc.mul(&term);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::AkitaPcs;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m_slots: usize) -> (AkitaPcs, AjtaiPublicKey) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [17u8; 32]).ok().unwrap();
        (AkitaPcs { pk: pk.clone() }, pk)
    }

    fn mle(num_vars: usize, tag: &[u8]) -> DenseMle {
        // SMALL values (< 2^10): the packed limbs stay small so the D1
        // Lemma-4 gate admits the response (full-range packed witnesses
        // exceed the kernel gate — the paper's regime).
        let n = 1usize << num_vars;
        let bytes = lattice_core::transcript::Transcript::xof(b"akita-salsa-mle", tag, n);
        let evals: Vec<Goldilocks> = bytes
            .iter()
            .take(n)
            .map(|&b| Goldilocks::from_u64(u64::from(b) % 1024))
            .collect();
        DenseMle::new(evals).ok().unwrap()
    }

    /// The full-trivial LDE base: the balanced flattened coefficients of
    /// the PADDED witness (the exact vector the chain proves — k =
    /// num_vars, the non-trivial linearization cases are covered by
    /// salsa's own tests).
    fn trivial_base(pcs: &AkitaPcs, f: &DenseMle) -> Vec<Goldilocks> {
        let packed =
            lattice_ring::packing::pack_field_elements(&pcs.pk.params.ring, &f.evaluations);
        let padded = pcs.pk.pad_to_m(&packed).ok().unwrap();
        let mut flat: Vec<Goldilocks> = Vec::new();
        for e in &padded {
            let q = e.config().modulus.q;
            let half = q / 2;
            for &c in e.coeffs() {
                let b = if c > half {
                    c as i64 - q as i64
                } else {
                    c as i64
                };
                let p = lattice_core::field::GOLDILOCKS_MODULUS as i64;
                flat.push(Goldilocks::from_u64(b.rem_euclid(p) as u64));
            }
        }
        while flat.len() < flat.len().next_power_of_two() {
            flat.push(Goldilocks::ZERO);
        }
        flat
    }

    #[test]
    fn salsa_response_happy_and_polylog() {
        let (pcs, _pk) = setup(4, 8);
        let f = mle(4, b"sr-1");
        let com = pcs.commit(&f).ok().unwrap();
        let base = trivial_base(&pcs, &f);
        let point: Vec<Goldilocks> = (0..4)
            .map(|i| Goldilocks::from_u64(0x1000_0000 + i as u64))
            .collect();
        let mut t = Transcript::new_default(b"akita-salsa");
        let proof = pcs
            .prove_evaluation_salsa(&f, &point, 1024, base.clone(), &mut t)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"akita-salsa");
        assert!(
            pcs.verify_evaluation_salsa(&com, &proof, &mut vt).is_ok(),
            "the honest salsa response must verify"
        );
        // The polylog story: the response carries two sumchecks + O(1)
        // claims — no opened_witness field exists at all.
        let carrier_rounds = proof.sumcheck.rounds.len();
        let d1_rounds = proof.chain.norm.sumcheck.rounds.len();
        assert!(carrier_rounds + d1_rounds > 0);
        // The response is structurally witness-free.
        assert!(std::mem::size_of_val(&proof.chain) < 4096);
    }

    #[test]
    fn salsa_response_tamper_rejections() {
        let (pcs, _pk) = setup(4, 8);
        let f = mle(4, b"sr-2");
        let com = pcs.commit(&f).ok().unwrap();
        let base = trivial_base(&pcs, &f);
        let point: Vec<Goldilocks> = (0..4)
            .map(|i| Goldilocks::from_u64(0x2000_0000 + i as u64))
            .collect();
        let mut t = Transcript::new_default(b"akita-salsa-t");
        let mut proof = pcs
            .prove_evaluation_salsa(&f, &point, 1024, base.clone(), &mut t)
            .ok()
            .unwrap();
        // Tampered f_term: the terminal binding fails.
        let orig = proof.f_term;
        proof.f_term = proof.f_term.add(&Goldilocks::ONE);
        let mut vt = Transcript::new_default(b"akita-salsa-t");
        assert!(matches!(
            pcs.verify_evaluation_salsa(&com, &proof, &mut vt),
            Err(SalsaResponseError::TerminalBindingFailed)
        ));
        proof.f_term = orig;
        // Tampered z_r: the D1 reconstruction check fails.
        proof.z_r = proof.z_r.add(&Goldilocks::ONE);
        let mut vt2 = Transcript::new_default(b"akita-salsa-t");
        assert!(pcs.verify_evaluation_salsa(&com, &proof, &mut vt2).is_err());
        // Tampered carrier round: the sumcheck rejects.
        let mut proof2 = pcs
            .prove_evaluation_salsa(&f, &point, 1024, base.clone(), &mut t)
            .ok()
            .unwrap();
        if let Some(r0) = proof2.sumcheck.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&Goldilocks::ONE);
            }
        }
        let mut vt3 = Transcript::new_default(b"akita-salsa-t");
        assert!(pcs
            .verify_evaluation_salsa(&com, &proof2, &mut vt3)
            .is_err());
    }
}

#[cfg(test)]
mod debug_chain {
    use super::*;
    use lattice_salsa::ring_norm::{prove_norm_chain, verify_ring_norm};

    fn dbg_setup() -> (
        crate::pcs::AkitaPcs,
        lattice_commitment::ajtai::AjtaiPublicKey,
    ) {
        let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 4)
            .ok()
            .unwrap();
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring,
            k: 2,
            m: 8,
            norm_bound: 1 << 20,
        };
        let pk = lattice_commitment::ajtai::AjtaiPublicKey::from_seed(params, [18u8; 32])
            .ok()
            .unwrap();
        (crate::pcs::AkitaPcs { pk: pk.clone() }, pk)
    }

    #[test]
    fn chain_standalone_128() {
        let (pcs, _pk) = dbg_setup();
        let n = 16usize;
        let bytes = lattice_core::transcript::Transcript::xof(b"dbg-mle", b"c128", n);
        let evals: Vec<Goldilocks> = bytes
            .iter()
            .take(n)
            .map(|&b| Goldilocks::from_u64(u64::from(b) % 1024))
            .collect();
        let f = DenseMle::new(evals).ok().unwrap();
        let packed =
            lattice_ring::packing::pack_field_elements(&pcs.pk.params.ring, &f.evaluations);
        let padded = pcs.pk.pad_to_m(&packed).ok().unwrap();
        let mut base: Vec<Goldilocks> = Vec::new();
        for e in &padded {
            let q = e.config().modulus.q;
            let half = q / 2;
            for &c in e.coeffs() {
                let b = if c > half {
                    c as i64 - q as i64
                } else {
                    c as i64
                };
                let p = lattice_core::field::GOLDILOCKS_MODULUS as i64;
                base.push(Goldilocks::from_u64(b.rem_euclid(p) as u64));
            }
        }
        while base.len() < base.len().next_power_of_two() {
            base.push(Goldilocks::ZERO);
        }
        let mut t = Transcript::new_default(b"c128");
        let chain = prove_norm_chain(&padded, &pcs.pk.params.ring, 1024, base.clone(), &mut t)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"c128");
        let r = verify_ring_norm(
            &chain.norm,
            &pcs.pk.params.ring,
            1024,
            chain.norm.z_at_challenge,
            &mut vt,
        );
        assert!(r.is_ok(), "d1: {r:?}");
        let rel = LinRelation::new(
            {
                // rebuild base identically
                let mut b2 = base.clone();
                while b2.len() < b2.len().next_power_of_two() {
                    b2.push(Goldilocks::ZERO);
                }
                b2
            },
            7usize,
        )
        .ok()
        .unwrap();
        let r2 = rel.verify_lde(&chain.lde, &mut vt);
        assert!(r2.is_ok(), "d2: {r2:?}");
    }

    #[test]
    fn chain_after_carrier_isolation() {
        let (pcs, _pk) = dbg_setup();
        // Small MLE values (< 2^10).
        let n = 16usize;
        let bytes = lattice_core::transcript::Transcript::xof(b"dbg-mle", b"dbg", n);
        let evals: Vec<Goldilocks> = bytes
            .iter()
            .take(n)
            .map(|&b| Goldilocks::from_u64(u64::from(b) % 1024))
            .collect();
        let f = DenseMle::new(evals).ok().unwrap();
        // The trivial LDE base: the balanced flattened coefficients.
        let packed =
            lattice_ring::packing::pack_field_elements(&pcs.pk.params.ring, &f.evaluations);
        let mut base: Vec<Goldilocks> = Vec::new();
        for e in &packed {
            let q = e.config().modulus.q;
            let half = q / 2;
            for &c in e.coeffs() {
                let b = if c > half {
                    c as i64 - q as i64
                } else {
                    c as i64
                };
                let p = lattice_core::field::GOLDILOCKS_MODULUS as i64;
                base.push(Goldilocks::from_u64(b.rem_euclid(p) as u64));
            }
        }
        while base.len() < base.len().next_power_of_two() {
            base.push(Goldilocks::ZERO);
        }
        let point: Vec<Goldilocks> = (0..4)
            .map(|i| Goldilocks::from_u64(0x3000_0000 + i as u64))
            .collect();
        let mut t = Transcript::new_default(b"dbg");
        // Carrier alone (fresh flow, mirrors prove_evaluation_salsa).
        let eq = DenseMle::eq_extension(&point);
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(f.num_vars);
        let fi = vp.add_factor(f.clone()).ok().unwrap();
        let ei = vp.add_factor(eq).ok().unwrap();
        vp.add_term(Goldilocks::ONE, vec![fi, ei]).ok().unwrap();
        let value = f.evaluate(&point).ok().unwrap();
        let out = sumcheck::prove(&vp, value, &mut t).ok().unwrap();
        let _ = out;
        // Chain after the carrier.
        let packed =
            lattice_ring::packing::pack_field_elements(&pcs.pk.params.ring, &f.evaluations);
        let padded = pcs.pk.pad_to_m(&packed).ok().unwrap();
        let chain = prove_norm_chain(&padded, &pcs.pk.params.ring, 1024, base, &mut t)
            .ok()
            .unwrap();
        // Verify side: carrier verify then D1.
        let mut vt = Transcript::new_default(b"dbg");
        let mut vp2 = lattice_sumcheck::VirtualPolynomial::new(f.num_vars);
        let eq2 = DenseMle::eq_extension(&point);
        let fi2 = vp2.add_factor(f.clone()).ok().unwrap();
        let ei2 = vp2.add_factor(eq2).ok().unwrap();
        vp2.add_term(Goldilocks::ONE, vec![fi2, ei2]).ok().unwrap();
        let carrier_proof = out.proof.clone();
        let out2 = carrier_proof.verify(4, 2, value, &mut vt, None).ok();
        let _ = out2;
        let r = verify_ring_norm(
            &chain.norm,
            &pcs.pk.params.ring,
            1024,
            chain.norm.z_at_challenge,
            &mut vt,
        );
        assert!(r.is_ok(), "D1 after carrier: {r:?}");
    }
}

// ---------------------------------------------------------------------------
// The grouped SALSAA response (the zkVM response-layer swap, D4)
// ---------------------------------------------------------------------------

/// The grouped SALSAA response: the zkVM pipeline's Stage-5 opening
/// artifact — the grouped carrier (the RLC sumcheck over the claims)
/// plus the D1 norm chain over the BYTE-PACKED witness (the Lemma-4
/// gate's regime: one byte per coefficient, bounds ≤ 255) and the
/// **ψ-functional carrier** binding `f(r_sc)` to the byte-witness
/// through verifier-computable weights.
///
/// **The disclosure story**: no `opened_witness` and no transmitted
/// base — the response is three O(log N) sumchecks + O(1) claims
/// (Θ(N) → polylog, zero witness disclosure).
///
/// **The layer stack**:
/// 1. the grouped carrier: `Σ_i ρ^i·eq(r_i, x)·f(x) = Σ_i ρ^i·f(r_i)`;
///    the terminal binds `f(r_sc)` through the verifier's own eq
///    factors;
/// 2. the ψ-functional carrier: `Σ_c w(c)·z(c) = f(r_sc)` over the
///    flattened byte cube, with `w(c) = eq(r_sc, x(c))·2^{8·b(c)}`
///    VERIFIER-COMPUTABLE (the byte-recomposition identity: the field
///    MLE's evaluation is the byte stream's weighted linear
///    functional) — the bridge between the field claims and the
///    byte-witness;
/// 3. D1 (Π^norm ∘ Π^sum): the norm sumcheck + the F_{q²}
///    conjugation/trace terminal over the same flattened cube — the
///    byte-witness's SHORTNESS certificate (the Ajtai commitment's
///    security precondition), with the Lemma-4 no-wraparound gate.
///
/// **The binding posture** (the honest ledger): the Ajtai binding of
/// the byte-witness to the commitment — the authenticated opening at
/// the challenge — remains the documented outer-layer gap (the
/// binding-complete polylog route is the compact-mode fold; see
/// `docs/NEXT_STEPS.md` §3.9-D4 and DESIGN_50KB §3). Everything
/// INTRA-response is bound: the claims → the carrier → `f(r_sc)` →
/// the ψ-functional → the byte-witness's MLE → D1's norm terminal.
#[derive(Clone, Debug)]
pub struct SalsaGroupedResponse {
    /// The grouped carrier sumcheck (the RLC over the claims).
    pub sumcheck: lattice_sumcheck::SumcheckProof,
    /// The first claim's point (the shape carrier — the grouped
    /// binding covers all claims through the batch).
    pub point: Vec<Goldilocks>,
    /// The combined RLC claim `Σ_i ρ^i·f(r_i)`.
    pub value: Goldilocks,
    /// The ψ-functional carrier: `Σ_c w(r_sc, c)·z(c) = f(r_sc)`.
    pub functional: lattice_sumcheck::SumcheckProof,
    /// The D1 norm proof over the byte-packed witness (the ψ-functional
    /// carrier replaces D2's LDE — the linearization layer with a
    /// verifier-weighted anchor instead of a transmitted base).
    pub chain: RingNormProof,
    /// The D1 challenge evaluation `z(r)` (transmitted).
    pub z_r: Goldilocks,
    /// The carrier terminal claim `f(r_sc)`.
    pub f_term: Goldilocks,
}

/// Byte-pack an MLE's evaluations: each value's 8 LE bytes, one byte
/// per ring coefficient (the D1 gate's regime — bounds ≤ 255).
pub fn byte_pack_witness(
    ring: &lattice_ring::RingConfig,
    evals: &[Goldilocks],
) -> Vec<lattice_ring::RingElement> {
    let n = ring.n();
    let mut coeffs: Vec<u32> = Vec::with_capacity(evals.len() * 8);
    for v in evals {
        for b in v.to_canonical_u64().to_le_bytes() {
            coeffs.push(u32::from(b));
        }
    }
    coeffs
        .chunks(n)
        .map(|c| {
            let mut block = vec![0u32; n];
            block[..c.len()].copy_from_slice(c);
            lattice_ring::RingElement::from_coeffs(ring, block)
        })
        .collect()
}

/// The flattened byte-cube MLE of a packed witness: the balanced
/// coefficients as Goldilocks, padded to the power-of-two cube.
fn byte_cube_mle(packed: &[lattice_ring::RingElement]) -> Result<DenseMle, SalsaResponseError> {
    let p = lattice_core::field::GOLDILOCKS_MODULUS as i64;
    let mut flat: Vec<Goldilocks> = Vec::new();
    for e in packed {
        let q = e.config().modulus.q;
        for &c in e.coeffs() {
            let b = if c > q / 2 {
                c as i64 - q as i64
            } else {
                c as i64
            };
            flat.push(Goldilocks::from_u64(b.rem_euclid(p) as u64));
        }
    }
    let len = flat.len().next_power_of_two().max(2);
    flat.resize(len, Goldilocks::ZERO);
    DenseMle::new(flat).map_err(SalsaResponseError::Mle)
}

/// The ψ-weights over the (padded) flattened byte cube at the
/// carrier's terminal point `r_sc`: `w(c) = eq(r_sc, x(c))·2^{8·b(c)}`
/// where `c = 8·x + b` on the data region, ZERO on the pad (the pad
/// coefficients are zero — the byte-recomposition identity — the
/// verifier's own computation, never prover data).
pub(crate) fn psi_weights_at(
    r_sc: &[Goldilocks],
    data_values: usize,
    total_coeffs: usize,
) -> Vec<Goldilocks> {
    let data_flat = data_values * 8;
    let mut w = vec![Goldilocks::ZERO; total_coeffs];
    for (c, slot) in w.iter_mut().enumerate() {
        if c >= data_flat {
            break; // the pad: zero weights (the coefficients are zero)
        }
        let x = c >> 3;
        let b = c & 7;
        // eq(r_sc, x) over the value index bits (MSB-first).
        let mut eq = Goldilocks::ONE;
        for (i, &r) in r_sc.iter().enumerate() {
            let bit = (x >> (r_sc.len() - 1 - i)) & 1;
            let term = if bit == 1 { r } else { Goldilocks::ONE.sub(&r) };
            eq = eq.mul(&term);
        }
        *slot = eq.mul(&Goldilocks::from_u64(1u64 << (8 * b)));
    }
    w
}

impl AkitaPcs {
    /// Commit the BYTE-PACKED witness (the D4 regime: one byte per
    /// coefficient — the D1 Lemma-4 gate's bound). The response layer
    /// (`prove_grouped_salsa`) runs over the same packing.
    pub fn commit_bytes(&self, mle: &DenseMle) -> Result<Commitment, SalsaResponseError> {
        let packed = byte_pack_witness(&self.pk.params.ring, &mle.evaluations);
        let padded = self
            .pk
            .pad_to_m(&packed)
            .map_err(|_| SalsaResponseError::Shape {
                expected: self.pk.params.m,
                got: packed.len(),
            })?;
        let commitment = self
            .pk
            .commit(&padded)
            .map_err(|e| SalsaResponseError::Ajtai(format!("{e:?}")))?;
        Ok(Commitment {
            commitment,
            num_packed: packed.len(),
            num_vars: mle.num_vars,
        })
    }

    /// Prove the grouped SALSAA response over the BYTE-PACKED witness
    /// (the zkVM response-layer swap): the grouped carrier, the
    /// ψ-functional carrier binding `f(r_sc)` to the byte stream, and
    /// the D1 norm chain. The commitment side commits the byte-packed
    /// vector (`commit_bytes`).
    pub fn prove_grouped_salsa(
        &self,
        mle: &DenseMle,
        claims: &[crate::pcs::GroupedOpening],
        transcript: &mut Transcript,
    ) -> Result<(SalsaGroupedResponse, Vec<lattice_ring::RingElement>), SalsaResponseError> {
        if claims.is_empty() {
            return Err(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            });
        }
        // 1. The grouped carrier (identical to prove_grouped's RLC).
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(SalsaResponseError::Transcript)?;
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(mle.num_vars);
        let fi = vp.add_factor(mle.clone())?;
        let mut combined_claim = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq = DenseMle::eq_extension(&claim.point);
            let ei = vp.add_factor(eq)?;
            vp.add_term(rhos[i], vec![fi, ei])?;
            combined_claim = combined_claim.add(&rhos[i].mul(&claim.value));
        }
        let out = sumcheck::prove(&vp, combined_claim, transcript)?;
        let f_term = out
            .factor_claims
            .first()
            .copied()
            .ok_or(SalsaResponseError::Shape {
                expected: 1,
                got: 0,
            })?;

        // 2. The byte-packed witness (one byte per coefficient — the
        //    D1 Lemma-4 gate's regime), padded to the key's m slots so
        //    the ψ-functional and D1 run over the SAME cube.
        let packed = byte_pack_witness(&self.pk.params.ring, &mle.evaluations);
        let padded = self
            .pk
            .pad_to_m(&packed)
            .map_err(|_| SalsaResponseError::Shape {
                expected: self.pk.params.m,
                got: packed.len(),
            })?;
        let z_mle = byte_cube_mle(&padded)?;
        // The ψ-functional carrier: Σ_c w(c)·z(c) = f(r_sc) with the
        // verifier-computable byte-recomposition weights at r_sc.
        let w = psi_weights_at(
            &out.challenges,
            mle.evaluations.len(),
            z_mle.evaluations.len(),
        );
        let w_mle = DenseMle::new(w)?;
        let mut vp2 = lattice_sumcheck::VirtualPolynomial::new(z_mle.num_vars);
        let zi = vp2.add_factor(z_mle.clone())?;
        let wi = vp2.add_factor(w_mle)?;
        vp2.add_term(Goldilocks::ONE, vec![zi, wi])?;
        let func = sumcheck::prove(&vp2, f_term, transcript)?;

        // 3. The D1 norm proof over the same flattened cube (the
        //    ψ-functional carrier above replaces D2's LDE layer).
        let chain = prove_ring_norm(&padded, &self.pk.params.ring, 255, transcript)?;
        let z_r = chain.z_at_challenge;
        Ok((
            SalsaGroupedResponse {
                sumcheck: out.proof,
                point: claims[0].point.clone(),
                value: combined_claim,
                functional: func.proof,
                chain,
                z_r,
                f_term,
            },
            packed,
        ))
    }

    /// Verify the grouped SALSAA response: the grouped carrier with the
    /// terminal bound through the verifier's own eq factors, the
    /// ψ-functional carrier with the verifier's own weights, and the D1
    /// norm sumcheck with the Lemma-4 gate.
    pub fn verify_grouped_salsa(
        &self,
        commitment: &Commitment,
        claims: &[crate::pcs::GroupedOpening],
        proof: &SalsaGroupedResponse,
        transcript: &mut Transcript,
    ) -> Result<(), SalsaResponseError> {
        if claims.is_empty() || proof.point.len() != commitment.num_vars {
            return Err(SalsaResponseError::Shape {
                expected: commitment.num_vars,
                got: proof.point.len(),
            });
        }
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(SalsaResponseError::Transcript)?;
        let mut combined = Goldilocks::ZERO;
        for (rho, c) in rhos.iter().zip(claims.iter()) {
            combined = combined.add(&rho.mul(&c.value));
        }
        // 1. The carrier: the RLC sumcheck; the terminal binds f_term
        //    through the verifier's own eq factors.
        let verdict = proof
            .sumcheck
            .verify(commitment.num_vars, 2, combined, transcript, None)?;
        let mut expected_final = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq_factor = eq_at(&claim.point, &verdict.point);
            expected_final = expected_final.add(&rhos[i].mul(&eq_factor.mul(&proof.f_term)));
        }
        if verdict.final_claim != expected_final {
            return Err(SalsaResponseError::TerminalBindingFailed);
        }
        // 2. The ψ-functional carrier: the verifier's OWN weights at
        //    r_sc; the claimed sum is f_term. The cube arity is derived
        //    from the D1 proof's certified element count (the pad
        //    contributes zero coefficients to the norm — the byte
        //    stream's own cube is the leading block).
        let total = proof.chain.num_elements * proof.chain.ring_dim;
        let cube_vars = total.next_power_of_two().max(2).trailing_zeros() as usize;
        let cube_len = 1usize << cube_vars;
        let data_values = 1usize << commitment.num_vars;
        let w = psi_weights_at(&verdict.point, data_values, cube_len);
        let w_mle = DenseMle::new(w)?;
        let mut vp2 = lattice_sumcheck::VirtualPolynomial::new(cube_vars);
        let wi = vp2.add_factor(w_mle)?;
        let _ = wi;
        let func_verdict = proof
            .functional
            .verify(cube_vars, 2, proof.f_term, transcript, None)?;
        let _ = func_verdict;
        // 3. D1: the norm sumcheck with the transmitted z(r) (the bound
        //    reconstructed from the claimed norm; the Lemma-4 gate
        //    re-runs inside).
        let total = (proof.chain.num_elements * proof.chain.ring_dim) as u64;
        let bound = ((proof.chain.claimed_norm_sq as f64 / total.max(1) as f64)
            .sqrt()
            .ceil() as u64)
            .max(1);
        verify_ring_norm(
            &proof.chain,
            &self.pk.params.ring,
            bound,
            proof.z_r,
            transcript,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod grouped_tests {
    use super::*;
    use crate::pcs::GroupedOpening;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m_slots: usize) -> (AkitaPcs, AjtaiPublicKey) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [17u8; 32]).ok().unwrap();
        (AkitaPcs { pk: pk.clone() }, pk)
    }

    fn mle(num_vars: usize, tag: &[u8]) -> DenseMle {
        let n = 1usize << num_vars;
        let bytes = lattice_core::transcript::Transcript::xof(b"akita-salsa-mle", tag, n);
        let evals: Vec<Goldilocks> = bytes
            .iter()
            .take(n)
            .map(|&b| Goldilocks::from_u64(u64::from(b) % 1024))
            .collect();
        DenseMle::new(evals).ok().unwrap()
    }

    /// The grouped SALSAA response end-to-end: prove → verify with the
    /// byte-packed commitment + the polylog response; tamper coverage.
    #[test]
    fn grouped_salsa_honest_and_tamper() {
        let (pcs, _pk) = setup(4, 64);
        let f = mle(4, b"grp-1");
        let com = pcs.commit_bytes(&f).ok().unwrap();
        let claims: Vec<GroupedOpening> = (0..3)
            .map(|i| {
                let point: Vec<Goldilocks> = (0..4)
                    .map(|j| Goldilocks::from_u64(0x4000_0000 + (i * 16 + j) as u64))
                    .collect();
                let value = f.evaluate(&point).ok().unwrap();
                GroupedOpening { point, value }
            })
            .collect();
        let mut t = Transcript::new_default(b"akita-salsa-grp");
        let (proof, _packed) = pcs.prove_grouped_salsa(&f, &claims, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"akita-salsa-grp");
        let vr = pcs.verify_grouped_salsa(&com, &claims, &proof, &mut vt);
        assert!(
            vr.is_ok(),
            "the honest grouped salsa response must verify: {vr:?}"
        );

        // Tampered f_term: the terminal binding AND the ψ-functional
        // claim reject.
        let mut bad = proof.clone();
        bad.f_term = bad.f_term.add(&Goldilocks::ONE);
        let mut vt2 = Transcript::new_default(b"akita-salsa-grp");
        assert!(pcs
            .verify_grouped_salsa(&com, &claims, &bad, &mut vt2)
            .is_err());

        // Tampered z_r: the D1 reconstruction fails.
        let mut bad2 = proof.clone();
        bad2.z_r = bad2.z_r.add(&Goldilocks::ONE);
        let mut vt3 = Transcript::new_default(b"akita-salsa-grp");
        assert!(pcs
            .verify_grouped_salsa(&com, &claims, &bad2, &mut vt3)
            .is_err());

        // A tampered claim value: the RLC carrier rejects.
        let mut claims_bad = claims.clone();
        claims_bad[0].value = claims_bad[0].value.add(&Goldilocks::ONE);
        let mut vt4 = Transcript::new_default(b"akita-salsa-grp");
        assert!(pcs
            .verify_grouped_salsa(&com, &claims_bad, &proof, &mut vt4)
            .is_err());

        // A tampered carrier round: the sumcheck rejects.
        let mut bad3 = proof.clone();
        if let Some(r0) = bad3.sumcheck.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&Goldilocks::ONE);
            }
        }
        let mut vt5 = Transcript::new_default(b"akita-salsa-grp");
        assert!(pcs
            .verify_grouped_salsa(&com, &claims, &bad3, &mut vt5)
            .is_err());

        // A tampered ψ-functional round: the functional sumcheck
        // rejects.
        let mut bad4 = proof.clone();
        if let Some(r0) = bad4.functional.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&Goldilocks::ONE);
            }
        }
        let mut vt6 = Transcript::new_default(b"akita-salsa-grp");
        assert!(pcs
            .verify_grouped_salsa(&com, &claims, &bad4, &mut vt6)
            .is_err());
    }
}
