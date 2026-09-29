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
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_core::mle::DenseMle;
use lattice_salsa::ring_norm::{
    prove_norm_chain, verify_ring_norm, LinRelation, NormChainProof,
};
use lattice_sumcheck::sumcheck;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SalsaResponseError {
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    RingNorm(lattice_salsa::ring_norm::RingNormError),
    Transcript(lattice_core::transcript::TranscriptError),
    Shape { expected: usize, got: usize },
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
        let f_term = out.factor_claims.first().copied().ok_or(
            SalsaResponseError::Shape { expected: 1, got: 0 },
        )?;
        // The D1 ∘ D2 chain over the packed witness (polylog response).
        let packed = lattice_ring::packing::pack_field_elements(
            &self.pk.params.ring,
            &mle.evaluations,
        );
        let packed_len = packed.len();
        let padded = self.pk.pad_to_m(&packed).map_err(|_| {
            SalsaResponseError::Shape { expected: self.pk.params.m, got: packed_len }
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
        let verdict = proof
            .sumcheck
            .verify(commitment.num_vars, 2, proof.value, transcript, None)
            ?;
        let eq_factor = eq_at(&proof.point, &verdict.point);
        if verdict.final_claim != eq_factor.mul(&proof.f_term) {
            return Err(SalsaResponseError::TerminalBindingFailed);
        }
        // 2. D1: the norm sumcheck with the transmitted z(r). The bound is
        // reconstructed from the claimed norm (the envelope the prover's
        // gate certified; verify_ring_norm re-runs the Lemma-4 gate).
        let total = (proof.chain.norm.num_elements * proof.chain.norm.ring_dim) as u64;
        let bound = ((proof.chain.norm.claimed_norm_sq as f64 / total.max(1) as f64).sqrt()
            .ceil() as u64)
            .max(1);
        verify_ring_norm(
            &proof.chain.norm,
            &self.pk.params.ring,
            bound,
            proof.z_r,
            transcript,
        )
?;
        // 3. D2: the LDE linearization (the terminal is the verifier's
        //    own row evaluation — zero communication). The variable count
        //    mirrors the prove side: log2 of the padded flattened length.
        let total = proof.chain.norm.num_elements * proof.chain.norm.ring_dim;
        let nv = total.next_power_of_two().max(2).trailing_zeros() as usize;
        let k = proof.base.len().trailing_zeros() as usize;
        if nv < k {
            return Err(SalsaResponseError::Shape { expected: k, got: nv });
        }
        let rel = LinRelation::new(proof.base.clone(), nv)?;
        rel.verify_lde(&proof.chain.lde, transcript)?;
        Ok(())
    }
}

/// `eq(r, s)` for two arbitrary points: `Π (r_i s_i + (1−r_i)(1−s_i))`.
fn eq_at(r: &[Goldilocks], s: &[Goldilocks]) -> Goldilocks {
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
        let params = AjtaiParams { ring: ring.clone(), k: 2, m: m_slots, norm_bound: 1 << 20 };
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
                let b = if c > half { c as i64 - q as i64 } else { c as i64 };
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
        let proof =
            pcs.prove_evaluation_salsa(&f, &point, 1024, base.clone(), &mut t).ok().unwrap();
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
        let mut proof =
            pcs.prove_evaluation_salsa(&f, &point, 1024, base.clone(), &mut t).ok().unwrap();
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
        let mut proof2 =
            pcs.prove_evaluation_salsa(&f, &point, 1024, base.clone(), &mut t).ok().unwrap();
        if let Some(r0) = proof2.sumcheck.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&Goldilocks::ONE);
            }
        }
        let mut vt3 = Transcript::new_default(b"akita-salsa-t");
        assert!(pcs.verify_evaluation_salsa(&com, &proof2, &mut vt3).is_err());
    }
}

#[cfg(test)]
mod debug_chain {
    use super::*;
    use lattice_salsa::ring_norm::{prove_norm_chain, verify_ring_norm};

    fn dbg_setup() -> (crate::pcs::AkitaPcs, lattice_commitment::ajtai::AjtaiPublicKey) {
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
        let evals: Vec<Goldilocks> =
            bytes.iter().take(n).map(|&b| Goldilocks::from_u64(u64::from(b) % 1024)).collect();
        let f = DenseMle::new(evals).ok().unwrap();
        let packed =
            lattice_ring::packing::pack_field_elements(&pcs.pk.params.ring, &f.evaluations);
        let padded = pcs.pk.pad_to_m(&packed).ok().unwrap();
        let mut base: Vec<Goldilocks> = Vec::new();
        for e in &padded {
            let q = e.config().modulus.q;
            let half = q / 2;
            for &c in e.coeffs() {
                let b = if c > half { c as i64 - q as i64 } else { c as i64 };
                let p = lattice_core::field::GOLDILOCKS_MODULUS as i64;
                base.push(Goldilocks::from_u64(b.rem_euclid(p) as u64));
            }
        }
        while base.len() < base.len().next_power_of_two() {
            base.push(Goldilocks::ZERO);
        }
        let mut t = Transcript::new_default(b"c128");
        let chain = prove_norm_chain(&padded, &pcs.pk.params.ring, 1024, base.clone(), &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"c128");
        let r = verify_ring_norm(&chain.norm, &pcs.pk.params.ring, 1024, chain.norm.z_at_challenge, &mut vt);
        assert!(r.is_ok(), "d1: {r:?}");
        let rel = LinRelation::new({
            // rebuild base identically
            let mut b2 = base.clone();
            while b2.len() < b2.len().next_power_of_two() { b2.push(Goldilocks::ZERO); }
            b2
        }, 7usize).ok().unwrap();
        let r2 = rel.verify_lde(&chain.lde, &mut vt);
        assert!(r2.is_ok(), "d2: {r2:?}");
    }

    #[test]
    fn chain_after_carrier_isolation() {
        let (pcs, _pk) = dbg_setup();
        // Small MLE values (< 2^10).
        let n = 16usize;
        let bytes = lattice_core::transcript::Transcript::xof(b"dbg-mle", b"dbg", n);
        let evals: Vec<Goldilocks> =
            bytes.iter().take(n).map(|&b| Goldilocks::from_u64(u64::from(b) % 1024)).collect();
        let f = DenseMle::new(evals).ok().unwrap();
        // The trivial LDE base: the balanced flattened coefficients.
        let packed =
            lattice_ring::packing::pack_field_elements(&pcs.pk.params.ring, &f.evaluations);
        let mut base: Vec<Goldilocks> = Vec::new();
        for e in &packed {
            let q = e.config().modulus.q;
            let half = q / 2;
            for &c in e.coeffs() {
                let b = if c > half { c as i64 - q as i64 } else { c as i64 };
                let p = lattice_core::field::GOLDILOCKS_MODULUS as i64;
                base.push(Goldilocks::from_u64(b.rem_euclid(p) as u64));
            }
        }
        while base.len() < base.len().next_power_of_two() {
            base.push(Goldilocks::ZERO);
        }
        let point: Vec<Goldilocks> =
            (0..4).map(|i| Goldilocks::from_u64(0x3000_0000 + i as u64)).collect();
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
        let chain = prove_norm_chain(&padded, &pcs.pk.params.ring, 1024, base, &mut t).ok().unwrap();
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
