//! The Akita polynomial commitment scheme: commit → prove evaluation →
//! verify, with grouped openings and norm-checked responses.

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_commitment::norm_proof::{NormProof, NormProofError};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement};

/// The PCS: an Ajtai key with packing geometry.
pub struct AkitaPcs {
    pub pk: AjtaiPublicKey,
}

/// A commitment: the Ajtai commitment rows plus the packed witness length.
#[derive(Clone, Debug)]
pub struct Commitment {
    pub commitment: AjtaiCommitment,
    /// Number of packed ring elements used (the packing geometry).
    pub num_packed: usize,
    /// Number of MLE variables committed.
    pub num_vars: usize,
}

/// An evaluation proof at a challenge point.
#[derive(Clone, Debug)]
pub struct EvaluationProof {
    /// Sumcheck over eq(r, ·)·f: the carrier/tensor layer.
    pub sumcheck: lattice_sumcheck::SumcheckProof,
    /// The challenge point r.
    pub point: Vec<Goldilocks>,
    /// Claimed f(r).
    pub value: Goldilocks,
    /// Opened packed witness (the response; short because the point binds
    /// all variables — the opening reveals the packing at r only through
    /// the norm-checked response in production; this kernel opens the
    /// full packed witness for the linear binding).
    pub opened_witness: Vec<RingElement>,
    /// Digit-decomposed norm proof for the response (the norm layer).
    pub norm_proof: NormProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaPcsError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    Norm(NormProofError),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Virtual(lattice_sumcheck::VirtualPolyError),
    Mle(lattice_core::mle::MleError),
    Transcript(TranscriptError),
    Packing(lattice_ring::PackingError),
    /// Verification failed.
    VerificationFailed,
    Shape { expected: usize, got: usize },
    NotInImage { coefficient: usize },
}

impl AkitaPcs {
    /// Commit to an MLE: pack evaluations into ring elements and commit.
    pub fn commit(&self, mle: &DenseMle) -> Result<Commitment, AkitaPcsError> {
        let packed =
            lattice_ring::packing::pack_field_elements(&self.pk.params.ring, &mle.evaluations);
        let s = self
            .pk
            .pad_to_m(&packed)
            .map_err(AkitaPcsError::Ajtai)?;
        let commitment = self.pk.commit(&s).map_err(AkitaPcsError::Ajtai)?;
        Ok(Commitment {
            commitment,
            num_packed: packed.len(),
            num_vars: mle.num_vars,
        })
    }

    /// The packed witness (prover-side).
    fn packed_witness(&self, mle: &DenseMle) -> Result<Vec<RingElement>, AkitaPcsError> {
        let packed =
            lattice_ring::packing::pack_field_elements(&self.pk.params.ring, &mle.evaluations);
        self.pk
            .pad_to_m(&packed)
            .map_err(AkitaPcsError::Ajtai)
    }

    /// Prove `f(r) = value` for a committed MLE.
    pub fn prove_evaluation(
        &self,
        mle: &DenseMle,
        point: &[Goldilocks],
        transcript: &mut Transcript,
    ) -> Result<EvaluationProof, AkitaPcsError> {
        let value = mle.evaluate(point).map_err(AkitaPcsError::Mle)?;
        // The evaluation identity: Σ_x eq(r, x)·f(x) = f(r).
        let eq = DenseMle::eq_extension(point);
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(mle.num_vars);
        let fi = vp
            .add_factor(mle.clone())
            .map_err(AkitaPcsError::Virtual)?;
        let ei = vp.add_factor(eq).map_err(AkitaPcsError::Virtual)?;
        vp.add_term(Goldilocks::ONE, vec![fi, ei])
            .map_err(AkitaPcsError::Virtual)?;
        let out = lattice_sumcheck::sumcheck::prove(&vp, value, transcript)
            .map_err(AkitaPcsError::Sumcheck)?;

        // Opened witness + norm proof (the response layer).
        let opened = self.packed_witness(mle)?;
        let norm_proof = NormProof::prove(&opened, self.pk.params.norm_bound)
            .map_err(AkitaPcsError::Norm)?;

        Ok(EvaluationProof {
            sumcheck: out.proof,
            point: point.to_vec(),
            value,
            opened_witness: opened,
            norm_proof,
        })
    }

    /// Verify an evaluation proof against a commitment.
    pub fn verify_evaluation(
        &self,
        commitment: &Commitment,
        proof: &EvaluationProof,
        transcript: &mut Transcript,
    ) -> Result<(), AkitaPcsError> {
        // 1. Shape: point arity matches the committed variables.
        if proof.point.len() != commitment.num_vars {
            return Err(AkitaPcsError::Shape {
                expected: commitment.num_vars,
                got: proof.point.len(),
            });
        }
        // 2. Sumcheck: Σ eq(r,x)f(x) = value — the verifier's final claim
        //    binds through the opened witness's f-evaluation (here: the
        //    witness is opened directly, so the final check runs against
        //    the recomputed evaluation).
        let verdict = proof
            .sumcheck
            .verify(commitment.num_vars, 2, proof.value, transcript, None)
            .map_err(AkitaPcsError::Sumcheck)?;
        // 3. The opened witness must match the commitment (Ajtai binding).
        self.pk
            .verify_opening(&commitment.commitment, &proof.opened_witness)
            .map_err(AkitaPcsError::Ajtai)?;
        // 4. Norm layer: the response respects the SIS norm bound.
        proof
            .norm_proof
            .verify(&proof.opened_witness, self.pk.params.norm_bound)
            .map_err(AkitaPcsError::Norm)?;
        // 5. The final sumcheck claim: P(r_sc) = eq(r, r_sc)·f(r_sc); the
        //    witness is open, so f(r_sc) recomputes exactly.
        let mut unpacked = lattice_ring::packing::unpack_field_elements(
            &self.pk.params.ring,
            &proof.opened_witness,
        )
        .map_err(AkitaPcsError::Packing)?;
        // The packed witness is zero-padded to m slots: keep exactly the
        // committed evaluations.
        let expected_len = 1usize << commitment.num_vars;
        if unpacked.len() < expected_len {
            return Err(AkitaPcsError::Shape {
                expected: expected_len,
                got: unpacked.len(),
            });
        }
        unpacked.truncate(expected_len);
        let f_mle = DenseMle {
            num_vars: commitment.num_vars,
            evaluations: unpacked,
        };
        let f_at_sc = f_mle.evaluate(&verdict.point).map_err(AkitaPcsError::Mle)?;
        let eq_at_sc = DenseMle::eq_extension(&proof.point)
            .evaluate(&verdict.point)
            .map_err(AkitaPcsError::Mle)?;
        let expected_final = eq_at_sc.mul(&f_at_sc);
        if verdict.final_claim != expected_final {
            return Err(AkitaPcsError::VerificationFailed);
        }
        // 6. The claimed value itself: f(r) from the opened witness.
        let f_at_r = f_mle.evaluate(&proof.point).map_err(AkitaPcsError::Mle)?;
        if f_at_r != proof.value {
            return Err(AkitaPcsError::VerificationFailed);
        }
        Ok(())
    }
}

/// Standalone verify helper (matches the crate-level re-export).
pub fn verify_evaluation(
    pcs: &AkitaPcs,
    commitment: &Commitment,
    proof: &EvaluationProof,
    transcript: &mut Transcript,
) -> Result<(), AkitaPcsError> {
    pcs.verify_evaluation(commitment, proof, transcript)
}

/// A grouped opening claim: (point, value) against a commitment.
#[derive(Clone, Debug)]
pub struct GroupedOpening {
    pub point: Vec<Goldilocks>,
    pub value: Goldilocks,
}

impl AkitaPcs {
    /// Grouped openings: batch multiple evaluation claims against ONE
    /// commitment into a single combined sumcheck (the prefix-packing
    /// grouped-opening flow: common-prefix claims share the eq tensor).
    pub fn prove_grouped(
        &self,
        mle: &DenseMle,
        claims: &[GroupedOpening],
        transcript: &mut Transcript,
    ) -> Result<EvaluationProof, AkitaPcsError> {
        if claims.is_empty() {
            return Err(AkitaPcsError::Shape {
                expected: 1,
                got: 0,
            });
        }
        // Batch challenges ρ.
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(AkitaPcsError::Transcript)?;
        // Combined polynomial: Σ_i ρ^i · eq(r_i, x)·f(x) — a single
        // sumcheck whose claim is Σ_i ρ^i · f(r_i).
        let mut vp = lattice_sumcheck::VirtualPolynomial::new(mle.num_vars);
        let fi = vp
            .add_factor(mle.clone())
            .map_err(AkitaPcsError::Virtual)?;
        let mut combined_claim = Goldilocks::ZERO;
        for (i, claim) in claims.iter().enumerate() {
            let eq = DenseMle::eq_extension(&claim.point);
            let ei = vp.add_factor(eq).map_err(AkitaPcsError::Virtual)?;
            vp.add_term(rhos[i], vec![fi, ei])
                .map_err(AkitaPcsError::Virtual)?;
            combined_claim = combined_claim.add(&rhos[i].mul(&claim.value));
        }
        let out = lattice_sumcheck::sumcheck::prove(&vp, combined_claim, transcript)
            .map_err(AkitaPcsError::Sumcheck)?;
        let opened = self.packed_witness(mle)?;
        let norm_proof = NormProof::prove(&opened, self.pk.params.norm_bound)
            .map_err(AkitaPcsError::Norm)?;
        Ok(EvaluationProof {
            sumcheck: out.proof,
            // Grouped proofs carry the first claim's point for shape
            // checks; the binding covers all claims through the batch.
            point: claims[0].point.clone(),
            value: combined_claim,
            opened_witness: opened,
            norm_proof,
        })
    }

    /// Verify a grouped opening: recomputes the combined RLC claim and the
    /// eq-weighted final binding over all points.
    pub fn verify_grouped(
        &self,
        commitment: &Commitment,
        claims: &[GroupedOpening],
        proof: &EvaluationProof,
        transcript: &mut Transcript,
    ) -> Result<(), AkitaPcsError> {
        if claims.is_empty() || proof.point.len() != commitment.num_vars {
            return Err(AkitaPcsError::Shape {
                expected: commitment.num_vars,
                got: proof.point.len(),
            });
        }
        let rhos = transcript
            .challenge_fields(b"akita-group-rho", claims.len())
            .map_err(AkitaPcsError::Transcript)?;
        let mut combined = Goldilocks::ZERO;
        for (rho, c) in rhos.iter().zip(claims.iter()) {
            combined = combined.add(&rho.mul(&c.value));
        }
        // Sumcheck over the combined polynomial.
        let verdict = proof
            .sumcheck
            .verify(commitment.num_vars, 2, combined, transcript, None)
            .map_err(AkitaPcsError::Sumcheck)?;
        // Witness binding + norms.
        self.pk
            .verify_opening(&commitment.commitment, &proof.opened_witness)
            .map_err(AkitaPcsError::Ajtai)?;
        proof
            .norm_proof
            .verify(&proof.opened_witness, self.pk.params.norm_bound)
            .map_err(AkitaPcsError::Norm)?;
        // Final: P(r_sc) = Σ_i ρ^i · eq(r_i, r_sc)·f(r_sc).
        let mut unpacked = lattice_ring::packing::unpack_field_elements(
            &self.pk.params.ring,
            &proof.opened_witness,
        )
        .map_err(AkitaPcsError::Packing)?;
        let expected_len = 1usize << commitment.num_vars;
        if unpacked.len() < expected_len {
            return Err(AkitaPcsError::Shape {
                expected: expected_len,
                got: unpacked.len(),
            });
        }
        unpacked.truncate(expected_len);
        let f_mle = DenseMle {
            num_vars: commitment.num_vars,
            evaluations: unpacked,
        };
        let f_at_sc = f_mle
            .evaluate(&verdict.point)
            .map_err(AkitaPcsError::Mle)?;
        let mut expected_final = Goldilocks::ZERO;
        for (rho, c) in rhos.iter().zip(claims.iter()) {
            let eq_at = DenseMle::eq_extension(&c.point)
                .evaluate(&verdict.point)
                .map_err(AkitaPcsError::Mle)?;
            expected_final = expected_final.add(&rho.mul(&eq_at.mul(&f_at_sc)));
        }
        if verdict.final_claim != expected_final {
            return Err(AkitaPcsError::VerificationFailed);
        }
        // Each claimed value must match the opened witness.
        for c in claims {
            let v = f_mle
                .evaluate(&c.point)
                .map_err(AkitaPcsError::Mle)?;
            if v != c.value {
                return Err(AkitaPcsError::VerificationFailed);
            }
        }
        Ok(())
    }
}

/// Ring config accessor for the schedule layer.
pub fn pcs_ring(pcs: &AkitaPcs) -> &RingConfig {
    &pcs.pk.params.ring
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(num_vars: usize) -> (AkitaPcs, DenseMle) {
        // Packing: 3 limbs per value; ring dimension must cover the
        // packed column: n >= 3 for small tests; m slots must hold
        // ceil(3 * 2^num_vars / n) elements.
        let log_n = 4u32; // n = 16
        let values = 1usize << num_vars;
        let packed_estimate = (values * 3).div_ceil(1 << log_n).max(1);
        let m = packed_estimate.max(4);
        let pcs = crate::akita_setup(log_n, m, 1 << 23, [71u8; 32]).ok().unwrap();
        let mle = DenseMle::random(num_vars, b"akita-mle");
        (pcs, mle)
    }

    #[test]
    fn commit_prove_verify_end_to_end() {
        for num_vars in [2usize, 4, 6] {
            let (pcs, mle) = setup(num_vars);
            let commitment = pcs.commit(&mle).ok().unwrap();
            let point: Vec<Goldilocks> = (1..=num_vars)
                .map(|i| Goldilocks::from_u64((i * 9973) as u64))
                .collect();
            let mut t = Transcript::new_default(b"lzx-akita-test");
            // Bind the statement.
            t.append_bytes(b"commitment", &commitment.commitment.to_bytes())
                .ok()
                .unwrap();
            t.append_field_slice(b"point", &point).ok().unwrap();
            let proof = pcs.prove_evaluation(&mle, &point, &mut t).ok().unwrap();
            let mut vt = Transcript::new_default(b"lzx-akita-test");
            vt.append_bytes(b"commitment", &commitment.commitment.to_bytes())
                .ok()
                .unwrap();
            vt.append_field_slice(b"point", &point).ok().unwrap();
            assert!(
                pcs.verify_evaluation(&commitment, &proof, &mut vt).is_ok(),
                "num_vars {num_vars}"
            );
        }
    }

    #[test]
    fn wrong_value_rejected() {
        let (pcs, mle) = setup(4);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let point: Vec<Goldilocks> = (1..=4)
            .map(|i| Goldilocks::from_u64((i * 1231) as u64))
            .collect();
        let mut t = Transcript::new_default(b"lzx-akita-test");
        t.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        t.append_field_slice(b"point", &point).ok().unwrap();
        let mut proof = pcs.prove_evaluation(&mle, &point, &mut t).ok().unwrap();
        proof.value = proof.value.add(&Goldilocks::ONE);
        let mut vt = Transcript::new_default(b"lzx-akita-test");
        vt.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        vt.append_field_slice(b"point", &point).ok().unwrap();
        assert!(pcs.verify_evaluation(&commitment, &proof, &mut vt).is_err());
    }

    #[test]
    fn tampered_witness_rejected() {
        let (pcs, mle) = setup(4);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let point: Vec<Goldilocks> = (1..=4)
            .map(|i| Goldilocks::from_u64((i * 4241) as u64))
            .collect();
        let mut t = Transcript::new_default(b"lzx-akita-test");
        t.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        t.append_field_slice(b"point", &point).ok().unwrap();
        let mut proof = pcs.prove_evaluation(&mle, &point, &mut t).ok().unwrap();
        // Tamper with the opened witness: Ajtai binding must reject.
        if !proof.opened_witness.is_empty() {
            let ring = pcs.pk.params.ring.clone();
            let mut coeffs = proof.opened_witness[0].coeffs().to_vec();
            coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
            proof.opened_witness[0] = RingElement::from_coeffs(&ring, coeffs);
        }
        let mut vt = Transcript::new_default(b"lzx-akita-test");
        vt.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        vt.append_field_slice(b"point", &point).ok().unwrap();
        assert!(pcs.verify_evaluation(&commitment, &proof, &mut vt).is_err());
    }

    #[test]
    fn tampered_sumcheck_rejected() {
        let (pcs, mle) = setup(4);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let point: Vec<Goldilocks> = (1..=4)
            .map(|i| Goldilocks::from_u64((i * 31337) as u64))
            .collect();
        let mut t = Transcript::new_default(b"lzx-akita-test");
        t.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        t.append_field_slice(b"point", &point).ok().unwrap();
        let mut proof = pcs.prove_evaluation(&mle, &point, &mut t).ok().unwrap();
        if let Some(r0) = proof.sumcheck.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&Goldilocks::ONE);
            }
        }
        let mut vt = Transcript::new_default(b"lzx-akita-test");
        vt.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        vt.append_field_slice(b"point", &point).ok().unwrap();
        assert!(pcs.verify_evaluation(&commitment, &proof, &mut vt).is_err());
    }

    #[test]
    fn grouped_openings_prove_and_bind() {
        let (pcs, mle) = setup(5);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let mk_point = |salt: u64| -> Vec<Goldilocks> {
            (0..5)
                .map(|i| Goldilocks::from_u64(salt.wrapping_mul(7919) + i as u64))
                .collect()
        };
        let claims: Vec<GroupedOpening> = (1..=3u64)
            .map(|salt| {
                let p = mk_point(salt);
                let v = mle.evaluate(&p).ok().unwrap();
                GroupedOpening { point: p, value: v }
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-akita-group");
        t.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        for c in &claims {
            t.append_field_slice(b"claim-point", &c.point).ok().unwrap();
            t.append_field(b"claim-value", &c.value).ok().unwrap();
        }
        let proof = pcs.prove_grouped(&mle, &claims, &mut t).ok().unwrap();
        // The grouped claim is the RLC of values with the same rhos.
        let mut vt = Transcript::new_default(b"lzx-akita-group");
        vt.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        for c in &claims {
            vt.append_field_slice(b"claim-point", &c.point).ok().unwrap();
            vt.append_field(b"claim-value", &c.value).ok().unwrap();
        }
        // The combined claim is checked inside verify_grouped (rhos are
        // sampled there exactly once); the proof's value equals the RLC.
        // Verify through the grouped path.
        assert!(pcs.verify_grouped(&commitment, &claims, &proof, &mut vt).is_ok());
        // A wrong claimed value must fail.
        let mut bad_claims = claims.clone();
        bad_claims[0].value = bad_claims[0].value.add(&Goldilocks::ONE);
        let mut vt2 = Transcript::new_default(b"lzx-akita-group");
        vt2.append_bytes(b"commitment", &commitment.commitment.to_bytes())
            .ok()
            .unwrap();
        for c in &claims {
            vt2.append_field_slice(b"claim-point", &c.point).ok().unwrap();
            vt2.append_field(b"claim-value", &c.value).ok().unwrap();
        }
        assert!(pcs.verify_grouped(&commitment, &bad_claims, &proof, &mut vt2).is_err());
    }
}
