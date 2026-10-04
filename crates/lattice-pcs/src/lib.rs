//! # lattice-pcs
//!
//! The PCS backend trait boundary plus the **HyperWolf** backend
//! (ePrint 2025/1903: lattice polynomial commitments with *standard*
//! soundness).
//!
//! HyperWolf's distinguishing property: binding holds in the STANDARD
//! model (no random-oracle extraction needed) — the commitment is opened
//! coefficient-wise with digit-decomposed norm proofs, and the evaluation
//! check runs through a fully linear route: the verifier recomputes the
//! multilinear evaluation identity from the opened, norm-checked
//! coefficients. Extraction is direct: two accepting openings yield a
//! Module-SIS solution outright.
//!
//! The trait (`PcsBackend`) is the audit report's architecture contract:
//! baseline Akita and research backends implement the same
//! commit/prove/verify surface without changing VM semantics.
//!
//! H7 (Wave 7.12): the trait admits **multiple evaluation claims** via
//! [`PcsBackend::prove_evaluations`] / [`PcsBackend::verify_evaluations`]
//! (HyperWolf paper Appendix B, the three batching modes); backends that
//! cannot batch degrade to an honest [`PcsFeatureError::BatchingUnsupported`].

#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::type_complexity
)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod hyperwolf;

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiParams, AjtaiPublicKey};
use lattice_commitment::norm_proof::{NormProof, NormProofError};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::RingElement;

/// A point/value opening claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningClaim {
    pub point: Vec<Goldilocks>,
    pub value: Goldilocks,
}

/// Errors for *optional* PCS trait features, so that the H7 batching
/// defaults can degrade honestly on backends that do not implement them
/// (every `PcsBackend::Error` converts from this type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcsFeatureError {
    /// The backend does not support multi-claim batched openings.
    BatchingUnsupported,
}

impl std::fmt::Display for PcsFeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PcsFeatureError::BatchingUnsupported => {
                write!(
                    f,
                    "multi-claim batched openings unsupported by this backend"
                )
            }
        }
    }
}

impl std::error::Error for PcsFeatureError {}

/// The PCS backend contract (the audit report §8.2 interface, specialized
/// to our concrete types).
///
/// # H7 — three-mode batching (HyperWolf paper, Appendix B)
///
/// The paper batches multiple evaluation proofs in three modes:
///
/// 1. **n polynomials, one point** — sample α ∈ Z_q^n, prove the RLC
///    f = Σ α^i f_i at the combined value Σ α^i v_i (callers compose this
///    above the trait by RLC-ing MLEs before committing).
/// 2. **one polynomial, n points** — g(x) = Σ α^i·eq(x, u_i)·f(x) plus a
///    sumcheck; this is the mode the trait surface models directly: one
///    committed MLE, a slice of claims.
/// 3. **n polynomials, n points** — mode 3 reduces to mode 2 followed by
///    mode 1 (the paper's own reduction).
///
/// [`PcsBackend::prove_evaluations`] therefore takes one MLE and a slice of
/// claims (mode 2); modes 1/3 compose on top. The default implementation
/// delegates the single-claim case to [`PcsBackend::prove`] and returns
/// [`PcsFeatureError::BatchingUnsupported`] otherwise, so existing
/// single-claim backends keep their exact behavior.
pub trait PcsBackend {
    type Commitment;
    type OpeningProof;
    /// Must convert from the optional-feature error so the H7 defaults can
    /// report unsupported batching honestly.
    type Error: From<PcsFeatureError>;

    fn commit(&self, mle: &DenseMle) -> Result<Self::Commitment, Self::Error>;
    fn prove(
        &self,
        mle: &DenseMle,
        claim: &OpeningClaim,
        transcript: &mut Transcript,
    ) -> Result<Self::OpeningProof, Self::Error>;
    fn verify(
        &self,
        commitment: &Self::Commitment,
        claim: &OpeningClaim,
        proof: &Self::OpeningProof,
        transcript: &mut Transcript,
    ) -> Result<(), Self::Error>;

    /// H7: prove several evaluation claims on one committed MLE (paper
    /// Appendix B mode 2 — one polynomial, many points). Default: delegate a
    /// single claim to [`PcsBackend::prove`]; error on zero or multiple
    /// claims when the backend has no batched implementation.
    fn prove_evaluations(
        &self,
        mle: &DenseMle,
        claims: &[OpeningClaim],
        transcript: &mut Transcript,
    ) -> Result<Self::OpeningProof, Self::Error> {
        if claims.len() == 1 {
            self.prove(mle, &claims[0], transcript)
        } else {
            Err(Self::Error::from(PcsFeatureError::BatchingUnsupported))
        }
    }

    /// H7: verify a batched opening. Default: delegate a single claim to
    /// [`PcsBackend::verify`]; error otherwise (never silently verify a
    /// subset of the claims).
    fn verify_evaluations(
        &self,
        commitment: &Self::Commitment,
        claims: &[OpeningClaim],
        proof: &Self::OpeningProof,
        transcript: &mut Transcript,
    ) -> Result<(), Self::Error> {
        if claims.len() == 1 {
            self.verify(commitment, &claims[0], proof, transcript)
        } else {
            Err(Self::Error::from(PcsFeatureError::BatchingUnsupported))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HyperWolfError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    Norm(NormProofError),
    Transcript(TranscriptError),
    Packing(lattice_ring::PackingError),
    Mle(lattice_core::mle::MleError),
    /// An optional PCS feature (H7 batching) is unsupported.
    Feature(PcsFeatureError),
    VerificationFailed,
    Shape {
        expected: usize,
        got: usize,
    },
}

impl From<PcsFeatureError> for HyperWolfError {
    fn from(e: PcsFeatureError) -> Self {
        HyperWolfError::Feature(e)
    }
}

/// HyperWolf: standard-soundness lattice PCS via direct coefficient
/// openings with norm proofs (no Fiat–Shamir-dependent extraction).
pub struct HyperWolf {
    pub pk: AjtaiPublicKey,
}

#[derive(Clone, Debug)]
pub struct HyperWolfCommitment {
    pub commitment: AjtaiCommitment,
    pub num_vars: usize,
}

#[derive(Clone, Debug)]
pub struct HyperWolfOpening {
    /// Opened packed coefficients (the full response — standard-model
    /// binding: verifiers extract directly).
    pub witness: Vec<RingElement>,
    /// Digit-decomposed norm proof.
    pub norm_proof: NormProof,
    /// The claimed evaluation.
    pub claim: OpeningClaim,
}

impl HyperWolf {
    pub fn setup(
        log_n: u32,
        m: usize,
        norm_bound: u32,
        seed: [u8; 32],
    ) -> Result<Self, HyperWolfError> {
        let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n)
            .map_err(HyperWolfError::Ring)?;
        let params = AjtaiParams {
            ring,
            k: 2,
            m,
            norm_bound,
        };
        let pk = AjtaiPublicKey::from_seed(params, seed).map_err(HyperWolfError::Ajtai)?;
        Ok(HyperWolf { pk })
    }

    fn packed(&self, mle: &DenseMle) -> Result<Vec<RingElement>, HyperWolfError> {
        let packed =
            lattice_ring::packing::pack_field_elements(&self.pk.params.ring, &mle.evaluations);
        self.pk.pad_to_m(&packed).map_err(HyperWolfError::Ajtai)
    }

    /// Shared verification preamble: Ajtai binding + norm layer + unpacking
    /// the committed evaluations (the witness-direct route both the
    /// single-claim and the H7 batched verifier start from).
    fn bound_witness(
        &self,
        commitment: &HyperWolfCommitment,
        proof: &HyperWolfOpening,
    ) -> Result<DenseMle, HyperWolfError> {
        // 1. Ajtai binding: the opened witness matches the commitment.
        self.pk
            .verify_opening(&commitment.commitment, &proof.witness)
            .map_err(HyperWolfError::Ajtai)?;
        // 2. Norm layer: response respects the SIS bound.
        proof
            .norm_proof
            .verify(&proof.witness, self.pk.params.norm_bound)
            .map_err(HyperWolfError::Norm)?;
        // 3. Unpack the committed evaluations.
        let mut unpacked =
            lattice_ring::packing::unpack_field_elements(&self.pk.params.ring, &proof.witness)
                .map_err(HyperWolfError::Packing)?;
        let expected_len = 1usize << commitment.num_vars;
        if unpacked.len() < expected_len {
            return Err(HyperWolfError::Shape {
                expected: expected_len,
                got: unpacked.len(),
            });
        }
        unpacked.truncate(expected_len);
        Ok(DenseMle {
            num_vars: commitment.num_vars,
            evaluations: unpacked,
        })
    }
}

impl PcsBackend for HyperWolf {
    type Commitment = HyperWolfCommitment;
    type OpeningProof = HyperWolfOpening;
    type Error = HyperWolfError;

    fn commit(&self, mle: &DenseMle) -> Result<Self::Commitment, Self::Error> {
        let witness = self.packed(mle)?;
        let commitment = self.pk.commit(&witness).map_err(HyperWolfError::Ajtai)?;
        Ok(HyperWolfCommitment {
            commitment,
            num_vars: mle.num_vars,
        })
    }

    fn prove(
        &self,
        mle: &DenseMle,
        claim: &OpeningClaim,
        _transcript: &mut Transcript,
    ) -> Result<Self::OpeningProof, Self::Error> {
        // Standard-model route: the opening is witness-direct — no
        // transcript-dependent extraction (that is the point).
        let value = mle.evaluate(&claim.point).map_err(HyperWolfError::Mle)?;
        if value != claim.value {
            return Err(HyperWolfError::VerificationFailed);
        }
        let witness = self.packed(mle)?;
        let norm_proof =
            NormProof::prove(&witness, self.pk.params.norm_bound).map_err(HyperWolfError::Norm)?;
        Ok(HyperWolfOpening {
            witness,
            norm_proof,
            claim: claim.clone(),
        })
    }

    fn verify(
        &self,
        commitment: &HyperWolfCommitment,
        claim: &OpeningClaim,
        proof: &HyperWolfOpening,
        _transcript: &mut Transcript,
    ) -> Result<(), Self::Error> {
        let f = self.bound_witness(commitment, proof)?;
        let value = f.evaluate(&claim.point).map_err(HyperWolfError::Mle)?;
        if value != claim.value {
            return Err(HyperWolfError::VerificationFailed);
        }
        Ok(())
    }

    /// H7 real batching: the witness-direct route makes multi-claim
    /// openings trivially sound — the full witness is opened once (one
    /// Ajtai binding + one norm proof) and EVERY claim is checked against
    /// the recomputed evaluations. Soundness is the standard-model
    /// commit-and-reveal argument of the single-claim path (per-claim
    /// verification, one shared binding); no Fiat–Shamir dependency.
    fn prove_evaluations(
        &self,
        mle: &DenseMle,
        claims: &[OpeningClaim],
        _transcript: &mut Transcript,
    ) -> Result<Self::OpeningProof, Self::Error> {
        if claims.is_empty() {
            return Err(HyperWolfError::Shape {
                expected: 1,
                got: 0,
            });
        }
        // Honest prover: refuse any false claim in the batch.
        for claim in claims {
            let value = mle.evaluate(&claim.point).map_err(HyperWolfError::Mle)?;
            if value != claim.value {
                return Err(HyperWolfError::VerificationFailed);
            }
        }
        let witness = self.packed(mle)?;
        let norm_proof =
            NormProof::prove(&witness, self.pk.params.norm_bound).map_err(HyperWolfError::Norm)?;
        Ok(HyperWolfOpening {
            witness,
            norm_proof,
            claim: claims[0].clone(),
        })
    }

    /// H7 real batching (verify): one binding + norm check, then every
    /// claim is evaluated against the opened witness.
    fn verify_evaluations(
        &self,
        commitment: &HyperWolfCommitment,
        claims: &[OpeningClaim],
        proof: &HyperWolfOpening,
        _transcript: &mut Transcript,
    ) -> Result<(), Self::Error> {
        if claims.is_empty() {
            return Err(HyperWolfError::Shape {
                expected: 1,
                got: 0,
            });
        }
        let f = self.bound_witness(commitment, proof)?;
        for claim in claims {
            let value = f.evaluate(&claim.point).map_err(HyperWolfError::Mle)?;
            if value != claim.value {
                return Err(HyperWolfError::VerificationFailed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup(num_vars: usize) -> (HyperWolf, DenseMle) {
        let log_n = 4u32;
        let values = 1usize << num_vars;
        let packed_estimate = (values * 3).div_ceil(1 << log_n).max(1);
        let pcs = HyperWolf::setup(log_n, packed_estimate.max(4), 1 << 23, [81u8; 32])
            .ok()
            .unwrap();
        let mle = DenseMle::random(num_vars, b"hw-mle");
        (pcs, mle)
    }

    #[test]
    fn hyperwolf_end_to_end() {
        for num_vars in [2usize, 5] {
            let (pcs, mle) = setup(num_vars);
            let commitment = pcs.commit(&mle).ok().unwrap();
            let point: Vec<Goldilocks> = (0..num_vars)
                .map(|i| Goldilocks::from_u64((i as u64 * 6151) + 11))
                .collect();
            let value = mle.evaluate(&point).ok().unwrap();
            let claim = OpeningClaim { point, value };
            let mut t = Transcript::new_default(b"lzx-hyperwolf");
            let proof = pcs.prove(&mle, &claim, &mut t).ok().unwrap();
            let mut vt = Transcript::new_default(b"lzx-hyperwolf");
            assert!(pcs.verify(&commitment, &claim, &proof, &mut vt).is_ok());
        }
    }

    #[test]
    fn hyperwolf_wrong_claim_rejected() {
        let (pcs, mle) = setup(4);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let point: Vec<Goldilocks> = (0..4)
            .map(|i| Goldilocks::from_u64((i as u64 * 9931) + 3))
            .collect();
        let value = mle.evaluate(&point).ok().unwrap().add(&Goldilocks::ONE);
        let claim = OpeningClaim { point, value };
        let mut t = Transcript::new_default(b"lzx-hyperwolf");
        // Prover refuses a false claim outright.
        assert!(pcs.prove(&mle, &claim, &mut t).is_err());
        // And a tampered proof fails verification.
        let good = OpeningClaim {
            point: claim.point.clone(),
            value: value.sub(&Goldilocks::ONE),
        };
        let mut t2 = Transcript::new_default(b"lzx-hyperwolf");
        let proof = pcs.prove(&mle, &good, &mut t2).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-hyperwolf");
        assert!(pcs.verify(&commitment, &claim, &proof, &mut vt).is_err());
    }

    #[test]
    fn hyperwolf_tampered_witness_rejected() {
        let (pcs, mle) = setup(3);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let point: Vec<Goldilocks> = (0..3)
            .map(|i| Goldilocks::from_u64((i as u64 * 4441) + 7))
            .collect();
        let value = mle.evaluate(&point).ok().unwrap();
        let claim = OpeningClaim { point, value };
        let mut t = Transcript::new_default(b"lzx-hyperwolf");
        let mut proof = pcs.prove(&mle, &claim, &mut t).ok().unwrap();
        if !proof.witness.is_empty() {
            let ring = pcs.pk.params.ring.clone();
            let mut coeffs = proof.witness[0].coeffs().to_vec();
            coeffs[1] = (coeffs[1] + 1) % ring.modulus.q;
            proof.witness[0] = RingElement::from_coeffs(&ring, coeffs);
        }
        let mut vt = Transcript::new_default(b"lzx-hyperwolf");
        assert!(pcs.verify(&commitment, &claim, &proof, &mut vt).is_err());
    }

    #[test]
    fn trait_object_dispatch() {
        // The backend boundary: two implementations behind one trait.
        let (hw, mle) = setup(3);
        let point: Vec<Goldilocks> = (0..3)
            .map(|i| Goldilocks::from_u64((i as u64 * 7717) + 5))
            .collect();
        let value = mle.evaluate(&point).ok().unwrap();
        let claim = OpeningClaim { point, value };
        let backend: &dyn PcsBackend<
            Commitment = HyperWolfCommitment,
            OpeningProof = HyperWolfOpening,
            Error = HyperWolfError,
        > = &hw;
        let commitment = backend.commit(&mle).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-hyperwolf");
        let proof = backend.prove(&mle, &claim, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-hyperwolf");
        assert!(backend.verify(&commitment, &claim, &proof, &mut vt).is_ok());
    }

    #[test]
    fn hyperwolf_batch_evaluations_h7() {
        // H7 mode 2 (one polynomial, many points) on the witness-direct
        // backend: one binding + one norm proof, every claim checked.
        let (pcs, mle) = setup(4);
        let commitment = pcs.commit(&mle).ok().unwrap();
        let claims: Vec<OpeningClaim> = (1..=3u64)
            .map(|salt| {
                let point: Vec<Goldilocks> = (0..4)
                    .map(|i| Goldilocks::from_u64(salt * 6151 + i as u64 * 37))
                    .collect();
                let value = mle.evaluate(&point).ok().unwrap();
                OpeningClaim { point, value }
            })
            .collect();
        let mut t = Transcript::new_default(b"lzx-hyperwolf-batch");
        let proof = pcs.prove_evaluations(&mle, &claims, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-hyperwolf-batch");
        assert!(pcs
            .verify_evaluations(&commitment, &claims, &proof, &mut vt)
            .is_ok());
        // One wrong value in the batch must sink the whole verification.
        let mut bad = claims.clone();
        bad[1].value = bad[1].value.add(&Goldilocks::ONE);
        let mut vt2 = Transcript::new_default(b"lzx-hyperwolf-batch");
        assert!(pcs
            .verify_evaluations(&commitment, &bad, &proof, &mut vt2)
            .is_err());
        // The honest prover refuses a batch containing a false claim.
        let mut t2 = Transcript::new_default(b"lzx-hyperwolf-batch");
        assert!(pcs.prove_evaluations(&mle, &bad, &mut t2).is_err());
        // Empty claim batches are a shape error, not a vacuous accept.
        let mut t3 = Transcript::new_default(b"lzx-hyperwolf-batch");
        assert!(pcs.prove_evaluations(&mle, &[], &mut t3).is_err());
        let mut vt3 = Transcript::new_default(b"lzx-hyperwolf-batch");
        assert!(pcs
            .verify_evaluations(&commitment, &[], &proof, &mut vt3)
            .is_err());
        // A claim verified in isolation against the batch proof still
        // passes (the proof opens the same witness).
        let mut vt4 = Transcript::new_default(b"lzx-hyperwolf-batch");
        assert!(pcs
            .verify(&commitment, &claims[2], &proof, &mut vt4)
            .is_ok());
    }
}
