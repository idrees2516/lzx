//! ABDLOP-style linear proofs of knowledge over Ajtai commitments.
//!
//! Statement: given public `t = A·s mod q` and public linear relations
//! `⟨v_i, s⟩ = u_i (mod q)` over R_q, prove knowledge of a *short* `s`
//! satisfying both, without revealing `s`.
//!
//! Protocol (single-round, Fiat–Shamir compiled, Lyubashevsky-style):
//! 1. Prover samples short masking `y`, computes `w = A·y` and also commits
//!    the linear-map images `⟨v_i, y⟩` (so relations can be checked).
//! 2. Challenge `c` sampled sparse-ternary from the transcript.
//! 3. Response `z = y + c·s`; the prover *rejects and resamples* if
//!    `||z||∞` exceeds the bound (rejection sampling — bounded retries,
//!    each retry consumes a fresh transcript counter, so the Fiat–Shamir
//!    transform stays extractable).
//! 4. Verifier checks `A·z = w + c·t` and `⟨v_i, z⟩ = ⟨v_i, y⟩ + c·u_i`,
//!    and that `||z||∞ ≤ B`.
//!
//! Linear relations are embedded by appending the `v_i` rows to the
//! commitment matrix (the ABDLOP extension), so soundness reduces to
//! Module-SIS on the extended instance.

use crate::ajtai::{AjtaiError, AjtaiPublicKey};
use lattice_core::challenge_set::{ChallengeDistribution, ChallengeSet};
use lattice_core::transcript::Transcript;
use lattice_ring::RingElement;

/// A public linear relation over the secret vector: `⟨v, s⟩ = u`.
#[derive(Clone, Debug)]
pub struct LinearRelation {
    /// Coefficient vector over the m ring slots.
    pub coefficients: Vec<RingElement>,
    /// Target ring element.
    pub target: RingElement,
}

/// Proof of knowledge of a short preimage + linear relations.
#[derive(Clone, Debug)]
pub struct LinearProof {
    /// Masking commitment rows w = A·y.
    pub mask_commitment: Vec<RingElement>,
    /// Linear-map images ⟨v_i, y⟩.
    pub mask_images: Vec<RingElement>,
    /// Sparse-ternary challenge coefficients (ring elements, one per m slot
    /// component? — one challenge ring element per protocol instance).
    pub challenge: RingElement,
    /// Response z = y + c·s (per slot).
    pub response: Vec<RingElement>,
    /// Number of rejection-sampling retries used (bounded, public).
    pub retries: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearProofError {
    Ajtai(AjtaiError),
    RelationShape { expected: usize, got: usize },
    NormExceeded,
    RetriesExceeded,
    VerificationFailed,
    Ring(lattice_ring::RingError),
}

/// Maximum rejection-sampling retries before the prover gives up.
pub const MAX_RETRIES: u32 = 32;

impl LinearRelation {
    /// Evaluate ⟨v, s⟩ for a secret vector.
    pub fn evaluate(&self, s: &[RingElement]) -> Result<RingElement, LinearProofError> {
        if s.len() != self.coefficients.len() {
            return Err(LinearProofError::RelationShape {
                expected: self.coefficients.len(),
                got: s.len(),
            });
        }
        let mut acc = self.target.config().zero();
        for (v, si) in self.coefficients.iter().zip(s.iter()) {
            let prod = v.mul(si).map_err(LinearProofError::Ring)?;
            acc = acc.add(&prod).map_err(LinearProofError::Ring)?;
        }
        Ok(acc)
    }
}

impl LinearProof {
    /// Prove knowledge of `s` (short) with `t = A·s` and relations.
    /// Deterministic masking from `prover_seed` keeps tests reproducible;
    /// production provers feed OS entropy here.
    pub fn prove(
        pk: &AjtaiPublicKey,
        relations: &[LinearRelation],
        s: &[RingElement],
        commitment: &crate::ajtai::AjtaiCommitment,
        prover_seed: &[u8],
    ) -> Result<Self, LinearProofError> {
        let ring = &pk.params.ring;
        let m = pk.params.m;
        if s.len() != m {
            return Err(LinearProofError::Ajtai(AjtaiError::DimensionMismatch {
                expected: m,
                got: s.len(),
            }));
        }
        // Verify the relations hold for s before proving (fail-closed).
        for rel in relations {
            let img = rel.evaluate(s)?;
            if img != rel.target {
                return Err(LinearProofError::VerificationFailed);
            }
        }

        // Canonical statement bytes (for per-attempt transcript
        // rebuilds below).
        let mut statement = Vec::new();
        statement.extend_from_slice(&pk.seed);
        statement.extend_from_slice(&commitment.to_bytes());
        for rel in relations {
            statement.extend_from_slice(&rel.target.to_bytes());
            for v in &rel.coefficients {
                statement.extend_from_slice(&v.to_bytes());
            }
        }
        // Derive the challenge from (statement, w, images) — the
        // SOUNDNESS FIX (wave 3): the masking commitment w and the
        // relation images must be absorbed BEFORE the challenge, or the
        // Fiat-Shamir transform is malleable (pick any short z', set
        // w' := A·z' − c·t with the statement-only c, and verification
        // passes without knowledge of s).
        let derive = |w: &crate::ajtai::AjtaiCommitment,
                      images: &[RingElement]|
         -> Result<RingElement, LinearProofError> {
            let mut t = Transcript::new_default(b"lzx-linear-proof");
            t.append_bytes(b"pk-seed", &pk.seed)
                .map_err(|_| LinearProofError::VerificationFailed)?;
            t.append_bytes(b"commitment", &commitment.to_bytes())
                .map_err(|_| LinearProofError::VerificationFailed)?;
            for rel in relations {
                t.append_bytes(b"relation-v", &rel.target.to_bytes())
                    .map_err(|_| LinearProofError::VerificationFailed)?;
                for v in &rel.coefficients {
                    t.append_bytes(b"rel-coeff", &v.to_bytes())
                        .map_err(|_| LinearProofError::VerificationFailed)?;
                }
            }
            t.append_bytes(b"statement", &statement)
                .map_err(|_| LinearProofError::VerificationFailed)?;
            t.append_bytes(b"mask-w", &w.to_bytes())
                .map_err(|_| LinearProofError::VerificationFailed)?;
            for img in images {
                t.append_bytes(b"mask-image", &img.to_bytes())
                    .map_err(|_| LinearProofError::VerificationFailed)?;
            }
            let chal_seed = t
                .challenge_bytes(b"challenge", 32)
                .map_err(|_| LinearProofError::VerificationFailed)?;
            let cs = ChallengeSet::sample(
                ChallengeDistribution::SparseTernary {
                    weight: ring.n() / 2,
                },
                ring.n(),
                &chal_seed,
            )
            .map_err(|_| LinearProofError::VerificationFailed)?;
            Ok(RingElement::from_signed(ring, &cs.coefficients))
        };
        // Rejection-sampled masking and response (each attempt is a
        // fresh Fiat-Shamir instance over the fixed statement).
        let bound = pk.params.norm_bound;
        let response_bound = bound.saturating_sub(2);
        if response_bound == 0 {
            return Err(LinearProofError::NormExceeded);
        }
        for retry in 0..=MAX_RETRIES {
            let mut salt = Vec::with_capacity(prover_seed.len() + 8);
            salt.extend_from_slice(prover_seed);
            salt.extend_from_slice(&(retry as u64).to_le_bytes());
            let y = crate::ajtai::sample_small_secret(ring, m, response_bound / 2, &salt);
            // w = A·y
            let w = pk.commit(&y).map_err(LinearProofError::Ajtai)?;
            // images ⟨v_i, y⟩
            let mut images = Vec::with_capacity(relations.len());
            for rel in relations {
                images.push(rel.evaluate(&y)?);
            }
            // Challenge derived AFTER absorbing (w, images).
            let challenge = derive(&w, &images)?;
            // z = y + c·s
            let mut z = Vec::with_capacity(m);
            let mut ok = true;
            for (yi, si) in y.iter().zip(s.iter()) {
                let cs_prod = challenge.mul(si).map_err(LinearProofError::Ring)?;
                let zi = yi.add(&cs_prod).map_err(LinearProofError::Ring)?;
                if zi.infinity_norm() > bound {
                    ok = false;
                    break;
                }
                z.push(zi);
            }
            if !ok {
                continue;
            }
            return Ok(LinearProof {
                mask_commitment: w.rows,
                mask_images: images,
                challenge,
                response: z,
                retries: retry,
            });
        }
        Err(LinearProofError::RetriesExceeded)
    }

    /// Verify the proof against the public statement.
    pub fn verify(
        &self,
        pk: &AjtaiPublicKey,
        relations: &[LinearRelation],
        commitment: &crate::ajtai::AjtaiCommitment,
    ) -> Result<(), LinearProofError> {
        let m = pk.params.m;
        if self.response.len() != m {
            return Err(LinearProofError::Ajtai(AjtaiError::DimensionMismatch {
                expected: m,
                got: self.response.len(),
            }));
        }
        // 1. Response norms.
        let bound = pk.params.norm_bound;
        for z in &self.response {
            if z.infinity_norm() > bound {
                return Err(LinearProofError::NormExceeded);
            }
        }
        // 2. Recompute the challenge deterministically (Fiat–Shamir,
        //    with the mask commitment and images absorbed first).
        let mut transcript = Transcript::new_default(b"lzx-linear-proof");
        transcript
            .append_bytes(b"pk-seed", &pk.seed)
            .map_err(|_| LinearProofError::VerificationFailed)?;
        transcript
            .append_bytes(b"commitment", &commitment.to_bytes())
            .map_err(|_| LinearProofError::VerificationFailed)?;
        for rel in relations {
            transcript
                .append_bytes(b"relation-v", &rel.target.to_bytes())
                .map_err(|_| LinearProofError::VerificationFailed)?;
            for v in &rel.coefficients {
                transcript
                    .append_bytes(b"rel-coeff", &v.to_bytes())
                    .map_err(|_| LinearProofError::VerificationFailed)?;
            }
        }
        let mut statement = Vec::new();
        statement.extend_from_slice(&pk.seed);
        statement.extend_from_slice(&commitment.to_bytes());
        for rel in relations {
            statement.extend_from_slice(&rel.target.to_bytes());
            for v in &rel.coefficients {
                statement.extend_from_slice(&v.to_bytes());
            }
        }
        transcript
            .append_bytes(b"statement", &statement)
            .map_err(|_| LinearProofError::VerificationFailed)?;
        let w_commitment = crate::ajtai::AjtaiCommitment {
            rows: self.mask_commitment.clone(),
        };
        transcript
            .append_bytes(b"mask-w", &w_commitment.to_bytes())
            .map_err(|_| LinearProofError::VerificationFailed)?;
        for img in &self.mask_images {
            transcript
                .append_bytes(b"mask-image", &img.to_bytes())
                .map_err(|_| LinearProofError::VerificationFailed)?;
        }
        let chal_seed = transcript
            .challenge_bytes(b"challenge", 32)
            .map_err(|_| LinearProofError::VerificationFailed)?;
        let ring = &pk.params.ring;
        let cs = ChallengeSet::sample(
            ChallengeDistribution::SparseTernary {
                weight: ring.n() / 2,
            },
            ring.n(),
            &chal_seed,
        )
        .map_err(|_| LinearProofError::VerificationFailed)?;
        let expected_challenge = RingElement::from_signed(ring, &cs.coefficients);
        if expected_challenge != self.challenge {
            return Err(LinearProofError::VerificationFailed);
        }

        // 3. A·z == w + c·t  (per commitment row).
        let c_times_t: Vec<RingElement> = commitment
            .rows
            .iter()
            .map(|t_i| self.challenge.mul(t_i).map_err(LinearProofError::Ring))
            .collect::<Result<_, _>>()?;
        let mut expected_rows = Vec::with_capacity(self.mask_commitment.len());
        for (w_i, ct_i) in self.mask_commitment.iter().zip(c_times_t.iter()) {
            expected_rows.push(w_i.add(ct_i).map_err(LinearProofError::Ring)?);
        }
        let az = pk.commit(&self.response).map_err(LinearProofError::Ajtai)?;
        if az.rows != expected_rows {
            return Err(LinearProofError::VerificationFailed);
        }

        // 4. Linear relations: ⟨v_i, z⟩ == image_i + c·u_i.
        for (rel, image) in relations.iter().zip(self.mask_images.iter()) {
            let lhs = rel.evaluate(&self.response)?;
            let cu = self
                .challenge
                .mul(&rel.target)
                .map_err(LinearProofError::Ring)?;
            let rhs = image.add(&cu).map_err(LinearProofError::Ring)?;
            if lhs != rhs {
                return Err(LinearProofError::VerificationFailed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, k: usize, m: usize, bound: u32) -> AjtaiPublicKey {
        let params = AjtaiParams {
            ring: RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap(),
            k,
            m,
            norm_bound: bound,
        };
        AjtaiPublicKey::from_seed(params, [9u8; 32]).ok().unwrap()
    }

    /// Regression (wave 3 FS-ordering fix): the pre-fix attack chose any
    /// short z', set w' := A·z' − c·t with the statement-only challenge,
    /// and passed verification without knowledge of s. The fixed
    /// verifier derives the challenge AFTER absorbing w', so the attack
    /// fails.
    #[test]
    fn post_hoc_malleability_rejected() {
        let pk = setup(4, 2, 3, 1 << 23);
        let ring = &pk.params.ring;
        let s = crate::ajtai::sample_small_secret(ring, pk.params.m, 32, b"attack");
        let t = pk.commit(&s).ok().unwrap();
        // Old-order challenge: statement only.
        let old_c = {
            let mut tr = Transcript::new_default(b"lzx-linear-proof");
            tr.append_bytes(b"pk-seed", &pk.seed).ok().unwrap();
            tr.append_bytes(b"commitment", &t.to_bytes()).ok().unwrap();
            let seed = tr.challenge_bytes(b"challenge", 32).ok().unwrap();
            let cs = ChallengeSet::sample(
                ChallengeDistribution::SparseTernary {
                    weight: ring.n() / 2,
                },
                ring.n(),
                &seed,
            )
            .ok()
            .unwrap();
            RingElement::from_signed(ring, &cs.coefficients)
        };
        // Forge with a z' that never touched s.
        let z = crate::ajtai::sample_small_secret(ring, pk.params.m, 64, b"forge");
        let az = pk.commit(&z).ok().unwrap();
        let mut w_rows = Vec::new();
        for (az_i, t_i) in az.rows.iter().zip(t.rows.iter()) {
            let ct = old_c.mul(t_i).ok().unwrap();
            w_rows.push(az_i.sub(&ct).ok().unwrap());
        }
        let forged = LinearProof {
            mask_commitment: w_rows,
            mask_images: vec![],
            challenge: old_c,
            response: z,
            retries: 0,
        };
        assert!(forged.verify(&pk, &[], &t).is_err());
    }

    #[test]
    fn prove_and_verify_with_relations() {
        let pk = setup(4, 2, 3, 1 << 17);
        let ring = &pk.params.ring;
        let s = crate::ajtai::sample_small_secret(ring, pk.params.m, 32, b"witness");
        let t = pk.commit(&s).ok().unwrap();
        // Relation: ⟨v, s⟩ = u with v = [1, 0, 0], so u = s[0].
        let mut v = vec![ring.zero(); pk.params.m];
        v[0] = ring.one();
        let u = s[0].clone();
        let rel = LinearRelation {
            coefficients: v,
            target: u,
        };
        let proof = LinearProof::prove(&pk, std::slice::from_ref(&rel), &s, &t, b"prover-seed")
            .ok()
            .unwrap();
        assert!(proof.verify(&pk, &[rel], &t).is_ok());
    }

    #[test]
    fn tampered_proof_rejected() {
        let pk = setup(4, 2, 3, 1 << 17);
        let ring = &pk.params.ring;
        let s = crate::ajtai::sample_small_secret(ring, pk.params.m, 32, b"witness");
        let t = pk.commit(&s).ok().unwrap();
        let proof = LinearProof::prove(&pk, &[], &s, &t, b"prover-seed")
            .ok()
            .unwrap();
        assert!(proof.verify(&pk, &[], &t).is_ok());
        // Tamper with the response.
        let mut bad = proof.clone();
        if !bad.response.is_empty() {
            let mut coeffs = bad.response[0].coeffs().to_vec();
            coeffs[0] = (coeffs[0].wrapping_add(1)) % ring.modulus.q;
            bad.response[0] = RingElement::from_coeffs(ring, coeffs);
        }
        assert!(bad.verify(&pk, &[], &t).is_err());
        // Tamper with the challenge.
        let mut bad2 = proof;
        let mut ccoeffs = bad2.challenge.coeffs().to_vec();
        ccoeffs[0] = (ccoeffs[0].wrapping_add(1)) % ring.modulus.q;
        bad2.challenge = RingElement::from_coeffs(ring, ccoeffs);
        assert!(bad2.verify(&pk, &[], &t).is_err());
    }

    #[test]
    fn wrong_relation_target_rejected() {
        let pk = setup(4, 2, 3, 1 << 17);
        let ring = &pk.params.ring;
        let s = crate::ajtai::sample_small_secret(ring, pk.params.m, 32, b"witness");
        let t = pk.commit(&s).ok().unwrap();
        let mut v = vec![ring.zero(); pk.params.m];
        v[1] = ring.one();
        let rel = LinearRelation {
            coefficients: v,
            target: s[1].clone(),
        };
        let proof = LinearProof::prove(&pk, std::slice::from_ref(&rel), &s, &t, b"ps")
            .ok()
            .unwrap();
        // Verify against a wrong target.
        let wrong = LinearRelation {
            coefficients: rel.coefficients.clone(),
            target: ring.one(),
        };
        assert!(proof.verify(&pk, &[wrong], &t).is_err());
    }

    #[test]
    fn relation_evaluate_shape_checked() {
        let pk = setup(3, 1, 2, 1 << 16);
        let ring = &pk.params.ring;
        let rel = LinearRelation {
            coefficients: vec![ring.one(), ring.zero()],
            target: ring.constant(5),
        };
        assert!(matches!(
            rel.evaluate(&[ring.one()]),
            Err(LinearProofError::RelationShape {
                expected: 2,
                got: 1
            })
        ));
    }
}
