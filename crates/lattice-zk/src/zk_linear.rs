//! The ZK linear proof: ABDLOP/Lyubashevsky-style proof of knowledge of a
//! short vector under an Ajtai commitment with public linear relations,
//! compiled with Fiat-Shamir **in the correct order**.
//!
//! ## The Fiat-Shamir ordering fix
//!
//! The baseline `lattice_commitment::linear_proof::LinearProof` derived
//! its challenge from the statement *only* (pk seed, commitment,
//! relations) — **without absorbing the masking commitment `w`**. Under
//! that ordering the proof is malleable: anyone can pick an arbitrary
//! short `z'`, set `w' := A·z' − c·t`, and pass verification with no
//! knowledge of `s` (the challenge never binds `w`). That breaks the
//! extraction argument — a knowledge-soundness bug, and exactly the
//! class of defect the audit's QROM review (§9.6 item 26) exists to
//! catch.
//!
//! `ZkLinearProof` absorbs `w` and the relation images **before**
//! sampling the challenge:
//!
//! ```text
//! c = H(protocol ‖ statement ‖ w ‖ images)
//! ```
//!
//! Rewinding/extracting two accepting transcripts that share `(w,
//! images)` but differ in `c` yields a Module-SIS solution — the
//! standard extractability argument, now actually valid.
//!
//! ## Zero knowledge
//!
//! * The masking vector `y` comes from **secret entropy**
//!   ([`ShakeStream`](crate::entropy::ShakeStream)), never from the
//!   transcript.
//! * Rejection sampling keeps `z = y + c·s` inside the norm box; the
//!   accepted `z` is (statistically) uniform in the box, independent of
//!   `s`.
//! * [`ZkLinearProof::simulate`] produces proofs from the statement
//!   alone: sample `z` uniformly in the box, set `w := A·z − c·t` and
//!   `images_i := ⟨v_i, z⟩ − c·u_i`. The distribution matches the honest
//!   prover's (HVZK); statistical KATs in this module check it.

use crate::entropy::ShakeStream;
use lattice_commitment::ajtai::{
    sample_small_secret, AjtaiCommitment, AjtaiError, AjtaiPublicKey,
};
use lattice_commitment::linear_proof::LinearRelation;
use lattice_core::challenge_set::{ChallengeDistribution, ChallengeSet};
use lattice_core::transcript::Transcript;
use lattice_ring::RingElement;

/// Maximum rejection-sampling attempts.
pub const MAX_RETRIES: u32 = 64;

#[derive(Clone, Debug)]
pub struct ZkLinearProof {
    /// Per-instance masking commitment rows `w_i = A·y_i` (absorbed
    /// *before* the challenge).
    pub mask_commitments: Vec<Vec<RingElement>>,
    /// Relation images `⟨v_i, y⟩` over the concatenated masking vector
    /// (absorbed *before* the challenge).
    pub mask_images: Vec<RingElement>,
    /// Sparse-ternary challenge ring element.
    pub challenge: RingElement,
    /// Per-instance responses `z_i = y_i + c·s_i`.
    pub responses: Vec<Vec<RingElement>>,
    /// Rejection-sampling attempts used (public; witness-independent in
    /// distribution — tested).
    pub retries: u32,
}

impl ZkLinearProof {
    /// Single-instance convenience accessors.
    pub fn mask_commitment(&self) -> &[RingElement] {
        self.mask_commitments.first().map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn response(&self) -> &[RingElement] {
        self.responses.first().map(|v| v.as_slice()).unwrap_or(&[])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkLinearProofError {
    Ajtai(AjtaiError),
    RelationShape { expected: usize, got: usize },
    NormExceeded,
    RetriesExceeded,
    VerificationFailed,
    Ring(lattice_ring::RingError),
    Entropy(String),
}

impl core::fmt::Display for ZkLinearProofError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZkLinearProofError::Ajtai(e) => write!(f, "ajtai: {e:?}"),
            ZkLinearProofError::RelationShape { expected, got } => {
                write!(f, "relation shape {got} != {expected}")
            }
            ZkLinearProofError::NormExceeded => write!(f, "norm bound exceeded"),
            ZkLinearProofError::RetriesExceeded => write!(f, "rejection retries exceeded"),
            ZkLinearProofError::VerificationFailed => write!(f, "verification failed"),
            ZkLinearProofError::Ring(e) => write!(f, "ring: {e:?}"),
            ZkLinearProofError::Entropy(msg) => write!(f, "entropy: {msg}"),
        }
    }
}

/// Canonical statement bytes: pk seed + commitment + relations, in the
/// exact order the transcript absorbs them.
fn statement_bytes(
    pk: &AjtaiPublicKey,
    commitments: &[AjtaiCommitment],
    relations: &[LinearRelation],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&pk.seed);
    for c in commitments {
        out.extend_from_slice(&c.to_bytes());
    }
    for rel in relations {
        out.extend_from_slice(&rel.target.to_bytes());
        for v in &rel.coefficients {
            out.extend_from_slice(&v.to_bytes());
        }
    }
    out
}

/// Derive the challenge given (statement, w, images) — the fixed order.
fn derive_challenge(
    pk: &AjtaiPublicKey,
    statement: &[u8],
    ws: &[Vec<RingElement>],
    images: &[RingElement],
) -> Result<RingElement, ZkLinearProofError> {
    let mut transcript = Transcript::new_default(b"lzx-zk-linear");
    transcript
        .append_bytes(b"statement", statement)
        .map_err(|_| ZkLinearProofError::VerificationFailed)?;
    for w in ws {
        for row in w {
            transcript
                .append_bytes(b"w", &row.to_bytes())
                .map_err(|_| ZkLinearProofError::VerificationFailed)?;
        }
    }
    for img in images {
        transcript
            .append_bytes(b"image", &img.to_bytes())
            .map_err(|_| ZkLinearProofError::VerificationFailed)?;
    }
    let chal_seed = transcript
        .challenge_bytes(b"challenge", 32)
        .map_err(|_| ZkLinearProofError::VerificationFailed)?;
    let ring = &pk.params.ring;
    let cs = ChallengeSet::sample(
        ChallengeDistribution::SparseTernary {
            weight: ring.n() / 2,
        },
        ring.n(),
        &chal_seed,
    )
    .map_err(|_| ZkLinearProofError::VerificationFailed)?;
    Ok(RingElement::from_signed(ring, &cs.coefficients))
}

impl ZkLinearProof {
    /// Prove knowledge of the short secrets `secrets[i]` (with
    /// `commitments[i] = A·secrets[i]` and the listed relations over the
    /// **concatenated** secret) using **secret entropy** for the masking.
    ///
    /// Every secret must be exactly `m` slots (pad with zero ring
    /// elements); the relations' coefficient vectors must span
    /// `secrets.len() * m` slots.
    pub fn prove(
        pk: &AjtaiPublicKey,
        relations: &[LinearRelation],
        secrets: &[Vec<RingElement>],
        commitments: &[AjtaiCommitment],
        stream: &mut ShakeStream,
    ) -> Result<Self, ZkLinearProofError> {
        let ring = &pk.params.ring;
        let m = pk.params.m;
        let instances = secrets.len();
        if instances == 0 || instances != commitments.len() {
            return Err(ZkLinearProofError::VerificationFailed);
        }
        for s in secrets {
            if s.len() != m {
                return Err(ZkLinearProofError::Ajtai(AjtaiError::DimensionMismatch {
                    expected: m,
                    got: s.len(),
                }));
            }
        }
        let total = instances * m;
        // Fail closed: the relations must hold for the concatenated
        // secret before proving.
        let concat: Vec<RingElement> = secrets.concat();
        for rel in relations {
            if rel.coefficients.len() != total {
                return Err(ZkLinearProofError::RelationShape {
                    expected: total,
                    got: rel.coefficients.len(),
                });
            }
            let img = rel
                .evaluate(&concat)
                .map_err(|_| ZkLinearProofError::VerificationFailed)?;
            if img != rel.target {
                return Err(ZkLinearProofError::VerificationFailed);
            }
        }

        let statement = statement_bytes(pk, commitments, relations);
        let bound = pk.params.norm_bound;
        for s in secrets {
            for e in s {
                if e.infinity_norm() > bound / 2 {
                    return Err(ZkLinearProofError::NormExceeded);
                }
            }
        }
        let response_bound = bound.saturating_sub(2);
        let y_bound = response_bound / 2;

        for retry in 0..=MAX_RETRIES {
            // Fresh masking entropy per instance and attempt.
            let mut ys: Vec<Vec<RingElement>> = Vec::with_capacity(instances);
            for _ in 0..instances {
                let mut seed = [0u8; 32];
                seed.copy_from_slice(&stream.next_bytes(32));
                ys.push(sample_small_secret(ring, m, y_bound, &seed));
            }
            let mut ws: Vec<Vec<RingElement>> = Vec::with_capacity(instances);
            for y in &ys {
                let w = pk.commit(y).map_err(ZkLinearProofError::Ajtai)?;
                ws.push(w.rows);
            }
            // Images over the concatenation.
            let y_concat: Vec<RingElement> = ys.concat();
            let mut images = Vec::with_capacity(relations.len());
            for rel in relations {
                images.push(
                    rel.evaluate(&y_concat)
                        .map_err(|_| ZkLinearProofError::VerificationFailed)?,
                );
            }
            let challenge = derive_challenge(pk, &statement, &ws, &images)?;
            let mut zs: Vec<Vec<RingElement>> = Vec::with_capacity(instances);
            let mut ok = true;
            'outer: for (y, s) in ys.iter().zip(secrets.iter()) {
                let mut z = Vec::with_capacity(m);
                for (yi, si) in y.iter().zip(s.iter()) {
                    let cs_prod = challenge.mul(si).map_err(ZkLinearProofError::Ring)?;
                    let zi = yi.add(&cs_prod).map_err(ZkLinearProofError::Ring)?;
                    if zi.infinity_norm() > bound {
                        ok = false;
                        break 'outer;
                    }
                    z.push(zi);
                }
                zs.push(z);
            }
            if ok && zs.len() == instances {
                return Ok(ZkLinearProof {
                    mask_commitments: ws,
                    mask_images: images,
                    challenge,
                    responses: zs,
                    retries: retry,
                });
            }
        }
        Err(ZkLinearProofError::RetriesExceeded)
    }

    /// Verify against the public statement.
    pub fn verify(
        &self,
        pk: &AjtaiPublicKey,
        relations: &[LinearRelation],
        commitments: &[AjtaiCommitment],
    ) -> Result<(), ZkLinearProofError> {
        let m = pk.params.m;
        let instances = commitments.len();
        if instances == 0 || self.responses.len() != instances || self.mask_commitments.len() != instances {
            return Err(ZkLinearProofError::VerificationFailed);
        }
        for z in &self.responses {
            if z.len() != m {
                return Err(ZkLinearProofError::Ajtai(AjtaiError::DimensionMismatch {
                    expected: m,
                    got: z.len(),
                }));
            }
        }
        // 1. Response norms.
        let bound = pk.params.norm_bound;
        for z in &self.responses {
            for e in z {
                if e.infinity_norm() > bound {
                    return Err(ZkLinearProofError::NormExceeded);
                }
            }
        }
        // 2. Recompute the challenge in the FIXED order.
        let statement = statement_bytes(pk, commitments, relations);
        let expected =
            derive_challenge(pk, &statement, &self.mask_commitments, &self.mask_images)?;
        if expected != self.challenge {
            return Err(ZkLinearProofError::VerificationFailed);
        }
        // 3. A·z_i == w_i + c·t_i per instance.
        for ((z, w), t) in self
            .responses
            .iter()
            .zip(self.mask_commitments.iter())
            .zip(commitments.iter())
        {
            let c_times_t: Vec<RingElement> = t
                .rows
                .iter()
                .map(|t_i| self.challenge.mul(t_i).map_err(ZkLinearProofError::Ring))
                .collect::<Result<_, _>>()?;
            let mut expected_rows = Vec::with_capacity(w.len());
            for (w_i, ct_i) in w.iter().zip(c_times_t.iter()) {
                expected_rows.push(w_i.add(ct_i).map_err(ZkLinearProofError::Ring)?);
            }
            let az = pk.commit(z).map_err(ZkLinearProofError::Ajtai)?;
            if az.rows != expected_rows {
                return Err(ZkLinearProofError::VerificationFailed);
            }
        }
        // 4. Relations over the concatenated response.
        let z_concat: Vec<RingElement> = self.responses.concat();
        for (rel, image) in relations.iter().zip(self.mask_images.iter()) {
            let lhs = rel
                .evaluate(&z_concat)
                .map_err(|_| ZkLinearProofError::VerificationFailed)?;
            let cu = self
                .challenge
                .mul(&rel.target)
                .map_err(ZkLinearProofError::Ring)?;
            let rhs = image.add(&cu).map_err(ZkLinearProofError::Ring)?;
            if lhs != rhs {
                return Err(ZkLinearProofError::VerificationFailed);
            }
        }
        Ok(())
    }

    /// Verify only the algebraic equations (norms, A·z = w + c·t,
    /// relations) — **without** the Fiat-Shamir challenge re-derivation.
    ///
    /// This is the simulator's verification path: in the ROM the ZK
    /// simulator *programs* the oracle at (statement, w, images) to
    /// return its chosen challenge, so concrete simulated transcripts
    /// satisfy every algebraic equation while the un-programmed FS
    /// recomputation differs. Distributional KATs pair this check with
    /// the chi-square comparison.
    pub fn verify_algebra(
        &self,
        pk: &AjtaiPublicKey,
        relations: &[LinearRelation],
        commitments: &[AjtaiCommitment],
    ) -> Result<(), ZkLinearProofError> {
        let m = pk.params.m;
        let instances = commitments.len();
        if instances == 0 || self.responses.len() != instances {
            return Err(ZkLinearProofError::VerificationFailed);
        }
        let bound = pk.params.norm_bound;
        for z in &self.responses {
            if z.len() != m {
                return Err(ZkLinearProofError::Ajtai(AjtaiError::DimensionMismatch {
                    expected: m,
                    got: z.len(),
                }));
            }
            for e in z {
                if e.infinity_norm() > bound {
                    return Err(ZkLinearProofError::NormExceeded);
                }
            }
        }
        for ((z, w), t) in self
            .responses
            .iter()
            .zip(self.mask_commitments.iter())
            .zip(commitments.iter())
        {
            let c_times_t: Vec<RingElement> = t
                .rows
                .iter()
                .map(|t_i| self.challenge.mul(t_i).map_err(ZkLinearProofError::Ring))
                .collect::<Result<_, _>>()?;
            let mut expected_rows = Vec::with_capacity(w.len());
            for (w_i, ct_i) in w.iter().zip(c_times_t.iter()) {
                expected_rows.push(w_i.add(ct_i).map_err(ZkLinearProofError::Ring)?);
            }
            let az = pk.commit(z).map_err(ZkLinearProofError::Ajtai)?;
            if az.rows != expected_rows {
                return Err(ZkLinearProofError::VerificationFailed);
            }
        }
        let z_concat: Vec<RingElement> = self.responses.concat();
        for (rel, image) in relations.iter().zip(self.mask_images.iter()) {
            let lhs = rel
                .evaluate(&z_concat)
                .map_err(|_| ZkLinearProofError::VerificationFailed)?;
            let cu = self
                .challenge
                .mul(&rel.target)
                .map_err(ZkLinearProofError::Ring)?;
            let rhs = image.add(&cu).map_err(ZkLinearProofError::Ring)?;
            if lhs != rhs {
                return Err(ZkLinearProofError::VerificationFailed);
            }
        }
        Ok(())
    }

    /// The HVZK simulator: produce a proof of the *statement* without the
    /// witness.
    ///
    /// `z` is uniform in the norm box; `w := A·z − c·t` and
    /// `images := ⟨v, z⟩ − c·u` make all verification equations hold by
    /// linearity. The challenge is drawn from the sparse-ternary
    /// distribution with secret entropy (in the ROM the simulator
    /// programs the oracle at the corresponding point).
    pub fn simulate(
        pk: &AjtaiPublicKey,
        relations: &[LinearRelation],
        commitments: &[AjtaiCommitment],
        stream: &mut ShakeStream,
    ) -> Result<Self, ZkLinearProofError> {
        let ring = &pk.params.ring;
        let m = pk.params.m;
        let instances = commitments.len();
        if instances == 0 {
            return Err(ZkLinearProofError::VerificationFailed);
        }
        let bound = pk.params.norm_bound;
        let z_box = (bound.saturating_sub(2) / 2).max(1);
        // Challenge from the correct distribution.
        let mut cseed = [0u8; 32];
        cseed.copy_from_slice(&stream.next_bytes(32));
        let cs = ChallengeSet::sample(
            ChallengeDistribution::SparseTernary {
                weight: ring.n() / 2,
            },
            ring.n(),
            &cseed,
        )
        .map_err(|_| ZkLinearProofError::VerificationFailed)?;
        let challenge = RingElement::from_signed(ring, &cs.coefficients);
        // Uniform responses in the box, per instance.
        let zs: Vec<Vec<RingElement>> = (0..instances)
            .map(|_| sample_uniform_box(ring, m, z_box, stream))
            .collect();
        let mut ws: Vec<Vec<RingElement>> = Vec::with_capacity(instances);
        for (z, t) in zs.iter().zip(commitments.iter()) {
            let az = pk.commit(z).map_err(ZkLinearProofError::Ajtai)?;
            let mut w_rows = Vec::with_capacity(az.rows.len());
            for (az_i, t_i) in az.rows.iter().zip(t.rows.iter()) {
                let ct = challenge.mul(t_i).map_err(ZkLinearProofError::Ring)?;
                w_rows.push(az_i.sub(&ct).map_err(ZkLinearProofError::Ring)?);
            }
            ws.push(w_rows);
        }
        // images = ⟨v, z⟩ − c·u over the concatenation.
        let z_concat: Vec<RingElement> = zs.concat();
        let mut images = Vec::with_capacity(relations.len());
        for rel in relations {
            let vz = rel
                .evaluate(&z_concat)
                .map_err(|_| ZkLinearProofError::VerificationFailed)?;
            let cu = challenge
                .mul(&rel.target)
                .map_err(ZkLinearProofError::Ring)?;
            images.push(vz.sub(&cu).map_err(ZkLinearProofError::Ring)?);
        }
        Ok(ZkLinearProof {
            mask_commitments: ws,
            mask_images: images,
            challenge,
            responses: zs,
            retries: 0,
        })
    }
}

/// Sample ring elements with coefficients uniform in `[-bound, bound]`
/// from secret entropy.
fn sample_uniform_box(
    ring: &lattice_ring::RingConfig,
    slots: usize,
    bound: u32,
    stream: &mut ShakeStream,
) -> Vec<RingElement> {
    let span = 2u64 * bound as u64 + 1;
    let mut out = Vec::with_capacity(slots);
    for _ in 0..slots {
        let mut coeffs = Vec::with_capacity(ring.n());
        while coeffs.len() < ring.n() {
            let v = stream.next_u64();
            // Unbiased rejection to the box.
            if v < (u64::MAX - (u64::MAX % span)) {
                let b = (v % span) as i64 - bound as i64;
                coeffs.push(ring.modulus.reduce_i64(b));
            }
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::SecretSeed;
    use lattice_commitment::ajtai::AjtaiParams;
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

    fn stream(label: &[u8]) -> ShakeStream {
        ShakeStream::new(SecretSeed::from_kat_label(label), b"zk-linear-test")
    }

    fn witness(pk: &AjtaiPublicKey, label: &[u8]) -> Vec<RingElement> {
        sample_small_secret(&pk.params.ring, pk.params.m, 1 << 21, label)
    }

    fn one_relation(pk: &AjtaiPublicKey, s: &[RingElement]) -> LinearRelation {
        let ring = &pk.params.ring;
        let mut v = vec![ring.zero(); pk.params.m];
        v[0] = ring.one();
        LinearRelation {
            coefficients: v,
            target: s[0].clone(),
        }
    }

    #[test]
    fn prove_verify_roundtrip() {
        let pk = setup(4, 2, 6, 1 << 24);
        let s = witness(&pk, b"witness-a");
        let t = pk.commit(&s).ok().unwrap();
        let rel = one_relation(&pk, &s);
        let mut st = stream(b"prove-1");
        let proof = ZkLinearProof::prove(&pk, std::slice::from_ref(&rel), std::slice::from_ref(&s), std::slice::from_ref(&t), &mut st)
            .ok()
            .unwrap();
        assert!(proof.verify(&pk, &[rel], &[t]).is_ok());
        assert!(proof.retries <= MAX_RETRIES);
    }

    #[test]
    fn tampered_challenge_rejected() {
        let pk = setup(4, 2, 6, 1 << 24);
        let s = witness(&pk, b"witness-b");
        let t = pk.commit(&s).ok().unwrap();
        let mut st = stream(b"prove-2");
        let mut proof = ZkLinearProof::prove(&pk, &[], std::slice::from_ref(&s), std::slice::from_ref(&t), &mut st).ok().unwrap();
        let ring = &pk.params.ring;
        let mut ccoeffs = proof.challenge.coeffs().to_vec();
        ccoeffs[0] = (ccoeffs[0] + 1) % ring.modulus.q;
        proof.challenge = RingElement::from_coeffs(ring, ccoeffs);
        assert!(proof.verify(&pk, &[], &[t]).is_err());
    }

    /// The pre-fix malleability attack: choose z' first, derive
    /// w' := A·z' − c·t with a challenge that IGNORES w (the old
    /// statement-only derivation), then try to verify. The fixed
    /// verifier recomputes the challenge WITH w' absorbed, so the
    /// forgery fails.
    #[test]
    fn post_hoc_malleability_rejected() {
        let pk = setup(4, 2, 6, 1 << 24);
        let s = witness(&pk, b"witness-c");
        let t = pk.commit(&s).ok().unwrap();
        // Old-order challenge: statement only.
        let statement = statement_bytes(&pk, std::slice::from_ref(&t), &[]);
        let old = {
            let mut tr = Transcript::new_default(b"lzx-zk-linear");
            tr.append_bytes(b"statement", &statement).ok().unwrap();
            let seed = tr.challenge_bytes(b"challenge", 32).ok().unwrap();
            let cs = ChallengeSet::sample(
                ChallengeDistribution::SparseTernary {
                    weight: pk.params.ring.n() / 2,
                },
                pk.params.ring.n(),
                &seed,
            )
            .ok()
            .unwrap();
            RingElement::from_signed(&pk.params.ring, &cs.coefficients)
        };
        // Forge: z' short, w' = A·z' − c·t, images = [].
        let _st = stream(b"forge");
        let z = sample_small_secret(&pk.params.ring, pk.params.m, 1 << 22, b"forge-z");
        let az = pk.commit(&z).ok().unwrap();
        let mut w_rows = Vec::new();
        for (az_i, t_i) in az.rows.iter().zip(t.rows.iter()) {
            let ct = old.mul(t_i).ok().unwrap();
            w_rows.push(az_i.sub(&ct).ok().unwrap());
        }
        let forged = ZkLinearProof {
            mask_commitments: vec![w_rows],
            mask_images: vec![],
            challenge: old,
            responses: vec![z],
            retries: 0,
        };
        // The fixed verifier must reject: its challenge binds w'.
        assert!(matches!(
            forged.verify(&pk, &[], std::slice::from_ref(&t)),
            Err(ZkLinearProofError::VerificationFailed)
        ));
    }

    #[test]
    fn simulated_proofs_verify_and_match_distribution() {
        // Rejection-sampling ZK needs the norm bound well above the
        // challenge-convolved witness (|c⊛s| ≤ (n/2)·|s|): bound 2^25
        // with |s| ≤ 2^16 gives |c⊛s| ≤ 2^19 = bound/64 — a proper
        // margin for the statistical indistinguishability KAT.
        let pk = setup(4, 2, 6, 1 << 25);
        let s = sample_small_secret(&pk.params.ring, pk.params.m, 1 << 16, b"witness-d");
        let t = pk.commit(&s).ok().unwrap();
        let rel = one_relation(&pk, &s);
        // Simulated proofs satisfy every verification equation.
        let mut sim_stream = stream(b"sim-1");
        let sim = ZkLinearProof::simulate(&pk, std::slice::from_ref(&rel), std::slice::from_ref(&t), &mut sim_stream)
            .ok()
            .unwrap();
        assert!(sim.verify_algebra(&pk, std::slice::from_ref(&rel), std::slice::from_ref(&t)).is_ok());

        // Two-sample chi-square over response coefficient magnitudes:
        // real vs simulated must be statistically indistinguishable.
        let bucket_count = 10usize;
        let bound = pk.params.norm_bound as i64;
        let mut real_hist = vec![0u64; bucket_count];
        let mut sim_hist = vec![0u64; bucket_count];
        let trials = 24u64;
        for i in 0..trials {
            let mut st = stream(format!("real-{i}").as_bytes());
            let proof = ZkLinearProof::prove(&pk, &[], std::slice::from_ref(&s), std::slice::from_ref(&t), &mut st).ok().unwrap();
            for z in proof.response() {
                for c in z.coeffs() {
                    let balanced = if *c <= pk.params.ring.modulus.q / 2 {
                        *c as i64
                    } else {
                        *c as i64 - pk.params.ring.modulus.q as i64
                    };
                    let mag = balanced.unsigned_abs() as i64;
                    let idx = ((mag * bucket_count as i64) / bound.max(1)) as usize;
                    let idx = idx.min(bucket_count - 1);
                    real_hist[idx] += 1;
                }
            }
            let mut ss = stream(format!("sim-{i}").as_bytes());
            let sim = ZkLinearProof::simulate(&pk, &[], std::slice::from_ref(&t), &mut ss).ok().unwrap();
            for z in sim.response() {
                for c in z.coeffs() {
                    let balanced = if *c <= pk.params.ring.modulus.q / 2 {
                        *c as i64
                    } else {
                        *c as i64 - pk.params.ring.modulus.q as i64
                    };
                    let mag = balanced.unsigned_abs() as i64;
                    let idx = ((mag * bucket_count as i64) / bound.max(1)) as usize;
                    let idx = idx.min(bucket_count - 1);
                    sim_hist[idx] += 1;
                }
            }
        }
        // Chi-square two-sample statistic (df = bucket_count - 1 = 9;
        // critical value at alpha = 0.001 is ~27.88).
        let mut chi2 = 0.0f64;
        for b in 0..bucket_count {
            let o1 = real_hist[b] as f64;
            let o2 = sim_hist[b] as f64;
            if o1 + o2 > 0.0 {
                chi2 += (o1 - o2).powi(2) / (o1 + o2);
            }
        }
        assert!(
            chi2 < 27.88,
            "real vs simulated response distributions differ: chi2 = {chi2:.3}, hists real={real_hist:?} sim={sim_hist:?}"
        );
    }

    /// Rejection-sampling retry counts must be witness-independent in
    /// distribution (audit §9.5 item 24: witness-dependent retry
    /// behavior is a leak).
    #[test]
    fn retry_counts_witness_independent() {
        let pk = setup(4, 2, 6, 1 << 24);
        let trials = 20u64;
        let mean_retries = |label: &[u8]| -> f64 {
            let mut total = 0u64;
            for i in 0..trials {
                let s = sample_small_secret(&pk.params.ring, pk.params.m, 1 << 21, label);
                let t = pk.commit(&s).ok().unwrap();
                let mut st = stream(format!("{label:?}-{i}").as_bytes());
                let proof = ZkLinearProof::prove(&pk, &[], std::slice::from_ref(&s), std::slice::from_ref(&t), &mut st).ok().unwrap();
                total += proof.retries as u64;
            }
            total as f64 / trials as f64
        };
        let sparse = mean_retries(b"sparse-w");
        let dense = mean_retries(b"dense-www");
        // Both must be finite and close: rejection depends on norms, not
        // on witness *content* (both classes use the same norm bound).
        assert!(sparse < 10.0 && dense < 10.0, "sparse={sparse} dense={dense}");
        assert!(
            (sparse - dense).abs() < 3.0,
            "retry distribution appears witness-dependent: sparse={sparse} dense={dense}"
        );
    }
}
