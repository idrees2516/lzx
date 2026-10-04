//! The NARK `FS[Π_sps^cm]` of §5.2 — the committed-message special-sound
//! protocol compiled with Fiat–Shamir.
//!
//! * `G`: samples the vector-Pedersen key (the commitment key `ck`).
//! * `I`: binds a relation + key into proving/verification keys.
//! * `P`: from `(x, w)`, produces the messages, commits each (hiding), and
//!   derives the challenge chain `r₁ = ρ(x)`, `rᵢ = ρ(rᵢ₋₁, Cᵢ)`.
//! * `V`: re-derives the challenges, checks the commitments open the
//!   messages, and checks the algebraic map evaluates to zero.
//!
//! This is the **non-zero-knowledge** NARK the paper's §4 construction
//! deliberately starts from: zero-knowledge is *not* needed here because
//! the proof is never transmitted — it exists only to be accumulated by
//! the (zero-knowledge) accumulation scheme of `accum`. That decoupling is
//! the paper's first contribution.

use crate::pedersen::PedersenKey;
use crate::sps::{
    check_predicate, commit_messages, derive_challenges, SpsInstance, SpsRelation, SpsWitness,
};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NarkError {
    Sps(crate::sps::SpsError),
}

impl core::fmt::Display for NarkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NarkError::Sps(e) => write!(f, "sps: {e}"),
        }
    }
}

impl From<crate::sps::SpsError> for NarkError {
    fn from(e: crate::sps::SpsError) -> Self {
        NarkError::Sps(e)
    }
}

/// A NARK proof: `π = (π.x = (x, [Cᵢ], [rᵢ]), π.w = ([mᵢ], blinds))`.
#[derive(Clone, Debug)]
pub struct NarkProof {
    pub instance: SpsInstance,
    pub witness: SpsWitness,
}

/// `NARK.P`: produce the committed-message proof for `(x, w)` where `w` is
/// the prover's message vector bundle (for µ=1 relations, a single vector).
pub fn prove<R: SpsRelation>(
    rel: &R,
    key: &PedersenKey,
    x: &[Fp256],
    messages: &[Vec<Fp256>],
    transcript: &mut Transcript,
) -> Result<NarkProof, NarkError> {
    if messages.len() != rel.num_rounds() {
        return Err(NarkError::Sps(crate::sps::SpsError::Shape(
            "message round count",
        )));
    }
    let (mut inst, wit) = commit_messages(messages, key, transcript)?;
    inst.x = x.to_vec();
    inst.challenges = derive_challenges(&inst)?;
    Ok(NarkProof {
        instance: inst,
        witness: wit,
    })
}

/// `NARK.V`: verify a proof.
pub fn verify<R: SpsRelation>(
    rel: &R,
    key: &PedersenKey,
    proof: &NarkProof,
) -> Result<bool, NarkError> {
    let mut t = Transcript::new_default(b"nark-verify");
    Ok(check_predicate(
        rel,
        &proof.instance,
        &proof.witness,
        key,
        &mut t,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::R1csSps;

    #[test]
    fn nark_roundtrip_and_rejection() {
        let mut rel = R1csSps::random(2, 5, 3, b"nark-seed");
        let (x, w) = rel.sample_satisfying(b"nw").ok().unwrap();
        let mut z = vec![Fp256::from_canonical_u64(1)];
        z.extend(x.iter().cloned());
        z.extend(w.iter().cloned());
        let key = PedersenKey::derive(&[21u8; 32], 32).ok().unwrap();
        let mut t = Transcript::new_default(b"nark");
        let proof = prove(&rel, &key, &x, &[z.clone()], &mut t).ok().unwrap();
        assert!(verify(&rel, &key, &proof).ok().unwrap());
        // Tampered witness message → reject.
        let mut bad = proof.clone();
        bad.witness.messages[0][2] = bad.witness.messages[0][2].add(&Fp256::from_canonical_u64(1));
        assert!(!verify(&rel, &key, &bad).ok().unwrap());
        // Tampered instance → reject.
        let mut bad2 = proof.clone();
        bad2.instance.x[0] = bad2.instance.x[0].add(&Fp256::from_canonical_u64(1));
        assert!(!verify(&rel, &key, &bad2).ok().unwrap());
        // Tampered challenge → reject (re-derivation mismatch).
        let mut bad3 = proof.clone();
        if !bad3.instance.challenges.is_empty() {
            bad3.instance.challenges[0] =
                bad3.instance.challenges[0].add(&Fp256::from_canonical_u64(1));
            assert!(!verify(&rel, &key, &bad3).ok().unwrap());
        }
    }
}
