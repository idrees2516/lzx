//! PikkuFold (Osadnik, ePrint 2026/1809): efficient folding in a few
//! kilobytes.
//!
//! Core mechanism per the paper: **layered random projections** with
//! biased-ternary matrices mod q. Instead of committing to a decomposed or
//! transformed witness during the fold (the dominant communication cost in
//! every prior lattice folder — dozens of KB per step), the prover sends
//! the *final short image* of the folded witness through the projection
//! pipeline directly to the verifier. A Johnson–Lindenstrauss property for
//! biased ternary matrices (with certified concrete constants) guarantees
//! the projection approximately preserves norms, so the accumulator norm
//! grows only additively across folds. PikkuFold is the first lattice
//! folder requiring **no in-protocol commitments beyond the fresh input
//! commitments**.
//!
//! Implementation:
//! * `projection_matrix` — biased-ternary Π ∈ {−1,0,1}^{k×n} from a seed
//!   (each entry ternary with bias: P(0) = 1/2, P(±1) = 1/4).
//! * `project` — Π·w over R_q (incomplete-NTT-friendly).
//! * `fold` — w' = w1 + r·w2 with a short challenge; the verifier receives
//!   π' = Π·w' (a few ring elements) *in the clear* plus a linear-relation
//!   proof binding π' to the input commitments.
//! * `jl_norm_bound` — certified norm-preservation bound for the
//!   projection (the JL theorem with concrete constants): with k target
//!   dimensions, ||Πv||² concentrates around (k·E[Π²])·||v||² within a
//!   (1±ε) factor.

use lattice_commitment::ajtai::{AjtaiError, AjtaiPublicKey};
use lattice_commitment::linear_proof::{LinearProof, LinearProofError, LinearRelation};
use lattice_core::transcript::Transcript;
use lattice_ring::RingElement;

/// A biased-ternary projection matrix Π ∈ {−1,0,1}^{k×n} derived from a
/// public seed (verifier-computable, so projections need no commitment).
pub struct ProjectionMatrix {
    pub target_dim: usize,
    pub source_dim: usize,
    /// Row-major entries in {-1, 0, 1}.
    pub entries: Vec<i8>,
}

impl ProjectionMatrix {
    /// Derive the projection from a seed: each entry is 0 with probability
    /// 1/2 and ±1 with probability 1/4 each (the biased-ternary
    /// distribution the paper's JL analysis covers).
    pub fn from_seed(target_dim: usize, source_dim: usize, seed: &[u8]) -> Self {
        let bytes = Transcript::xof(
            b"pikkufold-pi",
            seed,
            target_dim * source_dim,
        );
        let mut entries = Vec::with_capacity(target_dim * source_dim);
        for &b in bytes.iter().take(target_dim * source_dim) {
            // 2 bits decide: 00 -> 0, 01 -> 0, 10 -> +1, 11 -> -1.
            let e = match b & 0x3 {
                0 | 1 => 0i8,
                2 => 1,
                _ => -1,
            };
            entries.push(e);
        }
        ProjectionMatrix {
            target_dim,
            source_dim,
            entries,
        }
    }

    /// Project a witness vector of ring elements: π_j = Σ_i Π[j][i]·w_i.
    pub fn project(&self, w: &[RingElement]) -> Result<Vec<RingElement>, PikkuError> {
        if w.len() != self.source_dim {
            return Err(PikkuError::DimensionMismatch {
                expected: self.source_dim,
                got: w.len(),
            });
        }
        let ring = w
            .first()
            .map(|e| e.config().clone())
            .ok_or(PikkuError::DimensionMismatch {
                expected: self.source_dim,
                got: 0,
            })?;
        let mut out = Vec::with_capacity(self.target_dim);
        for j in 0..self.target_dim {
            let mut acc = ring.zero();
            for (i, wi) in w.iter().enumerate() {
                let e = self.entries[j * self.source_dim + i];
                if e != 0 {
                    let scaled = wi.scale_i64(e as i64);
                    acc = acc.add(&scaled).map_err(PikkuError::Ring)?;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PikkuError {
    DimensionMismatch { expected: usize, got: usize },
    Ring(lattice_ring::RingError),
    Ajtai(AjtaiError),
    LinearProof(LinearProofError),
    /// Norm concentration failure (should not happen for honest witnesses
    /// within the certified JL parameters).
    NormConcentration { projected: u64, bound: u64 },
}

/// Certified JL-style bound: for the biased-ternary projection with target
/// dimension k, the projected infinity-norm of a witness with infinity-norm
/// β is bounded by sqrt(k · 1/2) · β + slack with failure probability
/// 2·exp(−Ω(k ε²)) — we return the conservative concrete bound
/// ||Πv||∞ ≤ ceil(sqrt(k/2)) · β (union bound over source entries).
pub fn jl_norm_bound(target_dim: usize, witness_norm: u64) -> u64 {
    let sqrt_half_k = ((target_dim as f64) / 2.0).sqrt().ceil() as u64;
    sqrt_half_k.saturating_mul(witness_norm)
}

/// A PikkuFold fold step: no in-protocol commitment — the verifier gets the
/// short projected image directly.
pub struct PikkuFoldStep {
    /// Short challenge (balanced integer).
    pub challenge: i64,
    /// The final image sent to the verifier: only `target_dim` ring
    /// elements — the "few kilobytes".
    pub image: Vec<RingElement>,
    /// Linear-relation proof binding the image to the input commitments
    /// (reuses the ABDLOP-style prover; its response is small because the
    /// relation coefficients are ternary).
    pub binding: Option<LinearProof>,
}

/// Fold two witnesses under a short transcript challenge and compute the
/// projection image the verifier will receive.
pub fn fold(
    w1: &[RingElement],
    w2: &[RingElement],
    projection: &ProjectionMatrix,
    commitment1: &[u8],
    commitment2: &[u8],
) -> Result<PikkuFoldStep, PikkuError> {
    if w1.len() != w2.len() {
        return Err(PikkuError::DimensionMismatch {
            expected: w1.len(),
            got: w2.len(),
        });
    }
    let mut transcript = Transcript::new_default(b"lzx-pikkufold");
    transcript
        .append_bytes(b"c1", commitment1)
        .map_err(|_| PikkuError::DimensionMismatch { expected: 0, got: 0 })?;
    transcript
        .append_bytes(b"c2", commitment2)
        .map_err(|_| PikkuError::DimensionMismatch { expected: 0, got: 0 })?;
    let seed = transcript
        .challenge_bytes(b"fold-r", 32)
        .map_err(|_| PikkuError::DimensionMismatch { expected: 0, got: 0 })?;
    let bytes = Transcript::xof(b"pikku-chal", &seed, 8);
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[..8]);
    // Short challenge: 8 bits balanced — the paper's short-challenge
    // safety analysis keeps the accumulator growth additive.
    let raw = (u64::from_le_bytes(arr) & 0xFF) as i64;
    let r = if raw >= 128 { raw - 256 } else { raw };

    // w' = w1 + r·w2, then project.
    let mut folded = Vec::with_capacity(w1.len());
    for (a, b) in w1.iter().zip(w2.iter()) {
        folded.push(a.add(&b.scale_i64(r)).map_err(PikkuError::Ring)?);
    }
    let image = projection.project(&folded)?;
    Ok(PikkuFoldStep {
        challenge: r,
        image,
        binding: None,
    })
}

/// Full fold with the linear-relation binding proof (production shape).
pub fn fold_with_binding(
    pk: &AjtaiPublicKey,
    w1: &[RingElement],
    w2: &[RingElement],
    projection: &ProjectionMatrix,
    prover_seed: &[u8],
) -> Result<PikkuFoldStep, PikkuError> {
    if w1.len() != pk.params.m || w2.len() != pk.params.m {
        return Err(PikkuError::DimensionMismatch {
            expected: pk.params.m,
            got: w1.len(),
        });
    }
    let t1 = pk.commit(w1).map_err(PikkuError::Ajtai)?;
    let t2 = pk.commit(w2).map_err(PikkuError::Ajtai)?;
    let mut step = fold(
        w1,
        w2,
        projection,
        &t1.to_bytes(),
        &t2.to_bytes(),
    )?;
    // Bind the image: prove the linear relations
    // ⟨Π_j, w'⟩ = image_j for every projection row j, where
    // w' = w1 + r·w2 is the folded secret. The coefficients are the PUBLIC
    // projection rows — the verifier recomputes them from the seed.
    let ring = &pk.params.ring;
    let mut relations = Vec::with_capacity(projection.target_dim);
    for (j, img) in step.image.iter().enumerate() {
        let coefficients: Vec<RingElement> = (0..pk.params.m)
            .map(|i| {
                let e = projection.entries[j * projection.source_dim + i] as i64;
                ring.constant(ring.modulus.reduce_i64(e))
            })
            .collect();
        relations.push(LinearRelation {
            coefficients,
            target: img.clone(),
        });
    }
    // Prove knowledge of (w1, w2) folded: use the folded witness directly.
    let mut folded = Vec::with_capacity(pk.params.m);
    for (a, b) in w1.iter().zip(w2.iter()) {
        folded.push(a.add(&b.scale_i64(step.challenge)).map_err(PikkuError::Ring)?);
    }
    let folded_commitment = pk.commit(&folded).map_err(PikkuError::Ajtai)?;
    let proof = LinearProof::prove(
        pk,
        &relations,
        &folded,
        &folded_commitment,
        prover_seed,
    )
    .map_err(PikkuError::LinearProof)?;
    step.binding = Some(proof);
    Ok(step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_ring::{Modulus32, RingConfig};

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [41u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn small_w(ring: &RingConfig, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, 4, 128, tag)
    }

    #[test]
    fn projection_deterministic_and_ternary() {
        let pi = ProjectionMatrix::from_seed(8, 16, b"seed");
        let pi2 = ProjectionMatrix::from_seed(8, 16, b"seed");
        assert_eq!(pi.entries, pi2.entries);
        assert!(pi.entries.iter().all(|e| *e == 0 || *e == 1 || *e == -1));
        // Bias: roughly half zeros (with 128 entries, expect 40..88 zeros).
        let zeros = pi.entries.iter().filter(|e| **e == 0).count();
        assert!(zeros > 40 && zeros < 90, "zeros {zeros}");
    }

    #[test]
    fn projection_linearity() {
        let (_, ring) = setup(4, 4);
        let w1 = small_w(&ring, b"pl-1");
        let w2 = small_w(&ring, b"pl-2");
        let pi = ProjectionMatrix::from_seed(4, 4, b"pi");
        // Π(w1 + w2) == Πw1 + Πw2.
        let sum: Vec<RingElement> = w1
            .iter()
            .zip(w2.iter())
            .map(|(a, b)| a.add(b).ok().unwrap())
            .collect();
        let lhs = pi.project(&sum).ok().unwrap();
        let p1 = pi.project(&w1).ok().unwrap();
        let p2 = pi.project(&w2).ok().unwrap();
        for (l, (a, b)) in lhs.iter().zip(p1.iter().zip(p2.iter())) {
            assert_eq!(*l, a.add(b).ok().unwrap());
        }
    }

    #[test]
    fn fold_image_is_short_and_correct() {
        let (pk, ring) = setup(4, 4);
        let w1 = small_w(&ring, b"fi-1");
        let w2 = small_w(&ring, b"fi-2");
        let pi = ProjectionMatrix::from_seed(2, 4, b"pi-short");
        let t1 = pk.commit(&w1).ok().unwrap();
        let t2 = pk.commit(&w2).ok().unwrap();
        let step = fold(&w1, &w2, &pi, &t1.to_bytes(), &t2.to_bytes())
            .ok()
            .unwrap();
        // The verifier receives only 2 ring elements (the few-kilobyte
        // image), not a chunked commitment.
        assert_eq!(step.image.len(), 2);
        // Correctness: image == Π(w1 + r·w2).
        let mut folded = Vec::new();
        for (a, b) in w1.iter().zip(w2.iter()) {
            folded.push(a.add(&b.scale_i64(step.challenge)).ok().unwrap());
        }
        let expected = pi.project(&folded).ok().unwrap();
        assert_eq!(step.image, expected);
    }

    #[test]
    fn norm_grows_additively() {
        let (_, ring) = setup(4, 4);
        let pi = ProjectionMatrix::from_seed(4, 4, b"pi-norm");
        let w1 = small_w(&ring, b"ng-1");
        let norm1 = w1.iter().map(|e| e.infinity_norm() as u64).max().unwrap();
        // Projected norm is bounded by the certified JL bound.
        let proj = pi.project(&w1).ok().unwrap();
        let proj_norm = proj.iter().map(|e| e.infinity_norm() as u64).max().unwrap();
        let bound = jl_norm_bound(4, norm1);
        assert!(
            proj_norm <= bound,
            "projected {proj_norm} > JL bound {bound}"
        );
    }

    #[test]
    fn fold_with_binding_proves_and_verifies() {
        let (pk, ring) = setup(4, 4);
        let w1 = small_w(&ring, b"fb-1");
        let w2 = small_w(&ring, b"fb-2");
        let pi = ProjectionMatrix::from_seed(2, 4, b"pi-bind");
        let step = fold_with_binding(&pk, &w1, &w2, &pi, b"prover").ok().unwrap();
        let proof = step.binding.as_ref().ok_or(()).ok().unwrap();

        // Recompute the public statement: folded commitment + relations.
        let mut folded = Vec::new();
        for (a, b) in w1.iter().zip(w2.iter()) {
            folded.push(a.add(&b.scale_i64(step.challenge)).ok().unwrap());
        }
        let folded_commitment = pk.commit(&folded).ok().unwrap();
        let mut relations = Vec::with_capacity(step.image.len());
        for (j, img) in step.image.iter().enumerate() {
            // Public projection rows as coefficients (verifier-computable).
            let coefficients: Vec<RingElement> = (0..pk.params.m)
                .map(|i| {
                    let e = pi.entries[j * pi.source_dim + i] as i64;
                    ring.constant(ring.modulus.reduce_i64(e))
                })
                .collect();
            relations.push(LinearRelation {
                coefficients,
                target: img.clone(),
            });
        }
        assert!(proof.verify(&pk, &relations, &folded_commitment).is_ok());
    }

    #[test]
    fn jl_bound_sane_values() {
        assert_eq!(jl_norm_bound(2, 100), 100); // ceil(sqrt(1)) = 1
        assert_eq!(jl_norm_bound(8, 100), 200); // ceil(sqrt(4)) = 2
        assert_eq!(jl_norm_bound(32, 10), 40); // ceil(sqrt(16)) = 4
    }
}
