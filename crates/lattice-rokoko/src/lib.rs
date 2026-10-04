//! # lattice-rokoko
//!
//! RoKoko (ePrint 2026/575): lattice-based succinct arguments — a
//! committed refinement.
//!
//! Per the paper and its reference implementation (lattice-arguments/
//! rokoko): power-of-two cyclotomic rings, almost-splitting factors,
//! incomplete NTTs, a modular sumcheck interface, and **coarse/fine
//! committed random projections**: the verifier's check applies to a short
//! random projection of the committed witness, where the projection matrix
//! is sampled in two refinement stages (coarse then fine), and the
//! projection's correctness is proven sumcheck-aided.
//!
//! Components:
//! * `projection` — coarse/fine two-stage random projections over R_q
//!   (biased-ternary entries; the fine stage narrows the coarse image).
//! * `refine` — the committed refinement protocol: commit the coarse
//!   projection, prove it, then refine to the fine projection whose
//!   binding the verifier checks directly.
//! * `sumcheck_hook` — the modular sumcheck interface the projections
//!   plug into (linear-map identity as a virtual polynomial).
//! * `com` — the recursive Ajtai commitment COM (paper Fig. 1, Wave 7
//!   item 7.13 component 3): G^{-1} gadget recursion with power-of-two
//!   padding and the b0/b1/b2 verification gates.
//! * `protocol` — the committed-linear relation Ξ^lin_COM, Π^fold-split
//!   (Fig. 4), the sumcheckify constraint system (Fig. 5), Π^lin
//!   (Fig. 6) and the round driver with the terminal opening (items
//!   4+5), over the lattice-salsa ring sumcheck engine.

#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod com;
pub mod pcs_front;
pub mod proj_f;
pub mod protocol;
pub mod schedule;

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiParams, AjtaiPublicKey};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::RingElement;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RokokoError {
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    Transcript(TranscriptError),
    ProjectionShape { expected: usize, got: usize },
    RefinementFailed,
}

/// A random projection stage: entries in {-1, 0, 1} derived from a seed,
/// applied to ring-element witness vectors.
pub struct RandomProjection {
    pub target_dim: usize,
    pub source_dim: usize,
    entries: Vec<i8>,
}

impl RandomProjection {
    /// Derive a projection matrix from a seed (domain-separated stage).
    pub fn from_seed(stage: &[u8], target_dim: usize, source_dim: usize, seed: &[u8]) -> Self {
        let bytes = Transcript::xof(
            b"rokoko-projection",
            &[stage, seed].concat(),
            target_dim * source_dim,
        );
        let mut entries = Vec::with_capacity(target_dim * source_dim);
        for &b in bytes.iter().take(target_dim * source_dim) {
            entries.push(match b % 3 {
                0 => 0i8,
                1 => 1,
                _ => -1,
            });
        }
        RandomProjection {
            target_dim,
            source_dim,
            entries,
        }
    }

    /// Project a witness: π_j = Σ_i Π[j][i]·w_i over R_q.
    pub fn project(&self, w: &[RingElement]) -> Result<Vec<RingElement>, RokokoError> {
        if w.len() != self.source_dim {
            return Err(RokokoError::ProjectionShape {
                expected: self.source_dim,
                got: w.len(),
            });
        }
        let ring = w
            .first()
            .map(|e| e.config().clone())
            .ok_or(RokokoError::ProjectionShape {
                expected: self.source_dim,
                got: 0,
            })?;
        let mut out = Vec::with_capacity(self.target_dim);
        for j in 0..self.target_dim {
            let mut acc = ring.zero();
            for (i, wi) in w.iter().enumerate() {
                let e = self.entries[j * self.source_dim + i];
                if e != 0 {
                    acc = acc
                        .add(&wi.scale_i64(e as i64))
                        .map_err(RokokoError::Ring)?;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }

    /// Entry accessor (for proof recomputation).
    pub fn entry(&self, j: usize, i: usize) -> Option<i8> {
        if j < self.target_dim && i < self.source_dim {
            self.entries.get(j * self.source_dim + i).copied()
        } else {
            None
        }
    }
}

/// The committed refinement statement: coarse commitment + fine image.
pub struct RefinedProjection {
    /// Coarse stage: commitment to the coarse projection image.
    pub coarse_commitment: AjtaiCommitment,
    /// Fine stage: the short image the verifier checks directly.
    pub fine_image: Vec<RingElement>,
    /// The fine-stage challenge (absorbed into the transcript).
    pub fine_seed: [u8; 32],
}

/// The committed refinement protocol:
/// 1. Coarse projection Π_c (wide) — image committed under Ajtai.
/// 2. Fine projection Π_f (narrow) applied to the coarse image — sent
///    directly to the verifier (short).
/// 3. The verifier recomputes Π_f from the transcript seed and checks the
///    linear binding through the coarse commitment opening.
pub fn refine(
    pk: &AjtaiPublicKey,
    w: &[RingElement],
    coarse_target: usize,
    fine_target: usize,
) -> Result<RefinedProjection, RokokoError> {
    if w.len() != pk.params.m {
        return Err(RokokoError::ProjectionShape {
            expected: pk.params.m,
            got: w.len(),
        });
    }
    // Stage 1: coarse projection from a public seed (derived from the key).
    let coarse_seed: [u8; 32] = {
        let bytes = Transcript::xof(b"rokoko-coarse-seed", &pk.seed, 32);
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        arr
    };
    let coarse = RandomProjection::from_seed(b"coarse", coarse_target, w.len(), &coarse_seed);
    let coarse_image = coarse.project(w)?;
    // Commit the coarse image (padded).
    let padded = pk.pad_to_m(&coarse_image).map_err(RokokoError::Ajtai)?;
    let coarse_commitment = pk.commit(&padded).map_err(RokokoError::Ajtai)?;

    // Stage 2: fine projection over the coarse image.
    let mut transcript = Transcript::new_default(b"lzx-rokoko");
    transcript
        .append_bytes(b"coarse", &coarse_commitment.to_bytes())
        .map_err(RokokoError::Transcript)?;
    let fine_seed_vec = transcript
        .challenge_bytes(b"fine-seed", 32)
        .map_err(RokokoError::Transcript)?;
    let mut fine_seed = [0u8; 32];
    fine_seed.copy_from_slice(&fine_seed_vec);
    let fine = RandomProjection::from_seed(b"fine", fine_target, coarse_image.len(), &fine_seed);
    let fine_image = fine.project(&coarse_image)?;

    Ok(RefinedProjection {
        coarse_commitment,
        fine_image,
        fine_seed,
    })
}

/// Verify the refinement binding: given the prover's coarse opening
/// (witness-level check in this kernel; production replaces it with the
/// linear-proof binding), recompute the fine image and compare.
pub fn verify_refinement(
    pk: &AjtaiPublicKey,
    statement: &RefinedProjection,
    coarse_witness: &[RingElement],
    _coarse_target: usize,
    fine_target: usize,
) -> Result<bool, RokokoError> {
    // Coarse commitment must open to the provided coarse image.
    let padded = pk.pad_to_m(coarse_witness).map_err(RokokoError::Ajtai)?;
    if pk
        .verify_opening(&statement.coarse_commitment, &padded)
        .is_err()
    {
        return Ok(false);
    }
    // Recompute the fine projection from the transcript-derived seed.
    let fine = RandomProjection::from_seed(
        b"fine",
        fine_target,
        coarse_witness.len(),
        &statement.fine_seed,
    );
    let expected = fine.project(coarse_witness)?;
    Ok(expected == statement.fine_image)
}

/// Parameters helper.
pub fn rokoko_params(log_n: u32, m: usize, coarse_target: usize) -> Option<AjtaiParams> {
    let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n).ok()?;
    Some(AjtaiParams {
        ring,
        k: 2,
        m: m.max(coarse_target),
        norm_bound: 1 << 22,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_w(ring: &lattice_ring::RingConfig, m: usize, tag: &[u8]) -> Vec<RingElement> {
        lattice_commitment::ajtai::sample_small_secret(ring, m, 512, tag)
    }

    #[test]
    fn projection_deterministic_ternary() {
        let pi = RandomProjection::from_seed(b"stage", 4, 8, b"seed");
        let pi2 = RandomProjection::from_seed(b"stage", 4, 8, b"seed");
        assert_eq!(pi.entry(0, 0), pi2.entry(0, 0));
        assert_eq!(pi.target_dim, 4);
        // Ternary entries.
        for j in 0..4 {
            for i in 0..8 {
                let e = pi.entry(j, i).ok_or(()).ok().unwrap();
                assert!(e == 0 || e == 1 || e == -1);
            }
        }
        // Different stage -> different matrix.
        let other = RandomProjection::from_seed(b"other", 4, 8, b"seed");
        let mut differ = false;
        for j in 0..4 {
            for i in 0..8 {
                if pi.entry(j, i) != other.entry(j, i) {
                    differ = true;
                }
            }
        }
        assert!(differ);
    }

    #[test]
    fn committed_refinement_roundtrip() {
        let params = rokoko_params(4, 8, 4).unwrap();
        let pk = AjtaiPublicKey::from_seed(params, [61u8; 32]).ok().unwrap();
        let ring = pk.params.ring.clone();
        let w = small_w(&ring, pk.params.m, b"rokoko-w");
        let stmt = refine(&pk, &w, 4, 2).ok().unwrap();
        // Verifier: recompute the coarse image (here via the witness; the
        // production path replaces this with the committed linear proof),
        // then check the fine binding.
        let coarse_seed: [u8; 32] = {
            let bytes = Transcript::xof(b"rokoko-coarse-seed", &pk.seed, 32);
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        };
        let coarse = RandomProjection::from_seed(b"coarse", 4, w.len(), &coarse_seed);
        let coarse_image = coarse.project(&w).ok().unwrap();
        assert!(verify_refinement(&pk, &stmt, &coarse_image, 4, 2)
            .ok()
            .unwrap());
    }

    #[test]
    fn refinement_tamper_detected() {
        let params = rokoko_params(4, 8, 4).unwrap();
        let pk = AjtaiPublicKey::from_seed(params, [62u8; 32]).ok().unwrap();
        let ring = pk.params.ring.clone();
        let w = small_w(&ring, pk.params.m, b"rokoko-t");
        let mut stmt = refine(&pk, &w, 4, 2).ok().unwrap();
        // Tamper with the fine image.
        if !stmt.fine_image.is_empty() {
            let mut coeffs = stmt.fine_image[0].coeffs().to_vec();
            coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
            stmt.fine_image[0] = RingElement::from_coeffs(&ring, coeffs);
        }
        let coarse_seed: [u8; 32] = {
            let bytes = Transcript::xof(b"rokoko-coarse-seed", &pk.seed, 32);
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        };
        let coarse = RandomProjection::from_seed(b"coarse", 4, w.len(), &coarse_seed);
        let coarse_image = coarse.project(&w).ok().unwrap();
        assert!(!verify_refinement(&pk, &stmt, &coarse_image, 4, 2)
            .ok()
            .unwrap());
    }

    #[test]
    fn refinement_shape_errors() {
        let params = rokoko_params(4, 8, 4).unwrap();
        let pk = AjtaiPublicKey::from_seed(params, [63u8; 32]).ok().unwrap();
        let ring = pk.params.ring.clone();
        let w = small_w(&ring, 3, b"short");
        assert!(matches!(
            refine(&pk, &w, 4, 2),
            Err(RokokoError::ProjectionShape { .. })
        ));
    }
}
