//! # lattice-akita
//!
//! Akita (ePrint 2026/1983): a high-performance lattice-based polynomial
//! commitment scheme — the baseline post-quantum PCS of the LZX stack.
//!
//! End-to-end evaluation protocol:
//! 1. **Commit** — the MLE's evaluations are packed into ring elements
//!    (3×22-bit limbs per Goldilocks value) and committed under an Ajtai
//!    Module-SIS key.
//! 2. **Prove evaluation** — `f(r) = v` reduces to the multilinear
//!    identity `Σ_x eq(r, x)·f(x) = f(r)`, proven by sumcheck over the
//!    packed tensor structure (Akita's carrier/tensor layer); the final
//!    factor claims bind the committed packing.
//! 3. **Open** — the response reveals the packed witness at the challenge
//!    point with a digit-decomposed norm proof (the norm layer) plus the
//!    Ajtai linear-relation binding (the fold layer).
//! 4. **Grouped openings** — multiple (point, value) claims against one
//!    commitment batch through a single combined sumcheck.
//! 5. **Schedule / security metadata** — the trusted parameter catalog
//!    structure (digest-selected, verifier-registered).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod fold;
pub mod pcs;
pub mod ring_check;
pub mod salsa_binding;
pub mod salsa_response;
pub mod schedule;

pub use pcs::{
    verify_evaluation, AkitaPcs, AkitaPcsError, Commitment, EvaluationProof, GroupedOpening,
};
pub use schedule::{ScheduleCatalog, ScheduleEntry, SecurityProfile};

use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};

/// Construct an Akita PCS instance: the Ajtai key plus packing geometry.
pub fn akita_setup(
    log_n: u32,
    m: usize,
    norm_bound: u32,
    seed: [u8; 32],
) -> Result<AkitaPcs, AkitaPcsError> {
    let ring = lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, log_n)
        .map_err(AkitaPcsError::Ring)?;
    let params = AjtaiParams {
        ring,
        k: 2,
        m,
        norm_bound,
    };
    let pk = AjtaiPublicKey::from_seed(params, seed).map_err(AkitaPcsError::Ajtai)?;
    Ok(AkitaPcs { pk })
}
