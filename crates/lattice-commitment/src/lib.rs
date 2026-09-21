//! # lattice-commitment
//!
//! Ajtai-style commitments over module lattices — the binding core of the
//! whole protocol stack (Akita commitments, folding scheme cross terms,
//! LatticeFold+ tensor rings, SALSA linear commitments).
//!
//! * `ajtai` — Module-SIS vector commitment `t = A·s mod q` with
//!   deterministic seed-derived public matrices, small-norm opening
//!   support, and gadget-decomposed response folding.
//! * `linear_proof` — ABDLOP-style linear proof of knowledge: prove a
//!   committed vector satisfies public linear equations with a short
//!   response (the "linear-relation kernel" every lattice PCS builds on).
//! * `norm_proof` — proofs that committed vectors have bounded norm
//!   (direct and digit-decomposed routes; the sumcheck-accelerated route
//!   lives in lattice-salsa).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod ajtai;
pub mod linear_proof;
pub mod norm_proof;

pub use ajtai::{AjtaiCommitment, AjtaiParams, AjtaiPublicKey};
