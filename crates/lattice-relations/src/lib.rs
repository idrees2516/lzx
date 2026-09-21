//! # lattice-relations
//!
//! Constraint-system substrate consumed by the folding schemes
//! (SuperNeo/Neo fold CCS instances; LatticeFold+ folds tensor-ring
//! relaxations; the zkVM's residual CPU constraints compile to CCS).
//!
//! * `ccs` — Customizable Constraint Systems (Setty, Thaler 2023): the
//!   sparse "R1CS of lookup arguments" generalization that folding
//!   protocols natively consume: `Σ_j c_j · Π_k ⟨w, A_{σ(j,k)}⟩ = ⟨z, z⟩`
//!   style structure with sparse matrices and multiset equality vectors.
//! * `rlc` — random linear combination utilities for combining witness
//!   vectors and folding cross-terms with transcript challenges.
//! * `satisfaction` — CCS witness satisfaction checks (prover-side ground
//!   truth and test oracle; the proof layer replaces direct checks with
//!   sumcheck-based arguments).

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod ccs;
pub mod rlc;
pub mod satisfaction;

pub use ccs::{Ccs, CcsError, SparseMatrix};
