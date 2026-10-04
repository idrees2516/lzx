//! # lattice-folding
//!
//! Folding-scheme implementations for the six lattice folding papers:
//!
//! | Module | Paper | Core mechanism |
//! |--------|-------|----------------|
//! | `protogalattice` | ProtogaLattice (2026/1317) | constant-round folding of degree-d polynomial relations via tensor challenge powers |
//! | `latticefold_plus` | LatticeFold+ (2025/247) | tensor-ring Ajtai commitments with exact norm accounting |
//! | `cyclo` | Cyclo (2026/359) | norm-refreshing via partial range checks on high digits |
//! | `pikkufold` | PikkuFold (2026/1809) | few-kilobyte folding via seed-compressed witnesses |
//! | `symphony` | Symphony (2025/1905) | high-arity (μ-ary) folding in the ROM |
//! | `superneo` | Neo/SuperNeo (2026/242) | pay-per-bit CCS folding over small fields |
//!
//! Every module exposes the same shape: instance/witness types, a `fold`
//! operation producing the folded instance plus cross-term commitments, a
//! prover that proves the fold, and a verifier that checks it — with tests
//! proving the algebraic fold identities hold and tampering is rejected.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod cyclo;
pub mod cyclo_r1cs;
pub mod neo;
pub mod cyclo_protocols;
pub mod fq2_sumcheck;
pub mod latticefold_plus;
pub mod lfplus_mon;
pub mod lfplus_l2;
pub mod pgl;
pub mod pikkufold;
pub mod pikkufold_lrp;
pub mod protogalattice;
pub mod superneo;
pub mod superneo_committed;
pub mod symphony;
pub mod pi_ccs;
pub mod symphony_protocols;
