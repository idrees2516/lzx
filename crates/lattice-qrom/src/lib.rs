//! # lattice-qrom
//!
//! The QROM (Quantum Random Oracle Model) accountability layer of the
//! LZX stack — audit report §9.6 item 26 (gate G5): *"Finalize the
//! interactive protocol and extraction tree accounting; then perform
//! the QROM Fiat–Shamir proof for the full composition."*
//!
//! This crate makes the Fiat-Shamir composition **reviewable**:
//!
//! * [`ledger`] — worst-case random-oracle query accounting per protocol
//!   stage, including rejection-sampling amplification. A proof that
//!   exceeds its declared query budget is rejected before any expensive
//!   arithmetic runs.
//! * [`domains`] — the workspace-wide domain-separation registry: every
//!   production transcript label, uniqueness-checked at test time, with
//!   a runtime registry for dynamic registration.
//! * [`attestation`] — the `QromAttestation` artifact: the prover
//!   declares the stage list and per-stage query counts; the verifier
//!   checks the digest binding, uniqueness, and budget.
//! * [`review`] — the composition review: machine-checkable items
//!   (challenge-after-message ordering, budget coverage, domain
//!   uniqueness) plus the manual sign-off checklist; a passing review
//!   is the only path that grants the `QromFiatShamir` capability
//!   token (via [`lattice_zk::privacy_spec`]).
//!
//! ## Query accounting model
//!
//! Every `challenge_fields` / `challenge_bytes` call on a transcript
//! counts as one random-oracle query (the transcript's internal
//! counter). Rejection sampling inside a call amplifies the worst case:
//! with acceptance probability `1 − ε` per candidate and a hard retry
//! cap `R`, the worst case is `1 + R` queries (the cap is enforced by
//! the transcript's `RejectionBudgetExceeded` error). The ledger's
//! [`StageRecord`](attestation::StageRecord) carries the amplification
//! factor explicitly so reviewers see the inflated totals, not just
//! the happy path.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod attestation;
pub mod domains;
pub mod ledger;
pub mod review;

pub use attestation::{QromAttestation, StageRecord};
pub use domains::{DomainCollision, DomainRegistry, PRODUCTION_DOMAINS};
pub use ledger::{QueryLedger, RejectionModel};
pub use review::{review_composition, ReviewError, ReviewReport};
