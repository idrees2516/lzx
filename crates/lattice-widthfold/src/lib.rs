//! # lattice-widthfold — the width-reducing fold core (shared)
//!
//! The LaBRADOR-tail width fold (DESIGN_50KB.md Stage 5.2), extracted
//! from `lattice-zkvm::width_fold` into its own crate so BOTH consumers
//! compose the same security-critical machinery with zero duplication:
//!
//! * `lattice-zkvm` — the Sound compact profile (the memory argument's
//!   binding layer) and the recursive staging (`chain`, the benchmark-
//!   stream coverage);
//! * `lattice-folding` — the Cyclo §7 bridge's compact-PCS terminal
//!   (`ring_fold` — the decider that stops opening the witness).
//!
//! # Module map
//!
//! * [`codec`] — the response wire format (rANS-coded balanced
//!   coefficients) + ring-element serialization + the Q_32 column ring;
//! * [`helpers`] — the shared fold kernels: `apply_key`, the
//!   Goldilocks functional `functional_of` / `phi_term`, and the
//!   estimator gate `msis_bits` (the fail-closed posture primitive);
//! * [`fold`] — the quadratic-garbage width fold itself
//!   (`prove_width_fold` / `verify_width_fold`, the (W0)–(W4) checks);
//! * [`chain`] — the RECURSIVE width-collapse staging (log-stages of
//!   the sound rows; the benchmark-stream coverage extension);
//! * [`extraction`] — the multi-stage LaBRADOR extraction ledger (the
//!   degree-law unwind, the composed knowledge gap, the norm law, the
//!   grinding ledger, and the extractor-feasibility cap — the
//!   machine-checked half of `docs/analysis/MULTISTAGE_EXTRACTION.md`);
//! * [`ring_fold`] — the ring-functional width fold (the Cyclo bridge's
//!   compact terminal layer: exact mod-q linear identities ride the
//!   fold instead of the Goldilocks functional).
//!
//! # The security floor
//!
//! Every fold instance here terminates in MSIS on `[A₂ | −T]` at width
//! `w + r₂` with the extraction's relaxed bound `2·β₂`. The estimator
//! verdict ([`SECURITY_FLOOR_BITS`]) gates every profile fail-closed at
//! prove AND verify time — no fold ships a sub-floor instance.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::double_parens)]

pub mod chain;
pub mod codec;
pub mod extraction;
pub mod fold;
pub mod helpers;
pub mod ring_fold;

pub use codec::{
    decode_response, deserialize_elements, encode_response, q32_ring, serialize_elements,
    ResponseWire,
};
pub use fold::{prove_width_fold, verify_width_fold, WidthFoldParams, WidthFoldProof};

/// The security floor (classical bits) every profile enforces.
pub const SECURITY_FLOOR_BITS: f64 = 128.0;
