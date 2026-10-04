//! # lattice-zkvm
//!
//! The end-to-end zkVM: program → trace → claim DAG → staged sumchecks →
//! Akita PCS openings → bounded proof envelope → verification.
//!
//! Architecture (the audit report's claim-DAG stages):
//! 1. **Execute** — the canonical RV64IMAC executor produces trace rows
//!    (register reads/writes, memory accesses, PC transitions).
//! 2. **Claim DAG** — each trace stream becomes typed claims:
//!    register-Twist (read/write timeline), RAM-Twist (memory), and the
//!    instruction-semantics relation over the witness columns.
//! 3. **Prove** — every claim reduces to virtual-polynomial sumchecks
//!    (through `lattice-relations`' statement builder); the final factor
//!    claims open through the Akita PCS (packed commitment + norm-checked
//!    response + grouped openings).
//! 4. **Envelope** — the bounded canonical proof binding program digest,
//!    public input digest, schedule, and public output digest before the
//!    first challenge.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::double_parens)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod columns;
pub mod compact;
pub mod constraints;
pub mod envelope;
pub mod ledger;
pub mod legbatch;
pub mod lookup_memory;
pub mod memory;
pub mod memproof;
pub mod norm_check;
pub mod pipeline;
pub mod pipeline2;
pub mod pipeline3;
pub mod prove;
pub mod second_fold;
pub mod semantics;
pub mod streaming;
pub mod ttrp;
pub mod width_fold {
    //! The extracted width-fold core (see the `lattice-widthfold` crate).
    pub use lattice_widthfold::chain::{
        prove_width_fold_chain, verify_width_fold_chain, WidthChainParams, WidthChainProof,
        CHAIN_GRINDING_BITS,
    };
    pub use lattice_widthfold::fold::{
        prove_width_fold, prove_width_fold_ex, verify_width_fold, verify_width_fold_ex,
        WidthFoldParams, WidthFoldProof,
    };
}

#[cfg(test)]
mod fuzz;

pub use envelope::{ProofEnvelope, MAX_SECTIONS, MAX_SECTION_BYTES};
pub use prove::{prove_program, verify_program, PublicOutput, ZkvmError};

use lattice_core::transcript::Transcript;

/// Statement digests (program + public I/O) — the envelope preamble.
pub fn program_digest(program: &[u8]) -> [u8; 32] {
    Transcript::hash_domain(b"zkvm-program", program)
}

pub fn public_input_digest(public_input: &[u8]) -> [u8; 32] {
    Transcript::hash_domain(b"zkvm-public-input", public_input)
}
