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

pub mod claimsfold;
pub mod blockfold;
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
pub mod pipeline4;
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
pub use streaming::{
    prove_program_streaming, verify_program_streaming, StreamingProof, StreamingZkvmError,
};

use lattice_core::transcript::Transcript;

/// Statement digests (program + public I/O) — the envelope preamble.
pub fn program_digest(program: &[u8]) -> [u8; 32] {
    Transcript::hash_domain(b"zkvm-program", program)
}

pub fn public_input_digest(public_input: &[u8]) -> [u8; 32] {
    Transcript::hash_domain(b"zkvm-public-input", public_input)
}

// ---------------------------------------------------------------------------
// The default prover backend (Wave 10, SOTA mechanism #3): the streaming
// O(K + log T) path is the DEFAULT — unbounded T at the client budget,
// no O(T) materialization on the prover or verifier path.
// ---------------------------------------------------------------------------

/// The prover backend selection.
#[derive(Clone, Debug)]
pub enum ProverBackend {
    /// The streaming client-side prover (ePrint 2025/611): the
    /// checkpointed regeneration oracles drive every component from the
    /// VM's own step function — `O(K + log T)` space, unbounded cycle
    /// count at the caller's memory budget. **The default.**
    Streaming {
        /// The client memory budget in field elements (the hybrid
        /// space/time switch derives its chunking from it).
        budget_field_elements: usize,
    },
    /// The materialized in-memory path (`prove_program`'s envelope
    /// route) — retained for differential/benchmark use.
    Materialized,
}

impl Default for ProverBackend {
    fn default() -> Self {
        // 64 MiB worth of field elements — the comfortable client budget
        // (the pure O(K + log T) regime clamps from here).
        ProverBackend::Streaming {
            budget_field_elements: 8 * 1024 * 1024,
        }
    }
}

/// A proof from either backend.
#[derive(Clone, Debug)]
pub enum ZkvmProof {
    /// The streaming proof (`O(√T)`-sized artifacts, no materialized
    /// witness).
    Streaming(Box<StreamingProof>),
    /// The materialized-path envelope.
    Envelope(ProofEnvelope),
}

/// Prove with the DEFAULT backend (the streaming `O(K + log T)` path,
/// SOTA mechanism #3 — the cycle caps of the materialized posture do
/// not apply).
pub fn prove_program_default(
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(PublicOutput, ZkvmProof), ZkvmError> {
    let backend = ProverBackend::default();
    prove_program_with(backend, program, public_input, max_steps)
}

/// Prove with an explicit backend selection.
pub fn prove_program_with(
    backend: ProverBackend,
    program: &[u8],
    public_input: &[u8],
    max_steps: u64,
) -> Result<(PublicOutput, ZkvmProof), ZkvmError> {
    match backend {
        ProverBackend::Streaming {
            budget_field_elements,
        } => {
            let config = lattice_streaming::client::ClientProverConfig {
                max_field_elements: budget_field_elements,
                progress: None,
            };
            let (out, proof) =
                streaming::prove_program_streaming(program, public_input, max_steps, &config)
                    .map_err(|_| ZkvmError::VerificationFailed)?;
            Ok((out, ZkvmProof::Streaming(Box::new(proof))))
        }
        ProverBackend::Materialized => {
            // The materialized route needs the caller's PCS; use the
            // canonical test geometry (the envelope route is documented
            // differential-only at kernel scale).
            let pcs = lattice_akita::akita_setup(8, 64, 1 << 23, [7u8; 32]).map_err(|_| {
                ZkvmError::Pcs(lattice_akita::pcs::AkitaPcsError::VerificationFailed)
            })?;
            let (out, envelope) = prove::prove_program(&pcs, program, public_input, max_steps)?;
            Ok((out, ZkvmProof::Envelope(envelope)))
        }
    }
}

/// Verify a proof from either backend against the public output.
pub fn verify_program_default(
    program: &[u8],
    public_input: &[u8],
    public_output: &PublicOutput,
    proof: &ZkvmProof,
    max_steps: u64,
) -> Result<(), ZkvmError> {
    match proof {
        ZkvmProof::Streaming(p) => {
            streaming::verify_program_streaming(program, public_input, public_output, p, max_steps)
                .map_err(|_| ZkvmError::VerificationFailed)
        }
        ZkvmProof::Envelope(e) => {
            let pcs = lattice_akita::akita_setup(8, 64, 1 << 23, [7u8; 32]).map_err(|_| {
                ZkvmError::Pcs(lattice_akita::pcs::AkitaPcsError::VerificationFailed)
            })?;
            prove::verify_program(&pcs, program, public_input, public_output, e, max_steps)
        }
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    fn demo_program() -> Vec<u8> {
        // addi x1, x0, 8; addi x2, x0, 7; add x3, x1, x2; ecall
        let words = [0x0080_0093u32, 0x0070_0113, 0x0020_81b3, 0x0000_0073];
        let mut v = Vec::new();
        for w in words {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }

    #[test]
    fn default_backend_is_streaming_and_roundtrips() {
        // SOTA mechanism #3: the DEFAULT route is the streaming O(K+log T)
        // path — no cycle caps, no O(T) materialization.
        let program = demo_program();
        let (out, proof) = prove_program_default(&program, &[], 64).ok().unwrap();
        assert!(matches!(proof, ZkvmProof::Streaming(_)));
        assert_eq!(out.final_regs[3], 15);
        assert!(verify_program_default(&program, &[], &out, &proof, 64).is_ok());
        // A tampered public output is rejected.
        let mut bad = out.clone();
        bad.final_regs[3] += 1;
        assert!(verify_program_default(&program, &[], &bad, &proof, 64).is_err());
    }

    #[test]
    fn materialized_backend_still_selectable() {
        let program = demo_program();
        let (out, proof) = prove_program_with(ProverBackend::Materialized, &program, &[], 64)
            .ok()
            .unwrap();
        assert!(matches!(proof, ZkvmProof::Envelope(_)));
        assert!(verify_program_default(&program, &[], &out, &proof, 64).is_ok());
    }
}
