//! # lattice-vm
//!
//! The canonical RV64IMAC semantics layer: ONE decoded instruction enum
//! shared by execution, witness generation, and (later) verifier
//! preprocessing — the audit report's "one canonical semantics" principle.
//!
//! * `decode` — RV64IMAC decoder: base integer ops (RV64I), multiply
//!   (M), atomics (A), compressed (C) expansions to canonical micro-ops.
//! * `state` — machine state: 32 registers, PC, memory (word-addressed
//!   sparse map for the kernel; the zkVM layer owns the paged layout).
//! * `exec` — deterministic step function producing a `TraceRow` with the
//!   exact operands/flags/addresses the claim DAG consumes.

#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::double_parens
)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod decode;
#[cfg(test)]
mod differential;
pub mod exec;
pub mod golden;
pub mod reference;
pub mod state;

pub use decode::{decode, Instr, InstrFormat};
pub use exec::{run, step, ExecError, TraceRow};
pub use state::MachineState;
