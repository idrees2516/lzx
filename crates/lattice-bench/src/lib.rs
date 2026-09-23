//! # lattice-bench
//!
//! The reproducible benchmark matrix (audit gate G7): pure-std timing
//! harness plus size accounting. Run the binary:
//! `cargo run -p lattice-bench --release`.

pub mod harness;

pub use harness::{fmt_ns, measure, Timing};
