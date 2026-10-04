//! # lattice-streaming
//!
//! **Small-space, streaming, client-side proving** — a full Rust
//! implementation of *"Proving CPU Executions in Small Space"*
//! (Nair, Thaler, Zhu; ePrint 2025/611) over the LZX Goldilocks stack.
//!
//! The paper's thesis: the fastest known zkVM provers (Jolt's
//! sum-check-based components — Spartan, Twist, Shout) are *already
//! almost streaming*; with the right prover algorithms the whole
//! pipeline runs in `O(K + log T)` (or the simpler `O(√T)`) space
//! **without SNARK recursion**, with a concrete slowdown well under
//! 2× — because the `O(T log T)` term carries a constant ~75× smaller
//! than the linear-time term's (§1.2, §7.1).
//!
//! ## Module map
//!
//! * [`oracle`] — the witness-stream interfaces: sequential
//!   `StreamOracle`s, Algorithm-1 `IndexOracle`s, stateful
//!   witness generators, and **checkpointed regeneration** (the
//!   client-side random-access realization with parallelizable chunk
//!   replay).
//! * [`window_schedule`] — **the streaming window schedule**
//!   (2026/587 §5.2/C.4.2, Figure 2): `EvalProductStream_{k},SC` —
//!   geometrically-growing then space-capped windows with the bound
//!   tables *emulated* by eq-folds, `O(M^{1/k})`-class space,
//!   bit-identical round messages.
//! * [`small_space`] — **Algorithm 1**: the `O(n + ℓ²)`-space sum-check
//!   prover with Gray-coded eq walks; round messages bit-identical to
//!   the in-memory engine.
//! * [`hybrid`] — the **space/time switch**: Algorithm-1 rounds until
//!   the bound arrays fit a caller-chosen budget, one materializing
//!   sequential pass, then the in-memory linear-time finish. `c = n/2`
//!   is the paper's `O(√T)` regime.
//! * [`prefix_suffix`] — the **prefix-suffix inner product protocol**
//!   (Appendix A): the paper's new linear-time, `O(C·k·N^{1/C})`-space
//!   prover for `Σ ũ·ã` with prefix-suffix-structured `ã` — the
//!   pcnext-evaluation and M-evaluation sum-checks.
//! * [`grand_product`] — the **streaming grand product check**
//!   (Appendix D): the depth-first product-tree walk with an `O(n)`
//!   stack (Quarks' `f(x,1) = f(0,x)·f(1,x)` identity via sum-check).
//! * [`pcs_stream`] — the **matrix-layout streaming commitment**
//!   (§6.1): the Ligero/Brakedown-style `√N × √N` row-encoded
//!   commitment with `O(√N)`-space, single-pass row streaming.
//! * [`client`] — the client-side facade: memory budgets, peak-RSS
//!   metering, progress callbacks, and the WASM-friendly discipline
//!   (no threads, no mmap, no file I/O in the core path).

#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod client;
pub mod grand_product;
pub mod hybrid;
pub mod oracle;
pub mod pcs_stream;
pub mod prefix_suffix;
pub mod small_space;
pub mod window_schedule;
