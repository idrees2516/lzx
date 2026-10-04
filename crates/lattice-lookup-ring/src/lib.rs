//! # lattice-lookup-ring
//!
//! Lookup arguments over rings (Bootle–Guskind–Patranabis–Sotiraki,
//! ePrint 2026/471) and their Greyhound-style compile onto the
//! workspace's Ajtai/carrier stack.
//!
//! * `ring_d` — the split ring `R = Z_q[X]/(X^d+1) ≅ F_{q^{d/2}}²`,
//!   CRT projections, slot inversion, the binary challenge space `C`,
//!   the index map `g`, and the MLE machinery over `R`.
//! * `attacks` — Section 4's attacks (CRT-component swapping on
//!   Plookup/Lasso grand-products, the zero-divisor attacks on LogUp)
//!   as executable demonstrations.
//! * `ring_sumcheck` — the sum-check protocol over rings (Appendix A.3)
//!   over a multilinear virtual-polynomial system.
//! * `subprotocols` — Appendix B's toolkit: scalar product, Hadamard
//!   product, cyclic shift, entry product, integer check, and the
//!   LatticeFold+-derived binary check (Construction B.19).
//! * `ring_plookup` — Ring-Plookup (Construction 5.5).
//! * `ring_logup` — Ring-LogUp (Construction 5.11).
//! * `ram` — Section 6: offline memory checking, memory consistency,
//!   almost-identical RAM states, batch verification of RAM updates.
//! * `carrier` — the Ajtai commitment layer over the split ring (the
//!   carrier stack ported off the NTT kernel; MSIS binding unchanged).
//! * `windowed` — the windowed engine: digit-window decompositions on
//!   the √N × √N grid with the two-phase tensor binding pass
//!   (Greyhound-style MLE evaluation arguments).
//! * `compile` — the Greyhound-style compile: PIOP oracles become
//!   Ajtai commitments, evaluation queries ride the windowed engine.
//! * `fp256_port` — the Fp256 (BN254 scalar field) port of the windowed
//!   engine's binding pass on the CIOS Montgomery grid.

pub mod attacks;
pub mod carrier;
pub mod compile;
pub mod fp256_port;
pub mod ram;
pub mod ring_d;
pub mod ring_logup;
pub mod ring_plookup;
pub mod ring_sumcheck;
pub mod subprotocols;
pub mod windowed;
pub use ring_d::{Elem, RingD, RingDError, Q_SPLIT, SQRT_M1};
