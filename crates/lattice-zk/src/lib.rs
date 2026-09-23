//! # lattice-zk
//!
//! The lattice-native zero-knowledge layer of the LZX stack (audit report
//! §9.5 P4 / gate G4).
//!
//! Design principles (from the audit):
//! * **Privacy is a capability, not a default.** Everything that hides a
//!   witness is gated behind [`privacy_spec::SecurityCapability`] tokens
//!   bound to a protocol digest; a proof is never labeled "zk" merely
//!   because its PCS is lattice-based.
//! * **Private masks come from OS entropy.** Transcript-derived randomness
//!   is public by definition and can never supply hiding randomness
//!   (§9.5 item 23). The type system separates the two entropy worlds.
//! * **Choose published mechanisms, do not ad-hoc mask.** The ZK linear
//!   proof is the ABDLOP/Lyubashevsky Σ-protocol compiled with Fiat-Shamir
//!   *in the correct order* (mask commitment absorbed before the
//!   challenge); the ZK sumcheck masks round polynomials with secret
//!   uniform masks whose accumulated contribution is bound through a
//!   CRT-carrier linear relation over the committed mask limbs.
//! * **Simulators and negative tests ship with the code.** Every ZK
//!   construction in this crate has a distributional simulator and
//!   statistical KATs, plus negative tests for reused randomness,
//!   deterministic RNG misuse, and forged (post-hoc malleated) proofs.
//!
//! ## Module map
//! * [`entropy`] — OS/secret entropy, seeded XOF streams, nonce ledger.
//! * [`privacy_spec`] — leakage declarations and capability tokens.
//! * [`carrier`] — limb decomposition and the CRT-carrier linear relation
//!   that bridges field-level claims to mod-q ring relations exactly.
//! * [`zk_linear`] — the fixed, properly ordered ABDLOP ZK linear proof
//!   with simulator.
//! * [`zk_sumcheck`] — the secret-mask zero-knowledge sumcheck.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(clippy::needless_range_loop)]

pub mod carrier;
pub mod entropy;
pub mod privacy_spec;
pub mod zk_linear;
pub mod zk_sumcheck;

pub use carrier::{CarrierError, CarrierRelation};
pub use entropy::{NonceCollision, NonceLedger, OsEntropy, SecretSeed, ShakeStream};
pub use privacy_spec::{
    protocol_digest, CapabilityError, CapabilitySet, LeakageItem, PrivacySpec, SecurityCapability,
    SimulatorStatus,
};
pub use zk_linear::{ZkLinearProof, ZkLinearProofError};
pub use zk_sumcheck::{
    zk_prove, zk_simulate, zk_verify, ZkSumcheckError, ZkSumcheckProof, ZkSumcheckStatement,
};
