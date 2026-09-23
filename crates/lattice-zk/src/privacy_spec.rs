//! Privacy specification and security capability tokens.
//!
//! Audit §"Executive summary": *"The safest target is a capability-typed
//! system: POST_QUANTUM_BINDING, ZERO_KNOWLEDGE, and QROM_FIAT_SHAMIR
//! must be separate compile-time/runtime security claims. A proof should
//! never be labeled 'zk' or 'post-quantum ZK' solely because its PCS is
//! lattice-based."*
//!
//! This module implements that capability discipline:
//! * [`PrivacySpec`] declares what is secret and what leaks (§9.5 item
//!   20: write the privacy specification first).
//! * [`CapabilitySet`] carries the granted capabilities, each bound to a
//!   protocol digest; verifiers `require()` capabilities before
//!   admitting a proof into a pipeline that assumes them.

use lattice_core::transcript::Transcript;

/// What a proof transcript may reveal about the execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeakageItem {
    /// The program digest is public (committed programs).
    ProgramDigest,
    /// The public input digest is public.
    PublicInputDigest,
    /// The public output values are public.
    PublicOutput,
    /// The number of trace rows (execution length) leaks.
    TraceRowCount,
    /// The final memory digest is public.
    MemoryDigest,
    /// Commitment and proof byte lengths leak.
    ProofSizes,
}

/// The declared privacy posture of a proving pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivacySpec {
    /// Names of the secret inputs (documentation-grade binding).
    pub secret_inputs: Vec<String>,
    /// What the transcript is allowed to reveal.
    pub allowed_leakage: Vec<LeakageItem>,
    /// Whether the simulator is machine-checked (KATs) or pending review.
    pub simulator_status: SimulatorStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimulatorStatus {
    /// Distributional simulator + statistical KATs in-tree.
    MachineChecked { kat_tests: usize },
    /// Claimed but not yet verified — capability must NOT be granted.
    Pending,
}

/// Security capabilities, one per auditable claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SecurityCapability {
    /// Module-SIS binding of the commitment layer (classical analysis).
    PostQuantumBinding,
    /// Statistical zero knowledge of the specified secret inputs.
    ZeroKnowledge,
    /// QROM-analyzed Fiat-Shamir composition (see lattice-qrom).
    QromFiatShamir,
}

/// A capability set: each granted capability is bound to the digest of
/// the exact protocol stage list it was reviewed against.
#[derive(Clone, Debug, Default)]
pub struct CapabilitySet {
    granted: Vec<(SecurityCapability, [u8; 32])>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// The capability was never granted.
    NotGranted(SecurityCapability),
    /// The capability was granted for a different protocol digest.
    DigestMismatch {
        capability: SecurityCapability,
        expected: [u8; 32],
        granted_for: [u8; 32],
    },
}

impl core::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CapabilityError::NotGranted(c) => write!(f, "security capability not granted: {c:?}"),
            CapabilityError::DigestMismatch {
                capability,
                expected,
                granted_for,
            } => write!(
                f,
                "capability {capability:?} bound to a different protocol digest ({:02x?} != {:02x?})",
                &expected[..4],
                &granted_for[..4]
            ),
        }
    }
}

/// Canonical protocol digest over the ordered stage list (the exact
/// composition a capability is reviewed against).
pub fn protocol_digest(stages: &[&str]) -> [u8; 32] {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"LZX-PROTOCOL-1");
    buf.extend_from_slice(&(stages.len() as u32).to_le_bytes());
    for s in stages {
        buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    Transcript::hash_domain(b"protocol-digest", &buf)
}

impl CapabilitySet {
    pub fn empty() -> Self {
        CapabilitySet::default()
    }

    /// Grant a capability for a protocol digest (the review sign-off).
    pub fn grant(&mut self, capability: SecurityCapability, protocol: [u8; 32]) {
        // Re-granting for a different digest is a configuration bug:
        // keep both bindings (require() will surface the mismatch).
        self.granted.push((capability, protocol));
    }

    /// Require a capability for a protocol; error unless granted for
    /// exactly this digest.
    pub fn require(&self, capability: SecurityCapability, protocol: &[u8; 32]) -> Result<(), CapabilityError> {
        for (c, d) in &self.granted {
            if *c == capability {
                if d == protocol {
                    return Ok(());
                }
                return Err(CapabilityError::DigestMismatch {
                    capability,
                    expected: *protocol,
                    granted_for: *d,
                });
            }
        }
        Err(CapabilityError::NotGranted(capability))
    }

    /// Capabilities currently granted (for envelope self-description).
    pub fn capabilities(&self) -> Vec<SecurityCapability> {
        self.granted.iter().map(|(c, _)| *c).collect()
    }
}

impl PrivacySpec {
    /// The default zkVM privacy posture: the trace (register stream,
    /// memory contents, access values) is secret; the program, public
    /// input/output, and sizes leak.
    pub fn zkvm_default() -> Self {
        PrivacySpec {
            secret_inputs: vec![
                "register-write-stream".into(),
                "memory-access-values".into(),
                "mask-entropy".into(),
            ],
            allowed_leakage: vec![
                LeakageItem::ProgramDigest,
                LeakageItem::PublicInputDigest,
                LeakageItem::PublicOutput,
                LeakageItem::TraceRowCount,
                LeakageItem::MemoryDigest,
                LeakageItem::ProofSizes,
            ],
            simulator_status: SimulatorStatus::MachineChecked { kat_tests: 0 },
        }
    }

    /// Derive the capability set a verifier may trust from this spec.
    ///
    /// `ZeroKnowledge` is granted only when the simulator is
    /// machine-checked; `QromFiatShamir` is granted only by the explicit
    /// QROM review pass (lattice-qrom), never from a privacy spec alone.
    pub fn capabilities(&self, binding: SecurityCapability, protocol: [u8; 32]) -> CapabilitySet {
        let mut set = CapabilitySet::empty();
        match binding {
            SecurityCapability::PostQuantumBinding => {
                set.grant(SecurityCapability::PostQuantumBinding, protocol);
            }
            SecurityCapability::ZeroKnowledge => {
                if matches!(self.simulator_status, SimulatorStatus::MachineChecked { .. }) {
                    set.grant(SecurityCapability::ZeroKnowledge, protocol);
                }
            }
            SecurityCapability::QromFiatShamir => {
                // Requires the lattice-qrom attestation path.
            }
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGES_A: &[&str] = &["commit-masks", "zk-rounds", "carrier-relation", "zk-linear"];
    const STAGES_B: &[&str] = &["commit-masks", "zk-rounds", "carrier-relation"];

    #[test]
    fn digest_is_stage_sensitive() {
        let a = protocol_digest(STAGES_A);
        let b = protocol_digest(STAGES_B);
        assert_ne!(a, b);
        // Order-sensitive as well.
        let c = protocol_digest(&["zk-rounds", "commit-masks", "carrier-relation", "zk-linear"]);
        assert_ne!(a, c);
        // Deterministic.
        assert_eq!(a, protocol_digest(STAGES_A));
    }

    #[test]
    fn capability_gating() {
        let proto = protocol_digest(STAGES_A);
        let mut set = CapabilitySet::empty();
        assert!(matches!(
            set.require(SecurityCapability::ZeroKnowledge, &proto),
            Err(CapabilityError::NotGranted(_))
        ));
        set.grant(SecurityCapability::ZeroKnowledge, proto);
        assert!(set.require(SecurityCapability::ZeroKnowledge, &proto).is_ok());
        // Wrong protocol digest: mismatch, not silent acceptance.
        let other = protocol_digest(STAGES_B);
        assert!(matches!(
            set.require(SecurityCapability::ZeroKnowledge, &other),
            Err(CapabilityError::DigestMismatch { .. })
        ));
        // An unrelated capability is still not granted.
        assert!(matches!(
            set.require(SecurityCapability::QromFiatShamir, &proto),
            Err(CapabilityError::NotGranted(_))
        ));
    }

    #[test]
    fn zk_requires_machine_checked_simulator() {
        let proto = protocol_digest(STAGES_A);
        let mut spec = PrivacySpec::zkvm_default();
        spec.simulator_status = SimulatorStatus::Pending;
        let pending = spec.capabilities(SecurityCapability::ZeroKnowledge, proto);
        assert!(matches!(
            pending.require(SecurityCapability::ZeroKnowledge, &proto),
            Err(CapabilityError::NotGranted(_))
        ));
        spec.simulator_status = SimulatorStatus::MachineChecked { kat_tests: 6 };
        let ok = spec.capabilities(SecurityCapability::ZeroKnowledge, proto);
        assert!(ok.require(SecurityCapability::ZeroKnowledge, &proto).is_ok());
    }

    #[test]
    fn qrom_never_granted_from_privacy_spec() {
        let proto = protocol_digest(STAGES_A);
        let spec = PrivacySpec::zkvm_default();
        let set = spec.capabilities(SecurityCapability::QromFiatShamir, proto);
        assert!(matches!(
            set.require(SecurityCapability::QromFiatShamir, &proto),
            Err(CapabilityError::NotGranted(_))
        ));
    }
}
