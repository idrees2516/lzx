//! Akita schedule catalog and security profiles: the trusted, digest-
//! selected parameter artifacts (planner output is offline; the verifier
//! accepts only registered digests).

use lattice_core::transcript::Transcript;

/// A schedule entry: one named stage with its geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleEntry {
    pub stage: String,
    /// Round structure: per-stage (degree, rounds) pairs.
    pub rounds: Vec<(usize, usize)>,
}

/// A schedule catalog: digest-bound, versioned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleCatalog {
    pub version: u32,
    pub entries: Vec<ScheduleEntry>,
    /// Catalog digest (transcript-bound).
    pub digest: [u8; 32],
}

impl ScheduleCatalog {
    /// Build a catalog with its digest.
    pub fn new(version: u32, entries: Vec<ScheduleEntry>) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&version.to_le_bytes());
        for e in &entries {
            bytes.extend_from_slice(&(e.stage.len() as u32).to_le_bytes());
            bytes.extend_from_slice(e.stage.as_bytes());
            for (deg, rounds) in &e.rounds {
                bytes.extend_from_slice(&(*deg as u32).to_le_bytes());
                bytes.extend_from_slice(&(*rounds as u32).to_le_bytes());
            }
        }
        let digest = Transcript::hash_domain(b"akita-schedule", &bytes);
        ScheduleCatalog {
            version,
            entries,
            digest,
        }
    }

    /// Look up an entry by stage name.
    pub fn entry(&self, stage: &str) -> Option<&ScheduleEntry> {
        self.entries.iter().find(|e| e.stage == stage)
    }
}

/// Security profile: the Module-SIS evidence bundle structure (the audit
/// report's parameter-discipline checklist — machine-readable, digest-
/// bound; the verifier only accepts registered profiles).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecurityProfile {
    pub profile_name: String,
    pub ring_log_n: u32,
    pub modulus_q: u32,
    pub module_k: usize,
    pub module_m: usize,
    /// Challenge distribution tag (e.g. "sparse-ternary-w512").
    pub challenge_tag: String,
    /// Norm bound on responses.
    pub norm_bound: u32,
    /// Claimed classical security bits (estimator output; review-gated).
    pub classical_bits: u32,
    /// Claimed quantum (QROM) security bits — requires the QROM evidence
    /// review per the audit checklist.
    pub quantum_bits: u32,
    /// External review digest (zeros until reviewed).
    pub review_digest: [u8; 32],
}

impl SecurityProfile {
    /// Profile digest for registration.
    pub fn digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.profile_name.as_bytes());
        bytes.extend_from_slice(&self.ring_log_n.to_le_bytes());
        bytes.extend_from_slice(&self.modulus_q.to_le_bytes());
        bytes.extend_from_slice(&(self.module_k as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.module_m as u32).to_le_bytes());
        bytes.extend_from_slice(self.challenge_tag.as_bytes());
        bytes.extend_from_slice(&self.norm_bound.to_le_bytes());
        bytes.extend_from_slice(&self.classical_bits.to_le_bytes());
        bytes.extend_from_slice(&self.quantum_bits.to_le_bytes());
        bytes.extend_from_slice(&self.review_digest);
        Transcript::hash_domain(b"akita-security-profile", &bytes)
    }

    /// Whether the profile has completed external review.
    pub fn is_reviewed(&self) -> bool {
        self.review_digest.iter().any(|b| *b != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_digest_deterministic_and_sensitive() {
        let c1 = ScheduleCatalog::new(
            1,
            vec![ScheduleEntry {
                stage: "commit".into(),
                rounds: vec![(2, 10)],
            }],
        );
        let c2 = ScheduleCatalog::new(
            1,
            vec![ScheduleEntry {
                stage: "commit".into(),
                rounds: vec![(2, 10)],
            }],
        );
        assert_eq!(c1.digest, c2.digest);
        let c3 = ScheduleCatalog::new(
            1,
            vec![ScheduleEntry {
                stage: "commit".into(),
                rounds: vec![(3, 10)], // different geometry
            }],
        );
        assert_ne!(c1.digest, c3.digest);
        assert!(c1.entry("commit").is_some());
        assert!(c1.entry("open").is_none());
    }

    #[test]
    fn security_profile_review_gate() {
        let mut p = SecurityProfile {
            profile_name: "akita-128-basic".into(),
            ring_log_n: 10,
            modulus_q: 3221225473,
            module_k: 4,
            module_m: 6,
            challenge_tag: "sparse-ternary-w512".into(),
            norm_bound: 1 << 22,
            classical_bits: 128,
            quantum_bits: 0,
            review_digest: [0u8; 32],
        };
        assert!(!p.is_reviewed());
        p.review_digest = [7u8; 32];
        assert!(p.is_reviewed());
        // Digest distinguishes profiles.
        let q = SecurityProfile {
            profile_name: "akita-192-basic".into(),
            ..p.clone()
        };
        assert_ne!(p.digest(), q.digest());
    }
}
