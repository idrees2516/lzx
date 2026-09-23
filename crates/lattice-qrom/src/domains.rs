//! Workspace-wide domain-separation registry.
//!
//! Every production `Transcript::new*` label in the LZX workspace is
//! listed in [`PRODUCTION_DOMAINS`]. The uniqueness test below fails
//! the build if a new domain collides with an existing one (which
//! would make two protocols share a Fiat-Shamir derivation domain).
//! The runtime [`DomainRegistry`] provides the same check for dynamic
//! registrations (e.g. per-instance sub-protocol labels).

/// All production transcript domain labels (harvested from the
/// workspace source; see the `all_production_domains_unique` test).
pub const PRODUCTION_DOMAINS: &[&str] = &[
    // Commitment / proof-system cores.
    "lzx-linear-proof",
    "lzx-zk-linear",
    "lzx-zk-sumcheck",
    // PCS layer.
    "lzx-akita-group",
    "lzx-hyperwolf",
    // Folding papers.
    "lzx-protogalattice",
    "lzx-latticefold-plus",
    "lzx-cyclo",
    "lzx-pikkufold",
    "lzx-symphony",
    "lzx-superneo",
    // Succinct-argument papers.
    "lzx-rokoko",
    "lzx-salsa-lde",
    "lzx-salsa-zk",
    // VM layer.
    "lzx-zkvm",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainCollision {
    pub domain: String,
}

impl core::fmt::Display for DomainCollision {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "domain label collision: {:?}", self.domain)
    }
}

/// Runtime registry for dynamically chosen domain labels.
#[derive(Default)]
pub struct DomainRegistry {
    seen: std::collections::HashSet<String>,
}

impl DomainRegistry {
    pub fn new() -> Self {
        DomainRegistry::default()
    }

    /// Register a domain; `Err` on collision.
    pub fn register(&mut self, domain: &str) -> Result<(), DomainCollision> {
        if self.seen.insert(domain.to_string()) {
            Ok(())
        } else {
            Err(DomainCollision {
                domain: domain.to_string(),
            })
        }
    }

    /// Whether a domain is already registered.
    pub fn contains(&self, domain: &str) -> bool {
        self.seen.contains(domain)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Check a label list for duplicates (used by the static test and by
/// attestation verification).
pub fn find_duplicates(labels: &[&str]) -> Option<String> {
    let mut seen = std::collections::HashSet::new();
    for l in labels {
        if !seen.insert(*l) {
            return Some((*l).to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_production_domains_unique() {
        // A duplicate here means two protocols share a Fiat-Shamir
        // domain — cross-protocol transcript interference.
        assert!(
            find_duplicates(PRODUCTION_DOMAINS).is_none(),
            "duplicate production domain: {:?}",
            find_duplicates(PRODUCTION_DOMAINS)
        );
        // The registry must cover the core protocol families.
        for expected in [
            "lzx-zkvm",
            "lzx-zk-linear",
            "lzx-zk-sumcheck",
            "lzx-linear-proof",
        ] {
            assert!(PRODUCTION_DOMAINS.contains(&expected), "missing {expected}");
        }
    }

    #[test]
    fn runtime_registry_detects_collisions() {
        let mut reg = DomainRegistry::new();
        assert!(reg.register("instance-1").is_ok());
        assert!(reg.register("instance-2").is_ok());
        assert!(matches!(
            reg.register("instance-1"),
            Err(DomainCollision { .. })
        ));
        assert_eq!(reg.len(), 2);
        assert!(reg.contains("instance-2"));
    }

    #[test]
    fn find_duplicates_works() {
        assert!(find_duplicates(&["a", "b", "c"]).is_none());
        assert_eq!(find_duplicates(&["a", "b", "a"]), Some("a".into()));
    }
}
