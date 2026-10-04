//! QROM attestations: the prover's declared query accounting, verified
//! by the verifier before any expensive replay.

use crate::domains::find_duplicates;
use crate::ledger::{QueryLedger, RejectionModel, StageQueries};
use lattice_zk::privacy_spec::protocol_digest;

/// One stage's declared accounting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageRecord {
    pub label: String,
    /// Happy-path oracle queries.
    pub queries: u64,
    /// Worst-case rejection retries declared for the stage.
    pub max_retries: u32,
    /// Free-form note (e.g. "sumcheck over 8 vars, degree 2").
    pub note: String,
}

impl StageRecord {
    /// Worst-case queries for the stage.
    pub fn worst_case_queries(&self) -> u64 {
        let model = RejectionModel::Capped {
            max_retries: self.max_retries,
        };
        self.queries.saturating_mul(model.worst_case())
    }
}

/// The attestation artifact carried alongside a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QromAttestation {
    /// Digest of the ordered stage-label list (binds the composition).
    pub protocol_digest: [u8; 32],
    /// The ordered stage records.
    pub stages: Vec<StageRecord>,
    /// The declared total query budget for the composition.
    pub budget: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationError {
    /// The stage list does not match the expected protocol digest.
    DigestMismatch,
    /// Duplicate stage labels.
    DuplicateStage(String),
    /// The worst-case total exceeds the budget.
    BudgetExceeded { total: u64, budget: u64 },
    /// The declared budget exceeds the verifier's configured maximum.
    BudgetTooLarge { declared: u64, max: u64 },
    /// A stage declares zero queries (vacuous accounting).
    VacuousStage(String),
}

impl core::fmt::Display for AttestationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AttestationError::DigestMismatch => write!(f, "attestation digest mismatch"),
            AttestationError::DuplicateStage(s) => write!(f, "duplicate stage {s:?}"),
            AttestationError::BudgetExceeded { total, budget } => {
                write!(f, "worst-case {total} > budget {budget}")
            }
            AttestationError::BudgetTooLarge { declared, max } => {
                write!(f, "declared budget {declared} > verifier max {max}")
            }
            AttestationError::VacuousStage(s) => write!(f, "stage {s:?} declares zero queries"),
        }
    }
}

impl QromAttestation {
    /// Build an attestation over a stage list; computes the digest and
    /// checks the budget against `verifier_max`.
    pub fn build(
        stages: Vec<StageRecord>,
        budget: u64,
        verifier_max: u64,
    ) -> Result<Self, AttestationError> {
        if budget > verifier_max {
            return Err(AttestationError::BudgetTooLarge {
                declared: budget,
                max: verifier_max,
            });
        }
        let labels: Vec<&str> = stages.iter().map(|s| s.label.as_str()).collect();
        if let Some(dup) = find_duplicates(&labels) {
            return Err(AttestationError::DuplicateStage(dup));
        }
        for s in &stages {
            if s.queries == 0 {
                return Err(AttestationError::VacuousStage(s.label.clone()));
            }
        }
        let total: u64 = stages.iter().map(|s| s.worst_case_queries()).sum();
        if total > budget {
            return Err(AttestationError::BudgetExceeded { total, budget });
        }
        let digest = protocol_digest(&labels);
        Ok(QromAttestation {
            protocol_digest: digest,
            stages,
            budget,
        })
    }

    /// The worst-case total.
    pub fn worst_case_total(&self) -> u64 {
        self.stages.iter().map(|s| s.worst_case_queries()).sum()
    }

    /// Verify against the expected protocol digest and a verifier-side
    /// maximum budget. Called **before** transcript replay (audit §10.2:
    /// "Validate proof topology and schedule identity before transcript
    /// replay or expensive arithmetic").
    pub fn verify(
        &self,
        expected_digest: &[u8; 32],
        verifier_max: u64,
    ) -> Result<(), AttestationError> {
        if &self.protocol_digest != expected_digest {
            return Err(AttestationError::DigestMismatch);
        }
        if self.budget > verifier_max {
            return Err(AttestationError::BudgetTooLarge {
                declared: self.budget,
                max: verifier_max,
            });
        }
        let labels: Vec<&str> = self.stages.iter().map(|s| s.label.as_str()).collect();
        if let Some(dup) = find_duplicates(&labels) {
            return Err(AttestationError::DuplicateStage(dup));
        }
        let total = self.worst_case_total();
        if total > self.budget {
            return Err(AttestationError::BudgetExceeded {
                total,
                budget: self.budget,
            });
        }
        Ok(())
    }

    /// Convert to a ledger (for composition review).
    pub fn to_ledger(&self) -> QueryLedger {
        let mut ledger = QueryLedger::new(self.budget);
        for s in &self.stages {
            let label: &'static str = Box::leak(s.label.clone().into_boxed_str());
            let _ = ledger.record(StageQueries {
                stage: label,
                queries: s.queries,
                model: RejectionModel::Capped {
                    max_retries: s.max_retries,
                },
            });
        }
        ledger
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stages() -> Vec<StageRecord> {
        vec![
            StageRecord {
                label: "commit-masks".into(),
                queries: 2,
                max_retries: 0,
                note: "absorb + challenge".into(),
            },
            StageRecord {
                label: "zk-rounds".into(),
                queries: 8,
                max_retries: 0,
                note: "one challenge per round".into(),
            },
            StageRecord {
                label: "zk-linear".into(),
                queries: 1,
                max_retries: 64,
                note: "ABDLOP rejection sampling".into(),
            },
        ]
    }

    #[test]
    fn attestation_build_and_verify() {
        let stages = sample_stages();
        let labels: Vec<&str> = stages.iter().map(|s| s.label.as_str()).collect();
        let digest = protocol_digest(&labels);
        let att = QromAttestation::build(stages, 1 << 16, 1 << 20)
            .ok()
            .unwrap();
        // Worst case: 2 + 8 + 1·65 = 75.
        assert_eq!(att.worst_case_total(), 75);
        assert!(att.verify(&digest, 1 << 20).is_ok());
        // The ledger view matches.
        let ledger = att.to_ledger();
        assert_eq!(ledger.worst_case_total(), 75);
        assert!(ledger.check().is_ok());
    }

    #[test]
    fn digest_mismatch_rejected() {
        let att = QromAttestation::build(sample_stages(), 1 << 16, 1 << 20)
            .ok()
            .unwrap();
        let wrong = protocol_digest(&["other", "stage", "list"]);
        assert!(matches!(
            att.verify(&wrong, 1 << 20),
            Err(AttestationError::DigestMismatch)
        ));
    }

    #[test]
    fn budget_violations_rejected() {
        // Worst case 75 > budget 10.
        assert!(matches!(
            QromAttestation::build(sample_stages(), 10, 1 << 20),
            Err(AttestationError::BudgetExceeded {
                total: 75,
                budget: 10
            })
        ));
        // Declared budget above the verifier maximum.
        assert!(matches!(
            QromAttestation::build(sample_stages(), 1 << 30, 1 << 20),
            Err(AttestationError::BudgetTooLarge { .. })
        ));
    }

    #[test]
    fn duplicates_and_vacuous_stages_rejected() {
        let mut stages = sample_stages();
        stages.push(stages[0].clone());
        assert!(matches!(
            QromAttestation::build(stages, 1 << 16, 1 << 20),
            Err(AttestationError::DuplicateStage(_))
        ));
        let vacuous = vec![StageRecord {
            label: "nothing".into(),
            queries: 0,
            max_retries: 0,
            note: String::new(),
        }];
        assert!(matches!(
            QromAttestation::build(vacuous, 16, 1 << 20),
            Err(AttestationError::VacuousStage(_))
        ));
    }
}
