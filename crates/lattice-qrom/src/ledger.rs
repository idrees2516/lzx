//! Worst-case Fiat-Shamir query accounting.

/// How rejection sampling amplifies a stage's query count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectionModel {
    /// One oracle query, no rejection (pure absorbs + single squeezes).
    Exact,
    /// `1 + max_retries` queries: rejection sampling with a hard cap
    /// (e.g. Lyubashevsky-style responses, challenge re-derivation).
    Capped { max_retries: u32 },
}

impl RejectionModel {
    /// The worst-case query multiplier.
    pub fn worst_case(&self) -> u64 {
        match self {
            RejectionModel::Exact => 1,
            RejectionModel::Capped { max_retries } => 1 + *max_retries as u64,
        }
    }
}

/// A per-stage query record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageQueries {
    /// Protocol stage label (must be domain-unique).
    pub stage: &'static str,
    /// Happy-path oracle queries the stage consumes.
    pub queries: u64,
    /// Rejection amplification.
    pub model: RejectionModel,
}

impl StageQueries {
    /// Worst-case query count for the stage.
    pub fn worst_case_queries(&self) -> u64 {
        self.queries.saturating_mul(self.model.worst_case())
    }
}

/// The composed protocol's query ledger.
#[derive(Clone, Debug)]
pub struct QueryLedger {
    budget: u64,
    stages: Vec<StageQueries>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    /// Duplicate stage label (ambiguous accounting).
    DuplicateStage(&'static str),
    /// The worst-case total exceeds the declared budget.
    BudgetExceeded { total: u64, budget: u64 },
}

impl core::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LedgerError::DuplicateStage(s) => write!(f, "duplicate stage label {s:?}"),
            LedgerError::BudgetExceeded { total, budget } => {
                write!(f, "worst-case queries {total} exceed budget {budget}")
            }
        }
    }
}

impl QueryLedger {
    /// A ledger with the protocol's total query budget.
    pub fn new(budget: u64) -> Self {
        QueryLedger {
            budget,
            stages: Vec::new(),
        }
    }

    /// Record a stage (labels must be unique).
    pub fn record(&mut self, stage: StageQueries) -> Result<(), LedgerError> {
        if self.stages.iter().any(|s| s.stage == stage.stage) {
            return Err(LedgerError::DuplicateStage(stage.stage));
        }
        self.stages.push(stage);
        Ok(())
    }

    /// Happy-path total.
    pub fn happy_path_total(&self) -> u64 {
        self.stages.iter().map(|s| s.queries).sum()
    }

    /// Worst-case total (with rejection amplification).
    pub fn worst_case_total(&self) -> u64 {
        self.stages.iter().map(|s| s.worst_case_queries()).sum()
    }

    /// The declared budget.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// The stages recorded so far.
    pub fn stages(&self) -> &[StageQueries] {
        &self.stages
    }

    /// Check the worst case against the budget.
    pub fn check(&self) -> Result<(), LedgerError> {
        let total = self.worst_case_total();
        if total > self.budget {
            return Err(LedgerError::BudgetExceeded {
                total,
                budget: self.budget,
            });
        }
        Ok(())
    }

    /// Safety margin: budget − worst case.
    pub fn margin(&self) -> u64 {
        self.budget.saturating_sub(self.worst_case_total())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_accounts_worst_case() {
        let mut ledger = QueryLedger::new(1 << 20);
        ledger
            .record(StageQueries {
                stage: "commit",
                queries: 2,
                model: RejectionModel::Exact,
            })
            .ok()
            .unwrap();
        ledger
            .record(StageQueries {
                stage: "zk-linear",
                queries: 1,
                model: RejectionModel::Capped { max_retries: 64 },
            })
            .ok()
            .unwrap();
        assert_eq!(ledger.happy_path_total(), 3);
        // 2 + 1·65 = 67 worst case.
        assert_eq!(ledger.worst_case_total(), 2 + 65);
        assert!(ledger.check().is_ok());
        assert_eq!(ledger.margin(), (1 << 20) - 67);
    }

    #[test]
    fn duplicate_stage_rejected() {
        let mut ledger = QueryLedger::new(100);
        ledger
            .record(StageQueries {
                stage: "x",
                queries: 1,
                model: RejectionModel::Exact,
            })
            .ok()
            .unwrap();
        assert!(matches!(
            ledger.record(StageQueries {
                stage: "x",
                queries: 1,
                model: RejectionModel::Exact,
            }),
            Err(LedgerError::DuplicateStage("x"))
        ));
    }

    #[test]
    fn budget_enforced_on_worst_case() {
        let mut ledger = QueryLedger::new(10);
        ledger
            .record(StageQueries {
                stage: "big",
                queries: 5,
                model: RejectionModel::Capped { max_retries: 8 },
            })
            .ok()
            .unwrap();
        // 5·9 = 45 > 10.
        assert!(matches!(
            ledger.check(),
            Err(LedgerError::BudgetExceeded { total: 45, budget: 10 })
        ));
        assert_eq!(ledger.margin(), 0);
    }
}
