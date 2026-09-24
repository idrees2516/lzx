//! Symbolic norm budgets with hard wraparound gates (Wave 6 substrate,
//! `NEXT_STEPS.md` §2.4).
//!
//! The folding modules track witness norms informationally today — nothing
//! refuses a fold that pushes the balanced representative past `q/2`. When
//! that happens the folded witness *wraps* mod q: the SIS binding argument
//! needs the short **integer** vector, not its residue, so wraparound
//! silently destroys binding while every algebraic identity still checks.
//!
//! This module provides the shared accounting type:
//!
//! * `β` — current worst-case ℓ∞ bound over all accumulator witnesses,
//! * `γ_max` — largest certified challenge operator-norm Γ_C consumed
//!   (bookkeeping for auditors),
//! * `folds` — folds since the last refresh (Cyclo-style counting),
//! * [`NormBudget::fold`] — the rigorous growth law
//!   `β' = β + Γ_C · ⌈√N⌉ · β_in`, **hard-gated** against
//!   `min(q/2, β*)`: the fold refuses to happen rather than wrap,
//! * [`NormBudget::fold_scalar`] — the scalar-challenge growth law
//!   `β' = β + |r| · β_in` used by the pre-Wave-6 fold paths, same gate.
//!
//! All arithmetic is u128-safe (β can legitimately exceed u32 range: the
//! papers' multi-fold growth is `γ^k · β` with γ ≈ 8-15).

/// Symbolic norm budget for a folding accumulator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormBudget {
    /// Worst-case ℓ∞ bound of the accumulator witness (integer, not mod q).
    pub beta: u64,
    /// Folds since the last refresh.
    pub folds: u64,
    /// Largest certified Γ_C consumed (operator-norm growth factor).
    pub gamma_max: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormBudgetError {
    /// The fold would push β past the wraparound gate — refuse to fold.
    /// Carries (beta_after, cap) so callers surface the exact violation.
    Wraparound { beta_after: u128, cap: u64 },
}

impl NormBudgetError {
    /// True iff the error is the hard-gate violation.
    pub fn is_wraparound(&self) -> bool {
        matches!(self, NormBudgetError::Wraparound { .. })
    }
}

impl NormBudget {
    /// Fresh accumulator budget from an input witness bound.
    pub fn fresh(beta: u64) -> Self {
        NormBudget {
            beta,
            folds: 0,
            gamma_max: 0,
        }
    }

    /// The hard gate: β must stay below `min(q/2, β*)` — `q/2` protects the
    /// balanced representative (wraparound), `β*` protects the SIS norm
    /// bound of the opening proof.
    pub fn gate(beta: u128, q_half: u64, beta_star: u64) -> Result<u64, NormBudgetError> {
        let cap = q_half.min(beta_star) as u128;
        if beta >= cap {
            return Err(NormBudgetError::Wraparound { beta_after: beta, cap: cap as u64 });
        }
        Ok(beta as u64)
    }

    /// Fold under a **ring-element** challenge `c` with certified operator
    /// norm Γ_C (from [`crate::short_challenge`]):
    /// `w' = w_acc + c·w_in` gives
    /// `‖w'‖∞ ≤ β + Γ_C · ⌈√N⌉ · β_in`
    /// (the √N accounts for `‖w_in‖₂ ≤ √N · ‖w_in‖∞`).
    #[must_use = "the folded budget must be stored in the new accumulator"]
    pub fn fold(
        &self,
        gamma_c: u64,
        sqrt_n: u64,
        beta_input: u64,
        q_half: u64,
        beta_star: u64,
    ) -> Result<NormBudget, NormBudgetError> {
        let growth = (gamma_c as u128)
            .saturating_mul(sqrt_n as u128)
            .saturating_mul(beta_input as u128);
        let beta_after = (self.beta as u128).saturating_add(growth);
        let beta = Self::gate(beta_after, q_half, beta_star)?;
        Ok(NormBudget {
            beta,
            folds: self.folds + 1,
            gamma_max: self.gamma_max.max(gamma_c),
        })
    }

    /// Fold under a **scalar** challenge `r` (the current fold paths):
    /// `w' = w_acc + r·w_in` gives `β' = β + |r|·β_in` exactly.
    #[must_use = "the folded budget must be stored in the new accumulator"]
    pub fn fold_scalar(
        &self,
        r_abs: u64,
        beta_input: u64,
        q_half: u64,
        beta_star: u64,
    ) -> Result<NormBudget, NormBudgetError> {
        let beta_after = (self.beta as u128)
            .saturating_mul(1)
            .saturating_add((r_abs as u128).saturating_mul(beta_input as u128));
        let beta = Self::gate(beta_after, q_half, beta_star)?;
        Ok(NormBudget {
            beta,
            folds: self.folds + 1,
            gamma_max: self.gamma_max.max(r_abs),
        })
    }

    /// Refresh (Cyclo extension commitment): the chunked representation
    /// restarts the budget at the refreshed bound.
    #[must_use = "the refreshed budget must be stored in the new accumulator"]
    pub fn refresh(&self, refreshed_beta: u64) -> NormBudget {
        NormBudget {
            beta: refreshed_beta,
            folds: 0,
            gamma_max: 0,
        }
    }

    /// Current β bound.
    pub fn beta(&self) -> u64 {
        self.beta
    }

    /// Folds since refresh.
    pub fn folds(&self) -> u64 {
        self.folds
    }
}

/// `⌈√n⌉` for the fold growth law (exact integer ceiling sqrt).
pub fn ceil_sqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let r = (n as f64).sqrt() as u64;
    let mut r = r.max(1);
    while r.saturating_mul(r) < n {
        r += 1;
    }
    while r > 1 && (r - 1).saturating_mul(r - 1) >= n {
        r -= 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q32_HALF: u64 = 3221225473 / 2;

    #[test]
    fn scalar_fold_growth_exact() {
        let b = NormBudget::fresh(64);
        let f1 = b.fold_scalar(1 << 11, 256, Q32_HALF, u64::MAX).ok().unwrap();
        assert_eq!(f1.beta(), 64 + (1 << 11) * 256);
        assert_eq!(f1.folds(), 1);
        // The exact integer law, no hidden slack.
        let f2 = f1.fold_scalar(3, 5, Q32_HALF, u64::MAX).ok().unwrap();
        assert_eq!(f2.beta(), 64 + (1 << 11) * 256 + 15);
        assert_eq!(f2.folds(), 2);
        assert_eq!(f2.gamma_max, 1 << 11);
    }

    #[test]
    fn ring_fold_uses_gamma_and_sqrt_n() {
        // Γ_C = 80 (PikkuFold certified), N = 256 → ⌈√256⌉ = 16.
        let b = NormBudget::fresh(100);
        let f = b.fold(80, 16, 1000, Q32_HALF, u64::MAX).ok().unwrap();
        assert_eq!(f.beta(), 100 + 80 * 16 * 1000);
        assert_eq!(f.gamma_max, 80);
    }

    #[test]
    fn hard_gate_refuses_wraparound() {
        // Cyclo-class violation: β_in = 2^20, |r| = 2^11, q/2 ≈ 2^30.58.
        // One fold from β = 2^20 exceeds q/2 → refuse.
        let b = NormBudget::fresh(1 << 20);
        let err = b.fold_scalar(1 << 11, 1 << 20, Q32_HALF, u64::MAX).err().unwrap();
        assert!(err.is_wraparound());
        match err {
            NormBudgetError::Wraparound { beta_after, cap } => {
                assert_eq!(beta_after, (1 << 20) + (1 << 11) * (1 << 20));
                assert_eq!(cap, Q32_HALF);
            }
        }
        // Just at/above the gate fails; below passes.
        let tight = NormBudget::fresh(Q32_HALF - 3);
        assert!(tight.fold_scalar(1, 5, Q32_HALF, u64::MAX).is_err());
        let ok = NormBudget::fresh(Q32_HALF - 16);
        assert!(ok.fold_scalar(1, 15, Q32_HALF, u64::MAX).is_ok());
    }

    #[test]
    fn beta_star_limits_below_q_half() {
        let b = NormBudget::fresh(100);
        // β* = 200 dominates q/2 ≈ 1.6·10^9: β' = 100 + 1·101 = 201 ≥ 200
        // → refused; 100 + 1·99 = 199 < 200 → allowed.
        assert!(b.fold_scalar(1, 101, Q32_HALF, 200).is_err());
        assert!(b.fold_scalar(1, 99, Q32_HALF, 200).is_ok());
    }

    #[test]
    fn refresh_resets() {
        let b = NormBudget::fresh(1 << 20).fold_scalar(7, 3, Q32_HALF, u64::MAX).ok().unwrap();
        let r = b.refresh(128);
        assert_eq!(r.beta(), 128);
        assert_eq!(r.folds(), 0);
        assert_eq!(r.gamma_max, 0);
    }

    #[test]
    fn saturating_growth_never_panics() {
        let b = NormBudget::fresh(u64::MAX - 5);
        // Saturating arithmetic then the gate catches it.
        assert!(b.fold(u64::MAX, ceil_sqrt(1024), u64::MAX, Q32_HALF, u64::MAX).is_err());
    }

    #[test]
    fn ceil_sqrt_boundaries() {
        assert_eq!(ceil_sqrt(0), 0);
        assert_eq!(ceil_sqrt(1), 1);
        assert_eq!(ceil_sqrt(3), 2);
        assert_eq!(ceil_sqrt(4), 2);
        assert_eq!(ceil_sqrt(1024), 32);
        assert_eq!(ceil_sqrt(1025), 33);
        assert_eq!(ceil_sqrt(u64::MAX), 1u64 << 32);
    }
}
