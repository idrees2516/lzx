//! Statistical machinery of ePrint 2026/2146: the moment bounds (Lemma 3),
//! the modular ℓ∞ overflow bound (Lemma 4 / Theorem 2), the Cantelli
//! ℓ₂ concentration (Lemma 5 / Theorem 3), and the concrete parameter
//! search of §7.2 (Table 4).
//!
//! All formulas are closed-form functions of (µ, c, k, d, θ); the unit
//! tests cross-validate them against Monte-Carlo simulation at small
//! scale, and the parameter search replicates the paper's constraint
//! system:
//! * overflow: `(1/2 + 2^{µ−1}/2^{c+1})^k ≤ 2^{−p}`;
//! * Cantelli: `max(1 − (η₂−θ)²/(η₄−2η₂θ+θ²), 1/2 + 2^{µ−1}/2^{c+1})^k ≤ 2^{−p}`;
//! * no-overflow input regime: `B < q / (2^{µ+1}·√(θ·c^{µ−1}·m))`;
//! * slack `B̃/B = √(k·(c/2)^{µ−1}/θ)` under a target maximum;
//!
//! minimising the TT-format representation `k·d·(2c + (µ−2)c²)`.

/// Second moment factor η₂ (Lemma 3): `E[y²] = η₂·||x||²` for the
/// one-row projection.
///
/// Implemented as `c^{µ−1}/2^µ` — the value the paper's own A.2 derivation
/// produces (`E[pJ pJᵀ] = c^{µ−1}·σ^{2µ}·I` with `σ² = Var(D_ghl) = 1/2`).
/// Eq. (11)'s printed `(c/2)^{µ−1}` is a factor of 2 larger; the
/// Monte-Carlo test in tests/ttrp.rs pins the corrected value (deviation
/// documented in docs/papers/implemented/ttrp.md).
pub fn eta2(mu: usize, c: usize) -> f64 {
    (c as f64).powi(mu as i32 - 1) / 2f64.powi(mu as i32)
}

/// Fourth moment factor η₄ = (3/4)·(c(c+2)/4)^{µ−1} (Lemma 3, Eq. (12)):
/// `E[y⁴] ≤ η₄·||x||⁴`.
pub fn eta4(mu: usize, c: usize) -> f64 {
    (3.0 / 4.0) * (c as f64 * (c + 2) as f64 / 4.0).powi(mu as i32 - 1)
}

/// Per-row modular overflow failure bound of Lemma 4 / Theorem 2:
/// `Pr[|f_TT(x) mod q| < B/2^µ] ≤ 1/2 + 2^{µ−1}/2^{c+1}`.
pub fn overflow_row_failure(mu: usize, c: usize) -> f64 {
    0.5 + (1u64 << (mu - 1)) as f64 / (1u64 << (c + 1)) as f64
}

/// k-row ℓ∞ failure probability (Theorem 2).
pub fn overflow_k_failure(mu: usize, c: usize, k: usize) -> f64 {
    overflow_row_failure(mu, c).powi(k as i32)
}

/// Lemma 5 single-row success bound: for θ ∈ (0, η₂),
/// `Pr[y² > θ||x||²] ≥ (η₂−θ)²/(η₄−2η₂θ+θ²)`.
/// Returns None when θ ≥ η₂ (out of the lemma's regime).
pub fn cantelli_success(mu: usize, c: usize, theta: f64) -> Option<f64> {
    let e2 = eta2(mu, c);
    let e4 = eta4(mu, c);
    if theta <= 0.0 || theta >= e2 {
        return None;
    }
    let num = (e2 - theta).powi(2);
    let den = e4 - 2.0 * e2 * theta + theta * theta;
    if den <= 0.0 {
        return None;
    }
    Some(num / den)
}

/// Theorem 3 overall failure probability for the k-row projection:
/// `p = max(1 − success, overflow_row)^k`, the knowledge-error core of
/// Π₀ (Theorem 4: δ₀ = p + q^{−k'}).
pub fn theorem3_failure(mu: usize, c: usize, k: usize, theta: f64) -> Option<f64> {
    let success = cantelli_success(mu, c, theta)?;
    Some(
        (1.0 - success)
            .max(overflow_row_failure(mu, c))
            .powi(k as i32),
    )
}

/// The completeness bound B̂ = √(k·η₂)·B (Figure 1 / Theorem 4, with the
/// corrected η₂): the honest prover passes `||y0||₂ ≤ B̂` with probability
/// ≥ 1/2 by Markov (E[||y0||²] = k·η₂·B² = B̂²).
pub fn completeness_bound(mu: usize, c: usize, k: usize, b: f64) -> f64 {
    (k as f64 * eta2(mu, c)).sqrt() * b
}

/// The extraction slack B̃/B = √(k·η₂/θ) (Theorem 4, corrected η₂).
pub fn slack(mu: usize, c: usize, k: usize, theta: f64) -> f64 {
    (k as f64 * eta2(mu, c) / theta).sqrt()
}

/// The no-overflow input-norm ceiling of Theorem 3:
/// `B < q / (2^{µ+1}·√(θ·c^{µ−1}·m))` with m the coefficient length.
pub fn max_input_norm(mu: usize, c: usize, theta: f64, q: f64, m: usize) -> f64 {
    let denom =
        (1u64 << (mu + 1)) as f64 * (theta * (c as f64).powi(mu as i32 - 1) * m as f64).sqrt();
    q / denom
}

/// A valid parameter configuration found by the search.
#[derive(Clone, Debug, PartialEq)]
pub struct TtrpChoice {
    pub ell: usize,
    pub mu1: usize,
    pub mu2: usize,
    pub c: usize,
    pub k: usize,
    /// The θ actually used (derived from the slack target).
    pub theta: f64,
    /// log2 of the achieved failure probability (≤ −target_p).
    pub log2_failure: f64,
    /// TT-format representation size in entries.
    pub representation: usize,
    /// Extraction slack B̃/B.
    pub slack: f64,
    /// Maximum provable input norm under q (Theorem 3 regime).
    pub max_norm: f64,
}

/// The concrete parameter search of §7.2 (Table 4's generator).
///
/// Given the total coefficient length `cols = m̄r·φ = d^µ`, a slack target
/// `B̃/B ≤ slack_target`, a failure exponent target `p` (the paper fixes
/// p ≈ 90), the modulus `q`, and a row-count cap `k_max`, iterate over all
/// valid (ℓ, µ₁, µ₂, c, k) with θ derived from the slack target
/// (`θ = k(c/2)^{µ−1}/slack²`), discard configurations violating the
/// overflow or Cantelli bounds, and return the choices minimising the TT
/// representation, smallest-first.
pub fn search_parameters(
    cols: usize,
    q: u64,
    slack_target: f64,
    p_bits: usize,
    k_max: usize,
    c_max: usize,
    ell_max: usize,
) -> Vec<TtrpChoice> {
    let target_failure = 2.0f64.powi(-(p_bits as i32));
    let mut out = Vec::new();
    for ell in 1..=ell_max {
        let d = 1usize << ell;
        if cols.count_ones() != 1 || cols % (d * d) != 0 {
            // need d^µ = cols with µ = µ1+µ2 ≥ 2
            if cols.count_ones() != 1 {
                continue;
            }
        }
        let total_mu = {
            // µ = log_d(cols) if integral
            let mut l = 0usize;
            let mut v = cols;
            while v > 1 {
                if v % d != 0 {
                    break;
                }
                v /= d;
                l += 1;
            }
            if v != 1 {
                continue;
            }
            l
        };
        if total_mu < 2 {
            continue;
        }
        for mu1 in 1..total_mu {
            let mu2 = total_mu - mu1;
            for c in 1..=c_max {
                for k in 1..=k_max {
                    // θ from the slack target: slack = √(k·η₂/θ)
                    let theta = k as f64 * eta2(total_mu, c) / (slack_target * slack_target);
                    if theta >= eta2(total_mu, c) {
                        continue; // out of Cantelli's regime
                    }
                    let failure = match theorem3_failure(total_mu, c, k, theta) {
                        Some(f) => f,
                        None => continue,
                    };
                    if failure > target_failure {
                        continue;
                    }
                    // Theorem 3's no-overflow regime: B < q/(2^{µ+1}√(θ c^{µ−1} m))
                    let m = cols;
                    let max_norm = max_input_norm(total_mu, c, theta, q as f64, m);
                    if max_norm < 2.0 {
                        continue; // cannot even prove tiny norms
                    }
                    let representation = d * (2 * c + total_mu.saturating_sub(2) * c * c) * k;
                    out.push(TtrpChoice {
                        ell,
                        mu1,
                        mu2,
                        c,
                        k,
                        theta,
                        log2_failure: failure.log2(),
                        representation,
                        slack: slack(total_mu, c, k, theta),
                        max_norm,
                    });
                }
            }
        }
    }
    out.sort_by_key(|c| c.representation);
    out
}

/// The minimum k achieving the overflow bound at internal rank c
/// (Theorem 2's prescription: `k ≥ λ / (−log2(1/2 + 2^{−δ−1}))` with
/// `c ≥ ⌈log2(µ−1)⌉ + δ`).
pub fn min_rows_for_lambda(mu: usize, c: usize, lambda: usize) -> Option<usize> {
    let base = overflow_row_failure(mu, c);
    if base >= 1.0 {
        return None;
    }
    let mut k = 1usize;
    while base.powi(k as i32) > 2.0f64.powi(-(lambda as i32)) {
        k += 1;
        if k > 100_000 {
            return None;
        }
    }
    Some(k)
}
