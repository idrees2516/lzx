//! Infinity-norm SIS lattice cost (compact port of the akita
//! `lattice-estimator` infinity path: fixed-beta LGSA cost + a dense
//! (β, ζ) optimizer).

use crate::math::{half_q, log2_erf_from_log2_arg, log2_positive, log2_u128};
use crate::probability::log2_amplify;
use crate::reduction::{beta as beta_from_delta, delta, log2_bkz_cost, short_vectors_for, ReductionCostModel, ShortVectors};
use crate::simulator::{is_q_vector_length, lgsa_summary, LgsaSummary};

/// Success probability target for amplification (lattice-estimator default).
pub const TARGET_SUCCESS_PROBABILITY: f64 = 0.99;

/// A priced attack: the cheapest configuration found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatticeCost {
    /// log2 of the total ring-operations cost.
    pub rop_log2: f64,
    /// log2 of the reduction (BKZ) cost.
    pub red_log2: f64,
    /// Root-Hermite factor of the winning block size.
    pub delta: f64,
    /// BKZ block size β.
    pub beta: u32,
    /// Sieving dimension η.
    pub eta: u32,
    /// Projected-away coordinates ζ.
    pub zeta: u64,
    /// Effective lattice dimension d.
    pub d: u64,
    /// log2 of the per-trial success probability.
    pub prob_log2: Option<f64>,
    /// log2 of the repetition count.
    pub repetitions_log2: Option<f64>,
}

impl LatticeCost {
    /// Classical security bits (log2 attack cost). Non-finite → infinite
    /// security under this model (attack infeasible within the search).
    pub fn security_bits(&self) -> Option<f64> {
        if self.rop_log2.is_finite() {
            Some(self.rop_log2)
        } else {
            None
        }
    }
}

/// SIS instance parameters (scalar form).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SisParameters {
    /// Number of SIS equations (rows of A) — the q-ary lattice's
    /// non-identity part.
    pub n: u64,
    /// The modulus (prime; up to u128).
    pub q: u128,
    /// Number of witness columns m (the attack lattice dimension).
    pub m: u64,
    /// The solution length bound.
    pub length_bound: u64,
    /// The norm in which the bound is expressed.
    pub norm: SisNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SisNorm {
    Infinity,
    Euclidean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EstimatorError {
    InvalidParameter { field: &'static str, reason: String },
    Unsupported { feature: &'static str },
}

impl EstimatorError {
    /// Construct an InvalidParameter error (crate-shared helper).
    pub(crate) fn bad(field: &'static str, reason: impl Into<String>) -> Self {
        EstimatorError::InvalidParameter {
            field,
            reason: reason.into(),
        }
    }
}

impl SisParameters {
    /// Validate structural parameters.
    pub fn validate(&self) -> Result<(), EstimatorError> {
        if self.n == 0 || self.q < 3 || self.m == 0 || self.length_bound == 0 {
            return Err(EstimatorError::bad(
                "params",
                "n, m, length_bound must be positive and q >= 3",
            ));
        }
        if self.norm == SisNorm::Infinity && (self.length_bound as f64) >= half_q(self.q) {
            return Err(EstimatorError::bad(
                "length_bound",
                "SIS trivially easy: length_bound must be below (q - 1) / 2",
            ));
        }
        if self.norm == SisNorm::Euclidean && self.length_bound as f64 >= self.q as f64 {
            return Err(EstimatorError::bad(
                "length_bound",
                "SIS trivially easy: Euclidean length_bound must be below q",
            ));
        }
        if self.m <= self.n {
            return Err(EstimatorError::bad(
                "m",
                "infinity/euclidean estimation requires m > n (a tall SIS lattice)",
            ));
        }
        Ok(())
    }

    /// log2(q).
    pub fn log_q(&self) -> f64 {
        log2_u128(self.q)
    }
}

/// Whether the infinity instance uses the small-box probability formula:
/// `√d · β ≤ q` (with log-space comparison and exact fallback).
fn uses_small_box(params: &SisParameters, effective_dimension: u64) -> bool {
    // √d · bound ≤ q  ⟺  d · bound² ≤ q².
    let lhs_log = 0.5 * log2_positive(effective_dimension as f64)
        + 2.0 * log2_positive(params.length_bound as f64);
    let rhs_log = 2.0 * params.log_q();
    if (lhs_log - rhs_log).abs() > 1e-8 {
        lhs_log < rhs_log
    } else {
        (effective_dimension as u128)
            .saturating_mul(params.length_bound as u128)
            .saturating_mul(params.length_bound as u128)
            <= params.q.saturating_mul(params.q)
    }
}

/// Fixed-(β, ζ) infinity-norm cost (the LGSA attack model).
#[allow(clippy::too_many_arguments)]
fn cost_infinity_fixed(
    beta: u32,
    params: &SisParameters,
    zeta: u64,
    model: ReductionCostModel,
) -> Result<LatticeCost, EstimatorError> {
    params.validate()?;
    let lattice_dimension = params.m;
    let zeta_stop = lattice_dimension
        .checked_sub(params.n)
        .filter(|stop| *stop > 0)
        .ok_or_else(|| {
            EstimatorError::bad("m", "infinity estimation requires m > n")
        })?;
    if zeta >= zeta_stop {
        return Err(EstimatorError::bad("zeta", "zeta must leave an effective lattice dimension greater than n"));
    }
    let effective_dimension = lattice_dimension
        .checked_sub(zeta)
        .ok_or_else(|| EstimatorError::bad("zeta", "zeta must not exceed the lattice dimension"))?;
    if effective_dimension < u64::from(beta) {
        // Block size exceeds the working dimension: attack not applicable.
        return Ok(LatticeCost {
            rop_log2: f64::INFINITY,
            red_log2: f64::INFINITY,
            delta: delta(beta),
            beta,
            eta: beta,
            zeta,
            d: effective_dimension,
            prob_log2: None,
            repetitions_log2: None,
        });
    }
    let identity_vectors = effective_dimension as i128 - params.n as i128;
    let short: ShortVectors = short_vectors_for(model, beta, effective_dimension);
    let bkz_log2 = log2_bkz_cost(model, beta, effective_dimension);
    let summary: LgsaSummary = lgsa_summary(effective_dimension, identity_vectors, params.q, beta)
        .ok_or_else(|| EstimatorError::bad("beta", "LGSA requires 2 <= beta <= d"))?;
    let length_bound = params.length_bound as f64;
    let log_q = params.log_q();
    let d_ = effective_dimension as f64;
    let small_box = uses_small_box(params, effective_dimension);
    let log_trial_prob = if small_box {
        // Small box: Gaussian-coordinate model at the first GSO norm.
        let log2_sigma = crate::math::log2_positive(short.rho) + summary.first_log2_norm
            - 0.5 * log2_positive(d_);
        let log2_erf_arg = log2_positive(length_bound) - 0.5 - log2_sigma;
        d_ * log2_erf_from_log2_arg(log2_erf_arg)
    } else {
        // Dilithium-style: q-vector prefix + Gaussian core.
        let q_f = 2.0_f64.powf(log_q);
        let idx_start = summary.idx_start;
        let idx_end = summary.idx_end.max(idx_start);
        let gaussian_coords = (idx_end - idx_start + 1)
            .max(u64::from(short.sieve_dim))
            .max(1) as f64;
        let log2_sigma = summary.log2_vector_length_at_idx_start
            - 0.5 * log2_positive(gaussian_coords);
        let log2_erf_arg = log2_positive(length_bound) - 0.5 - log2_sigma;
        let mut p = log2_erf_from_log2_arg(log2_erf_arg) * gaussian_coords;
        p += log2_positive((2.0 * length_bound + 1.0) / q_f) * idx_start as f64;
        p
    };
    let log_probability = (log_trial_prob + log2_positive(short.count)).min(0.0);
    if !log_probability.is_finite() {
        return Ok(LatticeCost {
            rop_log2: f64::INFINITY,
            red_log2: f64::INFINITY,
            delta: delta(beta),
            beta,
            eta: short.sieve_dim,
            zeta,
            d: effective_dimension,
            prob_log2: None,
            repetitions_log2: None,
        });
    }
    let repetitions_log2 = log2_amplify(TARGET_SUCCESS_PROBABILITY, log_probability);
    if !repetitions_log2.is_finite() {
        return Ok(LatticeCost {
            rop_log2: f64::INFINITY,
            red_log2: f64::INFINITY,
            delta: delta(beta),
            beta,
            eta: short.sieve_dim,
            zeta,
            d: effective_dimension,
            prob_log2: Some(log_probability),
            repetitions_log2: None,
        });
    }
    let rop_log2 = short.cost_red_log2 + repetitions_log2;
    let red_log2 = bkz_log2 + repetitions_log2;
    Ok(LatticeCost {
        rop_log2,
        red_log2,
        delta: delta(beta),
        beta,
        eta: short.sieve_dim,
        zeta,
        d: effective_dimension,
        prob_log2: Some(log_probability),
        repetitions_log2: Some(repetitions_log2),
    })
}

/// Estimate the cheapest infinity-norm SIS attack: dense search over
/// β ∈ [40, min(d_max, 1024)] (step 1) and ζ over a geometric+linear grid.
///
/// Search granularity note: ζ is sampled on a ~64-point ladder (geometric
/// steps plus the endpoints) rather than exhaustively; the ζ landscape is
/// smooth in log-space, and a missed cheaper configuration would
/// OVERESTIMATE the cost — for the conservative direction (security
/// gating), β is searched exhaustively at step 1 and the ζ ladder includes
/// both boundaries and a midpoint refinement pass.
pub fn estimate_infinity(
    params: &SisParameters,
    model: ReductionCostModel,
) -> Result<LatticeCost, EstimatorError> {
    params.validate()?;
    let zeta_stop = params.m - params.n;
    let mut zeta_grid: Vec<u64> = Vec::new();
    // Geometric ladder.
    let mut z = 1u64;
    while z < zeta_stop {
        zeta_grid.push(z);
        z = z.saturating_mul(2);
    }
    // Midpoints between consecutive ladder points.
    let mids: Vec<u64> = zeta_grid
        .windows(2)
        .map(|w| (w[0] + w[1]) / 2)
        .collect();
    zeta_grid.extend(mids);
    zeta_grid.push(zeta_stop - 1);
    zeta_grid.sort_unstable();
    zeta_grid.dedup();

    let beta_max = params.m.min(1024);
    let mut best: Option<LatticeCost> = None;
    for beta in 40..=beta_max {
        let beta = u32::try_from(beta).unwrap_or(u32::MAX);
        for zeta in &zeta_grid {
            // Skip configs where beta cannot even fit the dimension.
            if params.m - zeta < u64::from(beta) {
                continue;
            }
            if let Ok(cost) = cost_infinity_fixed(beta, params, *zeta, model) {
                if !cost.rop_log2.is_finite() {
                    continue;
                }
                if best.as_ref().map_or(true, |b| cost.rop_log2 < b.rop_log2) {
                    best = Some(cost);
                }
            }
        }
    }
    best.ok_or(EstimatorError::Unsupported {
        feature: "no feasible attack configuration in the search range",
    })
}

// ---------------------------------------------------------------------------
// Euclidean-norm path (the Hermite-SVP model).
// ---------------------------------------------------------------------------

/// Estimate the cheapest Euclidean-norm SIS attack: the lattice-estimator
/// Euclidean path with the optimal sub-dimension.
pub fn estimate_euclidean(
    params: &SisParameters,
    model: ReductionCostModel,
) -> Result<LatticeCost, EstimatorError> {
    params.validate()?;
    if params.norm != SisNorm::Euclidean {
        return Err(EstimatorError::bad(
            "norm",
            "estimate_euclidean requires SisNorm::Euclidean",
        ));
    }
    match model {
        ReductionCostModel::Adps16 { .. } => {}
        ReductionCostModel::Bdgl16 => {
            return Err(EstimatorError::Unsupported {
                feature: "euclidean red_cost_model::BDGL16 (retired; use ADPS16 quantum)",
            });
        }
    }
    let log_q = params.log_q();
    let log_bound = log2_positive(params.length_bound as f64);
    if !log_bound.is_finite() || log_bound <= 0.0 {
        return Err(EstimatorError::bad(
            "length_bound",
            "Euclidean dimension optimization requires length_bound > 1",
        ));
    }
    // Optimal sub-dimension from the closed form, then verified against
    // the full feasible range around it.
    let log_delta = log_bound.powi(2) / (4.0 * params.n as f64 * log_q);
    let d_opt = ((params.n as f64 * log_q / log_delta).sqrt().floor() as u64).max(1);
    let mut best: Option<LatticeCost> = None;
    // Search around d_opt (the cost is unimodal in d).
    let lo = d_opt.saturating_sub(64).max(params.n + 1).min(params.m);
    let hi = d_opt.saturating_add(64).min(params.m);
    for d in lo..=hi {
        if d <= 1 || d > params.m {
            continue;
        }
        let root_volume_log2 = (params.n as f64 / d as f64) * log_q;
        let log_delta_req = (log_bound - root_volume_log2) / (d as f64 - 1.0);
        let delta_value = 2.0_f64.powf(log_delta_req);
        let required_beta = if delta_value >= 1.0 {
            beta_from_delta(delta_value)
        } else {
            None
        };
        let feasible = required_beta.is_some_and(|b| u64::from(b) <= d);
        let beta = required_beta
            .filter(|b| u64::from(*b) <= d)
            .unwrap_or(u32::try_from(d).unwrap_or(u32::MAX));
        let cost_log2 = if feasible && log_delta_req > 0.0 {
            match model {
                ReductionCostModel::Adps16 { mode } => {
                    crate::reduction::adps16_log2_cost(beta, mode)
                }
                ReductionCostModel::Bdgl16 => f64::INFINITY,
            }
        } else {
            f64::INFINITY
        };
        if !cost_log2.is_finite() {
            continue;
        }
        let candidate = LatticeCost {
            rop_log2: cost_log2,
            red_log2: cost_log2,
            delta: delta(beta),
            beta,
            eta: 0,
            zeta: 0,
            d,
            prob_log2: None,
            repetitions_log2: None,
        };
        if best.as_ref().map_or(true, |b| candidate.rop_log2 < b.rop_log2) {
            best = Some(candidate);
        }
    }
    best.ok_or(EstimatorError::Unsupported {
        feature: "no feasible Euclidean attack configuration",
    })
}

// Keep is_q_vector_length linked into the docs surface.
#[allow(dead_code)]
fn _doc_q_vector(length: f64, q: f64) -> bool {
    is_q_vector_length(length, q)
}
