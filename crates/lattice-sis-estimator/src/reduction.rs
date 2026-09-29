//! Reduction cost models and the root-Hermite factor machinery
//! (zero-dependency port of `akita-sis-estimator`'s `reduction/` and the
//! `lattice-estimator` upstream it mirrors — ADPS16 and BDGL16 with the
//! Core-SVP exponents; golden doctests preserved).

/// Upper bracket used by beta inversion (lattice-estimator convention).
pub const BETA_SEARCH_MAX: u32 = 1 << 16;

const SMALL_DELTA: [(u32, f64); 8] = [
    (2, 1.02190),
    (5, 1.01862),
    (10, 1.01616),
    (15, 1.01485),
    (20, 1.01420),
    (25, 1.01342),
    (28, 1.01331),
    (40, 1.01295),
];
const BETA_INVERSION_DELTA_TOLERANCE: f64 = 1e-13;

/// Compute δ from block size β (mirrors `ReductionCost._delta`).
#[must_use]
pub fn delta(beta: u32) -> f64 {
    let beta = beta.max(2);
    if beta <= 2 {
        return 1.0219;
    }
    if beta < 40 {
        for window in SMALL_DELTA.windows(2) {
            if window[1].0 > beta {
                return window[0].1;
            }
        }
        return SMALL_DELTA.last().copied().map_or(1.01295, |(_, d)| d);
    }
    if beta == 40 {
        return SMALL_DELTA.last().copied().map_or(1.01295, |(_, d)| d);
    }
    let beta_f = f64::from(beta);
    let pi = std::f64::consts::PI;
    let e = std::f64::consts::E;
    (beta_f / (2.0 * pi * e) * (pi * beta_f).powf(1.0 / beta_f))
        .powf(1.0 / (2.0 * (beta_f - 1.0)))
}

/// Invert a root-Hermite factor to the smallest supported BKZ block size
/// (lattice-estimator `_beta_find_root` integer semantics: values that
/// would require β < 40 return 40; beyond the bracket → `None`).
#[must_use]
pub fn beta(delta_target: f64) -> Option<u32> {
    if !delta_target.is_finite() {
        return None;
    }
    if delta(40) < delta_target {
        return Some(40);
    }
    if delta_target < delta(BETA_SEARCH_MAX) {
        return None;
    }
    let mut low = 40u32;
    let mut high = BETA_SEARCH_MAX;
    while low < high {
        let mid = low + (high - low) / 2;
        if delta(mid) <= delta_target + BETA_INVERSION_DELTA_TOLERANCE {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Some(low)
}

/// ADPS16 cost mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adps16Mode {
    /// Classical sieve cost `2^{0.292 β}`.
    Classical,
    /// Quantum sieve cost `2^{0.265 β}`.
    Quantum,
    /// Paranoid cost `2^{0.2075 β}`.
    Paranoid,
}

/// ADPS16 exponent for each cost mode.
#[must_use]
pub const fn adps16_exponent(mode: Adps16Mode) -> f64 {
    match mode {
        Adps16Mode::Classical => 0.2920,
        Adps16Mode::Quantum => 0.2650,
        Adps16Mode::Paranoid => 0.2075,
    }
}

/// ADPS16 BKZ cost `2^{c·β}` in log2 space.
#[must_use]
pub fn adps16_log2_cost(beta: u32, mode: Adps16Mode) -> f64 {
    adps16_exponent(mode) * f64::from(beta)
}

/// BDGL16 reduction cost model (`estimator.reduction.BDGL16`).
///
/// Number of SVP calls in BKZ-β (Chen13 experiments, loosely).
#[must_use]
pub const fn svp_repeat(beta: u32, d: u64) -> u64 {
    if (beta as u64) < d {
        8 * d
    } else {
        1
    }
}

/// LLL preprocessing cost with `B=None` (entry bit-size ignored).
#[must_use]
pub fn lll(d: u64) -> f64 {
    (d as f64).powi(3)
}

/// BDGL16 asymptotic sieve cost `LLL(d) + 2^{0.292·β + 16.4 + log₂ repeat}`.
#[must_use]
pub fn bdgl16_cost(beta: u32, d: u64) -> f64 {
    let repeat = svp_repeat(beta, d) as f64;
    let exponent = 0.292 * f64::from(beta) + 16.4 + repeat.log2();
    lll(d) + 2.0_f64.powf(exponent)
}

/// BDGL16 BKZ cost in log₂ space.
#[must_use]
pub fn bdgl16_log2_cost(beta: u32, d: u64) -> f64 {
    let repeat = svp_repeat(beta, d) as f64;
    let sieve_log2 = 0.292 * f64::from(beta) + 16.4 + repeat.log2();
    let lll_log2 = crate::math::log2_positive(lll(d));
    log2_sum(lll_log2, sieve_log2)
}

fn log2_sum(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    let min = a.min(b);
    if !max.is_finite() {
        return max;
    }
    max + 2.0_f64.powf(min - max).ln_1p() / std::f64::consts::LN_2
}

/// Output of a reduction model's short-vector sieve path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortVectors {
    /// Scaling factor ρ relative to the shortest BKZ vector.
    pub rho: f64,
    /// Total short-vector generation cost in log2 space.
    pub cost_red_log2: f64,
    /// Number of output vectors.
    pub count: f64,
    /// Sieving dimension η.
    pub sieve_dim: u32,
}

/// Shared sieve amortization (`ReductionCost._short_vectors_sieve`).
#[must_use]
pub fn sieve_short_vectors(beta: u32, bkz_log2: f64) -> ShortVectors {
    let sieve_dim = beta;
    let n_default = 2.0_f64.powf(0.2075 * f64::from(beta));
    let c = 1.0; // c0/c1 with equal defaults
    if c > 2.0_f64.powi(1000) {
        return ShortVectors {
            rho: f64::INFINITY,
            cost_red_log2: f64::INFINITY,
            count: f64::INFINITY,
            sieve_dim,
        };
    }
    let ceil_c = c.ceil();
    let rho = (4.0_f64 / 3.0).sqrt()
        * delta(sieve_dim).powi(sieve_dim as i32 - 1)
        * delta(beta).powf(1.0 - f64::from(sieve_dim));
    ShortVectors {
        rho,
        cost_red_log2: crate::math::log2_positive(ceil_c) + bkz_log2,
        count: ceil_c * n_default.floor(),
        sieve_dim,
    }
}

/// The configured reduction cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReductionCostModel {
    /// ADPS16 with the given mode.
    Adps16 { mode: Adps16Mode },
    /// BDGL16.
    Bdgl16,
}

impl Default for ReductionCostModel {
    fn default() -> Self {
        Self::Adps16 {
            mode: Adps16Mode::Classical,
        }
    }
}

/// BKZ cost in log2 space for the configured model.
#[must_use]
pub fn log2_bkz_cost(model: ReductionCostModel, beta: u32, d: u64) -> f64 {
    match model {
        ReductionCostModel::Adps16 { mode } => adps16_log2_cost(beta, mode),
        ReductionCostModel::Bdgl16 => bdgl16_log2_cost(beta, d),
    }
}

/// Short-vector sieve output for the configured model.
#[must_use]
pub fn short_vectors_for(model: ReductionCostModel, beta: u32, d: u64) -> ShortVectors {
    sieve_short_vectors(beta, log2_bkz_cost(model, beta, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_matches_small_table() {
        assert!((delta(40) - 1.01295).abs() < 1e-5);
    }

    #[test]
    fn beta_inversion_matches_lattice_estimator_doctests() {
        assert_eq!(beta(1.0121), Some(50));
        assert_eq!(beta(1.0093), Some(100));
        assert_eq!(beta(1.0024), Some(808));
        assert_eq!(beta(1.000_000_000_045_374_4), None);
    }

    #[test]
    fn adps16_policy_costs_scale_with_beta() {
        assert!((adps16_log2_cost(500, Adps16Mode::Classical) - 146.0).abs() < 1e-9);
        assert!((adps16_log2_cost(500, Adps16Mode::Quantum) - 132.5).abs() < 1e-9);
    }

    #[test]
    fn bdgl16_asymptotic_matches_lattice_estimator_doctest() {
        let log2 = bdgl16_log2_cost(500, 1024);
        assert!((log2 - 175.4).abs() < 1e-9);
    }

    #[test]
    fn svp_repeat_switches_at_beta_equals_d() {
        assert_eq!(svp_repeat(63, 64), 8 * 64);
        assert_eq!(svp_repeat(64, 64), 1);
    }
}
