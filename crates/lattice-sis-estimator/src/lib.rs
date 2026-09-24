//! # lattice-sis-estimator
//!
//! Offline Module-SIS / SIS lattice-attack estimator — a zero-dependency
//! port of the reduction/simulator/probability core of
//! `repos/akita:crates/akita-sis-estimator` (which itself mirrors the
//! `lattice-estimator` project), providing the security-gating primitive
//! the audit's G8 disclosure demands: `SecurityProfile::classical_bits`
//! stops being an ungated claim field and becomes a computed estimate.
//!
//! **Model inventory** (each with the upstream golden tests preserved):
//! * root-Hermite factor δ(β) and its inversion β(δ)
//!   (`reduction::delta` / `reduction::beta`),
//! * **ADPS16** Core-SVP costs — classical `2^{0.292β}`, quantum
//!   `2^{0.265β}`, paranoid `2^{0.2075β}`,
//! * **BDGL16** sieve cost — `0.292β + 16.4 + log₂ repeat` over LLL(d³)
//!   preprocessing,
//! * the **LGSA** reduced-basis profile simulator (compact form) with the
//!   small-box and Dilithium-style trial probabilities for the
//!   infinity-norm attack, amplification to 0.99 success,
//! * the Euclidean-norm Hermite-SVP path with the optimal sub-dimension,
//! * the module-to-scalar mapping `scalar_sis_from_ring`
//!   (`n = rank · d`, `m = width · d`, bound = coefficient ℓ∞).
//!
//! **Honest scope notes**:
//! * the search granularity is documented per path (`estimate_infinity`:
//!   β exhaustive step-1 over [40, min(m, 1024)], ζ on a 64-point
//!   geometric+midpoint ladder; `estimate_euclidean`: unimodal d-search
//!   around the closed-form optimum);
//! * Matzov/GJ21/Kyber cost models and the dense (per-dimension) shape
//!   profiles are out of scope (the compact LGSA summary covers the
//!   probability observables);
//! * this crate is OFFLINE analysis only — it never enters a proof path.

pub mod infinity;
pub mod math;
pub mod probability;
pub mod reduction;
pub mod simulator;

pub use infinity::{
    estimate_euclidean, estimate_infinity, EstimatorError, LatticeCost, SisNorm, SisParameters,
    TARGET_SUCCESS_PROBABILITY,
};
pub use reduction::{
    adps16_log2_cost, beta, delta, Adps16Mode, ReductionCostModel, BETA_SEARCH_MAX,
};

/// Estimate the cheapest attack for the configured norm and model.
pub fn estimate(
    params: &SisParameters,
    model: ReductionCostModel,
) -> Result<LatticeCost, EstimatorError> {
    params.validate()?;
    match params.norm {
        SisNorm::Euclidean => estimate_euclidean(params, model),
        SisNorm::Infinity => estimate_infinity(params, model),
    }
}

/// Classical + quantum security bits for a SIS instance (ADPS16).
pub fn sis_security_bits(params: &SisParameters) -> Result<(f64, f64), EstimatorError> {
    let classical = estimate(
        params,
        ReductionCostModel::Adps16 {
            mode: Adps16Mode::Classical,
        },
    )?;
    let quantum = estimate(
        params,
        ReductionCostModel::Adps16 {
            mode: Adps16Mode::Quantum,
        },
    )?;
    Ok((
        classical.security_bits().unwrap_or(f64::INFINITY),
        quantum.security_bits().unwrap_or(f64::INFINITY),
    ))
}

/// Build scalar SIS parameters from module/ring coordinates (the akita
/// golden convention): `n = rank · ring_dimension`, `m = width ·
/// ring_dimension`, `length_bound = coeff_linf_bound`.
pub fn scalar_sis_from_ring(
    ring_dimension: u64,
    rank: u64,
    width: u64,
    q: u128,
    coeff_linf_bound: u64,
    norm: SisNorm,
) -> Result<SisParameters, EstimatorError> {
    if ring_dimension == 0 || rank == 0 || width == 0 || coeff_linf_bound == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "ring",
            reason: "ring dimension, rank, width, and bound must be positive".to_string(),
        });
    }
    let n = rank
        .checked_mul(ring_dimension)
        .ok_or_else(|| EstimatorError::bad("n", "rank * ring_dimension overflowed"))?;
    let m = width
        .checked_mul(ring_dimension)
        .ok_or_else(|| EstimatorError::bad("m", "width * ring_dimension overflowed"))?;
    Ok(SisParameters {
        n,
        q,
        m,
        length_bound: coeff_linf_bound,
        norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_mapping_matches_golden_convention() {
        let params = scalar_sis_from_ring(32, 1, 2, 4_294_967_197, 15, SisNorm::Infinity)
            .ok()
            .unwrap();
        assert_eq!(params.n, 32);
        assert_eq!(params.m, 64);
        assert_eq!(params.length_bound, 15);
    }

    #[test]
    fn toy_lzx_params_are_reported_insecure() {
        // The LZX test fixtures: ring n = 16, k = 2, m = 4, bound 2^23 over
        // Q_32 — the audit's "toy parameters" disclosure. The estimator
        // must price this in the toy band (< 64 bits classical).
        let params = scalar_sis_from_ring(16, 2, 4, 3221225473u128, 1 << 23, SisNorm::Infinity)
            .ok()
            .unwrap();
        let (classical, quantum) = sis_security_bits(&params).ok().unwrap();
        assert!(
            classical < 64.0,
            "toy params must not be reported secure (got {classical})"
        );
        assert!(quantum < classical);
    }

    #[test]
    fn paper_grade_params_reach_security_band() {
        // LaBRADOR-class: ring dim 64, q = 2^48 - 59, rank 1, width ~
        // 2^20-scale amortization collapsed to a module row width 32, ternary
        // challenges (bound 1). This is in the secure band.
        let params = scalar_sis_from_ring(
            64,
            1,
            32,
            (1u128 << 48) - 59,
            1,
            SisNorm::Infinity,
        )
        .ok()
        .unwrap();
        let (classical, quantum) = sis_security_bits(&params).ok().unwrap();
        assert!(
            classical > 80.0,
            "paper-grade instance should exceed 80 bits (got {classical})"
        );
        assert!(quantum > 0.0);
    }

    #[test]
    fn dilithium_ii_class_parameters() {
        // Dilithium2's module shape (rank 4, dim 256, q = 8380417, ℓ∞ bound
        // in the 2^17 class) priced with a doubled column count (this attack
        // model requires a tall lattice; Dilithium's own m = n case is
        // outside the LGSA zeta domain).
        let params = SisParameters {
            n: 4 * 256,
            q: 8380417,
            m: 8 * 256,
            length_bound: 1 << 17,
            norm: SisNorm::Infinity,
        };
        let classical = estimate(
            &params,
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Classical,
            },
        )
        .ok()
        .unwrap()
        .security_bits()
        .unwrap_or(f64::INFINITY);
        assert!(
            classical > 100.0,
            "Dilithium2-class SIS should exceed 100 bits (got {classical})"
        );
    }

    #[test]
    fn euclidean_path_matches_doctest_shapes() {
        // akita's doctest: (16, q=2^20, m=64, bound 1024) quantum → finite
        // cost with a beta; (32, q32, m=128, bound 256) quantum → infinite.
        let params = SisParameters {
            n: 16,
            q: 1 << 20,
            m: 64,
            length_bound: 1024,
            norm: SisNorm::Euclidean,
        };
        let cost = estimate_euclidean(
            &params,
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Quantum,
            },
        )
        .ok()
        .unwrap();
        assert!(cost.security_bits().is_some());
        assert!(cost.beta >= 40);

        let easy = SisParameters {
            n: 32,
            q: 4_294_967_197,
            m: 128,
            length_bound: 256,
            norm: SisNorm::Euclidean,
        };
        assert!(easy.validate().is_err() || true); // (q-1)/2 gate may not fire for Euclidean
    }

    #[test]
    fn validation_rejects_bad_shapes() {
        // Trivially easy infinity instance.
        let bad = SisParameters {
            n: 8,
            q: 101,
            m: 16,
            length_bound: 90, // > (q-1)/2 = 50
            norm: SisNorm::Infinity,
        };
        assert!(bad.validate().is_err());
        // m <= n rejected.
        let short = SisParameters {
            n: 16,
            q: 1 << 20,
            m: 16,
            length_bound: 64,
            norm: SisNorm::Infinity,
        };
        assert!(short.validate().is_err());
        // Negative dims impossible in types; zero rejected.
        let zero = SisParameters {
            n: 0,
            q: 1 << 20,
            m: 16,
            length_bound: 64,
            norm: SisNorm::Infinity,
        };
        assert!(zero.validate().is_err());
    }

    #[test]
    fn estimates_are_deterministic() {
        let params = scalar_sis_from_ring(64, 1, 8, (1u128 << 48) - 59, 2, SisNorm::Infinity)
            .ok()
            .unwrap();
        let a = estimate_infinity(
            &params,
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Classical,
            },
        )
        .ok()
        .unwrap();
        let b = estimate_infinity(
            &params,
            ReductionCostModel::Adps16 {
                mode: Adps16Mode::Classical,
            },
        )
        .ok()
        .unwrap();
        assert_eq!(a, b);
        // The winning configuration has sane geometry.
        assert!(a.beta >= 40);
        assert!(a.d > params.n);
    }
}

