//! LGSA shape model (compact port of `estimator.simulator.LGSA` with
//! `xi=1`, `tau=False`) — the reduced-basis Gram-Schmidt profile facts the
//! infinity-norm probability path consumes, without per-dimension
//! allocation.

use crate::math::log2_u128;
use crate::reduction::delta;

/// Relative tolerance for the q-vector length test.
const Q_VECTOR_RELATIVE_TOLERANCE: f64 = 1e-8;

/// Compact facts about an LGSA profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LgsaSummary {
    /// Effective lattice dimension.
    pub effective_dimension: u64,
    /// Base-2 log of the first Gram-Schmidt length.
    pub first_log2_norm: f64,
    /// Dilithium-style q-vector prefix length.
    pub idx_start: u64,
    /// Last coordinate whose Gram-Schmidt length is materially above one.
    pub idx_end: u64,
    /// Base-2 log of the Gram-Schmidt length at `idx_start`.
    pub log2_vector_length_at_idx_start: f64,
}

/// Whether `length` is within tolerance of `q` (the q-vector test).
#[must_use]
pub fn is_q_vector_length(length: f64, q: f64) -> bool {
    length.is_finite()
        && q.is_finite()
        && q > 0.0
        && (length - q).abs() <= Q_VECTOR_RELATIVE_TOLERANCE * length.abs().max(q)
}

/// Smallest dimension at which the LGSA profile has its full GSA prefix
/// (the finite transition point used by pruned searches).
fn lgsa_stable_dimension(n: u64, q: u128, beta: u32) -> Option<u64> {
    if beta < 2 {
        return None;
    }
    let log_vol = n as f64 * log2_u128(q);
    let step = 2.0 * delta(beta).log2();
    let mut count = if log_vol <= 0.0 {
        1
    } else {
        (((1.0 + 8.0 * log_vol / step).sqrt() - 1.0) / 2.0).floor() as u64
    }
    .max(1);
    let profile_sum = |candidate: u64| step * candidate as f64 * (candidate as f64 + 1.0) / 2.0;
    while profile_sum(count) <= log_vol {
        count = count.checked_add(1)?;
    }
    while count > 1 && profile_sum(count - 1) > log_vol {
        count -= 1;
    }
    Some(count)
}

/// Compact LGSA profile facts for dimension `d` with `identity_vectors`
/// zero-norm coordinates (the q-ary embedding structure).
pub fn lgsa_summary(d: u64, identity_vectors: i128, q: u128, beta: u32) -> Option<LgsaSummary> {
    if beta < 2 || u64::from(beta) > d {
        return None;
    }
    let log_q = log2_u128(q);
    let n = u64::try_from(d as i128 - identity_vectors).ok()?;
    let log_vol = n as f64 * log_q;
    let step = 2.0 * delta(beta).log2();
    let num_gsa_vec = lgsa_stable_dimension(n, q, beta)?.min(d);
    let profile_sum = |count: u64| step * count as f64 * (count as f64 + 1.0) / 2.0;
    let profile_log_vol = profile_sum(num_gsa_vec);
    let shift = if num_gsa_vec > 0 {
        (profile_log_vol - log_vol) / num_gsa_vec as f64
    } else {
        0.0
    };
    let log_norm_at = |index: u64| -> f64 {
        if index < num_gsa_vec {
            (num_gsa_vec - index) as f64 * step - shift
        } else {
            0.0
        }
    };
    let first_log_norm = log_norm_at(0);
    let first_length = 2.0_f64.powf(first_log_norm);
    let q_f = 2.0_f64.powf(log_q);
    let mut idx_start = if is_q_vector_length(first_length, q_f) && num_gsa_vec > 1 {
        1
    } else {
        0
    };
    let unit_threshold = (1.0_f64 + 1e-8).log2();
    let positive_count = if step > 0.0 {
        let first_positive = ((unit_threshold + shift) / step).floor() as u64 + 1;
        if first_positive > num_gsa_vec {
            0
        } else {
            num_gsa_vec - first_positive + 1
        }
    } else {
        0
    };
    let idx_end = positive_count
        .checked_sub(1)
        .unwrap_or_else(|| d.saturating_sub(1));
    idx_start = idx_start.min(d.saturating_sub(1));
    Some(LgsaSummary {
        effective_dimension: d,
        first_log2_norm: first_log_norm,
        idx_start,
        idx_end,
        log2_vector_length_at_idx_start: log_norm_at(idx_start),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgsa_profile_finite_and_positive() {
        let q: u128 = 4_294_967_197;
        let s = lgsa_summary(96, 64, q, 40).unwrap();
        assert!(s.first_log2_norm.is_finite());
        assert!(s.first_log2_norm > 0.0);
        assert!(s.idx_start <= s.idx_end);
        assert!(s.idx_end < 96);
    }

    #[test]
    fn lgsa_summary_across_shapes() {
        let q128: u128 = u128::MAX - 4_294_944_758u128;
        let cases: [(u64, i128, u128); 4] = [
            (96, 32, 4_294_967_197u128),
            (384, 32, 4_294_967_197),
            (768, 64, 18_446_744_073_709_551_557u128),
            (768, 128, q128),
        ];
        for (d, identity_vectors, q) in cases {
            for beta in [40u32, 63, 128, 256, 343, 484, 651] {
                if u64::from(beta) > d {
                    continue;
                }
                let s = lgsa_summary(d, identity_vectors, q, beta).unwrap();
                assert!(s.first_log2_norm.is_finite());
                assert!(s.idx_start <= s.idx_end);
            }
        }
    }

    #[test]
    fn q_vector_length_test() {
        assert!(is_q_vector_length(100.0, 100.0));
        assert!(is_q_vector_length(100.0 * 1.000000001, 100.0));
        assert!(!is_q_vector_length(100.0 * 1.001, 100.0));
        assert!(!is_q_vector_length(f64::NAN, 100.0));
    }
}
