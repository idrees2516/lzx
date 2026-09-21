//! Challenge sets with prescribed distributions for Module-SIS security.
//!
//! Lattice protocols derive their hardness from *structured* challenge
//! distributions, not uniform field elements:
//! * **sparse ternary** — fixed Hamming weight `w` over `{-1,0,1}^n`
//!   (Akita sparse challenges; Module-SIS with sparse challenges keeps
//!   parameters small while making the challenge space polynomially large
//!   in n — enough for Fiat–Shamir extraction),
//! * **uniform ternary** — each coefficient uniform in `{-1,0,1}`,
//! * **small interval** — coefficients in `[-B, B]` (folding relaxations).
//!
//! Sampling is via rejection from SHAKE-256 streams with the same
//! bounded-rejection discipline as the transcript, and every sampler is
//! deterministic given the seed so prover/verifier derive identical sets.

use crate::keccak::shake256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeDistribution {
    /// Fixed Hamming weight w, coefficients ±1.
    SparseTernary { weight: usize },
    /// Coefficients uniform in {-1, 0, 1}.
    UniformTernary,
    /// Coefficients uniform in [-bound, bound].
    SmallInterval { bound: u32 },
}

/// A deterministically-sampled challenge vector over Z^n (coefficients
/// reduced mod q by the consumer ring).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeSet {
    pub distribution: ChallengeDistribution,
    /// Signed coefficients (small integers).
    pub coefficients: Vec<i64>,
}

impl ChallengeSet {
    /// Deterministically sample from a seed under a distribution.
    /// Uses bounded rejection: at most `4 * n * bound` candidates consumed,
    /// else fails closed (never silently narrows the distribution).
    pub fn sample(
        distribution: ChallengeDistribution,
        n: usize,
        seed: &[u8],
    ) -> Result<Self, ChallengeError> {
        match distribution {
            ChallengeDistribution::SparseTernary { weight } => {
                Self::sample_sparse_ternary(weight, n, seed)
            }
            ChallengeDistribution::UniformTernary => Self::sample_uniform_ternary(n, seed),
            ChallengeDistribution::SmallInterval { bound } => {
                Self::sample_small_interval(bound, n, seed)
            }
        }
    }

    fn sample_uniform_ternary(n: usize, seed: &[u8]) -> Result<Self, ChallengeError> {
        // Rejection over mod-3 residues: take bytes, keep b % 3 in {0,1,2}
        // only when b < 252 (252 divisible by 3 → unbiased).
        let stream = shake256(&Self::frame(b"ternary", seed, n), n * 4);
        let mut coefficients = Vec::with_capacity(n);
        let mut idx = 0usize;
        while coefficients.len() < n {
            if idx >= stream.len() {
                return Err(ChallengeError::RejectionBudgetExceeded);
            }
            let b = stream[idx];
            idx += 1;
            if b < 252 {
                coefficients.push(match b % 3 {
                    0 => 0i64,
                    1 => 1,
                    _ => -1,
                });
            }
        }
        Ok(ChallengeSet {
            distribution: ChallengeDistribution::UniformTernary,
            coefficients,
        })
    }

    fn sample_small_interval(bound: u32, n: usize, seed: &[u8]) -> Result<Self, ChallengeError> {
        // Unbiased rejection: need span = 2*bound+1 values per byte-window;
        // accept u16 < floor(65536 / span) * span.
        let span = 2 * bound as u64 + 1;
        let limit = (65536 / span) * span;
        let stream = shake256(&Self::frame(b"interval", seed, n * 2), n * 8);
        let mut coefficients = Vec::with_capacity(n);
        let mut idx = 0usize;
        while coefficients.len() < n {
            if idx + 2 > stream.len() {
                return Err(ChallengeError::RejectionBudgetExceeded);
            }
            let v = u16::from_le_bytes([stream[idx], stream[idx + 1]]) as u64;
            idx += 2;
            if v < limit {
                coefficients.push(v as i64 % span as i64 - bound as i64);
            }
        }
        Ok(ChallengeSet {
            distribution: ChallengeDistribution::SmallInterval { bound },
            coefficients,
        })
    }

    fn sample_sparse_ternary(weight: usize, n: usize, seed: &[u8]) -> Result<Self, ChallengeError> {
        // Sample `weight` distinct positions (Fisher–Yates over rejection-
        // sampled indices) and a sign bit per position.
        if weight > n {
            return Err(ChallengeError::InvalidParameters);
        }
        let stream = shake256(&Self::frame(b"sparse", seed, n), n * 8 + 16);
        let mut coefficients = vec![0i64; n];
        let mut chosen: Vec<usize> = Vec::with_capacity(weight);
        let mut idx = 0usize;
        // Rejection-swap sampling of distinct positions: position j must be
        // in [0, n - j) to keep uniform distinct draws.
        let mut remaining = n;
        while chosen.len() < weight {
            if idx + 4 > stream.len() {
                return Err(ChallengeError::RejectionBudgetExceeded);
            }
            let raw = u32::from_le_bytes([
                stream[idx],
                stream[idx + 1],
                stream[idx + 2],
                stream[idx + 3],
            ]) as u64;
            idx += 4;
            // Unbiased modulo rejection for the shrinking range.
            let range = remaining as u64;
            let limit = (u32::MAX / range as u32) as u64 * range;
            if raw >= limit {
                continue;
            }
            let pos = (raw % range) as usize;
            // The `pos`-th not-yet-chosen slot: since chosen slots are marked
            // by compaction, remap through the live list.
            let live = Self::nth_live(&coefficients, pos);
            if let Some(p) = live {
                let sign = if (stream[idx % stream.len()] & 1) == 0 { 1i64 } else { -1 };
                idx += 1;
                coefficients[p] = sign;
                chosen.push(p);
                remaining -= 1;
            }
        }
        Ok(ChallengeSet {
            distribution: ChallengeDistribution::SparseTernary { weight },
            coefficients,
        })
    }

    /// Index of the `k`-th slot that is still zero (un-chosen).
    fn nth_live(coefficients: &[i64], k: usize) -> Option<usize> {
        let mut seen = 0usize;
        for (i, c) in coefficients.iter().enumerate() {
            if *c == 0 {
                if seen == k {
                    return Some(i);
                }
                seen += 1;
            }
        }
        None
    }

    fn frame(tag: &[u8], seed: &[u8], n: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + tag.len() + seed.len());
        buf.extend_from_slice(tag);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(seed.len() as u32).to_le_bytes());
        buf.extend_from_slice(seed);
        buf
    }

    /// The infinity norm of the challenge (bound bookkeeping for SIS).
    pub fn infinity_norm(&self) -> i64 {
        self.coefficients
            .iter()
            .map(|c| c.abs())
            .max()
            .unwrap_or(0)
    }

    /// Hamming weight (number of non-zero coefficients).
    pub fn hamming_weight(&self) -> usize {
        self.coefficients.iter().filter(|c| **c != 0).count()
    }

    /// Minimum entropy (log2 of challenge space size) — Akita-style
    /// security bookkeeping so callers can refuse undersized sets.
    pub fn min_entropy_bits(&self) -> f64 {
        match self.distribution {
            ChallengeDistribution::SparseTernary { weight } => {
                // log2 C(n, w) + w  (signs)
                let n = self.coefficients.len() as f64;
                let w = weight as f64;
                let ln_binom = ln_factorial(n) - ln_factorial(w) - ln_factorial(n - w);
                (ln_binom + w * std::f64::consts::LN_2) / std::f64::consts::LN_2
            }
            ChallengeDistribution::UniformTernary => {
                let n = self.coefficients.len() as f64;
                n * (3.0f64).log2()
            }
            ChallengeDistribution::SmallInterval { bound } => {
                let n = self.coefficients.len() as f64;
                n * (2.0 * bound as f64 + 1.0).log2()
            }
        }
    }
}

fn ln_factorial(x: f64) -> f64 {
    // Stirling approximation, adequate for entropy bookkeeping.
    if x <= 1.0 {
        0.0
    } else {
        x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI * x).ln()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeError {
    RejectionBudgetExceeded,
    InvalidParameters,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_ternary_distribution() {
        let cs = ChallengeSet::sample(ChallengeDistribution::UniformTernary, 1024, b"seed-a").unwrap();
        assert_eq!(cs.coefficients.len(), 1024);
        for c in &cs.coefficients {
            assert!(*c == 0 || *c == 1 || *c == -1);
        }
        // Deterministic.
        let cs2 = ChallengeSet::sample(ChallengeDistribution::UniformTernary, 1024, b"seed-a").unwrap();
        assert_eq!(cs, cs2);
        // Different seed -> different set (whp).
        let cs3 = ChallengeSet::sample(ChallengeDistribution::UniformTernary, 1024, b"seed-b").unwrap();
        assert_ne!(cs, cs3);
    }

    #[test]
    fn sparse_ternary_weight_exact() {
        // log2 C(512,w) + w: w=32 -> ~204 bits, w=64 -> ~333, w=128 -> ~543.
        for (weight, min_bits) in [(32usize, 180.0f64), (64, 300.0), (128, 500.0)] {
            let cs = ChallengeSet::sample(
                ChallengeDistribution::SparseTernary { weight },
                512,
                b"sparse",
            )
            .unwrap();
            assert_eq!(cs.hamming_weight(), weight);
            for c in &cs.coefficients {
                assert!(*c == 0 || *c == 1 || *c == -1);
            }
            assert!(
                cs.min_entropy_bits() > min_bits,
                "weight {weight}: entropy {} <= {min_bits}",
                cs.min_entropy_bits()
            );
        }
    }

    #[test]
    fn small_interval_bounded() {
        let bound = 15u32;
        let cs = ChallengeSet::sample(
            ChallengeDistribution::SmallInterval { bound },
            768,
            b"int",
        )
        .unwrap();
        assert_eq!(cs.infinity_norm() as u32, 15); // whp hits both ends
        for c in &cs.coefficients {
            assert!(c.abs() <= bound as i64);
        }
    }

    #[test]
    fn rejects_bad_parameters() {
        assert_eq!(
            ChallengeSet::sample(ChallengeDistribution::SparseTernary { weight: 10 }, 4, b"x")
                .err(),
            Some(ChallengeError::InvalidParameters)
        );
    }
}
