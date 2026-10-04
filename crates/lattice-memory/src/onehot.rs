//! The d-dimensional one-hot indicator layout (Twist & Shout §2.5.3,
//! §2.8, §3.7): commitment-key size control for one-hot addressing.
//!
//! A memory of `K = 2^log_k` cells is viewed as a `d`-dimensional cube of
//! side `N = 2^(log_k/d)`. An address is committed as `d` per-dimension
//! one-hot matrices `ra_1..ra_d`, each of `N·T` entries, instead of one
//! `K·T` matrix — the committed volume shrinks from `K·T` to
//! `d·N·T = d·K^(1/d)·T` per cycle side, and (with a HyperKZG-style key)
//! the commitment key from `K` to `d·N = d·K^(1/d)` group elements.
//! Choosing `d` trades commitment size against prover time (§2.8: small
//! `d` minimizes prover time and proof size; §2.9.2: the "0s are free"
//! doctrine makes even `d = 1` attractive for elliptic-curve commitments).
//!
//! The **chunking policy** bounds the per-chunk matrix footprint: the `T`
//! cycles are split into chunks of `2^chunk_log_t` cycles so a prover (or
//! a sharded folding scheme, §2.9.3) materializes at most
//! `d·N·2^chunk_log_t` indicator entries at a time.

use crate::{FactorId, PiopError};
use lattice_core::{DenseMle, Goldilocks};

/// Layout of the d-dimensional one-hot indicator witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotLayout {
    /// Number of one-hot dimensions (paper parameter `d`).
    pub d: usize,
    /// Total address bits: memory size `K = 2^log_k`.
    pub log_k: usize,
    /// Cycle bits: `T = 2^log_t` memory operations.
    pub log_t: usize,
    /// Chunking policy: cycles per chunk (`2^chunk_log_t`).
    pub chunk_log_t: usize,
}

impl OneHotLayout {
    /// Build a layout with a chunking policy that keeps per-chunk
    /// indicator entries within `max_chunk_entries`.
    ///
    /// `d` must divide `log_k` exactly (§2.5.3 assumes `K^(1/d)` integral);
    /// when a single cycle already exceeds the budget the policy
    /// degenerates to one cycle per chunk (`chunk_log_t = 0`).
    pub fn new(
        log_k: usize,
        log_t: usize,
        d: usize,
        max_chunk_entries: usize,
    ) -> Result<Self, PiopError> {
        if d == 0 || log_k == 0 || log_k % d != 0 {
            return Err(PiopError::BadLayout { log_k, d });
        }
        let log_n = log_k / d;
        let n = 1usize
            .checked_shl(log_n as u32)
            .ok_or(PiopError::BadLayout { log_k, d })?;
        let per_cycle = d.checked_mul(n).ok_or(PiopError::BadLayout { log_k, d })?;
        // Largest chunk with d*N*2^c <= max_chunk_entries (>= 0).
        let mut chunk_log_t = 0usize;
        while chunk_log_t < log_t {
            let next =
                (1usize.checked_shl(chunk_log_t as u32 + 1)).and_then(|c| per_cycle.checked_mul(c));
            match next {
                Some(entries) if entries <= max_chunk_entries => chunk_log_t += 1,
                _ => break,
            }
        }
        Ok(OneHotLayout {
            d,
            log_k,
            log_t,
            chunk_log_t,
        })
    }

    /// Bits per one-hot dimension: `log N = log_k / d`.
    pub fn log_n(&self) -> usize {
        self.log_k / self.d
    }

    /// Side length `N = 2^log_n` of each one-hot dimension.
    pub fn n(&self) -> usize {
        1usize << self.log_n()
    }

    /// Memory size `K = 2^log_k`.
    pub fn k(&self) -> usize {
        1usize << self.log_k
    }

    /// Number of memory operations `T = 2^log_t`.
    pub fn t(&self) -> usize {
        1usize << self.log_t
    }

    /// Total committed indicator entries: `d·N·T` (the quantity that
    /// shrinks from `K·T` as `d` grows).
    pub fn committed_entries(&self) -> usize {
        self.d * self.n() * self.t()
    }

    /// Per-cycle commitment-key size: `d·N = d·K^(1/d)` group elements
    /// (§2.5.3: HyperKZG's key shrinks from K to d·N).
    pub fn key_elements(&self) -> usize {
        self.d * self.n()
    }

    /// Number of cycle chunks under the chunking policy.
    pub fn num_chunks(&self) -> usize {
        // ceil(T / 2^chunk_log_t), at least 1.
        let chunk = 1usize << self.chunk_log_t;
        self.t().div_ceil(chunk)
    }

    /// Half-open cycle range `[lo, hi)` of chunk `idx`.
    pub fn chunk_cycles(&self, idx: usize) -> (usize, usize) {
        let chunk = 1usize << self.chunk_log_t;
        let lo = idx * chunk;
        let hi = ((idx + 1) * chunk).min(self.t());
        (lo, hi.max(lo))
    }

    /// Maximum indicator entries materialized per chunk.
    pub fn max_chunk_entries(&self) -> usize {
        let chunk = 1usize << self.chunk_log_t;
        self.d * self.n() * chunk
    }

    /// The `d` base-`N` digits of `address` (digit 0 = most significant).
    /// Errors when `address >= K`.
    pub fn digits(&self, address: u64) -> Result<Vec<u32>, PiopError> {
        if address >= self.k() as u64 {
            return Err(PiopError::AddressOutOfRange {
                address,
                k: self.k(),
            });
        }
        let n = self.n() as u64;
        let mut out = vec![0u32; self.d];
        let mut rem = address;
        for slot in out.iter_mut().rev() {
            *slot = (rem % n) as u32;
            rem /= n;
        }
        Ok(out)
    }

    /// Inverse of [`Self::digits`]: recombine base-`N` digits.
    pub fn from_digits(&self, digits: &[u32]) -> Result<u64, PiopError> {
        if digits.len() != self.d {
            return Err(PiopError::Shape {
                expected: self.d,
                got: digits.len(),
            });
        }
        let n = self.n() as u64;
        let mut acc = 0u64;
        for dgt in digits {
            if *dgt as u64 >= n {
                return Err(PiopError::AddressOutOfRange {
                    address: *dgt as u64,
                    k: self.n(),
                });
            }
            acc = acc * n + *dgt as u64;
        }
        Ok(acc)
    }

    /// The per-dimension one-hot factor id for a read (`Ra`) or write
    /// (`Wa`) matrix.
    pub fn factor_id(&self, dim: usize, write: bool) -> FactorId {
        if write {
            FactorId::Wa(dim)
        } else {
            FactorId::Ra(dim)
        }
    }
}

/// Build the `(k_i, j)` one-hot matrix for one dimension: entry
/// `matrix[k_i·T + j]` is 1 iff `hot[j] == k_i`.
///
/// Only `T` of the `N·T` entries are nonzero — the sparse prover
/// ([`crate::sparse`]) never materializes the zeros.
pub fn one_hot_dim_matrix(hot: &[u32], log_n: usize, log_t: usize) -> Result<DenseMle, PiopError> {
    let n = 1usize << log_n;
    let t = 1usize << log_t;
    if hot.len() != t {
        return Err(PiopError::Shape {
            expected: t,
            got: hot.len(),
        });
    }
    let mut evals = vec![Goldilocks::ZERO; n * t];
    for (j, &h) in hot.iter().enumerate() {
        if h as usize >= n {
            return Err(PiopError::AddressOutOfRange {
                address: h as u64,
                k: n,
            });
        }
        evals[h as usize * t + j] = Goldilocks::ONE;
    }
    Ok(DenseMle {
        num_vars: log_n + log_t,
        evaluations: evals,
    })
}

/// Embed a per-dimension `(k_i, j)` matrix into the full `(k, j)` space:
/// entry at `(k_1..k_d, j)` equals `matrix[k_i, j]`.
///
/// Kernel-scale dense materialization (`K·T` entries) for the generic
/// sumcheck engine; the production route is the sparse prover, which
/// never forms this array. For `d == 1` this is a copy.
pub fn embed_dim(
    matrix: &DenseMle,
    layout: &OneHotLayout,
    dim: usize,
) -> Result<DenseMle, PiopError> {
    if matrix.num_vars != layout.log_n() + layout.log_t {
        return Err(PiopError::Shape {
            expected: layout.log_n() + layout.log_t,
            got: matrix.num_vars,
        });
    }
    if dim >= layout.d {
        return Err(PiopError::BadLayout {
            log_k: layout.log_k,
            d: layout.d,
        });
    }
    let log_t = layout.log_t;
    let log_k = layout.log_k;
    if layout.d == 1 {
        return Ok(matrix.clone());
    }
    let mut evals = vec![Goldilocks::ZERO; 1usize << (log_k + log_t)];
    let mask_j = (1usize << log_t) - 1;
    let mask_i = (1usize << layout.log_n()) - 1;
    // k = k_1·N^(d-1) + ... + k_d: dimension `dim` occupies bits
    // [(d-1-dim)·log_n, (d-dim)·log_n) of k.
    let shift = (layout.d - 1 - dim) * layout.log_n();
    for (full, e) in evals.iter_mut().enumerate() {
        let j = full & mask_j;
        let k = full >> log_t;
        let k_i = (k >> shift) & mask_i;
        *e = matrix.evaluations[(k_i << log_t) | j];
    }
    Ok(DenseMle {
        num_vars: log_k + log_t,
        evaluations: evals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_rejects_bad_d() {
        assert!(OneHotLayout::new(5, 3, 0, 1 << 20).is_err());
        assert!(OneHotLayout::new(5, 3, 2, 1 << 20).is_err()); // 5 % 2 != 0
        assert!(OneHotLayout::new(5, 3, 5, 1 << 20).is_ok());
        assert!(OneHotLayout::new(0, 3, 1, 1 << 20).is_err());
    }

    #[test]
    fn digits_roundtrip() {
        let layout = OneHotLayout::new(8, 4, 2, 1 << 20).ok().unwrap();
        assert_eq!(layout.n(), 16);
        for a in [0u64, 1, 15, 16, 255, 100] {
            let dg = layout.digits(a).ok().unwrap();
            assert_eq!(dg.len(), 2);
            assert_eq!(layout.from_digits(&dg).ok().unwrap(), a);
        }
        assert!(layout.digits(256).is_err());
        assert!(layout.from_digits(&[16, 0]).is_err());
        assert!(layout.from_digits(&[0]).is_err());
    }

    /// P0-3 size accounting: committed volume and key size shrink with d.
    #[test]
    fn size_accounting_shrinks_with_d() {
        let log_k = 16;
        let log_t = 10;
        let d1 = OneHotLayout::new(log_k, log_t, 1, u64::MAX as usize)
            .ok()
            .unwrap();
        let d2 = OneHotLayout::new(log_k, log_t, 2, u64::MAX as usize)
            .ok()
            .unwrap();
        let d4 = OneHotLayout::new(log_k, log_t, 4, u64::MAX as usize)
            .ok()
            .unwrap();
        let d8 = OneHotLayout::new(log_k, log_t, 8, u64::MAX as usize)
            .ok()
            .unwrap();
        // Committed entries per cycle side (K = 2^16, T = 2^10):
        // d=1: 65536·T; d=2: 2·256·T = 512·T; d=4: 4·16·T = 64·T; d=8: 8·2·T = 16·T.
        assert_eq!(d1.committed_entries(), (1 << 16) * (1 << log_t));
        assert_eq!(d2.committed_entries(), 2 * (1 << 8) * (1 << log_t));
        assert_eq!(d4.committed_entries(), 4 * (1 << 4) * (1 << log_t));
        assert_eq!(d8.committed_entries(), 8 * (1 << 2) * (1 << log_t));
        // Strictly decreasing (§2.5.3 commitment-key control).
        assert!(d2.committed_entries() < d1.committed_entries() / 100);
        assert!(d4.committed_entries() < d2.committed_entries());
        assert!(d8.committed_entries() < d4.committed_entries());
        // Commitment-key elements: K -> d·K^(1/d).
        assert_eq!(d1.key_elements(), 1 << 16);
        assert_eq!(d2.key_elements(), 2 * (1 << 8));
        assert_eq!(d4.key_elements(), 4 * (1 << 4));
    }

    /// P0-3 chunking policy: chunk count and per-chunk bounds.
    #[test]
    fn chunking_policy_bounds_chunks() {
        // K = 2^8, d = 2 (N = 16, per-cycle = 32 entries), T = 2^10.
        // Budget 2^12 entries -> chunk = 2^12 / 32 = 2^7 cycles.
        let layout = OneHotLayout::new(8, 10, 2, 1 << 12).ok().unwrap();
        assert_eq!(layout.chunk_log_t, 7);
        assert_eq!(layout.num_chunks(), 8);
        assert!(layout.max_chunk_entries() <= 1 << 12);
        assert_eq!(layout.chunk_cycles(0), (0, 128));
        assert_eq!(layout.chunk_cycles(7), (896, 1024));
        // Tight budget: one cycle per chunk.
        let tiny = OneHotLayout::new(8, 4, 2, 32).ok().unwrap();
        assert_eq!(tiny.chunk_log_t, 0);
        assert_eq!(tiny.num_chunks(), 16);
        // No budget pressure: whole timeline in one chunk.
        let big = OneHotLayout::new(8, 4, 2, u64::MAX as usize).ok().unwrap();
        assert_eq!(big.chunk_log_t, 4);
        assert_eq!(big.num_chunks(), 1);
        assert_eq!(big.chunk_cycles(0), (0, 16));
    }

    #[test]
    fn one_hot_matrix_shape_and_hotness() {
        let hot = vec![3u32, 0, 1, 1];
        let m = one_hot_dim_matrix(&hot, 2, 2).ok().unwrap();
        assert_eq!(m.num_vars, 4);
        assert_eq!(m.evaluations.len(), 16);
        for (j, &hj) in hot.iter().enumerate().take(4) {
            for k in 0..4usize {
                let expect = if k == hj as usize {
                    lattice_core::Goldilocks::ONE
                } else {
                    lattice_core::Goldilocks::ZERO
                };
                assert_eq!(m.evaluations[k * 4 + j], expect, "k={k} j={j}");
            }
        }
        assert!(one_hot_dim_matrix(&[5u32], 2, 0).is_err()); // out of range
        assert!(one_hot_dim_matrix(&[0u32; 2], 2, 2).is_err()); // wrong T
    }

    #[test]
    fn embed_dim_matches_native_entries() {
        // d = 2, log_k = 4 (N = 4), log_t = 2.
        let layout = OneHotLayout::new(4, 2, 2, u64::MAX as usize).ok().unwrap();
        // Dimension 0 hot indices and dimension 1 hot indices per cycle.
        let hot0 = vec![1u32, 3, 0, 2];
        let hot1 = vec![2u32, 0, 3, 1];
        let m0 = one_hot_dim_matrix(&hot0, 2, 2).ok().unwrap();
        let m1 = one_hot_dim_matrix(&hot1, 2, 2).ok().unwrap();
        let e0 = embed_dim(&m0, &layout, 0).ok().unwrap();
        let e1 = embed_dim(&m1, &layout, 1).ok().unwrap();
        assert_eq!(e0.num_vars, 6);
        for j in 0..4usize {
            for k in 0..16usize {
                // k = k_1·4 + k_2: dim 0 = bits [2,4), dim 1 = bits [0,2).
                let k1 = (k >> 2) & 3;
                let k2 = k & 3;
                let v0 = if k1 == hot0[j] as usize {
                    lattice_core::Goldilocks::ONE
                } else {
                    lattice_core::Goldilocks::ZERO
                };
                let v1 = if k2 == hot1[j] as usize {
                    lattice_core::Goldilocks::ONE
                } else {
                    lattice_core::Goldilocks::ZERO
                };
                assert_eq!(e0.evaluations[k * 4 + j], v0, "dim0 k={k} j={j}");
                assert_eq!(e1.evaluations[k * 4 + j], v1, "dim1 k={k} j={j}");
            }
        }
        // d = 1 embedding is a copy.
        let l1 = OneHotLayout::new(2, 2, 1, u64::MAX as usize).ok().unwrap();
        let m = one_hot_dim_matrix(&[1u32, 0, 3, 2], 2, 2).ok().unwrap();
        assert_eq!(embed_dim(&m, &l1, 0).ok().unwrap(), m);
        // Wrong arity rejected: a (2^3 × 2^2) matrix does not fit the
        // layout's (2^2 × 2^2) dimension slots.
        let bad = one_hot_dim_matrix(&[1u32, 0, 3, 2], 3, 2).ok().unwrap();
        assert!(embed_dim(&bad, &layout, 0).is_err());
    }
}
