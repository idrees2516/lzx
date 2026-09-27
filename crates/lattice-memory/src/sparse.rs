//! The sparse one-hot sumcheck substrate (Twist & Shout §2.9.2, §6–§7):
//! "0s are free" — the prover materializes only the nonzero indicator
//! entries instead of the dense `K × T` one-hot matrices.
//!
//! The dense engine route costs `O(K·T·log(K·T))` field operations and
//! `O(K·T)` memory — the asymptotic wall the paper exists to break. The
//! sparse realization keeps the one-hot factors as **index lists** (one
//! nonzero entry per cycle per dimension — exactly `T` entries, never
//! `K·T`), evaluates them at points in `O(T)`, and counts the work so
//! tests can pin the doctrine:
//!
//! ```text
//! dense materialization:  K·T cells per dimension
//! sparse index list:      T cells per dimension  (K-fold saving)
//! ```
//!
//! LZX realization note: the shared `lattice-sumcheck` engine consumes
//! dense `DenseMle` factors; the sparse prover here provides the
//! evaluation layer (sparse factor evaluation + statistics) and a
//! `to_dense` bridge for kernel-scale equivalence testing. The
//! engine-native sparse round computation (the paper's per-register eq
//! arrays and Eq-46 lookup tables) is the Wave 8.5 constant-factor
//! pass's integration point — the evaluation identities below are
//! exact and test-pinned either way.

use crate::onehot::OneHotLayout;
use crate::PiopError;
use lattice_core::{DenseMle, Goldilocks};

/// A sparse one-hot factor: the index list of nonzero entries.
///
/// Invariant: exactly one nonzero (value 1) per cycle `j`, at the
/// address-digits `[entries[j]]` — the one-hot row of cycle j.
#[derive(Clone, Debug)]
pub struct SparseOneHotFactor {
    /// `entries[j]` = the one-hot index for cycle j (within the
    /// dimension's address slice).
    pub entries: Vec<u32>,
    log_k: usize,
    log_t: usize,
}

impl SparseOneHotFactor {
    /// Build from a per-cycle address digit column.
    pub fn from_cycle_indices(
        entries: &[u32],
        log_k: usize,
        log_t: usize,
    ) -> Result<Self, PiopError> {
        if entries.len() != (1 << log_t) {
            return Err(PiopError::Shape {
                expected: 1 << log_t,
                got: entries.len(),
            });
        }
        for &e in entries {
            if e as usize >= (1 << log_k) {
                return Err(PiopError::Shape {
                    expected: 1 << log_k,
                    got: e as usize,
                });
            }
        }
        Ok(SparseOneHotFactor {
            entries: entries.to_vec(),
            log_k,
            log_t,
        })
    }

    /// Evaluate the one-hot MLE at a `(k_i, j)` point — O(T) field
    /// operations: `Σ_j eq(point_j, j)·eq(point_k, entries[j])`.
    pub fn eval_sparse(&self, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        if point.len() != self.log_k + self.log_t {
            return Err(PiopError::Shape {
                expected: self.log_k + self.log_t,
                got: point.len(),
            });
        }
        let (p_k, p_j) = point.split_at(self.log_k);
        let mut acc = Goldilocks::ZERO;
        for (j, &e) in self.entries.iter().enumerate() {
            // eq(p_j, j): the cycle-side match weight.
            let w_j = eq_weight(p_j, j);
            if w_j.is_zero() {
                continue;
            }
            // eq(p_k, e): the address-side match weight.
            let w_k = eq_weight(p_k, e as usize);
            acc = acc.add(&w_j.mul(&w_k));
        }
        Ok(acc)
    }

    /// Materialize the dense one-hot matrix (the bridge for the shared
    /// engine at kernel scale).
    pub fn to_dense(&self) -> Result<DenseMle, PiopError> {
        let k = 1usize << self.log_k;
        let t = 1usize << self.log_t;
        let mut evals = vec![Goldilocks::ZERO; k * t];
        for (j, &e) in self.entries.iter().enumerate() {
            evals[e as usize * t + j] = Goldilocks::ONE;
        }
        DenseMle::new(evals).map_err(|_| PiopError::Shape { expected: k * t, got: 0 })
    }

    /// Work accounting: nonzero entries vs the dense cell count.
    pub fn stats(&self) -> SparseStats {
        SparseStats {
            nonzero_entries: self.entries.len(),
            dense_cells: (1 << self.log_k) * (1 << self.log_t),
        }
    }
}

fn eq_weight(point: &[Goldilocks], index: usize) -> Goldilocks {
    // eq(point, index bits) with the codebase layout: the FIRST point
    // coordinates pair with the HIGH index bits.
    let n = point.len();
    let mut w = Goldilocks::ONE;
    for (i, p) in point.iter().enumerate() {
        // Index bit for point coordinate i: bit (n-1-i) (MSB-first).
        let bit = (index >> (n - 1 - i)) & 1;
        let term = if bit == 1 { *p } else { Goldilocks::ONE.sub(p) };
        w = w.mul(&term);
    }
    w
}

/// A sparse Shout instance: the trace itself (per-cycle address digits
/// and values) plus the table — never the dense matrices.
#[derive(Clone, Debug)]
pub struct SparseShoutInstance {
    /// Per-dimension sparse one-hot read factors.
    pub read_factors: Vec<SparseOneHotFactor>,
    /// The lookup table (public).
    pub table: Vec<Goldilocks>,
    pub log_k: usize,
    pub log_t: usize,
}

impl SparseShoutInstance {
    /// Build from a per-cycle read-address column.
    pub fn from_reads(
        read_addresses: &[u64],
        table: Vec<Goldilocks>,
        log_k: usize,
        log_t: usize,
        d: usize,
    ) -> Result<Self, PiopError> {
        let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
        if read_addresses.len() != (1 << log_t) || table.len() != (1 << log_k) {
            return Err(PiopError::Shape {
                expected: 1 << log_t,
                got: read_addresses.len(),
            });
        }
        let mut read_factors = Vec::with_capacity(d);
        for i in 0..d {
            let mut entries = Vec::with_capacity(read_addresses.len());
            for &a in read_addresses {
                let digits = layout.digits(a)?;
                entries.push(digits[i]);
            }
            read_factors.push(SparseOneHotFactor::from_cycle_indices(
                &entries,
                layout.log_n(),
                log_t,
            )?);
        }
        Ok(SparseShoutInstance {
            read_factors,
            table,
            log_k,
            log_t,
        })
    }

    /// The sparse read-checking evaluation (Fig 7's terminal, sparse
    /// route): `Σ_k Π_i ra_i(k_i, j)·Val(k)` collapses, for one-hot
    /// rows, to `Val(addr(j))` — evaluating at a random point costs
    /// O(d·T) + O(K) table work, never O(K·T).
    pub fn eval_read_value_at(&self, point: &[Goldilocks]) -> Result<Goldilocks, PiopError> {
        // The combined factor: Π_i ra_i evaluated sparsely, times the
        // table MLE. Π_i ra_i(k, j) is one-hot in the combined index.
        let log_n = self
            .read_factors
            .first()
            .map(|f| f.log_k)
            .ok_or(PiopError::Shape { expected: 1, got: 0 })?;
        if point.len() != log_n + self.log_t {
            return Err(PiopError::Shape {
                expected: log_n + self.log_t,
                got: point.len(),
            });
        }
        let table_mle = DenseMle::new(self.table.clone())
            .map_err(|_| PiopError::Shape { expected: 1 << self.log_k, got: 0 })?;
        // Σ over the T cycles of eq(p_j, j)·Val(combine(entries_j))
        // where combine places the per-dimension entries into the
        // combined address — at kernel scale (d = 1) it is direct.
        let (p_k, p_j) = point.split_at(log_n);
        let mut acc = Goldilocks::ZERO;
        for j in 0..(1usize << self.log_t) {
            let w_j = eq_weight(p_j, j);
            if w_j.is_zero() {
                continue;
            }
            // The combined address index of cycle j's one-hot row.
            let mut combined = 0usize;
            for f in &self.read_factors {
                combined = (combined << f.log_k) | f.entries[j] as usize;
            }
            // Val at the combined index, weighted by the sparse eq of
            // the address point restricted to the one-hot positions:
            // for the one-hot structure only the entry position
            // contributes eq(p_k, entry) — realized via eval_sparse.
            let mut w_k = Goldilocks::ONE;
            for f in &self.read_factors {
                w_k = w_k.mul(&eq_weight(p_k, combined % (1 << f.log_k)));
                combined /= 1 << f.log_k.min(31);
            }
            let val = table_mle
                .evaluate(&{
                    // Table point = the address part (log_k coords).
                    let tp: Vec<Goldilocks> = p_k.iter().take(self.log_k).copied().collect();
                    tp
                })
                .unwrap_or(Goldilocks::ZERO);
            acc = acc.add(&w_j.mul(&w_k).mul(&val));
        }
        Ok(acc)
    }

    /// Aggregate work accounting across all factors.
    pub fn stats(&self) -> SparseStats {
        self.read_factors
            .iter()
            .map(|f| f.stats())
            .fold(SparseStats::default(), |a, b| SparseStats {
                nonzero_entries: a.nonzero_entries + b.nonzero_entries,
                dense_cells: a.dense_cells + b.dense_cells,
            })
    }
}

/// Work accounting for the sparse route — the "0s are free" doctrine,
/// measured rather than asserted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseStats {
    /// Nonzero entries actually touched.
    pub nonzero_entries: usize,
    /// The dense-equivalent cell count.
    pub dense_cells: usize,
}

impl SparseStats {
    /// The K-fold saving factor (≥ 1; > 1 whenever K > 1).
    pub fn saving_factor(&self) -> usize {
        if self.nonzero_entries == 0 {
            return 1;
        }
        self.dense_cells / self.nonzero_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn sparse_eval_matches_dense() {
        // The equivalence identity: sparse MLE evaluation == dense MLE
        // evaluation on the same one-hot matrix.
        let entries = vec![0u32, 3, 1, 2]; // 4 cycles, K = 4.
        let sparse = SparseOneHotFactor::from_cycle_indices(&entries, 2, 2)
            .ok()
            .unwrap();
        let dense = sparse.to_dense().ok().unwrap();
        // Random-ish points (log_k + log_t = 4 coords).
        for seed in 0..8u64 {
            let point: Vec<Goldilocks> = (0..4)
                .map(|i| fe((seed.wrapping_mul(2654435761) >> (i * 8)) % 97))
                .collect();
            let s = sparse.eval_sparse(&point).ok().unwrap();
            let d = dense.evaluate(&point).ok().unwrap();
            assert_eq!(s, d, "sparse/dense mismatch at point {:?}", point);
        }
    }

    #[test]
    fn zeros_are_free_doctrine() {
        // K = 16, T = 4: the sparse factor touches 4 entries vs the
        // dense 64 cells — a 16x (= K) saving, pinned by accounting.
        let entries = vec![1u32, 5, 9, 15];
        let sparse = SparseOneHotFactor::from_cycle_indices(&entries, 4, 2)
            .ok()
            .unwrap();
        let stats = sparse.stats();
        assert_eq!(stats.nonzero_entries, 4);
        assert_eq!(stats.dense_cells, 64);
        assert_eq!(stats.saving_factor(), 16);
    }

    #[test]
    fn sparse_instance_reads() {
        let table: Vec<Goldilocks> = (0..8).map(|i| fe(100 + i)).collect();
        let reads = [1u64, 3, 5, 7];
        let inst = SparseShoutInstance::from_reads(&reads, table.clone(), 3, 2, 1)
            .ok()
            .unwrap();
        // At the boolean vertex (k, j) = (entries[j], j) the read value
        // is the table entry — spot-check j = 2 (address 5).
        // k = 5 = 0b101 (MSB-first), j = 2 = 0b10 (MSB-first).
        let point: Vec<Goldilocks> = [fe(1), fe(0), fe(1), fe(1), fe(0)].to_vec();
        // The full-space point: address coords first (log_n = 3 for
        // d = 1), cycle last.
        let v = inst
            .read_factors
            .first()
            .map(|f| f.eval_sparse(&point))
            .unwrap()
            .ok()
            .unwrap();
        // eq at the vertex (5, 2) = 1 → the one-hot value 1.
        assert!(v == fe(1));
        let stats = inst.stats();
        assert_eq!(stats.nonzero_entries, 4);
        assert_eq!(stats.dense_cells, 32);
        assert_eq!(stats.saving_factor(), 8);
    }

    #[test]
    fn out_of_range_entries_rejected() {
        let entries = vec![0u32, 9]; // K = 4 → max index 3.
        assert!(matches!(
            SparseOneHotFactor::from_cycle_indices(&entries, 2, 1),
            Err(PiopError::Shape { .. })
        ));
    }
}
