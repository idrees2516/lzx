//! MLE-structured verifier tables (Stage 5.3, the §9.4 constant-root
//! discipline): the public read-only tables of the lookup layer evaluated
//! through their ALGEBRAIC STRUCTURE instead of materialized arrays.
//!
//! The lookup Shouts' tables are public functions of the table index:
//! the identity family (`range8` / `range7` / `range12` / `al8`,
//! `T[k] = k`), the power table (`pow2`, `T[k] = 2^k`), and the
//! program-derived decode table (arbitrary but deterministic). The
//! verifier's terminal identity needs `T(ρ_k)` — the MLE of the table at
//! the sumcheck's terminal address point. For the structured families
//! that evaluation is a CLOSED FORM in `O(log K)`:
//!
//! * identity: `MLE(r) = Σ_i r_i · 2^{m−1−i}` (the value is linear in
//!   the index bits — `T` is its own multilinear extension);
//! * pow2: `MLE(r) = Π_i (1 + r_i · (2^{2^{m−1−i}} − 1))` (the value is
//!   the product of per-bit factors).
//!
//! The table absorb of the per-lookup flow (an `O(K)` field-slice hash
//! per lookup) is replaced by the `(kind, log_k)` tag; the terminal
//! identity itself is what binds the prover to the verifier's table, so
//! skipping the value absorb costs nothing soundness-wise and removes
//! both the materialization and the hashing from the verify path.

use lattice_core::{DenseMle, Goldilocks};

/// The public table families of the lookup layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TableFamily {
    /// `T[k] = k` (range8 / range7 / range12 / al8).
    Identity,
    /// `T[k] = 2^k` (pow2; `k ≤ 63` per the cap discipline).
    Pow2,
    /// Arbitrary but verifier-deterministic values (the decode table).
    Values,
}

impl TableFamily {
    /// Stable transcript tag (absorbed in place of the table values).
    pub fn tag(self) -> u64 {
        match self {
            TableFamily::Identity => 1,
            TableFamily::Pow2 => 2,
            TableFamily::Values => 3,
        }
    }
}

/// A structured public table.
#[derive(Clone, Debug)]
pub struct StructuredTable {
    pub family: TableFamily,
    pub log_k: usize,
    /// The materialized values (built lazily / once; the prover's dense
    /// factor and the `Values` family need them).
    values: Vec<Goldilocks>,
}

impl StructuredTable {
    /// A structured table without materializing the values (the
    /// verifier's closed-form route).
    pub const fn new(family: TableFamily, log_k: usize) -> Self {
        StructuredTable {
            family,
            log_k,
            values: Vec::new(),
        }
    }

    /// The materialized values (built once on first use).
    pub fn values(&self) -> Vec<Goldilocks> {
        if self.values.len() == (1usize << self.log_k).max(1) {
            return self.values.clone();
        }
        let n = 1usize << self.log_k;
        match self.family {
            TableFamily::Identity => (0..n).map(|j| Goldilocks::from_u64(j as u64)).collect(),
            TableFamily::Pow2 => (0..n)
                .map(|j| Goldilocks::from_u64(1u64 << j.min(63)))
                .collect(),
            TableFamily::Values => Vec::new(),
        }
    }

    /// Build a `Values` table from explicit verifier-deterministic data.
    pub fn from_values(values: Vec<Goldilocks>, log_k: usize) -> Result<Self, TableError> {
        if values.len() != (1usize << log_k) {
            return Err(TableError::Shape {
                expected: 1usize << log_k,
                got: values.len(),
            });
        }
        Ok(StructuredTable {
            family: TableFamily::Values,
            log_k,
            values,
        })
    }

    /// The table's MLE at an address point — the closed form for the
    /// structured families, the eq-dot for `Values`.
    ///
    /// Bit-exact with `DenseMle::evaluate` over the materialized values
    /// (test-pinned) — the identity value is linear in the index bits
    /// and the pow2 value factors per bit, so the multilinear extension
    /// IS the closed form.
    pub fn eval_mle(&self, point: &[Goldilocks]) -> Result<Goldilocks, TableError> {
        if point.len() != self.log_k {
            return Err(TableError::Shape {
                expected: self.log_k,
                got: point.len(),
            });
        }
        let m = self.log_k;
        match self.family {
            TableFamily::Identity => {
                // Σ_i r_i · 2^{m−1−i}.
                let mut acc = Goldilocks::ZERO;
                for (i, &r) in point.iter().enumerate() {
                    let weight = 1u64 << (m - 1 - i);
                    acc = acc.add(&r.mul(&Goldilocks::from_u64(weight)));
                }
                Ok(acc)
            }
            TableFamily::Pow2 => {
                // Π_i (1 + r_i · (2^{2^{m−1−i}} − 1)).
                let mut acc = Goldilocks::ONE;
                for (i, &r) in point.iter().enumerate() {
                    let base = 1u64 << (m - 1 - i);
                    let factor = (1u64 << base) - 1; // base ≤ 6 here (log_k ≤ 6)
                    acc = acc.mul(&Goldilocks::ONE.add(&r.mul(&Goldilocks::from_u64(factor))));
                }
                Ok(acc)
            }
            TableFamily::Values => {
                if self.values.len() != (1usize << m) {
                    return Err(TableError::Unmaterialized);
                }
                let mle =
                    DenseMle::new(self.values.clone()).map_err(|_| TableError::Unmaterialized)?;
                mle.evaluate(point).map_err(|_| TableError::Unmaterialized)
            }
        }
    }

    /// Absorb the table's STATEMENT (family tag + log_k) — the
    /// structured-families route; `Values` tables additionally absorb
    /// their values.
    pub fn absorb_statement(
        &self,
        transcript: &mut lattice_core::transcript::Transcript,
        label: &[u8],
    ) -> Result<(), lattice_core::transcript::TranscriptError> {
        let head = [
            self.family.tag().to_le_bytes(),
            (self.log_k as u64).to_le_bytes(),
        ]
        .concat();
        let mut buf = head;
        if self.family == TableFamily::Values {
            for v in &self.values {
                buf.extend_from_slice(&v.to_canonical_u64().to_le_bytes());
            }
        }
        transcript.append_bytes(label, &buf)
    }
}

/// Errors of the structured-table layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableError {
    Shape {
        expected: usize,
        got: usize,
    },
    /// A `Values` table was evaluated before materialization.
    Unmaterialized,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(bits: &[u64]) -> Vec<Goldilocks> {
        bits.iter().map(|&b| Goldilocks::from_u64(b)).collect()
    }

    #[test]
    fn identity_closed_form_matches_dense_mle() {
        for log_k in [3usize, 5, 8, 12] {
            let t = StructuredTable::new(TableFamily::Identity, log_k);
            let values = t.values();
            let mle = DenseMle::new(values).ok().unwrap();
            // Random-ish points: binary fractions of a counter.
            for seed in 0..8u64 {
                let point: Vec<Goldilocks> = (0..log_k)
                    .map(|i| Goldilocks::from_u64((seed >> i) & 1))
                    .collect();
                let closed = t.eval_mle(&point).ok().unwrap();
                let dense = mle.evaluate(&point).ok().unwrap();
                assert_eq!(closed, dense, "identity m={log_k} seed={seed}");
            }
        }
    }

    #[test]
    fn pow2_closed_form_matches_dense_mle() {
        for log_k in [4usize, 6] {
            let t = StructuredTable::new(TableFamily::Pow2, log_k);
            let values = t.values();
            let mle = DenseMle::new(values).ok().unwrap();
            for seed in 0..8u64 {
                let point: Vec<Goldilocks> = (0..log_k)
                    .map(|i| Goldilocks::from_u64((seed >> i) & 1))
                    .collect();
                let closed = t.eval_mle(&point).ok().unwrap();
                let dense = mle.evaluate(&point).ok().unwrap();
                assert_eq!(closed, dense, "pow2 m={log_k} seed={seed}");
            }
        }
    }

    #[test]
    fn values_table_roundtrip() {
        let vals: Vec<Goldilocks> = (0..16u64)
            .map(|i| Goldilocks::from_u64(i * 7 + 3))
            .collect();
        let t = StructuredTable::from_values(vals.clone(), 4).ok().unwrap();
        let mle = DenseMle::new(vals).ok().unwrap();
        let point = pt(&[1, 0, 1, 1]);
        assert_eq!(
            t.eval_mle(&point).ok().unwrap(),
            mle.evaluate(&point).ok().unwrap()
        );
        // Wrong arity rejected.
        assert!(t.eval_mle(&pt(&[1, 0])).is_err());
        // Bad construction rejected.
        assert!(StructuredTable::from_values(vec![Goldilocks::ZERO], 4).is_err());
    }

    #[test]
    fn absorb_statement_distinguishes_families() {
        use lattice_core::transcript::Transcript;
        let t1 = StructuredTable::new(TableFamily::Identity, 8);
        let t2 = StructuredTable::new(TableFamily::Pow2, 8);
        let mut a = Transcript::new_default(b"tbl");
        let mut b = Transcript::new_default(b"tbl");
        t1.absorb_statement(&mut a, b"tbl").ok().unwrap();
        t2.absorb_statement(&mut b, b"tbl").ok().unwrap();
        let c1 = a.challenge_field(b"chal").ok().unwrap();
        let c2 = b.challenge_field(b"chal").ok().unwrap();
        assert_ne!(c1, c2);
    }
}
