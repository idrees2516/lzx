//! The SuperNeo embedding machinery (§2.4.1) that connects field vectors,
//! ring vectors, structural matrices and multilinear extensions.
//!
//! * **Coefficient embedding**: a field vector z ∈ F^{nF} partitions into
//!   d-sized blocks z = [z¹, ..., z^{nR}] and embeds as a ring vector
//!   z ∈ R_F^{nR}; the embedding is norm-preserving.
//! * **Matrix lift**: M_j ∈ F^{m×nF} lifts to M̄_j ∈ R_F^{m×nR} by embedding
//!   each row with the **inner-product transform** (§2.4.1: ct(ā·b) =
//!   ⟨a, b⟩), so that ct(M̄_j·z) = M_j·z — "all m row-wise inner products
//!   are thus computed in parallel by a single ring product".
//! * **Ring-valued MLEs**: for the ring vector u = M̄_j z ∈ R_F^m, the
//!   multilinear extension (Σ_x eq(X,x)·u_x) is computed coefficient-wise
//!   after lifting to R_K; ct(M̄_j z(r)) = (M_j z)^(r) — the evaluation
//!   hint semantics of Definition 2.22.
//! * **Field MLEs**: ẑ(X⃗) for z ∈ F^{nF} over log2(nF) variables — the
//!   norm-check polynomial's base.
//! * **eq-array binding** for the Sum-Check engine: the standard fold
//!   new[s] = (1−r)·arr[0s] + r·arr[1s] over K.

use crate::fp::Fq;
use crate::fq2::K;
use crate::ring::Poly;
use crate::rk::PolyK;

/// A field vector in F^{nF} — the native witness representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldVec(pub Vec<Fq>);

impl FieldVec {
    pub fn n(&self) -> usize {
        self.0.len()
    }

    pub fn norm_inf(&self) -> i64 {
        self.0.iter().map(|c| c.sym()).map(|s| s.abs()).max().unwrap_or(0)
    }

    /// Embed into R_F^{nR} (coefficient embedding, §2.4.1).
    pub fn to_ring(&self, d: usize) -> Vec<Poly> {
        debug_assert_eq!(self.0.len() % d, 0);
        let nr = self.0.len() / d;
        (0..nr)
            .map(|i| Poly(self.0[i * d..(i + 1) * d].to_vec()))
            .collect()
    }

    /// The multilinear extension evaluation ẑ(r) over log2(n) variables
    /// (n a power of two).
    pub fn mle_eval(&self, r: &[K]) -> K {
        let n = self.0.len();
        debug_assert!(n.is_power_of_two());
        debug_assert_eq!(r.len(), n.trailing_zeros() as usize);
        // Iterative eq-based folding: maintain K-vector of eq weights.
        let mut cur: Vec<K> = self.0.iter().map(|&c| K::from_fp(c)).collect();
        let mut len = n;
        for ri in r {
            let half = len / 2;
            let mut next = Vec::with_capacity(half);
            for i in 0..half {
                // arr index bit: high half = X_i = 1
                let lo = cur[i];
                let hi = cur[half + i];
                next.push(
                    K::ONE.sub(ri).mul(&lo).add(&ri.mul(&hi)),
                );
            }
            cur = next;
            len = half;
        }
        cur[0]
    }

    pub fn uniform(n: usize, seed: &[u8], counter: &mut u64) -> FieldVec {
        FieldVec((0..n).map(|_| Fq::uniform(seed, counter)).collect())
    }

    /// Small-bounded sampler: each coordinate uniform in [−(b−1), b−1].
    pub fn small_b(n: usize, b: i64, seed: &[u8], counter: &mut u64) -> FieldVec {
        let spread = (2 * b - 1) as u64;
        FieldVec(
            (0..n)
                .map(|_| {
                    let v = Fq::uniform(seed, counter);
                    let m = (v.0 % spread) as i64;
                    Fq::from_i64(m - (b - 1))
                })
                .collect(),
        )
    }
}

/// Extract a field vector from a ring vector (inverse of to_ring).
pub fn from_ring(zr: &[Poly]) -> FieldVec {
    let d = zr.first().map(|p| p.d()).unwrap_or(0);
    let mut out = Vec::with_capacity(zr.len() * d);
    for p in zr {
        out.extend_from_slice(&p.0);
    }
    FieldVec(out)
}

/// A structural matrix M_j ∈ F^{m×nF} with its ring lift M̄_j ∈ R_F^{m×nR}
/// (rows embedded through the inner-product transform so that
/// ct(M̄_j·z)_row = ⟨(M_j)_row, z⟩).
#[derive(Clone, Debug)]
pub struct StructMatrix {
    /// Field rows, m × nF (row-major).
    pub rows: Vec<Vec<Fq>>,
    /// The ring lift: m ring elements per column-block — stored as
    /// m × nR ring elements (row i is the transformed embedding of
    /// rows[i] as nR ring elements).
    pub lift: Vec<Vec<Poly>>, // lift[row][col_block]
    pub m: usize,
    pub nf: usize,
}

impl StructMatrix {
    /// Build from field rows, applying the inner-product transform per row.
    pub fn new(rows: Vec<Vec<Fq>>, d: usize) -> StructMatrix {
        let m = rows.len();
        let nf = rows.first().map(|r| r.len()).unwrap_or(0);
        debug_assert!(nf % d == 0);
        let nr = nf / d;
        let mut lift = Vec::with_capacity(m);
        for row in &rows {
            // Split the row into nR blocks of d coefficients, embed each
            // block, and apply the transform to each block.
            let mut ring_row = Vec::with_capacity(nr);
            for c in 0..nr {
                let block: Vec<Fq> = row[c * d..(c + 1) * d].to_vec();
                let p = Poly(block);
                ring_row.push(p.inner_transform());
            }
            lift.push(ring_row);
        }
        StructMatrix {
            rows,
            lift,
            m,
            nf,
        }
    }

    /// The identity matrix M_1 = I_m (Remark 4.1.(3)).
    pub fn identity(m: usize, d: usize) -> StructMatrix {
        let mut rows = vec![vec![Fq::ZERO; m]; m];
        for i in 0..m {
            rows[i][i] = Fq::ONE;
        }
        StructMatrix::new(rows, d)
    }

    /// Field matrix-vector product M·z ∈ F^m.
    pub fn mul_field(&self, z: &FieldVec) -> Vec<Fq> {
        self.rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(z.0.iter())
                    .fold(Fq::ZERO, |acc, (a, b)| acc.add(&a.mul(b)))
            })
            .collect()
    }

    /// The ring-lifted product M̄·z ∈ R_F^m (z a ring vector of length nR).
    /// ct((M̄z)_row) = ⟨(M)_row, z⟩ — the packed inner products.
    pub fn mul_ring(&self, zr: &[Poly]) -> Vec<Poly> {
        debug_assert_eq!(
            zr.len(),
            self.lift.first().map(|r| r.len()).unwrap_or(0),
            "z must be a ring vector of length nR"
        );
        let mut out = Vec::with_capacity(self.m);
        for ring_row in &self.lift {
            let mut acc = Poly::zero(zr.first().map(|p| p.d()).unwrap_or(0));
            for (mp, zp) in ring_row.iter().zip(zr.iter()) {
                acc.add_assign(&mp.mul(zp));
            }
            out.push(acc);
        }
        out
    }
}

/// The ring-valued multilinear extension of a ring vector u ∈ R_F^m
/// (m a power of two), maintained as a cube array of R_K elements with
/// K-binding.
#[derive(Clone, Debug)]
pub struct RingMle {
    pub cube: Vec<PolyKRef>,
    pub log_len: usize,
}

/// A lightweight R_K value for cube arrays (avoids re-allocating d-sized
/// vectors per entry: store the R_F pair directly).
pub type PolyKRef = PolyK;

impl RingMle {
    pub fn from_ring_vec(u: &[Poly]) -> RingMle {
        let m = u.len();
        debug_assert!(m.is_power_of_two());
        RingMle {
            cube: u.iter().map(|p| PolyK::from_poly(p.clone())).collect(),
            log_len: m.trailing_zeros() as usize,
        }
    }

    /// Bind the next variable to r ∈ K: new[s] = (1−r)·cube[0s] + r·cube[1s].
    pub fn bind(&mut self, r: &K) {
        let half = self.cube.len() / 2;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            let lo = &self.cube[i];
            let hi = &self.cube[half + i];
            // (1−r)·lo + r·hi with K-scalar scaling of R_K elements.
            next.push(
                lo.scale_k(&K::ONE.sub(r)).add(&hi.scale_k(r)),
            );
        }
        self.cube = next;
        self.log_len -= 1;
    }

    /// Evaluate at a full point (binds a copy).
    pub fn eval(&self, point: &[K]) -> PolyK {
        let mut copy = self.clone();
        for r in point {
            copy.bind(r);
        }
        copy.cube[0].clone()
    }

    /// The full evaluation sum over the suffix cube (for Sum-Check round
    /// polys): Σ_s EQ[s]·cube[s] with EQ the bound eq-array.
    pub fn weighted_sum(&self, eq_arr: &[K]) -> PolyK {
        let mut acc = PolyK::zero(self.cube.first().map(|c| c.d()).unwrap_or(0));
        for (val, w) in self.cube.iter().zip(eq_arr.iter()) {
            if w.is_zero() {
                continue;
            }
            acc.add_assign(&val.scale_k(w));
        }
        acc
    }
}



/// eq-array utilities over K (for the Sum-Check engine).
pub struct EqArray;

impl EqArray {
    /// The full eq array over {0,1}^ℓ for the point α: eq[x] = eq(x, α).
    pub fn full(log_len: usize, alpha: &[K]) -> Vec<K> {
        let n = 1usize << log_len;
        let mut arr = vec![K::ONE; n];
        for (i, a) in alpha.iter().enumerate().take(log_len) {
            let half = n >> (i + 1);
            for s in 0..n {
                let bit = (s / half) % 2;
                arr[s] = arr[s].mul(&if bit == 1 { *a } else { K::ONE.sub(a) });
            }
        }
        arr
    }

    /// Bind an eq array at r ∈ K: new[s] = (1−r)·arr[0s] + r·arr[1s].
    pub fn bind(arr: &[K], r: &K) -> Vec<K> {
        let half = arr.len() / 2;
        (0..half)
            .map(|i| K::ONE.sub(r).mul(&arr[i]).add(&r.mul(&arr[half + i])))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_embedding_roundtrip() {
        let mut ctr = 0u64;
        let z = FieldVec::small_b(64, 2, b"emb", &mut ctr);
        let zr = z.to_ring(8);
        assert_eq!(zr.len(), 8);
        assert_eq!(from_ring(&zr), z);
        assert_eq!(zr[0].norm_inf(), {
            let m = z.0[..8].iter().map(|c| c.sym().abs()).max().unwrap();
            m
        });
    }

    #[test]
    fn matrix_lift_packs_inner_products() {
        // ct((M̄z)_row) = ⟨M_row, z⟩ for random M, z.
        let d = 4;
        let m = 16;
        let nf = 32;
        let mut ctr = 0u64;
        let rows: Vec<Vec<Fq>> = (0..m)
            .map(|_| (0..nf).map(|_| Fq::uniform(b"ml", &mut ctr)).collect())
            .collect();
        let mat = StructMatrix::new(rows.clone(), d);
        let z = FieldVec::small_b(nf, 2, b"mlz", &mut ctr);
        let zr = z.to_ring(d);
        let prod = mat.mul_ring(&zr);
        let field_prod = mat.mul_field(&z);
        for row in 0..m {
            assert_eq!(
                prod[row].ct(),
                field_prod[row],
                "row {row}: ct(M̄z) = Mz"
            );
        }
    }

    #[test]
    fn ring_mle_binding_matches_direct_eval() {
        let d = 4;
        let m = 8;
        let mut ctr = 0u64;
        let u: Vec<Poly> = (0..m).map(|_| Poly::uniform(d, b"rm", &mut ctr)).collect();
        let r: Vec<K> = (0..3).map(|_| K::uniform(b"rmr", &mut ctr)).collect();
        let mut mle = RingMle::from_ring_vec(&u);
        for ri in &r {
            mle.bind(ri);
        }
        // Direct: Σ_x eq(x, r)·u_x
        let mut expect = Poly::zero(d);
        for (i, p) in u.iter().enumerate() {
            let mut w = K::ONE;
            for (bit_idx, ri) in r.iter().enumerate() {
                let bit = (i >> (2 - bit_idx)) & 1;
                w = w.mul(&if bit == 1 { *ri } else { K::ONE.sub(ri) });
            }
            expect.add_assign(&p.scale(&w.0));
        }
        let got = mle.cube[0].clone();
        // got is R_K = w·u summed — with K weights the b-part may be nonzero
        // when w has b-components; compare coefficient-wise against the
        // a-part of the manual sum plus the ν-mixed b-part.
        let manual_k = PolyK::from_poly(expect);
        // Full manual with K weights:
        let mut expect_k = PolyK::zero(d);
        for (i, p) in u.iter().enumerate() {
            let mut w = K::ONE;
            for (bit_idx, ri) in r.iter().enumerate() {
                let bit = (i >> (2 - bit_idx)) & 1;
                w = w.mul(&if bit == 1 { *ri } else { K::ONE.sub(ri) });
            }
            expect_k.add_assign(&PolyK::from_poly(p.clone()).scale_k(&w));
        }
        assert_eq!(got, expect_k);
        let _ = manual_k;
    }

    #[test]
    fn field_mle_eval_matches_eq_sum() {
        let mut ctr = 0u64;
        let z = FieldVec::small_b(16, 2, b"fm", &mut ctr);
        let r: Vec<K> = (0..4).map(|_| K::uniform(b"fmr", &mut ctr)).collect();
        let got = z.mle_eval(&r);
        let mut expect = K::ZERO;
        for (i, &c) in z.0.iter().enumerate() {
            let mut w = K::ONE;
            for (bit_idx, ri) in r.iter().enumerate() {
                let bit = (i >> (3 - bit_idx)) & 1;
                w = w.mul(&if bit == 1 { *ri } else { K::ONE.sub(ri) });
            }
            expect = expect.add(&w.mul(&K::from_fp(c)));
        }
        assert_eq!(got, expect);
    }

    #[test]
    fn eq_array_binds_correctly() {
        let alpha: Vec<K> = (0..3).map(|i| K::from_fp(Fq::new(3 + i as u64 * 5))).collect();
        let arr = EqArray::full(3, &alpha);
        for (i, &v) in arr.iter().enumerate() {
            let mut w = K::ONE;
            for (bit_idx, a) in alpha.iter().enumerate() {
                let bit = (i >> (2 - bit_idx)) & 1;
                w = w.mul(&if bit == 1 { *a } else { K::ONE.sub(a) });
            }
            assert_eq!(v, w);
        }
        // Σ_x eq(x, α) = 1
        let total = arr.iter().fold(K::ZERO, |a, b| a.add(b));
        assert_eq!(total, K::ONE);
    }
}
