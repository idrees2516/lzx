//! The direct-boundary width theory, executable (the paper's §4.1–4.4):
//! the challenge-coefficient space `Va = span{a_i} ∪ {a_i a_j}`, the
//! separation condition (diagonal independence modulo `Va`), the exact
//! width `m_min = r·dim Va` (Theorem 4.1), the `k`-dimensional lower bound
//! under separation (Lemma 4.2), and the scaled-Cauchy attainment
//! `dim Va = k` (Corollary 4.3) — pinned by explicit linear algebra over
//! `K` on the concrete challenge families.

use crate::cauchy::{CauchyParams, QuadraticMap};
use crate::field_k::K4;

/// A function on the challenge support `C`, stored by its values.
#[derive(Clone, Debug, PartialEq)]
pub struct FnOnC {
    pub values: Vec<K4>,
}

/// The rank of a set of `K^m`-valued vectors (Gaussian elimination over
/// `K`, exact).
pub fn k4_rank(rows: &[Vec<K4>]) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let m = rows[0].len();
    let mut mat: Vec<Vec<K4>> = rows.to_vec();
    let mut rank = 0;
    for col in 0..m {
        // Find a pivot at or below `rank`.
        let mut piv = None;
        for r in rank..mat.len() {
            if !mat[r][col].is_zero() {
                piv = Some(r);
                break;
            }
        }
        let Some(p) = piv else { continue };
        mat.swap(rank, p);
        let inv = mat[rank][col].inv().expect("pivot invertible");
        for c in 0..m {
            mat[rank][c] = mat[rank][c].scale(&inv);
        }
        for r in 0..mat.len() {
            if r != rank && !mat[r][col].is_zero() {
                let f = mat[r][col];
                for c in 0..m {
                    mat[r][c] = mat[r][c].sub(&mat[rank][c].mul(&f));
                }
            }
        }
        rank += 1;
        if rank == mat.len() {
            break;
        }
    }
    rank
}

/// The challenge family analysis on a concrete support `C`.
pub struct BoundaryAnalysis {
    /// `dim Va` — the dimension of `span{a_i} ∪ {a_i a_j}` as functions on C.
    pub dim_va: usize,
    /// The separation condition holds: `{1, a_i²}` independent modulo `Va`,
    /// i.e. `dim span{1, a_i², a_i, a_i a_j} = 2k + 1`.
    pub separated: bool,
    /// `r = dim Y_B` — the mixed-output dimension of the polarized map.
    pub r: usize,
    /// The exact minimal direct-boundary width `m_min = r · dim Va`.
    pub m_min: usize,
    /// The raw count of visible mixed terms: `k + C(k, 2)`.
    pub visible_terms: usize,
}

/// Analyze a Cauchy family on the support `C` (values off the poles).
pub fn analyze(params: &CauchyParams, support_c: &[K4], q_map: &QuadraticMap) -> BoundaryAnalysis {
    let k = params.k();
    assert!(support_c.len() > 2 * k, "|C| > 2k required");
    // The functions a_i on C.
    let a_fns: Vec<Vec<K4>> = support_c
        .iter()
        .map(|&c| {
            (0..k)
                .map(|i| params.a_i(i, &c).expect("c off the poles"))
                .collect()
        })
        .collect();
    // As rows: function i's values across C.
    let a_rows: Vec<Vec<K4>> = (0..k)
        .map(|i| a_fns.iter().map(|v| v[i]).collect())
        .collect();
    // Pairwise products a_i·a_j.
    let mut pair_rows = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            pair_rows.push(a_fns.iter().map(|v| v[i].mul(&v[j])).collect::<Vec<K4>>());
        }
    }
    let mut va_rows = a_rows.clone();
    va_rows.extend(pair_rows.iter().cloned());
    let dim_va = k4_rank(&va_rows);

    // The separation condition: dim span{1} ∪ {a_i²} ∪ Va = 2k + 1.
    let one_row = vec![K4::ONE; support_c.len()];
    let sq_rows: Vec<Vec<K4>> = (0..k)
        .map(|i| a_fns.iter().map(|v| v[i].mul(&v[i])).collect())
        .collect();
    let mut full = vec![one_row];
    full.extend(sq_rows);
    full.extend(va_rows);
    let dim_full = k4_rank(&full);
    let separated = dim_full == 2 * k + 1;

    // r = dim Y_B: the span of {B(e_l, e_m)} over the standard basis
    // (the polarization is bilinear, so its image is spanned by the
    // basis-pair images).
    let s = q_map.s;
    let mut images = Vec::new();
    for l in 0..s {
        let mut e_l = vec![K4::ZERO; s];
        e_l[l] = K4::ONE;
        for m in 0..s {
            let mut e_m = vec![K4::ZERO; s];
            e_m[m] = K4::ONE;
            images.push(q_map.polarize(&e_l, &e_m));
        }
    }
    let r = k4_rank(&images);

    BoundaryAnalysis {
        dim_va,
        separated,
        r,
        m_min: r * dim_va,
        visible_terms: k + k * (k - 1) / 2,
    }
}

/// The mixed-output dimension of the polarized map for the R1CS shape —
/// the closed form `r = y` when `A`, `B` have full constraint rank
/// (verified numerically by [`analyze`]).
pub fn mixed_output_dim(q_map: &QuadraticMap) -> usize {
    let s = q_map.s;
    let mut images = Vec::new();
    for l in 0..s {
        let mut e_l = vec![K4::ZERO; s];
        e_l[l] = K4::ONE;
        for m in 0..s {
            let mut e_m = vec![K4::ZERO; s];
            e_m[m] = K4::ONE;
            images.push(q_map.polarize(&e_l, &e_m));
        }
    }
    k4_rank(&images)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(k: usize, extra: usize) -> Vec<K4> {
        // Distinct points avoiding the poles 1..=k.
        (0..2 * k + 1 + extra)
            .map(|i| {
                K4::from_coeffs([
                    (i as u64 * 997 + 5000 + k as u64 * 7 + 1) % Q48_,
                    i as u64 * 31 + 3,
                    7,
                    11,
                ])
            })
            .collect()
    }
    use crate::field_k::Q48 as Q48_;

    #[test]
    fn cauchy_family_attains_exact_width() {
        // Corollary 4.3: for the scaled Cauchy family on |C| > 2k,
        // dim Va = k and the separation condition holds.
        for k in [2, 3, 4, 8] {
            let params = CauchyParams::paper(k);
            let q_map = QuadraticMap::benchmark(4, 3, 17 + k as u64);
            let c = support(k, 2);
            let analysis = analyze(&params, &c, &q_map);
            assert_eq!(analysis.dim_va, k, "dim Va = k at arity {k}");
            assert!(analysis.separated, "separation at arity {k}");
            assert_eq!(analysis.m_min, analysis.r * k);
            // The visible-term count is quadratic; the width is linear.
            assert!(analysis.visible_terms > analysis.dim_va || k == 1);
        }
    }

    #[test]
    fn challenge_support_lower_bound() {
        // The paper's §4.3 counting: whenever the separation condition
        // holds on the support, |C| ≥ dim Va + k + 1 (so ≥ 2k + 1 at the
        // Cauchy width dim Va = k).
        let k = 4;
        let params = CauchyParams::paper(k);
        let q_map = QuadraticMap::benchmark(4, 3, 5);
        let c = support(k, 2);
        let analysis = analyze(&params, &c, &q_map);
        assert!(analysis.separated);
        assert_eq!(analysis.dim_va, k);
        assert!(c.len() > analysis.dim_va + k);
        // The minimal support 2k+1 still separates.
        let c_min = support(k, 0);
        assert_eq!(c_min.len(), 2 * k + 1);
        let a_min = analyze(&params, &c_min, &q_map);
        assert!(a_min.separated);
        assert_eq!(a_min.dim_va, k);
    }

    #[test]
    fn linear_family_needs_more_state() {
        // A non-Cauchy family with dependent products — e.g. the power
        // family a_i(c) = c^i on a large support — has dim Va < the
        // visible-term count but larger than k (it does not attain the
        // Cauchy width). The boundary machinery measures it.
        let k = 4;
        let c: Vec<K4> = (0..2 * k + 5)
            .map(|i| K4::from_coeffs([(i as u64 * 1234 + 7) % Q48_, 3, 5, 7]))
            .collect();
        // a_i = c^i, i = 1..=k.
        let a_fns: Vec<Vec<K4>> = c
            .iter()
            .map(|&x| {
                let mut acc = K4::ONE;
                (0..k)
                    .map(|_| {
                        let v = acc;
                        acc = acc.mul(&x);
                        v
                    })
                    .collect::<Vec<K4>>()
            })
            .collect();
        let a_rows: Vec<Vec<K4>> = (0..k)
            .map(|i| a_fns.iter().map(|v| v[i]).collect())
            .collect();
        let mut pair_rows = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                pair_rows.push(a_fns.iter().map(|v| v[i].mul(&v[j])).collect::<Vec<K4>>());
            }
        }
        let mut va = a_rows.clone();
        va.extend(pair_rows);
        let dim_va = k4_rank(&va);
        // The power family has products a_i a_j = c^{i+j} outside the
        // linear span of {c^1..c^k} — dim Va > k for k ≥ 2.
        assert!(dim_va > k, "power family does not attain the Cauchy width");
    }

    #[test]
    fn k4_rank_basics() {
        let r = k4_rank(&[
            vec![K4::ONE, K4::ZERO, K4::ONE],
            vec![K4::ZERO, K4::ONE, K4::ONE],
            vec![K4::ONE, K4::ONE, K4::ZERO],
        ]);
        assert_eq!(r, 3);
        // A dependent third row drops the rank.
        let r2 = k4_rank(&[
            vec![K4::ONE, K4::ZERO, K4::ONE],
            vec![K4::ZERO, K4::ONE, K4::ONE],
            vec![K4::ONE, K4::ONE, K4::from_coeffs([2, 0, 0, 0])],
        ]);
        assert_eq!(r2, 2);
        let r2 = k4_rank(&[
            vec![K4::ONE, K4::ONE],
            vec![K4::from_coeffs([2, 0, 0, 0]), K4::from_coeffs([2, 0, 0, 0])],
        ]);
        assert_eq!(r2, 1);
    }
}
