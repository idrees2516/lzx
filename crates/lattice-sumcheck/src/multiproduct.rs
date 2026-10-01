//! Multiproduct evaluation in evaluation form (ePrint 2026/587 §4,
//! Procedures 1–2; the engine behind the `OptLinearTime` prover and the
//! window grids of §5).
//!
//! Given `n` multilinear factors over the first `v` window variables (each
//! a `2^v`-entry boolean-grid table, first variable outermost), compute
//! the product's evaluations on the extended grid `U(n+1)^v` where
//! `U(k) = {∞, 0, 1, …, k−1}` (side `k+1` per axis): the ∞ slot per axis
//! carries the polynomial's degree-k leading coefficient (Lemma 2.2's
//! interpolation-with-∞), and every integer slot `0..n` the round
//! messages need is present (the product has degree n, so U(n+1)
//! over-determines it by one point — harmless and convenient).
//!
//! * **Procedure 1 (MultiProductEval)**: split the factors in half,
//!   recursively evaluate each half-product on its own grid, extrapolate
//!   both to the full grid (Procedure 2), and multiply point-wise. The
//!   big-by-big count obeys `a(n) = a(⌊n/2⌋) + a(⌈n/2⌉) + (n+2)^v` —
//!   `Θ(n^v)` for fixed `v ≥ 2` and `O(n log n)` for `v = 1`, versus the
//!   naive `(n+1)(n−1)`-per-point baseline.
//! * **Procedure 2 (MultiExtrapolate)**: axis-by-axis univariate
//!   extrapolation via the shifted-evaluation recurrence — exclusively
//!   small-by-big multiplications and additions.
//!
//! Grid layout: side `n+2` per axis; axis digit 0 = ∞, digit `i+1` = the
//! integer `i` (i ≤ n); axis 0 (the first / most-significant window
//! variable) is the outermost stride.

use crate::extrapolate::extrapolate_in_place;
use lattice_core::Goldilocks;

/// Instrumentation: big-by-big multiplication count (the dominant cost).
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductStats {
    pub bb_mults: u64,
    pub sb_mults: u64,
}

/// Expand one factor's boolean table (`2^v` entries) to the single-factor
/// grid `U(1)^v` (side 2 per axis): per axis, the pair `(lo, hi)` becomes
/// `(hi−lo, lo)` — the ∞ slot holds the degree-1 leading coefficient.
/// Axis conversion runs innermost-first (the layout keeps the first
/// window variable outermost throughout).
fn bool_table_to_grid(table: &[Goldilocks], v: usize) -> Vec<Goldilocks> {
    let mut out = table.to_vec();
    let mut stride = 2usize; // covers the axes processed so far (innermost first)
    for _ in 0..v {
        let half = stride / 2;
        let mut next = vec![Goldilocks::ZERO; out.len()];
        for block in 0..(out.len() / stride) {
            for j in 0..half {
                let lo = out[block * stride + j];
                let hi = out[block * stride + half + j];
                next[block * stride + j] = hi.sub(&lo);
                next[block * stride + half + j] = lo;
            }
        }
        out = next;
        stride *= 2;
    }
    out
}

/// Multivariate extrapolation (Procedure 2): extend a grid with per-axis
/// side `k + 2` (∞ + integers `0..k`) to side `h + 2` (∞ + `0..h`),
/// axis-by-axis. Every line along an axis is extrapolated with the
/// shifted recurrence; cost per line: `k·(h−k)` small multiplications.
pub fn multi_extrapolate(
    grid: &mut Vec<Goldilocks>,
    v: usize,
    k: usize,
    h: usize,
    stats: &mut ProductStats,
) {
    if h <= k {
        return;
    }
    let mut sides: Vec<usize> = vec![k + 1; v];
    for a in 0..v {
        let old_side = sides[a];
        let new_side = h + 1;
        let outer: usize = sides[..a].iter().product();
        let inner: usize = sides[a + 1..].iter().product();
        let mut next = vec![Goldilocks::ZERO; outer * new_side * inner];
        for o in 0..outer {
            for i in 0..inner {
                // The line along axis a: entries at (o, s, i).
                let mut line: Vec<Goldilocks> = Vec::with_capacity(old_side);
                for s in 0..old_side {
                    line.push(grid[o * old_side * inner + s * inner + i]);
                }
                extrapolate_in_place(&mut line, k, h);
                stats.sb_mults += (k * (h - k)) as u64;
                for (s, &val) in line.iter().enumerate() {
                    next[o * new_side * inner + s * inner + i] = val;
                }
            }
        }
        *grid = next;
        sides[a] = new_side;
    }
}

/// MultiProductEval (Procedure 1): the product of `tables.len()` factors
/// (each a `2^v` boolean-grid table) on the grid `{∞, 0..n}^v` where
/// `n = tables.len()` is the product's per-variable degree.
///
/// Returns `(grid, stats)`.
pub fn multi_product_eval(tables: &[Vec<Goldilocks>], v: usize) -> (Vec<Goldilocks>, ProductStats) {
    let mut stats = ProductStats::default();
    let mut grid = product_recursive(tables, v, &mut stats);
    // The recursion returns the degree-faithful U(n) grid; extend one step
    // to U(n+1) so every message integer 0..=n is present.
    let n = tables.len();
    multi_extrapolate(&mut grid, v, n, n + 1, &mut stats);
    debug_assert_eq!(grid.len(), (n + 2).pow(v as u32));
    (grid, stats)
}

fn product_recursive(
    tables: &[Vec<Goldilocks>],
    v: usize,
    stats: &mut ProductStats,
) -> Vec<Goldilocks> {
    let n = tables.len();
    debug_assert!(n >= 1);
    if n == 1 {
        return bool_table_to_grid(&tables[0], v);
    }
    let m = n / 2;
    let (left, right) = tables.split_at(m);
    let mut lg = product_recursive(left, v, stats);
    let mut rg = product_recursive(right, v, stats);
    // The half-products (degrees m and n−m) arrive as U(m) / U(n−m) grids;
    // extrapolate both to U(n) and multiply point-wise. The ∞ slots
    // compose: (deg-m lead)·(deg-(n−m) lead) = the degree-n lead.
    multi_extrapolate(&mut lg, v, m, n, stats);
    multi_extrapolate(&mut rg, v, n - m, n, stats);
    // Point-wise product (big-by-big).
    stats.bb_mults += lg.len() as u64;
    lg.iter().zip(rg.iter()).map(|(x, y)| x.mul(y)).collect()
}

/// The flat index of the grid point whose axis-`a` coordinate is
/// `coord[a]` (0 = ∞, `i+1` = integer `i`), given the per-axis side.
#[inline]
pub fn grid_index(sides: &[usize], coord: &[usize]) -> usize {
    let mut flat = 0usize;
    for (a, &c) in coord.iter().enumerate() {
        flat = flat * sides[a] + c;
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::Goldilocks;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// The MLE of a 2^v boolean table at an integer point
    /// `x ∈ {0..n}^v` (binding every variable to its integer coordinate;
    /// first variable outermost).
    fn mle_at(table: &[Goldilocks], v: usize, point: &[u64]) -> Goldilocks {
        let mut cur = table.to_vec();
        let mut len = cur.len();
        for &x in point.iter().take(v) {
            let half = len / 2;
            let xf = fe(x);
            for j in 0..half {
                cur[j] = cur[j].add(&cur[half + j].sub(&cur[j]).mul(&xf));
            }
            cur.truncate(half);
            len = half;
        }
        debug_assert_eq!(cur.len(), 1);
        cur[0]
    }

    #[test]
    fn univariate_product_matches_naive() {
        for n in [2usize, 3, 4, 5, 7, 8, 9, 16] {
            let tables: Vec<Vec<Goldilocks>> = (0..n)
                .map(|k| vec![fe(1 + ((k * 7) % 13) as u64), fe(2 + ((k * 11) % 17) as u64)])
                .collect();
            let (grid, stats) = multi_product_eval(&tables, 1);
            assert_eq!(grid.len(), n + 2);
            // Index 0 = ∞ = Π (hi−lo) (the product's leading coefficient).
            let mut want_inf = fe(1);
            for t in &tables {
                want_inf = want_inf.mul(&t[1].sub(&t[0]));
            }
            assert_eq!(grid[0], want_inf, "n={n}: ∞ entry");
            // Index i+1 = Π_k p_k(i) for i ∈ 0..=n.
            for x in 0..=n as u64 {
                let mut want = fe(1);
                for t in &tables {
                    let val = t[0].add(&t[1].sub(&t[0]).mul(&fe(x)));
                    want = want.mul(&val);
                }
                assert_eq!(grid[(x + 1) as usize], want, "n={n}, x={x}");
            }
            // The naive per-point evaluation costs (n+1)(n−1) big mults;
            // ours is strictly below for n ≥ 4 (Table 2's ratio).
            if n >= 4 {
                assert!(
                    stats.bb_mults < ((n + 1) * (n - 1)) as u64,
                    "n={n}: bb {} not better than naive {}",
                    stats.bb_mults,
                    (n + 1) * (n - 1)
                );
            }
        }
    }

    #[test]
    fn multivariate_product_matches_naive() {
        let (v, n) = (2usize, 3usize);
        let tables: Vec<Vec<Goldilocks>> = (0..n)
            .map(|k| {
                (0..4)
                    .map(|i| fe((k * 13 + i * 7 + 1) as u64 % 101))
                    .collect()
            })
            .collect();
        let (grid, _stats) = multi_product_eval(&tables, v);
        let side = n + 2;
        assert_eq!(grid.len(), side * side);
        for x in 0..=n as u64 {
            for y in 0..=n as u64 {
                let mut want = fe(1);
                for t in &tables {
                    want = want.mul(&mle_at(t, v, &[x, y]));
                }
                let got = grid[(x as usize + 1) * side + (y as usize + 1)];
                assert_eq!(got, want, "v=2 n=3 at ({x},{y})");
            }
        }
        // ∞ on axis 0, integer on axis 1: the leading coefficient in var0
        // of the product, evaluated at y.
        for y in 0..=n as u64 {
            let mut want = fe(1);
            for t in &tables {
                let lo = mle_at(&t[..2], 1, &[y]);
                let hi = mle_at(&t[2..], 1, &[y]);
                want = want.mul(&hi.sub(&lo));
            }
            let got = grid[y as usize + 1];
            assert_eq!(got, want, "v=2 n=3 ∞-axis0 at y={y}");
        }
        // ∞ on both axes: the mixed leading coefficient.
        {
            let mut want = fe(1);
            for t in &tables {
                let lo = t[0];
                let hi = t[1];
                let delta_inner = hi.sub(&lo);
                let lo2 = t[2];
                let hi2 = t[3];
                let delta_inner2 = hi2.sub(&lo2);
                want = want.mul(&delta_inner2.sub(&delta_inner));
            }
            assert_eq!(grid[0], want, "v=2 n=3 ∞-both");
        }
    }

    #[test]
    fn bb_counts_v1_follow_closed_form() {
        // v = 1: the combine at a node of size m costs (m+1) point-wise
        // multiplications over U(m); for powers of two the recursion tree
        // is exact: Σ over levels of (n/level)·(level+1) = n·(log2(n)+1) −
        // hmm — the final message extension adds no bb. Cross-check
        // against the direct recurrence a(n) = a(n/2)·2 + (n+1).
        for n in [2usize, 4, 8, 16] {
            let tables: Vec<Vec<Goldilocks>> =
                (0..n).map(|k| vec![fe(k as u64 + 1), fe(k as u64 + 2)]).collect();
            let (_, stats) = multi_product_eval(&tables, 1);
            // Direct recurrence for powers of two.
            let mut exact = 0u64;
            let mut level = n;
            while level > 1 {
                exact += (n / level) as u64 * (level + 1) as u64;
                level /= 2;
            }
            assert_eq!(stats.bb_mults, exact, "n={n}");
            // And the naive per-point baseline is strictly worse for n ≥ 4.
            if n >= 4 {
                assert!(stats.bb_mults < ((n + 1) * (n - 1)) as u64, "n={n}");
            }
        }
    }
}
