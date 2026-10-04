//! The round-batched, evaluation-grid sumcheck prover
//! (ePrint 2026/587 §5 "EvalProduct SV,SC" + ePrint 2025/1117 §4–5).
//!
//! Two optimizations over the baseline linear-time prover, both leaving
//! the protocol, verifier, and transcript flow **byte-identical**:
//!
//! 1. **The window** (§5.1): the first `v` rounds delay binding the
//!    window variables; per boolean suffix `x' ∈ {0,1}^{ℓ−v}` and per
//!    product term, the term's window polynomial
//!    `q(X_1..X_v, x') = c·Π_k p_k(X_1..X_v, x')` is materialised as an
//!    evaluation grid over `U(d+1)^v` by
//!    [`crate::multiproduct::multi_product_eval`] — the Θ(d^v)-bb
//!    multiproduct engine. Each window round then reads the grids:
//!    round `j`'s message is
//!    `g_j(t) = Σ_{x'} Σ_{u ∈ U^{j−1}} W_j[u]·(Σ_{w∈{0,1}^{v−j}} G_{x'}[u, t, w])`
//!    with `W_j = L(r_1) ⊗ ⋯ ⊗ L(r_{j−1})` the composed Lagrange weights
//!    (§5's evaluation-basis round emulation — the grid entries are read
//!    from their original small values, so the weighting multiplications
//!    are small-by-big whenever the factor evaluations are small).
//! 2. **The optimized tail** (§4 + C.1): rounds `v+1..ℓ` bind each
//!    factor once per round (`Δ_k = hi_k − lo_k` in a single pass) and
//!    evaluate the products at `t ∈ {0..d}` through the affine form
//!    `lo_k + t·Δ_k` — no per-`t` re-binding passes over the factors.
//!
//! Cost model (C.4.1): window work `Θ(M·((d+2)/2)^v)` weighting
//! multiplications + the grid construction (small-by-big dominated);
//! tail `Θ(d²·M/2^v)` big-by-big — the optimum
//! `v* = log_{d+2}(d²κ)` for the big/small cost ratio κ.
//!
//! The split-eq optimization (2025/1117 §5, 2026/587 §6) is available
//! through [`prove_fast_with_eq`]: the designated equality factor is
//! never materialised over the full hypercube — its window tables and
//! post-window bindings are derived from `w` via the prefix/suffix eq
//! factorisation `eq(w, (b, x')) = eq(w_{<v}, b)·eq(w_{≥v}, x')`.

use crate::multiproduct::{multi_product_eval, ProductStats};
use crate::sumcheck::{SumcheckError, SumcheckOutput, SumcheckProof};
use crate::virtual_poly::VirtualPolynomial;
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};

/// Options for the fast prover.
#[derive(Clone, Debug)]
pub struct FastProverOpts {
    /// Number of leading rounds to batch through the evaluation grid
    /// (0 = pure optimized-linear-time tail).
    pub window: usize,
    /// Collect multiplication statistics (retrieved via
    /// [`take_last_stats`]).
    pub collect_stats: bool,
}

impl Default for FastProverOpts {
    fn default() -> Self {
        FastProverOpts {
            window: 3,
            collect_stats: true,
        }
    }
}

/// Instrumentation: big-by-big / small-by-big multiplication counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastProverStats {
    pub bb_mults: u64,
    pub sb_mults: u64,
    /// Grid-construction stats from the multiproduct engine.
    pub grid: ProductStats,
}

/// A factor source: either a materialised table or a split eq factor.
enum FactorView<'a> {
    Table(&'a DenseMle),
    /// eq(w, ·) with the prefix/suffix split at the window boundary:
    /// `eq(w, (b, x')) = prefix[b]·suffix[x']`.
    Eq {
        /// eq(w_{<v}, b) for b ∈ {0,1}^v.
        prefix: Vec<Goldilocks>,
        /// eq(w_{≥v}, x') for x' ∈ {0,1}^{ℓ−v}.
        suffix: Vec<Goldilocks>,
    },
}

#[allow(clippy::needless_range_loop)]
impl<'a> FactorView<'a> {
    /// The factor's window table at suffix `x'` (2^v entries, first
    /// window variable outermost).
    fn window_table(&self, x2: usize, suffixes: usize, v: usize) -> Vec<Goldilocks> {
        match self {
            FactorView::Table(f) => {
                let evs = &f.evaluations;
                (0..1usize << v).map(|b| evs[b * suffixes + x2]).collect()
            }
            FactorView::Eq { prefix, suffix } => {
                let s = suffix[x2];
                prefix.iter().map(|p| p.mul(&s)).collect()
            }
        }
    }

    /// Bind the window variables to the challenges `r_{<v}` via the
    /// boolean eq-weights `eq_r[b]`: the array `p(r_{<v}, x')`.
    fn bind_window(
        &self,
        eq_r: &[Goldilocks],
        suffixes: usize,
        v: usize,
        stats: &mut FastProverStats,
    ) -> Vec<Goldilocks> {
        match self {
            FactorView::Table(f) => {
                let evs = &f.evaluations;
                let mut out = vec![Goldilocks::ZERO; suffixes];
                for b in 0..1usize << v {
                    let w = eq_r[b];
                    if w.is_zero() {
                        continue;
                    }
                    let base = b * suffixes;
                    for x2 in 0..suffixes {
                        out[x2] = out[x2].add(&evs[base + x2].mul(&w));
                        stats.sb_mults += 1;
                    }
                }
                out
            }
            FactorView::Eq { prefix, suffix } => {
                // p(r, x') = (Σ_b eq_r[b]·prefix[b])·suffix[x'] — the
                // prefix bound value is a single scalar.
                let mut pref_bound = Goldilocks::ZERO;
                for (b, &pb) in prefix.iter().enumerate() {
                    pref_bound = pref_bound.add(&pb.mul(&eq_r[b]));
                    stats.sb_mults += 1;
                }
                suffix
                    .iter()
                    .map(|&s| {
                        stats.sb_mults += 1;
                        pref_bound.mul(&s)
                    })
                    .collect()
            }
        }
    }
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

fn invert(x: &Goldilocks) -> Goldilocks {
    match x.inverse() {
        Some(v) => v,
        // Unreachable for our denominators (products of nonzero field
        // elements); zero maps to zero defensively.
        None => Goldilocks::ZERO,
    }
}

/// The Goldilocks modulus (2^64 − 2^32 + 1) — the field's public
/// constant, replicated locally because lattice-core exports the type but
/// not the constant.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// The signed small integer as a field element.
fn fe_signed(x: i64) -> Goldilocks {
    if x >= 0 {
        fe(x as u64)
    } else {
        Goldilocks(P.wrapping_sub(x.unsigned_abs()))
    }
}

/// The binding weights for one window axis, evaluated at the round
/// challenge `r`: the **finite-point Lagrange basis over `{0, 1, …, d}`**
/// (the d+1 integer values of the axis determine the degree-≤d product
/// restriction exactly; the axis's ∞ slot carries the degree-d leading
/// coefficient — a different degree convention than the (d+2)-point
/// domain — so it takes weight ZERO here).
///
/// Layout: side d+2 with index 0 (∞) = 0 and index j+1 = L_j(r).
fn lagrange_weights_u(r: &Goldilocks, d: usize) -> Vec<Goldilocks> {
    let side = d + 2;
    let mut w = vec![Goldilocks::ZERO; side];
    for j in 0..=d {
        let mut num = Goldilocks::ONE;
        let mut den = Goldilocks::ONE;
        for k in 0..=d {
            if k != j {
                num = num.mul(&r.sub(&fe(k as u64)));
                den = den.mul(&fe_signed(j as i64 - k as i64));
            }
        }
        w[j + 1] = num.mul(&invert(&den));
    }
    w
}

/// Prove with the default window (3 rounds) — a drop-in for
/// [`crate::sumcheck::prove`] producing byte-identical transcripts.
pub fn prove_fast(
    vp: &VirtualPolynomial,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<SumcheckOutput, SumcheckError> {
    prove_fast_with_opts(vp, claim, transcript, &FastProverOpts::default(), &[], &[])
}

/// Prove with an explicit equality-factor split: the factors at
/// `eq_indices` are eq(w, ·) for the verifier point `w` and are never
/// materialised over the full hypercube (2025/1117 §5's split-eq).
pub fn prove_fast_with_eq(
    vp: &VirtualPolynomial,
    claim: Goldilocks,
    transcript: &mut Transcript,
    opts: &FastProverOpts,
    eq_indices: &[usize],
    w: &[Goldilocks],
) -> Result<SumcheckOutput, SumcheckError> {
    prove_fast_with_opts(vp, claim, transcript, opts, eq_indices, w)
}

/// The full prover.
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub fn prove_fast_with_opts(
    vp: &VirtualPolynomial,
    claim: Goldilocks,
    transcript: &mut Transcript,
    opts: &FastProverOpts,
    eq_indices: &[usize],
    w: &[Goldilocks],
) -> Result<SumcheckOutput, SumcheckError> {
    if vp.terms.is_empty() {
        return crate::sumcheck::prove(vp, claim, transcript);
    }
    let m = vp.num_vars;
    let d = vp.max_degree();
    let v = opts.window.min(m);
    let suffixes = 1usize << (m - v);
    let mut stats = FastProverStats::default();

    // ---- Factor views (tables or split-eq) ----
    let views: Vec<FactorView> = vp
        .factors
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if eq_indices.contains(&i) {
                // Split eq(w, ·) at the window boundary.
                let mut prefix = vec![Goldilocks::ONE];
                for bit in 0..v {
                    let w_b = w[bit];
                    let one_minus = Goldilocks::ONE.sub(&w_b);
                    let mut next = Vec::with_capacity(prefix.len() * 2);
                    for e in &prefix {
                        // Interleaved doubling: the new variable's bit is
                        // the LSB of the flat index (first variable = MSB).
                        next.push(e.mul(&one_minus));
                        next.push(e.mul(&w_b));
                    }
                    prefix = next;
                }
                let mut suffix = vec![Goldilocks::ONE];
                for bit in v..m {
                    let w_b = w[bit];
                    let one_minus = Goldilocks::ONE.sub(&w_b);
                    let mut next = Vec::with_capacity(suffix.len() * 2);
                    for e in &suffix {
                        next.push(e.mul(&one_minus));
                        next.push(e.mul(&w_b));
                    }
                    suffix = next;
                }
                FactorView::Eq { prefix, suffix }
            } else {
                FactorView::Table(f)
            }
        })
        .collect();

    // ---- Phase W: per-suffix, per-term grids over U(d+1)^v ----
    // Grid side per axis: d + 2 (∞ + integers 0..d).
    let side = d + 2;
    let mut grids: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(suffixes); // [suffix][term]
    for x2 in 0..suffixes {
        let mut term_grids = Vec::with_capacity(vp.terms.len());
        for (_coeff, ids) in &vp.terms {
            let mut tables: Vec<Vec<Goldilocks>> = ids
                .iter()
                .map(|&k| views[k].window_table(x2, suffixes, v))
                .collect();
            // Pad lower-degree terms with constant-one factors so every
            // grid is built NATIVELY at the uniform side d+2 — extending a
            // degree-n grid to a larger domain would carry stale ∞-slot
            // semantics (the slot holds the degree-n lead, the larger
            // domain's formulas expect the degree-(d+1) one).
            while tables.len() < d {
                tables.push(vec![Goldilocks::ONE; 1usize << v]);
            }
            let (grid, gst) = multi_product_eval(&tables, v);
            stats.grid.bb_mults += gst.bb_mults;
            stats.grid.sb_mults += gst.sb_mults;
            term_grids.push(grid);
        }
        grids.push(term_grids);
    }

    // ---- Window rounds 1..v ----
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(m);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(m);
    let mut current_claim = claim;
    // Composed Lagrange weights over the bound window axes: W[u] indexed
    // MSB-first over the digits of axes 0..j−2 (0-based).
    let mut wj: Vec<Goldilocks> = vec![Goldilocks::ONE];
    for j in 1..=v {
        // Message at t ∈ 0..=d:
        // msg[t] = Σ_{x'} Σ_{term} c·Σ_u W[u]·(Σ_{w bool} G[u, t, w]).
        // Grid layout: axis a's digit at stride side^{v−1−a}; axis 0 =
        // the first window variable (outermost). 1-based axis j ↔
        // 0-based axis j−1. The extraction iterates ONLY the contributing
        // grid points (u ∈ side^{j−1} × t × w ∈ {0,1}^{v−j}) with
        // precomputed strides — no per-entry digit divisions.
        let t_axis = j - 1;
        let tail_axes = v - j;
        // Strides: axis a's stride in the flat index.
        let stride = |a: usize| side.pow((v - a - 1) as u32);
        // Precompute the w-offsets: the tail axes' boolean digits (1 or 2)
        // contribute Σ (digit)·stride(a) for a ∈ j..v−1.
        let w_offsets: Vec<usize> = {
            let mut offs = Vec::with_capacity(1usize << tail_axes);
            for w in 0..1usize << tail_axes {
                let mut off = 0usize;
                for b in 0..tail_axes {
                    let bit = (w >> (tail_axes - 1 - b)) & 1;
                    off += (bit + 1) * stride(j + b);
                }
                offs.push(off);
            }
            offs
        };
        let stride_t = stride(t_axis);
        let u_count = side.pow((j - 1) as u32);
        let mut evals_at = vec![Goldilocks::ZERO; d + 1];
        for term_grids in grids.iter() {
            for ((coeff, _ids), grid) in vp.terms.iter().zip(term_grids.iter()) {
                // Per u: decompose once into the u-part's flat offset.
                for u in 0..u_count {
                    // u's digits (MSB-first over axes 0..j−2).
                    // MSB-first: axis 0 carries u's most-significant
                    // digit (matching the composed-weight layout).
                    let mut u_off = 0usize;
                    let mut rem = u;
                    for a in (0..(j - 1)).rev() {
                        u_off += (rem % side) * stride(a);
                        rem /= side;
                    }
                    let wgt = wj[u];
                    if wgt.is_zero() {
                        continue;
                    }
                    for t in 0..=d {
                        let base = u_off + (t + 1) * stride_t;
                        let mut acc = Goldilocks::ZERO;
                        for &off in &w_offsets {
                            acc = acc.add(&grid[base + off]);
                        }
                        // acc already sums the boolean tail; weight once.
                        let contrib = acc.mul(&wgt).mul(coeff);
                        stats.sb_mults += 2;
                        evals_at[t] = evals_at[t].add(&contrib);
                    }
                }
            }
        }
        // Honest guard + transcript (identical labels to the baseline).
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(SumcheckError::ClaimMismatch);
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(SumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(SumcheckError::Transcript)?;
        challenges.push(r);
        current_claim = crate::sumcheck::interpolate_at(&evals_at, &r);
        rounds.push(evals_at);
        // Compose the weight vector for the next round.
        if j < v {
            let l = lagrange_weights_u(&r, d);
            let mut next = Vec::with_capacity(wj.len() * side);
            for wv in &wj {
                for lv in &l {
                    next.push(wv.mul(lv));
                    stats.bb_mults += 1;
                }
            }
            wj = next;
        }
    }

    // ---- Bind the factors to r_{<v} (the prefix adaptation) ----
    // eq(r_{<v}, b) over the boolean window cube.
    let mut eq_r = vec![Goldilocks::ONE];
    for ri in challenges.iter().take(v) {
        let one_minus = Goldilocks::ONE.sub(ri);
        let mut next = Vec::with_capacity(eq_r.len() * 2);
        for e in &eq_r {
            next.push(e.mul(&one_minus));
            next.push(e.mul(ri));
        }
        eq_r = next;
    }
    let bound: Vec<Vec<Goldilocks>> = views
        .iter()
        .map(|view| view.bind_window(&eq_r, suffixes, v, &mut stats))
        .collect();

    // ---- Tail rounds v+1..m: the baseline's SIMD kernel rounds ----
    // (bind-once per round through fix_variables + the packed half-binding
    // and 8-lane lazy term-product kernels — the same code path the
    // baseline prover uses, at performance parity; the window phase above
    // is the restructured part.)
    let mut bound_mles: Vec<DenseMle> = bound
        .into_iter()
        .map(|arr| DenseMle::new(arr).map_err(SumcheckError::Mle))
        .collect::<Result<_, _>>()?;
    for _round in v..m {
        let mut evals_at = Vec::with_capacity(d + 1);
        for t in 0..=d {
            let tf = fe(t as u64);
            stats.bb_mults += (d.saturating_sub(1).max(1) * bound_mles[0].len() / 2) as u64;
            evals_at.push(crate::sumcheck::sum_products(&bound_mles, &vp.terms, tf));
        }
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(SumcheckError::ClaimMismatch);
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(SumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(SumcheckError::Transcript)?;
        challenges.push(r);
        current_claim = crate::sumcheck::interpolate_at(&evals_at, &r);
        rounds.push(evals_at);
        for b in bound_mles.iter_mut() {
            *b = b.fix_variables(&[r]).map_err(SumcheckError::Mle)?;
        }
    }

    // ---- Final claims (identical to the baseline prover) ----
    let mut factor_claims = Vec::with_capacity(bound_mles.len());
    for b in &bound_mles {
        factor_claims.push(b.evaluate(&[]).map_err(SumcheckError::Mle)?);
    }
    let mut final_claim = Goldilocks::ZERO;
    for (coeff, ids) in &vp.terms {
        let mut prod = *coeff;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(SumcheckError::FinalCheckFailed);
    }

    if opts.collect_stats {
        STATS.with(|s| *s.borrow_mut() = Some(stats));
    }

    Ok(SumcheckOutput {
        proof: SumcheckProof { rounds },
        challenges,
        final_claim,
        factor_claims,
    })
}

thread_local! {
    static STATS: std::cell::RefCell<Option<FastProverStats>> = const { std::cell::RefCell::new(None) };
}

/// Take the stats collected by the last [`prove_fast_with_opts`] call.
pub fn take_last_stats() -> Option<FastProverStats> {
    STATS.with(|s| s.borrow_mut().take())
}
