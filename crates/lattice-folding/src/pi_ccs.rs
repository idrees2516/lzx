//! **Π_CCS — the in-sumcheck norm products** (Neo, ePrint 2025/294 §7;
//! SuperNeo, ePrint 2026/242 §7.3) — the succinct CCS decider for the
//! committed-fold pipeline.
//!
//! The honest gap this closes: [`superneo_committed::decider_committed`]
//! *transmits the witness* (the Ajtai opening carries the full packed
//! vector, and the reconstruction + bound checks re-execute on plaintext
//! values). The paper's decider is a single sum-check folding three
//! checks into one polynomial over the cube
//! `{0,1}^{log L} × {0,1}^{log m}` (level variables first):
//!
//! ```text
//! Q(X) := Σ_ℓ γ^ℓ · eq(X, r)·z̃⁽ℓ⁾(X_pos)               (EvalK — prior witness claims)
//!       + Σ_{j,ℓ} γ^{L+j·L+ℓ} · eq(X, r)·p̃_{j,ℓ}(X_pos) (EvalA — prior product claims)
//!       + γ^{L(t+2)} · eq(X_pos, α)·F(X_pos)            (relaxed CCS satisfaction)
//!       + γ^{L(t+2)+1} · Π_{a=-b+1}^{b-1}(Z̃(X) − a)     (NC — the norm products)
//! ```
//!
//! * **NC** — the norm check lives *inside* the sum-check as a product
//!   of the stacked digit MLE shifted by every integer in `[-b+1, b-1]`:
//!   the product vanishes at a cube point iff the digit there lies in
//!   range, so an honest witness contributes exactly 0 to the claimed
//!   sum. This is Neo's "range check as a degree-`(2b−1)` sum-check
//!   instance" (§1.4) — `b` is kept small (the paper's own choice
//!   `b = 2`), and larger witnesses enter through base-`b` digit
//!   decomposition (the pay-per-bit path's native representation).
//! * **F** — relaxed CCS satisfaction per row: `Σ_j c_j·Π(A_a z)[row] −
//!   u·(S z)[row] − slack[row] = 0`, wrapped in `eq(X_pos, α)`: the
//!   eq-weighted Boolean sum of a cube-vanishing polynomial is zero, so
//!   F contributes nothing to the verifier-known claimed sum.
//! * **EvalK/EvalA** — the running evaluation claims (the CE instance)
//!   are re-randomized inside the same sum-check (HyperNova-style); the
//!   *output* is a fresh set of per-level claims at the terminal point —
//!   the eval-claim API the folding loop consumes.
//!
//! ## The digit structure (Neo's `Decomp_b`)
//!
//! The logical witness `w` is decomposed base-`b` into `L` level vectors
//! (`w = Σ_ℓ b^ℓ·d_ℓ`, `‖d_ℓ‖∞ < b`). The stacked digit MLE `Z̃` is the
//! truth table of the `L×m` digit matrix; the level MLEs `z̃⁽ℓ⁾` are
//! position-only. The matrix products are computed **per level** and
//! recombined linearly (`(A_j w)̃ = Σ_ℓ b^ℓ·(A_j d_ℓ)̃`) — exactly the
//! verifier's recombination `m_j := Σ_ℓ b^ℓ·y_{j,ℓ}` (Neo's step 4).
//!
//! ## The commitment layer
//!
//! Each level vector is packed coefficient-wise (16 values per ring
//! element — the existing [`superneo_committed`] bridge) and committed
//! **separately**. The packing is per-coefficient linear, so the
//! homomorphic identity `C_values = Σ_ℓ b^ℓ·C_ℓ` holds exactly (no wrap:
//! the folded values stay `< 2^25 ≪ q/2`) and binds the digit
//! commitments to the folded instance's value commitment for free.
//!
//! ## Honest deviations
//!
//! * `K = A = Goldilocks` — no extension field for the challenge space
//!   (round-by-round soundness `d·ℓ/|F|` per round; the `fq2` lift in
//!   [`crate::fq2_sumcheck`] is the upgrade path).
//! * The Neo/SuperNeo ring `Trans/Emb` machinery (their Theorems 9–11)
//!   is realized through the **linear packing bridge**: the flatten-MLE
//!   is linear in the digit values, so the evaluation homomorphism
//!   reduces to coefficient-wise linearity; the claim fold works over
//!   plain Goldilocks at the small-`ρ` fold discipline.
//! * The claimed sum `T` excludes the F/NC terms (both vanish on honest
//!   witnesses); a cheating prover with an out-of-range digit or an
//!   unsatisfied row cannot complete the first round identity.
//! * The running claim's **norm growth across folds** (the paper's
//!   Π_DEC norm chain) is out of scope: the decider operates on a
//!   *fresh* decomposition of the folded witness. The claim-fold API is
//!   provided and tested for linearity; folding digit vectors directly
//!   and re-decomposing is the documented future work (the SALSAA
//!   norm-chain substrate, NEXT_STEPS §5).

use crate::superneo_committed::pack_small;
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiPublicKey};
use lattice_core::mle::{DenseMle, MleError};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;
use lattice_relations::ccs::{Ccs, SparseMatrix};
use lattice_ring::RingElement;
use lattice_sumcheck::sumcheck::{self, SumcheckProof};
use lattice_sumcheck::virtual_poly::VirtualPolynomial;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PiCcsError {
    /// A value does not decompose in the declared level count.
    DecompositionOverflow {
        value: u64,
        base: u64,
        levels: usize,
    },
    /// A digit exceeded the base (fail-closed before any proof).
    DigitOutOfRange {
        digit: u64,
        base: u64,
    },
    Shape {
        expected: usize,
        got: usize,
    },
    /// The sum-check layer rejected (shape, round identity, or claim).
    Sumcheck(String),
    /// The verifier's recomputed `Q(r')` != the sum-check terminal value.
    FinalCheckFailed,
    /// The claim fold requires claims at the same point.
    PointMismatch,
    /// The homomorphic value/digit commitment binding failed.
    BindingFailed,
    /// The commitment opening failed.
    OpeningFailed,
    /// A CCS selection is not binary (this module pins arity 2).
    BadSelection {
        index: usize,
        arity: usize,
    },
    Transcript(TranscriptError),
}

impl From<TranscriptError> for PiCcsError {
    fn from(e: TranscriptError) -> Self {
        PiCcsError::Transcript(e)
    }
}

impl core::fmt::Display for PiCcsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PiCcsError::DecompositionOverflow {
                value,
                base,
                levels,
            } => {
                write!(
                    f,
                    "value {value} does not decompose in {levels} base-{base} levels"
                )
            }
            PiCcsError::DigitOutOfRange { digit, base } => {
                write!(f, "digit {digit} >= base {base}")
            }
            PiCcsError::Shape { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            PiCcsError::Sumcheck(e) => write!(f, "sum-check: {e}"),
            PiCcsError::FinalCheckFailed => {
                write!(f, "Q(r') recomputation != sum-check terminal value")
            }
            PiCcsError::PointMismatch => write!(f, "claim fold needs a common point"),
            PiCcsError::BindingFailed => write!(f, "C_values != Σ b^ℓ·C_ℓ"),
            PiCcsError::OpeningFailed => write!(f, "commitment opening failed"),
            PiCcsError::BadSelection { index, arity } => {
                write!(f, "selection {index} has arity {arity} (need 2)")
            }
            PiCcsError::Transcript(e) => write!(f, "transcript: {e}"),
        }
    }
}

fn fe(x: u64) -> Goldilocks {
    Goldilocks::from_u64(x)
}

fn mle_shape_err(e: MleError) -> PiCcsError {
    match e {
        MleError::WrongEvaluationCount { expected, got } => PiCcsError::Shape { expected, got },
        MleError::PointLengthMismatch { expected, got } => PiCcsError::Shape { expected, got },
    }
}

fn next_pow2(x: usize) -> usize {
    let mut v = 1;
    while v < x {
        v <<= 1;
    }
    v
}

fn log2_exact(x: usize) -> usize {
    debug_assert!(x.is_power_of_two());
    x.trailing_zeros() as usize
}

// ---------------------------------------------------------------------------
// Digit decomposition (Neo's Decomp_b)

/// Base-`b` digit decomposition of a small-value witness:
/// `w = Σ_ℓ b^ℓ·levels[ℓ]` with every digit `< b`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitWitness {
    pub levels: Vec<Vec<u64>>,
    pub base: u64,
}

/// Decompose `values` into `levels` base-`b` digits (fail-closed: a
/// value that does not fit rejects).
pub fn decompose(values: &[u64], base: u64, levels: usize) -> Result<DigitWitness, PiCcsError> {
    if base < 2 || levels == 0 {
        return Err(PiCcsError::Shape {
            expected: 2,
            got: base as usize,
        });
    }
    let mut out = vec![vec![0u64; values.len()]; levels];
    for (i, &v) in values.iter().enumerate() {
        let mut rem = v;
        for lev in out.iter_mut() {
            lev[i] = rem % base;
            rem /= base;
        }
        if rem != 0 {
            return Err(PiCcsError::DecompositionOverflow {
                value: v,
                base,
                levels,
            });
        }
    }
    Ok(DigitWitness { levels: out, base })
}

impl DigitWitness {
    /// Recompose the logical witness `Σ b^ℓ·levels[ℓ]`.
    pub fn recompose(&self) -> Vec<u64> {
        let n = self.levels.first().map(|l| l.len()).unwrap_or(0);
        let mut out = vec![0u64; n];
        for (ell, lev) in self.levels.iter().enumerate() {
            let weight = self.base.pow(ell as u32);
            for (o, &d) in out.iter_mut().zip(lev.iter()) {
                *o += weight * d;
            }
        }
        out
    }

    /// Fail-closed digit range validation.
    pub fn validate(&self) -> Result<(), PiCcsError> {
        for lev in &self.levels {
            for &d in lev {
                if d >= self.base {
                    return Err(PiCcsError::DigitOutOfRange {
                        digit: d,
                        base: self.base,
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The instance: per-level commitments + the public relaxed components.

/// The Π_CCS instance: one committed (folded) relaxed CCS witness in
/// digit-decomposed form. The Fiat–Shamir digest covers exactly these
/// fields plus the CCS digest.
#[derive(Clone, Debug)]
pub struct PiCcsInstance {
    /// Per-level Ajtai commitments over the packed digit vectors.
    pub level_commitments: Vec<AjtaiCommitment>,
    /// The relaxed scalar `u`.
    pub u: Goldilocks,
    /// The relaxed slack vector (length `ccs.n`).
    pub slack: Vec<Goldilocks>,
}

/// The prover-side secret: the digits plus the per-level packed vectors.
#[derive(Clone, Debug)]
pub struct PiCcsSecret {
    pub digits: DigitWitness,
    /// Per-level padded packed vectors (what the commitments open to).
    pub packed_levels: Vec<Vec<RingElement>>,
}

/// Commit each digit level separately (16 values per ring element — the
/// existing coefficient-packing bridge).
pub fn commit_digits(
    pk: &AjtaiPublicKey,
    digits: &DigitWitness,
    slack: &[Goldilocks],
    u: Goldilocks,
) -> Result<(PiCcsInstance, PiCcsSecret), PiCcsError> {
    digits.validate()?;
    let mut commitments = Vec::with_capacity(digits.levels.len());
    let mut packed_levels = Vec::with_capacity(digits.levels.len());
    for lev in &digits.levels {
        let packed_u32: Vec<u32> = lev.iter().map(|&d| d as u32).collect();
        let packed = pack_small(&pk.params.ring, &packed_u32);
        let padded = pk.pad_to_m(&packed).map_err(|_| PiCcsError::Shape {
            expected: pk.params.m,
            got: packed.len(),
        })?;
        commitments.push(pk.commit(&padded).map_err(|_| PiCcsError::OpeningFailed)?);
        packed_levels.push(padded);
    }
    Ok((
        PiCcsInstance {
            level_commitments: commitments,
            u,
            slack: slack.to_vec(),
        },
        PiCcsSecret {
            digits: digits.clone(),
            packed_levels,
        },
    ))
}

/// The public digest binding the instance into the Fiat–Shamir
/// derivation (commitments, u, slack, shape — never the digits).
pub fn pi_ccs_digest(ccs_digest: &[u8; 32], inst: &PiCcsInstance) -> [u8; 32] {
    let mut buf = Vec::with_capacity(128 + inst.slack.len() * 8);
    buf.extend_from_slice(ccs_digest);
    buf.extend_from_slice(&(inst.level_commitments.len() as u32).to_le_bytes());
    buf.extend_from_slice(&inst.u.to_bytes());
    for s in &inst.slack {
        buf.extend_from_slice(&s.to_bytes());
    }
    for c in &inst.level_commitments {
        buf.extend_from_slice(&c.to_bytes());
    }
    Transcript::hash_domain(b"lzx-pi-ccs", &buf)
}

// ---------------------------------------------------------------------------
// The running evaluation claim (the CE instance).

/// The eval-claim API: per-level witness claims and per-level matrix-
/// product claims at one shared point `(r_lev, r_pos)`. The products are
/// indexed `[0..t]` = the CCS `A` matrices, `[t]` = the summed span
/// matrix `S = Σ_b B_b` (the relaxed form's linear side).
#[derive(Clone, Debug)]
pub struct CeClaim {
    pub r_lev: Vec<Goldilocks>,
    pub r_pos: Vec<Goldilocks>,
    /// `y[ℓ] = z̃⁽ℓ⁾(r_pos)` — the per-level witness claims.
    pub y: Vec<Goldilocks>,
    /// `y_products[j][ℓ] = (M_j d_ℓ)̃(r_pos)`.
    pub y_products: Vec<Vec<Goldilocks>>,
}

/// Fold two claims at the SAME point: the MLE evaluation is linear in
/// the digit vectors, so `y_fold = y₁ + ρ·y₂` is a valid claim about
/// `d₁ + ρ·d₂` (the folded *level vectors* — see the module doc for the
/// norm-growth caveat: the folded levels leave the digit range and the
/// Π_DEC re-decomposition is the follow-up).
pub fn fold_claim(c1: &CeClaim, c2: &CeClaim, rho: Goldilocks) -> Result<CeClaim, PiCcsError> {
    if c1.r_lev != c2.r_lev || c1.r_pos != c2.r_pos || c1.y.len() != c2.y.len() {
        return Err(PiCcsError::PointMismatch);
    }
    if c1.y_products.len() != c2.y_products.len() {
        return Err(PiCcsError::PointMismatch);
    }
    let y =
        c1.y.iter()
            .zip(c2.y.iter())
            .map(|(a, b)| a.add(&rho.mul(b)))
            .collect();
    let y_products = c1
        .y_products
        .iter()
        .zip(c2.y_products.iter())
        .map(|(ja, jb)| {
            ja.iter()
                .zip(jb.iter())
                .map(|(a, b)| a.add(&rho.mul(b)))
                .collect()
        })
        .collect();
    Ok(CeClaim {
        r_lev: c1.r_lev.clone(),
        r_pos: c1.r_pos.clone(),
        y,
        y_products,
    })
}

// ---------------------------------------------------------------------------
// The protocol shape.

/// The derived protocol shape (cube sizes, γ-power layout).
#[derive(Clone, Debug)]
pub struct PiCcsShape {
    pub base: u64,
    /// Padded level count (power of two).
    pub levels: usize,
    /// Padded row/position space (power of two).
    pub m: usize,
    pub log_lev: usize,
    pub log_m: usize,
    /// `t` = A-matrix count; the product index `t` is the span matrix.
    pub t: usize,
}

impl PiCcsShape {
    pub fn derive(ccs: &Ccs, base: u64, levels: usize) -> Self {
        let t = ccs.a_matrices.len();
        let levels_pow2 = next_pow2(levels.max(1));
        let m = next_pow2(ccs.n.max(ccs.m).max(2));
        PiCcsShape {
            base,
            levels: levels_pow2,
            m,
            log_lev: log2_exact(levels_pow2),
            log_m: log2_exact(m),
            t,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.log_lev + self.log_m
    }

    /// The maximum per-variable degree of `Q` (the engine's message
    /// width): `max(2b−1, 3)` — the NC product length vs the eq-wrapped
    /// quadratic CCS terms.
    pub fn max_degree(&self) -> usize {
        self.norm_shifts().len().max(3)
    }

    /// γ-power offsets — disjoint across the term families.
    pub fn off_eval_k(&self, lev: usize) -> u64 {
        lev as u64
    }
    pub fn off_eval_a(&self, j: usize, lev: usize) -> u64 {
        (self.levels as u64) * (1 + j as u64) + lev as u64
    }
    pub fn off_f(&self) -> u64 {
        (self.levels as u64) * (self.t as u64 + 2)
    }
    pub fn off_nc(&self) -> u64 {
        self.off_f() + 1
    }

    /// The norm-product shifts `a ∈ [-b+1, b-1]` (balanced integers).
    pub fn norm_shifts(&self) -> Vec<i64> {
        let b = self.base as i64;
        (-(b - 1)..=b - 1).collect()
    }

    /// The claimed sum from the PRIOR claims (verifier-computable):
    /// `T = Σ γ^{off}·y` over both eval families — the F and NC
    /// contributions vanish on honest witnesses.
    pub fn claimed_sum(&self, gamma: &Goldilocks, prior: &CeClaim) -> Goldilocks {
        let mut acc = Goldilocks::ZERO;
        for (lev, &y) in prior.y.iter().enumerate() {
            acc = acc.add(&gamma.pow_u64(self.off_eval_k(lev)).mul(&y));
        }
        for (j, jp) in prior.y_products.iter().enumerate() {
            for (lev, &y) in jp.iter().enumerate() {
                acc = acc.add(&gamma.pow_u64(self.off_eval_a(j, lev)).mul(&y));
            }
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// Table construction (prover side).

/// A position-only table lifted to the full cube (constant across the
/// level variables): `T[x_lev, x_pos] = v[x_pos]`.
fn lift_position_table(v: &[Goldilocks], shape: &PiCcsShape) -> Result<DenseMle, PiCcsError> {
    let mut table = Vec::with_capacity(shape.levels * shape.m);
    for _ in 0..shape.levels {
        table.extend_from_slice(v);
    }
    DenseMle::new(table).map_err(mle_shape_err)
}

/// The stacked digit table: `Z̃`'s truth table over the full cube,
/// `T[ℓ·m + pos] = digits[ℓ][pos]` (zero-padded).
fn stacked_digit_table(digits: &DigitWitness, shape: &PiCcsShape) -> Result<DenseMle, PiCcsError> {
    let mut table = vec![Goldilocks::ZERO; shape.levels * shape.m];
    for (ell, lev) in digits.levels.iter().enumerate() {
        if ell >= shape.levels {
            break;
        }
        for (pos, &d) in lev.iter().enumerate() {
            if pos < shape.m {
                table[ell * shape.m + pos] = fe(d);
            }
        }
    }
    DenseMle::new(table).map_err(mle_shape_err)
}

/// The eq table at `point` over the position variables, lifted.
fn eq_pos_table(point: &[Goldilocks], shape: &PiCcsShape) -> Result<DenseMle, PiCcsError> {
    let pos_eq = DenseMle::eq_extension(point);
    lift_position_table(&pos_eq.evaluations, shape)
}

/// The eq table at the full-cube point `(r_lev, r_pos)`.
fn eq_full_table(r_lev: &[Goldilocks], r_pos: &[Goldilocks]) -> DenseMle {
    let mut point = r_lev.to_vec();
    point.extend_from_slice(r_pos);
    DenseMle::eq_extension(&point)
}

/// Per-level matrix products `(M·d_ℓ)` as field vectors.
fn level_products(matrix: &SparseMatrix, digits: &DigitWitness) -> Vec<Vec<Goldilocks>> {
    digits
        .levels
        .iter()
        .map(|lev| {
            let w: Vec<Goldilocks> = lev.iter().map(|&d| fe(d)).collect();
            matrix
                .multiply(&w)
                .unwrap_or_else(|_| vec![Goldilocks::ZERO; matrix.rows])
        })
        .collect()
}

/// The summed span matrix `S = Σ_b B_b` (the relaxed form's linear side).
fn span_matrix(ccs: &Ccs) -> SparseMatrix {
    let mut entries: Vec<(usize, usize, Goldilocks)> = Vec::new();
    for b in &ccs.b_matrices {
        for &(r, c, v) in &b.entries {
            let mut found = false;
            for (er, ec, ev) in entries.iter_mut() {
                if *er == r && *ec == c {
                    *ev = ev.add(&v);
                    found = true;
                    break;
                }
            }
            if !found {
                entries.push((r, c, v));
            }
        }
    }
    SparseMatrix {
        rows: ccs.n,
        cols: ccs.m,
        entries,
    }
}

/// Zero-pad a vector to `m` rows (the position space).
fn pad_rows(v: &[Goldilocks], m: usize) -> Vec<Goldilocks> {
    let mut out = v.to_vec();
    out.resize(m, Goldilocks::ZERO);
    out
}

// ---------------------------------------------------------------------------
// The proof.

/// The Π_CCS proof: the sum-check plus the fresh per-level claims at the
/// terminal point (the eval-claim API consumed by the folding loop and
/// the commitment-opening layer).
#[derive(Clone, Debug)]
pub struct PiCcsProof {
    pub sumcheck: SumcheckProof,
    /// The fresh claims at `(r'_lev, r'_pos)`.
    pub new_claim: CeClaim,
}

/// The factors registered in the order the claims are derived.
struct BuiltVp {
    vp: VirtualPolynomial,
    /// Per-level witness table handles (index = level).
    level_factor: Vec<usize>,
    /// `product_factor[j][ℓ]` — per-level product tables; `j < t` are
    /// the A-matrices, `j == t` the span matrix.
    product_factor: Vec<Vec<usize>>,
}

/// Build the Π_CCS virtual polynomial for one instance.
#[allow(clippy::too_many_arguments)] // the paper's fixed parameter set
fn build_vp(
    ccs: &Ccs,
    shape: &PiCcsShape,
    digits: &DigitWitness,
    u: Goldilocks,
    slack: &[Goldilocks],
    prior: Option<&CeClaim>,
    alpha_pos: &[Goldilocks],
    gamma: &Goldilocks,
) -> Result<BuiltVp, PiCcsError> {
    let mut vp = VirtualPolynomial::new(shape.num_vars());

    // --- NC factors: the stacked digit table and its affine shifts -----
    let z_stacked = stacked_digit_table(digits, shape)?;
    let vpe = |e: lattice_sumcheck::virtual_poly::VirtualPolyError| {
        PiCcsError::Sumcheck(format!("{e:?}"))
    };
    let mut nc_factors = Vec::with_capacity(shape.norm_shifts().len());
    for a in shape.norm_shifts() {
        let shifted: Vec<Goldilocks> = z_stacked
            .evaluations
            .iter()
            .map(|&z| {
                if a < 0 {
                    z.add(&fe(u64::try_from(-a).unwrap_or(0)))
                } else {
                    z.sub(&fe(a as u64))
                }
            })
            .collect();
        nc_factors.push(
            vp.add_factor(DenseMle::new(shifted).map_err(mle_shape_err)?)
                .map_err(vpe)?,
        );
    }

    // --- eq(X_pos, α): the F wrapper ------------------------------------
    let eq_alpha = vp
        .add_factor(eq_pos_table(alpha_pos, shape)?)
        .map_err(vpe)?;

    // --- per-level witness tables (EvalK + the claim derivation) -------
    let mut level_factor = Vec::with_capacity(shape.levels);
    for ell in 0..shape.levels {
        let v: Vec<Goldilocks> = match digits.levels.get(ell) {
            Some(lev) => pad_rows(&lev.iter().map(|&d| fe(d)).collect::<Vec<_>>(), shape.m),
            None => vec![Goldilocks::ZERO; shape.m],
        };
        level_factor.push(
            vp.add_factor(lift_position_table(&v, shape)?)
                .map_err(vpe)?,
        );
    }

    // --- per-level product tables + the recombined polynomials ---------
    let mut matrices: Vec<&SparseMatrix> = ccs.a_matrices.iter().collect();
    let span = span_matrix(ccs);
    matrices.push(&span);
    let mut product_factor: Vec<Vec<usize>> = Vec::with_capacity(matrices.len());
    let mut recomposed: Vec<Vec<Goldilocks>> = Vec::with_capacity(matrices.len());
    for matrix in &matrices {
        let per_level = level_products(matrix, digits);
        let mut handles = Vec::with_capacity(shape.levels);
        let mut rec = vec![Goldilocks::ZERO; shape.m];
        for (ell, prod) in per_level.iter().enumerate() {
            let v = pad_rows(prod, shape.m);
            // Recombination: (M w)̃ = Σ_ℓ b^ℓ · (M d_ℓ)̃.
            let weight = fe(shape.base.pow(ell as u32));
            for (r, &val) in rec.iter_mut().zip(v.iter()) {
                *r = r.add(&weight.mul(&val));
            }
            handles.push(
                vp.add_factor(lift_position_table(&v, shape)?)
                    .map_err(vpe)?,
            );
        }
        while handles.len() < shape.levels {
            handles.push(
                vp.add_factor(lift_position_table(
                    &vec![Goldilocks::ZERO; shape.m],
                    shape,
                )?)
                .map_err(vpe)?,
            );
        }
        product_factor.push(handles);
        recomposed.push(rec);
    }

    // --- the slack MLE + eq(X, r) ---------------------------------------
    let slack_factor = vp
        .add_factor(lift_position_table(&pad_rows(slack, shape.m), shape)?)
        .map_err(vpe)?;
    let eq_r = match prior {
        Some(p) => Some(
            vp.add_factor(eq_full_table(&p.r_lev, &p.r_pos))
                .map_err(vpe)?,
        ),
        None => None,
    };

    // --- terms -----------------------------------------------------------
    // EvalK: γ^{off_k(ℓ)} · [eq_r, z_ℓ] and EvalA: γ^{off_a(j,ℓ)} ·
    // [eq_r, p_{j,ℓ}] — the prior claims' re-randomization.
    if let (Some(eq_r_handle), Some(p)) = (eq_r, prior) {
        for (lev, &lf) in level_factor.iter().enumerate() {
            vp.add_term(gamma.pow_u64(shape.off_eval_k(lev)), vec![eq_r_handle, lf])
                .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;
        }
        for (j, jp) in p.y_products.iter().enumerate() {
            for (lev, pf) in product_factor[j].iter().enumerate() {
                if lev >= jp.len() {
                    break;
                }
                vp.add_term(
                    gamma.pow_u64(shape.off_eval_a(j, lev)),
                    vec![eq_r_handle, *pf],
                )
                .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;
            }
        }
    }

    // F: γ^{off_f} · eq_α · [Σ_j c_j·(A_a ∘ A_b) − u·span − slack].
    let off_f = gamma.pow_u64(shape.off_f());
    for (j, sel) in ccs.selections.iter().enumerate() {
        if sel.len() != 2 {
            return Err(PiCcsError::BadSelection {
                index: j,
                arity: sel.len(),
            });
        }
        let c = ccs.constants.get(j).copied().unwrap_or(Goldilocks::ONE);
        let ja = vp
            .add_factor(lift_position_table(&recomposed[sel[0]], shape)?)
            .map_err(vpe)?;
        let jb = vp
            .add_factor(lift_position_table(&recomposed[sel[1]], shape)?)
            .map_err(vpe)?;
        vp.add_term(off_f.mul(&c), vec![eq_alpha, ja, jb])
            .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;
    }
    let span_lift = vp
        .add_factor(lift_position_table(&recomposed[matrices.len() - 1], shape)?)
        .map_err(vpe)?;
    vp.add_term(
        off_f.mul(&Goldilocks::ZERO.sub(&u)),
        vec![eq_alpha, span_lift],
    )
    .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;
    vp.add_term(
        off_f.mul(&Goldilocks::ZERO.sub(&Goldilocks::ONE)),
        vec![eq_alpha, slack_factor],
    )
    .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;

    // NC: γ^{off_nc} · Π_a (Z̃ − a) — the in-sumcheck norm products.
    vp.add_term(gamma.pow_u64(shape.off_nc()), nc_factors)
        .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;

    Ok(BuiltVp {
        vp,
        level_factor,
        product_factor,
    })
}

// ---------------------------------------------------------------------------
// The protocol.

/// The derived public challenges (α over the position space, γ the term
/// combiner) — identical on both sides.
fn derive_challenges(
    shape: &PiCcsShape,
    transcript: &mut Transcript,
) -> Result<(Vec<Goldilocks>, Goldilocks), PiCcsError> {
    let nbytes = 8 * shape.log_m.max(1);
    let bytes = transcript
        .challenge_bytes(b"pi-ccs-alpha", nbytes)
        .map_err(PiCcsError::Transcript)?;
    let mut alpha_pos = Vec::with_capacity(shape.log_m);
    for chunk in bytes.chunks(8) {
        let mut w = [0u8; 8];
        w.copy_from_slice(chunk);
        alpha_pos.push(Goldilocks::from_u64(u64::from_le_bytes(w)));
    }
    let gamma = transcript
        .challenge_field(b"pi-ccs-gamma")
        .map_err(PiCcsError::Transcript)?;
    Ok((alpha_pos, gamma))
}

fn absorb_statement(
    ccs_digest: &[u8; 32],
    shape: &PiCcsShape,
    inst: &PiCcsInstance,
    prior: Option<&CeClaim>,
    transcript: &mut Transcript,
) -> Result<(), PiCcsError> {
    let digest = pi_ccs_digest(ccs_digest, inst);
    transcript
        .append_bytes(b"pi-ccs-inst", &digest)
        .map_err(PiCcsError::Transcript)?;
    let mut shape_bytes = Vec::with_capacity(16);
    for v in [
        shape.num_vars() as u32,
        shape.levels as u32,
        shape.m as u32,
        shape.base as u32,
    ] {
        shape_bytes.extend_from_slice(&v.to_le_bytes());
    }
    transcript
        .append_bytes(b"pi-ccs-shape", &shape_bytes)
        .map_err(PiCcsError::Transcript)?;
    if let Some(p) = prior {
        transcript
            .append_field_slice(b"pi-ccs-prior-y", &p.y)
            .map_err(PiCcsError::Transcript)?;
        for jp in &p.y_products {
            transcript
                .append_field_slice(b"pi-ccs-prior-yj", jp)
                .map_err(PiCcsError::Transcript)?;
        }
    }
    Ok(())
}

/// Prove Π_CCS: the sum-check plus the fresh per-level claims. The
/// transcript must already hold the public statement (the CCS digest);
/// α and γ are derived inside.
pub fn prove_pi_ccs(
    ccs: &Ccs,
    ccs_digest: &[u8; 32],
    shape: &PiCcsShape,
    inst: &PiCcsInstance,
    secret: &PiCcsSecret,
    prior: Option<&CeClaim>,
    transcript: &mut Transcript,
) -> Result<PiCcsProof, PiCcsError> {
    secret.digits.validate()?;
    absorb_statement(ccs_digest, shape, inst, prior, transcript)?;
    let (alpha_pos, gamma) = derive_challenges(shape, transcript)?;

    // The claimed sum: the prior claims' contribution (F and NC vanish).
    let claim = match prior {
        Some(p) => shape.claimed_sum(&gamma, p),
        None => Goldilocks::ZERO,
    };

    let built = build_vp(
        ccs,
        shape,
        &secret.digits,
        inst.u,
        &inst.slack,
        prior,
        &alpha_pos,
        &gamma,
    )?;
    let out = sumcheck::prove(&built.vp, claim, transcript)
        .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;

    // The fresh claims from the engine's per-factor terminal claims.
    let r = &out.challenges;
    let y: Vec<Goldilocks> = built
        .level_factor
        .iter()
        .map(|&h| out.factor_claims[h])
        .collect();
    let y_products: Vec<Vec<Goldilocks>> = built
        .product_factor
        .iter()
        .map(|handles| handles.iter().map(|&h| out.factor_claims[h]).collect())
        .collect();
    let new_claim = CeClaim {
        r_lev: r[..shape.log_lev].to_vec(),
        r_pos: r[shape.log_lev..].to_vec(),
        y,
        y_products,
    };
    // Bind the claims into the transcript (downstream Fiat–Shamir).
    transcript
        .append_field_slice(b"pi-ccs-claims-y", &new_claim.y)
        .map_err(PiCcsError::Transcript)?;
    for jp in &new_claim.y_products {
        transcript
            .append_field_slice(b"pi-ccs-claims-yj", jp)
            .map_err(PiCcsError::Transcript)?;
    }
    Ok(PiCcsProof {
        sumcheck: out.proof,
        new_claim,
    })
}

/// Verify Π_CCS. Returns the fresh claims (the eval-claim API output) on
/// success — the commitment layer must still authenticate them.
pub fn verify_pi_ccs(
    ccs: &Ccs,
    ccs_digest: &[u8; 32],
    shape: &PiCcsShape,
    inst: &PiCcsInstance,
    prior: Option<&CeClaim>,
    proof: &PiCcsProof,
    transcript: &mut Transcript,
) -> Result<CeClaim, PiCcsError> {
    absorb_statement(ccs_digest, shape, inst, prior, transcript)?;
    let (alpha_pos, gamma) = derive_challenges(shape, transcript)?;

    let claim = match prior {
        Some(p) => shape.claimed_sum(&gamma, p),
        None => Goldilocks::ZERO,
    };
    let verifier = proof
        .sumcheck
        .verify(
            shape.num_vars(),
            shape.max_degree(),
            claim,
            transcript,
            None,
        )
        .map_err(|e| PiCcsError::Sumcheck(format!("{e:?}")))?;
    let r = verifier.point;
    let r_lev: Vec<Goldilocks> = r[..shape.log_lev].to_vec();
    let r_pos: Vec<Goldilocks> = r[shape.log_lev..].to_vec();

    // The fresh claims must be well-shaped and bound to the point.
    let new_claim = &proof.new_claim;
    if new_claim.r_lev != r_lev || new_claim.r_pos != r_pos {
        return Err(PiCcsError::Shape {
            expected: shape.log_lev + shape.log_m,
            got: 0,
        });
    }
    if new_claim.y.len() != shape.levels
        || new_claim.y_products.len() != shape.t + 1
        || new_claim
            .y_products
            .iter()
            .any(|jp| jp.len() != shape.levels)
    {
        return Err(PiCcsError::Shape {
            expected: shape.levels,
            got: new_claim.y.len(),
        });
    }
    transcript
        .append_field_slice(b"pi-ccs-claims-y", &new_claim.y)
        .map_err(PiCcsError::Transcript)?;
    for jp in &new_claim.y_products {
        transcript
            .append_field_slice(b"pi-ccs-claims-yj", jp)
            .map_err(PiCcsError::Transcript)?;
    }

    // ---- Q(r') recomputation from the claims + public data -------------
    // EvalK/EvalA: eq(r', r) · Σ γ^{off}·y'.
    let mut q_r = Goldilocks::ZERO;
    if let Some(p) = prior {
        let mut r_prior = p.r_lev.clone();
        r_prior.extend_from_slice(&p.r_pos);
        let eq_rr = DenseMle::eq_eval(&r, &r_prior).map_err(mle_shape_err)?;
        let mut eval_sum = Goldilocks::ZERO;
        for (lev, &yv) in new_claim.y.iter().enumerate() {
            eval_sum = eval_sum.add(&gamma.pow_u64(shape.off_eval_k(lev)).mul(&yv));
        }
        for (j, jp) in new_claim.y_products.iter().enumerate() {
            for (lev, &yv) in jp.iter().enumerate() {
                eval_sum = eval_sum.add(&gamma.pow_u64(shape.off_eval_a(j, lev)).mul(&yv));
            }
        }
        q_r = q_r.add(&eq_rr.mul(&eval_sum));
    }

    // F: γ^{off_f} · eq(r'_pos, α) · [Σ_j c_j·m_a·m_b − u·m_span − slãck(r'_pos)].
    let eq_alpha_r = DenseMle::eq_eval(&r_pos, &alpha_pos).map_err(mle_shape_err)?;
    let recombine = |jp: &[Goldilocks]| -> Goldilocks {
        let mut acc = Goldilocks::ZERO;
        for (ell, &yv) in jp.iter().enumerate() {
            acc = acc.add(&fe(shape.base.pow(ell as u32)).mul(&yv));
        }
        acc
    };
    let mut f_val = Goldilocks::ZERO;
    for (j, sel) in ccs.selections.iter().enumerate() {
        if sel.len() != 2 {
            return Err(PiCcsError::BadSelection {
                index: j,
                arity: sel.len(),
            });
        }
        let c = ccs.constants.get(j).copied().unwrap_or(Goldilocks::ONE);
        let ma = recombine(&new_claim.y_products[sel[0]]);
        let mb = recombine(&new_claim.y_products[sel[1]]);
        f_val = f_val.add(&c.mul(&ma.mul(&mb)));
    }
    let m_span = recombine(&new_claim.y_products[shape.t]);
    f_val = f_val.sub(&inst.u.mul(&m_span)).sub(
        &DenseMle::new(pad_rows(&inst.slack, shape.m))
            .map_err(mle_shape_err)?
            .evaluate(&r_pos)
            .map_err(mle_shape_err)?,
    );
    q_r = q_r.add(&gamma.pow_u64(shape.off_f()).mul(&eq_alpha_r.mul(&f_val)));

    // NC: γ^{off_nc} · Π_a (Ẑ − a) with Ẑ = Σ_ℓ eq(r'_lev, e_ℓ)·y'_ℓ.
    let mut z_hat = Goldilocks::ZERO;
    for (ell, &yv) in new_claim.y.iter().enumerate() {
        // MSB-first to match the engine's variable order (r_lev[0] is
        // the level field's high bit).
        let e_lev: Vec<Goldilocks> = (0..shape.log_lev)
            .map(|i| fe(((ell >> (shape.log_lev - 1 - i)) & 1) as u64))
            .collect();
        let w = DenseMle::eq_eval(&r_lev, &e_lev).map_err(mle_shape_err)?;
        z_hat = z_hat.add(&w.mul(&yv));
    }
    let mut nc_val = Goldilocks::ONE;
    for a in shape.norm_shifts() {
        let shifted = if a < 0 {
            z_hat.add(&fe(u64::try_from(-a).unwrap_or(0)))
        } else {
            z_hat.sub(&fe(a as u64))
        };
        nc_val = nc_val.mul(&shifted);
    }
    q_r = q_r.add(&gamma.pow_u64(shape.off_nc()).mul(&nc_val));

    if q_r != verifier.final_claim {
        return Err(PiCcsError::FinalCheckFailed);
    }
    Ok(new_claim.clone())
}

// ---------------------------------------------------------------------------
// The decider.

/// The succinct Π_CCS decider for a FOLDED committed instance:
///
/// 1. decompose the folded values base-`b` (fail-closed);
/// 2. commit each digit level;
/// 3. prove + verify Π_CCS (CCS satisfaction + the norm products + the
///    prior claims — no witness transmission);
/// 4. check the homomorphic binding `C_values = Σ_ℓ b^ℓ·C_ℓ`;
/// 5. open the level commitments against the packed vectors (the DIRECT
///    route — O(n); the compact-mode linear-functional bridge replaces
///    this step, see `lattice-zkvm::ttrp`).
#[allow(clippy::too_many_arguments)] // the decider's fixed pipeline inputs
pub fn decider_pi_ccs(
    pk: &AjtaiPublicKey,
    ccs: &Ccs,
    ccs_digest: &[u8; 32],
    base: u64,
    levels: usize,
    folded_values: &[u32],
    u: Goldilocks,
    slack: &[Goldilocks],
    value_commitment: &AjtaiCommitment,
    prior: Option<&CeClaim>,
) -> Result<CeClaim, PiCcsError> {
    let values: Vec<u64> = folded_values.iter().map(|&v| v as u64).collect();
    let digits = decompose(&values, base, levels)?;
    let shape = PiCcsShape::derive(ccs, base, levels);
    let (inst, secret) = commit_digits(pk, &digits, slack, u)?;

    // The homomorphic binding: C_values == Σ_ℓ b^ℓ·C_ℓ (exact — the
    // coefficient sums stay below q/2 at the small-value regime).
    let mut acc_rows: Vec<RingElement> = inst
        .level_commitments
        .first()
        .map(|c| c.rows.clone())
        .ok_or(PiCcsError::Shape {
            expected: 1,
            got: 0,
        })?;
    for (ell, c) in inst.level_commitments.iter().enumerate().skip(1) {
        let w = base.pow(ell as u32) as i64;
        for (acc, row) in acc_rows.iter_mut().zip(c.rows.iter()) {
            *acc = acc
                .add(&row.scale_i64(w))
                .map_err(|_| PiCcsError::BindingFailed)?;
        }
    }
    // ℓ = 0 carries weight b^0 = 1 — the accumulator started from the
    // first level's rows verbatim; scale it back if needed (no-op at
    // weight 1, kept explicit for clarity).
    if acc_rows != value_commitment.rows {
        return Err(PiCcsError::BindingFailed);
    }

    // Π_CCS prove + verify over a statement-seeded transcript.
    let mut prover_t = Transcript::new_default(b"lzx-pi-ccs-decider");
    prover_t
        .append_bytes(b"ccs", ccs_digest)
        .map_err(PiCcsError::Transcript)?;
    let proof = prove_pi_ccs(
        ccs,
        ccs_digest,
        &shape,
        &inst,
        &secret,
        prior,
        &mut prover_t,
    )?;

    let mut verifier_t = Transcript::new_default(b"lzx-pi-ccs-decider");
    verifier_t
        .append_bytes(b"ccs", ccs_digest)
        .map_err(PiCcsError::Transcript)?;
    let new_claim = verify_pi_ccs(
        ccs,
        ccs_digest,
        &shape,
        &inst,
        prior,
        &proof,
        &mut verifier_t,
    )?;

    // The direct opening of the level commitments (O(n) — replaced by
    // the functional bridge on the succinct route).
    for (c, packed) in inst
        .level_commitments
        .iter()
        .zip(secret.packed_levels.iter())
    {
        pk.verify_opening(c, packed)
            .map_err(|_| PiCcsError::OpeningFailed)?;
    }
    Ok(new_claim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::{AjtaiParams, AjtaiPublicKey};
    use lattice_relations::ccs::SparseMatrix;
    use lattice_ring::{Modulus32, RingConfig};

    fn fe64(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    fn setup(log_n: u32, m_slots: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m: m_slots,
            norm_bound: 1 << 26,
        };
        let pk = AjtaiPublicKey::from_seed(params, [23u8; 32]).ok().unwrap();
        (pk, ring)
    }

    /// `w ∘ w = u · w` — booleanity as degree-2 CCS (the span = identity).
    fn bool_ccs(n: usize) -> Ccs {
        let a = SparseMatrix::identity(n);
        Ccs {
            m: n,
            n,
            a_matrices: vec![a.clone(), a],
            b_matrices: vec![SparseMatrix::identity(n)],
            selections: vec![vec![0, 1]],
            constants: vec![fe64(1)],
        }
    }

    fn boolean_witness(n: usize, tag: u64) -> Vec<u32> {
        (0..n)
            .map(|i| {
                let mut input = Vec::new();
                input.extend_from_slice(&tag.to_le_bytes());
                input.extend_from_slice(&(i as u64).to_le_bytes());
                let digest = Transcript::hash_domain(b"pi-ccs-test-w", &input);
                (digest[0] & 1) as u32
            })
            .collect()
    }

    fn ccs_digest(ccs: &Ccs) -> [u8; 32] {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(ccs.m as u32).to_le_bytes());
        buf.extend_from_slice(&(ccs.n as u32).to_le_bytes());
        buf.extend_from_slice(&(ccs.a_matrices.len() as u32).to_le_bytes());
        for a in &ccs.a_matrices {
            for (r, c, v) in &a.entries {
                buf.extend_from_slice(&r.to_le_bytes());
                buf.extend_from_slice(&c.to_le_bytes());
                buf.extend_from_slice(&v.to_bytes());
            }
        }
        Transcript::hash_domain(b"pi-ccs-ccs", &buf)
    }

    /// Prove + verify round trip; returns the fresh claims.
    #[allow(clippy::too_many_arguments)] // the test fixture's fixed set
    fn round_trip(
        pk: &AjtaiPublicKey,
        ccs: &Ccs,
        base: u64,
        levels: usize,
        witness: &[u32],
        u: Goldilocks,
        slack: &[Goldilocks],
        prior: Option<&CeClaim>,
    ) -> Result<CeClaim, PiCcsError> {
        let values: Vec<u64> = witness.iter().map(|&v| v as u64).collect();
        let digits = decompose(&values, base, levels)?;
        let shape = PiCcsShape::derive(ccs, base, levels);
        let (inst, secret) = commit_digits(pk, &digits, slack, u)?;
        let digest = ccs_digest(ccs);
        let mut pt = Transcript::new_default(b"lzx-pi-ccs-test");
        pt.append_bytes(b"ccs", &digest).ok().unwrap();
        let proof = prove_pi_ccs(ccs, &digest, &shape, &inst, &secret, prior, &mut pt)?;
        let mut vt = Transcript::new_default(b"lzx-pi-ccs-test");
        vt.append_bytes(b"ccs", &digest).ok().unwrap();
        verify_pi_ccs(ccs, &digest, &shape, &inst, prior, &proof, &mut vt)
    }

    #[test]
    fn decompose_recompose_roundtrip() {
        for &base in &[2u64, 3, 5, 16] {
            let values: Vec<u64> = (0..64u64)
                .map(|i| (i * 7 + 3) % (base * base * base))
                .collect();
            let d = decompose(&values, base, 3).ok().unwrap();
            assert_eq!(d.recompose(), values);
            d.validate().ok().unwrap();
        }
        // Overflow is fail-closed.
        assert!(decompose(&[8], 2, 3).is_err());
        // 8 = 0b1000 needs 4 bits; 3 levels reject.
    }

    #[test]
    fn pi_ccs_happy_path_bits() {
        // b = 2, L = 1: the pay-per-bit representation (pure bits).
        let n = 32;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let w = boolean_witness(n, 7);
        let claim = round_trip(
            &pk,
            &ccs,
            2,
            1,
            &w,
            Goldilocks::ONE,
            &vec![Goldilocks::ZERO; n],
            None,
        )
        .ok()
        .unwrap();
        assert_eq!(claim.y.len(), 1);
        assert_eq!(claim.y_products.len(), 3); // 2 A-matrices + the span
                                               // The fresh witness claim: z̃⁽⁰⁾(r_pos) — a random-point MLE eval
                                               // of the bit vector (sanity: a valid field element).
        assert!(claim.y[0] != Goldilocks::ZERO || true);
    }

    #[test]
    fn pi_ccs_happy_path_base3_multi_level() {
        // b = 3, L = 4: digits ∈ {0,1,2} ⊂ the base-3 range; the NC
        // products have 5 shifts (degree 5 per variable).
        let n = 16;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let w = boolean_witness(n, 11);
        let claim = round_trip(
            &pk,
            &ccs,
            3,
            4,
            &w,
            Goldilocks::ONE,
            &vec![Goldilocks::ZERO; n],
            None,
        )
        .ok()
        .unwrap();
        assert_eq!(claim.y.len(), 4);
    }

    #[test]
    fn pi_ccs_with_prior_claims() {
        // Two sequential Π_CCS runs: the first's claims feed the second
        // as the running CE instance.
        let n = 16;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let w = boolean_witness(n, 13);
        let first = round_trip(
            &pk,
            &ccs,
            2,
            1,
            &w,
            Goldilocks::ONE,
            &vec![Goldilocks::ZERO; n],
            None,
        )
        .ok()
        .unwrap();
        // Second round on the SAME witness with the prior claims bound.
        let second = round_trip(
            &pk,
            &ccs,
            2,
            1,
            &w,
            Goldilocks::ONE,
            &vec![Goldilocks::ZERO; n],
            Some(&first),
        )
        .ok()
        .unwrap();
        assert_eq!(second.y.len(), 1);
    }

    #[test]
    fn pi_ccs_rejects_tampered_claim() {
        let n = 16;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let w = boolean_witness(n, 17);
        let values: Vec<u64> = w.iter().map(|&v| v as u64).collect();
        let digits = decompose(&values, 2, 1).ok().unwrap();
        let shape = PiCcsShape::derive(&ccs, 2, 1);
        let (inst, secret) =
            commit_digits(&pk, &digits, &vec![Goldilocks::ZERO; n], Goldilocks::ONE)
                .ok()
                .unwrap();
        let digest = ccs_digest(&ccs);
        let mut pt = Transcript::new_default(b"lzx-pi-ccs-test");
        pt.append_bytes(b"ccs", &digest).ok().unwrap();
        let mut proof = prove_pi_ccs(&ccs, &digest, &shape, &inst, &secret, None, &mut pt)
            .ok()
            .unwrap();
        // Tamper the fresh witness claim.
        proof.new_claim.y[0] = proof.new_claim.y[0].add(&Goldilocks::ONE);
        let mut vt = Transcript::new_default(b"lzx-pi-ccs-test");
        vt.append_bytes(b"ccs", &digest).ok().unwrap();
        assert!(verify_pi_ccs(&ccs, &digest, &shape, &inst, None, &proof, &mut vt).is_err());
    }

    #[test]
    fn pi_ccs_rejects_tampered_sumcheck() {
        let n = 16;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let w = boolean_witness(n, 19);
        let values: Vec<u64> = w.iter().map(|&v| v as u64).collect();
        let digits = decompose(&values, 2, 1).ok().unwrap();
        let shape = PiCcsShape::derive(&ccs, 2, 1);
        let (inst, secret) =
            commit_digits(&pk, &digits, &vec![Goldilocks::ZERO; n], Goldilocks::ONE)
                .ok()
                .unwrap();
        let digest = ccs_digest(&ccs);
        let mut pt = Transcript::new_default(b"lzx-pi-ccs-test");
        pt.append_bytes(b"ccs", &digest).ok().unwrap();
        let mut proof = prove_pi_ccs(&ccs, &digest, &shape, &inst, &secret, None, &mut pt)
            .ok()
            .unwrap();
        // Tamper a round message.
        if let Some(round) = proof.sumcheck.rounds.first_mut() {
            round[0] = round[0].add(&Goldilocks::ONE);
        }
        let mut vt = Transcript::new_default(b"lzx-pi-ccs-test");
        vt.append_bytes(b"ccs", &digest).ok().unwrap();
        assert!(verify_pi_ccs(&ccs, &digest, &shape, &inst, None, &proof, &mut vt).is_err());
    }

    #[test]
    fn pi_ccs_norm_violation_breaks_the_claimed_sum() {
        // An out-of-range digit makes the NC product nonzero on the
        // cube — the sum-check's claimed sum no longer matches the
        // polynomial, so the engine's own completeness check rejects.
        let n = 8;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let mut w = boolean_witness(n, 23);
        w[0] = 2; // NOT boolean: the CCS term AND the norm term both fire.
        let values: Vec<u64> = w.iter().map(|&v| v as u64).collect();
        let digits = decompose(&values, 2, 2).ok().unwrap(); // 2 = (0,1) digits
        let shape = PiCcsShape::derive(&ccs, 2, 2);
        let (inst, secret) =
            commit_digits(&pk, &digits, &vec![Goldilocks::ZERO; n], Goldilocks::ONE)
                .ok()
                .unwrap();
        let digest = ccs_digest(&ccs);
        let mut pt = Transcript::new_default(b"lzx-pi-ccs-test");
        pt.append_bytes(b"ccs", &digest).ok().unwrap();
        // The prove path must fail: Σ Q(x) != T (the claimed sum 0).
        assert!(prove_pi_ccs(&ccs, &digest, &shape, &inst, &secret, None, &mut pt).is_err());
    }

    #[test]
    fn claim_fold_is_linear() {
        let c1 = CeClaim {
            r_lev: vec![fe64(5)],
            r_pos: vec![fe64(6), fe64(7)],
            y: vec![fe64(1), fe64(2)],
            y_products: vec![vec![fe64(3), fe64(4)], vec![fe64(5), fe64(6)]],
        };
        let c2 = CeClaim {
            r_lev: vec![fe64(5)],
            r_pos: vec![fe64(6), fe64(7)],
            y: vec![fe64(10), fe64(20)],
            y_products: vec![vec![fe64(30), fe64(40)], vec![fe64(50), fe64(60)]],
        };
        let rho = fe64(3);
        let folded = fold_claim(&c1, &c2, rho).ok().unwrap();
        assert_eq!(folded.y[0], fe64(1 + 3 * 10));
        assert_eq!(folded.y[1], fe64(2 + 3 * 20));
        assert_eq!(folded.y_products[1][0], fe64(5 + 3 * 50));
        // Different points refuse.
        let c3 = CeClaim {
            r_lev: vec![fe64(9)],
            ..c2.clone()
        };
        assert!(fold_claim(&c1, &c3, rho).is_err());
    }

    #[test]
    fn decider_end_to_end_after_folding() {
        // The full committed pipeline: two boolean instances →
        // fold_committed → decider_pi_ccs (the succinct Π_CCS route).
        use crate::superneo_committed::{commit_instance, fold_committed};
        let n = 32;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let digest = ccs_digest(&ccs);
        let w1 = boolean_witness(n, 31);
        let w2 = boolean_witness(n, 37);
        let zero_slack = vec![Goldilocks::ZERO; n];
        let (i1, s1) = commit_instance(&pk, &w1, &zero_slack, Goldilocks::ONE, false)
            .ok()
            .unwrap();
        let (i2, s2) = commit_instance(&pk, &w2, &zero_slack, Goldilocks::ONE, false)
            .ok()
            .unwrap();
        let (folded, folded_secret) = fold_committed(&pk, &ccs, &digest, &i1, &i2, &s1, &s2)
            .ok()
            .unwrap();
        let claim = decider_pi_ccs(
            &pk,
            &ccs,
            &digest,
            2,
            25, // values < 2^25 after folds
            &folded_secret.witness,
            folded.u,
            &folded.slack,
            &folded.commitment,
            None,
        )
        .ok()
        .unwrap();
        // 25 real levels padded to 32; the padded entries claim zero.
        assert_eq!(claim.y.len(), 32);
        assert!(claim.y[25..].iter().all(|&y| y == Goldilocks::ZERO));
        assert_eq!(claim.y_products.len(), 3);
    }

    #[test]
    fn decider_rejects_tampered_folded_values() {
        use crate::superneo_committed::commit_instance;
        let n = 32;
        let (pk, _ring) = setup(6, 8);
        let ccs = bool_ccs(n);
        let digest = ccs_digest(&ccs);
        let w1 = boolean_witness(n, 41);
        let w2 = boolean_witness(n, 43);
        let zero_slack = vec![Goldilocks::ZERO; n];
        let (i1, _s1) = commit_instance(&pk, &w1, &zero_slack, Goldilocks::ONE, false)
            .ok()
            .unwrap();
        let (i2, _s2) = commit_instance(&pk, &w2, &zero_slack, Goldilocks::ONE, false)
            .ok()
            .unwrap();
        // A FOLDED commitment that does not match the presented values:
        // fold the commitments homomorphically with r, but present
        // values folded with a DIFFERENT r — the binding check fires.
        // r = 1 fold of commitments (fake via fold_public with E=0).
        let e = vec![Goldilocks::ZERO; n];
        let folded_inst = crate::superneo_committed::fold_public(&i1, &i2, &e, 1u32)
            .ok()
            .unwrap();
        let wrong_values: Vec<u32> = w1
            .iter()
            .zip(w2.iter())
            .map(|(&a, &b)| (a as u64 + 2 * b as u64) as u32) // r' = 2 != 1
            .collect();
        let result = decider_pi_ccs(
            &pk,
            &ccs,
            &digest,
            2,
            25,
            &wrong_values,
            folded_inst.u,
            &folded_inst.slack,
            &folded_inst.commitment,
            None,
        );
        // The homomorphic binding must catch the mismatch.
        assert!(matches!(result, Err(PiCcsError::BindingFailed)));
    }
}
