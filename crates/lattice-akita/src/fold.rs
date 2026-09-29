//! A1 — the Akita fold core (ePrint 2026/1983, §5, "One Fold for an
//! Opening Batch"): the keystone that replaces the full-witness opening
//! path of `pcs.rs` with the paper's committed fold.
//!
//! Structure realized here (Fig 5, Eq 2-8, Eq 104-109):
//!
//! * **Two-tier keys** — tier A (the *witness tier*, Eq 4: `t_i = A·s_i`
//!   over the source digits), tier B/D (the *sliced outer commitment
//!   tier*: the sliced outer matrix `B` binds the inner-image digits `t̂_i`
//!   block by block — Eq 5 — and the shared opening matrix `D` binds the
//!   concatenated partial digits `ê` — Eq 6).
//! * **G⁻¹ source decomposition** (Eq 3): every source block `f_i ∈ R^M`
//!   is digit-decomposed into short ring coordinates
//!   `s_i = G⁻¹_{b,M}(f_i) ∈ R^{Mδ}` so the fold operates on short data.
//! * **Sparse fixed-weight challenges** `c_i` (§5.3, the TAU-class
//!   families of `lattice_core::short_challenge`), sampled from the
//!   transcript **only after** the partial-binding payloads `u`, `v` and
//!   the scalar claim `vR` are absorbed — "both the inner images and the
//!   evaluation partials are fixed before the coefficients used to
//!   compare them are known" (§4, p.13).
//! * **Response** `z = Σ_i c_i·s_i` (Eq 104) with **response
//!   digitization** `z = G_z·ẑ` at response base `b_z` (Eq 105 and the
//!   `G_z := I_{Mδ} ⊗ (1, b_z, …)` map).
//! * **Fold equations** — the two challenge-dependent identities the fold
//!   must preserve:
//!   * Eq 7 (inner consistency): `A·z = Σ_i c_i·G_{b1,nA}(t̂_i)`;
//!   * Eq 8 (fold evaluation): `a^⊤·G_{b,M}(z) = Σ_i c_i·G_{b1,1}(ê_i)`.
//! * **Successor witness** `L = [ẑ | ê | t̂]` (the main part of Eq 106),
//!   committed under the successor Ajtai key and passed to the next level.
//!
//! LZX realization notes (kernel scale, honestly stated): in the paper the
//! digit segments `t̂`, `ê` are *bound* by the public payloads `u`, `v` and
//! authenticated by the fused relation sum-check of §7.4 against the
//! committed successor witness (items A2-A4); the terminal of §8.2 instead
//! reveals and directly checks. This module follows the terminal
//! discipline — the proof carries the digit segments in the clear, the
//! verifier checks the binding chain `u = B·t̂`, `v = D·ê`, both fold
//! equations, the response digitization, and the Ajtai successor
//! commitment. The fold structure itself (keys, decomposition, challenge
//! ordering, equations, successor) is the paper's, and the recursion
//! driver (item A5) chains these folds into the terminal.

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiParams, AjtaiPublicKey};
use lattice_core::norm_budget::{NormBudget, NormBudgetError};
use lattice_core::short_challenge::{
    ShortChallenge, ShortChallengeError, ShortChallengeFamily, ShortChallengeSpec,
};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_ring::{RingConfig, RingElement};

/// Fold geometry and digit depths (the public `Fold` record of Eq 83).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldParams {
    /// Ring dimension `d_A = 2^{log_n}` (all rings share it here; the
    /// paper's mixed-dimension rows collapse to the common maximum).
    pub log_n: u32,
    /// Number of source blocks `B` (Eq 81).
    pub num_blocks: usize,
    /// Block length `M` (ring elements per source block).
    pub block_len: usize,
    /// Source digit base `b` (power of two).
    pub source_base: u64,
    /// Source digit depth `δ` per coefficient (`b^δ ≥ q`).
    pub source_digits: usize,
    /// Inner matrix rows `n_A` (tier A height).
    pub inner_rows: usize,
    /// Inner-image digit base `b1`.
    pub inner_base: u64,
    /// Inner-image digit depth `δ1` (`b1^{δ1} ≥ q`).
    pub inner_digits: usize,
    /// Outer sliced matrix rows `n_B` (tier B height, reused per slice —
    /// one slice per block, §2.4 "divide these images into consecutive
    /// slices and reuse one narrower matrix on each slice").
    pub outer_rows: usize,
    /// Shared opening matrix rows `n_D` (tier D height).
    pub opening_rows: usize,
    /// Response base `b_z` (Eq 105).
    pub response_base: u64,
    /// Response digit depth `τ` (must cover the certified response bound).
    pub response_digits: usize,
    /// Fixed-weight challenge family weight `TAU`.
    pub challenge_weight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldError {
    Ring(lattice_ring::RingError),
    Ajtai(AjtaiError),
    ShortChallenge(ShortChallengeError),
    Transcript(TranscriptError),
    NormBudget(NormBudgetError),
    Decomposition { coefficient: usize },
    /// The response does not fit the response digit depth (Eq 111).
    ResponseTooLarge { bound: u64, capacity: u64 },
    /// A digit lies outside its balanced alphabet.
    DigitOutOfRange { digit: i64, bound: i64 },
    ChallengeMismatch,
    OuterBinding { block: usize },
    OpeningBinding,
    Equation7,
    Equation8,
    ScalarClaim,
    SuccessorBinding,
    /// The revealed response does not recompose from its digits.
    ResponseRecomposition,
    Shape { expected: usize, got: usize },
}

impl FoldError {
    /// True iff the failure is the norm-budget hard gate.
    #[allow(clippy::needless_range_loop)]
pub fn is_wraparound(&self) -> bool {
        matches!(self, FoldError::NormBudget(NormBudgetError::Wraparound { .. }))
    }
}

impl From<NormBudgetError> for FoldError {
    fn from(e: NormBudgetError) -> Self {
        FoldError::NormBudget(e)
    }
}

/// The two-tier key material: A (witness tier), B (sliced outer tier),
/// D (shared opening tier), and the successor commitment key.
pub struct FoldKeys {
    pub ring: RingConfig,
    /// `A ∈ R^{n_A × Mδ}` — tier A, the inner commitment matrix (Eq 4).
    pub a_matrix: Vec<Vec<RingElement>>,
    /// `B ∈ R^{n_B × n_A·δ1}` — tier B, the sliced outer matrix (Eq 5),
    /// applied once per block (one slice per block).
    pub b_matrix: Vec<Vec<RingElement>>,
    /// `D ∈ R^{n_D × B·δ1}` — the shared opening matrix (Eq 6).
    pub d_matrix: Vec<Vec<RingElement>>,
    /// The successor commitment key (binds `L = [ẑ|ê|t̂]`).
    pub successor_pk: AjtaiPublicKey,
}

impl FoldKeys {
    /// Derive all tiers from a seed (public-coin setup; the schedule
    /// digest binds the shapes).
    #[allow(clippy::needless_range_loop)]
pub fn from_seed(params: &FoldParams, seed: [u8; 32]) -> Result<Self, FoldError> {
        let ring = RingConfig::new(lattice_ring::Modulus32::Q_32, params.log_n)
            .map_err(FoldError::Ring)?;
        let md = params.block_len * params.source_digits;
        let a_matrix = derive_matrix(
            &ring,
            params.inner_rows,
            md,
            b"akita-fold-A",
            &seed,
        );
        let slice_width = params.inner_rows * params.inner_digits;
        let b_matrix = derive_matrix(
            &ring,
            params.outer_rows,
            slice_width,
            b"akita-fold-B",
            &seed,
        );
        let open_width = params.num_blocks * params.inner_digits;
        let d_matrix = derive_matrix(
            &ring,
            params.opening_rows,
            open_width,
            b"akita-fold-D",
            &seed,
        );
        // Successor Ajtai key sized for |L| = Mδτ + B·δ1 + B·n_A·δ1.
        let l_len = successor_len(params);
        let ajtai_params = AjtaiParams {
            ring: ring.clone(),
            k: 1,
            m: l_len,
            // Successor digits are bounded by the largest carried alphabet.
            norm_bound: u32::try_from(
                params
                    .inner_base
                    .max(params.response_base)
                    .div_ceil(2),
            )
            .unwrap_or(u32::MAX),
        };
        let successor_pk = AjtaiPublicKey::from_seed(ajtai_params, seed).map_err(FoldError::Ajtai)?;
        Ok(FoldKeys {
            ring,
            a_matrix,
            b_matrix,
            d_matrix,
            successor_pk,
        })
    }
}

/// `|L| = Mδτ + B·δ1 + B·n_A·δ1` (Eq 106 main segments).
#[allow(clippy::needless_range_loop)]
pub fn successor_len(params: &FoldParams) -> usize {
    params.block_len * params.source_digits * params.response_digits
        + params.num_blocks * params.inner_digits
        + params.num_blocks * params.inner_rows * params.inner_digits
}

fn derive_matrix(
    ring: &RingConfig,
    rows: usize,
    cols: usize,
    domain: &[u8],
    seed: &[u8],
) -> Vec<Vec<RingElement>> {
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            let idx = (r * cols + c) as u64;
            row.push(ring.uniform_from_seed(domain, seed, idx));
        }
        out.push(row);
    }
    out
}

// ---------------------------------------------------------------------------
// Gadget decomposition G / G⁻¹ over ring elements (Eq 3, §3.3).
// ---------------------------------------------------------------------------

/// Balanced representative of a coefficient in `[−q/2, q/2)`.
fn balanced(coeff: u32, q: u32) -> i64 {
    if coeff <= q / 2 {
        coeff as i64
    } else {
        coeff as i64 - q as i64
    }
}

/// `G⁻¹_{base,δ}` on one ring element: decompose every balanced
/// coefficient into `δ` balanced base-`b` digits; digit `u` of all
/// coefficients forms ring element `u` of the output (the layout of
/// Eq 107 restricted to one block).
#[allow(clippy::needless_range_loop)]
pub fn decompose_element(
    ring: &RingConfig,
    elem: &RingElement,
    base: u64,
    digits: usize,
) -> Result<Vec<RingElement>, FoldError> {
    let q = ring.modulus.q;
    let n = ring.n();
    let mut out = vec![vec![0u32; n]; digits];
    for (j, &c) in elem.coeffs().iter().enumerate() {
        let mut rem = balanced(c, q);
        for u in 0..digits {
            let du = rem.rem_euclid(base as i64);
            let d = if du > base as i64 / 2 {
                du - base as i64
            } else {
                du
            };
            out[u][j] = ring.modulus.reduce_i64(d);
            rem = (rem - d) / base as i64;
        }
        if rem != 0 {
            return Err(FoldError::Decomposition { coefficient: j });
        }
    }
    Ok(out.into_iter().map(|cs| RingElement::from_coeffs(ring, cs)).collect())
}

/// `G_{base}` recomposition of one ring element's digits (the inverse of
/// [`decompose_element`]): `f = Σ_u b^u·s_u mod q`.
#[allow(clippy::needless_range_loop)]
pub fn recompose_element(
    ring: &RingConfig,
    digits: &[RingElement],
    base: u64,
) -> Result<RingElement, FoldError> {
    let q = ring.modulus.q as i128;
    let n = ring.n();
    let mut coeffs = vec![0u32; n];
    for u in 0..digits.len() {
        let power = (base as i128).pow(u as u32) % q;
        for j in 0..n {
            let d = balanced(digits[u].coeffs()[j], ring.modulus.q) as i128;
            coeffs[j] = ring.modulus.reduce_u64(
                ((coeffs[j] as i128 + d * power).rem_euclid(q)) as u64,
            );
        }
    }
    Ok(RingElement::from_coeffs(ring, coeffs))
}

/// Decompose a whole block vector `f_i ∈ R^M` into `s_i ∈ R^{Mδ}`
/// (the paper's `s_i = G⁻¹_{b,M}(f_i)`).
#[allow(clippy::needless_range_loop)]
pub fn decompose_block(
    ring: &RingConfig,
    block: &[RingElement],
    base: u64,
    digits: usize,
) -> Result<Vec<RingElement>, FoldError> {
    let mut out = Vec::with_capacity(block.len() * digits);
    for f in block {
        out.extend(decompose_element(ring, f, base, digits)?);
    }
    Ok(out)
}

/// Recompose a block vector from its digit segments.
#[allow(clippy::needless_range_loop)]
pub fn recompose_block(
    ring: &RingConfig,
    digits: &[RingElement],
    base: u64,
    digits_per_elem: usize,
) -> Result<Vec<RingElement>, FoldError> {
    if digits.len() % digits_per_elem != 0 {
        return Err(FoldError::Shape {
            expected: digits_per_elem,
            got: digits.len(),
        });
    }
    let mut out = Vec::with_capacity(digits.len() / digits_per_elem);
    for chunk in digits.chunks(digits_per_elem) {
        out.push(recompose_element(ring, chunk, base)?);
    }
    Ok(out)
}

/// Check every digit lies in the balanced alphabet `[−base/2, base/2]`
/// (the premise Eq 9 certifies via item A3; the decomposer of
/// [`decompose_element`] emits exactly this alphabet).
#[allow(clippy::needless_range_loop)]
pub fn digits_in_range(ring: &RingConfig, elems: &[RingElement], base: u64) -> Result<(), FoldError> {
    let q = ring.modulus.q;
    let half = base as i64 / 2;
    for e in elems {
        for &c in e.coeffs() {
            let d = balanced(c, q);
            if d < -half || d > half {
                return Err(FoldError::DigitOutOfRange { digit: d, bound: half });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The fold prover and verifier.
// ---------------------------------------------------------------------------

/// The public opening geometry: the point segments that fix the
/// within-block weights `a` (Eq 96: `a_x = eq(ρ_pos, x)`) and the block
/// weights `χ_blk` (Eq 131).
#[derive(Clone, Debug)]
pub struct OpeningPoint {
    /// `ρ_pos` — one coordinate per block-length axis (log₂ M values).
    pub pos: Vec<Goldilocks>,
    /// `ρ_blk` — one coordinate per block axis (log₂ B values).
    pub blk: Vec<Goldilocks>,
    /// The claimed field value forwarded as the scalar target (Eq 103
    /// with a single claim: weight one).
    pub value: Goldilocks,
}

impl OpeningPoint {
    /// Within-block ring weights `a ∈ R^M` (eq over `ρ_pos`, embedded as
    /// ring constants).
    #[allow(clippy::needless_range_loop)]
pub fn block_weights(&self, ring: &RingConfig) -> Vec<RingElement> {
        let eq = DenseMle::eq_extension(&self.pos);
        (0..eq.evaluations.len())
            .map(|i| ring.constant(eq.evaluations[i].to_canonical_u64() as u32))
            .collect()
    }

    /// Block weights `χ_blk(i) = eq(ρ_blk, i)` embedded as ring constants.
    #[allow(clippy::needless_range_loop)]
pub fn chi_blk(&self, ring: &RingConfig, num_blocks: usize) -> Vec<RingElement> {
        let eq = DenseMle::eq_extension(&self.blk);
        (0..num_blocks)
            .map(|i| {
                let v = eq.evaluations.get(i).copied().unwrap_or(Goldilocks::ZERO);
                ring.constant(v.to_canonical_u64() as u32)
            })
            .collect()
    }
}

/// A fold proof: the public payloads, the response layer, and the
/// successor binding.
#[derive(Clone, Debug)]
pub struct FoldProof {
    /// Sliced outer images `u_i = B·t̂_i` per block (Eq 5, one slice per
    /// block — the sliced tier).
    pub outer_images: Vec<Vec<RingElement>>,
    /// Shared opening image `v = D·ê` (Eq 6).
    pub opening_image: Vec<RingElement>,
    /// The scalar claim `vR = Σ_i χ_blk(i)·e_i` (Eq 2).
    pub scalar_claim: RingElement,
    /// Partial digits `ê_i` (revealed — the §8.2 terminal discipline;
    /// the paper binds these through `v` and the fused sum-check).
    pub partial_digits: Vec<Vec<RingElement>>,
    /// Inner-image digits `t̂_i` (revealed; bound through `u_i`).
    pub inner_digits: Vec<Vec<RingElement>>,
    /// The response `z = Σ_i c_i·s_i` (Eq 104), revealed.
    pub response: Vec<RingElement>,
    /// Response digits `ẑ` with `z = G_z·ẑ` (Eq 105).
    pub response_digits: Vec<RingElement>,
    /// The fold challenges `c_i` (recomputed by the verifier from the
    /// transcript; carried for the driver).
    pub challenges: Vec<RingElement>,
    /// The successor commitment binding `L = [ẑ | ê | t̂]`.
    pub successor_commitment: AjtaiCommitment,
    /// Norm-budget bookkeeping after this fold.
    pub budget: NormBudget,
}

/// Certified response bound for the fold (the growth law of
/// `NormBudget::fold` evaluated for the worst challenge): every
/// coefficient of `z = Σ_i c_i·s_i` satisfies
/// `‖z‖∞ ≤ Σ_i Γ_{c_i}·⌈√n⌉·(b/2)`.
#[allow(clippy::needless_range_loop)]
pub fn certified_response_bound(
    params: &FoldParams,
    challenges: &[ShortChallenge],
) -> u64 {
    let sqrt_n = lattice_core::norm_budget::ceil_sqrt(1u64 << params.log_n);
    let digit_bound = params.source_base / 2;
    challenges
        .iter()
        .map(|c| c.gamma_c().saturating_mul(sqrt_n).saturating_mul(digit_bound))
        .sum()
}

/// Sample the fold challenges from the transcript **after** the partial
/// payloads are absorbed (the Eq-7/8 ordering: the compared data is fixed
/// before the comparison coefficients are drawn).
fn sample_challenges(
    params: &FoldParams,
    transcript: &mut Transcript,
) -> Result<Vec<ShortChallenge>, FoldError> {
    let spec = ShortChallengeSpec {
        n: 1usize << params.log_n,
        family: ShortChallengeFamily::FixedWeight {
            weight: params.challenge_weight,
            amplitude: 1,
        },
    };
    spec.validate().map_err(FoldError::ShortChallenge)?;
    let mut out = Vec::with_capacity(params.num_blocks);
    for i in 0..params.num_blocks {
        let seed = transcript
            .challenge_bytes(b"akita-fold-challenge", 32)
            .map_err(FoldError::Transcript)?;
        let mut domain = Vec::with_capacity(seed.len() + 8);
        domain.extend_from_slice(&seed);
        domain.extend_from_slice(&(i as u64).to_le_bytes());
        out.push(spec.sample(&domain).map_err(FoldError::ShortChallenge)?);
    }
    Ok(out)
}

fn embed(ring: &RingConfig, c: &ShortChallenge) -> RingElement {
    RingElement::from_signed(ring, &c.coefficients)
}

/// Prover-side fold state (the source list).
pub struct FoldSource {
    /// The source blocks `f_i ∈ R^M`.
    pub blocks: Vec<Vec<RingElement>>,
    /// Norm budget of the incoming witness (digits at depth δ).
    pub budget: NormBudget,
}

impl FoldSource {
    #[allow(clippy::needless_range_loop)]
pub fn new(blocks: Vec<Vec<RingElement>>, digit_bound: u64) -> Self {
        FoldSource {
            blocks,
            budget: NormBudget::fresh(digit_bound),
        }
    }
}

/// Run one fold: bind the partials, sample the challenges, form the
/// response, and commit the successor witness (Fig 5 steps 1-4).
#[allow(clippy::needless_range_loop)]
pub fn prove_fold(
    params: &FoldParams,
    keys: &FoldKeys,
    source: &FoldSource,
    point: &OpeningPoint,
    transcript: &mut Transcript,
) -> Result<FoldProof, FoldError> {
    let ring = &keys.ring;
    if source.blocks.len() != params.num_blocks {
        return Err(FoldError::Shape {
            expected: params.num_blocks,
            got: source.blocks.len(),
        });
    }
    let md = params.block_len * params.source_digits;

    // ---- Step 1: bind the partial opening evaluations (Eq 2, 4, 6). ----
    let a = point.block_weights(ring);
    if a.len() != params.block_len {
        return Err(FoldError::Shape {
            expected: params.block_len,
            got: a.len(),
        });
    }
    let mut s_blocks: Vec<Vec<RingElement>> = Vec::with_capacity(params.num_blocks);
    let mut t_hat: Vec<Vec<RingElement>> = Vec::with_capacity(params.num_blocks);
    let mut e_hat: Vec<Vec<RingElement>> = Vec::with_capacity(params.num_blocks);
    let mut partials: Vec<RingElement> = Vec::with_capacity(params.num_blocks);
    for block in &source.blocks {
        if block.len() != params.block_len {
            return Err(FoldError::Shape {
                expected: params.block_len,
                got: block.len(),
            });
        }
        // s_i = G⁻¹_{b,M}(f_i).
        let s = decompose_block(ring, block, params.source_base, params.source_digits)?;
        // e_i = ⟨a, f_i⟩ (Eq 2).
        let mut e = ring.zero();
        for (aw, f) in a.iter().zip(block.iter()) {
            e = e.add(&aw.mul(f).map_err(FoldError::Ring)?).map_err(FoldError::Ring)?;
        }
        // t_i = A·s_i (Eq 4).
        let mut t = vec![ring.zero(); params.inner_rows];
        for (r, row) in keys.a_matrix.iter().enumerate() {
            let mut acc = ring.zero();
            for (a_ent, sj) in row.iter().zip(s.iter()) {
                acc = acc
                    .add(&a_ent.mul(sj).map_err(FoldError::Ring)?)
                    .map_err(FoldError::Ring)?;
            }
            t[r] = acc;
        }
        // Digits of the inner images and the partial.
        let mut that = Vec::with_capacity(params.inner_rows * params.inner_digits);
        for te in &t {
            that.extend(decompose_element(ring, te, params.inner_base, params.inner_digits)?);
        }
        let ehat = decompose_element(ring, &e, params.inner_base, params.inner_digits)?;
        s_blocks.push(s);
        t_hat.push(that);
        e_hat.push(ehat);
        partials.push(e);
    }
    // Sliced outer images: u_i = B·t̂_i per block (Eq 5, one slice/block).
    let mut outer_images = Vec::with_capacity(params.num_blocks);
    for that in &t_hat {
        let mut img = Vec::with_capacity(params.outer_rows);
        for row in &keys.b_matrix {
            let mut acc = ring.zero();
            for (b_ent, d) in row.iter().zip(that.iter()) {
                acc = acc
                    .add(&b_ent.mul(d).map_err(FoldError::Ring)?)
                    .map_err(FoldError::Ring)?;
            }
            img.push(acc);
        }
        outer_images.push(img);
    }
    // Shared opening image: v = D·ê over the concatenated partial digits.
    let e_concat: Vec<RingElement> = e_hat.iter().flatten().cloned().collect();
    let mut opening_image = Vec::with_capacity(params.opening_rows);
    for row in &keys.d_matrix {
        let mut acc = ring.zero();
        for (d_ent, e) in row.iter().zip(e_concat.iter()) {
            acc = acc
                .add(&d_ent.mul(e).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
        opening_image.push(acc);
    }
    // Scalar claim vR = Σ_i χ_blk(i)·e_i (Eq 2).
    let chi = point.chi_blk(ring, params.num_blocks);
    let mut v_r = ring.zero();
    for (ch, e) in chi.iter().zip(partials.iter()) {
        v_r = v_r
            .add(&ch.mul(e).map_err(FoldError::Ring)?)
            .map_err(FoldError::Ring)?;
    }

    // Absorb the payloads BEFORE any fold challenge (the §4 ordering).
    for img in &outer_images {
        for e in img {
            transcript
                .append_bytes(b"akita-fold-u", &e.to_bytes())
                .map_err(FoldError::Transcript)?;
        }
    }
    for e in &opening_image {
        transcript
            .append_bytes(b"akita-fold-v", &e.to_bytes())
            .map_err(FoldError::Transcript)?;
    }
    transcript
        .append_bytes(b"akita-fold-vR", &v_r.to_bytes())
        .map_err(FoldError::Transcript)?;

    // ---- Step 2: challenges and the response (Eq 104). ----
    let challenges = sample_challenges(params, transcript)?;
    let c_ring: Vec<RingElement> = challenges.iter().map(|c| embed(ring, c)).collect();
    let mut z = vec![ring.zero(); md];
    for (c, s) in c_ring.iter().zip(s_blocks.iter()) {
        for (zj, sj) in z.iter_mut().zip(s.iter()) {
            *zj = zj.add(&c.mul(sj).map_err(FoldError::Ring)?).map_err(FoldError::Ring)?;
        }
    }
    // Response bound: certified law, then the norm-budget hard gate.
    let bound = certified_response_bound(params, &challenges);
    let capacity = params.response_base.pow(params.response_digits as u32) / 2;
    if bound >= capacity {
        return Err(FoldError::ResponseTooLarge { bound, capacity });
    }
    let budget = source.budget.fold(
        challenges
            .iter()
            .map(|c| c.gamma_c())
            .max()
            .unwrap_or(0),
        lattice_core::norm_budget::ceil_sqrt(ring.n() as u64),
        params.source_base / 2,
        ring.modulus.q as u64 / 2,
        u64::MAX,
    )?;
    // Response digitization ẑ = G⁻¹_z(z) (Eq 105).
    let mut response_digits = Vec::with_capacity(md * params.response_digits);
    for ze in &z {
        response_digits.extend(decompose_element(
            ring,
            ze,
            params.response_base,
            params.response_digits,
        )?);
    }

    // ---- Step 3: bind the successor witness (Eq 106). ----
    let mut l_vec: Vec<RingElement> = Vec::with_capacity(successor_len(params));
    l_vec.extend(response_digits.iter().cloned());
    for ehat in &e_hat {
        l_vec.extend(ehat.iter().cloned());
    }
    for that in &t_hat {
        l_vec.extend(that.iter().cloned());
    }
    let l_padded = keys
        .successor_pk
        .pad_to_m(&l_vec)
        .map_err(FoldError::Ajtai)?;
    let successor_commitment = keys
        .successor_pk
        .commit(&l_padded)
        .map_err(FoldError::Ajtai)?;
    transcript
        .append_bytes(b"akita-fold-L", &successor_commitment.to_bytes())
        .map_err(FoldError::Transcript)?;

    Ok(FoldProof {
        outer_images,
        opening_image,
        scalar_claim: v_r,
        partial_digits: e_hat,
        inner_digits: t_hat,
        response: z,
        response_digits,
        challenges: c_ring,
        successor_commitment,
        budget,
    })
}

/// Verify one fold: the binding chain, both fold equations, the response
/// digitization, and the successor commitment.
#[allow(clippy::needless_range_loop)]
pub fn verify_fold(
    params: &FoldParams,
    keys: &FoldKeys,
    point: &OpeningPoint,
    proof: &FoldProof,
    transcript: &mut Transcript,
) -> Result<(), FoldError> {
    let ring = &keys.ring;
    let md = params.block_len * params.source_digits;
    if proof.response.len() != md
        || proof.partial_digits.len() != params.num_blocks
        || proof.inner_digits.len() != params.num_blocks
        || proof.challenges.len() != params.num_blocks
    {
        return Err(FoldError::Shape {
            expected: params.num_blocks,
            got: proof.partial_digits.len(),
        });
    }

    // 1. Replay the payload absorption and the challenge sampling.
    for img in &proof.outer_images {
        for e in img {
            transcript
                .append_bytes(b"akita-fold-u", &e.to_bytes())
                .map_err(FoldError::Transcript)?;
        }
    }
    for e in &proof.opening_image {
        transcript
            .append_bytes(b"akita-fold-v", &e.to_bytes())
            .map_err(FoldError::Transcript)?;
    }
    transcript
        .append_bytes(b"akita-fold-vR", &proof.scalar_claim.to_bytes())
        .map_err(FoldError::Transcript)?;
    let challenges = sample_challenges(params, transcript)?;
    let c_ring: Vec<RingElement> = challenges.iter().map(|c| embed(ring, c)).collect();
    for (got, expect) in proof.challenges.iter().zip(c_ring.iter()) {
        if got != expect {
            return Err(FoldError::ChallengeMismatch);
        }
    }
    transcript
        .append_bytes(b"akita-fold-L", &proof.successor_commitment.to_bytes())
        .map_err(FoldError::Transcript)?;

    // 2. Digit alphabets (Eq 9 premise; the sumcheck of A3 replaces the
    //    coefficient-wise scan in the production path).
    for that in &proof.inner_digits {
        digits_in_range(ring, that, params.inner_base)?;
    }
    for ehat in &proof.partial_digits {
        digits_in_range(ring, ehat, params.inner_base)?;
    }
    digits_in_range(ring, &proof.response_digits, params.response_base)?;
    // Response digitization: z = G_z·ẑ (Eq 105).
    let z_check = recompose_block(
        ring,
        &proof.response_digits,
        params.response_base,
        params.response_digits,
    )?;
    if z_check != proof.response {
        return Err(FoldError::ResponseRecomposition);
    }

    // 3. Recompose the inner images and partials from their digits.
    let mut t_images: Vec<Vec<RingElement>> = Vec::with_capacity(params.num_blocks);
    for that in &proof.inner_digits {
        t_images.push(recompose_block(
            ring,
            that,
            params.inner_base,
            params.inner_digits,
        )?);
    }
    let mut partials: Vec<RingElement> = Vec::with_capacity(params.num_blocks);
    for ehat in &proof.partial_digits {
        partials.push(recompose_element(ring, ehat, params.inner_base)?);
    }

    // 4. Outer binding: u_i = B·t̂_i per slice (Eq 5).
    for (i, that) in proof.inner_digits.iter().enumerate() {
        let img = proof.outer_images.get(i).ok_or(FoldError::Shape {
            expected: params.num_blocks,
            got: proof.outer_images.len(),
        })?;
        for (r, row) in keys.b_matrix.iter().enumerate() {
            let mut acc = ring.zero();
            for (b_ent, d) in row.iter().zip(that.iter()) {
                acc = acc
                    .add(&b_ent.mul(d).map_err(FoldError::Ring)?)
                    .map_err(FoldError::Ring)?;
            }
            let got = img.get(r).ok_or(FoldError::Shape {
                expected: params.outer_rows,
                got: img.len(),
            })?;
            if &acc != got {
                return Err(FoldError::OuterBinding { block: i });
            }
        }
    }
    // 5. Opening binding: v = D·ê (Eq 6).
    let e_concat: Vec<RingElement> = proof.partial_digits.iter().flatten().cloned().collect();
    for (r, row) in keys.d_matrix.iter().enumerate() {
        let mut acc = ring.zero();
        for (d_ent, e) in row.iter().zip(e_concat.iter()) {
            acc = acc
                .add(&d_ent.mul(e).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
        let got = proof.opening_image.get(r).ok_or(FoldError::Shape {
            expected: params.opening_rows,
            got: proof.opening_image.len(),
        })?;
        if &acc != got {
            return Err(FoldError::OpeningBinding);
        }
    }

    // 6. Fold equation Eq 7: A·z = Σ_i c_i·G_{b1,nA}(t̂_i).
    for (r, row) in keys.a_matrix.iter().enumerate() {
        let mut lhs = ring.zero();
        for (a_ent, zj) in row.iter().zip(proof.response.iter()) {
            lhs = lhs
                .add(&a_ent.mul(zj).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
        let mut rhs = ring.zero();
        for (c, t) in c_ring.iter().zip(t_images.iter()) {
            rhs = rhs
                .add(&c.mul(&t[r]).map_err(FoldError::Ring)?)
                .map_err(FoldError::Ring)?;
        }
        if lhs != rhs {
            return Err(FoldError::Equation7);
        }
    }

    // 7. Fold equation Eq 8: a^⊤ G_{b,M}(z) = Σ_i c_i·G_{b1,1}(ê_i).
    let a = point.block_weights(ring);
    let folded_source = recompose_block(
        ring,
        &proof.response,
        params.source_base,
        params.source_digits,
    )?;
    let mut lhs = ring.zero();
    for (aw, f) in a.iter().zip(folded_source.iter()) {
        lhs = lhs
            .add(&aw.mul(f).map_err(FoldError::Ring)?)
            .map_err(FoldError::Ring)?;
    }
    let mut rhs = ring.zero();
    for (c, e) in c_ring.iter().zip(partials.iter()) {
        rhs = rhs
            .add(&c.mul(e).map_err(FoldError::Ring)?)
            .map_err(FoldError::Ring)?;
    }
    if lhs != rhs {
        return Err(FoldError::Equation8);
    }

    // 8. Scalar claim (Eq 2): vR = Σ_i χ_blk(i)·e_i.
    let chi = point.chi_blk(ring, params.num_blocks);
    let mut v_r = ring.zero();
    for (ch, e) in chi.iter().zip(partials.iter()) {
        v_r = v_r
            .add(&ch.mul(e).map_err(FoldError::Ring)?)
            .map_err(FoldError::Ring)?;
    }
    if v_r != proof.scalar_claim {
        return Err(FoldError::ScalarClaim);
    }

    // 9. Successor binding (Eq 106): C_L opens [ẑ | ê | t̂].
    let mut l_vec: Vec<RingElement> = Vec::with_capacity(successor_len(params));
    l_vec.extend(proof.response_digits.iter().cloned());
    for ehat in &proof.partial_digits {
        l_vec.extend(ehat.iter().cloned());
    }
    for that in &proof.inner_digits {
        l_vec.extend(that.iter().cloned());
    }
    let l_padded = keys
        .successor_pk
        .pad_to_m(&l_vec)
        .map_err(FoldError::Ajtai)?;
    keys.successor_pk
        .verify_opening(&proof.successor_commitment, &l_padded)
        .map_err(FoldError::Ajtai)?;

    Ok(())
}

/// The successor witness `L = [ẑ | ê | t̂]` (Eq 106 main segments) of a
/// proven fold — the next level's source.
#[allow(clippy::needless_range_loop)]
pub fn successor_witness(params: &FoldParams, proof: &FoldProof) -> Vec<RingElement> {
    let mut l_vec: Vec<RingElement> = Vec::with_capacity(successor_len(params));
    l_vec.extend(proof.response_digits.iter().cloned());
    for ehat in &proof.partial_digits {
        l_vec.extend(ehat.iter().cloned());
    }
    for that in &proof.inner_digits {
        l_vec.extend(that.iter().cloned());
    }
    l_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> FoldParams {
        FoldParams {
            log_n: 4,
            num_blocks: 4,
            block_len: 2,
            source_base: 256,
            source_digits: 4,
            inner_rows: 1,
            inner_base: 2048,
            inner_digits: 3,
            outer_rows: 1,
            opening_rows: 1,
            response_base: 256,
            response_digits: 2,
            challenge_weight: 8,
        }
    }

    fn keys(p: &FoldParams) -> FoldKeys {
        FoldKeys::from_seed(p, [41u8; 32]).ok().unwrap()
    }

    fn source(p: &FoldParams, ring: &RingConfig, tag: &[u8]) -> FoldSource {
        let blocks: Vec<Vec<RingElement>> = (0..p.num_blocks)
            .map(|i| {
                let mut tag_i = tag.to_vec();
                tag_i.extend_from_slice(&(i as u64).to_le_bytes());
                (0..p.block_len)
                    .map(|j| {
                        let mut t2 = tag_i.clone();
                        t2.push(j as u8);
                        ring.random(&t2)
                    })
                    .collect()
            })
            .collect();
        FoldSource::new(blocks, p.source_base / 2)
    }

    fn point(p: &FoldParams) -> OpeningPoint {
        OpeningPoint {
            pos: (1..=p.block_len.ilog2().max(1) as usize)
                .map(|i| Goldilocks::from_u64((i * 997) as u64))
                .collect(),
            blk: (1..=p.num_blocks.ilog2().max(1) as usize)
                .map(|i| Goldilocks::from_u64((i * 1231) as u64))
                .collect(),
            value: Goldilocks::from_u64(7),
        }
    }

    #[test]
    fn gadget_roundtrip_exact() {
        let ring = RingConfig::new(lattice_ring::Modulus32::Q_32, 4).ok().unwrap();
        for seed in [b"a".as_slice(), b"b".as_slice(), b"c".as_slice()] {
            let f = ring.random(seed);
            let digits = decompose_element(&ring, &f, 256, 4).ok().unwrap();
            let back = recompose_element(&ring, &digits, 256).ok().unwrap();
            assert_eq!(back, f, "G⁻¹ then G must be the identity mod q");
            digits_in_range(&ring, &digits, 256).ok().unwrap();
        }
    }

    #[test]
    fn fold_equations_identity_exact() {
        // Eq 7-8 hold EXACTLY for honest folds — the algebraic identity
        // behind the protocol, checked independently of any transcript.
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"id");
        let point = point(&p);
        // Honest prover internals.
        let a = point.block_weights(ring);
        let mut s_blocks = Vec::new();
        let mut t_images = Vec::new();
        let mut partials = Vec::new();
        for block in &src.blocks {
            let s = decompose_block(ring, block, p.source_base, p.source_digits)
                .ok()
                .unwrap();
            let mut e = ring.zero();
            for (aw, f) in a.iter().zip(block.iter()) {
                e = e.add(&aw.mul(f).ok().unwrap()).ok().unwrap();
            }
            let mut t = vec![ring.zero(); p.inner_rows];
            for (r, row) in keys.a_matrix.iter().enumerate() {
                let mut acc = ring.zero();
                for (a_ent, sj) in row.iter().zip(s.iter()) {
                    acc = acc.add(&a_ent.mul(sj).ok().unwrap()).ok().unwrap();
                }
                t[r] = acc;
            }
            s_blocks.push(s);
            t_images.push(t);
            partials.push(e);
        }
        // Fixed challenges (deterministic short vectors).
        let c: Vec<RingElement> = (0..p.num_blocks)
            .map(|i| {
                let signed: Vec<i64> = (0..ring.n())
                    .map(|j| if (i + j) % 5 == 0 { 1 } else if (i + j) % 7 == 0 { -1 } else { 0 })
                    .collect();
                RingElement::from_signed(ring, &signed)
            })
            .collect();
        // z = Σ c_i s_i.
        let md = p.block_len * p.source_digits;
        let mut z = vec![ring.zero(); md];
        for (ci, s) in c.iter().zip(s_blocks.iter()) {
            for (zj, sj) in z.iter_mut().zip(s.iter()) {
                *zj = zj.add(&ci.mul(sj).ok().unwrap()).ok().unwrap();
            }
        }
        // Eq 7: A·z == Σ c_i t_i — linearity of A.
        for (r, row) in keys.a_matrix.iter().enumerate() {
            let mut lhs = ring.zero();
            for (a_ent, zj) in row.iter().zip(z.iter()) {
                lhs = lhs.add(&a_ent.mul(zj).ok().unwrap()).ok().unwrap();
            }
            let mut rhs = ring.zero();
            for (ci, t) in c.iter().zip(t_images.iter()) {
                rhs = rhs.add(&ci.mul(&t[r]).ok().unwrap()).ok().unwrap();
            }
            assert_eq!(lhs, rhs, "Eq 7 (inner consistency) must hold exactly");
        }
        // Eq 8: ⟨a, G(z)⟩ == Σ c_i e_i — linearity through the gadget.
        let folded = recompose_block(ring, &z, p.source_base, p.source_digits)
            .ok()
            .unwrap();
        let mut lhs = ring.zero();
        for (aw, f) in a.iter().zip(folded.iter()) {
            lhs = lhs.add(&aw.mul(f).ok().unwrap()).ok().unwrap();
        }
        let mut rhs = ring.zero();
        for (ci, e) in c.iter().zip(partials.iter()) {
            rhs = rhs.add(&ci.mul(e).ok().unwrap()).ok().unwrap();
        }
        assert_eq!(lhs, rhs, "Eq 8 (fold evaluation) must hold exactly");
    }

    #[test]
    fn fold_prove_verify_honest() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"honest");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-akita-fold");
        assert!(verify_fold(&p, &keys, &point, &proof, &mut vt).is_ok());
        // Successor witness length matches Eq 106 accounting.
        let l = successor_witness(&p, &proof);
        assert_eq!(l.len(), successor_len(&p));
        // Norm budget accounting: one fold consumed.
        assert_eq!(proof.budget.folds(), 1);
        assert!(proof.budget.beta() > 0);
    }

    #[test]
    fn fold_zero_sources_identity() {
        // The identity fold: all-zero sources give the zero response, the
        // trivial equations, and an all-zero successor.
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let blocks: Vec<Vec<RingElement>> = (0..p.num_blocks)
            .map(|_| (0..p.block_len).map(|_| ring.zero()).collect())
            .collect();
        let src = FoldSource::new(blocks, p.source_base / 2);
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold-0");
        let proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        assert!(proof.response.iter().all(|z| z.is_zero()));
        let mut vt = Transcript::new_default(b"lzx-akita-fold-0");
        assert!(verify_fold(&p, &keys, &point, &proof, &mut vt).is_ok());
    }

    #[test]
    fn fold_tampered_response_rejected() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"tamper-z");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let mut proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        // Tamper one response coefficient: Eq 7 breaks.
        let mut coeffs = proof.response[0].coeffs().to_vec();
        coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
        proof.response[0] = RingElement::from_coeffs(ring, coeffs);
        let mut vt = Transcript::new_default(b"lzx-akita-fold");
        assert!(verify_fold(&p, &keys, &point, &proof, &mut vt).is_err());
    }

    #[test]
    fn fold_tampered_digits_rejected() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"tamper-t");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let mut proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        // Tamper an inner-image digit: outer binding (Eq 5) or Eq 7 breaks.
        let mut coeffs = proof.inner_digits[0][0].coeffs().to_vec();
        coeffs[1] = (coeffs[1] + 1) % ring.modulus.q;
        proof.inner_digits[0][0] = RingElement::from_coeffs(ring, coeffs);
        let mut vt = Transcript::new_default(b"lzx-akita-fold");
        assert!(verify_fold(&p, &keys, &point, &proof, &mut vt).is_err());
    }

    #[test]
    fn fold_out_of_alphabet_digit_rejected() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"alphabet");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let mut proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        // Force a response digit outside the balanced alphabet: a value
        // near q/2 has balanced magnitude ≈ q/2, far beyond ±b/2.
        let mut coeffs = proof.response_digits[0].coeffs().to_vec();
        coeffs[0] = (ring.modulus.q / 2 + 5) % ring.modulus.q;
        proof.response_digits[0] = RingElement::from_coeffs(ring, coeffs);
        let mut vt = Transcript::new_default(b"lzx-akita-fold");
        let err = verify_fold(&p, &keys, &point, &proof, &mut vt).err().unwrap();
        assert!(matches!(err, FoldError::DigitOutOfRange { .. }));
    }

    #[test]
    fn fold_challenge_ordering_enforced() {
        // The Eq-7/8 challenges must be drawn AFTER the payloads: a proof
        // whose challenges were sampled from a payload-free transcript
        // (the buggy ordering) does not verify under the correct one.
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"ordering");
        let point = point(&p);
        // Buggy prover: sample challenges BEFORE absorbing u, v, vR.
        let mut t_bug = Transcript::new_default(b"lzx-akita-fold");
        let stale = sample_challenges(&p, &mut t_bug).ok().unwrap();
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let mut proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        // Substitute the stale challenges into the proof.
        proof.challenges = stale.iter().map(|c| embed(ring, c)).collect();
        let mut vt = Transcript::new_default(b"lzx-akita-fold");
        let err = verify_fold(&p, &keys, &point, &proof, &mut vt).err().unwrap();
        assert_eq!(err, FoldError::ChallengeMismatch);
    }

    #[test]
    fn fold_successor_tamper_rejected() {
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"tamper-L");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let mut proof = prove_fold(&p, &keys, &src, &point, &mut t).ok().unwrap();
        // Tamper the successor commitment: Ajtai binding must reject.
        let mut coeffs = proof.successor_commitment.rows[0].coeffs().to_vec();
        coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
        proof.successor_commitment.rows[0] = RingElement::from_coeffs(ring, coeffs);
        let mut vt = Transcript::new_default(b"lzx-akita-fold");
        assert!(verify_fold(&p, &keys, &point, &proof, &mut vt).is_err());
    }

    #[test]
    fn fold_response_capacity_gate() {
        // A response depth too small for the certified bound is refused
        // (Eq 111 encodability), not silently wrapped.
        let mut p = params();
        p.response_digits = 1; // capacity 128 << certified bound
        let keys = keys(&p);
        let ring = &keys.ring;
        let src = source(&p, ring, b"capacity");
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let err = prove_fold(&p, &keys, &src, &point, &mut t).err().unwrap();
        assert!(matches!(err, FoldError::ResponseTooLarge { .. }));
    }

    #[test]
    fn fold_norm_budget_gate() {
        // The hard wraparound gate: an accumulator already near q/2
        // refuses the fold rather than wrapping mod q.
        let p = params();
        let keys = keys(&p);
        let ring = &keys.ring;
        let mut src = source(&p, ring, b"gate");
        src.budget = NormBudget::fresh((ring.modulus.q as u64 / 2).saturating_sub(100));
        let point = point(&p);
        let mut t = Transcript::new_default(b"lzx-akita-fold");
        let err = prove_fold(&p, &keys, &src, &point, &mut t).err().unwrap();
        assert!(err.is_wraparound());
    }
}
