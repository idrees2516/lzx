//! The zero-knowledge sum-check machinery of §5.1/§5.2 (ePrint 2026/289).
//!
//! * **`MaskPoly`** — the XZZ+19 compact mask of Eq. (3):
//!   `G(X) = r₀ + Σ_k r_k(X_k)` per coordinate, with `r_k` a random
//!   univariate of individual degree `D` — `O(n·(1 + L·D))` coefficients.
//!   The mask's individual degree equals the **statement's** individual
//!   degree (map degree `d` plus one for the multilinear `eq(·, α)` batching
//!   factor — the paper's "same variables and individual degrees as f"
//!   convention), so the round messages are fully blinded.
//! * **`prove_masked_batched` / `verify_masked_batched`** — the §5.2
//!   accumulation sum-check: one Fiat–Shamir transcript over `L` variables
//!   carrying `n + m·n` statement coordinates with shared challenges:
//!   ```text
//!   first  (n):   Σ_b [ eq(b,α)·F̃_c(b) + γ·G'_c(b) ] = γ·e'_g,c
//!   update (m·n): Σ_b [ eq(β_j,b)·Ĝ_{j,c}(b) ]        = v_g,j,c
//!   ```
//!   where `F̃ := F − e` vanishes on the whole boolean cube (so the first
//!   claimed sum is the *public* `γ·e'_g`), and `Ĝ_j` is the
//!   **multilinearization** of the old mask `G_j` (its cube values' MLE).
//!   The multilinearization is the load-bearing detail the paper's update
//!   statements require: for degree-`D` masks the kernel identity
//!   `Σ_b eq(β,b)·G(b) = MLE(G|_cube)(β)` holds for the MLE — *not* for the
//!   raw polynomial — so the accumulator's stored claim `v_g` is the
//!   **kernel value** `MLE(G|_cube)(β)`, and the update statements'
//!   sum-check ends at the kernel values at the *new* point β (the [KS24]
//!   point-update, which keeps the accumulator from growing). The fresh
//!   mask's *true* evaluation `G'(β)` is transmitted separately because it
//!   enters the first statement's final claim (from which the fresh error
//!   `ẽ = (v − γ·G'(β))·eq(β,α)⁻¹` is extracted — the paper's Eq. (5)).
//! * Round messages are ascending coefficient vectors of length `d+2`
//!   (per-variable degree ≤ d+1); the multilinear update statements are
//!   zero-padded to the same width (a few redundant field elements).
//!
//! The prover evaluates `F̃` through a black-box closure at concrete
//! points and interpolates each round's univariate from `d+2` node
//! evaluations (`X = 0..d+1`) — the standard `(d+2)·2^{L−k}` per-round work.

use crate::util::{challenge_fp_vec, eq_eval, mle_eval};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkScError {
    Shape(&'static str),
    Transcript(lattice_core::transcript::TranscriptError),
    RoundConsistency { round: usize, coord: usize },
    DegreeBound { round: usize, got: usize, expected: usize },
}

impl core::fmt::Display for ZkScError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ZkScError::Shape(s) => write!(f, "shape: {s}"),
            ZkScError::Transcript(e) => write!(f, "transcript: {e}"),
            ZkScError::RoundConsistency { round, coord } => {
                write!(f, "round {round} inconsistent at coordinate {coord}")
            }
            ZkScError::DegreeBound { round, got, expected } => write!(
                f,
                "round {round} degree bound: got {got} coefficients, expected {expected}"
            ),
        }
    }
}

impl From<lattice_core::transcript::TranscriptError> for ZkScError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        ZkScError::Transcript(e)
    }
}

// ---------------------------------------------------------------------------
// MaskPoly
// ---------------------------------------------------------------------------

/// The XZZ+19 mask for an `n`-coordinate, `L`-variable system with
/// individual degree `D`: per coordinate `c`,
/// `G_c(X) = r_{c,0} + Σ_{k<L} r_{c,k}(X_k)` with each `r_{c,k}` of degree
/// ≤ D (coefficients indexed 1..=D — the constant lives in `r_{c,0}`).
#[derive(Clone, Debug)]
pub struct MaskPoly {
    /// Number of coordinates `n`.
    pub n: usize,
    /// Number of variables `L`.
    pub num_vars: usize,
    /// Maximum individual degree `D`.
    pub degree: usize,
    /// `const_terms[c]` = `r_{c,0}`.
    pub const_terms: Vec<Fp256>,
    /// `per_var[k][c][j-1]` = the `X_k^j` coefficient, j ∈ [1, D].
    pub per_var: Vec<Vec<Vec<Fp256>>>,
}

impl MaskPoly {
    fn from_coeff_slice(
        n: usize,
        num_vars: usize,
        degree: usize,
        coeffs: &[Fp256],
    ) -> MaskPoly {
        let mut const_terms = Vec::with_capacity(n);
        let mut per_var = vec![Vec::with_capacity(n); num_vars];
        let mut idx = 0;
        for _ in 0..n {
            const_terms.push(coeffs[idx]);
            idx += 1;
        }
        for k in 0..num_vars {
            for _ in 0..n {
                per_var[k].push(coeffs[idx..idx + degree].to_vec());
                idx += degree;
            }
        }
        MaskPoly {
            n,
            num_vars,
            degree,
            const_terms,
            per_var,
        }
    }

    /// Sample a fresh random mask from the transcript.
    pub fn sample(
        n: usize,
        num_vars: usize,
        degree: usize,
        transcript: &mut Transcript,
    ) -> Result<MaskPoly, ZkScError> {
        let total = n * (1 + num_vars * degree);
        let coeffs = challenge_fp_vec(transcript, b"mask-poly", total)?;
        Ok(Self::from_coeff_slice(n, num_vars, degree, &coeffs))
    }

    /// A random mask from a raw seed (the simulator's sampler).
    pub fn from_seed(n: usize, num_vars: usize, degree: usize, seed: &[u8]) -> MaskPoly {
        let total = n * (1 + num_vars * degree);
        let coeffs = crate::util::fp_vec_from_seed(b"mask-seed", seed, total);
        Self::from_coeff_slice(n, num_vars, degree, &coeffs)
    }

    /// The zero mask.
    pub fn zero(n: usize, num_vars: usize, degree: usize) -> MaskPoly {
        MaskPoly {
            n,
            num_vars,
            degree,
            const_terms: vec![Fp256::ZERO; n],
            per_var: vec![vec![vec![Fp256::ZERO; degree]; n]; num_vars],
        }
    }

    /// Evaluate all coordinates at a concrete point (the TRUE polynomial
    /// evaluation — used for the fresh mask's `G'(β)`).
    pub fn eval_at(&self, point: &[Fp256]) -> Result<Vec<Fp256>, ZkScError> {
        if point.len() != self.num_vars {
            return Err(ZkScError::Shape("mask point length"));
        }
        let mut out = self.const_terms.clone();
        for (k, xk) in point.iter().enumerate() {
            for c in 0..self.n {
                let coeffs = &self.per_var[k][c];
                let mut acc = Fp256::ZERO;
                for j in (0..self.degree).rev() {
                    acc = acc.mul(xk).add(&coeffs[j]);
                }
                out[c] = out[c].add(&acc.mul(xk));
            }
        }
        Ok(out)
    }

    /// The boolean-cube values, row-major over the cube (`values[b][c]`).
    pub fn cube_values(&self) -> Vec<Vec<Fp256>> {
        let count = 1usize << self.num_vars;
        let mut out = vec![vec![Fp256::ZERO; self.n]; count];
        for b in 0..count {
            let pt: Vec<Fp256> = (0..self.num_vars)
                .map(|k| Fp256::from_canonical_u64(((b >> k) & 1) as u64))
                .collect();
            if let Ok(v) = self.eval_at(&pt) {
                out[b] = v;
            }
        }
        out
    }

    /// The kernel value `MLE(G|_cube)(β)` — the accumulator's claim
    /// semantics (the KS24 point-update's invariant).
    pub fn kernel_eval(&self, beta: &[Fp256]) -> Result<Vec<Fp256>, ZkScError> {
        if beta.len() != self.num_vars {
            return Err(ZkScError::Shape("mask kernel point length"));
        }
        let vals = self.cube_values();
        let flat: Vec<Fp256> = vals.concat();
        // mle_eval works per coordinate: transpose.
        let mut out = Vec::with_capacity(self.n);
        for c in 0..self.n {
            let col: Vec<Fp256> = (0..vals.len()).map(|b| flat[b * self.n + c]).collect();
            out.push(mle_eval(&col, beta));
        }
        Ok(out)
    }

    /// The cube sum `Σ_b G(b)` per coordinate — closed form:
    /// `2^L·r_{c,0} + Σ_k 2^{L−1}·Σ_{j≥1} coeff_j`.
    pub fn cube_sum(&self) -> Vec<Fp256> {
        let two = Fp256::from_canonical_u64(2);
        let inv2 = two
            .inverse()
            .unwrap_or(Fp256::from_canonical_u64(1));
        let mut scale = Fp256::from_canonical_u64(1);
        for _ in 0..self.num_vars {
            scale = scale.mul(&two);
        }
        let half = scale.mul(&inv2);
        let mut out: Vec<Fp256> = self.const_terms.iter().map(|v| v.mul(&scale)).collect();
        for k in 0..self.num_vars {
            for c in 0..self.n {
                let s: Fp256 = self.per_var[k][c]
                    .iter()
                    .fold(Fp256::ZERO, |acc, v| acc.add(v));
                out[c] = out[c].add(&s.mul(&half));
            }
        }
        out
    }

    /// In-place acc += λ·other — the linear-combination closure under which
    /// the representation is stable (the KS24 no-growth property).
    pub fn add_scaled(&mut self, lambda: &Fp256, other: &MaskPoly) -> Result<(), ZkScError> {
        if self.n != other.n || self.num_vars != other.num_vars || self.degree != other.degree {
            return Err(ZkScError::Shape("mask combination shape"));
        }
        for c in 0..self.n {
            let t = self.const_terms[c].add(&lambda.mul(&other.const_terms[c]));
            self.const_terms[c] = t;
        }
        for k in 0..self.num_vars {
            for c in 0..self.n {
                for j in 0..self.degree {
                    let t = self.per_var[k][c][j].add(&lambda.mul(&other.per_var[k][c][j]));
                    self.per_var[k][c][j] = t;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Univariate helpers (integer-node interpolation)
// ---------------------------------------------------------------------------

/// Interpolate the ascending coefficient vector of the degree-≤`y.len()−1`
/// polynomial through the nodes `X = 0..y.len()−1` with values `y`.
pub fn interpolate_int_nodes(y: &[Fp256]) -> Vec<Fp256> {
    let d = y.len() - 1;
    let mut a = vec![vec![Fp256::ZERO; d + 2]; d + 1];
    for (i, row) in a.iter_mut().enumerate().take(d + 1) {
        let xi = Fp256::from_canonical_u64(i as u64);
        let mut p = Fp256::from_canonical_u64(1);
        for j in 0..=d {
            row[j] = p;
            p = p.mul(&xi);
        }
        row[d + 1] = y[i];
    }
    for col in 0..=d {
        let mut piv = col;
        while piv <= d && a[piv][col].is_zero() {
            piv += 1;
        }
        if piv > d {
            continue;
        }
        a.swap(col, piv);
        let inv = match a[col][col].inverse() {
            Some(v) => v,
            None => continue,
        };
        for j in col..=d + 1 {
            let t = a[col][j].mul(&inv);
            a[col][j] = t;
        }
        for r in 0..=d {
            if r != col && !a[r][col].is_zero() {
                let factor = a[r][col];
                for j in col..=d + 1 {
                    let t = a[r][j].sub(&factor.mul(&a[col][j]));
                    a[r][j] = t;
                }
            }
        }
    }
    (0..=d).map(|i| a[i][d + 1]).collect()
}

/// Evaluate an ascending coefficient vector at `x`.
pub fn poly_eval(coeffs: &[Fp256], x: &Fp256) -> Fp256 {
    let mut acc = Fp256::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc.mul(x).add(c);
    }
    acc
}

// ---------------------------------------------------------------------------
// The masked batched sum-check
// ---------------------------------------------------------------------------

/// A proof of the §5.2 accumulation sum-check.
#[derive(Clone, Debug)]
pub struct ZkSumcheckProof {
    /// `e'_g` — the fresh mask's cube sums (n coordinates).
    pub mask_cube_sums: Vec<Fp256>,
    /// Per round, per coordinate: ascending univariate coefficients
    /// (length d+2). Coordinate order: [first-statement 0..n) then
    /// [old-mask j: n..n+m·n), j-major.
    pub rounds: Vec<Vec<Vec<Fp256>>>,
    /// `v'_g = G'(β)` — the fresh mask's TRUE evaluation at the final point
    /// (enters the first statement's final claim; Eq. (5)'s `v'_g`).
    pub fresh_mask_eval: Vec<Fp256>,
    /// `κ' = MLE(G'|_cube)(β)` — the fresh mask's kernel value (the new
    /// accumulator's `v_g` contribution).
    pub fresh_mask_kernel: Vec<Fp256>,
}

/// Derived verification outputs (the claims the caller pins).
#[derive(Clone, Debug)]
pub struct ZkSumcheckOutput {
    /// The final point β.
    pub beta: Vec<Fp256>,
    /// The first statement's final evals `v_c = eq(β,α)·F̃_c(β) + γ·G'_c(β)`.
    pub first_final: Vec<Fp256>,
    /// The old masks' kernel values at the new point:
    /// `κ_{j,c} = MLE(G_j|_cube)(β)`.
    pub old_kernel_evals: Vec<Vec<Fp256>>,
    /// The fresh mask `G'` itself (prover-side: it becomes the new
    /// accumulator's `G` building block; the verifier ignores it).
    pub fresh_mask: MaskPoly,
}

/// The statement bundle for the accumulation sum-check.
pub struct MaskedBatchStatement<'a> {
    /// Number of variables `L`.
    pub num_vars: usize,
    /// Map degree `d` (round messages have degree ≤ d+1 → d+2 coefficients;
    /// masks carry individual degree d+1).
    pub map_degree: usize,
    /// First-statement coordinate count `n`.
    pub n: usize,
    /// The eq-batching point α (first statement).
    pub alpha: Vec<Fp256>,
    /// The mask challenge γ.
    pub gamma: Fp256,
    /// Black-box evaluation of `F̃` at a concrete L-point → n values.
    pub f_tilde: &'a dyn Fn(&[Fp256]) -> Vec<Fp256>,
    /// Old masks `G_j` with their claim points `β_j` and stored kernel
    /// claims `v_g,j = MLE(G_j|_cube)(β_j)` (the update statements' claimed
    /// cube sums).
    pub old_masks: Vec<(MaskPoly, Vec<Fp256>, Vec<Fp256>)>,
}

/// Transcript order (both sides): [α, γ, old claims] → mask sampling →
/// [e'_g] → rounds. The mask's cube sums are absorbed *after* sampling so
/// the verifier replays the identical challenge stream (draw-and-discard
/// for the mask itself).
fn absorb_statement_pre_mask(
    transcript: &mut Transcript,
    alpha: &[Fp256],
    gamma: &Fp256,
    old_claims: &[(Vec<Fp256>, Vec<Fp256>)],
) -> Result<(), ZkScError> {
    for v in alpha {
        transcript
            .append_message(b"zksc-alpha", &v.from_mont().canon_bytes())
            .map_err(ZkScError::Transcript)?;
    }
    transcript
        .append_message(b"zksc-gamma", &gamma.from_mont().canon_bytes())
        .map_err(ZkScError::Transcript)?;
    for (bj, vg) in old_claims {
        crate::util::absorb_fp_slice(transcript, b"zksc-beta-j", bj)?;
        crate::util::absorb_fp_slice(transcript, b"zksc-vg-j", vg)?;
    }
    Ok(())
}

fn absorb_mask_cube_sums(
    transcript: &mut Transcript,
    mask_cube_sums: &[Fp256],
) -> Result<(), ZkScError> {
    crate::util::absorb_fp_slice(transcript, b"zksc-eg", mask_cube_sums)?;
    Ok(())
}

/// Prove the batched masked sum-check (Fiat–Shamir).
pub fn prove_masked_batched(
    st: &MaskedBatchStatement<'_>,
    transcript: &mut Transcript,
) -> Result<(ZkSumcheckProof, ZkSumcheckOutput), ZkScError> {
    let l = st.num_vars;
    let d = st.map_degree;
    let n = st.n;
    let m = st.old_masks.len();
    let node_count = d + 2;

    // Old masks' multilinearizations (cube values, per coordinate columns).
    let old_mles: Vec<Vec<Vec<Fp256>>> = st
        .old_masks
        .iter()
        .map(|(mask, _, _)| {
            // column c over the cube
            let cv = mask.cube_values();
            (0..n)
                .map(|c| cv.iter().map(|row| row[c]).collect())
                .collect()
        })
        .collect();

    // Claimed cube sums (filled after the mask is sampled below).
    let mut claims: Vec<Fp256> = Vec::with_capacity(n + m * n);

    let old_claims: Vec<(Vec<Fp256>, Vec<Fp256>)> = st
        .old_masks
        .iter()
        .map(|(_, bj, vg)| (bj.clone(), vg.clone()))
        .collect();
    // Transcript order: pre-mask header → mask sampling → cube sums.
    absorb_statement_pre_mask(transcript, &st.alpha, &st.gamma, &old_claims)?;
    let fresh_mask = MaskPoly::sample(n, l, d + 1, transcript)?;
    let eg = fresh_mask.cube_sum();
    for c in 0..n {
        claims.push(st.gamma.mul(&eg[c]));
    }
    for (_, _, vg) in &st.old_masks {
        for c in 0..n {
            claims.push(vg[c]);
        }
    }
    absorb_mask_cube_sums(transcript, &eg)?;

    let mut rounds: Vec<Vec<Vec<Fp256>>> = Vec::with_capacity(l);
    let mut beta: Vec<Fp256> = Vec::with_capacity(l);
    let mut prev_evals = claims;

    for k in 0..l {
        let tail_len = l - k - 1;
        let total_coords = n + m * n;

        // Combined statement value at a concrete full point.
        let point_value = |pt: &[Fp256]| -> Vec<Fp256> {
            let eq_alpha = eq_eval(pt, &st.alpha);
            let ft = (st.f_tilde)(pt);
            let gm = fresh_mask.eval_at(pt).unwrap_or_default();
            let mut out = Vec::with_capacity(total_coords);
            for c in 0..n {
                out.push(eq_alpha.mul(&ft[c]).add(&st.gamma.mul(&gm[c])));
            }
            for (j, (_, bj, _)) in st.old_masks.iter().enumerate() {
                let eq_bj = eq_eval(bj, pt);
                for c in 0..n {
                    // Ĝ_j,c evaluated at pt (multilinear):
                    out.push(eq_bj.mul(&mle_eval(&old_mles[j][c], pt)));
                }
            }
            out
        };

        // Node evaluations: for each node y ∈ {0..d+1} at position k, sum
        // over the boolean tail.
        let mut node_sums: Vec<Vec<Fp256>> = vec![vec![Fp256::ZERO; total_coords]; node_count];
        let tail_count = 1usize << tail_len;
        let mut tail_point = vec![Fp256::ZERO; l];
        for t in 0..tail_count {
            for i in 0..tail_len {
                tail_point[k + 1 + i] = Fp256::from_canonical_u64(((t >> i) & 1) as u64);
            }
            for y in 0..node_count {
                let mut pt = tail_point.clone();
                pt[..k].copy_from_slice(&beta[..k]);
                pt[k] = Fp256::from_canonical_u64(y as u64);
                let vals = point_value(&pt);
                for (c, v) in vals.iter().enumerate() {
                    node_sums[y][c] = node_sums[y][c].add(v);
                }
            }
        }

        let mut coords: Vec<Vec<Fp256>> = Vec::with_capacity(total_coords);
        for c in 0..total_coords {
            let ys: Vec<Fp256> = node_sums.iter().map(|ns| ns[c]).collect();
            coords.push(interpolate_int_nodes(&ys));
        }
        // Prover-side sanity: the round identity must hold exactly.
        let zero = Fp256::ZERO;
        let one = Fp256::from_canonical_u64(1);
        for (c, poly) in coords.iter().enumerate() {
            let s = poly_eval(poly, &zero).add(&poly_eval(poly, &one));
            if s != prev_evals[c] {
                return Err(ZkScError::RoundConsistency { round: k, coord: c });
            }
        }
        for poly in &coords {
            for coeff in poly {
                transcript
                    .append_message(b"zksc-msg", &coeff.from_mont().canon_bytes())
                    .map_err(ZkScError::Transcript)?;
            }
        }
        let ch = crate::util::challenge_fp(transcript, b"zksc-challenge")?;
        beta.push(ch);
        for (c, poly) in coords.iter().enumerate() {
            prev_evals[c] = poly_eval(poly, &ch);
        }
        rounds.push(coords);
    }

    let first_final: Vec<Fp256> = (0..n).map(|c| prev_evals[c]).collect();
    // The update statements end at eq(β_j, β)·κ_j — divide out the eq
    // factor to output the kernel claims at the new point (mirroring the
    // verifier's derivation).
    let mut old_kernel_evals: Vec<Vec<Fp256>> = Vec::with_capacity(m);
    for j in 0..m {
        let eqv = eq_eval(&st.old_masks[j].1, &beta);
        let inv = eqv
            .inverse()
            .ok_or(ZkScError::Shape("degenerate eq(β_j, β)"))?;
        old_kernel_evals.push(
            (0..n)
                .map(|c| prev_evals[n + j * n + c].mul(&inv))
                .collect(),
        );
    }
    let fresh_mask_eval = fresh_mask.eval_at(&beta)?;
    let fresh_mask_kernel = fresh_mask.kernel_eval(&beta)?;

    Ok((
        ZkSumcheckProof {
            mask_cube_sums: eg,
            rounds,
            fresh_mask_eval,
            fresh_mask_kernel,
        },
        ZkSumcheckOutput {
            beta,
            first_final,
            old_kernel_evals,
            fresh_mask,
        },
    ))
}

/// Verify the batched masked sum-check; returns the final point and the
/// final claims (first-statement evals and old-mask kernel values at β).
pub fn verify_masked_batched(
    num_vars: usize,
    map_degree: usize,
    n: usize,
    alpha: &[Fp256],
    gamma: &Fp256,
    old_claims: &[(Vec<Fp256>, Vec<Fp256>)],
    proof: &ZkSumcheckProof,
    transcript: &mut Transcript,
) -> Result<ZkSumcheckOutput, ZkScError> {
    let l = num_vars;
    let m = old_claims.len();
    let total_coords = n + m * n;
    let node_count = map_degree + 2;

    let mut claims: Vec<Fp256> = Vec::with_capacity(total_coords);
    if proof.mask_cube_sums.len() != n {
        return Err(ZkScError::Shape("mask cube sums length"));
    }
    for c in 0..n {
        claims.push(gamma.mul(&proof.mask_cube_sums[c]));
    }
    for (_, vg) in old_claims {
        if vg.len() != n {
            return Err(ZkScError::Shape("old claim value length"));
        }
        for c in 0..n {
            claims.push(vg[c]);
        }
    }

    absorb_statement_pre_mask(transcript, alpha, gamma, old_claims)?;
    // Replay the prover's mask sampling: draw and discard the same number
    // of challenge words (n·(1 + L·(d+1))) so the round challenges agree.
    let mask_words = n * (1 + l * (map_degree + 1));
    let _ = challenge_fp_vec(transcript, b"mask-poly", mask_words)?;
    absorb_mask_cube_sums(transcript, &proof.mask_cube_sums)?;

    if proof.rounds.len() != l {
        return Err(ZkScError::Shape("round count"));
    }
    let mut prev = claims;
    let mut beta = Vec::with_capacity(l);
    let zero = Fp256::ZERO;
    let one = Fp256::from_canonical_u64(1);
    for (k, coords) in proof.rounds.iter().enumerate() {
        if coords.len() != total_coords {
            return Err(ZkScError::Shape("coordinate count"));
        }
        for poly in coords {
            if poly.len() != node_count {
                return Err(ZkScError::DegreeBound {
                    round: k,
                    got: poly.len(),
                    expected: node_count,
                });
            }
        }
        for (c, poly) in coords.iter().enumerate() {
            let s = poly_eval(poly, &zero).add(&poly_eval(poly, &one));
            if s != prev[c] {
                return Err(ZkScError::RoundConsistency { round: k, coord: c });
            }
        }
        for poly in coords {
            for coeff in poly {
                transcript
                    .append_message(b"zksc-msg", &coeff.from_mont().canon_bytes())
                    .map_err(ZkScError::Transcript)?;
            }
        }
        let ch = crate::util::challenge_fp(transcript, b"zksc-challenge")?;
        beta.push(ch);
        prev = coords.iter().map(|poly| poly_eval(poly, &ch)).collect();
    }

    let first_final: Vec<Fp256> = (0..n).map(|c| prev[c]).collect();
    // The update statements end at eq(β_j, β)·MLE(G_j)(β) — the kernel
    // values at the new point (no division needed: the last-round eval IS
    // eq(β_j, β)·κ_j; the caller uses κ_j directly in the combination
    // checks, so we divide out eq(β_j, β) here for a clean claim).
    let mut old_kernel_evals = Vec::with_capacity(m);
    for (j, (bj, _vg)) in old_claims.iter().enumerate() {
        let eqv = eq_eval(bj, &beta);
        let inv = eqv
            .inverse()
            .ok_or(ZkScError::Shape("degenerate eq(β_j, β)"))?;
        let mut vals = Vec::with_capacity(n);
        for c in 0..n {
            vals.push(prev[n + j * n + c].mul(&inv));
        }
        old_kernel_evals.push(vals);
    }

    Ok(ZkSumcheckOutput {
        beta,
        first_final,
        old_kernel_evals,
        fresh_mask: MaskPoly::zero(n, l, map_degree + 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn mask_cube_sum_matches_direct() {
        let mut t = Transcript::new_default(b"mask-test");
        let mask = MaskPoly::sample(3, 3, 2, &mut t).ok().unwrap();
        let mut direct = vec![Fp256::ZERO; 3];
        for b in 0..8usize {
            let pt: Vec<Fp256> = (0..3).map(|k| fr(((b >> k) & 1) as u64)).collect();
            let v = mask.eval_at(&pt).ok().unwrap();
            for c in 0..3 {
                direct[c] = direct[c].add(&v[c]);
            }
        }
        assert_eq!(mask.cube_sum(), direct);
    }

    #[test]
    fn mask_kernel_eval_is_mle_of_cube() {
        let mut t = Transcript::new_default(b"mask-test3");
        let mask = MaskPoly::sample(2, 3, 4, &mut t).ok().unwrap();
        let beta = vec![fr(3), fr(5), fr(7)];
        let kern = mask.kernel_eval(&beta).ok().unwrap();
        // kernel ≠ true eval for degree ≥ 2 masks (the load-bearing detail).
        let tru = mask.eval_at(&beta).ok().unwrap();
        assert_ne!(kern, tru);
        // kernel equals the MLE of the cube values.
        let cv = mask.cube_values();
        for c in 0..2 {
            let col: Vec<Fp256> = cv.iter().map(|row| row[c]).collect();
            assert_eq!(kern[c], mle_eval(&col, &beta));
        }
    }

    #[test]
    fn mask_linear_combination_eval() {
        let mut t = Transcript::new_default(b"mask-test2");
        let g1 = MaskPoly::sample(2, 2, 3, &mut t).ok().unwrap();
        let g2 = MaskPoly::sample(2, 2, 3, &mut t).ok().unwrap();
        let lam = fr(0x1234_5678_9abc_def0u64);
        let mut comb = g1.clone();
        comb.add_scaled(&lam, &g2).ok().unwrap();
        let pt = vec![fr(3), fr(5)];
        let v1 = g1.eval_at(&pt).ok().unwrap();
        let v2 = g2.eval_at(&pt).ok().unwrap();
        let vc = comb.eval_at(&pt).ok().unwrap();
        for c in 0..2 {
            assert_eq!(vc[c], v1[c].add(&lam.mul(&v2[c])));
        }
        // kernel values also combine linearly (cube values combine).
        let k1 = g1.kernel_eval(&pt).ok().unwrap();
        let k2 = g2.kernel_eval(&pt).ok().unwrap();
        let kc = comb.kernel_eval(&pt).ok().unwrap();
        for c in 0..2 {
            assert_eq!(kc[c], k1[c].add(&lam.mul(&k2[c])));
        }
    }

    #[test]
    fn interpolate_matches_evaluations() {
        let coeffs = vec![fr(3), fr(1), fr(0), fr(2), fr(7)];
        let ys: Vec<Fp256> = (0..5usize)
            .map(|i| poly_eval(&coeffs, &fr(i as u64)))
            .collect();
        let back = interpolate_int_nodes(&ys);
        assert_eq!(back, coeffs);
    }

    #[test]
    fn masked_sumcheck_roundtrip_and_tamper() {
        // F̃ vanishing on the cube: F̃_c(X) = Σ_k (X_k² − X_k)·w_k[c].
        let l = 3;
        let d = 2; // map degree; statement degree d+1 = 3; masks degree 3.
        let n = 2;
        let w = [vec![fr(5), fr(7)], vec![fr(11), fr(13)], vec![fr(17), fr(19)]];
        let f_tilde = |pt: &[Fp256]| -> Vec<Fp256> {
            let mut out = vec![Fp256::ZERO; n];
            for (k, wk) in w.iter().enumerate() {
                let x = pt[k];
                let g = x.mul(&x).sub(&x);
                for c in 0..n {
                    out[c] = out[c].add(&g.mul(&wk[c]));
                }
            }
            out
        };
        let mut t = Transcript::new_default(b"zksc");
        let alpha = crate::util::challenge_fp_vec(&mut t, b"alpha", l).ok().unwrap();
        let gamma = crate::util::challenge_fp(&mut t, b"gamma").ok().unwrap();
        // Old masks are prior-accumulator inputs: sample them from a
        // separate transcript (their claims enter via the header).
        let mut told = Transcript::new_default(b"zksc-old");
        let old_masks: Vec<(MaskPoly, Vec<Fp256>, Vec<Fp256>)> = {
            let g1 = MaskPoly::sample(n, l, d + 1, &mut told).ok().unwrap();
            let bj = vec![fr(2), fr(3), fr(4)];
            let vg = g1.kernel_eval(&bj).ok().unwrap();
            vec![(g1, bj, vg)]
        };
        let st = MaskedBatchStatement {
            num_vars: l,
            map_degree: d,
            n,
            alpha: alpha.clone(),
            gamma,
            f_tilde: &f_tilde,
            old_masks: old_masks.clone(),
        };
        let mut tprov = Transcript::new_default(b"zksc");
        let proven = prove_masked_batched(&st, &mut tprov);
        let (proof, out) = match proven {
            Ok(v) => v,
            Err(e) => {
                println!("PROVER ERROR: {:?}", e);
                panic!("prover failed: {e}");
            }
        };

        let old_claims: Vec<(Vec<Fp256>, Vec<Fp256>)> = old_masks
            .iter()
            .map(|(_, bj, vg)| (bj.clone(), vg.clone()))
            .collect();
        let mut tver = Transcript::new_default(b"zksc");
        let vout = match verify_masked_batched(l, d, n, &alpha, &gamma, &old_claims, &proof, &mut tver) {
            Ok(v) => v,
            Err(e) => { println!("VERIFIER ERROR: {e}"); panic!("verifier failed: {e}"); }
        };
        assert_eq!(vout.beta, out.beta);
        assert_eq!(vout.first_final, out.first_final);
        assert_eq!(vout.old_kernel_evals, out.old_kernel_evals);

        // Consistency of the derived claims:
        let beta = &out.beta;
        let eqb = eq_eval(beta, &alpha);
        let ft = f_tilde(beta);
        let gm = out.fresh_mask.eval_at(beta).ok().unwrap();
        for c in 0..n {
            let expect = eqb.mul(&ft[c]).add(&gamma.mul(&gm[c]));
            assert_eq!(out.first_final[c], expect);
        }
        assert_eq!(proof.fresh_mask_eval, gm);
        assert_eq!(
            proof.fresh_mask_kernel,
            out.fresh_mask.kernel_eval(beta).ok().unwrap()
        );
        // Old-mask kernel evals at the new point:
        let (g1, _, _) = &old_masks[0];
        assert_eq!(out.old_kernel_evals[0], g1.kernel_eval(beta).ok().unwrap());

        // Tampering: flip one round coefficient → verifier rejects.
        let mut bad = proof.clone();
        bad.rounds[0][0][1] = bad.rounds[0][0][1].add(&fr(1));
        let mut tver2 = Transcript::new_default(b"zksc");
        assert!(
            verify_masked_batched(l, d, n, &alpha, &gamma, &old_claims, &bad, &mut tver2).is_err()
        );
        // Tampering the mask cube sums → reject.
        let mut bad2 = proof.clone();
        bad2.mask_cube_sums[0] = bad2.mask_cube_sums[0].add(&fr(1));
        let mut tver3 = Transcript::new_default(b"zksc");
        assert!(
            verify_masked_batched(l, d, n, &alpha, &gamma, &old_claims, &bad2, &mut tver3).is_err()
        );
    }
}
