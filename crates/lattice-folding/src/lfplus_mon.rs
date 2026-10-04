//! LatticeFold+ (Boneh–Chen, ePrint 2025/247) §4 toolbox — Wave 7.7: the
//! monomial-set / ψ machinery that replaces LatticeFold's superseded
//! bit-decomposition range proof.
//!
//! * **Monomial set** `M = {0, 1, X, …, X^{d−1}} ⊆ R_q` (paper Eq (2)) with
//!   the evaluation map `ev_a(β) = Σ_i a_i β^i` over `F_{q^u}`. **Corollary
//!   4.1**: `a ∈ M ⟹ ev_a(β)² = ev_a(β²)`, and for `a ∉ M` the identity
//!   fails except with probability `2d/|F_{q^u}|`. LZX realization: the
//!   challenge field is `Fq2` over Goldilocks with the R_q coefficients
//!   canonically embedded — sound because a nonzero coefficient discrepancy
//!   mod q embeds to a nonzero Goldilocks value (0 < c < q < p) and
//!   Schwartz–Zippel applies to the degree-`< 2d` difference polynomial.
//! * **Π^mon (Construction 4.2)** — the degree-3 sumcheck over the challenge
//!   field: per column `j`, `Σ_i eq(c, ⟨i⟩)·(m_j(⟨i⟩)² − m'_j(⟨i⟩)) = 0`
//!   with `m_j = MLE[ev_{M_{*,j}}(β)]`, `m'_j = MLE[ev_{M_{*,j}}(β²)]`; the
//!   columns are batched with α-powers; the prover sends `e_j = M_{*,j}(r)`
//!   and the verifier checks Eq (12):
//!   `eq(c, r) · Σ_j α^j · (ev_{e_j}(β)² − ev_{e_j}(β²)) = v`.
//!   **Remark 4.3 (O(n)-add trick)**: because every entry is a monomial,
//!   `e_j = Σ_i tensor(r)_i · M_{i,j}` accumulates into `d` coefficient
//!   buckets — O(n) field additions, O(d) multiplications for the final
//!   evaluation — realized verbatim in [`monomial_column_eval_o_n`].
//! * **ψ layer (Lemma 2.2, Construction 4.3)** — the range core:
//!   `ψ := Σ_{i∈[1,d')} i·(X^{−i} + X^i) ∈ R_q` satisfies
//!   `ct(ψ·b) = a ⟺ a ∈ (−d', d') ∧ b ∈ EXP(a)` for monomial `b`. The
//!   Π^range warm-up protocol: run Π^mon on `m_τ = exp(τ)`, send
//!   `a = ⟨τ, tensor(r)⟩`, verify `ct(ψ·b) = a` where `b` is Π^mon's
//!   evaluation output.
//! * **split/pow double commitments (Construction 4.1, Lemma 4.1)** —
//!   `dcom(M) := com(split(com(M)))`: commit the monomial matrix, gadget-
//!   decompose the commitment into `τ ∈ (−d', d')^n` (base-`d'` digits),
//!   commit `τ`. `pow` reconstructs `com(M)` from `τ` by power-sums;
//!   binding of `dcom` reduces to binding of `com` (Lemma 4.1) — tested on
//!   the actual Ajtai binding surface.
//!
//! Asymptotic honesty: kernel scale (ring dim d = 16, n ≤ 512 rows,
//! ℓ = ⌈log_{d'}(q)⌉ = 11 digits per coefficient). The paper's
//! `F_{q^u}`-native sumcheck arithmetic is realized over `Fq2` (Goldilocks
//! base) with the canonical embedding — exact for the small monomial
//! coefficients involved (|c| ≤ 1) — and the module keeps the paper's
//! communication shape: degree-3 rounds (4 field values each) plus one
//! `e_j` per column.

use crate::fq2_sumcheck::{self, Fq2SumcheckError, Fq2VirtualPoly};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::extension::{challenge_fq2, challenge_fq2_vec, Fq2};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::RingElement;

/// Balanced integer → Goldilocks (canonical representative).
fn fe_i64(x: i64) -> Goldilocks {
    let m = lattice_core::field::GOLDILOCKS_MODULUS as i128;
    Goldilocks::from_u64((x as i128).rem_euclid(m) as u64)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LfPlusMonError {
    Sumcheck(Fq2SumcheckError),
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    TranscriptFailure,
    /// A monomial code is out of range for the ring dimension.
    BadMonomialCode {
        code: u32,
        ring_dim: u32,
    },
    /// Shape mismatch (rows/columns/digit counts).
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    /// A τ entry lies outside `(−d', d')` (Lemma 2.2's range).
    TauOutOfRange {
        value: i64,
        bound: u64,
    },
    /// The Π^mon O(n)-add consistency guard failed (internal bug guard —
    /// fail closed rather than transcribe an inconsistent proof).
    OnAddConsistency,
    /// The ψ range identity `ct(ψ·b) = a` failed.
    PsiRangeFailed,
    /// The `pow(τ) ≠ com(M)` double-commitment identity failed.
    PowMismatch,
}

impl From<Fq2SumcheckError> for LfPlusMonError {
    fn from(e: Fq2SumcheckError) -> Self {
        LfPlusMonError::Sumcheck(e)
    }
}

impl From<AjtaiError> for LfPlusMonError {
    fn from(e: AjtaiError) -> Self {
        LfPlusMonError::Ajtai(e)
    }
}

impl From<lattice_ring::RingError> for LfPlusMonError {
    fn from(e: lattice_ring::RingError) -> Self {
        LfPlusMonError::Ring(e)
    }
}

// ---------------------------------------------------------------------------
// Monomial set M ⊆ R_q (paper Eq (2)) and the evaluation map (Cor 4.1)
// ---------------------------------------------------------------------------

/// A monomial matrix in the code view: `entries[i*m_cols + j]` is `0` for
/// the zero monomial and `k+1` for `X^k` (`k < d = ring dimension`).
/// All entries of a [`MonomialMatrix`] lie in the paper's monomial set `M`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonomialMatrix {
    pub n_rows: usize,
    pub m_cols: usize,
    /// Codes in `[0, d]`: 0 = zero monomial, `k+1` = `X^k`.
    pub entries: Vec<u32>,
}

impl MonomialMatrix {
    /// Validate the codes against the ring dimension.
    pub fn validate(&self, ring_dim: u32) -> Result<(), LfPlusMonError> {
        if self.entries.len() != self.n_rows * self.m_cols {
            return Err(LfPlusMonError::ShapeMismatch {
                expected: self.n_rows * self.m_cols,
                got: self.entries.len(),
            });
        }
        for &c in &self.entries {
            if c > ring_dim {
                return Err(LfPlusMonError::BadMonomialCode { code: c, ring_dim });
            }
        }
        Ok(())
    }

    /// Column `j` as ring elements (the Ajtai-commitment view).
    pub fn column_ring_elements(
        &self,
        ring: &lattice_ring::RingConfig,
    ) -> Result<Vec<Vec<RingElement>>, LfPlusMonError> {
        self.validate(ring.n() as u32)?;
        let mut cols = Vec::with_capacity(self.m_cols);
        for j in 0..self.m_cols {
            let mut col = Vec::with_capacity(self.n_rows);
            for i in 0..self.n_rows {
                col.push(monomial_ring(ring, self.entries[i * self.m_cols + j]));
            }
            cols.push(col);
        }
        Ok(cols)
    }

    /// Deterministic pseudorandom monomial matrix from a seed (entries
    /// uniform over `M` — used by tests and the double-commitment demos).
    pub fn from_seed(
        n_rows: usize,
        m_cols: usize,
        ring_dim: u32,
        seed: &[u8],
    ) -> Result<Self, LfPlusMonError> {
        let bytes = Transcript::xof(b"lfplus-mon-matrix", seed, n_rows * m_cols);
        let mut entries = Vec::with_capacity(n_rows * m_cols);
        for &b in bytes.iter().take(n_rows * m_cols) {
            // Code in [0, ring_dim]: 0 = zero, k+1 = X^k.
            entries.push(u32::from(b) % (ring_dim + 1));
        }
        Ok(MonomialMatrix {
            n_rows,
            m_cols,
            entries,
        })
    }
}

/// The ring element of monomial code `c` (0 = zero, `k+1` = `X^k`).
pub fn monomial_ring(ring: &lattice_ring::RingConfig, code: u32) -> RingElement {
    let mut coeffs = vec![0u32; ring.n()];
    if code > 0 {
        let k = (code - 1) as usize;
        if k < ring.n() {
            coeffs[k] = 1;
        }
    }
    RingElement::from_coeffs(ring, coeffs)
}

/// `ev_a(β)` for the monomial with code `c` (Cor 4.1's evaluation map):
/// zero for `c = 0`, `β^{c-1}` otherwise.
pub fn monomial_ev(code: u32, beta_powers: &[Fq2]) -> Fq2 {
    if code == 0 || beta_powers.is_empty() {
        return Fq2::ZERO;
    }
    beta_powers
        .get((code - 1) as usize)
        .copied()
        .unwrap_or(Fq2::ZERO)
}

/// Powers `β^0 ..= β^{d-1}` (the evaluation-map table).
pub fn beta_powers(beta: Fq2, d: usize) -> Vec<Fq2> {
    let mut out = Vec::with_capacity(d);
    let mut acc = Fq2::ONE;
    for _ in 0..d {
        out.push(acc);
        acc = acc.mul(&beta);
    }
    out
}

/// `ev_a(β)` for an arbitrary R_q element with (small) balanced integer
/// coefficients — the Cor-4.1 identity oracle for tests and the ψ layer.
pub fn ev_of_coeffs(coeffs: &[i64], beta: Fq2) -> Fq2 {
    let mut acc = Fq2::ZERO;
    let mut pow = Fq2::ONE;
    for &c in coeffs {
        acc = acc.add(&Fq2::from_base(fe_i64(c)).mul(&pow));
        pow = pow.mul(&beta);
    }
    acc
}

/// `tensor(r)_i` — the eq-table entry of the MLE hypercube (row `i`, MSB
/// first), the multilinear weight used by the O(n)-add trick.
pub fn tensor_at(r: &[Fq2], i: usize) -> Fq2 {
    let num_vars = r.len();
    let mut acc = Fq2::ONE;
    for (var, rv) in r.iter().enumerate() {
        let bit = (i >> (num_vars - 1 - var)) & 1;
        let term = if bit == 1 { *rv } else { Fq2::ONE.sub(rv) };
        acc = acc.mul(&term);
    }
    acc
}

/// **Remark 4.3's O(n)-add evaluation trick**: `e_j = M_{*,j}(r) =
/// Σ_i tensor(r)_i · M_{i,j}` accumulated into `d` coefficient buckets —
/// O(n) field additions. Returns the `d` Fq2 coefficients of `e_j` (the
/// `R_q ⊗ C` view of the paper's `e_j ∈ R_q`).
pub fn monomial_column_eval_o_n(
    matrix: &MonomialMatrix,
    col: usize,
    ring_dim: usize,
    r: &[Fq2],
) -> Result<Vec<Fq2>, LfPlusMonError> {
    if r.len() != matrix.n_rows.trailing_zeros() as usize {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: matrix.n_rows.trailing_zeros() as usize,
            got: r.len(),
        });
    }
    let mut coeffs = vec![Fq2::ZERO; ring_dim];
    for i in 0..matrix.n_rows {
        let code = matrix.entries[i * matrix.m_cols + col];
        if code == 0 {
            continue; // zero-skip: the pay-per-nonzero discipline of Remark 4.3
        }
        let bucket = (code - 1) as usize;
        if bucket >= ring_dim {
            return Err(LfPlusMonError::BadMonomialCode {
                code,
                ring_dim: ring_dim as u32,
            });
        }
        coeffs[bucket] = coeffs[bucket].add(&tensor_at(r, i));
    }
    Ok(coeffs)
}

/// `ev_{e}(β)` for an `e` given as Fq2 coefficients (O(d) multiplications).
pub fn ev_of_fq2_coeffs(coeffs: &[Fq2], beta: Fq2) -> Fq2 {
    let mut acc = Fq2::ZERO;
    let mut pow = Fq2::ONE;
    for &c in coeffs {
        acc = acc.add(&c.mul(&pow));
        pow = pow.mul(&beta);
    }
    acc
}

// ---------------------------------------------------------------------------
// Π^mon (Construction 4.2) — degree-3 sumcheck with α-power column batching
// ---------------------------------------------------------------------------

/// Π^mon public statement: the committed monomial matrix's shape.
#[derive(Clone, Debug)]
pub struct PiMonStatement {
    /// Rows per column (power of two).
    pub n_rows: usize,
    /// Columns.
    pub m_cols: usize,
    /// Ring dimension `d` (defines the monomial set M).
    pub ring_dim: usize,
}

/// A Π^mon proof: the degree-3 sumcheck plus the per-column evaluation
/// claims `e_j = M_{*,j}(r)` (Fq2 coefficient vectors of length `d`).
#[derive(Clone, Debug)]
pub struct PiMonProof {
    /// Round polynomials (4 values per round — degree 3).
    pub rounds: Vec<Vec<Fq2>>,
    /// The challenge point `r` (the prover's view). The verifier derives
    /// its own point from the transcript and MUST NOT trust this field —
    /// it exists so protocol layers wrapping Π^mon (Construction 4.3) can
    /// compute point-dependent claims (`a = ⟨τ, tensor(r)⟩`) without
    /// transcript replay: the surrounding protocol absorbs its statement
    /// BEFORE Π^mon's challenges, so a fresh-transcript replay would
    /// desynchronize and round-check-fail.
    pub r: Vec<Fq2>,
    /// `e_j` per column: `d` Fq2 coefficients each (the paper sends one
    /// `R_q` element per column; the `R_q ⊗ C` realization carries the
    /// challenge-field coefficients).
    pub e: Vec<Vec<Fq2>>,
}

/// The Π^mon output statement (paper Eq (10)): `(C_M, r, e)` with the
/// witness bound by `Mᵀ tensor(r) = e`.
#[derive(Clone, Debug)]
pub struct PiMonOutput {
    /// The sumcheck challenge point (per-row variables).
    pub r: Vec<Fq2>,
    /// The per-column evaluations.
    pub e: Vec<Vec<Fq2>>,
}

/// Prove Π^mon for a monomial matrix whose columns are committed under `pk`
/// (the commitments are absorbed into the transcript BEFORE the challenges —
/// the statement-binding discipline).
pub fn prove_mon(
    statement: &PiMonStatement,
    matrix: &MonomialMatrix,
    commitments: &[AjtaiCommitment],
    transcript: &mut Transcript,
) -> Result<PiMonProof, LfPlusMonError> {
    matrix.validate(statement.ring_dim as u32)?;
    if commitments.len() != statement.m_cols {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: statement.m_cols,
            got: commitments.len(),
        });
    }
    if !statement.n_rows.is_power_of_two() || statement.n_rows == 0 {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: 0,
            got: statement.n_rows,
        });
    }
    // Absorb the statement and the column commitments.
    absorb_mon_statement(transcript, statement, commitments)?;
    // Step 1: c ← C^{log n}, β ← C, α ← C (Construction 4.2 + Remark 2.6).
    let num_vars = statement.n_rows.trailing_zeros() as usize;
    let c = challenge_fq2_vec(transcript, b"lfplus-mon-c", num_vars)
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let beta = challenge_fq2(transcript, b"lfplus-mon-beta")
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let alpha = challenge_fq2(transcript, b"lfplus-mon-alpha")
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let beta2 = beta.square();
    let powers = beta_powers(beta, statement.ring_dim);
    let powers2 = beta_powers(beta2, statement.ring_dim);

    // eq(c, ·) table and the per-column m / m' tables.
    let eq_table = eq_table_fq2(&c);
    let mut vp = Fq2VirtualPoly::new(num_vars);
    let eq_id = vp.add_factor(eq_table)?;
    let mut alpha_pow = Fq2::ONE;
    for j in 0..statement.m_cols {
        let m_j: Vec<Fq2> = (0..statement.n_rows)
            .map(|i| monomial_ev(matrix.entries[i * statement.m_cols + j], &powers))
            .collect();
        let m2_j: Vec<Fq2> = (0..statement.n_rows)
            .map(|i| monomial_ev(matrix.entries[i * statement.m_cols + j], &powers2))
            .collect();
        let m_id = vp.add_factor(m_j)?;
        let m2_id = vp.add_factor(m2_j)?;
        // eq·m² and − eq·m' per column, weighted by α^j (Eq (11) batched).
        vp.add_term(alpha_pow, vec![eq_id, m_id, m_id])?;
        vp.add_term(alpha_pow.neg(), vec![eq_id, m2_id])?;
        alpha_pow = alpha_pow.mul(&alpha);
    }
    let out = fq2_sumcheck::prove(&vp, Fq2::ZERO, transcript)?;
    // e_j via the O(n)-add trick; the consistency guard pins
    // ev_{e_j}(β) == m_j(r) (the factor claim) — fail closed on mismatch.
    let mut e_cols = Vec::with_capacity(statement.m_cols);
    for j in 0..statement.m_cols {
        let e = monomial_column_eval_o_n(matrix, j, statement.ring_dim, &out.challenges)?;
        let ev = ev_of_fq2_coeffs(&e, beta);
        let expected = out.factor_claims[1 + 2 * j]; // m_j factor id: 1 + 2j
        if ev != expected {
            return Err(LfPlusMonError::OnAddConsistency);
        }
        e_cols.push(e);
    }
    Ok(PiMonProof {
        rounds: out.proof.rounds,
        r: out.challenges.clone(),
        e: e_cols,
    })
}

/// Verify Π^mon (Fig-1 step 4 / Eq (12)): replays the degree-3 sumcheck and
/// checks the terminal value from the prover's `e_j` claims. Returns the
/// output statement `(r, e)` for the decider / folding layers.
pub fn verify_mon(
    statement: &PiMonStatement,
    commitments: &[AjtaiCommitment],
    proof: &PiMonProof,
    transcript: &mut Transcript,
) -> Result<PiMonOutput, LfPlusMonError> {
    if commitments.len() != statement.m_cols || proof.e.len() != statement.m_cols {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: statement.m_cols,
            got: proof.e.len(),
        });
    }
    for e in &proof.e {
        if e.len() != statement.ring_dim {
            return Err(LfPlusMonError::ShapeMismatch {
                expected: statement.ring_dim,
                got: e.len(),
            });
        }
    }
    absorb_mon_statement(transcript, statement, commitments)?;
    let num_vars = statement.n_rows.trailing_zeros() as usize;
    let c = challenge_fq2_vec(transcript, b"lfplus-mon-c", num_vars)
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let beta = challenge_fq2(transcript, b"lfplus-mon-beta")
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let alpha = challenge_fq2(transcript, b"lfplus-mon-alpha")
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let beta2 = beta.square();
    let verdict = fq2_sumcheck::Fq2SumcheckProof {
        rounds: proof.rounds.clone(),
    }
    .verify(num_vars, 3, Fq2::ZERO, transcript, None)?;
    let r = verdict.point;
    // Eq (12): eq(c, r) · Σ_j α^j · (ev_{e_j}(β)² − ev_{e_j}(β²)) = v.
    let eq_cr = eq_point_fq2(&c, &r);
    let mut acc = Fq2::ZERO;
    let mut alpha_pow = Fq2::ONE;
    for e in &proof.e {
        let ev_b = ev_of_fq2_coeffs(e, beta);
        let ev_b2 = ev_of_fq2_coeffs(e, beta2);
        acc = acc.add(&alpha_pow.mul(&ev_b.square().sub(&ev_b2)));
        alpha_pow = alpha_pow.mul(&alpha);
    }
    let expected_final = eq_cr.mul(&acc);
    if verdict.final_claim != expected_final {
        return Err(LfPlusMonError::Sumcheck(
            fq2_sumcheck::Fq2SumcheckError::FinalCheckFailed,
        ));
    }
    Ok(PiMonOutput {
        r: r.clone(),
        e: proof.e.clone(),
    })
}

/// Decider-side Π^mon check (the `R_m,out` linear relation): the commitments
/// open the monomial matrix and `e_j = M_{*,j}ᵀ tensor(r)` for every column
/// (recomputed with the O(n)-add trick — O(nm) additions).
pub fn verify_mon_opening(
    pk: &AjtaiPublicKey,
    statement: &PiMonStatement,
    matrix: &MonomialMatrix,
    commitments: &[AjtaiCommitment],
    output: &PiMonOutput,
) -> Result<(), LfPlusMonError> {
    let ring = &pk.params.ring;
    if matrix.n_rows != statement.n_rows || matrix.m_cols != statement.m_cols {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: statement.n_rows,
            got: matrix.n_rows,
        });
    }
    let cols = matrix.column_ring_elements(ring)?;
    for (j, col) in cols.iter().enumerate() {
        let padded = pk.pad_to_m(col)?;
        let commitment = commitments.get(j).ok_or(LfPlusMonError::ShapeMismatch {
            expected: statement.m_cols,
            got: j,
        })?;
        pk.verify_opening(commitment, &padded)?;
        let e = monomial_column_eval_o_n(matrix, j, statement.ring_dim, &output.r)?;
        if e != output.e[j] {
            return Err(LfPlusMonError::OnAddConsistency);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ψ layer — Lemma 2.2 and the Construction-4.3 range check
// ---------------------------------------------------------------------------

/// `ψ := Σ_{i∈[1,d')} i·(X^{−i} + X^i) ∈ R_q` (Lemma 2.2): coefficient `i`
/// at `X^i` and `−i` at `X^{d−i}` (using `X^{−i} = −X^{d−i}` in
/// `R_q = Z_q[X]/(X^d+1)`).
pub fn psi_element(ring: &lattice_ring::RingConfig) -> RingElement {
    let d = ring.n();
    let dp = d / 2;
    let mut coeffs = vec![0u32; d];
    for i in 1..dp {
        coeffs[i] = ring.modulus.reduce_i64(i as i64);
        coeffs[d - i] = ring.modulus.reduce_i64(-(i as i64));
    }
    RingElement::from_coeffs(ring, coeffs)
}

/// `exp(a)` (paper Def above Lemma 2.2) as a monomial code: for
/// `a ∈ (−d', d')` — `a > 0 ⟹ X^a`, `a < 0 ⟹ X^{a+d}` (since
/// `−X^{a} = X^{a+d}` in R_q), `a = 0 ⟹ 0 ∈ EXP(0)`.
pub fn exp_code(a: i64, ring_dim: usize) -> Result<u32, LfPlusMonError> {
    let d = ring_dim as i64;
    let dp = d / 2;
    if a <= -dp || a >= dp {
        return Err(LfPlusMonError::TauOutOfRange {
            value: a,
            bound: dp as u64,
        });
    }
    let k = if a >= 0 { a } else { a + d };
    Ok((k + 1) as u32)
}

/// Lemma 2.2 constant-term oracle over R_q: `ct(ψ·b)` for monomial code `b`
/// (returned as a balanced integer — it always lies in `(−d', d')` for
/// monomial `b`, which is exactly the lemma's content).
pub fn psi_ct_of_code(ring: &lattice_ring::RingConfig, code: u32) -> i64 {
    let psi = psi_element(ring);
    let b = monomial_ring(ring, code);
    let prod = b.mul(&psi).unwrap_or_else(|_| ring.zero());
    let raw = prod.coeff(0);
    let half = ring.modulus.q / 2;
    if raw <= half {
        raw as i64
    } else {
        raw as i64 - ring.modulus.q as i64
    }
}

/// The ψ range proof (Construction 4.3): Π^mon over `m_τ = exp(τ)` plus the
/// prover's `a = ⟨τ, tensor(r)⟩` and the verifier's `ct(ψ·b) = a` check.
#[derive(Clone, Debug)]
pub struct PsiRangeProof {
    /// The Π^mon proof for the monomial column `m_τ`.
    pub mon: PiMonProof,
    /// `a = ⟨τ, tensor(r)⟩ ∈ C` (prover-sent, Step 2).
    pub a: Fq2,
}

/// Prove the Construction-4.3 range statement for `τ ∈ (−d', d')^n`.
/// `cm_τ` commits τ (as constant ring elements), `cm_mτ` commits the
/// monomial column `m_τ = exp(τ)` — both absorbed before the challenges.
pub fn prove_psi_range(
    ring: &lattice_ring::RingConfig,
    tau: &[i64],
    cm_tau: &AjtaiCommitment,
    cm_mtau: &AjtaiCommitment,
    transcript: &mut Transcript,
) -> Result<PsiRangeProof, LfPlusMonError> {
    let d = ring.n();
    let dp = d / 2;
    if !tau.len().is_power_of_two() || tau.is_empty() {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: 0,
            got: tau.len(),
        });
    }
    for &t in tau {
        if t <= -(dp as i64) || t >= dp as i64 {
            return Err(LfPlusMonError::TauOutOfRange {
                value: t,
                bound: dp as u64,
            });
        }
    }
    let codes: Vec<u32> = tau.iter().map(|&t| exp_code(t, d).unwrap_or(0)).collect();
    let matrix = MonomialMatrix {
        n_rows: tau.len(),
        m_cols: 1,
        entries: codes,
    };
    let statement = PiMonStatement {
        n_rows: tau.len(),
        m_cols: 1,
        ring_dim: d,
    };
    // Absorb the range statement (τ commitment + m_τ commitment + shape)
    // BEFORE the Π^mon challenges.
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&(tau.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(d as u32).to_le_bytes());
    buf.extend_from_slice(&(dp as u32).to_le_bytes());
    transcript
        .append_bytes(b"lfplus-psi-stmt", &buf)
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    transcript
        .append_bytes(b"lfplus-psi-cm-tau", &cm_tau.to_bytes())
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    transcript
        .append_bytes(b"lfplus-psi-cm-mtau", &cm_mtau.to_bytes())
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let mon = prove_mon(
        &statement,
        &matrix,
        std::slice::from_ref(cm_mtau),
        transcript,
    )?;
    // The prover's view of the challenge point r (identical to the
    // verifier's transcript-derived point — the rounds bind them).
    let r = mon.r.clone();
    // a = Σ_i τ_i · tensor(r)_i over the challenge field.
    let mut a = Fq2::ZERO;
    for (i, &t) in tau.iter().enumerate() {
        a = a.add(&Fq2::from_base(fe_i64(t)).mul(&tensor_at(&r, i)));
    }
    Ok(PsiRangeProof { mon, a })
}

/// Verify the Construction-4.3 range proof: Π^mon verification plus the
/// Lemma-2.2 check `ct(ψ·b) = a` where `b = e_0` (the Π^mon evaluation
/// output, Fq2 coefficients). Returns the Π^mon output `(r, e)` for the
/// decider / folding layers.
pub fn verify_psi_range(
    ring: &lattice_ring::RingConfig,
    tau_len: usize,
    cm_tau: &AjtaiCommitment,
    cm_mtau: &AjtaiCommitment,
    proof: &PsiRangeProof,
    transcript: &mut Transcript,
) -> Result<PiMonOutput, LfPlusMonError> {
    let d = ring.n();
    let dp = d / 2;
    if !tau_len.is_power_of_two() || tau_len == 0 {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: 0,
            got: tau_len,
        });
    }
    if proof.mon.e.len() != 1 {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: 1,
            got: proof.mon.e.len(),
        });
    }
    let statement = PiMonStatement {
        n_rows: tau_len,
        m_cols: 1,
        ring_dim: d,
    };
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&(tau_len as u32).to_le_bytes());
    buf.extend_from_slice(&(d as u32).to_le_bytes());
    buf.extend_from_slice(&(dp as u32).to_le_bytes());
    transcript
        .append_bytes(b"lfplus-psi-stmt", &buf)
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    transcript
        .append_bytes(b"lfplus-psi-cm-tau", &cm_tau.to_bytes())
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    transcript
        .append_bytes(b"lfplus-psi-cm-mtau", &cm_mtau.to_bytes())
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    let output = verify_mon(
        &statement,
        std::slice::from_ref(cm_mtau),
        &proof.mon,
        transcript,
    )?;
    // Step 3 (Lemma 2.2): ct(ψ·b) = a over the challenge field — the ψ
    // coefficients are small integers (exact under the embedding).
    let psi = psi_element(ring);
    let b = &proof.mon.e[0];
    let mut ct = Fq2::ZERO;
    for (k, &bk) in b.iter().enumerate() {
        let raw = psi.coeff(k);
        let half = ring.modulus.q / 2;
        let balanced = if raw <= half {
            raw as i64
        } else {
            raw as i64 - ring.modulus.q as i64
        };
        ct = ct.add(&Fq2::from_base(fe_i64(balanced)).mul(&bk));
    }
    if ct != proof.a {
        return Err(LfPlusMonError::PsiRangeFailed);
    }
    Ok(output)
}

/// Decider-side ψ range check (the `R'` output relation): `cm_τ` opens τ
/// (constants), `cm_mτ` opens `m_τ`, and `[τ, m_τ]ᵀ tensor(r) = (a, b)` at
/// the **verifier-derived** Π^mon output point (soundness: `r` comes from
/// the transcript-bound verification, never from the prover).
#[allow(clippy::too_many_arguments)] // decider surface: opening + point + claims
pub fn verify_psi_opening(
    pk: &AjtaiPublicKey,
    ring: &lattice_ring::RingConfig,
    tau: &[i64],
    cm_tau: &AjtaiCommitment,
    mtau_codes: &[u32],
    cm_mtau: &AjtaiCommitment,
    proof: &PsiRangeProof,
    output: &PiMonOutput,
) -> Result<(), LfPlusMonError> {
    let d = ring.n();
    // Commitments open the witnesses.
    let tau_elems: Vec<RingElement> = tau
        .iter()
        .map(|&t| ring.constant(ring.modulus.reduce_i64(t)))
        .collect();
    let padded_tau = pk.pad_to_m(&tau_elems)?;
    pk.verify_opening(cm_tau, &padded_tau)?;
    let mtau_elems: Vec<RingElement> = mtau_codes.iter().map(|&c| monomial_ring(ring, c)).collect();
    let padded_mtau = pk.pad_to_m(&mtau_elems)?;
    pk.verify_opening(cm_mtau, &padded_mtau)?;
    // Linear output relation at the VERIFIER-derived Π^mon point r.
    let r = &output.r;
    let mut a = Fq2::ZERO;
    for (i, &t) in tau.iter().enumerate() {
        a = a.add(&Fq2::from_base(fe_i64(t)).mul(&tensor_at(r, i)));
    }
    if a != proof.a {
        return Err(LfPlusMonError::PsiRangeFailed);
    }
    let e = monomial_column_eval_o_n(
        &MonomialMatrix {
            n_rows: tau.len(),
            m_cols: 1,
            entries: mtau_codes.to_vec(),
        },
        0,
        d,
        r,
    )?;
    if e != proof.mon.e[0] {
        return Err(LfPlusMonError::OnAddConsistency);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// split / pow double commitments (Construction 4.1, Lemma 4.1)
// ---------------------------------------------------------------------------

/// Gadget decomposition of one coefficient into `ℓ` balanced base-`d'`
/// digits (the `G^{-1}_{d',ℓ}` of Construction 4.1; digits in
/// `[-d'/2+1, d'/2] ⊂ (−d', d')` — base `d'`, the paper's digit radix).
fn split_coefficient(c: i64, dprime: u32, ell: usize) -> Vec<i64> {
    let base = dprime as i64;
    let mut out = Vec::with_capacity(ell);
    let mut rem = c;
    for _ in 0..ell {
        let half = base / 2;
        let mut digit = rem.rem_euclid(base);
        if digit > half {
            digit -= base;
        }
        rem = (rem - digit) / base;
        out.push(digit);
    }
    out
}

/// `split(com(M))` (Construction 4.1): flatten the per-column commitments'
/// coefficients and gadget-decompose each into `ℓ` digits. Returns the
/// digit vector (length `k · m_cols · d · ℓ`, entries in `(−d', d')`).
pub fn split_commitment(
    ring: &lattice_ring::RingConfig,
    com_m: &[AjtaiCommitment],
    dprime: u32,
    ell: usize,
) -> Result<Vec<i64>, LfPlusMonError> {
    let half = ring.modulus.q / 2;
    let mut tau = Vec::new();
    for com in com_m {
        for row in &com.rows {
            for &c in row.coeffs() {
                let balanced = if c <= half {
                    c as i64
                } else {
                    c as i64 - ring.modulus.q as i64
                };
                tau.extend(split_coefficient(balanced, dprime, ell));
            }
        }
    }
    Ok(tau)
}

/// `pow` (Construction 4.1): reassemble the committed ring elements from the
/// digit vector — `pow(split(D)) = D` for every `D` (the gadget power-sum).
/// Returns the `k × m_cols` commitment entries in the same **column-major**
/// traversal order as [`split_commitment`] (column outer, key row inner).
pub fn pow(
    ring: &lattice_ring::RingConfig,
    tau: &[i64],
    k: usize,
    m_cols: usize,
    dprime: u32,
    ell: usize,
) -> Result<Vec<RingElement>, LfPlusMonError> {
    let d = ring.n();
    let expected = k * m_cols * d * ell;
    if tau.len() < expected {
        return Err(LfPlusMonError::ShapeMismatch {
            expected,
            got: tau.len(),
        });
    }
    let base: i128 = dprime as i128;
    let mut out = Vec::with_capacity(k * m_cols);
    // Column-major entry order: entry = col * k + row.
    for entry in 0..k * m_cols {
        let mut coeffs = vec![0u32; d];
        #[allow(clippy::needless_range_loop)]
        for cpos in 0..d {
            let start = (entry * d + cpos) * ell;
            let mut acc: i128 = 0;
            let mut weight: i128 = 1;
            for i in 0..ell {
                acc += tau
                    .get(start + i)
                    .copied()
                    .ok_or(LfPlusMonError::ShapeMismatch {
                        expected: start + i + 1,
                        got: tau.len(),
                    })? as i128
                    * weight;
                weight *= base;
            }
            coeffs[cpos] = ring.modulus.reduce_i64(acc as i64);
        }
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    Ok(out)
}

/// The double commitment `dcom(M) = com(split(com(M)))` (Eq (7)): the digit
/// vector τ (padded with zeros to the key's slot count) committed under the
/// double-commitment key. Returns the padded τ as constant ring elements.
pub fn dcom_commit(
    pk_dcom: &AjtaiPublicKey,
    ring: &lattice_ring::RingConfig,
    com_m: &[AjtaiCommitment],
    dprime: u32,
    ell: usize,
) -> Result<(AjtaiCommitment, Vec<i64>, Vec<RingElement>), LfPlusMonError> {
    let tau = split_commitment(ring, com_m, dprime, ell)?;
    if tau.len() > pk_dcom.params.m {
        return Err(LfPlusMonError::ShapeMismatch {
            expected: pk_dcom.params.m,
            got: tau.len(),
        });
    }
    // Hard digit-norm gate: ∥τ∥∞ ≤ d'/2 (the balanced base-d' digit
    // invariant of G^{-1}_{d',ℓ}; digits live in (−d'/2, d'/2] ⊂ (−d', d'),
    // Lemma 2.2's opening precondition).
    let bound = (dprime / 2) as u64;
    for &t in &tau {
        if t.unsigned_abs() > bound {
            return Err(LfPlusMonError::TauOutOfRange { value: t, bound });
        }
    }
    let mut padded = tau.clone();
    while padded.len() < pk_dcom.params.m {
        padded.push(0);
    }
    let elems: Vec<RingElement> = padded
        .iter()
        .map(|&t| ring.constant(ring.modulus.reduce_i64(t)))
        .collect();
    let commitment = pk_dcom.commit(&elems)?;
    Ok((commitment, tau, elems))
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn absorb_mon_statement(
    transcript: &mut Transcript,
    statement: &PiMonStatement,
    commitments: &[AjtaiCommitment],
) -> Result<(), LfPlusMonError> {
    let mut buf = Vec::with_capacity(16 + commitments.len() * 8);
    buf.extend_from_slice(&(statement.n_rows as u32).to_le_bytes());
    buf.extend_from_slice(&(statement.m_cols as u32).to_le_bytes());
    buf.extend_from_slice(&(statement.ring_dim as u32).to_le_bytes());
    transcript
        .append_bytes(b"lfplus-mon-stmt", &buf)
        .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    for com in commitments {
        transcript
            .append_bytes(b"lfplus-mon-cm", &com.to_bytes())
            .map_err(|_| LfPlusMonError::TranscriptFailure)?;
    }
    Ok(())
}

/// eq(X; η) evaluations over the boolean hypercube (MSB-first variables).
fn eq_table_fq2(eta: &[Fq2]) -> Vec<Fq2> {
    let mut evals = vec![Fq2::ONE; 1usize << eta.len()];
    for (var, e) in eta.iter().enumerate() {
        let shift = eta.len() - 1 - var;
        for (idx, val) in evals.iter_mut().enumerate() {
            let bit = (idx >> shift) & 1;
            let term = if bit == 1 { *e } else { Fq2::ONE.sub(e) };
            *val = val.mul(&term);
        }
    }
    evals
}

/// eq(u; η) at an arbitrary point (the Eq-12 leaf factor).
fn eq_point_fq2(eta: &[Fq2], u: &[Fq2]) -> Fq2 {
    let mut acc = Fq2::ONE;
    for (e, x) in eta.iter().zip(u.iter()) {
        let term = x.mul(e).add(&Fq2::ONE.sub(x).mul(&Fq2::ONE.sub(e)));
        acc = acc.mul(&term);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_commitment::ajtai::AjtaiParams;
    use lattice_ring::{Modulus32, RingConfig};

    fn fe(x: i64) -> Fq2 {
        Fq2::from_base(fe_i64(x))
    }

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 20,
        };
        let pk = AjtaiPublicKey::from_seed(params, [91u8; 32]).ok().unwrap();
        (pk, ring)
    }

    // ---- Corollary 4.1: ev_a(β)² = ev_a(β²) iff a ∈ M ----

    #[test]
    fn cor_4_1_ev_squared_identity() {
        // Monomials satisfy the identity at a fixed random β.
        let beta = Fq2::new(
            Goldilocks::from_u64(0x1234_5678_9ABC_DEF0),
            Goldilocks::from_u64(0x9876_5432_1FED_CBA0),
        );
        for k in 0usize..16 {
            let mut coeffs = vec![0i64; 16];
            coeffs[k] = 1;
            let ev = ev_of_coeffs(&coeffs, beta);
            let ev2 = ev_of_coeffs(&coeffs, beta.square());
            assert_eq!(ev.square(), ev2, "monomial X^{k}");
        }
        // Zero is in M.
        assert_eq!(
            ev_of_coeffs(&[0i64; 16], beta).square(),
            ev_of_coeffs(&[0i64; 16], beta.square())
        );
        // Non-monomials fail (2·X^0, 1+X, X²+X, 2·X^5): the Cor-4.1
        // soundness direction at a random challenge point.
        for bad in [
            vec![2i64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![1i64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0i64, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![0i64, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ] {
            let ev = ev_of_coeffs(&bad, beta);
            let ev2 = ev_of_coeffs(&bad, beta.square());
            assert_ne!(ev.square(), ev2, "non-monomial must fail: {bad:?}");
        }
    }

    // ---- Lemma 2.2: ct(ψ·b) = a iff a ∈ (−d', d') and b ∈ EXP(a) ----

    #[test]
    fn lemma_2_2_psi_iff() {
        let (_, ring) = setup(4, 4);
        let d = ring.n() as i64;
        let dp = d / 2;
        // Forward: for every a ∈ (−d', d'), ct(ψ·exp(a)) = a.
        for a in -(dp - 1)..=(dp - 1) {
            let code = exp_code(a, ring.n()).ok().unwrap();
            assert_eq!(psi_ct_of_code(&ring, code), a, "a = {a}");
        }
        // EXP(0) also contains 1 and X^{d'} (ct = 0 for all three).
        assert_eq!(psi_ct_of_code(&ring, 1), 0); // b = 1
        assert_eq!(psi_ct_of_code(&ring, dp as u32 + 1), 0); // b = X^{d'}
                                                             // Converse: monomial b ∉ EXP(a) fails — e.g. ct(ψ·X^k) = k ≠ a for
                                                             // a < k < d', and the negated variants.
        assert_ne!(psi_ct_of_code(&ring, 5), 5); // X^4: ct(X^4·ψ)=4 ≠ a=5
        assert_ne!(psi_ct_of_code(&ring, 3), 4);
        assert_ne!(psi_ct_of_code(&ring, dp as u32 + 2), 0); // X^{d'+1}: ct = -d'+1 ≠ 0
                                                             // ψ itself has zero constant term.
        let psi = psi_element(&ring);
        assert_eq!(psi.coeff(0), 0);
    }

    // ---- Π^mon honest path + tamper ----

    #[test]
    fn pi_mon_happy_path_and_opening() {
        let (pk, ring) = setup(4, 64);
        let d = ring.n();
        let matrix = MonomialMatrix::from_seed(32, 2, d as u32, b"pi-mon-ok")
            .ok()
            .unwrap();
        let statement = PiMonStatement {
            n_rows: 32,
            m_cols: 2,
            ring_dim: d,
        };
        let cols = matrix.column_ring_elements(&ring).ok().unwrap();
        let commitments: Vec<AjtaiCommitment> = cols
            .iter()
            .map(|c| pk.commit(&pk.pad_to_m(c).ok().unwrap()).ok().unwrap())
            .collect();
        let mut t = Transcript::new_default(b"lfplus-pi-mon-test");
        let proof = prove_mon(&statement, &matrix, &commitments, &mut t)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"lfplus-pi-mon-test");
        let output = verify_mon(&statement, &commitments, &proof, &mut vt)
            .ok()
            .unwrap();
        // Degree-3 communication shape: 4 values per round.
        for round in &proof.rounds {
            assert_eq!(round.len(), 4);
        }
        // Decider: openings + the R_m,out linear relation.
        assert!(verify_mon_opening(&pk, &statement, &matrix, &commitments, &output).is_ok());
    }

    #[test]
    fn pi_mon_o_n_add_trick_matches_direct() {
        // Remark 4.3: the bucket accumulation equals the direct
        // ev-table MLE evaluation at the same point.
        let (_, ring) = setup(4, 4);
        let d = ring.n();
        let matrix = MonomialMatrix::from_seed(16, 1, d as u32, b"o-n-add")
            .ok()
            .unwrap();
        let beta = Fq2::new(
            Goldilocks::from_u64(0x1234_5678_9ABC_DEF0),
            Goldilocks::from_u64(0xFEDC_BA09_8765_4321),
        );
        let powers = beta_powers(beta, d);
        let r: Vec<Fq2> = (0..4)
            .map(|i| challenge_from_seed(b"o-n-r", i as u64))
            .collect();
        let e = monomial_column_eval_o_n(&matrix, 0, d, &r).ok().unwrap();
        // Direct: m(b) = ev_{M_{b,0}}(β), then MLE at r = Σ_b m(b)·tensor(r)_b.
        let mut direct = Fq2::ZERO;
        for b in 0..16 {
            let m_b = monomial_ev(matrix.entries[b], &powers);
            direct = direct.add(&m_b.mul(&tensor_at(&r, b)));
        }
        assert_eq!(ev_of_fq2_coeffs(&e, beta), direct);
    }

    #[test]
    fn pi_mon_tampered_rejected() {
        let (pk, ring) = setup(4, 64);
        let d = ring.n();
        let matrix = MonomialMatrix::from_seed(16, 1, d as u32, b"pi-mon-bad")
            .ok()
            .unwrap();
        let statement = PiMonStatement {
            n_rows: 16,
            m_cols: 1,
            ring_dim: d,
        };
        let cols = matrix.column_ring_elements(&ring).ok().unwrap();
        let commitments: Vec<AjtaiCommitment> = cols
            .iter()
            .map(|c| pk.commit(&pk.pad_to_m(c).ok().unwrap()).ok().unwrap())
            .collect();
        let mut t = Transcript::new_default(b"lfplus-pi-mon-tamper");
        let mut proof = prove_mon(&statement, &matrix, &commitments, &mut t)
            .ok()
            .unwrap();
        // Tampered round value: the round check fails.
        let mut bad_rounds = proof.clone();
        if let Some(r0) = bad_rounds.rounds.first_mut() {
            if let Some(v) = r0.first_mut() {
                *v = v.add(&fe(1));
            }
        }
        let mut vt = Transcript::new_default(b"lfplus-pi-mon-tamper");
        assert!(verify_mon(&statement, &commitments, &bad_rounds, &mut vt).is_err());
        // Tampered e_0: the Eq-12 terminal identity fails.
        proof.e[0][0] = proof.e[0][0].add(&fe(1));
        let mut vt2 = Transcript::new_default(b"lfplus-pi-mon-tamper");
        assert!(verify_mon(&statement, &commitments, &proof, &mut vt2).is_err());
        // Wrong commitment (statement binding): challenge desync.
        let other = MonomialMatrix::from_seed(16, 1, d as u32, b"pi-mon-other")
            .ok()
            .unwrap();
        let other_cols = other.column_ring_elements(&ring).ok().unwrap();
        let other_cms: Vec<AjtaiCommitment> = other_cols
            .iter()
            .map(|c| pk.commit(&pk.pad_to_m(c).ok().unwrap()).ok().unwrap())
            .collect();
        let mut t3 = Transcript::new_default(b"lfplus-pi-mon-tamper");
        let proof3 = prove_mon(&statement, &matrix, &commitments, &mut t3)
            .ok()
            .unwrap();
        let mut vt3 = Transcript::new_default(b"lfplus-pi-mon-tamper");
        assert!(verify_mon(&statement, &other_cms, &proof3, &mut vt3).is_err());
    }

    // ---- Construction 4.3: ψ range protocol ----

    #[test]
    fn psi_range_happy_path() {
        let (pk, ring) = setup(4, 16);
        let tau: Vec<i64> = [-7i64, 6, -1, 0, 3, -5, 2, 1].to_vec();
        let codes: Vec<u32> = tau
            .iter()
            .map(|&t| exp_code(t, ring.n()).ok().unwrap())
            .collect();
        // Commit τ (constants) and m_τ (monomials).
        let tau_elems: Vec<RingElement> = tau
            .iter()
            .map(|&t| ring.constant(ring.modulus.reduce_i64(t)))
            .collect();
        let cm_tau = pk
            .commit(&pk.pad_to_m(&tau_elems).ok().unwrap())
            .ok()
            .unwrap();
        let mtau_elems: Vec<RingElement> = codes.iter().map(|&c| monomial_ring(&ring, c)).collect();
        let cm_mtau = pk
            .commit(&pk.pad_to_m(&mtau_elems).ok().unwrap())
            .ok()
            .unwrap();
        let mut t = Transcript::new_default(b"lfplus-psi-test");
        let proof = prove_psi_range(&ring, &tau, &cm_tau, &cm_mtau, &mut t)
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"lfplus-psi-test");
        let output = verify_psi_range(&ring, tau.len(), &cm_tau, &cm_mtau, &proof, &mut vt)
            .ok()
            .unwrap();
        // Decider: openings + the linear output relation at the
        // verifier-derived point.
        assert!(
            verify_psi_opening(&pk, &ring, &tau, &cm_tau, &codes, &cm_mtau, &proof, &output)
                .is_ok()
        );
    }

    #[test]
    fn psi_range_out_of_range_and_tamper() {
        let (pk, ring) = setup(4, 16);
        // τ with an entry outside (−d', d'): the honest prover refuses.
        let bad_tau: Vec<i64> = [7i64, 8, 0, 0, 0, 0, 0, 0].to_vec(); // 8 ∉ (−8, 8)
        let tau: Vec<i64> = [1i64, -2, 3, 0, -7, 5, 0, 2].to_vec();
        let codes: Vec<u32> = tau
            .iter()
            .map(|&t| exp_code(t, ring.n()).ok().unwrap())
            .collect();
        let tau_elems: Vec<RingElement> = tau
            .iter()
            .map(|&t| ring.constant(ring.modulus.reduce_i64(t)))
            .collect();
        let cm_tau = pk
            .commit(&pk.pad_to_m(&tau_elems).ok().unwrap())
            .ok()
            .unwrap();
        let mtau_elems: Vec<RingElement> = codes.iter().map(|&c| monomial_ring(&ring, c)).collect();
        let cm_mtau = pk
            .commit(&pk.pad_to_m(&mtau_elems).ok().unwrap())
            .ok()
            .unwrap();
        let mut t = Transcript::new_default(b"lfplus-psi-bad");
        assert!(matches!(
            prove_psi_range(&ring, &bad_tau, &cm_tau, &cm_mtau, &mut t),
            Err(LfPlusMonError::TauOutOfRange { value: 8, .. })
        ));
        // Honest proof, tampered a: the ct(ψ·b) = a check fails.
        let mut t2 = Transcript::new_default(b"lfplus-psi-bad");
        let mut proof = prove_psi_range(&ring, &tau, &cm_tau, &cm_mtau, &mut t2)
            .ok()
            .unwrap();
        proof.a = proof.a.add(&fe(1));
        let mut vt = Transcript::new_default(b"lfplus-psi-bad");
        assert!(matches!(
            verify_psi_range(&ring, tau.len(), &cm_tau, &cm_mtau, &proof, &mut vt),
            Err(LfPlusMonError::PsiRangeFailed)
        ));
        // Tampered b (the e_0 evaluation): Π^mon rejects.
        let mut t3 = Transcript::new_default(b"lfplus-psi-bad");
        let mut proof_b = prove_psi_range(&ring, &tau, &cm_tau, &cm_mtau, &mut t3)
            .ok()
            .unwrap();
        proof_b.mon.e[0][3] = proof_b.mon.e[0][3].add(&fe(2));
        let mut vt3 = Transcript::new_default(b"lfplus-psi-bad");
        assert!(verify_psi_range(&ring, tau.len(), &cm_tau, &cm_mtau, &proof_b, &mut vt3).is_err());
    }

    // ---- Construction 4.1: split/pow double commitments ----

    #[test]
    fn double_commitment_pow_identity_and_binding() {
        // d = 16, d' = 8, ℓ = 11 digits, k = 2, m = 1 column:
        // τ length = 2·1·16·11 = 352 → padded to n = 512.
        let (pk, ring) = setup(4, 512);
        let (pk_dcom, _) = setup(4, 512);
        let d = ring.n();
        let matrix = MonomialMatrix::from_seed(512, 1, d as u32, b"dcom-M")
            .ok()
            .unwrap();
        let cols = matrix.column_ring_elements(&ring).ok().unwrap();
        let com_m: Vec<AjtaiCommitment> = cols
            .iter()
            .map(|c| pk.commit(&pk.pad_to_m(c).ok().unwrap()).ok().unwrap())
            .collect();
        let (dcom, tau, padded_elems) = dcom_commit(&pk_dcom, &ring, &com_m, 8, 11).ok().unwrap();
        // Digit norm gate: ∥τ∥∞ ≤ d'/2 = 4 (within (−d', d')).
        assert!(tau.iter().all(|t| t.unsigned_abs() <= 4));
        // pow identity (Lemma 4.1's precondition): pow(split(com(M))) = com(M).
        let rebuilt = pow(&ring, &tau, 2, 1, 8, 11).ok().unwrap();
        for (row, entry) in rebuilt.iter().enumerate() {
            assert_eq!(com_m[0].rows[row], *entry, "pow row {row}");
        }
        // dcom opens the padded τ.
        assert!(pk_dcom.verify_opening(&dcom, &padded_elems).is_ok());
        // Binding surface (Lemma 4.1): a DIFFERENT τ' with the same
        // pow-image must fail the dcom opening — pow is not injective, but
        // com binds τ. Tamper one digit:
        let mut tau_bad = padded_elems_as_ints(&tau, 512);
        tau_bad[0] += 1;
        let bad_elems: Vec<RingElement> = tau_bad
            .iter()
            .map(|&t| ring.constant(ring.modulus.reduce_i64(t)))
            .collect();
        assert!(pk_dcom.verify_opening(&dcom, &bad_elems).is_err());
        // Pad-region freedom: τ' differing only in the padding has the same
        // pow image but a different commitment — not a valid opening of C.
        let mut tau_pad = tau.clone();
        while tau_pad.len() < 512 {
            tau_pad.push(3);
        }
        let pad_elems: Vec<RingElement> = tau_pad
            .iter()
            .map(|&t| ring.constant(ring.modulus.reduce_i64(t)))
            .collect();
        assert_eq!(pow(&ring, &tau_pad, 2, 1, 8, 11).ok().unwrap(), rebuilt);
        assert!(pk_dcom.verify_opening(&dcom, &pad_elems).is_err());
        // Distinct matrices give distinct double commitments (whp).
        let other = MonomialMatrix::from_seed(512, 1, d as u32, b"dcom-M2")
            .ok()
            .unwrap();
        let other_cols = other.column_ring_elements(&ring).ok().unwrap();
        let com_m2: Vec<AjtaiCommitment> = other_cols
            .iter()
            .map(|c| pk.commit(&pk.pad_to_m(c).ok().unwrap()).ok().unwrap())
            .collect();
        let (dcom2, _, _) = dcom_commit(&pk_dcom, &ring, &com_m2, 8, 11).ok().unwrap();
        assert_ne!(dcom.rows, dcom2.rows);
    }

    fn padded_elems_as_ints(tau: &[i64], n: usize) -> Vec<i64> {
        let mut out = tau.to_vec();
        while out.len() < n {
            out.push(0);
        }
        out
    }

    #[test]
    fn split_covers_all_coefficients() {
        // The gadget with d' = 8, ℓ = 11 covers every balanced coefficient
        // in (−q/2, q/2] (q ≈ 2^31 < 4·(8^11−1)/7 ≈ 2^32.2).
        for c in [
            0i64,
            1,
            -1,
            4,
            -4,
            100,
            -1000,
            1 << 20,
            -(1 << 29),
            (1 << 30) - 1,
        ] {
            let digits = split_coefficient(c, 8, 11);
            let base: i128 = 8;
            let mut acc: i128 = 0;
            let mut w: i128 = 1;
            for &d in &digits {
                acc += d as i128 * w;
                w *= base;
            }
            assert_eq!(acc as i64, c, "coefficient {c}");
            assert!(digits.iter().all(|d| d.unsigned_abs() <= 4));
        }
    }

    /// Deterministic test challenge (not part of the protocol).
    fn challenge_from_seed(seed: &[u8], idx: u64) -> Fq2 {
        let mut input = seed.to_vec();
        input.extend_from_slice(&idx.to_le_bytes());
        let bytes = Transcript::xof(b"lfplus-mon-test-chal", &input, 16);
        let c0 = Goldilocks::from_u64(u64::from_le_bytes(bytes[..8].try_into().ok().unwrap()));
        let c1 = Goldilocks::from_u64(u64::from_le_bytes(bytes[8..16].try_into().ok().unwrap()));
        Fq2::new(c0, c1)
    }
}
