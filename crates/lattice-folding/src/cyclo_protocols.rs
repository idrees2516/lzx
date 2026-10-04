//! Cyclo Π^range and Π^ext (Wave 7.6): the two RoK building blocks of the
//! Cyclo folding scheme (Garreta–Lipmaa–Luhääär–Osadnik, ePrint 2026/359),
//! realized as real prover/verifier protocols over `F_{q^2}`.
//!
//! * **Π^range** (paper §4, Fig 1, Thm 1) proves every coefficient of the
//!   *chunked* (digit-decomposed) witness of an extension commitment lies
//!   in `[−b, b]` — the digit-decomposition MLE structure. The sumcheck
//!   polynomial is `f̂(X) = eq(X; η) · Π_{j=−b}^{b} (f(X) − j)` where
//!   `f = MLE[cf(v)]` over the flattened digit vector: individual degree
//!   `1 + (2b+1) = 2b+2` (Thm 1's degree law). For `b = 1` the range
//!   product is exactly the **X³−X Karatsuba-style identity**
//!   `Π_{j=−1}^{1}(t−j) = t³ − t = (t−1)·t·(t+1)`, and the leaf
//!   evaluation short-circuits zero factors (the product vanishes on
//!   in-range hypercube leaves, so whole factor chains are skipped). The
//!   verifier checks the terminal claim against a *verifier-computable
//!   digit reconstruction*: `t = Σ_i eq(u_head, ⟨i⟩)·d_i` (the layered-ML
//!   decomposition) and `c_at = Σ_i 2^{i·chunk_log}·d_i` (the digit
//!   reconstruction of the ORIGINAL committed coefficients — the caller
//!   authenticates `c_at` through the PCS layer, exactly like
//!   `latticefold_plus::verify_range`'s `coeff_claim_at_point`).
//!
//!   **LZX realization note on the range bound**: the paper's Thm-1 law is
//!   "coefficients in `[−b, b]`, individual degree `2b+2`". The chunked
//!   digits have `|d| ≤ b = 2^{chunk_log−1}`; certifying those digits
//!   in-range certifies the ORIGINAL committed ring element's coefficients
//!   lie in `[−2^B, 2^B]` for `B` with
//!   `b·(2^{num_chunks·chunk_log}−1)/(2^{chunk_log}−1) ≤ 2^B` (see
//!   [`PiRangeStatement::certified_coefficient_bound`]).
//!
//! * **Π^ext** (paper §5, Fig 2, Thm 2) commits to the digit decomposition
//!   `v = (w_0, …, w_{ℓ−1})` of a witness `w = Σ_i (2b)^i·w_i` under the
//!   extension key, with the **RoK constraint rows of Fig-2 steps 3–4**:
//!   the verifier samples a short challenge `c` (tensor(ĉ) in the paper)
//!   AFTER the transcript absorbs the INPUT commitment, and the output
//!   instance carries the challenge-batched constraint row
//!   `⟨c, ((2b)^i ⊗ A)·v⟩ = ⟨c, t⟩`, which certifies the committed `v`
//!   recomposes a witness of the input relation (`A·w = t`). The
//!   FS hole (challenge not bound to the input commitment) is structurally
//!   closed here: the challenge is derived from
//!   `(t_input, t_ext, params, fold counter)` in that order. The
//!   post-refresh length assertion (witness length must equal the key's
//!   `m`) is enforced by the base module's `fold`/`fold_ring_challenge`
//!   (Wave 6.4) and re-checked here on the extension side.
//!
//! Asymptotic honesty: kernel-scale parameters (n = 16–64, m ≤ 256,
//! b ≤ 8) keep every loop trivial; the paper's communication profile
//! (`1 R_{q^e} element + (2b+2)·ℓ + 1 F_{q^e} elements` for Π^range,
//! `a′ R_q elements` for Π^ext) is preserved in shape — the Fq2 engine
//! sends `(2b+3)·ℓ` round values plus the leaf claims.

use crate::fq2_sumcheck::{self, Fq2SumcheckError, Fq2VirtualPoly};
use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiError, AjtaiPublicKey};
use lattice_core::extension::{challenge_fq2, Fq2};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;
use lattice_ring::RingElement;

/// Balanced integer → Goldilocks (canonical representative).
fn fe_i64(x: i64) -> Goldilocks {
    let m = lattice_core::field::GOLDILOCKS_MODULUS as i128;
    Goldilocks::from_u64((x as i128).rem_euclid(m) as u64)
}

/// The range product `Π_{j=−b}^{b} (t − j)` over F_{q^2}, with the
/// Karatsuba-style zero-skip: the product vanishes as soon as one factor
/// is zero (every in-range hypercube leaf), so remaining factors are
/// skipped. For `b = 1` this is exactly `t³ − t`.
pub(crate) fn range_product(t: Fq2, b: u64) -> Fq2 {
    let mut acc = Fq2::ONE;
    for j in 0..=2 * b {
        // j enumerates the shifted values −b..=b.
        let shift = j as i64 - b as i64;
        let factor = t.sub(&Fq2::from_base(fe_i64(shift)));
        if factor.is_zero() {
            // Karatsuba-style shortcut: the whole product is zero.
            return Fq2::ZERO;
        }
        acc = acc.mul(&factor);
    }
    acc
}

/// Extract the signed digit layers of a ring element's coefficients:
/// layer `i` holds every coefficient's `i`-th base-`2^chunk_log` digit
/// (balanced, `|d| ≤ b = 2^{chunk_log−1}`).
#[allow(dead_code)] // utility path retained for the scale-up
pub(crate) fn digit_layers(
    ring: &lattice_ring::RingConfig,
    e: &RingElement,
    chunk_log: u32,
) -> Vec<Vec<i64>> {
    crate::cyclo::chunk_element(ring, e, chunk_log)
        .into_iter()
        .map(|layer| {
            let half = ring.modulus.q / 2;
            layer
                .coeffs()
                .iter()
                .map(|c| {
                    if *c <= half {
                        *c as i64
                    } else {
                        *c as i64 - ring.modulus.q as i64
                    }
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycloProtocolError {
    Sumcheck(Fq2SumcheckError),
    Ajtai(AjtaiError),
    Ring(lattice_ring::RingError),
    ShortChallenge(lattice_core::short_challenge::ShortChallengeError),
    TranscriptFailure,
    /// A digit lies outside `[−b, b]` (the honest prover refuses; the
    /// verifier's leaf check rejects the certificate).
    DigitOutOfRange {
        value: i64,
        bound: u64,
    },
    /// The Π^ext constraint row failed (committed digits do not
    /// recompose a witness of the input relation, or the FS challenge
    /// does not replay).
    ConstraintRowFailed,
    /// Terminal digit-reconstruction mismatch.
    ReconstructionFailed,
    /// Shape mismatch (layer count / length bookkeeping).
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    /// The norm gate `‖v‖∞ ≤ b` refused the opening (hard gate).
    NormGateExceeded {
        norm: u64,
        bound: u64,
    },
}

impl From<Fq2SumcheckError> for CycloProtocolError {
    fn from(e: Fq2SumcheckError) -> Self {
        CycloProtocolError::Sumcheck(e)
    }
}

impl From<AjtaiError> for CycloProtocolError {
    fn from(e: AjtaiError) -> Self {
        CycloProtocolError::Ajtai(e)
    }
}

impl From<lattice_ring::RingError> for CycloProtocolError {
    fn from(e: lattice_ring::RingError) -> Self {
        CycloProtocolError::Ring(e)
    }
}

impl From<lattice_core::short_challenge::ShortChallengeError> for CycloProtocolError {
    fn from(e: lattice_core::short_challenge::ShortChallengeError) -> Self {
        CycloProtocolError::ShortChallenge(e)
    }
}

/// Π^range public statement (Fig 1): digit layers of a committed ring
/// element, each of length `num_coeffs` (power of two), digit bound `b`.
#[derive(Clone, Debug)]
pub struct PiRangeStatement {
    /// Coefficients per layer (power of two).
    pub num_coeffs: usize,
    /// Digit layers (power of two; zero layers are in range).
    pub num_layers: usize,
    /// Digit shift between layers (the chunk geometry).
    pub chunk_log: u32,
    /// Digit bound `b` (the paper's range; individual degree 2b+2).
    pub bound: u64,
}

impl PiRangeStatement {
    /// Certified coefficient bound of the ORIGINAL committed element:
    /// `|c| ≤ b·(2^{num_layers·chunk_log} − 1)/(2^{chunk_log} − 1)`.
    pub fn certified_coefficient_bound(&self) -> u128 {
        let span: u128 = 1u128 << self.chunk_log;
        let b = self.bound as u128;
        let mut acc = 0u128;
        let mut pow = 1u128;
        for _ in 0..self.num_layers {
            acc += b * pow;
            pow *= span;
        }
        acc
    }

    fn num_vars(&self) -> usize {
        (self.num_coeffs * self.num_layers).trailing_zeros() as usize
    }
}

/// A Π^range proof: the degree-(2b+2) Fq2 sumcheck plus the leaf claims.
#[derive(Clone, Debug)]
pub struct PiRangeProof {
    /// Round polynomials (values at `X = 0..=2b+2` per round).
    pub rounds: Vec<Vec<Fq2>>,
    /// Leaf claim `t = f(u)` (paper: `t̃ = ⟨u′, w⟩` with `Trace(t̃) = t`
    /// by the dual-basis trace lemma — realized here through the layer
    /// decomposition check below).
    pub leaf: Fq2,
    /// Per-layer digit claims `d_i = MLE[v_i](u_tail)`.
    pub digit_claims: Vec<Fq2>,
}

/// Prove Π^range: the honest prover runs the sumcheck on the true digit
/// MLE. If any digit is out of `[−b, b]`, the claimed sum `Σ f̂ = 0` is
/// false and the prover-side consistency guard refuses to transcribe a
/// lying round polynomial (fail closed — a cheating prover must instead
/// forge rounds, which the verifier's round checks reject).
pub fn prove_range(
    statement: &PiRangeStatement,
    layers: &[Vec<i64>],
    transcript: &mut Transcript,
) -> Result<PiRangeProof, CycloProtocolError> {
    let total = statement.num_coeffs * statement.num_layers;
    if layers.len() != statement.num_layers
        || layers.iter().any(|l| l.len() != statement.num_coeffs)
    {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: statement.num_layers,
            got: layers.len(),
        });
    }
    if !total.is_power_of_two() || !statement.num_coeffs.is_power_of_two() {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: 0,
            got: total,
        });
    }
    // Hard in-protocol norm gate on the digits themselves (the paper's
    // ∥w∥ ≤ b precondition of Thm 1).
    for layer in layers {
        for d in layer {
            if d.unsigned_abs() > statement.bound {
                return Err(CycloProtocolError::DigitOutOfRange {
                    value: *d,
                    bound: statement.bound,
                });
            }
        }
    }
    // f := MLE[cf(v)] — the flattened digit vector, digit-major (the
    // layer index is the high part of the hypercube index).
    let mut flat: Vec<Fq2> = Vec::with_capacity(total);
    for layer in layers {
        for d in layer {
            flat.push(Fq2::from_base(fe_i64(*d)));
        }
    }
    // η ← F_{q^2}^ℓ (paper step 2) — after absorbing the statement.
    let num_vars = statement.num_vars();
    absorb_statement(transcript, statement)?;
    let eta = challenge_fq2_vec(transcript, b"cyclo-range-eta", num_vars)?;
    // eq(X; η) evaluations on the hypercube.
    let eq_evals = eq_table(&eta);
    // f̂ = eq · Π_{j=−b}^{b}(f − j): one eq factor plus (2b+1) shifted
    // copies of the digit MLE (the affine shifts realize the range
    // product without materializing a degree-(2b+1) factor per term).
    let mut vp = Fq2VirtualPoly::new(num_vars);
    let eq_id = vp.add_factor(eq_evals)?;
    let mut shifted_ids = Vec::with_capacity(2 * statement.bound as usize + 1);
    for j in 0..=2 * statement.bound {
        let shift = j as i64 - statement.bound as i64;
        let shifted: Vec<Fq2> = flat
            .iter()
            .map(|v| v.sub(&Fq2::from_base(fe_i64(shift))))
            .collect();
        shifted_ids.push(vp.add_factor(shifted)?);
    }
    let mut ids = Vec::with_capacity(2 * statement.bound as usize + 2);
    ids.push(eq_id);
    ids.extend(shifted_ids);
    vp.add_term(Fq2::ONE, ids)?;

    let out = fq2_sumcheck::prove(&vp, Fq2::ZERO, transcript)?;
    // The unshifted digit MLE is the factor at position j = b (shift 0)
    // among the shifted copies — factor id 1 + b.
    let unshifted_id = 1 + statement.bound as usize;
    // Layer digit claims at the tail of the challenge point (the low
    // log2(num_coeffs) coordinates — the layer variables were bound
    // first as the high bits).
    let tail_vars = statement.num_coeffs.trailing_zeros() as usize;
    let head_vars = num_vars - tail_vars;
    let u = &out.challenges;
    let tail: Vec<Fq2> = u[head_vars..].to_vec();
    let mut digit_claims = Vec::with_capacity(statement.num_layers);
    for layer in layers {
        digit_claims.push(mle_eval(layer, &tail)?);
    }
    Ok(PiRangeProof {
        rounds: out.proof.rounds,
        leaf: out.factor_claims[unshifted_id],
        digit_claims,
    })
}

/// Verifier-side Π^range: replays the sumcheck and checks the terminal
/// claim against the verifier-computable digit reconstruction. Returns
/// the challenge point (for the PCS layer).
pub fn verify_range(
    statement: &PiRangeStatement,
    proof: &PiRangeProof,
    coeff_claim: Fq2,
    transcript: &mut Transcript,
) -> Result<Vec<Fq2>, CycloProtocolError> {
    let num_vars = statement.num_vars();
    if proof.digit_claims.len() != statement.num_layers {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: statement.num_layers,
            got: proof.digit_claims.len(),
        });
    }
    absorb_statement(transcript, statement)?;
    let eta = challenge_fq2_vec(transcript, b"cyclo-range-eta", num_vars)?;
    let degree = 2 * statement.bound as usize + 2;
    let fq_proof = fq2_sumcheck::Fq2SumcheckProof {
        rounds: proof.rounds.clone(),
    };
    let verdict = fq_proof.verify(num_vars, degree, Fq2::ZERO, transcript, None)?;
    let u = verdict.point;
    // Leaf check (paper step 6): s = eq(u; η) · Π_{j=−b}^{b}(t − j) — the
    // verifier recomputes the range product from the claimed leaf `t`
    // (the digit-reconstruction binding below authenticates it).
    let eq_u = eq_point(&eta, &u);
    let expected_final = eq_u.mul(&range_product(proof.leaf, statement.bound));
    if verdict.final_claim != expected_final {
        return Err(CycloProtocolError::ReconstructionFailed);
    }
    // Digit reconstruction, level 1: the flattened-MLE leaf decomposes as
    // t = Σ_i eq(u_head, ⟨i⟩)·d_i (MLE over the layer index).
    let tail_vars = statement.num_coeffs.trailing_zeros() as usize;
    let head_vars = num_vars - tail_vars;
    let head: Vec<Fq2> = u[..head_vars].to_vec();
    let mut leaf_rec = Fq2::ZERO;
    for (i, d) in proof.digit_claims.iter().enumerate() {
        let mut w = Fq2::ONE;
        for (var, hv) in head.iter().enumerate() {
            let bit = (i >> (head.len() - 1 - var)) & 1;
            let term = if bit == 1 { *hv } else { Fq2::ONE.sub(hv) };
            w = w.mul(&term);
        }
        leaf_rec = leaf_rec.add(&d.mul(&w));
    }
    if leaf_rec != proof.leaf {
        return Err(CycloProtocolError::ReconstructionFailed);
    }
    // Digit reconstruction, level 2: the ORIGINAL coefficient MLE claim
    // (PCS-authenticated by the caller) equals Σ_i 2^{i·chunk_log}·d_i.
    let mut coeff_rec = Fq2::ZERO;
    for (i, d) in proof.digit_claims.iter().enumerate() {
        let shift = (i as u32).checked_mul(statement.chunk_log).ok_or(
            CycloProtocolError::ShapeMismatch {
                expected: usize::MAX,
                got: i,
            },
        )?;
        if shift >= 64 {
            return Err(CycloProtocolError::ShapeMismatch {
                expected: 64,
                got: shift as usize,
            });
        }
        coeff_rec = coeff_rec.add(&d.mul(&Fq2::from_base(Goldilocks::from_u64(1u64 << shift))));
    }
    if coeff_rec != coeff_claim {
        return Err(CycloProtocolError::ReconstructionFailed);
    }
    Ok(u)
}

/// The Π^ext RoK fold result: the extension commitment plus the
/// verifier-checkable constraint rows (Fig 2 steps 3–4).
#[derive(Clone, Debug)]
pub struct ExtFoldProof {
    /// Extension commitment `t_ext = R·v` (the chunked witness).
    pub ext_commitment: AjtaiCommitment,
    /// Per-row challenge `c ∈ C^k` (FS-derived AFTER absorbing the input
    /// commitment — the statement-binding discipline).
    pub challenge: Vec<i64>,
    /// The challenge-batched constraint row value `⟨c, t⟩ ∈ R_q`
    /// (verifier-computable from the input commitment; carried in the
    /// folded instance per Fig-2 step 4).
    pub batched_rhs: RingElement,
}

/// Commit the digit decomposition of a witness under the extension key,
/// with the Fig-2 RoK constraint rows:
///
/// * `v = chunk(w)` (`w = Σ_i (2b)^i w_i`, `‖v‖∞ ≤ b`),
/// * `t_ext = R·v` committed,
/// * challenge `c ← C^k` sampled from the transcript AFTER absorbing
///   `(t_input, t_ext, params, fold counter)`,
/// * constraint rows: for every base-key row `r`,
///   `Σ_{j,i} (2b)^i·A[r][j]·v[i·m+j] = t[r]` — the digit-folded linear
///   map recomposes the input relation — carried batched as
///   `⟨c, ((2b)^i ⊗ A)v⟩ = ⟨c, t⟩`.
pub fn ext_commit_rok(
    pk: &AjtaiPublicKey,
    pk_ext: &AjtaiPublicKey,
    input_commitment: &AjtaiCommitment,
    witness: &[RingElement],
    chunk_log: u32,
    folds_since_refresh: usize,
    challenge_spec: &lattice_core::short_challenge::ShortChallengeSpec,
) -> Result<ExtFoldProof, CycloProtocolError> {
    let ring = &pk.params.ring;
    if witness.len() != pk.params.m {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: pk.params.m,
            got: witness.len(),
        });
    }
    if pk_ext.params.ring.modulus.q != ring.modulus.q || pk_ext.params.ring.log_n != ring.log_n {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: ring.n(),
            got: pk_ext.params.ring.n(),
        });
    }
    let digit_bound = 1u64 << (chunk_log - 1);
    // v = chunk(w), DIGIT-MAJOR (the paper's v^T = (w_0^T, …, w_{ℓ−1}^T)):
    // v[i·m + j] = chunk i of witness element j, so that the constraint
    // row Σ_{j,i} (2b)^i·A[r][j]·v[i·m+j] recomposes A·w exactly.
    let num_chunks = ((32 + chunk_log - 1) / chunk_log.max(1)) as usize;
    let mut v: Vec<RingElement> = vec![ring.zero(); witness.len() * num_chunks];
    for (j, e) in witness.iter().enumerate() {
        for (i, c) in crate::cyclo::chunk_element(ring, e, chunk_log)
            .into_iter()
            .enumerate()
        {
            v[i * witness.len() + j] = c;
        }
    }
    if v.len() > pk_ext.params.m {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: pk_ext.params.m,
            got: v.len(),
        });
    }
    let mut padded_v = v;
    while padded_v.len() < pk_ext.params.m {
        padded_v.push(ring.zero());
    }
    // Hard digit norm gate (the paper's ∥v∥∞ ≤ b precondition).
    for d in &padded_v {
        if d.infinity_norm() as u64 > digit_bound {
            return Err(CycloProtocolError::NormGateExceeded {
                norm: d.infinity_norm() as u64,
                bound: digit_bound,
            });
        }
    }
    let ext_commitment = pk_ext.commit(&padded_v)?;
    // FS: absorb the INPUT commitment (and params + fold counter) BEFORE
    // the challenge — the Wave-6.4 fix, structural in Π^ext.
    let mut transcript = Transcript::new_default(b"lzx-cyclo-ext");
    transcript
        .append_bytes(b"input", &input_commitment.to_bytes())
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    transcript
        .append_bytes(b"ext", &ext_commitment.to_bytes())
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    absorb_ext_params(&mut transcript, pk, pk_ext, chunk_log, folds_since_refresh)?;
    let seed = transcript
        .challenge_bytes(b"ext-c", 32)
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    // Per-row challenge c ∈ C^k (C = the spec's family).
    let challenge = challenge_spec.sample(&seed)?;
    if challenge.coefficients.len() != pk.params.k {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: pk.params.k,
            got: challenge.coefficients.len(),
        });
    }
    // Challenge-batched RHS ⟨c, t⟩ = Σ_r c_r·t_r — the folded instance's
    // new constraint value (Fig-2 step 4's ⟨c, y⟩).
    let mut batched = ring.zero();
    for (row, c_r) in input_commitment
        .rows
        .iter()
        .zip(challenge.coefficients.iter())
    {
        if *c_r == 0 {
            continue;
        }
        batched = batched.add(&row.scale_i64(*c_r))?;
    }
    Ok(ExtFoldProof {
        ext_commitment,
        challenge: challenge.coefficients.clone(),
        batched_rhs: batched,
    })
}

/// Verify a Π^ext fold (the previously-missing verifier): re-derives the
/// FS challenge from the public statement, checks the extension
/// commitment opens the chunked witness, checks the hard digit norm gate,
/// and checks the **RoK constraint rows** — the digit-recomposition
/// `Σ_{j,i} (2b)^i·A[r][j]·v[i·m+j] = t[r]` for every base row (which
/// implies the challenge-batched row `⟨c, ((2b)^i ⊗ A)v⟩ = ⟨c, t⟩`) —
/// against the supplied opening `v` (the decider model: the opening is
/// PCS-authenticated upstream; tests supply the prover's `v`).
#[allow(clippy::too_many_arguments)]
pub fn verify_ext_fold(
    pk: &AjtaiPublicKey,
    pk_ext: &AjtaiPublicKey,
    input_commitment: &AjtaiCommitment,
    proof: &ExtFoldProof,
    opening_v: &[RingElement],
    chunk_log: u32,
    folds_since_refresh: usize,
    challenge_spec: &lattice_core::short_challenge::ShortChallengeSpec,
) -> Result<(), CycloProtocolError> {
    let ring = &pk.params.ring;
    if opening_v.len() != pk_ext.params.m {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: pk_ext.params.m,
            got: opening_v.len(),
        });
    }
    // 1. The extension commitment opens v.
    pk_ext.verify_opening(&proof.ext_commitment, opening_v)?;
    // 2. Hard digit norm gate.
    let digit_bound = 1u64 << (chunk_log - 1);
    for d in opening_v {
        if d.infinity_norm() as u64 > digit_bound {
            return Err(CycloProtocolError::NormGateExceeded {
                norm: d.infinity_norm() as u64,
                bound: digit_bound,
            });
        }
    }
    // 3. Re-derive the FS challenge from PUBLIC data only.
    let mut transcript = Transcript::new_default(b"lzx-cyclo-ext");
    transcript
        .append_bytes(b"input", &input_commitment.to_bytes())
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    transcript
        .append_bytes(b"ext", &proof.ext_commitment.to_bytes())
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    absorb_ext_params(&mut transcript, pk, pk_ext, chunk_log, folds_since_refresh)?;
    let seed = transcript
        .challenge_bytes(b"ext-c", 32)
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    let challenge = challenge_spec.sample(&seed)?;
    if challenge.coefficients != proof.challenge {
        return Err(CycloProtocolError::ConstraintRowFailed);
    }
    // 4. RoK constraint rows: for each base row r,
    //    Σ_{j<m} Σ_{i<num_chunks} (2b)^i·A[r][j]·v[i·m+j] = t[r].
    let num_chunks = ((32 + chunk_log - 1) / chunk_log.max(1)) as usize;
    for (r, t_row) in input_commitment.rows.iter().enumerate() {
        let mut acc = ring.zero();
        for j in 0..pk.params.m {
            for i in 0..num_chunks {
                let idx = i * pk.params.m + j;
                let d = opening_v
                    .get(idx)
                    .ok_or(CycloProtocolError::ShapeMismatch {
                        expected: idx + 1,
                        got: opening_v.len(),
                    })?;
                if d.is_zero() {
                    continue;
                }
                let weight = 1i64 << (i as u32 * chunk_log);
                let a_rj = pk.entry(r, j).ok_or(CycloProtocolError::ShapeMismatch {
                    expected: pk.params.m,
                    got: j,
                })?;
                let prod = a_rj.mul(&d.scale_i64(weight))?;
                acc = acc.add(&prod)?;
            }
        }
        if acc != *t_row {
            return Err(CycloProtocolError::ConstraintRowFailed);
        }
    }
    // 5. The carried batched RHS matches ⟨c, t⟩ (Fig-2 step 4).
    let mut batched = ring.zero();
    for (row, c_r) in input_commitment.rows.iter().zip(proof.challenge.iter()) {
        if *c_r == 0 {
            continue;
        }
        batched = batched.add(&row.scale_i64(*c_r))?;
    }
    if batched != proof.batched_rhs {
        return Err(CycloProtocolError::ConstraintRowFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Small helpers shared by the two protocols.
// ---------------------------------------------------------------------------

fn absorb_ext_params(
    transcript: &mut Transcript,
    pk: &AjtaiPublicKey,
    pk_ext: &AjtaiPublicKey,
    chunk_log: u32,
    folds_since_refresh: usize,
) -> Result<(), CycloProtocolError> {
    let ring = &pk.params.ring;
    let mut params_bytes = Vec::with_capacity(24);
    params_bytes.extend_from_slice(&ring.modulus.q.to_le_bytes());
    params_bytes.extend_from_slice(&ring.log_n.to_le_bytes());
    params_bytes.extend_from_slice(&(pk.params.k as u32).to_le_bytes());
    params_bytes.extend_from_slice(&(pk.params.m as u32).to_le_bytes());
    params_bytes.extend_from_slice(&(pk_ext.params.m as u32).to_le_bytes());
    params_bytes.extend_from_slice(&chunk_log.to_le_bytes());
    params_bytes.extend_from_slice(&(folds_since_refresh as u32).to_le_bytes());
    transcript
        .append_bytes(b"params", &params_bytes)
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    Ok(())
}

fn absorb_statement(
    transcript: &mut Transcript,
    statement: &PiRangeStatement,
) -> Result<(), CycloProtocolError> {
    let mut buf = Vec::with_capacity(24);
    buf.extend_from_slice(&(statement.num_coeffs as u32).to_le_bytes());
    buf.extend_from_slice(&(statement.num_layers as u32).to_le_bytes());
    buf.extend_from_slice(&statement.chunk_log.to_le_bytes());
    buf.extend_from_slice(&statement.bound.to_le_bytes());
    transcript
        .append_bytes(b"cyclo-range-stmt", &buf)
        .map_err(|_| CycloProtocolError::TranscriptFailure)?;
    Ok(())
}

fn challenge_fq2_vec(
    transcript: &mut Transcript,
    label: &[u8],
    n: usize,
) -> Result<Vec<Fq2>, CycloProtocolError> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut label_buf = label.to_vec();
        label_buf.extend_from_slice(&(i as u32).to_le_bytes());
        out.push(
            challenge_fq2(transcript, &label_buf)
                .map_err(|_| CycloProtocolError::TranscriptFailure)?,
        );
    }
    Ok(out)
}

/// eq(X; η) evaluations over the boolean hypercube.
fn eq_table(eta: &[Fq2]) -> Vec<Fq2> {
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

/// eq(u; η) at an arbitrary point (the leaf factor).
fn eq_point(eta: &[Fq2], u: &[Fq2]) -> Fq2 {
    let mut acc = Fq2::ONE;
    for (e, x) in eta.iter().zip(u.iter()) {
        let term = x.mul(e).add(&Fq2::ONE.sub(x).mul(&Fq2::ONE.sub(e)));
        acc = acc.mul(&term);
    }
    acc
}

/// MLE evaluation of an integer table at a point (balanced digits lifted
/// into Fq2).
fn mle_eval(table: &[i64], point: &[Fq2]) -> Result<Fq2, CycloProtocolError> {
    let num_vars = table.len().trailing_zeros() as usize;
    if point.len() != num_vars {
        return Err(CycloProtocolError::ShapeMismatch {
            expected: num_vars,
            got: point.len(),
        });
    }
    let mut acc = Fq2::ZERO;
    for (idx, v) in table.iter().enumerate() {
        let mut w = Fq2::ONE;
        for (var, pv) in point.iter().enumerate() {
            let bit = (idx >> (num_vars - 1 - var)) & 1;
            let term = if bit == 1 { *pv } else { Fq2::ONE.sub(pv) };
            w = w.mul(&term);
        }
        acc = acc.add(&Fq2::from_base(fe_i64(*v)).mul(&w));
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_core::short_challenge::{ShortChallengeFamily, ShortChallengeSpec};
    use lattice_ring::{Modulus32, RingConfig};

    fn fe(x: i64) -> Fq2 {
        Fq2::from_base(fe_i64(x))
    }

    fn setup(log_n: u32, m: usize) -> (AjtaiPublicKey, RingConfig) {
        let ring = RingConfig::new(Modulus32::Q_32, log_n).ok().unwrap();
        let params = lattice_commitment::ajtai::AjtaiParams {
            ring: ring.clone(),
            k: 2,
            m,
            norm_bound: 1 << 26,
        };
        let pk = AjtaiPublicKey::from_seed(params, [77u8; 32]).ok().unwrap();
        (pk, ring)
    }

    fn challenge_spec(k: usize) -> ShortChallengeSpec {
        ShortChallengeSpec {
            n: k,
            family: ShortChallengeFamily::SmallSet {
                values: vec![0, 1, -1, 2, -2],
            },
        }
    }

    /// Derive the sumcheck challenge point by replaying the verifier's
    /// transcript (test oracle for computing caller-side claims).
    fn derive_point(
        statement: &PiRangeStatement,
        proof: &PiRangeProof,
        protocol: &[u8],
    ) -> Vec<Fq2> {
        let mut rp = Transcript::new_default(protocol);
        absorb_statement(&mut rp, statement).ok().unwrap();
        let _eta = challenge_fq2_vec(&mut rp, b"cyclo-range-eta", statement.num_vars())
            .ok()
            .unwrap();
        let fq_proof = fq2_sumcheck::Fq2SumcheckProof {
            rounds: proof.rounds.clone(),
        };
        fq_proof
            .verify(
                statement.num_vars(),
                2 * statement.bound as usize + 2,
                Fq2::ZERO,
                &mut rp,
                None,
            )
            .ok()
            .unwrap()
            .point
    }

    #[test]
    fn range_product_x_cubed_minus_x() {
        // b = 1: Π_{j=−1}^{1}(t−j) = t³ − t.
        for t in [0i64, 1, -1, 2, 5, -3] {
            let z = fe(t);
            let naive = z.mul(&z).mul(&z).sub(&z);
            assert_eq!(range_product(z, 1), naive, "t = {t}");
        }
        // In-range digits are roots of the product.
        assert!(range_product(fe(0), 1).is_zero());
        assert!(range_product(fe(1), 1).is_zero());
        assert!(range_product(fe(-1), 1).is_zero());
        // Out-of-range values are not.
        assert!(!range_product(fe(2), 1).is_zero());
        // General b: integer cross-check for small b.
        let t = 3i64;
        let mut expected = 1i64;
        for j in -2i64..=2 {
            expected *= t - j;
        }
        assert_eq!(range_product(fe(t), 2), fe(expected));
    }

    #[test]
    fn pi_range_happy_path() {
        // Digits of a small witness, chunk_log = 2 (b = 2), 8 chunks of
        // 16 coefficients → hypercube of 128 points, degree 2b+2 = 6.
        let (_, ring) = setup(4, 4);
        let w = lattice_commitment::ajtai::sample_small_secret(&ring, 1, 1 << 8, b"range-ok");
        let layers = digit_layers(&ring, &w[0], 2);
        let statement = PiRangeStatement {
            num_coeffs: layers[0].len(),
            num_layers: layers.len(),
            chunk_log: 2,
            bound: 2,
        };
        let mut t = Transcript::new_default(b"cyclo-pi-range-test");
        let proof = prove_range(&statement, &layers, &mut t).ok().unwrap();
        // Coefficient MLE claim at the tail point (test oracle: locally
        // computed from the ORIGINAL coefficients; the PCS layer
        // authenticates it in the full stack).
        let half = ring.modulus.q / 2;
        let coeffs: Vec<i64> = w[0]
            .coeffs()
            .iter()
            .map(|c| {
                if *c <= half {
                    *c as i64
                } else {
                    *c as i64 - ring.modulus.q as i64
                }
            })
            .collect();
        let u = derive_point(&statement, &proof, b"cyclo-pi-range-test");
        let tail_vars = statement.num_coeffs.trailing_zeros() as usize;
        let head_vars = statement.num_vars() - tail_vars;
        let tail: Vec<Fq2> = u[head_vars..].to_vec();
        let coeff_claim = mle_eval(&coeffs, &tail).ok().unwrap();
        let mut vt = Transcript::new_default(b"cyclo-pi-range-test");
        assert!(verify_range(&statement, &proof, coeff_claim, &mut vt).is_ok());
        // A wrong coefficient claim is rejected (the reconstruction is
        // part of the verifier).
        let mut vt2 = Transcript::new_default(b"cyclo-pi-range-test");
        assert!(matches!(
            verify_range(&statement, &proof, coeff_claim.add(&fe(1)), &mut vt2),
            Err(CycloProtocolError::ReconstructionFailed)
        ));
    }

    #[test]
    fn pi_range_out_of_range_rejected() {
        // One digit over the bound: the honest prover fails closed…
        let layers = vec![vec![0i64, 1, -1, 0, 2, -2, 0, 1], vec![0i64; 8]];
        let statement = PiRangeStatement {
            num_coeffs: 8,
            num_layers: 2,
            chunk_log: 2,
            bound: 2,
        };
        let mut bad = layers.clone();
        bad[0][3] = 3; // out of [−2, 2]
        let mut t = Transcript::new_default(b"cyclo-pi-range-bad");
        assert!(matches!(
            prove_range(&statement, &bad, &mut t),
            Err(CycloProtocolError::DigitOutOfRange { value: 3, bound: 2 })
        ));
        // …and a tampered leaf claim is rejected by the range-product /
        // reconstruction checks (the round polynomials desync or the
        // layer decomposition mismatches).
        let mut t2 = Transcript::new_default(b"cyclo-pi-range-bad");
        let proof = prove_range(&statement, &layers, &mut t2).ok().unwrap();
        let mut tampered = proof.clone();
        tampered.leaf = tampered.leaf.add(&fe(1));
        let mut vt2 = Transcript::new_default(b"cyclo-pi-range-bad");
        let res = verify_range(&statement, &tampered, fe(0), &mut vt2);
        assert!(matches!(
            res,
            Err(CycloProtocolError::Sumcheck(
                fq2_sumcheck::Fq2SumcheckError::RoundCheckFailed { .. }
            )) | Err(CycloProtocolError::ReconstructionFailed)
        ));
        // Tampered digit claim: the layer decomposition breaks.
        let mut tampered_d = proof;
        tampered_d.digit_claims[0] = tampered_d.digit_claims[0].add(&fe(1));
        let mut vt3 = Transcript::new_default(b"cyclo-pi-range-bad");
        assert!(matches!(
            verify_range(&statement, &tampered_d, fe(0), &mut vt3),
            Err(CycloProtocolError::ReconstructionFailed)
        ));
    }

    #[test]
    fn pi_range_tampered_round_rejected() {
        let layers = vec![vec![1i64, 0, -1, 2], vec![0i64, 1, 0, -2]];
        let statement = PiRangeStatement {
            num_coeffs: 4,
            num_layers: 2,
            chunk_log: 2,
            bound: 2,
        };
        let mut t = Transcript::new_default(b"cyclo-pi-range-tamper");
        let mut proof = prove_range(&statement, &layers, &mut t).ok().unwrap();
        if let Some(r0) = proof.rounds.first_mut() {
            if let Some(e0) = r0.first_mut() {
                *e0 = e0.add(&fe(1));
            }
        }
        let mut vt = Transcript::new_default(b"cyclo-pi-range-tamper");
        assert!(verify_range(&statement, &proof, fe(0), &mut vt).is_err());
    }

    #[test]
    fn pi_range_identity_all_zero() {
        // The all-zero witness: every digit 0 ∈ [−b, b]; the sumcheck is
        // the zero polynomial; leaf and reconstruction are zero.
        let layers = vec![vec![0i64; 8], vec![0i64; 8]];
        let statement = PiRangeStatement {
            num_coeffs: 8,
            num_layers: 2,
            chunk_log: 2,
            bound: 2,
        };
        let mut t = Transcript::new_default(b"cyclo-pi-range-zero");
        let proof = prove_range(&statement, &layers, &mut t).ok().unwrap();
        assert!(proof.leaf.is_zero());
        let mut vt = Transcript::new_default(b"cyclo-pi-range-zero");
        assert!(verify_range(&statement, &proof, Fq2::ZERO, &mut vt).is_ok());
    }

    #[test]
    fn pi_range_degree_law() {
        // The round polynomials of an honest proof carry exactly
        // 2b+3 values (degree 2b+2) — Thm 1's communication shape.
        let layers = vec![vec![1i64, 0, -1, 0], vec![0i64, 1, 0, -1]];
        let statement = PiRangeStatement {
            num_coeffs: 4,
            num_layers: 2,
            chunk_log: 2,
            bound: 3,
        };
        let mut t = Transcript::new_default(b"cyclo-pi-range-deg");
        let proof = prove_range(&statement, &layers, &mut t).ok().unwrap();
        for (r, round) in proof.rounds.iter().enumerate() {
            assert_eq!(round.len(), 2 * 3 + 3, "round {r}");
        }
        // Certified coefficient bound: b·(2^{k·s}−1)/(2^s−1) with b=2,
        // k=2, s=2 → 2·(1+4) = 10.
        let st2 = PiRangeStatement {
            num_coeffs: 4,
            num_layers: 2,
            chunk_log: 2,
            bound: 2,
        };
        assert_eq!(st2.certified_coefficient_bound(), 10);
    }

    #[test]
    fn ext_rok_happy_path_and_identity() {
        let (pk, ring) = setup(4, 4);
        let (pk_ext, _) = setup(4, 64);
        let w = lattice_commitment::ajtai::sample_small_secret(&ring, 4, 1 << 8, b"ext-rok");
        let t_input = pk.commit(&w).ok().unwrap();
        let spec = challenge_spec(pk.params.k);
        let proof = ext_commit_rok(&pk, &pk_ext, &t_input, &w, 4, 3, &spec)
            .ok()
            .unwrap();
        // The chunked opening (padded to pk_ext.m), digit-major.
        let num_chunks = ((32 + 4 - 1) / 4) as usize;
        let mut v: Vec<RingElement> = vec![ring.zero(); w.len() * num_chunks];
        for (j, e) in w.iter().enumerate() {
            for (i, c) in crate::cyclo::chunk_element(&ring, e, 4)
                .into_iter()
                .enumerate()
            {
                v[i * w.len() + j] = c;
            }
        }
        while v.len() < pk_ext.params.m {
            v.push(ring.zero());
        }
        // Fold identity: the constraint rows hold (v recomposes w under A).
        assert!(verify_ext_fold(&pk, &pk_ext, &t_input, &proof, &v, 4, 3, &spec).is_ok());
        // Recomposition identity (the paper's Step-1 split law): Σ (2b)^i
        // w_i = w — pinned via unchunk_elements.
        for (j, e) in w.iter().enumerate() {
            let chunks: Vec<RingElement> = (0..num_chunks)
                .map(|i| v[i * w.len() + j].clone())
                .collect();
            let rec = crate::cyclo::unchunk_elements(&ring, &chunks, 4);
            assert_eq!(rec, *e);
        }
    }

    #[test]
    fn ext_rok_tampered_opening_rejected() {
        let (pk, ring) = setup(4, 4);
        let (pk_ext, _) = setup(4, 64);
        let w = lattice_commitment::ajtai::sample_small_secret(&ring, 4, 1 << 8, b"ext-tamper");
        let t_input = pk.commit(&w).ok().unwrap();
        let spec = challenge_spec(pk.params.k);
        let proof = ext_commit_rok(&pk, &pk_ext, &t_input, &w, 4, 0, &spec)
            .ok()
            .unwrap();
        let num_chunks = ((32 + 4 - 1) / 4) as usize;
        let mut v: Vec<RingElement> = vec![ring.zero(); w.len() * num_chunks];
        for (j, e) in w.iter().enumerate() {
            for (i, c) in crate::cyclo::chunk_element(&ring, e, 4)
                .into_iter()
                .enumerate()
            {
                v[i * w.len() + j] = c;
            }
        }
        while v.len() < pk_ext.params.m {
            v.push(ring.zero());
        }
        // Tamper one digit: the constraint row (and the opening) breaks.
        let mut bad = v.clone();
        let mut coeffs = bad[0].coeffs().to_vec();
        coeffs[1] = (coeffs[1] + 1) % ring.modulus.q;
        bad[0] = RingElement::from_coeffs(&ring, coeffs);
        assert!(matches!(
            verify_ext_fold(&pk, &pk_ext, &t_input, &proof, &bad, 4, 0, &spec),
            Err(CycloProtocolError::Ajtai(_)) | Err(CycloProtocolError::ConstraintRowFailed)
        ));
        // Oversized digit (norm gate): craft an element with a huge
        // coefficient — the hard gate refuses.
        let mut huge = v.clone();
        let mut coeffs = vec![0u32; ring.n()];
        coeffs[0] = 1 << 20;
        huge[1] = RingElement::from_coeffs(&ring, coeffs);
        assert!(matches!(
            verify_ext_fold(&pk, &pk_ext, &t_input, &proof, &huge, 4, 0, &spec),
            Err(CycloProtocolError::Ajtai(_))
                | Err(CycloProtocolError::NormGateExceeded { bound: 8, .. })
        ));
        // Wrong fold counter desyncs the FS challenge → the challenge
        // replay fails (constraint-row error).
        assert!(matches!(
            verify_ext_fold(&pk, &pk_ext, &t_input, &proof, &v, 4, 1, &spec),
            Err(CycloProtocolError::ConstraintRowFailed)
        ));
        // Tampered input commitment: neither the challenge nor the
        // constraint rows replay.
        let w2 = lattice_commitment::ajtai::sample_small_secret(&ring, 4, 1 << 8, b"ext-other");
        let t_other = pk.commit(&w2).ok().unwrap();
        assert!(matches!(
            verify_ext_fold(&pk, &pk_ext, &t_other, &proof, &v, 4, 0, &spec),
            Err(CycloProtocolError::ConstraintRowFailed)
        ));
    }

    #[test]
    fn ext_rok_challenge_binds_input_commitment() {
        // The FS challenge is a function of the INPUT commitment: a
        // different input commitment yields a different challenge —
        // grinding w.r.t. the input is hard (the FS hole, closed
        // structurally in Π^ext).
        let (pk, ring) = setup(4, 4);
        let (pk_ext, _) = setup(4, 64);
        let w1 = lattice_commitment::ajtai::sample_small_secret(&ring, 4, 64, b"cb-1");
        let w2 = lattice_commitment::ajtai::sample_small_secret(&ring, 4, 64, b"cb-2");
        let t1 = pk.commit(&w1).ok().unwrap();
        let t2 = pk.commit(&w2).ok().unwrap();
        let spec = challenge_spec(pk.params.k);
        let p1 = ext_commit_rok(&pk, &pk_ext, &t1, &w1, 4, 0, &spec)
            .ok()
            .unwrap();
        let p2 = ext_commit_rok(&pk, &pk_ext, &t2, &w2, 4, 0, &spec)
            .ok()
            .unwrap();
        assert_ne!(p1.challenge, p2.challenge);
        // Deterministic replay.
        let p1b = ext_commit_rok(&pk, &pk_ext, &t1, &w1, 4, 0, &spec)
            .ok()
            .unwrap();
        assert_eq!(p1.challenge, p1b.challenge);
        assert_eq!(p1.ext_commitment.rows, p1b.ext_commitment.rows);
        assert_eq!(p1.batched_rhs, p1b.batched_rhs);
    }

    #[test]
    fn ext_rok_shape_errors() {
        let (pk, ring) = setup(4, 4);
        let (pk_ext, _) = setup(4, 64);
        let w = lattice_commitment::ajtai::sample_small_secret(&ring, 4, 64, b"ext-shape");
        let t_input = pk.commit(&w).ok().unwrap();
        let spec = challenge_spec(pk.params.k);
        // Witness length mismatch (the post-refresh truncation guard).
        let short = &w[..3];
        assert!(matches!(
            ext_commit_rok(&pk, &pk_ext, &t_input, short, 4, 0, &spec),
            Err(CycloProtocolError::ShapeMismatch {
                expected: 4,
                got: 3
            })
        ));
        // Extension key too small for the chunked witness.
        let (pk_small, _) = setup(4, 4);
        assert!(matches!(
            ext_commit_rok(&pk, &pk_small, &t_input, &w, 4, 0, &spec),
            Err(CycloProtocolError::ShapeMismatch { .. })
        ));
    }
}
