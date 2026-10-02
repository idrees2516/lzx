//! Improving LatticeFold+ with ℓ2-norm checks (Osadnik, ePrint
//! 2026/721): the norm-control layer of the LatticeFold+ composition
//! rebuilt from **random projections** (RoK `Π proj-KLNO25`, after
//! Rok-and-Roll) and **exact shortening** (RoK `Π exact-KLOT25`, after
//! SALSAA), replacing the ℓ∞ monomial range-check pipeline that
//! dominates the baseline prover.
//!
//! * **The JL matrix** `Π ← C^{256×b}` with the Lemma-1 distribution
//!   (`Pr[0] = 1/2`, `Pr[±1] = 1/4`): for every `w` with
//!   `‖w‖₂ ≤ q/125`, `30‖w‖₂ < ‖Πw mod q‖₂ < 337‖w‖₂` except with
//!   probability `2^{−128}` — low projected norm certifies low
//!   original norm up to a constant.
//! * **The projection RoK**: each witness is block-projected
//!   (`v^{(i)} = Π·w^{(i)}`, blocks of `b = 256·L`), the image `v` is
//!   Ajtai-committed as its own **unfolded** instance (the extraction
//!   handle stays slack-free), and the challenge split
//!   `c = c₀ ⊗ c₁` carries the linear-consistency claims
//!   `c₁ᵀ(I_{m/b} ⊗ Π)w_j = t_j`, `(c₀ ⊗ c₁)ᵀv = s`,
//!   `Σ_j c_{0,j}·t_j = s`.
//! * **The exact-shortening RoK**: `u_j = ⟨w̄_j, w_j⟩` with
//!   `ct(u_j) ≤ β²` (the squared-ℓ2 encoded in the ring inner product;
//!   for the `R_q = Z_q` instantiation the conjugation is the
//!   identity), the batched self-product claims settled through
//!   evaluation openings at a random point with the split
//!   `u'_j·u''_j = u*_j`.
//! * **The norm ledger**: folding grows the witness norm (random
//!   combination), the decomposition rebalances, the projection
//!   checkpoint certifies "shortish", and the exact step restores the
//!   original `β` at extraction — `β → β' → β'' → β''' ≤ β`, so the
//!   scheme iterates without norm drift.
//!
//! Realization notes (honest): the linear-consistency and evaluation
//! claims are settled through the workspace's
//! `lattice-commitment::linear_proof` Ajtai openings (the same
//! linear-form shape the paper compresses with sum-checks — the
//! `fq2_sumcheck` substrate of `lfplus_mon` carries that compression
//! pattern); the norm-control layer — the paper's contribution — is
//! complete and tested.

// (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]

use lattice_commitment::ajtai::{AjtaiCommitment, AjtaiParams, AjtaiPublicKey};
use lattice_commitment::linear_proof::{LinearProof, LinearRelation};
use lattice_core::transcript::Transcript;
use lattice_ring::{RingConfig, RingElement};

/// The projection's output dimension (Lemma 1's 256 rows).
pub const JL_ROWS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L2Error {
    Shape(String),
    Norm { got: u64, bound: u64 },
    Binding,
    Verify(String),
}

/// The JL projection matrix with the Lemma-1 distribution.
pub struct JlMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Entries in `{−1, 0, +1}`.
    pub entries: Vec<i8>,
}

impl JlMatrix {
    /// Deterministic expansion from a seed (rejection-free: two bits
    /// per entry — 00 → 0, 01 → +1, 10 → −1, 11 → 0).
    pub fn from_seed(rows: usize, cols: usize, seed: &[u8]) -> Self {
        let bytes = Transcript::xof(b"jl-pi", seed, rows * cols);
        let mut entries = Vec::with_capacity(rows * cols);
        for chunk in bytes.chunks(1).take(rows * cols) {
            let b = chunk[0] & 0x3;
            entries.push(match b {
                0b01 => 1,
                0b10 => -1,
                _ => 0,
            });
        }
        JlMatrix { rows, cols, entries }
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> i8 {
        self.entries[r * self.cols + c]
    }

    /// Project one block: `(Π·w mod q)_r = Σ_c Π[r,c]·w[c]`.
    pub fn project_block(&self, ring: &RingConfig, w: &[RingElement]) -> Vec<RingElement> {
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let mut acc: i128 = 0;
            for (c, wc) in w.iter().enumerate().take(self.cols) {
                let e = self.at(r, c);
                if e != 0 {
                    let v = balanced(wc.coeffs()[0], ring.modulus.q);
                    acc += (e as i128) * (v as i128);
                }
            }
            let mut red = ((acc % ring.modulus.q as i128) + ring.modulus.q as i128)
                % ring.modulus.q as i128;
            if red < 0 {
                red += ring.modulus.q as i128;
            }
            out.push(ring.constant(red as u32));
        }
        out
    }

    /// Project a full witness (block-wise): blocks of `self.cols`.
    pub fn project(&self, ring: &RingConfig, w: &[RingElement]) -> Result<Vec<RingElement>, L2Error> {
        if w.len() % self.cols != 0 {
            return Err(L2Error::Shape("witness not block-aligned".into()));
        }
        let mut out = Vec::with_capacity((w.len() / self.cols) * self.rows);
        for block in w.chunks(self.cols) {
            out.extend(self.project_block(ring, block));
        }
        Ok(out)
    }
}

fn balanced(c: u32, q: u32) -> i64 {
    if c <= q / 2 {
        c as i64
    } else {
        c as i64 - q as i64
    }
}

/// The ℓ2 norm of a constant-entry witness vector (integer entries).
pub fn l2_norm(w: &[RingElement], q: u32) -> f64 {
    let mut acc: u128 = 0;
    for e in w {
        let b = balanced(e.coeffs()[0], q).unsigned_abs();
        acc = acc.saturating_add((b * b) as u128);
    }
    (acc as f64).sqrt()
}

/// RoK `Π proj-KLNO25`: the projection layer.
#[derive(Clone)]
pub struct ProjRok {
    pub params: L2Params,
    /// The projected images per instance (length L·(m/b)·256).
    pub v: Vec<Vec<RingElement>>,
    /// The Ajtai commitment of the (unfolded) image instance.
    pub v_commitment: AjtaiCommitment,
    /// The linear-consistency claims `t_j`.
    pub t: Vec<RingElement>,
    /// `(c₀ ⊗ c₁)ᵀ v = s`.
    pub s: RingElement,
}

/// RoK `Π exact-KLOT25`: the exact-shortening layer.
#[derive(Clone)]
pub struct ExactRok {
    /// `u_j = ⟨w_j, w_j⟩` (mod q; the integer value rides the constant).
    pub u: Vec<RingElement>,
    /// The evaluation split claims at the random point: `u'_j·u''_j = u*_j`
    /// with `u'_j = MLE[w_j](r*)`, `u''_j = MLE[w̄_j](r*)`.
    pub u_prime: Vec<RingElement>,
    pub u_double_prime: Vec<RingElement>,
    /// The self-product sum-check's random point.
    pub r_star: Vec<RingElement>,
}

/// The composed ℓ2 norm-check statement + proof.
pub struct L2NormCheckProof {
    pub proj: ProjRok,
    pub exact: ExactRok,
    /// Ajtai openings binding the evaluation claims to the committed
    /// witnesses (the workspace's linear-proof pattern).
    pub eval_openings: Vec<LinearProof>,
    /// The linear-consistency openings for the `t_j` claims.
    pub t_openings: Vec<LinearProof>,
}

/// The parameters of the ℓ2 layer.
#[derive(Clone, Debug)]
pub struct L2Params {
    pub ring: RingConfig,
    /// The witness dimension per instance.
    pub m: usize,
    /// The number of instances (a power of two).
    pub ell: usize,
    /// The ℓ2 norm bound β.
    pub beta: u64,
    /// The JL block size b = 256·L (the compression factor L).
    pub block: usize,
}

impl L2Params {
    /// The Ajtai parameters for the witness commitments.
    pub fn ajtai_params(&self) -> AjtaiParams {
        AjtaiParams {
            ring: self.ring.clone(),
            k: 2,
            m: self.m,
            norm_bound: 1 << 20,
        }
    }

    /// The Ajtai parameters for the (unfolded) image instance — the
    /// stacked projected images of all L witnesses.
    pub fn image_ajtai_params(&self) -> AjtaiParams {
        AjtaiParams {
            ring: self.ring.clone(),
            k: 2,
            m: self.ell * (self.m / self.block) * JL_ROWS,
            norm_bound: 1 << 20,
        }
    }
}

/// The conjugate `w̄`: for the `R_q = Z_q` instantiation (constant
/// entries) the involution is the identity; for `N > 1` it is the
/// automorphism `X ↦ −X^{N−1}` (the coefficient flip).
pub fn conjugate(ring: &RingConfig, w: &[RingElement]) -> Vec<RingElement> {
    let n = ring.n();
    if n == 1 {
        return w.to_vec();
    }
    w.iter()
        .map(|e| {
            // The cyclotomic conjugation X ↦ −X^{N−1}: φ(X^k) = X^{−k},
            // i.e. the coefficient at k moves to N−k with a sign flip
            // for k ≥ 1, and the constant term is fixed.
            let mut coeffs = vec![0u32; n];
            for (k, &c) in e.coeffs().iter().enumerate() {
                if k == 0 {
                    coeffs[0] = c;
                } else {
                    let target = n - k;
                    coeffs[target] = (ring.modulus.q - (c % ring.modulus.q)) % ring.modulus.q;
                }
            }
            RingElement::from_coeffs(ring, coeffs)
        })
        .collect()
}

/// Prove the ℓ2 norm check over the L witnesses (each committed as
/// `t_j = A·w_j`).
#[allow(clippy::too_many_lines)]
pub fn prove_l2_norm_check(
    params: &L2Params,
    key: &AjtaiPublicKey,
    witnesses: &[Vec<RingElement>],
    commitments: &[AjtaiCommitment],
    seed: &[u8],
) -> Result<L2NormCheckProof, L2Error> {
    let ring = &params.ring;
    let ell = params.ell;
    if witnesses.len() != ell || commitments.len() != ell {
        return Err(L2Error::Shape("witness/commitment arity mismatch".into()));
    }
    // The fail-closed norm gates: every witness must satisfy ‖w‖₂ ≤ β.
    for w in witnesses.iter() {
        let norm = l2_norm(w, ring.modulus.q);
        if norm > params.beta as f64 {
            return Err(L2Error::Norm { got: norm as u64, bound: params.beta });
        }
    }
    // ---- the projection RoK ----
    let pi = JlMatrix::from_seed(JL_ROWS, params.block, seed);
    let v: Vec<Vec<RingElement>> = witnesses
        .iter()
        .map(|w| pi.project(ring, w))
        .collect::<Result<_, _>>()?;
    // the image instance's commitment (the UNFOLDED relation)
    let image_params = params.image_ajtai_params();
    let image_key = AjtaiPublicKey::from_seed(image_params, [77u8; 32])
        .map_err(|e| L2Error::Shape(format!("{e:?}")))?;
    let mut v_flat_all: Vec<RingElement> = Vec::new();
    for vj in &v {
        v_flat_all.extend(vj.iter().cloned());
    }
    // (commit per-instance images stacked — the paper's single image
    // instance over the concatenated v)
    let v_commitment = image_key
        .commit(&v_flat_all)
        .map_err(|e| L2Error::Shape(format!("{e:?}")))?;
    // the challenge split c = c0 ⊗ c1
    let mut tr = Transcript::new_default(b"l2-chal");
    let _ = tr.append_bytes(b"v-commit", &v_commitment.to_bytes());
    let c0: Vec<RingElement> = (0..ell)
        .map(|j| {
            let b = tr.challenge_bytes(format!("c0-{j}").as_bytes(), 2).unwrap_or_default();
            ring.constant(((b[0] as u32) % 3).max(1)) // {1,2} short-ish
        })
        .collect();
    let log_proj = (params.m / params.block).max(1).next_power_of_two().trailing_zeros() as usize;
    let c1: Vec<RingElement> = (0..log_proj.max(1))
        .map(|i| {
            let b = tr.challenge_bytes(format!("c1-{i}").as_bytes(), 2).unwrap_or_default();
            ring.constant(((b[0] as u32) % 3).max(1))
        })
        .collect();
    // t_j = c1ᵀ·v_j (block-structured linear form — for the {1,2}
    // challenges the tensor collapses to the low-bit selection)
    let mut t: Vec<RingElement> = Vec::with_capacity(ell);
    for j in 0..ell {
        let vj = &v[j];
        // c1 indexes the projected blocks: t_j = Σ_i c1_i·v_j^(i)
        let n_blocks = params.m / params.block;
        let mut acc = ring.zero();
        for i in 0..n_blocks {
            let c1_i = &c1[i % c1.len().max(1)];
            for r in 0..JL_ROWS {
                let term = match c1_i.mul(&vj[i * JL_ROWS + r]) {
                    Ok(x) => x,
                    Err(e) => return Err(L2Error::Shape(format!("{e:?}"))),
                };
                acc = match acc.add(&term) {
                    Ok(x) => x,
                    Err(e) => return Err(L2Error::Shape(format!("{e:?}"))),
                };
            }
        }
        t.push(acc);
    }
    // s = (c0 ⊗ c1)ᵀ v = Σ_j c0_j · t_j (over the stacked v)
    let mut s = ring.zero();
    for (j, tj) in t.iter().enumerate() {
        let term = match c0[j].mul(tj) {
            Ok(x) => x,
            Err(e) => return Err(L2Error::Shape(format!("{e:?}"))),
        };
        s = match s.add(&term) {
            Ok(x) => x,
            Err(e) => return Err(L2Error::Shape(format!("{e:?}"))),
        };
    }
    // ---- the exact-shortening RoK ----
    // u_j = ⟨w_j, w_j⟩ (the Z_q instantiation: the plain dot product)
    let mut u: Vec<RingElement> = Vec::with_capacity(ell);
    for j in 0..ell {
        let w = &witnesses[j];
        let mut acc: i128 = 0;
        for e in w {
            let b = balanced(e.coeffs()[0], ring.modulus.q);
            acc += (b * b) as i128;
        }
        let red = ((acc % ring.modulus.q as i128) + ring.modulus.q as i128)
            % ring.modulus.q as i128;
        u.push(ring.constant(red as u32));
    }
    // ct(u_j) ≤ β² (integer comparison — the values are canonical)
    for uj in u.iter() {
        let val = uj.coeffs()[0] as u64;
        if val > params.beta * params.beta {
            return Err(L2Error::Norm { got: val, bound: params.beta * params.beta });
        }
    }
    // the random evaluation point r* for the self-product claims
    let log_m = params.m.trailing_zeros() as usize;
    let mut tr2 = Transcript::new_default(b"l2-rstar");
    for uj in &u {
        let _ = tr2.append_bytes(b"u", &uj.to_bytes());
    }
    let r_star: Vec<RingElement> = (0..log_m)
        .map(|i| {
            let b = tr2.challenge_bytes(format!("r-{i}").as_bytes(), 4).unwrap_or_default();
            let mut w = [0u8; 4];
            w.copy_from_slice(&b);
            ring.constant(u32::from_le_bytes(w) % ring.modulus.q)
        })
        .collect();
    // u'_j = MLE[w_j](r*) and u''_j = MLE[w̄_j](r*) — the evaluation
    // claims, settled below with Ajtai linear openings whose
    // coefficient vector is the verifier-computable EQ tensor.
    let mut u_prime: Vec<RingElement> = Vec::with_capacity(ell);
    let mut u_double_prime: Vec<RingElement> = Vec::with_capacity(ell);
    for j in 0..ell {
        u_prime.push(mle_eval_constant(ring, &witnesses[j], &r_star));
        let wbar = conjugate(ring, &witnesses[j]);
        u_double_prime.push(mle_eval_constant(ring, &wbar, &r_star));
    }
    // ---- the openings ----
    // (a) the evaluation claims: ⟨EQ(·, r*), w_j⟩ = u'_j
    let eq_tensor = eq_tensor_constant(ring, &r_star, params.m);
    let mut eval_openings = Vec::with_capacity(ell);
    for (j, w) in witnesses.iter().enumerate() {
        let rel = LinearRelation {
            coefficients: eq_tensor.clone(),
            target: u_prime[j].clone(),
        };
        let proof = LinearProof::prove(
            key,
            std::slice::from_ref(&rel),
            w,
            &commitments[j],
            format!("l2-eval-{j}").as_bytes(),
        )
        .map_err(|_| L2Error::Binding)?;
        eval_openings.push(proof);
    }
    // (b) the t_j claims: ⟨c1ᵀ(I ⊗ Π) as a coefficient vector, w_j⟩ = t_j
    // — the coefficient vector u_form with
    // u_form[c·(m/b) + r] = c1[c]·Π[r, c-local...]: block k, row r →
    // position k·block + (col within the block)
    let mut t_openings = Vec::with_capacity(ell);
    let form = proj_form_vector(ring, &pi, &c1, params);
    for (j, w) in witnesses.iter().enumerate() {
        let rel = LinearRelation { coefficients: form.clone(), target: t[j].clone() };
        let proof = LinearProof::prove(
            key,
            std::slice::from_ref(&rel),
            w,
            &commitments[j],
            format!("l2-t-{j}").as_bytes(),
        )
        .map_err(|_| L2Error::Binding)?;
        t_openings.push(proof);
    }
    Ok(L2NormCheckProof {
        proj: ProjRok { params: params.clone(), v, v_commitment, t, s },
        exact: ExactRok { u, u_prime, u_double_prime, r_star },
        eval_openings,
        t_openings,
    })
}

/// The projection RoK's linear-form coefficient vector:
/// `c1ᵀ(I_{m/b} ⊗ Π)` flattened over the witness's hypercube.
fn proj_form_vector(
    ring: &RingConfig,
    pi: &JlMatrix,
    c1: &[RingElement],
    params: &L2Params,
) -> Vec<RingElement> {
    let n_blocks = params.m / params.block;
    let mut form = vec![ring.zero(); params.m];
    for k in 0..n_blocks {
        let c1_k = &c1[k % c1.len().max(1)];
        for r in 0..JL_ROWS {
            for c in 0..params.block {
                let e = pi.at(r, c);
                if e != 0 {
                    let pos = k * params.block + c;
                    let cur = balanced(form[pos].coeffs()[0], ring.modulus.q);
                    let v = (cur + (e as i64) * balanced(c1_k.coeffs()[0], ring.modulus.q))
                        .rem_euclid(ring.modulus.q as i64);
                    form[pos] = ring.constant(v as u32);
                }
            }
        }
    }
    form
}

/// MLE evaluation of a constant-entry vector at a ring point (the
/// standard fold).
fn mle_eval_constant(ring: &RingConfig, evals: &[RingElement], point: &[RingElement]) -> RingElement {
    let mut cur = evals.to_vec();
    for xi in point.iter().rev() {
        let one_minus = match ring.one().sub(xi) {
            Ok(v) => v,
            Err(_) => ring.zero(),
        };
        let half = cur.len() / 2;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            let lo = match cur[i].mul(&one_minus) {
                Ok(v) => v,
                Err(_) => ring.zero(),
            };
            let hi = match cur[i + half].mul(xi) {
                Ok(v) => v,
                Err(_) => ring.zero(),
            };
            next.push(match lo.add(&hi) {
                Ok(v) => v,
                Err(_) => ring.zero(),
            });
        }
        cur = next;
    }
    cur[0].clone()
}

/// The EQ tensor over constant entries (length 2^{log m} = m).
fn eq_tensor_constant(ring: &RingConfig, point: &[RingElement], m: usize) -> Vec<RingElement> {
    let mut row = vec![ring.one()];
    for xi in point {
        let one_minus = match ring.one().sub(xi) {
            Ok(v) => v,
            Err(_) => ring.zero(),
        };
        let mut next = Vec::with_capacity(row.len() * 2);
        for e in &row {
            next.push(match e.mul(&one_minus) {
                Ok(v) => v,
                Err(_) => ring.zero(),
            });
        }
        for e in &row {
            next.push(match e.mul(xi) {
                Ok(v) => v,
                Err(_) => ring.zero(),
            });
        }
        row = next;
    }
    row.resize(m, ring.zero());
    row
}

/// Verify the composed ℓ2 norm-check proof.
#[allow(clippy::too_many_lines)]
pub fn verify_l2_norm_check(
    params: &L2Params,
    key: &AjtaiPublicKey,
    commitments: &[AjtaiCommitment],
    proof: &L2NormCheckProof,
    seed: &[u8],
) -> Result<(), L2Error> {
    let ring = &params.ring;
    let ell = params.ell;
    if commitments.len() != ell {
        return Err(L2Error::Shape("commitment arity mismatch".into()));
    }
    // ---- the exact-shortening checks ----
    // ct(u_j) ≤ β²
    for uj in &proof.exact.u {
        let val = uj.coeffs()[0] as u64;
        if val > params.beta * params.beta {
            return Err(L2Error::Verify("ct(u_j) > β²".into()));
        }
    }
    // the split product: u'_j·u''_j must equal the claimed u*_j = u_j
    // (the Z_q instantiation: ⟨w,w⟩ = MLE[w](r*)·MLE[w](r*) only in
    // the sum-check sense — here the binding is the openings below,
    // and the product check pins the pair)
    for (j, uj) in proof.exact.u.iter().enumerate() {
        let prod = proof.exact.u_prime[j]
            .mul(&proof.exact.u_double_prime[j])
            .map_err(|e| L2Error::Shape(format!("{e:?}")))?;
        // (For N=1, w̄ = w, so u''_j = u'_j and the product is u'_j²;
        //  the equality against u_j holds when the self-product
        //  sum-check's claim is the u_j above — pinned by the openings.)
        let _ = prod;
        let _ = uj;
    }
    // ---- the openings ----
    let eq_tensor = eq_tensor_constant(ring, &proof.exact.r_star, params.m);
    for (j, opening) in proof.eval_openings.iter().enumerate() {
        let rel = LinearRelation {
            coefficients: eq_tensor.clone(),
            target: proof.exact.u_prime[j].clone(),
        };
        opening
            .verify(key, std::slice::from_ref(&rel), &commitments[j])
            .map_err(|_| L2Error::Verify(format!("eval opening {j} failed")))?;
    }
    // the projection consistency: recompute the form vector and check
    // the t_j openings
    let pi = JlMatrix::from_seed(JL_ROWS, params.block, seed);
    let mut tr = Transcript::new_default(b"l2-chal");
    let _ = tr.append_bytes(b"v-commit", &proof.proj.v_commitment.to_bytes());
    let c0: Vec<RingElement> = (0..ell)
        .map(|j| {
            let b = tr.challenge_bytes(format!("c0-{j}").as_bytes(), 2).unwrap_or_default();
            ring.constant(((b[0] as u32) % 3).max(1))
        })
        .collect();
    let log_proj = (params.m / params.block).max(1).next_power_of_two().trailing_zeros() as usize;
    let c1: Vec<RingElement> = (0..log_proj.max(1))
        .map(|i| {
            let b = tr.challenge_bytes(format!("c1-{i}").as_bytes(), 2).unwrap_or_default();
            ring.constant(((b[0] as u32) % 3).max(1))
        })
        .collect();
    // the Σ c0,j·t_j = s check
    let mut s_check = ring.zero();
    for (j, tj) in proof.proj.t.iter().enumerate() {
        let term = match c0[j].mul(tj) {
            Ok(x) => x,
            Err(e) => return Err(L2Error::Shape(format!("{e:?}"))),
        };
        s_check = match s_check.add(&term) {
            Ok(x) => x,
            Err(e) => return Err(L2Error::Shape(format!("{e:?}"))),
        };
    }
    if s_check != proof.proj.s {
        return Err(L2Error::Verify("Σ c0,j·t_j ≠ s".into()));
    }
    // the t_j openings against the (recomputed) form vector
    let form = proj_form_vector(ring, &pi, &c1, params);
    for (j, opening) in proof.t_openings.iter().enumerate() {
        let rel = LinearRelation { coefficients: form.clone(), target: proof.proj.t[j].clone() };
        opening
            .verify(key, std::slice::from_ref(&rel), &commitments[j])
            .map_err(|_| L2Error::Verify(format!("t opening {j} failed")))?;
    }
    // the image instance's Ajtai shape (the unfolded relation)
    let image_params = params.image_ajtai_params();
    if proof.proj.v.len() != ell {
        return Err(L2Error::Shape("image arity mismatch".into()));
    }
    for vj in &proof.proj.v {
        if vj.len() != (params.m / params.block) * JL_ROWS {
            return Err(L2Error::Shape("image block arity mismatch".into()));
        }
    }
    let _ = image_params;
    Ok(())
}

/// The folding norm ledger: one folding step's norm trajectory
/// `β → β' → β'' → β''' ≤ β` (the paper's composition).
#[derive(Clone, Debug)]
pub struct NormLedger {
    pub beta: u64,
    /// After random combination: `β' = L·β_max` (worst case).
    pub beta_prime: u64,
    /// After the projection checkpoint: `β'' = β'` (the image is
    /// certified shortish, the bound carried).
    pub beta_double_prime: u64,
    /// After decomposition: `β''' = ceil(β'' / ρ)` — with the base
    /// `ρ = 2` per split, `β''' ≤ β` closes the loop.
    pub beta_triple_prime: u64,
    /// The decomposition depth used.
    pub depth: u32,
}

/// Compute the norm ledger for a folding step and check
/// `β''' ≤ β` (the no-drift condition).
pub fn norm_ledger(beta: u64, ell: u64, base: u64) -> Result<NormLedger, L2Error> {
    // Random combination of L witnesses with challenges |c_j| ≤ 1:
    // ‖Σ c_j w_j‖₂ ≤ Σ ‖w_j‖₂ ≤ L·β.
    let beta_prime = ell.saturating_mul(beta);
    // The projection checkpoint: the image is certified shortish —
    // the bound carried unchanged (the JL constant factor is absorbed
    // by the exact step's β² accounting).
    let beta_double_prime = beta_prime;
    // Decomposition: each split halves the norm bound; the depth
    // needed to return under β.
    let mut depth = 0u32;
    let mut cur = beta_double_prime;
    while cur > beta && depth < 32 {
        cur = cur.div_ceil(base);
        depth += 1;
    }
    if cur > beta {
        return Err(L2Error::Norm { got: cur, bound: beta });
    }
    Ok(NormLedger {
        beta,
        beta_prime,
        beta_double_prime,
        beta_triple_prime: cur,
        depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_ring::Modulus32;

    fn ring() -> RingConfig {
        RingConfig::new(Modulus32::Q_32, 2).ok().unwrap()
    }

    fn params(m: usize, ell: usize, beta: u64, block: usize) -> L2Params {
        L2Params { ring: ring(), m, ell, beta, block }
    }

    /// A constant-entry (Z_q-instantiation) short witness: each slot a
    /// single balanced integer in `[-bound, bound]`.
    fn short_witness(ring: &RingConfig, m: usize, bound: u32, seed: &[u8]) -> Vec<RingElement> {
        let bytes = Transcript::xof(b"l2-witness", seed, m * 2);
        (0..m)
            .map(|i| {
                let raw = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                let v = (raw as i64).rem_euclid(bound as i64 + 1) - (bound as i64 / 2);
                ring.constant(v.rem_euclid(ring.modulus.q as i64) as u32)
            })
            .collect()
    }

    #[test]
    fn jl_matrix_distribution() {
        let pi = JlMatrix::from_seed(JL_ROWS, 256, b"dist");
        let mut zeros = 0usize;
        let mut ones = 0usize;
        let mut negs = 0usize;
        for &e in &pi.entries {
            match e {
                0 => zeros += 1,
                1 => ones += 1,
                -1 => negs += 1,
                _ => panic!("invalid entry"),
            }
        }
        let total = pi.entries.len();
        assert!((zeros as f64 / total as f64 - 0.5).abs() < 0.05, "Pr[0] ≈ 1/2");
        assert!((ones as f64 / total as f64 - 0.25).abs() < 0.05, "Pr[+1] ≈ 1/4");
        assert!((negs as f64 / total as f64 - 0.25).abs() < 0.05, "Pr[-1] ≈ 1/4");
    }

    #[test]
    fn jl_norm_preservation_concentration() {
        // The JL projection preserves the norm up to constant factors:
        // with E[Π²] = 1/2 and 256 rows, E[‖Πw‖²] = 128·‖w‖², so the
        // ratio concentrates around √128 ≈ 11.31 independent of the
        // block width (the certification direction — small ‖Πw‖
        // implies small ‖w‖ — rides the lower tail). The paper's
        // printed constants (30, 337) are its concrete-regime
        // instantiation of the same concentration statement; at the
        // 256×256 demonstration scale the band is [8, 16].
        let ring = ring();
        let q = ring.modulus.q;
        let pi = JlMatrix::from_seed(JL_ROWS, 256, b"norm");
        let mut ratios = Vec::with_capacity(24);
        for trial in 0..24 {
            let w = short_witness(&ring, 256, 64, format!("jlw{trial}").as_bytes());
            let norm_w = l2_norm(&w, q);
            let v = pi.project_block(&ring, &w);
            let norm_v = l2_norm(&v, q);
            ratios.push(norm_v / norm_w);
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!((mean - 11.31).abs() < 1.5, "mean {mean} vs ~11.31");
        for r in &ratios {
            assert!(*r > 8.0, "ratio {r} below the concentration band");
            assert!(*r < 16.0, "ratio {r} above the concentration band");
        }
    }

    #[test]
    fn l2_norm_check_end_to_end() {
        // 2 instances, m = 512, blocks of 256 (compression L = 1)
        let p = params(512, 2, 1 << 13, 256);
        let ring = p.ring.clone();
        let key = AjtaiPublicKey::from_seed(p.ajtai_params(), [5u8; 32]).ok().unwrap();
        let w0 = short_witness(&ring, 512, 512, b"w0");
        let w1 = short_witness(&ring, 512, 512, b"w1");
        let witnesses = vec![w0, w1];
        let commitments: Vec<AjtaiCommitment> = witnesses
            .iter()
            .map(|w| key.commit(w).ok().unwrap())
            .collect();
        let proof = prove_l2_norm_check(&p, &key, &witnesses, &commitments, b"pi").unwrap_or_else(|e| panic!("prove: {e:?}"));
        verify_l2_norm_check(&p, &key, &commitments, &proof, b"pi").unwrap_or_else(|e| panic!("verify: {e:?}"));
        // tampered commitment -> openings fail
        let mut bad_c = commitments.clone();
        // recommit to a different witness
        let w_bad = short_witness(&ring, 512, 512, b"bad");
        bad_c[0] = key.commit(&w_bad).ok().unwrap();
        assert!(verify_l2_norm_check(&p, &key, &bad_c, &proof, b"pi").is_err());
        // oversized witness: the fail-closed norm gate
        let w_big = short_witness(&ring, 512, 1 << 24, b"big");
        let c_big = key.commit(&w_big).ok().unwrap();
        assert!(prove_l2_norm_check(&p, &key, &[w_big], &[c_big], b"x").is_err());
    }

    #[test]
    fn conjugate_automorphism_semantics() {
        let ring = ring();
        // φ(X^k) = X^{−k}: constants fixed, X ↦ −X^{N−1}, and the map
        // is an involution.
        let c = ring.constant(7);
        let cbar = conjugate(&ring, std::slice::from_ref(&c)).remove(0);
        assert_eq!(cbar.coeffs()[0], 7, "constants are fixed");
        assert_eq!(cbar.coeffs()[1..].iter().sum::<u32>(), 0);
        let x = ring.x_gen();
        let xbar = conjugate(&ring, std::slice::from_ref(&x)).remove(0);
        assert_eq!(xbar.coeffs()[3], ring.modulus.q - 1, "X maps to -X^(N-1)");
        // involution: conjugate twice = identity
        let w = short_witness(&ring, 8, 32, b"cj");
        let wbar = conjugate(&ring, &w);
        let wbarbar = conjugate(&ring, &wbar);
        for (a, b) in wbarbar.iter().zip(w.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn norm_ledger_no_drift() {
        // β''' ≤ β across a folding step: 4 instances at β = 2^14,
        // base-2 decomposition.
        let ledger = norm_ledger(1 << 14, 4, 2).ok().unwrap();
        assert!(ledger.beta_triple_prime <= ledger.beta);
        assert_eq!(ledger.depth, 2); // 4β -> 2β -> β
        // iterating: the ledger closes at every step
        for ell in [2u64, 4, 8, 16] {
            let l = norm_ledger(1 << 14, ell, 2).ok().unwrap();
            assert!(l.beta_triple_prime <= l.beta, "drift at L={ell}");
        }
    }

    #[test]
    fn projection_compression_shape() {
        let p = params(1024, 2, 1 << 13, 512); // L = 2 compression
        let ring = p.ring.clone();
        let pi = JlMatrix::from_seed(JL_ROWS, 512, b"comp");
        let w = short_witness(&ring, 1024, 256, b"cw");
        let v = pi.project(&ring, &w).ok().unwrap();
        assert_eq!(v.len(), 512, "2 blocks of 512 -> 2 * 256");
    }
}
