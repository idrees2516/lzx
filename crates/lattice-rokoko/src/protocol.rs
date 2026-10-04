//! RoKoko protocol core (Wave 7 item 7.13, components 4+5): the
//! committed-linear relation Ξ^lin_COM, the fold-split, the sumcheckify
//! constraint system, the Π^lin linearisation and the round driver —
//! ported from the lattice-zk-lab reference implementation.
//!
//! * **Ξ^lin_COM** (§4.4): `F_i W = H_i Y_i mod q` with COM-opened
//!   `vec(Y_i)`, left/right linear claims `ℓ_j^T W r_j = t_j`, and the
//!   norm bound (W implicitly committed via the vSIS keys).
//! * **Π^fold-split** (Fig. 4): fold the r columns with challenge c,
//!   gadget-decompose `W·c`, pack (folded witness + all aux data) into ŵ
//!   (decreasing-dimension sort + zero-pad, Lemma 3), re-commit under a
//!   fresh vSIS key, send (com, v) with `v = ⟨ŵ, ŵ̄⟩` the Hermitian
//!   self-inner product; the verifier checks `ct(v) ≤ β̃²` (Remark 3's
//!   power-of-two shortcut: the constant term equals ‖cf(ŵ)‖² exactly).
//! * **sumcheckify** (Fig. 5): every constraint becomes a degree-2
//!   sumcheck claim over the packed vector (with the conjugate packed
//!   vector as the second argument for the norm claim).
//! * **Π^lin** (Fig. 6): batch the k_sc claims with `eq(bin(i), γ)`
//!   combiners, run the (ring-valued) sumcheck, output the two
//!   evaluation rows z0/z1.
//! * **Round loop**: `(Π^fold-split ∘ Π^proj): Ξ^lin → Ξ^sum`, then
//!   `Π^lin: Ξ^sum → Ξ^lin`; the terminal instance is opened directly.
//!
//! Documented deviations (kernel scale, per the Python lab's gap ledger):
//! the NTT-slot / subfield batching `Φ = δ^T ∘ θ_a` is replaced by
//! full-ring challenges (soundness |C| = q^n); Π^proj-f (the fine
//! projection) is not implemented — the round loop runs without
//! projections (a norm-slack/extraction optimisation, not needed for
//! completeness); COM runs at depth 1 in the driver (the recursion is
//! exercised in `com.rs` unit tests).

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement};

use crate::com::{ComError, ComKey, ComOpening};
use lattice_salsa::ring_sc::{
    challenge_ring_elt, conj, mle_eval_ring, norm_conjugate_inner, ring_dot, ring_sc_prove,
    ring_sc_verify, ProductClaim, RingScError, RingScProof,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    Com(ComError),
    RingSc(RingScError),
    Transcript(TranscriptError),
    Ring(lattice_ring::RingError),
    /// The Hermitian self-inner-product gate failed (ct(v) > β̃² or < 0).
    NormGateFailed {
        ct_v: i64,
        beta_tilde_sq: i64,
    },
    /// A constraint failed against the revealed packed witness.
    ConstraintFailed,
    /// Commitment binding failed at the terminal opening.
    BindingFailed,
    /// The claim-batching/derivation replay mismatched (tampered).
    ReplayMismatch,
    Shape {
        expected: usize,
        got: usize,
    },
}

impl From<ComError> for ProtocolError {
    fn from(e: ComError) -> Self {
        ProtocolError::Com(e)
    }
}
impl From<RingScError> for ProtocolError {
    fn from(e: RingScError) -> Self {
        ProtocolError::RingSc(e)
    }
}
impl From<TranscriptError> for ProtocolError {
    fn from(e: TranscriptError) -> Self {
        ProtocolError::Transcript(e)
    }
}
impl From<lattice_ring::RingError> for ProtocolError {
    fn from(e: lattice_ring::RingError) -> Self {
        ProtocolError::Ring(e)
    }
}

// ---------------------------------------------------------------------------
// Packing (§3.6 Lemma 3)
// ---------------------------------------------------------------------------

/// `pack(w_0..w_{k−1})`: blocks sorted by DECREASING dimension,
/// concatenated, zero-padded to a power of two. Records offsets so the
/// sumcheck can address sub-blocks via eq(prefix, ·) selectors.
#[derive(Clone, Debug)]
pub struct Packing {
    pub flat: Vec<RingElement>,
    /// (offset, length, log2 length) per block, in packed order.
    pub blocks: Vec<(usize, usize, usize)>,
    pub total: usize,
}

fn min_pow_two(v: usize) -> usize {
    if v <= 1 {
        1
    } else if v.is_power_of_two() {
        v
    } else {
        1usize << (v.ilog2() + 1)
    }
}

impl Packing {
    pub fn pack(blocks: &[Vec<RingElement>], ring: &RingConfig) -> Packing {
        let mut order: Vec<usize> = (0..blocks.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(blocks[i].len()));
        let mut flat: Vec<RingElement> = Vec::new();
        let mut meta = Vec::with_capacity(blocks.len());
        for i in order {
            let b = &blocks[i];
            let log_len = if b.len() > 1 {
                (b.len() - 1).ilog2() as usize
            } else {
                0
            };
            meta.push((flat.len(), b.len(), log_len));
            flat.extend_from_slice(b);
        }
        let total = min_pow_two(flat.len().max(1));
        flat.resize(total, ring.zero());
        Packing {
            flat,
            blocks: meta,
            total,
        }
    }

    /// `p_i = bin(offset_i / m_i)`: the bit prefix selecting block i
    /// inside the packed hypercube (Lemma 3).
    pub fn prefix(&self, block_index: usize) -> Vec<u32> {
        let (off, m, _) = self.blocks[block_index];
        if m == self.total {
            return Vec::new();
        }
        let shift = (self.total / m).trailing_zeros() as usize;
        let base = off / m;
        (0..shift).rev().map(|b| ((base >> b) & 1) as u32).collect()
    }
}

// ---------------------------------------------------------------------------
// Ξ^lin_COM relation
// ---------------------------------------------------------------------------

/// stmt = ((com_i, F_i, H_i)_i, (ℓ_j, r_j, t_j)_j) + witness.
#[derive(Clone, Debug)]
pub struct LinComInstance {
    /// k_lin matrices, each n_i x m_w.
    pub f: Vec<Vec<Vec<RingElement>>>,
    /// k_lin matrices, each n_i x m_{y,i}.
    pub h: Vec<Vec<Vec<RingElement>>>,
    /// COM outputs for vec(Y_i).
    pub coms: Vec<Vec<RingElement>>,
    /// per-i COM openings (aux data).
    pub aux: Vec<ComOpening>,
    /// k_lr left vectors (length m_w).
    pub ell: Vec<Vec<RingElement>>,
    /// k_lr right vectors (length r).
    pub rr: Vec<Vec<RingElement>>,
    /// k_lr targets.
    pub tt: Vec<RingElement>,
    pub m_w: usize,
    pub r: usize,
    pub beta_w: u64,
    /// witness, m_w x r (list of COLUMNS — witness_columns()).
    pub w_cols: Option<Vec<Vec<RingElement>>>,
    /// per i: the Y_i matrices as columns (m_{y,i} x r).
    pub ys: Option<Vec<Vec<Vec<RingElement>>>>,
}

impl LinComInstance {
    /// All Ξ^lin_COM constraints for the embedded witness (per column).
    pub fn check_honest(&self) -> Result<bool, ProtocolError> {
        let w = self.w_cols.as_ref().ok_or(ProtocolError::Shape {
            expected: 1,
            got: 0,
        })?;
        let ys = self.ys.as_ref().ok_or(ProtocolError::Shape {
            expected: 1,
            got: 0,
        })?;
        for i in 0..self.f.len() {
            for col in 0..self.r {
                let wcol = &w[col];
                let fw = mat_vec(&self.f[i], wcol)?;
                // the col-th column of Y_i (ys[i][k] is row k, length r)
                let ycol: Vec<RingElement> = ys[i].iter().map(|row| row[col].clone()).collect();
                let hy = mat_vec(&self.h[i], &ycol)?;
                if fw != hy {
                    return Ok(false);
                }
            }
        }
        // left/right linear claims against column 0
        for j in 0..self.ell.len() {
            if ring_dot(&self.ell[j], &w[0])? != self.tt[j] {
                return Ok(false);
            }
        }
        // norm bound over the whole witness
        let mut norm_sq: u64 = 0;
        for col in w {
            for e in col {
                norm_sq = norm_sq.saturating_add(e.euclidean_norm_squared());
            }
        }
        if norm_sq
            > self
                .beta_w
                .saturating_mul(self.beta_w)
                .saturating_mul(self.r as u64)
        {
            return Ok(false);
        }
        Ok(true)
    }
}

/// Matrix-vector product over ring matrices (rows x cols).
pub fn mat_vec(
    m: &[Vec<RingElement>],
    v: &[RingElement],
) -> Result<Vec<RingElement>, ProtocolError> {
    let mut out = Vec::with_capacity(m.len());
    for row in m {
        out.push(ring_dot(row, v)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// sumcheckify (Figure 5)
// ---------------------------------------------------------------------------

/// One sumcheckified constraint (Figure 5 / §6.2 worked example).
///
/// Every constraint is a difference/constant of INNER PRODUCTS over the
/// packed vector ŵ, encoded as PRODUCT groups for the ring sumcheck:
/// * `Lindiff`: `⟨a_L, ŵ⟩ − ⟨a_R, ŵ⟩ = 0` (a_R sign-folded);
/// * `Lin`: `⟨a, ŵ⟩ = value` (public constant claim);
/// * `Norm`: `⟨ŵ, conj(ŵ)⟩ = value` (the exact norm check).
///
/// The private slot (ŵ / ŵ-bar) is supplied at prove time; the verifier
/// substitutes the sent z0/z1 at the terminal check.
#[derive(Clone, Debug)]
pub enum ScConstraint {
    Lindiff {
        a_l: Vec<RingElement>,
        a_r: Vec<RingElement>,
    },
    Lin {
        a: Vec<RingElement>,
        value: RingElement,
    },
    Norm {
        value: RingElement,
    },
}

impl ScConstraint {
    /// The prover-side product groups (private slot = ŵ tables).
    fn groups(&self, w_hat: &[RingElement]) -> Result<Vec<ProductClaim>, ProtocolError> {
        let ring = w_hat
            .first()
            .map(|e| e.config().clone())
            .ok_or(ProtocolError::Shape {
                expected: 1,
                got: 0,
            })?;
        match self {
            ScConstraint::Lindiff { a_l, a_r } => {
                let neg_r: Vec<RingElement> = a_r.iter().map(|x| x.neg()).collect();
                Ok(vec![
                    ProductClaim {
                        tables: vec![a_l.clone(), w_hat.to_vec()],
                        value: ring.zero(),
                    },
                    ProductClaim {
                        tables: vec![neg_r, w_hat.to_vec()],
                        value: ring.zero(),
                    },
                ])
            }
            ScConstraint::Lin { a, value } => Ok(vec![ProductClaim {
                tables: vec![a.clone(), w_hat.to_vec()],
                value: value.clone(),
            }]),
            ScConstraint::Norm { value } => {
                let wbar: Vec<RingElement> = w_hat.iter().map(conj).collect();
                Ok(vec![ProductClaim {
                    tables: vec![w_hat.to_vec(), wbar],
                    value: value.clone(),
                }])
            }
        }
    }

    /// The verifier-side groups with the private slot replaced by the
    /// length-1 [z0]/[z1] tables (their MLE at any point is the value).
    fn verifier_groups(&self, z0: &RingElement, z1: &RingElement) -> Vec<Vec<Vec<RingElement>>> {
        match self {
            ScConstraint::Lindiff { a_l, a_r } => {
                let neg_r: Vec<RingElement> = a_r.iter().map(|x| x.neg()).collect();
                vec![
                    vec![a_l.clone(), vec![z0.clone()]],
                    vec![neg_r, vec![z0.clone()]],
                ]
            }
            ScConstraint::Lin { a, .. } => vec![vec![a.clone(), vec![z0.clone()]]],
            ScConstraint::Norm { .. } => vec![vec![vec![z0.clone()], vec![z1.clone()]]],
        }
    }

    /// The public value the constraint contributes to the batched target
    /// (Lindiff contributes zero).
    fn public_value(&self) -> Option<&RingElement> {
        match self {
            ScConstraint::Lindiff { .. } => None,
            ScConstraint::Lin { value, .. } => Some(value),
            ScConstraint::Norm { value } => Some(value),
        }
    }
}

/// `eq(bin(i), γ)` combiners — the paper's claim-batching weights.
fn eq_bin_combiners(gamma: &[u32], k: usize, ring: &RingConfig) -> Vec<RingElement> {
    let log_k = gamma.len();
    let q = ring.modulus.q as i128;
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let mut val: i128 = 1;
        for (b, &g) in gamma.iter().enumerate() {
            let bit = (i >> (log_k - 1 - b)) & 1;
            let factor = if bit == 1 {
                i128::from(g)
            } else {
                1 - i128::from(g)
            };
            val = (val * factor).rem_euclid(q);
        }
        out.push(ring.constant(val as u32));
    }
    out
}

// ---------------------------------------------------------------------------
// Π^fold-split (Figure 4)
// ---------------------------------------------------------------------------

/// The fold-split proof: the new packed commitment + the Hermitian
/// self-inner product value.
#[derive(Clone, Debug)]
pub struct FoldSplitProof {
    pub com: Vec<RingElement>,
    pub v: RingElement,
}

/// The Π^fold-split prover output (proof + the constraint system + the
/// packed witness + the new key matrix for the driver).
pub struct FoldSplitOutput {
    pub proof: FoldSplitProof,
    pub constraints: Vec<ScConstraint>,
    pub w_hat: Vec<RingElement>,
    pub c_chal: Vec<RingElement>,
    /// the fresh vSIS key rows (n0 x total) — public, seed-derived.
    pub f_new: Vec<Vec<RingElement>>,
    pub fu: Vec<RingElement>,
}

/// Π^fold-split prover (Figure 4): fold the columns, gadget-decompose,
/// pack, re-commit, and build the sumcheckified constraint system.
pub fn fold_split_prove(
    inst: &LinComInstance,
    ck: &mut ComKey,
    ring: &RingConfig,
    transcript: &mut Transcript,
    l_prime: usize,
) -> Result<FoldSplitOutput, ProtocolError> {
    let w = inst.w_cols.as_ref().ok_or(ProtocolError::Shape {
        expected: 1,
        got: 0,
    })?;
    let (m_w, r) = (inst.m_w, inst.r);
    transcript.append_bytes(b"rk:fold-split:start", b"")?;
    for cm_i in &inst.coms {
        for x in cm_i {
            transcript.append_bytes(b"rk:com", &x.to_bytes())?;
        }
    }
    let c_chal: Vec<RingElement> = (0..r)
        .map(|i| challenge_ring_elt(transcript, format!("rk:c{}", i).as_bytes(), ring))
        .collect::<Result<Vec<_>, _>>()?;
    // fold: folded[j] = Σ_i W[j][i]·c_i
    let mut folded = vec![ring.zero(); m_w];
    for (col, c) in w.iter().zip(&c_chal) {
        for (j, wj) in col.iter().enumerate() {
            folded[j] = folded[j].add(&wj.mul(c)?)?;
        }
    }
    // gadget-decompose the folded witness
    let w_tilde = crate::com::g_inv_vec(ring, &folded, l_prime);
    // pack: w_tilde + all Y_i blocks (row-major over columns) + aux x blocks
    let mut blocks: Vec<Vec<RingElement>> = vec![w_tilde];
    if let Some(ys) = &inst.ys {
        for y_i in ys {
            // vec(Y_i): row k's r entries, k = 0..m_y (row-major columns)
            let mut blk = Vec::with_capacity(y_i.len() * r);
            for row in y_i {
                blk.extend_from_slice(row);
            }
            blocks.push(blk);
        }
    }
    for aux in &inst.aux {
        if let Some(x) = &aux.x {
            blocks.push(x.clone());
        }
    }
    let packing = Packing::pack(&blocks, ring);
    let w_hat = packing.flat.clone();
    let v = norm_conjugate_inner(&w_hat)?;
    // fresh key + commitment (depth-1 COM: the Ajtai output IS com)
    let fu = ck.commit(ring, &w_hat, ck.params.n0)?;
    let f_new = {
        let key = ck.key(ring, ck.params.n0, packing.total)?.clone();
        let mut rows = Vec::with_capacity(ck.params.n0);
        for i in 0..ck.params.n0 {
            let mut row = Vec::with_capacity(packing.total);
            for j in 0..packing.total {
                row.push(key.entry(i, j).cloned().ok_or(ProtocolError::Shape {
                    expected: 1,
                    got: 0,
                })?);
            }
            rows.push(row);
        }
        rows
    };
    let constraints = build_constraints(inst, &packing, &c_chal, ring, &f_new, &fu, &v, l_prime);
    Ok(FoldSplitOutput {
        proof: FoldSplitProof { com: fu.clone(), v },
        constraints,
        w_hat,
        c_chal,
        f_new,
        fu,
    })
}

/// The sumcheckified constraint system (Figure 4 step 7 / §6.2 example):
/// folded linear blocks as lin-diff claims (LHS row(x)g_l on the w_tilde
/// block vs RHS gadget-folded selector on the Y blocks), commitment
/// well-formedness rows, and the exact norm claim.
fn build_constraints(
    inst: &LinComInstance,
    packing: &Packing,
    c_chal: &[RingElement],
    ring: &RingConfig,
    f_new: &[Vec<RingElement>],
    fu: &[RingElement],
    v: &RingElement,
    l_prime: usize,
) -> Vec<ScConstraint> {
    let total = packing.total;
    let mut cons: Vec<ScConstraint> = Vec::new();
    // (a) folded linear blocks: <f^j (x) g_{l'}, w_tilde> = <y_j-block, c-folded>
    for i in 0..inst.f.len() {
        for (j, row) in inst.f[i].iter().enumerate() {
            let a_l = row_gadget_on_block(row, l_prime, packing, 0, ring, total);
            let a_r = c_folded_selector(inst, i, j, c_chal, packing, ring, total);
            cons.push(ScConstraint::Lindiff { a_l, a_r });
        }
    }
    // (b) commitment well-formedness: <F_new[j], w_hat> = com_new[j]
    for (j, key_row) in f_new.iter().enumerate() {
        let mut a0 = vec![ring.zero(); total];
        for (k, kv) in key_row.iter().enumerate() {
            if k < total {
                a0[k] = kv.clone();
            }
        }
        cons.push(ScConstraint::Lin {
            a: a0,
            value: fu.get(j).cloned().unwrap_or_else(|| ring.zero()),
        });
    }
    // (c) exact norm: <w_hat, conj(w_hat)> = v
    cons.push(ScConstraint::Norm { value: v.clone() });
    cons
}

/// Public table: the row (x) g_l placed on the w_tilde block of the
/// packed vector (§6.2 worked example — prefix selector folded in).
fn row_gadget_on_block(
    row: &[RingElement],
    l: usize,
    packing: &Packing,
    block_idx: usize,
    ring: &RingConfig,
    total: usize,
) -> Vec<RingElement> {
    let mut out = vec![ring.zero(); total];
    let (off, m, _) = packing.blocks[block_idx];
    for (t, rv) in row.iter().enumerate() {
        for e in 0..l {
            let pos = off + t * l + e;
            if pos < total && pos < off + m {
                out[pos] = rv.scale_i64(((1i128 << e) % i128::from(ring.modulus.q)) as i64);
            }
        }
    }
    out
}

/// RHS public table: the c-weighted (H_i Y_i) row placed on the packed
/// Y-block: `(H_i Y_i)[j, col] = Σ_k H_i[j][k]·Y_i[k][col]`; the folded
/// target is `Σ_col c_col·(H_i Y_i)[j, col]`, so the table places
/// `H_i[j][k]·c_col` at the packed position of `Y_i[k][col]`.
fn c_folded_selector(
    inst: &LinComInstance,
    i: usize,
    j: usize,
    c_chal: &[RingElement],
    packing: &Packing,
    ring: &RingConfig,
    total: usize,
) -> Vec<RingElement> {
    let mut out = vec![ring.zero(); total];
    // the Y_i block follows the w_tilde block
    let y_block = 1 + i;
    if packing.blocks.len() <= y_block {
        return out;
    }
    let (off, m, _) = packing.blocks[y_block];
    let h_row = match inst.h.get(i).and_then(|h| h.get(j)) {
        Some(row) => row,
        None => return out,
    };
    let r = c_chal.len().max(1);
    // packed position of Y_i[k][col] is off + k·r + col (row-major columns)
    for k in 0..(m / r).min(h_row.len()) {
        for col in 0..r.min(c_chal.len()) {
            let pos = off + k * r + col;
            if pos < total && pos < off + m {
                out[pos] = h_row[k].mul(&c_chal[col]).unwrap_or_else(|_| ring.zero());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Π^lin — linearisation (Figure 6)
// ---------------------------------------------------------------------------

/// The Π^lin proof: the sumcheck + the two final openings + gamma.
#[derive(Clone, Debug)]
pub struct LinProof {
    pub sumcheck: RingScProof,
    pub z0: RingElement,
    pub z1: RingElement,
    pub gamma: Vec<u32>,
    pub point: Vec<u32>,
}

/// Π^lin prover (Figure 6): batch all constraints' product groups with
/// eq(bin(i), γ) combiners, run ONE sumcheck, output z0/z1.
pub fn lin_prove(
    w_hat: &[RingElement],
    cons: &[ScConstraint],
    ring: &RingConfig,
    transcript: &mut Transcript,
) -> Result<LinProof, ProtocolError> {
    let k = cons.len();
    let log_k = if k > 1 { (k - 1).ilog2() as usize } else { 0 };
    let gamma: Vec<u32> = (0..log_k)
        .map(|i| {
            lattice_salsa::ring_sc::challenge_zq(
                transcript,
                format!("rk:gamma{}", i).as_bytes(),
                ring.modulus.q,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let comb = eq_bin_combiners(&gamma, k, ring);
    // assemble the groups + per-group combiners
    let mut groups: Vec<ProductClaim> = Vec::new();
    let mut combiners: Vec<RingElement> = Vec::new();
    for (ci, con) in comb.iter().zip(cons) {
        for g in con.groups(w_hat)? {
            groups.push(g);
            combiners.push(ci.clone());
        }
    }
    let sumcheck = ring_sc_prove(ring, &groups, &combiners, transcript)?;
    let point = sumcheck.point.clone();
    let wbar: Vec<RingElement> = w_hat.iter().map(conj).collect();
    let z0 = mle_eval_ring(w_hat, &point)?;
    let z1 = mle_eval_ring(&wbar, &point)?;
    Ok(LinProof {
        sumcheck,
        z0,
        z1,
        gamma,
        point,
    })
}

/// Π^lin verifier: re-derive gamma, check the batched sumcheck rounds and
/// the TERMINAL identity with z0/z1-substituted groups. Returns the last
/// combined claim for the caller's cross-checks.
pub fn lin_verify(
    cons: &[ScConstraint],
    proof: &LinProof,
    ring: &RingConfig,
    transcript: &mut Transcript,
) -> Result<(), ProtocolError> {
    let k = cons.len();
    let log_k = if k > 1 { (k - 1).ilog2() as usize } else { 0 };
    let gamma: Vec<u32> = (0..log_k)
        .map(|i| {
            lattice_salsa::ring_sc::challenge_zq(
                transcript,
                format!("rk:gamma{}", i).as_bytes(),
                ring.modulus.q,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if gamma != proof.gamma {
        return Err(ProtocolError::ReplayMismatch);
    }
    let comb = eq_bin_combiners(&gamma, k, ring);
    // combined target: Σ_i comb_i · (constraint value)
    let mut target = ring.zero();
    for (ci, con) in comb.iter().zip(cons) {
        if let Some(value) = con.public_value() {
            target = target.add(&ci.mul(value)?)?;
        }
    }
    let mu = proof.sumcheck.rounds.len();
    let last = ring_sc_verify(ring, 2, mu, &target, &proof.sumcheck, transcript)?;
    // terminal: Σ_g comb_g · Π MLE[T_g](point) == last, with private
    // slots substituted by [z0]/[z1]
    let mut total = ring.zero();
    let mut gi = 0;
    for (ci, con) in comb.iter().zip(cons) {
        for vg in con.verifier_groups(&proof.z0, &proof.z1) {
            let mut prod = mle_eval_ring(&vg[0], &proof.point)?;
            for t in &vg[1..] {
                prod = prod.mul(&mle_eval_ring(t, &proof.point)?)?;
            }
            total = total.add(&ci.mul(&prod)?)?;
            gi += 1;
        }
    }
    let _ = gi;
    if total != last {
        return Err(ProtocolError::ConstraintFailed);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Round driver: full argument with terminal opening
// ---------------------------------------------------------------------------

/// The full RoKoko proof: fold-split + lin + the terminal opening.
pub struct RoKokoProof {
    pub fold_proof: FoldSplitProof,
    pub lin_proof: LinProof,
    pub constraints: Vec<ScConstraint>,
    /// terminal opening (direct reveal of ŵ).
    pub w_hat: Vec<RingElement>,
    pub f_new: Vec<Vec<RingElement>>,
}

/// One full committed-refinement round (fold-split → sumcheckify → lin) +
/// direct terminal opening of the packed witness. Multi-round shrinking
/// repeats the same flow on the packed vector (structural support at
/// rounds=1; see the gap ledger).
pub fn rokoko_prove(
    inst: &LinComInstance,
    ck: &mut ComKey,
    ring: &RingConfig,
    transcript: &mut Transcript,
    l_prime: usize,
) -> Result<RoKokoProof, ProtocolError> {
    let fs = fold_split_prove(inst, ck, ring, transcript, l_prime)?;
    let lin = lin_prove(&fs.w_hat, &fs.constraints, ring, transcript)?;
    Ok(RoKokoProof {
        fold_proof: fs.proof,
        lin_proof: lin,
        constraints: fs.constraints,
        w_hat: fs.w_hat,
        f_new: fs.f_new,
    })
}

/// The full verifier: replay the fold-split transcript flow, the norm
/// gate, the lin verifier, and the terminal opening checks.
pub fn rokoko_verify(
    inst: &LinComInstance,
    ck: &mut ComKey,
    ring: &RingConfig,
    proof: &RoKokoProof,
    transcript: &mut Transcript,
    l_prime: usize,
) -> Result<(), ProtocolError> {
    let _ = l_prime;
    // ---- fold-split verifier (Figure 4 right): replay the transcript
    // operations so the lin verifier's challenges re-derive identically
    transcript.append_bytes(b"rk:fold-split:start", b"")?;
    for cm_i in &inst.coms {
        for x in cm_i {
            transcript.append_bytes(b"rk:com", &x.to_bytes())?;
        }
    }
    let r = inst.r;
    let _c_chal: Vec<RingElement> = (0..r)
        .map(|i| challenge_ring_elt(transcript, format!("rk:c{}", i).as_bytes(), ring))
        .collect::<Result<Vec<_>, _>>()?;
    // ct(v) <= beta_tilde^2 (Remark 3 power-of-two shortcut)
    let q = ring.modulus.q as i64;
    let ct_v = {
        let c0 = proof.fold_proof.v.coeff(0) as i64;
        if c0 > q / 2 {
            c0 - q
        } else {
            c0
        }
    };
    let beta_tilde =
        i128::from(inst.beta_w) * 4 * i128::from(r as u64) * (1i128 << (l_prime + 2).min(120));
    let beta_tilde_sq = beta_tilde.saturating_mul(beta_tilde);
    if i128::from(ct_v) < 0 || i128::from(ct_v) > beta_tilde_sq {
        return Err(ProtocolError::NormGateFailed {
            ct_v,
            beta_tilde_sq: beta_tilde_sq.min(i64::MAX as i128) as i64,
        });
    }
    // ---- lin verifier: rounds + terminal identity with z0/z1
    lin_verify(&proof.constraints, &proof.lin_proof, ring, transcript)?;
    // ---- terminal opening checks (direct reveal of w_hat):
    let w_hat = &proof.w_hat;
    // (1) commitment binding: F_new · w_hat == com
    let fu = mat_vec(&proof.f_new, w_hat)?;
    if fu != proof.fold_proof.com {
        return Err(ProtocolError::BindingFailed);
    }
    // (2) exact norm: <w_hat, conj(w_hat)> == v
    if norm_conjugate_inner(w_hat)? != proof.fold_proof.v {
        return Err(ProtocolError::BindingFailed);
    }
    // (3) every constraint holds directly against the revealed w_hat
    for con in &proof.constraints {
        match con {
            ScConstraint::Lindiff { a_l, a_r } => {
                if ring_dot(a_l, w_hat)? != ring_dot(a_r, w_hat)? {
                    return Err(ProtocolError::ConstraintFailed);
                }
            }
            ScConstraint::Lin { a, value } => {
                if ring_dot(a, w_hat)? != *value {
                    return Err(ProtocolError::ConstraintFailed);
                }
            }
            ScConstraint::Norm { value } => {
                if norm_conjugate_inner(w_hat)? != *value {
                    return Err(ProtocolError::ConstraintFailed);
                }
            }
        }
    }
    let _ = ck;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::com::RokokoParams;

    fn ring() -> RingConfig {
        lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 3)
            .ok()
            .unwrap()
    }

    fn small_vec(ring: &RingConfig, m: usize, tag: &[u8], span: u32) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"rokoko-proto-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    4 * ring.n(),
                );
                let coeffs: Vec<u32> = bytes
                    .chunks(4)
                    .take(ring.n())
                    .map(|c| {
                        let mut a = [0u8; 4];
                        a.copy_from_slice(&c[..4]);
                        u32::from_le_bytes(a) % (2 * span + 1)
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    fn make_instance(ring: &RingConfig, ck: &mut ComKey) -> LinComInstance {
        let (m_w, r, m_y) = (4, 2, 4);
        // witness columns (small)
        let w_cols: Vec<Vec<RingElement>> = (0..r)
            .map(|col| small_vec(ring, m_w, format!("w{}", col).as_bytes(), 4))
            .collect();
        // F (1 x m_w), H (1 x m_y) single linear relation; force an
        // invertible pivot at H[0][0] = 1 so the honest Y is trivial to
        // derive
        let f = vec![vec![small_vec(ring, m_w, b"f0", 6)]];
        let mut h_rows = vec![small_vec(ring, m_y, b"h0", 6)];
        h_rows[0][0] = ring.one();
        let h = vec![h_rows];
        // Y as ROWS (m_y x r): Y[0][0] = F·w0 (pivot 1), everything else 0
        let ys: Vec<Vec<Vec<RingElement>>> = {
            let fw = mat_vec(&f[0], &w_cols[0]).ok().unwrap();
            let mut rows: Vec<Vec<RingElement>> = Vec::new();
            let mut row0 = vec![ring.zero(); r];
            row0[0] = fw[0].clone();
            rows.push(row0);
            for _ in 1..m_y {
                rows.push(vec![ring.zero(); r]);
            }
            vec![rows]
        };
        // w col 1 = 0 to keep H·Y consistent
        let mut w_cols = w_cols;
        w_cols[1] = vec![ring.zero(); m_w];
        // COM commitments for vec(Y_i)
        let (com, aux) = {
            let y_flat: Vec<RingElement> = ys[0].iter().flatten().cloned().collect();
            crate::com::com_commit(ck, ring, &y_flat, 1, 32)
                .ok()
                .unwrap()
        };
        // left/right linear claims: ℓ^T W r = t with r = (1,0): t = ℓ·w0
        let ell = vec![small_vec(ring, m_w, b"ell", 6)];
        let rr = vec![vec![ring.one(), ring.zero()]];
        let tt = vec![ring_dot(&ell[0], &w_cols[0]).ok().unwrap()];
        LinComInstance {
            f,
            h,
            coms: vec![com],
            aux: vec![aux],
            ell,
            rr,
            tt,
            m_w,
            r,
            beta_w: 64,
            w_cols: Some(w_cols),
            ys: Some(ys),
        }
    }

    #[test]
    fn packing_decreasing_order_and_prefixes() {
        let ring = ring();
        let blocks = vec![
            small_vec(&ring, 4, b"p0", 3),
            small_vec(&ring, 16, b"p1", 3),
            small_vec(&ring, 8, b"p2", 3),
        ];
        let packing = Packing::pack(&blocks, &ring);
        // decreasing-dimension order: 16, 8, 4
        assert_eq!(packing.blocks[0].1, 16);
        assert_eq!(packing.blocks[1].1, 8);
        assert_eq!(packing.blocks[2].1, 4);
        // total padded to a power of two
        assert!(packing.total.is_power_of_two());
        assert_eq!(packing.total, 32);
        // block prefixes select the blocks
        for bi in 0..3 {
            let prefix = packing.prefix(bi);
            let (off, m, _) = packing.blocks[bi];
            if m != packing.total {
                let base = off / m;
                let mut v: usize = 0;
                for &b in &prefix {
                    v = (v << 1) | b as usize;
                }
                assert_eq!(v, base);
            }
        }
    }

    #[test]
    fn rokoko_end_to_end_and_tampered() {
        let ring = ring();
        let params = RokokoParams {
            n_ring: 3,
            n0: 2,
            gadget_len: 32,
            com_depth: 1,
            r: 2,
            beta_w: 64,
        };
        let mut ck = ComKey::new(params, [51u8; 32]);
        let inst = make_instance(&ring, &mut ck);
        assert!(inst.check_honest().ok().unwrap());
        let mut t = Transcript::new_default(b"lzx-rokoko-proto");
        let proof = rokoko_prove(&inst, &mut ck, &ring, &mut t, 32)
            .map_err(|e| panic!("prove: {:?}", e))
            .ok()
            .unwrap();
        let mut vt = Transcript::new_default(b"lzx-rokoko-proto");
        let vres = rokoko_verify(&inst, &mut ck, &ring, &proof, &mut vt, 32);
        assert!(vres.is_ok(), "verify failed: {:?}", vres.err());
        // tampered com fails the binding check
        let mut bad = RoKokoProof {
            fold_proof: proof.fold_proof.clone(),
            lin_proof: proof.lin_proof.clone(),
            constraints: proof.constraints.clone(),
            w_hat: proof.w_hat.clone(),
            f_new: proof.f_new.clone(),
        };
        if !bad.fold_proof.com.is_empty() {
            let mut coeffs = bad.fold_proof.com[0].coeffs().to_vec();
            coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
            bad.fold_proof.com[0] = RingElement::from_coeffs(&ring, coeffs);
        }
        let mut vt2 = Transcript::new_default(b"lzx-rokoko-proto");
        assert!(rokoko_verify(&inst, &mut ck, &ring, &bad, &mut vt2, 32).is_err());
        // tampered w_hat fails the constraint checks
        let mut bad2 = RoKokoProof {
            fold_proof: proof.fold_proof.clone(),
            lin_proof: proof.lin_proof.clone(),
            constraints: proof.constraints.clone(),
            w_hat: proof.w_hat.clone(),
            f_new: proof.f_new.clone(),
        };
        let mut coeffs = bad2.w_hat[0].coeffs().to_vec();
        coeffs[0] = (coeffs[0] + 1) % ring.modulus.q;
        bad2.w_hat[0] = RingElement::from_coeffs(&ring, coeffs);
        let mut vt3 = Transcript::new_default(b"lzx-rokoko-proto");
        assert!(rokoko_verify(&inst, &mut ck, &ring, &bad2, &mut vt3, 32).is_err());
        // tampered lin sumcheck rejected
        let mut bad3 = RoKokoProof {
            fold_proof: proof.fold_proof.clone(),
            lin_proof: proof.lin_proof.clone(),
            constraints: proof.constraints.clone(),
            w_hat: proof.w_hat.clone(),
            f_new: proof.f_new.clone(),
        };
        let r0 = bad3.lin_proof.sumcheck.rounds[0][0].clone();
        bad3.lin_proof.sumcheck.rounds[0][0] = r0.add(&ring.one()).ok().unwrap();
        let mut vt4 = Transcript::new_default(b"lzx-rokoko-proto");
        assert!(rokoko_verify(&inst, &mut ck, &ring, &bad3, &mut vt4, 32).is_err());
    }
}
