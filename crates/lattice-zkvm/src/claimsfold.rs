//! **The Stage-5.2 claims fold (DESIGN_50KB's residual term; SOTA
//! mechanism #4 — "claims 3.5 KB → ~1 KB")**: the GKR-style product-tree
//! fold that replaces the values-only claims list of the compact memory
//! argument.
//!
//! # The problem
//!
//! The batched legs terminate in *factor evaluation claims* — the ledger's
//! values-only list (292–345 Goldilocks at the benchmark shapes, ~3.5 KB
//! on the wire). Every consumer of those values is one of:
//!
//! * a stage's `claimed` vector (one per class-instance: the address and
//!   read-value entries — 18 values, transmitted in the clear), or
//! * a stage-terminal **expect identity** — a known-coefficient polynomial
//!   in the popped values (products up to arity `log_k + 2`), or
//! * the carriers' right-hand side (a linear form).
//!
//! The values are full-width Goldilocks (MLE evaluations at random
//! sumcheck terminals), so no width-aware packing helps: the list must be
//! *folded*.
//!
//! # The construction (the two-level fold)
//!
//! **Level 1 — the deferred expects.** Every expect identity is
//! transcribed into *deferred checks*: monomials
//! `κ_c · Π_t u_{c,t}` over the claim slots, where each leaf holds an
//! **affine form** `u = α·v + β` of one slot value (this absorbs
//! `digit_affine`'s `α = 2ρ−1, β = 1−ρ` and the `inc − INC_OFFSET`
//! shift without any exponential monomial expansion). The verifier draws
//! one `γ_s` per stage; the folded identity is
//!
//! `Σ_c κ_c · Π_t u_{c,t} = TARGET`,
//!
//! with `TARGET = Σ_s γ_s·(final_s − const_s)` fully verifier-computable
//! (the stage finals are bound by the legs' sumchecks; the constant parts
//! of the expects are public).
//!
//! **Level 2 — the product-tree sumchecks.** The leaf values form the
//! level-0 table `U_0` over the `(check, position)` cube; the products
//! build a balanced binary tree of `s = ⌈log₂ max_arity⌉` levels. Layer
//! `j` (over the level-`(s−j)` cube) proves
//!
//! `Σ_n Ẽ_j(n) · U_{s−j−1}(n∥0) · U_{s−j−1}(n∥1) = [the entering claim]`
//!
//! — a degree-3 sumcheck per layer whose factors are the reshaped child
//! tables (variable renaming keeps them multilinear over the layer cube).
//! Layer 0 merges the γ-fold (its `Ẽ_0` is the coefficient MLE `K̃` and
//! its expected total is `TARGET`); consecutive layers chain through
//! μ-combinations of the two child claims each layer's terminal produces.
//!
//! **The leaf binding.** The last layer terminates in two claims about
//! `U_0` at points `P_L = (ρ_c, ρ_g∥0)`, `P_R = (ρ_c, ρ_g∥1)`. By the
//! MLE collapse, `U_0(P) = Σ_{(c,t)} eq(P,(c,t))·(α_{c,t}·v_{slot} +
//! β_{c,t})` — a **linear form in the claim values** with
//! verifier-computable weights. After one final μ-combination the
//! per-slot weights `W_i` and the constant are fixed; the prover
//! transmits the two per-bundle partial sums `S_bits`, `S_vals` (16 B),
//! and the *carriers* — the same grouped sumchecks as the clear-list
//! design — bind `Σ_i W_i·v_i` to the committed bundles with `E(x) =
//! Σ_i W_i·eq(P_i, x)`. The 18 pre-leg values (address / read-value
//! entries) ride the same carriers with fresh `ρ'` weights; their
//! contribution to the right-hand side is verifier-computable since
//! their values are transmitted.
//!
//! # Soundness
//!
//! The γ-, μ- and ρ′-challenges are all drawn after the material they
//! bind; the layer sumchecks are the workspace engine's (round-chain
//! soundness); the leaf linear forms are MLE identities; the carriers and
//! the compact openings are unchanged from the audited clear-list
//! design — the values list is replaced by objects every one of which
//! terminates in those carriers. The claims never appear in the clear.
//!
//! # Size (the benchmark shapes)
//!
//! `s·m + s(s−1)/2` sumcheck rounds at 4 coefficients (8 B) plus 2·s
//! claim field elements: at fib (m = 8, s = 3) ≈ 27 rounds ≈ **0.9 KB**
//! replacing the 2.9–3.5 KB list — the SOTA ledger's mechanism-#4 target.

use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck::{self, SumcheckProof};
use lattice_sumcheck::VirtualPolynomial;

/// A leaf of a deferred product: the affine form `u = α·v + β` of one
/// claim slot's value. Padding leaves use `α = 0, β = 1` (the neutral
/// element) with an arbitrary slot index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldLeaf {
    pub slot: usize,
    pub alpha: Goldilocks,
    pub beta: Goldilocks,
}

/// One deferred monomial: `coeff · Π leaves`.
#[derive(Clone, Debug)]
pub struct DeferredCheck {
    pub coeff: Goldilocks,
    pub leaves: Vec<FoldLeaf>,
}

/// The derived check set + the verifier-computable stage constants.
#[derive(Clone, Debug, Default)]
pub struct DeferredCheckSet {
    /// The monomials (in derivation order).
    pub checks: Vec<DeferredCheck>,
    /// The per-stage constant expect parts, `const_s` (the TARGET
    /// subtracts `γ_s·const_s` per stage).
    pub stage_consts: Vec<Goldilocks>,
    /// The number of stages (the γ arity).
    pub num_stages: usize,
    /// The γ-weighted stage residual the layer-0 sumcheck must total
    /// to: `Σ_s γ_s·(final_s − const_s)` — verifier-computable, and
    /// re-derived identically by the prover.
    pub target: Goldilocks,
}

/// The claims-fold proof: the layer sumchecks + the terminal claims.
#[derive(Clone, Debug)]
pub struct ClaimsFoldProof {
    /// The `s` layer sumchecks (layer 0 = the merged γ-fold/top).
    pub layers: Vec<SumcheckProof>,
    /// The per-layer terminal child claims `(L, R)` — `2s` field
    /// elements; the last layer's pair are the leaf claims.
    pub claims: Vec<[Goldilocks; 2]>,
    /// The per-bundle leaf linear-form values (transmitted, bound by the
    /// split check against the leaf claims and checked by the carriers).
    pub s_bits: Goldilocks,
    pub s_vals: Goldilocks,
    /// The 18 pre-leg claimed-vector entries transmitted in the clear
    /// (the address entries in stage-A pop order, then the read-value
    /// entries in stage-B pop order — both class-major, instance-minor).
    pub addr_claims: Vec<Goldilocks>,
    pub rv_claims: Vec<Goldilocks>,
}

/// The fold's error surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldError {
    Shape(String),
    Sumcheck(lattice_sumcheck::SumcheckError),
    Transcript(lattice_core::transcript::TranscriptError),
    FinalCheck(&'static str),
}

impl From<lattice_sumcheck::SumcheckError> for FoldError {
    fn from(e: lattice_sumcheck::SumcheckError) -> Self {
        FoldError::Sumcheck(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for FoldError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        FoldError::Transcript(e)
    }
}

/// `⌈log₂ x⌉` for `x ≥ 1`.
pub fn ceil_log2(x: usize) -> usize {
    debug_assert!(x >= 1);
    (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// The fold's shape: the check cube's variable count and the tree depth.
pub fn fold_shape(num_checks: usize, max_arity: usize) -> (usize, usize) {
    let m = ceil_log2(num_checks.max(1));
    // At least one tree level: single-leaf checks (arity 1) pad to 2 so
    // the layer machinery is uniform.
    let s = ceil_log2(max_arity.max(2));
    (m, s)
}

/// The per-check leaf list with arity padding applied: each check's
/// leaves truncated/padded to `2^s` entries with neutral leaves.
fn padded_leaves(checks: &[DeferredCheck], s: usize) -> Vec<Vec<FoldLeaf>> {
    let width = 1usize << s;
    checks
        .iter()
        .map(|c| {
            let mut leaves = c.leaves.clone();
            leaves.truncate(width);
            while leaves.len() < width {
                leaves.push(FoldLeaf {
                    slot: 0,
                    alpha: Goldilocks::ZERO,
                    beta: Goldilocks::ONE,
                });
            }
            leaves
        })
        .collect()
}

/// Build the level-0 leaf table over the `(check, position)` cube from
/// the slot values (prover side).
fn leaf_table(
    padded: &[Vec<FoldLeaf>],
    m: usize,
    s: usize,
    values: &[Goldilocks],
) -> Result<DenseMle, FoldError> {
    let size = 1usize << (m + s);
    let mut evals = Vec::with_capacity(size);
    for c in 0..(1usize << m) {
        for t in 0..(1usize << s) {
            let leaf = padded
                .get(c)
                .and_then(|row| row.get(t))
                .cloned()
                .unwrap_or(FoldLeaf {
                    slot: 0,
                    alpha: Goldilocks::ZERO,
                    beta: Goldilocks::ONE,
                });
            let v = values
                .get(leaf.slot)
                .copied()
                .ok_or_else(|| FoldError::Shape(format!("leaf slot {} out of range", leaf.slot)))?;
            evals.push(leaf.alpha.mul(&v).add(&leaf.beta));
        }
    }
    DenseMle::new(evals).map_err(|e| FoldError::Shape(format!("leaf table: {e:?}")))
}

/// The level-`ℓ` product table over the `(check, h)` cube: cell
/// `(c, h)` = the product of `2^{s−ℓ}` leaves under node `h`.
fn level_table(u0: &DenseMle, m: usize, s: usize, level: usize) -> DenseMle {
    // Node axis bits for `level`: `s − level` per check.
    let nodes = 1usize << (s - level);
    let mut evals = Vec::with_capacity((1usize << m) * nodes);
    let width = 1usize << s;
    for c in 0..(1usize << m) {
        for h in 0..nodes {
            // The leaf positions under node h: h·(width/nodes) ..< (h+1)·(width/nodes).
            let stride = width / nodes;
            let mut acc = Goldilocks::ONE;
            for t in (h * stride)..((h + 1) * stride) {
                acc = acc.mul(&u0.evaluations[c * width + t]);
            }
            evals.push(acc);
        }
    }
    DenseMle {
        num_vars: m + (s - level),
        evaluations: evals,
    }
}

/// Reshape a level-`ℓ` table into the left/right child factors over the
/// level-`(ℓ+1)` cube: cell `(c, g)` = the level-`ℓ` value at
/// `(c, g∥bit)`.
fn child_factor(u: &DenseMle, m: usize, s: usize, level: usize, bit: usize) -> DenseMle {
    // `u` lives on the level-`level` cube: (c, h) with h ∈ {0,1}^{s−level}.
    // The level-(level+1) cube: (c, g) with g ∈ {0,1}^{s−level−1}.
    let hvars = s - level;
    let nodes = 1usize << hvars;
    let gnodes = nodes >> 1;
    let mut evals = Vec::with_capacity((1usize << m) * gnodes);
    for c in 0..(1usize << m) {
        for g in 0..gnodes {
            // h = g∥bit (MSB-first bit layout: g occupies the high
            // h-vars, `bit` the last one).
            let h = (g << 1) | bit;
            evals.push(u.evaluations[c * nodes + h]);
        }
    }
    DenseMle {
        num_vars: m + (s - level - 1),
        evaluations: evals,
    }
}

/// The per-layer combined selector `Ẽ` as a dense MLE over the layer's
/// cube: `μ·eq(P_L, ·) + eq(P_R, ·)` (or the coefficient MLE `K̃` for
/// layer 0 — handled by the caller).
fn combined_selector(
    mu: Goldilocks,
    p_l: &[Goldilocks],
    p_r: &[Goldilocks],
) -> Result<DenseMle, FoldError> {
    let l = DenseMle::eq_extension(p_l);
    let r = DenseMle::eq_extension(p_r);
    let evals = l
        .evaluations
        .iter()
        .zip(r.evaluations.iter())
        .map(|(a, b)| mu.mul(a).add(b))
        .collect();
    DenseMle::new(evals).map_err(|e| FoldError::Shape(format!("selector: {e:?}")))
}

/// The leaf-point / weight computation shared by prover and verifier:
/// given the last layer's terminal `(ρ_c, ρ_g)` and the final μ, compute
/// (a) the per-slot combined weight `W_i` and (b) the constant
/// `Σ Ẽ·β` of the leaf linear form.
pub fn leaf_weights(
    padded: &[Vec<FoldLeaf>],
    m: usize,
    s: usize,
    num_slots: usize,
    terminal: &[Goldilocks],
    mu_leaf: Goldilocks,
) -> (Vec<Goldilocks>, Goldilocks) {
    // The two leaf points.
    let (rc, rg) = terminal.split_at(m);
    let mut p_l = rc.to_vec();
    p_l.extend_from_slice(rg);
    p_l.push(Goldilocks::ZERO);
    let mut p_r = rc.to_vec();
    p_r.extend_from_slice(rg);
    p_r.push(Goldilocks::ONE);
    let mut weights = vec![Goldilocks::ZERO; num_slots];
    let mut constant = Goldilocks::ZERO;
    let neutral = FoldLeaf {
        slot: 0,
        alpha: Goldilocks::ZERO,
        beta: Goldilocks::ONE,
    };
    for c in 0..(1usize << m) {
        for t in 0..(1usize << s) {
            // Every cell of the (padded) leaf cube participates in the
            // MLE collapse — the padding cells hold the neutral leaf
            // (α = 0, β = 1) and contribute their selector weight to the
            // constant.
            let leaf = padded
                .get(c)
                .and_then(|row| row.get(t))
                .cloned()
                .unwrap_or(neutral.clone());
            // eq((c,t) as a cube point, P) for both leaf points.
            let cell = idx_point(m + s, c * (1usize << s) + t);
            let e_l = DenseMle::eq_eval(&cell, &p_l).unwrap_or(Goldilocks::ZERO);
            let e_r = DenseMle::eq_eval(&cell, &p_r).unwrap_or(Goldilocks::ZERO);
            let sel = mu_leaf.mul(&e_l).add(&e_r);
            if leaf.alpha != Goldilocks::ZERO {
                weights[leaf.slot] = weights[leaf.slot].add(&sel.mul(&leaf.alpha));
            }
            constant = constant.add(&sel.mul(&leaf.beta));
        }
    }
    (weights, constant)
}

/// Prove the claims fold. `slot_values` are the claim values in slot
/// order; `slot_bits` marks each slot's bundle (true = bits).
#[allow(clippy::too_many_lines)]
pub fn prove_claims_fold(
    set: &DeferredCheckSet,
    slot_values: &[Goldilocks],
    slot_bits: &[bool],
    addr_claims: Vec<Goldilocks>,
    rv_claims: Vec<Goldilocks>,
    transcript: &mut Transcript,
) -> Result<(ClaimsFoldProof, FoldBinding), FoldError> {
    let checks = &set.checks;
    if checks.is_empty() {
        return Err(FoldError::Shape("no deferred checks".into()));
    }
    let max_arity = checks.iter().map(|c| c.leaves.len()).max().unwrap_or(1);
    let (m, s) = fold_shape(checks.len(), max_arity);
    let padded = padded_leaves(checks, s);

    // The coefficient MLE K̃ over the (padded) check cube.
    let mut kappa = vec![Goldilocks::ZERO; 1usize << m];
    for (c, chk) in checks.iter().enumerate() {
        kappa[c] = chk.coeff;
    }
    let kappa_mle = DenseMle {
        num_vars: m,
        evaluations: kappa,
    };

    // The tables.
    let u0 = leaf_table(&padded, m, s, slot_values)?;
    let levels: Vec<DenseMle> = (0..=s).map(|l| level_table(&u0, m, s, l)).collect();

    // ---- Layer 0: the merged top (K̃ · L · R over the check cube) ----
    let l0 = child_factor(&levels[s - 1], m, s, s - 1, 0);
    let r0 = child_factor(&levels[s - 1], m, s, s - 1, 1);
    let mut vp = VirtualPolynomial::new(m);
    let ki = vp
        .add_factor(kappa_mle)
        .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
    let li = vp
        .add_factor(l0)
        .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
    let ri = vp
        .add_factor(r0)
        .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
    vp.add_term(Goldilocks::ONE, vec![ki, li, ri])
        .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
    let out = sumcheck::prove(&vp, set.target, transcript)?;
    // Fail-closed: the γ-folded residual the legs pinned must equal the
    // folded products of the prover's own slot values.
    let mut folded = Goldilocks::ZERO;
    for (c, chk) in checks.iter().enumerate() {
        let mut prod = chk.coeff;
        for leaf in &chk.leaves {
            let v = slot_values
                .get(leaf.slot)
                .copied()
                .ok_or_else(|| FoldError::Shape("slot range".into()))?;
            prod = prod.mul(&leaf.alpha.mul(&v).add(&leaf.beta));
        }
        let _ = c;
        folded = folded.add(&prod);
    }
    if folded != set.target {
        return Err(FoldError::FinalCheck("prover target mismatch"));
    }
    let mut layers = vec![out.proof];
    let mut claims = vec![[out.factor_claims[li], out.factor_claims[ri]]];
    absorb_pair(transcript, &claims[0])?;
    // The entering points for layer 1: the terminal with the child bit.
    let mut terminal = out.challenges.clone();

    // ---- Layers 1 .. s−1 ----
    for j in 1..s {
        let mu = transcript.challenge_field(b"fold-mu")?;
        // The entering points: (terminal_{j-1} ∥ 0/1).
        let mut p_l = terminal.clone();
        p_l.push(Goldilocks::ZERO);
        let mut p_r = terminal.clone();
        p_r.push(Goldilocks::ONE);
        // The layer cube: m + j vars (the level-(s−j) cube).
        let sel = combined_selector(mu, &p_l, &p_r)?;
        let lu = child_factor(&levels[s - j - 1], m, s, s - j - 1, 0);
        let ru = child_factor(&levels[s - j - 1], m, s, s - j - 1, 1);
        let mut vp = VirtualPolynomial::new(m + j);
        let ei = vp
            .add_factor(sel)
            .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
        let li = vp
            .add_factor(lu)
            .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
        let ri = vp
            .add_factor(ru)
            .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
        vp.add_term(Goldilocks::ONE, vec![ei, li, ri])
            .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
        // Expected: μ·L_{j-1} + R_{j-1}.
        let expected = mu.mul(&claims[j - 1][0]).add(&claims[j - 1][1]);
        let out = sumcheck::prove(&vp, expected, transcript)?;
        layers.push(out.proof);
        claims.push([out.factor_claims[li], out.factor_claims[ri]]);
        absorb_pair(transcript, &claims[j])?;
        terminal = out.challenges.clone();
    }

    // ---- The leaf binding ----
    let mu_leaf = transcript.challenge_field(b"fold-mu-leaf")?;
    let (weights, constant) = leaf_weights(&padded, m, s, slot_values.len(), &terminal, mu_leaf);
    let mut s_bits = Goldilocks::ZERO;
    let mut s_vals = Goldilocks::ZERO;
    for (i, w) in weights.iter().enumerate() {
        let v = slot_values
            .get(i)
            .copied()
            .ok_or_else(|| FoldError::Shape("slot range".into()))?;
        let acc = if slot_bits.get(i).copied().unwrap_or(false) {
            &mut s_bits
        } else {
            &mut s_vals
        };
        *acc = acc.add(&w.mul(&v));
    }
    transcript.append_field(b"fold-s-bits", &s_bits)?;
    transcript.append_field(b"fold-s-vals", &s_vals)?;

    Ok((
        ClaimsFoldProof {
            layers,
            claims,
            s_bits,
            s_vals,
            addr_claims,
            rv_claims,
        },
        FoldBinding {
            weights,
            leaf_l: Goldilocks::ZERO,
            leaf_r: Goldilocks::ZERO,
            mu_leaf,
            constant,
        },
    ))
}

/// Absorb one layer's claim pair (before the next layer's challenges).
fn absorb_pair(transcript: &mut Transcript, pair: &[Goldilocks; 2]) -> Result<(), FoldError> {
    transcript.append_field(b"fold-claim-l", &pair[0])?;
    transcript.append_field(b"fold-claim-r", &pair[1])?;
    Ok(())
}

/// The verifier's fold output: the per-slot carrier weights and the
/// per-bundle transmitted partial sums.
#[derive(Clone, Debug)]
pub struct FoldBinding {
    /// The per-slot weights for the carrier (zero for unreferenced
    /// slots — the addr/rv entries get their fresh ρ′ from the carrier).
    pub weights: Vec<Goldilocks>,
    /// The leaf claims (the last layer's pair).
    pub leaf_l: Goldilocks,
    pub leaf_r: Goldilocks,
    pub mu_leaf: Goldilocks,
    /// The verifier-computable constant of the leaf linear form.
    pub constant: Goldilocks,
}

/// Verify the claims fold; returns the carrier binding.
#[allow(clippy::too_many_lines)]
pub fn verify_claims_fold(
    proof: &ClaimsFoldProof,
    set: &DeferredCheckSet,
    num_slots: usize,
    transcript: &mut Transcript,
) -> Result<FoldBinding, FoldError> {
    let checks = &set.checks;
    if checks.is_empty() {
        return Err(FoldError::Shape("no deferred checks".into()));
    }
    let max_arity = checks.iter().map(|c| c.leaves.len()).max().unwrap_or(1);
    let (m, s) = fold_shape(checks.len(), max_arity);
    if proof.layers.len() != s || proof.claims.len() != s {
        return Err(FoldError::Shape("layer count".into()));
    }
    let padded = padded_leaves(checks, s);
    let kappa_mle = {
        let mut kappa = vec![Goldilocks::ZERO; 1usize << m];
        for (c, chk) in checks.iter().enumerate() {
            kappa[c] = chk.coeff;
        }
        DenseMle {
            num_vars: m,
            evaluations: kappa,
        }
    };

    // TARGET is the γ-folded stage residual (the caller computed it
    // identically on both sides — the stage finals bound by the legs'
    // sumchecks minus the public constants).
    let target = set.target;
    let verdict = proof.layers[0].verify(m, 3, target, transcript, None)?;
    // Final check: K̃(ρ)·L·R.
    let k_at = kappa_mle
        .evaluate(&verdict.point)
        .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
    let expect = k_at.mul(&proof.claims[0][0]).mul(&proof.claims[0][1]);
    if verdict.final_claim != expect {
        return Err(FoldError::FinalCheck("fold layer 0"));
    }
    absorb_pair(transcript, &proof.claims[0])?;
    let mut terminal = verdict.point.clone();

    for j in 1..s {
        let mu = transcript.challenge_field(b"fold-mu")?;
        let expected = mu.mul(&proof.claims[j - 1][0]).add(&proof.claims[j - 1][1]);
        let verdict = proof.layers[j].verify(m + j, 3, expected, transcript, None)?;
        // Ẽ(ρ) = μ·eq(P_L, ρ) + eq(P_R, ρ).
        let mut p_l = terminal.clone();
        p_l.push(Goldilocks::ZERO);
        let mut p_r = terminal.clone();
        p_r.push(Goldilocks::ONE);
        let e_l = DenseMle::eq_eval(&p_l, &verdict.point)
            .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
        let e_r = DenseMle::eq_eval(&p_r, &verdict.point)
            .map_err(|e| FoldError::Shape(format!("{e:?}")))?;
        let sel_at = mu.mul(&e_l).add(&e_r);
        let expect = sel_at.mul(&proof.claims[j][0]).mul(&proof.claims[j][1]);
        if verdict.final_claim != expect {
            return Err(FoldError::FinalCheck("fold layer chain"));
        }
        absorb_pair(transcript, &proof.claims[j])?;
        terminal = verdict.point.clone();
    }

    // The leaf binding: weights + constant + the split check.
    let mu_leaf = transcript.challenge_field(b"fold-mu-leaf")?;
    let (weights, constant) = leaf_weights(&padded, m, s, num_slots, &terminal, mu_leaf);
    transcript.append_field(b"fold-s-bits", &proof.s_bits)?;
    transcript.append_field(b"fold-s-vals", &proof.s_vals)?;
    let [leaf_l, leaf_r] = proof.claims[s - 1];
    let combined = mu_leaf.mul(&leaf_l).add(&leaf_r);
    if combined != proof.s_bits.add(&proof.s_vals).add(&constant) {
        return Err(FoldError::FinalCheck("leaf split"));
    }
    Ok(FoldBinding {
        weights,
        leaf_l,
        leaf_r,
        mu_leaf,
        constant,
    })
}

/// Unit-vector over `nbits` vars selecting index `idx` (MSB-first) —
/// matches `ledger::idx_point`.
pub fn idx_point(nbits: usize, idx: usize) -> Vec<Goldilocks> {
    (0..nbits)
        .map(|i| Goldilocks::from_u64(((idx >> (nbits - 1 - i)) & 1) as u64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// A random-ish deferred set: products over random slots.
    fn synthetic_set(
        num_checks: usize,
        max_arity: usize,
        num_slots: usize,
        seed: u64,
    ) -> (DeferredCheckSet, Vec<Goldilocks>) {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut checks = Vec::new();
        for _ in 0..num_checks {
            let arity = 1 + (next() as usize) % max_arity;
            let mut leaves = Vec::new();
            for _ in 0..arity {
                leaves.push(FoldLeaf {
                    slot: (next() as usize) % num_slots,
                    alpha: fe(1 + (next() % 7)),
                    beta: fe(next() % 5),
                });
            }
            checks.push(DeferredCheck {
                coeff: fe(1 + (next() % 11)),
                leaves,
            });
        }
        let values: Vec<Goldilocks> = (0..num_slots).map(|i| fe(100 + i as u64)).collect();
        // The target the "legs" would pin: Σ coeff·Π leaves.
        let mut target = Goldilocks::ZERO;
        for c in &checks {
            let mut prod = c.coeff;
            for l in &c.leaves {
                prod = prod.mul(&l.alpha.mul(&values[l.slot]).add(&l.beta));
            }
            target = target.add(&prod);
        }
        let set = DeferredCheckSet {
            checks,
            stage_consts: Vec::new(),
            num_stages: 1,
            target,
        };
        (set, values)
    }

    #[test]
    fn fold_honest_roundtrip() {
        for &(n, a, slots) in &[(9usize, 6usize, 20usize), (33, 14, 50), (128, 5, 64)] {
            let (set, values) = synthetic_set(n, a, slots, 42 + n as u64);
            let slot_bits: Vec<bool> = (0..slots).map(|i| i % 2 == 0).collect();
            let mut t = Transcript::new_default(b"fold-test");
            let (proof, _binding) =
                prove_claims_fold(&set, &values, &slot_bits, vec![], vec![], &mut t).unwrap();
            let mut t = Transcript::new_default(b"fold-test");
            let binding = verify_claims_fold(&proof, &set, slots, &mut t).unwrap();
            // The weights must reproduce the split sums.
            let mut s_bits = Goldilocks::ZERO;
            let mut s_vals = Goldilocks::ZERO;
            for (i, w) in binding.weights.iter().enumerate() {
                let acc = if slot_bits[i] {
                    &mut s_bits
                } else {
                    &mut s_vals
                };
                *acc = acc.add(&w.mul(&values[i]));
            }
            assert_eq!(s_bits, proof.s_bits, "bits split at n={n}");
            assert_eq!(s_vals, proof.s_vals, "vals split at n={n}");
        }
    }

    #[test]
    fn fold_tamper_fails() {
        let (set, values) = synthetic_set(40, 10, 30, 7);
        let slot_bits: Vec<bool> = (0..30).map(|i| i % 3 == 0).collect();
        let mut t = Transcript::new_default(b"fold-test");
        let (proof, _binding) =
            prove_claims_fold(&set, &values, &slot_bits, vec![], vec![], &mut t).unwrap();

        // A wrong leaf claim: rejected.
        let mut bad = proof.clone();
        let last = bad.claims.len() - 1;
        bad.claims[last][0] = bad.claims[last][0].add(&fe(1));
        let mut t = Transcript::new_default(b"fold-test");
        assert!(verify_claims_fold(&bad, &set, 30, &mut t).is_err());

        // A wrong S value: rejected (the split check).
        let mut bad2 = proof.clone();
        bad2.s_bits = bad2.s_bits.add(&fe(1));
        let mut t = Transcript::new_default(b"fold-test");
        assert!(verify_claims_fold(&bad2, &set, 30, &mut t).is_err());

        // A wrong target (the legs' finals tampered): the prover's own
        // fold fails closed on the inconsistent total.
        let (mut set2, _values2) = synthetic_set(40, 10, 30, 7);
        set2.target = set2.target.add(&fe(1));
        let mut t = Transcript::new_default(b"fold-test");
        assert!(prove_claims_fold(&set2, &_values2, &slot_bits, vec![], vec![], &mut t).is_err());

        // A tampered layer round message: rejected.
        let mut bad3 = proof.clone();
        if let Some(r) = bad3.layers[0].rounds.first_mut() {
            if let Some(x) = r.first_mut() {
                *x = x.add(&fe(1));
            }
        }
        let mut t = Transcript::new_default(b"fold-test");
        assert!(verify_claims_fold(&bad3, &set, 30, &mut t).is_err());
    }

    #[test]
    fn fold_shape_math() {
        assert_eq!(fold_shape(1, 1), (0, 1));
        assert_eq!(fold_shape(9, 6), (4, 3));
        assert_eq!(fold_shape(200, 14), (8, 4));
    }
}
