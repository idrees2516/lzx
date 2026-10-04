//! A3 — Akita digit-range sumcheck + response-norm certification
//! (ePrint 2026/1983, §6): the norm statements that certify the recursive
//! witness before it is used in another fold (Wave 7 item 7.11, A3).
//!
//! * **§6.1 Digit Range Check** — every Boolean evaluation `w(x)` of the
//!   level-`j+1` witness must lie in the balanced alphabet
//!   `A_{b*} = {−b*/2, …, b*/2 − 1}` for the schedule's common range
//!   parameter `b* ∈ {4, 8, 16, 32, 64}`.
//!   * **Degree halving (Eq 114)**: pairing the factors of the vanishing
//!     product `Q_{b*}(w) = Π_a (w − a)` as `(w − k)(w + k + 1) =
//!     w(w + 1) − k(k + 1)` rewrites the degree-`b*` predicate as the
//!     degree-`b*/2` predicate `Q_sq(s)` in the derived value
//!     `s(x) := w(x)·(w(x) + 1)`; both roots `k, −(k+1)` of a pair share
//!     the image `k(k+1)`. The anchored identity (Eq 115)
//!     `Σ_x eq(τ̃₀, x)·Q_sq(s(x)) = 0` is proven by sumchecks reducing to
//!     the single claim `s_claim = s̃(r_virt)`.
//!   * **The product tree (§6.1 shapes)**: `b* ≤ 8` uses ONE sumcheck of
//!     degree `b*/2`; larger `b*` organize the product as a 2-or-4-ary
//!     tree — `(d₀..d_{S−1})/(n₀..n_S)` from the paper's table — proven
//!     level-by-level with the claim-and-prove chain: each level's
//!     batched sumcheck takes the previous level's node evaluation
//!     claims as its initial claim (the multilinear evaluation identity
//!     `ĉ_j(p) = Σ_x eq(p, x)·Π_k c_{j,k}(x)` holds for honest node
//!     data) and outputs the children's claims at a fresh point.
//!   * **Leaf collapse**: the leaves `s(x) − k(k+1)` are affine in `s`,
//!     so the final leaf claims collapse to one claim on `s̃` — the only
//!     claim that leaves the range pipeline.
//!   * **The s-binding sumcheck**: `s̃` is not part of the recursive
//!     witness; `Σ_x eq(r, x)·w̃(x)·(w̃(x) + 1) = s_claim` binds it back
//!     to the committed witness (the kernel realization of the §7
//!     reduction of the `s̃` claim to a `w̃` claim; at commitment scale
//!     this row feeds the fused relation sum-check of A2).
//!   * **Remark 6.2 fused binariness**: compression digits restricted to
//!     `{−1, 0}` get their binariness from the SAME quadratic `w(w+1)`,
//!     fused into ONE sumcheck with the carried `s` claim under the
//!     verifier-sampled coefficients `(γ, ζ_bin)`; the binariness weight
//!     is the *restricted equality* — the MLE of
//!     `x ↦ eq(r_virt, x)·1_{I_bin}(x)` — explicitly NOT the product of
//!     the two MLEs (the paper's warning, realized here and pinned by a
//!     dedicated test).
//! * **§6.2 Certifying the ℓ2 response norm**:
//!   * **Direct route (Eq 118–120, Lemma 6.3)**: when
//!     `max{U_dir, S_max} < q` with `U_dir = N_A·(Σ_h b^h·B_{dig,h})²`
//!     (Eq 119), the field identity `Σ_x z_int(x)² = E_resp` over F_q
//!     determines the integer squared norm exactly; `E_resp` is carried
//!     in a canonical fixed-width encoding gated by `S_max`.
//!   * **Digit-expanded route (Eq 121–123, Lemma 6.4)**: when the full
//!     norm can exceed `q`, the squared norm expands into per-segment
//!     digit-plane inner products `p_{t,h,k}` whose centered lifts are
//!     unique under the segment bound `|I_t|·B_{dig,h}·B_{dig,k} < q/2`
//!     (Eq 122); the verifier reconstructs
//!     `E_resp = Σ_t (Σ_h b^{2h}·p̄_{t,h,h} + 2·Σ_{h<k} b^{h+k}·p̄_{t,h,k})`
//!     with exact integer arithmetic and gates it against `S_max`.
//!
//! LZX realization notes (kernel scale, honestly stated): the §3.5
//! equality-factored message compression (omit the constant coefficient,
//! recover it from the running claim) is realized as full round messages
//! with the eq factor carried as an ordinary sumcheck factor — a
//! wire-format deviation of one coefficient per round that does not
//! change the binding structure; the derived `s` values follow the
//! paper's "multiply at each Boolean index and then interpolate"
//! exactly. All norm integers are computed in `i128`/`u128` and every
//! bound gate (Eq 119/122, `S_max`, canonicality) is enforced fail-closed
//! on BOTH sides.

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck::{self, SumcheckError, SumcheckProof};
use lattice_sumcheck::virtual_poly::{VirtualPolyError, VirtualPolynomial};

/// The Goldilocks modulus `q` (the direct-route gate compares against it).
const GOLDILOCKS_Q: u128 = 0xFFFF_FFFF_0000_0001;

/// The schedule's range/norm parameters (§6.1 + §6.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeParams {
    /// The common range parameter `b* ∈ {4, 8, 16, 32, 64}`.
    pub b_star: u32,
    /// The response digit depth `δ_f` (§6.2).
    pub digit_depth: usize,
    /// The response base `b` (Eq 116).
    pub response_base: u64,
    /// Per-plane digit alphabet ceilings `B_{dig,h}` (Eq 119).
    pub digit_bounds: Vec<u64>,
    /// The schedule's squared-norm ceiling `S_max` (Eq 118).
    pub s_max: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeError {
    Transcript(TranscriptError),
    Sumcheck(SumcheckError),
    /// Virtual-polynomial construction failure (shape mismatch).
    Virtual(VirtualPolyError),
    /// MLE evaluation failure (point/shape mismatch).
    Mle(lattice_core::mle::MleError),
    /// `b*` outside the schedule's alphabet `{4, 8, 16, 32, 64}`.
    BadRangeBase(u32),
    /// A witness entry outside the balanced alphabet (fail-closed at
    /// prove; the anchored identity would be false).
    DigitOutOfRange {
        index: usize,
        value: i64,
    },
    /// A compression-digit cell outside `{−1, 0}` (Remark 6.2).
    BinaryViolated {
        index: usize,
        value: i64,
    },
    /// Leaf claims disagree on `s̃(r)` (tampered leaf claim).
    LeafInconsistent,
    /// The terminal/level identity failed (tampered proof).
    TerminalFailed,
    /// The carried `s`-claim failed the binding sumcheck.
    BindingFailed,
    /// `U_dir ≥ q` — the direct route is inadmissible (Eq 119 gate).
    DirectRouteInadmissible {
        u_dir: u128,
    },
    /// `E_resp` outside `[0, S_max]` or non-canonically encoded.
    NonCanonicalNorm {
        value: u128,
    },
    /// The segment bound `|I_t|·B_{dig,h}·B_{dig,k} < q/2` failed.
    SegmentBoundViolated {
        segment: usize,
        h: usize,
        k: usize,
        bound: u128,
    },
    /// The reconstructed integer norm mismatched the claimed `E_resp`.
    ReconstructionMismatch {
        claimed: u128,
        reconstructed: i128,
    },
    /// Shape mismatch (witness length vs. the declared hypercube).
    Shape {
        expected: usize,
        got: usize,
    },
}

impl From<TranscriptError> for RangeError {
    fn from(e: TranscriptError) -> Self {
        RangeError::Transcript(e)
    }
}

impl From<SumcheckError> for RangeError {
    fn from(e: SumcheckError) -> Self {
        RangeError::Sumcheck(e)
    }
}

impl From<VirtualPolyError> for RangeError {
    fn from(e: VirtualPolyError) -> Self {
        RangeError::Virtual(e)
    }
}

impl From<lattice_core::mle::MleError> for RangeError {
    fn from(e: lattice_core::mle::MleError) -> Self {
        RangeError::Mle(e)
    }
}

/// One product-tree level: node count + product degree (§6.1's table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeLevel {
    /// Product degree `d_i` at this level.
    pub degree: usize,
    /// Node count `n_i` at this level.
    pub nodes: usize,
}

/// The paper's product-tree shapes:
/// `b* : (d₀, …) / (n₀, …)` = `4:(2)/(1,2)`, `8:(4)/(1,4)`,
/// `16:(2,4)/(1,2,8)`, `32:(4,4)/(1,4,16)`, `64:(2,4,4)/(1,2,8,32)`.
pub fn range_tree(b_star: u32) -> Result<Vec<TreeLevel>, RangeError> {
    match b_star {
        4 => Ok(vec![TreeLevel {
            degree: 2,
            nodes: 1,
        }]),
        8 => Ok(vec![TreeLevel {
            degree: 4,
            nodes: 1,
        }]),
        16 => Ok(vec![
            TreeLevel {
                degree: 2,
                nodes: 1,
            },
            TreeLevel {
                degree: 4,
                nodes: 2,
            },
        ]),
        32 => Ok(vec![
            TreeLevel {
                degree: 4,
                nodes: 1,
            },
            TreeLevel {
                degree: 4,
                nodes: 4,
            },
        ]),
        64 => Ok(vec![
            TreeLevel {
                degree: 2,
                nodes: 1,
            },
            TreeLevel {
                degree: 4,
                nodes: 2,
            },
            TreeLevel {
                degree: 4,
                nodes: 8,
            },
        ]),
        other => Err(RangeError::BadRangeBase(other)),
    }
}

/// The involution representatives `U_{b*} = {0, …, b*/2 − 1}` (§6.1).
fn u_set(b_star: u32) -> Vec<u64> {
    (0..u64::from(b_star) / 2).collect()
}

/// The derived Boolean evaluation `s(x) = w(x)·(w(x)+1)` over F_q
/// (§6.1: multiply at each Boolean index, then interpolate).
fn derived_s(w: &[i64]) -> Vec<Goldilocks> {
    w.iter()
        .map(|&v| {
            let s = i128::from(v) * i128::from(v + 1);
            Goldilocks::from_u64(s.rem_euclid(GOLDILOCKS_Q as i128) as u64)
        })
        .collect()
}

/// `Q_{b*}(w) = Q_sq(w·(w+1))` — the degree-halving identity (Eq 114),
/// exact over the integers (pure math; pinned by tests).
pub fn vanishing_product(b_star: u32, w: i64) -> Result<i128, RangeError> {
    range_tree(b_star)?;
    let mut acc: i128 = 1;
    for k in u_set(b_star) {
        let s = w * (w + 1);
        acc *= i128::from(s) - i128::from(k) * i128::from(k + 1);
    }
    Ok(acc)
}

/// Whether `w` lies in the balanced alphabet `A_{b*}`.
pub fn in_alphabet(b_star: u32, w: i64) -> bool {
    let half = i64::from(b_star) / 2;
    (-half..half).contains(&w)
}

fn to_field(v: i64) -> Goldilocks {
    Goldilocks::from_u64(i128::from(v).rem_euclid(GOLDILOCKS_Q as i128) as u64)
}

fn mle_of(values: Vec<Goldilocks>) -> Result<DenseMle, RangeError> {
    let n = values.len();
    DenseMle::new(values).map_err(|_| RangeError::Shape {
        expected: n,
        got: n,
    })
}

/// Build the node-value tree bottom-up from the derived `s` values.
/// Returns `tree[0]` = root (one node), `tree[i]` = level-`i` node values;
/// the LAST entry is the leaf level (`b*/2` nodes).
fn build_tree(b_star: u32, s_vals: &[Goldilocks]) -> Result<Vec<Vec<Vec<Goldilocks>>>, RangeError> {
    let levels = range_tree(b_star)?;
    // Leaves: s(x) − k(k+1) for k ∈ U_{b*}, affine in s.
    let mut current: Vec<Vec<Goldilocks>> = u_set(b_star)
        .into_iter()
        .map(|k| {
            let kk = Goldilocks::from_u64(k * (k + 1));
            s_vals.iter().map(|&s| s.sub(&kk)).collect()
        })
        .collect();
    let mut out: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(levels.len() + 1);
    for level in levels.iter().rev() {
        out.push(current.clone());
        if current.len() != level.nodes * level.degree {
            return Err(RangeError::Shape {
                expected: level.nodes * level.degree,
                got: current.len(),
            });
        }
        let mut parents: Vec<Vec<Goldilocks>> = Vec::with_capacity(level.nodes);
        for chunk in current.chunks(level.degree) {
            let mut pv = Vec::with_capacity(s_vals.len());
            for i in 0..s_vals.len() {
                let mut acc = Goldilocks::ONE;
                for child in chunk {
                    acc = acc.mul(&child[i]);
                }
                pv.push(acc);
            }
            parents.push(pv);
        }
        current = parents;
    }
    out.push(current);
    out.reverse();
    Ok(out)
}

/// One product level's proof: the batched sumcheck + the carried
/// child-node evaluation claims at the level's output point.
#[derive(Clone, Debug)]
pub struct LevelProof {
    pub sumcheck: SumcheckProof,
    /// The children's claimed evaluations at the output point (the
    /// prover's factor claims minus the eq factor, in node order).
    pub child_claims: Vec<Goldilocks>,
}

/// The digit-range proof (§6.1's full pipeline).
#[derive(Clone, Debug)]
pub struct DigitRangeProof {
    /// The per-level batched product sumchecks (level 0 = the anchored
    /// identity with claim 0).
    pub levels: Vec<LevelProof>,
    /// The claimed leaf values at the final point (one per `k ∈ U_{b*}`).
    pub leaf_claims: Vec<Goldilocks>,
    /// The collapsed claim `s_claim = s̃(r_virt)`.
    pub s_claim: Goldilocks,
    /// The (possibly fused) s-binding sumcheck + the output point +
    /// the claimed `ŵ(r')` (which the binding's terminal pins).
    pub binding: SumcheckProof,
    pub binding_point: Vec<Goldilocks>,
    pub w_claim: Goldilocks,
    /// The fused coefficients `(γ, ζ_bin)` (absorbed when spans exist).
    pub fused_coeffs: Option<(Goldilocks, Goldilocks)>,
}

/// Prove the digit-range statement for a witness whose entries are
/// balanced digits in `A_{b*}` and whose compression spans `i_bin`
/// (indices into `w`) hold `{−1, 0}` cells (Remark 6.2). FAIL-CLOSED on
/// both premises before any proof is built.
#[allow(clippy::too_many_lines)]
pub fn prove_digit_range(
    w: &[i64],
    b_star: u32,
    i_bin: &[usize],
    transcript: &mut Transcript,
) -> Result<DigitRangeProof, RangeError> {
    let n = w.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(RangeError::Shape {
            expected: 0,
            got: n,
        });
    }
    let num_vars = n.trailing_zeros() as usize;
    for (i, &v) in w.iter().enumerate() {
        if !in_alphabet(b_star, v) {
            return Err(RangeError::DigitOutOfRange { index: i, value: v });
        }
    }
    for &i in i_bin {
        if i >= n {
            return Err(RangeError::Shape {
                expected: n,
                got: i,
            });
        }
        if !(-1..=0).contains(&w[i]) {
            return Err(RangeError::BinaryViolated {
                index: i,
                value: w[i],
            });
        }
    }

    let s_vals = derived_s(w);
    let tree = build_tree(b_star, &s_vals)?;
    let levels_meta = range_tree(b_star)?;
    let mut levels: Vec<LevelProof> = Vec::with_capacity(levels_meta.len());

    // ---- Level 0: the anchored identity (Eq 115) -------------------------
    // Σ_x eq(τ₀,x)·Π_{j≤d₀} c_j(x) = 0: the root is the product of its
    // children and Q_sq vanishes at every Boolean s-value.
    let tau0: Vec<Goldilocks> = transcript.challenge_fields(b"a3-range-tau0", num_vars)?;
    let out0 = {
        let children = &tree[1];
        let mut vp = VirtualPolynomial::new(num_vars);
        let eq_idx = vp.add_factor(DenseMle::eq_extension(&tau0))?;
        let mut idxs = vec![eq_idx];
        for child in children {
            idxs.push(vp.add_factor(mle_of(child.clone())?)?);
        }
        vp.add_term(Goldilocks::ONE, idxs)?;
        sumcheck::prove(&vp, Goldilocks::ZERO, transcript)?
    };
    levels.push(LevelProof {
        sumcheck: out0.proof,
        child_claims: out0.factor_claims[1..].to_vec(),
    });

    // ---- Levels 1..S−1: claim-and-prove product relations ---------------
    // Σ_x eq(p,x)·Σ_j ν_j·Π_k c_{j,k}(x) = Σ_j ν_j·ĉ_j(p): the honest
    // identity is the multilinear evaluation identity on the parent
    // nodes (whose Boolean values are the children products).
    let mut point = out0.challenges.clone();
    let mut node_claims = levels[0].child_claims.clone();
    for (li, level) in levels_meta.iter().enumerate().skip(1) {
        let children_level = &tree[li + 1];
        if node_claims.len() != level.nodes {
            return Err(RangeError::Shape {
                expected: level.nodes,
                got: node_claims.len(),
            });
        }
        let nus: Vec<Goldilocks> = transcript.challenge_fields(b"a3-range-nu", level.nodes)?;
        let mut claim = Goldilocks::ZERO;
        for (j, nu) in nus.iter().enumerate() {
            claim = claim.add(&nu.mul(&node_claims[j]));
        }
        let mut vp = VirtualPolynomial::new(num_vars);
        let eq_idx = vp.add_factor(DenseMle::eq_extension(&point))?;
        for (j, nu) in nus.iter().enumerate() {
            let chunk = &children_level[j * level.degree..(j + 1) * level.degree];
            let mut idxs = vec![eq_idx];
            for child in chunk {
                idxs.push(vp.add_factor(mle_of(child.clone())?)?);
            }
            vp.add_term(*nu, idxs)?;
        }
        let out = sumcheck::prove(&vp, claim, transcript)?;
        levels.push(LevelProof {
            sumcheck: out.proof,
            child_claims: out.factor_claims[1..].to_vec(),
        });
        point = out.challenges;
        node_claims = levels[li].child_claims.clone();
    }
    let r_virt = point;

    // ---- Leaf collapse ---------------------------------------------------
    let leaf_count = node_claims.len();
    if leaf_count != u_set(b_star).len() {
        return Err(RangeError::Shape {
            expected: u_set(b_star).len(),
            got: leaf_count,
        });
    }
    let s_claim = node_claims[0].add(&Goldilocks::from_u64(0));
    for (k, claim) in node_claims.iter().enumerate() {
        let kk = u64::try_from(k).unwrap_or(u64::MAX);
        if claim.add(&Goldilocks::from_u64(kk * (kk + 1))) != s_claim {
            return Err(RangeError::LeafInconsistent);
        }
    }

    // ---- The (fused) s-binding sumcheck ----------------------------------
    // Plain: Σ_x eq(r,x)·w̃·(w̃+1) = s_claim.
    // Fused (Remark 6.2): + ζ_bin·w̃_bin(x)·w̃·(w̃+1) with the restricted
    // equality weight, under the carried-claim coefficient γ.
    let w_mle = mle_of(w.iter().map(|&v| to_field(v)).collect())?;
    let w_plus_one = mle_of(w.iter().map(|&v| to_field(v + 1)).collect())?;
    let fused_coeffs = if i_bin.is_empty() {
        None
    } else {
        let gamma = transcript.challenge_field(b"a3-range-gamma")?;
        let zeta = transcript.challenge_field(b"a3-range-zetabin")?;
        Some((gamma, zeta))
    };
    let mut claim = s_claim;
    let mut vp = VirtualPolynomial::new(num_vars);
    {
        let eq_idx = vp.add_factor(DenseMle::eq_extension(&r_virt))?;
        let w_idx = vp.add_factor(w_mle.clone())?;
        let wp_idx = vp.add_factor(w_plus_one)?;
        match fused_coeffs {
            Some((gamma, _)) => {
                vp.add_term(gamma, vec![eq_idx, w_idx, wp_idx])?;
            }
            None => {
                vp.add_term(Goldilocks::ONE, vec![eq_idx, w_idx, wp_idx])?;
            }
        }
        if let Some((gamma, zeta)) = fused_coeffs {
            // The restricted-equality weight: the MLE of
            // x ↦ eq(r_virt, x)·1_{I_bin}(x) — NOT eq·1̃ (the paper's
            // warning; see test restricted_eq_is_not_the_product).
            let mut vals = Vec::with_capacity(n);
            let bin_set: std::collections::HashSet<usize> = i_bin.iter().copied().collect();
            for (x, &e) in DenseMle::eq_extension(&r_virt)
                .evaluations
                .iter()
                .enumerate()
            {
                vals.push(if bin_set.contains(&x) {
                    e
                } else {
                    Goldilocks::ZERO
                });
            }
            let ridx = vp.add_factor(mle_of(vals)?)?;
            vp.add_term(zeta, vec![ridx, w_idx, wp_idx])?;
            claim = s_claim.mul(&gamma);
        }
    }
    let out = sumcheck::prove(&vp, claim, transcript)?;
    let w_claim = w_mle.evaluate(&out.challenges)?;
    Ok(DigitRangeProof {
        levels,
        leaf_claims: node_claims.clone(),
        s_claim,
        binding: out.proof,
        binding_point: out.challenges,
        w_claim,
        fused_coeffs,
    })
}

/// Verify the digit-range proof. `w` is the (revealed, kernel-scale)
/// witness the statement is about; at commitment scale the `w_claim`
/// binding becomes the PCS opening of the committed `w̃`.
#[allow(clippy::too_many_lines)]
pub fn verify_digit_range(
    proof: &DigitRangeProof,
    w: &[i64],
    b_star: u32,
    i_bin: &[usize],
    transcript: &mut Transcript,
) -> Result<(), RangeError> {
    let n = w.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(RangeError::Shape {
            expected: 0,
            got: n,
        });
    }
    let num_vars = n.trailing_zeros() as usize;
    let levels_meta = range_tree(b_star)?;
    if proof.levels.len() != levels_meta.len() {
        return Err(RangeError::Shape {
            expected: levels_meta.len(),
            got: proof.levels.len(),
        });
    }
    // Level 0 replay + shape checks.
    let tau0: Vec<Goldilocks> = transcript.challenge_fields(b"a3-range-tau0", num_vars)?;
    let v0 = proof.levels[0].sumcheck.verify(
        num_vars,
        levels_meta[0].degree + 1,
        Goldilocks::ZERO,
        transcript,
        None,
    )?;
    let mut point = v0.point;
    let mut node_claims = proof.levels[0].child_claims.clone();
    if node_claims.len() != levels_meta[0].nodes * levels_meta[0].degree {
        return Err(RangeError::Shape {
            expected: levels_meta[0].nodes * levels_meta[0].degree,
            got: node_claims.len(),
        });
    }
    // Level 0 terminal: 0 = eq(τ₀, r)·Π_j ĉ_j(r).
    let mut term = DenseMle::eq_extension(&tau0)
        .evaluate(&point)
        .map_err(|_| RangeError::Shape {
            expected: num_vars,
            got: point.len(),
        })?;
    for c in &node_claims {
        term = term.mul(c);
    }
    if v0.final_claim != term {
        return Err(RangeError::TerminalFailed);
    }
    // Inner levels.
    for (li, level) in levels_meta.iter().enumerate().skip(1) {
        let nus: Vec<Goldilocks> = transcript.challenge_fields(b"a3-range-nu", level.nodes)?;
        if node_claims.len() != level.nodes {
            return Err(RangeError::Shape {
                expected: level.nodes,
                got: node_claims.len(),
            });
        }
        let mut claim = Goldilocks::ZERO;
        for (j, nu) in nus.iter().enumerate() {
            claim = claim.add(&nu.mul(&node_claims[j]));
        }
        let lp = &proof.levels[li];
        if lp.child_claims.len() != level.nodes * level.degree {
            return Err(RangeError::Shape {
                expected: level.nodes * level.degree,
                got: lp.child_claims.len(),
            });
        }
        let v = lp
            .sumcheck
            .verify(num_vars, level.degree + 1, claim, transcript, None)?;
        // Terminal: Σ_j ν_j·Π_k ĉ_{j,k}(r)·eq(p, r) == final claim.
        let eqp = DenseMle::eq_extension(&point)
            .evaluate(&v.point)
            .map_err(|_| RangeError::Shape {
                expected: num_vars,
                got: v.point.len(),
            })?;
        let mut t = Goldilocks::ZERO;
        for (j, nu) in nus.iter().enumerate() {
            let mut prod = *nu;
            for k in 0..level.degree {
                prod = prod.mul(&lp.child_claims[j * level.degree + k]);
            }
            t = t.add(&prod);
        }
        if v.final_claim != t.mul(&eqp) {
            return Err(RangeError::TerminalFailed);
        }
        point = v.point;
        node_claims = lp.child_claims.clone();
    }
    // Leaf consistency: every leaf claim + k(k+1) agrees on s_claim.
    let r_virt = point;
    if proof.leaf_claims.len() != u_set(b_star).len() {
        return Err(RangeError::Shape {
            expected: u_set(b_star).len(),
            got: proof.leaf_claims.len(),
        });
    }
    for (k, lc) in proof.leaf_claims.iter().enumerate() {
        let kk = u64::try_from(k).unwrap_or(u64::MAX);
        if lc.add(&Goldilocks::from_u64(kk * (kk + 1))) != proof.s_claim {
            return Err(RangeError::LeafInconsistent);
        }
    }
    // The (fused) binding.
    let w_mle = mle_of(w.iter().map(|&v| to_field(v)).collect())?;
    let w_plus_one = mle_of(w.iter().map(|&v| to_field(v + 1)).collect())?;
    let (coeff0, claim) = match proof.fused_coeffs {
        Some((gamma, zeta)) => {
            let g = transcript.challenge_field(b"a3-range-gamma")?;
            let z = transcript.challenge_field(b"a3-range-zetabin")?;
            if (g, z) != (gamma, zeta) {
                return Err(RangeError::BindingFailed);
            }
            (gamma, proof.s_claim.mul(&gamma))
        }
        None => (Goldilocks::ONE, proof.s_claim),
    };
    let v = proof.binding.verify(num_vars, 3, claim, transcript, None)?;
    if v.point != proof.binding_point {
        return Err(RangeError::BindingFailed);
    }
    // Terminal: final = [γ·eq(r,r') (+ ζ·w̃_bin(r'))]·ŵ(r')·(ŵ(r')+1),
    // with ŵ(r') cross-checked against the witness MLE (kernel scale).
    let true_w_claim = w_mle.evaluate(&v.point)?;
    if true_w_claim != proof.w_claim {
        return Err(RangeError::BindingFailed);
    }
    let mut weight = DenseMle::eq_extension(&r_virt)
        .evaluate(&v.point)
        .map_err(|_| RangeError::Shape {
            expected: num_vars,
            got: v.point.len(),
        })?
        .mul(&coeff0);
    if let Some((_, zeta)) = proof.fused_coeffs {
        let bin_set: std::collections::HashSet<usize> = i_bin.iter().copied().collect();
        let mut vals = Vec::with_capacity(n);
        for (x, &e) in DenseMle::eq_extension(&r_virt)
            .evaluations
            .iter()
            .enumerate()
        {
            vals.push(if bin_set.contains(&x) {
                e
            } else {
                Goldilocks::ZERO
            });
        }
        let rw = mle_of(vals)?.evaluate(&v.point)?;
        weight = weight.add(&rw.mul(&zeta));
    }
    let wp1 = w_plus_one
        .evaluate(&v.point)
        .map_err(|_| RangeError::Shape {
            expected: num_vars,
            got: v.point.len(),
        })?;
    let expect = weight.mul(&proof.w_claim).mul(&wp1);
    if v.final_claim != expect {
        return Err(RangeError::BindingFailed);
    }
    Ok(())
}

/// The direct-route norm proof (Eq 118–120).
#[derive(Clone, Debug)]
pub struct DirectNormProof {
    /// `E_resp` in the canonical fixed-width unsigned encoding (the width
    /// is derived from `S_max`; non-canonical encodings are rejected).
    pub e_resp_bytes: Vec<u8>,
    /// The degree-2 sumcheck `Σ_x z_int(x)² = E_resp` (Eq 120).
    pub sumcheck: SumcheckProof,
    /// The final point (carried for the caller's factor-claim binding).
    pub point: Vec<Goldilocks>,
}

/// `U_dir = N_A·(Σ_h b^h·B_{dig,h})²` (Eq 119) — the admissibility gate.
pub fn u_dir(params: &RangeParams, n_a: usize) -> u128 {
    let mut acc: u128 = 0;
    for (h, bound) in params.digit_bounds.iter().enumerate() {
        acc += u128::from(params.response_base.pow(h as u32) * bound);
    }
    (n_a as u128) * acc * acc
}

/// Canonical fixed-width big-endian encoding of `E_resp`.
fn canonical_encode(value: u128, s_max: u64) -> Vec<u8> {
    let tz = s_max.max(1).next_power_of_two().trailing_zeros() as usize;
    let width = tz.div_ceil(8).clamp(1, 16);
    value.to_be_bytes()[16 - width..].to_vec()
}

/// Decode + the canonicality/out-of-range gates.
pub fn canonical_decode(bytes: &[u8], s_max: u64) -> Result<u128, RangeError> {
    if bytes.is_empty() || bytes.len() > 16 {
        return Err(RangeError::NonCanonicalNorm { value: 0 });
    }
    let mut buf = [0u8; 16];
    buf[16 - bytes.len()..].copy_from_slice(bytes);
    let value = u128::from_be_bytes(buf);
    if canonical_encode(value, s_max) != bytes {
        return Err(RangeError::NonCanonicalNorm { value });
    }
    if value > u128::from(s_max) {
        return Err(RangeError::NonCanonicalNorm { value });
    }
    Ok(value)
}

/// Prove the response squared-norm on the direct route (Eq 120).
/// `z_int` are the reconstructed response coefficients (Eq 116's
/// `z_u^Z`), zero-padded to the Boolean domain by §6.2's rule.
pub fn prove_norm_direct(
    z_int: &[i64],
    params: &RangeParams,
    transcript: &mut Transcript,
) -> Result<DirectNormProof, RangeError> {
    let n_a = z_int.len();
    let gate = u_dir(params, n_a);
    if gate >= GOLDILOCKS_Q || u128::from(params.s_max) >= GOLDILOCKS_Q {
        return Err(RangeError::DirectRouteInadmissible { u_dir: gate });
    }
    let e_int: u128 = z_int
        .iter()
        .map(|&v| {
            let a = v.unsigned_abs();
            u128::from(a) * u128::from(a)
        })
        .sum();
    if e_int > u128::from(params.s_max) {
        return Err(RangeError::NonCanonicalNorm { value: e_int });
    }
    let padded_len = n_a.max(1).next_power_of_two();
    let mut padded = vec![0i64; padded_len];
    padded[..n_a].copy_from_slice(z_int);
    let num_vars = padded_len.trailing_zeros() as usize;
    let z_mle = mle_of(padded.iter().map(|&v| to_field(v)).collect())?;
    let claim = Goldilocks::from_u64((e_int % GOLDILOCKS_Q) as u64);
    let mut vp = VirtualPolynomial::new(num_vars);
    let f = vp.add_factor(z_mle.clone())?;
    let f2 = vp.add_factor(z_mle)?;
    vp.add_term(Goldilocks::ONE, vec![f, f2])?;
    let out = sumcheck::prove(&vp, claim, transcript)?;
    Ok(DirectNormProof {
        e_resp_bytes: canonical_encode(e_int, params.s_max),
        sumcheck: out.proof,
        point: out.challenges,
    })
}

/// Verify the direct-route norm proof (Lemma 6.3: both integers live in
/// `[0, q)` so the accepted field identity pins the integer equality).
pub fn verify_norm_direct(
    proof: &DirectNormProof,
    params: &RangeParams,
    n_a: usize,
    transcript: &mut Transcript,
) -> Result<u128, RangeError> {
    let gate = u_dir(params, n_a);
    if gate >= GOLDILOCKS_Q || u128::from(params.s_max) >= GOLDILOCKS_Q {
        return Err(RangeError::DirectRouteInadmissible { u_dir: gate });
    }
    let e_resp = canonical_decode(&proof.e_resp_bytes, params.s_max)?;
    let claim = Goldilocks::from_u64((e_resp % GOLDILOCKS_Q) as u64);
    let padded_len = n_a.max(1).next_power_of_two();
    let num_vars = padded_len.trailing_zeros() as usize;
    let v = proof
        .sumcheck
        .verify(num_vars, 2, claim, transcript, None)?;
    if v.point != proof.point {
        return Err(RangeError::TerminalFailed);
    }
    Ok(e_resp)
}

/// The digit-expanded route (Eq 121–123).
#[derive(Clone, Debug)]
pub struct ExpandedNormProof {
    /// `E_resp` in the canonical encoding.
    pub e_resp_bytes: Vec<u8>,
    /// Per-(t, h, k) claims `p_{t,h,k}` (Eq 121), row-major over
    /// `(t, h, k)` with `0 ≤ h ≤ k < δ_f`.
    pub p_claims: Vec<Goldilocks>,
    /// The ONE batched degree-2 sumcheck over the padded hypercube:
    /// `Σ_x Σ_{t,h,k} λ_{t,h,k}·1̃_{I_t}(x)·z_h(x)·z_k(x) = Σ λ·p`.
    pub sumcheck: SumcheckProof,
    /// The final point.
    pub point: Vec<Goldilocks>,
}

/// Consecutive power-of-two-aligned segment partition of `[N_A]`.
pub fn segment_partition(n_a: usize, seg_len: usize) -> Result<Vec<(usize, usize)>, RangeError> {
    if seg_len == 0 || n_a % seg_len != 0 {
        return Err(RangeError::Shape {
            expected: 0,
            got: n_a,
        });
    }
    Ok((0..n_a)
        .step_by(seg_len)
        .map(|s| (s, s + seg_len))
        .collect())
}

fn pair_count(depth: usize) -> usize {
    depth * (depth + 1) / 2
}

#[allow(clippy::too_many_lines)]
fn segment_bound_check(
    segments: &[(usize, usize)],
    params: &RangeParams,
) -> Result<(), RangeError> {
    let depth = params.digit_depth;
    for (t, (start, end)) in segments.iter().enumerate() {
        let card = end - start;
        for h in 0..depth {
            for k in h..depth {
                let bound = (card as u128)
                    * u128::from(params.digit_bounds[h])
                    * u128::from(params.digit_bounds[k]);
                if bound >= GOLDILOCKS_Q / 2 {
                    return Err(RangeError::SegmentBoundViolated {
                        segment: t,
                        h,
                        k,
                        bound,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Prove the response norm on the digit-expanded route. `digit_planes`
/// holds `z_h(u) = w(π(u, h))` per plane `h`, each zero-padded to the
/// (power-of-two) hypercube in the scheduled order.
#[allow(clippy::too_many_lines)]
pub fn prove_norm_expanded(
    digit_planes: &[Vec<i64>],
    segments: &[(usize, usize)],
    params: &RangeParams,
    transcript: &mut Transcript,
) -> Result<ExpandedNormProof, RangeError> {
    let depth = params.digit_depth;
    if digit_planes.len() != depth {
        return Err(RangeError::Shape {
            expected: depth,
            got: digit_planes.len(),
        });
    }
    let cube = digit_planes[0].len();
    if cube == 0 || !cube.is_power_of_two() {
        return Err(RangeError::Shape {
            expected: 0,
            got: cube,
        });
    }
    segment_bound_check(segments, params)?;
    // Exact per-(t,h,k) integer inner products (Eq 121) + the true norm.
    let mut p_claims: Vec<Goldilocks> = Vec::with_capacity(segments.len() * pair_count(depth));
    let mut e_int: i128 = 0;
    for &(start, end) in segments {
        for h in 0..depth {
            for k in h..depth {
                let mut ip: i128 = 0;
                for (&zh, &zk) in digit_planes[h][start..end]
                    .iter()
                    .zip(&digit_planes[k][start..end])
                {
                    ip += i128::from(zh) * i128::from(zk);
                }
                p_claims.push(Goldilocks::from_u64(
                    ip.rem_euclid(GOLDILOCKS_Q as i128) as u64
                ));
                if h == k {
                    e_int += i128::from(params.response_base.pow(2 * h as u32)) * ip;
                } else {
                    e_int += 2 * i128::from(params.response_base.pow((h + k) as u32)) * ip;
                }
            }
        }
    }
    if e_int < 0 || e_int > u128::from(params.s_max) as i128 {
        return Err(RangeError::NonCanonicalNorm {
            value: e_int.max(0) as u128,
        });
    }
    // The batched sumcheck over the full hypercube.
    let num_vars = cube.trailing_zeros() as usize;
    let total_pairs = segments.len() * pair_count(depth);
    let lambdas: Vec<Goldilocks> = transcript.challenge_fields(b"a3-norm-lambda", total_pairs)?;
    let mut claim = Goldilocks::ZERO;
    for (l, p) in lambdas.iter().zip(p_claims.iter()) {
        claim = claim.add(&l.mul(p));
    }
    let mut vp = VirtualPolynomial::new(num_vars);
    let plane_idx: Vec<usize> = digit_planes
        .iter()
        .map(|plane| {
            vp.add_factor(mle_of(plane.iter().map(|&v| to_field(v)).collect())?)
                .map_err(RangeError::from)
        })
        .collect::<Result<Vec<_>, RangeError>>()?;
    let seg_idx: Vec<usize> = segments
        .iter()
        .map(|&(start, end)| {
            let mut vals = vec![Goldilocks::ZERO; cube];
            for v in vals.iter_mut().take(end).skip(start) {
                *v = Goldilocks::ONE;
            }
            vp.add_factor(mle_of(vals)?).map_err(RangeError::from)
        })
        .collect::<Result<Vec<_>, RangeError>>()?;
    let mut li = 0;
    for (t, _) in segments.iter().enumerate() {
        for h in 0..depth {
            for k in h..depth {
                vp.add_term(lambdas[li], vec![seg_idx[t], plane_idx[h], plane_idx[k]])?;
                li += 1;
            }
        }
    }
    let out = sumcheck::prove(&vp, claim, transcript)?;
    Ok(ExpandedNormProof {
        e_resp_bytes: canonical_encode(e_int as u128, params.s_max),
        p_claims,
        sumcheck: out.proof,
        point: out.challenges,
    })
}

/// Verify the digit-expanded norm proof (Lemma 6.4's unique lifting).
#[allow(clippy::too_many_lines)]
pub fn verify_norm_expanded(
    proof: &ExpandedNormProof,
    segments: &[(usize, usize)],
    params: &RangeParams,
    transcript: &mut Transcript,
) -> Result<u128, RangeError> {
    let depth = params.digit_depth;
    let expected_claims = segments.len() * pair_count(depth);
    if proof.p_claims.len() != expected_claims {
        return Err(RangeError::Shape {
            expected: expected_claims,
            got: proof.p_claims.len(),
        });
    }
    segment_bound_check(segments, params)?;
    let e_resp = canonical_decode(&proof.e_resp_bytes, params.s_max)?;
    // Centered lifts p̄_{t,h,k} ∈ (−q/2, q/2).
    let lifted: Vec<i128> = proof
        .p_claims
        .iter()
        .map(|p| {
            let c = p.to_canonical_u64() as i128;
            if c > GOLDILOCKS_Q as i128 / 2 {
                c - GOLDILOCKS_Q as i128
            } else {
                c
            }
        })
        .collect();
    // λ replay + the batched sumcheck.
    let lambdas: Vec<Goldilocks> =
        transcript.challenge_fields(b"a3-norm-lambda", expected_claims)?;
    let mut claim = Goldilocks::ZERO;
    for (l, p) in lambdas.iter().zip(proof.p_claims.iter()) {
        claim = claim.add(&l.mul(p));
    }
    let num_vars = proof.sumcheck.rounds.len();
    if num_vars == 0 || num_vars > 30 {
        return Err(RangeError::Shape {
            expected: 1,
            got: num_vars,
        });
    }
    // Degree 3 per variable: the indicator factor + the two planes.
    let v = proof
        .sumcheck
        .verify(num_vars, 3, claim, transcript, None)?;
    if v.point != proof.point {
        return Err(RangeError::TerminalFailed);
    }
    // Integer reconstruction (Eq 123), exact in i128.
    let mut e_rec: i128 = 0;
    let mut li = 0;
    for _ in segments {
        for h in 0..depth {
            for k in h..depth {
                let p = lifted[li];
                if h == k {
                    e_rec += i128::from(params.response_base.pow(2 * h as u32)) * p;
                } else {
                    e_rec += 2 * i128::from(params.response_base.pow((h + k) as u32)) * p;
                }
                li += 1;
            }
        }
    }
    if e_rec < 0 || e_rec != e_resp as i128 {
        return Err(RangeError::ReconstructionMismatch {
            claimed: e_resp,
            reconstructed: e_rec,
        });
    }
    Ok(e_resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> Transcript {
        Transcript::new_default(b"lzx-a3-test")
    }

    fn small_witness(n: usize, b_star: u32, seed: &[u8]) -> Vec<i64> {
        let half = i64::from(b_star) / 2;
        (0..n)
            .map(|i| {
                let h = lattice_core::keccak::sha3_256(&[seed, &(i as u64).to_be_bytes()].concat());
                i64::from(u32::from(h[0]) % b_star) - half
            })
            .collect()
    }

    #[test]
    fn degree_halving_identity() {
        // Q_{b*}(w) = Q_sq(w(w+1)) vanishes exactly on A_{b*}.
        for b_star in [4u32, 8, 16] {
            for w in -40i64..40 {
                let qp = vanishing_product(b_star, w).unwrap();
                assert_eq!(qp == 0, in_alphabet(b_star, w), "b*={b_star} w={w}");
            }
        }
    }

    #[test]
    fn range_tree_shapes_match_paper() {
        assert_eq!(
            range_tree(4).unwrap(),
            vec![TreeLevel {
                degree: 2,
                nodes: 1
            }]
        );
        assert_eq!(
            range_tree(8).unwrap(),
            vec![TreeLevel {
                degree: 4,
                nodes: 1
            }]
        );
        let t16 = range_tree(16).unwrap();
        assert_eq!(
            t16[0],
            TreeLevel {
                degree: 2,
                nodes: 1
            }
        );
        assert_eq!(
            t16[1],
            TreeLevel {
                degree: 4,
                nodes: 2
            }
        );
        let t64 = range_tree(64).unwrap();
        assert_eq!(t64.len(), 3);
        assert_eq!(t64[2].nodes, 8);
        assert!(range_tree(12).is_err());
    }

    #[test]
    fn digit_range_proves_and_verifies() {
        for b_star in [4u32, 8, 16, 32, 64] {
            let w = small_witness(16, b_star, b"a3-happy");
            let mut t = transcript();
            let proof = prove_digit_range(&w, b_star, &[], &mut t).unwrap();
            let mut tv = transcript();
            assert!(
                verify_digit_range(&proof, &w, b_star, &[], &mut tv).is_ok(),
                "b*={b_star}"
            );
        }
    }

    #[test]
    fn digit_range_fused_binary_proves_and_verifies() {
        let mut w = small_witness(8, 16, b"a3-fused");
        // Compression spans: force {−1, 0} cells.
        w[0] = -1;
        w[1] = 0;
        let ibin = vec![0usize, 1];
        let mut t = transcript();
        let proof = prove_digit_range(&w, 16, &ibin, &mut t).unwrap();
        let mut tv = transcript();
        assert!(verify_digit_range(&proof, &w, 16, &ibin, &mut tv).is_ok());
        // A binary violation (cell 1 = 3) fails closed at prove.
        let mut bad = w.clone();
        bad[1] = 3;
        let mut t2 = transcript();
        assert!(matches!(
            prove_digit_range(&bad, 16, &ibin, &mut t2),
            Err(RangeError::BinaryViolated { index: 1, .. })
        ));
    }

    #[test]
    fn out_of_range_fails_closed() {
        let mut w = small_witness(8, 4, b"a3-bad");
        w[3] = 2; // A_4 = {−2,−1,0,1}
        let mut t = transcript();
        assert!(matches!(
            prove_digit_range(&w, 4, &[], &mut t),
            Err(RangeError::DigitOutOfRange { index: 3, .. })
        ));
    }

    #[test]
    fn tampered_round_rejected() {
        let w = small_witness(8, 8, b"a3-tamper");
        let mut t = transcript();
        let mut proof = prove_digit_range(&w, 8, &[], &mut t).unwrap();
        // Flip one round value in the anchored identity's sumcheck.
        let r0 = &mut proof.levels[0].sumcheck.rounds[0];
        r0[0] = r0[0].add(&Goldilocks::ONE);
        let mut tv = transcript();
        assert!(matches!(
            verify_digit_range(&proof, &w, 8, &[], &mut tv),
            Err(RangeError::Sumcheck(SumcheckError::RoundCheckFailed { .. }))
        ));
    }

    #[test]
    fn tampered_leaf_claim_rejected() {
        let w = small_witness(8, 8, b"a3-leaf");
        let mut t = transcript();
        let mut proof = prove_digit_range(&w, 8, &[], &mut t).unwrap();
        proof.leaf_claims[1] = proof.leaf_claims[1].add(&Goldilocks::ONE);
        let mut tv = transcript();
        assert!(matches!(
            verify_digit_range(&proof, &w, 8, &[], &mut tv),
            Err(RangeError::LeafInconsistent)
        ));
    }

    #[test]
    fn tampered_w_claim_rejected() {
        let w = small_witness(8, 4, b"a3-wclaim");
        let mut t = transcript();
        let mut proof = prove_digit_range(&w, 4, &[], &mut t).unwrap();
        proof.w_claim = proof.w_claim.add(&Goldilocks::ONE);
        let mut tv = transcript();
        assert!(matches!(
            verify_digit_range(&proof, &w, 4, &[], &mut tv),
            Err(RangeError::BindingFailed)
        ));
    }

    #[test]
    fn tampered_s_claim_rejected() {
        let w = small_witness(8, 4, b"a3-sclaim");
        let mut t = transcript();
        let mut proof = prove_digit_range(&w, 4, &[], &mut t).unwrap();
        // Shift s_claim AND the leaf claims consistently — the binding
        // sumcheck's terminal identity then fails (the witness is fixed).
        let shift = Goldilocks::ONE;
        proof.s_claim = proof.s_claim.add(&shift);
        for lc in proof.leaf_claims.iter_mut() {
            *lc = lc.add(&shift);
        }
        let mut tv = transcript();
        assert!(verify_digit_range(&proof, &w, 4, &[], &mut tv).is_err());
    }

    #[test]
    fn restricted_eq_is_not_the_product() {
        // The paper's warning: eq̃·1_{I_bin} ≠ eq·1̃ as polynomials.
        let r: Vec<Goldilocks> = (0..3)
            .map(|i| Goldilocks::from_u64(0x1234 + i as u64))
            .collect();
        let n = 8usize;
        let bin_set: std::collections::HashSet<usize> = [0usize, 5].into_iter().collect();
        let eq = DenseMle::eq_extension(&r);
        let mut restricted_vals = Vec::with_capacity(n);
        for (x, &e) in eq.evaluations.iter().enumerate() {
            restricted_vals.push(if bin_set.contains(&x) {
                e
            } else {
                Goldilocks::ZERO
            });
        }
        let restricted = mle_of(restricted_vals).unwrap();
        // The indicator MLE of {0, 5}:
        let mut ind_vals = vec![Goldilocks::ZERO; n];
        ind_vals[0] = Goldilocks::ONE;
        ind_vals[5] = Goldilocks::ONE;
        let ind = mle_of(ind_vals).unwrap();
        // eq·1̃ at a non-Boolean point differs from the restricted MLE.
        let probe: Vec<Goldilocks> = (0..3)
            .map(|i| Goldilocks::from_u64(0x0f0f + i as u64))
            .collect();
        let a = restricted.evaluate(&probe).unwrap();
        let b = eq
            .evaluate(&probe)
            .unwrap()
            .mul(&ind.evaluate(&probe).unwrap());
        assert_ne!(a, b);
        // …but they agree on the Boolean cube (both equal eq·1_{I_bin}).
        for x in 0..n {
            let pt: Vec<Goldilocks> = (0..3)
                .map(|j| Goldilocks::from_u64(((x >> j) & 1) as u64))
                .collect();
            assert_eq!(
                restricted.evaluate(&pt).unwrap(),
                eq.evaluate(&pt).unwrap().mul(&ind.evaluate(&pt).unwrap())
            );
        }
    }

    #[test]
    fn norm_direct_happy_and_gates() {
        let params = RangeParams {
            b_star: 8,
            digit_depth: 2,
            response_base: 16,
            digit_bounds: vec![4, 4],
            s_max: 1 << 20,
        };
        let z: Vec<i64> = (0..8).map(|i| i64::from((i * 37) % 29) - 14).collect();
        let mut t = transcript();
        let proof = prove_norm_direct(&z, &params, &mut t).unwrap();
        let mut tv = transcript();
        let e = verify_norm_direct(&proof, &params, z.len(), &mut tv).unwrap();
        let e_true: u128 = z.iter().map(|&v| u128::from(v.unsigned_abs().pow(2))).sum();
        assert_eq!(e, e_true);
        // U_dir gate past q.
        let mut bad = params.clone();
        bad.digit_bounds = vec![1 << 40, 1 << 40];
        assert!(matches!(
            prove_norm_direct(&z, &bad, &mut t),
            Err(RangeError::DirectRouteInadmissible { .. })
        ));
        // S_max gate.
        let mut tight = params.clone();
        tight.s_max = 1;
        assert!(matches!(
            prove_norm_direct(&z, &tight, &mut t),
            Err(RangeError::NonCanonicalNorm { .. })
        ));
    }

    #[test]
    fn norm_direct_tampered_round_rejected() {
        let params = RangeParams {
            b_star: 8,
            digit_depth: 2,
            response_base: 16,
            digit_bounds: vec![4, 4],
            s_max: 1 << 20,
        };
        let z: Vec<i64> = (0..8).map(|i| i64::from((i * 31) % 21) - 10).collect();
        let mut t = transcript();
        let mut proof = prove_norm_direct(&z, &params, &mut t).unwrap();
        let r0 = &mut proof.sumcheck.rounds[0];
        r0[0] = r0[0].add(&Goldilocks::ONE);
        let mut tv = transcript();
        assert!(matches!(
            verify_norm_direct(&proof, &params, z.len(), &mut tv),
            Err(RangeError::Sumcheck(SumcheckError::RoundCheckFailed { .. }))
        ));
    }

    #[test]
    fn norm_expanded_reconstruction() {
        let params = RangeParams {
            b_star: 16,
            digit_depth: 3,
            response_base: 16,
            digit_bounds: vec![8, 8, 8],
            s_max: 1 << 26,
        };
        let cube = 16usize;
        let planes: Vec<Vec<i64>> = (0..3)
            .map(|h| (0..cube).map(|u| ((u * (h + 3)) % 15) as i64 - 7).collect())
            .collect();
        let segs = segment_partition(cube, 4).unwrap();
        let mut t = transcript();
        let proof = prove_norm_expanded(&planes, &segs, &params, &mut t).unwrap();
        let mut tv = transcript();
        let e = verify_norm_expanded(&proof, &segs, &params, &mut tv).unwrap();
        // The reconstruction is exact: E = Σ_u (Σ_h b^h z_h(u))².
        let mut e_true: i128 = 0;
        for u in 0..cube {
            let mut z = 0i128;
            for (h, plane) in planes.iter().enumerate() {
                z += i128::from(params.response_base.pow(h as u32)) * i128::from(plane[u]);
            }
            e_true += z * z;
        }
        assert_eq!(e, e_true as u128);
        // Tampered p-claim → the reconstruction mismatches (or the
        // round checks fail first).
        let mut bad = proof.clone();
        bad.p_claims[0] = bad.p_claims[0].add(&Goldilocks::ONE);
        let mut tv2 = transcript();
        assert!(verify_norm_expanded(&bad, &segs, &params, &mut tv2).is_err());
    }

    #[test]
    fn segment_bound_violation_fails_closed() {
        let params = RangeParams {
            b_star: 16,
            digit_depth: 2,
            response_base: 16,
            digit_bounds: vec![1 << 34, 1 << 34],
            s_max: 1 << 40,
        };
        let planes: Vec<Vec<i64>> = vec![vec![0; 8], vec![0; 8]];
        let segs = segment_partition(8, 4).unwrap();
        let mut t = transcript();
        assert!(matches!(
            prove_norm_expanded(&planes, &segs, &params, &mut t),
            Err(RangeError::SegmentBoundViolated { .. })
        ));
    }

    #[test]
    fn canonical_encoding_gates() {
        assert_eq!(canonical_encode(0, 255).len(), 1);
        assert_eq!(canonical_encode(255, 255).len(), 1);
        assert!(canonical_decode(&[0, 255], 255).is_err()); // width ≠ canonical
        assert!(canonical_decode(&[255], 254).is_err()); // > S_max
        assert_eq!(canonical_decode(&[254], 255).unwrap(), 254);
    }
}
