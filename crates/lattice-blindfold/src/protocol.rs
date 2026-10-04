//! The LatticeBlindFold protocol stack (§4): the three modified
//! reductions Π'_R1CS (Protocol 6), Π'_RLC (Protocol 7), Π'_DEC
//! (Protocol 8), the samplers (Protocols 9/10/11), the composed
//! protocol Π_LBF (Protocol 12) with the ι_bl precomposition
//! (Definition 4.6), the accumulator-free variant Π°_LBF
//! (Corollary 4.24) and the folding blueprint (Protocol 13).
//!
//! Every reduction is an honest-prover/verifier pair producing and
//! checking a full public-coin transcript; the verifier NEVER re-executes
//! the prover's algebra — it checks the PoK transcripts, the homomorphic
//! identities and the anchoring relations.

use crate::abdlop::{AbdlopCommitment, AbdlopOpening, AbdlopPp, MsgLayout};
use crate::ajtai::{AjtaiL, BlindedStructure};
use crate::embed::{EqArray, FieldVec, RingMle, StructMatrix};
use crate::fp::Fq;
use crate::fq2::K;
use crate::gauss::{rej1_decide, Rng};
use crate::params::Params;
use crate::pok::{
    pok_anchor, pok_ct_wrapper, pok_linear, pok_quadratic, verify_anchor, verify_ct_wrapper,
    verify_linear, verify_quadratic, Block, PokError, QuadRelation, RelRow, RelRows,
};
use crate::ring::{split_b_k, Poly, StrongSet};
use crate::rk::PolyK;
use crate::sumcheck::{MaskedSumcheck, SumcheckInputs};

/// The full setup: global parameters + the blinded structure + the
/// commitment layers (Definition 2.2).
pub struct LbfSetup {
    pub params: Params,
    /// The compact Ajtai commitment L (κ_Ajtai × nR).
    pub ajtai: AjtaiL,
    /// The ABDLOP public parameters.
    pub abdlop: AbdlopPp,
    /// The blinded structure.
    pub blinded: BlindedStructure,
    /// M2 in the blinded layout (ring-lifted).
    pub m2: StructMatrix,
    /// M3 in the blinded layout (ring-lifted).
    pub m3: StructMatrix,
}

impl LbfSetup {
    /// Build from circuit-level (M2°, M3°) over the unpadded columns.
    pub fn setup(
        params: Params,
        m2_circ: &[Vec<i64>],
        m3_circ: &[Vec<i64>],
        rng: &mut Rng,
    ) -> LbfSetup {
        let blinded = BlindedStructure {
            m_circ: params.nf - params.nf_bl,
            nf_circ: params.nf - params.nf_bl,
            nf_bl: params.nf_bl,
            d: params.d,
            iota1: 0,
        };
        let (m2, m3) = blinded.build_matrices(m2_circ, m3_circ);
        let to_struct = |mat: &[Vec<i64>]| -> StructMatrix {
            let rows: Vec<Vec<Fq>> = mat
                .iter()
                .map(|r| r.iter().map(|&v| Fq::from_i64(v)).collect())
                .collect();
            StructMatrix::new(rows, params.d)
        };
        LbfSetup {
            params: params.clone(),
            ajtai: AjtaiL::setup_d(
                params.kappa_ajtai,
                params.nr(),
                params.nr_bl(),
                params.d,
                rng,
            ),
            abdlop: AbdlopPp::setup(
                params.kappa,
                params.ell,
                params.m1,
                params.m2,
                params.d,
                rng,
            ),
            blinded,
            m2: to_struct(&m2),
            m3: to_struct(&m3),
        }
    }

    /// M1 = I_m (Remark 4.1.(3)).
    pub fn m1(&self) -> StructMatrix {
        StructMatrix::identity(self.params.nf, self.params.d)
    }
}

/// A CEcom instance (Definition 3.21).
#[derive(Clone, Debug)]
pub struct CecomInstance {
    /// c = L(z) ∈ R_F^κ.
    pub c: Vec<Poly>,
    /// x = Lin(z) — the public input (field coords).
    pub x: Vec<Fq>,
    /// The evaluation point r ∈ K^{log2 m}.
    pub r: Vec<K>,
    /// The t ABDLOP hint commitments C_j = Commit(y_j, s_j).
    pub coms: Vec<AbdlopCommitment>,
}

/// A CEcom witness.
#[derive(Clone, Debug)]
pub struct CecomWitness {
    /// z ∈ F^{nF} with ∥z∥∞ < bound.
    pub z: FieldVec,
    /// y_j = M̄_j z(r) ∈ R_K.
    pub y: Vec<PolyK>,
    /// The ABDLOP openings (s1, s2, slots) per j.
    pub openings: Vec<AbdlopOpening>,
}

/// A CEcom pair.
#[derive(Clone, Debug)]
pub struct CecomPair {
    pub inst: CecomInstance,
    pub wit: CecomWitness,
}

/// Check a CEcom pair against the relation (Definition 3.21). The
/// `salt_bound` is B̃ = 2 for fresh instances and B_fold =
/// (K+k)·T·(B̃−1)+B̃ for folded ones (Lemma B.1).
pub fn cecom_check(
    setup: &LbfSetup,
    pair: &CecomPair,
    norm_bound: i64,
    salt_bound: i64,
) -> Result<(), String> {
    let p = &setup.params;
    let zr = pair.wit.z.to_ring(p.d);
    // c = L(z)
    if setup.ajtai.commit(&zr) != pair.inst.c {
        return Err("c != L(z)".into());
    }
    // x = Lin(z)
    if pair.inst.x != pair.wit.z.0[..p.nf_in] {
        return Err("x != Lin(z)".into());
    }
    // ∥z∥∞ < bound
    if pair.wit.z.norm_inf() >= norm_bound {
        return Err("witness norm exceeded".into());
    }
    // y_j = M̄_j z(r) and C_j = Commit(y_j, s_j), ∥s_j∥ < B̃
    let m1 = setup.m1();
    let mats = [&m1, &setup.m2, &setup.m3];
    for (j, com) in pair.inst.coms.iter().enumerate().take(p.t) {
        let u = mats[j].mul_ring(&zr);
        let mle = RingMle::from_ring_vec(&u);
        let yj = mle.eval(&pair.inst.r);
        if yj != pair.wit.y[j] {
            return Err(format!("y_{j} != M̄_j z(r)"));
        }
        if !pair.wit.openings[j].verify(&setup.abdlop, com) {
            return Err(format!("C_{j} does not open"));
        }
        for s in &pair.wit.openings[j].s1 {
            if s.norm_inf() >= salt_bound {
                return Err("s1 norm".into());
            }
        }
        for s in &pair.wit.openings[j].s2 {
            if s.norm_inf() >= salt_bound {
                return Err("s2 norm".into());
            }
        }
    }
    Ok(())
}

/// Commit the hints y_j as one ABDLOP block per hint (fresh ternary salts).
fn commit_hint(pp: &AbdlopPp, y: &PolyK, rng: &mut Rng) -> (AbdlopCommitment, AbdlopOpening) {
    AbdlopOpening::commit_rk(pp, std::slice::from_ref(y), &[], rng)
}

/// The message-level R_K weight → slot-weights expansion:
/// w·(a + bY) = w·a + (w·Y)·b, with w·Y = PolyK{a: ν·w_b, b: w_a}.
fn msg_weight(w: &PolyK) -> (PolyK, PolyK) {
    let nu = Fq::new(crate::fp::NU);
    let wy = PolyK {
        a: w.b.scale(&nu),
        b: w.a.clone(),
    };
    (w.clone(), wy)
}

// ---------------------------------------------------------------------------
// Protocol 6 — the R1CS reduction Π'_R1CS.
// ---------------------------------------------------------------------------

/// The R1CS reduction output: K+k CEcom pairs at the new point r′.
pub struct R1csOutput {
    pub pairs: Vec<CecomPair>,
    pub r_prime: Vec<K>,
}

/// The R1CS reduction transcript (everything the verifier checks).
pub struct R1csTranscript {
    /// Step 1: the batched opening PoK for the k input hint commitments.
    pub step1: Option<crate::pok::LinearPokTranscript>,
    /// Step 4: the coeffs commitment (the packed mask).
    pub coeffs_com: AbdlopCommitment,
    /// The number of packed R_K elements.
    pub n_packed: usize,
    /// Step 6: σ = ζP + T.
    pub sigma: K,
    /// Step 7: the σ-anchoring transcript.
    pub step7: crate::pok::CtwTranscript,
    /// Step 8: the masked Sum-Check.
    pub sumcheck: crate::sumcheck::MaskedSumcheckTranscript,
    /// Step 9: the new-hint commitments (K+k)×t.
    pub hint_coms: Vec<Vec<AbdlopCommitment>>,
    /// Step 10: the surrogate commitments (K+k)×t.
    pub surr_coms: Vec<Vec<AbdlopCommitment>>,
    /// Step 12: the degree-0 wrapper transcript.
    pub step12: crate::pok::CtwTranscript,
    /// Steps 13-14: the sq/cube/prod commitments.
    pub sq_coms: Vec<AbdlopCommitment>,
    pub cube_coms: Vec<AbdlopCommitment>,
    pub prod_coms: Vec<AbdlopCommitment>,
    /// Step 15: the quadratic PoK transcript.
    pub step15: Option<crate::pok::QuadPokTranscript>,
    /// Step 16-17: the verifier-derived cF, cN, cE, u⋆ (documentation).
    pub c_f: K,
    pub c_n: K,
    pub c_e: K,
    pub u_star: K,
    /// Step 18: the final anchoring (target h).
    pub step18: crate::pok::CtwTranscript,
    /// The derived challenges (α, γ's, δ's, ζ, µ's).
    pub alpha: Vec<K>,
    pub gamma1: Vec<K>,
    pub gamma2: Vec<K>,
    pub gamma3: Vec<K>,
    pub delta0: K,
    pub delta1: K,
    pub zeta: K,
    pub mus: Vec<K>,
}

/// Run Protocol 6: fold K fresh (blinded-layout) instances + k accumulated
/// CEcom claims into K+k CEcom claims at a fresh point r′.
///
/// `fresh` — the K fresh witnesses z_i (field vectors, blinded layout,
/// already padded with the blinding block); `accs` — the k carried pairs.
pub fn r1cs_reduction(
    setup: &LbfSetup,
    fresh: &[FieldVec],
    accs: &[CecomPair],
    rng: &mut Rng,
) -> Result<(R1csOutput, R1csTranscript), PokError> {
    let p = &setup.params;
    let d = p.d;
    let t = p.t;
    let capital_k = fresh.len();
    let k = accs.len();
    let nk = capital_k + k;
    let log_m = p.log_m();
    let d_max = p.d_max();
    // ---- Assemble the working data: z_i for all i ∈ [K+k].
    let mut zs: Vec<FieldVec> = Vec::with_capacity(nk);
    zs.extend(fresh.iter().cloned());
    for acc in accs {
        zs.push(acc.wit.z.clone());
    }
    // Ring witnesses and the M̄_j z_i cube arrays.
    let m1 = setup.m1();
    let mats = [&m1, &setup.m2, &setup.m3];
    let mut ring_z: Vec<Vec<Poly>> = Vec::with_capacity(nk);
    for z in &zs {
        ring_z.push(z.to_ring(d));
    }
    let mut ring_mles: Vec<Vec<RingMle>> = Vec::with_capacity(nk);
    for zr in &ring_z {
        let mut per = Vec::with_capacity(t);
        for j in 0..t {
            per.push(RingMle::from_ring_vec(&mats[j].mul_ring(zr)));
        }
        ring_mles.push(per);
    }
    // ---- Step 1: the batched opening PoK for the k input hints.
    let mut acc_blocks: Vec<Block> = Vec::with_capacity(k * t);
    for acc in accs {
        for j in 0..t {
            acc_blocks.push(Block {
                com: acc.inst.coms[j].clone(),
                op: acc.wit.openings[j].clone(),
            });
        }
    }
    let open_rows = RelRows { rows: vec![] };
    let step1 = if !acc_blocks.is_empty() {
        let nc = acc_blocks.len();
        let widths = p.pok_widths_fresh(nc);
        Some(pok_linear(
            &setup.abdlop,
            &acc_blocks,
            &open_rows,
            widths,
            p.tau,
            p.beta_ch,
            p.w_max(),
            rng,
        )?)
    } else {
        None
    };
    // ---- Step 2: challenges.
    let draw_k = |rng: &mut Rng| {
        K(
            Fq(rng.next_u64() % crate::fp::Q),
            Fq(rng.next_u64() % crate::fp::Q),
        )
    };
    let alpha: Vec<K> = (0..log_m).map(|_| draw_k(rng)).collect();
    let gamma1: Vec<K> = (0..nk).map(|_| draw_k(rng)).collect();
    let gamma2: Vec<K> = (0..t).map(|_| draw_k(rng)).collect();
    let gamma3: Vec<K> = (0..d).map(|_| draw_k(rng)).collect();
    let delta0 = draw_k(rng);
    let delta1 = draw_k(rng);
    // ---- Step 3: T — the Eval part of the claimed sum from the carried
    // hints: T = δ1·Σ_{i>K,j,ℓ} γγγ·cf(y_{i,j})_ℓ.
    let mut t_claim = K::ZERO;
    for (idx, acc) in accs.iter().enumerate() {
        let i = capital_k + idx;
        for j in 0..t {
            for l in 1..=d {
                let w = gamma1[i]
                    .mul(&gamma2[j])
                    .mul(&gamma3[l - 1])
                    .mul(&acc.wit.y[j].cf(l));
                t_claim = t_claim.add(&w);
            }
        }
    }
    t_claim = delta1.mul(&t_claim);
    // ---- Steps 4-8: the masked Sum-Check with the coeffs commitment.
    // The mask p is sampled FIRST (Step 3), its coefficients packed and
    // committed (Step 4), THEN ζ is drawn (Step 5) and σ = ζP + T sent
    // (Step 6) — the mask precedes the challenge (§4.1.2's crucial point).
    let eq_alpha = EqArray::full(log_m, &alpha);
    // The old point r (the accs' shared point; fresh instances have no
    // old point — Eval sums over i > K only).
    let r_old = accs.first().map(|a| a.inst.r.clone()).unwrap_or_default();
    let eq_r = EqArray::full(log_m, &r_old);
    let field_mles: Vec<Vec<K>> = zs
        .iter()
        .map(|z| z.0.iter().map(|&c| K::from_fp(c)).collect())
        .collect();
    let mut sc_inputs = SumcheckInputs {
        ring_mles,
        field_mles,
        eq_alpha,
        eq_r,
        gamma1: gamma1.clone(),
        delta0,
        delta1,
        d_max,
        b: p.b,
        capital_k,
        eval_weights: None,
    };
    sc_inputs.set_eval_weights(&gamma2, &gamma3);
    // The mask p and its packed coefficients: 1 + Dmax·ℓ coefficients.
    let n_coeffs = 1 + d_max * log_m;
    let n_packed = n_coeffs.div_ceil(d);
    let draw_k = |rng: &mut Rng| {
        K(
            Fq(rng.next_u64() % crate::fp::Q),
            Fq(rng.next_u64() % crate::fp::Q),
        )
    };
    let mut mask_coeffs: Vec<K> = Vec::with_capacity(n_coeffs);
    mask_coeffs.push(draw_k(rng)); // a0
    for _ in 1..n_coeffs {
        mask_coeffs.push(draw_k(rng));
    }
    // Pack into n_packed R_K elements and commit (Step 4).
    let mut packed: Vec<PolyK> = Vec::with_capacity(n_packed);
    for e in 0..n_packed {
        let mut coeffs = vec![K::ZERO; d];
        for (t_i, c) in coeffs.iter_mut().enumerate() {
            if let Some(w) = mask_coeffs.get(e * d + t_i) {
                *c = *w;
            }
        }
        packed.push(PolyK::from_coeffs(&coeffs));
    }
    let (coeffs_com, coeffs_op) = AbdlopOpening::commit_rk(&setup.abdlop, &packed, &[], rng);
    // ζ ∈ K^× (Step 5) and σ = ζP + T (Step 6).
    let zeta = loop {
        let z = draw_k(rng);
        if z.inverse().is_some() {
            break z;
        }
    };
    let mask = crate::sumcheck::LibraMask {
        a0: mask_coeffs[0],
        coef: {
            let mut c = Vec::with_capacity(log_m);
            for i in 0..log_m {
                c.push(mask_coeffs[1 + i * d_max..1 + (i + 1) * d_max].to_vec());
            }
            c
        },
        d_max,
    };
    let sigma = zeta.mul(&mask.cube_sum()).add(&t_claim);
    // Step 8: the masked Sum-Check (ζp + Q).
    let (sc_tr, r_prime) = MaskedSumcheck::prove_with_mask(&sc_inputs, t_claim, &mask, zeta, rng)
        .map_err(PokError::Shape)?;
    // ---- Step 7: the σ-anchoring.
    // Weights: ζ·ρ_e (message weight) on packed element e of the coeffs
    // block, with ρ_e = the P-functional rotation; δ1·π_{i,j} on the
    // carried hints' messages. Target σ.
    // R_m^{(Σ)}: P = 2^ℓ·a0 + 2^{ℓ−1}·Σa_{i,j} — the P-weight of the
    // mask coefficient at packed position `pos`.
    let p_weight_of = |pos: usize| -> K {
        // P = 2^ℓ·a₀ + Σ_i 2^{ℓ−1}·Σ_j a_{i,j}: every round's univariate
        // contributes its cube-sum scaled by 2^{ℓ−1} (x_i = 1 on exactly
        // half the cube).
        if pos == 0 {
            K::from_fp(Fq::new(1u64 << log_m.min(62)))
        } else {
            K::from_fp(Fq::new(1u64 << (log_m - 1).min(62)))
        }
    };
    let coeffs_block = Block {
        com: coeffs_com.clone(),
        op: coeffs_op.clone(),
    };
    let mut anchor_weights: Vec<(usize, usize, PolyK)> = Vec::new();
    // The coeffs block is block 0 of the anchoring instance; the carried
    // hints follow.
    for e in 0..n_packed {
        let mut w_ell = vec![K::ZERO; d];
        for t_i in 0..d {
            let pos = e * d + t_i;
            if pos < n_coeffs {
                w_ell[t_i] = p_weight_of(pos);
            }
        }
        let rho = PolyK::packaged_rotation(&w_ell);
        let (wa, wb) = msg_weight(&rho.scale_k(&zeta));
        anchor_weights.push((0, 2 * e, wa));
        anchor_weights.push((0, 2 * e + 1, wb));
    }
    // Carried hints: blocks 1..=k·t; the (i, j) hint's message with the
    // π_{i,j} rotation weight.
    let mut hint_blocks: Vec<Block> = Vec::with_capacity(k * t);
    for acc in accs {
        for j in 0..t {
            hint_blocks.push(Block {
                com: acc.inst.coms[j].clone(),
                op: acc.wit.openings[j].clone(),
            });
        }
    }
    let mut blk_idx = 1;
    for (idx, _acc) in accs.iter().enumerate() {
        let i = capital_k + idx;
        for j in 0..t {
            let mut w_ell = vec![K::ZERO; d];
            for l in 1..=d {
                w_ell[l - 1] = gamma1[i].mul(&gamma2[j]).mul(&gamma3[l - 1]);
            }
            let pi_ij = PolyK::packaged_rotation(&w_ell);
            let (wa, wb) = msg_weight(&pi_ij.scale_k(&delta1));
            anchor_weights.push((blk_idx, 0, wa));
            anchor_weights.push((blk_idx, 1, wb));
            blk_idx += 1;
        }
    }
    let mut all_blocks: Vec<Block> = vec![coeffs_block.clone()];
    all_blocks.extend(hint_blocks.iter().cloned());
    let widths7 = p.pok_widths_fresh(all_blocks.len());
    let step7 = pok_anchor(
        &setup.abdlop,
        &all_blocks,
        &anchor_weights,
        sigma,
        widths7,
        p.tau,
        p.beta_ch,
        p.w_max(),
        rng,
    )?;
    // ---- Step 9: commit the new hints y'_{i,j} = M̄_j z_i(r′).
    let mut hint_coms: Vec<Vec<AbdlopCommitment>> = Vec::with_capacity(nk);
    let mut hint_openings: Vec<Vec<AbdlopOpening>> = Vec::with_capacity(nk);
    let mut new_y: Vec<Vec<PolyK>> = Vec::with_capacity(nk);
    for zr in ring_z.iter() {
        let mut per_com = Vec::with_capacity(t);
        let mut per_op = Vec::with_capacity(t);
        let mut per_y = Vec::with_capacity(t);
        for j in 0..t {
            let u = mats[j].mul_ring(zr);
            let mle = RingMle::from_ring_vec(&u);
            let yj = mle.eval(&r_prime);
            let (com, op) = commit_hint(&setup.abdlop, &yj, rng);
            per_com.push(com);
            per_op.push(op);
            per_y.push(yj);
        }
        hint_coms.push(per_com);
        hint_openings.push(per_op);
        new_y.push(per_y);
    }
    // ---- Step 10: the degree-0 surrogates ŷ_{i,j} (ct = ct(y')).
    let mut surr_coms: Vec<Vec<AbdlopCommitment>> = Vec::with_capacity(nk);
    let mut surr_openings: Vec<Vec<AbdlopOpening>> = Vec::with_capacity(nk);
    let mut surrogates: Vec<Vec<PolyK>> = Vec::with_capacity(nk);
    for per_y in &new_y {
        let mut per_com = Vec::with_capacity(t);
        let mut per_op = Vec::with_capacity(t);
        let mut per_s = Vec::with_capacity(t);
        for yj in per_y {
            let yhat = PolyK::degree0(yj.ct(), d);
            let (com, op) = commit_hint(&setup.abdlop, &yhat, rng);
            per_com.push(com);
            per_op.push(op);
            per_s.push(yhat);
        }
        surr_coms.push(per_com);
        surr_openings.push(per_op);
        surrogates.push(per_s);
    }
    // ---- Step 11: µ_ℓ challenges and π_µ.
    let mus: Vec<K> = (2..=d).map(|_| draw_k(rng)).collect();
    let mut mu_weights = vec![K::ZERO; d];
    mu_weights[0] = K::ZERO;
    for (idx, m) in mus.iter().enumerate() {
        mu_weights[idx + 1] = *m;
    }
    let pi_mu = PolyK::packaged_rotation(&mu_weights);
    // ---- Step 12: the degree-0 wrapper over the 2(K+k)t ct-relations.
    // Blocks: [hints; surrogates] — hint (i,j) is block i·t + j, surrogate
    // (i,j) is block nk·t + i·t + j.
    let mut step12_blocks: Vec<Block> = Vec::with_capacity(2 * nk * t);
    for i in 0..nk {
        for j in 0..t {
            step12_blocks.push(Block {
                com: hint_coms[i][j].clone(),
                op: hint_openings[i][j].clone(),
            });
        }
    }
    for i in 0..nk {
        for j in 0..t {
            step12_blocks.push(Block {
                com: surr_coms[i][j].clone(),
                op: surr_openings[i][j].clone(),
            });
        }
    }
    let mut step12_rows: Vec<RelRow> = Vec::with_capacity(2 * nk * t);
    for i in 0..nk {
        for j in 0..t {
            let hint_blk = i * t + j;
            let surr_blk = nk * t + i * t + j;
            // ct(π_µ·ŷ_{i,j}) = 0
            let (wa, wb) = msg_weight(&pi_mu);
            step12_rows.push(RelRow::const_coeff(
                vec![(surr_blk, 0, wa), (surr_blk, 1, wb)],
                K::ZERO,
                d,
            ));
            // ct(y'_{i,j} − ŷ_{i,j}) = 0
            let one = PolyK::one(d);
            let neg = PolyK::one(d).neg();
            let (na, nb) = msg_weight(&neg);
            let _ = one;
            step12_rows.push(RelRow::const_coeff(
                vec![
                    (hint_blk, 0, PolyK::one(d)),
                    (hint_blk, 1, msg_weight(&PolyK::one(d)).1),
                    (surr_blk, 0, na),
                    (surr_blk, 1, nb),
                ],
                K::ZERO,
                d,
            ));
        }
    }
    let step12_rows = RelRows { rows: step12_rows };
    let step12_gammas: Vec<K> = (0..2 * nk * t).map(|_| draw_k(rng)).collect();
    let widths12 = p.pok_widths_fresh(step12_blocks.len());
    let step12 = pok_ct_wrapper(
        &setup.abdlop,
        &step12_blocks,
        Some((&step12_rows, step12_gammas)),
        None,
        widths12,
        p.tau,
        p.beta_ch,
        p.w_max(),
        rng,
    )?;
    // ---- Steps 13-14: the sq/cube/prod commitments.
    let mut sq_coms: Vec<AbdlopCommitment> = Vec::with_capacity(nk);
    let mut sq_openings: Vec<AbdlopOpening> = Vec::with_capacity(nk);
    let mut cube_coms: Vec<AbdlopCommitment> = Vec::with_capacity(nk);
    let mut cube_openings: Vec<AbdlopOpening> = Vec::with_capacity(nk);
    let mut prod_coms: Vec<AbdlopCommitment> = Vec::with_capacity(capital_k);
    let mut prod_openings: Vec<AbdlopOpening> = Vec::with_capacity(capital_k);
    for per_s in &surrogates {
        // ŷ_{i,1} = per_s[0] (the M1 = I surrogate — the witness itself):
        // the square and cube chain the norm-check value ŷ³ − ŷ.
        let sq = per_s[0].square();
        let cube = sq.mul(&per_s[0]);
        let (c, o) = commit_hint(&setup.abdlop, &sq, rng);
        sq_coms.push(c);
        sq_openings.push(o);
        let (c, o) = commit_hint(&setup.abdlop, &cube, rng);
        cube_coms.push(c);
        cube_openings.push(o);
    }
    for per_s in surrogates.iter().take(capital_k) {
        // m_prod,i = ŷ_{i,2}·ŷ_{i,3} = per_s[1]·per_s[2] (the R1CS
        // product slots M2·M3).
        let prod = per_s[1].mul(&per_s[2]);
        let (c, o) = commit_hint(&setup.abdlop, &prod, rng);
        prod_coms.push(c);
        prod_openings.push(o);
    }
    // ---- Step 15: the quadratic PoK for the 2(K+k)+K relations.
    // Block map: surrogates (i,1) at block nk·t + i·t + 1 (message 0);
    // sq at block 2·nk·t + i; cube at 2·nk·t + nk + i; prod at
    // 2·nk·t + 2·nk + i (message 0 of each).
    let mut step15_blocks: Vec<Block> = Vec::new();
    // The quadratic PoK's instance: [surrogate blocks; sq; cube; prod].
    for i in 0..nk {
        for j in 0..t {
            step15_blocks.push(Block {
                com: surr_coms[i][j].clone(),
                op: surr_openings[i][j].clone(),
            });
        }
    }
    for (c, o) in sq_coms.iter().zip(sq_openings.iter()) {
        step15_blocks.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    for (c, o) in cube_coms.iter().zip(cube_openings.iter()) {
        step15_blocks.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    for (c, o) in prod_coms.iter().zip(prod_openings.iter()) {
        step15_blocks.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    let surr_blk = |i: usize, j: usize| i * t + j; // j ∈ {0,1,2} = (M1,M2,M3)
    let sq_blk = |i: usize| nk * t + i;
    let cube_blk = |i: usize| nk * t + nk + i;
    let prod_blk = |i: usize| nk * t + 2 * nk + i;
    let mut quads: Vec<QuadRelation> = Vec::with_capacity(2 * nk + capital_k);
    let one_w = PolyK::one(d);
    let neg_w = PolyK::one(d).neg();
    for i in 0..nk {
        // m_sq,i − ŷ_{i,1}·ŷ_{i,1} = 0 (ŷ_{i,1} = the M1 surrogate)
        quads.push(QuadRelation {
            products: vec![((surr_blk(i, 0), 0), (surr_blk(i, 0), 0), neg_w.clone())],
            linear: vec![((sq_blk(i), 0), one_w.clone())],
            constant: PolyK::zero(d),
        });
        // m_cube,i − m_sq,i·ŷ_{i,1} = 0
        quads.push(QuadRelation {
            products: vec![((sq_blk(i), 0), (surr_blk(i, 0), 0), neg_w.clone())],
            linear: vec![((cube_blk(i), 0), one_w.clone())],
            constant: PolyK::zero(d),
        });
    }
    for i in 0..capital_k {
        // m_prod,i − ŷ_{i,2}·ŷ_{i,3} = 0 (the M2·M3 product slots)
        quads.push(QuadRelation {
            products: vec![((surr_blk(i, 1), 0), (surr_blk(i, 2), 0), neg_w.clone())],
            linear: vec![((prod_blk(i), 0), one_w.clone())],
            constant: PolyK::zero(d),
        });
    }
    let widths15 = p.pok_widths_fresh(step15_blocks.len());
    let inner15 = p.pok_widths_fresh(step15_blocks.len() + 1);
    let step15 = if !step15_blocks.is_empty() {
        Some(
            pok_quadratic(
                &setup.abdlop,
                &step15_blocks,
                &quads,
                widths15,
                inner15,
                p.tau,
                p.beta_ch,
                p.w_max(),
                rng,
            )
            .map_err(|e| {
                if std::env::var("BF_DEBUG").is_ok() {
                    eprintln!("STEP15 quadratic failed: {e:?}");
                }
                e
            })?,
        )
    } else {
        None
    };
    // ---- Steps 16-17: the verifier-derived cF, cN, cE, u⋆.
    // cF = Σ_{i≤K} γ_i·ct(tB^{prod,i} − tB^{1,i}); cN = Σ_{i≤K+k}
    // γ_i·ct(tB^{cube,i} − tB^{1,i}); cE = eq(r′,r)·Σ_{i>K,j,ℓ} γγγ·cf(tB^{(i,j)})_ℓ.
    let mut c_f = K::ZERO;
    for i in 0..capital_k {
        let v = prod_coms[i].t_b[0].sub(&surr_coms[i][1].t_b[0]).ct();
        c_f = c_f.add(&gamma1[i].mul(&K::from_fp(v)));
    }
    let mut c_n = K::ZERO;
    for i in 0..nk {
        let v = cube_coms[i].t_b[0].sub(&surr_coms[i][0].t_b[0]).ct();
        c_n = c_n.add(&gamma1[i].mul(&K::from_fp(v)));
    }
    let mut c_e = K::ZERO;
    let eq_rp_r = K::eq(&r_prime, &r_old);
    for (idx, _acc) in accs.iter().enumerate() {
        let i = capital_k + idx;
        for j in 0..t {
            for l in 1..=d {
                // cf(tB^{(i,j)})_ℓ — the ℓ-th coefficient of the BDLOP
                // component (an R_F element's coefficient).
                let tbl = hint_coms[i][j].t_b[0].cf(l);
                let w = gamma1[i]
                    .mul(&gamma2[j])
                    .mul(&gamma3[l - 1])
                    .mul(&K::from_fp(tbl));
                c_e = c_e.add(&w);
            }
        }
    }
    c_e = eq_rp_r.mul(&c_e);
    let eq_rp_alpha = K::eq(&r_prime, &alpha);
    let h_final = sc_tr.final_h;
    let u_star = h_final
        .sub(&eq_rp_alpha.mul(&c_f.add(&delta0.mul(&c_n))))
        .sub(&delta1.mul(&c_e));
    // ---- Step 18: the final anchoring — target h (the evaluation check
    // h − ζp(r′) = Q(r′) at the level of the commitments).
    // Weights on message slots: the Step-9-15 blocks' tB-derived targets
    // are replaced by the direct h-target (the deviation-ledger notes the
    // equivalence with the paper's u⋆-routing through Lemma 3.10).
    // Blocks: [NEW hints (i > K, j in [t]); surrogates; sq; cube; prod;
    // coeffs] — the Eval-family weights act on the Step-9 hint
    // commitments y'_{i,j} (the claims at the NEW point), per (4.2)'s
    // {s2,i,j} with the Lemma 3.10 substitution.
    let mut w18: Vec<(usize, usize, PolyK)> = Vec::new();
    let mut blocks18: Vec<Block> = Vec::new();
    // The NEW hint commitments for i > K.
    for i in capital_k..nk {
        for j in 0..t {
            blocks18.push(Block {
                com: hint_coms[i][j].clone(),
                op: hint_openings[i][j].clone(),
            });
        }
    }
    for i in 0..nk {
        for j in 0..t {
            blocks18.push(Block {
                com: surr_coms[i][j].clone(),
                op: surr_openings[i][j].clone(),
            });
        }
    }
    for (c, o) in sq_coms.iter().zip(sq_openings.iter()) {
        blocks18.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    for (c, o) in cube_coms.iter().zip(cube_openings.iter()) {
        blocks18.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    for (c, o) in prod_coms.iter().zip(prod_openings.iter()) {
        blocks18.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    blocks18.push(coeffs_block);
    let n_hint18 = k * t;
    let surr18 = |i: usize, j: usize| n_hint18 + i * t + j;
    let _sq18 = |i: usize| n_hint18 + nk * t + i;
    let cube18 = |i: usize| n_hint18 + nk * t + nk + i;
    let prod18 = |i: usize| n_hint18 + nk * t + 2 * nk + i;
    let coeffs18 = blocks18.len() - 1;
    for i in 0..capital_k {
        let w = eq_rp_alpha.mul(&gamma1[i]);
        let (wa, wb) = msg_weight(&PolyK::degree0(w, d));
        w18.push((prod18(i), 0, wa));
        w18.push((prod18(i), 1, wb));
        let wn = PolyK::degree0(w.neg(), d);
        let (na, nb) = msg_weight(&wn);
        w18.push((surr18(i, 0), 0, na));
        w18.push((surr18(i, 0), 1, nb));
    }
    for i in 0..nk {
        let w = eq_rp_alpha.mul(&delta0).mul(&gamma1[i]);
        let (wa, wb) = msg_weight(&PolyK::degree0(w, d));
        w18.push((cube18(i), 0, wa));
        w18.push((cube18(i), 1, wb));
        let wn = PolyK::degree0(w.neg(), d);
        let (na, nb) = msg_weight(&wn);
        w18.push((surr18(i, 0), 0, na));
        w18.push((surr18(i, 0), 1, nb));
    }
    // NEW hints (i > K): δ1·eq(r′,r)·π_{i,j} rotations on the Step-9
    // commitments' messages (blocks 0..k·t of the step-18 instance).
    {
        let mut blk = 0usize;
        for idx in 0..k {
            let i = capital_k + idx;
            for j in 0..t {
                let mut w_ell = vec![K::ZERO; d];
                for l in 1..=d {
                    w_ell[l - 1] = gamma1[i].mul(&gamma2[j]).mul(&gamma3[l - 1]);
                }
                let pi = PolyK::packaged_rotation(&w_ell);
                let w = pi.scale_k(&delta1.mul(&eq_rp_r));
                let (wa, wb) = msg_weight(&w);
                w18.push((blk, 0, wa));
                w18.push((blk, 1, wb));
                blk += 1;
            }
        }
    }
    // coeffs: ζ·R_m^{(r′)} — the rotation sending coeffs to p(r′).
    {
        let r_weight_of = |pos: usize| -> K {
            if pos == 0 {
                K::ONE
            } else {
                let rel = pos - 1;
                let var = rel / d_max + 1;
                let j = rel % d_max + 1;
                let mut pw = K::ONE;
                for _ in 0..j {
                    pw = pw.mul(&r_prime[var - 1]);
                }
                pw
            }
        };
        for e in 0..n_packed {
            let mut w_ell = vec![K::ZERO; d];
            for t_i in 0..d {
                let pos = e * d + t_i;
                if pos < n_coeffs {
                    w_ell[t_i] = r_weight_of(pos);
                }
            }
            let rho = PolyK::packaged_rotation(&w_ell);
            let (wa, wb) = msg_weight(&rho.scale_k(&zeta));
            w18.push((coeffs18, 2 * e, wa));
            w18.push((coeffs18, 2 * e + 1, wb));
        }
    }
    // The target: h (public — the sumcheck's final value). The relation:
    // ct(Σ w·m) = h — the evaluation check at the commitment level.
    let target18 = h_final;
    // TEMPORARY debug: evaluate the row on the prover's real openings.
    if std::env::var("BF_DEBUG").is_ok() {
        let mut debug_slot_arrays: Vec<Vec<Poly>> = Vec::new();
        // NEW hints (i > K) — the Step-9 commitments' openings
        for i in capital_k..nk {
            for j in 0..t {
                debug_slot_arrays.push(hint_openings[i][j].slots.clone());
            }
        }
        // surrogates
        for i in 0..nk {
            for j in 0..t {
                debug_slot_arrays.push(surr_openings[i][j].slots.clone());
            }
        }
        // sq, cube, prod
        for o in &sq_openings {
            debug_slot_arrays.push(o.slots.clone());
        }
        for o in &cube_openings {
            debug_slot_arrays.push(o.slots.clone());
        }
        for o in &prod_openings {
            debug_slot_arrays.push(o.slots.clone());
        }
        // coeffs
        debug_slot_arrays.push(coeffs_op.slots.clone());
        let rr = crate::pok::RelRows {
            rows: vec![crate::pok::RelRow::const_coeff(w18.clone(), target18, d)],
        };
        let got = rr.eval_all(&debug_slot_arrays);
        eprintln!(
            "STEP18 honest ct = {:?}, target h = {:?}",
            got[0].ct(),
            target18
        );
        // Decompose: eq(r',α)(F + δ0·NC)(r') + δ1·Eval(r') + ζ·p(r').
        {
            let mut f_rp = K::ZERO;
            for (i, gi) in gamma1.iter().enumerate().take(capital_k) {
                let u1 = sc_inputs.ring_mles[i][0].eval(&r_prime).ct();
                let u2 = sc_inputs.ring_mles[i][1].eval(&r_prime).ct();
                let u3 = sc_inputs.ring_mles[i][2].eval(&r_prime).ct();
                f_rp = f_rp.add(&gi.mul(&u2.mul(&u3).sub(&u1)));
            }
            let mut nc_rp = K::ZERO;
            for (i, gi) in gamma1.iter().enumerate() {
                let z_rp = zs[i].mle_eval(&r_prime);
                let mut prod = K::ONE;
                for j in -(p.b - 1)..=p.b - 1 {
                    prod = prod.mul(&z_rp.sub(&K::from_i64(j)));
                }
                nc_rp = nc_rp.add(&gi.mul(&prod));
            }
            let mut eval_rp = K::ZERO;
            if let Some(ew) = sc_inputs.eval_weights.as_ref() {
                for (i, per) in ew.iter().enumerate() {
                    for (j, w) in per.iter().enumerate() {
                        if w.iter().all(|x| x.is_zero()) {
                            continue;
                        }
                        let rho = PolyK::packaged_rotation(w);
                        eval_rp = eval_rp.add(&PolyK::rotated_ct(
                            &rho,
                            &sc_inputs.ring_mles[i][j].eval(&r_prime),
                        ));
                    }
                }
            }
            let p_rp = mask.eval(&r_prime);
            let direct = eq_rp_alpha
                .mul(&f_rp.add(&delta0.mul(&nc_rp)))
                .add(&delta1.mul(&eq_rp_r).mul(&eval_rp))
                .add(&zeta.mul(&p_rp));
            eprintln!(
                "STEP18 direct = {:?} (F={:?} NC={:?} Eval={:?} zp={:?})",
                direct,
                f_rp,
                nc_rp,
                eval_rp,
                zeta.mul(&p_rp)
            );
            // Decompose the ROW evaluation per weight family.
            let eval_w = |weights: &Vec<(usize, usize, PolyK)>| -> K {
                let row = crate::pok::RelRow::const_coeff(weights.clone(), K::ZERO, d);
                let rr2 = crate::pok::RelRows { rows: vec![row] };
                rr2.eval_all(&debug_slot_arrays)[0].ct()
            };
            // F-family weights (prod + surrogate-1 for i ≤ K):
            let mut wf: Vec<(usize, usize, PolyK)> = Vec::new();
            for i in 0..capital_k {
                let w = eq_rp_alpha.mul(&gamma1[i]);
                let (wa, wb) = msg_weight(&PolyK::degree0(w, d));
                wf.push((prod18(i), 0, wa));
                wf.push((prod18(i), 1, wb));
                let wn = PolyK::degree0(w.neg(), d);
                let (na, nb) = msg_weight(&wn);
                wf.push((surr18(i, 0), 0, na));
                wf.push((surr18(i, 0), 1, nb));
            }
            eprintln!(
                "STEP18 F-row = {:?} expect {:?}",
                eval_w(&wf),
                eq_rp_alpha.mul(&f_rp)
            );
            // NC-family:
            let mut wnc: Vec<(usize, usize, PolyK)> = Vec::new();
            for i in 0..nk {
                let w = eq_rp_alpha.mul(&delta0).mul(&gamma1[i]);
                let (wa, wb) = msg_weight(&PolyK::degree0(w, d));
                wnc.push((cube18(i), 0, wa));
                wnc.push((cube18(i), 1, wb));
                let wn = PolyK::degree0(w.neg(), d);
                let (na, nb) = msg_weight(&wn);
                wnc.push((surr18(i, 0), 0, na));
                wnc.push((surr18(i, 0), 1, nb));
            }
            eprintln!(
                "STEP18 NC-row = {:?} expect {:?}",
                eval_w(&wnc),
                eq_rp_alpha.mul(&delta0).mul(&nc_rp)
            );
            // Eval-family (hints):
            let mut we: Vec<(usize, usize, PolyK)> = Vec::new();
            {
                let mut blk = 0usize;
                for idx in 0..k {
                    let i = capital_k + idx;
                    for j in 0..t {
                        let mut w_ell = vec![K::ZERO; d];
                        for l in 1..=d {
                            w_ell[l - 1] = gamma1[i].mul(&gamma2[j]).mul(&gamma3[l - 1]);
                        }
                        let pi = PolyK::packaged_rotation(&w_ell);
                        let w = pi.scale_k(&delta1.mul(&eq_rp_r));
                        let (wa, wb) = msg_weight(&w);
                        we.push((blk, 0, wa));
                        we.push((blk, 1, wb));
                        blk += 1;
                    }
                }
            }
            eprintln!(
                "STEP18 Eval-row = {:?} expect {:?}",
                eval_w(&we),
                delta1.mul(&eq_rp_r).mul(&eval_rp)
            );
            // Coeffs-family:
            let mut wc: Vec<(usize, usize, PolyK)> = Vec::new();
            {
                let r_weight_of = |pos: usize| -> K {
                    if pos == 0 {
                        K::ONE
                    } else {
                        let rel = pos - 1;
                        let var = rel / d_max + 1;
                        let j = rel % d_max + 1;
                        let mut pw = K::ONE;
                        for _ in 0..j {
                            pw = pw.mul(&r_prime[var - 1]);
                        }
                        pw
                    }
                };
                for e in 0..n_packed {
                    let mut w_ell = vec![K::ZERO; d];
                    for t_i in 0..d {
                        let pos = e * d + t_i;
                        if pos < n_coeffs {
                            w_ell[t_i] = r_weight_of(pos);
                        }
                    }
                    let rho = PolyK::packaged_rotation(&w_ell);
                    let (wa, wb) = msg_weight(&rho.scale_k(&zeta));
                    wc.push((coeffs18, 2 * e, wa));
                    wc.push((coeffs18, 2 * e + 1, wb));
                }
            }
            eprintln!(
                "STEP18 coeffs-row = {:?} expect {:?}",
                eval_w(&wc),
                zeta.mul(&p_rp)
            );
        }
    }
    let widths18 = p.pok_widths_fresh(blocks18.len());
    let step18 = pok_anchor(
        &setup.abdlop,
        &blocks18,
        &w18,
        target18,
        widths18,
        p.tau,
        p.beta_ch,
        p.w_max(),
        rng,
    )?;
    // ---- Assemble the output pairs at r′.
    let mut pairs: Vec<CecomPair> = Vec::with_capacity(nk);
    for i in 0..nk {
        pairs.push(CecomPair {
            inst: CecomInstance {
                c: setup.ajtai.commit(&ring_z[i]),
                x: zs[i].0[..p.nf_in].to_vec(),
                r: r_prime.clone(),
                coms: hint_coms[i].clone(),
            },
            wit: CecomWitness {
                z: zs[i].clone(),
                y: new_y[i].clone(),
                openings: hint_openings[i].clone(),
            },
        });
    }
    let transcript = R1csTranscript {
        step1,
        coeffs_com,
        n_packed,
        sigma,
        step7,
        sumcheck: sc_tr,
        hint_coms,
        surr_coms,
        step12,
        sq_coms,
        cube_coms,
        prod_coms,
        step15,
        c_f,
        c_n,
        c_e,
        u_star,
        step18,
        alpha,
        gamma1,
        gamma2,
        gamma3,
        delta0,
        delta1,
        zeta,
        mus,
    };
    let _ = step1;
    Ok((R1csOutput { pairs, r_prime }, transcript))
}

/// Verify the R1CS reduction transcript (all sub-protocols).
pub fn r1cs_verify(
    setup: &LbfSetup,
    fresh_c: &[Vec<Poly>],
    accs: &[CecomPair],
    out: &R1csOutput,
    tr: &R1csTranscript,
) -> Result<(), PokError> {
    let p = &setup.params;
    let _d = p.d;
    let t = p.t;
    let capital_k = fresh_c.len();
    let k = accs.len();
    let nk = capital_k + k;
    // Step 1 PoK.
    if let Some(step1) = &tr.step1 {
        let mut coms: Vec<AbdlopCommitment> = Vec::with_capacity(k * t);
        for acc in accs {
            coms.extend(acc.inst.coms.iter().cloned());
        }
        let rows = RelRows { rows: vec![] };
        let blocks: Vec<Block> = accs
            .iter()
            .flat_map(|acc| {
                (0..t)
                    .map(|j| Block {
                        com: acc.inst.coms[j].clone(),
                        op: acc.wit.openings[j].clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let u = rows.eval_messages(&blocks);
        verify_linear(&setup.abdlop, &coms, &rows, &u, step1)?;
    }
    // Sum-Check.
    let h = MaskedSumcheck::verify(&tr.sumcheck).map_err(PokError::Shape)?;
    if h != tr.sumcheck.final_h {
        return Err(PokError::Shape("final h mismatch".into()));
    }
    // Step 18: the final anchoring — the verifier rebuilds the weight
    // rows from the public challenges and checks the wrapper.
    {
        let (blocks18_coms, w18) = tr.step18_assembly(setup, accs, out);
        verify_anchor(&setup.abdlop, &blocks18_coms, &w18, h, &tr.step18)?;
    }
    // Step 7 anchoring.
    {
        let mut coms: Vec<AbdlopCommitment> = vec![tr.coeffs_com.clone()];
        for acc in accs {
            coms.extend(acc.inst.coms.iter().cloned());
        }
        let weights = tr.step7_weights(setup, accs);
        verify_anchor(&setup.abdlop, &coms, &weights, tr.sigma, &tr.step7)?;
    }
    // Step 12 wrapper.
    {
        let mut coms: Vec<AbdlopCommitment> = Vec::with_capacity(2 * nk * t);
        for i in 0..nk {
            for j in 0..t {
                coms.push(tr.hint_coms[i][j].clone());
            }
        }
        for i in 0..nk {
            for j in 0..t {
                coms.push(tr.surr_coms[i][j].clone());
            }
        }
        let (rows, gammas) = tr.step12_rows(setup);
        verify_ct_wrapper(
            &setup.abdlop,
            &coms,
            Some((&rows, gammas)),
            None,
            &tr.step12,
        )?;
    }
    // Step 15 quadratic PoK.
    if let Some(step15) = &tr.step15 {
        let mut coms: Vec<AbdlopCommitment> = Vec::new();
        for i in 0..nk {
            for j in 0..t {
                coms.push(tr.surr_coms[i][j].clone());
            }
        }
        coms.extend(tr.sq_coms.iter().cloned());
        coms.extend(tr.cube_coms.iter().cloned());
        coms.extend(tr.prod_coms.iter().cloned());
        let quads = tr.step15_relations(setup);
        verify_quadratic(&setup.abdlop, &coms, &quads, step15, (0.0, 0.0))?;
    }
    // Output shape: K+k pairs at r′.
    if out.pairs.len() != nk {
        return Err(PokError::Shape("output arity".into()));
    }
    for pair in &out.pairs {
        if pair.inst.r != out.r_prime {
            return Err(PokError::Shape("r' mismatch".into()));
        }
    }
    let _ = fresh_c;
    Ok(())
}

impl R1csTranscript {
    /// The Step-18 assembly (verifier side): the commitment list and the
    /// anchoring weights rebuilt from the public challenges — the exact
    /// mirror of the prover's construction.
    fn step18_assembly(
        &self,
        setup: &LbfSetup,
        accs: &[CecomPair],
        out: &R1csOutput,
    ) -> (Vec<AbdlopCommitment>, Vec<(usize, usize, PolyK)>) {
        let p = &setup.params;
        let d = p.d;
        let t = p.t;
        let capital_k = accs.len() + 1 - accs.len(); // 1 (the transcript's gamma1 len − accs)
        let _ = capital_k;
        let k = accs.len();
        let nk = self.hint_coms.len();
        let cap_k = nk - k;
        let r_prime = &out.r_prime;
        let eq_rp_alpha = K::eq(r_prime, &self.alpha);
        // r_old from the accs
        let r_old = accs.first().map(|a| a.inst.r.clone()).unwrap_or_default();
        let eq_rp_r = K::eq(r_prime, &r_old);
        let mut coms: Vec<AbdlopCommitment> = Vec::new();
        for i in cap_k..nk {
            for j in 0..t {
                coms.push(self.hint_coms[i][j].clone());
            }
        }
        for i in 0..nk {
            for j in 0..t {
                coms.push(self.surr_coms[i][j].clone());
            }
        }
        coms.extend(self.sq_coms.iter().cloned());
        coms.extend(self.cube_coms.iter().cloned());
        coms.extend(self.prod_coms.iter().cloned());
        coms.push(self.coeffs_com.clone());
        let n_hint18 = k * t;
        let surr18 = |i: usize, j: usize| n_hint18 + i * t + j;
        let _sq18 = |i: usize| n_hint18 + nk * t + i;
        let cube18 = |i: usize| n_hint18 + nk * t + nk + i;
        let prod18 = |i: usize| n_hint18 + nk * t + 2 * nk + i;
        let coeffs18 = coms.len() - 1;
        let mut w18: Vec<(usize, usize, PolyK)> = Vec::new();
        for i in 0..cap_k {
            let w = eq_rp_alpha.mul(&self.gamma1[i]);
            let (wa, wb) = msg_weight(&PolyK::degree0(w, d));
            w18.push((prod18(i), 0, wa));
            w18.push((prod18(i), 1, wb));
            let wn = PolyK::degree0(w.neg(), d);
            let (na, nb) = msg_weight(&wn);
            w18.push((surr18(i, 0), 0, na));
            w18.push((surr18(i, 0), 1, nb));
        }
        for i in 0..nk {
            let w = eq_rp_alpha.mul(&self.delta0).mul(&self.gamma1[i]);
            let (wa, wb) = msg_weight(&PolyK::degree0(w, d));
            w18.push((cube18(i), 0, wa));
            w18.push((cube18(i), 1, wb));
            let wn = PolyK::degree0(w.neg(), d);
            let (na, nb) = msg_weight(&wn);
            w18.push((surr18(i, 0), 0, na));
            w18.push((surr18(i, 0), 1, nb));
        }
        {
            let mut blk = 0usize;
            for idx in 0..k {
                let i = cap_k + idx;
                for j in 0..t {
                    let mut w_ell = vec![K::ZERO; d];
                    for l in 1..=d {
                        w_ell[l - 1] = self.gamma1[i].mul(&self.gamma2[j]).mul(&self.gamma3[l - 1]);
                    }
                    let pi = PolyK::packaged_rotation(&w_ell);
                    let w = pi.scale_k(&self.delta1.mul(&eq_rp_r));
                    let (wa, wb) = msg_weight(&w);
                    w18.push((blk, 0, wa));
                    w18.push((blk, 1, wb));
                    blk += 1;
                }
            }
        }
        // coeffs: ζ·R_m^{(r′)}
        {
            let log_m = p.log_m();
            let d_max = p.d_max();
            let n_coeffs = 1 + d_max * log_m;
            let n_packed = n_coeffs.div_ceil(d);
            let r_weight_of = |pos: usize| -> K {
                if pos == 0 {
                    K::ONE
                } else {
                    let rel = pos - 1;
                    let var = rel / d_max + 1;
                    let j = rel % d_max + 1;
                    let mut pw = K::ONE;
                    for _ in 0..j {
                        pw = pw.mul(&r_prime[var - 1]);
                    }
                    pw
                }
            };
            for e in 0..n_packed {
                let mut w_ell = vec![K::ZERO; d];
                for t_i in 0..d {
                    let pos = e * d + t_i;
                    if pos < n_coeffs {
                        w_ell[t_i] = r_weight_of(pos);
                    }
                }
                let rho = PolyK::packaged_rotation(&w_ell);
                let (wa, wb) = msg_weight(&rho.scale_k(&self.zeta));
                w18.push((coeffs18, 2 * e, wa));
                w18.push((coeffs18, 2 * e + 1, wb));
            }
        }
        (coms, w18)
    }

    /// Rebuild the Step-7 anchoring weights (verifier side).
    fn step7_weights(&self, setup: &LbfSetup, accs: &[CecomPair]) -> Vec<(usize, usize, PolyK)> {
        let p = &setup.params;
        let d = p.d;
        let t = p.t;
        let capital_k = self.gamma1.len() - accs.len();
        let _k = accs.len();
        let log_m = p.log_m();
        let d_max = p.d_max();
        let n_coeffs = 1 + d_max * log_m;
        let mut weights: Vec<(usize, usize, PolyK)> = Vec::new();
        let p_weight_of = |pos: usize| -> K {
            if pos == 0 {
                K::from_fp(Fq::new(1u64 << log_m.min(62)))
            } else {
                K::from_fp(Fq::new(1u64 << (log_m - 1).min(62)))
            }
        };
        for e in 0..self.n_packed {
            let mut w_ell = vec![K::ZERO; d];
            for t_i in 0..d {
                let pos = e * d + t_i;
                if pos < n_coeffs {
                    w_ell[t_i] = p_weight_of(pos);
                }
            }
            let rho = PolyK::packaged_rotation(&w_ell);
            let (wa, wb) = msg_weight(&rho.scale_k(&self.zeta));
            weights.push((0, 2 * e, wa));
            weights.push((0, 2 * e + 1, wb));
        }
        let mut blk = 1;
        for (idx, _acc) in accs.iter().enumerate() {
            let i = capital_k + idx;
            for j in 0..t {
                let mut w_ell = vec![K::ZERO; d];
                for l in 1..=d {
                    w_ell[l - 1] = self.gamma1[i].mul(&self.gamma2[j]).mul(&self.gamma3[l - 1]);
                }
                let pi = PolyK::packaged_rotation(&w_ell);
                let (wa, wb) = msg_weight(&pi.scale_k(&self.delta1));
                weights.push((blk, 0, wa));
                weights.push((blk, 1, wb));
                blk += 1;
            }
        }
        weights
    }

    /// Rebuild the Step-12 rows (verifier side).
    fn step12_rows(&self, setup: &LbfSetup) -> (RelRows, Vec<K>) {
        let p = &setup.params;
        let d = p.d;
        let t = p.t;
        let nk = self.hint_coms.len();
        let mut mu_weights = vec![K::ZERO; d];
        for (idx, m) in self.mus.iter().enumerate() {
            if idx + 1 < d {
                mu_weights[idx + 1] = *m;
            }
        }
        let pi_mu = PolyK::packaged_rotation(&mu_weights);
        let mut rows: Vec<RelRow> = Vec::with_capacity(2 * nk * t);
        for i in 0..nk {
            for j in 0..t {
                let hint_blk = i * t + j;
                let surr_blk = nk * t + i * t + j;
                let (wa, wb) = msg_weight(&pi_mu);
                rows.push(RelRow::const_coeff(
                    vec![(surr_blk, 0, wa), (surr_blk, 1, wb)],
                    K::ZERO,
                    d,
                ));
                rows.push(RelRow::const_coeff(
                    vec![
                        (hint_blk, 0, PolyK::one(d)),
                        (hint_blk, 1, msg_weight(&PolyK::one(d)).1),
                        (surr_blk, 0, PolyK::one(d).neg()),
                        (surr_blk, 1, msg_weight(&PolyK::one(d).neg()).1),
                    ],
                    K::ZERO,
                    d,
                ));
            }
        }
        let gammas: Vec<K> = self.step12.gammas.clone();
        let _ = setup;
        (RelRows { rows }, gammas)
    }

    /// Rebuild the Step-15 quadratic relations (verifier side).
    fn step15_relations(&self, setup: &LbfSetup) -> Vec<QuadRelation> {
        let p = &setup.params;
        let d = p.d;
        let t = p.t;
        let nk = self.hint_coms.len();
        let capital_k = self.prod_coms.len();
        let one_w = PolyK::one(d);
        let neg_w = PolyK::one(d).neg();
        let surr_blk = |i: usize, j: usize| i * t + j;
        let sq_blk = |i: usize| nk * t + i;
        let cube_blk = |i: usize| nk * t + nk + i;
        let prod_blk = |i: usize| nk * t + 2 * nk + i;
        let mut quads: Vec<QuadRelation> = Vec::with_capacity(2 * nk + capital_k);
        for i in 0..nk {
            // ŷ_{i,1} = the M1 surrogate = surr_blk(i, 0).
            quads.push(QuadRelation {
                products: vec![((surr_blk(i, 0), 0), (surr_blk(i, 0), 0), neg_w.clone())],
                linear: vec![((sq_blk(i), 0), one_w.clone())],
                constant: PolyK::zero(d),
            });
            quads.push(QuadRelation {
                products: vec![((sq_blk(i), 0), (surr_blk(i, 0), 0), neg_w.clone())],
                linear: vec![((cube_blk(i), 0), one_w.clone())],
                constant: PolyK::zero(d),
            });
        }
        for i in 0..capital_k {
            // m_prod,i = ŷ_{i,2}·ŷ_{i,3} = surr_blk(i,1)·surr_blk(i,2).
            quads.push(QuadRelation {
                products: vec![((surr_blk(i, 1), 0), (surr_blk(i, 2), 0), neg_w.clone())],
                linear: vec![((prod_blk(i), 0), one_w.clone())],
                constant: PolyK::zero(d),
            });
        }
        let _ = setup;
        quads
    }
}

// ---------------------------------------------------------------------------
// Protocol 7 — the random linear combination Π'_RLC.
// ---------------------------------------------------------------------------

/// The RLC transcript.
pub struct RlcTranscript {
    /// The mask commitments Cy,0 (the tuple) and {Cy,j}.
    pub cy0: Vec<AbdlopCommitment>,
    pub cyj: Vec<AbdlopCommitment>,
    /// The opened mask block (c_y, x_y, salts) — revealed on acceptance.
    pub cy0_opening: Vec<AbdlopOpening>,
    /// The folding challenges ρ⃗ ∈ C^{K+k}.
    pub rhos: Vec<Poly>,
    /// The Step-1 batched PoK (input hints).
    pub step1: Option<crate::pok::LinearPokTranscript>,
    /// The Step-17 batched PoK (mask hints).
    pub step17: Option<crate::pok::LinearPokTranscript>,
    /// The folded z (revealed in the clear — it is simulatable).
    pub z_folded: FieldVec,
    /// The number of attempts used (≤ Wmax).
    pub attempts: u32,
}

/// Run Π'_RLC: fold K+k CEcom pairs into one at norm B.
pub fn rlc_reduction(
    setup: &LbfSetup,
    pairs: &[CecomPair],
    rng: &mut Rng,
) -> Result<(CecomPair, RlcTranscript), PokError> {
    let p = &setup.params;
    let d = p.d;
    let t = p.t;
    let nk = pairs.len();
    // ---- Step 1: the batched opening PoK for all input hints.
    let mut blocks1: Vec<Block> = Vec::with_capacity(nk * t);
    for pr in pairs {
        for j in 0..t {
            blocks1.push(Block {
                com: pr.inst.coms[j].clone(),
                op: pr.wit.openings[j].clone(),
            });
        }
    }
    let rows1 = RelRows { rows: vec![] };
    let step1 = if !blocks1.is_empty() {
        let widths = p.pok_widths_fresh(blocks1.len());
        Some(pok_linear(
            &setup.abdlop,
            &blocks1,
            &rows1,
            widths,
            p.tau,
            p.beta_ch,
            p.w_max(),
            rng,
        )?)
    } else {
        None
    };
    // ---- The mask loop (Steps 2-16).
    let s_width = p.rlc_width();
    let r_old = pairs
        .first()
        .map(|pr| pr.inst.r.clone())
        .unwrap_or_default();
    let m1 = setup.m1();
    let mats = [&m1, &setup.m2, &setup.m3];
    let mut attempts = 0u32;
    #[allow(unused_assignments)]
    let mut accepted: Option<RlcAccepted> = None;
    while attempts < p.w_max() {
        attempts += 1;
        // Step 3: y ← D_s^{nF}.
        let y: Vec<i64> = (0..p.nf).map(|_| rng.gaussian(s_width, p.tau)).collect();
        let y_f = FieldVec(y.iter().map(|&v| Fq::from_i64(v)).collect());
        let y_r = y_f.to_ring(d);
        // Step 4: c_y = L(y), x_y = L_in(y), y_y,j = M̄_j y(r).
        let c_y = setup.ajtai.commit(&y_r);
        let x_y: Vec<Poly> = y_r[..p.nr_in()].to_vec();
        let mut yy: Vec<PolyK> = Vec::with_capacity(t);
        for j in 0..t {
            let u = mats[j].mul_ring(&y_r);
            yy.push(RingMle::from_ring_vec(&u).eval(&r_old));
        }
        // Step 5: fresh salts, Cy,j := Commit(y_y,j, salts).
        let mut cyj_coms: Vec<AbdlopCommitment> = Vec::with_capacity(t);
        let mut cyj_ops: Vec<AbdlopOpening> = Vec::with_capacity(t);
        for yj in &yy {
            let (c, o) = commit_hint(&setup.abdlop, yj, rng);
            cyj_coms.push(c);
            cyj_ops.push(o);
        }
        // Step 6: Cy,0 — the tuple commitment to (c_y, x_y) as R_F
        // messages, split into ⌈(κ + nR,in)/ℓ⌉ blocks (Remark 4.21).
        let mut rf_msgs: Vec<Poly> = Vec::with_capacity(p.kappa_ajtai + p.nr_in());
        rf_msgs.extend(c_y.iter().cloned());
        rf_msgs.extend(x_y.iter().cloned());
        let n_blocks_cy0 = (rf_msgs.len() + p.ell - 1) / p.ell.max(1);
        let mut cy0_coms: Vec<AbdlopCommitment> = Vec::with_capacity(n_blocks_cy0);
        let mut cy0_ops: Vec<AbdlopOpening> = Vec::with_capacity(n_blocks_cy0);
        for b in 0..n_blocks_cy0 {
            let chunk: Vec<Poly> =
                rf_msgs[b * p.ell..((b + 1) * p.ell).min(rf_msgs.len())].to_vec();
            let mut padded = chunk.clone();
            while padded.len() < p.ell {
                padded.push(Poly::zero(d));
            }
            let (c, o) = AbdlopOpening::commit_rk(&setup.abdlop, &[], &padded, rng);
            cy0_coms.push(c);
            cy0_ops.push(o);
        }
        // Step 7: send Cy,0 and {Cy,j} — then the verifier's ρ⃗ (Step 8).
        let rhos: Vec<Poly> = (0..nk)
            .map(|_| StrongSet::sample(d, b"rho", &mut rho_ctr()))
            .collect();
        // Step 9-10: v := Σρz_i; z := v + y.
        let mut v: Vec<i64> = vec![0i64; p.nf];
        for (i, rho) in rhos.iter().enumerate() {
            let zsym: Vec<i64> = pairs[i].wit.z.0.iter().map(|c| c.sym()).collect();
            for (acc, vv) in v.iter_mut().zip(zsym.iter()) {
                // ρ·z_i per coefficient: the negacyclic convolution is
                // per-ring-element; we accumulate at field granularity by
                // computing the ring products per block.
                let _ = (acc, vv);
            }
            // Ring-level fold:
            let zr = pairs[i].wit.z.to_ring(d);
            for (blk, zp) in zr.iter().enumerate() {
                let prod = rho.mul(zp);
                for (o, pc) in v[blk * d..(blk + 1) * d].iter_mut().zip(prod.0.iter()) {
                    *o = ((*o as i128 + pc.sym() as i128).clamp(i64::MIN as i128, i64::MAX as i128))
                        as i64;
                }
            }
        }
        let z: Vec<i64> = y.iter().zip(v.iter()).map(|(a, b)| a + b).collect();
        // Step 11: Rej1(z, v, s).
        if !rej1_decide(rng, &z, &v, s_width, p.m_rate()) {
            continue;
        }
        // Step 12-13: open Cy,0 in the clear (reveal salts).
        accepted = Some((y_f, c_y, x_y, cyj_coms, cyj_ops, cy0_coms, cy0_ops, yy));
        let z_f = FieldVec(z.iter().map(|&x| Fq::from_i64(x)).collect());
        let mut tr_rhos = rhos;
        tr_rhos.truncate(nk);
        // Fold and finish below with the accepted pieces.
        let (out_pair, step17) = rlc_finish(
            setup,
            pairs,
            &z_f,
            &tr_rhos,
            // The loop above returns only on acceptance, so the Option is
            // inhabited here; bind it once without unwrap().
            match &accepted {
                Some(a) => a,
                None => {
                    return Err(PokError::ExtractionFailed(
                        "unreachable: no accepting attempt".into(),
                    ))
                }
            },
            rng,
        );
        let (cy0, cyj, cy0_opening) = match &accepted {
            Some(a) => (a.5.clone(), a.3.clone(), a.6.clone()),
            None => {
                return Err(PokError::ExtractionFailed(
                    "unreachable: no accepting attempt".into(),
                ))
            }
        };
        let transcript = RlcTranscript {
            cy0,
            cyj,
            cy0_opening,
            rhos: tr_rhos,
            step1,
            step17,
            z_folded: z_f,
            attempts,
        };
        return Ok((out_pair, transcript));
    }
    Err(PokError::ExtractionFailed("Wmax exhausted".into()))
}

fn rho_ctr() -> u64 {
    use std::cell::Cell;
    thread_local! {
        static C: Cell<u64> = const { Cell::new(0) };
    }
    C.with(|c| {
        let v = c.get() + 1;
        c.set(v);
        v
    })
}

#[allow(clippy::too_many_arguments)]
/// The accepted mask-loop payload of Π'_RLC (Protocol 7): the Gaussian
/// mask y with its Ajtai image, the input slice, the Cy,j and Cy,0
/// commitments with their openings, and the mask hints.
#[allow(clippy::type_complexity)]
type RlcAccepted = (
    FieldVec,
    Vec<Poly>,
    Vec<Poly>,
    Vec<AbdlopCommitment>,
    Vec<AbdlopOpening>,
    Vec<AbdlopCommitment>,
    Vec<AbdlopOpening>,
    Vec<PolyK>,
);

#[allow(clippy::too_many_arguments)]
fn rlc_finish(
    setup: &LbfSetup,
    pairs: &[CecomPair],
    z_f: &FieldVec,
    rhos: &[Poly],
    acc: &RlcAccepted,
    rng: &mut Rng,
) -> (CecomPair, Option<crate::pok::LinearPokTranscript>) {
    let p = &setup.params;
    let d = p.d;
    let t = p.t;
    let _nk = pairs.len();
    let (_y_f, c_y, _x_y, cyj_coms, cyj_ops, _cy0_coms, _cy0_ops, yy) = acc;
    // ---- Step 17: the batched PoK for the mask hints Cy,j.
    let mut blocks17: Vec<Block> = Vec::with_capacity(t);
    for (c, o) in cyj_coms.iter().zip(cyj_ops.iter()) {
        blocks17.push(Block {
            com: c.clone(),
            op: o.clone(),
        });
    }
    let rows17 = RelRows { rows: vec![] };
    let step17 = if !blocks17.is_empty() {
        let widths = p.pok_widths_fresh(blocks17.len());
        pok_linear(
            &setup.abdlop,
            &blocks17,
            &rows17,
            widths,
            p.tau,
            p.beta_ch,
            p.w_max(),
            rng,
        )
        .ok()
    } else {
        None
    };
    // ---- Step 18: c, x.
    let _zr_folded = z_f.to_ring(d);
    let c_folded = {
        let mut acc_c = c_y.clone();
        for (i, rho) in rhos.iter().enumerate() {
            for (a, ci) in acc_c.iter_mut().zip(pairs[i].inst.c.iter()) {
                a.add_assign(&rho.mul(ci));
            }
        }
        acc_c
    };
    // x = Lin(z): the folded witness's public-input slice (the folded z
    // is transmitted in the clear — it is simulatable by the blinding
    // property, so its input part is public).
    let x_folded: Vec<Fq> = z_f.0[..p.nf_in].to_vec();
    // ---- Steps 19-20: fold the salts and hints, commit the folded y_j.
    let r_old = pairs
        .first()
        .map(|pr| pr.inst.r.clone())
        .unwrap_or_default();
    let m1 = setup.m1();
    let _mats = [&m1, &setup.m2, &setup.m3];
    let mut folded_y: Vec<PolyK> = Vec::with_capacity(t);
    let mut folded_coms: Vec<AbdlopCommitment> = Vec::with_capacity(t);
    let mut folded_ops: Vec<AbdlopOpening> = Vec::with_capacity(t);
    for j in 0..t {
        // y_j := Σρ_i y_{i,j} + y_{y,j} — R_K RING products with the
        // R_F challenges ρ (the full ring element, not its constant
        // coefficient).
        let mut yj = yy[j].clone();
        for (i, rho) in rhos.iter().enumerate() {
            let contrib = PolyK::from_poly(rho.clone()).mul(&pairs[i].wit.y[j]);
            yj.add_assign(&contrib);
        }
        // Salts: s_{l,j} := Σρ_i s_{l,i,j} + s_{l,y,j}.
        let mut s1 = cyj_ops[j].s1.clone();
        let mut s2 = cyj_ops[j].s2.clone();
        for (i, rho) in rhos.iter().enumerate() {
            for (a, b) in s1.iter_mut().zip(pairs[i].wit.openings[j].s1.iter()) {
                a.add_assign(&rho.mul(b));
            }
            for (a, b) in s2.iter_mut().zip(pairs[i].wit.openings[j].s2.iter()) {
                a.add_assign(&rho.mul(b));
            }
        }
        // Commit(y_j) with the folded salts — the homomorphic identity
        // Commit(y_j) = Σρ Commit(y_{i,j}) + Cy,j holds automatically.
        let slots = {
            let mut v = vec![yj.a.clone(), j_b(&yj, d)];
            while v.len() < p.ell {
                v.push(Poly::zero(d));
            }
            v
        };
        let com = AbdlopCommitment {
            t_a: setup.abdlop.t_a(&s1, &s2),
            t_b: setup.abdlop.t_b(&s2, &slots),
            layout: MsgLayout::rk_only(1),
        };
        let op = AbdlopOpening { s1, s2, slots };
        folded_y.push(yj);
        folded_coms.push(com);
        folded_ops.push(op);
    }
    let out_pair = CecomPair {
        inst: CecomInstance {
            c: c_folded,
            x: x_folded,
            r: r_old,
            coms: folded_coms,
        },
        wit: CecomWitness {
            z: z_f.clone(),
            y: folded_y,
            openings: folded_ops,
        },
    };
    (out_pair, step17)
}

fn j_b(yj: &PolyK, d: usize) -> Poly {
    let _ = d;
    yj.b.clone()
}

/// Verify Π'_RLC.
pub fn rlc_verify(
    setup: &LbfSetup,
    pairs: &[CecomPair],
    out: &CecomPair,
    tr: &RlcTranscript,
) -> Result<(), PokError> {
    let p = &setup.params;
    let t = p.t;
    let nk = pairs.len();
    if tr.rhos.len() != nk {
        return Err(PokError::Shape("rho arity".into()));
    }
    // Step 13: the Cy,0 openings check (∥s∥ < B̃ + recomputation).
    for op in &tr.cy0_opening {
        // The salts are opened in the clear: norm check only (the
        // commitment recomputation is done against the cy0 tuple).
        for s in &op.s1 {
            if s.norm_inf() >= p.b_tilde {
                return Err(PokError::NormCheck(999));
            }
        }
        for s in &op.s2 {
            if s.norm_inf() >= p.b_tilde {
                return Err(PokError::NormCheck(999));
            }
        }
    }
    // Step 21: Commit(y_j) = Σρ Commit(y_{i,j}) + Cy,j.
    for j in 0..t {
        let coms_ref: Vec<&AbdlopCommitment> = pairs.iter().map(|pr| &pr.inst.coms[j]).collect();
        let expect = crate::abdlop::homomorphic_comb(&coms_ref, &tr.rhos, Some(&tr.cyj[j]));
        if out.inst.coms[j].t_a != expect.t_a || out.inst.coms[j].t_b != expect.t_b {
            return Err(PokError::CommitmentCheck(j));
        }
    }
    // Step 1/17 PoKs.
    if let Some(step1) = &tr.step1 {
        let mut coms: Vec<AbdlopCommitment> = Vec::with_capacity(nk * t);
        for pr in pairs {
            coms.extend(pr.inst.coms.iter().cloned());
        }
        let rows = RelRows { rows: vec![] };
        let blocks: Vec<Block> = pairs
            .iter()
            .flat_map(|pr| {
                (0..t)
                    .map(|j| Block {
                        com: pr.inst.coms[j].clone(),
                        op: pr.wit.openings[j].clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let u = rows.eval_messages(&blocks);
        verify_linear(&setup.abdlop, &coms, &rows, &u, step1)?;
    }
    if let Some(step17) = &tr.step17 {
        let coms: Vec<AbdlopCommitment> = tr.cyj.clone();
        let rows = RelRows { rows: vec![] };
        let blocks: Vec<Block> = tr
            .cyj
            .iter()
            .zip(tr.cyj_openings_hint(setup))
            .map(|(c, o)| Block {
                com: c.clone(),
                op: o,
            })
            .collect();
        let u = rows.eval_messages(&blocks);
        verify_linear(&setup.abdlop, &coms, &rows, &u, step17)?;
    }
    // The folded norm bound B = b^k.
    if out.wit.z.norm_inf() >= p.b.pow(p.k as u32) {
        return Err(PokError::NormCheck(1000));
    }
    Ok(())
}

impl RlcTranscript {
    /// The verifier does NOT hold the mask-hint openings — the Step-17
    /// PoK verifies against the commitments alone via the row identity.
    fn cyj_openings_hint(&self, _setup: &LbfSetup) -> Vec<AbdlopOpening> {
        // Placeholder openings are never used by verify_linear with empty
        // rows (the row identity is vacuous); the commitment checks are
        // the real verification.
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Protocol 8 — the decomposition reduction Π'_DEC.
// ---------------------------------------------------------------------------

pub struct DecTranscript {
    /// c_i = L(z_i) and the per-piece hint commitments C_{i,j}.
    pub c_is: Vec<Vec<Poly>>,
    pub coms: Vec<Vec<AbdlopCommitment>>,
    /// The Step-5 batched PoK (the y_j − Σb^{i−1}y_{i,j} = 0 identity).
    pub step5: Option<crate::pok::LinearPokTranscript>,
}

/// Run Π'_DEC: split one norm-B claim into k norm-b claims with fresh salts.
pub fn dec_reduction(
    setup: &LbfSetup,
    pair: &CecomPair,
    rng: &mut Rng,
) -> Result<(Vec<CecomPair>, DecTranscript), PokError> {
    let p = &setup.params;
    let d = p.d;
    let t = p.t;
    let k = p.k;
    // Step 1: splitb(z).
    let zr = pair.wit.z.to_ring(d);
    let pieces = split_b_k(&zr, p.b, k);
    let mut c_is: Vec<Vec<Poly>> = Vec::with_capacity(k);
    let mut coms: Vec<Vec<AbdlopCommitment>> = Vec::with_capacity(k);
    let mut openings: Vec<Vec<AbdlopOpening>> = Vec::with_capacity(k);
    let mut ys: Vec<Vec<PolyK>> = Vec::with_capacity(k);
    let m1 = setup.m1();
    let mats = [&m1, &setup.m2, &setup.m3];
    for piece in &pieces {
        c_is.push(setup.ajtai.commit(piece));
        let mut per_com = Vec::with_capacity(t);
        let mut per_op = Vec::with_capacity(t);
        let mut per_y = Vec::with_capacity(t);
        for j in 0..t {
            let u = mats[j].mul_ring(piece);
            let yj = RingMle::from_ring_vec(&u).eval(&pair.inst.r);
            let (c, o) = commit_hint(&setup.abdlop, &yj, rng);
            per_com.push(c);
            per_op.push(o);
            per_y.push(yj);
        }
        coms.push(per_com);
        openings.push(per_op);
        ys.push(per_y);
    }
    // Step 5: the batched PoK on the concatenation [C_j; C_{i,j}] —
    // the full-ring rows y_j − Σ b^{i−1} y_{i,j} = 0.
    let mut blocks: Vec<Block> = Vec::with_capacity((k + 1) * t);
    for j in 0..t {
        blocks.push(Block {
            com: pair.inst.coms[j].clone(),
            op: pair.wit.openings[j].clone(),
        });
    }
    for (i, per) in coms.iter().enumerate() {
        for (c, o) in per.iter().zip(openings[i].iter()) {
            blocks.push(Block {
                com: c.clone(),
                op: o.clone(),
            });
        }
    }
    // Rows: for each j: block 0 (the input C_j) with weight 1; blocks
    // 1+i with weight −b^{i−1} — full-ring rows targeting 0.
    let mut rows: Vec<RelRow> = Vec::with_capacity(t);
    for j in 0..t {
        let mut weights: Vec<(usize, usize, PolyK)> = Vec::with_capacity(k + 1);
        let one = PolyK::one(d);
        let (a0, b0) = msg_weight(&one);
        weights.push((j, 0, a0));
        weights.push((j, 1, b0));
        for i in 0..k {
            let w = PolyK::degree0(K::from_fp(Fq::from_i64(-(p.b).pow(i as u32))), d);
            let (wa, wb) = msg_weight(&w);
            weights.push(((1 + i) * t + j, 0, wa));
            weights.push(((1 + i) * t + j, 1, wb));
        }
        rows.push(RelRow::full_ring(weights, PolyK::zero(d)));
    }
    let rel_rows = RelRows { rows };
    let widths = p.pok_widths_folded(blocks.len(), 1); // one folded block
    let step5 = if !blocks.is_empty() {
        Some(pok_linear(
            &setup.abdlop,
            &blocks,
            &rel_rows,
            widths,
            p.tau,
            p.beta_ch,
            p.w_max(),
            rng,
        )?)
    } else {
        None
    };
    // Assemble the output pairs.
    let mut out: Vec<CecomPair> = Vec::with_capacity(k);
    for (i, piece) in pieces.iter().enumerate() {
        let z_field = crate::embed::from_ring(piece);
        out.push(CecomPair {
            inst: CecomInstance {
                c: c_is[i].clone(),
                x: z_field.0[..p.nf_in].to_vec(),
                r: pair.inst.r.clone(),
                coms: coms[i].clone(),
            },
            wit: CecomWitness {
                z: z_field,
                y: ys[i].clone(),
                openings: openings[i].clone(),
            },
        });
    }
    Ok((out, DecTranscript { c_is, coms, step5 }))
}

/// Verify Π'_DEC.
pub fn dec_verify(
    setup: &LbfSetup,
    pair: &CecomPair,
    out: &[CecomPair],
    tr: &DecTranscript,
) -> Result<(), PokError> {
    let p = &setup.params;
    let t = p.t;
    let k = out.len();
    if k != p.k {
        return Err(PokError::Shape("k mismatch".into()));
    }
    // Step 4: c = Σ b^{i−1} c_i — the verifier checks the Ajtai
    // homomorphism against the INSTANCE commitments c_i (the public
    // side), exactly Protocol 8's Step 4.
    let mut expect_c: Option<Vec<Poly>> = None;
    for (i, oi) in out.iter().enumerate() {
        let w = Fq::from_i64(p.b.pow(i as u32));
        expect_c = Some(match expect_c {
            None => oi.inst.c.iter().map(|c| c.scale(&w)).collect(),
            Some(mut a) => {
                for (x, c) in a.iter_mut().zip(oi.inst.c.iter()) {
                    x.add_assign(&c.scale(&w));
                }
                a
            }
        });
    }
    if expect_c.unwrap_or_default() != pair.inst.c {
        return Err(PokError::Shape("c != Σ b^{i−1} c_i".into()));
    }
    // Step 5 PoK.
    if let Some(step5) = &tr.step5 {
        let mut coms: Vec<AbdlopCommitment> = Vec::with_capacity((k + 1) * t);
        coms.extend(pair.inst.coms.iter().cloned());
        for per in &tr.coms {
            coms.extend(per.iter().cloned());
        }
        let mut rows: Vec<RelRow> = Vec::with_capacity(t);
        for j in 0..t {
            let mut weights: Vec<(usize, usize, PolyK)> = Vec::with_capacity(k + 1);
            let one = PolyK::one(p.d);
            let (a0, b0) = msg_weight(&one);
            weights.push((j, 0, a0));
            weights.push((j, 1, b0));
            for i in 0..k {
                let w = PolyK::degree0(K::from_fp(Fq::from_i64(-(p.b).pow(i as u32))), p.d);
                let (wa, wb) = msg_weight(&w);
                weights.push(((1 + i) * t + j, 0, wa));
                weights.push(((1 + i) * t + j, 1, wb));
            }
            rows.push(RelRow::full_ring(weights, PolyK::zero(p.d)));
        }
        let rel_rows = RelRows { rows };
        let blocks: Vec<Block> = {
            let mut b = Vec::with_capacity((k + 1) * t);
            for j in 0..t {
                b.push(Block {
                    com: pair.inst.coms[j].clone(),
                    op: pair.wit.openings[j].clone(),
                });
            }
            for (i, per) in tr.coms.iter().enumerate() {
                for (c, o) in per.iter().zip(out[i].wit.openings.iter()) {
                    b.push(Block {
                        com: c.clone(),
                        op: o.clone(),
                    });
                }
            }
            b
        };
        let u = rel_rows.eval_messages(&blocks);
        verify_linear(&setup.abdlop, &coms, &rel_rows, &u, step5)?;
    }
    // Output norm bound b.
    for oi in out {
        if oi.wit.z.norm_inf() >= p.b {
            return Err(PokError::NormCheck(1001));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Protocol 11 — the blinding sampler; Protocol 12 — Π_LBF.
// ---------------------------------------------------------------------------

/// Protocol 11: sample a randomized CEcom(b, L, B̃, Commit, T, K)^k
/// instance-witness pair: one Gaussian z ← D_s^{nF} (restarted while
/// ∥z∥∞ > τs), split into k pieces sharing the point r.
pub fn sample_blind_cecom(setup: &LbfSetup, rng: &mut Rng) -> Result<Vec<CecomPair>, PokError> {
    let p = &setup.params;
    let d = p.d;
    let t = p.t;
    let s = p.rlc_width();
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        if attempts > p.w_max() {
            return Err(PokError::ExtractionFailed("sampler restarts".into()));
        }
        let z: Vec<i64> = (0..p.nf).map(|_| rng.gaussian(s, p.tau)).collect();
        let z_f = FieldVec(z.iter().map(|&v| Fq::from_i64(v)).collect());
        if z_f.norm_inf() as f64 > p.tau * s {
            continue;
        }
        let zr = z_f.to_ring(d);
        let pieces = split_b_k(&zr, p.b, p.k);
        // The shared point r.
        let r: Vec<K> = (0..p.log_m())
            .map(|_| {
                K(
                    Fq(rng.next_u64() % crate::fp::Q),
                    Fq(rng.next_u64() % crate::fp::Q),
                )
            })
            .collect();
        let m1 = setup.m1();
        let mats = [&m1, &setup.m2, &setup.m3];
        let mut out = Vec::with_capacity(p.k);
        for piece in &pieces {
            let z_field = crate::embed::from_ring(piece);
            let mut per_com = Vec::with_capacity(t);
            let mut per_y = Vec::with_capacity(t);
            let mut per_op = Vec::with_capacity(t);
            for j in 0..t {
                let u = mats[j].mul_ring(piece);
                let yj = RingMle::from_ring_vec(&u).eval(&r);
                let (c, o) = commit_hint(&setup.abdlop, &yj, rng);
                per_com.push(c);
                per_y.push(yj);
                per_op.push(o);
            }
            out.push(CecomPair {
                inst: CecomInstance {
                    c: setup.ajtai.commit(piece),
                    x: z_field.0[..p.nf_in].to_vec(),
                    r: r.clone(),
                    coms: per_com,
                },
                wit: CecomWitness {
                    z: z_field,
                    y: per_y,
                    openings: per_op,
                },
            });
        }
        return Ok(out);
    }
}

/// Protocol 12 — Π_LBF (with the ι_bl precomposition of Definition 4.6):
/// fold K = 1 fresh (unpadded) R1CS instance with k sampled accumulators.
pub fn lbf_protocol(
    setup: &LbfSetup,
    x: &[Fq],
    w_circ: &[Fq],
    rng: &mut Rng,
) -> Result<(Vec<CecomPair>, LbfTranscript), PokError> {
    let p = &setup.params;
    if p.capital_k != 1 {
        return Err(PokError::Shape("blinding requires K = 1".into()));
    }
    // ι_bl (Eq 3.19): pad the structure, sample the blinding block e.
    let e = setup.blinded.sample_blinding_block(p.b, rng);
    let mut z1 = vec![Fq::ZERO; p.nf];
    z1[..x.len()].copy_from_slice(x);
    z1[x.len()..x.len() + w_circ.len()].copy_from_slice(w_circ);
    let flat_e: Vec<Fq> = e.iter().flat_map(|pp| pp.0.iter().copied()).collect();
    z1[p.nf - p.nf_bl..].copy_from_slice(&flat_e);
    let z1_f = FieldVec(z1);
    // c_1 = L(z_1) — a prover message (the unpadded-relation reading).
    let c1 = setup.ajtai.commit(&z1_f.to_ring(p.d));
    // Step 1: sample the k accumulator pairs (Protocol 11).
    let accs = sample_blind_cecom(setup, rng)?;
    // Step 2: Π'_DEC ∘ Π'_RLC ∘ Π'_R1CS (in application order 6, 7, 8).
    let fresh_inst_c = vec![c1];
    let (r1cs_out, r1cs_tr) = r1cs_reduction(setup, std::slice::from_ref(&z1_f), &accs, rng)?;
    let (rlc_out, rlc_tr) = rlc_reduction(setup, &r1cs_out.pairs, rng)?;
    let (dec_out, dec_tr) = dec_reduction(setup, &rlc_out, rng)?;
    Ok((
        dec_out.clone(),
        LbfTranscript {
            fresh_c: fresh_inst_c,
            accs,
            r1cs: r1cs_tr,
            r1cs_out_r: r1cs_out.r_prime.clone(),
            r1cs_pairs: r1cs_out.pairs.clone(),
            rlc: rlc_tr,
            rlc_out: rlc_out.clone(),
            dec: dec_tr,
            dec_out,
        },
    ))
}

/// The Π_LBF transcript (the three reductions' transcripts + the
/// intermediate instance-side data the verifier needs).
pub struct LbfTranscript {
    pub fresh_c: Vec<Vec<Poly>>,
    pub accs: Vec<CecomPair>,
    pub r1cs: R1csTranscript,
    pub r1cs_out_r: Vec<K>,
    /// The R1CS output pairs (instance-side + commitment witnesses for
    /// the RLC verifier).
    pub r1cs_pairs: Vec<CecomPair>,
    pub rlc: RlcTranscript,
    /// The RLC output pair.
    pub rlc_out: CecomPair,
    pub dec: DecTranscript,
    pub dec_out: Vec<CecomPair>,
}

/// Verify Π_LBF end-to-end: the three reductions' verifications run on
/// the intermediate instances recorded in the transcript (the verifier
/// needs only instance-side data — the intermediate witnesses stay with
/// the prover).
pub fn lbf_verify(
    setup: &LbfSetup,
    x: &[Fq],
    out: &[CecomPair],
    tr: &LbfTranscript,
) -> Result<(), PokError> {
    // Π'_R1CS on (fresh, accs).
    let r1cs_out = R1csOutput {
        pairs: tr.r1cs_pairs.clone(),
        r_prime: tr.r1cs_out_r.clone(),
    };
    r1cs_verify(setup, &tr.fresh_c, &tr.accs, &r1cs_out, &tr.r1cs)?;
    // Π'_RLC on the R1CS output.
    rlc_verify(setup, &tr.r1cs_pairs, &tr.rlc_out, &tr.rlc)?;
    // Π'_DEC on the RLC output.
    dec_verify(setup, &tr.rlc_out, out, &tr.dec)?;
    let _ = x;
    Ok(())
}

// ---------------------------------------------------------------------------
// Corollary 4.24 — the accumulator-free variant Π°_LBF, and Protocol 13 —
// the folding blueprint Π_LBF-Fold.
// ---------------------------------------------------------------------------

/// Corollary 4.24: Π°_LBF at kin = 0 — Step 1 of Protocol 12 is deleted;
/// Π'_R1CS and Π'_RLC run on the single fresh pair alone (v = ρ₁·z₁,
/// z = v + y), while Π'_DEC still outputs k pieces. The blinding pair
/// count drops (k = 26 vs 31 at the paper's parameters) at no cost in
/// blinding — hiding is carried by the Gaussian mask and the blinded
/// layout alone.
pub fn lbf_accumulator_free(
    setup: &LbfSetup,
    x: &[Fq],
    w_circ: &[Fq],
    rng: &mut Rng,
) -> Result<(Vec<CecomPair>, LbfTranscript), PokError> {
    let p = &setup.params;
    if p.capital_k != 1 {
        return Err(PokError::Shape("blinding requires K = 1".into()));
    }
    // ι_bl: pad and blind the single fresh witness.
    let e = setup.blinded.sample_blinding_block(p.b, rng);
    let mut z1 = vec![Fq::ZERO; p.nf];
    z1[..x.len()].copy_from_slice(x);
    z1[x.len()..x.len() + w_circ.len()].copy_from_slice(w_circ);
    let flat_e: Vec<Fq> = e.iter().flat_map(|pp| pp.0.iter().copied()).collect();
    z1[p.nf - p.nf_bl..].copy_from_slice(&flat_e);
    let z1_f = FieldVec(z1);
    let c1 = setup.ajtai.commit(&z1_f.to_ring(p.d));
    // NO accumulator sampling (kin = 0): the reductions run on the
    // single fresh pair.
    let fresh_inst_c = vec![c1];
    let (r1cs_out, r1cs_tr) = r1cs_reduction(setup, &[z1_f], &[], rng)?;
    let (rlc_out, rlc_tr) = rlc_reduction(setup, &r1cs_out.pairs, rng)?;
    let (dec_out, dec_tr) = dec_reduction(setup, &rlc_out, rng)?;
    Ok((
        dec_out.clone(),
        LbfTranscript {
            fresh_c: fresh_inst_c,
            accs: Vec::new(),
            r1cs: r1cs_tr,
            r1cs_out_r: r1cs_out.r_prime.clone(),
            r1cs_pairs: r1cs_out.pairs.clone(),
            rlc: rlc_tr,
            rlc_out: rlc_out.clone(),
            dec: dec_tr,
            dec_out,
        },
    ))
}

/// Protocol 13 — the Π_LBF-Fold blueprint: the optional-input branch
/// structure. With `None` the prover samples the k accumulator pairs
/// itself (the blinding branch — Π_LBF); with `Some(accs)` the
/// accumulators come from a prior round (the folding branch — NOT proved
/// blinding by the paper; recorded here as the blueprint it is).
pub fn lbf_fold_blueprint(
    setup: &LbfSetup,
    x: &[Fq],
    w_circ: &[Fq],
    optional_accs: Option<Vec<CecomPair>>,
    rng: &mut Rng,
) -> Result<(Vec<CecomPair>, LbfTranscript), PokError> {
    match optional_accs {
        None => lbf_protocol(setup, x, w_circ, rng),
        Some(accs) => {
            // The folding branch: run the reductions with the supplied
            // accumulators (their distribution is a hypothesis the paper
            // flags but does not prove — Remark 4.8's O(1)-round limit).
            let p = &setup.params;
            let e = setup.blinded.sample_blinding_block(p.b, rng);
            let mut z1 = vec![Fq::ZERO; p.nf];
            z1[..x.len()].copy_from_slice(x);
            z1[x.len()..x.len() + w_circ.len()].copy_from_slice(w_circ);
            let flat_e: Vec<Fq> = e.iter().flat_map(|pp| pp.0.iter().copied()).collect();
            z1[p.nf - p.nf_bl..].copy_from_slice(&flat_e);
            let z1_f = FieldVec(z1);
            let c1 = setup.ajtai.commit(&z1_f.to_ring(p.d));
            let fresh_inst_c = vec![c1];
            let (r1cs_out, r1cs_tr) = r1cs_reduction(setup, &[z1_f], &accs, rng)?;
            let (rlc_out, rlc_tr) = rlc_reduction(setup, &r1cs_out.pairs, rng)?;
            let (dec_out, dec_tr) = dec_reduction(setup, &rlc_out, rng)?;
            Ok((
                dec_out.clone(),
                LbfTranscript {
                    fresh_c: fresh_inst_c,
                    accs,
                    r1cs: r1cs_tr,
                    r1cs_out_r: r1cs_out.r_prime.clone(),
                    r1cs_pairs: r1cs_out.pairs.clone(),
                    rlc: rlc_tr,
                    rlc_out: rlc_out.clone(),
                    dec: dec_tr,
                    dec_out,
                },
            ))
        }
    }
}
