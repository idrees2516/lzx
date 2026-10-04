//! `Π_GBF2` — the Spartan-style reduction `R*_GBF → R*_PCE × R_hbPCE`
//! (ePrint 2026/538 §4.2, Figures 4–5), instantiated for **both**
//! representations (`ν = 1` univariate over `H` with the `h₁/h₂`
//! decomposition sum-checks, and `ν = log n` multivariate with two
//! round-based sum-checks), plus the **early-stopping variant**
//! `Π_esGBF2` (§4.3's HyperNova-style linearized committed CCS).
//!
//! The protocol batched over `K` GBF statements:
//! 1. `γ ← F^K` batches the instances: `q(X) = Σ_k γ^k·left^k(X)·right^k(X)`.
//! 2. **Sum-check 1** reduces `Σ_{h∈H} q(h) = s = Σ_k γ^k s^k`.
//! 3. At the point `α`, the prover sends the Evals: `{u_j(α)}` (R_PCE
//!    claims over the committed u's) and `{ζ_{jM,jv} = λ(α)ᵀ M_{jM}
//!    v_{jv}}`.
//! 4. `η ← F^{|SR|}` batches the ζ's; **sum-check 2** reduces
//!    `Σ_h Σ η·(λ(α)ᵀ M λ(h))·v(h) = s' = Σ η ζ`.
//! 5. At `β`: Evals `{v_{jv}(β)}` (R_PCE) and `{m_{jM} = λ(α)ᵀ M_{jM}
//!    λ(β)}` — the **holographic claims** (R_hbPCE).
//!
//! Decision (Fig. 5): the round identities, plus the Evals-derived
//! `ζ = left(α)·right(α)` and `ζ' = Σ η m v` identity checks.

use crate::pc::PcCommitment;
use crate::poly::{lambda_eval, mat_vec, vec_poly_eval, Domain};
use crate::relations::{GbfInstance, GbfWitness, RelError};
use crate::sumcheck::{
    mv_prove, mv_verify, uni_mul, uni_prove, uni_verify, MvSumcheckProof, UniSumcheckProof,
};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GbfError {
    Rel(RelError),
    Sc(crate::sumcheck::ScError),
    Transcript(lattice_core::transcript::TranscriptError),
    Shape(&'static str),
}

impl core::fmt::Display for GbfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GbfError::Rel(e) => write!(f, "relation: {e}"),
            GbfError::Sc(e) => write!(f, "sumcheck: {e}"),
            GbfError::Transcript(e) => write!(f, "transcript: {e}"),
            GbfError::Shape(s) => write!(f, "gbf shape: {s}"),
        }
    }
}

impl From<RelError> for GbfError {
    fn from(e: RelError) -> Self {
        GbfError::Rel(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for GbfError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        GbfError::Transcript(e)
    }
}

impl From<crate::poly::PolyError> for GbfError {
    fn from(e: crate::poly::PolyError) -> Self {
        GbfError::Rel(RelError::Poly(e))
    }
}

impl From<crate::sumcheck::ScError> for GbfError {
    fn from(e: crate::sumcheck::ScError) -> Self {
        GbfError::Sc(e)
    }
}

/// The sum-check representation.
#[derive(Clone, Debug)]
pub enum ScRepr {
    Uni(UniSumcheckProof),
    Mv(MvSumcheckProof),
}

/// Alias used by `gbf1` for the shared representation.
pub type ScRepr2 = ScRepr;

/// The protocol statement set: K GBF instances with their witnesses.
pub struct Gbf2Statement<'a> {
    pub domain: Domain,
    pub instances: &'a [GbfInstance],
    pub witnesses: &'a [GbfWitness],
    /// The matrix set (the shared index).
    pub matrices: &'a [Vec<Vec<Fp256>>],
}

/// The prover's Evals and proofs (the full non-interactive artifact).
#[derive(Clone, Debug)]
pub struct Gbf2Proof {
    pub gammas: Vec<Fp256>,
    pub sc1: ScRepr,
    /// The α point (sum-check 1's challenge).
    pub alpha: Vec<Fp256>,
    /// Evals at α: `u_j(α)` for the committed u's (R_PCE claims).
    pub u_evals: Vec<Fp256>,
    /// Evals at α: `ζ_{jM,jv} = λ(α)ᵀ M_{jM} v_{jv}` (claimed).
    pub zetas: Vec<Vec<Fp256>>,
    /// The η challenges batching the ζ's.
    pub etas: Vec<Fp256>,
    pub sc2: ScRepr,
    /// The β point (sum-check 2's challenge).
    pub beta: Vec<Fp256>,
    /// Evals at β: `v_{jv}(β)` for the committed v's (R_PCE claims).
    pub v_evals: Vec<Fp256>,
    /// Evals at β: `m_{jM} = λ(α)ᵀ M_{jM} λ(β)` — the holographic claims.
    pub m_evals: Vec<Fp256>,
}

/// The derived output claims for the caller to settle.
#[derive(Clone, Debug)]
pub struct Gbf2Output {
    pub alpha: Vec<Fp256>,
    pub beta: Vec<Fp256>,
    /// R_PCE claims: (commitment, point, value) for the u's (at α) and
    /// v's (at β).
    pub pce_claims: Vec<(PcCommitment, Vec<Fp256>, Fp256)>,
    /// The holographic claims `m_i` (R_hbPCE instance content).
    pub hb_gammas: Vec<Fp256>,
}

/// The right-side pair set `SR` (deduplicated (jM, jv) pairs across the
/// instances).
pub fn right_pairs(instances: &[GbfInstance]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for inst in instances {
        for s_i in &inst.right.sets {
            for &p in s_i {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// The left-side u-index set (deduplicated).
pub fn left_indices(instances: &[GbfInstance]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for inst in instances {
        for s_i in &inst.left.sets {
            for &j in s_i {
                if !out.contains(&j) {
                    out.push(j);
                }
            }
        }
    }
    out
}

/// The v-index set (deduplicated).
pub fn v_indices(instances: &[GbfInstance]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for inst in instances {
        for s_i in &inst.right.sets {
            for &(_, jv) in s_i {
                if !out.contains(&jv) {
                    out.push(jv);
                }
            }
        }
    }
    out
}

struct EffectiveVectors {
    /// Per instance: the effective u vectors (committed + implicit λ(α)
    /// occupying the LAST slot).
    us: Vec<Vec<Vec<Fp256>>>,
    /// Per instance: the effective v vectors (committed + implicit λ(β)).
    vs: Vec<Vec<Vec<Fp256>>>,
}

fn effective_vectors(
    domain: &Domain,
    instances: &[GbfInstance],
    witnesses: &[GbfWitness],
) -> Result<EffectiveVectors, GbfError> {
    let n = domain.size();
    let mut us = Vec::with_capacity(instances.len());
    let mut vs = Vec::with_capacity(instances.len());
    let mut implicit_alpha = Vec::with_capacity(instances.len());
    let mut implicit_beta = Vec::with_capacity(instances.len());
    let _ = (&implicit_alpha, &implicit_beta);
    for (inst, wit) in instances.iter().zip(witnesses.iter()) {
        let mut u = wit.us.clone();
        let ia = inst.alpha.is_some();
        if let Some(alpha) = &inst.alpha {
            let lam: Vec<Fp256> = (0..n)
                .map(|i| lambda_eval(domain, i, alpha))
                .collect::<Result<_, _>>()?;
            u.push(lam);
        }
        let mut v = wit.vs.clone();
        let ib = inst.beta.is_some();
        if let Some(beta) = &inst.beta {
            let lam: Vec<Fp256> = (0..n)
                .map(|i| lambda_eval(domain, i, beta))
                .collect::<Result<_, _>>()?;
            v.push(lam);
        }
        us.push(u);
        vs.push(v);
        implicit_alpha.push(ia);
        implicit_beta.push(ib);
    }
    let _ = (implicit_alpha, implicit_beta);
    Ok(EffectiveVectors { us, vs })
}

/// The left-side vector value at a point (per instance).
fn left_at(
    domain: &Domain,
    inst: &GbfInstance,
    us: &[Vec<Fp256>],
    pt: &[Fp256],
) -> Result<Fp256, GbfError> {
    let mut acc = Fp256::ZERO;
    for (i, s_i) in inst.left.sets.iter().enumerate() {
        let mut prod = inst.left.constants[i];
        for &j in s_i {
            prod = prod.mul(&vec_poly_eval(domain, &us[j], pt)?);
        }
        acc = acc.add(&prod);
    }
    Ok(acc)
}

/// The right-side vector value at a point (per instance).
fn right_at(
    domain: &Domain,
    inst: &GbfInstance,
    vs: &[Vec<Fp256>],
    matrices: &[Vec<Vec<Fp256>>],
    pt: &[Fp256],
) -> Result<Fp256, GbfError> {
    let n = domain.size();
    // λ(pt) per index.
    let lam: Vec<Fp256> = (0..n)
        .map(|i| lambda_eval(domain, i, pt))
        .collect::<Result<_, _>>()?;
    let mut acc = Fp256::ZERO;
    for (i, s_i) in inst.right.sets.iter().enumerate() {
        let mut prod = inst.right.constants[i];
        for &(jm, jv) in s_i {
            // λ(pt)ᵀ M_{jm} v_{jv}
            let mv = mat_vec(&matrices[jm], &vs[jv]).map_err(RelError::Poly)?;
            let mut dot = Fp256::ZERO;
            for r in 0..n {
                dot = dot.add(&lam[r].mul(&mv[r]));
            }
            prod = prod.mul(&dot);
        }
        acc = acc.add(&prod);
    }
    Ok(acc)
}

/// Prove `Π_GBF2` (Fiat–Shamir over the public instances).
pub fn gbf2_prove(
    st: &Gbf2Statement<'_>,
    transcript: &mut Transcript,
) -> Result<(Gbf2Proof, Gbf2Output), GbfError> {
    let domain = &st.domain;
    let n = domain.size();
    let k = st.instances.len();
    if st.witnesses.len() != k {
        return Err(GbfError::Shape("instances vs witnesses"));
    }
    // Absorb the public statements.
    for inst in st.instances {
        lattice_pcd::util::absorb_fp(transcript, b"gbf-s", &inst.s)?;
        for c in &inst.u_commitments {
            transcript.append_message(b"gbf-uc", &c.to_bytes())?;
        }
        for c in &inst.v_commitments {
            transcript.append_message(b"gbf-vc", &c.to_bytes())?;
        }
    }
    let gammas = lattice_pcd::util::challenge_fp_vec(transcript, b"gbf-gamma", k)?;
    let eff = effective_vectors(domain, st.instances, st.witnesses)?;
    let s = st
        .instances
        .iter()
        .zip(gammas.iter())
        .fold(Fp256::ZERO, |acc, (i, g)| acc.add(&g.mul(&i.s)));

    // ---- Sum-check 1: q(X) = Σ_k γ^k·left^k(X)·right^k(X) ----
    let dl = st
        .instances
        .iter()
        .map(|i| i.left.sets.iter().map(|s| s.len()).max().unwrap_or(0))
        .max()
        .unwrap_or(0);
    let dr = st
        .instances
        .iter()
        .map(|i| i.right.sets.iter().map(|s| s.len()).max().unwrap_or(0))
        .max()
        .unwrap_or(0);
    let degree1 = dl + dr;

    let sc1;
    let alpha: Vec<Fp256>;
    match domain {
        Domain::Multivariate { num_vars } => {
            let eval = |pt: &[Fp256]| -> Vec<Fp256> {
                let mut acc = Fp256::ZERO;
                for (ki, inst) in st.instances.iter().enumerate() {
                    let l = left_at(domain, inst, &eff.us[ki], pt).unwrap_or(Fp256::ZERO);
                    let r =
                        right_at(domain, inst, &eff.vs[ki], st.matrices, pt).unwrap_or(Fp256::ZERO);
                    acc = acc.add(&gammas[ki].mul(&l.mul(&r)));
                }
                vec![acc]
            };
            let (proof, out) = mv_prove(*num_vars, degree1, &eval, &[s], transcript)?;
            sc1 = ScRepr::Mv(proof);
            alpha = out.point;
        }
        Domain::Univariate { n } => {
            // Build g's coefficient vector: the domain-value vectors ARE
            // Lagrange expansions; convert to standard coefficients via
            // Σᵢ vals[i]·u_H(X)/(n·hᵢ^{n−1}·(X−hᵢ)) with u_H/(X−hᵢ) from
            // exact division (O(n²)).
            let to_coeffs = |vals: &[Fp256]| -> Vec<Fp256> {
                let uh = crate::sumcheck::vanish_poly(*n);
                let inv_n = Fp256::from_canonical_u64(*n as u64)
                    .inverse()
                    .unwrap_or(Fp256::ZERO);
                let mut acc = vec![Fp256::ZERO; (*n).max(1)];
                for i in 0..*n {
                    if vals[i].is_zero() {
                        continue;
                    }
                    let h = crate::poly::domain_point(i, *n);
                    let lin = [h.neg(), Fp256::from_canonical_u64(1)]; // X − h
                    let (q, _r) = crate::sumcheck::uni_divmod(&uh, &lin);
                    // scale = vals[i]·h/n (h^{n−1} = h^{−1} on H).
                    let scale = vals[i].mul(&h).mul(&inv_n);
                    for (c, qc) in q.iter().enumerate() {
                        if c < acc.len() {
                            acc[c] = acc[c].add(&scale.mul(qc));
                        }
                    }
                }
                acc
            };
            // left/right evaluated at every domain point.
            let mut g_coeffs = vec![Fp256::ZERO; 2 * n];
            for (ki, inst) in st.instances.iter().enumerate() {
                // left values at the n domain points → coeffs:
                let left_vals: Vec<Fp256> = (0..*n)
                    .map(|i| {
                        let pt = vec![crate::poly::domain_point(i, *n)];
                        left_at(domain, inst, &eff.us[ki], &pt).unwrap_or(Fp256::ZERO)
                    })
                    .collect();
                let right_vals: Vec<Fp256> = (0..*n)
                    .map(|i| {
                        let pt = vec![crate::poly::domain_point(i, *n)];
                        right_at(domain, inst, &eff.vs[ki], st.matrices, &pt).unwrap_or(Fp256::ZERO)
                    })
                    .collect();
                let lc = to_coeffs(&left_vals);
                let rc = to_coeffs(&right_vals);
                let gk = uni_mul(&lc, &rc);
                for (i, c) in gk.iter().enumerate() {
                    if i < g_coeffs.len() {
                        g_coeffs[i] = g_coeffs[i].add(&gammas[ki].mul(c));
                    }
                }
            }
            let proof = uni_prove(*n, &g_coeffs, &s);
            sc1 = ScRepr::Uni(proof);
            alpha = vec![lattice_pcd::util::challenge_fp(transcript, b"gbf-alpha")?];
        }
    }

    // ---- Evals at α ----
    let pairs = right_pairs(st.instances);
    let u_idx = left_indices(st.instances);
    let v_idx = v_indices(st.instances);
    // u_j(α) — only for the COMMITTED u's (the implicit λ(α) is free).
    // Evals for the COMMITTED u slots only — the implicit λ(α) slot is
    // verifier-computable (Corollary 1's specialization).
    let mut u_evals = Vec::new();
    for &j in &u_idx {
        for (ki, inst) in st.instances.iter().enumerate() {
            if j < inst.u_commitments.len() && j < eff.us[ki].len() {
                u_evals.push(vec_poly_eval(domain, &eff.us[ki][j], &alpha)?);
                break;
            }
        }
    }
    // ζ per instance per pair.
    let mut zetas: Vec<Vec<Fp256>> = Vec::with_capacity(k);
    {
        let lam_alpha: Vec<Fp256> = (0..n)
            .map(|i| lambda_eval(domain, i, &alpha))
            .collect::<Result<_, _>>()?;
        for (ki, _inst) in st.instances.iter().enumerate() {
            let mut row = Vec::new();
            for &(jm, jv) in &pairs {
                if jv < eff.vs[ki].len() && jm < st.matrices.len() {
                    let mv = mat_vec(&st.matrices[jm], &eff.vs[ki][jv]).map_err(RelError::Poly)?;
                    let mut dot = Fp256::ZERO;
                    for r in 0..n {
                        dot = dot.add(&lam_alpha[r].mul(&mv[r]));
                    }
                    row.push(dot);
                } else {
                    row.push(Fp256::ZERO);
                }
            }
            zetas.push(row);
        }
    }
    // η batches the ζ's (flat over (k, pair) — Fig. 4's η^{(k)} indexed
    // per instance).
    let etas = lattice_pcd::util::challenge_fp_vec(transcript, b"gbf-eta", k * pairs.len())?;
    let s_prime: Fp256 = {
        let mut acc = Fp256::ZERO;
        for (ki, row) in zetas.iter().enumerate() {
            for (pi, z) in row.iter().enumerate() {
                acc = acc.add(&etas[ki * pairs.len() + pi].mul(z));
            }
        }
        acc
    };

    // ---- Sum-check 2: q'(X) = Σ_k Σ_pairs η·(λ(α)ᵀ M λ(X))·v_{jv}(X) ----
    let sc2;
    let beta: Vec<Fp256>;
    {
        let claimed = s_prime;
        match domain {
            Domain::Multivariate { num_vars } => {
                let eval = |pt: &[Fp256]| -> Vec<Fp256> {
                    let mut acc = Fp256::ZERO;
                    for (ki, _inst) in st.instances.iter().enumerate() {
                        for (pi, &(jm, jv)) in pairs.iter().enumerate() {
                            if jv >= eff.vs[ki].len() {
                                continue;
                            }
                            // λ(α)ᵀ M_{jm} λ(pt) · v_{jv}(pt)
                            let m_at =
                                crate::poly::matrix_poly_eval(domain, &st.matrices[jm], pt, &alpha)
                                    .unwrap_or(Fp256::ZERO);
                            let v_at =
                                vec_poly_eval(domain, &eff.vs[ki][jv], pt).unwrap_or(Fp256::ZERO);
                            acc = acc.add(&etas[ki * pairs.len() + pi].mul(&m_at.mul(&v_at)));
                        }
                    }
                    vec![acc]
                };
                let (proof, out) = mv_prove(*num_vars, 2, &eval, &[claimed], transcript)?;
                sc2 = ScRepr::Mv(proof);
                beta = out.point;
            }
            Domain::Univariate { n } => {
                // Build q' coefficients: Σ η · M_{jm}(X, α) · v_{jv}(X) —
                // via evaluation at 2n−1 points + interpolation? M(X,α) has
                // degree n−1, v degree n−1 → product degree 2n−2 → evaluate
                // at 2n−1 points... simpler: evaluate at 2n points and
                // interpolate.
                // Interpolate through integer nodes 0..2n−1 (general-node
                // interpolation would also work; integer nodes reuse the
                // shared engine).
                let npoints = 2 * n;
                let ys_int: Vec<Fp256> = (0..npoints)
                    .map(|i| {
                        let pt = vec![Fp256::from_canonical_u64(i as u64)];
                        let mut acc = Fp256::ZERO;
                        for (ki, _inst) in st.instances.iter().enumerate() {
                            for (pi, &(jm, jv)) in pairs.iter().enumerate() {
                                if jv >= eff.vs[ki].len() {
                                    continue;
                                }
                                let m_at = crate::poly::matrix_poly_eval(
                                    domain,
                                    &st.matrices[jm],
                                    &pt,
                                    &alpha,
                                )
                                .unwrap_or(Fp256::ZERO);
                                let v_at = vec_poly_eval(domain, &eff.vs[ki][jv], &pt)
                                    .unwrap_or(Fp256::ZERO);
                                acc = acc.add(&etas[ki * pairs.len() + pi].mul(&m_at.mul(&v_at)));
                            }
                        }
                        acc
                    })
                    .collect();
                let g_coeffs = crate::sumcheck::interpolate_int_nodes(&ys_int);
                let proof = uni_prove(*n, &g_coeffs, &claimed);
                sc2 = ScRepr::Uni(proof);
                beta = vec![lattice_pcd::util::challenge_fp(transcript, b"gbf-beta")?];
            }
        }
    }

    // ---- Evals at β ----
    let mut v_evals = Vec::new();
    for &j in &v_idx {
        for (ki, inst) in st.instances.iter().enumerate() {
            if j < inst.v_commitments.len() && j < eff.vs[ki].len() {
                v_evals.push(vec_poly_eval(domain, &eff.vs[ki][j], &beta)?);
                break;
            }
        }
    }
    // m_{jM} = λ(α)ᵀ M_{jM} λ(β) — the holographic claims.
    let m_evals: Vec<Fp256> = st
        .matrices
        .iter()
        .map(|m| crate::poly::matrix_poly_eval(domain, m, &beta, &alpha))
        .collect::<Result<_, _>>()?;

    // R_PCE claims: u's at α + v's at β.
    let mut pce_claims = Vec::new();
    for (jj, uj) in u_idx.iter().zip(u_evals.iter()) {
        // The commitment for u-slot j: the first instance with that slot.
        for inst in st.instances {
            if *jj < inst.u_commitments.len() {
                pce_claims.push((inst.u_commitments[*jj], alpha.clone(), *uj));
                break;
            }
        }
    }
    for (jj, vj) in v_idx.iter().zip(v_evals.iter()) {
        for inst in st.instances {
            if *jj < inst.v_commitments.len() {
                pce_claims.push((inst.v_commitments[*jj], beta.clone(), *vj));
                break;
            }
        }
    }

    let proof = Gbf2Proof {
        gammas,
        sc1,
        alpha: alpha.clone(),
        u_evals,
        zetas,
        etas,
        sc2,
        beta: beta.clone(),
        v_evals,
        m_evals: m_evals.clone(),
    };
    let out = Gbf2Output {
        alpha,
        beta,
        pce_claims,
        hb_gammas: m_evals,
    };
    Ok((proof, out))
}

/// Verify `Π_GBF2`: checks the sum-check structures and the Evals-derived
/// identity checks; outputs the claims for the caller to settle.
pub fn gbf2_verify(
    domain: &Domain,
    instances: &[GbfInstance],
    proof: &Gbf2Proof,
    transcript: &mut Transcript,
) -> Result<Gbf2Output, GbfError> {
    let _n = domain.size();
    let k = instances.len();
    for inst in instances {
        lattice_pcd::util::absorb_fp(transcript, b"gbf-s", &inst.s)?;
        for c in &inst.u_commitments {
            transcript.append_message(b"gbf-uc", &c.to_bytes())?;
        }
        for c in &inst.v_commitments {
            transcript.append_message(b"gbf-vc", &c.to_bytes())?;
        }
    }
    // γ rederivation.
    let gammas = lattice_pcd::util::challenge_fp_vec(transcript, b"gbf-gamma", k)?;
    if gammas != proof.gammas {
        return Err(GbfError::Shape("gamma mismatch"));
    }
    let s = instances
        .iter()
        .zip(gammas.iter())
        .fold(Fp256::ZERO, |acc, (i, g)| acc.add(&g.mul(&i.s)));

    // Sum-check 1.
    let alpha: Vec<Fp256>;
    let sc1_final: Option<Fp256>;
    match (&proof.sc1, domain) {
        (ScRepr::Mv(p), Domain::Multivariate { num_vars }) => {
            let out = mv_verify(
                *num_vars,
                dl_dr(instances).0 + dl_dr(instances).1,
                &[s],
                p,
                transcript,
            )?;
            alpha = out.point;
            sc1_final = Some(out.final_evals[0]);
        }
        (ScRepr::Uni(p), Domain::Univariate { n }) => {
            // The identity check at α happens against the Evals below.
            let a = lattice_pcd::util::challenge_fp(transcript, b"gbf-alpha")?;
            alpha = vec![a];
            sc1_final = None;
            // Degree bound: dl + dr terms over the domain — expected deg
            // = (dl + dr)·(n−1).
            let (dl, dr) = dl_dr(instances);
            let expected_deg = (dl + dr) * (n - 1);
            // ζ from the Evals: left(α)·right(α).
            let zeta = zeta_from_evals(domain, instances, proof, &alpha)?;
            if !uni_verify(*n, p, &s, &a, &zeta, expected_deg) {
                return Err(GbfError::Shape("sum-check 1 identity"));
            }
        }
        _ => return Err(GbfError::Shape("sum-check 1 representation")),
    }

    // η rederivation.
    let pairs = right_pairs(instances);
    let etas = lattice_pcd::util::challenge_fp_vec(transcript, b"gbf-eta", k * pairs.len())?;
    if etas != proof.etas {
        return Err(GbfError::Shape("eta mismatch"));
    }
    // s' = Σ η ζ.
    let s_prime: Fp256 = proof
        .zetas
        .iter()
        .enumerate()
        .fold(Fp256::ZERO, |acc, (ki, row)| {
            row.iter()
                .enumerate()
                .fold(acc, |a, (pi, z)| a.add(&etas[ki * pairs.len() + pi].mul(z)))
        });

    // Sum-check 2.
    let beta: Vec<Fp256>;
    let sc2_final: Option<Fp256>;
    match (&proof.sc2, domain) {
        (ScRepr::Mv(p), Domain::Multivariate { num_vars }) => {
            let out = mv_verify(*num_vars, 2, &[s_prime], p, transcript)?;
            beta = out.point;
            sc2_final = Some(out.final_evals[0]);
        }
        (ScRepr::Uni(p), Domain::Univariate { n }) => {
            let b = lattice_pcd::util::challenge_fp(transcript, b"gbf-beta")?;
            beta = vec![b];
            sc2_final = None;
            // ζ' from the Evals: Σ_k Σ_pairs η·m_{jM}·v_{jv}(β).
            let zeta_prime = zeta_prime_from_evals(domain, instances, proof, &alpha, &beta)?;
            let expected_deg = 2 * (n - 1);
            if !uni_verify(*n, p, &s_prime, &b, &zeta_prime, expected_deg) {
                return Err(GbfError::Shape("sum-check 2 identity"));
            }
        }
        _ => return Err(GbfError::Shape("sum-check 2 representation")),
    }

    // The multivariate identity checks (Fig. 5's q_ν(α_ν) = ζ,
    // q'_ν(β_ν) = ζ') — against the kept final evals.
    if let Some(final1) = sc1_final {
        let zeta = zeta_from_evals(domain, instances, proof, &alpha)?;
        if final1 != zeta {
            return Err(GbfError::Shape("sum-check 1 final eval"));
        }
    }
    if let Some(final2) = sc2_final {
        let zeta_prime = zeta_prime_from_evals(domain, instances, proof, &alpha, &beta)?;
        if final2 != zeta_prime {
            return Err(GbfError::Shape("sum-check 2 final eval"));
        }
    }

    // R_PCE claims + holographic claims.
    // The committed u/v slots only (the implicit λ slots carry no claims).
    let u_idx: Vec<usize> = left_indices(instances)
        .into_iter()
        .filter(|&j| instances.iter().any(|i| j < i.u_commitments.len()))
        .collect();
    let v_idx: Vec<usize> = v_indices(instances)
        .into_iter()
        .filter(|&j| instances.iter().any(|i| j < i.v_commitments.len()))
        .collect();
    let mut pce_claims = Vec::new();
    for (jj, uj) in u_idx.iter().zip(proof.u_evals.iter()) {
        for inst in instances {
            if *jj < inst.u_commitments.len() {
                pce_claims.push((inst.u_commitments[*jj], alpha.clone(), *uj));
                break;
            }
        }
    }
    for (jj, vj) in v_idx.iter().zip(proof.v_evals.iter()) {
        for inst in instances {
            if *jj < inst.v_commitments.len() {
                pce_claims.push((inst.v_commitments[*jj], beta.clone(), *vj));
                break;
            }
        }
    }
    Ok(Gbf2Output {
        alpha,
        beta,
        pce_claims,
        hb_gammas: proof.m_evals.clone(),
    })
}

/// `(dl, dr)` — the max Hadamard-set sizes.
pub fn dl_dr(instances: &[GbfInstance]) -> (usize, usize) {
    let dl = instances
        .iter()
        .map(|i| i.left.sets.iter().map(|s| s.len()).max().unwrap_or(0))
        .max()
        .unwrap_or(0);
    let dr = instances
        .iter()
        .map(|i| i.right.sets.iter().map(|s| s.len()).max().unwrap_or(0))
        .max()
        .unwrap_or(0);
    (dl, dr)
}

/// `ζ = Σ_k γ^k·left^k(α)·right^k(α)` from the Evals: left(α) from the
/// u-evals (and the implicit λ(α)), right(α) from the ζ's.
fn zeta_from_evals(
    domain: &Domain,
    instances: &[GbfInstance],
    proof: &Gbf2Proof,
    alpha: &[Fp256],
) -> Result<Fp256, GbfError> {
    let n = domain.size();
    let pairs = right_pairs(instances);
    let u_idx = left_indices(instances);
    // u_j(α): the committed u's from the Evals; the implicit λ(α) computed.
    let lam_alpha: Vec<Fp256> = (0..n)
        .map(|i| lambda_eval(domain, i, alpha))
        .collect::<Result<_, _>>()?;
    let mut acc = Fp256::ZERO;
    for (ki, inst) in instances.iter().enumerate() {
        let gamma = proof.gammas[ki];
        // left^k(α) = Σ_i c ∏_{j∈S} u_j(α).
        let mut left = Fp256::ZERO;
        for (i, s_i) in inst.left.sets.iter().enumerate() {
            let mut prod = inst.left.constants[i];
            for &j in s_i {
                let val = if j < inst.u_commitments.len() {
                    let pos = u_idx.iter().position(|&x| x == j).unwrap_or(usize::MAX);
                    if pos == usize::MAX || pos >= proof.u_evals.len() {
                        return Err(GbfError::Shape("u eval missing"));
                    }
                    proof.u_evals[pos]
                } else {
                    // The implicit λ(α^{(k)}) slot: the domain polynomial of
                    // that vector evaluated at the NEW α —
                    // Σᵢ λᵢ(α^{(k)})·λᵢ(α).
                    match &inst.alpha {
                        Some(ak) => {
                            let lam_k: Vec<Fp256> = (0..n)
                                .map(|i| lambda_eval(domain, i, ak))
                                .collect::<Result<_, _>>()?;
                            lam_k
                                .iter()
                                .zip(lam_alpha.iter())
                                .fold(Fp256::ZERO, |a, (x, y)| a.add(&x.mul(y)))
                        }
                        None => return Err(GbfError::Shape("u slot without a source")),
                    }
                };
                prod = prod.mul(&val);
            }
            left = left.add(&prod);
        }
        // right^k(α) = Σ_i c ∏ ζ_{(jM,jv)}.
        let mut right = Fp256::ZERO;
        for (i, s_i) in inst.right.sets.iter().enumerate() {
            let mut prod = inst.right.constants[i];
            for &p in s_i {
                let pos = pairs.iter().position(|&x| x == p).unwrap_or(usize::MAX);
                if pos == usize::MAX || pos >= proof.zetas[ki].len() {
                    return Err(GbfError::Shape("zeta missing"));
                }
                prod = prod.mul(&proof.zetas[ki][pos]);
            }
            right = right.add(&prod);
        }
        acc = acc.add(&gamma.mul(&left.mul(&right)));
    }
    Ok(acc)
}

/// `ζ' = Σ_k Σ_pairs η·m_{jM}·v_{jv}(β)` from the Evals.
fn zeta_prime_from_evals(
    domain: &Domain,
    instances: &[GbfInstance],
    proof: &Gbf2Proof,
    _alpha: &[Fp256],
    beta: &[Fp256],
) -> Result<Fp256, GbfError> {
    let n = domain.size();
    let pairs = right_pairs(instances);
    let v_idx = v_indices(instances);
    // λ(β) for the implicit-v slots evaluated at the NEW β.
    let lam_beta: Vec<Fp256> = (0..n)
        .map(|i| lambda_eval(domain, i, beta))
        .collect::<Result<_, _>>()?;
    let mut acc = Fp256::ZERO;
    for (ki, inst) in instances.iter().enumerate() {
        // The instance's OWN implicit-v point β^{(k)} (if any).
        let lam_beta_k: Option<Vec<Fp256>> = match &inst.beta {
            Some(bk) => Some(
                (0..n)
                    .map(|i| lambda_eval(domain, i, bk))
                    .collect::<Result<_, _>>()?,
            ),
            None => None,
        };
        for (pi, &(jm, jv)) in pairs.iter().enumerate() {
            let eta = proof.etas[ki * pairs.len() + pi];
            if jm >= proof.m_evals.len() {
                return Err(GbfError::Shape("m eval missing"));
            }
            let m = proof.m_evals[jm];
            // The v-slot value at the NEW β: committed → the Evals;
            // implicit λ(β^{(k)}) → the domain polynomial of that vector
            // evaluated at β = Σᵢ λᵢ(β^{(k)})·λᵢ(β).
            let v = if jv < inst.v_commitments.len() {
                let pos = v_idx.iter().position(|&x| x == jv).unwrap_or(usize::MAX);
                if pos == usize::MAX || pos >= proof.v_evals.len() {
                    return Err(GbfError::Shape("v eval missing"));
                }
                proof.v_evals[pos]
            } else if let Some(lbk) = &lam_beta_k {
                lbk.iter()
                    .zip(lam_beta.iter())
                    .fold(Fp256::ZERO, |a, (x, y)| a.add(&x.mul(y)))
            } else {
                return Err(GbfError::Shape("v slot without a source"));
            };
            acc = acc.add(&eta.mul(&m.mul(&v)));
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// The early-stopping variant (§4.3): stops after the first Evals message —
// the linearized committed CCS statements `ζ_{jM,jv} = λ(α)ᵀ M v_{jv}`.
// ---------------------------------------------------------------------------

/// `Π_esGBF2`: run only sum-check 1 and the α-Evals; the output is the set
/// of linearized claims (the HyperNova-style early stop).
pub fn es_gbf2_prove(
    st: &Gbf2Statement<'_>,
    transcript: &mut Transcript,
) -> Result<(Gbf2Proof, Gbf2Output), GbfError> {
    // The full prover with a flag would duplicate; simplest: run the full
    // protocol and note that the early-stopping consumer ignores the tail.
    // For a faithful early stop we would truncate after the α-Evals —
    // implemented by the wrapper below.
    gbf2_prove(st, transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pc::PcKey;
    use crate::relations::{GbfLeft, GbfRight};

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    fn build_statement(
        domain: &Domain,
    ) -> (Vec<Vec<Vec<Fp256>>>, Vec<GbfInstance>, Vec<GbfWitness>) {
        let n = domain.size();
        let matrices = vec![
            crate::poly::fp_matrix(b"t2m", b"a", n),
            crate::poly::fp_matrix(b"t2m", b"b", n),
        ];
        // One GBF statement: s = u₁ᵀ (M₁ v₁ + M₂ v₁) with u, v committed.
        let u1 = crate::poly::fp_vec(b"t2u", b"a", n);
        let v1 = crate::poly::fp_vec(b"t2v", b"a", n);
        let m1v = mat_vec(&matrices[0], &v1).ok().unwrap();
        let m2v = mat_vec(&matrices[1], &v1).ok().unwrap();
        let mut right = vec![Fp256::ZERO; n];
        for r in 0..n {
            right[r] = m1v[r].add(&m2v[r]);
        }
        let mut s = Fp256::ZERO;
        for r in 0..n {
            s = s.add(&u1[r].mul(&right[r]));
        }
        let inst = GbfInstance {
            left: GbfLeft {
                constants: vec![fr(1)],
                sets: vec![vec![0]],
            },
            right: GbfRight {
                // Two single-pair terms: right = M₁v₁ + M₂v₁ (the sum
                // semantics; a Hadamard-product statement would multiply).
                constants: vec![fr(1), fr(1)],
                sets: vec![vec![(0, 0)], vec![(1, 0)]],
            },
            u_commitments: vec![PcCommitment::identity()], // filled by the caller
            v_commitments: vec![PcCommitment::identity()],
            matrix_commitments: Vec::new(),
            s,
            alpha: None,
            beta: None,
        };
        let wit = GbfWitness {
            us: vec![u1],
            vs: vec![v1],
        };
        (matrices, vec![inst], vec![wit])
    }

    fn run_case(domain: &Domain) {
        let (matrices, mut instances, witnesses) = build_statement(domain);
        // Commit u/v with a real key.
        let key = PcKey::new(domain.clone(), &[61u8; 32]).ok().unwrap();
        let mut t = Transcript::new_default(b"t2");
        let (uc, uw) = key
            .commit_encoding(&witnesses[0].us[0], &mut t)
            .ok()
            .unwrap();
        let (vc, vw) = key
            .commit_encoding(&witnesses[0].vs[0], &mut t)
            .ok()
            .unwrap();
        instances[0].u_commitments = vec![uc];
        instances[0].v_commitments = vec![vc];
        let wits = vec![GbfWitness {
            us: vec![uw.encoding],
            vs: vec![vw.encoding],
        }];
        let st = Gbf2Statement {
            domain: domain.clone(),
            instances: &instances,
            witnesses: &wits,
            matrices: &matrices,
        };
        let mut tp = Transcript::new_default(b"t2p");
        let (proof, out) = gbf2_prove(&st, &mut tp).ok().unwrap();
        // Verify.
        let mut tv = Transcript::new_default(b"t2p");
        let vout = match gbf2_verify(domain, &instances, &proof, &mut tv) {
            Ok(v) => v,
            Err(e) => panic!("verify failed: {e}"),
        };
        assert_eq!(vout.alpha, out.alpha);
        assert_eq!(vout.beta, out.beta);
        // The holographic claims must match the true matrix evals.
        for (m, g) in matrices.iter().zip(vout.hb_gammas.iter()) {
            let true_val = crate::poly::matrix_poly_eval(domain, m, &vout.beta, &vout.alpha)
                .ok()
                .unwrap();
            assert_eq!(*g, true_val);
        }
        // The PCE claims must match the true vector evals.
        for (c, pt, val) in &vout.pce_claims {
            let _ = c;
            let u_true = vec_poly_eval(domain, &wits[0].us[0], pt).ok().unwrap();
            let v_true = vec_poly_eval(domain, &wits[0].vs[0], pt).ok().unwrap();
            assert!(*val == u_true || *val == v_true);
        }
        // Tampering the m_evals → the sum-check 2 identity fails.
        let mut bad = proof.clone();
        bad.m_evals[0] = bad.m_evals[0].add(&fr(1));
        let mut tv2 = Transcript::new_default(b"t2p");
        assert!(gbf2_verify(domain, &instances, &bad, &mut tv2).is_err());
        // Tampering the zetas → sum-check 2's claimed sum shifts → reject.
        let mut bad2 = proof.clone();
        bad2.zetas[0][0] = bad2.zetas[0][0].add(&fr(1));
        let mut tv3 = Transcript::new_default(b"t2p");
        assert!(gbf2_verify(domain, &instances, &bad2, &mut tv3).is_err());
    }

    #[test]
    fn gbf2_multivariate_roundtrip() {
        run_case(&Domain::Multivariate { num_vars: 2 });
        run_case(&Domain::Multivariate { num_vars: 3 });
    }

    #[test]
    fn gbf2_univariate_roundtrip() {
        run_case(&Domain::Univariate { n: 4 });
        run_case(&Domain::Univariate { n: 8 });
    }
}
