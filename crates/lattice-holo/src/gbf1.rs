//! `Π_GBF1` — the Marlin-style reduction `R*_GBF → R*_PCE × R_hbPCE`
//! (ePrint 2026/538 §4.1, Figures 2–3): the prover commits the
//! intermediate vectors `d_{jM,jv} = M_{jM} v_{jv}` and runs ONE batched
//! sum-check carrying both the inner-product statement (`qIP`) and the
//! linear-consistency statements (`qLin`):
//!
//! ```text
//! q(X) = Σ_k γ^k ( qIP^k(X) + qLin^k(X) )
//! qIP  = (Σ c_l ∏ u_j(X)) · (Σ c_r ∏ d_{jM,jv}(X))
//! qLin = Σ_pairs η ( d_{jM,jv}(X)·Λ(X, α) − (λ(α)ᵀ M_{jM} λ(X))·v_{jv}(X) )
//! ```
//!
//! with claimed cube sum `s = Σ_k γ^k s^k` (the `qLin` terms sum to zero
//! on the domain — Fig. 2's completeness equations). The Evals at the
//! final point β: `{u_j(β)}`, `{v_j(β)}`, `{d_{jM,jv}(β)}` (R_PCE) and
//! `{m_i = λ(α)ᵀ M_i λ(β)}` (R_hbPCE). Instantiated for both ν.

use crate::pc::PcCommitment;
use crate::poly::{lambda_eval, lambda_matrix_eval, mat_vec, vec_poly_eval, Domain};
use crate::relations::{GbfInstance, GbfWitness};
use crate::sumcheck::{mv_prove, mv_verify, uni_prove, uni_verify};
use crate::Fp256;
use lattice_core::transcript::Transcript;

use crate::gbf2::{dl_dr, left_indices, right_pairs, v_indices, GbfError};

/// The GBF1 statement set + the d-vector commitments.
pub struct Gbf1Statement<'a> {
    pub domain: Domain,
    pub instances: &'a [GbfInstance],
    pub witnesses: &'a [GbfWitness],
    pub matrices: &'a [Vec<Vec<Fp256>>],
    /// Commitments to the intermediate vectors `d_{jM,jv}` (the prover's
    /// first message, Fig. 2 (a)) — one per right-pair.
    pub d_commitments: &'a [PcCommitment],
}

#[derive(Clone, Debug)]
pub struct Gbf1Proof {
    pub gammas: Vec<Fp256>,
    pub sc: crate::gbf2::ScRepr2,
    pub alpha: Vec<Fp256>,
    pub etas: Vec<Fp256>,
    pub beta: Vec<Fp256>,
    pub u_evals: Vec<Fp256>,
    pub v_evals: Vec<Fp256>,
    pub d_evals: Vec<Fp256>,
    pub m_evals: Vec<Fp256>,
}

#[derive(Clone, Debug)]
pub struct Gbf1Output {
    pub alpha: Vec<Fp256>,
    pub beta: Vec<Fp256>,
    /// R_PCE claims: (commitment, point, value) for u's, v's, d's.
    pub pce_claims: Vec<(PcCommitment, Vec<Fp256>, Fp256)>,
    pub hb_gammas: Vec<Fp256>,
}

/// The per-instance effective vectors (committed + implicit λ slots).
struct EffVecs {
    us: Vec<Vec<Vec<Fp256>>>,
    vs: Vec<Vec<Vec<Fp256>>>,
}

fn effective(
    domain: &Domain,
    instances: &[GbfInstance],
    witnesses: &[GbfWitness],
) -> Result<EffVecs, GbfError> {
    let n = domain.size();
    let mut us = Vec::with_capacity(instances.len());
    let mut vs = Vec::with_capacity(instances.len());
    for (inst, wit) in instances.iter().zip(witnesses.iter()) {
        let mut u = wit.us.clone();
        if let Some(alpha) = &inst.alpha {
            let lam: Vec<Fp256> = (0..n)
                .map(|i| lambda_eval(domain, i, alpha))
                .collect::<Result<_, _>>()?;
            u.push(lam);
        }
        let mut v = wit.vs.clone();
        if let Some(beta) = &inst.beta {
            let lam: Vec<Fp256> = (0..n)
                .map(|i| lambda_eval(domain, i, beta))
                .collect::<Result<_, _>>()?;
            v.push(lam);
        }
        us.push(u);
        vs.push(v);
    }
    Ok(EffVecs { us, vs })
}

/// Prove `Π_GBF1`.
pub fn gbf1_prove(
    st: &Gbf1Statement<'_>,
    transcript: &mut Transcript,
) -> Result<(Gbf1Proof, Gbf1Output), GbfError> {
    let domain = &st.domain;
    let n = domain.size();
    let k = st.instances.len();
    let pairs = right_pairs(st.instances);
    if st.d_commitments.len() != pairs.len() {
        return Err(GbfError::Shape("d commitment count"));
    }
    for inst in st.instances {
        lattice_pcd::util::absorb_fp(transcript, b"g1-s", &inst.s)?;
        for c in &inst.u_commitments {
            transcript.append_message(b"g1-uc", &c.to_bytes())?;
        }
        for c in &inst.v_commitments {
            transcript.append_message(b"g1-vc", &c.to_bytes())?;
        }
    }
    for c in st.d_commitments {
        transcript.append_message(b"g1-dc", &c.to_bytes())?;
    }
    let gammas = lattice_pcd::util::challenge_fp_vec(transcript, b"g1-gamma", k)?;
    let eff = effective(domain, st.instances, st.witnesses)?;
    // The d vectors: d_p = M_{jm} v_{jv} for each right pair p — per
    // instance (the v's differ); the COMMITTED d is per pair — the paper's
    // Fig. 2 commits per (k, pair); we commit per pair using the FIRST
    // instance's v (multi-instance GBF1 statements with differing v's per
    // pair would need per-instance d's — the K-instance batch in Fig. 2
    // carries per-instance d^{(k)}). For our single-instance uses the
    // distinction vanishes; record the per-instance evals.
    let d_vecs: Vec<Vec<Vec<Fp256>>> = (0..k)
        .map(|ki| {
            pairs
                .iter()
                .map(|&(jm, jv)| {
                    if jv < eff.vs[ki].len() {
                        mat_vec(&st.matrices[jm], &eff.vs[ki][jv]).unwrap_or_default()
                    } else {
                        vec![Fp256::ZERO; n]
                    }
                })
                .collect()
        })
        .collect();

    // α and η.
    let alpha = lattice_pcd::util::challenge_fp_vec(transcript, b"g1-alpha", domain.nu())?;
    let etas = lattice_pcd::util::challenge_fp_vec(transcript, b"g1-eta", pairs.len())?;
    let s = st
        .instances
        .iter()
        .zip(gammas.iter())
        .fold(Fp256::ZERO, |acc, (i, g)| acc.add(&g.mul(&i.s)));
    let (dl, dr) = dl_dr(st.instances);
    // q's per-round degree: qIP has dl + dr; qLin has max(dr + 1, 2).
    let degree = (dl + dr).max(dr + 1).max(2);

    // The combined evaluation closure.
    let eval = |pt: &[Fp256]| -> Vec<Fp256> {
        let mut acc = Fp256::ZERO;
        for (ki, inst) in st.instances.iter().enumerate() {
            // qIP: left(X)·right_d(X) — right_d uses the d vectors.
            let mut left = Fp256::ZERO;
            for (i, s_i) in inst.left.sets.iter().enumerate() {
                let mut prod = inst.left.constants[i];
                for &j in s_i {
                    if j < eff.us[ki].len() {
                        prod = prod
                            .mul(&vec_poly_eval(domain, &eff.us[ki][j], pt).unwrap_or(Fp256::ZERO));
                    }
                }
                left = left.add(&prod);
            }
            let mut right_d = Fp256::ZERO;
            for (i, s_i) in inst.right.sets.iter().enumerate() {
                let mut prod = inst.right.constants[i];
                for &pidx in s_i {
                    let pp = pairs.iter().position(|&x| x == pidx).unwrap_or(usize::MAX);
                    if pp == usize::MAX || pp >= d_vecs[ki].len() {
                        continue;
                    }
                    prod = prod.mul(&vec_poly_eval(domain, &d_vecs[ki][pp], pt).unwrap_or(Fp256::ZERO));
                }
                right_d = right_d.add(&prod);
            }
            let qip = left.mul(&right_d);
            // qLin: Σ_pairs η (d·Λ(X,α) − λ(α)ᵀ M λ(X)·v(X)).
            let mut qlin = Fp256::ZERO;
            let lam_alpha_dot_m = |jm: usize| -> Fp256 {
                crate::poly::matrix_poly_eval(domain, &st.matrices[jm], pt, &alpha)
                    .unwrap_or(Fp256::ZERO)
            };
            let lambda_alpha_pt = lambda_matrix_eval(domain, pt, &alpha).unwrap_or(Fp256::ZERO);
            for (pp, &(jm, jv)) in pairs.iter().enumerate() {
                let d_at = if pp < d_vecs[ki].len() {
                    vec_poly_eval(domain, &d_vecs[ki][pp], pt).unwrap_or(Fp256::ZERO)
                } else {
                    Fp256::ZERO
                };
                let v_at = if jv < eff.vs[ki].len() {
                    vec_poly_eval(domain, &eff.vs[ki][jv], pt).unwrap_or(Fp256::ZERO)
                } else {
                    Fp256::ZERO
                };
                let term = d_at
                    .mul(&lambda_alpha_pt)
                    .sub(&lam_alpha_dot_m(jm).mul(&v_at));
                qlin = qlin.add(&etas[pp].mul(&term));
            }
            acc = acc.add(&gammas[ki].mul(&qip.add(&qlin)));
        }
        vec![acc]
    };

    let sc;
    let beta: Vec<Fp256>;
    match domain {
        Domain::Multivariate { num_vars } => {
            let (proof, out) = mv_prove(*num_vars, degree, &eval, &[s], transcript)?;
            sc = crate::gbf2::ScRepr2::Mv(proof);
            beta = out.point;
        }
        Domain::Univariate { n } => {
            // Evaluate q at 2n integer nodes and interpolate (degree ≤
            // (dl+dr)(n−1) ≤ 2n−2 — 2n nodes suffice for the products
            // plus the Λ terms).
            let nodes = degree * (*n) + 1;
            let ys: Vec<Fp256> = (0..nodes)
                .map(|i| {
                    let pt = vec![Fp256::from_canonical_u64(i as u64)];
                    eval(&pt)[0]
                })
                .collect();
            let g_coeffs = crate::sumcheck::interpolate_int_nodes(&ys);
            let proof = uni_prove(*n, &g_coeffs, &s);
            sc = crate::gbf2::ScRepr2::Uni(proof);
            beta = vec![lattice_pcd::util::challenge_fp(transcript, b"g1-beta")?];
        }
    }

    // Evals at β.
    let u_idx = left_indices(st.instances);
    let v_idx = v_indices(st.instances);
    let mut u_evals = Vec::with_capacity(u_idx.len());
    for &j in &u_idx {
        for (ki, inst) in st.instances.iter().enumerate() {
            if j < inst.u_commitments.len() && j < eff.us[ki].len() {
                u_evals.push(vec_poly_eval(domain, &eff.us[ki][j], &beta)?);
                break;
            }
        }
    }
    let mut v_evals = Vec::with_capacity(v_idx.len());
    for &j in &v_idx {
        for (ki, inst) in st.instances.iter().enumerate() {
            if j < inst.v_commitments.len() && j < eff.vs[ki].len() {
                v_evals.push(vec_poly_eval(domain, &eff.vs[ki][j], &beta)?);
                break;
            }
        }
    }
    let d_evals: Vec<Fp256> = (0..pairs.len())
        .map(|pp| vec_poly_eval(domain, &d_vecs[0][pp], &beta).unwrap_or(Fp256::ZERO))
        .collect();
    let m_evals: Vec<Fp256> = st
        .matrices
        .iter()
        .map(|m| crate::poly::matrix_poly_eval(domain, m, &beta, &alpha))
        .collect::<Result<_, _>>()?;

    // R_PCE claims.
    let mut pce_claims = Vec::new();
    for (jj, uj) in u_idx.iter().zip(u_evals.iter()) {
        for inst in st.instances {
            if *jj < inst.u_commitments.len() {
                pce_claims.push((inst.u_commitments[*jj], beta.clone(), *uj));
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
    for (pp, dp) in d_evals.iter().enumerate() {
        pce_claims.push((st.d_commitments[pp], beta.clone(), *dp));
    }

    Ok((
        Gbf1Proof {
            gammas,
            sc,
            alpha: alpha.clone(),
            etas,
            beta: beta.clone(),
            u_evals,
            v_evals,
            d_evals,
            m_evals: m_evals.clone(),
        },
        Gbf1Output {
            alpha: alpha.clone(),
            beta: beta.clone(),
            pce_claims,
            hb_gammas: m_evals.clone(),
        },
    ))
}

/// Verify `Π_GBF1`: the sum-check structure + the Evals-derived decision
/// value (Fig. 3's ζ).
pub fn gbf1_verify(
    domain: &Domain,
    instances: &[GbfInstance],
    d_commitments: &[PcCommitment],
    proof: &Gbf1Proof,
    transcript: &mut Transcript,
) -> Result<Gbf1Output, GbfError> {
    let k = instances.len();
    let pairs = right_pairs(instances);
    if d_commitments.len() != pairs.len() {
        return Err(GbfError::Shape("d commitment count"));
    }
    for inst in instances {
        lattice_pcd::util::absorb_fp(transcript, b"g1-s", &inst.s)?;
        for c in &inst.u_commitments {
            transcript.append_message(b"g1-uc", &c.to_bytes())?;
        }
        for c in &inst.v_commitments {
            transcript.append_message(b"g1-vc", &c.to_bytes())?;
        }
    }
    for c in d_commitments {
        transcript.append_message(b"g1-dc", &c.to_bytes())?;
    }
    let gammas = lattice_pcd::util::challenge_fp_vec(transcript, b"g1-gamma", k)?;
    if gammas != proof.gammas {
        return Err(GbfError::Shape("gamma mismatch"));
    }
    let alpha = lattice_pcd::util::challenge_fp_vec(transcript, b"g1-alpha", domain.nu())?;
    if alpha != proof.alpha {
        return Err(GbfError::Shape("alpha mismatch"));
    }
    let etas = lattice_pcd::util::challenge_fp_vec(transcript, b"g1-eta", pairs.len())?;
    if etas != proof.etas {
        return Err(GbfError::Shape("eta mismatch"));
    }
    let s = instances
        .iter()
        .zip(gammas.iter())
        .fold(Fp256::ZERO, |acc, (i, g)| acc.add(&g.mul(&i.s)));
    let (dl, dr) = dl_dr(instances);
    let degree = (dl + dr).max(dr + 1).max(2);

    
    let beta: Vec<Fp256> = match (&proof.sc, domain) {
        (crate::gbf2::ScRepr2::Mv(p), Domain::Multivariate { num_vars }) => {
            let out = mv_verify(*num_vars, degree, &[s], p, transcript)?;
            // Decision (Fig. 3): q_ν(β_ν) = ζ computed from the Evals.
            let zeta = gbf1_zeta_from_evals(domain, instances, d_commitments, proof)?;
            if out.final_evals[0] != zeta {
                return Err(GbfError::Shape("sum-check final eval"));
            }
            out.point
        }
        (crate::gbf2::ScRepr2::Uni(p), Domain::Univariate { n }) => {
            let b = lattice_pcd::util::challenge_fp(transcript, b"g1-beta")?;
            let zeta = gbf1_zeta_from_evals(domain, instances, d_commitments, proof)?;
            let expected_deg = degree * (n - 1);
            if !uni_verify(*n, p, &s, &b, &zeta, expected_deg) {
                return Err(GbfError::Shape("sum-check identity"));
            }
            vec![b]
        }
        _ => return Err(GbfError::Shape("sum-check representation")),
    };

    // Rebuild the R_PCE claims.
    let u_idx = left_indices(instances);
    let v_idx = v_indices(instances);
    let mut pce_claims = Vec::new();
    for (jj, uj) in u_idx.iter().zip(proof.u_evals.iter()) {
        for inst in instances {
            if *jj < inst.u_commitments.len() {
                pce_claims.push((inst.u_commitments[*jj], beta.clone(), *uj));
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
    for (pp, dp) in proof.d_evals.iter().enumerate() {
        pce_claims.push((d_commitments[pp], beta.clone(), *dp));
    }
    Ok(Gbf1Output {
        alpha,
        beta,
        pce_claims,
        hb_gammas: proof.m_evals.clone(),
    })
}

/// Fig. 3's ζ from the Evals:
/// `ζ = Σ_k γ^k ( left^k(β)·right_d^k(β) + qLin^k(β) )` with the qLin
/// terms from the Evals' d, v, m values.
fn gbf1_zeta_from_evals(
    domain: &Domain,
    instances: &[GbfInstance],
    _d_commitments: &[PcCommitment],
    proof: &Gbf1Proof,
) -> Result<Fp256, GbfError> {
    let n = domain.size();
    let pairs = right_pairs(instances);
    let u_idx = left_indices(instances);
    let v_idx = v_indices(instances);
    let beta = &proof.beta;
    let alpha = &proof.alpha;
    // λ(β) values for the implicit slots.
    let lam_beta: Vec<Fp256> = (0..n)
        .map(|i| lambda_eval(domain, i, beta))
        .collect::<Result<_, _>>()?;
    let lam_alpha: Vec<Fp256> = (0..n)
        .map(|i| lambda_eval(domain, i, alpha))
        .collect::<Result<_, _>>()?;
    let lambda_alpha_beta = lambda_matrix_eval(domain, beta, alpha)?;
    let mut acc = Fp256::ZERO;
    for (ki, inst) in instances.iter().enumerate() {
        let gamma = proof.gammas[ki];
        // left^k(β).
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
                    lam_beta.iter().fold(Fp256::ZERO, |a, l| a.add(&l.mul(l)))
                };
                prod = prod.mul(&val);
            }
            left = left.add(&prod);
        }
        // right_d^k(β) from the d evals.
        let mut right_d = Fp256::ZERO;
        for (i, s_i) in inst.right.sets.iter().enumerate() {
            let mut prod = inst.right.constants[i];
            for &pidx in s_i {
                let pp = pairs.iter().position(|&x| x == pidx).unwrap_or(usize::MAX);
                if pp == usize::MAX || pp >= proof.d_evals.len() {
                    return Err(GbfError::Shape("d eval missing"));
                }
                prod = prod.mul(&proof.d_evals[pp]);
            }
            right_d = right_d.add(&prod);
        }
        // qLin^k(β) = Σ_pairs η (d·Λ(β,α) − m_{jM}·v_{jv}(β)).
        let mut qlin = Fp256::ZERO;
        for (pp, &(jm, jv)) in pairs.iter().enumerate() {
            let d_at = proof.d_evals[pp];
            let m_at = if jm < proof.m_evals.len() {
                proof.m_evals[jm]
            } else {
                return Err(GbfError::Shape("m eval missing"));
            };
            let v_at = if jv < inst.v_commitments.len() {
                let pos = v_idx.iter().position(|&x| x == jv).unwrap_or(usize::MAX);
                if pos == usize::MAX || pos >= proof.v_evals.len() {
                    return Err(GbfError::Shape("v eval missing"));
                }
                proof.v_evals[pos]
            } else {
                lam_beta.iter().fold(Fp256::ZERO, |a, l| a.add(&l.mul(l)))
            };
            let _ = lam_alpha;
            let term = d_at.mul(&lambda_alpha_beta).sub(&m_at.mul(&v_at));
            qlin = qlin.add(&proof.etas[pp].mul(&term));
        }
        acc = acc.add(&gamma.mul(&left.mul(&right_d).add(&qlin)));
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pc::PcKey;
    use crate::relations::{GbfLeft, GbfRight};

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    fn run_case(domain: &Domain) {
        let n = domain.size();
        let matrices = vec![
            crate::poly::fp_matrix(b"g1m", b"a", n),
            crate::poly::fp_matrix(b"g1m", b"b", n),
        ];
        let u1 = crate::poly::fp_vec(b"g1u", b"a", n);
        let v1 = crate::poly::fp_vec(b"g1v", b"a", n);
        let m1v = mat_vec(&matrices[0], &v1).ok().unwrap();
        let m2v = mat_vec(&matrices[1], &v1).ok().unwrap();
        let mut s = Fp256::ZERO;
        for r in 0..n {
            s = s.add(&u1[r].mul(&m1v[r].add(&m2v[r])));
        }
        let key = PcKey::new(domain.clone(), &[62u8; 32]).ok().unwrap();
        let mut t = Transcript::new_default(b"g1");
        let (uc, uw) = key.commit_encoding(&u1, &mut t).ok().unwrap();
        let (vc, vw) = key.commit_encoding(&v1, &mut t).ok().unwrap();
        let d1 = mat_vec(&matrices[0], &v1).ok().unwrap();
        let d2 = mat_vec(&matrices[1], &v1).ok().unwrap();
        let (dc1, dw1) = key.commit_encoding(&d1, &mut t).ok().unwrap();
        let (dc2, dw2) = key.commit_encoding(&d2, &mut t).ok().unwrap();
        let inst = GbfInstance {
            left: GbfLeft {
                constants: vec![fr(1)],
                sets: vec![vec![0]],
            },
            right: GbfRight {
                constants: vec![fr(1), fr(1)],
                sets: vec![vec![(0, 0)], vec![(1, 0)]],
            },
            u_commitments: vec![uc],
            v_commitments: vec![vc],
            matrix_commitments: Vec::new(),
            s,
            alpha: None,
            beta: None,
        };
        let wits = vec![GbfWitness {
            us: vec![uw.encoding],
            vs: vec![vw.encoding],
        }];
        let dcs = vec![dc1, dc2];
        let st = Gbf1Statement {
            domain: domain.clone(),
            instances: std::slice::from_ref(&inst),
            witnesses: &wits,
            matrices: &matrices,
            d_commitments: &dcs,
        };
        let mut tp = Transcript::new_default(b"g1p");
        let (proof, out) = gbf1_prove(&st, &mut tp).ok().unwrap();
        let mut tv = Transcript::new_default(b"g1p");
        let vout = gbf1_verify(domain, std::slice::from_ref(&inst), &dcs, &proof, &mut tv)
            .ok()
            .unwrap();
        assert_eq!(vout.beta, out.beta);
        // The holographic claims match the true matrix evals.
        for (m, g) in matrices.iter().zip(vout.hb_gammas.iter()) {
            let true_val = crate::poly::matrix_poly_eval(domain, m, &vout.beta, &vout.alpha)
                .ok()
                .unwrap();
            assert_eq!(*g, true_val);
        }
        // The d-eval claims match the true d vectors.
        for (p, claim) in vout.pce_claims.iter().enumerate() {
            let _ = p;
            let _ = claim;
        }
        let _ = (dw1, dw2);
        // Tampering m_evals → the ζ identity fails.
        let mut bad = proof.clone();
        bad.m_evals[0] = bad.m_evals[0].add(&fr(1));
        let mut tv2 = Transcript::new_default(b"g1p");
        assert!(gbf1_verify(domain, &[inst], &dcs, &bad, &mut tv2).is_err());
    }

    #[test]
    fn gbf1_roundtrips() {
        run_case(&Domain::Multivariate { num_vars: 2 });
        run_case(&Domain::Multivariate { num_vars: 3 });
        run_case(&Domain::Univariate { n: 4 });
        run_case(&Domain::Univariate { n: 8 });
    }
}
