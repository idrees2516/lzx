//! The non-uniform PCD **decider** (ePrint 2026/538 §5.3): for `ℓ`
//! functions each encoded with `t_M` CCS matrices, the final output
//! carries `ℓ` accumulators; the decider runs `Π_GBF,α,β` over the `ℓ+1`
//! statements, `Π_batchM` folds them to a single claim
//! `Σᵢ Σⱼ ηⱼ Mⱼ^{(i)}(β, α) = γ`, and settles it with ONE homomorphic
//! linear combination of the `ℓ·t_M` matrix commitments plus a single
//! evaluation check — instead of `2ℓ·t_M` matrix evaluations.
//!
//! With the transparent linear-opening PC the "single evaluation proof"
//! is the long opening of the combined matrix polynomial (O(n²) revealed
//! coefficients); with a homomorphic short-opening PC2 (KZG, IPA) the
//! same call site yields the paper's O(ℓ·t_M)-group-op decider.

use crate::barebones::AccStatement;
use crate::batch::batch_m;
use crate::gbf2::{gbf2_prove, gbf2_verify, Gbf2Statement};
use crate::pc::{combine_matrix_commitments, LinearOpening, PcCommitment, PcError, PcKey};
use crate::relations::{GbfInstance, GbfWitness};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeciderError {
    Pc(PcError),
    Gbf2(crate::gbf2::GbfError),
    Batch(crate::batch::BatchError),
    Fold(crate::fold::FoldError),
    Shape(&'static str),
}

impl core::fmt::Display for DeciderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DeciderError::Pc(e) => write!(f, "pc: {e}"),
            DeciderError::Gbf2(e) => write!(f, "gbf2: {e}"),
            DeciderError::Batch(e) => write!(f, "batch: {e}"),
            DeciderError::Fold(e) => write!(f, "fold: {e}"),
            DeciderError::Shape(s) => write!(f, "decider shape: {s}"),
        }
    }
}

impl From<PcError> for DeciderError {
    fn from(e: PcError) -> Self {
        DeciderError::Pc(e)
    }
}

impl From<crate::gbf2::GbfError> for DeciderError {
    fn from(e: crate::gbf2::GbfError) -> Self {
        DeciderError::Gbf2(e)
    }
}

impl From<crate::batch::BatchError> for DeciderError {
    fn from(e: crate::batch::BatchError) -> Self {
        DeciderError::Batch(e)
    }
}

impl From<crate::fold::FoldError> for DeciderError {
    fn from(e: crate::fold::FoldError) -> Self {
        DeciderError::Fold(e)
    }
}

/// The decider's interactive output: the combined claim + the combined
/// commitment (the prover side provides the opening).
#[derive(Clone, Debug)]
pub struct DeciderClaim {
    pub alpha: Vec<Fp256>,
    pub beta: Vec<Fp256>,
    /// `γ = Σᵢ Σⱼ ηⱼ·Mⱼ^{(i)}(β, α)`.
    pub gamma: Fp256,
    /// The η weights (per function i, per matrix j — flattened i·t_M + j).
    pub etas: Vec<Fp256>,
    /// The combined matrix commitment `[M] = Σ Σ ηⱼ·[Mⱼ^{(i)}]`.
    pub combined: PcCommitment,
    /// The long opening of the combined matrix polynomial.
    pub opening: LinearOpening,
}

/// Run the decider over the final accumulators of `ℓ` functions (the
/// prover side — it provides the combined opening).
#[allow(clippy::type_complexity)]
pub fn decide(
    key: &PcKey,
    accs: &[AccStatement],
    matrices_per_fn: &[Vec<Vec<Vec<Fp256>>>],
    matrix_commitments_per_fn: &[Vec<PcCommitment>],
    transcript: &mut Transcript,
) -> Result<bool, DeciderError> {
    if accs.len() != matrices_per_fn.len()
        || matrices_per_fn.len() != matrix_commitments_per_fn.len()
    {
        return Err(DeciderError::Shape("per-function arity"));
    }
    // 1. Π_GBF,α,β over the ℓ statements (all-implicit vectors), with the
    //    matrix indices re-based into the union index (function i's matrix
    //    j becomes union index offset_i + j — the paper's "index containing
    //    all the matrices").
    let instances: Vec<GbfInstance> = accs
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let mut g = a.gbf_alpha_beta.clone();
            let offset = matrix_commitments_per_fn
                .iter()
                .take(i)
                .map(|c| c.len())
                .sum::<usize>();
            for s_i in &mut g.right.sets {
                for p in s_i.iter_mut() {
                    p.0 += offset;
                }
            }
            g.matrix_commitments = Vec::new();
            g
        })
        .collect();
    let witnesses: Vec<GbfWitness> = (0..instances.len())
        .map(|_| GbfWitness {
            us: Vec::new(),
            vs: Vec::new(),
        })
        .collect();
    // The matrices differ per function — the multi-index regime. The GBF
    // run needs a single matrix set: use the union (the paper's decider
    // runs Π_GBF,α,β "on the index containing all the matrices"). Flatten.
    let mut all_matrices: Vec<Vec<Vec<Fp256>>> = Vec::new();
    let mut all_comms: Vec<PcCommitment> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new(); // per function: the matrix offset
    for (mats, coms) in matrices_per_fn.iter().zip(matrix_commitments_per_fn.iter()) {
        offsets.push(all_matrices.len());
        all_matrices.extend(mats.iter().cloned());
        all_comms.extend(coms.iter().cloned());
    }
    let st = Gbf2Statement {
        domain: key.domain.clone(),
        instances: &instances,
        witnesses: &witnesses,
        matrices: &all_matrices,
    };
    // The GBF run is self-contained (a dedicated transcript seeded by the
    // statements' own absorptions — replayable by the verifier below).
    let mut tgbf = Transcript::new_default(b"decider-gbf");
    let (gbf2_proof, gbf2_out) = gbf2_prove(&st, &mut tgbf)?;
    // 2. Π_batchM over the fresh matrix claims.
    let (folded, c) = batch_m(
        &all_comms,
        &gbf2_out.hb_gammas,
        &gbf2_out.alpha,
        &gbf2_out.beta,
        transcript,
    )?;
    // The folded statement's s = Σ cⱼ·γⱼ where γⱼ = Mⱼ(β,α) over the
    // UNION index — the paper's Eq. (17) form.
    let gamma = folded.s;
    // 3. The combined commitment + the single evaluation check.
    let items: Vec<(Fp256, PcCommitment)> = c
        .iter()
        .zip(all_comms.iter())
        .map(|(eta, com)| (*eta, *com))
        .collect();
    let combined = combine_matrix_commitments(&items);
    // The prover's opening: the combined matrix polynomial evaluated at
    // (β, α) — the long opening over the flattened coefficient vector.
    let combined_matrix: Vec<Vec<Fp256>> = {
        let n = key.domain.size();
        let mut m = vec![vec![Fp256::ZERO; n]; n];
        for (j, eta) in c.iter().enumerate() {
            for r in 0..n {
                for cc in 0..n {
                    m[r][cc] = m[r][cc].add(&eta.mul(&all_matrices[j][r][cc]));
                }
            }
        }
        m
    };
    let opening = key.open_matrix(&key.domain, &combined_matrix)?;
    let _t = Transcript::new_default(b"decider");
    let ok_open = key.verify_matrix_opening(&key.domain, &opening)?;
    if !ok_open {
        return Ok(false);
    }
    // [M] == Σ η [Mⱼ].
    if opening.commitment != combined {
        return Ok(false);
    }
    // M(β, α) == γ.
    let eval = key.eval_matrix(
        &key.domain,
        &combined_matrix,
        &gbf2_out.beta,
        &gbf2_out.alpha,
    )?;
    if eval != gamma {
        return Ok(false);
    }
    // 4. The GBF run's own verification — the self-contained replay.
    let mut treplay = Transcript::new_default(b"decider-gbf");
    let vout = gbf2_verify(&key.domain, &instances, &gbf2_proof, &mut treplay)?;
    if vout.alpha != gbf2_out.alpha || vout.beta != gbf2_out.beta {
        return Ok(false);
    }
    let _ = offsets;
    let _ = LinearOpening {
        commitment: PcCommitment::identity(),
        encoding: Vec::new(),
        blind: Fp256::ZERO,
    };
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barebones::barebones_prove;
    use crate::relations::Ccs;

    #[test]
    fn decider_two_functions() {
        let domain = crate::poly::Domain::Multivariate { num_vars: 3 };
        // Two functions (two CCS instances).
        let (ccs1, x1, w1) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"dec1");
        let (ccs2, x2, w2) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"dec2");
        let key = PcKey::new(domain.clone(), &[68u8; 32]).ok().unwrap();
        let coms1: Vec<PcCommitment> = ccs1
            .matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        let coms2: Vec<PcCommitment> = ccs2
            .matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        let mut accs = Vec::new();
        for (ccs, x, w, coms, seed) in [
            (&ccs1, &x1, &w1, &coms1, b"dec-a" as &[u8]),
            (&ccs2, &x2, &w2, &coms2, b"dec-b"),
        ] {
            let mut t = Transcript::new_default(seed);
            let (_p, acc) = barebones_prove(ccs, &key, coms, x, w, &mut t).ok().unwrap();
            accs.push(acc);
        }
        let mut td = Transcript::new_default(b"decider");
        let ok = match decide(
            &key,
            &accs,
            &[ccs1.matrices.clone(), ccs2.matrices.clone()],
            &[coms1, coms2],
            &mut td,
        ) {
            Ok(v) => v,
            Err(e) => panic!("decider failed: {e}"),
        };
        assert!(ok);
    }
}
