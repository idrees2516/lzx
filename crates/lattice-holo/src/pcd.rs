//! The PCD construction (ePrint 2026/538, Corollary 3): NI-Barebones as
//! the argument `ARG` and NI-ΠFold as the accumulation scheme `ACC`,
//! composed via the [BMNW25 Thm. 4.3] transformation into a PCD scheme
//! for constant-depth compliance predicates — the **stateless recursion**
//! the paper targets: each step's prover needs only the previous proofs
//! and the public accumulator, never the previous witnesses.
//!
//! This module drives a PCD chain: each node proves its compliance step
//! (a CCS instance whose public input carries the incoming messages) with
//! Barebones, folds the incoming accumulators with `Π_Fold`, and passes
//! `(proof, acc)` forward; the final verifier runs the §5.3 decider.

use crate::barebones::{barebones_prove, barebones_verify, AccStatement, BarebonesProof};
use crate::decider::decide;
use crate::fold::{fold_verify, fold_with_matrices, FoldProof};
use crate::pc::{PcCommitment, PcKey};
use crate::relations::{Ccs, RelError};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcdHoloError {
    Rel(RelError),
    Barebones(crate::barebones::BarebonesError),
    Fold(crate::fold::FoldError),
    Decider(crate::decider::DeciderError),
    Shape(&'static str),
}

impl core::fmt::Display for PcdHoloError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PcdHoloError::Rel(e) => write!(f, "relation: {e}"),
            PcdHoloError::Barebones(e) => write!(f, "barebones: {e}"),
            PcdHoloError::Fold(e) => write!(f, "fold: {e}"),
            PcdHoloError::Decider(e) => write!(f, "decider: {e}"),
            PcdHoloError::Shape(s) => write!(f, "pcd shape: {s}"),
        }
    }
}

impl From<RelError> for PcdHoloError {
    fn from(e: RelError) -> Self {
        PcdHoloError::Rel(e)
    }
}

impl From<crate::barebones::BarebonesError> for PcdHoloError {
    fn from(e: crate::barebones::BarebonesError) -> Self {
        PcdHoloError::Barebones(e)
    }
}

impl From<crate::fold::FoldError> for PcdHoloError {
    fn from(e: crate::fold::FoldError) -> Self {
        PcdHoloError::Fold(e)
    }
}

impl From<crate::decider::DeciderError> for PcdHoloError {
    fn from(e: crate::decider::DeciderError) -> Self {
        PcdHoloError::Decider(e)
    }
}

/// A node's PCD proof: the Barebones argument + the fold + the accumulator.
#[derive(Clone, Debug)]
pub struct PcdHoloProof {
    pub barebones: BarebonesProof,
    pub fold: Option<FoldProof>,
    /// The public input of this node's CCS instance.
    pub x: Vec<Fp256>,
}

/// `PCD.P` at a node: prove the step and fold the incoming accumulators.
#[allow(clippy::too_many_arguments)]
pub fn pcd_prove_step(
    ccs: &Ccs,
    key: &PcKey,
    matrix_commitments: &[PcCommitment],
    x: &[Fp256],
    w: &[Fp256],
    incoming: &[AccStatement],
    transcript: &mut Transcript,
) -> Result<(PcdHoloProof, AccStatement), PcdHoloError> {
    // 1. NI-Barebones for this step's CCS instance.
    let (bb, mut acc) = barebones_prove(ccs, key, matrix_commitments, x, w, transcript)?;
    // 2. NI-ΠFold over the incoming accumulators + this step's fresh one.
    let fold = if incoming.is_empty() {
        None
    } else {
        let mut all: Vec<AccStatement> = incoming.to_vec();
        all.push(acc.clone());
        let (fp, new_acc) =
            fold_with_matrices(key, &ccs.matrices, matrix_commitments, &all, transcript)?;
        acc = new_acc;
        Some(fp)
    };
    Ok((
        PcdHoloProof {
            barebones: bb,
            fold,
            x: x.to_vec(),
        },
        acc,
    ))
}

/// `PCD.V` (the chain driver): verify a step's proof given the incoming
/// accumulator statements.
pub fn pcd_verify_step(
    ccs: &Ccs,
    key: &PcKey,
    matrix_commitments: &[PcCommitment],
    incoming: &[AccStatement],
    proof: &PcdHoloProof,
    acc: &AccStatement,
    transcript: &mut Transcript,
) -> Result<bool, PcdHoloError> {
    // 1. Verify the Barebones argument.
    let ok_bb = barebones_verify(
        ccs,
        key,
        matrix_commitments,
        &proof.x,
        &proof.barebones,
        transcript,
    )?;
    if !ok_bb {
        return Ok(false);
    }
    // 2. Verify the fold (if present) and the resulting accumulator.
    match &proof.fold {
        None => Ok(acc == &proof.barebones_pseudo_acc()),
        Some(fp) => {
            let mut all: Vec<AccStatement> = incoming.to_vec();
            all.push(proof.barebones_pseudo_acc());
            let folded = fold_verify(key, matrix_commitments, &all, fp, transcript)?;
            Ok(folded == *acc)
        }
    }
}

impl PcdHoloProof {
    /// The barebones step's own accumulator statement (reconstructed —
    /// in the driver the prover's `acc` from step 1 is needed; we re-run
    /// the batchM half from the proof's data).
    fn barebones_pseudo_acc(&self) -> AccStatement {
        AccStatement {
            gbf_alpha_beta: self.barebones.gbf_alpha_beta.clone(),
            pcep: self.barebones.pcep.clone(),
            pcep_claims: self.barebones.pcep_claims.clone(),
            pcep_point: self.barebones.pcep_point(),
            pcep_claimed_sum: self.barebones.pcep_claimed_sum,
        }
    }
}

/// The final decider over a chain's accumulator (single function).
pub fn pcd_decide(
    key: &PcKey,
    ccs: &Ccs,
    matrix_commitments: &[PcCommitment],
    acc: &AccStatement,
    transcript: &mut Transcript,
) -> Result<bool, PcdHoloError> {
    let ok = decide(
        key,
        std::slice::from_ref(acc),
        std::slice::from_ref(&ccs.matrices),
        std::slice::from_ref(&matrix_commitments.to_vec()),
        transcript,
    )?;
    Ok(ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn pcd_chain_three_steps() {
        let domain = crate::poly::Domain::Multivariate { num_vars: 3 };
        let (ccs, _x0, w0) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"pcdh-seed");
        let key = PcKey::new(domain.clone(), &[69u8; 32]).ok().unwrap();
        let matrix_comms: Vec<PcCommitment> = ccs
            .matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        // The compliance predicate: each step's public input encodes the
        // previous message + a step counter (a "constant-depth" chain).
        let mut accs: Vec<AccStatement> = Vec::new();
        let mut proofs: Vec<PcdHoloProof> = Vec::new();
        let mut prev_msg = Fp256::ZERO;
        for step in 0..3u64 {
            // x = [prev_msg, step] + padding to s = 2.
            let x = vec![prev_msg, fr(step + 1)];
            // A satisfying witness (the zeroed-matrix construction accepts
            // any w — the compliance semantics live in x).
            let w = {
                let mut ww = w0.clone();
                ww[0] = ww[0].add(&fr(step * 7));
                ww
            };
            let mut t = Transcript::new_default(b"pcdh-step");
            t.append_message(b"pcdh-step", &step.to_le_bytes()).ok().unwrap();
            let (proof, acc) = match pcd_prove_step(&ccs, &key, &matrix_comms, &x, &w, &accs, &mut t) {
                Ok(v) => v,
                Err(e) => panic!("pcd prove failed: {e}"),
            };
            prev_msg = fr(step + 1);
            accs.push(acc);
            proofs.push(proof);
        }
        // Verify each step.
        for step in 0..3usize {
            let incoming: Vec<AccStatement> = accs[..step].to_vec();
            let mut tv = Transcript::new_default(b"pcdh-step");
            tv.append_message(b"pcdh-step", &(step as u64).to_le_bytes()).ok().unwrap();
            // The prover's fold input: incoming + the barebones acc —
            // reconstruct via the proof.
            let ok = pcd_verify_step(
                &ccs,
                &key,
                &matrix_comms,
                &incoming,
                &proofs[step],
                &accs[step],
                &mut tv,
            )
            .ok()
            .unwrap();
            assert!(ok, "step {step} verification failed");
        }
        // The final decider.
        let mut td = Transcript::new_default(b"pcdh-dec");
        let ok = match pcd_decide(&key, &ccs, &matrix_comms, &accs[2], &mut td) {
            Ok(v) => v,
            Err(e) => panic!("decider failed: {e}"),
        };
        assert!(ok);

        // Tampering the final accumulator → decider rejects.
        let mut bad = accs[2].clone();
        bad.gbf_alpha_beta.s = bad.gbf_alpha_beta.s.add(&fr(1));
        let mut td2 = Transcript::new_default(b"pcdh-dec");
        let ok_bad = pcd_decide(&key, &ccs, &matrix_comms, &bad, &mut td2).unwrap_or(false);
        assert!(!ok_bad);
    }
}
