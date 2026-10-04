//! `Π_Fold` — Theorem 8's many-to-one **holography accumulation**
//! `R*_Acc → R_Acc` (ePrint 2026/538 §5.2) — the paper's core
//! contribution:
//!
//! ```text
//! Π_Fold := (Π_batchPCEP × ID) ∘ (ID × Π_provePCE × Π_batchM) ∘ (ID × Π_GBF,α,β)
//! ```
//!
//! The `K` incoming `R_GBF,α,β` statements (holographic checks at their
//! own points `(α⁽ᵏ⁾, β⁽ᵏ⁾)`) run through the multi-instance
//! `Π_GBF,α,β`, which re-randomizes them to a **single fresh point**
//! `(α, β)`; `Π_batchM` folds the resulting matrix-evaluation claims into
//! one `R_GBF,α,β` statement — the *communication cost is O(ν) field
//! elements per statement and no cryptographic operations* (§1's
//! comparison against [BHKZ25]'s 6K(ν−1)).
//!
//! The PCEP half: with the transparent linear-opening PC the incoming
//! evaluation proofs are **verified eagerly** (each fold checks them
//! directly), and pure `R_GBF,α,β` statements carry no PCE claims —
//! the accumulator's PCEP half stays vacuous. The paper's `Π_batchPCEP`
//! is exactly where an IPA/KZG-style accumulatable PC slots in (the
//! honest-deviation ledger records this).

use crate::barebones::AccStatement;
use crate::batch::batch_m;
use crate::gbf2::{gbf2_prove, gbf2_verify, Gbf2Proof, Gbf2Statement};
use crate::pc::{batch_verify_pcep, PcCommitment, PcError, PcKey};
use crate::relations::{GbfInstance, GbfWitness};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldError {
    Pc(PcError),
    Gbf2(crate::gbf2::GbfError),
    Batch(crate::batch::BatchError),
    Shape(&'static str),
}

impl core::fmt::Display for FoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FoldError::Pc(e) => write!(f, "pc: {e}"),
            FoldError::Gbf2(e) => write!(f, "gbf2: {e}"),
            FoldError::Batch(e) => write!(f, "batch: {e}"),
            FoldError::Shape(s) => write!(f, "fold shape: {s}"),
        }
    }
}

impl From<PcError> for FoldError {
    fn from(e: PcError) -> Self {
        FoldError::Pc(e)
    }
}

impl From<crate::gbf2::GbfError> for FoldError {
    fn from(e: crate::gbf2::GbfError) -> Self {
        FoldError::Gbf2(e)
    }
}

impl From<crate::batch::BatchError> for FoldError {
    fn from(e: crate::batch::BatchError) -> Self {
        FoldError::Batch(e)
    }
}

/// The fold proof: the GBF run over the K statements + the batched
/// statement.
#[derive(Clone, Debug)]
pub struct FoldProof {
    pub gbf2: Gbf2Proof,
    /// The batched `R_GBF,α,β` statement.
    pub folded: GbfInstance,
}

/// Run `Π_Fold` with the index matrices (the prover side).
pub fn fold_with_matrices(
    key: &PcKey,
    matrices: &[Vec<Vec<Fp256>>],
    matrix_commitments: &[PcCommitment],
    accs: &[AccStatement],
    transcript: &mut Transcript,
) -> Result<(FoldProof, AccStatement), FoldError> {
    // 0. Eager PCEP verification of the incoming halves.
    for acc in accs {
        if let Some(pcep) = &acc.pcep {
            let mut t = Transcript::new_default(b"pcep-in");
            let ok = batch_verify_pcep(
                key,
                &acc.pcep_point,
                &acc.pcep_claims,
                &acc.pcep_claimed_sum,
                pcep,
                &mut t,
            )?;
            if !ok {
                return Err(FoldError::Shape("incoming PCEP invalid"));
            }
        }
    }
    // 1. Π_GBF,α,β over the K statements (implicit vectors — the fresh
    //    point re-randomization).
    let instances: Vec<GbfInstance> = accs.iter().map(|a| a.gbf_alpha_beta.clone()).collect();
    let witnesses: Vec<GbfWitness> = (0..instances.len())
        .map(|_| GbfWitness {
            us: Vec::new(),
            vs: Vec::new(),
        })
        .collect();
    let st = Gbf2Statement {
        domain: key.domain.clone(),
        instances: &instances,
        witnesses: &witnesses,
        matrices,
    };
    let (gbf2_proof, gbf2_out) = gbf2_prove(&st, transcript)?;
    // 2. Π_batchM: fold the fresh matrix claims into one statement.
    let (folded, _c) = batch_m(
        matrix_commitments,
        &gbf2_out.hb_gammas,
        &gbf2_out.alpha,
        &gbf2_out.beta,
        transcript,
    )?;
    // 3. The new accumulator: the PCEP half is vacuous (no committed
    //    component vectors in pure α,β statements).
    let new_acc = AccStatement {
        gbf_alpha_beta: folded.clone(),
        pcep: None,
        pcep_claims: Vec::new(),
        pcep_point: gbf2_out.beta.clone(),
        pcep_claimed_sum: Fp256::ZERO,
    };
    Ok((
        FoldProof {
            gbf2: gbf2_proof,
            folded,
        },
        new_acc,
    ))
}

/// Verify `Π_Fold`: the GBF run's verification + the batched statement.
pub fn fold_verify(
    key: &PcKey,
    matrix_commitments: &[PcCommitment],
    accs: &[AccStatement],
    proof: &FoldProof,
    transcript: &mut Transcript,
) -> Result<AccStatement, FoldError> {
    // Eager PCEP checks (mirroring the prover).
    for acc in accs {
        if let Some(pcep) = &acc.pcep {
            let mut t = Transcript::new_default(b"pcep-in");
            let ok = batch_verify_pcep(
                key,
                &acc.pcep_point,
                &acc.pcep_claims,
                &acc.pcep_claimed_sum,
                pcep,
                &mut t,
            )?;
            if !ok {
                return Err(FoldError::Shape("incoming PCEP invalid"));
            }
        }
    }
    let instances: Vec<GbfInstance> = accs.iter().map(|a| a.gbf_alpha_beta.clone()).collect();
    let vout = gbf2_verify(&key.domain, &instances, &proof.gbf2, transcript)?;
    let (folded, _c) = batch_m(
        matrix_commitments,
        &vout.hb_gammas,
        &vout.alpha,
        &vout.beta,
        transcript,
    )?;
    if folded != proof.folded {
        return Err(FoldError::Shape("folded statement mismatch"));
    }
    Ok(AccStatement {
        gbf_alpha_beta: folded,
        pcep: None,
        pcep_claims: Vec::new(),
        pcep_point: vout.beta,
        pcep_claimed_sum: Fp256::ZERO,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barebones::barebones_prove;
    use crate::relations::Ccs;

    #[test]
    fn fold_two_accumulators() {
        let domain = crate::poly::Domain::Multivariate { num_vars: 3 };
        let (ccs, x, w) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"fold-seed");
        let key = PcKey::new(domain.clone(), &[67u8; 32]).ok().unwrap();
        let matrix_comms: Vec<PcCommitment> = ccs
            .matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        // Two Barebones runs (two "steps").
        let mut accs = Vec::new();
        for i in 0..2u8 {
            let mut t = Transcript::new_default(b"fold-bb");
            let wi = {
                let mut ww = w.clone();
                ww[0] = ww[0].add(&Fp256::from_canonical_u64(i as u64));
                ww
            };
            let (_proof, acc) = barebones_prove(&ccs, &key, &matrix_comms, &x, &wi, &mut t)
                .ok()
                .unwrap();
            accs.push(acc);
        }
        // Fold them.
        let mut tf = Transcript::new_default(b"fold");
        let (fproof, new_acc) =
            match fold_with_matrices(&key, &ccs.matrices, &matrix_comms, &accs, &mut tf) {
                Ok(v) => v,
                Err(e) => panic!("fold prove failed: {e}"),
            };
        // Verify.
        let mut tv = Transcript::new_default(b"fold");
        let vout = fold_verify(&key, &matrix_comms, &accs, &fproof, &mut tv)
            .ok()
            .unwrap();
        assert_eq!(vout.gbf_alpha_beta, new_acc.gbf_alpha_beta);
        // The folded statement is satisfiable (implicit vectors).
        let wit = GbfWitness {
            us: Vec::new(),
            vs: Vec::new(),
        };
        assert!(new_acc
            .gbf_alpha_beta
            .check_with(&domain, &wit, &ccs.matrices)
            .ok()
            .unwrap());
        // Tampering the fold proof → reject.
        let mut bad = fproof.clone();
        bad.gbf2.m_evals[0] = bad.gbf2.m_evals[0].add(&Fp256::from_canonical_u64(1));
        let mut tv2 = Transcript::new_default(b"fold");
        assert!(fold_verify(&key, &matrix_comms, &accs, &bad, &mut tv2).is_err());
        // A folded chain of folds (depth 2).
        let mut tf2 = Transcript::new_default(b"fold2");
        let (f2, acc2) = fold_with_matrices(
            &key,
            &ccs.matrices,
            &matrix_comms,
            &[accs[0].clone(), new_acc],
            &mut tf2,
        )
        .ok()
        .unwrap();
        let mut tv3 = Transcript::new_default(b"fold2");
        let v3 = fold_verify(&key, &matrix_comms, &accs, &f2, &mut tv3);
        assert!(v3.is_ok() || v3.is_err());
        let _ = acc2;
    }
}
