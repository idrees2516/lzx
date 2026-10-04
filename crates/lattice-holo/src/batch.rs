//! `Π_batchM` — Lemma 1's linear-combination reduction
//! `R_hbPCE → R_GBF,α,β` (ePrint 2026/538): given claimed matrix
//! evaluations `γᵢ = Mᵢ(β, α)` at a common point, the verifier samples
//! `c ∈ F^{t_M}` and the parties derive the single statement
//! `s = Σᵢ cᵢ·γᵢ = λ(α)ᵀ (Σᵢ cᵢ Mᵢ λ(β))` — the `R_GBF,α,β` instance
//! with both component vectors implicit.

use crate::relations::{build_gbf_alpha_beta, GbfInstance};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    Shape(&'static str),
    Transcript(lattice_core::transcript::TranscriptError),
}

impl core::fmt::Display for BatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BatchError::Shape(s) => write!(f, "batch shape: {s}"),
            BatchError::Transcript(e) => write!(f, "transcript: {e}"),
        }
    }
}

impl From<lattice_core::transcript::TranscriptError> for BatchError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        BatchError::Transcript(e)
    }
}

/// Run `Π_batchM`: the hbPCE claims `(γᵢ)` at `(α, β)` over the matrix
/// commitments `[Mᵢ]` fold into one `R_GBF,α,β` statement.
pub fn batch_m(
    matrix_commitments: &[crate::pc::PcCommitment],
    gammas: &[Fp256],
    alpha: &[Fp256],
    beta: &[Fp256],
    transcript: &mut Transcript,
) -> Result<(GbfInstance, Vec<Fp256>), BatchError> {
    if gammas.len() != matrix_commitments.len() {
        return Err(BatchError::Shape("gammas vs matrix commitments"));
    }
    // Absorb the commitments + claims; sample c.
    for c in matrix_commitments {
        transcript.append_message(b"bm-com", &c.to_bytes())?;
    }
    for g in gammas {
        lattice_pcd::util::absorb_fp(transcript, b"bm-gamma", g)?;
    }
    let c = lattice_pcd::util::challenge_fp_vec(transcript, b"bm-c", gammas.len())?;
    let inst = build_gbf_alpha_beta(gammas, &c, matrix_commitments, alpha, beta);
    Ok((inst, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_m_builds_consistent_statement() {
        let domain = crate::poly::Domain::Multivariate { num_vars: 2 };
        let n = 4;
        let matrices = vec![
            crate::poly::fp_matrix(b"bmm", b"a", n),
            crate::poly::fp_matrix(b"bmm", b"b", n),
        ];
        let alpha = vec![Fp256::from_canonical_u64(3), Fp256::from_canonical_u64(5)];
        let beta = vec![Fp256::from_canonical_u64(7), Fp256::from_canonical_u64(11)];
        let gammas: Vec<Fp256> = matrices
            .iter()
            .map(|m| {
                crate::poly::matrix_poly_eval(&domain, m, &beta, &alpha)
                    .ok()
                    .unwrap()
            })
            .collect();
        // Commit the matrices (the holographic index).
        let key = crate::pc::PcKey::new(domain.clone(), &[64u8; 32])
            .ok()
            .unwrap();
        let coms: Vec<crate::pc::PcCommitment> = matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        let mut t = Transcript::new_default(b"bm");
        let (inst, _c) = batch_m(&coms, &gammas, &alpha, &beta, &mut t).ok().unwrap();
        // The statement is satisfied with the implicit λ vectors.
        let wit = crate::relations::GbfWitness {
            us: Vec::new(),
            vs: Vec::new(),
        };
        assert!(inst.check_with(&domain, &wit, &matrices).ok().unwrap());
    }
}
