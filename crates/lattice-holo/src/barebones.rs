//! **Barebones** — Theorem 7's composed argument `R_CCS → R_PCEP ×
//! R_GBF,α,β` (ePrint 2026/538 §5.1):
//!
//! ```text
//! Barebones := (Π_provePCE × ID) ∘ (ID × Π_batchM) ∘ Π_GBF,α ∘ Π_Collapse
//! ```
//!
//! Instantiated with `Π_GBF2` at `ν = log n` this recovers **SuperSpartan**
//! [STW23]; at `ν = 1` with `Π_GBF1` it recovers **SuperMarlin** (§5.1's
//! remark). The output is the accumulator relation `R_Acc`'s two halves:
//! one batched evaluation-proof statement (`R_PCEP`) and one holographic
//! statement (`R_GBF,α,β`).

use crate::batch::batch_m;
use crate::collapse::collapse;
use crate::gbf2::{gbf2_prove, gbf2_verify, Gbf2Proof, Gbf2Statement};
use crate::pc::{
    batch_claimed_sum, batch_prove_pce, batch_verify_pcep, PcCommitment, PcError, PcKey,
    PceBatchProof, PceClaim,
};
use crate::relations::{Ccs, GbfInstance, GbfWitness, RelError};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarebonesError {
    Rel(RelError),
    Pc(PcError),
    Gbf2(crate::gbf2::GbfError),
    Collapse(crate::collapse::CollapseError),
    Batch(crate::batch::BatchError),
    Shape(&'static str),
}

impl core::fmt::Display for BarebonesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BarebonesError::Rel(e) => write!(f, "relation: {e}"),
            BarebonesError::Pc(e) => write!(f, "pc: {e}"),
            BarebonesError::Gbf2(e) => write!(f, "gbf2: {e}"),
            BarebonesError::Collapse(e) => write!(f, "collapse: {e}"),
            BarebonesError::Batch(e) => write!(f, "batch: {e}"),
            BarebonesError::Shape(s) => write!(f, "barebones shape: {s}"),
        }
    }
}

impl From<lattice_core::transcript::TranscriptError> for BarebonesError {
    fn from(_e: lattice_core::transcript::TranscriptError) -> Self {
        BarebonesError::Shape("transcript")
    }
}

impl From<crate::poly::PolyError> for BarebonesError {
    fn from(e: crate::poly::PolyError) -> Self {
        BarebonesError::Rel(RelError::Poly(e))
    }
}

impl From<RelError> for BarebonesError {
    fn from(e: RelError) -> Self {
        BarebonesError::Rel(e)
    }
}

impl From<PcError> for BarebonesError {
    fn from(e: PcError) -> Self {
        BarebonesError::Pc(e)
    }
}

impl From<crate::gbf2::GbfError> for BarebonesError {
    fn from(e: crate::gbf2::GbfError) -> Self {
        BarebonesError::Gbf2(e)
    }
}

impl From<crate::collapse::CollapseError> for BarebonesError {
    fn from(e: crate::collapse::CollapseError) -> Self {
        BarebonesError::Collapse(e)
    }
}

impl From<crate::batch::BatchError> for BarebonesError {
    fn from(e: crate::batch::BatchError) -> Self {
        BarebonesError::Batch(e)
    }
}

/// The full Barebones proof (the non-interactive argument for one CCS
/// instance).
#[derive(Clone, Debug)]
pub struct BarebonesProof {
    /// The collapsed `R_GBF,α` statement (with the w commitment).
    pub gbf_alpha: GbfInstance,
    /// The z encoding (prover-side; never transmitted in the succinct
    /// regime — here part of the transparent artifact).
    pub z_encoding: Vec<Fp256>,
    pub w_commitment: PcCommitment,
    pub x: Vec<Fp256>,
    /// The GBF2 proof over the collapsed statement.
    pub gbf2: Gbf2Proof,
    /// The batched evaluation proof for the PCE claims (R_PCEP) — `None`
    /// when the GBF run produced no PCE claims.
    pub pcep: Option<PceBatchProof>,
    /// The claimed sum of the PCE batch.
    pub pcep_claimed_sum: Fp256,
    /// The PCE claims the batch proves (for the fold's eager checks).
    pub pcep_claims: Vec<crate::pc::PceClaim>,
    /// The resulting holographic statement (R_GBF,α,β).
    pub gbf_alpha_beta: GbfInstance,
}

/// `R_Acc` — the accumulator relation's instance (the argument's deferred
/// statement): `(R_PCEP, R_GBF,α,β)`.
#[derive(Clone, Debug, PartialEq)]
pub struct AccStatement {
    /// The holographic half.
    pub gbf_alpha_beta: GbfInstance,
    /// The PCEP half (transparent long-opening regime: the batched proof
    /// is settled eagerly — see the honest-deviation ledger).
    pub pcep: Option<PceBatchProof>,
    pub pcep_claims: Vec<PceClaim>,
    pub pcep_point: Vec<Fp256>,
    pub pcep_claimed_sum: Fp256,
}

/// Prove Barebones for `(ccs, x, w)`; returns the proof + the `R_Acc`
/// statement.
pub fn barebones_prove(
    ccs: &Ccs,
    key: &PcKey,
    matrix_commitments: &[PcCommitment],
    x: &[Fp256],
    w: &[Fp256],
    transcript: &mut Transcript,
) -> Result<(BarebonesProof, AccStatement), BarebonesError> {
    // 1. Π_Collapse.
    let col = collapse(ccs, key, x, w, transcript)?;
    // 2. Π_GBF2,α over the collapsed statement (the implicit λ(α) left
    //    side; the z vector as v-slot 0).
    let wits = vec![GbfWitness {
        us: Vec::new(),
        vs: vec![col.z_encoding.clone()],
    }];
    let st = Gbf2Statement {
        domain: key.domain.clone(),
        instances: std::slice::from_ref(&col.gbf),
        witnesses: &wits,
        matrices: &ccs.matrices,
    };
    let (gbf2_proof, gbf2_out) = gbf2_prove(&st, transcript)?;
    // 3. Π_batchM over the holographic claims at (α, β).
    let (gbf_ab, _c) = batch_m(
        matrix_commitments,
        &gbf2_out.hb_gammas,
        &gbf2_out.alpha,
        &gbf2_out.beta,
        transcript,
    )?;
    // 4. Π_provePCE: settle the PCE claims (the v-slot claim at β; the
    //    claimed value is z(β) = x(β) + w(β) — the public part adjusted
    //    into the claim).
    let mut pce_claims = Vec::new();
    let mut witnesses = Vec::new();
    for (com, point, val) in &gbf2_out.pce_claims {
        // The v-slot claim: the value covers z; the commitment covers the
        // padded w encoding — adjust the claim to the w-only eval.
        let mut adjusted = *val;
        if *com == col.w_commitment {
            // z(β) − x(β) = w(β).
            let x_at = crate::poly::vec_poly_eval(&key.domain, &pad_x(x, key.domain.size()), point)
                .map_err(crate::gbf2::GbfError::from)?;
            adjusted = val.sub(&x_at);
        }
        pce_claims.push(PceClaim {
            poly_id: 0,
            commitment: *com,
            claimed: adjusted,
        });
        witnesses.push((col.w_witness.encoding.clone(), col.w_witness.blind));
    }
    let pcep_point = gbf2_out.beta.clone();
    let claims = pce_claims;
    let pcep_claims = claims.clone();
    let (pcep, pcep_claimed_sum) = if claims.is_empty() {
        (None, Fp256::ZERO)
    } else {
        let proof = batch_prove_pce(key, &pcep_point, &claims, &witnesses, transcript)?;
        let sum = batch_claimed_sum(&proof.etas, &claims);
        (Some(proof), sum)
    };

    let proof = BarebonesProof {
        gbf_alpha: col.gbf.clone(),
        z_encoding: col.z_encoding,
        w_commitment: col.w_commitment,
        x: x.to_vec(),
        gbf2: gbf2_proof,
        pcep: pcep.clone(),
        pcep_claimed_sum,
        pcep_claims: claims,
        gbf_alpha_beta: gbf_ab.clone(),
    };
    let acc = AccStatement {
        gbf_alpha_beta: gbf_ab,
        pcep,
        pcep_claims,
        pcep_point,
        pcep_claimed_sum,
    };
    Ok((proof, acc))
}

/// The public input padded to the domain size (positions 0..s).
fn pad_x(x: &[Fp256], n: usize) -> Vec<Fp256> {
    let mut out = vec![Fp256::ZERO; n];
    for (i, v) in x.iter().enumerate() {
        if i < n {
            out[i] = *v;
        }
    }
    out
}

/// Verify a Barebones proof: re-run the collapse statement check, GBF2,
/// batchM, and the PCEP settlement.
pub fn barebones_verify(
    ccs: &Ccs,
    key: &PcKey,
    matrix_commitments: &[PcCommitment],
    x: &[Fp256],
    proof: &BarebonesProof,
    transcript: &mut Transcript,
) -> Result<bool, BarebonesError> {
    // 1. Collapse replay: the public absorptions → α → the discarded
    // w-blind → the w commitment.
    lattice_pcd::util::absorb_fp_slice(transcript, b"col-x", x)?;
    lattice_pcd::util::absorb_fp_slice(transcript, b"col-ccs-c", &ccs.constants)?;
    let alpha = lattice_pcd::util::challenge_fp_vec(transcript, b"col-alpha", key.domain.nu())?;
    let _ = lattice_pcd::util::challenge_fp(transcript, b"col-w-blind")?;
    transcript.append_message(b"col-wc", &proof.w_commitment.to_bytes())?;
    if proof.gbf_alpha.alpha.as_deref() != Some(&alpha) {
        return Ok(false);
    }
    if proof.w_commitment != proof.gbf_alpha.v_commitments[0] {
        return Ok(false);
    }
    // The collapsed statement's right side must match the CCS structure.
    if proof.gbf_alpha.right.constants != ccs.constants {
        return Ok(false);
    }
    // 2. GBF2 verification.
    let vout = gbf2_verify(
        &key.domain,
        std::slice::from_ref(&proof.gbf_alpha),
        &proof.gbf2,
        transcript,
    )?;
    // 3. batchM replay.
    let (_gbf_ab, _c) = batch_m(
        matrix_commitments,
        &vout.hb_gammas,
        &vout.alpha,
        &vout.beta,
        transcript,
    )?;
    // 4. PCEP settlement: the v-slot claim (adjusted for the public part)
    //    must open.
    if let Some(pcep) = &proof.pcep {
        // Rebuild the adjusted claims as in the prover.
        let mut claims = Vec::new();
        for (com, point, val) in &vout.pce_claims {
            let mut adjusted = *val;
            if *com == proof.w_commitment {
                let x_at =
                    crate::poly::vec_poly_eval(&key.domain, &pad_x(x, key.domain.size()), point)?;
                adjusted = val.sub(&x_at);
            }
            claims.push(PceClaim {
                poly_id: 0,
                commitment: *com,
                claimed: adjusted,
            });
        }
        let ok = batch_verify_pcep(
            key,
            &proof.pcep_point(),
            &claims,
            &proof.pcep_claimed_sum,
            pcep,
            transcript,
        )?;
        if !ok {
            return Ok(false);
        }
    }
    Ok(true)
}

impl BarebonesProof {
    pub fn pcep_point(&self) -> Vec<Fp256> {
        // The β point of the GBF2 run (the PCE claims' point).
        self.gbf2.beta.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barebones_roundtrip_multivariate() {
        let domain = crate::poly::Domain::Multivariate { num_vars: 3 };
        let (ccs, x, w) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"bb-seed");
        let key = PcKey::new(domain.clone(), &[65u8; 32]).ok().unwrap();
        let matrix_comms: Vec<PcCommitment> = ccs
            .matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        let mut t = Transcript::new_default(b"bb");
        let (proof, acc) = match barebones_prove(&ccs, &key, &matrix_comms, &x, &w, &mut t) {
            Ok(v) => v,
            Err(e) => panic!("barebones prove failed: {e}"),
        };
        // Verify.
        let mut tv = Transcript::new_default(b"bb");
        let ok = barebones_verify(&ccs, &key, &matrix_comms, &x, &proof, &mut tv)
            .ok()
            .unwrap();
        assert!(ok);
        // The accumulator statement is a satisfiable GBF,α,β (implicit
        // vectors) — the holographic half.
        let wit = GbfWitness {
            us: Vec::new(),
            vs: Vec::new(),
        };
        assert!(acc
            .gbf_alpha_beta
            .check_with(&domain, &wit, &ccs.matrices)
            .ok()
            .unwrap());
        // Tampered proof: corrupt the m_evals → GBF2 rejects.
        let mut bad = proof.clone();
        bad.gbf2.m_evals[0] = bad.gbf2.m_evals[0].add(&Fp256::from_canonical_u64(1));
        let mut tv2 = Transcript::new_default(b"bb");
        let ok2 = barebones_verify(&ccs, &key, &matrix_comms, &x, &bad, &mut tv2).unwrap_or(false);
        assert!(!ok2);
    }

    #[test]
    fn barebones_roundtrip_univariate() {
        let domain = crate::poly::Domain::Univariate { n: 8 };
        let (ccs, x, w) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 2, b"bbu-seed");
        let key = PcKey::new(domain.clone(), &[66u8; 32]).ok().unwrap();
        let matrix_comms: Vec<PcCommitment> = ccs
            .matrices
            .iter()
            .map(|m| key.commit_matrix(&domain, m).ok().unwrap())
            .collect();
        let mut t = Transcript::new_default(b"bbu");
        let (proof, _acc) = match barebones_prove(&ccs, &key, &matrix_comms, &x, &w, &mut t) {
            Ok(v) => v,
            Err(e) => panic!("barebones(u) prove failed: {e}"),
        };
        let mut tv = Transcript::new_default(b"bbu");
        let ok = barebones_verify(&ccs, &key, &matrix_comms, &x, &proof, &mut tv)
            .ok()
            .unwrap();
        assert!(ok);
    }
}
