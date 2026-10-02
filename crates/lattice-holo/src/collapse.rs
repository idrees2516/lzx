//! `Π_Collapse` — Lemma 2's reduction `R_CCS → R_GBF,α` (ePrint
//! 2026/538): the preprocessing step + first round of virtually every CCS
//! argument. The prover commits the witness part `w(X)`; the verifier
//! samples `α`; both define the `R_GBF,α` statement
//! `λ(α)ᵀ (Σᵢ cᵢ ∘_{j∈Sᵢ} Mⱼ z) = 0` with `z = (x, w)` — the right side's
//! single v-slot carrying `z`, whose commitment covers `w` (the public
//! part `x` is folded into the evaluation claims — Remark 4's split).

use crate::pc::{PcCommitment, PcKey, PcWitness};
use crate::relations::{Ccs, GbfInstance, GbfLeft, GbfRight, RelError};
use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollapseError {
    Rel(RelError),
    Pc(crate::pc::PcError),
    Transcript(lattice_core::transcript::TranscriptError),
    Shape(&'static str),
}

impl core::fmt::Display for CollapseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CollapseError::Rel(e) => write!(f, "relation: {e}"),
            CollapseError::Pc(e) => write!(f, "pc: {e}"),
            CollapseError::Transcript(e) => write!(f, "transcript: {e}"),
            CollapseError::Shape(s) => write!(f, "collapse shape: {s}"),
        }
    }
}

impl From<RelError> for CollapseError {
    fn from(e: RelError) -> Self {
        CollapseError::Rel(e)
    }
}

impl From<lattice_core::transcript::TranscriptError> for CollapseError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        CollapseError::Transcript(e)
    }
}

impl From<crate::pc::PcError> for CollapseError {
    fn from(e: crate::pc::PcError) -> Self {
        CollapseError::Pc(e)
    }
}

/// The collapsed statement: the `R_GBF,α` instance + the witness context.
pub struct Collapsed {
    pub gbf: GbfInstance,
    /// The witness encoding of `z = (x, w)` (the prover-side vector).
    pub z_encoding: Vec<Fp256>,
    /// The commitment to `w` (the v-slot's commitment; the claim value
    /// adds the public part `x(α)`).
    pub w_commitment: PcCommitment,
    pub w_witness: PcWitness,
    /// The public `x` (for the caller's claim adjustment).
    pub x: Vec<Fp256>,
    pub alpha: Vec<Fp256>,
}

/// Run `Π_Collapse`: commit `w`, sample `α`, build the `R_GBF,α` instance
/// with `s = 0` (the CCS statement collapses to the zero check).
pub fn collapse(
    ccs: &Ccs,
    key: &PcKey,
    x: &[Fp256],
    w: &[Fp256],
    transcript: &mut Transcript,
) -> Result<Collapsed, CollapseError> {
    if x.len() != ccs.s || w.len() != ccs.n - ccs.s {
        return Err(CollapseError::Shape("ccs input lengths"));
    }
    let mut z = x.to_vec();
    z.extend(w.iter().cloned());
    // Transcript order (verifier-replayable): [publics → α → the w-blind
    // (drawn; the verifier discards) → the commitment] — no prover-side
    // absorptions precede the public challenges.
    lattice_pcd::util::absorb_fp_slice(transcript, b"col-x", x)?;
    lattice_pcd::util::absorb_fp_slice(transcript, b"col-ccs-c", &ccs.constants)?;
    let alpha = lattice_pcd::util::challenge_fp_vec(transcript, b"col-alpha", key.domain.nu())?;
    // Commit w(X) as the padded domain encoding (0..0, w) — Remark 4's
    // [w(X)] = [wᵀ(λ_{s+1},...,λ_n)]: the public part enters the claims.
    let mut w_encoding = vec![Fp256::ZERO; ccs.s];
    w_encoding.extend(w.iter().cloned());
    let blind = lattice_pcd::util::challenge_fp(transcript, b"col-w-blind")?;
    let wc = key
        .key
        .commit(&w_encoding, &blind)
        .map_err(|e| CollapseError::Pc(crate::pc::PcError::Pedersen(e)))?;
    transcript.append_message(b"col-wc", &wc.to_bytes())?;
    let wwit = PcWitness {
        encoding: w_encoding.clone(),
        blind,
    };
    // The GBF,α instance: left = λ(α) (implicit — one term, constant 1,
    // set {0} over the single implicit u slot); right = Σ_i c_i ∘_{j∈S_i}
    // (M_j z): one right term per CCS equation with the Hadamard set
    // {(j, 0) : j ∈ S_i}, v-slot 0 = z (its commitment = the w commitment
    // with the public-part convention).
    let gbf = GbfInstance {
        left: GbfLeft {
            constants: vec![Fp256::from_canonical_u64(1)],
            sets: vec![vec![0]],
        },
        right: GbfRight {
            constants: ccs.constants.clone(),
            sets: ccs.sets.iter().map(|s| s.iter().map(|&j| (j, 0)).collect()).collect(),
        },
        u_commitments: Vec::new(),
        v_commitments: vec![wc],
        matrix_commitments: Vec::new(),
        s: Fp256::ZERO,
        alpha: Some(alpha.clone()),
        beta: None,
    };
    Ok(Collapsed {
        gbf,
        z_encoding: z,
        w_commitment: wc,
        w_witness: PcWitness {
            encoding: w_encoding.clone(),
            blind: wwit.blind,
        },
        x: x.to_vec(),
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_statement_satisfied() {
        let domain = crate::poly::Domain::Multivariate { num_vars: 3 };
        let (ccs, x, w) = Ccs::random_with_solution(&domain, 2, 6, 3, 3, 3, b"col-seed");
        let key = PcKey::new(domain.clone(), &[63u8; 32]).ok().unwrap();
        let mut t = Transcript::new_default(b"col");
        let col = collapse(&ccs, &key, &x, &w, &mut t).ok().unwrap();
        // The GBF,α statement is satisfied with the z witness (the implicit
        // λ(α) + the z vector as v-slot 0).
        let wit = crate::relations::GbfWitness {
            us: Vec::new(),
            vs: vec![col.z_encoding.clone()],
        };
        assert!(col.gbf.check_with(&domain, &wit, &ccs.matrices).ok().unwrap());
    }
}
