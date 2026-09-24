//! Port of **osdnk/labinius** (`crates/pcs`) to the LZX pure-std stack: a lattice-based
//! polynomial commitment scheme for binary witnesses over `R_648 = Z_q[X]/Phi_1944(X)` with
//! the evaluation claim in `F162 = GF(2)[x]/(x^162 + x^81 + 1)`, and the opening sent in the
//! clear, bit-dropped, or replaced by a LaBRADOR proof ([`lattice-labrador`], the native port
//! of the vendored C backend).
//!
//! What is ported, and how (full map in `README-port.md`):
//! * `params` — both NTT trees, const-evaluated, verbatim;
//! * `scalar` — the exact reference NTT family (this port's *only* implementation: upstream's
//!   AVX-512 kernels are replaced by their defining reference paths);
//! * `fields`/`f162` — `F162`/`B128` with software carry-less multiplication (PCLMULQDQ
//!   replaced), the lift of an F162 stream into binary ring elements;
//! * `ring` — `R_162` slot elements, the 648 -> 4x162 decomposition (split + quadratic);
//! * `key` — the Ajtai commitment key and the per-chunk pointwise commitment;
//! * `challenge` — the Fiat-Shamir transcript (SHAKE-256 in place of blake3, same
//!   absorb/counter/label discipline) and weight-28 canonical-bounded challenges over `R_162`;
//! * `fold` — `v = sum_j c_j W_j` in the slot domain, the commitment fold, `A v`;
//! * `eval` — the binary shadow (eq tables, row evaluations, the claim and binary checks);
//! * `bd` — the bit-dropped opening (Garner digits, the residual norm check);
//! * `scheme` — `Params`/`Prover`/`Verifier`/`PublicParameters`, Clear + BitDropped modes;
//! * `recursion` — the recursive opening as a LaBRADOR statement (direct slot-domain encoding
//!   of the per-limb Ajtai identities; see the module docs for the documented simplifications
//!   vs upstream's chunked-chain encoding).
//!
//! Not ported: the AVX-512 SIMD kernels themselves, `wire`'s rANS entropy coder (bit-precise
//! size floors are reported by `Commitment::wire_bytes`), the Binius/Flock front ends
//! (`crates/binius`, `crates/flock`) and the competitor harness (`crates/competitors`), which
//! are external-dependency integration layers rather than protocol.

// Upstream kernel structure: loops index with strides and table positions
// (`batches[c * nr + i]`, `lut[3 * k + r]`), which the range-loop lint's iterator
// suggestions cannot express. The patterns are verbatim from the ported reference.
#![allow(clippy::needless_range_loop)]
pub mod bd;
pub mod binfield;
pub mod challenge;
pub mod eval;
pub mod fold;
pub mod hw;
pub mod key;
pub mod params;
pub mod recursion;
pub mod ring;
pub mod scalar;
pub mod scheme;
pub mod simd;

pub use binfield::{B128, F162};
pub use challenge::{Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
pub use ring::{Modulus, PowerOfThreeRing as RingElement162};
pub use scheme::{
    basic, Commitment, CommitmentOpening, CommitmentValue, EvaluationPoint, FoldedCommitment,
    FoldedWitness, Opening, ParamError, Params, Prover, PublicParameters, RowEvaluation, Suite,
    SUITES, Verifier, VerificationError, Witness,
};

/// The upstream reference round, as an executable example and a smoke test:
/// commit -> derive the point -> evaluate -> fold -> verify, in the Clear mode.
pub fn reference_round(witness_log_len: u32, column_log_len: u32) -> Result<(), String> {
    let params = Params::new(
        witness_log_len,
        column_log_len,
        vec![Modulus::Q2917_Q_S],
        Opening::Clear,
    )
    .map_err(|e| e.to_string())?;
    let matrix_seed = [7u8; 32];
    let pp = PublicParameters::from_seed(params.clone(), matrix_seed);
    let witness = Witness::random(&params, [42u8; 32]);
    let (prover, verifier) = (Prover::new(&pp), Verifier::new(&pp));
    let (commitment, opening) = prover.commit(&witness);
    let mut t = Transcript::new(b"labinius/reference");
    let point = verifier.derive_evaluation_point(&mut t, &commitment);
    let claimed = witness.mle_evaluate(&point);
    let row = witness.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut t, &row);
    let folded = prover.fold(opening, &challenges);
    verifier.verify_evaluation(&point, &claimed, &row).map_err(|e| e.to_string())?;
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row = verifier.fold_row_evaluation(&row, &challenges);
    verifier
        .verify_opening(&folded_commitment, &folded, &point, &folded_row)
        .map_err(|e| e.to_string())?;
    Ok(())
}
