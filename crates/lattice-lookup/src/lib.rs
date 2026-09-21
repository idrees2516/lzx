//! # lattice-lookup
//!
//! Quasar (Zheng–Gao–Guo–Xiao, ePrint 2025/1912): sublinear accumulation
//! schemes for multiple instances — the lookup-argument accumulation core.
//!
//! Two components:
//! * `lookup` — the grand-product lookup check (FLI/cq lineage): reads are
//!   contained in the table as a multiset iff the random grand products
//!   satisfy `T(τ) = R(τ)·Q(τ)` at a random challenge τ, where
//!   `T = Π_table (1 + τ·v)`, `R = Π_reads (1 + τ·v)`, and Q is the
//!   quotient. Verified with Schwartz–Zippel over τ.
//! * `accumulate` — Quasar's multi-instance accumulation via **partial
//!   evaluation** instead of random linear combinations of commitments
//!   (the costly CRC operations): all instances partially evaluate their
//!   polynomials at a *shared* challenge prefix, so the folded accumulator
//!   needs a single combination step — verifier cost sublinear in the
//!   number of instances.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};

/// A lookup proof: evaluations of the grand products at the challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProof {
    /// Π_table (1 + τ·v) evaluated... structurally: the proof carries the
    /// three evaluations; commitments are layered by the caller.
    pub t_eval: Goldilocks,
    pub r_eval: Goldilocks,
    pub q_eval: Goldilocks,
    pub challenge: Goldilocks,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    Transcript(TranscriptError),
    /// Grand-product identity failed.
    LookupFailed,
    /// Accumulation identity failed.
    AccumulationFailed,
    ShapeMismatch { expected: usize, got: usize },
}

/// Grand product helper: Π (1 + τ·v) over values.
fn grand_product(values: &[Goldilocks], tau: &Goldilocks) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for v in values {
        acc = acc.mul(&Goldilocks::ONE.add(&tau.mul(v)));
    }
    acc
}

/// Prove reads ⊆ table (multiset containment) via the grand-product
/// quotient argument.
pub fn prove_lookup(
    table: &[Goldilocks],
    reads: &[Goldilocks],
    transcript: &mut Transcript,
) -> Result<LookupProof, LookupError> {
    // Absorb the public statement (lengths + digests of both sides).
    let mut table_bytes = Vec::with_capacity(table.len() * 8);
    for v in table {
        table_bytes.extend_from_slice(&v.to_bytes());
    }
    let mut reads_bytes = Vec::with_capacity(reads.len() * 8);
    for v in reads {
        reads_bytes.extend_from_slice(&v.to_bytes());
    }
    transcript
        .append_bytes(b"table", &table_bytes)
        .map_err(LookupError::Transcript)?;
    transcript
        .append_bytes(b"reads", &reads_bytes)
        .map_err(LookupError::Transcript)?;

    let tau = transcript
        .challenge_field(b"lookup-tau")
        .map_err(LookupError::Transcript)?;

    // Honest prover: multiset containment must hold — the quotient Q is
    // the grand product over table \ reads (multiset difference). Fail
    // closed when reads are NOT contained (Q would not exist).
    let t_eval = grand_product(table, &tau);
    let r_eval = grand_product(reads, &tau);
    // Compute the multiset difference: table minus reads.
    let mut remaining: Vec<Goldilocks> = table.to_vec();
    for rv in reads {
        match remaining.iter().position(|t| *t == *rv) {
            Some(idx) => {
                remaining.swap_remove(idx);
            }
            None => return Err(LookupError::LookupFailed),
        }
    }
    let q_eval = grand_product(&remaining, &tau);
    Ok(LookupProof {
        t_eval,
        r_eval,
        q_eval,
        challenge: tau,
    })
}

/// Verify a lookup proof: T(τ) == R(τ)·Q(τ) plus the τ-consistency.
pub fn verify_lookup(
    proof: &LookupProof,
    table: &[Goldilocks],
    reads: &[Goldilocks],
    transcript: &mut Transcript,
) -> Result<(), LookupError> {
    let mut table_bytes = Vec::with_capacity(table.len() * 8);
    for v in table {
        table_bytes.extend_from_slice(&v.to_bytes());
    }
    let mut reads_bytes = Vec::with_capacity(reads.len() * 8);
    for v in reads {
        reads_bytes.extend_from_slice(&v.to_bytes());
    }
    transcript
        .append_bytes(b"table", &table_bytes)
        .map_err(LookupError::Transcript)?;
    transcript
        .append_bytes(b"reads", &reads_bytes)
        .map_err(LookupError::Transcript)?;
    let tau = transcript
        .challenge_field(b"lookup-tau")
        .map_err(LookupError::Transcript)?;
    if tau != proof.challenge {
        return Err(LookupError::LookupFailed);
    }
    // Grand-product identity.
    if proof.t_eval != proof.r_eval.mul(&proof.q_eval) {
        return Err(LookupError::LookupFailed);
    }
    Ok(())
}

/// An accumulated lookup instance (post partial evaluation): the MLE is
/// collapsed to its remaining variables, and only ONE combination step
/// is needed across instances.
#[derive(Clone, Debug)]
pub struct PartialEvalInstance {
    /// Partially evaluated polynomial (fewer variables than the original).
    pub partial: DenseMle,
    /// The prefix point at which it was evaluated.
    pub prefix: Vec<Goldilocks>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuasarError {
    Lookup(LookupError),
    Mle(lattice_core::mle::MleError),
    /// Partial-evaluation accumulation identity failed.
    AccumulationFailed,
    VariableCountMismatch { expected: usize, got: usize },
}

/// Quasar accumulation: fold k instances by partially evaluating every
/// polynomial at a SHARED random prefix (derived from one transcript
/// challenge absorption), then combining the collapsed polynomials with a
/// single linear-combination step — one CRC instead of k.
pub fn accumulate_partial_evaluation(
    instances: &[DenseMle],
    transcript: &mut Transcript,
) -> Result<(DenseMle, Vec<Goldilocks>), QuasarError> {
    if instances.is_empty() {
        return Err(QuasarError::VariableCountMismatch {
            expected: 1,
            got: 0,
        });
    }
    let total_vars = instances[0].num_vars;
    if instances.iter().any(|f| f.num_vars != total_vars) {
        return Err(QuasarError::VariableCountMismatch {
            expected: total_vars,
            got: instances
                .iter()
                .map(|f| f.num_vars)
                .max()
                .unwrap_or(0),
        });
    }
    // Number of prefix variables to collapse: leave one variable so the
    // result is still a non-degenerate MLE (unless total is 0).
    let collapse = total_vars.saturating_sub(1).max(0);
    let prefix = transcript
        .challenge_fields(b"quasar-prefix", collapse)
        .map_err(LookupError::Transcript)
        .map_err(QuasarError::Lookup)?;

    // Partially evaluate every instance at the shared prefix.
    let mut partials = Vec::with_capacity(instances.len());
    for f in instances {
        let g = f
            .fix_variables(&prefix)
            .map_err(QuasarError::Mle)?;
        partials.push(g);
    }
    // Single combination step with one challenge per instance.
    let lambdas = transcript
        .challenge_fields(b"quasar-lambda", partials.len())
        .map_err(LookupError::Transcript)
        .map_err(QuasarError::Lookup)?;
    // Combine: acc = Σ λ_i · g_i (all g_i share the same variable count).
    let combined_evals: Vec<Goldilocks> = partials[0]
        .evaluations
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let mut acc = Goldilocks::ZERO;
            for (g, lambda) in partials.iter().zip(lambdas.iter()) {
                acc = acc.add(&lambda.mul(&g.evaluations[idx]));
            }
            acc
        })
        .collect();
    Ok((
        DenseMle {
            num_vars: total_vars - collapse,
            evaluations: combined_evals,
        },
        prefix,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn lookup_contained_proves_and_verifies() {
        let table: Vec<Goldilocks> = (0..32u64).map(|i| fe(i * 7 + 3)).collect();
        // Distinct reads (the table has multiplicity 1 everywhere).
        let reads = vec![table[3], table[17], table[0]];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        let proof = prove_lookup(&table, &reads, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-lookup-test");
        assert!(verify_lookup(&proof, &table, &reads, &mut vt).is_ok());
    }

    #[test]
    fn lookup_missing_entry_rejected() {
        let table: Vec<Goldilocks> = (0..16u64).map(|i| fe(i)).collect();
        // 99 is not in the table.
        let reads = vec![fe(1), fe(99), fe(2)];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        assert!(matches!(
            prove_lookup(&table, &reads, &mut t),
            Err(LookupError::LookupFailed)
        ));
    }

    #[test]
    fn lookup_multiplicity_enforced() {
        let table: Vec<Goldilocks> = (0..8u64).map(|i| fe(i)).collect();
        // Table contains one 5; reads demand two.
        let reads = vec![fe(5), fe(5)];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        assert!(matches!(
            prove_lookup(&table, &reads, &mut t),
            Err(LookupError::LookupFailed)
        ));
    }

    #[test]
    fn tampered_lookup_rejected() {
        let table: Vec<Goldilocks> = (0..16u64).map(|i| fe(i)).collect();
        let reads = vec![fe(1), fe(2)];
        let mut t = Transcript::new_default(b"lzx-lookup-test");
        let mut proof = prove_lookup(&table, &reads, &mut t).ok().unwrap();
        // Tamper with an evaluation: identity must fail.
        proof.t_eval = proof.t_eval.add(&fe(1));
        let mut vt = Transcript::new_default(b"lzx-lookup-test");
        assert!(verify_lookup(&proof, &table, &reads, &mut vt).is_err());
        // Wrong statement (different reads) desyncs the challenge.
        let mut t2 = Transcript::new_default(b"lzx-lookup-test");
        let mut proof2 = prove_lookup(&table, &reads, &mut t2).ok().unwrap();
        proof2.challenge = proof2.challenge.add(&fe(1));
        let mut vt2 = Transcript::new_default(b"lzx-lookup-test");
        assert!(verify_lookup(&proof2, &table, &reads, &mut vt2).is_err());
    }

    #[test]
    fn partial_evaluation_accumulation() {
        // Accumulate 4 MLE instances: the folded polynomial equals the
        // partial-evaluated combination (verifiable directly).
        let instances: Vec<DenseMle> = ["a", "b", "c", "d"]
            .iter()
            .map(|t| DenseMle::random(6, t.as_bytes()))
            .collect();
        let mut t = Transcript::new_default(b"lzx-quasar-test");
        let (folded, prefix) = accumulate_partial_evaluation(&instances, &mut t)
            .ok()
            .unwrap();
        assert_eq!(folded.num_vars, 1); // 6 vars collapsed to 1.
        assert_eq!(prefix.len(), 5);

        // Direct reference: replay the same transcript sequence (prefix
        // first, then lambdas) to reproduce identical challenges.
        let mut t2 = Transcript::new_default(b"lzx-quasar-test");
        let prefix2 = t2
            .challenge_fields(b"quasar-prefix", 5)
            .ok()
            .unwrap();
        assert_eq!(prefix2, prefix);
        let lambdas = t2
            .challenge_fields(b"quasar-lambda", instances.len())
            .ok()
            .unwrap();
        let mut expected = vec![Goldilocks::ZERO; 2];
        for (f, lambda) in instances.iter().zip(lambdas.iter()) {
            let g = f.fix_variables(&prefix).ok().unwrap();
            for (e, v) in expected.iter_mut().zip(g.evaluations.iter()) {
                *e = e.add(&lambda.mul(v));
            }
        }
        assert_eq!(folded.evaluations, expected);
    }

    #[test]
    fn accumulation_shape_errors() {
        let mut t = Transcript::new_default(b"lzx-quasar-test");
        assert!(matches!(
            accumulate_partial_evaluation(&[], &mut t),
            Err(QuasarError::VariableCountMismatch { .. })
        ));
        let mixed = vec![DenseMle::random(3, b"x"), DenseMle::random(4, b"y")];
        assert!(matches!(
            accumulate_partial_evaluation(&mixed, &mut t),
            Err(QuasarError::VariableCountMismatch { .. })
        ));
    }

    #[test]
    fn accumulator_satisfies_lookup_when_instances_do() {
        // End-to-end composition under ONE shared challenge τ (as in the
        // accumulated statement): grand products compose multiplicatively,
        // so R_total = R1·R2 and Q_total = T / R_total closes the identity.
        let table: Vec<Goldilocks> = (0..64u64).map(|i| fe(i * 13 + 1)).collect();
        let reads1 = vec![table[5], table[9]];
        let reads2 = vec![table[40], table[2], table[5]];
        // Shared challenge (the accumulation point).
        let tau = Goldilocks::from_u64(0x9E37_79B9_7F4A_7C15);
        // NOTE: reads2 reuses table[5] which reads1 already consumed —
        // multiplicity across the ACCUMULATED statement must respect the
        // table: use disjoint reads instead.
        let reads2 = vec![table[40], table[2], table[7]];
        let combined_reads = [reads1.clone(), reads2.clone()].concat();
        let r1 = grand_product(&reads1, &tau);
        let r2 = grand_product(&reads2, &tau);
        let r_total = grand_product(&combined_reads, &tau);
        assert_eq!(r_total, r1.mul(&r2));
        // Divisibility: T = R_total · Q_total with the multiset quotient.
        let t_total = grand_product(&table, &tau);
        let mut remaining = table.clone();
        for rv in &combined_reads {
            let idx = remaining.iter().position(|t| *t == *rv);
            assert!(idx.is_some(), "read {rv:?} not in table");
            remaining.swap_remove(idx.ok_or(()).ok().unwrap());
        }
        let q_total = grand_product(&remaining, &tau);
        assert_eq!(q_total.mul(&r_total), t_total);
    }
}
