//! The core Shout PIOPs (Twist & Shout Figs 5 and 7, ePrint 2025/105
//! §2.5.2, §4.1.1, §4.2): read-checking sumchecks for read-only memories
//! (lookup tables).
//!
//! Given a size-K table `Val` and T committed reads (one-hot matrices
//! `ra_1..ra_d` plus the claimed read-value column `rv`), the PIOP proves
//!
//! `rv(j) = Σ_k Π_i ra_i(k_i, j) · Val(k)` for all cycles j.
//!
//! * **General d (Fig 7)**: the verifier samples `rcycle` and the sumcheck
//!   runs over the full `(k, j)` cube on
//!   `eq(rcycle, j)·Π_i ra_i(k_i, j)·Val(k)` with claim `rv(rcycle)`.
//!   Degree `d + 2` round polynomials; soundness
//!   `((d+2)·log T + 2·log K)/|F|` (Thm 3). The dense prover here
//!   materializes the embedded matrices (kernel scale); the production
//!   prover is the sparse one ([`crate::sparse`], §6.2/§7: O(K) + O(T)
//!   field operations).
//! * **d = 1 fast form (Fig 5)**: when d = 1 the RHS is multilinear in
//!   `rcycle`, so the j-sum disappears — the sumcheck runs over the log K
//!   address variables ONLY, on `ra_bound(k)·Val(k)` with
//!   `ra_bound = ra(·, rcycle)` (cycle variables pre-bound via
//!   `fix_last_variables`). This is the paper's headline T-field-
//!   multiplication prover (§6.1).
//!
//! The table is absorbed in full at kernel scale; MLE-structured tables
//! (Jolt's subtables) would be evaluated in O(log K) instead (§2.5.2).

use crate::onehot::{embed_dim, OneHotLayout};
use crate::{FactorId, FactorResolver, PiopError};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

/// A Shout proof: the read-checking sumcheck (Fig 7 form).
#[derive(Clone, Debug)]
pub struct ShoutProof {
    pub read_checking: SumcheckProof,
}

fn table_mle(table: &[Goldilocks], log_k: usize) -> Result<DenseMle, PiopError> {
    if table.len() != 1 << log_k {
        return Err(PiopError::Shape {
            expected: 1 << log_k,
            got: table.len(),
        });
    }
    Ok(DenseMle::new(table.to_vec())?)
}

fn absorb_shout_meta(
    log_k: usize,
    log_t: usize,
    d: usize,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let meta = [
        Goldilocks::from_u64(log_k as u64),
        Goldilocks::from_u64(log_t as u64),
        Goldilocks::from_u64(d as u64),
    ];
    transcript.append_field_slice(b"shout-meta", &meta)?;
    Ok(())
}

fn absorb_table(table: &[Goldilocks], transcript: &mut Transcript) -> Result<(), PiopError> {
    transcript.append_field_slice(b"shout-table", table)?;
    Ok(())
}

/// Prove the Fig 7 read-checking sumcheck for general d.
///
/// `matrices` are the per-dimension one-hot `(k_i, j)` matrices;
/// `rv(rcycle)` (the claim) is resolved from the committed read-value
/// column.
pub fn prove_shout(
    table: &[Goldilocks],
    matrices: &[DenseMle],
    log_k: usize,
    log_t: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<ShoutProof, PiopError> {
    let d = matrices.len();
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    for m in matrices {
        if m.num_vars != log_n + log_t {
            return Err(PiopError::Shape {
                expected: log_n + log_t,
                got: m.num_vars,
            });
        }
    }
    let table_m = table_mle(table, log_k)?;
    absorb_shout_meta(log_k, log_t, d, transcript)?;
    absorb_table(table, transcript)?;

    // Fig 7 line 2: V picks rcycle.
    let rcycle = transcript.challenge_fields(b"shout-rcycle", log_t)?;
    // The claim rv(rcycle), resolved from the committed read-value column.
    let claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"shout-claim", &claim)?;

    // VP: eq(rcycle, j) · Val(k) · Π_i ra_i(k_i, j) over (k, j).
    let mut vp = VirtualPolynomial::new(log_k + log_t);
    let eq_j = DenseMle::one(log_k).tensor(&DenseMle::eq_extension(&rcycle));
    let eq_idx = vp.add_factor(eq_j)?;
    let val_idx = vp.add_factor(table_m.tensor(&DenseMle::one(log_t)))?;
    let mut ra_ids = Vec::with_capacity(d);
    for (i, m) in matrices.iter().enumerate() {
        ra_ids.push(vp.add_factor(embed_dim(m, &layout, i)?)?);
    }
    let mut term = vec![eq_idx, val_idx];
    term.extend(ra_ids.iter().copied());
    vp.add_term(Goldilocks::ONE, term)?;
    let out = sumcheck::prove(&vp, claim, transcript)?;
    Ok(ShoutProof {
        read_checking: out.proof,
    })
}

/// Verify the Fig 7 read-checking sumcheck.
///
/// `d` is the one-hot dimension count (the caller's instance parameter).
pub fn verify_shout(
    proof: &ShoutProof,
    table: &[Goldilocks],
    log_k: usize,
    log_t: usize,
    d: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    let table_m = table_mle(table, log_k)?;
    absorb_shout_meta(log_k, log_t, d, transcript)?;
    absorb_table(table, transcript)?;

    let rcycle = transcript.challenge_fields(b"shout-rcycle", log_t)?;
    let claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"shout-claim", &claim)?;

    let verdict = proof
        .read_checking
        .verify(log_k + log_t, 2 + d, claim, transcript, None)?;
    let rho = verdict.point;
    let (rho_k, rho_j) = rho.split_at(log_k);
    // Terminal identity: eq(rcycle, ρ_j) · Val(ρ_k) · Π_i ra_i(ρ_k^{(i)}, ρ_j).
    let eq_v = DenseMle::eq_eval(&rcycle, rho_j)?;
    let val_v = table_m.evaluate(rho_k)?;
    let mut prod = eq_v.mul(&val_v);
    for i in 0..d {
        let mut native = rho_k[i * log_n..(i + 1) * log_n].to_vec();
        native.extend(rho_j.iter().copied());
        prod = prod.mul(&resolver.eval(FactorId::Ra(i), &native)?);
    }
    if verdict.final_claim != prod {
        return Err(PiopError::FinalCheckFailed("shout read-checking"));
    }
    Ok(())
}

/// Prove the Fig 5 fast form (d = 1 only): the sumcheck runs over the log K
/// address variables with the cycle block pre-bound to `rcycle` — the
/// paper's T-multiplication core Shout prover (§6.1).
pub fn prove_shout_core_d1(
    table: &[Goldilocks],
    ra: &DenseMle,
    log_k: usize,
    log_t: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<SumcheckProof, PiopError> {
    if ra.num_vars != log_k + log_t {
        return Err(PiopError::Shape {
            expected: log_k + log_t,
            got: ra.num_vars,
        });
    }
    let table_m = table_mle(table, log_k)?;
    transcript.append_field_slice(
        b"shout-core-meta",
        &[Goldilocks::from_u64(log_k as u64), Goldilocks::from_u64(log_t as u64)],
    )?;
    absorb_table(table, transcript)?;
    let rcycle = transcript.challenge_fields(b"shout-rcycle", log_t)?;
    let claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"shout-claim", &claim)?;

    // ra_bound(k) = ra(k, rcycle): bind the cycle block (least-significant
    // variables) while the address block stays free.
    let ra_bound = ra.fix_last_variables(&rcycle)?;
    let mut vp = VirtualPolynomial::new(log_k);
    let ra_idx = vp.add_factor(ra_bound)?;
    let val_idx = vp.add_factor(table_m)?;
    vp.add_term(Goldilocks::ONE, vec![ra_idx, val_idx])?;
    let out = sumcheck::prove(&vp, claim, transcript)?;
    Ok(out.proof)
}

/// Verify the Fig 5 fast form. The terminal identity binds
/// `ra(ρ_k, rcycle)` through the resolver (an opening claim on the full
/// matrix, per Fig 5 line 4).
pub fn verify_shout_core_d1(
    proof: &SumcheckProof,
    table: &[Goldilocks],
    log_k: usize,
    log_t: usize,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let table_m = table_mle(table, log_k)?;
    transcript.append_field_slice(
        b"shout-core-meta",
        &[Goldilocks::from_u64(log_k as u64), Goldilocks::from_u64(log_t as u64)],
    )?;
    absorb_table(table, transcript)?;
    let rcycle = transcript.challenge_fields(b"shout-rcycle", log_t)?;
    let claim = resolver.eval(FactorId::ReadValues, &rcycle)?;
    transcript.append_field(b"shout-claim", &claim)?;

    let verdict = proof.verify(log_k, 2, claim, transcript, None)?;
    // Terminal identity: ra(ρ_k, rcycle) · Val(ρ_k).
    let mut point = verdict.point.clone();
    point.extend(rcycle.iter().copied());
    let ra_v = resolver.eval(FactorId::Ra(0), &point)?;
    let val_v = table_m.evaluate(&verdict.point)?;
    if verdict.final_claim != ra_v.mul(&val_v) {
        return Err(PiopError::FinalCheckFailed("shout core d1"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onehot::one_hot_dim_matrix;
    use crate::WitnessResolver;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    struct Fixture {
        table: Vec<Goldilocks>,
        log_k: usize,
        log_t: usize,
        reads: Vec<(u64, u64)>,
    }

    fn fixture() -> Fixture {
        // K = 8 table, T = 4 reads.
        let table: Vec<Goldilocks> = [10u64, 21, 32, 43, 54, 65, 76, 87].map(fe).to_vec();
        let reads = vec![(2u64, 32u64), (7, 87), (0, 10), (2, 32)];
        Fixture { table, log_k: 3, log_t: 2, reads }
    }

    fn build(fx: &Fixture) -> (DenseMle, DenseMle) {
        let hot: Vec<u32> = fx.reads.iter().map(|&(a, _)| a as u32).collect();
        let matrix = one_hot_dim_matrix(&hot, fx.log_k, fx.log_t).ok().unwrap();
        let mut rv: Vec<Goldilocks> = fx.reads.iter().map(|&(_, v)| fe(v)).collect();
        rv.resize(1 << fx.log_t, Goldilocks::ZERO);
        (matrix, DenseMle::new(rv).ok().unwrap())
    }

    #[test]
    fn honest_d1_fig7_proves_and_verifies() {
        let fx = fixture();
        let (matrix, rv_col) = build(&fx);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        let proof = prove_shout(&fx.table, &[matrix.clone()], fx.log_k, fx.log_t, &resolver, &mut t)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"shout-test");
        assert!(
            verify_shout(&proof, &fx.table, fx.log_k, fx.log_t, 1, &resolver, &mut t2).is_ok()
        );
    }

    #[test]
    fn honest_d1_fig5_fast_form_proves_and_verifies() {
        let fx = fixture();
        let (matrix, rv_col) = build(&fx);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        let proof = prove_shout_core_d1(&fx.table, &matrix, fx.log_k, fx.log_t, &resolver, &mut t)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"shout-test");
        assert!(
            verify_shout_core_d1(&proof, &fx.table, fx.log_k, fx.log_t, &resolver, &mut t2).is_ok()
        );
    }

    #[test]
    fn honest_d2_fig7_proves_and_verifies() {
        // K = 16, d = 2 (N = 4), T = 4.
        let log_k = 4usize;
        let log_t = 2usize;
        let table: Vec<Goldilocks> = (0..16u64).map(|i| fe(i * 7 + 3)).collect();
        let reads = vec![(0u64, 3u64), (7, 52), (15, 108), (3, 24)];
        let layout = OneHotLayout::new(log_k, log_t, 2, usize::MAX).ok().unwrap();
        let digits: Vec<Vec<u32>> = reads.iter().map(|&(a, _)| layout.digits(a).ok().unwrap()).collect();
        let m0 = one_hot_dim_matrix(&digits.iter().map(|d| d[0]).collect::<Vec<_>>(), 2, log_t)
            .ok()
            .unwrap();
        let m1 = one_hot_dim_matrix(&digits.iter().map(|d| d[1]).collect::<Vec<_>>(), 2, log_t)
            .ok()
            .unwrap();
        let mut rv: Vec<Goldilocks> = reads.iter().map(|&(_, v)| fe(v)).collect();
        rv.resize(1 << log_t, Goldilocks::ZERO);
        let rv_col = DenseMle::new(rv).ok().unwrap();
        let m0c = m0.clone();
        let m1c = m1.clone();
        let resolver = WitnessResolver {
            ra: vec![Some(&m0c), Some(&m1c)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        let proof = prove_shout(&table, &[m0, m1], log_k, log_t, &resolver, &mut t)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"shout-test");
        assert!(verify_shout(&proof, &table, log_k, log_t, 2, &resolver, &mut t2).is_ok());
    }

    /// Soundness: a wrong read value (read does not match the table entry)
    /// — the exact bug class `shout_check` catches deterministically —
    /// makes the sumcheck fail closed at the claim.
    #[test]
    fn wrong_read_value_refused_by_prover() {
        let mut fx = fixture();
        fx.reads[1] = (7, 88); // table[7] = 87
        let (matrix, rv_col) = build(&fx);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        assert!(
            prove_shout(&fx.table, &[matrix.clone()], fx.log_k, fx.log_t, &resolver, &mut t).is_err()
        );
        let mut t2 = Transcript::new_default(b"shout-test");
        let (m2, rv2) = build(&fx);
        let res2 = WitnessResolver {
            ra: vec![Some(&m2)],
            read_values: Some(&rv2),
            ..Default::default()
        };
        assert!(
            prove_shout_core_d1(&fx.table, &m2, fx.log_k, fx.log_t, &res2, &mut t2).is_err()
        );
    }

    /// Soundness: the verifier resolving the claim against a TAMPERED
    /// read-value column desyncs the transcript and rejects.
    #[test]
    fn tampered_read_values_column_rejected_by_verifier() {
        let fx = fixture();
        let (matrix, rv_col) = build(&fx);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        let proof = prove_shout(&fx.table, &[matrix.clone()], fx.log_k, fx.log_t, &resolver, &mut t)
            .ok()
            .unwrap();
        // Tamper the rv column the verifier resolves against.
        let mut bad_vals = rv_col.evaluations.clone();
        bad_vals[0] = fe(999);
        let bad_col = DenseMle::new(bad_vals).ok().unwrap();
        let bad_resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&bad_col),
            ..Default::default()
        };
        let mut t2 = Transcript::new_default(b"shout-test");
        assert!(
            verify_shout(&proof, &fx.table, fx.log_k, fx.log_t, 1, &bad_resolver, &mut t2).is_err()
        );
    }

    /// Soundness: tampered sumcheck rounds rejected.
    #[test]
    fn tampered_rounds_rejected() {
        let fx = fixture();
        let (matrix, rv_col) = build(&fx);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        let mut proof = prove_shout(&fx.table, &[matrix.clone()], fx.log_k, fx.log_t, &resolver, &mut t)
            .ok()
            .unwrap();
        if let Some(round) = proof.read_checking.rounds.first_mut() {
            if let Some(v) = round.first_mut() {
                *v = v.add(&fe(1));
            }
        }
        let mut t2 = Transcript::new_default(b"shout-test");
        assert!(
            verify_shout(&proof, &fx.table, fx.log_k, fx.log_t, 1, &resolver, &mut t2).is_err()
        );
    }

    #[test]
    fn bad_shapes_rejected() {
        let fx = fixture();
        let (matrix, rv_col) = build(&fx);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_values: Some(&rv_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"shout-test");
        // Wrong table length.
        let short_table = fx.table[..4].to_vec();
        assert!(
            prove_shout(&short_table, &[matrix.clone()], fx.log_k, fx.log_t, &resolver, &mut t)
                .is_err()
        );
        // Wrong matrix arity.
        let small = one_hot_dim_matrix(&[0u32, 1, 2, 3], 2, fx.log_t).ok().unwrap();
        assert!(prove_shout(&fx.table, &[small], fx.log_k, fx.log_t, &resolver, &mut t).is_err());
    }
}
