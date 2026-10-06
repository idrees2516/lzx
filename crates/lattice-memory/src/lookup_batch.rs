//! Batched read-only-table lookups (the stream-concatenation +
//! eq-block-RLC design): L lookups over the SAME public table share ONE
//! sparse sumcheck.
//!
//! The v3 statement carries ~205 read-only-table Shouts (decode / pow2 /
//! al8 / the byte range8 / range7 / range12 carry chains). Run
//! individually, each spends a full `log_k + log_t`-round transcript
//! flow, its own table absorb, its own eq table, and its own opening
//! claim point. Run BATCHED over a group sharing one table, they become
//! ONE `SparseInstance` with L ρ-weighted terms over the same
//! `(k, j)` cube:
//!
//! ```text
//! Σ_{k,j} eq(rcycle, j) · T[k] · Σ_ℓ ρ_ℓ · Π_i ra^{(ℓ)}_i(k_i, j)
//!   = Σ_ℓ ρ_ℓ · v_ℓ(rcycle)
//! ```
//!
//! * ONE shared `rcycle` challenge (after all L value claims are
//!   absorbed) — so every claim of the group lives at ONE point, which
//!   is also what the cross-column opening fold wants;
//! * ONE shared eq table and ONE shared table factor (bound once per
//!   round instead of once per lookup);
//! * the round count collapses from `L·(log_k + log_t)` to
//!   `log_k + log_t`;
//! * the per-(lookup, digit) `ra` factor claims ride the proof
//!   (`ra_claims`, absorbed post-sumcheck) — the terminal identity
//!   `Σ_ℓ ρ_ℓ · eq · T(ρ_k) · Π_i ra^{(ℓ)}_i` is the same anchor the
//!   per-lookup flow's resolver claims provided.
//!
//! Soundness is the ρ-RLC of the per-lookup statements (each identity
//! necessary by challenge independence); the value claims are resolved
//! through the caller's resolvers exactly as before, so the
//! committed-column binding and the claim-table flow are unchanged.

use crate::sparse_engine::{
    prove_sparse_sumcheck_owned, ProjectedDense, SparseFactor, SparseInstance, SparseTerm,
};
use crate::structured_table::StructuredTable;
use crate::{FactorId, FactorResolver, PiopError};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::SumcheckProof;

/// One lookup in a batch: the per-cycle read addresses plus the resolver
/// that supplies the lookup's value column (`FactorId::ReadValues`).
pub struct BatchedLookup<'a> {
    /// The table index read at each cycle (`T` entries, each `< K`).
    pub read_addresses: &'a [u64],
    /// Resolves the value claim `v(rcycle)` against the committed column.
    pub resolver: &'a dyn FactorResolver,
}

/// The batched Shout proof for one table group.
#[derive(Clone, Debug)]
pub struct BatchedShoutProof {
    /// The single read-checking sumcheck over the `(k, j)` cube.
    pub sumcheck: SumcheckProof,
    /// The per-(lookup, digit) factor claim values — `L·d` field
    /// elements, lookup-major / digit-minor — at the batched terminal's
    /// native `(k_i, j)` points (recomputed from the shared challenges).
    pub ra_claims: Vec<Goldilocks>,
}

/// The batch-group metadata (prover and verifier must agree exactly).
#[derive(Clone, Debug)]
pub struct BatchGroup<'a> {
    /// The shared public table.
    pub table: &'a StructuredTable,
    pub log_k: usize,
    pub log_t: usize,
    /// One-hot dimension count (`d = log_k` gives bit-digits).
    pub d: usize,
}

/// Prove one batch group: ONE sumcheck over all lookups of the group.
pub fn prove_shout_batched(
    group: &BatchGroup,
    lookups: &[BatchedLookup<'_>],
    transcript: &mut Transcript,
) -> Result<BatchedShoutProof, PiopError> {
    let log_k = group.log_k;
    let log_t = group.log_t;
    let d = group.d;
    let ell_count = lookups.len();
    if ell_count == 0 {
        return Err(PiopError::BadLayout { log_k, d });
    }
    let layout = crate::onehot::OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    let t = layout.t();
    let k = layout.k();
    let _ = k;
    // ---- Transcript: meta + table statement + rcycle + claims + ρ. ----
    transcript
        .append_field_slice(
            b"shout-batch-meta",
            &[
                Goldilocks::from_u64(log_k as u64),
                Goldilocks::from_u64(log_t as u64),
                Goldilocks::from_u64(d as u64),
                Goldilocks::from_u64(ell_count as u64),
                Goldilocks::from_u64(group.table.family.tag()),
            ],
        )
        .map_err(PiopError::Transcript)?;
    group
        .table
        .absorb_statement(transcript, b"shout-batch-table")
        .map_err(PiopError::Transcript)?;
    let rcycle = transcript.challenge_fields(b"shout-batch-rcycle", log_t)?;
    let mut claims = Vec::with_capacity(ell_count);
    for lk in lookups {
        if lk.read_addresses.len() != t {
            return Err(PiopError::Shape {
                expected: t,
                got: lk.read_addresses.len(),
            });
        }
        for &a in lk.read_addresses {
            if a >= k as u64 {
                return Err(PiopError::AddressOutOfRange { address: a, k });
            }
        }
        let v = lk.resolver.eval(FactorId::ReadValues, &rcycle)?;
        claims.push(v);
        transcript.append_field(b"shout-batch-claim", &v)?;
    }
    let rho = transcript.challenge_fields(b"shout-batch-rho", ell_count)?;
    let mut combined_claim = Goldilocks::ZERO;
    for (r, v) in rho.iter().zip(claims.iter()) {
        combined_claim = combined_claim.add(&r.mul(v));
    }

    // ---- The instance: L terms, shared dense pair, per-lookup ra dims. ----
    let mut sparse: Vec<SparseFactor> = Vec::with_capacity(ell_count * d);
    let mut terms: Vec<SparseTerm> = Vec::with_capacity(ell_count);
    let table_values = group.table.values();
    if table_values.len() != k {
        return Err(PiopError::Shape {
            expected: k,
            got: table_values.len(),
        });
    }
    for (ell, lk) in lookups.iter().enumerate() {
        let positions: Vec<u64> = lk
            .read_addresses
            .iter()
            .enumerate()
            .map(|(j, &a)| (a << log_t) | j as u64)
            .collect();
        let mut digits_per_dim: Vec<Vec<u32>> = vec![Vec::with_capacity(t); d];
        for &a in lk.read_addresses {
            let digits = layout.digits(a)?;
            for (i, dg) in digits.iter().enumerate() {
                digits_per_dim[i].push(*dg);
            }
        }
        for (i, dim) in digits_per_dim.iter().enumerate() {
            let mut var_map: Vec<usize> = (i * log_n..(i + 1) * log_n).collect();
            var_map.extend(log_k..log_k + log_t);
            let entries: Vec<(u64, Goldilocks)> = dim
                .iter()
                .enumerate()
                .map(|(j, &dg)| (((dg as u64) << log_t) | j as u64, Goldilocks::ONE))
                .collect();
            sparse.push(SparseFactor { entries, var_map });
        }
        terms.push(SparseTerm {
            coeff: rho[ell],
            positions,
            sparse: (ell * d..ell * d + d).collect(),
            dense: vec![0, 1],
        });
    }
    let eq_j = ProjectedDense {
        mle: DenseMle::eq_extension(&rcycle),
        var_map: (log_k..log_k + log_t).collect(),
    };
    let val_f = ProjectedDense {
        mle: DenseMle::new(table_values)?,
        var_map: (0..log_k).collect(),
    };
    let inst = SparseInstance {
        num_vars: log_k + log_t,
        sparse,
        dense: vec![eq_j, val_f],
        terms,
    };
    let out = prove_sparse_sumcheck_owned(inst, combined_claim, transcript)?;

    // ---- The per-(lookup, digit) factor claims, absorbed post-rounds. ----
    let ra_claims = out.sparse_claims.clone();
    transcript
        .append_field_slice(b"shout-batch-ra", &ra_claims)
        .map_err(PiopError::Transcript)?;
    Ok(BatchedShoutProof {
        sumcheck: out.proof,
        ra_claims,
    })
}

/// Verify one batch group: the same transcript flow, the single
/// sumcheck, and the RLC terminal identity through the structured
/// table's closed-form MLE at the terminal address point.
#[allow(clippy::too_many_lines)]
pub fn verify_shout_batched(
    group: &BatchGroup,
    lookups: &[BatchedLookup<'_>],
    proof: &BatchedShoutProof,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let log_k = group.log_k;
    let log_t = group.log_t;
    let d = group.d;
    let ell_count = lookups.len();
    if ell_count == 0 || proof.ra_claims.len() != ell_count * d {
        return Err(PiopError::Shape {
            expected: ell_count * d,
            got: proof.ra_claims.len(),
        });
    }
    transcript
        .append_field_slice(
            b"shout-batch-meta",
            &[
                Goldilocks::from_u64(log_k as u64),
                Goldilocks::from_u64(log_t as u64),
                Goldilocks::from_u64(d as u64),
                Goldilocks::from_u64(ell_count as u64),
                Goldilocks::from_u64(group.table.family.tag()),
            ],
        )
        .map_err(PiopError::Transcript)?;
    group
        .table
        .absorb_statement(transcript, b"shout-batch-table")
        .map_err(PiopError::Transcript)?;
    let rcycle = transcript.challenge_fields(b"shout-batch-rcycle", log_t)?;
    let mut claims = Vec::with_capacity(ell_count);
    for lk in lookups {
        let v = lk.resolver.eval(FactorId::ReadValues, &rcycle)?;
        claims.push(v);
        transcript.append_field(b"shout-batch-claim", &v)?;
    }
    let rho = transcript.challenge_fields(b"shout-batch-rho", ell_count)?;
    let mut combined_claim = Goldilocks::ZERO;
    for (r, v) in rho.iter().zip(claims.iter()) {
        combined_claim = combined_claim.add(&r.mul(v));
    }
    let verdict = proof
        .sumcheck
        .verify(log_k + log_t, d + 2, combined_claim, transcript, None)?;
    transcript
        .append_field_slice(b"shout-batch-ra", &proof.ra_claims)
        .map_err(PiopError::Transcript)?;
    // Terminal identity: Σ_ℓ ρ_ℓ · eq(rcycle, ρ_j) · T(ρ_k) · Π_i ra_ℓ,i.
    let rho_full = &verdict.point;
    let (rho_k, rho_j) = rho_full.split_at(log_k);
    let eq_v = DenseMle::eq_eval(&rcycle, rho_j)?;
    let t_val = group
        .table
        .eval_mle(rho_k)
        .map_err(|_| PiopError::MissingFactor {
            factor: FactorId::Table,
        })?;
    let mut expect = Goldilocks::ZERO;
    for (ell, r) in rho.iter().enumerate() {
        let mut prod = r.mul(&eq_v).mul(&t_val);
        for i in 0..d {
            prod = prod.mul(&proof.ra_claims[ell * d + i]);
        }
        expect = expect.add(&prod);
    }
    if verdict.final_claim != expect {
        return Err(PiopError::FinalCheckFailed("batched shout terminal"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_table::{StructuredTable, TableFamily};
    use crate::WitnessResolver;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    /// Two lookups over one identity table (K=16, T=4, d=4 bit-digits).
    fn fixture() -> (Vec<Vec<u64>>, Vec<Vec<Goldilocks>>, Vec<Vec<Goldilocks>>) {
        let _log_k = 4usize;
        let log_t = 2usize;
        let addrs_l = vec![vec![0u64, 7, 15, 3], vec![5u64, 1, 9, 12]];
        let mut rvs = Vec::new();
        let mut cols = Vec::new();
        for a in &addrs_l {
            let rv: Vec<Goldilocks> = a.iter().map(|&x| fe(x)).collect();
            let mut col = rv.clone();
            col.resize(1 << log_t, Goldilocks::ZERO);
            rvs.push(rv);
            cols.push(col);
        }
        (addrs_l, rvs, cols)
    }

    /// Holds the column MLEs alive behind the borrowed resolvers.
    struct FixtureRes {
        mles: Vec<DenseMle>,
    }

    impl FixtureRes {
        fn new(cols: &[Vec<Goldilocks>]) -> Self {
            FixtureRes {
                mles: cols
                    .iter()
                    .map(|c| DenseMle::new(c.clone()).ok().unwrap())
                    .collect(),
            }
        }
        fn res(&self) -> Vec<WitnessResolver<'_>> {
            self.mles
                .iter()
                .map(|m| WitnessResolver {
                    read_values: Some(m),
                    ..Default::default()
                })
                .collect()
        }
    }

    #[test]
    fn batched_honest_proves_and_verifies() {
        let (addrs, _rvs, cols) = fixture();
        let fr = FixtureRes::new(&cols);
        let mles = fr.res();
        let table = StructuredTable::new(TableFamily::Identity, 4);
        let group = BatchGroup {
            table: &table,
            log_k: 4,
            log_t: 2,
            d: 4,
        };
        let lks: Vec<BatchedLookup> = addrs
            .iter()
            .zip(mles.iter())
            .map(|(a, r)| BatchedLookup {
                read_addresses: a,
                resolver: r,
            })
            .collect();
        let mut t = Transcript::new_default(b"batch-test");
        let proof = prove_shout_batched(&group, &lks, &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"batch-test");
        assert!(verify_shout_batched(&group, &lks, &proof, &mut t2).is_ok());
    }

    #[test]
    fn batched_prover_catches_dishonest_lookup() {
        // The second lookup claims a value that is NOT its address
        // (identity table): the combined claim mismatch fails closed.
        let (addrs, _rvs, mut cols) = fixture();
        cols[1][2] = fe(99); // true address 9, value 99
        let fr = FixtureRes::new(&cols);
        let mles = fr.res();
        let table = StructuredTable::new(TableFamily::Identity, 4);
        let group = BatchGroup {
            table: &table,
            log_k: 4,
            log_t: 2,
            d: 4,
        };
        let lks: Vec<BatchedLookup> = addrs
            .iter()
            .zip(mles.iter())
            .map(|(a, r)| BatchedLookup {
                read_addresses: a,
                resolver: r,
            })
            .collect();
        let mut t = Transcript::new_default(b"batch-test");
        assert!(prove_shout_batched(&group, &lks, &mut t).is_err());
    }

    #[test]
    fn batched_tampered_rounds_rejected() {
        let (addrs, _rvs, cols) = fixture();
        let fr = FixtureRes::new(&cols);
        let mles = fr.res();
        let table = StructuredTable::new(TableFamily::Identity, 4);
        let group = BatchGroup {
            table: &table,
            log_k: 4,
            log_t: 2,
            d: 4,
        };
        let lks: Vec<BatchedLookup> = addrs
            .iter()
            .zip(mles.iter())
            .map(|(a, r)| BatchedLookup {
                read_addresses: a,
                resolver: r,
            })
            .collect();
        let mut t = Transcript::new_default(b"batch-test");
        let mut proof = prove_shout_batched(&group, &lks, &mut t).ok().unwrap();
        if let Some(round) = proof.sumcheck.rounds.first_mut() {
            if let Some(v) = round.first_mut() {
                *v = v.add(&fe(1));
            }
        }
        let mut t2 = Transcript::new_default(b"batch-test");
        assert!(verify_shout_batched(&group, &lks, &proof, &mut t2).is_err());
    }

    #[test]
    fn batched_tampered_ra_claims_rejected() {
        let (addrs, _rvs, cols) = fixture();
        let fr = FixtureRes::new(&cols);
        let mles = fr.res();
        let table = StructuredTable::new(TableFamily::Identity, 4);
        let group = BatchGroup {
            table: &table,
            log_k: 4,
            log_t: 2,
            d: 4,
        };
        let lks: Vec<BatchedLookup> = addrs
            .iter()
            .zip(mles.iter())
            .map(|(a, r)| BatchedLookup {
                read_addresses: a,
                resolver: r,
            })
            .collect();
        let mut t = Transcript::new_default(b"batch-test");
        let mut proof = prove_shout_batched(&group, &lks, &mut t).ok().unwrap();
        proof.ra_claims[0] = proof.ra_claims[0].add(&fe(3));
        let mut t2 = Transcript::new_default(b"batch-test");
        assert!(verify_shout_batched(&group, &lks, &proof, &mut t2).is_err());
    }

    #[test]
    fn batched_tampered_claim_value_rejected() {
        // The verifier resolves a DIFFERENT value column: desync.
        let (addrs, _rvs, mut cols) = fixture();
        let fr = FixtureRes::new(&cols);
        let mles = fr.res();
        let table = StructuredTable::new(TableFamily::Identity, 4);
        let group = BatchGroup {
            table: &table,
            log_k: 4,
            log_t: 2,
            d: 4,
        };
        let lks: Vec<BatchedLookup> = addrs
            .iter()
            .zip(mles.iter())
            .map(|(a, r)| BatchedLookup {
                read_addresses: a,
                resolver: r,
            })
            .collect();
        let mut t = Transcript::new_default(b"batch-test");
        let proof = prove_shout_batched(&group, &lks, &mut t).ok().unwrap();
        cols[0][0] = fe(12345);
        let fr2 = FixtureRes::new(&cols);
        let mles2 = fr2.res();
        let lks2: Vec<BatchedLookup> = addrs
            .iter()
            .zip(mles2.iter())
            .map(|(a, r)| BatchedLookup {
                read_addresses: a,
                resolver: r,
            })
            .collect();
        let mut t2 = Transcript::new_default(b"batch-test");
        assert!(verify_shout_batched(&group, &lks2, &proof, &mut t2).is_err());
    }

    #[test]
    fn batched_pow2_family() {
        // One lookup over the pow2 table: T=4 cycles, shamt values.
        let log_k = 4usize;
        let log_t = 2usize;
        let addrs = vec![vec![3u64, 9, 1, 15]];
        let cols: Vec<Vec<Goldilocks>> =
            vec![addrs[0].iter().map(|&a| fe(1u64 << a)).collect::<Vec<_>>()];
        let fr = FixtureRes::new(&cols);
        let mles = fr.res();
        let table = StructuredTable::new(TableFamily::Pow2, log_k);
        let group = BatchGroup {
            table: &table,
            log_k,
            log_t,
            d: log_k,
        };
        let lks: Vec<BatchedLookup> = vec![BatchedLookup {
            read_addresses: &addrs[0],
            resolver: &mles[0],
        }];
        let mut t = Transcript::new_default(b"batch-test");
        let proof = prove_shout_batched(&group, &lks, &mut t).ok().unwrap();
        let mut t2 = Transcript::new_default(b"batch-test");
        assert!(verify_shout_batched(&group, &lks, &proof, &mut t2).is_ok());
        // Address out of range fails closed.
        let bad = vec![vec![3u64, 16, 1, 15]];
        let bad_lks: Vec<BatchedLookup> = vec![BatchedLookup {
            read_addresses: &bad[0],
            resolver: &mles[0],
        }];
        let mut t3 = Transcript::new_default(b"batch-test");
        assert!(prove_shout_batched(&group, &bad_lks, &mut t3).is_err());
    }
}
