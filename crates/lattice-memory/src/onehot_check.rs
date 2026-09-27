//! The one-hot constraint PIOP (Twist & Shout Figs 6 and 8, ePrint
//! 2025/105 §4.1.2 / §4.2): proves that the committed per-dimension
//! matrices `ra_i` (or `wa_i`) are well-formed d-dimensional one-hot
//! encodings of the committed address column, and grants the verifier
//! query access to the virtual address polynomial.
//!
//! Three checks per side:
//!
//! 1. **Booleanity** (Fig 6 line 3 / Fig 8 line 3): for each dimension i,
//!    `0 = Σ_{k_i,j} eq(r^i,k_i)·eq(r',j)·(ra_i(k_i,j)² − ra_i(k_i,j))`
//!    via the sumcheck protocol (degree-3 round polynomials).
//! 2. **Hamming weight one** (Fig 6 line 5 / Fig 8 line 4): per dimension,
//!    `1 = Σ_{k_i} ra_i(k_i, r')` — checked with the **2^-1 point trick**
//!    (`Σ_{k ∈ {0,1}^m} f̃(k) = 2^m · f̃(2^-1,...,2^-1)`), valid on
//!    Goldilocks since the field characteristic is 2^64 − 2^32 + 1 > 2.
//!    No sumcheck and no extra evaluation queries: the verifier needs one
//!    resolver evaluation per dimension at the fixed point
//!    `(2^-1,...,2^-1, r')` and checks `N · ra_i(...) = 1`.
//! 3. **raf-evaluation** (Fig 6 line 6 / Fig 8 line 5): the sumcheck
//!    `y = Σ_{k,j} eq(r',j)·w(k)·Π_i ra_i(k_i, j)` with the digit-weight
//!    polynomial `w(k) = int(k)` (its MLE is the affine extension, see
//!    `DenseMle::int_extension`). This binds the one-hot matrices to the
//!    committed address column's evaluation `y = raf(r')`.
//!
//! Soundness (paper Thm 2 / Thm 3): `(6·log K + 4·log T)/|F|` for d = 1
//! and `(4d·log T + 6·log K)/|F|` in general, plus the commitment layer's
//! binding of the resolver-supplied factor evaluations.
//!
//! The paper runs the d Booleanity sumchecks "in parallel" (shared round
//! challenges) as an optimization; this implementation runs them
//! sequentially with fresh randomness — identical soundness, one extra
//! transcript challenge per dimension.

use crate::onehot::{embed_dim, OneHotLayout};
use crate::{FactorId, FactorResolver, PiopError};
use lattice_core::transcript::Transcript;
use lattice_core::{DenseMle, Goldilocks};
use lattice_sumcheck::sumcheck;
use lattice_sumcheck::SumcheckProof;
use lattice_sumcheck::VirtualPolynomial;

/// Which side of the memory instance is being checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneHotSide {
    Read,
    Write,
}

impl OneHotSide {
    /// The per-dimension matrix factor for this side.
    pub fn matrix(self, dim: usize) -> FactorId {
        match self {
            OneHotSide::Read => FactorId::Ra(dim),
            OneHotSide::Write => FactorId::Wa(dim),
        }
    }

    /// The address column factor this side's `y` is resolved against.
    pub fn addr_column(self) -> FactorId {
        match self {
            OneHotSide::Read => FactorId::ReadAddr,
            OneHotSide::Write => FactorId::WriteAddr,
        }
    }
}

/// The one-hot constraint proof: d Booleanity sumchecks + the
/// raf-evaluation sumcheck.
#[derive(Clone, Debug)]
pub struct OneHotProof {
    pub booleanity: Vec<SumcheckProof>,
    pub raf: SumcheckProof,
}

/// Absorb the instance metadata (both prover and verifier).
fn absorb_meta(
    log_k: usize,
    log_t: usize,
    d: usize,
    y_factor: FactorId,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let meta = [
        Goldilocks::from_u64(log_k as u64),
        Goldilocks::from_u64(log_t as u64),
        Goldilocks::from_u64(d as u64),
        Goldilocks::from_u64(y_factor.discriminant()),
    ];
    transcript.append_field_slice(b"onehot-meta", &meta)?;
    Ok(())
}

/// Prove the one-hot constraint PIOP for one side.
///
/// `matrices` are the per-dimension `(k_i, j)` one-hot matrices (the
/// committed indicator witness); `side` selects which factors the
/// resolver resolves them (and the `y` address column) as.
pub fn prove_onehot(
    matrices: &[DenseMle],
    log_k: usize,
    log_t: usize,
    side: OneHotSide,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<OneHotProof, PiopError> {
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
    let y_factor = side.addr_column();
    absorb_meta(log_k, log_t, d, y_factor, transcript)?;

    // Shared cycle point r' (Fig 6 line 2).
    let r_prime = transcript.challenge_fields(b"onehot-rprime", log_t)?;
    // The claimed address-column evaluation y = raf(r').
    let y = resolver.eval(y_factor, &r_prime)?;
    transcript.append_field(b"onehot-y", &y)?;

    // 1. Booleanity per dimension.
    let mut booleanity = Vec::with_capacity(d);
    for matrix in matrices.iter() {
        let r_i = transcript.challenge_fields(b"onehot-bool-r", log_n)?;
        let mut point = r_i.clone();
        point.extend(r_prime.iter().copied());
        let eq = DenseMle::eq_extension(&point);
        let mut vp = VirtualPolynomial::new(log_n + log_t);
        let ra = vp.add_factor(matrix.clone())?;
        let eqi = vp.add_factor(eq)?;
        vp.add_term(Goldilocks::ONE, vec![ra, ra, eqi])?;
        vp.add_term(Goldilocks::ONE.neg(), vec![ra, eqi])?;
        let out = sumcheck::prove(&vp, Goldilocks::ZERO, transcript)?;
        booleanity.push(out.proof);
    }

    // 3. raf-evaluation sumcheck over the full (k, j) space.
    let mut vp = VirtualPolynomial::new(log_k + log_t);
    let eq_j = DenseMle::one(log_k).tensor(&DenseMle::eq_extension(&r_prime));
    let eq_idx = vp.add_factor(eq_j)?;
    // w(k) = int(k): the affine digit-weight polynomial.
    let w_evals: Vec<Goldilocks> = (0..layout.k())
        .map(|k| Goldilocks::from_u64(k as u64))
        .collect();
    let w_ext = DenseMle::new(w_evals)?.tensor(&DenseMle::one(log_t));
    let w_idx = vp.add_factor(w_ext)?;
    let mut ra_ids = Vec::with_capacity(d);
    for (i, matrix) in matrices.iter().enumerate() {
        let emb = embed_dim(matrix, &layout, i)?;
        ra_ids.push(vp.add_factor(emb)?);
    }
    let mut term = vec![eq_idx, w_idx];
    term.extend(ra_ids.iter().copied());
    vp.add_term(Goldilocks::ONE, term)?;
    let raf_out = sumcheck::prove(&vp, y, transcript)?;

    Ok(OneHotProof { booleanity, raf: raf_out.proof })
}

/// Verify the one-hot constraint PIOP for one side.
///
/// Resolver evaluations are treated as claimed factor evaluations; the
/// commitment layer must authenticate them (the zkVM passes a resolver
/// over its committed + opened columns).
pub fn verify_onehot(
    proof: &OneHotProof,
    log_k: usize,
    log_t: usize,
    side: OneHotSide,
    resolver: &dyn FactorResolver,
    transcript: &mut Transcript,
) -> Result<(), PiopError> {
    let d = proof.booleanity.len();
    let layout = OneHotLayout::new(log_k, log_t, d, usize::MAX)?;
    let log_n = layout.log_n();
    let y_factor = side.addr_column();
    absorb_meta(log_k, log_t, d, y_factor, transcript)?;

    let r_prime = transcript.challenge_fields(b"onehot-rprime", log_t)?;
    let y = resolver.eval(y_factor, &r_prime)?;
    transcript.append_field(b"onehot-y", &y)?;

    // 1. Booleanity: replay each sumcheck and check the terminal identity
    //    final == eq(ρ)·(ra(ρ)² − ra(ρ)).
    for (i, sc) in proof.booleanity.iter().enumerate() {
        let r_i = transcript.challenge_fields(b"onehot-bool-r", log_n)?;
        let verdict = sc.verify(log_n + log_t, 3, Goldilocks::ZERO, transcript, None)?;
        let ra_v = resolver.eval(side.matrix(i), &verdict.point)?;
        let mut eq_point = r_i.clone();
        eq_point.extend(r_prime.iter().copied());
        let eq_v = DenseMle::eq_eval(&eq_point, &verdict.point)?;
        let expect = eq_v.mul(&ra_v.square().sub(&ra_v));
        if verdict.final_claim != expect {
            return Err(PiopError::FinalCheckFailed("onehot booleanity"));
        }
    }

    // 2. Hamming weight one: the 2^-1 point trick. For each dimension,
    //    N · ra_i(2^-1,...,2^-1, r') == 1.
    let inv2 = Goldilocks::TWO
        .inverse()
        .ok_or(PiopError::InverseOfTwo)?;
    let mut weight_point = vec![inv2; log_n];
    weight_point.extend(r_prime.iter().copied());
    for i in 0..d {
        let v = resolver.eval(side.matrix(i), &weight_point)?;
        let n_fe = Goldilocks::from_u64(layout.n() as u64);
        if n_fe.mul(&v) != Goldilocks::ONE {
            return Err(PiopError::FinalCheckFailed("onehot hamming weight"));
        }
    }

    // 3. raf-evaluation: final == eq(r', ρ_j)·int(ρ_k)·Π_i ra_i(ρ_k^{(i)}, ρ_j).
    let verdict = proof
        .raf
        .verify(log_k + log_t, 2 + d, y, transcript, None)?;
    let rho = verdict.point;
    let (rho_k, rho_j) = rho.split_at(log_k);
    let eq_v = DenseMle::eq_eval(&r_prime, rho_j)?;
    let w_v = DenseMle::int_extension(rho_k)?;
    let mut prod = eq_v.mul(&w_v);
    for i in 0..d {
        // Native (k_i, j) point: dimension i's digits of ρ_k, then ρ_j.
        let mut native = rho_k[i * log_n..(i + 1) * log_n].to_vec();
        native.extend(rho_j.iter().copied());
        let ra_v = resolver.eval(side.matrix(i), &native)?;
        prod = prod.mul(&ra_v);
    }
    if verdict.final_claim != prod {
        return Err(PiopError::FinalCheckFailed("onehot raf evaluation"));
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

    fn bool_col(addr: &[u64], log_t: usize) -> DenseMle {
        // Address column padded to 2^log_t.
        let mut vals: Vec<Goldilocks> = addr.iter().map(|&a| fe(a)).collect();
        vals.resize(1 << log_t, Goldilocks::ZERO);
        DenseMle::new(vals).ok().unwrap()
    }

    fn hot_for(addr: &[u64]) -> Vec<u32> {
        addr.iter().map(|&a| a as u32).collect()
    }

    #[test]
    fn honest_d1_proves_and_verifies() {
        let log_k = 3usize;
        let log_t = 3usize;
        let addr = vec![1u64, 5, 2, 7, 0, 3, 6, 4];
        let matrix = one_hot_dim_matrix(&hot_for(&addr), log_k, log_t).ok().unwrap();
        let col = bool_col(&addr, log_t);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_addr: Some(&col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        let proof = prove_onehot(
            &[matrix.clone()],
            log_k,
            log_t,
            OneHotSide::Read,
            &resolver,
            &mut t,
        )
        .ok()
        .unwrap();
        let mut t2 = Transcript::new_default(b"onehot-test");
        assert!(verify_onehot(&proof, log_k, log_t, OneHotSide::Read, &resolver, &mut t2).is_ok());
        // Transcript desync (different protocol label) must reject.
        let mut t3 = Transcript::new_default(b"other");
        assert!(verify_onehot(&proof, log_k, log_t, OneHotSide::Read, &resolver, &mut t3).is_err());
    }

    #[test]
    fn honest_d2_proves_and_verifies() {
        // K = 16, d = 2 (N = 4), T = 4.
        let log_k = 4usize;
        let log_t = 2usize;
        let addr = vec![0u64, 7, 15, 3];
        let layout = OneHotLayout::new(log_k, log_t, 2, usize::MAX).ok().unwrap();
        let digits: Vec<Vec<u32>> = addr.iter().map(|&a| layout.digits(a).ok().unwrap()).collect();
        let hot0: Vec<u32> = digits.iter().map(|d| d[0]).collect();
        let hot1: Vec<u32> = digits.iter().map(|d| d[1]).collect();
        let m0 = one_hot_dim_matrix(&hot0, 2, log_t).ok().unwrap();
        let m1 = one_hot_dim_matrix(&hot1, 2, log_t).ok().unwrap();
        let col = bool_col(&addr, log_t);
        let resolver = WitnessResolver {
            ra: vec![Some(&m0), Some(&m1)],
            read_addr: Some(&col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        let proof = prove_onehot(
            &[m0.clone(), m1.clone()],
            log_k,
            log_t,
            OneHotSide::Read,
            &resolver,
            &mut t,
        )
        .ok()
        .unwrap();
        let mut t2 = Transcript::new_default(b"onehot-test");
        assert!(verify_onehot(&proof, log_k, log_t, OneHotSide::Read, &resolver, &mut t2).is_ok());
    }

    /// Soundness: a non-Boolean indicator entry makes the prover refuse
    /// (Booleanity claim mismatch — the sumcheck engine fails closed).
    #[test]
    fn non_boolean_matrix_refused_by_prover() {
        let log_k = 2usize;
        let log_t = 2usize;
        let mut matrix = one_hot_dim_matrix(&[1u32, 0, 3, 2], log_k, log_t)
            .ok()
            .unwrap();
        // A "2" entry: not Boolean.
        matrix.evaluations[0] = fe(2);
        let col = bool_col(&[1, 0, 3, 2], log_t);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_addr: Some(&col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        assert!(prove_onehot(&[matrix.clone()], log_k, log_t, OneHotSide::Read, &resolver, &mut t).is_err());
    }

    /// Soundness (the weight check): an all-zero row is Boolean, so the
    /// prover CAN complete — with a y consistent with the broken matrix —
    /// but the verifier rejects at the Hamming-weight check: exactly the
    /// class the old fingerprint statements never checked.
    #[test]
    fn zero_row_weight_violation_rejected_by_verifier() {
        let log_k = 2usize;
        let log_t = 2usize;
        // Cycle 1's indicator row is all-zero: Boolean, but weight zero.
        // Its decoded address is 0, matching the address column — so the
        // whole proof completes and ONLY the weight check catches it.
        let matrix = one_hot_dim_matrix(&[1u32, 0, 3, 2], log_k, log_t).ok().unwrap();
        let mut bad = matrix.clone();
        for k in 0..4usize {
            bad.evaluations[k * 4 + 1] = Goldilocks::ZERO;
        }
        let bad_col = bool_col(&[1, 0, 3, 2], log_t);
        let resolver = WitnessResolver {
            ra: vec![Some(&bad)],
            read_addr: Some(&bad_col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        let proof = prove_onehot(&[bad.clone()], log_k, log_t, OneHotSide::Read, &resolver, &mut t)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"onehot-test");
        assert!(matches!(
            verify_onehot(&proof, log_k, log_t, OneHotSide::Read, &resolver, &mut t2),
            Err(PiopError::FinalCheckFailed("onehot hamming weight"))
        ));
    }

    /// Soundness: a claimed address evaluation y inconsistent with the
    /// matrices makes the raf-evaluation sumcheck fail closed.
    #[test]
    fn mismatched_y_refused_by_prover() {
        let log_k = 2usize;
        let log_t = 2usize;
        let matrix = one_hot_dim_matrix(&[1u32, 0, 3, 2], log_k, log_t).ok().unwrap();
        // Column disagrees with the matrices at cycle 3 (address 2 vs 3).
        let col = bool_col(&[1, 0, 3, 7], log_t);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_addr: Some(&col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        assert!(prove_onehot(&[matrix.clone()], log_k, log_t, OneHotSide::Read, &resolver, &mut t).is_err());
    }

    /// Soundness: a resolver that answers with a DIFFERENT matrix than
    /// the one proven (the committed-vs-proven substitution attack) fails
    /// the terminal identities.
    #[test]
    fn resolver_substitution_rejected() {
        let log_k = 2usize;
        let log_t = 2usize;
        let matrix = one_hot_dim_matrix(&[1u32, 0, 3, 2], log_k, log_t).ok().unwrap();
        let other = one_hot_dim_matrix(&[2u32, 1, 0, 3], log_k, log_t).ok().unwrap();
        let col = bool_col(&[1, 0, 3, 2], log_t);
        let prover_resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_addr: Some(&col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        let proof = prove_onehot(&[matrix.clone()], log_k, log_t, OneHotSide::Read, &prover_resolver, &mut t)
            .ok()
            .unwrap();
        // Cheating resolver: different matrix at the booleanity points.
        let bad_col = bool_col(&[2, 1, 0, 3], log_t);
        let cheat = WitnessResolver {
            ra: vec![Some(&other)],
            read_addr: Some(&bad_col),
            ..Default::default()
        };
        let mut t2 = Transcript::new_default(b"onehot-test");
        assert!(verify_onehot(&proof, log_k, log_t, OneHotSide::Read, &cheat, &mut t2).is_err());
    }

    /// Soundness: tampered sumcheck rounds are rejected by the engine.
    #[test]
    fn tampered_rounds_rejected() {
        let log_k = 2usize;
        let log_t = 2usize;
        let matrix = one_hot_dim_matrix(&[1u32, 0, 3, 2], log_k, log_t).ok().unwrap();
        let col = bool_col(&[1, 0, 3, 2], log_t);
        let resolver = WitnessResolver {
            ra: vec![Some(&matrix)],
            read_addr: Some(&col),
            ..Default::default()
        };
        let mut t = Transcript::new_default(b"onehot-test");
        let mut proof = prove_onehot(&[matrix.clone()], log_k, log_t, OneHotSide::Read, &resolver, &mut t)
            .ok()
            .unwrap();
        if let Some(round) = proof.raf.rounds.first_mut() {
            if let Some(v) = round.first_mut() {
                *v = v.add(&fe(1));
            }
        }
        let mut t2 = Transcript::new_default(b"onehot-test");
        assert!(verify_onehot(&proof, log_k, log_t, OneHotSide::Read, &resolver, &mut t2).is_err());
    }
}
