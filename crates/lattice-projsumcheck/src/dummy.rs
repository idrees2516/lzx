//! Projective **batched sum-check with claim-preserving dummy rounds** —
//! §B.1 of ePrint 2026/762.
//!
//! Jolt-style zkVMs batch many sum-check instances of different sizes
//! into one lockstep execution ("front-loaded" batching). Under the
//! Boolean round identity `H(0) + H(1) = claim`, a dormant instance must
//! send `H(X) = claim/2` every dummy round, which forces the input claims
//! to be pre-scaled by `2^{n_max − n}` and halved once per dummy round.
//!
//! The projective round identity `H(0) + H(∞) = claim` dissolves this:
//! the constant univariate `H(X) = claim` satisfies it *honestly*
//! (`H(0) = claim`, `H(∞) = 0`), because a polynomial that does not use
//! the extra leading variables has **no monomials in them** — under
//! `{0,∞}` interpolation the dormant instance's data sits in the low
//! coefficient slots untouched, and its hypercube sum is invariant under
//! the padding. Concretely, with this crate's convention:
//!
//! * an instance with `n < n_max` variables is embedded by placing its
//!   coefficient array in the low `2^n` slots (all monomials involving
//!   the first `n_max − n` variables have coefficient zero);
//! * binding a leading variable `X ← r` of the dormant instance computes
//!   `lo[i] + r·hi[i]` where `hi` is identically zero — the array passes
//!   through **unchanged**, and the honest round message is the constant
//!   claim;
//! * no `2^{n_max − n}` pre-scaling, no per-round halving, no
//!   renormalization at the activation boundary (§B.2: "the arithmetic
//!   pre-scaling and claim renormalization disappear").

use crate::proj_mle::MonomialMle;
use crate::proj_sumcheck::{
    interpolate_with_infinity, ProjSumcheckError, ProjSumcheckProof,
};
use lattice_core::field_simd::{self, Sum8};
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;

/// One batched instance: a virtual polynomial plus its claimed sum.
#[derive(Clone, Debug)]
pub struct BatchedInstance {
    /// The instance's own variable count (`n ≤ n_max`).
    pub num_vars: usize,
    /// Virtual polynomial factors in *native* size (2^num_vars each).
    pub factors: Vec<MonomialMle>,
    /// `(coefficient, factor indices)` terms.
    pub terms: Vec<(Goldilocks, Vec<usize>)>,
    /// Claimed `Σ P_i` over the instance's own hypercube.
    pub claim: Goldilocks,
}

impl BatchedInstance {
    /// A single product instance.
    pub fn product(factors: Vec<MonomialMle>, claim: Goldilocks) -> Self {
        let terms = vec![(Goldilocks::ONE, (0..factors.len()).collect())];
        BatchedInstance {
            num_vars: factors.first().map(|f| f.num_vars).unwrap_or(0),
            factors,
            terms,
            claim,
        }
    }

    pub fn max_degree(&self) -> usize {
        self.terms.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1)
    }
}

/// Batched prover output.
#[derive(Clone, Debug)]
pub struct BatchedOutput {
    pub proof: ProjSumcheckProof,
    /// Challenges (variable 0 first).
    pub challenges: Vec<Goldilocks>,
    /// Terminal claims per factor of every instance (native order).
    pub factor_claims: Vec<Vec<Goldilocks>>,
    /// Terminal `P_i(r_i)` claims per instance.
    pub final_claims: Vec<Goldilocks>,
}

/// Embed an instance's coefficient array into `2^{n_max}` slots (low
/// slots; monomials of the leading extra variables are zero).
fn embed(coeffs: &[Goldilocks], n_max: usize) -> Vec<Goldilocks> {
    let mut out = vec![Goldilocks::ZERO; 1usize << n_max];
    out[..coeffs.len()].copy_from_slice(coeffs);
    out
}

/// `γ · Σ_j c_j · Π_k slices[k]` over a full slice set.
fn sum_terms_scaled(
    terms: &[(Goldilocks, Vec<usize>)],
    slices: &[&[Goldilocks]],
    gamma: Goldilocks,
) -> Goldilocks {
    let mut acc = Sum8::new();
    let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
    for (c, ids) in terms {
        fslices.clear();
        fslices.extend(ids.iter().map(|fi| slices[*fi]));
        acc.accumulate_term(*c, &fslices);
    }
    acc.finish().mul(&gamma)
}

/// Prove the batch: `Σ_i γ^i · Σ_{b ∈ {0,∞}^{n_max}} P_i^{(emb)}(b) =
/// Σ_i γ^i · claim_i` — one lockstep projective sum-check over `n_max`
/// rounds with **no claim pre-scaling** (the γ scalars are drawn from the
/// transcript before the first round).
pub fn prove_batched(
    instances: &[BatchedInstance],
    transcript: &mut Transcript,
) -> Result<BatchedOutput, ProjSumcheckError> {
    if instances.is_empty() {
        return Err(ProjSumcheckError::EmptyInstance);
    }
    let n_max = instances.iter().map(|i| i.num_vars).max().unwrap_or(0);
    let d = instances.iter().map(|i| i.max_degree()).max().unwrap_or(1);
    let n_fin = d.saturating_sub(1);

    // Batching scalars γ^i and the batched claim (no pre-scaling!).
    let mut gammas = Vec::with_capacity(instances.len());
    for _ in 0..instances.len() {
        gammas.push(
            transcript
                .challenge_field(b"projsumcheck-batch-gamma")
                .map_err(ProjSumcheckError::Transcript)?,
        );
    }
    let mut claim = Goldilocks::ZERO;
    for (inst, gam) in instances.iter().zip(gammas.iter()) {
        claim = claim.add(&gam.mul(&inst.claim));
    }

    // Embedded, bound coefficient arrays per instance.
    let mut bound: Vec<Vec<Vec<Goldilocks>>> = instances
        .iter()
        .map(|inst| inst.factors.iter().map(|f| embed(&f.coeffs, n_max)).collect())
        .collect();

    // An instance is *live* at round `round` iff its own variables are
    // being bound: round ≥ n_max − n.
    let mut live: Vec<bool> = instances.iter().map(|i| i.num_vars == n_max).collect();

    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n_max);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n_max);

    for round in 0..n_max {
        let rem = 1usize << (n_max - round);
        let half = rem / 2;
        let mut s0 = Goldilocks::ZERO;
        let mut s_inf = Goldilocks::ZERO;
        let mut s_fin = vec![Goldilocks::ZERO; n_fin.max(1)];

        for (i, inst) in instances.iter().enumerate() {
            if !live[i] {
                // Dormant: honest constant message γ^i · claim_i —
                // H(X) = const, H(0) = γ^i·claim_i, H(∞) = 0.
                let contrib = gammas[i].mul(&inst.claim);
                s0 = s0.add(&contrib);
                for sf in s_fin.iter_mut() {
                    *sf = sf.add(&contrib);
                }
                continue;
            }
            let facs = &bound[i];
            // s(0) over the first halves; s(∞) over the second halves.
            let first: Vec<&[Goldilocks]> = facs.iter().map(|f| &f[..half]).collect();
            s0 = s0.add(&sum_terms_scaled(&inst.terms, &first, gammas[i]));
            let second: Vec<&[Goldilocks]> = facs.iter().map(|f| &f[half..]).collect();
            s_inf = s_inf.add(&sum_terms_scaled(&inst.terms, &second, gammas[i]));

            // Finite points t = 1..d−1: materialize α + t·β per factor.
            for t in 1..d {
                let mut buffers: Vec<Vec<Goldilocks>> = Vec::with_capacity(facs.len());
                for f in facs {
                    let (lo, hi) = f.split_at(half);
                    let mut buf = vec![Goldilocks::ZERO; half];
                    if t == 1 {
                        // Pure addition: f(1) = f(0) + f(∞).
                        field_simd::add_slices(lo, hi, &mut buf);
                    } else {
                        let tf = Goldilocks::from_u64(t as u64);
                        field_simd::mul_scalar_slice(hi, tf, &mut buf);
                        for k in 0..half {
                            buf[k] = lo[k].add(&buf[k]);
                        }
                    }
                    buffers.push(buf);
                }
                let slices: Vec<&[Goldilocks]> = buffers.iter().map(|b| b.as_slice()).collect();
                let val = sum_terms_scaled(&inst.terms, &slices, gammas[i]);
                s_fin[t - 1] = s_fin[t - 1].add(&val);
            }
        }

        // Round identity guard: s(0) + s(∞) = C_{i−1}.
        if s0.add(&s_inf) != current_claim {
            return Err(ProjSumcheckError::ClaimMismatch);
        }

        let mut evals = Vec::with_capacity(d);
        evals.push(s_inf);
        evals.extend(s_fin.iter().copied());
        transcript
            .append_field_slice(b"projsumcheck-round", &evals)
            .map_err(ProjSumcheckError::Transcript)?;
        let r = transcript
            .challenge_field(b"projsumcheck-challenge")
            .map_err(ProjSumcheckError::Transcript)?;
        challenges.push(r);

        let mut finite = vec![s0];
        finite.extend(evals.iter().skip(1).copied());
        current_claim = interpolate_with_infinity(&finite, s_inf, r);

        // Projective binding of every live instance's factors; dormant
        // arrays pass through (their high halves are identically zero),
        // but every array truncates so live sizes stay in lockstep.
        for (i, facs) in bound.iter_mut().enumerate() {
            for f in facs.iter_mut() {
                if live[i] {
                    MonomialMle::bind_first_slice_in_place(f, r);
                }
                f.truncate(half);
            }
        }

        // Wake instances whose own variables begin binding next round.
        for (i, inst) in instances.iter().enumerate() {
            if !live[i] && round + 1 + inst.num_vars >= n_max {
                live[i] = true;
            }
        }
        rounds.push(evals);
    }

    // Terminal claims: after n_max rounds every array has length 1
    // (dormant instances bound through zero halves, preserving their
    // constants in slot 0).
    let mut factor_claims = Vec::with_capacity(instances.len());
    let mut final_claims = Vec::with_capacity(instances.len());
    for (i, inst) in instances.iter().enumerate() {
        let fc: Vec<Goldilocks> = bound[i].iter().map(|f| f[0]).collect();
        let mut fin = Goldilocks::ZERO;
        for (c, ids) in &inst.terms {
            let mut prod = *c;
            for fi in ids {
                prod = prod.mul(&fc[*fi]);
            }
            fin = fin.add(&prod);
        }
        factor_claims.push(fc);
        final_claims.push(fin);
    }
    let batched_terminal: Goldilocks = final_claims
        .iter()
        .zip(gammas.iter())
        .fold(Goldilocks::ZERO, |acc, (fin, gam)| acc.add(&fin.mul(gam)));
    if current_claim != batched_terminal {
        return Err(ProjSumcheckError::FinalCheckFailed);
    }

    Ok(BatchedOutput {
        proof: ProjSumcheckProof { rounds },
        challenges,
        factor_claims,
        final_claims,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Batch an instance of 6 variables with one of 3 variables: the
    /// lockstep proof verifies, the batched claim needs no pre-scaling,
    /// and each instance's terminal claim equals its own `P_i(r_i)` —
    /// with the small instance's point being the challenge suffix.
    #[test]
    fn batched_mixed_sizes_roundtrip() {
        let f_big = MonomialMle::random(6, b"b-big");
        let h_big = MonomialMle::random(6, b"b-big2");
        let f_small = MonomialMle::random(3, b"b-small");
        let h_small = MonomialMle::random(3, b"b-small2");

        let claim_big = f_big
            .coeffs
            .iter()
            .zip(h_big.coeffs.iter())
            .fold(Goldilocks::ZERO, |acc, (a, b)| acc.add(&a.mul(b)));
        let claim_small = f_small
            .coeffs
            .iter()
            .zip(h_small.coeffs.iter())
            .fold(Goldilocks::ZERO, |acc, (a, b)| acc.add(&a.mul(b)));

        let instances = vec![
            BatchedInstance::product(vec![f_big.clone(), h_big.clone()], claim_big),
            BatchedInstance::product(vec![f_small.clone(), h_small.clone()], claim_small),
        ];
        let mut ts = Transcript::new_default(b"batch-prove");
        let out = prove_batched(&instances, &mut ts).unwrap();

        // Verify the lockstep proof against the batched claim.
        let mut ts2 = Transcript::new_default(b"batch-prove");
        let mut gammas = Vec::new();
        for _ in 0..2 {
            gammas.push(ts2.challenge_field(b"projsumcheck-batch-gamma").unwrap());
        }
        let batched_claim = instances
            .iter()
            .zip(gammas.iter())
            .fold(Goldilocks::ZERO, |acc, (inst, gam)| acc.add(&gam.mul(&inst.claim)));
        let v = out.proof.verify(batched_claim, 6, 2, &mut ts2).unwrap();
        assert_eq!(v.point, out.challenges);

        // Terminal claims per instance.
        let big0 = f_big.evaluate(&out.challenges).unwrap();
        let big1 = h_big.evaluate(&out.challenges).unwrap();
        assert_eq!(out.factor_claims[0], vec![big0, big1]);

        let r_small: Vec<Goldilocks> = out.challenges[3..].to_vec();
        let small0 = f_small.evaluate(&r_small).unwrap();
        let small1 = h_small.evaluate(&r_small).unwrap();
        assert_eq!(out.factor_claims[1], vec![small0, small1]);

        // Batched terminal identity: Σ γ^i · P_i(r_i).
        let want = gammas[0]
            .mul(&out.final_claims[0])
            .add(&gammas[1].mul(&out.final_claims[1]));
        assert_eq!(v.final_claim, want);
    }

    /// Single-instance batch degenerates to the plain protocol shape.
    #[test]
    fn batched_single_instance() {
        let f = MonomialMle::random(4, b"s-f");
        let h = MonomialMle::random(4, b"s-h");
        let claim = f
            .coeffs
            .iter()
            .zip(h.coeffs.iter())
            .fold(Goldilocks::ZERO, |acc, (a, b)| acc.add(&a.mul(b)));
        let instances = vec![BatchedInstance::product(vec![f, h], claim)];
        let mut ts = Transcript::new_default(b"single-batch");
        let out = prove_batched(&instances, &mut ts).unwrap();
        let mut ts2 = Transcript::new_default(b"single-batch");
        let gam = ts2.challenge_field(b"projsumcheck-batch-gamma").unwrap();
        let v = out
            .proof
            .verify(gam.mul(&claim), 4, 2, &mut ts2)
            .unwrap();
        assert_eq!(v.final_claim, gam.mul(&out.final_claims[0]));
    }

    /// Three sizes at once (7 / 5 / 2 variables) — dormancy chains.
    #[test]
    fn batched_three_sizes() {
        let mk = |seed: &[u8], n: usize| MonomialMle::random(n, seed);
        let fa = mk(b"3-a", 7);
        let fb = mk(b"3-b", 7);
        let fc = mk(b"3-c", 5);
        let fd = mk(b"3-d", 5);
        let fe = mk(b"3-e", 2);
        let ca = fa.coeffs.iter().zip(fb.coeffs.iter()).fold(Goldilocks::ZERO, |a, (x, y)| a.add(&x.mul(y)));
        let cc = fc.coeffs.iter().zip(fd.coeffs.iter()).fold(Goldilocks::ZERO, |a, (x, y)| a.add(&x.mul(y)));
        let ce = fe.coeffs.iter().map(|x| x.mul(x)).fold(Goldilocks::ZERO, |a, x| a.add(&x));
        let instances = vec![
            BatchedInstance::product(vec![fa.clone(), fb.clone()], ca),
            BatchedInstance::product(vec![fc.clone(), fd.clone()], cc),
            BatchedInstance::product(vec![fe.clone(), fe.clone()], ce),
        ];
        let mut ts = Transcript::new_default(b"three-batch");
        let out = prove_batched(&instances, &mut ts).unwrap();
        let mut ts2 = Transcript::new_default(b"three-batch");
        let gammas: Vec<Goldilocks> = (0..3)
            .map(|_| ts2.challenge_field(b"projsumcheck-batch-gamma").unwrap())
            .collect();
        let batched = instances
            .iter()
            .zip(gammas.iter())
            .fold(Goldilocks::ZERO, |acc, (inst, gam)| acc.add(&gam.mul(&inst.claim)));
        let v = out.proof.verify(batched, 7, 2, &mut ts2).unwrap();
        assert_eq!(v.point, out.challenges);
        // Instance points are challenge suffixes of matching length.
        let r5: Vec<Goldilocks> = out.challenges[2..].to_vec();
        assert_eq!(
            out.factor_claims[1][0],
            fc.evaluate(&r5).unwrap()
        );
        let r2: Vec<Goldilocks> = out.challenges[5..].to_vec();
        assert_eq!(
            out.factor_claims[2][0],
            fe.evaluate(&r2).unwrap()
        );
    }
}
