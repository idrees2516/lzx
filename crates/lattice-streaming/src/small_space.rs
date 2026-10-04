//! **Algorithm 1 — the small-space sum-check prover** (ePrint 2025/611,
//! §3.1.2, Theorem 3.3): `O(n + ℓ²)` space, `O(ℓ²·n·2^n)` time.
//!
//! The protocol is the *standard Boolean* sum-check (bit-identical round
//! messages to `lattice-sumcheck`); only the prover's memory strategy
//! changes: instead of caching the bound factor arrays (which halve each
//! round but start at `O(ℓ·2^n)`), every round re-derives the bound
//! values from the oracles via the eq-weighted combination over the
//! already-bound prefix (Claim 3.2 / Equation 6):
//!
//! ```text
//! g_k(r_0..r_{i−1}, α, tobits(m)) =
//!     Σ_{j ∈ {0,1}^i} eqf(r_{0..i−1}, tobits(j)) ·
//!         [ (1−α) · A_k[(j, 0, tobits(m))] + α · A_k[(j, 1, tobits(m))] ]
//! ```
//!
//! Per round the prover makes one oracle sweep of `2^n` queries grouped
//! by the remaining-hypercube index `m` (the `2^i` prefix queries per
//! `m` land at `u_even = 2^i·2m + j` and `u_odd = 2^i·(2m+1) + j` in the
//! paper's LSB-first indexing — under this crate's MSB-first layout,
//! prefix `j` occupies the *high* bits and the round variable sits at
//! bit `n−1−i`).
//!
//! The eq weights over the prefix are maintained with a Gray-code walk
//! (`j → j+1` flips only the trailing-one bits), amortized `O(1)` per
//! weight — the lex-order enumeration technique of [CFFZE24 §4.1] the
//! paper cites. Random access is served by [`IndexOracle`]
//! implementations; client-side this is checkpointed regeneration.

use crate::oracle::IndexOracle;
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmallSpaceError {
    Transcript(TranscriptError),
    /// Instance/factor variable-count mismatch.
    BadShape {
        expected: usize,
        got: usize,
    },
    /// Prover-side round identity failed (construction bug guard).
    RoundCheckFailed {
        round: usize,
    },
    ClaimMismatch,
}

impl core::fmt::Display for SmallSpaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SmallSpaceError::Transcript(e) => write!(f, "transcript error: {e}"),
            SmallSpaceError::BadShape { expected, got } => {
                write!(f, "small-space instance shape {got} != {expected}")
            }
            SmallSpaceError::RoundCheckFailed { round } => {
                write!(f, "round identity failed at round {round}")
            }
            SmallSpaceError::ClaimMismatch => write!(f, "claim mismatch"),
        }
    }
}

/// A small-space sum-check instance: `Σ_j c_j · Π_k f_{j,k}` over
/// `{0,1}^n`, factors served by indexed oracles.
pub struct SmallSpaceInstance<'a> {
    pub num_vars: usize,
    /// One oracle per shared factor.
    pub factors: Vec<&'a mut dyn IndexOracle>,
    /// `(coefficient, factor indices)` terms.
    pub terms: Vec<(Goldilocks, Vec<usize>)>,
}

impl<'a> SmallSpaceInstance<'a> {
    pub fn max_degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(_, ids)| ids.len())
            .max()
            .unwrap_or(1)
    }
}

/// Prover output — same shape as the Boolean engine's, so proofs verify
/// with `lattice_sumcheck::SumcheckProof` and cross-validate directly.
#[derive(Clone, Debug)]
pub struct SmallSpaceOutput {
    /// Round polynomials: round `j` carries `g_j(0..=d)` (the standard
    /// Boolean compressed form, degree `d`).
    pub rounds: Vec<Vec<Goldilocks>>,
    pub challenges: Vec<Goldilocks>,
    pub final_claim: Goldilocks,
    /// Per-factor claimed evaluations at the terminal point.
    pub factor_claims: Vec<Goldilocks>,
}

/// eq over two bit-vectors given as field points (MSB-first).
fn eq_points(r: &[Goldilocks], j_bits: &[bool]) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for (i, &b) in j_bits.iter().enumerate() {
        let one_minus = Goldilocks::ONE.sub(&r[i]);
        acc = acc.mul(&if b { r[i] } else { one_minus });
    }
    acc
}

/// Gray-code walk of `eq(r_{<k}, tobits(j))` for `j = 0, 1, 2, …`.
///
/// `j → j+1` flips the trailing-one bits plus the next zero bit; each
/// flip multiplies the weight by the per-bit ratio
/// `eq(r_b, 0)/eq(r_b, 1) = (1−r_b)/r_b`. Degenerate challenges
/// (`r_b ∈ {0,1}`, probability `2^{-64}` each) fall back to direct
/// recomputation.
pub(crate) struct EqWalk {
    k: usize,
    /// Precomputed `(1−r_b)/r_b` for 1→0 flips (None when degenerate).
    ratios: Vec<Option<Goldilocks>>,
    /// Precomputed `r_b/(1−r_b)` for 0→1 flips.
    ratio_invs: Vec<Option<Goldilocks>>,
    j: u64,
    w: Goldilocks,
    r: Vec<Goldilocks>,
}

impl EqWalk {
    pub(crate) fn new(r: &[Goldilocks]) -> Self {
        let k = r.len();
        let mut w = Goldilocks::ONE;
        let mut ratios = Vec::with_capacity(k);
        let mut ratio_invs = Vec::with_capacity(k);
        for rb in r {
            let one_minus = Goldilocks::ONE.sub(rb);
            ratios.push(rb.inverse().map(|inv| one_minus.mul(&inv)));
            ratio_invs.push(one_minus.inverse().map(|inv| rb.mul(&inv)));
            w = w.mul(&one_minus); // j = 0: all bits zero.
        }
        EqWalk {
            k,
            ratios,
            ratio_invs,
            j: 0,
            w,
            r: r.to_vec(),
        }
    }

    pub(crate) fn weight(&self) -> Goldilocks {
        self.w
    }

    pub(crate) fn advance(&mut self) {
        self.j += 1;
        let degenerate = self.ratios.iter().any(|x| x.is_none());
        if degenerate {
            let bits: Vec<bool> = (0..self.k)
                .map(|b| (self.j >> (self.k - 1 - b)) & 1 == 1)
                .collect();
            self.w = eq_points(&self.r, &bits);
            return;
        }
        // j−1's trailing-one bits flip 1→0 (ratio = (1−r)/r); the
        // stopping zero bit flips 0→1 (inverse ratio = r/(1−r)).
        // Integer bit t of j is MSB-first variable k−1−t.
        let mut t = 0usize;
        while t < self.k && (self.j - 1) >> t & 1 == 1 {
            if let Some(ratio) = self.ratios[self.k - 1 - t] {
                self.w = self.w.mul(&ratio);
            }
            t += 1;
        }
        if t < self.k {
            if let Some(inv) = self.ratio_invs[self.k - 1 - t] {
                self.w = self.w.mul(&inv);
            }
        }
    }
}

/// One Algorithm-1 round sweep over indexed oracles: computes the round
/// polynomial `g_round(X)` evaluated at the nodes `0..=d` with the
/// bound prefix `bound_r`, in `O(ℓ·(d+1))` space beyond the oracles.
pub(crate) fn round_evals_sweep(
    factors: &mut [&mut dyn IndexOracle],
    terms: &[(Goldilocks, Vec<usize>)],
    bound_r: &[Goldilocks],
    n: usize,
    d: usize,
) -> Vec<Goldilocks> {
    let ell = factors.len();
    let k = bound_r.len();
    let n_points = d + 1;
    let rem_bits = n - k;
    let m_count = 1u64 << (rem_bits - 1);
    let points: Vec<Goldilocks> = (0..=d as u64).map(Goldilocks::from_u64).collect();
    let mut evals_at = vec![Goldilocks::ZERO; n_points];
    let mut witness_eval = vec![Goldilocks::ZERO; ell * n_points];
    for m in 0..m_count {
        for v in witness_eval.iter_mut() {
            *v = Goldilocks::ZERO;
        }
        let mut walk = EqWalk::new(bound_r);
        for j in 0..(1u64 << k) {
            let w = walk.weight();
            let base_even = (j << rem_bits) | m;
            let base_odd = base_even | (1u64 << (rem_bits - 1));
            for (fi, oracle) in factors.iter_mut().enumerate() {
                let even = oracle.eval(base_even);
                let odd = oracle.eval(base_odd);
                for (pi, alpha) in points.iter().enumerate() {
                    let one_minus = Goldilocks::ONE.sub(alpha);
                    let combo = one_minus.mul(&even).add(&alpha.mul(&odd));
                    witness_eval[fi * n_points + pi] =
                        witness_eval[fi * n_points + pi].add(&w.mul(&combo));
                }
            }
            if j + 1 < (1u64 << k) {
                walk.advance();
            }
        }
        for (c, ids) in terms {
            for (pi, _alpha) in points.iter().enumerate() {
                let mut prod = *c;
                for fi in ids {
                    prod = prod.mul(&witness_eval[(*fi) * n_points + pi]);
                }
                evals_at[pi] = evals_at[pi].add(&prod);
            }
        }
    }
    evals_at
}

/// Terminal factor claims: the eq-weighted combination over the full
/// point (all variables bound).
pub(crate) fn terminal_claims_sweep(
    factors: &mut [&mut dyn IndexOracle],
    bound_r: &[Goldilocks],
    n: usize,
) -> Vec<Goldilocks> {
    let ell = factors.len();
    let mut accs = vec![Goldilocks::ZERO; ell];
    let mut walk = EqWalk::new(bound_r);
    for j in 0..(1u64 << n) {
        let w = walk.weight();
        for (fi, oracle) in factors.iter_mut().enumerate() {
            accs[fi] = accs[fi].add(&w.mul(&oracle.eval(j)));
        }
        if j + 1 < (1u64 << n) {
            walk.advance();
        }
    }
    accs
}

/// Prove `Σ_{x∈{0,1}^n} P(x) = claim` in `O(n + ℓ²)` space.
///
/// The oracles serve factor evaluations at arbitrary indices; the prover
/// never materializes a bound array. Round messages are bit-identical to
/// the in-memory Boolean engine's (strong cross-validation in the tests).
pub fn prove_small_space(
    inst: &mut SmallSpaceInstance<'_>,
    claim: Goldilocks,
    transcript: &mut Transcript,
) -> Result<SmallSpaceOutput, SmallSpaceError> {
    let n = inst.num_vars;
    let d = inst.max_degree();
    let ell = inst.factors.len();
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut bound_r: Vec<Goldilocks> = Vec::with_capacity(n);

    // Scratch: per-factor bound values for one (m, point) — O(ℓ·(d+1)).
    let n_points = d + 1;
    let mut witness_eval = vec![Goldilocks::ZERO; ell * n_points];

    for round in 0..n {
        let k = round; // bound prefix length
        let rem_bits = n - k; // includes the round variable
        let m_count = 1u64 << (rem_bits - 1);
        let mut evals_at = vec![Goldilocks::ZERO; n_points];

        // Per-point small-value table for the linear selector.
        let points: Vec<Goldilocks> = (0..=d as u64).map(Goldilocks::from_u64).collect();

        for m in 0..m_count {
            // Reset the per-(factor, point) accumulation for this m.
            for v in witness_eval.iter_mut() {
                *v = Goldilocks::ZERO;
            }
            // Prefix walk with Gray-coded eq weights.
            let mut walk = EqWalk::new(&bound_r);
            for j in 0..(1u64 << k) {
                let w = walk.weight();
                // Query every factor at (j, 0, m) and (j, 1, m):
                // index = j · 2^{rem_bits} + b · 2^{rem_bits−1} + m.
                let base_even = (j << rem_bits) | m;
                let base_odd = base_even | (1u64 << (rem_bits - 1));
                for (fi, oracle) in inst.factors.iter_mut().enumerate() {
                    let even = oracle.eval(base_even);
                    let odd = oracle.eval(base_odd);
                    for (pi, alpha) in points.iter().enumerate() {
                        // (1−α)·even + α·odd.
                        let one_minus = Goldilocks::ONE.sub(alpha);
                        let combo = one_minus.mul(&even).add(&alpha.mul(&odd));
                        witness_eval[fi * n_points + pi] =
                            witness_eval[fi * n_points + pi].add(&w.mul(&combo));
                    }
                }
                if j + 1 < (1u64 << k) {
                    walk.advance();
                }
            }
            // Accumulate the term products for each point.
            for (c, ids) in &inst.terms {
                for (pi, _alpha) in points.iter().enumerate() {
                    let mut prod = *c;
                    for fi in ids {
                        prod = prod.mul(&witness_eval[(*fi) * n_points + pi]);
                    }
                    evals_at[pi] = evals_at[pi].add(&prod);
                }
            }
        }

        // Prover guard: g(0) + g(1) = C_{i−1}.
        let sum01 = evals_at[0].add(&evals_at[1]);
        if sum01 != current_claim {
            return Err(SmallSpaceError::RoundCheckFailed { round });
        }

        transcript
            .append_field_slice(b"sumcheck-round", &evals_at)
            .map_err(SmallSpaceError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(SmallSpaceError::Transcript)?;
        challenges.push(r);
        bound_r.push(r);

        // C_i = g_i(r): Lagrange over the nodes 0..=d.
        current_claim = interpolate_nodes(&evals_at, &r);
        rounds.push(evals_at);
    }

    // Terminal: each factor's evaluation at the full point — served by a
    // final eq-weighted sweep (the point has all variables bound).
    let factor_claims = terminal_claims_sweep(
        &mut inst
            .factors
            .iter_mut()
            .map(|o| &mut **o as &mut dyn IndexOracle)
            .collect::<Vec<_>>(),
        &bound_r,
        n,
    );
    let mut final_claim = Goldilocks::ZERO;
    for (c, ids) in &inst.terms {
        let mut prod = *c;
        for fi in ids {
            prod = prod.mul(&factor_claims[*fi]);
        }
        final_claim = final_claim.add(&prod);
    }
    if final_claim != current_claim {
        return Err(SmallSpaceError::ClaimMismatch);
    }

    Ok(SmallSpaceOutput {
        rounds,
        challenges,
        final_claim,
        factor_claims,
    })
}

/// Lagrange evaluation of the round polynomial (values at nodes 0..=d)
/// at `r` — the same interpolation the Boolean engine uses.
pub(crate) fn interpolate_nodes(evals: &[Goldilocks], r: &Goldilocks) -> Goldilocks {
    let n = evals.len();
    let mut acc = Goldilocks::ZERO;
    for (i, &ei) in evals.iter().enumerate() {
        let xi = Goldilocks::from_u64(i as u64);
        let mut weight = Goldilocks::ONE;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = Goldilocks::from_u64(j as u64);
            let num = r.sub(&xj);
            let den = xi.sub(&xj);
            weight = weight.mul(&num.mul(&den.inverse().unwrap_or(Goldilocks::ZERO)));
        }
        acc = acc.add(&ei.mul(&weight));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use lattice_core::DenseMle;
    use lattice_sumcheck::sumcheck::SumcheckProof;
    use lattice_sumcheck::virtual_poly::VirtualPolynomial;

    /// The decisive test: the small-space prover's round messages are
    /// **bit-identical** to the in-memory Boolean engine's for the same
    /// instance and transcript seed — the same protocol, different memory
    /// strategy.
    #[test]
    fn roundtrips_match_in_memory_engine() {
        for n in [4usize, 6, 8] {
            let df = DenseMle::random(n, b"ss-f");
            let dh = DenseMle::random(n, b"ss-h");
            let mut vp = VirtualPolynomial::new(n);
            let a = vp.add_factor(df.clone()).unwrap();
            let b = vp.add_factor(dh.clone()).unwrap();
            vp.add_term(Goldilocks::from_u64(3), vec![a, b]).unwrap();
            let claim: Goldilocks = (0..(1usize << n))
                .map(|i| Goldilocks::from_u64(3).mul(&df.evaluations[i].mul(&dh.evaluations[i])))
                .fold(Goldilocks::ZERO, |acc, v| acc.add(&v));

            // In-memory reference.
            let mut ts = Transcript::new_default(b"ss-seed");
            let reference = lattice_sumcheck::sumcheck::prove(&vp, claim, &mut ts).unwrap();

            // Small-space prover over owned oracles.
            let mut of = OwnedOracle::new(df.evaluations.clone());
            let mut oh = OwnedOracle::new(dh.evaluations.clone());
            let mut inst = SmallSpaceInstance {
                num_vars: n,
                factors: vec![&mut of, &mut oh],
                terms: vec![(Goldilocks::from_u64(3), vec![0, 1])],
            };
            let mut ts2 = Transcript::new_default(b"ss-seed");
            let out = prove_small_space(&mut inst, claim, &mut ts2).unwrap();

            assert_eq!(out.rounds, reference.proof.rounds, "round messages (n={n})");
            assert_eq!(out.challenges, reference.challenges, "challenges (n={n})");
            assert_eq!(out.final_claim, reference.final_claim);
            assert_eq!(out.factor_claims, reference.factor_claims);

            // The proof verifies with the standard Boolean verifier.
            let mut ts3 = Transcript::new_default(b"ss-seed");
            let proof = SumcheckProof {
                rounds: out.rounds.clone(),
            };
            let v = proof
                .verify(n, 2, claim, &mut ts3, Some(out.final_claim))
                .unwrap();
            assert_eq!(v.point, out.challenges);
            assert_eq!(v.final_claim, out.final_claim);
        }
    }

    /// A degree-3 virtual polynomial with two terms (mixed degrees)
    /// roundtrips and matches the reference engine.
    #[test]
    fn degree3_mixed_terms() {
        let n = 5;
        let df0 = DenseMle::random(n, b"m-f0");
        let df1 = DenseMle::random(n, b"m-f1");
        let df2 = DenseMle::random(n, b"m-f2");
        let mut vp = VirtualPolynomial::new(n);
        let a = vp.add_factor(df0.clone()).unwrap();
        let b = vp.add_factor(df1.clone()).unwrap();
        let c = vp.add_factor(df2.clone()).unwrap();
        vp.add_term(Goldilocks::from_u64(2), vec![a, b, c]).unwrap();
        vp.add_term(Goldilocks::from_u64(7), vec![a]).unwrap();
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| {
                Goldilocks::from_u64(2)
                    .mul(
                        &df0.evaluations[i]
                            .mul(&df1.evaluations[i])
                            .mul(&df2.evaluations[i]),
                    )
                    .add(&Goldilocks::from_u64(7).mul(&df0.evaluations[i]))
            })
            .fold(Goldilocks::ZERO, |acc, v| acc.add(&v));

        let mut ts = Transcript::new_default(b"m-seed");
        let reference = lattice_sumcheck::sumcheck::prove(&vp, claim, &mut ts).unwrap();

        let mut o0 = OwnedOracle::new(df0.evaluations.clone());
        let mut o1 = OwnedOracle::new(df1.evaluations.clone());
        let mut o2 = OwnedOracle::new(df2.evaluations.clone());
        let mut inst = SmallSpaceInstance {
            num_vars: n,
            factors: vec![&mut o0, &mut o1, &mut o2],
            terms: vec![
                (Goldilocks::from_u64(2), vec![0, 1, 2]),
                (Goldilocks::from_u64(7), vec![0]),
            ],
        };
        let mut ts2 = Transcript::new_default(b"m-seed");
        let out = prove_small_space(&mut inst, claim, &mut ts2).unwrap();
        assert_eq!(out.rounds, reference.proof.rounds);
        assert_eq!(out.challenges, reference.challenges);
        assert_eq!(out.final_claim, reference.final_claim);
    }

    /// Checkpointed-regeneration oracles (the client-side realization)
    /// drive the same protocol to the same messages.
    #[test]
    fn works_over_regen_oracles() {
        let n = 6;
        let df = DenseMle::random(n, b"r-f");
        let dh = DenseMle::random(n, b"r-h");
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| df.evaluations[i].mul(&dh.evaluations[i]))
            .fold(Goldilocks::ZERO, |acc, v| acc.add(&v));
        let values_f = df.evaluations.clone();
        let values_h = dh.evaluations.clone();
        let (mut of, _) =
            crate::oracle::ChunkedRegenOracle::build(n, 0u64, 7, |_: &mut u64, i: u64| {
                values_f[i as usize]
            });
        let (mut oh, _) =
            crate::oracle::ChunkedRegenOracle::build(n, 0u64, 7, |_: &mut u64, i: u64| {
                values_h[i as usize]
            });
        let mut inst = SmallSpaceInstance {
            num_vars: n,
            factors: vec![&mut of, &mut oh],
            terms: vec![(Goldilocks::ONE, vec![0, 1])],
        };
        let mut ts = Transcript::new_default(b"r-seed");
        let out = prove_small_space(&mut inst, claim, &mut ts).unwrap();
        // Terminal claims equal the MLE evaluations at the point.
        let df_ref = df.clone();
        let dh_ref = dh.clone();
        assert_eq!(
            out.factor_claims[0],
            df_ref.evaluate(&out.challenges).unwrap()
        );
        assert_eq!(
            out.factor_claims[1],
            dh_ref.evaluate(&out.challenges).unwrap()
        );
    }
}

#[cfg(test)]
mod eqwalk_tests {
    use super::*;

    /// The Gray-code walk must reproduce `eq(r, tobits(j))` for every j.
    #[test]
    fn eq_walk_matches_direct() {
        for k in [1usize, 2, 3, 5, 8] {
            let r: Vec<Goldilocks> = (1..=k as u64)
                .map(|i| Goldilocks::from_u64(i.wrapping_mul(1_000_000_007) + 13))
                .collect();
            let mut walk = EqWalk::new(&r);
            for j in 0..(1u64 << k) {
                let bits: Vec<bool> = (0..k).map(|b| (j >> (k - 1 - b)) & 1 == 1).collect();
                let want = eq_points(&r, &bits);
                assert_eq!(walk.weight(), want, "k={k} j={j}");
                if j + 1 < (1u64 << k) {
                    walk.advance();
                }
            }
        }
    }
}
