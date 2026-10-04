//! (Kernel loops use explicit indices by convention.)
#![allow(clippy::needless_range_loop)]
//! The sum-check protocol over rings (Appendix A.3 of ePrint 2026/471,
//! after [CCKP19]): `Σ_{x ∈ {0,1}^μ} P(x) = τ` for a multilinear
//! virtual polynomial `P = Σ_t c_t · ∏_j f_{t,j}` over the split ring,
//! with challenges sampled from the binary challenge space `C`
//! (Schwartz–Zippel for rings, Lemma 3.6, needs exactly this).
//!
//! Round structure: `μ` rounds; round `i` binds the *high* variable of
//! the remaining cube (the same fold order `RingD::mle_eval` uses), the
//! round message is the univariate polynomial
//!
//! ```text
//! q_i(X) = Σ_{rest} P(r_1..r_{i-1}, X, rest) ∈ R[X],  deg ≤ D_i,
//! ```
//!
//! transmitted as its `D_i + 1` ring-element coefficients (interpolated
//! from evaluations at the integer nodes `0..D_i`, which is legitimate
//! because the Vandermonde determinant `∏(s_j−s_i)` is a small nonzero
//! integer, invertible mod `q`). The verifier checks
//! `q_i(0) + q_i(1) = q_{i-1}(r_{i-1})` (`H = {0,1}`), samples
//! `r_i ← C` off the transcript, and after the last round obtains the
//! evaluation claim `P(r) = v` which it settles by querying the factor
//! MLEs at `r` through a caller-supplied resolver — the PIOP layer
//! answers from its oracle table, the compiled layer from Ajtai
//! openings.

use crate::ring_d::{Elem, RingD};
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone)]
pub enum RingFactor {
    /// An MLE over the μ-dimensional cube, given by its `2^μ` evaluations.
    Mle(Vec<Elem>),
    /// `EQ(x, y)` for a verifier point `y` — as its cube evaluations.
    Eq(Vec<Elem>),
    /// A constant.
    Const(Elem),
}

impl RingFactor {
    fn evals(&self) -> &[Elem] {
        match self {
            RingFactor::Mle(v) | RingFactor::Eq(v) => v,
            RingFactor::Const(_) => &[],
        }
    }

    fn is_const(&self) -> bool {
        matches!(self, RingFactor::Const(_))
    }
}

/// One product term `c · ∏_j f_j`.
#[derive(Debug, Clone)]
pub struct RingTerm {
    pub coeff: Elem,
    pub factors: Vec<RingFactor>,
}

/// The virtual polynomial plus its claimed cube-sum.
#[derive(Debug, Clone)]
pub struct RingVirtualPoly {
    pub num_vars: usize,
    pub claimed_sum: Elem,
    pub terms: Vec<RingTerm>,
}

impl RingVirtualPoly {
    /// The shape the verifier needs: no evaluations, just degrees and
    /// the public term coefficients (statement data).
    pub fn shape(&self) -> RingSumcheckShape {
        let terms = self
            .terms
            .iter()
            .map(|t| RingTermShape {
                coeff: t.coeff.clone(),
                num_factors: t.factors.iter().filter(|f| !f.is_const()).count(),
            })
            .collect();
        RingSumcheckShape {
            num_vars: self.num_vars,
            terms,
        }
    }

    /// Direct evaluation at a point (testing / final-claim checking).
    pub fn eval_at(&self, ring: &RingD, point: &[Elem]) -> Result<Elem, String> {
        if point.len() != self.num_vars {
            return Err(format!("point arity {} != {}", point.len(), self.num_vars));
        }
        let mut acc = ring.zero();
        for t in &self.terms {
            let mut prod = t.coeff.clone();
            for f in &t.factors {
                match f {
                    RingFactor::Const(c) => prod = ring.mul(&prod, c),
                    RingFactor::Mle(v) | RingFactor::Eq(v) => {
                        let e = ring.mle_eval(v, point).map_err(|e| format!("{e:?}"))?;
                        prod = ring.mul(&prod, &e);
                    }
                }
            }
            acc = ring.add(&acc, &prod);
        }
        Ok(acc)
    }
}

/// Verifier-side shape (factor counts per term, no secrets).
#[derive(Debug, Clone)]
pub struct RingSumcheckShape {
    pub num_vars: usize,
    pub terms: Vec<RingTermShape>,
}

#[derive(Debug, Clone)]
pub struct RingTermShape {
    /// Public term coefficient (part of the statement).
    pub coeff: Elem,
    pub num_factors: usize,
}

impl RingSumcheckShape {
    /// Maximum round degree (a product of `k` multilinear factors is
    /// degree `k` in the bound variable).
    pub fn max_degree(&self) -> usize {
        self.terms.iter().map(|t| t.num_factors).max().unwrap_or(0)
    }
}

/// A sum-check proof: per-round coefficient vectors of the round
/// polynomials, plus the (prover-recorded) challenge point.
#[derive(Debug, Clone)]
pub struct RingSumcheckProof {
    pub rounds: Vec<Vec<Elem>>,
    /// The challenge point in **little-endian variable order**
    /// (`point[j]` binds variable `x_j`) — identical to the
    /// `mle_eval` convention.
    pub point: Vec<Elem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SumcheckError {
    Shape(String),
    RoundDegree {
        round: usize,
        expected: usize,
        got: usize,
    },
    RoundCheck {
        round: usize,
    },
    FinalCheck,
    Resolver(String),
}

/// Prove the claimed cube-sum. `transcript` must already absorb the
/// statement (the caller's discipline).
pub fn prove_sumcheck(
    ring: &RingD,
    poly: &RingVirtualPoly,
    transcript: &mut Transcript,
) -> Result<RingSumcheckProof, SumcheckError> {
    let mu = poly.num_vars;
    let shape = poly.shape();
    let max_deg = shape.max_degree();
    // Bind each non-const factor, tracking its current cube evaluations.
    let mut bound: Vec<Vec<Option<Vec<Elem>>>> = poly
        .terms
        .iter()
        .map(|t| {
            t.factors
                .iter()
                .map(|f| {
                    if f.is_const() {
                        None
                    } else {
                        Some(f.evals().to_vec())
                    }
                })
                .collect()
        })
        .collect();
    let consts: Vec<Vec<Elem>> = poly
        .terms
        .iter()
        .map(|t| {
            t.factors
                .iter()
                .filter_map(|f| match f {
                    RingFactor::Const(c) => Some(c.clone()),
                    _ => None,
                })
                .collect()
        })
        .collect();

    let mut running_claim = poly.claimed_sum.clone();
    let mut rounds: Vec<Vec<Elem>> = Vec::with_capacity(mu);
    let mut challenges_binding_order: Vec<Elem> = Vec::with_capacity(mu);

    for round in 0..mu {
        let m = mu - round; // remaining variables (incl. the one being bound)
                            // The bound variable is x_{m-1} (the high bit of the remaining cube).
        let half = 1usize << (m - 1);
        let deg = max_deg;
        // Sample values at integer nodes 0..=deg
        let mut values_fixed: Vec<Elem> = Vec::with_capacity(deg + 1);
        for s in 0..=deg as u64 {
            let s_elem = ring.constant(s);
            let one_minus_s = ring.sub(&ring.one(), &s_elem);
            let mut total = ring.zero();
            for (ti, t) in poly.terms.iter().enumerate() {
                // elementwise product over the remaining cube
                let mut prod_vec: Option<Vec<Elem>> = None;
                if let Some(fs) = bound.get(ti) {
                    for fopt in fs.iter() {
                        let Some(f) = fopt else { continue };
                        let bound_f: Vec<Elem> = (0..half)
                            .map(|k| {
                                let lo = ring.mul(&f[k], &one_minus_s);
                                let hi = ring.mul(&f[k + half], &s_elem);
                                ring.add(&lo, &hi)
                            })
                            .collect();
                        prod_vec = Some(match prod_vec {
                            None => bound_f,
                            Some(pv) => pv
                                .into_iter()
                                .zip(bound_f)
                                .map(|(a, b)| ring.mul(&a, &b))
                                .collect(),
                        });
                    }
                }
                let term_sum = match prod_vec {
                    None => t.coeff.clone(),
                    Some(pv) => {
                        let mut acc = ring.zero();
                        for e in &pv {
                            acc = ring.add(&acc, e);
                        }
                        let mut with_coeff = ring.mul(&acc, &t.coeff);
                        for c in &consts[ti] {
                            with_coeff = ring.mul(&with_coeff, c);
                        }
                        with_coeff
                    }
                };
                total = ring.add(&total, &term_sum);
            }
            values_fixed.push(total);
        }
        // Interpolate the coefficients c_0..c_deg
        let coeffs = vandermonde_solve(ring, deg + 1, &values_fixed)?;
        // Verifier-side consistency the prover asserts: q(0)+q(1) = claim
        let q0 = eval_univariate(ring, &coeffs, &ring.zero());
        let q1 = eval_univariate(ring, &coeffs, &ring.one());
        let sum_halves = ring.add(&q0, &q1);
        if round == 0 {
            if sum_halves != running_claim {
                return Err(SumcheckError::RoundCheck { round });
            }
        } else {
            let prev = eval_univariate(
                ring,
                &rounds[round - 1],
                &challenges_binding_order[round - 1],
            );
            if sum_halves != prev {
                return Err(SumcheckError::RoundCheck { round });
            }
        }
        rounds.push(coeffs);
        // Sample the challenge for this round off the transcript.
        let ch = ring.sample_challenge(transcript, b"sc-chal");
        challenges_binding_order.push(ch.clone());
        running_claim = eval_univariate(ring, &rounds[round], &ch);
        // Bind every factor's high variable to ch.
        for ti in 0..bound.len() {
            for fopt in bound[ti].iter_mut() {
                let Some(f) = fopt else { continue };
                let one_minus = ring.sub(&ring.one(), &ch);
                let mut next = Vec::with_capacity(half);
                for k in 0..half {
                    let lo = ring.mul(&f[k], &one_minus);
                    let hi = ring.mul(&f[k + half], &ch);
                    next.push(ring.add(&lo, &hi));
                }
                *fopt = Some(next);
            }
        }
    }

    // point in little-endian order: binding order was x_{mu-1} .. x_0
    let mut point = challenges_binding_order.clone();
    point.reverse();
    Ok(RingSumcheckProof { rounds, point })
}

/// The factor-evaluation callback of [`verify_sumcheck`].
pub type Resolver<'a> = dyn FnMut(usize, usize, &[Elem]) -> Result<Elem, String> + 'a;

/// Verify a sum-check proof. `factor_eval` resolves factor MLE
/// evaluations at the final point (oracle table / committed openings).
/// Factor indexing: `(term_index, factor_index)` over non-const factors
/// in declaration order.
pub fn verify_sumcheck(
    ring: &RingD,
    shape: &RingSumcheckShape,
    claimed_sum: &Elem,
    proof: &RingSumcheckProof,
    transcript: &mut Transcript,
    factor_eval: &mut Resolver,
) -> Result<(), SumcheckError> {
    let mu = shape.num_vars;
    if proof.rounds.len() != mu {
        return Err(SumcheckError::Shape(format!(
            "rounds {} != vars {}",
            proof.rounds.len(),
            mu
        )));
    }
    let max_deg = shape.max_degree();
    let mut running = claimed_sum.clone();
    let mut challenges_binding_order: Vec<Elem> = Vec::with_capacity(mu);
    for (round, coeffs) in proof.rounds.iter().enumerate() {
        let expected_len = max_deg + 1;
        if coeffs.len() != expected_len {
            return Err(SumcheckError::RoundDegree {
                round,
                expected: expected_len,
                got: coeffs.len(),
            });
        }
        let q0 = eval_univariate(ring, coeffs, &ring.zero());
        let q1 = eval_univariate(ring, coeffs, &ring.one());
        let sum_halves = ring.add(&q0, &q1);
        if sum_halves != running {
            return Err(SumcheckError::RoundCheck { round });
        }
        let ch = ring.sample_challenge(transcript, b"sc-chal");
        challenges_binding_order.push(ch.clone());
        running = eval_univariate(ring, coeffs, &ch);
    }
    // Final check: running == P(point) via factor evaluations.
    let mut point = challenges_binding_order.clone();
    point.reverse();
    if point != proof.point {
        // The prover's recorded point must match the verifier's
        // transcript-derived challenges.
        return Err(SumcheckError::FinalCheck);
    }
    let mut acc = ring.zero();
    for (ti, t) in shape.terms.iter().enumerate() {
        let mut prod = t.coeff.clone();
        for fi in 0..t.num_factors {
            let e = factor_eval(ti, fi, &point).map_err(SumcheckError::Resolver)?;
            prod = ring.mul(&prod, &e);
        }
        acc = ring.add(&acc, &prod);
    }
    if acc != running {
        return Err(SumcheckError::FinalCheck);
    }
    Ok(())
}

/// Evaluate `Σ c_k · x^k` at a ring point `x`.
fn eval_univariate(ring: &RingD, coeffs: &[Elem], x: &Elem) -> Elem {
    let mut acc = ring.zero();
    let mut power = ring.one();
    for c in coeffs {
        let term = ring.mul(c, &power);
        acc = ring.add(&acc, &term);
        power = ring.mul(&power, x);
    }
    acc
}

/// Solve the `(D+1)×(D+1)` Vandermonde system `V·c = v` where
/// `V[s][k] = s^k` over `Z_q` (the integer nodes `s = 0..D`); the
/// determinant is `∏_{s>j}(s−j)`, a nonzero integer `< q`, so `V` is
/// invertible mod `q`. Returns the coefficient vector `c` (ring
/// elements scaled by integer matrix entries).
fn vandermonde_solve(ring: &RingD, n: usize, values: &[Elem]) -> Result<Vec<Elem>, SumcheckError> {
    if values.len() != n {
        return Err(SumcheckError::Shape("value arity mismatch".into()));
    }
    let q = ring.q;
    // Build V and augment with values as an extra column of RING elems —
    // we invert V over Z_q, then scale.
    let mut mat: Vec<Vec<u64>> = vec![vec![0u64; n]; n];
    for s in 0..n {
        for k in 0..n {
            mat[s][k] = mod_pow_small(s as u64, k as u64, q);
        }
    }
    // Gauss-Jordan inverse over Z_q
    let mut inv: Vec<Vec<u64>> = (0..n)
        .map(|i| {
            let mut row = vec![0u64; n];
            row[i] = 1;
            row
        })
        .collect();
    for col in 0..n {
        // pivot
        let mut piv = col;
        while piv < n && mat[piv][col] % q == 0 {
            piv += 1;
        }
        if piv == n {
            return Err(SumcheckError::Shape("vandermonde singular".into()));
        }
        mat.swap(col, piv);
        inv.swap(col, piv);
        let pv = mat[col][col] % q;
        let pv_inv = mod_pow_small(pv, q - 2, q);
        for k in 0..n {
            mat[col][k] = (mat[col][k] * pv_inv) % q;
            inv[col][k] = (inv[col][k] * pv_inv) % q;
        }
        for r in 0..n {
            if r != col && mat[r][col] % q != 0 {
                let f = mat[r][col];
                for k in 0..n {
                    mat[r][k] = (mat[r][k] + q - (f * mat[col][k]) % q) % q;
                    inv[r][k] = (inv[r][k] + q - (f * inv[col][k]) % q) % q;
                }
            }
        }
    }
    // c_k = Σ_s inv[k][s] · value_s
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut acc = ring.zero();
        for s in 0..n {
            let scale = inv[k][s] % q;
            if scale == 0 {
                continue;
            }
            let scaled = ring.scale(&values[s], scale);
            acc = ring.add(&acc, &scaled);
        }
        out.push(acc);
    }
    Ok(out)
}

fn mod_pow_small(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        base = (base as u128 * base as u128 % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_d::RingD;

    fn ring() -> RingD {
        RingD::new(8).ok().unwrap()
    }

    #[test]
    fn sumcheck_single_factor() {
        let r = ring();
        let mu = 4;
        let evals: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("f{i}").as_bytes()))
            .collect();
        let mut sum = r.zero();
        for e in &evals {
            sum = r.add(&sum, e);
        }
        let poly = RingVirtualPoly {
            num_vars: mu,
            claimed_sum: sum,
            terms: vec![RingTerm {
                coeff: r.one(),
                factors: vec![RingFactor::Mle(evals.clone())],
            }],
        };
        let mut tr = Transcript::new_default(b"sc1");
        let proof = prove_sumcheck(&r, &poly, &mut tr).ok().unwrap();
        let mut tr = Transcript::new_default(b"sc1");
        let shape = poly.shape();
        let claimed = poly.claimed_sum.clone();
        let evals_ref = &evals;
        let r_ref = &r;
        verify_sumcheck(&r, &shape, &claimed, &proof, &mut tr, &mut |ti, fi, pt| {
            assert_eq!((ti, fi), (0, 0));
            r_ref.mle_eval(evals_ref, pt).map_err(|e| format!("{e:?}"))
        })
        .ok()
        .unwrap();
    }

    #[test]
    fn sumcheck_product_of_factors() {
        let r = ring();
        let mu = 3;
        let a: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("a{i}").as_bytes()))
            .collect();
        let b: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("b{i}").as_bytes()))
            .collect();
        let c: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("c{i}").as_bytes()))
            .collect();
        // claimed sum of a*b + c (degree-2 and degree-1 terms)
        let mut sum = r.zero();
        for i in 0..(1 << mu) {
            let ab = r.mul(&a[i], &b[i]);
            sum = r.add(&sum, &r.add(&ab, &c[i]));
        }
        let poly = RingVirtualPoly {
            num_vars: mu,
            claimed_sum: sum,
            terms: vec![
                RingTerm {
                    coeff: r.one(),
                    factors: vec![RingFactor::Mle(a.clone()), RingFactor::Mle(b.clone())],
                },
                RingTerm {
                    coeff: r.one(),
                    factors: vec![RingFactor::Mle(c.clone())],
                },
            ],
        };
        let mut tr = Transcript::new_default(b"sc2");
        let proof = prove_sumcheck(&r, &poly, &mut tr).ok().unwrap();
        // tamper: flip a round coefficient
        let mut bad = proof.clone();
        if let Some(first) = bad.rounds.first_mut() {
            let flipped = r.add(&first[0], &r.one());
            first[0] = flipped;
        }
        let mut tr = Transcript::new_default(b"sc2");
        let shape = poly.shape();
        let claimed = poly.claimed_sum.clone();
        let res = verify_sumcheck(&r, &shape, &claimed, &bad, &mut tr, &mut |_, _, _| {
            Err("no resolver needed for rejection".into())
        });
        assert!(res.is_err(), "tampered round must fail");
        // honest verify
        let mut tr = Transcript::new_default(b"sc2");
        let (ra, rb, rc, rref) = (&a, &b, &c, &r);
        verify_sumcheck(&r, &shape, &claimed, &proof, &mut tr, &mut |ti, fi, pt| {
            let v = match (ti, fi) {
                (0, 0) => ra,
                (0, 1) => rb,
                (1, 0) => rc,
                _ => return Err("bad index".into()),
            };
            rref.mle_eval(v, pt).map_err(|e| format!("{e:?}"))
        })
        .ok()
        .unwrap();
    }

    #[test]
    fn sumcheck_wrong_claim_rejected() {
        let r = ring();
        let mu = 3;
        let a: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("w{i}").as_bytes()))
            .collect();
        let mut sum = r.zero();
        for e in &a {
            sum = r.add(&sum, e);
        }
        let wrong = r.add(&sum, &r.one());
        let poly = RingVirtualPoly {
            num_vars: mu,
            claimed_sum: wrong,
            terms: vec![RingTerm {
                coeff: r.one(),
                factors: vec![RingFactor::Mle(a.clone())],
            }],
        };
        let mut tr = Transcript::new_default(b"sc3");
        // The prover itself must detect the inconsistent claim at round 0.
        assert!(prove_sumcheck(&r, &poly, &mut tr).is_err());
    }

    #[test]
    fn sumcheck_with_eq_factor_zerocheck() {
        // Σ EQ(x, y)·(f(x)·g(x) − h(x)) = 0 style: zero-check flavor.
        let r = ring();
        let mu = 3;
        let mut tr = Transcript::new_default(b"zc");
        let y: Vec<Elem> = (0..mu)
            .map(|i| r.sample_challenge(&mut tr, format!("y{i}").as_bytes()))
            .collect();
        let eq = r.eq_row(&y);
        let f: Vec<Elem> = (0..(1 << mu))
            .map(|i| r.random(format!("zf{i}").as_bytes()))
            .collect();
        // h = f elementwise (so the zero-check passes)
        let h = f.clone();
        let poly = RingVirtualPoly {
            num_vars: mu,
            claimed_sum: r.zero(),
            terms: vec![
                RingTerm {
                    coeff: r.one(),
                    factors: vec![RingFactor::Eq(eq.clone()), RingFactor::Mle(f.clone())],
                },
                RingTerm {
                    coeff: r.neg(&r.one()),
                    factors: vec![RingFactor::Eq(eq.clone()), RingFactor::Mle(h.clone())],
                },
            ],
        };
        let mut tr2 = Transcript::new_default(b"sc4");
        let proof = prove_sumcheck(&r, &poly, &mut tr2).ok().unwrap();
        let mut tr2 = Transcript::new_default(b"sc4");
        let shape = poly.shape();
        let zero = r.zero();
        let (eqr, fr, hr, rref) = (&eq, &f, &h, &r);
        verify_sumcheck(&r, &shape, &zero, &proof, &mut tr2, &mut |ti, fi, pt| {
            // (0,0)=eq for term0, (0,1)=f; (1,0)=eq, (1,1)=h
            let v = match (ti, fi) {
                (0, 0) | (1, 0) => eqr,
                (0, 1) => fr,
                (1, 1) => hr,
                _ => return Err("bad index".into()),
            };
            rref.mle_eval(v, pt).map_err(|e| format!("{e:?}"))
        })
        .ok()
        .unwrap();
    }
}
