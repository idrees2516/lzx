//! **The prefix-suffix inner product protocol** — Appendix A of
//! ePrint 2025/611, the paper's new prover algorithm for
//!
//! ```text
//! S = Σ_{x∈{0,1}^n} ũ(x) · ã(x)
//! ```
//!
//! with `u` an arbitrary (dense or sparse) vector served by a stream and
//! `ã` **prefix-suffix structured**: at cutoff `c` (variables `y` = the
//! first `c`, MSB-side),
//!
//! ```text
//! ã(y, z) = Σ_{j=1..k} prefix_j(y) · suffix_j(z)
//! ```
//!
//! The protocol is the standard Boolean sum-check (round messages
//! bit-identical to the in-memory engine — verified in the tests); only
//! the prover's strategy changes. At `C = 2` (the paper's focus: "which
//! is sufficient to keep the prover space bounded by `O(√T)`", with
//! `c = n/2`):
//!
//! * **Stage 1** (rounds `0..c`): one streaming pass builds
//!   `Q_j[y] = Σ_z u[(y,z)]·suffix_j(z)` — `O(2^c)` per array — plus the
//!   prefix tables `P_j[y] = prefix_j(y)`; the stage's round messages
//!   equal those of the in-memory sum-check on `Σ_j P̂_j·Q̂_j` over the
//!   `2^c` cube (Expression 16/17's equivalence, which holds by
//!   eq-linearity of the binding).
//! * **Stage 2** (rounds `c..n`): one eq-weighted streaming pass
//!   materializes `u_bound[z] = Σ_y eq(r_y, y)·u[(y,z)]`, and the stage
//!   runs in memory on `u_bound · (Σ_j prefix_j(r_y)·suffix_j)(z)`.
//!
//! Space: `O(k·N^{1/2})` field elements — the paper's
//! `O(k·C·N^{1/C})` at `C = 2`; time: `O(k·N)` field operations plus two
//! passes over the stream (the paper's `C` passes).
//!
//! ## Structures (the paper's two applications, §A runtime analyses)
//!
//! * **M-evaluation (Twist)**: `ã = LT_f(r', ·)` decomposes at every
//!   cutoff as
//!   `LT_f(r', (y, z)) = LT_f(r'_y, y) · 1 + eq_f(r'_y, y) · LT_f(r'_z, z)`
//!   (k = 2 — "val(j) < val(r') iff the high half is smaller, or the
//!   halves agree and the low half is smaller").
//! * **pcnext-evaluation (Spartan)**: `ã = shift_f(r, ·)` (the no-wrap
//!   indicator `1[val(x)+1 = val(r)]`) decomposes as
//!   `eq_f(r_y, y)·shift_f(r_z, z) + shift_f(r_y, y)·[∏_z(1−r_zℓ)]·[∏_ℓ z_ℓ]`
//!   (k = 2 — the carry decomposition: either the low half increments
//!   with the high half agreeing, or the low half is all-ones, `r_z = 0`,
//!   and the high half increments).
//! * **eq** (the read-checking suffix): k = 1 trivially.

use crate::oracle::StreamOracle;
use crate::small_space::{interpolate_nodes, EqWalk};
use lattice_core::field_simd::{self, Sum8};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::{DenseMle, Goldilocks};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixSuffixError {
    Transcript(TranscriptError),
    RoundCheckFailed { round: usize },
    ClaimMismatch,
    BadShape { expected: usize, got: usize },
}

impl core::fmt::Display for PrefixSuffixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PrefixSuffixError::Transcript(e) => write!(f, "transcript error: {e}"),
            PrefixSuffixError::RoundCheckFailed { round } => {
                write!(f, "prefix-suffix round identity failed at round {round}")
            }
            PrefixSuffixError::ClaimMismatch => write!(f, "prefix-suffix claim mismatch"),
            PrefixSuffixError::BadShape { expected, got } => {
                write!(f, "prefix-suffix shape {got} != {expected}")
            }
        }
    }
}

/// The structured second operand `ã` of the inner product.
#[derive(Clone, Debug)]
pub enum Structure {
    /// `LT_f(r', ·)` — the M-evaluation sum-check's less-than factor.
    Lt { r: Vec<Goldilocks> },
    /// `shift_f(r, ·)` — the pcnext-evaluation kernel (no wrap).
    Shift { r: Vec<Goldilocks> },
    /// `eq_f(r, ·)` — the read-checking equality factor.
    Eq { r: Vec<Goldilocks> },
}

/// Affine evaluation helpers (Boolean-MLE forms, MSB-first bits).
fn eq_eval(r: &[Goldilocks], x: &[Goldilocks]) -> Goldilocks {
    let mut acc = Goldilocks::ONE;
    for (a, b) in r.iter().zip(x.iter()) {
        acc = acc.mul(
            &a.mul(b)
                .add(&Goldilocks::ONE.sub(a).mul(&Goldilocks::ONE.sub(b))),
        );
    }
    acc
}

/// `LT_f(r, x) = Σ_v (1−r_v)·x_v·Π_{u<v} eq_f(r_u, x_u)` — the standard
/// Boolean less-than MLE (`1[nat(r) < nat(x)]`).
pub fn lt_bool_eval(r: &[Goldilocks], x: &[Goldilocks]) -> Goldilocks {
    let n = r.len();
    let mut acc = Goldilocks::ZERO;
    let mut prefix = Goldilocks::ONE;
    for v in 0..n {
        let term = Goldilocks::ONE.sub(&r[v]).mul(&x[v]).mul(&prefix);
        acc = acc.add(&term);
        prefix = prefix.mul(&eq_eval(&r[v..v + 1], &x[v..v + 1]));
    }
    acc
}

/// `shift_f(r, x)` — the no-wrap `1[val(x)+1 = val(r)]` MLE:
///
/// ```text
/// Σ_v [r_v · Π_{u>v}(1−r_u)] · (1−x_v) · Π_{u>v} x_u · Π_{u<v} eq_f(r_u, x_u)
/// ```
///
/// The decisive position `v` requires `x_v = 0`, `x`'s bits below `v`
/// all ones (the carry chain), `r_v = 1`, **`r`'s bits below `v` all
/// zero** (the carry clears them), and agreement above — the
/// `Π_{u>v}(1−r_u)` factor is the r-side carry condition.
pub fn shift_bool_eval(r: &[Goldilocks], x: &[Goldilocks]) -> Goldilocks {
    let n = r.len();
    let mut x_suffix = vec![Goldilocks::ONE; n + 1]; // Π_{u>v} x_u
    let mut r_suffix = vec![Goldilocks::ONE; n + 1]; // Π_{u>v} (1−r_u)
    for v in (0..n).rev() {
        x_suffix[v] = x_suffix[v + 1].mul(&x[v]);
        r_suffix[v] = r_suffix[v + 1].mul(&Goldilocks::ONE.sub(&r[v]));
    }
    let mut acc = Goldilocks::ZERO;
    let mut prefix = Goldilocks::ONE;
    for v in 0..n {
        let term = r[v]
            .mul(&Goldilocks::ONE.sub(&x[v]))
            .mul(&x_suffix[v + 1])
            .mul(&r_suffix[v + 1])
            .mul(&prefix);
        acc = acc.add(&term);
        prefix = prefix.mul(&eq_eval(&r[v..v + 1], &x[v..v + 1]));
    }
    acc
}

impl Structure {
    /// The affine evaluation of `ã` at an arbitrary point.
    pub fn eval_affine(&self, point: &[Goldilocks]) -> Goldilocks {
        match self {
            Structure::Lt { r } => lt_bool_eval(r, point),
            Structure::Shift { r } => shift_bool_eval(r, point),
            Structure::Eq { r } => eq_eval(r, point),
        }
    }

    /// The k=2 prefix-suffix decomposition at cutoff `c` (variables
    /// `0..c` = `y`, the MSB side): returns `(a_j, b_j)` pairs as
    /// **truth tables** over the `y`- and `z`-cubes.
    ///
    /// * LT: `a_1 = LT(r_y)`, `b_1 = 1`; `a_2 = eq(r_y)`, `b_2 = LT(r_z)`.
    /// * Shift: `a_1 = eq(r_y)`, `b_1 = shift(r_z)`;
    ///   `a_2 = shift(r_y)·∏(1−r_z)`, `b_2 = all-ones(z) indicator`.
    /// * Eq: `a_1 = eq(r_y)`, `b_1 = eq(r_z)` (k=1; a zero second term).
    pub fn decompose_tables(
        &self,
        n: usize,
        c: usize,
    ) -> (Vec<Vec<Goldilocks>>, Vec<Vec<Goldilocks>>) {
        let r = match self {
            Structure::Lt { r } | Structure::Shift { r } | Structure::Eq { r } => r,
        };
        let y_len = 1usize << c;
        let z_len = 1usize << (n - c);
        let r_y = &r[..c.min(r.len())];
        let r_z = &r[c.min(r.len())..];
        let bits = |mut idx: usize, m: usize| -> Vec<Goldilocks> {
            (0..m)
                .map(|i| {
                    let b = Goldilocks::from_u64(((idx >> (m - 1 - i)) & 1) as u64);
                    let _ = &mut idx;
                    b
                })
                .collect()
        };
        match self {
            Structure::Lt { .. } => {
                // a_1 = LT(r_y, ·) over the y-cube; b_1 = 1 over z.
                let a1: Vec<Goldilocks> =
                    (0..y_len).map(|y| lt_bool_eval(r_y, &bits(y, c))).collect();
                let b1 = vec![Goldilocks::ONE; z_len];
                let a2: Vec<Goldilocks> = (0..y_len).map(|y| eq_eval(r_y, &bits(y, c))).collect();
                let b2: Vec<Goldilocks> = (0..z_len)
                    .map(|z| lt_bool_eval(r_z, &bits(z, n - c)))
                    .collect();
                (vec![a1, a2], vec![b1, b2])
            }
            Structure::Shift { .. } => {
                // a_1 = eq(r_y, ·); b_1 = shift(r_z, ·).
                let a1: Vec<Goldilocks> = (0..y_len).map(|y| eq_eval(r_y, &bits(y, c))).collect();
                let b1: Vec<Goldilocks> = (0..z_len)
                    .map(|z| shift_bool_eval(r_z, &bits(z, n - c)))
                    .collect();
                // a_2 = shift(r_y, ·) · ∏(1−r_z); b_2 = all-ones(z).
                let rz_scalar = r_z
                    .iter()
                    .fold(Goldilocks::ONE, |acc, rv| acc.mul(&Goldilocks::ONE.sub(rv)));
                let a2: Vec<Goldilocks> = (0..y_len)
                    .map(|y| shift_bool_eval(r_y, &bits(y, c)).mul(&rz_scalar))
                    .collect();
                let b2: Vec<Goldilocks> = (0..z_len)
                    .map(|z| {
                        if z == z_len - 1 {
                            Goldilocks::ONE
                        } else {
                            Goldilocks::ZERO
                        }
                    })
                    .collect();
                (vec![a1, a2], vec![b1, b2])
            }
            Structure::Eq { .. } => {
                let a1: Vec<Goldilocks> = (0..y_len).map(|y| eq_eval(r_y, &bits(y, c))).collect();
                let b1: Vec<Goldilocks> =
                    (0..z_len).map(|z| eq_eval(r_z, &bits(z, n - c))).collect();
                (
                    vec![a1, vec![Goldilocks::ZERO; y_len]],
                    vec![b1, vec![Goldilocks::ZERO; z_len]],
                )
            }
        }
    }
}

/// Prover output: standard Boolean sum-check rounds plus the terminal
/// claims (`ũ(r)` for the PCS layer; `ã(r)` is verifier-computable).
#[derive(Clone, Debug)]
pub struct PrefixSuffixOutput {
    pub rounds: Vec<Vec<Goldilocks>>,
    pub challenges: Vec<Goldilocks>,
    /// `u(r)` — the stream-side terminal claim (PCS-authenticated).
    pub u_claim: Goldilocks,
    /// `ã(r)` — the structure-side terminal value.
    pub a_claim: Goldilocks,
}

impl PrefixSuffixOutput {
    /// Total claim check: `u_claim · a_claim` (the PCS opens u at r).
    pub fn final_claim(&self) -> Goldilocks {
        self.u_claim.mul(&self.a_claim)
    }
}

/// Prove `Σ_{x} u(x)·ã(x) = claim` with the C=2 prefix-suffix protocol
/// (`c = n/2` — the `O(√N)`-space regime; `O(2)` stream passes).
///
/// `claim` may be `None`, in which case one extra pass computes it.
pub fn prove_prefix_suffix(
    stream: &mut dyn StreamOracle,
    structure: &Structure,
    n: usize,
    claim: Option<Goldilocks>,
    transcript: &mut Transcript,
) -> Result<PrefixSuffixOutput, PrefixSuffixError> {
    if n < 2 || n % 2 != 0 {
        return Err(PrefixSuffixError::BadShape {
            expected: 2,
            got: n,
        });
    }
    let c = n / 2;
    let z_bits = n - c;
    let y_len = 1usize << c;
    let z_len = 1usize << z_bits;
    let total = 1u64 << n;

    // ------------------------------------------------------------------
    // Pass 1 (optional): compute the claim Σ u·ã directly.
    // ------------------------------------------------------------------
    let claim = match claim {
        Some(v) => v,
        None => {
            stream.reset();
            // Materialize the b tables for the per-element ã evaluation…
            // cheaper: evaluate ã per element via the affine form on the
            // element's bits: O(n) per element (fine for a claim pass).
            let mut acc = Goldilocks::ZERO;
            for i in 0..total {
                let u = stream.next();
                let x: Vec<Goldilocks> = (0..n)
                    .map(|b| Goldilocks::from_u64((i >> (n - 1 - b)) & 1))
                    .collect();
                let a = structure.eval_affine(&x);
                acc = acc.add(&u.mul(&a));
            }
            acc
        }
    };

    // ------------------------------------------------------------------
    // Prefix/suffix tables (the O(√N) structure materialization).
    // ------------------------------------------------------------------
    let (a_tables, b_tables) = structure.decompose_tables(n, c);
    let k = 2usize;

    // ------------------------------------------------------------------
    // Pass 2: Q_j[y] = Σ_z u[(y,z)]·b_j(z) — one sequential sweep
    // (the y-blocks iterate slowly; z lookups hit the b tables).
    // ------------------------------------------------------------------
    let mut q_tables: Vec<Vec<Goldilocks>> =
        (0..k).map(|_| vec![Goldilocks::ZERO; y_len]).collect();
    {
        stream.reset();
        for i in 0..total {
            let u = stream.next();
            let y = (i >> z_bits) as usize;
            let z = (i & (z_len as u64 - 1)) as usize;
            for j in 0..k {
                let bz = b_tables[j][z];
                if !bz.is_zero() {
                    q_tables[j][y] = q_tables[j][y].add(&u.mul(&bz));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Stage 1: the in-memory linear-time sum-check on Σ_j P_j·Q_j over
    // the 2^c cube (Expression 16/17's equivalence).
    // ------------------------------------------------------------------
    let mut bound_a: Vec<Vec<Goldilocks>> = a_tables.clone();
    let mut bound_q: Vec<Vec<Goldilocks>> = q_tables.clone();
    let mut current_claim = claim;
    let mut rounds: Vec<Vec<Goldilocks>> = Vec::with_capacity(n);
    let mut challenges: Vec<Goldilocks> = Vec::with_capacity(n);
    let mut bound_r: Vec<Goldilocks> = Vec::with_capacity(n);

    for _round in 0..c {
        let evals = two_factor_round(&bound_a, &bound_q);
        let sum01 = evals[0].add(&evals[1]);
        if sum01 != current_claim {
            return Err(PrefixSuffixError::RoundCheckFailed {
                round: bound_r.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals)
            .map_err(PrefixSuffixError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(PrefixSuffixError::Transcript)?;
        current_claim = interpolate_nodes(&evals, &r);
        bound_r.push(r);
        challenges.push(r);
        for arr in bound_a.iter_mut().chain(bound_q.iter_mut()) {
            field_simd::bind_first_half_in_place(arr, r);
            let half = arr.len() / 2;
            arr.truncate(half);
        }
        rounds.push(evals);
    }
    // Stage-1 cross-check: Σ_j P_j·Q_j (now constants).
    let stage1_check: Goldilocks = (0..k)
        .map(|j| bound_a[j][0].mul(&bound_q[j][0]))
        .fold(Goldilocks::ZERO, |acc, v| acc.add(&v));
    if stage1_check != current_claim {
        return Err(PrefixSuffixError::ClaimMismatch);
    }

    // ------------------------------------------------------------------
    // Pass 3: u_bound[z] = Σ_y eq(r_y, y)·u[(y,z)] — the eq-weighted
    // materialization at the switch point (one sweep, Gray walk over y).
    // ------------------------------------------------------------------
    let r_y = bound_r.clone();
    let mut u_bound = vec![Goldilocks::ZERO; z_len];
    {
        stream.reset();
        let mut walk = EqWalk::new(&r_y);
        let y_block = z_len as u64;
        for y in 0..y_len as u64 {
            let w = walk.weight();
            for z in 0..y_block {
                let u = stream.next();
                if !w.is_zero() {
                    u_bound[z as usize] = u_bound[z as usize].add(&w.mul(&u));
                }
            }
            if y + 1 < y_len as u64 {
                walk.advance();
            }
        }
    }

    // ã_bound[z] = Σ_j a_j(r_y)·b_j(z).
    let a_at_ry: Vec<Goldilocks> = a_tables
        .iter()
        .map(|t| {
            let mle = DenseMle::new(t.clone()).unwrap_or(DenseMle::constant(Goldilocks::ZERO));
            mle.evaluate(&r_y).unwrap_or(Goldilocks::ZERO)
        })
        .collect();
    let mut a_bound = vec![Goldilocks::ZERO; z_len];
    for j in 0..k {
        if a_at_ry[j].is_zero() {
            continue;
        }
        for z in 0..z_len {
            a_bound[z] = a_bound[z].add(&a_at_ry[j].mul(&b_tables[j][z]));
        }
    }

    // ------------------------------------------------------------------
    // Stage 2: in-memory sum-check on u_bound·ã_bound over the z-cube.
    // ------------------------------------------------------------------
    let mut ub = u_bound;
    let mut ab = a_bound;
    for _round in 0..z_bits {
        let evals = two_factor_round(&[ub.clone()], &[ab.clone()]);
        let sum01 = evals[0].add(&evals[1]);
        if sum01 != current_claim {
            return Err(PrefixSuffixError::RoundCheckFailed {
                round: bound_r.len(),
            });
        }
        transcript
            .append_field_slice(b"sumcheck-round", &evals)
            .map_err(PrefixSuffixError::Transcript)?;
        let r = transcript
            .challenge_field(b"sumcheck-challenge")
            .map_err(PrefixSuffixError::Transcript)?;
        current_claim = interpolate_nodes(&evals, &r);
        bound_r.push(r);
        challenges.push(r);
        field_simd::bind_first_half_in_place(&mut ub, r);
        ub.truncate(ub.len() / 2);
        field_simd::bind_first_half_in_place(&mut ab, r);
        ab.truncate(ab.len() / 2);
        rounds.push(evals);
    }

    let u_claim = ub[0];
    let a_claim = ab[0];
    if u_claim.mul(&a_claim) != current_claim {
        return Err(PrefixSuffixError::ClaimMismatch);
    }

    Ok(PrefixSuffixOutput {
        rounds,
        challenges,
        u_claim,
        a_claim,
    })
}

/// Round polynomial of `Σ_j A_j·B_j` (degree 2) evaluated at nodes 0..=2,
/// with in-place-safe temporary binding.
fn two_factor_round(a: &[Vec<Goldilocks>], b: &[Vec<Goldilocks>]) -> Vec<Goldilocks> {
    let len = a[0].len();
    let half = len / 2;
    let mut evals = vec![Goldilocks::ZERO; 3];
    for (pi, t) in [0u64, 1, 2].iter().enumerate() {
        let tf = Goldilocks::from_u64(*t);
        // Bound values at X = t for each array.
        let bind = |arr: &Vec<Goldilocks>| -> Vec<Goldilocks> {
            match t {
                0 => arr[..half].to_vec(),
                1 => arr[half..].to_vec(),
                _ => {
                    let mut v = vec![Goldilocks::ZERO; half];
                    field_simd::bind_half_slices(&arr[..half], &arr[half..], tf, &mut v);
                    v
                }
            }
        };
        let bound_a: Vec<Vec<Goldilocks>> = a.iter().map(bind).collect();
        let bound_b: Vec<Vec<Goldilocks>> = b.iter().map(bind).collect();
        let mut acc = Sum8::new();
        let mut fslices: Vec<&[Goldilocks]> = Vec::with_capacity(8);
        for (av, bv) in bound_a.iter().zip(bound_b.iter()) {
            fslices.clear();
            fslices.push(av.as_slice());
            fslices.push(bv.as_slice());
            acc.accumulate_term(Goldilocks::ONE, &fslices);
        }
        evals[pi] = acc.finish();
    }
    evals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use lattice_sumcheck::virtual_poly::VirtualPolynomial;

    fn random_u(n: usize, seed: &[u8]) -> Vec<Goldilocks> {
        DenseMle::random(n, seed).evaluations
    }

    fn structure_point(_structure: &Structure, n: usize) -> Vec<Goldilocks> {
        (1..=n as u64)
            .map(|i| Goldilocks::from_u64(i.wrapping_mul(1_000_000_007) + 5))
            .collect()
    }

    /// The decisive equivalence: for each structure (LT / Shift / Eq),
    /// the prefix-suffix prover's round messages, challenges, and
    /// terminal claims match the in-memory Boolean engine run on the
    /// dense `u·ã` instance.
    #[test]
    fn matches_in_memory_engine_all_structures() {
        for n in [6usize, 8] {
            for (structure, tag) in [
                (
                    Structure::Lt {
                        r: (1..=n as u64)
                            .map(|i| Goldilocks::from_u64(i.wrapping_mul(997) + 3))
                            .collect(),
                    },
                    "LT",
                ),
                (
                    Structure::Shift {
                        r: (1..=n as u64)
                            .map(|i| Goldilocks::from_u64(i.wrapping_mul(1_234_567) + 11))
                            .collect(),
                    },
                    "shift",
                ),
                (
                    Structure::Eq {
                        r: (1..=n as u64)
                            .map(|i| Goldilocks::from_u64(i.wrapping_mul(555) + 7))
                            .collect(),
                    },
                    "eq",
                ),
            ] {
                let u = random_u(n, format!("u-{tag}").as_bytes());
                // Dense reference: a-table = ã's truth table.
                let a_table: Vec<Goldilocks> = (0..(1u64 << n))
                    .map(|i| {
                        let x: Vec<Goldilocks> = (0..n)
                            .map(|b| Goldilocks::from_u64((i >> (n - 1 - b)) & 1))
                            .collect();
                        structure.eval_affine(&x)
                    })
                    .collect();
                let claim: Goldilocks = (0..(1usize << n))
                    .map(|i| u[i].mul(&a_table[i]))
                    .fold(Goldilocks::ZERO, |acc, v| acc.add(&v));

                // In-memory reference via lattice-sumcheck.
                let du = DenseMle {
                    num_vars: n,
                    evaluations: u.clone(),
                };
                let da = DenseMle {
                    num_vars: n,
                    evaluations: a_table.clone(),
                };
                let mut vp = VirtualPolynomial::new(n);
                let iu = vp.add_factor(du.clone()).unwrap();
                let ia = vp.add_factor(da.clone()).unwrap();
                vp.add_term(Goldilocks::ONE, vec![iu, ia]).unwrap();
                let mut ts = Transcript::new_default(b"ps-seed");
                let reference = lattice_sumcheck::sumcheck::prove(&vp, claim, &mut ts).unwrap();

                // Prefix-suffix prover.
                let mut stream = OwnedOracle::new(u.clone());
                let mut ts2 = Transcript::new_default(b"ps-seed");
                let out =
                    prove_prefix_suffix(&mut stream, &structure, n, Some(claim), &mut ts2).unwrap();

                assert_eq!(out.rounds, reference.proof.rounds, "{tag} rounds n={n}");
                assert_eq!(
                    out.challenges, reference.challenges,
                    "{tag} challenges n={n}"
                );
                assert_eq!(
                    out.u_claim,
                    du.evaluate(&out.challenges).unwrap(),
                    "{tag} u(r)"
                );
                // a(r): the verifier recomputes it directly.
                assert_eq!(
                    out.a_claim,
                    structure.eval_affine(&out.challenges),
                    "{tag} a(r) n={n}"
                );
                assert_eq!(
                    out.final_claim(),
                    reference.final_claim,
                    "{tag} final n={n}"
                );
                let _ = structure_point(&structure, n);
            }
        }
    }

    /// Claim computation path (claim = None) matches the explicit one.
    #[test]
    fn claim_computation_path() {
        let n = 6;
        let structure = Structure::Lt {
            r: (1..=n as u64)
                .map(|i| Goldilocks::from_u64(i.wrapping_mul(31) + 1))
                .collect(),
        };
        let u = random_u(n, b"cc-u");
        let a_table: Vec<Goldilocks> = (0..(1u64 << n))
            .map(|i| {
                let x: Vec<Goldilocks> = (0..n)
                    .map(|b| Goldilocks::from_u64((i >> (n - 1 - b)) & 1))
                    .collect();
                structure.eval_affine(&x)
            })
            .collect();
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| u[i].mul(&a_table[i]))
            .fold(Goldilocks::ZERO, |acc, v| acc.add(&v));
        // The None path computes the same claim and the same proof as the
        // explicit-claim path; the terminal is the evaluation claim P(r)
        // (checked against the structure's own affine evaluation).
        let mut stream = OwnedOracle::new(u.clone());
        let mut ts = Transcript::new_default(b"cc-seed");
        let out_none = prove_prefix_suffix(&mut stream, &structure, n, None, &mut ts).unwrap();
        let mut stream2 = OwnedOracle::new(u);
        let mut ts2 = Transcript::new_default(b"cc-seed");
        let out_some =
            prove_prefix_suffix(&mut stream2, &structure, n, Some(claim), &mut ts2).unwrap();
        assert_eq!(out_none.rounds, out_some.rounds);
        assert_eq!(out_none.challenges, out_some.challenges);
        assert_eq!(out_none.final_claim(), out_some.final_claim());
        assert_eq!(
            out_none.a_claim,
            structure.eval_affine(&out_none.challenges)
        );
        assert_eq!(
            out_none.u_claim,
            DenseMle::new(random_u(n, b"cc-u"))
                .unwrap()
                .evaluate(&out_none.challenges)
                .unwrap()
        );
    }
}
