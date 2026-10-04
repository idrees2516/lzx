//! The batched sum-check machinery for ePrint 2026/538's protocols — both
//! polynomial representations:
//!
//! * **Multivariate (`ν = log n`)**: rounds over the boolean cube with
//!   per-round degree `D = dl + dr` (products of MLEs), vector-valued
//!   statements batched with shared challenges, prover round messages
//!   interpolated from `D+2` node evaluations of a black-box closure —
//!   the same engine pattern as `lattice-pcd`'s masked sum-check, minus
//!   the ZK masking (the holography paper's protocols are not ZK).
//! * **Univariate (`ν = 1`)**: the domain sum-check over `H` via the
//!   paper's `h₁/h₂` decomposition for degree-`> n` polynomials (§2's
//!   Figures 2–5):
//!   ```text
//!   g(X) = s/n + X·h₁(X) + u_H(X)·h₂(X),
//!   h₁ ∈ F[X]^{≤ n−2},  h₂ ∈ F[X]^{≤ deg(g) − n},
//!   verifier: s/n + β·h₁(β) + u_H(β)·h₂(β) = g(β)   (checked against the
//!   Evals-derived g(β) in the protocols' decision phases).
//!   ```
//!   The prover builds `g`'s coefficient vector (from the component
//!   polynomials' coefficients), reduces modulo `u_H`, and splits.

use crate::Fp256;
use lattice_core::transcript::Transcript;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScError {
    Shape(&'static str),
    Transcript(lattice_core::transcript::TranscriptError),
    RoundConsistency { round: usize, coord: usize },
}

impl core::fmt::Display for ScError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScError::Shape(s) => write!(f, "sumcheck shape: {s}"),
            ScError::Transcript(e) => write!(f, "transcript: {e}"),
            ScError::RoundConsistency { round, coord } => {
                write!(f, "round {round} inconsistent at coordinate {coord}")
            }
        }
    }
}

impl From<lattice_core::transcript::TranscriptError> for ScError {
    fn from(e: lattice_core::transcript::TranscriptError) -> Self {
        ScError::Transcript(e)
    }
}

// ---------------------------------------------------------------------------
// Shared univariate polynomial arithmetic (coefficient vectors, ascending)
// ---------------------------------------------------------------------------

/// Ascending-coefficient evaluation.
pub fn uni_eval(coeffs: &[Fp256], x: &Fp256) -> Fp256 {
    let mut acc = Fp256::ZERO;
    for c in coeffs.iter().rev() {
        acc = acc.mul(x).add(c);
    }
    acc
}

/// Polynomial multiplication (schoolbook).
pub fn uni_mul(a: &[Fp256], b: &[Fp256]) -> Vec<Fp256> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Fp256::ZERO; a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            out[i + j] = out[i + j].add(&ai.mul(bj));
        }
    }
    out
}

/// Polynomial addition.
pub fn uni_add(a: &[Fp256], b: &[Fp256]) -> Vec<Fp256> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = a.get(i).copied().unwrap_or(Fp256::ZERO);
        let bi = b.get(i).copied().unwrap_or(Fp256::ZERO);
        out.push(ai.add(&bi));
    }
    out
}

/// Polynomial subtraction `a − b`.
pub fn uni_sub(a: &[Fp256], b: &[Fp256]) -> Vec<Fp256> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = a.get(i).copied().unwrap_or(Fp256::ZERO);
        let bi = b.get(i).copied().unwrap_or(Fp256::ZERO);
        out.push(ai.sub(&bi));
    }
    out
}

/// `u_H(X) = Xⁿ − 1` as coefficients.
pub fn vanish_poly(n: usize) -> Vec<Fp256> {
    let mut out = vec![Fp256::ZERO; n + 1];
    out[0] = Fp256::from_canonical_u64(1).neg();
    out[n] = Fp256::from_canonical_u64(1);
    out
}

/// Long division `(q, r) = a / b` with `deg r < deg b`.
pub fn uni_divmod(a: &[Fp256], b: &[Fp256]) -> (Vec<Fp256>, Vec<Fp256>) {
    let db = b.len().saturating_sub(1);
    if b.iter().all(|v| v.is_zero()) || a.len().saturating_sub(1) < db {
        return (Vec::new(), a.to_vec());
    }
    let blc_inv = match b[db].inverse() {
        Some(v) => v,
        None => return (Vec::new(), a.to_vec()),
    };
    let mut rem = a.to_vec();
    let mut quot = vec![Fp256::ZERO; a.len() - db];
    while rem.len() >= b.len() {
        let dr = rem.len() - 1;
        if dr < db {
            break;
        }
        let factor = rem[dr].mul(&blc_inv);
        if factor.is_zero() {
            rem.pop();
            continue;
        }
        quot[dr - db] = factor;
        for (i, bi) in b.iter().enumerate() {
            let sub = factor.mul(bi);
            let idx = dr - db + i;
            if idx < rem.len() {
                rem[idx] = rem[idx].sub(&sub);
            }
        }
        rem.pop();
    }
    while rem.last().map(|v| v.is_zero()).unwrap_or(false) {
        rem.pop();
    }
    while quot.last().map(|v| v.is_zero()).unwrap_or(false) {
        quot.pop();
    }
    (quot, rem)
}

// ---------------------------------------------------------------------------
// The univariate domain sum-check (the h₁/h₂ decomposition)
// ---------------------------------------------------------------------------

/// The univariate sum-check proof: `(h₁, h₂)` with
/// `g(X) = s/n + X·h₁(X) + u_H(X)·h₂(X)`.
#[derive(Clone, Debug)]
pub struct UniSumcheckProof {
    pub h1: Vec<Fp256>,
    pub h2: Vec<Fp256>,
}

/// Prove `Σ_{h∈H} g(h) = s` from `g`'s coefficient vector.
pub fn uni_prove(n: usize, g_coeffs: &[Fp256], s: &Fp256) -> UniSumcheckProof {
    // a := g mod u_H (degree < n); Σ_{h∈H} a(h) = s.
    let uh = vanish_poly(n);
    let (_, mut a) = uni_divmod(g_coeffs, &uh);
    a.resize(n, Fp256::ZERO);
    // Σ_{h∈H} a(h) = n·a₀ (all other powers vanish over H).
    // h₁ := (a − s/n) / X.
    let inv_n = Fp256::from_canonical_u64(n as u64)
        .inverse()
        .unwrap_or(Fp256::ZERO);
    let s_over_n = s.mul(&inv_n);
    let mut shifted = a.clone();
    shifted[0] = shifted[0].sub(&s_over_n);
    // Divide by X: shift down (the constant must vanish — it does by the
    // sum identity).
    let h1: Vec<Fp256> = if shifted[0].is_zero() {
        shifted[1..].to_vec()
    } else {
        // Numerically impossible for the honest prover; emit garbage that
        // the verifier will reject.
        shifted
    };
    // h₂ := (g − a) / u_H.
    let g_minus_a = uni_sub(g_coeffs, &a);
    let (h2, _) = uni_divmod(&g_minus_a, &uh);
    UniSumcheckProof { h1, h2 }
}

/// Verify at a random `β` against the Evals-derived `g(β)`:
/// `s/n + β·h₁(β) + u_H(β)·h₂(β) == g(β)`, with the degree bounds
/// `deg h₁ ≤ n−2` and `deg h₂ ≤ expected_deg − n`.
pub fn uni_verify(
    n: usize,
    proof: &UniSumcheckProof,
    s: &Fp256,
    beta: &Fp256,
    g_at_beta: &Fp256,
    expected_deg: usize,
) -> bool {
    if proof.h1.len() > n.saturating_sub(1) {
        return false;
    }
    if proof.h2.len() + n > expected_deg + 1 {
        return false;
    }
    let inv_n = Fp256::from_canonical_u64(n as u64)
        .inverse()
        .unwrap_or(Fp256::ZERO);
    let s_over_n = s.mul(&inv_n);
    let h1b = uni_eval(&proof.h1, beta);
    let uh_b = crate::poly::vanish_eval(n, beta);
    let h2b = uni_eval(&proof.h2, beta);
    let rhs = s_over_n.add(&beta.mul(&h1b)).add(&uh_b.mul(&h2b));
    rhs == *g_at_beta
}

// ---------------------------------------------------------------------------
// The multivariate batched sum-check
// ---------------------------------------------------------------------------

/// A multivariate sum-check proof: per round, per coordinate, ascending
/// univariate coefficients (length degree + 2).
#[derive(Clone, Debug)]
pub struct MvSumcheckProof {
    pub rounds: Vec<Vec<Vec<Fp256>>>,
}

#[derive(Clone, Debug)]
pub struct MvSumcheckOutput {
    pub point: Vec<Fp256>,
    /// The final claimed evaluations per coordinate.
    pub final_evals: Vec<Fp256>,
}

/// Prove `Σ_{b∈{0,1}^ν} g_c(b) = s_c` for each coordinate through a
/// black-box evaluation closure; per-round degree bound `degree`.
pub fn mv_prove(
    num_vars: usize,
    degree: usize,
    eval: &dyn Fn(&[Fp256]) -> Vec<Fp256>,
    claims: &[Fp256],
    transcript: &mut Transcript,
) -> Result<(MvSumcheckProof, MvSumcheckOutput), ScError> {
    let node_count = degree + 2;
    let coords = claims.len();
    lattice_pcd::util::absorb_fp_slice(transcript, b"mvsc-claims", claims)?;
    let mut rounds: Vec<Vec<Vec<Fp256>>> = Vec::with_capacity(num_vars);
    let mut point: Vec<Fp256> = Vec::with_capacity(num_vars);
    let mut prev = claims.to_vec();
    for k in 0..num_vars {
        let tail_len = num_vars - k - 1;
        let tail_count = 1usize << tail_len;
        let mut node_sums = vec![vec![Fp256::ZERO; coords]; node_count];
        let mut pt = vec![Fp256::ZERO; num_vars];
        for t in 0..tail_count {
            for i in 0..tail_len {
                pt[k + 1 + i] = Fp256::from_canonical_u64(((t >> i) & 1) as u64);
            }
            for y in 0..node_count {
                let mut p = pt.clone();
                p[..k].copy_from_slice(&point[..k]);
                p[k] = Fp256::from_canonical_u64(y as u64);
                let vals = eval(&p);
                for (c, v) in vals.iter().enumerate() {
                    node_sums[y][c] = node_sums[y][c].add(v);
                }
            }
        }
        let mut msgs: Vec<Vec<Fp256>> = Vec::with_capacity(coords);
        for c in 0..coords {
            let ys: Vec<Fp256> = node_sums.iter().map(|ns| ns[c]).collect();
            msgs.push(interpolate_int_nodes(&ys));
        }
        // Prover-side sanity.
        let zero = Fp256::ZERO;
        let one = Fp256::from_canonical_u64(1);
        for (c, poly) in msgs.iter().enumerate() {
            let lhs = uni_eval(poly, &zero).add(&uni_eval(poly, &one));
            if lhs != prev[c] {
                return Err(ScError::RoundConsistency { round: k, coord: c });
            }
        }
        for poly in &msgs {
            for coeff in poly {
                transcript.append_message(b"mvsc-msg", &coeff.from_mont().canon_bytes())?;
            }
        }
        let ch = lattice_pcd::util::challenge_fp(transcript, b"mvsc-challenge")?;
        point.push(ch);
        prev = msgs.iter().map(|poly| uni_eval(poly, &ch)).collect();
        rounds.push(msgs);
    }
    Ok((
        MvSumcheckProof { rounds },
        MvSumcheckOutput {
            point,
            final_evals: prev,
        },
    ))
}

/// Verify the multivariate sum-check; returns the final point + claims.
pub fn mv_verify(
    num_vars: usize,
    degree: usize,
    claims: &[Fp256],
    proof: &MvSumcheckProof,
    transcript: &mut Transcript,
) -> Result<MvSumcheckOutput, ScError> {
    let node_count = degree + 2;
    let coords = claims.len();
    lattice_pcd::util::absorb_fp_slice(transcript, b"mvsc-claims", claims)?;
    if proof.rounds.len() != num_vars {
        return Err(ScError::Shape("round count"));
    }
    let mut prev = claims.to_vec();
    let mut point = Vec::with_capacity(num_vars);
    let zero = Fp256::ZERO;
    let one = Fp256::from_canonical_u64(1);
    for (k, msgs) in proof.rounds.iter().enumerate() {
        if msgs.len() != coords {
            return Err(ScError::Shape("coordinate count"));
        }
        for poly in msgs {
            if poly.len() > node_count {
                return Err(ScError::Shape("degree bound"));
            }
        }
        for (c, poly) in msgs.iter().enumerate() {
            let lhs = uni_eval(poly, &zero).add(&uni_eval(poly, &one));
            if lhs != prev[c] {
                return Err(ScError::RoundConsistency { round: k, coord: c });
            }
        }
        for poly in msgs {
            for coeff in poly {
                transcript.append_message(b"mvsc-msg", &coeff.from_mont().canon_bytes())?;
            }
        }
        let ch = lattice_pcd::util::challenge_fp(transcript, b"mvsc-challenge")?;
        point.push(ch);
        prev = msgs.iter().map(|poly| uni_eval(poly, &ch)).collect();
    }
    Ok(MvSumcheckOutput {
        point,
        final_evals: prev,
    })
}

/// Interpolate through integer nodes `0..y.len()−1` (ascending
/// coefficients) — shared with `lattice-pcd`'s engine, restated locally to
/// keep the crate self-contained.
pub fn interpolate_int_nodes(y: &[Fp256]) -> Vec<Fp256> {
    let d = y.len() - 1;
    let mut a = vec![vec![Fp256::ZERO; d + 2]; d + 1];
    for (i, row) in a.iter_mut().enumerate().take(d + 1) {
        let xi = Fp256::from_canonical_u64(i as u64);
        let mut p = Fp256::from_canonical_u64(1);
        for j in 0..=d {
            row[j] = p;
            p = p.mul(&xi);
        }
        row[d + 1] = y[i];
    }
    for col in 0..=d {
        let mut piv = col;
        while piv <= d && a[piv][col].is_zero() {
            piv += 1;
        }
        if piv > d {
            continue;
        }
        a.swap(col, piv);
        let inv = match a[col][col].inverse() {
            Some(v) => v,
            None => continue,
        };
        for j in col..=d + 1 {
            let t = a[col][j].mul(&inv);
            a[col][j] = t;
        }
        for r in 0..=d {
            if r != col && !a[r][col].is_zero() {
                let factor = a[r][col];
                for j in col..=d + 1 {
                    let t = a[r][j].sub(&factor.mul(&a[col][j]));
                    a[r][j] = t;
                }
            }
        }
    }
    (0..=d).map(|i| a[i][d + 1]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: u64) -> Fp256 {
        Fp256::from_canonical_u64(v)
    }

    #[test]
    fn uni_arithmetic() {
        let a = vec![fr(1), fr(2), fr(3)]; // 1 + 2x + 3x²
        let b = vec![fr(4), fr(5)]; // 4 + 5x
        let p = uni_mul(&a, &b); // 4 + 13x + 22x² + 15x³
        assert_eq!(p, vec![fr(4), fr(13), fr(22), fr(15)]);
        // Divmod by (x − 1):
        let d = vec![fr(1).neg(), fr(1)];
        let (q, r) = uni_divmod(&p, &d);
        // p(1) = 4+13+22+15 = 54 → remainder 54.
        assert_eq!(r, vec![fr(54)]);
        // The quotient satisfies q·(x−1) + 54 = p as polynomials:
        let recon = uni_add(&uni_mul(&q, &d), &r);
        assert_eq!(recon, p);
        // u_H for n=4: x⁴ − 1.
        let uh = vanish_poly(4);
        assert_eq!(uh, vec![fr(1).neg(), fr(0), fr(0), fr(0), fr(1)]);
    }

    #[test]
    fn uni_sumcheck_roundtrip() {
        // g = (1 + 2x + 3x²)·(4 + 5x) over H with n = 4: Σ g(h) = n·g₀mod
        let n = 4;
        let g = uni_mul(&[fr(1), fr(2), fr(3)], &[fr(4), fr(5)]);
        // Σ_{h∈H} g(h): compute directly.
        let mut s = Fp256::ZERO;
        for i in 0..n {
            let h = crate::poly::domain_point(i, n);
            s = s.add(&uni_eval(&g, &h));
        }
        let proof = uni_prove(n, &g, &s);
        let beta = fr(0x1234_5678u64);
        let g_beta = uni_eval(&g, &beta);
        assert!(uni_verify(n, &proof, &s, &beta, &g_beta, g.len() - 1));
        // Tampered proof → reject.
        let mut bad = proof.clone();
        bad.h1[0] = bad.h1[0].add(&fr(1));
        assert!(!uni_verify(n, &bad, &s, &beta, &g_beta, g.len() - 1));
        // Wrong s → reject.
        assert!(!uni_verify(
            n,
            &proof,
            &s.add(&fr(1)),
            &beta,
            &g_beta,
            g.len() - 1
        ));
    }

    #[test]
    fn mv_sumcheck_roundtrip_and_tamper() {
        // g_c(X) = (X₀² − X₀)·w_c + multilinear f_c — vanishing structure
        // not required here; any polynomial works. Use degree-3 products.
        let num_vars = 3;
        let degree = 3;
        let w = [fr(5), fr(7)];
        let f: [Vec<Fp256>; 2] = [
            vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6), fr(7), fr(8)],
            vec![fr(8), fr(7), fr(6), fr(5), fr(4), fr(3), fr(2), fr(1)],
        ];
        let eval = |pt: &[Fp256]| -> Vec<Fp256> {
            let g0 = pt[0].mul(&pt[0]).sub(&pt[0]);
            vec![
                g0.mul(&w[0]).add(&crate::poly::mle_eval(&f[0], pt)),
                g0.mul(&w[1]).add(&crate::poly::mle_eval(&f[1], pt)),
            ]
        };
        // Claims: Σ over the cube.
        let mut claims = vec![Fp256::ZERO; 2];
        for b in 0..8usize {
            let pt: Vec<Fp256> = (0..3).map(|k| fr(((b >> k) & 1) as u64)).collect();
            let v = eval(&pt);
            for c in 0..2 {
                claims[c] = claims[c].add(&v[c]);
            }
        }
        let mut t1 = Transcript::new_default(b"mv");
        let (proof, out) = mv_prove(num_vars, degree, &eval, &claims, &mut t1)
            .ok()
            .unwrap();
        let mut t2 = Transcript::new_default(b"mv");
        let vout = mv_verify(num_vars, degree, &claims, &proof, &mut t2)
            .ok()
            .unwrap();
        assert_eq!(vout.point, out.point);
        assert_eq!(vout.final_evals, out.final_evals);
        // Final evals match direct evaluation.
        let direct = eval(&out.point);
        assert_eq!(out.final_evals, direct);
        // Tamper → reject.
        let mut bad = proof.clone();
        bad.rounds[0][0][1] = bad.rounds[0][0][1].add(&fr(1));
        let mut t3 = Transcript::new_default(b"mv");
        assert!(mv_verify(num_vars, degree, &claims, &bad, &mut t3).is_err());
    }
}
