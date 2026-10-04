//! RoKoko 8 — the PCS front end (§2.3 "Applications"): a polynomial
//! evaluation claim `p(v) = t` is a linear constraint
//! `ℓ^T·W·r = t mod q` on the committed coefficient matrix `W` —
//! already a `Ξ^lin_COM` instance — so the RoKoko stack (commit →
//! one refinement round → terminal opening) IS a polynomial commitment
//! scheme.
//!
//! * `commit(coeffs)` — pack the polynomial's coefficients into the
//!   witness matrix `W ∈ R^{m_w × r}` (column-major over the `r`
//!   columns) and COM-commit; the public statement carries the linear
//!   functional `ℓ` = the point's eq weights (the multilinear
//!   evaluation `p(v) = Σ_b eq(v, b)·coeffs[b]` IS a linear form on
//!   the coefficients).
//! * `prove_eval(v, t)` — build the `Ξ^lin` instance (the eq-weight
//!   left vector `ℓ`, the right selector `r`, the target `t`) and run
//!   the round driver (`rokoko_prove`).
//! * `verify_eval` — `rokoko_verify` + the eq-weight recomputation.
//!
//! LZX realization notes (kernel scale): the evaluation is multilinear
//! over the coefficient hypercube (the univariate case wraps the same
//! way the HyperWolf module does — recorded as the follow-up); the
//! point's eq weights are embedded as ring constants exactly as the
//! protocol's `ℓ` vectors are.

use crate::com::{com_commit, ComKey, ComOpening};
use crate::protocol::{rokoko_prove, rokoko_verify, LinComInstance, RoKokoProof};
use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_core::Goldilocks;
use lattice_ring::{RingConfig, RingElement};

#[derive(Debug)]
pub enum PcsFrontError {
    Transcript(TranscriptError),
    /// Coefficient/point shape mismatch.
    Shape {
        expected: usize,
        got: usize,
    },
    /// The point is not on the coefficient hypercube's axes.
    BadPoint {
        axes: usize,
        got: usize,
    },
    /// The underlying RoKoko round failed.
    Protocol(String),
}

impl From<TranscriptError> for PcsFrontError {
    fn from(e: TranscriptError) -> Self {
        PcsFrontError::Transcript(e)
    }
}

/// The front-end commitment: the packed coefficient matrix + its COM
/// outputs.
pub struct RokokoPcsCommitment {
    /// The coefficient matrix's columns (`r` columns of `m_w` elements).
    pub w_cols: Vec<Vec<RingElement>>,
    /// The COM outputs for vec(Y) (the F·W = H·Y block's committed
    /// images — the identity block at the front end: `F = I`, `H = I`,
    /// `Y = W`).
    pub coms: Vec<Vec<RingElement>>,
    pub openings: Vec<ComOpening>,
    /// The axes count (log2 of the coefficient count).
    pub axes: usize,
}

/// Transpose the column-major witness into row-major Y rows
/// (`m_w × r`, `ys[i]` = row i of length r).
fn w_rows(ring: &RingConfig, w_cols: &[Vec<RingElement>]) -> Vec<Vec<RingElement>> {
    let r = w_cols.len();
    let m_w = w_cols.first().map(|c| c.len()).unwrap_or(0);
    let mut rows = Vec::with_capacity(m_w);
    for i in 0..m_w {
        let mut row = Vec::with_capacity(r);
        for c in 0..r {
            row.push(w_cols[c][i].clone());
        }
        rows.push(row);
    }
    let _ = ring;
    rows
}

/// Build the eq-weight left vector `ℓ` for the multilinear evaluation
/// at `v`: `ℓ_b = eq(v, b)` over the coefficient hypercube (the
/// LSB-first convention matching the eq table order), embedded as ring
/// constants.
fn eq_weights_ring(
    ring: &RingConfig,
    v: &[u64],
    m_w: usize,
) -> Result<Vec<RingElement>, PcsFrontError> {
    let axes = v.len();
    let n = 1usize << axes;
    if n < m_w {
        return Err(PcsFrontError::Shape {
            expected: m_w,
            got: n,
        });
    }
    // ℓ has m_w entries: the eq table over the FIRST m_w cube points
    // (the matrix's rows; the remaining axes live on the right side r).
    let q = u64::from(ring.modulus.q);
    let mut out = Vec::with_capacity(m_w);
    for b in 0..m_w {
        let mut acc: u128 = 1;
        for (j, &vj) in v.iter().enumerate() {
            // Variable j = the row-index bit j — only the leading axes
            // apply to the rows (the rest fold into r).
            let bj = ((b >> j) & 1) as u64;
            let term = if bj == 1 {
                u128::from(vj % q)
            } else {
                u128::from((q - vj % q) % q)
            };
            acc = acc * term % u128::from(q);
        }
        out.push(ring.constant((acc % u128::from(q)) as u32));
    }
    Ok(out)
}

/// `RokokoPcs::commit` — pack `2^{axes}` coefficients into the
/// `m_w × r` witness matrix and COM-commit the identity block.
pub fn pcs_commit(
    ck: &mut ComKey,
    ring: &RingConfig,
    coeffs: &[i64],
) -> Result<RokokoPcsCommitment, PcsFrontError> {
    let n = coeffs.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(PcsFrontError::Shape {
            expected: 0,
            got: n,
        });
    }
    let axes = n.trailing_zeros() as usize;
    // The kernel front end: `m_w = n` rows with the coefficients in
    // COLUMN 0 (the remaining columns zero — the evaluation claim is
    // the protocol's column-0 linear form `ℓ^T·W[0] = t`; r ≥ 2 keeps
    // the fold-split's column fold non-degenerate).
    let r = ck.params.r.max(2);
    let m_w = n;
    let q = u64::from(ring.modulus.q);
    let mut w_cols = Vec::with_capacity(r);
    for c in 0..r {
        let mut col = Vec::with_capacity(m_w);
        for i in 0..m_w {
            let v = if c == 0 {
                coeffs.get(i).copied().unwrap_or(0).rem_euclid(q as i64) as u32
            } else {
                0
            };
            col.push(ring.constant(v));
        }
        w_cols.push(col);
    }
    // The identity block's COM outputs: `com_klin` binds vec(Y) — the
    // FLATTENED Y rows (the protocol's committed linear block), with
    // Y = W (rows) at the front end.
    let y_flat: Vec<RingElement> = w_rows(ring, &w_cols).into_iter().flatten().collect();
    let (coms, openings) = com_commit(ck, ring, &y_flat, 1, ck.params.gadget_len)
        .map_err(|e| PcsFrontError::Protocol(format!("{e:?}")))?;
    let coms = vec![coms];
    let openings = vec![openings];
    Ok(RokokoPcsCommitment {
        w_cols,
        coms,
        openings,
        axes,
    })
}

/// The true multilinear evaluation over the packed matrix (the
/// reference for tests and the honest prover's target).
pub fn pcs_evaluate(ring: &RingConfig, commitment: &RokokoPcsCommitment, v: &[u64]) -> u64 {
    let q = u64::from(ring.modulus.q);
    let n = 1usize << commitment.axes;
    let mut coeffs = vec![0u64; n];
    let m_w = commitment.w_cols.first().map(|c| c.len()).unwrap_or(0);
    for i in 0..m_w.min(n) {
        coeffs[i] = u64::from(commitment.w_cols[0][i].coeff(0));
    }
    let mut acc: u128 = 0;
    for (b, &cb) in coeffs.iter().enumerate() {
        let mut eqv: u128 = 1;
        for (j, &vj) in v.iter().enumerate() {
            let bj = ((b >> j) & 1) as u64;
            let term = if bj == 1 { vj % q } else { (q - vj % q) % q };
            eqv = eqv * u128::from(term) % u128::from(q);
        }
        acc = (acc + eqv * u128::from(cb)) % u128::from(q);
    }
    acc as u64
}

/// Build the `Ξ^lin_COM` instance for the evaluation claim at `v`.
#[allow(clippy::too_many_lines)]
pub fn pcs_instance(
    ring: &RingConfig,
    commitment: &RokokoPcsCommitment,
    v: &[u64],
    value: u64,
) -> Result<LinComInstance, PcsFrontError> {
    if v.len() != commitment.axes {
        return Err(PcsFrontError::BadPoint {
            axes: commitment.axes,
            got: v.len(),
        });
    }
    let r = commitment.w_cols.len();
    let m_w = commitment.w_cols.first().map(|c| c.len()).unwrap_or(0);
    // ℓ: the eq weights over the leading axes (the row side).
    let ell = eq_weights_ring(ring, v, m_w)?;
    // The right selector: the claim selects COLUMN 0 (weight 1 on the
    // first column, zero elsewhere).
    let mut r_vec = vec![ring.zero(); r];
    r_vec[0] = ring.one();
    let rr = vec![r_vec];
    // The target: t = value (the claimed evaluation).
    let tt = vec![ring.constant((value % u64::from(ring.modulus.q)) as u32)];
    // The identity block: F = I, H = I, Y = W (kernel: the front end's
    // linear relation degenerates to the eq-weight row).
    let eye = |dim: usize| -> Vec<Vec<RingElement>> {
        (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| ring.constant(if i == j { 1 } else { 0 }))
                    .collect()
            })
            .collect()
    };
    Ok(LinComInstance {
        f: vec![eye(m_w)],
        h: vec![eye(m_w)],
        coms: commitment.coms.clone(),
        aux: commitment.openings.clone(),
        ell: vec![ell],
        rr,
        tt,
        m_w,
        r,
        beta_w: ck_beta_w(commitment),
        w_cols: Some(commitment.w_cols.clone()),
        ys: Some(vec![w_rows(ring, &commitment.w_cols)]),
    })
}

fn ck_beta_w(commitment: &RokokoPcsCommitment) -> u64 {
    // The kernel norm bound: the coefficient spread (small at the
    // front end).
    let mut max: u64 = 1;
    for col in &commitment.w_cols {
        for e in col {
            max = max.max(e.infinity_norm() as u64);
        }
    }
    max.saturating_mul(max)
}

/// `prove_eval`: run the RoKoko round driver over the evaluation
/// instance.
pub fn pcs_prove_eval(
    ring: &RingConfig,
    commitment: &RokokoPcsCommitment,
    ck: &mut ComKey,
    v: &[u64],
    value: u64,
    transcript: &mut Transcript,
    l_prime: usize,
) -> Result<RoKokoProof, PcsFrontError> {
    let inst = pcs_instance(ring, commitment, v, value)?;
    rokoko_prove(&inst, ck, ring, transcript, l_prime)
        .map_err(|e| PcsFrontError::Protocol(format!("{e:?}")))
}

/// `verify_eval`.
pub fn pcs_verify_eval(
    ring: &RingConfig,
    commitment: &RokokoPcsCommitment,
    ck: &mut ComKey,
    v: &[u64],
    value: u64,
    proof: &RoKokoProof,
    transcript: &mut Transcript,
    l_prime: usize,
) -> Result<bool, PcsFrontError> {
    let inst = pcs_instance(ring, commitment, v, value)?;
    rokoko_verify(&inst, ck, ring, proof, transcript, l_prime)
        .map_err(|e| PcsFrontError::Protocol(format!("{e:?}")))?;
    Ok(true)
}

/// The Goldilocks-flavored point helper (the trait-facing surface).
pub fn point_from_goldilocks(point: &[Goldilocks], q: u64) -> Vec<u64> {
    point.iter().map(|g| g.to_canonical_u64() % q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::com::RokokoParams;

    fn ring() -> RingConfig {
        RingConfig::new(lattice_ring::Modulus32::Q_32, 4).unwrap()
    }

    fn ck(r: usize) -> ComKey {
        ComKey::new(
            RokokoParams {
                n_ring: 4,
                n0: 4,
                gadget_len: 4,
                com_depth: 1,
                r,
                beta_w: 1 << 16,
            },
            [9u8; 32],
        )
    }

    fn coeffs(n: usize) -> Vec<i64> {
        (0..n).map(|i| ((i * 37) % 199) as i64 - 99).collect()
    }

    #[test]
    fn pcs_commit_prove_verify_roundtrip() {
        let ring = ring();
        let mut key = ck(2);
        let cs = coeffs(16);
        let com = pcs_commit(&mut key, &ring, &cs).unwrap();
        let v: Vec<u64> = vec![101, 211, 307, 401];
        let t = pcs_evaluate(&ring, &com, &v);
        // The evaluation is the true multilinear value (pinned against
        // the direct coefficient sum).
        let q = u64::from(ring.modulus.q);
        let mut direct: u128 = 0;
        for (b, &cb) in cs.iter().enumerate() {
            let mut eqv: u128 = 1;
            for (j, &vj) in v.iter().enumerate() {
                let bj = ((b >> j) & 1) as u64;
                let term = if bj == 1 { vj % q } else { (q - vj % q) % q };
                eqv = eqv * u128::from(term) % u128::from(q);
            }
            direct = (direct + eqv * u128::from(cb.rem_euclid(q as i64) as u64)) % u128::from(q);
        }
        assert_eq!(t, direct as u64);
        let mut tr = Transcript::new_default(b"rk-pcs");
        let proof = pcs_prove_eval(&ring, &com, &mut key, &v, t, &mut tr, 32).unwrap();
        let mut tv = Transcript::new_default(b"rk-pcs");
        assert!(pcs_verify_eval(&ring, &com, &mut key, &v, t, &proof, &mut tv, 32).unwrap());
        // A tampered commitment is rejected (the bound direction; the
        // wrong-TARGET direction rides the constraint checks — the
        // crate's documented claims-binding gap ledger entry).
        let mut bad_com = com;
        let mut cm0 = bad_com.coms[0][0].coeffs().to_vec();
        cm0[0] = (cm0[0] + 1) % ring.modulus.q;
        bad_com.coms[0][0] = RingElement::from_coeffs(&ring, cm0);
        let mut tv2 = Transcript::new_default(b"rk-pcs");
        assert!(pcs_verify_eval(&ring, &bad_com, &mut key, &v, t, &proof, &mut tv2, 32).is_err());
    }

    #[test]
    fn pcs_wrong_point_rejected() {
        let ring = ring();
        let mut key = ck(2);
        let cs = coeffs(8);
        let com = pcs_commit(&mut key, &ring, &cs).unwrap();
        assert!(matches!(
            pcs_instance(&ring, &com, &[1, 2], 0),
            Err(PcsFrontError::BadPoint { .. })
        ));
    }

    #[test]
    fn pcs_non_power_two_rejected() {
        let ring = ring();
        let mut key = ck(2);
        assert!(matches!(
            pcs_commit(&mut key, &ring, &[1, 2, 3]),
            Err(PcsFrontError::Shape { .. })
        ));
    }
}
