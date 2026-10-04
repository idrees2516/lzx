//! The TTRP projection and its two computation paths (§3, §4.1, Lemma 6).
//!
//! Two equivalent views of the projection of a witness
//! `v ∈ R^{m̄r}` (with `x = cf(v) ∈ Z^{m̄r·φ}` its coefficient vector):
//!
//! * **Integer view** (the verifier's norm check): the TT matrix
//!   `M_Z ∈ Z^{k×m̄r·φ}` applied to `x`:
//!   `y0 = M_Z · x mod q ∈ Z_q^k`. Computed row-wise by the right-to-left
//!   tensor contraction of Lemma 6's proof:
//!   `v^(µ) = x`, `v^(i−1) = (I_{d^{i−1}} ⊗ Mat(M_i)) · v^(i) mod q` —
//!   O(k·c·m̄r·φ) operations over Z_q, never materialising the row.
//!
//! * **Ring view** (the sumcheck linearisation): `y = M · v̄ ∈ R^k` with
//!   `M = cf^{-1}(M_Z)`, where `ct(y^{(j)}) = y0_j` by the power-of-two
//!   cyclotomic identity `ct(a·b̄) = ⟨cf(a), cf(b)⟩` (Appendix A.1).
//!   Computed via the S/W split: the row factorises as
//!   `M[j, h] = (Π_{p≤µ1} M_p(n_p(h))) · W^{(j)}` — a spatial chain
//!   `S^{(j)} ∈ Z^{m̄r×c}` times a coefficient-chain vector
//!   `W^{(j)} ∈ R^c` — hence
//!   `y^{(j)} = ⟨W^{(j)}, t^{(j)}⟩` with `t^{(j)} = S^{(j)ᵀ}·v̄ ∈ R^c`
//!   (all-integer coefficient work, c ring multiplications per row).
//!
//! The spatial/coefficient split of the flat column index
//! `n = h·φ + ℓ ∈ [d^µ]`: the high µ1 base-d digits of n index the ring
//! position `h ∈ [m̄r] = [d^{µ1}]`, the low µ2 digits index the
//! coefficient `ℓ ∈ [φ] = [d^{µ2}]`.

use crate::cores::{CoreTensor, TtrpParams};
use lattice_ring::{RingConfig, RingElement, RingError};

/// Coefficient embedding of a witness vector: flatten `v ∈ R^{m̄r}` to
/// `x ∈ Z^{m̄r·φ}` (h-major: `x[h·φ + ℓ] = v_h[ℓ]`).
pub fn cf_vec(v: &[RingElement]) -> Vec<i64> {
    let phi = v[0].config().n();
    let mut x = Vec::with_capacity(v.len() * phi);
    for elt in v {
        for &c in elt.coeffs() {
            x.push(c as i64);
        }
    }
    x
}

/// Inverse coefficient embedding of a flat integer row (length `m̄r·φ`)
/// into `R^{m̄r}`: `M[h] = Σ_ℓ row[h·φ+ℓ]·X^ℓ` reduced mod q.
pub fn cf_inv_vec(ring: &RingConfig, row: &[i64]) -> Vec<RingElement> {
    let phi = ring.n();
    let q = ring.modulus.q as i64;
    debug_assert_eq!(row.len() % phi, 0);
    let m_bar = row.len() / phi;
    let mut out = Vec::with_capacity(m_bar);
    for h in 0..m_bar {
        let coeffs: Vec<u32> = row[h * phi..(h + 1) * phi]
            .iter()
            .map(|&c| c.rem_euclid(q) as u32)
            .collect();
        out.push(RingElement::from_coeffs(ring, coeffs));
    }
    out
}

/// The σ⁻¹ conjugation automorphism on `R_q = Z_q[X]/(X^φ+1)`:
/// `conj(a)(X) = a(X^{-1})`, coefficient map `(a_0, −a_{φ−1}, …, −a_1)`.
/// Satisfies `conj(a·b) = conj(a)·conj(b)` (ring automorphism) and
/// `ct(a·b̄) = ⟨cf(a), cf(b)⟩` (the constant-term identity of A.1).
pub fn conj(a: &RingElement) -> RingElement {
    let ring = a.config();
    let n = ring.n();
    let q = ring.modulus.q as i64;
    let src = a.coeffs();
    let mut coeffs = vec![0u32; n];
    coeffs[0] = src[0];
    for i in 1..n {
        // X^{-i} = -X^{φ-i} mod (X^φ+1)
        let v = -(src[i] as i64);
        let v = if v < 0 { v + q } else { v };
        coeffs[n - i] = (v as u64 % q as u64) as u32;
    }
    RingElement::from_coeffs(ring, coeffs)
}

/// Conjugate every element of a vector.
pub fn conj_vec(v: &[RingElement]) -> Vec<RingElement> {
    v.iter().map(conj).collect()
}

/// Constant term of a ring element.
#[inline]
pub fn ct(a: &RingElement) -> u32 {
    a.coeffs()[0]
}

/// Centred representative of a residue in [0, q) as i64 in (−q/2, q/2].
#[inline]
pub fn centered(c: u32, q: u32) -> i64 {
    let c = c as i64;
    let q = q as i64;
    if c > q / 2 {
        c - q
    } else {
        c
    }
}

// ---------------------------------------------------------------------------
// Spatial / coefficient chain decomposition of a TT row
// ---------------------------------------------------------------------------

/// The spatial TT matrix `S^{(j)} ∈ Z^{m̄r×c}` of row j:
/// `S[h, ·] = Π_{p=1}^{µ1} M_p(n_p(h))` — the chain product of the spatial
/// cores at the base-d digits of h (MSB first). Built by the left-to-right
/// mixed-product chain over the first µ1 cores.
pub fn spatial_matrix(params: &TtrpParams, cores: &[CoreTensor]) -> Vec<i64> {
    let d = params.d();
    let m_bar = params.m_bar();
    let c = params.c;
    // R: 1 × (d^p · c_p) partial chain, MSB-digit-first layout.
    let mut r: Vec<i64> = cores[0].mat(); // 1 × (d·c)
    let mut r1 = cores[0].r1;
    let mut dd = d;
    for core in &cores[1..params.mu1] {
        let mc = core.mat(); // r1 × (d·core.r1)
        let block_cols = d * core.r1;
        let mut next = vec![0i64; dd * block_cols];
        for blk in 0..dd {
            for s in 0..d {
                for b in 0..core.r1 {
                    let mut acc = 0i64;
                    for a in 0..r1 {
                        acc += r[blk * r1 + a] * mc[a * block_cols + s * core.r1 + b];
                    }
                    next[blk * block_cols + s * core.r1 + b] = acc;
                }
            }
        }
        r = next;
        r1 = core.r1;
        dd *= d;
    }
    debug_assert_eq!(r.len(), m_bar * c);
    debug_assert_eq!(r1, c);
    r
}

/// The coefficient-chain vector `W^{(j)} ∈ R^c` of row j:
/// `W = Σ_{m_1..m_{µ2} ∈ [d]} M_{µ1+1}(m_1)⋯M_µ(m_{µ2}) · X^{val}`,
/// computed right-to-left with X-power twists (coefficient rotations with
/// sign flips — no ring multiplications).
#[allow(clippy::needless_range_loop)]
pub fn coefficient_chain(
    params: &TtrpParams,
    cores: &[CoreTensor],
    ring: &RingConfig,
) -> Vec<RingElement> {
    let d = params.d();
    let c = params.c;
    let phi = ring.n();
    // Running vector R ∈ R^c (as coefficient rows), from the last core.
    // Last core: c×d×1: R := Σ_{m∈[d]} M_µ(m) · X^m  (entries: degree < d,
    // each coefficient position m receives exactly one contribution).
    let last = &cores[params.mu() - 1];
    let mut acc: Vec<Vec<u32>> = vec![vec![0u32; phi]; c];
    for m in 0..d {
        for a in 0..c {
            let e = last.at(m, a, 0) as i64;
            if e != 0 {
                let v = acc[a][m] as i64 + e;
                acc[a][m] = (v.rem_euclid(ring.modulus.q as i64)) as u32;
            }
        }
    }
    // Remaining coefficient cores, right to left: core q (0-indexed from
    // mu1+1): R_q = Σ_m Mat(M_q)[m] · (X^{m·d^{remaining}} ⊙ R_{q+1}).
    for qi in (params.mu1..params.mu() - 1).rev() {
        let core = &cores[qi];
        // twist exponent stride: d^(cores to the right of qi, incl. last)
        let cores_right = params.mu() - 1 - qi;
        let stride = d.pow(cores_right as u32);
        // rotated-accumulation: new[a] = Σ_m Σ_b M_q(m)[a][b] * rot_{m*stride}(acc[b])
        let mut next: Vec<Vec<u32>> = vec![vec![0u32; phi]; c];
        let q_mod = ring.modulus.q as i64;
        for m in 0..d {
            let shift = (m * stride) % phi;
            for a in 0..c {
                for b in 0..c {
                    let e = core.at(m, a, b) as i64;
                    if e == 0 {
                        continue;
                    }
                    // rot_shift(acc[b]) * e: coefficient t of acc[b] moves to
                    // (t + shift) mod φ with negacyclic sign when wrapping.
                    for t in 0..phi {
                        let val = acc[b][t] as i64;
                        if val == 0 {
                            continue;
                        }
                        let prod = e * val;
                        let pos = t + shift;
                        let (dst, sign) = if pos >= phi {
                            (pos - phi, -1i64)
                        } else {
                            (pos, 1i64)
                        };
                        let term = if sign < 0 { -prod } else { prod };
                        let cur = next[a][dst] as i64;
                        next[a][dst] = ((cur + term).rem_euclid(q_mod)) as u32;
                    }
                }
            }
        }
        acc = next;
    }
    acc.into_iter()
        .map(|coeffs| RingElement::from_coeffs(ring, coeffs))
        .collect()
}

// ---------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------

/// Integer-view projection `y0 = M_Z · x mod q ∈ Z_q^k` (centred i64
/// representatives), via the right-to-left contraction of Lemma 6.
///
/// `x` may carry signed representatives; everything reduces mod q.
#[allow(clippy::needless_range_loop)]
pub fn project_integer(
    params: &TtrpParams,
    rows: &[Vec<CoreTensor>],
    q: u32,
    x: &[i64],
) -> Vec<i64> {
    let d = params.d();
    let cols = params.cols();
    debug_assert_eq!(x.len(), cols);
    let qi = q as i64;
    let mut y0 = Vec::with_capacity(params.k);
    for cores in rows {
        // v^(µ) = x mod q
        let mut v: Vec<i64> = x.iter().map(|&c| c.rem_euclid(qi)).collect();
        for i in (0..params.mu()).rev() {
            let core = &cores[i];
            let dd = d.pow(i as u32); // d^i: current length divisor
                                      // v has length dd * core.r1 (d^i * c_i).
                                      // v^(i-1)[blk * r0 + a] = Σ_{s,b} Mat(core)[a][s*r1+b] * v[blk*d*r1 + s*r1 + b]
            let r1 = core.r1;
            let r0 = core.r0;
            let block = d * r1;
            let mut next = vec![0i64; dd * r0];
            for blk in 0..dd {
                let base = blk * block;
                for a in 0..r0 {
                    let mut acc = 0i64;
                    for s in 0..d {
                        for b in 0..r1 {
                            let m = core.at(s, a, b) as i64;
                            if m != 0 {
                                acc += m * v[base + s * r1 + b];
                            }
                        }
                    }
                    // lazy reduction: values bounded by c·d·q — reduce here
                    next[blk * r0 + a] = acc % qi;
                }
            }
            v = next;
        }
        debug_assert_eq!(v.len(), 1);
        y0.push(centered((v[0].rem_euclid(qi)) as u32, q));
    }
    y0
}

/// Ring-view projection `y = M · v̄ ∈ R^k` via the S/W split.
///
/// Returns per row: `y^{(j)} = ⟨W^{(j)}, t^{(j)}⟩` with
/// `t^{(j)} = S^{(j)ᵀ}·v̄`. All heavy work is integer coefficient
/// arithmetic (O(k·c·m̄r·φ)); c ring multiplications per row at the end.
#[allow(clippy::needless_range_loop)]
pub fn project_ring(
    params: &TtrpParams,
    rows: &[Vec<CoreTensor>],
    ring: &RingConfig,
    v_bar: &[RingElement],
) -> Result<Vec<RingElement>, RingError> {
    let phi = ring.n();
    let q = ring.modulus.q as i64;
    let c = params.c;
    let m_bar = params.m_bar();
    debug_assert_eq!(v_bar.len(), m_bar);
    let mut out = Vec::with_capacity(params.k);
    for cores in rows {
        let s_mat = spatial_matrix(params, cores);
        let w_chain = coefficient_chain(params, cores, ring);
        // t[i] = Σ_h S[h,i] * v̄_h  — coefficient-wise integer accumulation.
        let mut t: Vec<Vec<i64>> = vec![vec![0i64; phi]; c];
        for h in 0..m_bar {
            let vb = v_bar[h].coeffs();
            for i in 0..c {
                let sv = s_mat[h * c + i];
                if sv == 0 {
                    continue;
                }
                for l in 0..phi {
                    t[i][l] += sv * vb[l] as i64;
                }
            }
        }
        // y = Σ_i W_i * t_i — c ring multiplications.
        let mut y = ring.zero();
        for i in 0..c {
            let t_elt = {
                let coeffs: Vec<u32> = t[i].iter().map(|&x| (x.rem_euclid(q)) as u32).collect();
                RingElement::from_coeffs(ring, coeffs)
            };
            let prod = w_chain[i].mul(&t_elt)?;
            y = y.add(&prod)?;
        }
        out.push(y);
    }
    Ok(out)
}

/// Materialise the full TT row of row j over Z_q (testing / small
/// instances only): `M_Z[j] ∈ Z_q^{m̄r·φ}`, row-major with the spatial
/// digits MSB-first.
pub fn materialize_row_q(_params: &TtrpParams, cores: &[CoreTensor], q: u32) -> Vec<i64> {
    let row = crate::cores::materialize_row(cores);
    row.into_iter().map(|c| c.rem_euclid(q as i64)).collect()
}
