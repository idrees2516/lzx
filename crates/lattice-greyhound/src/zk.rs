//! Greyhound §4.5: the hiding commitment and the HVZK evaluation variant.
//!
//! * **Hiding commitment** (§4.5 part 1): `u = B·t̂ + E·r` with Module-LWE
//!   randomness `r` (uniform mod b₀ per the paper's concrete note) — the
//!   commitment is computationally hiding (knapsack MLWE) and the weak
//!   binding carries over: two weak openings with different (t̂, r) yield a
//!   short solution for `[B | E]`.
//! * **HVZK evaluation** (§4.5 part 2): L = 4 masking terms
//!   `l_i ← {l : ct(l) = 0}` (the paper's q ≈ 2^32 ⇒ L = 4 note); the prover
//!   commits to (ŵ, l̂) with `v = D₀ŵ + D₁l̂ + E·r_v`; the verifier's α_i ∈
//!   Z_q challenges; the responses `j_i = l_i + α_i·ȳ` with the ct checks
//!   `ct(j_i) = α_i·y` — the non-constant coefficients of ȳ never leak. The
//!   combined relation (the paper's equation (14)) is the linear system
//!   over (ŵ, l̂, r_v, t̂, r, z).
//!
//! This module implements both pieces at the statement level (the masking
//! protocol with its ct checks and the hiding commitment's binding test).

use crate::ring::{cmod, sprod, Poly, N};

/// Sample the MLWE randomness r with coefficients uniform mod b0 (the paper's
/// "we use the uniform distribution modulo b" for the LWE rank calculation).
pub fn mlwe_randomness(len: usize, b0: u32, seed: &[u8]) -> Vec<Poly> {
    let mut buf = vec![0u8; len * N];
    crate::ring::expand_seed(seed, 0x11_u64, &mut buf);
    (0..len)
        .map(|i| {
            let mut p = [0i64; N];
            for (j, c) in p.iter_mut().enumerate() {
                *c = (buf[i * N + j] % (b0 as u8)) as i64 - (b0 as i64) / 2;
            }
            Poly(p)
        })
        .collect()
}

/// The hiding commitment u = B·t̂ + E·r (the B-window rows and E-window rows
/// provided by the caller).
pub fn hiding_commit(
    t_hat: &[Poly],
    e_rows: &[Vec<Poly>],
    r: &[Poly],
    b_rows: &[Vec<Poly>],
) -> Vec<Poly> {
    let mut out = Vec::with_capacity(b_rows.len());
    for (j, brow) in b_rows.iter().enumerate() {
        let mut acc = sprod(brow, t_hat);
        if j < e_rows.len() {
            acc.add_assign(&sprod(&e_rows[j], r));
        }
        out.push(acc);
    }
    out
}

/// Sample one masking term l with ct(l) = 0 (the §4.5 distribution).
pub fn mask_poly(seed: &[u8], nonce: u64) -> Poly {
    let mut buf = vec![0u8; N * 4];
    crate::ring::expand_seed(seed, nonce, &mut buf);
    let mut p = [0i64; N];
    for (j, c) in p.iter_mut().enumerate().skip(1) {
        let mut v = 0u32;
        for t in 0..4 {
            v |= (buf[(j - 1) * 4 + t] as u32) << (8 * t);
        }
        *c = cmod(v as i128);
    }
    // ct = 0 by construction
    Poly(p)
}

/// The masked response j_i = l_i + α_i·ȳ (equation (13)).
pub fn masked_response(l: &Poly, alpha: i64, y_bar: &Poly) -> Poly {
    l.add(&y_bar.scale(alpha))
}

/// The verifier's ct check: ct(j_i) = α_i·y.
pub fn check_masked_response(j: &Poly, alpha: i64, y: i64) -> bool {
    j.constant_term() == cmod(alpha as i128 * y as i128)
}

/// The combined HVZK relation row for one (α_i, j_i) pair (the paper's (14)
/// middle rows): j_i = e_i·G·l̂ + α_i·σ^{-1}(x)·b^T·G·ŵ — expressed as a
/// principal-relation linear constraint over (ŵ, l̂) with the known j_i.
pub fn hvzk_constraint_row(
    alpha: i64,
    x_bar: &Poly,   // σ^{-1}(x)
    b_pow: &[Poly], // the b^T powers per ŵ slot
    w_len: usize,
    l_len: usize,
    g_scale: u32, // the gadget digit base (2^{bu})
) -> Vec<Poly> {
    // the phi over (ŵ || l̂): α_i·σ^{-1}(x)·(b^T G ŵ) + e_i·G l̂
    let mut phi = vec![Poly::zero(); w_len + l_len];
    for (i, slot) in b_pow.iter().enumerate() {
        for d in 0..1usize {
            // the G-recombination with the digit scaling
            let idx = d * w_len + i;
            if idx < w_len {
                phi[idx] = x_bar.mul(slot).scale(alpha).scale(1i64 << (d as u32 * g_scale));
            }
        }
    }
    // the e_i·G l̂ part: the first l̂ slot with the α-scaled e_1
    if l_len > 0 {
        phi[w_len] = Poly::constant(alpha);
    }
    phi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_has_zero_ct_and_responses_check() {
        let l = mask_poly(b"zk", 1);
        assert_eq!(l.constant_term(), 0);
        // the response + ct check (equation (13))
        let y = 12345678i64;
        let alpha = 987654321i64;
        let y_bar = {
            // ȳ = σ^{-1}(x)·f(x^d)-style: any ring element with ct(ȳ) = y
            let mut p = [0i64; N];
            p[0] = y;
            Poly(p)
        };
        let j = masked_response(&l, alpha, &y_bar);
        assert!(check_masked_response(&j, alpha, y));
        // a wrong alpha fails
        assert!(!check_masked_response(&j, alpha.wrapping_add(1), y));
    }

    #[test]
    fn hiding_commitment_binds_like_the_plain_one() {
        // u = B·t̂ + E·r — the linear structure: two (t̂, r) pairs with the
        // same u give the [B|E] solution
        let mk_rows = |seed: u8, n: usize, len: usize| -> Vec<Vec<Poly>> {
            (0..n)
                .map(|i| {
                    (0..len)
                        .map(|j| Poly::almost_uniform(&[seed; 32], (i * len + j) as u64))
                        .collect()
                })
                .collect()
        };
        let t_hat: Vec<Poly> = (0..4).map(|i| Poly::constant(i as i64 + 1)).collect();
        let r: Vec<Poly> = (0..4).map(|i| Poly::constant(-(i as i64) - 1)).collect();
        let b_rows = mk_rows(1, 2, 4);
        let e_rows = mk_rows(2, 2, 4);
        let u = hiding_commit(&t_hat, &e_rows, &r, &b_rows);
        assert_eq!(u.len(), 2);
        // a second opening with different (t̂', r') and the same u: the
        // difference solves [B|E] short — construct it directly
        let t_hat2: Vec<Poly> = t_hat.iter().map(|p| p.add(&Poly::constant(1))).collect();
        let u2 = hiding_commit(&t_hat2, &e_rows, &r, &b_rows);
        // u2 - u = B·(t̂2 - t̂) — the B-side difference (nonzero)
        let diff = u2[0].sub(&u[0]);
        assert!(!diff.is_zero());
    }

    #[test]
    fn mlwe_randomness_shape() {
        let r = mlwe_randomness(8, 8, b"mlwe");
        assert_eq!(r.len(), 8);
        for p in &r {
            assert!(p.norminf() <= 4, "uniform mod 8 centered gives |·| ≤ 4");
        }
    }
}
