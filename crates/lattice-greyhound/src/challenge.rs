//! The challenge space C ⊂ R_q (LaBRADOR §2 / Greyhound §5) and the auxiliary
//! challenge distributions.
//!
//! Concrete set (both papers' instantiation): 64 - TAU1 - TAU2 = 24 zero
//! coefficients, TAU1 = 32 coefficients ±1, TAU2 = 8 coefficients ±2. Size ≈ 2^123
//! elements (the papers quote 2^128 for the 23/31/10 variant; the reference's set
//! is 32/8 — we follow the reference, matching the 53KB parameter tables).
//! Rejection keeps `‖c‖op ≤ T = 14` (LaBRADOR §2 uses T = 15, the reference T = 14;
//! we take the reference's bound, which is the stricter one).
//!
//! Differences of distinct challenges are invertible mod q ([LS18, Corollary 1.2]:
//! q ≡ 5 mod 8 gives the two-factor split; short differences with
//! `‖c1 - c2‖∞ ≤ 4` are nonzero in both factors). The amortized-opening division
//! `(z - z')/(c - c')` in the extraction tests uses [`crate::ring::poly_inv`].

use crate::ring::{expand_seed, poly_inv, Poly, N};

/// Number of ±1 coefficients.
pub const TAU1: usize = 32;
/// Number of ±2 coefficients.
pub const TAU2: usize = 8;
/// Operator-norm rejection bound (the reference's T; the paper's §2 quotes 15).
pub const T: f64 = 14.0;
/// τ = TAU1·1² + TAU2·2² = 48 — the l2² of a challenge (the papers' τ).
pub const TAU: f64 = (TAU1 + 4 * TAU2) as f64;
/// κ = TAU1·1 + TAU2·2 = 48 — the l1 norm of a challenge (Greyhound's κ).
pub const KAPPA_CH: f64 = (TAU1 + 2 * TAU2) as f64;

/// Sample `len` challenges from C with operator-norm rejection, from a
/// seed+nonce (deterministic — the Fiat-Shamir reproducible stream).
///
/// The sampler is the reference's partial Fisher-Yates: positions are drawn in
/// decreasing order from the random stream, signs from a packed bit prefix;
/// challenges with opnorm > T are rejected and resampled (fresh stream).
pub fn challenge_vec(len: usize, seed: &[u8], nonce: u64) -> Vec<Poly> {
    let mut out = Vec::with_capacity(len);
    let mut nonce = nonce;
    while out.len() < len {
        // ~120 position bytes + 8 sign bytes amortizes one challenge; the
        // expected draw count is Σ_{k=24..63} 64/k ≈ 64, with tail slack.
        let mut buf = vec![0u8; 256];
        expand_seed(seed, nonce, &mut buf);
        nonce = nonce.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if let Some(c) = sample_one(&buf) {
            if c.opnorm() <= T {
                out.push(c);
            }
        }
    }
    out
}

/// One challenge from a 256-byte buffer (positions from byte 8 onward).
fn sample_one(buf: &[u8]) -> Option<Poly> {
    let mut coeffs = [0i64; N];
    let mut k = N - TAU1 - TAU2; // next position to fix
    let mut signs: u64 = 0;
    for i in 0..8 {
        signs |= (buf[i] as u64) << (8 * i);
    }
    let mut j = 8usize;
    while k < N && j < buf.len() {
        let b = (buf[j] as usize) & (N - 1);
        j += 1;
        if b <= k {
            // Fisher-Yates step: place the new signed value at position b,
            // moving whatever was there to position k.
            let val = if k < N - TAU2 { 1i64 } else { 2i64 };
            let signed = if signs & 1 == 1 { -val } else { val };
            signs >>= 1;
            coeffs[k] = coeffs[b];
            coeffs[b] = signed;
            k += 1;
        }
    }
    if k < N {
        return None; // ran out of randomness — caller resamples
    }
    Some(Poly(coeffs))
}

/// Quarternary challenges: coefficients uniform in {-2, -1, 0, 1} (2 bits each).
/// Used for the four challenge groups (α, β, γ, δ) of the level-aggregation step
/// in the reference. The papers' description uses uniform R_q α/β for the F-family
/// aggregation; see `protocol.rs` for how this port keeps the paper's uniform
/// aggregation for statement constraints while using quarternary challenges only
/// where the reference does (the *internal* folding of the level's verification
/// equations into the next statement's single constraint).
pub fn quarternary_vec(len: usize, seed: &[u8], nonce: u64) -> Vec<Poly> {
    let bytes = (len * N).div_ceil(4);
    let mut buf = vec![0u8; bytes];
    expand_seed(seed, nonce, &mut buf);
    (0..len)
        .map(|i| {
            let mut p = [0i64; N];
            for j in 0..N {
                let bitpos = (i * N + j) * 2;
                let v = ((buf[bitpos / 8] >> (bitpos % 8)) & 3) as i64;
                p[j] = v - 2;
            }
            Poly(p)
        })
        .collect()
}

/// Uniform Z_q challenge polynomials (the papers' α, β ← R_q^K for the
/// F-family aggregation — Theorem 5.1's distribution).
pub fn uniform_rq_vec(len: usize, seed: &[u8], nonce: u64) -> Vec<Poly> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(Poly::almost_uniform(
            seed,
            nonce.wrapping_add((i as u64) << 40),
        ));
    }
    out
}

/// Scalar Z_q challenges (the papers' ψ ∈ Z_q^L, ω ∈ Z_q^256).
pub fn zq_scalars(len: usize, seed: &[u8], nonce: u64) -> Vec<i64> {
    let mut buf = vec![0u8; len * 4];
    expand_seed(seed, nonce, &mut buf);
    (0..len)
        .map(|i| {
            let mut v = 0u32;
            for k in 0..4 {
                v |= (buf[i * 4 + k] as u32) << (8 * k);
            }
            (v % crate::ring::Q as u32) as i64
        })
        .collect()
}

/// The inverse of 2 mod q (q odd): (q+1)/2 — used for the paper's h_ij = ½(…).
pub const INV2: i64 = (crate::ring::Q + 1) / 2;

/// Membership test for C (shape + operator norm).
pub fn is_challenge(c: &Poly) -> bool {
    let (mut p1, mut m1, mut p2, mut m2) = (0, 0, 0, 0);
    for &x in c.0.iter() {
        match x {
            1 => p1 += 1,
            -1 => m1 += 1,
            2 => p2 += 1,
            -2 => m2 += 1,
            0 => {}
            _ => return false,
        }
    }
    p1 + m1 == TAU1 && p2 + m2 == TAU2 && c.opnorm() <= T
}

/// Check that a difference of two distinct challenges is invertible mod q
/// (the weak-opening division). Returns the inverse if it exists.
pub fn challenge_diff_inverse(c1: &Poly, c2: &Poly) -> Option<Poly> {
    let d = c1.sub(c2);
    if d.is_zero() {
        return None;
    }
    poly_inv(&d)
}

/// Divide a ring element by a challenge difference (exact in R_q).
pub fn divide_by_diff(z: &Poly, c1: &Poly, c2: &Poly) -> Option<Poly> {
    let inv = challenge_diff_inverse(c1, c2)?;
    Some(z.mul(&inv))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_sampling_is_fast() {
        let t0 = std::time::Instant::now();
        let cs = challenge_vec(8, b"timing", 0);
        eprintln!("8 challenges in {:?}", t0.elapsed());
        assert_eq!(cs.len(), 8);
        // distribution of the opnorm
        let ops: Vec<f64> = cs.iter().map(|c| c.opnorm()).collect();
        eprintln!("opnorms: {ops:.2?}");
    }

    #[test]
    fn challenge_shape_and_norms() {
        let cs = challenge_vec(64, b"test", 0);
        assert_eq!(cs.len(), 64);
        for c in &cs {
            let mut plus1 = 0;
            let mut minus1 = 0;
            let mut plus2 = 0;
            let mut minus2 = 0;
            let mut zeros = 0;
            for &x in c.0.iter() {
                match x {
                    1 => plus1 += 1,
                    -1 => minus1 += 1,
                    2 => plus2 += 1,
                    -2 => minus2 += 1,
                    0 => zeros += 1,
                    _ => panic!("bad coefficient {x}"),
                }
            }
            assert_eq!(plus1 + minus1, TAU1);
            assert_eq!(plus2 + minus2, TAU2);
            assert_eq!(zeros, N - TAU1 - TAU2);
            assert_eq!(c.normsq(), (TAU1 + 4 * TAU2) as u64);
            assert!(c.opnorm() <= T, "operator norm rejection violated");
        }
    }

    #[test]
    fn distinct_challenges_have_invertible_differences() {
        let cs = challenge_vec(40, b"inv-test", 7);
        for i in 0..40 {
            for j in i + 1..40 {
                if cs[i] != cs[j] {
                    assert!(
                        challenge_diff_inverse(&cs[i], &cs[j]).is_some(),
                        "difference of distinct challenges must be invertible"
                    );
                }
            }
        }
    }

    #[test]
    fn quarternary_range() {
        let qs = quarternary_vec(8, b"q", 0);
        for q in &qs {
            assert!(q.0.iter().all(|&x| (-2..=1).contains(&x)));
        }
    }

    #[test]
    fn uniform_rq_shape() {
        let us = uniform_rq_vec(4, b"u", 3);
        for u in &us {
            assert!(u.norminf() <= crate::ring::Q / 2);
        }
    }

    #[test]
    fn zq_scalars_range() {
        let zs = zq_scalars(100, b"z", 5);
        assert!(zs.iter().all(|&z| (0..crate::ring::Q).contains(&z)));
    }

    #[test]
    fn division_by_diff_roundtrip() {
        let cs = challenge_vec(2, b"div", 11);
        let (c1, c2) = (&cs[0], &cs[1]);
        if c1 == c2 {
            return;
        }
        let z = Poly::almost_uniform(b"z", 9);
        let q = divide_by_diff(&z, c1, c2).expect("invertible");
        // (z / (c1-c2))·(c1-c2) = z
        let d = c1.sub(c2);
        assert_eq!(q.mul(&d), z);
    }
}
