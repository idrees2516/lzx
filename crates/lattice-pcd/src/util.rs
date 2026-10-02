//! Shared utilities over the protocol field `F_r` (BN254 scalar field via
//! `lattice_projsumcheck::fp256::Fp256`): full-width challenge sampling and
//! the equality-polynomial machinery of §5.1 (Eq. (3)'s `eq`, `eq_e`
//! multilinear Lagrange bases, and the boolean-hypercube sum-check
//! primitives the accumulation prover/verifier share).
//!
//! **Representation discipline**: `Fp256` limbs are Montgomery form; field
//! arithmetic (`add`/`sub`/`mul`) is Montgomery-native; transcript
//! absorption uses canonical bytes (`from_mont().canon_bytes()`); challenge
//! sampling reduces 32 XOF bytes to a full-width Montgomery element.

use crate::Fp256;
use lattice_core::transcript::Transcript;
use lattice_projsumcheck::fp256::reduce_wide_ref;

/// Sample a full-width field element from 32 big-endian bytes (reduced mod
/// r, returned in Montgomery form).
pub fn fp_from_be32(bytes: &[u8; 32]) -> Fp256 {
    let mut wide = [0u64; 8];
    let mut limb_idx = 4usize;
    let mut acc: u128 = 0;
    for (i, b) in bytes.iter().enumerate() {
        acc = (acc << 8) | (*b as u128);
        if i % 8 == 7 {
            limb_idx -= 1;
            wide[limb_idx] = acc as u64;
            acc = 0;
        }
    }
    let canon = reduce_wide_ref(&wide);
    Fp256 { limbs: canon }.to_mont()
}

/// Absorb a field element into a transcript in canonical byte form.
pub fn absorb_fp(t: &mut Transcript, label: &[u8], v: &Fp256) -> Result<(), lattice_core::transcript::TranscriptError> {
    t.append_bytes(label, &v.from_mont().canon_bytes())
}

/// Absorb a slice of field elements.
pub fn absorb_fp_slice(
    t: &mut Transcript,
    label: &[u8],
    vs: &[Fp256],
) -> Result<(), lattice_core::transcript::TranscriptError> {
    let mut buf = Vec::with_capacity(vs.len() * 32);
    for v in vs {
        buf.extend_from_slice(&v.from_mont().canon_bytes());
    }
    t.append_bytes(label, &buf)
}

/// Sample a full-width challenge field element from the transcript.
pub fn challenge_fp(
    t: &mut Transcript,
    label: &[u8],
) -> Result<Fp256, lattice_core::transcript::TranscriptError> {
    let bytes = t.challenge_bytes(label, 32)?;
    let mut b = [0u8; 32];
    b.copy_from_slice(&bytes);
    Ok(fp_from_be32(&b))
}

/// Sample a vector of full-width challenges.
pub fn challenge_fp_vec(
    t: &mut Transcript,
    label: &[u8],
    n: usize,
) -> Result<Vec<Fp256>, lattice_core::transcript::TranscriptError> {
    let bytes = t.challenge_bytes(label, 32 * n)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0u8; 32];
        b.copy_from_slice(&bytes[i * 32..(i + 1) * 32]);
        out.push(fp_from_be32(&b));
    }
    Ok(out)
}

/// Deterministic pseudo-random field vector from a seed (for tests and
/// dummy-instance sampling with a reproducible transcript).
pub fn fp_vec_from_seed(label: &[u8], seed: &[u8], n: usize) -> Vec<Fp256> {
    let bytes = Transcript::xof(label, seed, 32 * n);
    (0..n)
        .map(|i| {
            let mut b = [0u8; 32];
            b.copy_from_slice(&bytes[i * 32..(i + 1) * 32]);
            fp_from_be32(&b)
        })
        .collect()
}

/// `eq(X, Y) = ∏ᵢ (XᵢYᵢ + (1−Xᵢ)(1−Yᵢ))` evaluated at concrete points —
/// the equality function of §5.1 (Corollary 1's `eq`).
pub fn eq_eval(x: &[Fp256], y: &[Fp256]) -> Fp256 {
    let one = Fp256::from_canonical_u64(1);
    let mut acc = one;
    for (xi, yi) in x.iter().zip(y.iter()) {
        // xi·yi + (1−xi)(1−yi) = 1 − xi − yi + 2·xi·yi
        let prod = xi.mul(yi);
        let term = one.sub(xi).sub(yi).add(&prod).add(&prod);
        acc = acc.mul(&term);
    }
    acc
}

/// The multilinear Lagrange basis `eq_e_i(X)` evaluated at a concrete point
/// `x`: the indicator of `Bits(i)` on the boolean cube.
pub fn eq_basis_eval(i: usize, x: &[Fp256]) -> Fp256 {
    let one = Fp256::from_canonical_u64(1);
    let mut acc = one;
    for (k, xk) in x.iter().enumerate() {
        let bit = (i >> k) & 1;
        let term = if bit == 1 {
            *xk
        } else {
            one.sub(xk)
        };
        acc = acc.mul(&term);
    }
    acc
}

/// Evaluate the multilinear extension of `f` (evaluations over the boolean
/// cube in row-major order, `f[i]` at `Bits(i)`) at point `r` — the
/// standard MLE evaluation via eq bases.
pub fn mle_eval(f: &[Fp256], r: &[Fp256]) -> Fp256 {
    let n = f.len();
    let zero = Fp256::ZERO;
    if n == 0 {
        return zero;
    }
    // Iterative folding: fold dimension by dimension.
    let mut cur = f.to_vec();
    let mut len = n;
    for k in (0..r.len()).rev() {
        let rk = r[k];
        let one = Fp256::from_canonical_u64(1);
        let next_len = len / 2;
        for i in 0..next_len {
            let lo = cur[i];
            let hi = cur[i + next_len];
            // eq contribution: (1−rk)·lo + rk·hi
            cur[i] = one.sub(&rk).mul(&lo).add(&rk.mul(&hi));
        }
        len = next_len;
    }
    cur[0]
}

/// Sample `n` independent blinding scalars from the transcript.
pub fn blinds_from_transcript(
    t: &mut Transcript,
    label: &[u8],
    n: usize,
) -> Result<Vec<Fp256>, lattice_core::transcript::TranscriptError> {
    challenge_fp_vec(t, label, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_from_be32_matches_canonical() {
        // 1 as BE bytes → the field element 1.
        let mut b = [0u8; 32];
        b[31] = 1;
        let f = fp_from_be32(&b);
        assert_eq!(f, Fp256::from_canonical_u64(1));
        // 2^200: big-endian byte 6 (bits 192..199 region), bit 0 of byte 6
        // is bit 200 overall (byte 0 holds bits 248..255).
        let mut c = [0u8; 32];
        c[6] = 0x01;
        let g = fp_from_be32(&c);
        let two = Fp256::from_canonical_u64(2);
        let expected = two.pow(&[200, 0, 0, 0]);
        assert_eq!(g, expected);
    }

    #[test]
    fn eq_eval_basics() {
        let zero = Fp256::ZERO;
        let one = Fp256::from_canonical_u64(1);
        let x = vec![one, zero];
        let y = vec![one, one];
        // x = (1,0), y = (1,1): term0 = 1, term1 = (1−0)(1−1) = 0 → 0.
        assert!(eq_eval(&x, &y).is_zero());
        assert_eq!(eq_eval(&x, &x.clone()), one);
    }

    #[test]
    fn eq_basis_indicator() {
        let one = Fp256::from_canonical_u64(1);
        // Bits(2) = (bit0=0, bit1=1) under LSB-first bit order.
        let r = vec![Fp256::ZERO, one];
        assert_eq!(eq_basis_eval(2, &r), one);
        assert!(eq_basis_eval(0, &r).is_zero());
        assert!(eq_basis_eval(1, &r).is_zero());
        assert!(eq_basis_eval(3, &r).is_zero());
    }

    #[test]
    fn mle_eval_matches_direct() {
        // f = [5, 7, 11, 13] over the 2-cube; r = (a, b).
        let f: Vec<Fp256> = [5u64, 7, 11, 13]
            .iter()
            .map(|v| Fp256::from_canonical_u64(*v))
            .collect();
        let a = Fp256::from_canonical_u64(3);
        let b = Fp256::from_canonical_u64(5);
        let r = vec![a, b];
        // Direct: f(Bits(i)) with interpolation: (1−a)(1−b)·5 + a(1−b)·7 +
        // (1−a)b·11 + ab·13
        let one = Fp256::from_canonical_u64(1);
        let direct = one
            .sub(&a)
            .mul(&one.sub(&b))
            .mul(&f[0])
            .add(&a.mul(&one.sub(&b)).mul(&f[1]))
            .add(&one.sub(&a).mul(&b).mul(&f[2]))
            .add(&a.mul(&b).mul(&f[3]));
        assert_eq!(mle_eval(&f, &r), direct);
    }

    #[test]
    fn challenge_sampling_roundtrip_distinct() {
        let mut t = Transcript::new_default(b"test");
        let c1 = challenge_fp(&mut t, b"c").ok().unwrap();
        let c2 = challenge_fp(&mut t, b"c").ok().unwrap();
        assert_ne!(c1, c2);
    }
}
