//! The shared fold kernels (moved from `lattice-zkvm::{compact,
//! second_fold}`): the key application, the Goldilocks functional, and
//! the estimator's fail-closed MSIS gate.

use lattice_core::Goldilocks;
use lattice_ring::ring::{RingConfig, RingElement};

/// Apply a column-blocked key `blocks` (block c = the key's column c as
/// k ring elements) to a response `v`: `Σ_c blocks[c][rr]·v[c]`.
fn apply_key_inner(
    ring: &RingConfig,
    blocks: &[Vec<RingElement>],
    v: &[RingElement],
    out_rank: usize,
) -> Vec<RingElement> {
    let mut out = vec![ring.zero(); out_rank];
    for (c, blk) in blocks.iter().enumerate() {
        if c >= v.len() {
            break;
        }
        for (rr, e) in blk.iter().enumerate().take(out_rank) {
            if let Ok(prod) = v[c].mul(e) {
                if let Ok(s) = out[rr].add(&prod) {
                    out[rr] = s;
                }
            }
        }
    }
    out
}

/// The column-blocked key application (the level-1 link image
/// `F̄·v`): blocks are the key's COLUMNS (block c has k entries).
pub fn apply_key(
    ring: &RingConfig,
    blocks: &[Vec<RingElement>],
    v: &[RingElement],
    k: usize,
) -> Vec<RingElement> {
    apply_key_inner(ring, blocks, v, k)
}

/// The balanced-representative Goldilocks term of one coefficient:
/// `ψ · bal_q(c)` (the two-characteristic discipline: the balanced
/// integer is exact in BOTH fields because the gates keep |c| < q/2).
pub fn phi_term(weight: &Goldilocks, c: u32, q: u32) -> Goldilocks {
    let c_int = if c > q / 2 {
        c as i64 - q as i64
    } else {
        c as i64
    };
    let mag = weight.mul(&Goldilocks::from_u64(c_int.unsigned_abs()));
    if c_int < 0 {
        mag.neg()
    } else {
        mag
    }
}

/// The Goldilocks functional of a response: `Φ(v) = Σ ψ_m · bal(v_m)`
/// (weights beyond the response length are ZERO — the zero-padding
/// convention the recursive staging relies on).
pub fn functional_of(
    ring: &RingConfig,
    v: &[RingElement],
    psi_weights: &[Goldilocks],
    q: u64,
) -> Goldilocks {
    let n = ring.n();
    let mut acc = Goldilocks::ZERO;
    for (c, e) in v.iter().enumerate() {
        for d in 0..n {
            let w = psi_weights
                .get(c * n + d)
                .copied()
                .unwrap_or(Goldilocks::ZERO);
            if w != Goldilocks::ZERO {
                let term = phi_term(&w, e.coeffs()[d], q as u32);
                acc = acc.add(&term);
            }
        }
    }
    acc
}

/// The estimator verdict of an MSIS instance `[A | −T]` with `kappa`
/// rows over `width` ring-element columns at coefficient bound `bound`:
/// `(classical_bits, quantum_bits)` — the ADPS16 core-SVP costs of the
/// cheapest lattice attack the offline estimator models.
///
/// Fails (`Err`) when the instance is out of the estimator's model
/// (over-determined shapes `width ≤ kappa` are information-theoretically
/// binding — no short solution exists — but the honest posture refuses
/// to claim bits the model does not rate).
pub fn msis_bits(
    kappa: u64,
    width: u64,
    q: u64,
    ring_dim: u64,
    bound: u64,
) -> Result<(f64, f64), String> {
    if bound == 0 || bound >= q / 2 {
        return Err(format!("gate exceeds q/2: bound {bound} vs q {q}"));
    }
    let p = lattice_sis_estimator::scalar_sis_from_ring(
        ring_dim,
        kappa,
        width,
        q as u128,
        bound,
        lattice_sis_estimator::SisNorm::Infinity,
    )
    .map_err(|e| format!("estimator: {e:?}"))?;
    lattice_sis_estimator::sis_security_bits(&p).map_err(|e| format!("bits: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functional_zero_pads_short_weights() {
        let ring = crate::codec::q32_ring().unwrap();
        let n = ring.n();
        let v: Vec<RingElement> = (0..4)
            .map(|i| {
                let coeffs: Vec<u32> = (0..n).map(|c| ((i * 97 + c * 13) % 200) as u32).collect();
                RingElement::from_coeffs(&ring, coeffs)
            })
            .collect();
        // Weights only covering the first 2 elements: the tail is ZERO.
        let weights: Vec<Goldilocks> = (0..2 * n)
            .map(|i| Goldilocks::from_u64((i as u64 * 7919 + 3) % 1_000_003))
            .collect();
        let full = functional_of(&ring, &v[..2], &weights, u64::from(ring.modulus.q));
        let padded = functional_of(&ring, &v, &weights, u64::from(ring.modulus.q));
        assert_eq!(full, padded);
    }
}
