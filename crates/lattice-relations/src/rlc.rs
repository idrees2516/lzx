//! Random linear combination utilities for folding and claim batching.

use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;

/// Powers of a challenge: `[r^0, r^1, ..., r^{n-1}]`.
pub fn challenge_powers(r: &Goldilocks, n: usize) -> Vec<Goldilocks> {
    let mut out = Vec::with_capacity(n);
    let mut acc = Goldilocks::ONE;
    for _ in 0..n {
        out.push(acc);
        acc = acc.mul(r);
    }
    out
}

/// Fold two witness vectors: `w_fold = w1 + r · w2`.
pub fn fold_witnesses(
    w1: &[Goldilocks],
    w2: &[Goldilocks],
    r: &Goldilocks,
) -> Result<Vec<Goldilocks>, RlcError> {
    if w1.len() != w2.len() {
        return Err(RlcError::LengthMismatch {
            expected: w1.len(),
            got: w2.len(),
        });
    }
    Ok(w1
        .iter()
        .zip(w2.iter())
        .map(|(a, b)| a.add(&r.mul(b)))
        .collect())
}

/// RLC of many vectors: `Σ_i ρ^i · w_i`.
pub fn rlc_vectors(
    vectors: &[&[Goldilocks]],
    rho: &Goldilocks,
) -> Result<Vec<Goldilocks>, RlcError> {
    if vectors.is_empty() {
        return Err(RlcError::EmptyInput);
    }
    let len = vectors[0].len();
    for v in vectors {
        if v.len() != len {
            return Err(RlcError::LengthMismatch {
                expected: len,
                got: v.len(),
            });
        }
    }
    let powers = challenge_powers(rho, vectors.len());
    let mut out = vec![Goldilocks::ZERO; len];
    for (v, p) in vectors.iter().zip(powers.iter()) {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o = o.add(&p.mul(x));
        }
    }
    Ok(out)
}

/// RLC of scalars: `Σ_i ρ^i · c_i`.
pub fn rlc_scalars(scalars: &[Goldilocks], rho: &Goldilocks) -> Goldilocks {
    let powers = challenge_powers(rho, scalars.len());
    let mut acc = Goldilocks::ZERO;
    for (c, p) in scalars.iter().zip(powers.iter()) {
        acc = acc.add(&c.mul(p));
    }
    acc
}

/// Sample a folding challenge from the transcript after absorbing the
/// instances being folded (caller responsibility).
pub fn folding_challenge(
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<Goldilocks, RlcError> {
    transcript
        .challenge_field(label)
        .map_err(|_| RlcError::TranscriptFailure)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RlcError {
    LengthMismatch { expected: usize, got: usize },
    EmptyInput,
    TranscriptFailure,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fe(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn powers_correct() {
        let p = challenge_powers(&fe(3), 5);
        assert_eq!(p, vec![fe(1), fe(3), fe(9), fe(27), fe(81)]);
    }

    #[test]
    fn fold_witness_linearity() {
        let w1 = vec![fe(1), fe(2), fe(3)];
        let w2 = vec![fe(4), fe(5), fe(6)];
        let r = fe(10);
        // (w1 + r w2) with r=10: [41, 52, 63]
        assert_eq!(
            fold_witnesses(&w1, &w2, &r).ok().unwrap(),
            vec![fe(41), fe(52), fe(63)]
        );
        assert!(fold_witnesses(&w1, &[fe(1)], &r).is_err());
    }

    #[test]
    fn rlc_vectors_matches_manual() {
        let v1 = [fe(1), fe(2)];
        let v2 = [fe(3), fe(4)];
        let v3 = [fe(5), fe(6)];
        let rho = fe(2);
        let combined = rlc_vectors(&[&v1, &v2, &v3], &rho).ok().unwrap();
        // 1*v1 + 2*v2 + 4*v3 = [1+6+20, 2+8+24] = [27, 34]
        assert_eq!(combined, vec![fe(27), fe(34)]);
    }

    #[test]
    fn rlc_scalars_matches_manual() {
        assert_eq!(rlc_scalars(&[fe(1), fe(3), fe(5)], &fe(2)), fe(1 + 6 + 20));
    }

    #[test]
    fn folding_challenge_deterministic() {
        let mut t1 = Transcript::new_default(b"rlc-test");
        let mut t2 = Transcript::new_default(b"rlc-test");
        let c1 = folding_challenge(&mut t1, b"fold").ok().unwrap();
        let c2 = folding_challenge(&mut t2, b"fold").ok().unwrap();
        assert_eq!(c1, c2);
    }
}
