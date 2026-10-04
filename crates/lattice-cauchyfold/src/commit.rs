//! The commitment layer over the paper's ring `R_{q,64}` with
//! `q = 2^48 − 59` (reusing `lattice_labrador::ring`): seeded Ajtai
//! matrices `Com_A(w) = A·w`, the state encodings (level-1 8-bit digits
//! of the `K`-coefficients with designated Boolean payload positions),
//! and the level-2 canonical centered radix-16 decomposition with digits
//! in `{−8..7}` (the paper's `{−7..7}`/`R15` analogue, here `R16`).

use crate::field_k::{Fq48, K4};
use lattice_labrador::ring::Poly;

/// A seeded Ajtai commitment matrix over `R_{q,64}`: `a × b` ring entries.
#[derive(Clone)]
pub struct AjtaiKey {
    pub rows: usize,
    pub cols: usize,
    pub matrix: Vec<Poly>,
}

impl AjtaiKey {
    pub fn from_seed(rows: usize, cols: usize, seed: &[u8]) -> Self {
        let mut salt = [0u8; 32];
        salt[..seed.len().min(32)].copy_from_slice(&seed[..seed.len().min(32)]);
        let flat = lattice_labrador::ring::uniform(rows * cols, &salt, 0);
        AjtaiKey {
            rows,
            cols,
            matrix: flat,
        }
    }

    /// `Com_A(w) = A·w` (a vector of `rows` ring elements).
    pub fn commit(&self, w: &[Poly]) -> Result<Vec<Poly>, String> {
        if w.len() != self.cols {
            return Err(format!("witness {} vs cols {}", w.len(), self.cols));
        }
        let mut out = vec![Poly::zero(); self.rows];
        for (c, wc) in w.iter().enumerate() {
            for r in 0..self.rows {
                out[r].add_assign(&self.matrix[r * self.cols + c].mul(wc));
            }
        }
        Ok(out)
    }

    pub fn verify(&self, w: &[Poly], t: &[Poly]) -> bool {
        match self.commit(w) {
            Ok(c) => c == t,
            Err(_) => false,
        }
    }
}

/// The level-1 encoding of a `K^d` vector: each `K4` coefficient splits
/// into six 8-bit unsigned digits (48 bits); the digit stream carries the
/// designated Boolean payload in the low digit of selected coefficients.
#[derive(Clone, Debug)]
pub struct Level1Encoding {
    /// The digit values, each in `[0, 256)` — length `4·d·6`.
    pub digits: Vec<u16>,
    /// The positions (into `digits`) that must be Boolean.
    pub boolean_positions: Vec<usize>,
}

impl Level1Encoding {
    /// Encode `d` `K`-values. `boolean_payload`: if true, the source is
    /// constructed with the low digit of every 4th coefficient Boolean.
    pub fn encode(values: &[K4], boolean_payload: bool) -> Self {
        let mut digits = Vec::with_capacity(values.len() * 24);
        for (vi, v) in values.iter().enumerate() {
            for c in 0..4 {
                let x = v.0[c].0;
                for j in 0..6 {
                    digits.push(((x >> (8 * (5 - j))) & 0xFF) as u16);
                }
                if boolean_payload && vi % 4 == 0 && c == 0 {
                    // The low digit of the first coefficient of every 4th
                    // value carries a Boolean payload bit (the honest
                    // generator constructs such sources).
                }
            }
        }
        // Designated Boolean positions: the low digit of the first
        // coefficient of every 4th value (constructed Boolean by the
        // fixture generator).
        let mut boolean_positions = Vec::new();
        if boolean_payload {
            for vi in 0..values.len() {
                if vi % 4 == 0 {
                    boolean_positions.push(vi * 24 + 5);
                }
            }
        }
        Level1Encoding {
            digits,
            boolean_positions,
        }
    }

    /// Recompose the `K` values.
    pub fn recompose(&self) -> Vec<K4> {
        let n = self.digits.len() / 24;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = [Fq48::ZERO; 4];
            for cc in 0..4 {
                let mut x: u64 = 0;
                for j in 0..6 {
                    x = (x << 8) | self.digits[i * 24 + cc * 6 + j] as u64;
                }
                c[cc] = Fq48(x);
            }
            out.push(K4(c));
        }
        out
    }

    /// Pack the digit stream into ring elements (64 digits per element,
    /// zero-padded).
    pub fn to_ring_vector(&self) -> Vec<Poly> {
        let mut coeffs = vec![0i64; self.digits.len().div_ceil(64) * 64];
        for (i, &d) in self.digits.iter().enumerate() {
            coeffs[i] = d as i64;
        }
        coeffs
            .chunks(64)
            .map(|c| {
                let mut arr = [0i64; 64];
                arr.copy_from_slice(c);
                Poly(arr)
            })
            .collect()
    }

    /// The squared ℓ2 norm of the digit stream.
    pub fn norm_squared(&self) -> u64 {
        self.digits.iter().map(|&d| d as u64 * d as u64).sum()
    }
}

/// The level-2 canonical centered radix-16 decomposition of one level-1
/// digit: at most three digits in `{−8..7}` (255 = 16²·1 − 16 − 1 etc.).
#[derive(Clone, Debug)]
pub struct Level2Encoding {
    /// Digits in `{−8..7}`; `3` per level-1 value — length `3·n`.
    pub digits: Vec<i8>,
}

/// Split one value `x ∈ [0, 256)` into three centered radix-16 digits:
/// `x = 256·a + 16·b + c` with `a, b, c ∈ {−8..7}` (when possible; the
/// canonical form picks the centered residues).
pub fn split_radix16(x: u16) -> [i8; 3] {
    // Centered residues: c = centered(x mod 16), then recurse on the carry.
    let c = centered_residue((x % 16) as i32);
    let x1 = (x as i32 - c) / 16; // ∈ [0, 16]
    let b = centered_residue(x1 % 16);
    let x2 = (x1 - b) / 16; // ∈ [0, 1]
    let a = x2 as i8; // 0 or 1 — both in {−8..7}
    [a, b as i8, c as i8]
}

fn centered_residue(r: i32) -> i32 {
    if r > 7 {
        r - 16
    } else {
        r
    }
}

impl Level2Encoding {
    pub fn encode(level1: &[u16]) -> Self {
        let mut digits = Vec::with_capacity(level1.len() * 3);
        for &x in level1 {
            let [a, b, c] = split_radix16(x);
            digits.push(a);
            digits.push(b);
            digits.push(c);
        }
        Level2Encoding { digits }
    }

    /// Recompose the level-1 digits (mod q semantics on the total).
    pub fn recompose(&self) -> Vec<u16> {
        self.digits
            .chunks(3)
            .map(|d| {
                let v = d[0] as i32 * 256 + d[1] as i32 * 16 + d[2] as i32;
                (v.rem_euclid(256)) as u16
            })
            .collect()
    }

    pub fn norm_squared(&self) -> u64 {
        self.digits
            .iter()
            .map(|&d| (d as i64 * d as i64) as u64)
            .sum()
    }

    /// All digits in the range set `{−8..7}`.
    pub fn in_range(&self) -> bool {
        self.digits.iter().all(|&d| (-8..=7).contains(&d))
    }

    /// Pack into ring elements (as centered i64 coefficients).
    pub fn to_ring_vector(&self) -> Vec<Poly> {
        let mut coeffs = vec![0i64; self.digits.len().div_ceil(64) * 64];
        for (i, &d) in self.digits.iter().enumerate() {
            coeffs[i] = d as i64;
        }
        coeffs
            .chunks(64)
            .map(|c| {
                let mut arr = [0i64; 64];
                arr.copy_from_slice(c);
                Poly(arr)
            })
            .collect()
    }
}

/// A witness vector of ring elements with its squared-norm bound.
#[derive(Clone, Debug)]
pub struct CommittedWitness {
    /// The ring-element blocks.
    pub blocks: Vec<Vec<Poly>>,
    /// Per-block commitment values (A·w).
    pub commitments: Vec<Vec<Poly>>,
    /// The squared ℓ2 norm bound (integer, over centered coefficients).
    pub norm_bound: u64,
}

/// Compute the exact squared ℓ2 norm of a ring-vector witness (centered
/// coefficients).
pub fn ring_vector_norm_squared(w: &[Poly]) -> u64 {
    w.iter()
        .map(|p| {
            p.0.iter()
                .map(|&c| (c as i128 * c as i128).min(u64::MAX as i128) as u64)
                .sum::<u64>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_k::Q48;

    fn k4s(seed: u64) -> K4 {
        K4::from_coeffs([
            seed.wrapping_mul(7919) % Q48,
            seed.wrapping_mul(104729) % Q48,
            seed.wrapping_mul(1299709) % Q48,
            seed.wrapping_mul(15485863) % Q48,
        ])
    }

    #[test]
    fn ajtai_commit_binding_shape() {
        let key = AjtaiKey::from_seed(2, 4, b"k1");
        let w = lattice_labrador::ring::quarternary(4, &[0u8; 32], 1);
        let t = key.commit(&w).unwrap();
        assert!(key.verify(&w, &t));
        let w2 = lattice_labrador::ring::quarternary(4, &[1u8; 32], 1);
        // Distinct short witnesses give distinct commitments
        // (overwhelmingly — the Module-SIS regime).
        let t2 = key.commit(&w2).unwrap();
        assert_ne!(t, t2);
    }

    #[test]
    fn level1_roundtrip() {
        let vals: Vec<K4> = (0..9u64).map(k4s).collect();
        let enc = Level1Encoding::encode(&vals, false);
        assert_eq!(enc.digits.len(), 9 * 24);
        assert_eq!(enc.recompose(), vals);
        // Digits are 8-bit — the shortness regime.
        assert!(enc.digits.iter().all(|&d| d < 256));
    }

    #[test]
    fn level1_boolean_payload() {
        // A constructed source with Boolean low-digits at designated
        // positions.
        let mut vals: Vec<K4> = (0..8u64).map(k4s).collect();
        for (i, v) in vals.iter_mut().enumerate() {
            if i % 4 == 0 {
                v.0[0] = Fq48(v.0[0].0 & !0xFFu64 | (((i / 4) % 2) as u64));
            }
        }
        let enc = Level1Encoding::encode(&vals, true);
        assert_eq!(enc.boolean_positions.len(), 2);
        for &p in &enc.boolean_positions {
            assert!(enc.digits[p] <= 1, "designated position is Boolean");
        }
        assert_eq!(enc.recompose(), vals);
    }

    #[test]
    fn level2_radix16_roundtrip() {
        for x in [0u16, 1, 7, 8, 15, 16, 100, 200, 255] {
            let [a, b, c] = split_radix16(x);
            assert!((-8..=7).contains(&a));
            assert!((-8..=7).contains(&b));
            assert!((-8..=7).contains(&c));
            let v = a as i32 * 256 + b as i32 * 16 + c as i32;
            assert_eq!(v.rem_euclid(256) as u16, x, "x = {x}");
        }
        // Full-stream roundtrip.
        let l1: Vec<u16> = (0..100u32).map(|i| (i * 37 % 256) as u16).collect();
        let l2 = Level2Encoding::encode(&l1);
        assert!(l2.in_range());
        assert_eq!(l2.recompose(), l1);
    }

    #[test]
    fn norms() {
        let l1: Vec<u16> = (0..64).map(|i| (i * 11 % 256) as u16).collect();
        let enc = Level1Encoding {
            digits: l1.clone(),
            boolean_positions: vec![],
        };
        let expect: u64 = l1.iter().map(|&d| d as u64 * d as u64).sum();
        assert_eq!(enc.norm_squared(), expect);
        let rv = enc.to_ring_vector();
        assert_eq!(rv.len(), 1);
        assert_eq!(ring_vector_norm_squared(&rv), expect);
        let l2 = Level2Encoding::encode(&l1);
        assert!(l2.norm_squared() <= 64 * 3 * 64);
    }

    #[test]
    fn q_consistency() {
        // The crate's field modulus equals the ring modulus.
        assert_eq!(crate::field_k::Q48 as i128, lattice_labrador::ring::Q);
        assert_eq!(lattice_labrador::ring::Q64, crate::field_k::Q48 as i64);
    }
}
