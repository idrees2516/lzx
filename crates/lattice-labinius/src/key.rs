//! The Ajtai commitment key and what a commitment leaves behind for the fold.
//! Port of `labinius` `key.rs`, scalar path: `A` is one uniform row of `R_648` per modulus in
//! the NTT domain (centered i16, stored slot-major), and the commitment of one chunk is
//! `y = sum_i A_i * NTT(w_i)` — a pointwise inner product per limb.

use crate::binfield::{random_elems, F162};
use crate::params::{quadratic_slots, N};
use crate::ring::{components_of, Modulus, PowerOfThreeRing};
use crate::scalar::{pointwise_mul, Coeffs};
use crate::binfield::Rng;

/// A `4 x r` matrix of `R_162` slot elements, one per modulus: the public commitment.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct CommitmentMatrix {
    pub primes: Vec<u16>,
    pub columns: usize,
    /// `value[row][column][modulus]`.
    pub value: Vec<Vec<Vec<PowerOfThreeRing>>>,
}

impl CommitmentMatrix {
    pub fn element(&self, row: usize, column: usize, modulus: usize) -> &PowerOfThreeRing {
        &self.value[row][column][modulus]
    }
    pub fn rows(&self) -> usize {
        4
    }
    pub fn cols(&self) -> usize {
        self.columns
    }
}

/// The Ajtai matrix `A`: for limb `k`, `a[k][i]` is the NTT of `A_i` modulo `prime(k)`,
/// centered, stored as `[i16; N]`.
#[derive(Clone)]
pub struct CommitmentKey {
    pub(crate) a: Vec<Vec<[i16; N]>>,
    base: Modulus,
    additional: Vec<Modulus>,
    len_f162: usize,
}

/// Everything a commitment leaves behind that the folding step needs: the witness's transform
/// modulo the base limb, and the raw 648-row commitments of the `r` chunks per limb.
#[derive(Clone)]
pub struct AuxData {
    /// The transform modulo the base limb: `batches[i][u]` = slot u of ring element i, fully
    /// reduced `u32` in `[0, q)`.
    pub(crate) batches: Vec<Coeffs>,
    /// `raw[k][j]` = the 648-row commitment of chunk j for limb k, in `[0, q)`.
    pub raw: Vec<Vec<Coeffs>>,
    pub(crate) chunks: usize,
}

impl CommitmentKey {
    /// A uniformly random key for `len_f162` witness elements (a multiple of 4) over `base`
    /// and `additional`, deterministically from `seed` (upstream's xorshift64* `Rng`).
    pub fn random(len_f162: usize, seed: u64, base: Modulus, additional: &[Modulus]) -> Self {
        assert!(len_f162 > 0 && len_f162 % 4 == 0, "len_f162 must be a multiple of 4");
        let mut seen = vec![base];
        for l in additional {
            assert!(!seen.contains(l), "the limb {l:?} is listed twice");
            seen.push(*l);
        }
        let primes: Vec<u16> = core::iter::once(base.prime())
            .chain(additional.iter().map(|l| l.prime()))
            .collect();
        let nr = len_f162 / 4;
        let a = primes
            .iter()
            .enumerate()
            .map(|(k, &q)| {
                let half = ((q - 1) / 2) as i32;
                let mut rng = Rng::new(seed ^ (0x9E37_79B9_u64.wrapping_mul(k as u64 + 1)));
                (0..nr)
                    .map(|i| {
                        // uniform centered coefficients, then forward NTT of the lift of a
                        // fresh uniform ring element: A_i must be a uniform R_648 element in
                        // the NTT domain, which is slot-wise uniform.
                        let mut coeffs = [0u32; N];
                        for c in coeffs.iter_mut() {
                            *c = rng.below(q as u32) as u32;
                        }
                        let t = crate::ring::ntt_of(q, &coeffs);
                        let mut out = [0i16; N];
                        for u in 0..N {
                            out[u] = if t[u] as i32 > half {
                                t[u] as i32 - q as i32
                            } else {
                                t[u] as i32
                            } as i16;
                        }
                        let _ = i;
                        out
                    })
                    .collect()
            })
            .collect();
        CommitmentKey {
            a,
            base,
            additional: additional.to_vec(),
            len_f162,
        }
    }

    pub fn len_f162(&self) -> usize {
        self.len_f162
    }
    pub fn len_ring(&self) -> usize {
        self.len_f162 / 4
    }
    pub fn limbs(&self) -> usize {
        self.a.len()
    }
    pub fn base(&self) -> Modulus {
        self.base
    }
    pub fn additional(&self) -> &[Modulus] {
        &self.additional
    }
    pub fn limb(&self, k: usize) -> Modulus {
        if k == 0 {
            self.base
        } else {
            self.additional[k - 1]
        }
    }
    pub fn prime(&self, k: usize) -> u16 {
        self.limb(k).prime()
    }
    pub fn is_quadratic(&self, k: usize) -> bool {
        quadratic_slots(self.prime(k))
    }

    /// Bytes of `A` held, over all limbs.
    pub fn bytes(&self) -> usize {
        self.limbs() * self.len_ring() * 2 * N
    }

    /// Commit to `witness` in `r` chunks under the same key. Chunk `c` is
    /// `witness[c*len .. (c+1)*len]`, read as `len/4` binary ring elements, transformed once,
    /// and multiplied into `y = sum_i A_i * NTT(w_i)` per limb (pointwise). Each limb's `y` is
    /// decomposed into its four `R_162` components, which become column `c` of the matrix.
    pub fn commit(
        &self,
        witness: &[F162],
        r: usize,
    ) -> (CommitmentMatrix, AuxData) {
        assert!(r.is_power_of_two() && r >= 2, "r must be a power of two >= 2");
        assert_eq!(witness.len(), r * self.len_f162);
        let nr = self.len_ring();
        let mut aux = AuxData {
            batches: vec![[0u32; N]; r * nr],
            raw: (0..self.limbs()).map(|_| Vec::with_capacity(r)).collect(),
            chunks: r,
        };
        let mut matrix = CommitmentMatrix {
            primes: (0..self.limbs()).map(|k| self.prime(k)).collect(),
            columns: r,
            value: vec![vec![Vec::new(); r]; 4],
        };
        for k in 0..self.limbs() {
            let q = self.prime(k);
            for c in 0..r {
                let mut y = [0u64; N];
                for i in 0..nr {
                    let w = crate::binfield::lift_elem(&witness[c * self.len_f162..], i);
                    let t = crate::ring::ntt_of(q, &w);
                    if k == 0 {
                        aux.batches[c * nr + i] = t;
                    }
                    let arow = &self.a[k][i];
                    if crate::params::quadratic_slots(q) {
                        // quadratic limb: the leaf product is bilinear, not pointwise
                        let mut acoef = [0u32; N];
                        for u in 0..N {
                            acoef[u] = (arow[u].rem_euclid(q as i16) as i64 as u32) % q as u32;
                        }
                        let prod = match q {
                            2917 => crate::scalar::mul_quad_slots::<2917>(&acoef, &t),
                            4861 => crate::scalar::mul_quad_slots::<4861>(&acoef, &t),
                            12637 => crate::scalar::mul_quad_slots::<12637>(&acoef, &t),
                            _ => unreachable!(),
                        };
                        for u in 0..N {
                            y[u] += prod[u] as u64;
                        }
                    } else {
                        for u in 0..N {
                            y[u] += (arow[u].rem_euclid(q as i16) as i64 as u64) * t[u] as u64;
                        }
                    }
                }
                let mut yc = [0u32; N];
                for u in 0..N {
                    yc[u] = (y[u] % q as u64) as u32;
                }
                let comps = components_of(q, &yc);
                for row in 0..4 {
                    matrix.value[row][c].push(comps[row]);
                }
                aux.raw[k].push(yc);
            }
        }
        (matrix, aux)
    }
}

/// A uniform random witness of `n` F162 elements from a seed.
pub fn random_witness(n: usize, seed: u64) -> Vec<F162> {
    random_elems(n, seed)
}
