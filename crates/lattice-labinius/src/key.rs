//! The Ajtai commitment key and what a commitment leaves behind for the fold.
//! Port of `labinius` `key.rs`: `A` is one uniform row of `R_648` per modulus in
//! the NTT domain (centered i16, stored slot-major), and the commitment of one chunk is
//! `y = sum_i A_i * NTT(w_i)` — a pointwise inner product per limb.
//!
//! Two backends produce bit-identical commitments:
//! * the scalar reference path (the original port), and
//! * the AVX-512 backend ([`crate::simd`]) when the CPU has the feature set and every limb is
//!   one of `{3889, 9721, 2917, 4861, 12637}` (splitting primes above `2^14` and
//!   non-multiple-of-128 chunk lengths fall back to scalar) — the vertical-layout `A`, the
//!   batch-of-32 binary NTT kernels and the `vpmaddwd` raw-accumulation MAC of upstream,
//!   verified against the scalar path by `tests/simd.rs`.

use crate::binfield::{random_elems, F162};
use crate::params::{quadratic_slots, N};
use crate::ring::{components_of, Modulus, PowerOfThreeRing};
use crate::scalar::Coeffs;
use crate::simd::{commit as mac, Batch32};
use crate::simd::transpose::{slice_f162_into, BinaryIndex32};
use crate::binfield::Rng;
use std::sync::OnceLock;

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
    /// The same rows in the vertical batch-of-32 layout the AVX-512 kernels consume
    /// (`avx[k][b].v[j][p]` = slot `j` of `A_{32b+p}`), built once on first use.
    vertical: OnceLock<Vec<Vec<Batch32>>>,
    base: Modulus,
    additional: Vec<Modulus>,
    len_f162: usize,
}

/// Which commitment backend to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// Dispatch on CPU features and the limb list (the default of [`CommitmentKey::commit`]).
    Auto,
    /// The exact scalar reference path.
    Scalar,
    /// The AVX-512 batch-of-32 kernels with the `vpmaddwd` raw-accumulation MAC.
    Simd,
}

/// Everything a commitment leaves behind that the folding step needs: the witness's transform
/// modulo the base limb, and the raw 648-row commitments of the `r` chunks per limb.
#[derive(Clone)]
pub struct AuxData {
    /// The transform modulo the base limb, element-major: `batches[i][u]` = slot u of ring
    /// element i, fully reduced `u32` in `[0, q)`. Filled by the scalar backend; the AVX-512
    /// backend keeps the transform vertical instead (below). Public for the backend-equality
    /// tests and benchmarks.
    pub batches: Vec<Coeffs>,
    /// The transform modulo the base limb in the vertical `Batch32` layout the kernels
    /// produce — `vertical[c * bpc + b]` is batch `b` of chunk `c`, lanes at the binary
    /// kernel's declared output bound. Filled by the AVX-512 backend (no `store_transform`
    /// scatter); empty in the scalar backend.
    pub vertical: Vec<Batch32>,
    /// `raw[k][j]` = the 648-row commitment of chunk j for limb k, in `[0, q)`.
    pub raw: Vec<Vec<Coeffs>>,
    pub(crate) chunks: usize,
}

impl CommitmentKey {
    /// A uniformly random key for `len_f162` witness elements (a multiple of 4) over `base`
    /// and `additional`, deterministically from `seed` (upstream's xorshift64* `Rng`).
    pub fn random(len_f162: usize, seed: u64, base: Modulus, additional: &[Modulus]) -> Self {
        assert!(len_f162 > 0 && len_f162.is_multiple_of(4), "len_f162 must be a multiple of 4");
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
                            *c = rng.below(q as u32);
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
            vertical: OnceLock::new(),
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

    /// The vertical-layout `A` (built once, on first SIMD commitment).
    fn vertical(&self) -> &Vec<Vec<Batch32>> {
        self.vertical.get_or_init(|| {
            let nr = self.len_ring();
            let nb = nr / 32;
            (0..self.limbs())
                .map(|k| {
                    let mut batches =
                        vec![Batch32 { v: [[0i16; 32]; N] }; nb.max(1)];
                    for b in 0..nb {
                        for p in 0..32 {
                            let row = &self.a[k][32 * b + p];
                            for j in 0..N {
                                batches[b].v[j][p] = row[j];
                            }
                        }
                    }
                    batches
                })
                .collect()
        })
    }

    /// Can every limb run the AVX-512 backend? (All seven limb primes are ported.)
    fn simd_limbs(&self) -> bool {
        (0..self.limbs()).all(|k| {
            let q = self.prime(k);
            matches!(q, 3889 | 9721 | 17497 | 19441 | 2917 | 4861 | 12637)
        })
    }

    /// `A v` for limb `k` from centered coefficient batches (the verifier's recomputation):
    /// the vertical fast path (generic-input forward transform + packed-accumulator MAC against
    /// the vertical `A`) when the CPU has the feature set and the chunk length allows batches,
    /// the scalar reference otherwise.
    pub(crate) fn a_times_v_limb(&self, k: usize, v: &[[i16; N]]) -> Coeffs {
        let q = self.prime(k);
        if crate::simd::available() && self.len_f162.is_multiple_of(128) {
            let vert = &self.vertical()[k];
            crate::fold::a_times_v_forward(q, vert, v)
        } else {
            crate::fold::a_times_v_of(q, &self.a[k], v)
        }
    }

    /// Commit to `witness` in `r` chunks under the same key: AVX-512 backend when the CPU and
    /// the limb list allow it, the exact scalar reference otherwise.
    pub fn commit(&self, witness: &[F162], r: usize) -> (CommitmentMatrix, AuxData) {
        let backend = if crate::simd::available() && self.simd_limbs() && self.len_f162.is_multiple_of(128) {
            Backend::Simd
        } else {
            Backend::Scalar
        };
        self.commit_with(witness, r, backend)
    }

    /// Commit with an explicit backend: [`Backend::Scalar`] is always available, [`Backend::Simd`]
    /// panics on unsupported limb lists (primes above `2^14`) or CPU feature sets and falls back
    /// only for non-multiple-of-128 chunk lengths. Both produce bit-identical results
    /// (`tests/simd.rs`); the explicit form exists for the benchmark harness.
    pub fn commit_with(
        &self,
        witness: &[F162],
        r: usize,
        backend: Backend,
    ) -> (CommitmentMatrix, AuxData) {
        match backend {
            Backend::Scalar => self.commit_scalar(witness, r),
            Backend::Simd => {
                assert!(
                    crate::simd::available(),
                    "the AVX-512 PCS feature set is not available on this machine"
                );
                assert!(self.simd_limbs(), "the AVX-512 backend does not cover this limb list");
                if self.len_f162.is_multiple_of(128) {
                    self.commit_simd(witness, r)
                } else {
                    self.commit_scalar(witness, r)
                }
            }
            Backend::Auto => self.commit(witness, r),
        }
    }

    /// The AVX-512 commitment: one slicing pass and one kernel pass per limb per batch of 32
    /// ring elements (the index rows depend neither on `q` nor on the tree), each into the
    /// chunk's own accumulator, with the fold-backs on the compile-time periods. The base
    /// limb's transform is kept **vertical** (`aux.vertical`, the layout the kernels wrote —
    /// no `store_transform` scatter, non-temporal stores): the fold consumes it in place.
    fn commit_simd(&self, witness: &[F162], r: usize) -> (CommitmentMatrix, AuxData) {
        assert!(r.is_power_of_two() && r >= 2, "r must be a power of two >= 2");
        assert_eq!(witness.len(), r * self.len_f162);
        let nr = self.len_ring();
        let nb = nr / 32;
        assert!(nb > 0, "AVX-512 commit needs >= 32 ring elements per chunk");
        let mut aux = AuxData {
            batches: Vec::new(),
            vertical: vec![Batch32 { v: [[0i16; 32]; N] }; r * nb],
            raw: (0..self.limbs()).map(|_| Vec::with_capacity(r)).collect(),
            chunks: r,
        };
        let mut matrix = CommitmentMatrix {
            primes: (0..self.limbs()).map(|k| self.prime(k)).collect(),
            columns: r,
            value: vec![vec![Vec::new(); r]; 4],
        };
        // per-limb accumulators, reused chunk to chunk (cleared, not reallocated)
        enum LimbAcc {
            Split(Box<mac::Acc>),
            Quad(Box<mac::QuadAcc>),
        }
        let mut accs: Vec<LimbAcc> = (0..self.limbs())
            .map(|k| {
                if self.is_quadratic(k) {
                    LimbAcc::Quad(mac::QuadAcc::zero())
                } else {
                    LimbAcc::Split(mac::Acc::zero())
                }
            })
            .collect();
        let vertical = self.vertical();
        let mut idx = BinaryIndex32::zero();
        let mut out = Batch32 { v: [[0i16; 32]; N] };

        for c in 0..r {
            for b in 0..nb {
                let elems: &[F162; 128] = witness[c * self.len_f162 + 128 * b..]
                    [..128]
                    .try_into()
                    .unwrap();
                unsafe { slice_f162_into(elems, &mut idx) };
                for k in 0..self.limbs() {
                    let q = self.prime(k);
                    let av = &vertical[k][b];
                    let done = b + 1;
                    let is_base = k == 0;
                    // Safety: the AVX-512 PCS feature set was checked by `commit`.
                    unsafe {
                        match q {
                            3889 => {
                                const Q: u16 = 3889;
                                let LimbAcc::Split(acc) = &mut accs[k] else { unreachable!() };
                                mac::split_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            9721 => {
                                const Q: u16 = 9721;
                                let LimbAcc::Split(acc) = &mut accs[k] else { unreachable!() };
                                mac::split_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            17497 => {
                                const Q: u16 = 17497;
                                let LimbAcc::Split(acc) = &mut accs[k] else { unreachable!() };
                                mac::split_large_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            19441 => {
                                const Q: u16 = 19441;
                                let LimbAcc::Split(acc) = &mut accs[k] else { unreachable!() };
                                mac::split_large_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            2917 => {
                                const Q: u16 = 2917;
                                let LimbAcc::Quad(acc) = &mut accs[k] else { unreachable!() };
                                mac::quad_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            4861 => {
                                const Q: u16 = 4861;
                                let LimbAcc::Quad(acc) = &mut accs[k] else { unreachable!() };
                                mac::quad_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            12637 => {
                                const Q: u16 = 12637;
                                let LimbAcc::Quad(acc) = &mut accs[k] else { unreachable!() };
                                mac::quad_batch::<Q>(&idx, av, &mut out, acc, done);
                            }
                            _ => unreachable!("simd_limbs checked the prime list"),
                        }
                        if is_base {
                            // keep the transform vertical for the fold — no scatter, and the
                            // 41.5 KB per batch leaves through non-temporal stores (fenced
                            // once after the whole commitment)
                            mac::stream_copy_batch32(&mut aux.vertical[c * nb + b], &out);
                        }
                    }
                }
            }
            // finish every limb of this chunk
            for k in 0..self.limbs() {
                let q = self.prime(k);
                let yc = match (&mut accs[k], q) {
                    (LimbAcc::Split(acc), 3889) => mac::finish::<3889>(acc),
                    (LimbAcc::Split(acc), 9721) => mac::finish::<9721>(acc),
                    (LimbAcc::Split(acc), 17497) => mac::finish::<17497>(acc),
                    (LimbAcc::Split(acc), 19441) => mac::finish::<19441>(acc),
                    (LimbAcc::Quad(acc), 2917) => mac::finish_quad::<2917>(acc),
                    (LimbAcc::Quad(acc), 4861) => mac::finish_quad::<4861>(acc),
                    (LimbAcc::Quad(acc), 12637) => mac::finish_quad::<12637>(acc),
                    _ => unreachable!("simd_limbs checked the prime list"),
                };
                match &mut accs[k] {
                    LimbAcc::Split(acc) => acc.clear(),
                    LimbAcc::Quad(acc) => acc.clear(),
                }
                let comps = components_of(q, &yc);
                for row in 0..4 {
                    matrix.value[row][c].push(comps[row]);
                }
                aux.raw[k].push(yc);
            }
        }
        // the kept transform's non-temporal stores must be fenced before anyone reads them
        mac::sfence();
        (matrix, aux)
    }

    /// The scalar reference commitment (the original port's path, kept verbatim as the
    /// specification the AVX-512 backend is verified against).
    fn commit_scalar(&self, witness: &[F162], r: usize) -> (CommitmentMatrix, AuxData) {
        assert!(r.is_power_of_two() && r >= 2, "r must be a power of two >= 2");
        assert_eq!(witness.len(), r * self.len_f162);
        let nr = self.len_ring();
        let mut aux = AuxData {
            batches: vec![[0u32; N]; r * nr],
            vertical: Vec::new(),
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
