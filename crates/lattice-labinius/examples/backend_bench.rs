//! Backend benchmark: the AVX-512 path against the exact scalar reference, kernel by kernel
//! and end to end, at the reference-round scale (suite sizem: 2^18 F162, 128 columns,
//! 3889 + 2917).
//!
//! * `clmul` — the F162/B128 carry-less multiply primitive: `PCLMULQDQ` vs the software
//!   bitwise product (nine of these under every `F162` product: the eq tables, row
//!   evaluations, challenge sampling and fold parities).
//! * `ntt-split` — the split-tree forward transform of one batch of 32 binary ring elements:
//!   scalar `scalar::ntt` per element vs the `vpermb`-lookup + Montgomery-butterfly kernel.
//! * `ntt-quad` — the quadratic-slot tree of the same batch: scalar `scalar::ntt_quad` vs the
//!   folded-lookup kernel.
//! * `mac` — the Ajtai inner product over the batch: scalar pointwise / quadratic-slot
//!   products vs the `vpmaddwd` raw accumulator + `hsum8` + f64-exact `mod_q` finish.
//! * `commit` — the whole commitment of suite sizem (the reference round's dominant stage):
//!   `commit_with(Scalar)` vs `commit_with(Simd)`, the headline number.
//!
//! Everything the SIMD path prints is verified bit-equal by `tests/simd.rs`.

// loops index with strides; the lint's iterator forms do not apply
#![allow(clippy::needless_range_loop)]
use lattice_labinius::binfield::{random_elems, lift_elem, Rng};
use lattice_labinius::hw::{clmul64, clmul64_soft};
use lattice_labinius::key::{Backend, CommitmentKey};
use lattice_labinius::params::N;
use lattice_labinius::ring::Modulus;
use lattice_labinius::scalar::{mul_quad_slots, ntt, ntt_quad, pointwise_mul, Coeffs};
use lattice_labinius::simd::commit as mac;
use lattice_labinius::simd::ntt_quad;
use lattice_labinius::simd::ntt_small;
use lattice_labinius::simd::transpose::{slice_f162_into, BinaryIndex32};
use lattice_labinius::simd::Batch32;
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn bench<F: FnMut()>(mut f: F, runs: usize) -> f64 {
    let mut ts = vec![0f64; runs];
    for t in ts.iter_mut() {
        let start = Instant::now();
        f();
        *t = start.elapsed().as_secs_f64();
    }
    median(&mut ts)
}

fn main() {
    println!("lzx labinius backend bench (AVX-512 available: {})", lattice_labinius::simd::available());
    let runs = 5usize;

    // ------------------------------------------------------------------ clmul
    {
        let mut acc = 0u128;
        let hw = bench(
            || {
                for i in 0..100_000u64 {
                    acc ^= clmul64(0x9E37_79B9_7F4A_7C15 + i, 0x2545_F491_4F6C_DD1D - i);
                }
            },
            runs,
        );
        let sw = bench(
            || {
                for i in 0..100_000u64 {
                    acc ^= clmul64_soft(0x9E37_79B9_7F4A_7C15 + i, 0x2545_F491_4F6C_DD1D - i);
                }
            },
            runs,
        );
        println!("  clmul64 x100k       hw {:>9.3} ms   sw {:>9.3} ms   speedup {:>6.1}x   (acc {acc})",
            hw * 1e3, sw * 1e3, sw / hw);
    }

    // ------------------------------------------------------------------ kernels (one batch of 32)
    let elems = random_elems(128, 0xBEEF);
    let mut idx = BinaryIndex32::zero();
    let mut out = Batch32 { v: [[0i16; 32]; N] };
    unsafe { slice_f162_into(elems.as_slice().try_into().unwrap(), &mut idx) };

    // random A rows for the MAC, centered in the vertical layout
    let mut a_vert = Batch32 { v: [[0i16; 32]; N] };
    let mut a_rows: Vec<Coeffs> = Vec::with_capacity(32);
    {
        let mut rng = Rng::new(0xA11CE);
        for _p in 0..32 {
            let mut r = [0u32; N];
            for u in r.iter_mut() {
                *u = rng.below(3889);
            }
            a_rows.push(r);
        }
        for j in 0..N {
            for p in 0..32 {
                let c = a_rows[p][j] as i32;
                a_vert.v[j][p] = (if c > 3889 / 2 { c - 3889 } else { c }) as i16;
            }
        }
    }

    // split-tree kernel, q = 3889
    {
        let scalar = bench(
            || {
                for p in 0..32 {
                    let w = lift_elem(&elems, p);
                    let _ = ntt::<3889>(&w);
                }
            },
            runs,
        );
        let simd = bench(
            || unsafe { ntt_small::ntt_bin_batch32::<3889>(&idx, &mut out) },
            runs,
        );
        println!("  ntt split q=3889  batch32   scalar {:>9.3} ms   simd {:>9.3} ms   speedup {:>6.1}x",
            scalar * 1e3, simd * 1e3, scalar / simd);
    }
    // quad-tree kernel, q = 2917
    {
        let scalar = bench(
            || {
                for p in 0..32 {
                    let w = lift_elem(&elems, p);
                    let _ = ntt_quad::<2917>(&w);
                }
            },
            runs,
        );
        let simd = bench(
            || unsafe { ntt_quad::ntt_quad_bin_batch32::<2917>(&idx, &mut out) },
            runs,
        );
        println!("  ntt quad  q=2917  batch32   scalar {:>9.3} ms   simd {:>9.3} ms   speedup {:>6.1}x",
            scalar * 1e3, simd * 1e3, scalar / simd);
    }
    // the MAC over one batch, q = 3889 (scalar reference = pointwise products + mod)
    {
        let mut acc = mac::Acc::zero();
        let simd = bench(
            || unsafe {
                mac::mac_batch(
                    out.v.as_ptr() as *const i16,
                    a_vert.v.as_ptr() as *const i16,
                    acc.v.as_mut_ptr() as *mut i32,
                );
                mac::reduce_acc::<3889>(acc.v.as_mut_ptr() as *mut i32);
                let _ = mac::finish::<3889>(&acc);
            },
            runs,
        );
        let scalar = bench(
            || {
                let mut y = [0u64; N];
                for i in 0..32 {
                    let w = lift_elem(&elems, i);
                    let t = ntt::<3889>(&w);
                    let prod = pointwise_mul(&a_rows[i], &t, 3889);
                    for u in 0..N {
                        y[u] += prod[u] as u64;
                    }
                }
                let mut yc = [0u32; N];
                for u in 0..N {
                    yc[u] = (y[u] % 3889) as u32;
                }
            },
            runs,
        );
        println!("  mac+finish q=3889 batch32   scalar {:>9.3} ms   simd {:>9.3} ms   speedup {:>6.1}x  (mac-only simd {:>7.3} ms)",
            scalar * 1e3, simd * 1e3, scalar / simd, simd * 1e3);
    }
    // quad MAC over one batch, q = 2917
    {
        let mut acc = mac::QuadAcc::zero();
        let simd = bench(
            || unsafe {
                mac::mac_quad_batch::<2917>(
                    out.v.as_ptr() as *const i16,
                    a_vert.v.as_ptr() as *const i16,
                    acc.p01.as_mut_ptr() as *mut i32,
                    acc.p2.as_mut_ptr() as *mut i32,
                );
                mac::reduce_quad_acc::<2917>(&mut acc);
                let _ = mac::finish_quad::<2917>(&acc);
            },
            runs,
        );
        let scalar = bench(
            || {
                let mut y = [0u64; N];
                for i in 0..32 {
                    let w = lift_elem(&elems, i);
                    let t = ntt_quad::<2917>(&w);
                    let prod = mul_quad_slots::<2917>(&a_rows[i], &t);
                    for u in 0..N {
                        y[u] += prod[u] as u64;
                    }
                }
                let mut yc = [0u32; N];
                for u in 0..N {
                    yc[u] = (y[u] % 2917) as u32;
                }
            },
            runs,
        );
        println!("  mac+finish q=2917 batch32   scalar {:>9.3} ms   simd {:>9.3} ms   speedup {:>6.1}x",
            scalar * 1e3, simd * 1e3, scalar / simd);
    }

    // ------------------------------------------------------- full commitment, suite sizem
    // 2^18 F162 in 128 columns: len_f162 = 2048 per chunk, r = 128 chunks.
    {
        let len_f162 = 2048usize;
        let r = 128usize;
        println!("  commit sizem (2^18 F162, {r} columns, 3889+2917)");
        let key = CommitmentKey::random(len_f162, 0x5EED_C0DE, Modulus::Q3889_FS_S, &[Modulus::Q2917_Q_S]);
        let witness = random_elems(len_f162 * r, 0x5EED_CAFE);
        println!("    A: {} bytes", key.bytes());
        // warm the vertical layout + one pass so the SIMD numbers are steady-state
        let _ = key.commit_with(&witness, r, Backend::Simd);
        let a_vert_bytes = key.bytes();
        let scalar = bench(
            || {
                let _ = key.commit_with(&witness, r, Backend::Scalar);
            },
            2,
        );
        let simd = bench(
            || {
                let _ = key.commit_with(&witness, r, Backend::Simd);
            },
            runs,
        );
        println!("    scalar {:>9.3} ms   simd {:>9.3} ms   speedup {:>6.1}x   (vertical A {a_vert_bytes} B)",
            scalar * 1e3, simd * 1e3, scalar / simd);
        // smaller suites for the ratio curve
        for &(len, rr) in &[(512usize, 16usize), (1024usize, 32usize)] {
            let key = CommitmentKey::random(len, 0x5EED_C0DE, Modulus::Q3889_FS_S, &[Modulus::Q2917_Q_S]);
            let witness = random_elems(len * rr, 0x5EED_CAFE);
            let _ = key.commit_with(&witness, rr, Backend::Simd);
            let scalar = bench(|| { let _ = key.commit_with(&witness, rr, Backend::Scalar); }, 2);
            let simd = bench(|| { let _ = key.commit_with(&witness, rr, Backend::Simd); }, runs);
            println!("    2^{} F162 in {rr} columns: scalar {:>9.3} ms   simd {:>9.3} ms   speedup {:>6.1}x",
                (len * rr).trailing_zeros(), scalar * 1e3, simd * 1e3, scalar / simd);
        }
    }
}
