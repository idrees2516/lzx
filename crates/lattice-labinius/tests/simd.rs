//! The AVX-512 backend against the exact scalar reference: every kernel, every supported
//! prime, random inputs, bit-level equality of the commitments.
//!
//! These tests are the port's contract with itself: the SIMD path does not get to be *close*,
//! it has to produce the same fully-reduced values as the scalar path the whole protocol was
//! verified on. They skip silently on machines without the feature set.

// loops index with strides; the lint's iterator forms do not apply
#![allow(clippy::needless_range_loop)]
use lattice_labinius::binfield::{lift_elem, random_elems, F162};
use lattice_labinius::key::{Backend, CommitmentKey};
use lattice_labinius::params::{quadratic_slots, N};
use lattice_labinius::ring::Modulus;
use lattice_labinius::scalar::{ntt, ntt_quad, Coeffs};
use lattice_labinius::simd::commit as mac;
use lattice_labinius::simd::commit::{
    finish, finish_quad, Acc, QuadAcc, ACC_VECS, QBLOCKS, QACC01_PER_BLK, QACC2_PER_BLK,
};
use lattice_labinius::simd::ntt_quad;
use lattice_labinius::simd::ntt_small;
use lattice_labinius::simd::transpose::{slice_f162_into, BinaryIndex32};
use lattice_labinius::simd::Batch32;

fn gate() -> bool {
    if lattice_labinius::simd::available() {
        true
    } else {
        eprintln!("skipping: no AVX-512 PCS feature set on this machine");
        false
    }
}

/// The scalar reference of one batch: 32 ring elements, fully reduced `[u32; N]` each.
fn scalar_batch(elems: &[F162], q: u16) -> Vec<Coeffs> {
    (0..32)
        .map(|p| {
            let w = lift_elem(elems, p);
            if quadratic_slots(q) {
                match q {
                    2917 => ntt_quad::<2917>(&w),
                    4861 => ntt_quad::<4861>(&w),
                    _ => ntt_quad::<12637>(&w),
                }
            } else {
                match q {
                    3889 => ntt::<3889>(&w),
                    _ => ntt::<9721>(&w),
                }
            }
        })
        .collect()
}

/// The kernel lane vs the scalar slot value, modulo q, plus the declared bound.
fn check_kernel_output(out: &Batch32, reference: &[Coeffs], q: u16, bound: i32) {
    for (p, ref_p) in reference.iter().enumerate() {
        for (j, &want) in ref_p.iter().enumerate() {
            let lane = out.v[j][p] as i32;
            let want = want as i32;
            assert_eq!(
                lane.rem_euclid(q as i32),
                want,
                "q={q} element {p} slot {j}: SIMD lane {lane} != scalar {want}"
            );
            assert!(
                lane.abs() <= bound,
                "q={q} element {p} slot {j}: |lane| {lane} exceeds declared bound {bound}"
            );
        }
    }
}

fn test_bin_small(q: u16, seed: u64) {
    if !gate() {
        return;
    }
    let elems = random_elems(128, seed);
    let elems: &[F162; 128] = elems.as_slice().try_into().unwrap();
    let reference = scalar_batch(elems, q);
    let mut idx = BinaryIndex32::zero();
    let mut out = Batch32 { v: [[0i16; 32]; N] };
    unsafe {
        slice_f162_into(elems, &mut idx);
        match q {
            3889 => ntt_small::ntt_bin_batch32::<3889>(&idx, &mut out),
            _ => ntt_small::ntt_bin_batch32::<9721>(&idx, &mut out),
        }
    }
    let bound = (ntt_small::output_bound_milli_q(q) as i64 * q as i64 / 1000) as i32;
    check_kernel_output(&out, &reference, q, bound);
}

#[test]
fn bin_small_3889_matches_scalar() {
    test_bin_small(3889, 0x5EED_0001);
}

#[test]
fn bin_small_9721_matches_scalar() {
    test_bin_small(9721, 0x5EED_0002);
}

fn test_bin_quad(q: u16, seed: u64) {
    if !gate() {
        return;
    }
    let elems = random_elems(128, seed);
    let elems: &[F162; 128] = elems.as_slice().try_into().unwrap();
    let reference = scalar_batch(elems, q);
    let mut idx = BinaryIndex32::zero();
    let mut out = Batch32 { v: [[0i16; 32]; N] };
    unsafe {
        slice_f162_into(elems, &mut idx);
        match q {
            2917 => ntt_quad::ntt_quad_bin_batch32::<2917>(&idx, &mut out),
            4861 => ntt_quad::ntt_quad_bin_batch32::<4861>(&idx, &mut out),
            _ => ntt_quad::ntt_quad_bin_batch32::<12637>(&idx, &mut out),
        }
    }
    check_kernel_output(&out, &reference, q, ntt_quad::output_bound(q));
}

#[test]
fn bin_quad_2917_matches_scalar() {
    test_bin_quad(2917, 0x5EED_0011);
}

#[test]
fn bin_quad_4861_matches_scalar() {
    test_bin_quad(4861, 0x5EED_0012);
}

#[test]
fn bin_quad_12637_matches_scalar() {
    test_bin_quad(12637, 0x5EED_0013);
}

/// The accumulator machinery end to end: kernel + MAC + finish must equal the scalar pointwise
/// inner product over the same A rows and the same lifted witness.
fn test_mac_against_scalar(q: u16, seed: u64) {
    if !gate() {
        return;
    }
    let nb = 4; // 4 batches = 128 ring elements
    let elems = random_elems(128 * nb, seed);
    // a random centered A row per element, in the vertical layout
    let mut a_vert = vec![Batch32 { v: [[0i16; 32]; N] }; nb];
    let mut a_rows: Vec<Coeffs> = Vec::with_capacity(32 * nb);
    {
        use lattice_labinius::binfield::Rng;
        let mut rng = Rng::new(seed ^ 0xA11CE);
        for av in a_vert.iter_mut().take(nb) {
            for p in 0..32 {
                let mut r = [0u32; N];
                for x in r.iter_mut() {
                    *x = rng.below(q as u32);
                }
                for (j, &c) in r.iter().enumerate() {
                    let c = c as i32;
                    let centered = if c > q as i32 / 2 { c - q as i32 } else { c };
                    av.v[j][p] = centered as i16;
                }
                a_rows.push(r);
            }
        }
    }
    // the scalar reference: y = sum_i A_i * NTT(w_i) mod q (quadratic leaf products on the
    // quad tree, pointwise on the splitting tree), exactly as `commit_scalar` computes it
    let mut y_scalar = [0u64; N];
    for i in 0..32 * nb {
        let w = lift_elem(&elems, i);
        let t = if quadratic_slots(q) {
            match q {
                2917 => ntt_quad::<2917>(&w),
                4861 => ntt_quad::<4861>(&w),
                _ => ntt_quad::<12637>(&w),
            }
        } else {
            match q {
                3889 => ntt::<3889>(&w),
                _ => ntt::<9721>(&w),
            }
        };
        if quadratic_slots(q) {
            let prod = match q {
                2917 => lattice_labinius::scalar::mul_quad_slots::<2917>(&a_rows[i], &t),
                4861 => lattice_labinius::scalar::mul_quad_slots::<4861>(&a_rows[i], &t),
                _ => lattice_labinius::scalar::mul_quad_slots::<12637>(&a_rows[i], &t),
            };
            for j in 0..N {
                y_scalar[j] += prod[j] as u64;
            }
        } else {
            for j in 0..N {
                let a = a_rows[i][j] as u64;
                y_scalar[j] += a * t[j] as u64;
            }
        }
    }
    let mut y = [0u32; N];
    for j in 0..N {
        y[j] = (y_scalar[j] % q as u64) as u32;
    }

    // the SIMD path, batch by batch
    let mut idx = BinaryIndex32::zero();
    let mut out = Batch32 { v: [[0i16; 32]; N] };
    let run = |idx: &BinaryIndex32, a: &Batch32, out: &mut Batch32| unsafe {
        match q {
            3889 => {
                let mut acc = Acc::zero();
                ntt_small::ntt_bin_batch32::<3889>(idx, out);
                mac::mac_batch(
                    out.v.as_ptr() as *const i16,
                    a.v.as_ptr() as *const i16,
                    acc.v.as_mut_ptr() as *mut i32,
                );
                mac::reduce_acc::<3889>(acc.v.as_mut_ptr() as *mut i32);
                finish::<3889>(&acc)
            }
            9721 => {
                let mut acc = Acc::zero();
                ntt_small::ntt_bin_batch32::<9721>(idx, out);
                mac::mac_batch(
                    out.v.as_ptr() as *const i16,
                    a.v.as_ptr() as *const i16,
                    acc.v.as_mut_ptr() as *mut i32,
                );
                mac::reduce_acc::<9721>(acc.v.as_mut_ptr() as *mut i32);
                finish::<9721>(&acc)
            }
            2917 => {
                let mut acc = QuadAcc::zero();
                ntt_quad::ntt_quad_bin_batch32::<2917>(idx, out);
                mac::mac_quad_batch::<2917>(
                    out.v.as_ptr() as *const i16,
                    a.v.as_ptr() as *const i16,
                    acc.p01.as_mut_ptr() as *mut i32,
                    acc.p2.as_mut_ptr() as *mut i32,
                );
                mac::reduce_quad_acc::<2917>(&mut acc);
                finish_quad::<2917>(&acc)
            }
            4861 => {
                let mut acc = QuadAcc::zero();
                ntt_quad::ntt_quad_bin_batch32::<4861>(idx, out);
                mac::mac_quad_batch::<4861>(
                    out.v.as_ptr() as *const i16,
                    a.v.as_ptr() as *const i16,
                    acc.p01.as_mut_ptr() as *mut i32,
                    acc.p2.as_mut_ptr() as *mut i32,
                );
                mac::reduce_quad_acc::<4861>(&mut acc);
                finish_quad::<4861>(&acc)
            }
            _ => {
                let mut acc = QuadAcc::zero();
                ntt_quad::ntt_quad_bin_batch32::<12637>(idx, out);
                mac::mac_quad_batch::<12637>(
                    out.v.as_ptr() as *const i16,
                    a.v.as_ptr() as *const i16,
                    acc.p01.as_mut_ptr() as *mut i32,
                    acc.p2.as_mut_ptr() as *mut i32,
                );
                mac::reduce_quad_acc::<12637>(&mut acc);
                finish_quad::<12637>(&acc)
            }
        }
    };
    let mut y_simd = [0u32; N];
    for b in 0..nb {
        let elems_b: Box<[F162; 128]> = elems[128 * b..128 * b + 128].to_vec().try_into().unwrap();
        unsafe { slice_f162_into(&elems_b, &mut idx) };
        let yb = run(&idx, &a_vert[b], &mut out);
        for j in 0..N {
            y_simd[j] = (y_simd[j] as u64 + yb[j] as u64 % q as u64) as u32 % q as u32;
        }
    }
    assert_eq!(y_simd, y, "q={q}: SIMD MAC != scalar pointwise inner product");
}

#[test]
fn mac_3889_matches_scalar() {
    test_mac_against_scalar(3889, 0xC0FF_0001);
}

#[test]
fn mac_2917_matches_scalar() {
    test_mac_against_scalar(2917, 0xC0FF_0002);
}

/// The full commitment: `commit_with(Scalar)` vs `commit_with(Simd)`, bit-identical matrix,
/// raw commitments and kept transforms.
fn test_commit_backend_pair(base: Modulus, additional: &[Modulus], seed: u64) {
    if !gate() {
        return;
    }
    let len_f162 = 512; // multiple of 128 -> 128 ring elements per chunk, 4 batches
    let r = 4;
    let key = CommitmentKey::random(len_f162, seed, base, additional);
    let witness = random_elems(len_f162 * r, seed ^ 0xBEEF_C0DE);
    let (m_scalar, aux_scalar) = key.commit_with(&witness, r, Backend::Scalar);
    let (m_simd, aux_simd) = key.commit_with(&witness, r, Backend::Simd);
    assert_eq!(m_scalar, m_simd, "commitment matrices differ");
    assert_eq!(aux_scalar.raw, aux_simd.raw, "raw commitments differ");
    assert_eq!(aux_scalar.batches, aux_simd.batches, "kept transforms differ");
}

#[test]
fn commit_simd_equals_scalar_split_quad() {
    test_commit_backend_pair(Modulus::Q3889_FS_S, &[Modulus::Q2917_Q_S], 0xD1CE_0001);
}

#[test]
fn commit_simd_equals_scalar_split_split() {
    test_commit_backend_pair(Modulus::Q3889_FS_S, &[Modulus::Q9721_FS_S], 0xD1CE_0002);
}

#[test]
fn commit_simd_equals_scalar_three_limbs() {
    test_commit_backend_pair(
        Modulus::Q3889_FS_S,
        &[Modulus::Q2917_Q_S, Modulus::Q4861_Q_S],
        0xD1CE_0003,
    );
}

// keep the accumulator-shape constants honest against the kernels' layouts
#[test]
fn accumulator_shapes() {
    assert_eq!(core::mem::size_of::<Acc>(), ACC_VECS * 64);
    assert_eq!(
        core::mem::size_of::<QuadAcc>(),
        (QBLOCKS * QACC01_PER_BLK) * 64 + (QBLOCKS * QACC2_PER_BLK) * 64 + 8 * 64
    );
}
