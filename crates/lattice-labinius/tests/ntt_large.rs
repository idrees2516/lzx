//! The large-prime binary-input NTT kernel (`ntt_large`, q = 17497 / 19441) against the exact
//! scalar reference.
//!
//! The port's contract with itself, as in `tests/simd.rs`: the SIMD path does not get to be
//! *close*, it has to produce values congruent mod q to `scalar::ntt` of the same lift, per
//! lane, and every lane has to respect the kernel's declared lazy-reduction bound
//! (`ntt_large::output_bound`) — which is the level-6 column of the const-evaluated bound
//! recursion `bin_model` replays on the chosen reduction schedule. The const schedule's own
//! invariants (peak intermediate inside i16, the a0-only schedule failing, the documented
//! per-prime reduction shapes) are re-checked from the outside, and the [`BlockSink`] entry
//! point is verified to deliver the same 24 blocks the plain one writes.
//!
//! Everything skips silently on machines without the AVX-512 PCS feature set.

// the kernels index rows and table positions with strides; the lint's iterator forms do not apply
#![allow(clippy::needless_range_loop)]

use lattice_labinius::binfield::{lift_elem, random_elems, F162};
use lattice_labinius::params::QS_LARGE;
use lattice_labinius::scalar::{ntt, Coeffs};
use lattice_labinius::simd::ntt_large::{
    bar_kind, bar_levels, bin_model, is_large, output_bound, BlockSink, RED_LUT, RED_NONE,
};
use lattice_labinius::simd::ntt_large;
use lattice_labinius::simd::transpose::{slice_f162_into, BinaryIndex32};
use lattice_labinius::simd::Batch32;
use std::time::Instant;

fn gate() -> bool {
    if lattice_labinius::simd::available() {
        true
    } else {
        eprintln!("skipping: no AVX-512 PCS feature set on this machine");
        false
    }
}

/// One batch of 32 binary ring elements through the real GFNI front end and the kernel.
fn run_batch(elems: &[F162; 128], q: u16) -> Batch32 {
    let mut idx = BinaryIndex32::zero();
    let mut out = Batch32::zero();
    unsafe {
        slice_f162_into(elems, &mut idx);
        match q {
            17497 => ntt_large::ntt_bin_batch32::<17497>(&idx, &mut out),
            _ => ntt_large::ntt_bin_batch32::<19441>(&idx, &mut out),
        }
    }
    out
}

/// The scalar reference of one batch: 32 ring elements, fully reduced `[u32; N]` each.
fn reference(elems: &[F162; 128], q: u16) -> Vec<Coeffs> {
    (0..32)
        .map(|p| {
            let w = lift_elem(elems, p);
            match q {
                17497 => ntt::<17497>(&w),
                _ => ntt::<19441>(&w),
            }
        })
        .collect()
}

/// The kernel lane vs the scalar slot value, modulo q, plus the declared bound; returns the
/// worst |lane| seen.
fn check_output(out: &Batch32, reference: &[Coeffs], q: u16, bound: i32) -> i32 {
    let mut worst = 0i32;
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
            if lane.abs() > worst {
                worst = lane.abs();
            }
        }
    }
    worst
}

/// Eight random batches per prime, each checked lane by lane.
fn test_bin_large(q: u16, seed: u64) {
    if !gate() {
        return;
    }
    let bound = output_bound(q);
    let mut worst = 0i32;
    for b in 0..8 {
        let elems = random_elems(128, seed + b as u64);
        let elems: Box<[F162; 128]> = elems.into_boxed_slice().try_into().unwrap();
        let out = run_batch(&elems, q);
        let reference = reference(&elems, q);
        worst = worst.max(check_output(&out, &reference, q, bound));
    }
    println!(
        "q={q}: 8 random batches congruent to scalar, all lanes within the declared bound {bound} (observed max |lane| {worst})"
    );
}

#[test]
fn bin_large_17497_matches_scalar() {
    test_bin_large(17497, 0x5EED_0021);
}

#[test]
fn bin_large_19441_matches_scalar() {
    test_bin_large(19441, 0x5EED_0022);
}

/// Adversarial binary inputs: all-zero, all-ones, and single-bit taps at the low, middle and
/// top coefficient positions of the lift.
#[test]
fn bin_large_edge_inputs() {
    if !gate() {
        return;
    }
    let taps = [
        F162([0, 0, 0]),
        F162([u64::MAX, u64::MAX, (1u64 << 34) - 1]),
        F162([1, 0, 0]),
        F162([0, 1 << 49, 0]),
        F162([0, 0, 1 << 33]),
    ];
    let mut worst_tap = 0i32;
    for &fill in taps.iter() {
        let elems = [fill; 128];
        for q in QS_LARGE {
            let out = run_batch(&elems, q);
            let reference = reference(&elems, q);
            worst_tap = worst_tap.max(check_output(&out, &reference, q, output_bound(q)));
        }
    }
    println!("edge inputs: all taps congruent, worst |lane| {worst_tap}");
}

/// The const schedule's own story, re-checked from the outside: the peak intermediate the bound
/// model carries stays inside i16 for the chosen schedule, the declared output bound is the
/// model's level-6 column, and the schedule that only ever reduces `a0` with the lookup Barrett
/// (all the sub-2^14 kernels ever need) provably does not fit — which is why this kernel exists.
#[test]
fn schedule_model_fits_i16() {
    for q in QS_LARGE {
        assert!(is_large(q));
        let code = bar_levels(q);
        let (lm, peak) = bin_model(q, code);
        assert!(peak <= 32767, "q={q}: model peak {peak} leaves i16");
        for (l, &b) in lm.iter().enumerate() {
            assert!(b <= 32767, "q={q}: level {l} bound {b} leaves i16");
        }
        assert_eq!(output_bound(q), lm[4], "q={q}: declared bound != model output");
        println!(
            "q={q}: schedule {code}, per-level bounds {lm:?} (x1000/q: {:?}), peak {peak}",
            lm.map(|b| b as i64 * 1000 / q as i64)
        );
        // the a0-lookup-only schedule: RED_LUT at site 0 of every level, nothing elsewhere
        let a0_only = RED_LUT as u32 * (1 + 27 + 729 + 19683);
        assert_eq!(bar_kind(a0_only, 0, 0), RED_LUT);
        assert_eq!(bar_kind(a0_only, 3, 0), RED_LUT);
        assert_eq!(bar_kind(a0_only, 3, 1), RED_NONE);
        assert!(
            bin_model(q, a0_only).1 > 32767,
            "q={q}: the a0-only schedule would fit i16"
        );
    }
    // the documented shapes: 17497 reduces a0 and the twiddle products and leaves u alone,
    // spending the two-multiply Barrett on one of them per level; 19441 pays for u as well and
    // is all-lookup.
    let (c0, c1) = (bar_levels(QS_LARGE[0]), bar_levels(QS_LARGE[1]));
    for l in 0..4 {
        assert_ne!(bar_kind(c0, l, 0), RED_NONE);
        assert_ne!(bar_kind(c0, l, 1), RED_NONE);
        assert_eq!(bar_kind(c0, l, 2), RED_NONE);
        assert!(
            bar_kind(c0, l, 0) == ntt_large::RED_MUL || bar_kind(c0, l, 1) == ntt_large::RED_MUL,
            "17497 level {l} spends no vpmulhrsw Barrett"
        );
        for i in 0..3 {
            assert_eq!(bar_kind(c1, l, i), RED_LUT, "19441 level {l} site {i}");
        }
    }
}

/// A sink that writes the blocks into its own batch-shaped buffer and records the order they
/// arrive in, so the plumbing (`dst`/`block`, 24 blocks of 27 rows) is checked end to end.
struct CollectSink {
    base: *mut i16,
    order: Vec<usize>,
    dsts: Vec<*mut i16>,
}

impl CollectSink {
    fn new(out: &mut Batch32) -> Self {
        CollectSink {
            base: out.v.as_mut_ptr() as *mut i16,
            order: Vec::new(),
            dsts: Vec::new(),
        }
    }
}

impl BlockSink for CollectSink {
    unsafe fn dst(&mut self, blk: usize) -> *mut i16 {
        let p = self.base.add(32 * 27 * blk);
        self.dsts.push(p);
        p
    }
    unsafe fn block(&mut self, blk: usize, dst: *const i16) {
        self.order.push(blk);
        assert_eq!(
            dst, self.dsts[blk] as *const i16,
            "block {blk} handed a foreign pointer"
        );
    }
}

#[test]
fn sink_delivers_the_same_transform() {
    if !gate() {
        return;
    }
    let elems = random_elems(128, 0x5EED_0023);
    let elems: Box<[F162; 128]> = elems.into_boxed_slice().try_into().unwrap();
    let mut idx = BinaryIndex32::zero();
    unsafe { slice_f162_into(&elems, &mut idx) };
    for q in QS_LARGE {
        let plain = run_batch(&elems, q);
        let reference = reference(&elems, q);
        let mut sunk = Batch32::zero();
        let mut sink = CollectSink::new(&mut sunk);
        unsafe {
            match q {
                17497 => ntt_large::ntt_bin_batch32_sink::<17497, _>(&idx, &mut sink),
                _ => ntt_large::ntt_bin_batch32_sink::<19441, _>(&idx, &mut sink),
            }
        }
        assert_eq!(sink.order, (0..24).collect::<Vec<_>>(), "q={q}: block order");
        assert_eq!(sink.dsts.len(), 24, "q={q}: one dst per block");
        assert_eq!(sunk.v, plain.v, "q={q}: sink output != plain output");
        check_output(&sunk, &reference, q, output_bound(q));
    }
    println!("sink entry point: 24 blocks of 27 rows each, in order, == plain output");
}

/// Median-of-9 timing per prime over batches of 32 ring elements (648 slots each).
#[test]
fn timings() {
    if !gate() {
        return;
    }
    const BATCHES: usize = 8;
    for q in QS_LARGE {
        // prepare the inputs once; time only the transform
        let mut idxs = Vec::with_capacity(BATCHES);
        let mut outs = Vec::with_capacity(BATCHES);
        for b in 0..BATCHES {
            let elems = random_elems(128, 0x7107_0000 + b as u64);
            let elems: Box<[F162; 128]> = elems.into_boxed_slice().try_into().unwrap();
            let mut idx = BinaryIndex32::zero();
            unsafe { slice_f162_into(&elems, &mut idx) };
            idxs.push(idx);
            outs.push(Batch32::zero());
        }
        let mut runs = [0f64; 9];
        for r in 0..9 {
            let t0 = Instant::now();
            for b in 0..BATCHES {
                unsafe {
                    match q {
                        17497 => ntt_large::ntt_bin_batch32::<17497>(&idxs[b], &mut outs[b]),
                        _ => ntt_large::ntt_bin_batch32::<19441>(&idxs[b], &mut outs[b]),
                    }
                }
            }
            runs[r] = t0.elapsed().as_secs_f64();
        }
        runs.sort_by(|a, b| a.total_cmp(b));
        let med = runs[runs.len() / 2];
        let per_batch = med / BATCHES as f64;
        println!(
            "q={q}: median {:.2} us per batch of 32 ring elements ({:.1} ns per element; \
             {BATCHES} batches x 9 runs)",
            per_batch * 1e6,
            per_batch * 1e9 / 32.0
        );
    }
}
