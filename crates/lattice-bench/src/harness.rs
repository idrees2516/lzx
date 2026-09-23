//! Pure-std timing harness: warmup + repetitions, median and minimum.

use std::time::Instant;

/// One measured stage.
#[derive(Clone, Debug)]
pub struct Timing {
    pub name: String,
    pub median_ns: u128,
    pub min_ns: u128,
    pub reps: usize,
}

/// Measure `f`, with `warmup` warmup calls and `reps` timed reps.
pub fn measure<F: FnMut()>(name: &str, mut f: F, warmup: usize, reps: usize) -> Timing {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos());
    }
    samples.sort_unstable();
    Timing {
        name: name.to_string(),
        median_ns: samples[samples.len() / 2],
        min_ns: samples[0],
        reps,
    }
}

/// Human-readable duration.
pub fn fmt_ns(ns: u128) -> String {
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}
