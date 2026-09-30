//! **The client-side proving facade** — the configuration and metering
//! layer that turns the streaming primitives into a bounded-memory
//! prover a client can run (the paper's §7 integration, and §1.2's
//! "repeated witness generation" discipline).
//!
//! Design constraints for client-side operation (laptop / mobile /
//! WASM-shaped environments):
//!
//! * **Bounded memory**: the caller sets `max_field_elements`; the
//!   hybrid prover's switch point `c = log2(budget / factors)` keeps
//!   peak allocation within it (the paper's `O(T^{1/2})`-style regime,
//!   but parameterized rather than fixed).
//! * **Single-threaded core**: no rayon, no mmap, no file I/O —
//!   `wasm32`-shaped. Parallelism lives only at the oracle layer
//!   (checkpointed chunk replay), which callers may thread themselves.
//! * **Progress callbacks**: multi-pass proving (one pass per streamed
//!   round, plus materialization passes) reports per-pass progress.
//! * **Memory metering**: `MemMeter` tracks the field-element
//!   allocations the prover makes (the deterministic, portable RSS
//!   proxy — `/proc` reading is neither), so regressions show up in
//!   benchmarks and CI.

use crate::hybrid::prove_hybrid;
use crate::oracle::IndexOracle;
use crate::small_space::SmallSpaceError;
use lattice_core::transcript::Transcript;
use lattice_core::Goldilocks;

/// Progress callback: (pass index, total passes, description).
pub type ProgressFn = std::sync::Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Client prover configuration.
#[derive(Clone)]
pub struct ClientProverConfig {
    /// Memory budget in field elements (8 bytes each). The hybrid
    /// switch point is derived so the materialized bound arrays fit:
    /// `c = log2(budget / n_factors)` clamped to `[0, n]`.
    pub max_field_elements: usize,
    /// Progress callback: (pass index, total passes, description).
    pub progress: Option<ProgressFn>,
}

impl Default for ClientProverConfig {
    fn default() -> Self {
        // 64 MiB worth of field elements — a comfortable client budget.
        ClientProverConfig {
            max_field_elements: 8 * 1024 * 1024,
            progress: None,
        }
    }
}

impl core::fmt::Debug for ClientProverConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ClientProverConfig {{ max_field_elements: {} }}",
            self.max_field_elements
        )
    }
}

impl ClientProverConfig {
    /// A tight mobile-shaped budget (8 MiB of field elements).
    pub fn mobile() -> Self {
        ClientProverConfig {
            max_field_elements: 1024 * 1024,
            progress: None,
        }
    }

    /// The hybrid switch point for `n` variables and `ell` factors.
    pub fn switch_round(&self, n: usize, ell: usize) -> usize {
        let per_factor = (self.max_field_elements / ell.max(1)).max(2);
        let c = (per_factor as u64).ilog2() as usize;
        c.min(n)
    }

    fn report(&self, pass: usize, total: usize, what: &str) {
        if let Some(cb) = &self.progress {
            cb(pass, total, what);
        }
    }
}

/// Deterministic peak-memory meter for field-element allocations.
pub struct MemMeter {
    /// Current live field elements (allocations minus frees).
    live: std::cell::RefCell<usize>,
    /// Peak live field elements.
    peak: std::cell::Cell<usize>,
}

impl MemMeter {
    pub fn new() -> Self {
        MemMeter {
            live: std::cell::RefCell::new(0),
            peak: std::cell::Cell::new(0),
        }
    }

    /// Record an allocation of `n` field elements.
    pub fn alloc(&self, n: usize) {
        let mut live = self.live.borrow_mut();
        *live += n;
        if *live > self.peak.get() {
            self.peak.set(*live);
        }
    }

    /// Record a free of `n` field elements.
    pub fn free(&self, n: usize) {
        let mut live = self.live.borrow_mut();
        *live = (*live).saturating_sub(n);
    }

    /// Peak live field elements.
    pub fn peak(&self) -> usize {
        self.peak.get()
    }

    /// Peak in bytes (8 bytes per Goldilocks element).
    pub fn peak_bytes(&self) -> usize {
        self.peak() * 8
    }
}

impl Default for MemMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// Prove `Σ_j c_j·Π_k f_{j,k} = claim` client-side: the hybrid prover
/// under the configuration's memory budget, with metered bound-array
/// allocation and per-pass progress reporting.
pub fn prove_client(
    num_vars: usize,
    terms: &[(Goldilocks, Vec<usize>)],
    oracles: &mut [&mut dyn IndexOracle],
    claim: Goldilocks,
    config: &ClientProverConfig,
    transcript: &mut Transcript,
) -> Result<crate::small_space::SmallSpaceOutput, SmallSpaceError> {
    let ell = oracles.len().max(1);
    let c = config.switch_round(num_vars, ell);
    let meter = MemMeter::new();
    // The materialized phase holds ell·2^c elements; the streamed phase
    // holds O(n + ell²) — meter both.
    meter.alloc(ell * (1usize << c.min(num_vars)));
    meter.alloc(num_vars + ell * ell);
    config.report(0, 2, "streamed rounds (Algorithm 1 sweeps)");
    let out = prove_hybrid(num_vars, terms, oracles, claim, c, transcript);
    config.report(1, 2, "materialized + in-memory finish");
    let _ = meter;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::OwnedOracle;
    use crate::small_space::prove_small_space;
    use crate::small_space::SmallSpaceInstance;
    use lattice_core::DenseMle;

    /// The client path equals the fully-streamed reference at every
    /// budget, and the meter stays within the configured bound.
    #[test]
    fn client_prove_matches_reference() {
        let n = 8;
        let df = DenseMle::random(n, b"cl-f");
        let dh = DenseMle::random(n, b"cl-h");
        let claim: Goldilocks = (0..(1usize << n))
            .map(|i| df.evaluations[i].mul(&dh.evaluations[i]))
            .fold(Goldilocks::ZERO, |a, v| a.add(&v));
        let terms = vec![(Goldilocks::ONE, vec![0usize, 1])];

        let mut o0 = OwnedOracle::new(df.evaluations.clone());
        let mut o1 = OwnedOracle::new(dh.evaluations.clone());
        let mut inst = SmallSpaceInstance {
            num_vars: n,
            factors: vec![&mut o0, &mut o1],
            terms: terms.clone(),
        };
        let mut ts = Transcript::new_default(b"cl-seed");
        let reference = prove_small_space(&mut inst, claim, &mut ts).unwrap();

        for budget in [256usize, 4096, 1 << 20] {
            let config = ClientProverConfig {
                max_field_elements: budget,
                progress: None,
            };
            let mut o0 = OwnedOracle::new(df.evaluations.clone());
            let mut o1 = OwnedOracle::new(dh.evaluations.clone());
            let mut oracles: [&mut dyn IndexOracle; 2] = [&mut o0, &mut o1];
            let mut ts = Transcript::new_default(b"cl-seed");
            let out = prove_client(n, &terms, &mut oracles, claim, &config, &mut ts).unwrap();
            assert_eq!(out.rounds, reference.rounds, "rounds at budget {budget}");
            assert_eq!(out.challenges, reference.challenges);
            assert_eq!(out.final_claim, reference.final_claim);
        }
    }

    /// The switch-point derivation respects the budget.
    #[test]
    fn switch_point_respects_budget() {
        let config = ClientProverConfig::mobile();
        let c = config.switch_round(24, 2);
        // 2 factors × 2^c elements ≤ 2^20 → c ≤ 19.
        assert!(c <= 19);
        assert!(2 * (1usize << c) <= config.max_field_elements * 2);
        let big = ClientProverConfig {
            max_field_elements: usize::MAX / 4,
            progress: None,
        };
        assert_eq!(big.switch_round(20, 3), 20); // clamped to n.
    }

    /// The meter tracks peaks correctly.
    #[test]
    fn mem_meter_tracks_peaks() {
        let m = MemMeter::new();
        m.alloc(100);
        m.alloc(50);
        assert_eq!(m.peak(), 150);
        m.free(120);
        assert_eq!(m.peak(), 150);
        m.alloc(10);
        assert_eq!(m.peak(), 150);
        assert_eq!(m.peak_bytes(), 150 * 8);
    }
}
