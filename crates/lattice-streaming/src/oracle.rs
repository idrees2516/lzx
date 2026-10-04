//! Stream oracles — the prover's access pattern for small-space proving
//! (ePrint 2025/611, §3.1.2 / Observation 3.5).
//!
//! The paper's small-space sum-check prover (Algorithm 1) assumes the
//! evaluations of every factor `g_k` over `{0,1}^n` can be *enumerated*
//! in index order, each in `O(1)` time and space — the shape Jolt's
//! witness generation satisfies: "there is a witness generation algorithm
//! that computes each evaluation of `g_k` in `O(1)` time and logarithmic
//! space (on top of the space K needed simply to run the VM)".
//!
//! Two access disciplines:
//!
//! * **Sequential streaming** (`StreamOracle`) — `next()` yields the
//!   truth table in index order; used by the prefix-suffix protocol, the
//!   streaming grand product, and the matrix-layout commitment.
//! * **Indexed oracles** (`IndexOracle`) — Algorithm 1's jump pattern
//!   `(j, 0, tobits(m)) / (j, 1, tobits(m))`; realized *client-side* by
//!   checkpointed regeneration: the VM state is snapshotted at chunk
//!   boundaries during the (serial) first pass, so any window of the
//!   trace can be replayed on demand — the paper's "repeated witness
//!   generation", parallelizable across chunks (§1.2 item 4).

use lattice_core::Goldilocks;

/// Sequential truth-table stream in index order (variable 0 = most
/// significant index bit, matching `DenseMle`).
pub trait StreamOracle {
    /// Total stream length (a power of two).
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Next evaluation; callers must not over-read.
    fn next(&mut self) -> Goldilocks;
    /// Reset to the beginning (multi-pass protocols call this between
    /// rounds — one pass per sum-check round is the paper's shape).
    fn reset(&mut self);
}

/// Random-access oracle over the same data — Algorithm 1's discipline.
pub trait IndexOracle {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Evaluation at the given index.
    fn eval(&mut self, index: u64) -> Goldilocks;
}

/// In-memory slice oracle (tests, reference paths, and the hybrid
/// prover's materialized phase).
#[derive(Clone)]
pub struct SliceOracle<'a> {
    data: &'a [Goldilocks],
    pos: usize,
}

impl<'a> SliceOracle<'a> {
    pub fn new(data: &'a [Goldilocks]) -> Self {
        SliceOracle { data, pos: 0 }
    }
}

impl StreamOracle for SliceOracle<'_> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn next(&mut self) -> Goldilocks {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl IndexOracle for SliceOracle<'_> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn eval(&mut self, index: u64) -> Goldilocks {
        self.data[index as usize]
    }
}

/// Owned variant (the streaming prover owns its witness data).
pub struct OwnedOracle {
    pub data: Vec<Goldilocks>,
    pos: usize,
}

impl OwnedOracle {
    pub fn new(data: Vec<Goldilocks>) -> Self {
        OwnedOracle { data, pos: 0 }
    }
    pub fn from_fn(n_vars: usize, f: impl Fn(u64) -> Goldilocks) -> Self {
        let len = 1u64 << n_vars;
        let mut data = Vec::with_capacity(len as usize);
        for i in 0..len {
            data.push(f(i));
        }
        OwnedOracle { data, pos: 0 }
    }
}

impl StreamOracle for OwnedOracle {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn next(&mut self) -> Goldilocks {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl IndexOracle for OwnedOracle {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn eval(&mut self, index: u64) -> Goldilocks {
        self.data[index as usize]
    }
}

/// **Witness-generator oracle**: computes each evaluation on the fly from
/// a closure — the paper's Observation 3.5 shape (`O(1)` time and space
/// per element). The closure receives the index and returns the truth
/// table value; stateful generators (a VM stepping through the trace)
/// capture their state in the closure and rely on `reset` to re-arm via
/// the `restart` hook.
pub struct GenOracle<F, R>
where
    F: FnMut(u64) -> Goldilocks,
    R: FnMut(),
{
    n_vars: usize,
    gen: F,
    restart: R,
    pos: u64,
}

impl<F, R> GenOracle<F, R>
where
    F: FnMut(u64) -> Goldilocks,
    R: FnMut(),
{
    pub fn new(n_vars: usize, gen: F, restart: R) -> Self {
        GenOracle {
            n_vars,
            gen,
            restart,
            pos: 0,
        }
    }
}

impl<F, R> StreamOracle for GenOracle<F, R>
where
    F: FnMut(u64) -> Goldilocks,
    R: FnMut(),
{
    fn len(&self) -> u64 {
        1u64 << self.n_vars
    }
    fn next(&mut self) -> Goldilocks {
        let v = (self.gen)(self.pos);
        self.pos += 1;
        v
    }
    fn reset(&mut self) {
        (self.restart)();
        self.pos = 0;
    }
}

/// **Checkpointed regeneration oracle** — the client-side realization of
/// Algorithm 1's random access (the paper's §1.2 item 4): snapshots of
/// the generator state at chunk boundaries let any window be replayed in
/// `O(chunk)` time, and independent windows replay in parallel.
///
/// The generator is expressed as a fold over indices:
/// `state ↦ (state', value)` — e.g. a VM whose step `i` produces the
/// trace value at index `i`. `checkpoint_every` controls the snapshot
/// granularity (the memory/space tradeoff: `O(len / chunk)` snapshots).
/// The per-index generator fold: `state ↦ (state', value)`.
pub type StepFn<'a, S> = Box<dyn FnMut(&mut S, u64) -> Goldilocks + 'a>;

pub struct ChunkedRegenOracle<'a, S> {
    n_vars: usize,
    /// Initial generator state.
    initial: S,
    /// `(index, state)` snapshots; always includes `(0, initial)`.
    checkpoints: Vec<(u64, S)>,
    chunk: u64,
    /// Cached current position for sequential walks.
    pos: u64,
    state: S,
    step: StepFn<'a, S>,
}

impl<'a, S: Clone> ChunkedRegenOracle<'a, S> {
    /// Build with a single sequential pass, snapshotting every `chunk`
    /// indices (the paper's first serial witness-generation pass).
    pub fn build(
        n_vars: usize,
        initial: S,
        chunk: u64,
        mut step: impl FnMut(&mut S, u64) -> Goldilocks + 'a,
    ) -> (Self, Vec<Goldilocks>) {
        let len = 1u64 << n_vars;
        let mut checkpoints = vec![(0u64, initial.clone())];
        let mut values = Vec::with_capacity(len as usize);
        let mut st = initial.clone();
        for i in 0..len {
            if i > 0 && i % chunk == 0 {
                checkpoints.push((i, st.clone()));
            }
            values.push(step(&mut st, i));
        }
        let oracle = ChunkedRegenOracle {
            n_vars,
            initial: initial.clone(),
            checkpoints,
            chunk,
            pos: 0,
            state: initial,
            step: Box::new(step),
        };
        (oracle, values)
    }

    /// Build **without materializing the values** — the checkpoint-only
    /// pass (the memory-bounded path: `O(len/chunk)` snapshots of the
    /// generator state, no `O(len)` value array). The oracle then serves
    /// both sequential walks (`StreamOracle`, one re-generation pass per
    /// `reset`) and indexed access (`IndexOracle::eval`, `O(chunk)` per
    /// seek from the latest checkpoint) — the paper's §1.2 item 4.
    pub fn build_streaming(
        n_vars: usize,
        initial: S,
        chunk: u64,
        mut step: impl FnMut(&mut S, u64) -> Goldilocks + 'a,
    ) -> Self {
        let len = 1u64 << n_vars;
        let chunk = chunk.max(1);
        let mut checkpoints = vec![(0u64, initial.clone())];
        let mut st = initial.clone();
        for i in 0..len {
            if i > 0 && i % chunk == 0 {
                checkpoints.push((i, st.clone()));
            }
            let _ = step(&mut st, i);
        }
        ChunkedRegenOracle {
            n_vars,
            initial: initial.clone(),
            checkpoints,
            chunk,
            pos: 0,
            state: initial,
            step: Box::new(step),
        }
    }

    /// The number of live snapshots (the checkpoint-memory footprint in
    /// generator states).
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Seek to position 0 and drop to the initial state (a fresh
    /// re-generation arm — used instead of `reset` when the caller wants
    /// an explicitly rewound sequential walk).
    pub fn rewind(&mut self) {
        self.state = self.initial.clone();
        self.pos = 0;
    }

    fn seek(&mut self, index: u64) {
        // Fast path: sequential or small forward jumps.
        if index >= self.pos && index - self.pos <= self.chunk {
            while self.pos < index {
                let _ = (self.step)(&mut self.state, self.pos);
                self.pos += 1;
            }
            return;
        }
        // Restart from the latest checkpoint at or below `index`.
        let mut best = 0usize;
        for (ci, &(idx, _)) in self.checkpoints.iter().enumerate() {
            if idx <= index {
                best = ci;
            } else {
                break;
            }
        }
        let (start, st) = self.checkpoints[best].clone();
        self.state = st;
        self.pos = start;
        while self.pos < index {
            let _ = (self.step)(&mut self.state, self.pos);
            self.pos += 1;
        }
    }
}

impl<'a, S: Clone> IndexOracle for ChunkedRegenOracle<'a, S> {
    fn len(&self) -> u64 {
        1u64 << self.n_vars
    }
    fn eval(&mut self, index: u64) -> Goldilocks {
        self.seek(index);
        let v = (self.step)(&mut self.state, self.pos);
        self.pos += 1;
        v
    }
}

impl<'a, S: Clone> StreamOracle for ChunkedRegenOracle<'a, S> {
    fn len(&self) -> u64 {
        1u64 << self.n_vars
    }
    fn next(&mut self) -> Goldilocks {
        let v = (self.step)(&mut self.state, self.pos);
        self.pos += 1;
        v
    }
    fn reset(&mut self) {
        self.state = self.initial.clone();
        self.pos = 0;
    }
}

/// **Streaming MLE evaluation** — `Σ_z eq(point, z)·v[z]` in one pass
/// over the stream, `O(n)` space (the eq weight maintained per element,
/// `O(n)` work each — `O(n·2^n)` total). The memory-bounded replacement
/// for `DenseMle::new(values).evaluate(&point)`.
pub fn stream_mle_eval(
    stream: &mut dyn StreamOracle,
    n_vars: usize,
    point: &[Goldilocks],
) -> Option<Goldilocks> {
    if point.len() != n_vars {
        return None;
    }
    stream.reset();
    let len = 1u64 << n_vars;
    let mut acc = Goldilocks::ZERO;
    // Gray-code-free direct evaluation: O(n) per element.
    for z in 0..len {
        let v = stream.next();
        let mut w = Goldilocks::ONE;
        for (b, rb) in point.iter().enumerate() {
            let f = if (z >> (n_vars - 1 - b)) & 1 == 1 {
                *rb
            } else {
                Goldilocks::ONE.sub(rb)
            };
            w = w.mul(&f);
        }
        acc = acc.add(&w.mul(&v));
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(x: u64) -> Goldilocks {
        Goldilocks::from_u64(x)
    }

    #[test]
    fn owned_oracle_roundtrip() {
        let data = vec![g(1), g(2), g(3), g(4)];
        let mut o = OwnedOracle::new(data.clone());
        assert_eq!(StreamOracle::len(&o), 4);
        assert_eq!(o.next(), g(1));
        assert_eq!(o.next(), g(2));
        o.reset();
        assert_eq!(o.next(), g(1));
        assert_eq!(IndexOracle::eval(&mut o, 3), g(4));
    }

    /// Checkpointed regeneration: random access through a stateful
    /// generator (a running sum) matches the direct values, including
    /// backward jumps that force checkpoint restarts.
    #[test]
    fn chunked_regen_random_access() {
        let n = 6;
        let (mut o, values) = ChunkedRegenOracle::build(n, 0u64, 5, |st: &mut u64, i: u64| {
            *st = st.wrapping_add(i * 3 + 1);
            g(*st)
        });
        assert_eq!(values.len(), 64);
        for idx in [0u64, 1, 5, 6, 17, 40, 63, 20, 2, 63] {
            assert_eq!(
                IndexOracle::eval(&mut o, idx),
                values[idx as usize],
                "at {idx}"
            );
        }
        // Sequential stream agrees too.
        o.reset();
        for v in &values {
            assert_eq!(o.next(), *v);
        }
    }

    /// The generator oracle with state captured by the closure.
    #[test]
    fn gen_oracle_stateful() {
        let mut counter = 100u64;
        let mut o = GenOracle::new(
            4,
            move |_| {
                counter = counter.wrapping_add(7);
                g(counter)
            },
            || {},
        );
        let first = o.next();
        let second = o.next();
        assert_eq!(first, g(107));
        assert_eq!(second, g(114));
        // The restart hook cannot reach state moved into the generator
        // closure, so a reset rewinds the *index* but not the captured
        // state — re-arming requires shared state (see ChunkedRegenOracle).
        o.reset();
        assert_eq!(o.next(), g(121));
    }
}
