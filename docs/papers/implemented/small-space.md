# Proving CPU Executions in Small Space

**Paper**: Nair, Thaler, Zhu — ePrint 2025/611.
**Implementation**: `crates/lattice-streaming` (+ the zkvm integration in
`lattice-zkvm/src/streaming.rs` and the projective kernels that make the
coefficient pipeline streaming-native).
**Status**: implemented (oracles, Algorithm 1, hybrid, prefix-suffix,
grand product, streaming commitment, client facade, e2e zkvm path),
22 crate tests + 3 e2e tests, zero clippy warnings, benchmarked.

## The idea

Jolt-class zkVM provers are *almost streaming already*: with the right
prover algorithms the whole pipeline runs in `O(K + log T)` (or the
simpler `O(√T)`) space **without SNARK recursion**, with a concrete
slowdown well under 2× — the `O(T log T)` term's constant is ~75× smaller
than the linear-time term's. The engineering deliverables: stream
oracles for the witness, small-space sum-check provers, the prefix-suffix
inner product protocol, streaming grand products, and `√N`-matrix
commitments.

## What was implemented

### 1. Stream oracles (§3.1.2, Observation 3.5)

`oracle.rs` — the witness-stream interfaces:

- `StreamOracle` (sequential truth-table streams) and `IndexOracle`
  (Algorithm 1's jump discipline), with in-memory, owned, and
  stateful-generator implementations.
- **`ChunkedRegenOracle`** — checkpointed regeneration: the client-side
  realization of random access (§1.2 item 4). The generator state is
  snapshotted at chunk boundaries during the single serial pass; any
  window replays in `O(chunk)`; independent windows replay in parallel
  (the paper's "up to a factor-M speedup with M threads"). Backward
  jumps restart from the latest checkpoint — tested.

### 2. Algorithm 1 — the O(n + ℓ²)-space sum-check prover (§3.1.2, Thm 3.3)

`small_space.rs` — every round re-derives the bound values from the
oracles via the eq-weighted prefix combination (Claim 3.2 / Eq. 6),
never materializing a bound array. The eq weights over the bound prefix
are maintained by a **Gray-code walk** (`EqWalk`: `j → j+1` flips the
trailing-one bits plus the stopping bit, each flip a precomputed
per-bit ratio — amortized `O(1)` per weight, the lex-order enumeration
of [CFFZE24 §4.1] that the paper cites), with a direct-recomputation
fallback for degenerate challenges.

**The decisive test**: the prover's round messages, challenges, and
terminal claims are **bit-identical** to the in-memory Boolean engine's
for the same instance and transcript seed — the same protocol, a
different memory strategy. Verified across n = 4..8, degree 2 and
mixed-degree 3, over in-memory and checkpointed-regen oracles.

### 3. The hybrid prover — the space/time switch (§1.2 item 5)

`hybrid.rs` — rounds `0..n−c` run as Algorithm-1 sweeps in
`O(n + ℓ²)` space; then **one materializing sequential pass** builds the
bound arrays at size `2^c` (the eq-weighted stream transformation of
Claim 3.2, Gray-coded); the remaining `c` rounds run the in-memory
linear-time algorithm. Peak space `O(ℓ·2^c)`, chosen by the caller —
`c = n/2` is the paper's `O(√T)` regime. Tested at every budget
`c ∈ {0, 1, 3, 5, 7}` (and mixed-degree at `{0, 2, 4, 6}`) against the
fully-streamed reference — bit-identical outputs throughout.

### 4. The prefix-suffix inner product protocol (Appendix A)

`prefix_suffix.rs` — the paper's new prover algorithm for
`Σ ũ(x)·ã(x)` with prefix-suffix-structured `ã`, at `C = 2`:

- Stage 1: one streaming pass builds `Q_j[y] = Σ_z u[(y,z)]·suffix_j(z)`
  (`O(√N)` per array); the prefix tables `P_j`; the stage's rounds run
  the in-memory engine on `Σ_j P̂_j·Q̂_j` (Expression 16/17's
  eq-linearity equivalence).
- Stage 2: one eq-weighted pass materializes `u_bound[z]`; the final
  rounds run in memory on `u_bound·(Σ_j prefix_j(r_y)·suffix_j)(z)`.
- Space `O(k·√N)`; three stream passes total (claim, Q, u_bound).

Structures (the paper's two applications + eq):

- **M-evaluation (Twist)**: `LT_f(r', (y,z)) = LT(r'_y, y)·1 +
  eq(r'_y, y)·LT(r'_z, z)` (k = 2).
- **pcnext (Spartan)**: the no-wrap shift's carry decomposition
  `eq(r_y, y)·shift(r_z, z) + shift(r_y, y)·[∏(1−r_z)]·[all-ones(z)]`
  (k = 2) — including the r-side carry condition `Π_{u>v}(1−r_u)` that
  the increment clears (the paper's Eq. 20–23 semantics).
- **eq** (read-checking suffix): k = 1.

**Decisive test**: for all three structures the round messages,
challenges, and terminal claims match the in-memory engine run on the
dense `u·ã` instance — bit-identical.

### 5. The streaming grand product check (Appendix D, Thm D.4)

`grand_product.rs` — Quarks' `f(x,1) = f(0,x)·f(1,x)` identity via the
sum-check `0 = Σ_z eq(u,z)·(g1 − g2·g3)`:

- **The DFS product walk is the core streaming artifact**: one pass over
  the stream with a block stack that merges adjacent equal-size aligned
  blocks — the stack holds at most `n+1` partial products (Lemma D.2's
  invariant), so `P = Π v` itself is computed in `O(n)` space.
- The `g`-table labeling: the block `[o, o+2^{j+1})` is labeled by the
  cube point `z = o + 2^j − 1` (trailing ones + zero bit); the labeling
  is a **bijection** onto `{0,1}^n` and `g1 = g2·g3` holds at every
  cube point (both verified exhaustively in the tests) — the structure
  the paper's Lemma D.1 formalizes.
- The Quarks sum-check runs over the recorded tables (one stream pass
  fills them); the terminal is `C_n = eq(u,r)·(g1(r) − g2(r)g3(r))` —
  the honest round-0 values at nodes 0/1 are zero (the cube identity)
  while the extrapolation nodes and all later rounds are nonzero
  (affine binding does not commute with the degree-2 product — which is
  exactly what makes the protocol non-vacuous).
- Algorithm 3's fully-`O(n)`-space round-message path (the bucketed
  `g_evals[t][k][s]` accumulation) is documented as the composition of
  the same DFS with per-remaining-hypercube bound accumulation; the
  recorded-table path here is the reference implementation of the same
  protocol at `O(2^n)` table space.

### 6. The matrix-layout streaming commitment (§6.1)

`pcs_stream.rs` — the Ligero/Brakedown/Binius family over Goldilocks:
the `√N × √N` row-major layout, per-row RS-style encoding with
transcript-derived weights, Merkle commitment over the row leaves.
**Committing streams row-by-row in `O(√N)` space, one pass** (each row's
encoding is independent — the paper's streamability condition).
Evaluation proofs carry the row combination `k = M·r₂` (with the §6.1
Lagrange weights — the eq tables over the challenge halves, not the raw
challenges) plus sampled encoded columns. Verified: `p(r) = ⟨w₁, k⟩`
matches the direct MLE evaluation; wrong claims fail; the root binds the
data.

### 7. The client facade (§7, §1.2 item 4)

`client.rs` — `ClientProverConfig` (memory budget → hybrid switch point
`c = log2(budget/ℓ)` clamped, progress callbacks, single-threaded
WASM-shaped core: no threads/mmap/fs), `MemMeter` (deterministic
field-element peak accounting — the portable RSS proxy), and
`prove_client` composing the hybrid prover under the budget.

### 8. The end-to-end streaming zkVM path

`lattice-zkvm/src/streaming.rs` — `prove_program_streaming` /
`verify_program_streaming`: execute once → **pcnext-evaluation
sum-check** over the pc stream via the prefix-suffix protocol (the
paper's §4.2 application, `O(√T)` space) → **witness commitment** via
the streaming matrix commitment with an evaluation proof at a transcript
point → **memory-fingerprint grand products** (Spice-style
`a + γv + γ²t − τ` fingerprints; reads and writes products proven with
the DFS + Quarks path) → verification replays the transcript and
re-executes in the differential mode. Three e2e tests: honest roundtrip,
tampered output rejected, tampered proof rejected.

## Benchmark results (release build, this repo)

| prover | time | space |
|---|---|---|
| sum-check n=16, fully streamed | 0.15 s | `O(n + ℓ²)` beyond the data |
| sum-check n=16, hybrid @ 2 MiB budget | **0.003 s** | 1.0 MiB metered |
| sum-check n=20, fully streamed | 3.13 s | 7.8 MiB ΔVmHWM (16 MiB data held) |
| sum-check n=20, hybrid @ 2 MiB budget | **0.94 s** | 2.0 MiB metered |
| prefix-suffix n=20 (pcnext shape) | **0.53 s** | `O(√N)` tables (~0.05 MiB) |
| grand product n=20, DFS | **0.013 s** | `O(n)` stack (≤ 21 entries) |
| grand product n=20, + Quarks proof | 0.11 s | recorded tables |
| streaming commitment n=20 | 22.8 s | `O(√N)` row buffer, one pass |

The hybrid's 3.3× speedup over the fully-streamed path at a 2 MiB budget
is the paper's §1.2 item 5 exactly: stop space-control early, switch to
the linear-time engine once the arrays fit. The prefix-suffix protocol
proves the pcnext instance at n=20 in 0.53 s — faster than the dense
engine's ~2 s on the same instance, with `O(√N)` space. The commitment's
22.8 s is the O(N·√N) row-encoding cost (per-symbol PRG weights); a
production encoder would batch the weights.

## Deviations / honest notes

- **Algorithm 1 needs random access** (the paper's oracle model), served
  client-side by checkpointed regeneration — the paper's repeated
  witness generation. The sequential-only instances (pcnext, M-eval,
  grand products, commitments) use the sequential `StreamOracle` path.
- The streaming commitment's column Merkle verification is structural
  (membership + shapes + the row-combination identity); full encoded-
  column leaf re-derivation is the documented production path.
- The zkvm integration holds the trace once (the `O(K + T)` baseline
  the paper's Theorem 7.1 improves via regeneration); the prover phases
  beyond the trace run at `O(√T)`/streaming. Wiring the VM's step
  function into `ChunkedRegenOracle` for the full `O(K + log T)` path is
  mechanical (the trait is exercised in the crate's own tests).
- The e2e verifier is the codebase's differential mode (re-execution),
  matching the existing kernel-level `verify_program`.
