# Speeding Up Sum-Check Proving (+ the Extended Version)

**Paper**: Bagad, Dao, Domb, Thaler — ePrint 2025/1117; the extended
version Dao, DeStefano, Bagad, Domb, Thaler — ePrint 2026/587.
**Implementation**: `crates/lattice-sumcheck/src/{extrapolate,
multiproduct, fastprover}.rs` (+ `tests/fastprover.rs` and
`examples/fastprover_bench.rs`).
**Status**: implemented (the shifted-recurrence extrapolation, the
multiproduct evaluation engine, the round-batched window prover, the
optimized tail, the split-eq factor source), 10 new tests (27 total in
the crate's new modules), zero clippy warnings, benchmarked with
multiplication-count instrumentation. **Byte-identical transcripts to the
baseline prover** — the protocol, verifier, and Fiat–Shamir flow are
unchanged.

## The idea

The sumcheck prover's cost is (a) the per-round product
`Π_k p_k(r_{<i}, t, x')` evaluations and (b) the per-round factor
re-binding. Three optimizations attack both, trading big-by-big (bb)
field multiplications for small-by-small/small-by-big (ss/sb) ones:

1. **The multiproduct engine** (§4, Procedures 1–2): the product of d
   multilinear factors over a window of v variables, in evaluation form
   on the grid `U(d+1)^v` — split the factors in half, recurse,
   extrapolate both half-product grids to the full grid axis-by-axis,
   multiply point-wise. bb count: `Θ(d^v)` for fixed `v ≥ 2`,
   `Θ(d log d)` for `v = 1` — versus the naive `(d+1)(d−1)` per point.
2. **The window** (§5.1, C.4.1): the first `v` rounds delay binding —
   per boolean suffix `x' ∈ {0,1}^{ℓ−v}` and per term, the window
   polynomial's grid is materialised once; each round's message is read
   off the grids with composed Lagrange weights:
   `g_j(t) = Σ_{x'} Σ_{u∈U^{j−1}} W_j[u]·(Σ_{w∈{0,1}^{v−j}} G[u,t,w])`.
   Cost model (C.4.1): `d²M/2^v` bb + `M·((d+2)/2)^v` sb — the optimum
   `v* = log_{d+2}(d²κ)`.
3. **The split-eq factor source** (2025/1117 §5, 2026/587 §6): the
   equality factor `eq(w, ·)` is never materialised over the full
   hypercube — its window tables and post-window bindings derive from
   the prefix/suffix factorisation
   `eq(w, (b, x')) = eq(w_{<v}, b)·eq(w_{≥v}, x')`.

## What was implemented

### 1. The shifted-evaluation recurrence (`extrapolate.rs`, Appendix D.1)

Extrapolating a degree-≤k polynomial's evaluations `{∞, 0..k−1}` to
further integers via the sliding stencil

```text
p(k+c) = k!·p(∞) + Σ_j (−1)^{k−1−j}·C(k,j)·p(c+j)
```

— the same small-integer stencil (`±C(k,j)`, `k!` for the ∞ column) for
every new point, exactly Appendix D.1's `V₈V₄⁻¹` shift structure. Every
multiplication is small-by-big; the accumulation is **signed i128 with
exact reduction** — u128 wrapping arithmetic is subtly wrong here
because `2^128 ≢ 0 (mod p)` for Goldilocks (a live factor-2^32-off
corruption caught by the multiproduct tests). The ∞ slot carries the
degree-k leading coefficient (Lemma 2.2's interpolation-with-∞).

### 2. The multiproduct engine (`multiproduct.rs`, Procedures 1–2)

- `multi_product_eval(tables, v)` — the product of n factors over the
  grid `{∞, 0..n}^v` (side n+2 per axis: the ∞ slot for the
  extrapolation machinery + every message integer `0..n` present).
- `multi_extrapolate` (Procedure 2): axis-by-axis extension with the
  shifted recurrence; `Θ(k·(h−k))` sb per grid line, zero bb.
- The base case: a single factor's boolean table → the `U(1)^v` grid
  (`(hi−lo, lo)` per axis, innermost-first).
- The ∞ slots compose correctly through the recursion:
  `(deg-m lead)·(deg-(n−m) lead) = the degree-n lead`.
- Verified against direct MLE evaluation at every grid point (v=1 for
  n ∈ 2..16; v=2 mixed-∞ axes) and the closed-form bb recurrence
  `Σ_levels (n/level)·(level+1)`.

### 3. The round-batched window prover (`fastprover.rs`)

- Per suffix × per term grids over the uniform side `d+2` — **lower
  degree terms are padded with constant-one factors** rather than
  post-extended (extending a degree-n grid to a larger domain would
  carry stale ∞-slot semantics: the slot holds the degree-n lead while
  the larger domain's formulas expect the degree-(d+1) one).
- Round extraction: nested loops over `(u ∈ side^{j−1}) × t × (w ∈
  {0,1}^{v−j})` with precomputed strides — no per-entry digit
  divisions; the boolean tail is summed with additions before the single
  weight multiplication.
- The binding weights are the **finite-point Lagrange basis over
  `{0..d}`** with the ∞ slot at weight zero (the d+1 integer values
  determine the degree-≤d product exactly — the axis's ∞ slot carries a
  different degree convention than the (d+2)-point domain).
- The composed weights `W_j = L(r_1) ⊗ ⋯ ⊗ L(r_{j−1})` (new axis =
  lowest digit, matching the flat-index layout).
- Post-window prefix adaptation: the factors bound to `r_{<v}` through
  the boolean eq-weights (interleaved doubling — the new variable's bit
  is the LSB of the flat index, first variable = MSB).
- The tail rounds reuse the baseline's SIMD kernel path
  (`sum_products` + `fix_variables`) — performance parity with the
  linear-time prover for rounds `v+1..ℓ`.
- **The eq tables are built with the interleaved doubling order** — the
  two-block concatenation order silently swaps the bit significance
  (caught by the bit-identity tests).

### 4. The split-eq factor source

`prove_fast_with_eq(vp, claim, transcript, opts, eq_indices, w)`: the
designated factors' window tables (`prefix[b]·suffix[x']` — 2^v small
multiplications per suffix) and post-window bindings
(`(Σ_b eq_r[b]·prefix[b])·suffix[x']` — one scalar times the suffix
weights) are derived from `w`; the `2^ℓ` eq table is never built.

## The bit-identity contract

`fast_prover_bit_identical_to_baseline` pins, for five virtual-polynomial
shapes (d ∈ 2..5, mixed-degree terms) × windows 0..3: identical round
polynomials, identical challenges, identical final claims, identical
factor claims, and identical post-proof transcript ratchets. The
eq-split variant is pinned against the materialised-eq baseline in
`fast_prover_zerocheck_style_with_eq_split`.

## Benchmarks (`examples/fastprover_bench.rs`) — the honest Goldilocks finding

The instrumentation confirms the paper's asymptotics exactly: at
`d = 2, M = 2^14`, the total bb count drops 65 528 → 24 573 → 12 289 →
6 161 → **3 153** as the window grows 1→4 (the 2^v tail factor), while
the sb count grows `M·((d+2)/2)^v` as C.4.1 predicts.

But **the wall-clock trade is negative on Goldilocks**: a 64-bit field
has `κ = cost(bb)/cost(sb) ≈ 1` (a field multiplication is one u128
multiply plus a cheap reduction — there is no limb-count hierarchy), so
the paper's own optimum `v* = log_{d+2}(d²κ)` collapses to ≈1 and the
window cannot beat the baseline's SIMD 8-lane kernels. The optimization
regime the papers target — `κ ≈ 2N²+1 ≈ 33` for 256-bit Montgomery
fields (BN254/BLS12-381 scalar fields, Spartan-in-Jolt's setting) —
is realised in this workspace by `lattice-projsumcheck`'s 4-limb CIOS
Fp256; porting the window prover to that engine is the follow-up that
would realise the 2.5–4× (small-value) and 1.7–2.2× (high-degree)
speedups the papers measure. What does transfer to Goldilocks today:
the multiproduct engine (bb-count wins for high-degree univariate
products), the split-eq memory win (no `2^ℓ` eq materialisation), and
the bind-once tail structure.

## Honest deviations (the ledger)

1. **No small-value lazy kernel on Goldilocks** — the §3 ss/sb/bb
   hierarchy is vacuous on a 64-bit field (documented above); the
   deferred-reduction discipline is applied where it does pay (the
   stencil accumulation, the TTRP crate's i128 grid accumulation).
2. **The streaming window schedule** (§5.2, `EvalProductStream_k`,
   O(M^{1/k}) space) is not implemented — the window prover stores the
   per-suffix grids (linear space, like the baseline's factor arrays);
   the schedule belongs with the streaming engine in
   `lattice-streaming`.
3. **Univariate skip** (§7) modifies the protocol and verifier
   interpolation degree; not implemented (documented as future work —
   it composes with the window machinery but changes the proof format).
4. **The ∞-slot degree semantics** of the U-domains differ from the
   paper's Figure-2 domains (`Ũ_d = U_d \ {1}` with the ∞ point): our
   engine sends `g(0..d)` (integer points) as the existing verifier
   expects, so the grids carry `{∞, 0..d}` with the ∞ slot at the
   degree-n convention and the binding weights use the finite basis.

## Tests

`extrapolate.rs`: the stencil equals ±C(k,j) with `k!`; extrapolation
matches direct polynomial evaluation (degree 3 and 8, out to 20 points);
the boolean-pair-to-U₁ conversion. `multiproduct.rs`: univariate products
for n ∈ 2..16 against direct evaluation (including the ∞ entry and the
naive-bb superiority for n ≥ 4); the v=2 multivariate grid at every
point including mixed-∞ axes; the closed-form bb recurrence.
`tests/fastprover.rs`: the five-shape × four-window bit-identity suite;
the eq-split equivalence; window-0 tail equivalence; stats reporting;
the default drop-in.
