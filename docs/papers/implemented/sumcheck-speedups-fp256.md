# Sum-Check Prover Speedups — the Fp256 window port (ePrint 2025/1117 + 2026/587)

Status: **IMPLEMENTED** (`crates/lattice-projsumcheck/src/fastprover.rs`,
`crates/lattice-projsumcheck/src/fp256.rs` extensions,
`crates/lattice-projsumcheck/examples/fastprover_bench.rs`; the Goldilocks
engine lives on in `crates/lattice-sumcheck/src/{fastprover,extrapolate,
multiproduct}.rs`).

## What this adds over the Goldilocks engine

The Goldilocks port proved the papers' machinery (the multiproduct grid
engine, the shifted-evaluation recurrence, the round-batched window, the
byte-identity discipline) but could not demonstrate the wall-clock win:
on a 64-bit field a field multiplication costs one `u64` multiply, so the
big/small cost ratio is κ ≈ 1 and the window's bookwork never pays. The
Fp256 port targets the regime the papers actually benchmark: the 4-limb
CIOS field (BN254 Fr), where

* **bb** (two full-width Montgomery operands) = a 36-native-multiplication
  CIOS (measured 27 ns);
* **sb** (Montgomery × small canonical) = the zero-limb-skipped CIOS
  ([`Fp256::mul_small`], measured 11.8 ns — **2.7×** vs bb);
* **ss** (two small canonical integers) = one native `i128` multiply —
  and because `p > 2^254 > 2^127`, every small value is *exact* in `i128`
  with no reduction at all (the grid engine runs on pure native
  arithmetic).

## The module

* `FpVirtualPolynomial` — factors are `Small(Vec<i128>)` (digit/binary
  tables — the SV regime) or `Big(Vec<Fp256>)` (canonical full-width);
  mixed instances take the generic path with free canonical conversion
  of the small tables.
* **SV path** — the whole window phase is exact `i128`: the per-suffix
  multiproduct grids over `U(d+1)^v` (Procedure 1 with the
  shifted-evaluation stencils), the per-round message extraction
  `msg[t] = Σ_{x'} Σ_term c·Σ_u W[u]·(Σ_w G[u,t,w])` with `c·acc` folded
  into the `i128` layer (one ss) and a single sb weighting kernel
  `CIOS(W̄, small) = W·small` per contribution.
* **Generic path** — Montgomery grids (the multiproduct bb-reduction of
  Procedure 1 — `(d log d)^v` instead of naive `(d+1)^{v}·d`), the same
  window extraction with Montgomery accumulators.
* **The tail** (both provers) — bind-once + per-t affine values
  `v̄_k(t) = lō + CIOS(Δ̄, t̄)`, the §4 structure, all-Montgomery so
  every CIOS has a Montgomery operand.
* **Challenges** — the upper-limb set (2026/762 §5): always
  canonical-valid (no rejection), and every multiplication by the
  challenge's Montgomery form takes the CIOS zero-limb skips (~20 native
  mults).
* **Byte-identity** — `prove_fast` at every window size produces the
  same proofs and challenge paths as `prove_baseline` (SV, generic, and
  mixed suites), pinned by tests; the verifier replays the canonical
  round identities and finite-node Lagrange.
* **`optimal_window(d, κ, ℓ)`** — the Lemma-5/C.4.1 minimiser
  `v* = log_{d+1}(d²κ)`; `kappa_limbs(N) = 2N²+N`.

## The form calculus (the load-bearing implementation detail)

`CIOS(x, y) = x·y·R^{-1}` for operand *forms* `x = X·R^{f_x}`,
`y = Y·R^{f_y}` gives `CIOS = (XY)·R^{f_x+f_y-1}`:

| operands | result form | value |
|---|---|---|
| mont × mont | mont | `XY·R` — the TRUE product, Montgomery |
| mont × canon | canon | `XY` — the TRUE product, canonical |
| canon × canon | **neither** (`XY·R^{-1}`) | — |

So every hot-loop multiplication keeps a Montgomery operand (the
`(0,0)` case is a form error, caught immediately by the byte-identity
tests), and multiplying by Montgomery constants (`t̄`, `c̄`, the
challenge) is *form-preserving* while scaling truly. The field layer is
differentially pinned against an independent schoolbook +
`reduce_wide_ref` path (`field_layer_differential`), including
`to_mont/from_mont` roundtrips and `to_mont(1) == R`.

## The honest findings (the deviation ledger)

1. **A pre-existing modulus defect, found and fixed.** `BN254_FR`'s limb
   2 was mistyped `…58d2` for `…585d` — a digit transposition that
   silently replaced the prime field with a **composite modulus** (we
   verified compositeness; the true limb restores the canonical BN254
   scalar field `21888…95617`). Every local test had still passed
   because the reference reducer shared the same wrong constant. The
   port's cross-constants (`R`, `R²` computed against the true field)
   surfaced it. The fix is in `fp256.rs` with a comment documenting the
   defect.
2. **The measured κ regime differs from the papers' model.** The papers'
   `κ ≈ 2N²+N = 36` is the **bb/ss** ratio, and Lemma 5's optimum plugs
   it into a cost whose window term is *weighting multiplications* —
   which on our engine are **sb-class** (big Montgomery weights × small
   grid sums), measured at **2.7×**, not 36×. With the honest ratio the
   optimum collapses to `v* ≈ log_{d+1}(d²·2.7) ≈ 2` and the wall-clock
   on digit-table instances (`d = 2..3`, `M = 2^{13..14}`) is
   **break-even** (best ≈ 1.0×): the tail's `d²·M/2^v` bb shrinks as
   fast as the window's `M·((d+2)/2)^v` sb grows. The grid construction
   itself is pure ss and essentially free (confirming that half of the
   papers' claim); delivering the 2.5–4× end-to-end requires an
   ss-class weighting — e.g. small challenges (bounded soundness) or a
   restructured grid binding (Appendix C.2's grid-based linear-time
   emulation) — documented as the follow-up, not claimed as done.
3. **Window weights padded to the grid side.** The composed Lagrange
   weights include the ∞ slot at weight ZERO (the `{0..d}`-node
   interpolation never reads it) — the grid layout is side `d+2`.
4. **Not ported:** the split-eq factor source (§6, the decomposable-eq
   optimization) — the eq factor is big-valued, so it rides the generic
   path; the univariate skip (§7) is protocol-modifying and out of
   scope.

## Tests

`sv_byte_identity_all_windows` (3 shapes × 5 windows),
`generic_byte_identity`, `mixed_byte_identity`, `verify_roundtrip_sv`,
`tampered_round_rejected`, `wrong_claim_rejected`,
`field_layer_differential`, `small_inverse_correct`,
`interpolate_matches_direct`, `small_grid_matches_naive`,
`sv_stats_are_ss_dominant`, `optimal_window_matches_model`.
