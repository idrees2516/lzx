# The small-value windowed projective sum-check (2025/1117 + 2026/587 §5, applied to 2026/762)

**Status: implemented** (`lattice-projsumcheck/src/svsc.rs`, ~700 lines,
8 tests + `svsc_bench`)

The ss-class weighting restructure over `Fp256`: one integer-arithmetic
grid pass answers the first `v` rounds, with byte-identical transcripts
to the round-by-round reference prover (pinned by test).

## What changed

The baseline binds every round with a field challenge, so every message
coefficient from round 2 on is a **bb** Montgomery multiplication. At
the measured **κ ≈ 61.6** (bb:ss cost ratio; the paper's model
`2N²+N = 36`), the windowed prover restructures the weighting:

* **The window grid** — `q(X₁..X_v) = Σ_{x'∈{0,∞}^{ℓ−v}} P(X₁..X_v, x')`
  materialized as its evaluations on `U_d^v = {0,1,…,d−1,∞}^v` — every
  intermediate a **u128 integer**: the factors' window bindings
  `f(0) + u·f(∞)` multiply by small integers, the suffix term products
  accumulate unreduced.
* **Intra-window rounds** (Appendix C.5.5's two tracks): a scratch
  sum-out (f(0) + f(∞) per free axis — the projective round identity)
  produces the message `[s_j(∞), s_j(1..d−1)]` straight off the axis;
  the live grid binds at the challenge via Lemma-2.2 interpolation with
  the **upper-limb `mul_upper_limb` short-circuit** — the two
  optimizations stack.
* **The fail-closed u128 bit-width precondition** — the paper's
  small-value validity condition:
  `(ℓ−v) + (v−1) + d·(v·⌈log₂ d⌉ + κ_v) + log₂ c_max + margin ≤ 128`.

## Measured

| ℓ | v | reference | windowed | speedup |
|---|---|-----------|----------|---------|
| 12 | 3 | 55.0 ms | 27.2 ms | 2.02× |
| 16 | 4 | 878.2 ms | 395.0 ms | 2.22× |
| 18 | 4 | 3.5 s | 1.6 s | 2.23× |

(d = 2, 33-bit values, release build.) The honest gap to the paper's
2.5–4×: the plain DFS grid without `MultiProductEval`'s shared
extrapolation, the post-window rounds on the reference path, and no
AVX-512 kernels yet.

## The Fp256 modulus fix

Landed with this module: `BN254_FR[2]` carried a digit transposition
(`0x…8158d2` → `0x…81585d`) — the committed field computed modulo a
wrong modulus (off by `117·2^128`), masked until now because the crate's
tests were self-referential (CIOS vs `reduce_wide_ref`, never against
the true BN254 scalar field). Every prior Fp256 "result" was garbage in
the true field — a security-critical fix.

## Honest scope

Pure products (every term the same arity — the Shout/read-RA shape);
mixed-degree instances stay on the Goldilocks engine (the mixed message
needs the value at `d`, a `(d+2)`-point grid). The streaming schedule
(`Stream_k`'s geometric window growth) is exposed via `WindowSchedule`;
the progressive schedule is the `lattice-streaming` follow-up.
