# The Sum-Check Protocol over the Monomial Basis, and Other Optimizations

**Paper**: Dao, Biswas, Eagen, Milson, Papini, Thaler — ePrint 2026/762.
**Implementation**: `crates/lattice-projsumcheck` (+ the projective kernels
in `lattice-core/src/field_simd.rs`).
**Status**: implemented (protocol, tables, batching, Fp256 challenges),
27 tests, zero clippy warnings, benchmarked.

## The idea

The Boolean hypercube `{0,1}^n` is not the optimal interpolating set for
sum-check. Switching to the **infinity hypercube `{0,∞}^n** — where
evaluating a multilinear polynomial at ∞ means extracting a monomial
coefficient (Proposition 3.1: the monomial coefficients of the `{0,∞}`
interpolant are exactly the truth-table values, slot for slot) — makes
the whole protocol subtraction-free and aligns it with the monomial-basis
commitments (WHIR; this codebase's compact Ajtai opening).

## What was implemented

### 1. The projective MLE and binding (§3, Cor. 3.2)

- `MonomialMle` — coefficient-form multilinear polynomials sharing the
  `DenseMle` index layout (variable 0 = MSB), so `from_truth_table` is an
  array identity and the whole pipeline (trace → coefficients →
  sum-check → compact Ajtai opening) uses one representation with **no
  Möbius conversion anywhere** (§4.3's PCS alignment).
- **Subtraction-free binding** `p(r,x') = p(0,x') + r·p(∞,x')` — one
  multiplication and one addition per surviving coefficient, as the
  AVX-512 kernel `bind_projective_first_half_in_place` (plus the
  pairwise variant), saving `d·(2^n − 1)` subtractions per proof
  (Proposition 4.1).
- Möbius transforms both directions (`mobius_to_boolean_coeffs` /
  `mobius_from_boolean_coeffs`) for interoperation, per-block
  butterflies, roundtrip-tested.
- The affine-bridge identity `p(r) = f̂(φ(r))·Π(1+r_i)` with
  `φ_i = r_i/(1+r_i)` — the cross-validation anchor against the
  Boolean engine.

### 2. The projective sum-check protocol (Figure 3, Theorem 3.3)

- Round identity `s_i(0) + s_i(∞) = C_{i−1}`; the prover sends the
  compressed set `Ū_d = {∞} ∪ {1..d−1}` and the verifier derives
  `s_i(0) := C_{i−1} − s_i(∞)` — **one field element saved per round**
  (measured 1.5× smaller proofs at degree 2: 320 B vs 480 B at n=20).
- Lemma 2.2 interpolation with the ∞ node
  (`interpolate_with_infinity`).
- **Mixed-degree handling** (the implementation's own extension beyond
  the paper's pure-product setting): for virtual polynomials whose terms
  have different degrees, the round polynomial's ∞-value (the identity
  datum: `Σ_b g(r_{<i}, ∞, b)`) differs from its leading coefficient
  (the interpolation datum) — pure products keep the compressed d-entry
  form; mixed instances send d+1 entries with the finite-node Lagrange
  finish. Both paths roundtrip-tested.
- Terminal factor claims are coefficient-form evaluations at `r` —
  directly consumable by the compact opening.

### 3. Structured polynomials (§4.2 + Appendix A)

- The per-pair dictionary (Figure 1) as `PairFactor` — every entry
  Möbius-verified against its discrete gate (Prop. 3.1): `AND → X·Y`,
  `XOR → X+Y` (multiplication-free), `OR → X+Y+XY`, equality → `1+XY`,
  complement → the constant 1, ignored → `(1+X)(1+Y)`.
- `eq` closed form `Π(1 + r_i·Y_i)` and the full-domain table via the
  `(e, e·r)` recurrence — the left half of every doubling step stays in
  place (a free copy), measured **1.72× faster** than the Boolean table
  at n=20 (Table 4's 1.94× on BN254; Goldilocks' cheap subtraction
  trims the edge).
- `LT` closed form `Σ_v Y_v·Π_{u<v}E_u·Π_{u>v}Ω_u` and the
  subtraction-free doubling recurrence (one mul + one add per output
  pair — the recurrence reuses the even entry, beating the paper's
  count).
- The Jolt table families: bitwise (And/Andn/Or/Xor), comparisons
  (Eq/LT word-level), Movsign, XORROT(ρ)/XORROTW(ρ), Rev8W,
  MulUNoOverflow, Pow2 — all Möbius-verified against their discrete
  semantics (exhaustive at w=4/8; Rev8W structurally at w=16).
- The pcnext `shift` kernel in both bases (Boolean MLE here; projective
  in `tables.rs`) — the carry-chain formula including the r-side
  condition `Π_{u>v}(1−r_u)` that the carry clears.

### 4. Claim-preserving dummy rounds (§B.1/§B.2)

`prove_batched`: front-loaded batching where dormant instances send the
**honest constant `H(X) = claim`** (`H(∞) = 0` satisfies the round
identity) — no `2^{n_max−n}` pre-scaling, no per-round halving, no
renormalization at the activation boundary. Dormant coefficient arrays
pass through binding untouched (their high halves are identically zero).
Tested with three simultaneous sizes (7/5/2 variables).

### 5. Upper-limb Montgomery challenges (§5) + grinding (§5.3)

`fp256.rs` — the full 4-limb CIOS field over BN254's Fr:

- `mul`: the complete CIOS (36 native multiplications), bootstrap-tested
  against a shift-and-subtract reference reduction.
- `mul_upper_limb`: the short-circuit path for challenges whose
  Montgomery form has two zero low limbs — phase 1 skips the `b[i] = 0`
  steps (8 products) and phase 2's `m = 0` steps are pure shifts (10
  products): **18 of 36 multiplications eliminated**, bit-exact with
  full CIOS (chained, tested).
- `sample_upper_limb`: the Fiat–Shamir instantiation — truncate the
  transcript hash to λ = 125 bits, left-shift into the upper limbs,
  treat as already-Montgomery (|S| = 2^125, top 3 bits cleared for CIOS
  overflow safety).
- `grind`/`verify_grind`: the proof-of-work step closing the 12-bit gap
  to 128-bit security (a cheating prover must re-grind per modified
  commitment; the honest prover pays `2^γ` hashes once, outside the
  sum-check hot loop).
- Measured: chained mul **1.45×** (the paper's 1.92× on BN254 hardware
  with arkworks' baseline; our scalar-fallback CIOS keeps less of the
  edge), projective binding **1.19×** with upper-limb challenges.

## Benchmark results (Goldilocks, this repo, release build)

| kernel | Boolean baseline | projective | speedup |
|---|---|---|---|
| binding, 2^20 coefficients | 1.19 ms | 0.91 ms | **1.31×** |
| eq full-domain table, n=20 | 4.64 ms | 2.70 ms | **1.72×** |
| degree-2 sum-check, n=20 | 36.0 ms (480 B) | 24.7 ms (320 B) | **1.46× / 1.50× smaller** |
| degree-2 × eq, n=20 | 64.1 ms (640 B) | 48.6 ms (480 B) | **1.32× / 1.33×** |
| degree-2, n=22 | 170 ms | 123 ms | **1.38×** |
| Fp256 chained mul ×10^6 | 27.5 ms | 19.0 ms | **1.45×** |
| Fp256 projective binding 2^20 | 29.5 ms | 24.9 ms | **1.19×** |

The paper's measured end-to-end gains (≈10% on BN254/Fp128 eager, Table 6)
are against baselines carrying all of [4]'s optimizations; our Boolean
baseline is the pre-existing engine, so the deltas above mix the basis
change with the eager→projective kernel differences. The subtraction-free
identity, the compressed proofs, and the (e, e·r) table recurrences are
the structural wins; the Goldilocks field's cheap subtraction (no
conditional-borrow asymmetry like BN254's) mutes the per-subtraction
savings exactly as the paper's field-dependent analysis (§4.2, §6.6)
predicts.

## Not implemented

- The `{1,∞}^n` shifted basis variant (§4.2's alternative for
  add-expensive fields) — Goldilocks addition is cheap, so the variant
  buys nothing here.
- The BB Ext5 packed-kernel code-shaping case studies (§C) — the
  documented lessons (explicit temporaries, asymmetric inlining,
  intentional reloads) are AVX-512/NEON-specific register-pressure
  techniques; our kernels follow the existing `field_simd` discipline.
- The paper's 40-table Jolt inventory in full — the families here cover
  every closed-form pattern (paired dictionary, raw-index, recurrence,
  prefix-suffix combine) with exhaustive verification at feasible
  widths; the remaining tables are instantiations of the same patterns.
