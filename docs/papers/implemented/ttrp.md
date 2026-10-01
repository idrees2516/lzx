# Succinct Shortness Check Under a Few Kilobytes via Tensor Train Random Projections

**Paper**: Zhiyuan Geng, Maxime Plançon — ePrint 2026/2146.
**Implementation**: `crates/lattice-ttrp`.
**Status**: implemented (cores, both projection paths, the Π₀/Π_TTRP
reduction of knowledge with sumcheck linearisation, the verifier's
tensor-structured evaluation, the statistical machinery, the §7.2
parameter search), 15 tests, zero clippy warnings, benchmarked.

## The idea

Unstructured JL range proofs (LaBRADOR) make the verifier process a
`λ × m` ternary matrix (Table 1's 6492 MB); structured JL (RoK and Roll)
keeps the matrix small but blows the projection length up to `m/ρ`, which
must then be committed. The Tensor Train representation gets both:
`O(λ)` core entries AND a `O(λ)`-length projection sent in the clear.

A TT row is a chain of µ core tensors `M₁ ∈ Z^{1×d×c},
Mᵢ ∈ Z^{c×d×c}, M_µ ∈ Z^{c×d×1}` with i.i.d. `D_ghl` entries
(0 w.p. ½, ±1 w.p. ¼ each). Row `j`'s flat entry at column
`n = h·φ + ℓ` (spatial digits MSB-first, then coefficient digits) is the
matrix chain product `M₁(n₁)·M₂(n₂)⋯M_µ(n_µ)`.

## What was implemented

### 1. Cores and the TT structure (`cores.rs`)

- `CoreTensor` (flat `[slice][row][col]`), the deterministic `χ_TT`
  sampler (SHAKE-256 streams, 2 bits per entry — exactly `D_ghl`),
  `Mat` flattening, and the left-to-right mixed-product chain
  materialisation (Fact 1) with the naive chain product as the tested
  ground truth (`tt_materialization_matches_naive`).
- `TtrpParams` with the validated invariant `ℓ·(µ₁+µ₂) = ν + log₂ φ`
  (spatial cores index the `m̄r` ring positions; coefficient cores the
  `φ` coefficients; together `m̄r·φ = d^µ` integer columns) and the
  representation-size formula `k·d·(2c + (µ−2)c²)` (Table 4's "Repr."
  column).

### 2. The two projection paths (`projection.rs`)

- **Integer view** `y0 = M_Z·cf(v) mod q` — the right-to-left
  contraction of Lemma 6's proof: `v^(µ) = x`,
  `v^(i−1) = (I_{d^{i−1}} ⊗ Mat(Mᵢ))·v^(i) mod q`, per row, with lazy
  per-block reduction. O(k·c·m̄r·φ) integer work, never materialising
  the row.
- **Ring view** `y = M·v̄` via the **S/W split**: the row factorises as
  `M[j,h] = (Π_{p≤µ₁} M_p(n_p(h)))·W^{(j)}` — a spatial chain
  `S^{(j)} ∈ Z^{m̄r×c}` times a coefficient-chain vector `W^{(j)} ∈ R^c`
  (built with negacyclic X-power twists, no ring multiplications) —
  hence `y^{(j)} = ⟨W^{(j)}, t^{(j)}⟩` with
  `t^{(j)} = S^{(j)ᵀ}·v̄` accumulated coefficient-wise in i128.
- The σ⁻¹ conjugation automorphism `conj` and the central power-of-two
  cyclotomic identity `ct(a·b̄) = ⟨cf(a), cf(b)⟩` — pinned by
  `constant_term_identity_and_sw_projection` against the materialised
  row for every row (`ct(y^{(j)}) == y0_j`).

### 3. The protocol (`protocol.rs`, Figure 1 + Corollary 1)

1. **Cores** derived from the transcript after a public attempt counter
   (the honest prover retries under the Markov ½ completeness; the
   verifier replays the accepted attempt).
2. **y0** sent in the clear; the verifier checks `‖y0‖₂ ≤ B̂`
   (`b_hat_squared`: exact u128 arithmetic `k·c^{µ−1}·B²/2^µ`).
3. **Γ ∈ Z_q^{k′×k}** sampled after y0; the prover responds
   `y1 = Γ·y`; the verifier checks the constant-term identity
   `ct(y1ᵢ) = Σ_j Γ_{ij}·y0_j` (u64 accumulation).
4. **γ ∈ R^{k′}**; with `γ̃ = γ·Γ ∈ R^k` and `y* = Σ γᵢ·y1ᵢ`, the single
   batched claim `Σ_{z∈{0,1}^ν} mle(m*)(z)·mle(v̄)(z) = y*` with
   `m* = Σ_j γ̃_j·M[j]`.
5. **ν-round degree-2 ring sumcheck** on
   `g(z) = mle(m*)(z)·mle(v̄)(z) − y*·2^{−ν}` — round messages
   `[g(0), g(1), g(2)] ∈ R³`, uniform ring-element challenges
   (unbiased per-coefficient u32 rejection), the constant folded as
   `y*·2^{−j}` per round.
6. **Terminal**: the prover's `w_r = conj(mle(v̄)(r))` (it conjugates its
   final bound array — no extra evaluation); the verifier checks
   `C_ν = mle(m*)(r)·conj(w_r) − y*·2^{−ν}` with `mle(m*)(r)` computed
   **locally by tensor contraction** and returns the evaluation claim
   `(conj(r), w_r)` for the caller's outer `Ξ_poly` relation.
- The prover's `m*` construction folds γ̃ into the coefficient chains
  (`W̃^{(j)} = γ̃_j·W^{(j)}`, k·c ring multiplications total) and
  accumulates coefficient-wise with deferred reduction — the paper's
  O(k·c·m̄r·φ) prover cost without any `k×m̄r` materialisation.

### 4. The verifier's tensor-structured evaluation (Lemma 6)

`tensor_eval_mstar`: chunk the ν challenges into µ₁ blocks of ℓ; per
spatial core, evaluate the MLE of its slices at the chunk via the
multilinear basis (built by doubling — **challenges consumed in reverse
order so the digit's MSB pairs with the chunk's first challenge**); chain
the core evaluations into the 1×c boundary vector `V^{(j)}`; multiply by
the γ̃-scaled coefficient chain; sum over rows. Shared basis: O(µ₁·2^ℓ)
multiplications; per row O(µ₁·c² + c) — never touching the
`m̄r`-long rows. Cross-validated against successive-binding MLE
evaluation end-to-end.

### 5. The statistical machinery (`bounds.rs`)

- Lemma 3 moments, Lemma 4 / Theorem 2 overflow bounds
  (`1/2 + 2^{µ−1}/2^{c+1}` per row, k-row amplification, the
  `c ≥ ⌈log₂(µ−1)⌉+δ` prescription), Lemma 5 / Theorem 3 Cantelli
  ℓ₂ concentration, the completeness bound B̂, the slack
  `√(k·η₂/θ)`, and the no-overflow ceiling
  `q/(2^{µ+1}·√(θc^{µ−1}m))`.
- The §7.2 parameter search (Table 4's generator): iterate
  `(ℓ, µ₁, µ₂, c, k)` with θ derived from the slack target, discard
  configurations violating the overflow or Cantelli bounds, minimise the
  representation. The searched optimum lands in the paper's own league
  (~2.5M core entries at m = 2^20 coefficients).

## Benchmarks (`examples/ttrp_bench.rs`)

| instance | m̄r | φ | k | prove | verify (tensor) | naive row pass | proof |
|---|---|---|---|---|---|---|---|
| small | 2^8 | 16 | 16 | 5 ms | 2 ms | 1 ms | 1.8 KB |
| mid | 2^12 | 64 | 32 | 1.5 s | 79 ms | 233 ms | 10.6 KB |
| large | 2^14 | 64 | 48 | 4.6 s | 139 ms | 1405 ms | 12.2 KB |

The verifier's tensor evaluation beats the JL-style row materialisation
**10× at m̄r = 2^14** — the paper's Table-1 axis (the 6492 MB → 0.2 MB
verifier-matrix win). The prover cost is the honest O(k·c·m̄r·φ) paper
complexity (integer contraction + S/W accumulation dominate).

## Honest deviations (the ledger)

1. **Eq. (11) factor-2.** The paper's printed second moment
   `E[y²] = (c/2)^{µ−1}·‖x‖²` contradicts its own A.2 derivation
   (`E[pJ pJᵀ] = c^{µ−1}·σ^{2µ}·I` with `σ² = Var(D_ghl) = ½`), which
   gives `c^{µ−1}/2^µ·‖x‖²`. Our Monte-Carlo test
   (`monte_carlo_lemma3_moments`, 4000 trials) pins the A.2 value
   (measured 2369937 vs the printed formula's 4808000 — a clean factor
   2). `eta2`, `completeness_bound`, `slack`, and `b_hat_squared`
   implement the corrected value; the fourth-moment bound (Lemma 9's
   upper bound) is unchanged.
2. **Challenge space.** Lemma 1 assumes a challenge set with
   pairwise-invertible differences and error `ℓ·deg/|C|`; over the
   power-of-two cyclotomic `R_q` (totally split for NTT primes) the
   honest per-round CRT bound for uniform ring challenges is
   `deg·φ/q`-shaped, so the crate follows the workspace's 32-bit-q
   interactive posture (the same documented deviation as
   `lattice-salsa/ring_sc`): soundness amplification / grinding / a
   larger modulus belong to the outer composition. k′ is a caller
   parameter because the knowledge error carries the `q^{−k′}`
   Γ-collision term.
3. **k′-row aggregation.** The proof carries `y1 ∈ R^{k′}` in the clear
   (the paper's Remark 1 Lift-and-Batch size optimisation is future
   work); the sumcheck is γ-batched to a single instance as §4.2
   prescribes.
4. **The Ajtai layer.** The statement digest binds the caller's
   `Ξ_poly` context; `T = A·W` commitments and the claim-ledger wiring
   live with the caller (the zkvm pipeline's stage-4 pattern).

## Tests (`tests/ttrp.rs`)

TT materialisation vs the naive chain product; sampling determinism and
D_ghl marginals; the ct identity + S/W projection vs the materialised
row; the S/W row factorisation; Monte-Carlo Lemma 3 moments and the
Theorem 3 short/long separation; protocol completeness with the outer
evaluation claim; tampered y0 / round / w_r / y1 rejections; the
long-witness (16× norm) rejection over 12 fresh attempts; the
mid-parameter end-to-end; the parameter-search constraints.
