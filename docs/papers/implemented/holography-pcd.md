# Proof-Carrying Data via Holography Accumulation (ePrint 2026/538)

**Status: implemented in depth** — `crates/lattice-holo` (~3.6k lines
with tests), Paslis–Ràfols–Zacharakis (UPF / HPI).

## What was implemented

### The polynomial layer (`poly`)
- Both representations: `ν = 1` (the roots-of-unity domain `H` — BN254
  `F_r` has 2-adicity 28, the domain generator fixed per `n`) and
  `ν = log n` (the boolean cube).
- The Lagrange bases `λ_h` (the paper's §2 form
  `u_H(X)/(n·h^{n−1}(X−h))` — the PDF extraction's `u_H/(n(X−h))` is off
  by the `h^{n−1}` factor; the implemented basis satisfies
  `λ_h(h) = 1`, `λ_h(h') = 0`, pinned by the partition-of-unity test),
  the vanishing polynomial `u_H`, the identity polynomial `Λ(X,Y)` in
  both closed forms (univariate
  `(u_H(X)Y − u_H(Y)X)/(n(X−Y))` with the diagonal's analytic
  continuation; multivariate `∏(XᵢYᵢ + (1−Xᵢ)(1−Yᵢ))`), and the matrix
  polynomials `M(X,Y) = λ(Y)ᵀ M λ(X)` (cross-checked against the direct
  bilinear forms in tests for both ν).

### The relation family (`relations`)
- `R_PCE`, `R_PCEP`, `R_hbPCE` (Definitions 4–5), `R_CCS` (Definition 6)
  with satisfaction checks and a solution sampler, and the central
  **`R_GBF`** (Definition 7) — the generalized bilinear forms
  `s = (Σ c_l ∘ u)ᵀ (Σ c_r ∘ (M v))` — with the `R_GBF,α` and
  `R_GBF,α,β` specializations (implicit `λ(α)`/`λ(β)` component slots)
  and `build_gbf_alpha_beta` (Lemma 1's statement).

### The polynomial commitment (`pc`)
- Vector Pedersen over BN254 G1 (reusing `lattice-pcd`'s bucket-MSM
  backend), with evaluation proofs as **linear openings**: the
  transparent long-opening regime. `Π_provePCE` batches same-point claims
  via the homomorphic combination + one combined opening; `Π_batchPCEP`
  verifies the batch. The batch proofs are **self-contained** (their η
  challenges derive from a dedicated transcript seeded by the point +
  commitments + claims) so they are portable across later verification
  contexts — exactly what the fold's eager checks need.

### The sum-check machinery (`sumcheck`)
- **Multivariate**: batched vector statements over the boolean cube with
  per-round degree `dl + dr`, round messages interpolated from `D+2`
  node evaluations of a black-box closure.
- **Univariate**: the paper's `h₁/h₂` decomposition
  `g(X) = s/n + X·h₁(X) + u_H(X)·h₂(X)` for degree-`> n` polynomials,
  with exact polynomial arithmetic (multiply / divide / mod `u_H`) and
  the degree bounds of Figures 2–5.

### §4.1 — `Π_GBF1` (`gbf1`, Figures 2–3)
- The Marlin-style reduction: the prover commits the intermediate
  vectors `d_{jM,jv} = M_{jM} v_{jv}`, ONE batched sum-check carrying
  both `qIP` (the inner product) and `qLin` (the linear-consistency
  terms `d·Λ(X,α) − λ(α)ᵀMλ(X)·v`), the Evals at β, and the
  Evals-derived decision value ζ of Fig. 3. Both ν.

### §4.2 — `Π_GBF2` (`gbf2`, Figures 4–5)
- The Spartan-style reduction: sum-check 1 over
  `q = Σ γ^k·left^k·right^k`, the α-Evals (`u_j(α)` and the claimed
  `ζ_{jM,jv}`), the η batching, sum-check 2 over
  `q' = Σ η·(λ(α)ᵀ M λ(X))·v(X)`, the β-Evals (`v_{jv}(β)` and the
  **holographic claims** `m_{jM} = λ(α)ᵀ M λ(β)`). The multivariate
  final-eval identity checks (`q_ν(α_ν) = ζ`, `q'_ν(β_ν) = ζ'`) and the
  univariate `h₁/h₂` identity checks against the Evals-derived values.
  Both ν; tamper coverage on the `m`-evals and ζ's.
- The **early-stopping variant** `Π_esGBF2` (§4.3's HyperNova-style
  linearized committed CCS) is provided as the wrapper that stops after
  the α-Evals.

### Lemma 1 / Lemma 2 / Theorem 7 (`batch`, `collapse`, `barebones`)
- `Π_batchM`: the linear-combination reduction `R_hbPCE → R_GBF,α,β`
  (the verifier's `c` challenge; the folded statement's satisfiability
  pinned by tests).
- `Π_Collapse`: `R_CCS → R_GBF,α` — the `w(X)` commitment (the padded
  domain encoding of Remark 4), the α challenge, and the
  zero-check statement `λ(α)ᵀ(Σ c ∘ M z) = 0`.
- **Barebones** = `(Π_provePCE × ID) ∘ (ID × Π_batchM) ∘ Π_GBF,α ∘
  Π_Collapse` — recovering SuperSpartan (ν = log n) and SuperMarlin
  (ν = 1) as instantiations. The output is the `R_Acc` pair
  `(R_PCEP, R_GBF,α,β)`. The public-input split (`z(β) − x(β) = w(β)`)
  is handled at the claim-adjustment site.

### Theorem 8 — `Π_Fold` (`fold`)
- The **holography accumulation**: the K incoming `R_GBF,α,β` statements
  run through the multi-instance `Π_GBF,α,β` (all-implicit component
  vectors — their own `(α⁽ᵏ⁾, β⁽ᵏ⁾)` points), which re-randomizes them
  to a single fresh point; `Π_batchM` folds the resulting matrix claims
  into one statement. Communication: O(ν) field elements per statement
  and **no cryptographic operations** — the paper's headline vs
  [BHKZ25]'s 6K(ν−1). Chained folds (depth 2) and tamper coverage in
  tests.

### §5.3 — the non-uniform decider (`decider`)
- The ℓ-function decider: `Π_GBF,α,β` over the ℓ statements on the
  **union index** (per-function matrix offsets re-based — the paper's
  "index containing all the matrices"), `Π_batchM` to the single claim
  `Σᵢ Σⱼ ηⱼ Mⱼ^{(i)}(β,α) = γ`, ONE homomorphic linear combination of
  the `ℓ·t_M` matrix commitments, and a single evaluation check (the
  long opening of the combined matrix polynomial — see the ledger).

### Corollary 3 — the PCD construction (`pcd`)
- NI-Barebones as `ARG` + NI-ΠFold as `ACC`: the chain driver where each
  node proves its compliance step (the CCS instance whose public input
  carries the incoming messages), folds the incoming accumulators, and
  passes `(proof, acc)` forward — the **stateless recursion** (each
  prover needs only the previous proofs and the public accumulator,
  never the previous witnesses). The final verifier runs the §5.3
  decider; three-step chains, per-step verification, and accumulator
  tampering rejection in tests.

## Benchmarks (release, this container)

| protocol | configuration | prove | verify |
|---|---|---|---|
| Π_GBF2 / Π_GBF1 | mv n=4 | 1.5 / 1.8 ms | — |
| Π_GBF2 / Π_GBF1 | mv n=8 | 5.9 / 7.3 ms | — |
| Π_GBF2 / Π_GBF1 | uv n=8 | 32 / 50 ms | — |
| Barebones | mv n=8 | 13.3 ms | 4.7 ms |
| Barebones | uv n=8 | 34.5 ms | 5.3 ms |
| Π_Fold (K=2) | mv n=8 | 23.5 ms | — |
| Decider (ℓ=1) | mv n=8 | 59.8 ms | — |
| PCD chain | depth 2 / 4 (mv n=8) | 49 / 140 ms | 76 / 102 ms |

## The honest-deviation ledger

1. **The PC instantiation.** Vector Pedersen with **linear (long)
   openings**: transparent, binding, O(n) proofs. The paper keeps the PC
   abstract (KZG / Pedersen-with-IPA / FRI — §1's list); the swap to a
   short-opening PC is local to `pc::open`/`verify` and the decider's
   single-evaluation call site. Consequence: the `R_PCEP` half of `R_Acc`
   is **settled eagerly** (each fold verifies the incoming batched proofs
   directly — the "atomic accumulation" flavor on that half) and pure
   `R_GBF,α,β` statements carry no PCE claims, so the accumulator's PCEP
   half stays vacuous. The paper's `Π_batchPCEP` is exactly where an
   accumulatable PC (BCMS20-style) slots in. **[DONE — the follow-up
   wave]**: `lattice-holo/pc_short.rs` implements the accumulatable
   short-opening PC on the lattice — the Accordion (ePrint 2025/1325)
   module-sumcheck over the BN254 scalar field with 16-bit digit layers:
   `O(log n)` round messages instead of the linear openings, the
   γ-accumulation folding multiple claims into one deferred instance,
   and the amortized decider (`C ≜ Ĝ(r)`, once per batch) — 4 tests
   including the tampered-value/message rejections and the
   accumulate-and-decide roundtrip. The Pedersen backend remains the
   default; the swap surface is the documented `open`/`verify` pair.
2. **Round-by-round knowledge soundness** (Definition 2's state-function
   machinery) is not formalized in code — the protocols are implemented
   with Fiat–Shamir and tested for completeness + tamper rejection; the
   extractor-side guarantees are argued, not executed.
3. **`Λ(X,Y)`'s univariate diagonal.** The closed form is 0/0 at `X = Y`;
   the implemented evaluation falls back to the basis-product sum
   `Σᵢ λᵢ(x)²` on the diagonal (the analytic continuation — equals 1 on
   `H`, the correct polynomial value off it).
4. **`λ_h`'s denominator.** The PDF extraction of §2's
   `λ_h(X) = u_H(X)/(n(X−h))` misses the `h^{n−1}` factor; the
   implemented (correct) basis is `u_H(X)/(n·h^{n−1}(X−h))`, pinned by
   the indicator and partition-of-unity tests.
5. **The decider's single evaluation check** is the long opening of the
   combined matrix polynomial (O(n²) revealed coefficients) rather than
   a short KZG/IPA proof — the same PC swap as (1). **[DONE — the
   lattice route]**: the `pc_short` decider settles the evaluation with
   the module-sumcheck's deferred instance + the direct public
   `Ĝ(r)` — the accordion note records why the paper's FRI-based
   group-BaseFold decider does not port to Ajtai bindings.
6. **`Π_GBF1`'s d-vectors** are committed per right-pair (the paper's
   Fig. 2 commits per (k, pair) in the K-instance batch); for the
   single-instance statements used here the distinction vanishes, and
   the K-instance path records per-instance evaluations.
7. **The compliance predicate** in the PCD driver is the CCS relation
   itself (each step's public input carries the incoming messages) —
   the paper's constant-depth predicate class is represented by this
   single-relation family rather than an arbitrary-circuit arithmetizer.
8. **Early stopping** (`Π_esGBF2`) is provided as a wrapper that
   delegates to the full protocol (the tail is ignored by the
   linearized-CCS consumer); a standalone truncated implementation is
   mechanical follow-up.
