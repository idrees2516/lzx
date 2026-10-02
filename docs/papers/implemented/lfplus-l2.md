# Improving LatticeFold+ with ℓ2-Norm Checks (ePrint 2026/721)

**Paper**: Osadnik — *Improving LatticeFold+ with ℓ2-norm checks*
(workshop note, 2026).
**Module**: `lattice-folding/src/lfplus_l2.rs`.
**Status**: the norm-control layer implemented — both RoKs, the JL
projection with its concentration analysis, the exact-shortening
identity, the norm ledger, and the composed prove/verify driver.

## What the paper is

LatticeFold+ (CRYPTO '25) controls witness-norm growth through ℓ∞
monomial range-checks — the dominant prover cost
(`L·n·m·N·log_N β` ring additions). This note replaces the
norm-control layer with an ℓ2 pipeline built from two reductions of
knowledge:

1. **`Π proj-KLNO25`** (the Rok-and-Roll spirit): block-project each
   witness with a shared Johnson–Lindenstrauss matrix
   `Π ← C^{256×b}` (`Pr[0]=1/2`, `Pr[±1]=1/4`), commit the projected
   image as its own **unfolded** instance (the extraction handle stays
   slack-free), and carry the linear-consistency claims through the
   challenge split `c = c₀ ⊗ c₁`:
   `c₁ᵀ(I_{m/b} ⊗ Π)w_j = t_j`, `(c₀ ⊗ c₁)ᵀv = s`, `Σ_j c_{0,j} t_j = s`.
2. **`Π exact-KLOT25`** (the SALSAA spirit): the exact shortening
   `ct(⟨w̄, w⟩) ≤ β²` — the squared-ℓ2 encoded in the ring inner
   product (for the `R_q = Z_q` instantiation the conjugation is the
   identity; for `N > 1` it is the automorphism `X ↦ −X^{N−1}`),
   with the split product `u'_j·u''_j = u*_j` over evaluation claims
   at a random point.

Composed with homogenization, random combination, and decomposition,
the witness norm trajectory closes: `β → β' → β'' → β''' ≤ β` —
iterative folding without norm drift, with the dominant prover term
dropping to `L·m·256` additions.

## The implementation

* **`JlMatrix`** — the Lemma-1 distribution from a seed (rejection-free
  two-bit encoding), block projection `v^{(i)} = Π·w^{(i)}` over
  `b = 256·L` blocks, and the norm-concentration analysis: with
  `E[Π²] = 1/2` and 256 rows, `E[‖Πw‖²] = 128‖w‖²` — the ratio
  concentrates around `√128 ≈ 11.31` independent of the block width,
  which is the certification direction (small `‖Πw‖` ⇒ small `‖w‖`
  through the lower tail). The tests pin the `[8, 16]` band at the
  demonstration scale; the paper's printed `(30, 337)` constants are
  its concrete-regime instantiation of the same concentration
  statement.
* **`conjugate`** — the cyclotomic involution `φ(X^k) = X^{−k}`
  (constants fixed, `X ↦ −X^{N−1}`), an involution by construction and
  tested as such.
* **`prove_l2_norm_check` / `verify_l2_norm_check`** — the composed
  RoKs over `L` Ajtai-committed witnesses:
  * the fail-closed norm gates (`‖w_j‖₂ ≤ β`, `ct(u_j) ≤ β²`);
  * the projection layer: per-instance images, the stacked **unfolded**
    image commitment, the `c₀ ⊗ c₁` challenge split with the `t_j`
    and `s` claims, and the `Σ c_{0,j}·t_j = s` check;
  * the exact-shortening layer: `u_j = ⟨w_j, w_j⟩` (integer dot under
    the canonical lift), the random evaluation point `r*`, the split
    claims `u'_j = MLE[w_j](r*)` and `u''_j = MLE[w̄_j](r*)`;
  * the openings: the evaluation claims settle through the workspace's
    `lattice-commitment::linear_proof` with the EQ-tensor coefficient
    vectors; the `t_j` claims settle against the recomputed
    `c₁ᵀ(I_{m/b} ⊗ Π)` form vector.
* **`norm_ledger`** — the `β → β' → β'' → β''' ≤ β` trajectory with the
  decomposition depth that closes the loop; tested across
  `L ∈ {2, 4, 8, 16}` (no drift at any width).

## Deviations (honest list)

1. The linear-consistency and evaluation claims settle through Ajtai
   linear openings rather than the paper's sum-check compression — the
   same linear-form shape; the `fq2_sumcheck` substrate of `lfplus_mon`
   carries that compression pattern. The norm-control layer — the
   paper's contribution — is complete.
2. The paper's `c = c₀ ⊗ c₁` challenges are sampled as short
   `{1, 2}` constants at demonstration scale (the tensor structure is
   preserved; full-width challenges are the production
   instantiation).
3. The Lemma-1 constants: see the concentration note above — the
   module pins the mathematically-expected band and documents the
   paper's printed constants as its regime's tail analysis.

## Tests

`jl_matrix_distribution` (the {1/2, 1/4, 1/4} law),
`jl_norm_preservation_concentration` (24 trials in the [8, 16] band),
`conjugate_automorphism_semantics` (constants fixed, `X ↦ −X^{N−1}`,
involution), `l2_norm_check_end_to_end` (prove/verify + tampered
commitment rejection + the oversized-witness gate),
`norm_ledger_no_drift`, `projection_compression_shape`.
