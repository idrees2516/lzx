# Π_CCS — the in-sumcheck norm products (Neo 2025/294 §7 + SuperNeo 2026/242 §7.3)

**Status: implemented** (`lattice-folding/src/pi_ccs.rs`, ~950 lines, 10 tests)

The succinct CCS decider for the committed-fold pipeline. Closes the honest
gap in `superneo_committed::decider_committed`, which *transmitted the
witness* (the Ajtai opening carried the full packed vector).

## The reduction

One sum-check over the cube `{0,1}^{log L} × {0,1}^{log m}` (level
variables first) with four term families:

| Term | Checks | Degree |
|------|--------|--------|
| `NC` | the norm products `Π_{a=−b+1}^{b−1}(Z̃(X) − a)` — vanishes at a cube point iff the digit lies in range | `2b−1` |
| `F` | relaxed CCS satisfaction per row, wrapped in `eq(X_pos, α)` — the eq-weighted Boolean sum of a cube-vanishing polynomial is 0 | 2 |
| `EvalK` | the prior witness claims `z̃⁽ℓ⁾(r_pos)`, re-randomized via `eq(X, r)` | 2 |
| `EvalA` | the prior per-level matrix-product claims | 2 |

The claimed sum `T` is verifier-computable from the prior claims alone
(F and NC vanish on honest witnesses) — a cheating prover with an
out-of-range digit or an unsatisfied row cannot complete the first round
identity. The γ-power offsets are disjoint across the families.

## The digit structure

`Decomp_b`: the witness decomposes base-`b` into `L` level vectors (the
pay-per-bit path at `b = 2`). Matrix products are computed **per level**
and recombined linearly; the verifier's recombination
`m_j := Σ_ℓ b^ℓ·y_{j,ℓ}` is Neo's step 4 exactly. Per-level digit
commitments with the FREE homomorphic binding `C_values = Σ b^ℓ·C_ℓ`
(the coefficient-packing linearity — no extra proof).

## The eval-claim API

`CeClaim { r, y[ℓ], y_products[j][ℓ] }` + `fold_claim` (the linear
claim fold at a shared point). Π_CCS's output is a fresh claim set at
the terminal point — consumed by the folding loop and the
commitment-opening layer.

## Honest deviations

* `K = A = Goldilocks` — no extension field for the challenges (the
  `fq2_sumcheck` lift is the upgrade path to 128-bit).
* The `Trans/Emb` ring machinery (their Theorems 9–11) is realized
  through the linear packing bridge — the flatten-MLE is linear in the
  digit values, so the evaluation homomorphism reduces to
  coefficient-wise linearity.
* The running claim's norm growth across folds (the paper's Π_DEC norm
  chain) is out of scope: the decider operates on a *fresh*
  decomposition of the folded witness. Folding digit vectors directly
  and re-decomposing is the follow-up (the SALSAA norm-chain substrate).
* The terminal claims open via the direct Ajtai opening in
  `decider_pi_ccs`; the compact-mode linear-functional bridge
  (`lattice-zkvm::ttrp`) is the succinct route.
