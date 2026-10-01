# Neo (2025/294, Nguyen–Setty) — IMPLEMENTED (the core mechanisms)

Status: **IMPLEMENTED** (`crates/lattice-folding/src/neo.rs`, ~1050 lines
incl. tests) — the Neo half complementing the workspace's SuperNeo
(2026/242) in `superneo_committed.rs`.

## What is implemented

* **The small-field ring** (`NeoRing`): `R = F_p[X]/(X^d + 1)` directly
  over **Goldilocks** — the paper's headline ("Neo can use small prime
  fields"): no 32-bit bridge, no NTT representation, no ring switching.
  The negacyclic product *is* the rotation-matrix multiply
  `rot(a)·cf(b)` (§3.2's identity), so the Ajtai map is an
  **S-module homomorphism** (Theorem 2). The multiplicative inverse
  runs by exact Gaussian elimination on the rotation matrix (d ≤ 16).
* **`Decomp_b` / `split_b`** (Definition 11): the b-ary digit-matrix
  embedding `z ∈ F^m → Z ∈ F^{d×m}` (one ring element per column) with
  the recomposition identity `z = Σ b^{i−1}·Z^{(i)}` — the norm
  management that keeps folded openings committable.
* **Pay-per-bit Ajtai commitments** (§3.2): `c = M·cf^{-1}(Z)` via the
  rotation-column identity `cf(a·b) = Σ_l b_l·a_l` — **zero digit
  coefficients add nothing**, so the commitment work scales with the
  **popcount** of the digit columns. The measured cost profile
  (`pay_per_bit_cost_profile`): a binary witness pays ≥ 4× less than a
  full-width one and both agree bit-exactly with the naive `d²`-product
  reference on values.
* **The strong sampling set** (§3.4, Theorem 3): ternary fold
  challenges with pairwise-invertible differences (enforced by
  inversion), and the **expansion factor** measured against random
  probes — within the paper's bound `T ≤ 2·φ(η)·max‖ρ‖∞ = 2d` for
  ternary elements (`strong_sampling_within_bound`).
* **`Π_RLC`** (§4.5): the fold with **rotation-matrix challenges**
  `ρ ∈ C` — the verifier-side `c = Σ ρ_i·c_i`, `y_j = Σ ρ_i·y^{(i)}_j`,
  `x/u` scaled by the challenges' constant terms; the prover-side
  `Z = Σ ρ_i·Z_i`.
* **`Π_DEC`** (§4.6): the norm-reducing split — `split_b(Z)` into k
  low-norm parts with fresh pay-per-bit commitments and the
  verifier-side recomposition identities `c = Σ b^{i−1}·c_i`,
  `y_j = Σ b^{i−1}·y^{(i,j)}` (checked exactly).
* **The decider**: opening (`M·z' = c` exact), the fail-closed digit
  norm bound, the recomposition consistency, and relaxed CCS
  satisfaction through the shared `lattice_relations::ccs` engine.

## The honest deviation ledger

1. **`Π_CCS`'s in-sumcheck norm products**: the paper's `NC_i(X) =
   Π_{j=−b+1}^{b−1}(Ẑ_i(X) − j)` folds the range check into the one
   big HyperNova-style sum-check over `K = F_{p²}`. This
   implementation enforces the same invariant at the decider (the
   reconstructed digits are range-checked fail-closed) plus the
   DEC/RLC norm bookkeeping — the security-relevant core; the
   amortization (and the `Eval_{i,j}` rerandomization machinery) is
   the follow-up.
2. **The partial-evaluation claims**: the paper's `y_j = Z·M_j^T·r̄`
   are established by `Π_CCS`'s sum-check. The driver uses
   **linear claim functionals** `L_j(columns) = Σ w_{j,j'}·col_{j'}`
   with public hash-derived small weights — linear in exactly the way
   RLC and DEC consume (the identities hold bit-exactly), standing in
   for the sum-check-established evaluations. The `Fq2` extension
   (available and tested) is the challenge field for the future
   sum-check layer.
3. **Pair folds**: the paper's multi-folding (β instances at once)
   amortizes the decomposition overhead; this implementation folds
   pairs — identical mechanism, the amortization is follow-up.
4. The `y`-claim consistency across `Π_DEC` is enforced by
   construction (linearity), matching the paper's verifier checks.

## Tests (11)

`ring_basics` (product/inverse/commutativity/distributivity),
`decomp_roundtrip`, `pay_per_bit_cost_profile` (the headline
measurement), `strong_sampling_within_bound`, `neo_fold_roundtrip`
(commit → RLC → DEC → decide), `tampered_witness_rejected`,
`tampered_commitment_rejected`, `fq2_challenges_available`, plus the
shared-CCS tests.

## Relation to the SuperNeo half

`superneo_committed.rs` (2026/242) implements the successor design:
the `F_{2^16}` packing bridge, the `fold_public`/`fold_secret` split
with the Nova-style cross term, the committed-instance Fiat–Shamir
discipline, and the `π_CCS` decider. Neo's distinct mechanisms —
native Goldilocks ring arithmetic, the `Decomp_b`/`split_b` norm
pipeline, the rotation-matrix challenge set with its expansion-factor
discipline, and the popcount-scaled commitment path — are the ones
implemented here; together the two modules cover both papers' cores.
