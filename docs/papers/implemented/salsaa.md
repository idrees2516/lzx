# SALSAA (ePrint 2025/2124) — IMPLEMENTED (kernel scale)

Crates: `lattice-salsa` (`ring_sc.rs` + `salsaa.rs` + `air.rs` +
`ring_norm.rs` D1+D2) and `lattice-akita/salsa_response.rs` (the D4
response-layer swap).

The paper's protocol stack over the shared ring-sumcheck engine
(`ring_sc.rs`: ring-valued product-claim sumchecks with explicit
combiners, evaluation-form round messages `[g(0), …, g(deg)]`, Z_q
round challenges, the σ⁻¹ conjugation, conjugate inner products, and
balanced traces):

* **A2 — Π_norm/Π_norm+ (Fig 4)**: the O(m) direct-norm trick, the
  c-power row-batching ladder, the combined degree-2 sumcheck, the
  Π_mle eq-row append producing the reduced instance, the balanced-trace
  integer gate.
* **A3 — Π_bin (Fig 5) + the staircase RoK (Fig 6)**: binariness
  `t = ⟨w, 1° − w⟩` with the `Tr(t) = 0` gate (Lemma 4.11); Ξ^stair
  `A W_0 = Y0; B W_{j−1} + A W_j = 0; B W_{K−1} = Y1` folded into the
  single degree-3 claim with the c-power row batching and the
  geometric p/s derivation shared by prover and verifier.
* **A4 — the VDF (§6)**: the delay chain `y_{i+1} = A·G^{−1}(−y_i)`
  proved as the binary staircase `G W_0 = −y0; A W_{j−1} + G W_j = 0;
  A W_{K−1} = yT` (full-coverage gadget, L = 32 layers) composed with
  Π_bin over the flat witness (`Π_as = Π_staircase ∘ Π_bin`).
* **A5 — committed-AIR (Fig 7) + folding (§7)**: the transition /
  shift / boundary claim families in ONE combined degree-3 sumcheck
  with the (η, α, θ) public tables and the column-opening terminal;
  the Lova-style linear fold with commitment homomorphism and norm
  growth.

Tests (26 across the crate): the engine (automorphism, norm
constant-term, MLE-vs-direct, honest/tampered sumchecks, multi-claim
combiners, eq-table), A2 honest/tampered, Π_bin honest/non-binary
rejection, the staircase honest/tampered, the VDF e2e with wrong-output
and tamper rejections, the AIR e2e with dishonest-trace and tamper
rejections, and the folding completeness.
