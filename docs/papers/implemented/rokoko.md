# RoKoko (ePrint 2026/575) — IMPLEMENTED (kernel scale)

Crate: `lattice-rokoko` (`com.rs` + `protocol.rs` + the pre-existing
projection kernels). The committed-refinement core:

* **COM (Fig 1, item 3)**: the recursive Ajtai commitment — level-0
  `y = A_{n0,m}·w`, the binary gadget recursion `e = G^{−1}_l(y)`
  zero-padded to a power of two, the b0/b1/b2 verification gates
  (norm, gadget recomposition, recursive structure). Full-coverage
  gadget (l = 32 layers at Q_32) so `G·W ≡ −y mod q` exactly.
* **Ξ^lin_COM (§4.4)**: `F_i W = H_i Y_i` with COM-opened `vec(Y_i)`,
  the left/right linear claims, and the witness norm bound.
* **Π^fold-split (Fig 4, item 4)**: fold the r columns with the c
  challenges, gadget-decompose `W·c`, pack (decreasing-dimension sort +
  zero-pad, Lemma 3, prefix selectors), re-commit under a fresh vSIS
  key, send (com, v) with `v = ⟨ŵ, ŵ̄⟩` the Hermitian self-inner
  product; the verifier's `ct(v) ≤ β̃²` gate (Remark 3).
* **sumcheckify (Fig 5, item 4)**: the Lindiff/Lin/Norm constraint
  system over the packed vector — folded linear blocks as lin-diff
  claims (the row ⊗ g_{l'} selector vs the c-weighted H-row selector on
  the Y block), the commitment well-formedness rows, the exact norm
  claim.
* **Π^lin (Fig 6, item 5)**: eq(bin(i), γ) claim batching, ONE
  ring-valued sumcheck over the `lattice-salsa` `ring_sc` engine, the
  z0/z1 terminal substitution.
* **The round driver**: fold-split → sumcheckify → Π^lin → the direct
  terminal opening (binding + norm + every constraint against the
  revealed ŵ).
* **Π^proj-c / Π^proj-f (Fig 2/3, Lemmas 7/8 — `proj_f.rs` + the
  driver's inline coarse projection)**: the committed random
  projections — the coarse (ring-level) and the fine
  (coefficient-level, the trace-dual embedding) — each a self-reduction
  growing the statement; the fine variant's exact per-column trace
  identity pinned.
* **The norm schedule + the parbreak wiring (`schedule.rs` +
  `parbreak.rs`)**: the ParCom coherence, the dcmp gadget-norm map,
  Lemma 4's parbreak derivation — and the **estimator wiring**: the
  parbreak SIS set through the offline estimator (both norms, the
  cheaper attack governs), unioned with the fold-extraction instance,
  **fail-closed at driver setup on both sides** (`DriverError::Parbreak`),
  the verdict riding the proof as public metadata.
* **The statement-growth driver (`driver.rs`, §8.3)**: the full
  multi-round composition — coarse rounds while `m_w` is large
  (**Lemma 7: k_lin → k_lin + 1**), fine rounds below the switch
  (**Lemma 8: k_lin → k_lin + 2 and n → n + n_bat**, the exact
  trace-dual lift `Tr(V) = (I⊗J)·cf(W)` with `b^∨ = ±X^{n−k}/n`, the
  n_bat trace-consistency rows as the new `ScConstraint::TraceDiff`
  variant — sumcheckified + ct-gated), each round = projection →
  ℓ-block conversion (Fig 4 step 1) → fold → pack → successor (the
  column key `A_{n0,m_w'}` + the gadget product + the z0 eq-claim
  carry) → constraints → Π^lin; the direct terminal at `terminal_m`.
  The **growth ledger** records `(round, kind, k_lin, n, m_w, coms,
  β_y)` per round and the verifier replays it against Lemma 7/8's
  exact growth — any deviation fails closed.
* **The PCS front end (`pcs_front.rs`)**: the eq-weight linear form
  through the full Ξ^lin round driver.

Benchmark (`cargo run --release -p lattice-rokoko --example
driver_bench`): m_w = 512, r = 2, 2 rounds — prove 4 620 ms / verify
484 ms; the ledger: coarse `m_w 512→128, k_lin 1→2`; fine
`m_w 128→32, k_lin 1→3, n 0→2`; the parbreak verdict 11.7 bits at the
toy ring (fail-closed at any higher target).

Tests (37): COM depth-1/2 roundtrips + tamper, the gadget roundtrip,
the packing structure, the single-round e2e + 3-way tamper, the
projection kernels' identity pins, the norm-schedule coherence, **the
trace-dual duality, the fine-projection trace identity + ct(r_i) = 0,
the multi-round driver e2e (coarse +1 then fine +2/+n_bat with the
ledger replay), tampered r-rows / tampered ledger / tampered terminal
rejections, the parbreak gate fail-closed, and the parbreak
shape/union tables**.

Documented deviations: full-ring challenges (the paper's Φ_δ subfield
batching is a size optimisation; the driver's fold challenge is
ternary — the paper's small class C, needed for the ℓ-digit
compression); the r_i rows are realized as the TraceDiff constraint
variant (sumcheckified with Lindiff-shaped groups + the ct gate)
rather than the paper's `A' = diag(A, ·)` global-linear encoding —
knowledge error per Lemma 8's `κ`; the norm schedule is
saturating-u64 heuristics in the driver path; COM depth 1 in the live
path; the kernel story (incomplete NTT at q ≈ 2^50, Karatsuba 5→4,
AVX-512) remains unported; the signed balanced gadget `g_inv_signed`
replaces the unsigned form wherever balanced values appear.
