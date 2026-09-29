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

Tests (9): COM depth-1/depth-2 roundtrips with tamper and
inflated-witness rejections, the gadget roundtrip, the packing
order/prefix structure, and the full e2e argument with tampered-com /
tampered-witness / tampered-sumcheck rejections.

Documented deviations: full-ring challenges (the paper's Φ_δ subfield
batching is a size optimisation), Π^proj-f (the fine projection) is
not implemented (a norm-slack optimisation, not needed for
completeness), the driver runs at COM depth 1 (the recursion is
unit-tested in `com.rs`), and the kernel story (incomplete NTT at
q ≈ 2^50, Karatsuba 5→4, AVX-512) remains unported.
