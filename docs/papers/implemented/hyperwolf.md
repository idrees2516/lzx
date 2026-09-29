# HyperWolf (ePrint 2025/922) — IMPLEMENTED (kernel scale)

Crate: `lattice-pcs` (`hyperwolf.rs`, ~1,470 lines + the pre-existing
`PcsBackend` transparent backend). Full transcription of the paper's
Protocols 1/2/3:

* **PC.Commit (Protocol 2)**: ring-pack `M_R` → balanced gadget
  decomposition (carry-rebalanced signed digits, the +1 headroom digit)
  → k-dimensional hypercube `s^(k)` (axes b × … × b × b·ι) →
  block-tiled Ajtai slice commitments `A^(k) = 1^T ⊗ A` → the outer
  `B^(k) = 1^T ⊗ B` binding of the digit-decomposed stack.
* **Protocol 1 (guarded recursive evaluation)**: k−1 rounds — per round
  the fold `R_q^b`, the JL projections (σ⁻¹-conjugated seeded trit
  matrix, block-sum tiled), and the slice commitments; the verifier's
  four checks (evaluation identity, the 128-style JL norm bound at
  `jl_rows/2·β²`, the outer-commitment binding in the statement-chain
  form, the cross-round projection consistency) + the final pinning
  (`⟨conj(a0_ext), s^(1)⟩ = y`, `‖s^(1)‖ ≤ β^(0)`, the JL terminal,
  `A·s^(1) = Σ C_i t_i`).
* **PC.Eval (Protocol 3)**: univariate and multilinear a-vector
  builders with the integer-stride axis semantics, the gadget-transpose
  a0 expansion, and the direct-evaluation reference.
* **The ring layer (H2, partial)**: a self-contained u64 negacyclic
  ring at `q = 2^61 − 259 ≡ 5 (mod 8)` (Lemma-1 invertibility; the
  labrador precedent) — schoolbook multiply over i128 accumulators.
  The paper's q ≈ 2^128 regime still needs the RNS layer (Wave 8.3).
* **The challenge space (H5)**: certified fixed-weight signed
  challenges via `lattice_core::short_challenge::hyperwolf_spec`
  (weight 10, Γ_C = 4 per sample) replacing the Labrador SVD-rejection
  sampler — the NEXT_STEPS H5 design; the norm ladder keeps the
  paper's conservative T = 15 growth (√30 ≥ Γ_C).
* `paper_params()` + the Table-2 proof-size model (test-pinned; the
  uncompacted-vs-LaBRADOR-compacted delta documented).

Tests (13): balanced-digit roundtrip incl. the negative-value
carry-rebalance, gadget roundtrip, a-vector builders, hypercube
slice/fold, certified sampler, e2e k = 2..4 (univariate) with
wrong-y / tampered-fold / tampered-final / tampered-commitment
rejections, the multilinear path with the W2 fold == direct-evaluation
identity, the proof-size model, and the PC.Open roundtrip.

Remaining (efficiency/scale, not soundness): H6 LaBRADOR compaction
(out of scope until the RNS ring), H7 three-mode batching (trait
extension), H8 `impl PcsBackend for HyperWolfFull` unification, the
paper-scale q ≈ 2^128 parameters.
