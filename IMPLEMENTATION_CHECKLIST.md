# Implementation Checklist

**The per-component status grid.** Complements `IMPLEMENTATION_LOG.md`
(when things landed), `docs/papers/README.md` (per-paper part-by-part
coverage), `NEXT_STEPS.md` (the research backlog + honest ledger), and
`AUDIT_CHECKLIST.md` (the G1–G8 audit gates).

Legend: [x] landed + test-pinned · [~] partial (the gap stated) ·
[ ] open (the route documented).

---

## The core engine

- [x] `lattice-core` — Goldilocks field, Keccak/SHA3/SHAKE, the FS
      transcript, dense MLEs, gadget decomposition, challenge sets,
      short ring challenges with certified operator norms, the
      NormBudget hard gates, F_{q²}
- [x] `lattice-ring` — negacyclic NTT (CT/GS), R_q arithmetic,
      incomplete NTT + completion, split/CRT packing, the extension
      ring, the Modulus-50 quadratic-slot NTT
- [x] `lattice-commitment` — Ajtai commitments (seed-derived keys,
      cached-NTT fast path, zero-skipping MAC), ABDLOP-style linear
      proofs, digit-decomposed norm proofs, bit-packed one-hot columns
- [x] `lattice-sumcheck` — the virtual-polynomial engine, zerocheck,
      batching, the window/multiproduct fast provers, Fp256 variants
- [x] `lattice-sis-estimator` — the offline ADPS16/BDGL16/LGSA
      estimator (infinity + Euclidean paths)
- [x] `lattice-bench` — the reproducible benchmark matrix

## The papers (17 + 2 ports)

- [x] ProtogaLattice — PGL-Fold + PGL-Boot + range attach
- [x] SALSAA — D1 (norm + Lemma-4 gate), D2, D3 (Π_batch-star), D4
      (the response-layer swap), D6, the A2–A5 stack (Π_norm+, Π_bin,
      the staircase, the VDF, committed-AIR + folding)
- [x] Akita — the PCS (packed commitments, evaluation + grouped
      openings, the schedule catalog); **+ D4-bound** (this wave: the
      binding closure + the r-column split)
- [x] Cyclo — the extension commitment, partial range, the
      accumulator, **the §7 R1CS bridge + the compact-PCS terminal**
      (the witness-free decider)
- [x] HyperWolf — Protocols 1/2/3 in full (the standard-soundness PCS
      backend)
- [x] LatticeFold+ — range proof, double-commitment folding, tensor
      rings, the ℓ2-norm checks (2026/721)
- [x] PikkuFold — the layered LRP + certified-JL + Π_fold
- [x] Quasar — the committed lookup protocol + Q2/Q3 accumulation +
      the IVC loop
- [x] RoKoko — the recursive COM, Π_fold-split, sumcheckify, Π_lin,
      the round driver
- [x] Symphony — the μ-ary fold + Π_had + the tensor substrate
- [x] Twist & Shout — the PIOPs, the sparse engine, the constraint
      families (T2: ten legs)
- [x] ZK-PCD (2026/289) — the SPS framework + zk-Protogalaxy + the
      two-circuit construction
- [x] Holography PCD (2026/538) — the GBF family + Π_Fold + the
      non-uniform decider + the PQ follow-ups
- [x] Accordion (2025/1325) — the module sum-check + the accumulatable
      ml-PCS + the two-α extractor
- [x] CauchyFold (2026/2011) — the carrier algebra + the node
      protocol + the §6 extraction
- [x] LatticeBlindFold (2026/1857) — the blinding stack complete
- [x] TTRP (2026/2146) — cores, both projection paths, Π_TTRP
- [x] Monomial sum-check (2026/762) + small-space (2025/611) —
      `lattice-projsumcheck`, `lattice-streaming` (streaming +
      client-side, byte-identical)
- [x] Ring lookups (2026/471) — `lattice-lookup-ring` + the v3
      pipeline wiring
- [x] labinius port — the full PCS + the AVX-512 parity kernels + the
      Recursive mode
- [x] LaBRADOR — the native port (`lattice-labrador`) AND the
      paper-faithful engine + Greyhound (`lattice-greyhound`) AND the
      **H6 re-parameterization to the HyperWolf ring**
      (`lattice-pcs/src/hyperwolf_labrador.rs`): the amortized
      Dachshund at `q = 2^61 − 259` with per-round exact ℓ2 statements,
      the fold-consistency dot-products natively in-ring, the recursive
      outer-commitment compaction driver (the O(log log log N) route),
      and the full-fidelity `eval_prove_labrador`/`eval_verify_labrador`
      protocol — the faithful-gate pin + the KernelBypass demonstrator
      mode, the wraparound guard fail-closed
- [x] **the A5 commitment-scale recursion driver**
      (`lattice-akita/src/a5_committed.rs`): the Eq 5/6/7/8/2 row set
      over the committed successor witness, ONE A2 fused sum-check over
      the flat Goldilocks coordinates, the A3 digit-range deferred
      claims as eq-anchored rows (`verify_digit_range_deferred`), the
      A4 evaluation-trace rows (the Eq-135 c0/c1 coordinates), the
      App F.1 ordering, the per-level deferred Goldilocks claim, the
      terminal discharge — the split-field residual honestly recorded

## The zkVM stack

- [x] `lattice-vm` — the canonical RV64IMAC decoder + the traced
      executor + the differential conformance corpus
- [x] `lattice-memory` — Twist/Shout + the sparse engine
- [x] `lattice-zkvm` — v1 (envelope) · v2 (Twist & Shout live, the
      verifier never re-executes, Stage-5 D4 swap, **+ the Bound mode —
      the compact-fold composition, this wave**) · v3 (ring-lookup
      memory arguments) · streaming · leg batching
- [x] `lattice-widthfold` — the width fold (W0)–(W4), the recursive
      chain, the ring-functional fold (the Cyclo terminal layer)
- [x] `lattice-zk` / `lattice-qrom` — the ZK capability layer + the
      QROM accountability

## The soundness/analysis layer (this wave's focus)

- [x] **the multi-stage LaBRADOR extraction ledger**
      (`lattice-widthfold/src/extraction.rs`): the degree-law unwind
      (laws E1–E5), machine-checked, ENFORCED in `assert_sound_chain`
      at prove AND verify time; the tables published via
      `--example extraction_table`; the formal write-up
      `docs/analysis/MULTISTAGE_EXTRACTION.md`
- [x] **the D4 binding closure** (`lattice-akita/src/salsa_binding.rs`):
      the byte-witness↔commitment authenticated opening (the
      compact-fold composition), wrong-commitment tamper-pinned;
      `docs/analysis/D4_BINDING_CLOSURE.md`
- [x] **the r-column capacity split** (same module): `byte_capacity`
      (the exact Lemma-4 cap: 2,048 values at dim 16/Q_32) +
      `prove/verify_grouped_salsa_split` (μ-weighted ψ-decomposition,
      per-column D1 + chains)
- [x] the extraction-posture enforcement wired into the chain gate
- [x] debug-profile overflow fixes (the suite passes with overflow
      checks ON in both profiles)
- [x] CI workflow (`.github/workflows/ci.yml`): test + clippy + the
      evidence examples

## Production hygiene

- [x] README / ARCHITECTURE / PAPERS_MAP / INDEX / IMPLEMENTATION_LOG /
      IMPLEMENTATION_CHECKLIST / NEXT_STEPS / SECURITY /
      AUDIT_CHECKLIST / PERFORMANCE / BENCHMARKS maintained
- [x] 1,237 tests green (debug + release profiles); clippy clean on
      every touched crate
- [~] clippy `-D warnings` clean workspace-WIDE on all targets
      (pre-existing test-target lint debt in `lattice-folding` —
      tracked, untouched)
- [ ] GitHub topics/description set on the remote (this push)
- [ ] release tagging + CHANGELOG automation
- [ ] external audit (see AUDIT_CHECKLIST's external items)

## The open engineering ledger (top of NEXT_STEPS)

- [ ] the Modulus-50 class as the widthfold operating modulus (the
      norm-law headroom beyond the Q_32 ceiling)
- [ ] folding the D4 split's r column-chains into ONE accumulator
      (the Quasar/PCD route — the response-size recovery)
- [ ] the single-fold-over-columns composition (the LaBinius secondary
      discipline) for the split's size
- [ ] zero-knowledge for the fold/response layers (Wave 8.6,
      LatticeBlindFold's designated route)
- [ ] the CLOB guest port (Guests A/B/C of
      `docs/CLOB_WORKLOAD_RESEARCH.md`) + the ethrex/zoda adapter code
      (`docs/ETHREX_ZODA_INTEGRATION.md`'s design)

## Wave 7 completion (2026-10-05)

- [x] Akita A3 — the digit-range sumcheck (degree halving Eq 114, the
      product-tree shapes, the leaf collapse, the fused binariness) +
      the response-norm certification (direct Eq 118-120 + digit-expanded
      Eq 121-123, both routes fail-closed) — `a3_range.rs`
- [x] Akita A4 — the Diamond-Posen tensor reduction (the tensor step,
      the E-valued sumcheck, the transparent factor via the conjugate
      formula, the Theorem-3.11 batch) + the evaluation-trace row over
      F_{Q32²} (ψ, σ⁻¹, the pinned packing identity) — `a4_tensor.rs`
- [x] Akita A5 — the recursion driver + the §8.2 terminal (the grind
      discipline, Eq 163-165, the signed Rice budget) — `a5_terminal.rs`
- [x] HyperWolf H6 — the projection-compaction layer (commitment
      linearity + the terminal reveal; the re-parameterized-LaBRADOR
      route recorded as the follow-up) — `hyperwolf_compact.rs`
- [x] HyperWolf H7 — the three Appendix-B batching modes —
      `hyperwolf_batch.rs`
- [x] RoKoko 6 — Π^proj-f (the coefficient-level projection, the
      trace-dual identity, the batched traces) — `proj_f.rs`
- [x] RoKoko 7 — the norm schedule + the SIS parameter algebra
      (dcmp, parbreak, the κ composition, fail-closed) — `schedule.rs`
- [x] RoKoko 8 — the PCS front end through the Ξ^lin stack —
      `pcs_front.rs`
