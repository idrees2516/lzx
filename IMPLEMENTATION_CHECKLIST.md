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
      paper-faithful engine + Greyhound (`lattice-greyhound`)

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
