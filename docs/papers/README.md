# Paper Coverage Status

Part-by-part coverage of every paper realized in this workspace. Folders:

* [`implemented/`](implemented/) — papers whose core protocol is
  realized end-to-end (prover + verifier + adversarial tests) at kernel
  scale; remaining items are efficiency/scale, not soundness.
* [`partially-implemented/`](partially-implemented/) — papers with real
  algebraic kernels but missing protocol halves; each file states
  exactly which parts are implemented, partial, or absent.
* [`unimplemented/`](unimplemented/) — designated components not yet
  started (tracked so the roadmap stays honest).

## The matrix (Wave 7 end-state + the streaming wave)

| Paper | Crate | Wave-7 items | Status | Tests |
|---|---|---|---|---|
| ProtogaLattice (2026/1317) | lattice-folding/pgl.rs | 7.1, 7.14 | **implemented** (PGL-Fold + PGL-Boot + range attach) | 11 |
| SALSAA (2025/2124) | lattice-salsa/ring_norm.rs + lattice-akita/salsa_response.rs | 7.2, 7.3 | **implemented** (D1+D2 + the D4 response-layer swap consumed by the Akita PCS + D3 Pi-batch-star row-count-preserving batching + the D6 engine constant-factor pass) | 18 |
| labinius (upstream PCS) | lattice-labinius | 7.16 | **implemented** (wire/); 7.5 Recursive partial | 9 wire |
| Cyclo (2026/359) | lattice-folding/cyclo.rs | 7.6 | partial (Π^range/Π^ext designs; kernel fold exists) | — |
| LatticeFold+ (2025/247) | lattice-folding/latticefold_plus.rs | 7.7 | partial (monomial/ψ layer open; range proof + padding fix landed) | — |
| PikkuFold (2026/1809) | lattice-folding/pikkufold.rs | 7.8 | partial (layered LRP/RingSC open; projection + gates exist) | — |
| Symphony (2025/1905) | lattice-folding/symphony.rs | 7.9 | partial (μ-ary fold kernel; tensor ring/Π_had open) | — |
| SuperNeo (2026/242) | lattice-folding/superneo.rs | 7.15 | partial (committed-instance design; fold algebra exists) | — |
| Quasar (2025/1912) | lattice-lookup | 7.10 | partial (Q1 committed lookups from Wave 6; Q2/Q3 open) | — |
| Akita (2026/1983) | lattice-akita | 7.11 | **implemented** (A1–A5: the fold core, A2 ring checks with the App F.1 ordering, A3 digit-range + norm routes, A4 tensor reduction + trace rows, A5 the terminal + the grind + Rice — **+ the commitment-scale recursion driver** (`a5_committed.rs`: the rows/A3/A4 into ONE fused sum-check against the committed successor witness, the deferred claims, the terminal discharge)) | 77 |
| HyperWolf (2025/922) | lattice-pcs | 7.12 | **implemented** (Protocols 1/2/3 + H5 certified challenges + H6 projection-compaction **+ the full-fidelity route** (`hyperwolf_labrador.rs`: the LaBRADOR engine re-parameterized to the HyperWolf ring, the amortized Dachshund over the projections with per-round exact ℓ2, the recursive outer-commitment compaction — the O(log log log N) route) + H7 batching) | 34 |
| RoKoko (2026/575) | lattice-rokoko | 7.13 | partial (projection kernels; Π^proj-c/COM open) | — |
| Twist & Shout (2025/105) | lattice-memory + zkvm | 7.3, 7.4 | **implemented** (the sparse `0s are free` engine + the v1 instruction-semantics constraint families — see `implemented/constraints-families.md`; shifts/MUL/DIV gated out) | 8 |
| **Monomial-basis sum-check (2026/762)** | lattice-projsumcheck | — | **implemented** (projective protocol + structured tables + claim-preserving batching + Fp256 upper-limb challenges + grinding) | 27 |
| **Small-space CPU proving (2025/611)** | lattice-streaming + zkvm/streaming.rs | — | **implemented** (oracles + Algorithm 1 + hybrid + prefix-suffix + grand product + matrix commitment + client facade + e2e path) | 25 |
| Accordion / IPA-sumcheck (2025/1325) | lattice-accordion | Tier-1 (the accumulatable ml-PCS) | **implemented** (module sum-check + reduce/accumulate/decide + the two-α extractor) | 23 |
| CauchyFold (2026/2011) | lattice-cauchyfold | Tier-1 (high-arity folding) | **implemented** (carrier algebra + boundary theory + the node protocol + the chain + the extraction) | 58 |
| Tensor-Train Random Projections (2026/2146) | lattice-ttrp | Tier-0 (TTRP shortness) | **implemented** (cores, both projection paths, Π_TTRP RoK + verifier tensor evaluation + bounds/search) | 15 |
| Speeding Up Sum-Check (2025/1117 + 2026/587) | lattice-sumcheck/{extrapolate,multiproduct,fastprover} | Tier-0 (prover speedups) | **implemented** (window prover + multiproduct engine + split-eq, byte-identical transcripts; Goldilocks κ≈1 documented) | 10 |
| **Ring lookups (2026/471)** | lattice-lookup-ring + zkvm/{lookup_memory,pipeline3} | — | **implemented** (Ring-Plookup + Ring-LogUp + the Section-4 attacks + the Appendix-B toolkit + the Section-6 RAM batch verification + the Greyhound-style compile onto the Ajtai/carrier stack + the Fp256 binding-pass port + the zkVM memory-argument wiring) | 62 |
| **LF+ ℓ2-norm checks (2026/721)** | lattice-folding/lfplus_l2.rs | — | **implemented** (the JL projection RoK with the concentration analysis + the exact-shortening RoK + the no-drift norm ledger) | 6 |
| **ZK-PCD from Accumulation Schemes (2026/289)** | lattice-pcd | — | **implemented** (the SPS framework with R1CS/CCS/permutation instances + the CFS17/XZZ+19 masked sum-check with the KS24 point-update + the zk-Protogalaxy accumulation + the two-circuit ZK-PCD construction over vector-Pedersen BN254) | 43 |
| **PCD via Holography Accumulation (2026/538)** | lattice-holo | — | **implemented** (the GBF relation family both ν + Π_GBF1/Π_GBF2 + Π_batchM + Π_Collapse + Barebones + Π_Fold + the non-uniform decider + the PCD chain driver) | 25 |
| **LaBRADOR (2022/1341)** | lattice-greyhound (protocol/relation/jl/r1cs/recursion) | §5.2–§5.7, §6 | **implemented** (the paper-faithful engine at q=2^32-99: Figure 2/3, LIFTS, the §5.3 target relation, the §5.6 tail, the §6 R1CS reductions, §5.4's restart remedy) | 57 |
| **Greyhound (2024/1293)** | lattice-greyhound (greyhound/batch/cwss/zk/sizes) | §2.5, §3–§5 | **implemented** (Figure 1 + the CWSS extractor of Lemma 3.2, Figure 2 batching, Figure 4 PCS with the σ^{-1} translation, Lemma 2.11 weak binding, §4.5 hiding/HVZK, Table 4 + the 53KB accounting) | (shared) |
| **LatticeBlindFold (2026/1857)** | lattice-blindfold | all of §2–§4 | **implemented** (the blinding stack for SuperNeo: Libra-style masked Sum-Check with the never-opened mask, the componentwise ABDLOP commit-and-prove over R_K with the rank-doubling ψ, the Π_many^(1)/(2)/(ct)/Π_anc PoK family with Rej1/Rej2 and the (g₀,g₁,g₂) garbage triple, Protocols 6/7/8/9/10/11/12 + Corollary 4.24's accumulator-free variant + Protocol 13's blueprint, the blinded layout with Lemma 3.3 hiding, the Table-2 parameter sets with the consolidated error budget) | 71 |

Reading order for an auditor: `docs/ARCHITECTURE.md`, then the paper
files in status order (implemented → partial), then `NEXT_STEPS.md`
(the full 831-line gap analysis with per-paper tables and the Wave 6–8
roadmap), `PERFORMANCE.md`, `SECURITY.md`, `AUDIT_CHECKLIST.md`.

## Session updates (2026-10-04, evening — the follow-ups wave)

- **the multi-stage LaBRADOR extraction** (the degree-law unwind —
  the honest residual of the staging wave): CLOSED as an executable
  ledger — `lattice-widthfold/src/extraction.rs` (laws E1–E5,
  enforced in the chain gate) + `docs/analysis/MULTISTAGE_EXTRACTION.md`
  + `--example extraction_table`. The SALSAA/Akita rows below gain
  their formal security narrative.
- **SALSAA D4 — the binding closure** (the authenticated opening at
  the challenge): `lattice-akita/src/salsa_binding.rs` — the
  byte-witness↔commitment binding via the width-collapse chain,
  composed into the v2 pipeline (`Stage5Mode::Bound`);
  `docs/analysis/D4_BINDING_CLOSURE.md` + BENCHMARKS §2l.
- **the D4 capacity split** (the r-column discipline):
  `byte_capacity` (the exact Lemma-4 cap: 2,048 values/commitment at
  dim 16/Q_32) + `prove/verify_grouped_salsa_split` (the μ-weighted
  ψ-decomposition across r columns).
- The navigation layer: `docs/PAPERS_MAP.md` (the papers' inner
  connections — the lineages and the edge-by-edge flows) and
  `docs/INDEX.md` (the concept/keyword index); the Akita row in the
  matrix above now carries the bound/split response modes.

## Session updates (2026-10-03)

- **SALSAA D3 + D6** (2025/2124 Theorem 4 + the engine pass):
  `lattice-sumcheck/batch.rs` — `prove_batch_star`/`verify_batch_star`
  (m claims, heterogeneous variable counts, ONE sumcheck with round
  count = max(num_vars); the padding lift ignores the last k bits so the
  claim scales by 2^k and the terminal never scales) and the
  `lattice-salsa/ring_norm.rs` norm-layer instantiation
  (`prove_ring_norm_batch`/`verify_ring_norm_batch` with per-instance
  Lemma-4 gates and the single F_{q^2} batch terminal);
  `lattice-sumcheck/sumcheck.rs` — the single-binding round discipline
  (t=0/1 borrow the raw halves zero-copy; one reusable bind buffer per
  factor per round) + in-place factor binding (no per-round
  fix_variables clones), byte-identical transcripts pinned by a
  reference test.
- **The T&S instruction-semantics constraint families** (T2): the ten
  legs of `constraints-families.md` — booleanity (per-row),
  selectors/decode, flags, arith limb recurrences, full-width
  comparisons, the next-pc MUX, memory routing + bitwise ALU, and
  termination — with the fail-closed v1 coverage gate and tamper
  suites. Includes the structural fixes the never-executed substrate
  needed (consistent bit-tensor MLEs, MSB-first ledger row mapping,
  the per-row booleanity construction, the stage() pairing fix) and
  the Goldilocks negative-constant correction
  (`fe(0u64.wrapping_sub(k))` is wrong — `from_u64` reduces mod p).

## Session updates (2026-10-01)

- **Fp256 window fast prover** (2025/1117 + 2026/587, the SV setting):
  `lattice-projsumcheck/fastprover.rs` — see
  `implemented/sumcheck-speedups-fp256.md` (includes the pre-existing
  composite-modulus defect found and fixed in `fp256.rs`).
- **The streaming window schedule** (`EvalProductStream_k`, 2026/587
  §5.2/C.4.2): `lattice-streaming/window_schedule.rs` — the Figure-2
  schedule with emulated prefix access, `O(M^{1/k})`-class space,
  bit-identical round messages.
- **TTRP as the zkVM ledger's norm-check module** (2026/2146):
  `lattice-zkvm/norm_check.rs` — the eval-claim API wired into the
  bundle openings, replacing the JL-projection/digit-gadget norm
  semantics.
- **Neo** (2025/294): `lattice-folding/neo.rs` — the pay-per-bit Ajtai
  commitments (popcount-scaled), `Decomp_b`/`split_b`, the strong
  sampling set with the expansion-factor bound, `Π_RLC` + `Π_DEC`, and
  the CCS decider — see `partially-implemented/neo.md`.
