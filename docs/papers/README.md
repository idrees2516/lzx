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
| SALSAA (2025/2124) | lattice-salsa/ring_norm.rs + lattice-akita/salsa_response.rs | 7.2, 7.3 | **implemented** (D1+D2 + the D4 response-layer swap consumed by the Akita PCS) | 12 |
| labinius (upstream PCS) | lattice-labinius | 7.16 | **implemented** (wire/); 7.5 Recursive partial | 9 wire |
| Cyclo (2026/359) | lattice-folding/cyclo.rs | 7.6 | partial (Π^range/Π^ext designs; kernel fold exists) | — |
| LatticeFold+ (2025/247) | lattice-folding/latticefold_plus.rs | 7.7 | partial (monomial/ψ layer open; range proof + padding fix landed) | — |
| PikkuFold (2026/1809) | lattice-folding/pikkufold.rs | 7.8 | partial (layered LRP/RingSC open; projection + gates exist) | — |
| Symphony (2025/1905) | lattice-folding/symphony.rs | 7.9 | partial (μ-ary fold kernel; tensor ring/Π_had open) | — |
| SuperNeo (2026/242) | lattice-folding/superneo.rs | 7.15 | partial (committed-instance design; fold algebra exists) | — |
| Quasar (2025/1912) | lattice-lookup | 7.10 | partial (Q1 committed lookups from Wave 6; Q2/Q3 open) | — |
| Akita (2026/1983) | lattice-akita | 7.11 | partial (packed commit + sumcheck eval; A1–A5 open) | — |
| HyperWolf (2025/1903) | lattice-pcs | 7.12 | partial (PcsBackend + transparent mode; guarded IPA open) | — |
| RoKoko (2026/575) | lattice-rokoko | 7.13 | partial (projection kernels; Π^proj-c/COM open) | — |
| Twist & Shout (2025/105) | lattice-memory + zkvm | 7.3, 7.4 | partial/unimplemented (oracles + envelope strong; PIOPs + response-layer swap open) | — |
| **Monomial-basis sum-check (2026/762)** | lattice-projsumcheck | — | **implemented** (projective protocol + structured tables + claim-preserving batching + Fp256 upper-limb challenges + grinding) | 27 |
| **Small-space CPU proving (2025/611)** | lattice-streaming + zkvm/streaming.rs | — | **implemented** (oracles + Algorithm 1 + hybrid + prefix-suffix + grand product + matrix commitment + client facade + e2e path) | 25 |
| Tensor-Train Random Projections (2026/2146) | lattice-ttrp | Tier-0 (TTRP shortness) | **implemented** (cores, both projection paths, Π_TTRP RoK + verifier tensor evaluation + bounds/search) | 15 |
| Speeding Up Sum-Check (2025/1117 + 2026/587) | lattice-sumcheck/{extrapolate,multiproduct,fastprover} | Tier-0 (prover speedups) | **implemented** (window prover + multiproduct engine + split-eq, byte-identical transcripts; Goldilocks κ≈1 documented) | 10 |
| **Ring lookups (2026/471)** | lattice-lookup-ring + zkvm/{lookup_memory,pipeline3} | — | **implemented** (Ring-Plookup + Ring-LogUp + the Section-4 attacks + the Appendix-B toolkit + the Section-6 RAM batch verification + the Greyhound-style compile onto the Ajtai/carrier stack + the Fp256 binding-pass port + the zkVM memory-argument wiring) | 62 |
| **LF+ ℓ2-norm checks (2026/721)** | lattice-folding/lfplus_l2.rs | — | **implemented** (the JL projection RoK with the concentration analysis + the exact-shortening RoK + the no-drift norm ledger) | 6 |

Reading order for an auditor: `docs/ARCHITECTURE.md`, then the paper
files in status order (implemented → partial), then `NEXT_STEPS.md`
(the full 831-line gap analysis with per-paper tables and the Wave 6–8
roadmap), `PERFORMANCE.md`, `SECURITY.md`, `AUDIT_CHECKLIST.md`.

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
