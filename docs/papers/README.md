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

Reading order for an auditor: `docs/ARCHITECTURE.md`, then the paper
files in status order (implemented → partial), then `NEXT_STEPS.md`
(the full 831-line gap analysis with per-paper tables and the Wave 6–8
roadmap), `PERFORMANCE.md`, `SECURITY.md`, `AUDIT_CHECKLIST.md`.
