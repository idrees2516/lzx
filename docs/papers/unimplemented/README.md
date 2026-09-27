# Unimplemented designated components

Components designed, scoped, and tracked but not yet started. Each is a
Wave 8 line item (see NEXT_STEPS.md §4 for the dependency graph and
calibration targets).

| Component | Wave | What it is |
|---|---|---|
| gen_* transforms + vertical AuxData | 8.1 | labinius fold/verify kernel families (2,642 upstream LOC; round 222→~100 ms, removes the 170 MB scatter) |
| Goldilocks AVX-512 kernels | 8.2 | IFMA/VPCLMULQDQ kernels for lattice-core/sumcheck/akita/zkvm field-bound stages (4–8x) |
| lattice-rns 8-prime crate | 8.3 | RNS arithmetic for LaBRADOR polx (10–50x prove) + RoKoko CRT commitment |
| bin_large + block-sink fusion + A-prefetch + components_of SIMD | 8.4 | labinius encode kernels (910 upstream LOC) |
| Sumcheck constant-factor pass | 8.5 | single-binding rounds, in-place fix_variables, Gruen/Dao-Thaler |
| ZK wiring into the zkVM | 8.6 | per-stage blinding (Blindfold pattern), sparse masking, composed simulator KATs |
| QROM composition attestations + grinding + CT/timeout harness | 8.7 | regenerate from the real stage list; machine-checked grinding bounds |
| Production parameter instantiation | 8.8 | estimator-gated SecurityProfiles (digest-pinned) |
| Chunked-chain recursion encoding | 8.9 | labinius constraint count ↓ (shared-φ chunked chains with carry gadgets) |
| Akita planner/validator; HyperWolf compaction; Symphony CP-SNARK; Quasar IVC | 8.10 | the four "L each" production modules |
