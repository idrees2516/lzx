# Symphony (ePrint 2025/1905) — PARTIAL

Crate: `lattice-folding/src/symphony.rs`. Implemented: μ-ary degree-2
fold with explicit pairwise cross terms E_ij (exact identity tested at
μ = 3, 4) under the stacked cross commitment; LaBRADOR SmallSet
challenges {0,±1,±2} (Wave 6).

Open (item 7.9): the tensor-ring substrate (K = F_{q²}, TensorElement
dual views, ts(r)), Π_had (Fig 1 degree-3 sumcheck with α-power column
batching — the module's first real verifier), the O(μ) shared-randomness
fold (Fig 4: shared (J, s′, α), merge 2μ sumchecks into two via
α-power RLC Eq 45, β ← S^μ folding Eqs 48–49 — replacing the O(μ²)
enumeration that is exactly the paper's §1.2 strawman), Eq-50
feasibility, Π_rg, the CP-SNARK compiler (Constr 6.1), two-layer
folding (§8), the memory strategy (Remark 4.1).
