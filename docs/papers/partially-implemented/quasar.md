# Quasar (ePrint 2025/1912) — PARTIAL

Crate: `lattice-lookup/src/lib.rs`. Implemented (Wave 6.5, "Q1"):
committed T/R/Q with τ derived from commitments, counting-map
multiset difference, forged-triple negative test — the soundness hole
(`verify_lookup` binding nothing) is closed. Plus the deterministic
accumulation transform.

Landed (Wave 7.10, `quasar_acc.rs` over the R_q RingSC engine): Q2
`NIR_multicast` — ONE Ajtai commitment per union side covering all ℓ
instances (the linear Z_q packing keeps the homomorphism), the single
log ℓ-round sumcheck over `G(Y) = F(x̃(Y), w̃(Y))·eq(Y, r_y)` with the
sumcheck challenges AS the accumulation point τ, the verifier-derived
slack `e = G(τ)·eq(τ, r_y)^{-1}` with the degenerate-eq rejection, the
field-only accumulated vector `x = Σ eq̃_k(τ)·x^(k)`, and the
partial-evaluation consistency (`w̃ = Σ eq̃_k(τ)·w^(k)` — decider-bound
through the C commitment); Q3 the 2-to-1 fold with γ-power combination
`e* = e₀ + γ·T + γ²·e₁` (the cross term T binds via the decider's
`F(x*, w̃*) = e*`), Ajtai-homomorphic commitment folds, and the
ACC.V/ACC.D split (O(1) verifier, opening+constraint decider); Q5 the
multi-instance IVC loop (ℓ chunks per step: multicast → verify → fold
into the running accumulator — test-pinned over 3 steps). 5 tests:
happy+decider, unsatisfied-predicate fail-close, tampered-sumcheck
rejection, fold decider + cross-term binding, the IVC loop.

Open: Q4 (the SPS/CV wrapper), the paper's IOR_batch evaluation-claim
folding (kernel-simplified: the Reval subrelations are decider-verified
per accumulated instance before folding — documented), the O(n²)
multiset difference and the partial-MLE memory blowup.
