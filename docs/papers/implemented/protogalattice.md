# ProtogaLattice (ePrint 2026/1317) — IMPLEMENTED

Crate: `lattice-folding/src/pgl.rs` (Wave 7.1 + 7.14). Papers read from
source; protocol realized from Fig. 2 (PGL-Fold) and Fig. 3 (PGL-Boot).

## What is implemented (part by part)

| Paper part | Realization | Tests |
|---|---|---|
| Relaxed relation Σ_i pow_i(β)·f_i(w) = e | `PglConstraintSystem::relaxed_value` (ring-product semantics, pow-tower compression ⊗^t(1,β_j)) | `pow_tower_matches_tensor_products` |
| PGL-Fold round 1 — δ ← C, F(X) = Σ_i pow_i(β+Xδ)f_i(w₀), send F_1..F_t | `compute_f_poly` (δ-vector = powers δ^{2^j}; degree ≤ t = log n) | `fig2_fold_protocol_end_to_end` |
| Round 2 — α ← C, β* = β + α·δ, F(α) := e + ΣF_j α^j (e-substitution binds F to the instance) | `fig2_fold` round-2 block | `fig2_tampered_f_coeff_fails` |
| H(Y) = Σ_i pow_i(β*)·f_i(Σ L_j(Y)·w_j), Lagrange forms L₀=Y₀, L_j=Y_j−Y_{j−1}, ghost zero-witness L_{k+1}=1−Y_k | `compute_h_poly` (dense MPoly over R_q) | fold identity tests |
| Gröbner division H(Y) − Y₀F(α) = Σ_{a≤b} K_ab·(Y_aY_b − Y_a), remainder 0 asserted in code; deg(K) ≤ d−2 checked by the verifier | `groebner_divide` (min-index free reduction, value-preserving on the suffix-ones selection chain) | `groebner_division_properties` |
| Round 3 — y ← C^k, y₀ := 1 (Cyclo's trick: accumulator coefficient 1 → linear norm growth) | `fig2_fold` round-3 block | `fig2_iterated_folding_then_decide` |
| e*-check: e* = F(α) + Σ K_ab(y)·Z_ab(y); t* = t₀ + Σ(y_j−y_{j−1})t_j (homomorphism) | `fig2_verify` (full transcript replay re-derives δ, α, y) | `fig2_fold_protocol_end_to_end`, `fig2_tampered_quotient_fails_e_star` |
| Challenge space C = {−1,0,1,2}^N fixed weight T | `PglChallengeParams` → `ShortChallengeFamily::FixedWeightSmallSet` | shared substrate tests |
| Decider: open t*, check Σ pow_i(β*)f_i(w*) = e*, norms | `decide` | all fold tests |
| Invalid fresh instance detection (linear remainder ≠ 0) | prover-side division assertion | `fig2_fold_rejects_invalid_fresh_instance` |
| PGL-Boot (Fig 3): base-b balanced-digit decomposition, per-block commitments t_j, errors e_j (same β), D-point identity Σb^j e_j + ΣZ(D)K(D) = e, Σb^j t_j = t, fold at fresh y | `fig3_boot` + `fig3_boot_verify` | `fig3_boot_refreshes_norm`, `fig3_boot_tampered_block_fails` |
| Range-proof attachment (Π_rg): certify digit blocks are short | `boot_with_range` (LF+ algebraic range proofs per block) | `boot_range_attachment_produces_proofs` |

## Realization notes (documented in-code)

* The vanishing ideal is realized as `J = ⟨Y_aY_b − Y_a : a ≤ b⟩` — the
  vanishing ideal of the suffix-ones selection chain; the division is
  the paper's Prop.-5-style free reduction (`Y_aY_b → Y_a`), exact and
  remainder-asserted. Quotients include index-0 pairs (the paper
  substitutes y₀ := 1 before dividing — a constant-factor optimization).
* H(Y) is a dense MPoly: O(C(k+d, d)) terms — exact at kernel scale;
  the paper's structured per-constraint route avoids the blowup at scale.

## Not implemented (honest gaps)

* Multi-folding k > 2 at paper scale (k = 2 tested; the algebra is
  arity-generic), PCD accumulator composition, CCS bridge, Table-2
  production parameters (needs the SIS-estimator gate, Wave 8.8).
* Norm accounting is tracked with the shared `NormBudget` gates, not the
  paper's exact γ' = γ + (2kT+1)B theorem constants.
