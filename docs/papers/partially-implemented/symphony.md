# Symphony (ePrint 2025/1905) — PARTIAL

Crate: `lattice-folding/src/symphony.rs`. Implemented: μ-ary degree-2
fold with explicit pairwise cross terms E_ij (exact identity tested at
μ = 3, 4) under the stacked cross commitment; LaBRADOR SmallSet
challenges {0,±1,±2} (Wave 6).

Landed (Wave 7.9, `symphony_protocols.rs`): the tensor substrate `ts(r)`
(the eq weight vector of the Eq-23 linear output relation, MLE-consistent
by test); `Π_had` (Fig 1) as a real prover+verifier over the R_q-native
RingSC engine — the Eq-24 degree-3 sumcheck with α-power column batching,
the Eq-25 terminal cross-check against the prover's `U ∈ R_q^{3×d}`
claims, and the decider `⟨M_i·f, ts(r)⟩ = v_i` with commitment openings;
the O(μ) shared-randomness fold (Fig 4): ONE merged sumcheck for ℓ_np
instances via the Eq-45 α-power RLC (rounds stay log(m) — test-pinned,
the O(μ²) pairwise `E_ij` enumeration is never touched), `β ← S^{ℓ}`
fixed-weight folds of commitments/evaluations/witnesses (Eqs 48–49), and
the Eq-50 feasibility gate `B_bnd ≥ √ℓ·∥S∥_op·max(B·n^{d/ℓh}, √n)`
enforced fail-closed both on the folded norm (prover) and the q/2
wraparound margin (verifier). 7 tests: ts/MLE identity, Π_had happy +
decider, non-Hadamard witness fail-closed, tampered-U rejection, the
one-sumcheck-regardless-of-arity fold with a decider-valid folded
instance, the Eq-50 gate, and the bound formula.

Open: Π_rg (the approximate range/monomial check), the CP-SNARK compiler
(Constr 6.1), two-layer folding (§8), the memory strategy (Remark 4.1),
and the paper's F_{q²} challenge field (realized over Z_q — the engine's
subfield view).
