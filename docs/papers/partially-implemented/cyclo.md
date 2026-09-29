# Cyclo (ePrint 2026/359) — PARTIAL

Crate: `lattice-folding/src/cyclo.rs`. Implemented: exact signed base-2b
digit chunking (`chunk_element` — the only paper-faithful mechanism per
the Wave-6 audit), extension commitment as chunk→pad→Ajtai commit, fold
with homomorphic commitment, refresh = extension-commit the accumulated
witness, additive norm bookkeeping behind the shared `NormBudget` gate,
fixed-weight biased-ternary ring challenges (Wave 6.1).

Landed (Wave 7.6, `cyclo_protocols.rs` + `fq2_sumcheck.rs`): Π^range as
a real degree-(2b+2) sumcheck over F_{q²} with the digit-MLE structure;
Π^ext fold with RoK constraint rows + `verify_ext_fold`; the shared F_{q²}
sumcheck layer.

Open: the X³−X Karatsuba constant-factor trick;
the R1CS-over-F_q bridge (§7 θ_k); 50-bit production parameter regime
(modulus50 substrate exists); partial_range_check's 2x soundness slack.
