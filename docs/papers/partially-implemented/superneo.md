# Neo/SuperNeo (2026/242) — PARTIAL

Crate: `lattice-folding/src/superneo.rs`. Implemented: relaxed-CCS fold
algebra with vacuity probes.

Landed (Wave 7.15, `superneo_committed.rs`): `CommittedRelaxedCcsInstance`
— the witness committed via Ajtai BEFORE the Fiat–Shamir derivation
(the fold challenge derives from `hash(commitments, u, slack,
ccs-digest)` — public data only, closing the pre-Wave-6.4 hole where
the challenge absorbed `w₁‖w₂`); the R_q bridge via the linear
small-field packing (`F_{2^16}` witnesses, 16 values per ring element —
the Ajtai homomorphism EXACT across folds with short `r < 2^8`
challenges, fail-closed bounds at `2^25 ≪ q/2`); the
`fold_public`/`fold_secret` split (verifier-side commitment homomorphism
+ `u' = u₁ + r·u₂` + `slack' = slack₁ + r²·slack₂ + r·E` with the
Nova-style cross term `E = quad-cross − (u₁·span(w₂) + u₂·span(w₁))` —
full-field slack, the small bound applies only to the bridged witness);
the pay-per-bit zero-skipping commit path (bit decomposition — the
commitment work ∝ popcount via the Wave-6.8 zero-skip; folded bits
reconstruct linearly, `w = Σ_j 2^j·b_j`, exact for folded bits ≥ 2);
the `π_CCS` decider (opening + reconstruction + relaxed CCS
satisfaction through the shared `verify_folded`). 6 tests: both
packings fold+decide, the FS-challenge binding (determinism from
public digests, never the witness bytes), the wrong-cross-term decider
rejection, the out-of-bounds refusal, and a two-round IVC.

Open: the standalone Neo range proof (appendix A — LZX's equivalent is
the LF+ attachment realized for PGL-Boot) and the paper's native
small-field arithmetic (realized as `F_{2^16}`-embedded Goldilocks).
