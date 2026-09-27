# Quasar (ePrint 2025/1912) — PARTIAL

Crate: `lattice-lookup/src/lib.rs`. Implemented (Wave 6.5, "Q1"):
committed T/R/Q with τ derived from commitments, counting-map
multiset difference, forged-triple negative test — the soundness hole
(`verify_lookup` binding nothing) is closed. Plus the deterministic
accumulation transform.

Open (item 7.10): Q2 NIR_multicast — union polynomial w̃_∪ with one
commitment C_∪ covering ℓ instances, the log ℓ-round sumcheck over
G(Y) = F(x̃,w̃)·eq̃, the partial-evaluation consistency check; Q3 the
2-to-1 fold (Z-multilinear curves, γ-power combined one-round sumcheck,
IOR_batch via commitment homomorphism) + ACC.V/ACC.D decider; Q4 the
SPS/CV wrapper; Q5 the multi-instance IVC loop (the shard-parallel
zkVM payoff). P: O(n²) multiset difference and k× partial-MLE memory
blowup.
