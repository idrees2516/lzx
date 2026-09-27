# PikkuFold (ePrint 2026/1809) — PARTIAL

Crate: `lattice-folding/src/pikkufold.rs`. Implemented: single-layer
biased-ternary projection (entry distribution correct), fold with
fixed-weight challenges behind the norm gate, `fold_with_binding`
(ABDLOP LinearProof response — the anti-feature the paper's design
avoids; quarantining it as test-only is part of item 7.8).

Open (item 7.8): layered LRP (d layers, coarse ring lifts + final fine
coefficient layer, `Tr(Mw) = P·coeff(w)`), RingSC with subfield
batching (80% of the paper's prover time), the certified-ℓ2 JL gate
(Thm 2 Table 1 constants + the ‖v_tr‖ ≤ ω Fig-2 gate), relation upgrade
with MLE evaluation claims (s, t), AIR folding (§7.2), norm/SIS
accounting with periodic reset.
