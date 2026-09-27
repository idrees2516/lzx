# LatticeFold+ (ePrint 2025/247) — PARTIAL

Crate: `lattice-folding/src/latticefold_plus.rs`. Implemented: the
algebraic range proof (digit decomposition + booleanity + reconstruction
sumchecks, prover/verifier with tamper tests) — Wave 7 fixed the
zero-padding bug (padding entries reconstruct v = 0 + β, not 0) and
added Clone/Debug derives; `prove_range`/`verify_range` is the range
attachment used by PGL-Boot.

Partially landed (Wave 7.7, `lfplus_mon.rs`): the monomial-set machinery,
the ψ layer (Lemma 2.2), Π^mon, and the split/pow double commitments —
with five of its tests currently `#[ignore]`d on open defects (the ψ iff
boundary case and the split/pow digit bounds); the O(n)-add evaluation
trick is realized but untested at scale.

Open: fixing the ignored tests; Π^rgchk/Π^cm, R_lin,B + the R1CS reduction and the
Π^mlin/Π^decomp fold loop (Thm 5.1–5.2) — the ≲100 KB proof path.
