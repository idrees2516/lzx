# LatticeFold+ (ePrint 2025/247) — PARTIAL

Crate: `lattice-folding/src/latticefold_plus.rs`. Implemented: the
algebraic range proof (digit decomposition + booleanity + reconstruction
sumchecks, prover/verifier with tamper tests) — Wave 7 fixed the
zero-padding bug (padding entries reconstruct v = 0 + β, not 0) and
added Clone/Debug derives; `prove_range`/`verify_range` is the range
attachment used by PGL-Boot.

Landed (Wave 7.7 complete, `lfplus_mon.rs`): the monomial-set machinery,
the ψ layer (Lemma 2.2), Π^mon, and the split/pow double commitments —
all 9 module tests green. The five formerly-`#[ignore]`d defects are
fixed: the split/pow digit radix is the paper's base-`d'` (was `2^{d'}`),
the `dcom` norm gate is `∥τ∥∞ ≤ d'/2`, and the challenge-point recovery
no longer desynchronizes the transcript (the Π^mon proof now carries the
prover's point; the decider uses the verifier-derived one). The O(n)-add
evaluation trick is realized and tested against the direct evaluation.

Open: Π^rgchk/Π^cm, R_lin,B + the R1CS reduction and the
Π^mlin/Π^decomp fold loop (Thm 5.1–5.2) — the ≲100 KB proof path.
