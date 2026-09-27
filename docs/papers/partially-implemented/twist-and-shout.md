# Twist & Shout (ePrint 2025/105) + the zkVM stack — PARTIAL

Crates: `lattice-memory`, `lattice-zkvm`, `lattice-vm`.

## Implemented

* Sound memory-checking ORACLES (`twist_check`, `shout_check`) — ground
  truth for the protocol layer.
* Grand-product `shout_fingerprint` (the Lipton's-trick technique the
  paper eliminates) and `twist_fingerprint` (final-state consistency
  only — documented as unsound as a memory-correctness statement).
* The RV64IMAC conformance layer: decoder/executor/reference, golden
  vectors, a differential corpus (131 randomized programs) that fixed
  four real VM bugs (unaligned Ld/Sd word-straddle, SRLI/SRAI shamt
  decoding, SLLIW/SRLIW/SRAIW widths, LUI/AUIPC sign-extension).
* Envelope discipline: strict decoding + 4000-mutation fuzzing.

## Landed (Wave 7.4 P0-1..P0-3 — `lattice-memory`)

* `onehot.rs` — the d-dimensional one-hot layout + chunking policy
  (§2.5.3, §2.8, §3.7) with size accounting (committed entries and key
  elements shrink with d — the commitment-key control).
* `onehot_check.rs` — the one-hot constraint PIOP (Figs 6/8): Booleanity
  + Hamming-weight-one (the 2^{-1} point trick) + the raf-evaluation
  sumcheck; tamper/resolver-substitution tests.
* `shout.rs` — the core Shout read-checking sumchecks (Figs 5/7), both
  the general-d and the d=1 fast form (cycle-pre-bound, the paper's
  T-multiplication headline).
* `twist.rs` — the Twist increment commitment (Fig 9):
  Inc(k,j) = wa·(wv − Val) with the three sumchecks (read-checking,
  Inc-definition, telescoping against public init/final) — the
  memory-correctness statement the old fingerprint could not make; the
  stale-read blind spot is test-pinned as rejected.
* `sparse.rs` — the sparse one-hot substrate (§2.9.2): index-list
  factors, O(T) sparse evaluation, "0s are free" work accounting (K-fold
  savings pinned by tests); the engine-native sparse rounds are the
  Wave 8.5 integration point.
* `lt_extension` in lattice-core::mle (the less-than gadget).

## Open (items 7.3 + 7.4 remainder — the production path)

* 7.4 P0-4: column restructure of `prove_program` (ra_i/wa_i/Inc/wv per
  column) and the envelope carrying column commitments.
* 7.4 P0-5: DELETE verifier re-execution (prove.rs:211–215), replace the
  byte-length-only sumcheck check (:276–278), stop shipping the witness
  in the clear; tamper tests that must fail without re-execution.
* 7.3 D4: swap the response layer to the SALSAA chain (the
  `prove_norm_chain` API is the integration point) — Θ(N)→polylog,
  disclosure removed.
* P1 sparse-dense sumcheck for structured tables (§7), locality-aware
  binding orders (§8.2), Gruen/Dao-Thaler round optimizations
  (Wave 8.5), instruction-semantics constraints (bytecode-fetch as
  Shout over the program image), production parameters, RVC standard
  conformance (the current funct3 mapping is codebase-local).
