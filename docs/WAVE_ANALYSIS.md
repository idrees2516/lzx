# LZX Wave Analysis — Waves 6/7/8 In-Depth Gap Audit & Landing Ledger

Date: 2026-09-29. Method: every Wave 6/7/8 item from `NEXT_STEPS.md` §4 was
audited against the code (git history + file:line evidence), then the
remaining parts were implemented where the session budget allowed. This
document is the authoritative record of **what was audited, what landed
today, and what remains**.

---

## 1. Executive summary

| Wave | Items | Landed before today | Landed today | Remaining |
|---|---|---|---|---|
| 6 (substrate) | 10 | 10 | — | Modulus50/SIS-estimator *wiring* into protocols (substrate exists, unused) |
| 7 (protocols) | 16 | 13 (+ partial Akita A3) | **7.4's core: the zkVM production path** | 7.11 A3/A4/A5 completion, 7.12 H6/H8 completion, 7.13 items 6–8 |
| 8 (production) | 10 | 0 | **8.5 (partial), 8.8 (partial), 8.10 (partial)** via the memory-argument stack | 8.1–8.4, 8.6, 8.7, 8.9, most of 8.10 |

The headline of today's session: **Wave 7.4's P0-4/P0-5 core landed** — the
zkVM's memory argument is now proven and verified **without re-execution**
over real guest programs (see `docs/BENCHMARKS.md`), with the
virtual-`Val` Val-evaluation sumcheck that closes the stale-read soundness
hole of a bare materialized-`Val` formulation.

---

## 2. Wave 6 — shared substrate: COMPLETE (with wiring caveats)

Every Wave 6 item landed in commit `e69732e` (341 tests) and was re-verified
today:

| # | Item | Status | Evidence |
|---|---|---|---|
| 6.1 | `short_challenge` fixed-weight ternary + Γ_C | LANDED | `lattice-core/src/short_challenge.rs` (880 lines); consumed by pgl, cyclo, symphony, pikkufold, hyperwolf, akita |
| 6.2 | `NormBudget` hard gate | LANDED | `lattice-core/src/norm_budget.rs`; wired in 6 modules |
| 6.3 | Ajtai cached-NTT + Barrett + statement absorption | LANDED | `ajtai.rs:87-103`, `modulus.rs`, `absorb_statement` |
| 6.4 | FS hygiene (cyclo/superneo) | LANDED | `cyclo.rs:229-279`, `superneo_committed.rs:126-135` |
| 6.5 | Quasar Q1 committed lookup | LANDED | `lattice-lookup/src/lib.rs:291-432` (5 binding layers + forged-triple test) |
| 6.6 | F_{q^e} extension layer | LANDED (e=2) | `lattice-core/src/extension.rs` + `fq2_sumcheck.rs` |
| 6.7 | Modulus50 incomplete-NTT | LANDED (substrate) | `lattice-ring/src/modulus50.rs` (758 lines) — **not consumed by any protocol**; RoKoko/Cyclo/PikkuFold still run Q_32 |
| 6.8 | Zero-skipping Ajtai + bit-packed one-hot | LANDED | `sparse.rs` — **now consumed** by today's bits bundle (31 bits/coeff) |
| 6.9 | SIS estimator | LANDED (core) | `lattice-sis-estimator` (5 modules) — **not wired into SecurityProfile** (that is 8.8) |
| 6.10 | Folding/lookup/salsa benches | LANDED | `lattice-bench/src/bin/bench.rs` (35 stages) — **extended today** with the zkVM memory-argument benchmark |

Wiring gaps (6.7, 6.9) are Wave 8.8 work, not new substrate.

---

## 3. Wave 7 — protocol completion

### 3.1 The pre-existing landing ledger (verified today)

7.1 ProtogaLattice PGL-Fold (`pgl.rs`, 50 tests) — LANDED.
7.2 SALSAA D1+D2 (`ring_norm.rs`, wraparound gate) — LANDED.
7.3 D4 Akita response swap (`salsa_response.rs`) — LANDED for Akita; **the
zkVM half landed today** through the bundle openings (the zkVM now uses the
polylog-claim grouped-opening flow, not the Θ(N) witness reveal of the old
`prove_evaluation` path); the Ajtai-binding of `z(r)/f(r_sc)` to the
commitment remains the documented outer-layer gap.
7.5 labinius Recursive mode (`scheme.rs`/`recursion.rs`) — LANDED.
7.6 Cyclo Π^range + Π^ext RoK (`cyclo_protocols.rs`) — LANDED (decider
model: `opening_v` in the clear).
7.7 LF+ monomial/ψ (`lfplus_mon.rs`, 9 tests) — LANDED.
7.8 PikkuFold layered LRP (`pikkufold_lrp.rs`, 14 tests) — LANDED.
7.9 Symphony Π_had + O(μ) fold (`symphony_protocols.rs`, 7 tests) — LANDED.
7.10 Quasar Q2+Q3+IVC (`quasar_acc.rs`, 5 tests) — LANDED.
7.14 PGL-Boot — LANDED. 7.15 SuperNeo committed (`superneo_committed.rs`,
6 tests) — LANDED. 7.16 labinius wire/ (`wire.rs`, 9 tests) — LANDED.
7.11 SALSAA A2–A5 (`salsaa.rs`/`air.rs`) + 7.12 HyperWolf FULL
(`hyperwolf.rs`) + 7.13 RoKoko core (`com.rs`/`protocol.rs`) — LANDED in
`00dff23` (77 tests).

### 3.2 What landed TODAY: Wave 7.4's core — the zkVM production path

The pre-session audit (§3.11 of `NEXT_STEPS.md` confirmed): the zkVM
verifier **re-executed the whole program** (`prove.rs:211-215`), checked the
sumcheck **by byte length only** (`:276`), and used the unsound
`twist_fingerprint` memory statement. Today's landing, crate
`lattice-zkvm`:

| Module | Lines | What it is |
|---|---|---|
| `columns.rs` | 470 | The P0-4 column restructure: 64-bit value tensors (rs1/rs2/imm/rd/mem_old/mem_new), instruction tensor, flags, word-granular shadow replay with **initial-image seeding** (reads of never-written words see the initial state — the bug where an empty shadow produced `mem_old = 0` was found and fixed end-to-end), alignment fail-closed |
| `ledger.rs` | 700 | The resolver↔PCS glue (the audit's "single highest-leverage missing component"): the claim ledger with derived-factor expansion (limbs = 16 tensor-row claims, combos = 64), the **bits bundle** (bit-packed at 31 bits/coeff — Wave 6.8 finally consumed), the **values bundle**, grouped carrier openings with compact norm proofs, and the full binding chain (A·s = t, digit reconstruction, f(r_sc) recomputation from the response) |
| `memory.rs` | 1000 | The Twist & Shout instances over the digit-bit representation: **virtual one-hot matrices** (never committed — every claim proven by a matrix-evaluation sumcheck expanding the one-hot selection into affine digit-bit factors), **virtual `Val` with the Fig-9 Eq-11 Val-evaluation sumcheck** (closing the stale-read hole: a materialized-`Val` formulation admits mid-interval `Val` shifts that keep telescoping consistent — the stale-read rejection test pins this), telescoping against public init/final, per-limb instances (16-bit values, no combo aliasing), 13 legs per read-write instance in a fixed protocol order |
| `memproof.rs` | 750 | The orchestrator: `prove_memory_argument` / `verify_memory_argument` — nine instances (4 register limbs, 4 RAM limbs, 1 fetch Shout over the public program table), statement-derived seeds, **verification with zero re-execution** (the verifier recomputes only public tables), tamper tests (wrong final regs / final memory / claims / program all rejected) |
| `constraints.rs` | 700 | The instruction-semantics substrate: the auxiliary-column builder (selectors, flags, carries, eq-prefixes with the signed/unsigned **shared-eq-prefix** trick — the bit-63 flip preserves `eq`, so only `lt`'s head term differs), the leg driver with factor-view binding, and the booleanity family (paired prove/verify) |

The `PROVE_CYCLE_CAP` / dense-matrices posture: the twist legs materialize
`K × T_s` matrices prover-side — kernel-scale by design; the paper's sparse
provers (§6/§7) are the Wave 8.5 follow-up.

### 3.3 What remains in Wave 7

- **7.4 completion**: the arith/logic/comparison/control/routing/halted
  constraint families (the substrate and driver are in `constraints.rs`;
  the identities are enumerated in the module docs — ADD/SUB limb+carry,
  bit-level logic ops, the eq-prefix comparisons, branch/jump control,
  load/store bit routing through public indicator MLEs, halted
  propagation). Until they land, `prove_memory_argument` must be read as a
  proof of the memory-checking relation over committed streams, not yet of
  full instruction semantics — the module docs say exactly this.
- **7.11 A3/A4/A5**: Akita digit-range sumcheck (symmetry halving), tensor
  reduction + trace functional, and the recursion driver that chains folds
  into the terminal.
- **7.12 H6/H8**: HyperWolf LaBRADOR compaction + `PcsBackend` unification
  (a partial H8 landed in the working tree today: `evaluate_direct` and
  its claim-semantics test).
- **7.13 items 6–8**: Π^proj-f, the norm schedule, the PCS front end.

---

## 4. Wave 8 — production posture

Landed today (as byproducts of the memory-argument stack):

- **8.5 (partial)**: the claim-ledger expansion machinery (derived factors
  resolved through affine combinations of base claims) is exactly the
  constant-factor substrate the sparse prover needs; the sparse round
  computation itself (the paper's §6.3/§7 "0s are free" engines) remains.
- **8.8 (partial)**: statement-derived seeds and the two-bundle norm-bound
  discipline; the estimator-gated `SecurityProfile` wiring remains.
- **8.10 (partial)**: the memory argument replaces the `twist_fingerprint`
  shortcut in the live protocol path.

Everything else in Wave 8 (8.1–8.4, 8.6, 8.7, 8.9, the rest of 8.10)
remains as specified in `NEXT_STEPS.md` §4.

---

## 5. The honest soundness ledger

1. The memory argument is sound as a memory-checking statement (the
   paper's Thm-4 structure: read-checking + write-checking + Val-evaluation
   + telescoping, all terminals bound through matrix-evals and the
   bundles).
2. The instruction-semantics binding is NOT yet proven — the streams'
   values are committed but not pinned to ALU/decode semantics. This is
   the remaining P0-4 work item, stated in `memproof.rs`'s module docs.
3. The Ajtai-binding gap of `salsa_response.rs:18-22` (7.3's outer layer)
   is unchanged; the zkVM bundles avoid it by binding through the compact
   norm proof + A·s recomputation.
4. Kernel-scale parameters throughout (Q_32 rings, m ≤ a few thousand);
   paper parameters need 6.7's Modulus50 wiring or 8.3's RNS crate.

## 6. Guest programs (the Jolt-style suite)

`lattice-guest` (landed today, 18 + 8 tests): a two-pass RV64IM assembler
(labels, directives, every VM instruction, GAS-ish text front-end, full
decode round-trip tests) and the benchmark programs — fibonacci, collatz,
sorting, memory-ops, regex (table-driven DFA for `(ab|ba)*c`),
matrix-mul, modinv (Fermat inverses mod 2^61−1), muldiv — each with a
pure-std reference and a VM-vs-reference test. SHA-2/SHA-3/merkle are the
documented next guests.

## 8. The Stage-4 + streaming session (2026-09-30, evening)

The DESIGN_50KB final cut executed end-to-end, plus the two streaming
papers' client-side path:

| item | landed | evidence |
|---|---|---|
| **Stage 4 leg batching** | ~117 legs → 12 staged batched sumchecks (`legbatch.rs`); fibonacci **75.5 → 33.0 KB** (110× vs Clear; 27.0 KB at the k=2 prototype) | `memproof::compact_tests`, `zkvm-membench` |
| **Algorithm 3 (2025/611 App D)** | the bucketed `O(n)`-space grand-product round prover — LSB-first binding, open-bucket routing, completion-label flushes; the `O(2^n)` g-tables eliminated | `grand_product::tests::bucketed_*` (round-1 bit-identical to the direct evaluation; g-claims vs rebuilt tables) |
| **The O(K + log T) path** | the VM's step function wired into `ChunkedRegenOracle` (`build_streaming`, the budget-derived chunk); pc/witness/fingerprint columns are regeneration oracles — no `O(T)` materialization on prover or verifier | `streaming::tests` (incl. the materialized cross-check + indexed access) |
| **Stage 5.1 (the MSIS table)** | the estimator run + the honest verdict: `k=2` is `~2^12` at every response length; **shipped the `k=4`, `A=2^6` interim**; the sound posture needs the second-level fold | `fold_security_table.rs`, SECURITY.md |
| **Stage 5.4 (partial)** | the ledger `fix_last_variables` tail cache — the digit-row claim pattern's `(log_k+1)×` resolution win | `ledger.rs` |

Workspace: **685 tests green, 0 clippy warnings**. The remaining ledger:
the LaBRADOR decider (5.2 — now estimator-MANDATED), MLE-structured
verifier tables (5.3), the sparse prover + SIMD/RNS (5.4), and the
instruction-semantics families (5.5 — the substrate and aux columns are
built; the arith/logic/comparison/control/routing constraint
polynomials are the next focused session).

## §11 — The Π_CCS / SVSC-window / TTRP wave

Three modules landed, closing the three gaps named in the session brief:

1. **Π_CCS — the in-sumcheck norm products** (`lattice-folding/src/pi_ccs.rs`,
   `docs/papers/implemented/pi-ccs.md`): the succinct CCS decider — the NC
   term (the norm check) lives INSIDE the decider sum-check as
   `Π_{a=−b+1}^{b−1}(Z̃(X)−a)` products over the stacked digit MLE; the
   F/EvalK/EvalA terms carry relaxed satisfaction and the running claims;
   the output is the per-level eval-claim API. No witness transmission in
   the proof itself — the O(L·(t+2)) claims replace the O(n) reveal.
2. **The ss-class weighting restructure over Fp256**
   (`lattice-projsumcheck/src/svsc.rs`,
   `docs/papers/implemented/svsc-window.md`): the SVSC windowed projective
   prover — byte-identical transcripts, measured κ ≈ 61.6, 1.8–2.2×
   speedups at d=2/33-bit values. **Plus the critical BN254_FR modulus
   transposition fix** (the committed field was computing modulo a wrong
   modulus — every prior Fp256 result was garbage in the true field).
3. **TTRP + the digit-free opening** (`lattice-zkvm/src/ttrp.rs`,
   `docs/papers/implemented/ttrp.md`): the tensor-train random projection
   as the ledger's norm-check module (replacing the JL routes) with the
   compact-mode linear-functional bridge (replacing the digit reveal) —
   zero digits, the eval-claim API terminal.

Honest residuals: the Π_DEC norm chain (the claim-witness growth across
folds); the mixed-degree window message (the (d+2)-point grid); the
MultiProductEval shared-extrapolation grid (the remaining gap to the
paper's 2.5–4×); the Lift-and-Batch refinement; the full bundle-opening
swap inside `prove_grouped_carrier`.
