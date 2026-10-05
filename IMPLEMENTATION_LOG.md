# Implementation Log — LZX

**The timestamped build history of the workspace, session by session.**
Every entry is anchored to its commit (`git log --format="%h %ad %s"`);
this file is the human-readable index of *what landed when*, in
chronological order (newest first). Companion artifacts:
`NEXT_STEPS.md` (the research backlog + the per-session honest ledger),
`IMPLEMENTATION_CHECKLIST.md` (the per-component status grid),
`docs/BENCHMARKS.md` (the measured evidence per wave),
`docs/analysis/` (the formal analyses).

Test counts are the workspace totals at each wave's landing commit.

## 2026-10-05 (II) — the H6 full-fidelity route + the A5 commitment-scale driver

**Modules:** `lattice-pcs/src/hyperwolf_labrador.rs` (NEW, ~3,400
lines), `lattice-akita/src/a5_committed.rs` (NEW, ~1,800 lines), the
`HwRing::mul` adversarial-overflow hardening. The honest ledger's two
top residuals closed: (1) the LaBRADOR engine re-parameterized to the
HyperWolf ring — the amortized Dachshund over ALL rounds' projection
vectors with per-round exact ℓ2 statements (the σ⁻¹-conjugate
quadratics + the greedy square-decomposition slack) and the
fold-consistency dot-products natively in the same ring — plus the
recursive outer-commitment compaction driver (the O(log log log N)
route) and the `eval_prove_labrador`/`eval_verify_labrador` protocol;
(2) the A5 commitment-scale recursion driver — the Eq 5/6/7/8/2 rows,
the A3 digit-range deferred claims, and the A4 evaluation-trace rows
all feeding ONE A2 fused sum-check against the COMMITTED successor
witness, with the App F.1 ordering, the per-level deferred Goldilocks
claim, and the terminal discharge. 88 suites / 1,313 tests green
(was 1,300); clippy clean; fmt applied. The superseded
`docs/WAVE_ANALYSIS.md` + `docs/DESIGN_50KB.md` wiped (records live
here + `docs/BENCHMARKS.md`). See NEXT_STEPS's session update for the
full per-item mapping and the honest residual ledger.

---

## 2026-10-05 — Wave 7 completion: Akita A3/A4/A5, HyperWolf H6/H7, RoKoko 6/7/8

**Modules:** `lattice-akita/{a3_range,a4_tensor,a5_terminal}.rs` (NEW,
~2,900 lines), `lattice-pcs/{hyperwolf_compact,hyperwolf_batch}.rs`
(NEW, ~900 lines), `lattice-rokoko/{proj_f,schedule,pcs_front}.rs`
(NEW, ~1,100 lines). The six Wave-7 residuals after the audit (items
7.11 A3-A5, 7.12 H6/H7, 7.13 6-8; all other Wave-7 items verified
already landed across the 09-29..10-04 waves). 88 suites / 1,300
tests green; clippy clean; fmt applied. See NEXT_STEPS's session
update for the per-item protocol mapping and the honest residual
ledger.

---

## 2026-10-04 (21:30 PKT) — the extraction ledger + the D4 closure + the r-column split

**Commits:** this wave (`extraction.rs`, `salsa_binding.rs`, the
pipeline2 bound mode, the production docs) — the three honest-ledger
follow-ups of the staging wave, plus the repo productionization
(this log, the checklist, the papers map, the search index, CI).

- **the multi-stage LaBRADOR extraction ledger**
  (`lattice-widthfold/src/extraction.rs` + `--example
  extraction_table` + `docs/analysis/MULTISTAGE_EXTRACTION.md`): the
  degree-law unwind formalized — the stage-local affine degree law
  (fork width 2 by the garbage pre-commitment), the composed claim
  degree 2L, the unwind norm law `2·β_{L+1} < q/2`, the grinding
  ledger, the extractor-feasibility cap `2^L ≤ 2^16` (laws E1–E5,
  machine-checked) — now ENFORCED inside `assert_sound_chain` at prove
  AND verify time. The knowledge gap vs the binding security are
  reported as distinct quantities (the rewind abort is not the forgery
  security).
- **the D4 binding closure** (`lattice-akita/src/salsa_binding.rs` +
  `docs/analysis/D4_BINDING_CLOSURE.md`):
  `prove/verify_grouped_salsa_bound` — the byte-witness↔commitment
  authenticated opening via the width-collapse chain (the compact-fold
  composition); the ψ-functional sumcheck is subsumed by the chain's
  (W0')/(W3) thread; the closure is pinned by the wrong-commitment
  tamper test. The pipeline2 layer gains `Stage5Mode::Bound`
  (`prove_v2_with_stage5`) — the v2 pipeline runs end-to-end with
  every column's byte-witness chain-bound.
- **the r-column capacity split** (same module): `byte_capacity` states
  the Lemma-4 cap exactly (2,048 values at ring dim 16/Q_32 — the
  prior "~1,200" prose note, now executable and fail-closed);
  `prove/verify_grouped_salsa_split` scales past it with the compact
  mode's discipline (r columns, μ-weighted ψ-decomposition, per-column
  D1 + binding chains).
- **the measured evidence** (BENCHMARKS §2l): bound responses
  5.5–54.8 KB at 2^6–2^10 values (11–671× under Clear; 8.6–59× over
  the open mode — the honest price of the binding); the split at
  2^12/2^13 = 133.5/265.8 KB (18.7× under Clear, 2/4 columns).
- **productionization**: debug-mode overflow fixes (the test RNGs —
  the suite now passes with overflow checks ON in BOTH profiles), CI
  workflow, this log, the checklist, the papers map, the search index.
- **Tests:** 1,237 green (+10: 5 extraction, 4 binding/split, 1
  pipeline composition); clippy clean on every touched crate.

## 2026-10-04 (14:01 UTC) — the staging wave

**Commit `18c3e82`.** (1) the recursive width-collapse staging
(`lattice-widthfold` — NEW crate: `chain.rs`, the log-stages taking the
Sound profile's coverage from `n̄ ≤ 16` to the benchmark streams;
measured boundary n̄ ≤ 4096 at the byte gate; the Sound memory
argument 129.7 → 55.2 KB); (2) SALSA D4 — the Akita/zkVM
response-layer swap (640–928 B responses at 2^6–2^10, 61–671× vs
Clear, zero disclosure; the Ajtai binding recorded as the outer-layer
gap — closed later this day, above); (3) the Cyclo §7 bridge's
compact-PCS terminal (`cyclo_terminal.rs` — the witness-free decider).
Tests: 1,227.

## 2026-10-04 (10:42–10:58 UTC) — the LaBRADOR decider + the Cyclo §7 wave

**Commits `7b5fc5d`, `12f0a7f`.** The width fold (Stage 5.2: the
quadratic-garbage fold with (W1)–(W3), the estimator-gated `[A₂ | −T]`
binding); the Sound compact profile + the Sound memory argument
(129.7 KB at the test scale); the estimator evidence tables; Cyclo §7
— the R1CS-over-F_q bridge (the θ_k digit map, the bridge-local
F_{q²} extension, the HyperNova-style linearized sumcheck, the ring
lifts with the carry finding pinned) + `decide_principal_linear` (the
bridge end-to-end).

## 2026-10-03 — the zkVM semantics waves + SALSAA D3/D6

**Commits `f18b396` → `a91e70f`, `1242822`.** The Twist-and-Shout
instruction-semantics constraint families (T2: ten legs, fail-closed
coverage); the constraint legs wired into the live pipeline; the
verify-side carrier factoring (sub-second verify); SALSAA D3 + D6
(Π_batch-star row-count-preserving batching, Theorem 4); the Wave-8
research report (`docs/research/`).

## 2026-10-02 — the PCD / Accordion / CauchyFold / LaBRADOR+Greyhound / LatticeBlindFold waves

**Commits `780f870` → `b7f4391`, `3ab000a` (the union merge).**
ePrint 2026/471 (the ring-lookup wave: Ring-Plookup + Ring-LogUp over
the CRT-split ring, the v3 pipeline); ePrint 2022/1341 + 2024/1293
(`lattice-greyhound` — the paper-faithful LaBRADOR engine + the
Greyhound PCS, the 53KB accounting); ePrint 2026/289 + 2026/538 (the
PCD wave: ZK-PCD from accumulation + holography PCD); ePrint
2025/1325 + 2026/2011 (Accordion + CauchyFold); ePrint 2026/1857
(LatticeBlindFold — the blinding stack complete).

## 2026-10-01 — the Tier-0 wave + the streaming rebase + the Fp256 provers

**Commits `5668925` → `e15d365`, `8ab5103` (v2 pipeline), `95bf2d4`
(wave 8.5).** TTRP (2026/2146) + the sum-check speedups; Π_CCS (Neo
§7 / SuperNeo §7.3); the SVSC windowed projective prover; the
four-part Fp256 wave; the v2 pipeline (Twist & Shout in the live path,
the verifier never re-executes); wave 8.5 (the sparse "0s are free"
engine).

## 2026-09-30 — the streaming wave + the 50KB compact opening + DESIGN_50KB

**Commits `59a1a42` → `1c010d2`.** The monomial-basis sum-check
(2026/762) + proving in small space (2025/611) — streaming and
client-side proving; the 50KB compact-opening pipeline (fibonacci
3,640 → 75.5 KB); the Stage-4 leg batching (33 KB fibonacci proofs);
DESIGN_50KB.md (the full specification).

## 2026-09-29 — the §4 performance roadmap + wave 7.4 (the zkVM production path)

**Commits `f2e4983` → `5d184de` (the merge).** The memory argument
proven WITHOUT re-execution; the AVX-512 kernel parity wave (the
49× reference round); the entropy-coded folded opening; the LaBRADOR
big-integer vectorization; the Goldilocks SIMD kernels.

## 2026-09-28 — wave 7 in force (the protocol port day)

**Commits `303b133` → `d034759`.** ProtogaLattice PGL-Fold/PGL-Boot;
SALSAA D1+D2 (the Lemma-4 gate); labinius wire/ + Recursive; LF+ fixes;
PikkuFold LRP; Symphony; Quasar accumulation; SuperNeo; the labinius
Recursive wiring; **D4** (the SALSAA response-layer swap, first
landing); the SALSAA/HyperWolf/RoKoko/Serval/Hachi native-Rust port
(`00dff23` — 519 tests).

## 2026-09-27 — wave 7 partial landing + T&S PIOPs

**Commits `d383997` → `41d8a05`.** SALSAA D1+D2; the labinius wire
layer; the Twist & Shout PIOPs (one-hot/Shout/Twist/sparse engines);
the production docs.

## 2026-09-24 — waves 5 + 6 (efficiency + the soundness-critical substrate)

**Commits `cdb61a2`, `5eeb886`, `e69732e`.** The AVX-512 PCS backend +
PCLMULQDQ (341 tests); the wave-6 research (NEXT_STEPS.md born, 830
lines); the shared substrate wave: paper-calibrated challenge
distributions, the norm wraparound gates, the Ajtai cached-NTT path,
the committed Quasar protocol, F_{q²}, the SIS estimator.

## 2026-09-23 — waves 3 + 4 (QROM/ZK/bench + the external ports)

**Commits `6166ee6` → `9e38417`.** lattice-qrom (query ledgers,
attestations); lattice-zk (secret entropy, HVZK simulators,
chi-square KATs); lattice-bench (the reproducible matrix); the labinius
PCS port + the native LaBRADOR port (240 tests).

## 2026-09-21 — waves 1 + 2 (the foundation)

**Commits `6166ee6` → `b2c4172`.** lattice-core (Goldilocks, Keccak,
transcript, MLE); lattice-ring (negacyclic NTT, packing); the Ajtai
commitment; sumcheck + relations; lattice-folding (all six papers:
ProtogaLattice, LatticeFold+, Cyclo, PikkuFold, Symphony + the
precursors); lattice-lookup + lattice-salsa; RoKoko + Hachi
embeddings; lattice-vm (the canonical RV64IMAC decoder); lattice-memory
(Twist & Shout); lattice-zkvm (end-to-end prove/verify).
