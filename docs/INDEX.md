# The Repository Index — Search & Navigation

**The keyword map of the workspace.** GitHub's native code search
covers identifiers; this index covers CONCEPTS — the project's own
vocabulary (the honest-ledger terms, the wave names, the security
posture vocabulary) mapped to the files that own them. Browsable
top-to-bottom or searched in-page (`Ctrl-F` / GitHub's `in:file`).

---

## By concept

| Concept | Primary file(s) |
|---|---|
| Ajtai commitment / SIS hash | `crates/lattice-commitment/src/ajtai.rs`, `crates/lattice-labinius/src/commit.rs` |
| ABDLOP linear proofs | `crates/lattice-commitment/src/` (linear proofs), `crates/lattice-blindfold/src/protocol.rs` |
| Akita PCS / packed commitments | `crates/lattice-akita/src/pcs.rs` |
| D4 — the response-layer swap (open) | `crates/lattice-akita/src/salsa_response.rs`, `docs/papers/implemented/salsaa.md` |
| D4 — the binding closure (bound) | `crates/lattice-akita/src/salsa_binding.rs`, `docs/analysis/D4_BINDING_CLOSURE.md` |
| D4 — the r-column capacity split | `crates/lattice-akita/src/salsa_binding.rs` (`byte_capacity`, `prove_grouped_salsa_split`) |
| Lemma-4 gate (no-wraparound) | `crates/lattice-salsa/src/ring_norm.rs` (`wraparound_gate`) |
| ψ-functional / byte recomposition | `crates/lattice-akita/src/salsa_response.rs` (`psi_weights_at`) |
| width fold — (W0)–(W4) | `crates/lattice-widthfold/src/fold.rs` |
| recursive width-collapse chain | `crates/lattice-widthfold/src/chain.rs` |
| extraction ledger / degree-law unwind / laws E1–E5 | `crates/lattice-widthfold/src/extraction.rs`, `docs/analysis/MULTISTAGE_EXTRACTION.md` |
| sound coverage boundary (n̄ ≤ 16 → staged) | `crates/lattice-widthfold/src/chain.rs` (module docs), `crates/lattice-widthfold/examples/chain_coverage.rs` |
| ring-functional fold (the Cyclo terminal layer) | `crates/lattice-widthfold/src/ring_fold.rs` |
| compact-PCS terminal / witness-free decider | `crates/lattice-folding/src/cyclo_terminal.rs` |
| Cyclo §7 R1CS bridge | `crates/lattice-folding/src/cyclo_r1cs.rs` |
| the compact mode (LaBinius secondary structure) | `crates/lattice-zkvm/src/compact.rs` |
| the Sound profile / SoundOpening | `crates/lattice-zkvm/src/compact.rs`, `crates/lattice-zkvm/src/memproof.rs` |
| sparse "0s are free" engine | `crates/lattice-memory/src/sparse_engine.rs` |
| v2 pipeline / Stage-5 modes | `crates/lattice-zkvm/src/pipeline2.rs` (`Stage5Mode`, `prove_v2_with_stage5`) |
| v3 pipeline (ring-lookup memory) | `crates/lattice-zkvm/src/pipeline3.rs`, `crates/lattice-lookup-ring/` |
| streaming / client-side proving | `crates/lattice-streaming/`, `crates/lattice-zkvm/src/streaming.rs` |
| RV64IMAC semantics / differential conformance | `crates/lattice-vm/` (`reference.rs` — the independent interpreter) |
| MSIS estimator / ADPS16 | `crates/lattice-sis-estimator/` |
| QROM accountability | `crates/lattice-qrom/` |
| HVZK simulators / chi-square KATs | `crates/lattice-zk/` |
| the norm budget hard gates | `crates/lattice-core/` (NormBudget), every folding module |
| LaBRADOR paper-faithful engine + Greyhound | `crates/lattice-greyhound/` |
| LatticeBlindFold (the ZK/blinding stack) | `crates/lattice-blindfold/` |
| TTRP shortness | `crates/lattice-ttrp/` |
| the proof envelope | `crates/lattice-zkvm/src/envelope.rs` |

## By question

| Question | Answer file |
|---|---|
| What is implemented vs remaining? | `IMPLEMENTATION_CHECKLIST.md`, `NEXT_STEPS.md` (§ the honest ledger), `docs/papers/README.md` (the matrix) |
| When did X land? | `IMPLEMENTATION_LOG.md` (timestamped), `git log` |
| How do the papers connect? | `docs/PAPERS_MAP.md` |
| What are the measured numbers? | `docs/BENCHMARKS.md` (§2l = this wave), `PERFORMANCE.md` |
| What is the security posture? | `SECURITY.md`, `docs/analysis/` (the two formal analyses), `AUDIT_CHECKLIST.md` |
| Why is the chain sound? | `docs/analysis/MULTISTAGE_EXTRACTION.md` (the five laws, the three honesty layers) |
| Why is D4 now binding-complete? | `docs/analysis/D4_BINDING_CLOSURE.md` |
| What caps the byte-witness capacity? | `docs/analysis/D4_BINDING_CLOSURE.md` §3 (the Lemma-4 law), `byte_capacity()` |
| How do I run the evidence? | `README.md` §Build & test; the examples: `extraction_table`, `chain_coverage`, `salsa_bound_size`, `salsa_swap_size` |
| What is the CLOB/ethrex/zoda plan? | `docs/CLOB_WORKLOAD_RESEARCH.md`, `docs/ETHREX_ZODA_INTEGRATION.md` |
| What comes next? | `NEXT_STEPS.md` — the next highest-value items at the top |

## By directory

```text
README.md                     the front door (counts, papers table, quickstart)
IMPLEMENTATION_LOG.md         the timestamped build history
IMPLEMENTATION_CHECKLIST.md   the per-component status grid
NEXT_STEPS.md                 the research backlog + the honest ledger
PERFORMANCE.md / SECURITY.md / AUDIT_CHECKLIST.md
docs/
  ARCHITECTURE.md             the deep architecture (core engine, protocols, invariants)
  PAPERS_MAP.md               the papers' inner connections (this index's companion)
  BENCHMARKS.md               the measured evidence per wave (§2a–§2l)
  DESIGN_50KB.md              the compact-opening pipeline specification
  CLOB_WORKLOAD_RESEARCH.md   the CLOB guest port plan
  ETHREX_ZODA_INTEGRATION.md  the L1/L2 integration design
  WAVE_ANALYSIS.md            the wave planning history
  INDEX.md                    this file
  analysis/                   the formal analyses (extraction, D4 closure)
  papers/                     per-paper coverage docs (implemented / partial / unimplemented)
  research/                   the research reports (next-papers pipeline, landscape)
crates/                       33 crates — see README §Architecture
guests/                       the guest program crates
research/                     paper source material
.github/workflows/ci.yml      CI (test + clippy + the evidence examples)
```

## The vocabulary (grep-friendly identifiers)

`n_bar` (the fold width) · `beta1`/`beta2` (the response gates) ·
`r2`/`amplitude`/`kappa`/`w` (the stage shape) · `psi_weights` /
`mu_weight` (the functional weights) · `byte_capacity` /
`column_count_for` (the split planner) · `chain_extraction_ledger` /
`assert_extraction_sound` (the five laws) · `CHAIN_GRINDING_BITS` /
`REWIND_TREE_CAP` (the allowances) · `Stage5Mode` (the response-layer
selector) · `wraparound_gate` (Lemma-4) · `prove_width_fold_chain` /
`verify_width_fold_chain` (the staging) · `prove_grouped_salsa_bound` /
`verify_grouped_salsa_bound` (the closure) · `decide_principal_linear`
(the bridge terminal).
