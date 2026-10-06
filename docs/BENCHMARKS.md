# LZX zkVM Benchmarks — End-to-End Memory-Argument Proofs vs SOTA zkVMs

Date: 2026-09-29; re-run + extended 2026-10-05. Harness: `cargo run
--release -p lattice-bench --bin zkvm-membench`. Guest programs:
`lattice-guest` (Jolt-lineage benchmark set, real algorithms,
reference-checked).

## 0. The 2026-10-05 full-system re-run (this session)

| harness | workload | prove | verify | size |
|---|---|---|---|---|
| zkvm-membench (batched-compact, Stage-5.2 claims fold) | fibonacci, 185 cycles | **160 ms** | **58 ms** | **31.0 KB** |
| rokoko driver_bench | m_w=512, r=2, 2 rounds (coarse +1 → fine +2/+n_bat) | 4 620 ms | 484 ms | ledger-published |
| akita recursion_bench | 4-block fold chain + the Rice terminal | ~1 ms | ~1 ms | 600 B terminal |
| labinius lattice-bench | the PCS round suite (sizem) | 78 ms/round | 2.6 ms | — |
| projsumcheck proj_bench | degree-2 sumcheck n=20 | 18.8 ms | — | 320 B (1.50× smaller) |

The SOTA reading of these numbers lives in
[`SOTA_COMPARISON.md`](SOTA_COMPARISON.md) — the Airbender / Zisk /
Lattice-Jolt / Akita ledger and the mechanism-by-mechanism gap map.

## 0f. The Stage-5.2 claims fold (2026-10-05, this session)

**The values-only claims list is gone.** The compact and Sound memory
arguments now terminate their legs in the claims fold
(`lattice-zkvm/src/claimsfold.rs`, ~700 lines): the expect identities
become deferred monomials over the claim slots (affine leaves absorb
`digit_affine`'s `α = 2ρ−1, β = 1−ρ` and the `inc − INC_OFFSET` shift),
and a GKR-style product tree of `s = ⌈log₂ max arity⌉` degree-3
sumchecks folds all the products into two leaf claims whose MLE collapse
is a linear form in the (never-transmitted) values — bound by the same
carriers, now with the fold-derived weights plus fresh ρ′ for the 18
pre-leg entries (the address / read-value claimed-sum inputs, the only
values still in the clear).

Measured (`cargo run -p lattice-zkvm --example claims_probe
--release`):

| program | fold layers | fold rounds | **fold bytes** | total proof |
|---|---|---|---|---|
| fib-test (64 steps, k=4) | 3 | 24 | **1 024 B** | 17.1 KB |
| fib-bench (fibonacci(18)) | 3 | 24 | **1 024 B** | 22.5 KB |
| fib-40 (fibonacci(40)) | 3 | 24 | **1 024 B** | 32.2 KB |

The claims component: **2.9–3.5 KB → 1.02 KB** (the SOTA ledger's
mechanism-#4 target: "claims 3.5 KB → ~1 KB"). The end-to-end
batched-compact proof: **33.0 → 31.0 KB** at the 185-cycle fibonacci
(`zkvm-membench`). The fold's size is shape-flat (log in the check
count): the same 1 024 B at fib-test, fib(18) and fib(40), while the
clear-list term it replaces grows linearly with the claim count.

Tamper coverage (pinned by the test suite): a wrong pre-leg value, a
wrong layer round message, a wrong leaf-claim pair, and a wrong S value
all fail closed; the existing carrier/opening/commitment tamper pins
carry over unchanged.

This container's timings (the membench table's prove/verify columns are
the Clear mode's, per the harness's convention): the compact mode
itself — 1 173 ms prove / 136 ms verify at the 185-cycle fibonacci
(the fold adds ~90 ms of layer sumchecks over ~330 recorded slots;
verification's fold cost is ~20 ms — the two eq-table walks at the
leaf binding).

## 1. What is measured

For every guest program: **cycles** (executor steps), and — for programs
inside the dense prover's kernel-scale envelope — the full
**memory-argument proof**: 9 Twist & Shout instances (4 register limbs, 4
RAM limbs, 1 fetch Shout), 13 sumcheck legs each, ~120 legs total, with
virtual-`Val` Val-evaluation, matrix-evaluation sumchecks for every
virtual one-hot claim, and two Ajtai bundles authenticating every base
claim. **Verification never re-executes the program**: the verifier
recomputes only public tables (the program image, the initial memory
image) and O(λ) field work.

## 2. Results (release build, this container)

| program | cycles | prove (ms) | verify (ms) | proof (KB) | proved |
|---|---|---|---|---|---|
| fibonacci | 185 | 2 539 | 530 | 3 640 | yes |
| muldiv | 392 | 6 560 | 1 047 | 7 212 | yes |
| regex | 451 | 7 448 | 1 067 | 7 218 | yes |
| memory_ops | 2 831 | — | — | — | no (window > 256 words: dense prover cap) |
| modinv | 9 071 | — | — | — | no (cycle cap) |
| sorting | 12 648 | — | — | — | no (cycle cap) |
| matrix_mul | 12 154 | — | — | — | no (cycle cap) |
| collatz | 462 105 | — | — | — | no (cycle cap) |

The caps are the documented dense-prover posture: the twist legs
materialize `K × T_s` matrices prover-side, so the harness only proves
programs with `T ≤ 2^12` cycles **and** a RAM window ≤ 256 words. The
paper's sparse provers (T&S §6.3/§7 — "0s are free") remove the cap; they
are the Wave 8.5 follow-up.

## 2b. The compact mode (the 50 KB pipeline, 2026-09-30)

`prove_memory_argument_compact` replaces the Θ(N) digit-revealing
bundle openings with the folded amortized opening
(consolidated into this file's history): narrow byte packing (1 byte/coefficient — also
fixing the bits-bundle 31-bit vacuous-gate security defect), the
r-aligned column layout with per-column Ajtai commitments, the
scalar-challenge integer fold with the Goldilocks commuting functional,
the rANS-coded response, and the values-only claims list (points
re-derived by the verifier's leg replay).

| program | cycles | Clear mode | Compact mode | reduction | compact prove | compact verify |
|---|---|---|---|---|---|---|
| fibonacci | 185 | 3,640 KB | **75.5 KB** | 48× | 1,802 ms | 221 ms |
| regex | 451 | 7,218 KB | **108.4 KB** | 47× | 4,600 ms | 288 ms |
| muldiv | 392 | 7,212 KB | **~107 KB** | 48× | 4,831 ms | 295 ms |

Compact-mode composition (fibonacci): legs 55.0 KB (108 sumchecks —
the remaining term, see the roadmap), claims 3.5 KB (345 values-only),
column commitments 8.2 KB, carriers 0.8 KB, compact openings 9.3 KB,
statement 0.5 KB. The multi-megabyte digit tables — 95% of the old
proof — are gone; the opening machinery is now 13% of the proof.

## 2b′. The Stage-4 leg batching + the estimator-hardened fold (2026-09-30, this session)

**The 50 KB target is met**: `legbatch.rs` re-structures the ~117
per-instance legs into 12 staged, dependency-ordered batched sumchecks
(T&S §4.2.1 random-power RLC via `prove_batch`; the transmitted
per-leg evaluation claims became the batches' claimed-sum vectors,
pinned by their own stages' terminal identities). The fold's interim
hardening (`k = 4`, amplitude `2^6` — the estimator-run MSIS table's
verdict, SECURITY.md) doubles the column commitments and tightens the
relaxed bound 64×.

| program | cycles | Clear mode | Batched compact | reduction | compact prove | compact verify |
|---|---|---|---|---|---|---|
| fibonacci | 185 | 3,640 KB | **33.0 KB** (pre-fold; **31.0 KB** with the Stage-5.2 claims fold, §0f) | **117×** | 1,429 ms | 196 ms |

Batched-compact composition (fibonacci): legs ~6 KB (12 batched
sumchecks — was 55 KB / 108 legs), claims 3.5 KB (pre-fold — now the
1.0 KB Stage-5.2 fold, §0f), column commitments
~16 KB (the k=4 hardening), carriers 0.8 KB, compact openings ~6 KB,
statement 0.5 KB. The per-cycle prover cost includes the ledger's new
`fix_last_variables` tail cache (the digit-row claim pattern's
`(log_k+1)×` resolution win).

**The honest security caveat** (the estimator's verdict,
`SECURITY.md`): the single-level fold's MSIS binding at `k = 2` was
`~2^12` bits at every response length; the shipped `k = 4` interim
lifts the short-response regime but the sound 128-bit posture at the
benchmark response lengths requires the second-level fold (Stage 5.2,
the LaBRADOR decider) — the estimator table
(`lattice-sis-estimator/examples/fold_security_table.rs`) maps the
knob levers and the sound `n̄ ∈ {2, 4}` regime.

## 2c. The byte-guest ISA + the virtual-Val route (2026-10-06, this session)

Two bottlenecks from the throughput sprint's post-mortem landed:

**The byte-guest ISA** (`lattice-vm` + `lattice-zkvm` + `lattice-guest`):
the full sub-word load/store surface — LB/LBU/LH/LHU/SB/SH — at every
byte alignment (including straddling halfwords). The memory argument is
UNCHANGED (word-granular RAM; the sub-word access is a read-modify-write
splice on the containing word — the extraction/merge rides the
mem_old/mem_new tensors and the trace columns). Conformance: the
differential harness (120 random programs) now generates the sub-word
surface at all alignments; a dedicated all-offsets test pins LB/LBU/LH/
LHU/SB/SH semantics against ground truth. The new `byte_ops` benchmark
guest (byte-reverse + byte-sum + halfword swap-XOR, 1,284 cycles) runs
in the suite and proves end-to-end through BOTH the compact and
block-commit modes. This unlocks the byte-oriented workload family
(SHA-256, LEB128, CRC) that previously forced shift-and-mask whole-word
workarounds.

**The virtual-Val route** (`lattice-memory/sparse_engine.rs`): the
O(K·T) materialized Val matrix — the container-scale memory cap (128 MB
of field elements at K = T = 2^12; GB-scale beyond) — is REPLACED by the
O(K + T) write-event spec. The `VirtualValSpec`/`VirtualValState` factor
computes the Val MLE's round partials from (init, the write events):
the k-rounds via address-bucketed integer comparisons (LT at Boolean
points), the j-rounds via the LT-extension at the mixed points (the
paper's own pairwise cost profile — the dispatch heuristic
`pairwise_cost` keeps the materialized route when the address traffic is
hot relative to K, i.e. the register instances). The V0/V1 legs'
point-evaluations (`memory.rs`) also moved to the Eq-11 stream identity
(`Val(r_a, r_c) = init̃(r_a) + Σ Inc̃(r_a, j')·LT̃(j', r_c)`) — the
materialization is gone from the claim side too. **The proofs are
byte-identical to the materialized route** (pinned by
`twist_ports_virtual_matches_materialized_exactly`); the container-scale
test proves + verifies at K = T = 2^12 without the matrix (8.1 s in the
DEBUG build; the same workload would allocate 128 MB + O(K·T) round work
on the materialized route). The live `pipeline2` RAM twists dispatch to
the virtual route when `pairwise_cost·8 < K·T`.

## 2d. The block-commit batched opening (2026-10-06, this session)

The "~300 per-column Ajtai commitments dominate the prover" bottleneck:
the compact mode's commitment layer is r·k ring elements (16 KB of the
33 KB fibonacci proof at r=64 — the single largest component, growing
linearly in the column count). `blockfold.rs` replaces it with **ONE
packed block commitment per bundle** — the wide seeded key
F ∈ R^{k×m} over the whole column universe, ONE k-vector transmitted
(1 KB) — plus the **fused binding sumcheck** (the Akita Eq-160-style
"outer layer" the research consensus flagged as our documented gap): a
degree-2 Z_q sumcheck over the m-entry cube proving
`⟨β, Σ_p c_p·w_p⟩ = ⟨β, c_y + ⟨h, v⟩⟩` whose g-half pins the witness
to the ONE commitment and whose h-half pins the transmitted response v.
The compact machinery (carrier, ũ_j's, interpolation, fold challenges,
rANS response, norm gate, functional commute) is unchanged.

**The 2026-10-06 hardening update** (the estimator run + the streamed
keys, SECURITY.md's block-geometry table): the block-geometry MSIS
verdict ships `SECURITY_K = 16` (the conservative composed-gate reading
at 128 classical bits; k=8 already carries the byte-bound homogeneous
argument at 253 bits) and the verifier's key passes are STREAMED
(`streamed_c_array`: per-column seed regeneration, g and the public
C-array fused in one pass, O(m + n̄ + k) state — bit-identical, the
dedicated test pins it).

| program | cycles | Batched compact | **Block-commit (k=16)** | reduction | block prove | block verify |
|---|---|---|---|---|---|---|
| fibonacci | 185 | 31.0 KB | **25.0 KB** | **1.2×** | 1,477 ms | 240 ms |

The honest price of the 128-bit bar: the k=4 interim (98 bits in the
byte-bound regime — insecure posture) measured 19.0 KB; the
SECURITY_K=16 posture costs 6 KB of k-vector commitments (4.1 KB per
bundle) and buys the estimator-certified binding. The streamed keys
hold the verifier at 240 ms at 4× the rank (the materialized route's
memory would have scaled with k; the stream's does not). The win GROWS
with the column count: at richer workload scale (r → 256+ columns) the
compact layer grows to 64+ KB per bundle while the block layer stays at
4 KB — the estimator-certified posture at r× amortization. The
block-commit layer at k=16 is 4× smaller than the per-column layer
would be at the SAME security rank (r·k = 64 ring elements at r=4).


## 2c. The streaming / client-side prover (the small-space pipeline, 2026-09-30)

Papers: ePrint 2025/611 (Proving CPU Executions in Small Space) and
ePrint 2026/762 (the monomial-basis sum-check). Harness: `cargo run
--release -p lattice-streaming --example stream_bench` and `cargo run
--release -p lattice-projsumcheck --example proj_bench`.

### The projective (monomial-basis) sum-check — `lattice-projsumcheck`

| kernel | Boolean baseline | projective | speedup |
|---|---|---|---|
| binding, 2^20 coefficients | 1.19 ms | 0.91 ms | 1.31× |
| eq full-domain table, n=20 | 4.64 ms | 2.70 ms | 1.72× |
| degree-2 sum-check, n=20 | 36.0 ms / 480 B | 24.7 ms / 320 B | 1.46× / 1.50× smaller |
| degree-2 × eq, n=20 | 64.1 ms / 640 B | 48.6 ms / 480 B | 1.32× / 1.33× |
| degree-2, n=22 | 170 ms | 123 ms | 1.38× |
| Fp256 (BN254 Fr) chained mul ×10^6 | 27.5 ms | 19.0 ms | 1.45× |
| Fp256 projective binding 2^20 | 29.5 ms | 24.9 ms | 1.19× |

The proof-size column is structural: the projective message set
`{s(∞), s(1..d−1)}` carries one element fewer per round (the verifier
derives `s(0)` from the round identity `s(0) + s(∞) = C`). On
BN254-shaped fields the paper measures 1.92× for the upper-limb
challenge multiplication; our scalar-fallback CIOS keeps 1.45×.

### The streaming prover — `lattice-streaming`

| prover | time | space |
|---|---|---|
| sum-check n=16, fully streamed (Algorithm 1) | 0.15 s | O(n + ℓ²) beyond the data |
| sum-check n=16, hybrid @ 2 MiB | 0.003 s | 1.0 MiB metered peak |
| sum-check n=20, fully streamed | 3.13 s | 7.8 MiB ΔVmHWM (16 MiB data) |
| sum-check n=20, hybrid @ 2 MiB | 0.94 s | 2.0 MiB metered peak |
| prefix-suffix inner product n=20 (pcnext) | 0.53 s | O(√N) tables |
| grand product n=20, DFS | 0.013 s | O(n) stack (≤ 21 entries) |
| grand product n=20, + Quarks proof | 0.11 s | recorded tables |
| matrix-layout streaming commitment n=20 | 22.8 s | O(√N), one pass |

Round messages of the streamed/hybrid/prefix-suffix provers are
**bit-identical** to the in-memory engine's (tested), so the streaming
paths are drop-in prover strategies, not protocol variants. The end-to-end
`prove_program_streaming` composes the pcnext prefix-suffix sum-check,
the streaming witness commitment, and the memory-fingerprint grand
products over one VM execution.

## 2d. The 2026/2146 + 2025/1117+2026/587 waves (2026-10-01, this session)

**lattice-ttrp** (`examples/ttrp_bench.rs`, release, this container):

| instance | m̄r | φ | d | µ₁+µ₂ | c | k | prove | verify (tensor) | naive row pass | proof |
|---|---|---|---|---|---|---|---|---|---|---|
| small | 2^8 | 16 | 4 | 6 | 4 | 16 | 5 ms | 2 ms | 1 ms | 1.8 KB |
| mid | 2^12 | 64 | 4 | 9 | 8 | 32 | 1.5 s | 79 ms | 233 ms | 10.6 KB |
| large | 2^14 | 64 | 4 | 10 | 8 | 48 | 4.6 s | 139 ms | 1405 ms | 12.2 KB |

The verifier's tensor-structured MLE evaluation (Lemma 6: core-MLE
chunks → boundary chain → γ̃-scaled coefficient chains) beats the
JL-style row materialisation **10× at m̄r = 2^14** — the paper's Table-1
axis (6492 MB → sub-MB core tensors). The prover runs the honest
O(k·c·m̄r·φ) integer-contraction + S/W-split cost of Lemma 6.

**lattice-sumcheck fast prover** (`examples/fastprover_bench.rs`): the
multiplication-count instrumentation confirms the papers' asymptotics
exactly — at d = 2, M = 2^14, the total big-by-big count drops
65 528 → 3 153 as the window grows 1→4 (the 2^v tail factor) while the
small-by-big count grows `M·((d+2)/2)^v` per C.4.1 — but the wall-clock
trade is negative on Goldilocks: a 64-bit field has κ ≈ 1 (no limb
hierarchy), so the window's sb work costs the same as bb and the
baseline's SIMD 8-lane kernels win. The regime the papers target
(κ ≈ 33 for 256-bit Montgomery fields) is realised by
`lattice-projsumcheck`'s Fp256 — the port target for the real
2.5–4× small-value and 1.7–2.2× high-degree wins. What transfers to
Goldilocks today: the multiproduct engine's bb-count wins for
high-degree products, the split-eq memory win (no 2^ℓ eq
materialisation), and byte-identical drop-in transcripts.

## 2e. The Fp256 window + streaming-schedule + TTRP-ledger + Neo waves (2026-10-01, this session)

**The Fp256 window fast prover** (`lattice-projsumcheck/examples/fastprover_bench`,
release build): the measured kernel ratios on this container —
bb (full CIOS) 27 ns, sb (zero-limb-skipped CIOS, `mul_small`) 11.8 ns
(**2.7×** vs bb), ss (native `i128`) sub-ns — against the papers'
`κ ≈ 2N²+N = 36` at N = 4 limbs. The honest finding: the window's
*weighting* multiplications are **sb-class** (big Montgomery weights ×
small grid sums), not the ss-class the Lemma-5 optimum assumes, so the
measured optimum collapses to `v* ≈ 2` and digit-table instances
(`d = 2..3`, `M = 2^{13..14}`) come out **break-even** vs the
linear-time baseline (byte-identity holds at every window). The grid
construction itself is pure `i128` and essentially free — confirming
that half of the papers' claim; the 2.5–4× end-to-end needs ss-class
weighting (small challenges or Appendix C.2's grid-based binding),
documented as the follow-up in
`papers/implemented/sumcheck-speedups-fp256.md`. The port also
**found and fixed a pre-existing composite-modulus defect** in
`fp256.rs` (limb 2 mistyped `…58d2` for `…585d` — every local test had
passed against the wrong constant).

**The streaming window schedule** (`lattice-streaming/window_schedule.rs`):
the Figure-2 `EvalProductStream_{k},SC` — geometric-then-capped windows
with the bound tables *emulated* by eq-folds (never materialized), one
pass-equivalent of `Θ(d·M)` oracle work per window, peak storage
grid+scratch (`(d+2)^{⌊ℓ/(kδ)⌋} ≈ M^{1/k}`; the `space_profile` test
pins peak < M/8 at `ℓ = 12, k = 2`), bit-identical round messages to
the in-memory engine across 6 shape/k combinations.

**The TTRP norm-check module** (`lattice-zkvm/norm_check.rs`): the
ledger's bundle openings gain the `Π_TTRP` shortness layer — the
statement digest binds (commitment, shape, bound, carrier terminal),
the eval claim `mle(v)(conj(r)) = w_r` cross-checks the reconstructed
response (the soundness glue the JL/digit-gadget layer never had), and
the ABDLOP `LinearRelation` adapter is exposed for constraint
absorption. Tamper tests: digits, TTRP proof, wrong context, eval
mismatch all rejected.

**Neo** (`lattice-folding/neo.rs`): the pay-per-bit cost profile
measured — a binary witness commits with ≥ 4× fewer column-additions
than a full-width one, both bit-identical to the naive reference; the
strong sampling set's expansion factor stays within Theorem 3's
`2·φ(η)·max‖ρ‖∞`; the commit → `Π_RLC` → `Π_DEC` → decider round trip
and both tamper paths are test-pinned.
## 2d. The ring-lookup layer (ePrint 2026/471, this session)

The lookup layer rebuilt over the CRT-split ring
(`lattice-lookup-ring`), including the three follow-ups of
`docs/papers/implemented/ring-lookups.md` (release build, this
container; `cargo run -p lattice-lookup-ring --example lookup_bench`):

| Benchmark | Time |
|---|---|
| Ring-LogUp prove+verify M=4 N=4 (d=8) | 0.9 ms |
| Ring-LogUp prove+verify M=8 N=8 | 1.7 ms |
| Ring-LogUp prove+verify M=16 N=16 | 3.1 ms |
| Ring-Plookup prove+verify M=4 N=4 | 1.6 ms |
| Ring-Plookup prove+verify M=8 N=8 | 2.9 ms |
| Windowed binding pass (ring carrier) N=8 / 32 / 128 | 0.2 / 0.8 / 3.2 ms |
| Fp256 binding pass (BN254, CIOS grid) N=8 / 32 / 128 | 1.3 / 4.6 / 18.2 ms |
| Compiled Ring-LogUp (commitments + binding passes) M=N=4 | 2.2 ms |
| RAM batch verification (Section 6) M=4 k=8 | 9.5 ms |
| RAM batch verification M=8 k=16 | 17.8 ms |

Reading: the PIOPs run at the paper's `O(N + poly(d)·M)` prover shape
with the schoolbook split-ring kernel (`O(d²)` per product — the NTT
is incompatible with Lemma 5.8's `q ≡ 5 mod 8` two-component split);
the binding passes scale linearly in the slot count
(`N·K_w` digit windows per oracle); the Fp256 port runs the same
rounds at ~6x the ring carrier's cost — the CIOS grid's per-limb work
against the schoolbook u32 kernel, with the upper-limb short-circuit
already active. The v3 zkVM pipeline (the ring-lookup memory layer)
proves the 6-step demo program end-to-end in ~0.4 s wall (the four
bridge instances' RAM batch verifications dominate; the v2
Twist & Shout path on the same trace is ~30 ms — the honest
trade-off of the lookup approach at demonstration scale: unstructured
tables and ring compatibility against the field-tuned grand products).

## 2f. The PCD papers wave (ePrint 2026/289 + 2026/538, 2026-10-02, this session)

Two new crates benchmark the PCD layer over BN254 Fr/G1 with vector
Pedersen commitments (bucket MSM) — `cargo run --release -p lattice-pcd
--example pcd_bench` and `cargo run --release -p lattice-holo --example
holo_bench`.

**ZK-PCD from accumulation schemes (2026/289)** — the zk-Protogalaxy
accumulation (masking vector + eq-interpolated F(X) over the party
layout + the masked batched sum-check + the error commitment):

| relation | size | prove (ms) | verify (ms) | decide (ms) |
|---|---|---|---|---|
| R1CS (d=2, µ=1) | s=2 t=6 rows=4 | 5.7 | 1.3 | 1.9 |
| R1CS | s=4 t=12 rows=8 | 18.4 | 3.8 | 7.4 |
| R1CS | s=8 t=24 rows=16 | 26.7 | 4.1 | 11.1 |
| CCS (d=3) | rows=4 t_M=3 | 6.4 | — | — |
| CCS (d=6) | rows=8 t_M=5 | 14.8 | — | — |
| permutation (d=n, µ=2) | n=4/8/12 | 4.2 / 6.0 / 8.0 | — | — |
| ZK-PCD chain (arity 2) | depth 2/4/8 | 39 / 78 / 157 | ~12 (final) | — |

**PCD via holography accumulation (2026/538)** — the GBF protocols in
both representations, Barebones (the SuperSpartan/SuperMarlin
recovery), the holography fold, the decider, and the PCD chain:

| protocol | configuration | prove (ms) | verify (ms) |
|---|---|---|---|
| Π_GBF2 / Π_GBF1 | mv n=4 | 1.5 / 1.8 | — |
| Π_GBF2 / Π_GBF1 | mv n=8 | 5.9 / 7.3 | — |
| Π_GBF2 / Π_GBF1 | uv n=4 | 7.9 / 13.4 | — |
| Π_GBF2 / Π_GBF1 | uv n=8 | 32.0 / 49.9 | — |
| Barebones | mv n=8 | 13.3 | 4.7 |
| Barebones | uv n=8 | 34.5 | 5.3 |
| Π_Fold (K=2) | mv n=8 | 23.5 | — |
| Decider (ℓ=1) | mv n=8 | 59.8 | — |
| PCD chain | depth 2 (mv n=8) | 48.9 | 75.7 (steps+decider) |
| PCD chain | depth 4 (mv n=8) | 140.1 | 101.5 |

The univariate instantiation's higher constant is the O(n²)
Lagrange-to-monomial conversion and the 2n-node interpolation of the
h₁/h₂ decomposition — the paper's own trade-off table (univariate wins
only with a commitment whose evaluation proofs are O(1), e.g. KZG;
with the linear-opening PC here the multivariate path is cheaper).


## 2g. The Accordion + CauchyFold + PQ-follow-up wave (2026-10-02, this session)

**lattice-accordion** (ePrint 2025/1325 over the Ajtai module, `q = 2^50−2687`,
16-bit digit layers; `examples/accordion_bench.rs`, release):

| shape | reduce | verify | decide (4-fold, amortized) | proof |
|---|---|---|---|---|
| k=4, N=64, rows=1 | 0.22 ms | 0.07 ms | 0.35 ms | 9.2 KB |
| k=6, N=256 | 0.55 ms | 0.10 ms | 0.88 ms | 12.3 KB |
| k=8, N=1024 | 1.99 ms | 0.12 ms | 3.30 ms | 15.4 KB |
| k=10, N=4096 | 7.78 ms | 0.14 ms | 12.91 ms | 18.4 KB |
| k=8, N=1024, rows=2 | 3.49 ms | 0.21 ms | 5.94 ms | 30.7 KB |

The proof is exactly `3·(k+κ)` module points + one scalar (the paper's
communication), verify is `O(m)`, and the amortized decide is `O(N)`
ring-scalar operations once per batch — the Halo amortization preserved.

**lattice-cauchyfold** (ePrint 2026/2011 at the scaled profile,
`q = 2^48−59`, `K = Fq4`; `examples/cauchyfold_bench.rs`, release):

| k | carrier (direct) | carrier (fast) | boundary | node prove | node verify |
|---|---|---|---|---|---|
| 2 | 0.02 ms | 0.04 ms | 0.10 ms | 58.1 ms | 6.7 ms |
| 4 | 0.03 ms | 0.10 ms | 0.25 ms | 107.2 ms | 14.5 ms |
| 8 | 0.20 ms | 0.40 ms | 1.32 ms | 400.9 ms | 66.6 ms |
| 16 | 2.04 ms | 1.84 ms | — | 1600.4 ms | 248.0 ms |

The boundary column confirms `dim Va = k` (Corollary 4.3) by exact
K-linear algebra; the fast carrier agrees with the direct form at every
arity. The paper's own k=16 profiles (127,887 / 129,002 B wires,
9.4 ks pipelines) are recorded declaratively in `params.rs` — not
executed (57.5M-coefficient witnesses).

**The PQ follow-ups** (the deviation-ledger items):
`lattice-pcd::{ajtai_fr, pq}` — the Ajtai-over-`F_r` commitment layer
(the digit regime at radius `2^16−1`, the E-fold homomorphic closure,
the norm ledger, the decider, the double-open MSIS kernel) and
`lattice-holo::pc_short` — the Accordion module-sumcheck on `Fp256`
(`O(log n)` openings, the γ-accumulation, the amortized decider).

## 2h. The LatticeBlindFold wave (ePrint 2026/1857, 2026-10-03, this session)

The blinding layer for the folding stack — the first lattice-based
NovaBlindFold analogue. End-to-end timings of one full Π_LBF folding step
(Protocol 12: the ι_bl precomposition + Π'_R1CS + Π'_RLC + Π'_DEC, with
all the ABDLOP PoKs and their rejection-sampling loops), release profile:

| profile | prove | verify | commitments | masked openings |
|---------|-------|--------|-------------|-----------------|
| toy (d=4, nf=2^6, k=17) | 159 ms | 8.4 ms | 167 ABDLOP + 18 Ajtai | ~23k ring elems |
| medium (d=8, nf=2^8, k=20) | 1.10 s | 27 ms | 194 ABDLOP + 21 Ajtai | ~70k ring elems |

The k = Θ(log n_F) decomposition depth (17/20 at the toy scales) is the
paper's own price of blinding (Remark 4.18: SuperNeo at its native k=Θ(1)
is a log-factor cheaper — the gap the paper leaves as "the main open
engineering problem"). At the paper's parameters (n_F = 2^21, d = 128,
k = 31, ξ ≈ 20) §4.3.3.1 reports ≈ 8.2 MB of communication per folding
step (4.57 MB masked openings over 246 blocks, 1.98 MB output ABDLOP
commitments, 0.47 MB compact Ajtai) — the crate's budget calculator
reproduces the supporting security rows (≈ 121 bits interactive, binding
term ε_SC = 2^−121.6, blinding cap 2^−112).

The honest toy-scale caveat: |C| = 4^d gives no challenge-space security
at d = 4/8 (the budget prints −3/+5 interactive bits — recorded, not
hidden); the blinding machinery itself (the Rej1 distribution-flattening
test, the perfect Sum-Check masking, the S_ABDLOP simulator's accepting
transcripts) is exercised and verified statistically regardless.

## 2i. The semantics-stage factoring wave: verify-side carrier + the sparse shift route (2026-10-04, this session)

The instruction-semantics stage (`semantics.rs`, wired by the
family-completion wave) benchmarked with a phase-attributed harness
(`lattice-bench --bin semantics-bench`, the scalable mixed-instruction
loop). The pre-wave profile at log_t=6 (52 cycles): prove 14,696 ms —
of which the grouped-carrier openings 14,264 ms (97%) — verify 108 ms;
the 30,826-claim list dominated both time and proof size.

Four fixes landed (each pinned by differential tests):

1. **The prefix-factored carrier eq build** (`ledger.rs::rec_eq_acc`):
   the flat points' leading coordinates are BOOLEAN (the layout's
   head/slice bits and the bit-row heads), so claims ROUTE exactly
   through the trie levels and only the field-valued tails need dense
   per-group eq tables. The produced array is bit-identical (field
   addition commutes) — transcripts byte-identical — while the work
   drops from `claims x 2^log_flat` to `sum_groups claims_g x 2^{field
   vars} + 2^log_flat`: openings 14,264 ms -> 15 ms at log_t=6 (~10^3x).
2. **The seeded-matrix derivation fix** (`lattice-ring`): the
   per-coefficient counter-indexed XOF re-squeeze collapsed to ONE
   prefix-consistent squeeze per element (byte-identical derivation,
   test-pinned) — AND the rejection slack corrected for the actual
   modulus (q = 3·2^30+1 rejects a QUARTER of u32 candidates, so the
   slack-8 walk fell short on every element and silently fell back into
   the old loop): from_seed 21.8 -> 4.6 us/element. The verifier's
   bundle-key regeneration (the pk-derive) was the verify-side carrier
   cost: 73 ms -> 4 ms at log_t=6.
3. **The sparse-engine shift route** (`constraints.rs`): the shift
   family — the O(64^2)-per-class MUX term expansion, 74% of the
   families' prover time — now routes through the sparse engine
   ("0s are free"): the one-hot gates ride as single sparse factors
   with their OWN supports, the selectors as dense factors. The
   emitted proof is byte-identical to the dense engine over the same
   virtual polynomial (differential test pinned); the verifier is
   UNCHANGED. Shift 8,234 -> ~4,400 ms at log_t=12, and the claim list
   30,826 -> 9,066 (3.4x — the sparse route records each one-hot once).
   THE ENGINE-LEVEL FINDING (new differential test in
   `sparse_engine.rs::identity_differential`): a term with TWO sparse
   factors CANNOT use intersection-filtered supports — the multilinear
   products have suffix-level cross terms outside the boolean
   intersection (round polynomials at t >= 2 sample the extensions;
   t in {0,1} still agree, so the sum checks pass while the rounds
   diverge). Correct: one sparse factor per term (own support), or
   union-aligned zero-padded entries.
4. **The ctrl carry-gate completeness fix** (`constraints.rs`): the
   prover's next-pc MUX leaked the limb-0 OUT carry into the l=0
   identity via `l - 1.min(l)` — any taken branch/jal with a target
   crossing the 16-bit limb boundary (negative offsets!) failed
   ClaimMismatch. The verifier was already correct; the prover now
   gates the in-carries by l > 0 (fail-closed coverage retained).

| log_t | cycles | families | openings | prove | pk-derive | verify | ms/cycle | claims |
|---|---|---|---|---|---|---|---|---|
| 6 | 52 | 88 ms | 15 ms | 119 ms | 4 ms | 23 ms | 2.3 | 9,066 |
| 8 | 244 | 327 ms | 56 ms | 430 ms | 16 ms | 53 ms | 1.8 | 9,066 |
| 10 | 1,012 | 1.22 s | 200 ms | 1.57 s | 64 ms | 172 ms | 1.5 | 9,066 |
| 12 | 4,084 | 5.46 s | 846 ms | 7.04 s | 252 ms | **725 ms** | 1.7 | 9,066 |

**Verify is sub-second at the benchmark scale** (the pre-wave verify at
log_t=12 extrapolates to ~9.4 s: the pk-derive alone ~4.7 s + the
naive claim resolution). The proof size is claim-list dominated
(~1.1 KB/claim with points) — 9,066 claims ≈ 10 MB of the 34 MB proof
at log_t=12; the values-only claim compression (the compact mode's
discipline) is the natural next step.

**The honest remaining ledger**: the shift family is still ~4.4 s of
the log_t=12 prove — the term-count wall (25k per-(bit, shamt) MUX
terms, each with per-round fixed overhead regardless of support size).
The structural fix is the 2D (cycle x shamt) convolution sumcheck with
the bit-axis rand-checked — a verifier-visible restructure (the
paper-route; the sparse engine's home turf). The other moderate
families (bool-cols 277 ms, route 189 ms, cmp 185 ms, sel 178 ms) share
the selector-gated shape and would benefit from the same route.

## 2j. The LaBRADOR decider wave: the width fold + the Sound memory argument (2026-10-04, this session)

The 50KB pipeline's Stage 5.2 completion — the quadratic-garbage
width-reducing fold (`lattice-zkvm/src/width_fold.rs`) wired through
the Sound compact profile (`compact.rs::CompactProfile::Sound`) into
the live memory-argument path (`memproof.rs::
prove_memory_argument_sound`). The binding of the WHOLE compact opening
becomes the width fold's `[A₂ | −T]` MSIS instance at the estimator's
sound row — the level-1 `[F̄ | −y]` instance (the Stage 5.1 broken
regime at benchmark response lengths) never arises: the level-1
response is never transmitted, the `y_j` enter only through the public
target `t = Σ_j d_j·y_j` which `(W0)` pins.

The estimator evidence (`lattice-sis-estimator --example
width_fold_table` — the table now in the repo):

| β₁ (the level-1 gate) | w | r₂ | κ | A₂ | classical | quantum | garbage |
|---|---|---|---|---|---|---|---|
| 2^15 (r₁=2) | 8 | 2 | 8 | 2^4 | 1,855 | 1,838 | 32 elems |
| 2^16 (r₁=4) | 8 | 2 | 8 | 2^4 | 1,231 | 1,214 | 32 elems |
| 2^17 (r₁=8) | 8 | 2 | 8 | 2^4 | 648.6 | 631.3 | 32 elems |
| 2^20 (re-packed) | 8 | 2 | 8 | 2^2 | 206.7 | 189.4 | 8 elems |
| 2^17 (broken row) | 4 | 4 | 4 | 2^6 | 11.7 | — | (ruled out) |

The Sound profile ships `(w, r₂, κ, A₂) = (8, 2, 8, 2^2–2^4)` with the
level-1 amplitude `A₁ = 2^4` (`FoldParams::sound`) and the column
count that lands the response at `n̄ ≤ 16`
(`sound_fold_params_for`) — fail-closed beyond the ceilings (the
honest boundary: the recursive width-fold staging and the Modulus-50
class are the documented follow-ups).

The measured honest price at the test-scale memory argument
(fibonacci, 16 cycles — `sound_memproof_honest_and_tamper`):

| Mode | total proof | binding |
|---|---|---|
| Clear (single-level fold) | ~33 KB | `[F̄ \| −y]` — the Stage 5.1 broken regime |
| Compact (single-level fold) | ~60 KB | `[F̄ \| −y]` — the broken regime |
| **Sound (the width fold)** | **129.7 KB** | `[A₂ \| −T]` — **329+ classical bits, estimator-gated** |

The ~2.2× honest multiple is the quadratic garbage's price
(`r₂·(r₂−1)·k` ring elements + `r₂·κ` inner commitments + the
functional layer) at the small trace; at benchmark response lengths
the ratio inverts (the garbage stays constant while the replaced
response grows with `n̄`). The width fold's unit tests (11: honest
roundtrip, tampered z/inner/images/garbage/functional/target, the
profile gate's broken shapes, the padding path, the sound search's
fail-closed floors) and the Sound opening's tamper suite (wrong claim,
wrong point, tampered ũ/z/garbage/images/commitment, wrong seed) all
pin the `(W0)`–`(W4)` checks.

**The honest remaining ledger**: the single-stage sound coverage ends
at the streams where `n̄ ≤ 16` (the β₁ ceiling 2^20 at `A₁ = 2^4`); the
benchmark-scale streams need either the recursive staging
(log-stages of the cheap `(8, 2, 8, 2^2)` row — each stage's output
feeds the next) or the Modulus-50 class (the 50-bit modulus doubles the
β headroom). Both are the documented Stage-5 follow-ups; the profile
gate fail-closes in between rather than shipping a broken binding.


## 2k. The recursive staging + SALSA D4 + the compact terminal (2026-10-04, this session)

Three landings on the soundness/size frontier:

**(1) The recursive width-collapse staging** (`lattice-widthfold::chain`
— the extracted shared fold core, NEW crate): the Sound opening and the
Sound memory argument now fold through log-stages of estimator-sound
rows instead of the single stage (the `n̄ ≤ 16` boundary). The
fail-closed posture: every stage's `[A₂ | −T]` instance ≥ 128 classical
bits + the 32-bit replay-grinding allowance; the per-stage gates grow
geometrically (β_{ℓ+1} = r₂·A₂·β_ℓ) under the q/2 completeness cap.

The measured coverage boundary (`cargo run -p lattice-widthfold
--example chain_coverage`):

| β₁ (the level-1 gate) | max staged n̄ | stream ceiling (r₁=128) |
|---|---|---|
| 2^8 (the byte gate) | 4,096 | ~33.6 MB |
| 2^15 (r₁=8) | 2,048 | ~16.8 MB |
| 2^17 (r₁=32) | 512 | ~4.2 MB |
| 2^19 (r₁=128, the packing cap) | 128 | ~1.0 MB |

vs the single-stage boundary: n̄ ≤ 16 → ~131 KB of stream. The
schedule at n̄=512/β₁=2^15: 6 stages of `(r₂=2, κ=16, A₂=1)` halvings
landing `(8, 2, 8)` — 1,184 transmitted ring elements, 19.0 grinding
bits (under the 32-bit allowance). Beyond the boundary the search
fails closed (the honest Q_32 ceiling: the Modulus-50 class is the
documented follow-up — the norm headroom, not the machinery, binds).

The Sound memory argument at the test scale with the chain (and the
cheaper sound rows the extended amplitude search finds):

| Mode | total proof | binding |
|---|---|---|
| Compact (single-level fold) | ~60 KB | `[F̄ \| −y]` — the broken regime |
| **Sound (the recursive chain)** | **55.2 KB** | per-stage `[A₂ \| −T]` ≥ 128+32 bits |

(the §2j single-stage Sound row was 129.7 KB — the chain + the
amplitude-extended sound-row search land BELOW the compact mode's size
at the test scale while carrying the estimator-gated binding.)

**(2) SALSA D4 — the Akita/zkVM response-layer swap**
(`lattice-akita::salsa_response::SalsaGroupedResponse`): the v2
pipeline's Stage-5 grouped openings replace the opened witness + the
digit-revealing NormProof with the byte-packed SALSAA chain — the
grouped RLC carrier + the ψ-functional carrier (the verifier-weighted
byte-recomposition bridge `Σ_c eq(r_sc, x(c))·2^{8b(c)}·z(c) =
f(r_sc)`, replacing D2's transmitted base) + D1's norm sumcheck with
the Lemma-4 gate. Measured (`cargo run -p lattice-akita --example
salsa_swap_size`, ring dim 16, the pipeline's pk shape):

| column values | Clear response | SALSAA response | reduction |
|---|---|---|---|
| 2^6 | 39,056 B | **640 B** | 61× |
| 2^8 | 155,840 B | **784 B** | 199× |
| 2^10 | 622,832 B | **928 B** | 671× |

The response grows only with log N (three sumchecks' rounds) — the
Θ(N) → polylog + disclosure-removal claim is measured, not asserted.
The honest regime note: the byte-packed D1 chain's Lemma-4 gate caps
the per-commitment capacity at ~1,200 values (the r-column split is
the scaling route); the byte-witness-to-commitment Ajtai binding
remains the documented outer-layer gap (the compact-mode fold is the
binding-complete route).

**(3) The Cyclo §7 bridge's compact-PCS terminal**
(`lattice-folding::cyclo_terminal` over `lattice-widthfold::ring_fold`):
the decider stops opening the witness — `(D1)` rides the fold's (W0)
(part images ≟ the commitment), `(D2)` the six linear claims ride the
EXACT ring-functional layer, `(D3)` the two prefix claims ride
projected functionals (θ_k of the fold's public sums). The terminal's
communication at the bridge's test scale (m=8): the fold's ~40 ring
elements (~10 KB) replacing the opened lift (8 elements, 2 KB) + the
binding — the honest price of the decider's witness-freedom at small
m; the win is the disclosure removal and the binding posture (the
[A₂ | −T] instance at the digit gate β₁ = k−1, estimator-gated).

## 2l. The extraction ledger + the D4 binding closure + the r-column split (2026-10-04, this session)

The three honest-ledger follow-ups of §2k, measured:

**(1) The multi-stage LaBRADOR extraction ledger**
(`lattice-widthfold::extraction` + `docs/analysis/
MULTISTAGE_EXTRACTION.md`): the degree-law unwind as an executable,
fail-closed artifact — now enforced inside `assert_sound_chain` at
prove AND verify time. The published table (`cargo run -p
lattice-widthfold --example extraction_table`): the six boundary
schedules' rewind trees (2–128 leaves), unwind degrees (2L = 2–14),
grinding ledgers (3.2–31.7 bits under the 32-bit allowance), the
per-stage kernel verdicts (min 161.5–841.4 classical bits), and the
unwind-norm slack (2^20.6 down to 2^6.6 — the honest Q_32 ceiling,
now measured per schedule by law E3).

**(2) The D4 binding closure** (`lattice-akita::salsa_binding::
prove/verify_grouped_salsa_bound` — the compact-fold composition):
the byte-witness↔commitment authenticated opening. Measured
(`cargo run -p lattice-akita --example salsa_bound_size`, ring dim
16, the pipeline's pk shape):

| values | Clear (B) | Salsa open (B) | Salsa BOUND (B) | bound/open | bound prove | bound verify |
|---|---|---|---|---|---|---|
| 2^6 | 39,056 | 640 | **5,523** | 8.6× | 90 ms | 8 ms |
| 2^8 | 155,840 | 784 | **24,962** | 31.8× | 945 ms | 60 ms |
| 2^10 | 622,832 | 928 | **54,759** | 59.0× | 3,790 ms | 186 ms |

The honest reading: the bound response is 6–11× under Clear at
8.6–59× the open mode's size — the price of closing the documented
outer-layer gap (the chain's transmitted fold material grows with the
stream; the open mode stays polylog but unbound). The closure itself
is pinned by the wrong-commitment tamper test (the (W0) rejection
the open mode lacked); the pipeline-level composition
(`Stage5Mode::Bound`, `pipeline2.rs`) verifies end-to-end.

**(3) The r-column capacity split** (same module): `byte_capacity`
states the Lemma-4 cap exactly — **2,048 values per commitment** at
ring dim 16/Q_32 (the §2k "~1,200" prose note was this same cap,
margin-rounded). Beyond it, the compact mode's discipline:

| values | columns | split response (B) | vs Clear | prove / verify |
|---|---|---|---|---|
| 2^12 | 2 | **133,541** | 18.7× | 8.5 s / 384 ms |
| 2^13 | 4 | **265,805** | 18.7× | 16.9 s / 761 ms |

The response grows as `r·(D1 + chain)` — the honest O(r) scaling
price at fixed modulus; the single-fold-over-columns composition (the
LaBinius secondary discipline) and the accumulator folding (the
Quasar/PCD route) are the documented size-recovery follow-ups.

## 3. Comparison with SOTA zkVMs (published numbers)

Context, not competition: LZX is a lattice-SIS research zkVM at kernel
scale; the systems below are production elliptic-curve/FRI zkVMs. The
comparison fixes the *shape* of the gap.

| System (published) | guest flavor | prover | verifier | proof | notes |
|---|---|---|---|---|---|
| **Jolt** (a16z, 2024) | sha2/keccak/ecdsa guests | ~0.9–1.5 s / 2^20 cycles | ~50–100 ms | ~100–200 KB | Spartan-style + sumchecks, GPU paths exist |
| **SP1** (S1, 2024) | sha2/rsaecdsa | ~2–6 s / 2^20-ish cycles | ~10–100 ms | ~100–300 KB | Plonkish + STARK folding, SP1 Pro network |
| **Risc0** (2024) | sha2/rsa | ~2–10 s / 2^20 cycles | ~50–200 ms | ~100–500 KB | FRI STARKs |
| **lzx memory argument (Clear)** (this) | arithmetic/DFA guests, ≤ 2^12 cycles | ~2.5–7.5 s / ≤ 512 cycles | ~0.5–1.1 s | ~3.6–7.2 MB | SIS commitments, Clear-response norm proofs |
| **lzx memory argument (compact)** (this) | arithmetic/DFA guests, ≤ 2^12 cycles | ~1.8–4.8 s | ~0.2–0.3 s | **75–108 KB** | folded amortized openings (LaBinius/LaBRADOR-lineage), values-only claims |

Honest reading of the gap:

1. **Per-cycle prover throughput** is the headline difference: ~15 ms/cycle
   here vs ~1–5 µs/cycle for SOTA. Three compounding causes, all
   identified: (a) the dense `K × T_s` matrix materialization (the sparse
   prover removes it); (b) the claim ledger resolves each base claim with
   a fresh `DenseMle::evaluate` (O(N·vars) per claim, thousands of claims —
   the fix_last_variables caching design is written in `ledger.rs`'s docs);
   (c) no SIMD (Wave 8.2) and no `lattice-rns` (8.3).
2. **Verifier time** (0.5–1.1 s) is dominated by the same naive claim
   resolution plus the `A·s` recomputation over the response — the SOTA
   zkVMs' 10–100 ms verifiers are the target once the ledger is cached.
   Note what the verifier is *not* doing: it never touches the interpreter
   — the pre-wave verifier re-executed the program.
3. **Proof size** (3.6–7.2 MB vs 0.1–0.5 MB) is dominated by the compact
   norm proofs' digit tables and the per-bit claim expansion (16 row
   claims per limb claim). The paper routes are the Akita fold driver
   (7.11 A5), LaBRADOR compaction (7.12 H6), and the SALSAA norm chain —
   all on the Wave 7 remainder ledger in `NEXT_STEPS.md` (the consolidated
   planning history).
4. **The constraint layer**: the memory argument proves the memory
   timeline over committed streams; the instruction-semantics families
   (ALU/decode/control routing) are the remaining P0-4 work — the SOTA
   systems prove full instruction semantics. This is the single most
   important remaining item on the ledger.

## 4. What the numbers do establish

- A complete, non-re-executing prove/verify loop over **real RISC-V
  programs** (not toy relations): witness → nine memory instances → ~120
  sumchecks → two Ajtai bundles → envelope → verification.
- The virtual-`Val` Twist & Shout stack works end-to-end at kernel scale
  with the stale-read soundness property test-pinned.
- The bit-packed bundle discipline (31 bits/coeff) holds the *committed*
  universe at O(T) values — the design that makes the sparse prover the
  next constant-factor step rather than a redesign.

## 3. The LaBRADOR + Greyhound papers wave (ePrint 2022/1341 + 2024/1293, 2026-10-02, this session)

`cargo run --release -p lattice-greyhound --example greyhound_bench` — the
full PCS pipeline (commit → eval → the LaBRADOR recursion → verify) at
feasible scales, the Table 4 analytic accounting, and the LaBRADOR sub-proof
at the real Table-4-derived statement sizes:

| Instance | prove | verify | total proof | levels |
|---|---|---|---|---|
| PCS, 256 ring elements (degree 16K) | 1.92 s | 1.45 s | **34.8 KB** | 2 (1 tail) |
| PCS, 1024 ring elements (degree 64K) | 2.89 s | 2.26 s | **39.5 KB** | 2 (1 tail) |
| PCS, 4096 ring elements (degree 256K) | 4.72 s | 3.88 s | **45.4 KB** | 2 (1 tail) |
| LaBRADOR sub-proof at the 2^26 statement (34,791 ring elements = 2.2M coefficients) | 52.5 s | 44.3 s | **85.6 KB (measured)** | 2 |
| LaBRADOR sub-proof at the 2^30 statement (138,880 ring elements = 8.9M coefficients) | ~4× the 2^26 | ~4× | see the bench | — |

The analytic accounting (Table 4 parameters + the §5.7 level model):

| N | Greyhound contribution | LaBRADOR sub-proof (analytic) | total |
|---|---|---|---|
| 2^26 | 3.5 KB (paper: 3.75) | 31.2 KB | **34.2 KB** (paper: 46) |
| 2^28 | 3.5 KB (paper: 3.75) | 50.9 KB | **53.4 KB** (paper: 53) |
| 2^30 | 3.5 KB (paper: 4.25) | 45.1 KB | **48.2 KB** (paper: 53) |

Notes:
* the proof sizes are **near-constant in N** — the papers' headline property
  (the last recursion levels dominate, independent of the statement size);
* the measured 2^26 sub-proof (85.6 KB) exceeds the analytic model because
  our level parameters are locally optimized (the reference itself notes it
  optimizes "locally instead of globally" — the same caveat, honest here);
* the q = 2^32-99 modulus with the schoolbook i64/i128 arithmetic (no
  multi-modular RNS NTT, no AVX-512) — the papers' 132s commit at 2^30 on a
  Xeon is the optimized C reference's number; our port targets
  verifiability, not throughput;
* the §5.4 restart remedy is live: the level restarts with inflated input
  norms when the measured output norm exceeds the heuristic prediction
  (the [norm] lines in the bench show pred ≈ measured within ~1% at the
  paper's parameter scales).


## §7c — the H6 full-fidelity route (2026-10-05)

The amortized-Dachshund compaction of the HyperWolf projection payload
(`lattice-pcs/src/hyperwolf_labrador.rs`):

| measurement | value |
|---|---|
| kernel shape (d=64, b=2, k=3, jl_rows=32) — clear projection payload | (k−1)·b·jl_rows·d·61 bits |
| kernel amortized proof (`lab_proof_size_bytes`, tail-only at this size) | measured in-test, the level table published |
| paper shape (2^20, `lab_size_model`) — clear | (k−1)·b·256·64·61 bits ≈ 3.4 MB |
| paper shape — amortized (2 levels + the tail's final witness) | ~130 KB modelled |

The honest caveats: at kernel statement sizes the amortization is
tail-only (the shrink gate needs the paper scale); the faithful SIS
gates fail closed at N = 64 (the recorded parameter residual); the
paper-scale β ladder exceeds the `B_r < q/4` wraparound guard and is
modelled, not executed. All pinned by the module's tests
(`full_fidelity_size_beats_clear_payload`, `round_bound_gate_is_fail_closed`,
`faithful_gate_fails_at_kernel_scale`).
## 4. The v3 pipeline (2026-10-06, Wave 10): the COMPLETE protocol, the throughput wave landed

`zkvm_v3bench` measures the full v3 statement — the v2 memory arguments
PLUS the instruction-semantics AIR (all 17 families: ADD/ADDI/SUB/MUL/
DIV/REMU/SLLI/SRLI/SLTU/BRCH×4/JAL/JALR/LOAD/STORE/LUI/AUIPC/HALT),
byte-granular carry chains, five comparison containers, four schoolbooks,
~205 read-only-table lookups, the α-batched AIR sumcheck, the r_air
linear gates — over the sparse engine throughout, now with the **Wave-10
throughput layer**: the batched lookups (~205 Shouts → 6 sumchecks), the
r_air RLC fold (~300 per-column openings → chunked integer folds at the
distinct points), the SIMD engine inner loops, the de-cloned assembly,
and the O(1) claim index on the verify path.

| program | cycles | prove (ms) | verify (ms) | proof (KB) | cycles/s | status |
|---|---|---|---|---|---|---|
| fibonacci | 185 | 742 | 39 | 443 | ~249 | **verified** (complete statement) |
| scale_loop(n=60) | 121 | 376 | 37 | 374 | ~322 | **verified** |
| scale_loop(n=200) | 401 | 1,402 | 41 | 570 | ~286 | **verified** |
| scale_loop(n=500) | 1,001 | 2,828 | 45 | 815 | ~354 | **verified** |
| scale_loop(n=1000) | 2,001 | 5,715 | 51 | 1,294 | ~350 | **verified** |
| scale_loop(n=2000) | 4,001 | 11,617 | 62 | 2,240 | ~344 | **verified** |
| collatz | 462,105 | skipped | — | — | — | the witness estimate (24.9 GB) exceeds the 2 GB container budget — the O(K·T) Val matrix (the virtual-Val route is the documented follow-up) |
| regex/muldiv/sorting/... | — | — | — | — | — | unsupported instruction (sub-word loads/stores — the bounded, enumerated extension) |

The Wave-9 → Wave-10 deltas at the fibonacci shape: prove 1,795 → 742 ms
(2.4×), verify 377 → 39 ms (9.7×), proof 1,970 → 443 KB (4.4×),
throughput ~10² → ~2.5×10² cycles/s — and the scale ladder now runs
verified to 4,001 cycles (the 2-core/4 GB container's honest ceiling
before the Val-matrix budget guard trips).

The honest reading:

1. **Correctness is the win (unchanged)**: the statement is still "the
   execution of the program", the verifier never re-executes, and the
   tamper tests cover every layer — now including the batched-lookup
   and folded-opening tamper suites.
2. **The four enumerated engineering gaps are CLOSED**: (a) the ~205
   lookup Shouts collapsed into 6 batched sumchecks (one per table
   group — the shared rcycle, the shared eq/table factors, the
   per-(lookup, digit) ra claims riding the proofs); (b) the ~300
   per-column grouped openings replaced by the chunked integer folds
   at the distinct points (the Ajtai linear homomorphism — the verifier
   computes each folded commitment from the per-column ones; the exact
   `L·A·2^22 < q/2` integer-fold discipline); (c) the engine's round
   loop is the affine W0/W1 form through the AVX-512 slice kernels —
   each entry touched once per round regardless of round length;
   (d) the claim index is O(1) and the AIR assembly is de-cloned (the
   eq-table dot-product claim evaluation, the owned instances).
3. **Throughput is still NOT SOTA** — ~3×10² cycles/s vs Lattice
   Jolt's ~2×10⁶ (CPU) — and the remaining gap is a NEW enumerable
   ledger: (a) the ~300 per-column Ajtai commitments now dominate the
   prover (the block-commit batching — one packed block commitment for
   the column universe — is the next fold); (b) the byte-guest ISA
   extension (sub-word loads/stores) gates the richer workloads; (c)
   the O(K·T) Val matrix caps the container scale (the virtual-Val
   route); (d) no GPU backend (the excluded mechanism — the CPU floor
   is the AVX-512 kernels).
4. The P0-63 profile: values < 2^63 (the mod-p ±window cannot open —
   see pipeline3.rs's module doc for the soundness analysis);
   non-wrapping ADD/SUB; shamt ≤ 62. The 64-bit mode requires the
   binary-tensor bridge (the documented next layer).
