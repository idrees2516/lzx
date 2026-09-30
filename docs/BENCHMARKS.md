# LZX zkVM Benchmarks — End-to-End Memory-Argument Proofs vs SOTA zkVMs

Date: 2026-09-29. Harness: `cargo run --release -p lattice-bench --bin
zkvm-membench`. Guest programs: `lattice-guest` (Jolt-lineage benchmark
set, real algorithms, reference-checked).

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
(docs/DESIGN_50KB.md): narrow byte packing (1 byte/coefficient — also
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
   all on the Wave 7 remainder ledger in `docs/WAVE_ANALYSIS.md`.
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
