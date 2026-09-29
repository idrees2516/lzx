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

## 3. Comparison with SOTA zkVMs (published numbers)

Context, not competition: LZX is a lattice-SIS research zkVM at kernel
scale; the systems below are production elliptic-curve/FRI zkVMs. The
comparison fixes the *shape* of the gap.

| System (published) | guest flavor | prover | verifier | proof | notes |
|---|---|---|---|---|---|
| **Jolt** (a16z, 2024) | sha2/keccak/ecdsa guests | ~0.9–1.5 s / 2^20 cycles | ~50–100 ms | ~100–200 KB | Spartan-style + sumchecks, GPU paths exist |
| **SP1** (S1, 2024) | sha2/rsaecdsa | ~2–6 s / 2^20-ish cycles | ~10–100 ms | ~100–300 KB | Plonkish + STARK folding, SP1 Pro network |
| **Risc0** (2024) | sha2/rsa | ~2–10 s / 2^20 cycles | ~50–200 ms | ~100–500 KB | FRI STARKs |
| **lzx memory argument** (this) | arithmetic/DFA guests, ≤ 2^12 cycles | ~2.5–7.5 s / ≤ 512 cycles | ~0.5–1.1 s | ~3.6–7.2 MB | SIS commitments, Clear-response norm proofs |

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
