# LZX Performance Engineering: the labinius-parity analysis

This document is the deep audit of **how the LZX codebase was made as efficient as
[osdnk/labinius](https://github.com/osdnk/labinius)**: the technique inventory extracted from
upstream's ~7,800 lines of AVX-512 kernels, what was ported and measured here, what was
deliberately simplified, and what remains on the roadmap. Everything claimed below is
reproducible: `cargo run --release -p lattice-labinius --example backend_bench` and
`cargo run --release -p lattice-labinius --example round_bench m`.

Machine of record: 2-core Xeon, full AVX-512 (F/BW/VL/DQ/VBMI/VBMI2/VNNI/GFNI), VPCLMULQDQ,
BMI2, 504 MB L3, Rust 1.98.1 stable, `--release` (LTO thin, codegen-units 1).

---

## 1. The labinius efficiency doctrine (what upstream actually does)

Reading upstream's kernels at cycle-level granularity, seven principles carry the performance:

**D1 — Vertical batch-of-32 layout.** The unit of computation is *32 ring elements at once*:
one 64-byte vector holds slot `j` of 32 different polynomials. Every transform, every product,
every accumulation is 32-way SIMD across the batch, with the scalar loop structure gone
entirely.

**D2 — Never reduce what you don't have to.** Upstream tracks *bounds, not values*: each
transform level's output bound is known as an exact multiple of `q` (7.5 q for q = 3889,
2.31 q for 9721), and the commitment's slot products are **never individually reduced** —
`vpmaddwd` accumulates raw i16×i16 products into i32 lanes, with an exact fold-back period
computed at compile time (`red_period` = 8 for 3889, 4 for 9721), asserted by `const`
evaluations. One multiply-port uop per slot per batch replaces the four a Montgomery slot
product would cost.

**D3 — Fold the cheap levels into lookup tables.** Levels 0–2 of the binary NTT plus the
level-3 twiddles are a *linear* function of the 4-bit nibble `(b_i, b_{i+162}, b_{i+324},
b_{i+486})` — so they become one byte-split 16-entry `vpermb` lookup (1 uop, port 5) instead
of butterflies. On the quad tree, even the level-3 *butterfly* folds into the tables (three
lookups and two adds per output row): 1512 Montgomery products per batch instead of 2160.

**D4 — Choose the instruction, not the semantic.** `vpermb` over `vpermw` (1 uop vs 2),
`vpbroadcastd` from memory as a pure load (raw `asm!` because LLVM rebuilds the splat),
`vpmaddwd` accumulation instead of `vpdpwssd` because the packed accumulator's lane layout is
what fits L1, shuffle-port lookup-Barrett (`vpmultishiftqb`+`vpermb`) instead of a multiply-
port Barrett when port 0 is the bottleneck, f64-exact `mod_q` on the double unit for the
final reduction.

**D5 — Keep the working set in L1 and consume results while hot.** The accumulator is 21.5 KB
(two slots packed per vector), the transform's stack block is 10 KB, each finished block is
multiplied into the accumulator *while still in L1* (the `BlockSink` hook), and the A stream is
prefetched (`prefetcht1`, one batch ahead) only when A's footprint exceeds the 4 MB cache
crossover.

**D6 — Amortise the front end once for all limbs.** The bit-sliced index rows depend neither
on `q` nor on the tree: one slicing pass per 32 elements feeds every limb's kernel.

**D7 — Bound-work with exact compile-time models.** The quad kernel's reduction schedule is
*derived* by a const-evaluated replay of the tree's bound arithmetic (`bin_model_f3` /
`bin_model_split`), picking the cheapest schedule that fits i16; the assertions that pin
`red_period`, Karatsuba applicability and output bounds are all `const` — the safety argument
is checked by the compiler, not by a comment.

Upstream's honest baseline numbers (i7-11850H, one core, 2^18 F162 in 256 columns): commit
7.2 ms base limb + 6.0–10.4 ms per further limb, 247–533 cycles per ring element of transform,
58–74 cycles of MAC.

---

## 2. What this port adopted (and what it measures here)

### 2.1 `PCLMULQDQ` for the binary fields (`hw.rs`)

Every `F162`/`B128` product stood on a bitwise software carry-less multiply (a 64-iteration
shift-XOR loop, ~256 cycles). `hw::clmul64` routes through one `_mm_clmulepi64_si128`
behind a `OnceLock` runtime gate, with the software loop kept as the portable fallback.

**Measured: 66.0x** on the primitive; the whole `evaluate` stage of the reference round
(eq tables + row evaluations over F162) went **327 ms → 9.4 ms (35x)**.

### 2.2 The AVX-512 PCS backend (`simd/`, ~2,600 lines ported as pure `std`)

| Module | Upstream | Status |
|---|---|---|
| `transpose.rs` | `simd/transpose_f162.rs` | full port (GFNI 4-phase bit-slice) |
| `ntt_small.rs` | `simd/ntt/bin_small.rs` | full port (the pure-intrinsics reference kernel, incl. the three `asm!` micro-shims for `vpbroadcastd`/`vpmulhw` codegen) |
| `ntt_quad.rs` | `simd/ntt/bin_quad.rs` + `bin_asm`'s lookup-Barrett | full port (both schedules, const bound models, `BlockSink`) |
| `commit.rs` | `simd/commit.rs` | full port of the MAC machinery (raw `vpmaddwd` accumulation, compile-time fold-back periods, packed accumulator, `hsum8`, f64-exact `mod_q`, quadratic 3-accumulator Karatsuba) + `store_transform` scatter extraction + `mac_row_u64` for the fold |

Every kernel is **bit-exact against the scalar reference** (`tests/simd.rs`: outputs congruent
mod `q`, every lane within the declared bound, full commitments byte-identical between
backends) — the port's contract with itself.

**Measured (batch of 32 ring elements):**

| Kernel | scalar | AVX-512 | speedup |
|---|---|---|---|
| forward NTT, q = 3889 (split) | 0.768 ms | 0.003 ms | **274.6x** |
| forward NTT, q = 2917 (quad) | 0.676 ms | 0.003 ms | **259.9x** |
| MAC + finish, q = 3889 | 0.767 ms | 0.002 ms | **447.8x** |
| MAC + finish, q = 2917 (Karatsuba) | 0.564 ms | 0.004 ms | **147.3x** |

**Full commitment (`commit_with`, same key, same witness):**

| Size | scalar | AVX-512 | speedup |
|---|---|---|---|
| 2^18 F162, 128 columns (suite sizem) | 3197.6 ms | 84.8 ms | **37.7x** |
| 2^15 F162, 32 columns | 404.3 ms | 9.99 ms | **40.5x** |
| 2^13 F162, 16 columns | 105.0 ms | 3.30 ms | **31.8x** |

**The reference round end-to-end (suite sizem, Clear mode):**

| Stage | before | after | speedup |
|---|---|---|---|
| commit | 3134 ms | 83.8 ms | 37x |
| point | 2 ms | 2.1 ms | — |
| evaluate | 327 ms | 9.4 ms | 35x |
| challenge | 12 ms | 12.6 ms | — (SHAKE-bound) |
| fold | 332 ms | 47.8 ms | 7x |
| verify | 78 ms | 67 ms | 1.2x |
| **total** | **3885 ms** | **222 ms** | **17.5x** |

For calibration against upstream's own machine: their 2-limb commit at 2^18/256 columns is
~13 ms on one Tiger Lake core; this port's 85 ms at 2^18/128 columns *includes* the kept-
transform extraction (a 170 MB `store_transform` scatter the reference-round's fold consumes),
the scalar `components_of` decomposition, and the matrix assembly. Same kernel designs, same
layout, same arithmetic — the remaining distance is the port's scalar glue, itemised in §3.

### 2.3 The fold's arithmetic (`fold.rs`)

The fold was the second bottleneck (332 ms). Three exact changes:

1. **Hoist the challenge conversion.** `fold_witness` re-centred each challenge slot
   (`rem_euclid`) inside the `nr × r × N` inner loop — 512 redundant conversions per value.
   Challenges are now converted to `[0, q)` once (83 k conversions total instead of 42.5 M).
2. **AVX-512 u64-lane MAC** (`mac_row_u64`): the length-`r` inner product per slot runs as
   `vpmullq` on 8 u64 lanes with `vpmovzxdq` widening — no reduction at all until the row's
   single final `% q` (exact: `r · q² < 2^64` for every suite shape).
3. **Barrett Horner** in `challenge_r162_slots`: every `acc = (acc·x + c) % q` step feeds
   `barrett_mod_u64` — one u64 mulhi and one conditional subtract instead of a division —
   exhaustively checked per prime over the reachable `v < q² + q` range (`tests/barrett.rs`).

`a_times_v` (the verifier's `A v`) got the same preconversion + u64 MAC treatment.

**Measured: fold 332 ms → 47.8 ms (7x).**

### 2.4 LaBRADOR's reduction (`lattice-labrador/src/ring.rs`)

`Q = 2^48 − 59` gives `2^48 ≡ 59 (mod Q)`, so `cmod` is now four shift-and-multiply folds
(exact for the full i128 range, exhaustively tested against the division form including
`i128::MIN`/`MAX`). **Honest result: a wash** (0.9x–1.0x) — LLVM already strength-reduces
division by a *constant* modulus into multiply-shift sequences. The change is kept for
cross-compiler predictability (i128 constant division is not guaranteed strength-reduced
everywhere), and the measurement is recorded here precisely because the first instinct
("division is 40 cycles") was wrong at `-O`. The real LaBRADOR bottleneck is the schoolbook
O(N²) i128 product — upstream replaces it with an 8-prime RNS + AVX-512 `polx`; see §4.

---

## 3. The measured remaining distance (where the 85 ms and 222 ms still go)

Cycle-accounting the current round against upstream's numbers:

1. **`store_transform` (kept transform)**: the fold consumes per-element `[u32; N]` rows, so
   the SIMD backend scatters the vertical transform back to element-major — 170 MB of scattered
   u32 stores at sizem, roughly 15–20 ms. Upstream instead keeps the vertical layout and folds
   it with the `gen_*` SIMD kernels. *Fix: port `gen_small`/`gen_large` and keep `AuxData`
   vertical — an interface change through `fold.rs`.*
2. **`components_of` decomposition + matrix assembly** (~10–15 ms): scalar u64 arithmetic with
   `% q` per slot over 128 chunks × 2 limbs. *Fix: Barrett + SIMD inner loop; the twiddle
   tables already exist.*
3. **The fold's inverse transform** (`intt_of`, 512 scalar NTTs ≈ 11 ms): the amortised
   witness is *not* binary, so the `bin_*` kernels do not apply. *Fix: port `gen_*` (the
   general mixed-radix AVX-512 transforms).*
4. **`challenge` (12.6 ms)**: SHAKE-256 absorb/squeeze over the commitment — hash-bound,
   irreducible without changing the transcript discipline (which would change the proofs).
5. **`verify` (67 ms)**: dominated by `a_times_v`'s 512 scalar forward NTTs (same `gen_*`
   gap) plus the R_162 Horner fold (now Barretted).
6. **`fold_commitment`'s Horner**: 2 × 128 challenges × 162 slots × 162 terms — Barretted but
   still scalar; *~15 ms. Fix: vectorise across slots (vpmullq + vector Barrett), or
   batch-invert the per-slot powers.*

Itemising honestly: of the 222 ms round, ~120 ms sits behind the `gen_*` port (one further
~2,000-line kernel family), ~30 ms behind scalar decomposition/assembly, and ~70 ms is
hashing + protocol structure.

---

## 4. Roadmap to full upstream parity (mapped, with expected gains)

| Item | Upstream reference | Expected effect here | Effort |
|---|---|---|---|
| `bin_large` port (17497/19441) | `simd/ntt/bin_large.rs` | those primes 200x+; enables all-7-limb suites | medium (mechanical port) |
| `gen_small`/`gen_large`/`gen_quad` port | `simd/ntt/gen_*.rs` | fold/verify/kept-transform fully SIMD; round 222 → ~100 ms | large |
| Vertical `AuxData` + SIMD fold | upstream's fold consumes `Batch32` directly | removes the 170 MB scatter (~20 ms) | medium (interface change) |
| Block-sink fusion (`Mac`/`MacKeep` consuming each 27-slot block mid-transform) | `commit.rs`'s sinks | commit transform+MAC ~35% closer to upstream's 630 vs 874 cycles/element | small once sinks exist |
| A-stream prefetch + column grouping (`GROUP=8`) | `commit.rs` `A_PREFETCH_BYTES` | matters when A exceeds cache (large suites, many columns) | small |
| `components_of` SIMD | — | ~10–15 ms at sizem | small |
| LaBRADOR RNS (8 small primes, NTT per prime, CRT reconstruction) | vendored `lattice-dogs` `polx` | LaBRADOR prove 10–50x (replaces O(N²) i128 schoolbook) | large |
| rANS entropy coder for the opening | `wire/` | opening 664 KB → ~370 KB (upstream ~8.8 bits/coeff) | medium |
| Goldilocks AVX-512 kernels for the rest of LZX (lattice-core/sumcheck/akita/zkvm) | Plonky2-style packed mul + lazy reduction | the whole non-labinius stack; 4–8x on field-bound stages | large |

The last row is the strategic one: the labinius PCS now runs at kernel speed, and the same
vertical-batch doctrine applies to the Goldilocks half of the codebase — that is the next
wave, not a research question.

---

## 5. Methodology

* Timing: median over runs (5 for kernels/commit SIMD, 2–3 for the slow scalar paths), pure
  `std::time::Instant`, nothing else running on the machine, release profile
  (`lto = "thin"`, `codegen-units = 1`).
* Correctness under speed: every optimised path has an exactness test — SIMD kernels vs the
  scalar NTT (mod `q` + declared bounds), full commitments byte-identical across backends,
  Barrett exhaustively per prime, `cmod` vs the division form over the full i128 range. The
  speedups above are speedups of *verified-identical* computation, not of approximations.
* Portability: every gate is runtime-detected (`is_x86_feature_detected!` through
  `hw::avx512_pcs()` / `hw::pclmulqdq()`); non-x86-64 and feature-less machines run the exact
  scalar paths unchanged. The crate still has zero dependencies and builds on stable.
