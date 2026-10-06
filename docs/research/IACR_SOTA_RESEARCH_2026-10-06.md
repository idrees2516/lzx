# IACR / SOTA deep research — 2026-10-06 session

Scope: the papers, techniques, and concrete requirements to drive LZX to
lattice-zkVM SOTA (CPU-only, AVX-512), with the three mandated follow-ups
(the block-geometry MSIS re-run, the streamed verifier key passes, the P1
sub-word constraint polynomials) as the landing targets.

## 1. The SOTA reference points (October 2026)

| System | Prover | Verifier | Proof | Security | Source |
|---|---|---|---|---|---|
| Lattice Jolt (a16z) | **>2M RV64 cycles/s CPU-only** (MacBook); >10M with Apple Metal | — | **<100 KB** | Module-SIS, 128-bit target | a16z article "Entering the era of lattice SNARKs" (2026-09) |
| Lattice Jolt (curve-era baseline) | ~1M cycles/s CPU; ~4M Metal | — | ~200-600 KB (PQ hash peers) | — | same |
| Jolt + Akita vs Jolt + Dory | **1.3×–2.2× prover**, **2.2×–7.4× verifier** speedup | — | comparable, every proof <100 KB | Module-SIS | Akita ePrint 2026/1983 |
| Akita PCS alone | — | **19.2×–89.7× faster than Greyhound** (same lattice-security calibration) | **61–72 KB** | standard Module-SIS | ePrint 2026/1983 |
| Airbender (STARK, PQ baseline) | ~35 s per Ethereum L1 block (2×RTX 5090) | — | ~200+ KB | hash-based | House of ZK 2025-06 |
| Zisk | ~32 GPUs real-time L1 | — | — | — | SOTA_COMPARISON's earlier ledger |
| LZX (this repo, wave 10) | 742 ms / 185-cycle fib (~250 cycles/s at container scale; the scale ladder to 4,001 cycles) | 39 ms | 443 KB (v3 statement path) | MSIS-gated profiles | BENCHMARKS §4 |

**The bar to match (CPU-only)**: 2M cycles/s prover, <100 KB proofs,
~200 B/cycle prover memory, 128-bit Module-SIS from standard assumptions,
Õ(N) prover with sub-linear memory beyond the polynomial itself.

### The Lattice Jolt speed anatomy (why 2M is reachable on CPU)

1. **128-bit fields, not 256-bit**: the prover's dominant cost is field
   multiplication; halving operand width multiplies throughput. Goldilocks
   (64-bit) is already in this class — LZX's stance is right.
2. **Soundness scaling**: lattice SNARK soundness error scales like
   `log(n)/|F|` vs hash-based `n/|F|` — the full 128 bits survive on the
   small field at billion-step scale. LZX's P0-63 profile (the ±p-window
   analysis) is the same discipline taken further (63-bit recomposition).
3. **Sparse commitments**: commitment time proportional to NONZERO entries
   — the property Twist and Shout needs (enormous, almost-all-zero
   polynomials). LZX's sparse '0s are free' engine is this property.
4. **Sub-linear memory**: Akita's prover memory grows sub-linearly beyond
   the polynomial — LZX's streaming O(K+log T) default + virtual-Val are
   this property.
5. **The everything-SNARK architecture**: no recursion for scale; direct
   proving at program size.

## 2. The paper ledger (the extraction sources)

* **Akita** — ePrint 2026/1983 (Dao, Bodaghi, Khajehpour, Vitto,
  Badakhshan, Georghiades, Liu, Zhang, Thaler; LayerZero + CMU + USC +
  a16z; revised 2026-09-18). The PCS under Lattice Jolt. Techniques:
  - **Setup offloading** (verification): the public setup matrices are
    committed AHEAD OF TIME and the verifier's work in processing them is
    DEFERRED AND PROVED against these commitments — Õ(N^{1/K}) verify at
    fixed K ≥ 2 while keeping Õ(log N) proofs, Õ(N) prover, standard
    Module-SIS. This is the direct answer to our streamed-verifier-keys
    gap: LZX's seeded keys allow the strictly simpler streaming
    regeneration (no extra commitments needed — O(1) state).
  - **Commitments compressed to 128 bytes each** — the block commitment's
    size target per column group.
  - **Exact Euclidean norm checks for tighter Module-SIS parameters** —
    the estimator discipline: exact ℓ2 bounds (not √m·β heuristics) buy
    real parameter margin. Our block-geometry re-run adopts both norms.
  - Optimized digit range check (the involution pairing — landed here in
    Wave 7's A3), relation-specific ring dimensions and subring
    challenges, batched openings of separately committed polynomials (the
    block-commit batching's paper-side anchor), low-communication
    distributed proving, offline planner under configurable cost
    objectives (landed as planner.rs).
  - Rust implementation; benchmarks vs lattice- and hash-based PCSs.
* **Twist and Shout** — ePrint 2025/105 (Setty, Thaler; CRYPTO 2026,
  cited by 24). The memory-checking core of Jolt: one-hot addressing and
  increments. LZX's Twists/Shouts are this; the sub-word ISA extends the
  ports without changing the argument shape (word-granular RAM + the
  merge in the instruction-semantics layer).
* **LatticeFold+** — ePrint 2025/247 (Boneh, Chen, ...): folding over
  64-bit fields, ℓ2-norm response discipline (the estimates notebook is
  public — the GitHub `lattice-fold-plus-l2-norm-blog-estimates` repo
  runs the malb/lattice-estimator on the fold instances). Confirms the
  estimator-on-fold-instances workflow our fold_security_table follows.
* **LaBRADOR / Greyhound / SuperNeo / Hachi** — the lineage Akita
  completes; Hachi is the square-root-time-verifier predecessor Akita
  beats via setup offloading.
* **malb/lattice-estimator** (the reference estimator): `estimator.sis`
  module with `Estimate.strong`/`Estimate.weak` conventions, ADPS16
  Core-SVP cost models 2^{0.292β} classical / 2^{0.265β} quantum /
  2^{0.2075β} paranoid, used for Dilithium/Falcon-class instances. Our
  lattice-sis-estimator crate ports this core (infinity + Euclidean
  paths, LGSA shape simulator, golden tests preserved).

## 3. The three mandated follow-ups — technique grounding

### (1) Re-run the MSIS estimator for the block geometry

The block-commit's soundness terminates in MSIS on the WIDE `[F | −y]`
key: `F ∈ R_q^{k×m}` with `m = r·n̄^pad` ring columns (the whole
universe), q = Q_32 = 3·2^30+1, ring dim N = 64. The distinctive regime
vs the compact fold's `[F̄ | −y]`: the preimage's W-side is
**byte-bounded** (two openings differ by ≤ 2·255 = 510 per coefficient —
the honest W is byte-packed), INDEPENDENT of r and A — the tightest bound
any fold here has had. The estimator run must price (Akita's discipline):
- both norms: ℓ∞ (byte bound) AND exact ℓ2
  (`‖Δw‖₂ ≤ 510·√(m·N)` worst-case; the statistical form is tighter);
- the gate regime for the σ-side (`2·r·A·255` at A = 2^6) and the 6σ
  statistical form — for the [F | −y] composed instance;
- the k ladder (2 → 64): the wide instance's rank n = k·N — the honest
  question is whether ANY (k, bound-regime) at the benchmark shapes
  reaches 128 classical bits, and what the memory/proof cost of that k is
  (which the streamed verifier makes memory-free).

### (2) Stream the verifier's key passes

`verify_block_opening` materializes the full key (`O(k·m)` ring state ≈
10 MB at fib scale) to compute `g = Σ_l ρ_l·F[l]` and the public C-array.
The key is SEEDED (`uniform_from_seed(b"ajtai-A", seed, l·m + p)` flat
l-major) — so the verifier can regenerate each column on the fly and
accumulate `c_p = g_p + d_j·h_i` in ONE pass with `O(m + n̄ + k)` state,
bit-identical to the materialized computation (the generation is
deterministic per element). This is the strictly-simpler cousin of
Akita's setup offloading (no commitment of the setup matrix needed since
the key is transcript-derived and public). The k-ladder from (1) then
costs the verifier NOTHING in memory — the security knob becomes free.

### (3) The P1 sub-word constraint polynomials

The byte-guest ISA's trace substrate (columns.rs: merge_word splicing,
relaxed alignment, the RAM replay fix) is built and differentially
verified, but the CONSTRAINT FAMILY layer (constraints.rs — the
instruction-semantics statement path) still only knows the word-granular
loads/stores. The gap, precisely: the six selector rows (sel_lb/sel_lh/
sel_lbu/sel_lhu/sel_sb/sel_sh), the raw_funct3 arms, the off-bit aux
columns (addr bits 0..2), the routing identity extension
(byte: `addr = 8·word + Σ 2^i·off_i`; half: `addr = 8·word + 4·off_2 +
2·off_1` with the ALIGNMENT constraint `sel_half·off_0 = 0` fail-closed),
the load-extraction muxes (bit-grain: `rd[b] = Σ_p pos_p·old[8p+b]`,
sign-extension for LB/LH via the muxed sign bit), and the store-merge
identities (`new[8p+b] = pos_p·rs2[b] + (1−pos_p)·old[8p+b]`). All ride
the EXISTING bit columns (val_bits[T_MEM_OLD/NEW/RS2/RD] are 64-bit
decompositions) — no new value tensors, only 3 aux bit columns + the
degree-≤4 polynomials.

## 4. Requirements checklist to "match SOTA level" (the honest ledger)

1. **Throughput**: the wave-10 affine round loop + batched lookups +
   r_air RLC fold + SIMD + de-cloning closed the enumerated gaps (742 ms
   fib prove); the remaining distance to 2M cycles/s is the statement
   path's constant factors (the v3 AIR's per-family leg count) and the
   commitment layer's ring-op count — the block fold (1.7× measured) and
   the estimator-driven k-ladder are the levers.
2. **Proof size**: 19 KB block-mode at fib scale (already < 100 KB);
   Akita's 128-byte compressed commitments are the next compression.
3. **Memory**: virtual-Val + streaming default + the streamed verifier
   keys = sub-linear beyond the witness (the Akita property).
4. **Security**: the block geometry's MSIS table must be RUN and the
   profile gates updated to the verdict (this session's item 1) — 128
   classical bits from standard Module-SIS at the shipping shapes, or the
   honest boundary documented.
5. **The byte-guest ISA's statement-path coverage** (item 3) — richer
   workloads (byte-grain programs) provable through the canonical v3
   path, not just the v2 pipeline.

## 5. Source URLs (fetched this session)

- https://eprint.iacr.org/2026/1983 (Akita — abstract + metadata)
- https://a16zcrypto.com/posts/article/lattice-snarks-jolt-post-quantum-faster
- https://layerzero.network/blog/introducing-akita
- https://eprint.iacr.org/2025/105 (Twist and Shout)
- https://eprint.iacr.org/2025/247 (LatticeFold+)
- https://eprint.iacr.org/2024/257 (LatticeFold)
- https://github.com/malb/lattice-estimator (the reference estimator)
- https://eprint.iacr.org/2026/1080-adjacent: the LF+ ℓ2 estimates
  notebook (lattice-fold-plus-l2-norm-blog-estimates on GitHub)
