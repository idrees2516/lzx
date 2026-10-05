# LZX vs SOTA — the honest efficiency ledger

Date: 2026-10-05. Sources: public reporting + the systems' own
papers/repos (Akita ePrint 2026/1983 Table 8/Table 10; a16z crypto's
Lattice Jolt launch reporting; Matter Labs' Airbender disclosures; the
Zisk repository/releases; the Ethereum Foundation's real-time proving
target). **What LZX is**: a kernel-scale research workspace realizing
the protocol cores of 11+ post-quantum lattice papers end-to-end
(prover + verifier + adversarial tests) — not a production zkVM. The
comparison below is therefore structured as *target vs measured-gap*:
the SOTA row defines the bar; the LZX row states what the kernel
measures today and which specific mechanism separates the two.

## 1. The systems

| system | who | stack | headline (public, as of 2026-10) |
|---|---|---|---|
| **Airbender** | Matter Labs (zkSync) | RISC-V zkVM, Boojum successor | ~1 s L2 block proving; full **Ethereum L1 blocks in 9.4 s on 2× RTX 5090** (~$0.0026/block); billed as the fastest open-source RISC-V zkVM |
| **Zisk** | Jordi Baylina | GPU-first zkVM (Polygon lineage) | real-time L1 proving with **~32 GPUs** (most blocks), ~40 GPUs for 35 M-gas blocks; v1.3.0-alpha adds BLAKE3 recursive proving |
| **Lattice Jolt** | a16z crypto (+ LayerZero) | post-quantum zkVM (Jolt + lattice PCS) | **>10 M RV64IMAC cycles/s** GPU (Apple Metal, a laptop), **>2 M cycles/s** pure CPU; ~200 B/cycle prover memory; 3× over prior lattice zkVMs; Twist & Shout integration alone brought Jolt a 6× speedup |
| **Akita** | LayerZero (the PCS LZX implements) | lattice PCS inside Jolt | **61–70 KB proofs**, **8.1–15.9 ms verify** (19–90× faster than Greyhound), 128 B commitments; inside Jolt on an M4 Max: 1.477 MHz padded throughput at T=2²⁷ (1.3–2.2× faster than Dory), 98 KB full proofs |
| **the EF bar** | Ethereum Foundation | — | real-time L1 zkEVM: **≤10 s for 99% of blocks**, 128-bit security (100-bit minimum at launch) |

## 2. Where LZX stands (measured, this container, release build)

| workload | prove | verify | proof | note |
|---|---|---|---|---|
| zkVM end-to-end (fib, 185 cycles, batched-compact) | 1 366 ms | 198 ms | **33.0 KB** | the full memory argument: 9 T&S instances, 12 batched sumchecks, verifier never re-executes |
| zkVM end-to-end (regex, 451 cycles, compact) | ~4 600 ms | ~288 ms | 108.4 KB | pre-batching number, kept for the record |
| RoKoko statement-growth driver (m_w=512, 2 rounds: coarse +1 block, fine +2 blocks/+n_bat) | 4 620 ms | 484 ms | — | this session: Lemma 7/8 semantics, the parbreak estimator gate (11.7 bits at toy params — fail-closed) |
| Akita A5 recursion driver (4-block batch → fold → Rice terminal) | ~1 ms | ~1 ms | 600 B terminal (Rice) | this session: the §8.2 stop rule + signed-Rice reveal |
| labinius PCS round (sizem) | 78 ms | 2.6 ms | — | the AVX-512 kernels at upstream parity |

## 3. The honest gap analysis (LZX → the Akita / Lattice-Jolt bar)

1. **Throughput (~4 orders of magnitude).** LZX's dense kernel proves
   ~135 cycles/s vs Lattice Jolt's 2 M CPU / 10 M GPU cycles/s. The
   mechanism gap is enumerable: (a) the sparse "0s are free" provers
   exist in-tree (`lattice-memory/sparse_engine.rs`, Wave 8.5) but the
   zkVM's live path only partially routes through them; (b) no GPU
   backend (Zisk's and Lattice Jolt's headline numbers are
   GPU/Metal-first; our AVX-512 kernels are the CPU floor); (c) the
   cycle caps (T ≤ 2¹² dense) are the documented dense-prover posture
   — the streaming path (`lattice-streaming`, O(K+log T)) lifts them
   but is not yet the default.
2. **Proof size (within ~2× at the PCS layer, ahead at the
   memory-argument layer).** Our batched-compact memory-argument proof
   is 33 KB at 185 cycles; Akita's PCS opening is 61–70 KB at 2²⁷ bits
   — different objects, but the *shape* is right: the 50 KB design
   target was met via leg batching + amortized openings (110× over the
   clear mode). The remaining term is the 3.5 KB values-only claims
   list (345 claims) — the paper's recursive/2-level fold (Stage 5.2)
   compresses it further.
3. **Verification (within ~1 order at kernel scale).** 198 ms vs
   Akita's 8–16 ms single-thread at full scale — the gap is the
   O(cycles) public-table recomputation (the program image) plus the
   un-batched terminal checks; the MLE-structured verifier tables
   (Stage 5.3) are the documented fix.
4. **Security posture (the honest split).** The estimator-run verdict
   (SECURITY.md): the compact fold's MSIS binding at k=4/A≤2⁸ is 329+
   bits in the sound regime but the *benchmark* response lengths need
   the second-level fold; the RoKoko driver now ships the fail-closed
   **parbreak estimator gate** at every driver setup (this session),
   and the SIS tables publish the knob levers. Airbender/Zisk are
   elliptic-curve (not post-quantum); Lattice Jolt and Akita are the
   post-quantum cohort LZX belongs to.
5. **What LZX has that the production systems do not:** the 11-paper
   protocol matrix itself — ProtogaLattice bootstrapping, SALSAA
   D1/D2/D4, the full HyperWolf Protocol 1/2/3 with H6 compaction, the
   RoKoko recursive COM + Π^fold-split + the **statement-growth driver
   (Lemma 7/8, this session)**, Twist & Shout sparse PIOPs, the v2
   no-re-execution pipeline, the streaming (ePrint 2025/611) and
   monomial-basis (ePrint 2026/762) engines — all adversarially
   tested. This is the differentiating asset: the protocol cores the
   production systems are converging toward, realized and
   cross-connected.

## 4. The path to the bar (the prioritized mechanism list)

| # | mechanism | paper basis | expected effect | status |
|---|---|---|---|---|
| 1 | GPU/Metal prover backend for the sumcheck+fold kernels | Zisk/LJ practice | ~10× (the LJ CPU→Metal ratio) | not started |
| 2 | sparse engine as the zkVM default (the 0s-are-free path) | T&S §6.3/§7 | lifts the cycle cap; K=2¹⁰×T=2¹⁰ Shout already <60 ms | engine landed, wiring partial |
| 3 | streaming as the default prover (O(K+log T)) | ePrint 2025/611 | unbounded T at ~3.3× the hybrid switch speed | landed, opt-in |
| 4 | the 2-level fold (LaBRADOR decider, Stage 5.2) | LaBRADOR/§5.6 | 128-bit MSIS at benchmark lengths; claims 3.5 KB → ~1 KB | specified |
| 5 | MLE-structured verifier tables (Stage 5.3) | §9.4 constant-root | verify 198 ms → ~20 ms | specified |
| 6 | the A3/A4 commitment-scale substitution | Akita §7 | polylog private responses (kill the terminal reveals) | the documented gap |
| 7 | the per-level planner (digit-depth re-tuning) | Akita §12 | shrinking multi-level recursion (the A5 driver's chain grows at fixed depths) | the documented gap |

## 5. Reading the comparison honestly

Airbender and Zisk answer "how fast can a *deployed* elliptic zkVM
prove an Ethereum block today" (9.4 s on 2 GPUs; 32 GPUs). Lattice
Jolt and Akita answer "how fast can a *post-quantum* zkVM go" (10 M
cycles/s GPU; 61–70 KB proofs at 8–16 ms verify). LZX answers a
different question — "what do the eleven protocol cores beneath those
systems actually compute, and do they compose" — and measures itself
against the second cohort's mechanisms, not the first cohort's
deployments. The kernel's honest position: **proof-size machinery
within reach of Akita's shape; throughput 4 orders out pending the GPU
backend and the sparse/streaming defaults; security gated
fail-closed by the in-tree estimator with the second-level fold as
the named 128-bit closure.**
