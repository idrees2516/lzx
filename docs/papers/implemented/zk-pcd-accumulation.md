# Zero-Knowledge Proof-Carrying Data from Accumulation Schemes (ePrint 2026/289)

**Status: implemented in depth** — `crates/lattice-pcd` (~4.3k lines with
tests), Zheng–Gao–Liu (PolyU HK).

## What was implemented

### The base layer (`fp_base`, `g1`, `pedersen`)
- The BN254 **base field** `F_p` (CIOS Montgomery, 4×u64 — mirroring the
  workspace's `fp256` which is the scalar field `F_r`), with Legendre /
  square roots via `a^{(p+1)/4}` (p ≡ 3 mod 4).
- **BN254 G1** (`y² = x³ + 3`, Jacobian coordinates, cofactor 1 — the
  on-curve check is a subgroup check), windowed scalar multiplication,
  try-and-increment hashing to the curve. The generator and group order
  are cross-checked in tests (`r·G = ∞`).
- **Vector Pedersen** `Com(v; r) = Σ vᵢGᵢ + rH` over `F_r` scalars:
  homomorphic over `F_r`-linear combinations (the accumulation's core
  check — pinned by a dedicated test), statistically hiding, binding via
  discrete log. This is the instantiation the paper itself assumes
  ("conducted over cyclic groups for simplicity", §5's complexity notes);
  the PQ route is the Ajtai/shortness discipline of the LatticeFold line
  (out of scope here — see the ledger below). A 4-bit **bucket MSM**
  (`pedersen::msm`) makes commitments and combinations
  count-independent.

### §5.1 — the special-sound framework (`sps`)
- The homogeneous algebraic verifier map `V_sps = Σ_k f_k^V` as the
  `SpsRelation` trait with **black-box point evaluation** — the
  accumulation's interpolated `F(X) = V_sps(x(X), m(X), r(X))` is
  evaluated by interpolating the eq-combination inputs and applying the
  map; round polynomials interpolate from `d+2` node evaluations.
- Instances: **R1CS** (d=2, µ=1) with the `z₀ − 1` (relaxed-`u`) and
  public-input consistency coordinates; **CCS** (d=q, µ=1 — the
  high-degree headline); **permutation / grand product** (d=n, µ=2 — a
  challenge-dependent map: `∏(aᵢ+ρ) − ∏(b_{π(i)}+ρ)`).
- The committed-message NARK `FS[Π_sps^cm]`: hiding commitments, the
  **stateless** challenge chain `r₁ = ρ(x)`, `rᵢ = ρ(rᵢ₋₁, Cᵢ)`.

### §5.1/§5.2 — the ZK sum-check (`zk_sumcheck`)
- The CFS17/XZZ+19 mask of Eq. (3): `G(X) = r₀ + Σ_k r_k(X_k)` per
  coordinate, `O(n·(1 + L·D))` coefficients, closed under
  `F_r`-linear combinations — **the [KS24] no-growth property** the
  accumulator relies on (pinned by tests).
- The §5.2 batched accumulation sum-check: one Fiat–Shamir transcript
  over `L` variables carrying `n + m·n` statement coordinates — the
  masked first statement `Σ_b [eq(b,α)·F̃_c(b) + γ·G'_c(b)] = γ·e'_g,c`
  plus the **point-update statements** `Σ_b eq(β_j,b)·Ĝ_{j,c}(b) =
  v_g,j,c` that re-randomize the old masks' claims to the fresh point β
  (the technique the paper adopts from [KS24] to stop accumulator
  growth). Round messages are `d+2`-coefficient vectors; tamper coverage
  in tests.

### §5.2 — the zk-Protogalaxy accumulation scheme (`accum`)
- The accumulator `acc.x = (x, [Cᵢ], [rᵢ], E, β, v_g)` /
  `acc.w = ([mᵢ], G(X), blinds)`; the **masking vector** (the random
  dummy pair — the paper's ZK contribution #1); the eq-interpolated
  `F(X)` over the party layout `[dummy | predicates | accumulators |
  pads]` with Corollary 1's no-cross-term trick; the masked sum-check
  (contribution #2); the error commitment `E`; the decider.
- Chains (multi-step accumulation with per-step deciders) and six-way
  tamper coverage (instance `x`, `E`, sum-check rounds, `v_g`, witness
  messages, mask claims) in tests.

### §4 — the ZK-PCD construction (`pcd`)
- The **two-circuit split**: `R^(0) = R_φ` (the predicate over the
  witness) proven by the (non-ZK) SPS-NARK `π^(0)` which is **never
  transmitted** — only accumulated into `acc^(0)` — and `R^(1) = R_V`
  (the accumulation verification over public data). The final verifier's
  `b₀ ∧ b₁ ∧ b₂`: the R_V re-run, the decider on the predicate
  accumulator, and the incoming edges' deciders.
- DAG merge semantics: `accumulate` takes independent
  predicate/accumulator counts, so a node folds its `m` incoming
  accumulators against its own fresh predicate pair at the chain-fixed
  `L = ⌈log₂(2 + arity)⌉`.
- The PCD proof carries `π^(0).x` — the predicate instance's *hiding
  commitments* — inside the R_V bundle, exactly as the paper's
  `π^(1) ← NARK.P(…, (z, π^(0).x), …)` prescribes: zero-knowledge
  without a ZK-NARK.

## Benchmarks (release, this container)

| relation | size | prove | verify | decide |
|---|---|---|---|---|
| R1CS (d=2, µ=1) | s=2 t=6 rows=4 | 5.7 ms | 1.3 ms | 1.9 ms |
| R1CS | s=8 t=24 rows=16 | 26.7 ms | 4.1 ms | 11.1 ms |
| CCS (d=3..6) | rows 4..8 | 6.4–14.8 ms | — | — |
| permutation (d=4..12, µ=2) | n 4..12 | 4.2–8.0 ms | — | — |
| ZK-PCD chain (arity 2) | depth 2/4/8 | 39/78/157 ms | ~12 ms | — |

## The honest-deviation ledger

1. **The fresh error's commitment (Resolution 1).** The paper's verifier
   listing checks `E = Σⱼ eq·Eⱼ` with a `⊥` at the dummy's E-slot, which
   forces the dummy's error to zero and contradicts the "random dummy"
   sampling (completeness would fail for degree ≥ 2 maps). The consistent
   reading — implemented here — commits the *public* fresh error `ẽ`
   unblinded (`Com_pub(ẽ) = Σ ẽ_c·G_c`, binding, verifier-computable) and
   transmits the dummy's error commitment `C^e₀` in `pf` (the "one more
   proof-instance pair" overhead the paper itself cites). Completeness,
   the decider, and the extractor's binding argument close under this
   reading.
2. **Kernel claims (Resolution 2).** With individual-degree-`D` masks the
   update identity `Σ_b eq(β,b)·G(b) = G(β)` holds only for the
   **multilinearization**; the accumulator's stored claim is therefore
   the kernel value `MLE(G|_cube)(β)` throughout (prover, verifier,
   decider), and the update statements run over the masks'
   multilinearizations. The fresh mask's *true* evaluation `G'(β)` is
   transmitted separately (it enters Eq. (5)'s `ẽ` extraction).
3. **Mask degree.** The mask's individual degree is the *statement's*
   degree `d+1` (map degree plus the multilinear eq factor) so the round
   messages are fully blinded — the paper's "same variables and
   individual degrees as f" convention applied to the batched statement.
4. **Pad positions.** The paper's party count `2m+1 < 2^L` leaves
   unassigned cube positions (glossed there). Pad positions carry
   **copies of the dummy** so their map values equal `e₀`, `F̃` vanishes
   on the whole cube, and the E-fold closes with the public weight
   `w₀ = eq⁰(β) + Σ_pad eq_pad(β)` on the single transmitted `C^e₀`.
5. **The `R^(1)` accumulator.** Accumulating the R_V pairs through the
   SPS framework requires *arithmetizing the random oracle* inside the
   recursive relation (circuit-friendly hash) so the challenge
   consistency and tuple checks become algebraic — the paper's circuit
   regime, out of scope for a protocol-level implementation. The R_V
   NARK is instantiated as the **transparent argument** (the bundle is
   re-verified directly); the construction's structure — the split,
   `π^(0)` never transmitted, `b₀ ∧ b₁ ∧ b₂` — is preserved exactly, and
   the ZK contribution is fully live (the predicate witness only ever
   enters the hiding accumulator).
6. **The commitment instantiation.** Vector Pedersen over BN254 G1 —
   the paper's own cyclic-group assumption — is *not* post-quantum. The
   PQ route (Ajtai commitments with shortness/norm-budget discipline, à
   la LatticeFold+/Symphony in this workspace) changes the accumulator's
   algebra (field-scalar folds must become shortness-preserving); it is
   a separate implementation program. **[DONE — the follow-up wave]**:
   `lattice-pcd/ajtai_fr.rs` + `lattice-pcd/pq.rs` implement the PQ
   commitment layer over the scalar field (the d = 1 module, seeded
   uniform matrices, 16-bit digit layers at radius `2^16−1`, MSIS
   dimension rule `rows/cols ≤ 1 − 16/254`), the `Com_pub(ẽ)`
   digit-regime instantiation, the E-fold's homomorphic closure over
   the Ajtai layer (verified), the LatticeFold-style norm ledger with
   re-decomposition points, the decider-side opening with the norm
   check, and the double-opening MSIS kernel extraction — 8 tests.
7. **The `δ`-power check compression** (§5's optimization paragraph:
   compressing the `n` map outputs into one via `pow_j(δ)` weights) is
   not implemented — the vector-valued sum-check (the paper's primary
   `tr` structure) is.
8. **The ZK simulator** of Appendix A is realized distributionally (the
   dummy + fresh masks + hiding commitments make the accumulator's
   public parts uniformly random by construction); an explicit
   `Sim` transcript generator is future work.
