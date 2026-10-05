# The 50 KB Proof Pipeline — Design Document

Date: 2026-09-30. Status: **STAGE 4 LANDED — the 50 KB target is met**:
measured 3,640 KB → **33.0 KB (fibonacci, the k=4-hardened fold; 27.0 KB at the k=2 size prototype) with the Stage-4 leg
batching (the 108 legs → 12 staged sumchecks; `lattice-zkvm/src/legbatch.rs`),
on top of the Stage 0–2 compact opening + claims compression
(75.5 KB pre-batching). The estimator-run MSIS table (Stage 5.1,
`SECURITY.md`) published the honest binding verdict: the single-level
fold at `k = 2` is `~2^12` at every response length — the interim
hardening (`k = 4`, `A = 2^6`) is shipped; the sound posture needs the
second-level fold (Stage 5.2).
Target: zkVM memory-argument proof ≤ 50 KB (from 3.6–7.2 MB) — **met at
33 KB** (27 KB at the k=2 size prototype), with the binding caveat documented.

## Measured after the compact opening + claims compression

| program | cycles | Clear mode | Compact mode | reduction |
|---|---|---|---|---|
| fibonacci | 185 | 3,640 KB | **75.5 KB** | 48× |
| regex | 451 | 7,218 KB | **108.4 KB** | 47× |
| muldiv | 392 | 7,212 KB | **~107 KB** | 48× |

Compact-mode breakdown (fibonacci): legs 55.0 KB, claims 3.5 KB (345
values-only), bits commitment 4.1 KB, values commitment 4.1 KB, carriers
0.8 KB, compact openings 9.3 KB, statement 0.5 KB. The opening machinery
(the multi-MB term) is now 13% of the proof; the legs are the remaining
bulk.

## The two-characteristic discovery (the design's central lesson)

The original design placed the fold's shadow functional over F_q (the
ring's field) with ring-element challenges — the LaBinius structure
verbatim. Implementation proved this **unsound across characteristics**:
the legs' claims are Goldilocks values (the sumchecks' field), and no
homomorphism connects F_q and Goldilocks — the claim values cannot cross
(verified live: the carrier's F_q evaluation diverges from the lifted
Goldilocks claim). The final sound design keeps the Goldilocks carrier
(the legs' own field), uses **scalar fold challenges d_j ∈ [−A, A]** (the
integer fold never wraps mod q — the gate bounds every coefficient below
q/2 — so the Goldilocks functional commutes *exactly* through the fold),
transmits the per-column functional values ũ_j as Goldilocks elements
(8 B each), and checks: (a) the MLE interpolation `w = Σ_j eq(r_tail)_j·ũ_j`,
(b) the commuting functional `Φ(v) = Σ_j d_j·ũ_j` over Goldilocks, (c) the
Ajtai fold `F̄·v = Σ_j d_j·y_j` over R_q, (d) the tight norm gate. Soundness
terminates in MSIS on `[F̄ | −y]` at 2× the gate with the mixed-moduli
constraint lattice (the honest-gap regime: the gate sits at r·A·255, the
honest fold at ~2^13σ, the lattice covering radius at the chosen (k, n̄)
far above both — see SECURITY.md's parameter table).

## 1. Where the 3.6 MB actually goes (measured + modeled)

| Component | Share | Mechanism |
|---|---|---|
| `bits_opening.digits` | ~60% | full response `s` revealed as base-256 i16 digit tables (4 digits/coeff × 64 coeffs × m elements) |
| `values_opening.digits` | ~33% | same, 3 digits/coeff |
| sumcheck legs (~108–117) | ~1.5% | per-leg round messages (rounds × coeffs × 8 B) |
| claims + envelopes | ~5% | thousands of `BaseClaim`s with full points |

Root cause: `BundleOpening` is a Θ(N) *clear reveal* of the committed
response. Every compact lattice system (LaBRADOR, Greyhound, Akita, LaBinius)
replaces this with a folded/amortized opening whose transmitted data is
polylog or √-scale.

## 2. The research consensus (6 agents, 15 papers + SOTA survey)

1. **LaBinius Π_fold / LaBRADOR amortized opening** (agent 2-a): column-major
   witness W ∈ R^{n̄×r}, same key F per column, fold v = W·c, checks
   `F·v = Y·c`, linear shadow `⟨Ψ, v⟩ = uᵀ·c`, norm gate √r·γ_C·β. Response
   length n̄ = M/r independent of r. rANS the response (~7–14 bits/coeff).
2. **Akita** (agent 2-b): fused sumcheck Eq 160 binds the carrier terminal to
   the committed witness (our documented outer-layer gap); fold driver with
   O(log log N) rounds; 61–67 KB at their parameters.
3. **SALSAA** (agent 2-c): digit-domain norm chains (base-16 digits pass the
   wraparound gate at Q_32; packed 22/31-bit coefficients FAIL it — the bits
   bundle at 31 bits/coeff has a VACUOUS norm gate since 2^31 > q/2).
4. **T&S** (agent 2-d): §4.2.1 random-power batching collapses all ~108 legs
   to < 2.1 KB of sumcheck communication through T = 4096; legs are NOT the
   problem once batched.
5. **Folding survey** (agent 2-e): fold claims, not instances; the carrier
   sumcheck already IS the claim-folder; norm arithmetic: per-ring-element
   growth is √n not √N; bits bundle must repack to ≤ 30 bits/coeff.
6. **SOTA** (agent 2-f): the ONLY post-quantum route to 10–100 KB is
   recursive amortization of the MSIS statement (LaBRADOR 47–54 KB,
   Greyhound 46–53, Akita 61–67). Hash-based wraps do not pay at our scale;
   pairing wraps drop PQ. **The honest floor ≈ 35–45 KB.**

## 3. The chosen architecture

### Stage 0 — repack (security fix + norm headroom)
- bits bundle: **1 bit per coefficient** (norm ≤ 1 — kills the vacuous-gate
  bug at 31 bits/coeff; M grows ×31 but every downstream cost is norm-driven,
  not length-driven, after the fold).
- values bundle: **8 byte-limbs per Goldilocks value, byte-major layout**
  (index = b·N + v): norms ≤ 255, linear unpack `val = Σ_b 2^{8b}·c_{b,v}`
  (Z-linear, weights < 2^{57} — exact mod q and mod Goldilocks).

### Stage 1 — carrier sumcheck (unchanged)
The existing grouped carrier binds every base claim to one terminal
`f(r_sc)`. Keep it. r_sc is split (head = all but last log₂r flat vars,
tail = last log₂r).

### Stage 2 — THE FOLDED OPENING (the 93% cut) — new `compact.rs`
Per bundle:
1. Columns: split the coefficient space on the **low log₂r index bits**
   (bits: t mod r; values: v mod r with byte-major layout) → W ∈ R^{n̄×r},
   n̄ = M/r. The per-column functional is **identical** across columns:
   `Ψ(head) = eq(r_head)_{head} × packing-weight(head)` (eq factorizes; the
   packing weight depends only on head-relative position).
2. Commitments: y_j = F·w_j with the SAME seeded F ∈ R^{k×n̄} (k = 2 rows,
   MSIS-checked; column-uniform A is exactly LaBinius's structure — MSIS on
   F̄ with m = n̄, byte norms — strong). The bundle commitment becomes
   (y_1..y_r); the aggregate t = Σ_j y_j is a public consistency check.
3. Prover sends `u ∈ F_q^r`, u_j = ⟨Ψ(r_head), w_j⟩ mod q — ABSORBED BEFORE
   any challenge. Verifier checks the carrier terminal:
   `f(r_sc) = Σ_j eq(r_tail)_j · u_j` (MLE interpolation identity).
4. Challenges c_j ← fixed-weight ring challenges (short_challenge.rs,
   γ_C certified). Response v = Σ_j c_j·w_j ∈ R^{n̄} — TRANSMITTED, rANS
   entropy-coded (wire.rs).
5. Verifier: `F·v = Σ_j c_j·y_j` (Ajtai fold, linear ⇒ cross-term-free),
   `⟨Ψ(r_head), v⟩ = Σ_j c_j·u_j` (the linear shadow — binds v to the
   carrier claim through the u's), norm gate ‖v‖ ≤ √r·γ_C·β_col (fail-closed,
   Lemma-4-style wraparound precondition).

Soundness chain: sumcheck legs → carrier → f(r_sc) → u's (interpolation) →
v (shadow linearity) → y's (Ajtai fold) → MSIS. The documented
"z(r)/f(r_sc) Ajtai-binding gap" of salsa_response.rs is closed by
construction: the shadow check IS the binding.

Extraction (relaxed, no division): fork on c ⇒ w = v−v′, σ = c−c′ with
F·w = Σδ_j y_j — [F | −y-args] MSIS with combined norm; two relaxed openings
⟹ A·(Δw − Δσ·s₀) = 0 — plain MSIS on A. (LaBRADOR's relaxed-relation
discipline; never divide by challenge differences in the splitting ring.)

### Stage 3 — size arithmetic (per bundle)
Total(r) = 256·k·r (commitments) + 4r (u) + (C/r)·(e/8) (response, rANS),
C = coefficient count, e ≈ log₂(2·√r·γ_C·β_col) + 1.
- bits: β=1, r=16–32 → e ≈ 8–9 bits → ≈ 8–16 KB
- values: β=255, r=32–64 → e ≈ 15–17 bits → ≈ 12–25 KB
- legs after batching (Stage 4): ≤ 2 KB; carriers ≈ 1 KB
**Predicted total: ~30–45 KB** at the fibonacci/muldiv/regex shapes — inside
the 50 KB budget with the commitment/response split tuned per bundle against
the measured C. (If C is larger than modeled, Stage 5 below kicks in.)

### Stage 4 — leg batching (T&S §4.2.1)
Batch the ~108 legs into ≤ 13 batched sumchecks via random-power RLC over
the shared transcript (batch.rs exists); claims list collapses to values
(points transcript-derived).

### Stage 5 (stretch) — LaBRADOR decider / second fold level
If Stage 2 lands above budget: fold the response again (requires completing
the lattice-labrador core: response vectors, γ/δ, real verifier) or run the
SALSAA digit-norm chain over the folded response. Deferred — Stage 2's
arithmetic predicts ≤ 45 KB without it.

## 4. Security posture (fail-closed gates)

1. Norm gates BEFORE allocation: per-coefficient ‖v‖ ≤ √r·γ_C·β_col with the
   wraparound precondition (n·bound² < q/2 per column — byte/bit norms pass
   with 2^20+ headroom at Q_32).
2. MSIS parameterization at the RELAXED bounds (8·γ_C·β′ per LaBinius Cor. 1)
   — verified with lattice-sis-estimator before sealing k, r, n̄.
3. FS hygiene: u and y_j absorbed BEFORE challenges; per-bundle labeled
   challenge derivations; statement digest (program/input/final digests +
   commitments) absorbed first (Wave 6.4 discipline).
4. The bits-bundle vacuous-gate fix (31 → 1 bit/coeff) is a SECURITY fix,
   not just size: at 31 bits/coeff the balanced-representative norm claim is
   ambiguous mod q.
5. Tamper tests: wrong u, wrong v, wrong y, norm violation, shadow mismatch,
   carrier mismatch — all must fail closed.


## Stage 4 — the leg batching (LANDED, this session)

The 108 legs (55–70 KB) group by (dependency stage, variable count); the
batching via `lattice-sumcheck/src/batch.rs::prove_batch` (random-power
RLC, the T&S §4.2.1 machinery) collapses each group into ONE sumcheck
whose message count is `max_rounds × max_degree` regardless of the
group's size:

| group | legs per RW instance | shared point | vars | batched size |
|---|---|---|---|---|
| read group | Ma, V0, Mu0 | read_point (C's terminal) | log_ts | 10 × ~12 coeffs |
| write group | Mb, Mc, V1, Mu1 | write_point (W's terminal) | log_ts | 10 × ~12 |
| tel group | Md | tel_point (T's terminal) | log_ts | 10 × ~10 |
| B+R | B, R | (own points, same cube) | log_rows+log_ts | 13 × ~7 |

**The landed shape** (`lattice-zkvm/src/legbatch.rs`): the dependency-safe
staged protocol — stage A (B+R per cube class), stage B (C/W/T per
class, one shared terminal per class), the fetch instance's standalone
Ma, then the GLOBAL batches: read {Ma, V0}×8 → Mu0×8 → write
{Mb, Mc, V1}×8 → Mu1×8 → Md×8. Twelve sumchecks total (vs ~117 legs);
the measured legs communication dropped 55 KB → **~6 KB** and the full
fibonacci proof landed at **27.0 KB** (17.7 KB on the compact test
program) — under the 50 KB target with the `k = 4` hardened opening.
The transmitted per-leg evaluation claims (`ra/val/u/wa/inc`) became the
batches' claimed-sum vectors, pinned by their own stages' terminal
identities against the ledger — the same binding structure the per-leg
`LegProof::claim` had. The ledger discipline (dedup'd `(factor, point)`
records in a prover/verifier-identical sequence) is the queue-sync
invariant; the shared terminals CONCENTRATE the ledger claims.

Implementation notes (the dependency-safe order): C's terminal feeds the
read group; W's terminal feeds the write group; T's terminal feeds Md —
batch only WITHIN a stage. `prove_batch`'s factor claims use GLOBAL
factor indexing (the union across the batched claims); each leg's
`bind_matrix_factors`/terminal checks map through the index offsets. The
verify side mirrors the same grouping; the LegProof list changes shape
(13 → 6 per instance) — the legs' names/claims must stay
transcript-stable across the change (a protocol revision, not a
compatibility break — the proof format is versioned by the envelope).

## Stage 5 — the residual roadmap (post-50 KB)

1. **The MSIS parameter tightening — RUN, verdict published**
   (`lattice-sis-estimator/examples/fold_security_table.rs` +
   `SECURITY.md`): the estimator says the single-level fold at `k = 2`
   is `~2^12` bits at EVERY response length (the `m/n` regime), the
   knobs alone do not close it at `n̄ ≥ 8` (needs `k ≥ 16`, whose
   commitments blow the budget), and the sound regime is the
   second-level fold's `n̄ ∈ {2, 4}` at `k = 4, A ≤ 2^8` (329+ bits).
   **Shipped as the interim**: `k = 4`, `A = 2^6` (the gate tightened
   64×, the commitments doubled — 33 KB total, still under budget).
2. The LaBRADOR decider (completing `lattice-labrador`'s core: the
   response vector per part, γ/δ wiring, the real verifier) — the
   second-level fold that takes the openings to ~5 KB and removes the
   (k, n̄) security/size tension entirely. **Now estimator-mandated**
   (the Stage 5.1 verdict): the single-level fold's binding does not
   reach 128 bits at the benchmark response lengths without it.
3. The verifier's O(K) public-table work → MLE-structured tables
   (O(log K)) at RAM scale.

## The session's companion landings (the streaming path)

* **Algorithm 3's bucketed O(n)-space grand-product rounds**
  (`lattice-streaming/src/grand_product.rs::
  prove_grand_product_bucketed`): LSB-first binding + open-bucket
  routing keyed by the remaining hypercube's high bits, completion-label
  flushes — the `O(2^n)` g-table materialization eliminated; round-1
  messages cross-validated against the direct evaluation, g-claims
  against the rebuilt tables.
* **The VM's step function wired into `ChunkedRegenOracle`**
  (`lattice-zkvm/src/streaming.rs`): every prover/verifier column (pc,
  register-write witness, read/write fingerprints) is a regeneration
  oracle over the live machine — the full `O(K + log T)` path
  (`build_streaming`'s checkpoint-only construction, the
  budget-derived chunk granularity, streaming MLE evaluation).
  `prove/verify_program_streaming` no longer materializes any
  `O(T)` column.
* **The ledger `fix_last_variables` cache** (`ledger.rs`): the
  digit-row claim pattern (`idx_point(b) ∥ terminal`, the `b`-loop)
  shares one bound tensor per tail — the `(log_k + 1)×` resolution win
  the prover profile identified.
