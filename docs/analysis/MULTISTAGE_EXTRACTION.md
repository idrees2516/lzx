# The Multi-Stage LaBRADOR Extraction — The Degree-Law Unwind

**The formal analysis of the recursive width-collapse chain's
knowledge soundness.** This document closes the honest residual
recorded in `lattice-widthfold/src/chain.rs` and NEXT_STEPS's ledger:
"the full multi-stage extraction — the LaBRADOR special-soundness
degree law composed ACROSS stages, unwinding the chain to the level-1
response — is the open analysis."

The executable half of this document is
`lattice-widthfold/src/extraction.rs` (`chain_extraction_ledger`,
`assert_extraction_sound`) — **every numeric claim below is that
module's output** (`cargo run -p lattice-widthfold --example
extraction_table`), re-derived fail-closed at prove AND verify time by
`assert_sound_chain`. No number in this document is hand-typed.

---

## 1. The object under analysis

The chain protocol (`chain.rs`) proves, for a level-1 response
`v ∈ R^{n̄}` committed as `t₁ = F̄·v` with functional claim
`u₁ = Φ(v)`:

> **the staged statement**: there exist per-stage responses
> `z_ℓ ∈ R^{w_ℓ}` (the last transmitted, the rest bound by their
> stages' instances) such that every stage's `(W0)–(W4)` identity
> holds and the public claims thread: `t_{ℓ+1} = Σ_i γ_i^{(ℓ)}·T_i^{(ℓ)}`
> (degree 1 in the stage challenges), `u_{ℓ+1} = Σ_i (γ_i^{(ℓ)})²·u_i +
> Σ_{i≠j} γ_i^{(ℓ)}γ_j^{(ℓ)}·g_ij` (degree 2).

Stage parameters `(r₂^{(ℓ)}, A₂^{(ℓ)}, κ^{(ℓ)}, w^{(ℓ)})` with gates
`β_{ℓ+1} = r₂^{(ℓ)}·A₂^{(ℓ)}·β_ℓ` and per-stage challenge spaces
`C_ℓ = [−A₂^{(ℓ)}, A₂^{(ℓ)}]^{r₂^{(ℓ)}}`, `|C_ℓ| = (2A₂+1)^{r₂}`.

## 2. The per-stage extraction (the LaBRADOR degree law, stage-local)

**Lemma 1 (stage-local special soundness).** Fix a stage ℓ and its
pre-challenge material `(p_i, G_ij, T_i, u_i, g_ij)` — all absorbed
into the transcript before the challenges are drawn. From **two**
accepting transcripts differing in the challenge vector
(`γ ≠ γ′`, any coordinate), one extracts:

* the **response kernel**: `z − z′ = Σ_i (γ_i − γ′_i)·s_i` with
  `‖z − z′‖_∞ ≤ 2·β_{ℓ+1}` (each response is gated ≤ β_{ℓ+1}), and
* the **instance kernel** on `[A₂^{(ℓ)} | −T^{(ℓ)}]`: the
  `T_i`-differences and the `(W2)` identity yield a short vector
  `x ≠ 0` with `‖x‖_∞ ≤ 2·β_{ℓ+1}` and `[A₂ | −T]·x = 0`.

*Why width 2 suffices — the degree law, stage-local form:* the stage's
response is **affine** (degree 1) in its γ's *because every quadratic
term is transmitted pre-challenge as garbage* (`G_ij` for the images,
`g_ij` for the functional). This is the LaBRADOR discipline: a prover
whose response were degree-d in the challenge would require d+1
transcripts to solve for its coefficients; the garbage pre-commitment
pins d = 1, so the extractor needs exactly `d+1 = 2`. The fork width
is a property of the protocol's degree discipline, and this codebase's
folds keep it at 2 by construction (all quadratic cross-terms are
pre-committed) — `StageExtraction::fork_width` is asserted equal to 2
by law E2.

**The estimator layer (modeled, not proven).** The kernel instance's
hardness is the MSIS verdict `msis_bits(κ_ℓ, w_ℓ + r₂_ℓ, q, n,
2·β_{ℓ+1})` — the ADPS16 core-SVP model the offline estimator rates.
The chain's fail-closed posture requires every stage's verdict ≥
`SECURITY_FLOOR_BITS (128) + CHAIN_GRINDING_BITS (32)`.

## 3. The composition (the AND-composition that ships)

**Lemma 2 (cheat detection, one stage suffices).** A forger who
produces two full accepting proofs that differ at stage ℓ (and agree
on the pre-ℓ transcript prefix) yields Lemma 1's kernel on stage ℓ's
instance. Contrapositive: under the estimator's hardness model, no
efficient forger produces accepting forks at any stage — the chain's
binding is the conjunction over stages, each gated. This is the
cheat-detection route the chain shipped on before this analysis; it is
unchanged and restated here as ledger law **E1**.

## 4. The degree-law unwind (the composed analysis)

**Lemma 3 (the composed claim degree).** The derived-claim maps
compose: after L stages, the level-1 target `t₁` and functional `u₁`
are pinned by identities of degree **L** and **2L** respectively in
the staged challenges `(γ^{(1)}, …, γ^{(L)})`:

```text
t_L = P_t(γ^{(1..L)}),  deg P_t = L     (one γ-linear layer per stage)
u_L = P_u(γ^{(1..L)}),  deg P_u = 2L    (a γ² layer per stage)
```

The unwind degree `2L` is ledger law **E2** — asserted, not assumed.
The subtlety the degree law exposes: **the composed CLAIM is degree-2L
while the per-stage EXTRACTION stays affine** (Lemma 1). The rewind
extractor forks each stage once (width 2, depth L — `2^L` leaves, law
**E5** caps the tree at `2^16`); it never needs to solve a degree-2L
system, because the quadratic terms of each stage are already
pre-committed and the derived claims are *linear* functions of the
next stage's transmitted material.

**Lemma 4 (the unwind norm law).** The telescoped level-1
reconstruction carries

```text
‖v̂₁‖_∞ ≤ 2·β_{L+1},    β_{L+1} = Π_ℓ (r₂^{(ℓ)}·A₂^{(ℓ)})·β₁
```

(the fork's 2× factor on the final gate; the per-stage gates grow
`β_{ℓ+1} = r₂·A₂·β_ℓ` exactly because each fold's response is a
γ-weighted combination of its input's parts). Wraparound-free balanced
reading of the reconstruction requires `2·β_{L+1} < q/2` — ledger law
**E3**, the composed form of the same discipline as D1's Lemma-4 gate,
lifted from one commitment to the whole chain. The stages' own
`β < q/2` gates imply it transitively; the ledger asserts it
explicitly so the composition cannot silently erode the slack
(`norm_slack_bits` is printed per schedule — at the β₁ = 2^15
benchmark rows it is already down to ~2^8.6, the honest Q_32 ceiling
the Modulus-50 class addresses).

**Theorem (multi-stage knowledge soundness).** Under (i) the MSIS
hardness model at every stage's kernel instance (≥ floor + grinding
bits), (ii) the Fiat-Shamir transform's rewinding soundness with the
grinding allowance, and (iii) transcript collision resistance:

```text
Pr[forgery] ≤ Σ_ℓ 2^{−msis_ℓ}  +  2^{−grinding_bits_total}·(work factor)
```

and the knowledge extractor E_chain (fork every stage once; abort
probability ≤ Σ_ℓ |C_ℓ|^{−1} — the ledger's `composed_kappa_bits`)
either aborts, or outputs per-stage kernels (Lemma 2's witnesses) with
the level-1 claims pinned by the degree-2L thread (Lemma 3) within the
norm law (Lemma 4). The ledger reports the knowledge gap and the
binding security as **distinct quantities** — the gap
(`2^0.6`–`2^3.2` bits on shipped schedules — the rewinding
completeness) is NOT the forgery security (the MSIS floor, 161–841
bits on shipped schedules); conflating them would be the analysis
error this document exists to prevent.

**Law E4 (the grinding ledger).** Replay-grinding all L stages costs
`Σ_ℓ log₂|C_ℓ|` hash trials — the 32-bit `CHAIN_GRINDING_BITS`
allowance already charged against every stage's floor. On shipped
schedules it measures 3.2–31.7 bits; over-deep fabricated schedules
fail closed here first (~3.2 bits/stage accumulates).

## 5. The three honesty layers (what is proven, modeled, assumed)

| Layer | Contents | Status |
|---|---|---|
| **Arithmetic** | the degree law (2L, fork width 2), the norm law (`2β_{L+1} < q/2`), the grinding ledger, the tree size, the kernel bounds | **Proven + machine-checked** (`assert_extraction_sound`, laws E1–E5) |
| **Modeled** | the MSIS hardness of each kernel instance | the offline ADPS16 estimator — a model, not a theorem |
| **Assumed** | FS rewinding soundness with the grinding allowance; transcript collision resistance | standard assumptions, stated not proven |

## 6. The measured ledgers (the shipped schedules)

`cargo run -p lattice-widthfold --example extraction_table` — the
boundary rows of the coverage table:

| schedule | L | tree 2^L | unwind deg | grinding | min MSIS | unwind norm vs q/2 |
|---|---|---|---|---|---|---|
| n̄=16, β₁=2^8 (the single-stage terminal) | 1 | 2 | 2 | 3.2b | 841.4b | 2^20.6 slack |
| n̄=512, β₁=2^8 (the D4 byte-witness scale) | 5 | 32 | 10 | 19.0b | 177.5b | 2^15.6 slack |
| n̄=4096, β₁=2^8 (the byte-gate boundary) | 7 | 128 | 14 | 31.7b | 284.4b | 2^12.6 slack |
| n̄=512, β₁=2^15 (the benchmark stream) | 6 | 64 | 12 | 19.0b | 173.7b | 2^8.6 slack |
| n̄=2048, β₁=2^15 (the r₁=8 boundary) | 7 | 128 | 14 | 25.4b | 161.5b | 2^6.6 slack |
| n̄=128, β₁=2^19 (the packing cap) | 4 | 16 | 8 | 12.7b | 208.4b | 2^6.6 slack |

(The last row's slack is the honest Q_32 ceiling in its plainest form:
the norm law, not the machinery, binds — the Modulus-50 class is the
documented headroom route.)

## 7. What this analysis deliberately does NOT claim

* It does not claim the knowledge gap is cryptographically small — it
  is a rewind-completeness quantity and is reported as such.
* It does not claim a tight constant in the unwind abort analysis; the
  union bound `Σ|C_ℓ|^{−1}` is used as-is.
* It does not upgrade the estimator's model (classical/quantum MSIS
  costs remain the modeled layer).
* The level-1 `[F̄]` equation is *proven* by the fold's (W0) — the
  analysis never assumes the level-1 key's own MSIS hardness (this is
  what lets the D4 closure bind byte-witnesses wider than the level-1
  instance's own comfort zone; see `docs/analysis/D4_BINDING_CLOSURE.md`).
