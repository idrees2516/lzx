# LZX Next-Implementation Research

**Per-paper gap analysis: what must be implemented next in each of the 11 papers
(+ the labinius port and the zkVM stack) to make the system maximally efficient,
performant, and security-optimized.**

Research wave 6-prep, 2026-09-24. Method: each paper was read end-to-end and
cross-examined line-by-line against its crate in this workspace, with the
reference repositories (`repos/{akita,rokoko,zem,jolt,labinius}`, read-only)
used as ground truth for what a complete implementation looks like. Every claim
below carries a paper-section or file:line citation. This document is the
complete, prioritized implementation backlog; it intentionally does not soften
findings.

Companion documents: `PERFORMANCE.md` (the labinius-parity performance audit),
`AUDIT_CHECKLIST.md` (G1-G8 evidence map), `SECURITY.md`.

---

## 0. Executive summary — the brutal bottom line

1. **The algebra kernels are exact and well-tested (256 tests, 0 clippy
   warnings), but the majority of the paper protocols are missing their
   verifier-facing halves.** ProtogaLattice, Symphony, SuperNeo, Cyclo,
   LatticeFold+, PikkuFold, Quasar, RoKoko, and the zkVM's Twist/Shout path
   compute *fold identities* or *prover-side transforms* that a malicious
   prover is never forced to satisfy: there is no verifier equation consuming
   the cross terms, no randomized relaxation point, no committed lookup
   arguments, and (in the zkVM) the verifier literally re-executes the program
   and checks the sumcheck section **by byte length only** (`prove.rs:276`).
2. **The single dominant soundness defect across the folding family is the
   challenge space.** Every folding module samples small *integer* scalars
   (8-31 bits), while every paper uses short **ring-element** challenges
   (fixed-weight ternary with operator-norm rejection, |C| ≈ 2^100+). Knowledge
   error today: 2^-8 (PikkuFold) to 2^-31 (best case) — ~90 bits short of the
   papers at λ=128. This is one shared module away from fixed.
3. **The two LZX-native PCS crates are kernel shells of their papers.** Akita
   implements a sumcheck glued to a full-witness Ajtai opening (Θ(N) proofs,
   Θ(N) verification, full witness disclosure); HyperWolf is a trait object
   around the same full-witness opening. Neither implements its paper's
   central construction (the recursive fold / the guarded IPA), so neither
   earns the logarithmic proof size, the amortization, the parameter regime,
   or the standard-soundness theorem that defines the scheme.
4. **Twist & Shout's protocol machinery is 0% implemented**: `lattice-memory`
   contains grand-product fingerprints — the exact technique the paper
   eliminates — and `twist_fingerprint` checks final-state consistency only
   (a stale read followed by a consistent final state passes), which makes the
   currently-staged production statement **unsound for memory correctness**.
5. **The zkVM's genuine ZK/Succinctness posture is nil today**: the witness is
   shipped in the clear inside the proof envelope, the sumcheck transcript is
   unverified, the ZK crate is not a dependency of the zkVM crate, and no
   instruction-semantics constraints exist anywhere in the proof path.
6. **What IS strong**: the substrate (field/ring/transcript/NTT/sumcheck
   engine, envelope discipline, QROM ledger framework, VM conformance layer
   with differential testing, the labinius AVX-512 backend with bit-exact
   verification, 17.5x end-to-end on the reference round). The gap is
   *protocol completion on top of an honest substrate* — not a rewrite.
7. **Highest-leverage next implementations** (full detail in §3 and §4):
   a. `lattice-core::short_challenge` — fixed-weight ternary ring challenges
      with op-norm rejection (closes the #1 soundness gap of five papers).
   b. Ajtai cached-NTT commit + Barrett reduction (≈3x on every commit in the
      workspace; removes Θ(N) recompute at every verify).
   c. The SALSAA ring-norm sumcheck + LDE linearization, swapped into the
      Akita/zkVM response layer (turns end-to-end proofs from Θ(N)-and-public
      to polylog-and-private).
   d. Quasar committed grand products (closes an outright soundness hole).
   e. The real Twist & Shout PIOPs (one-hot + increments + three sumchecks)
      and deletion of verifier re-execution.
   f. labinius `Recursive` mode wiring (the port's declared-but-unwired compact
      opening mode).

---

## 1. Methodology

* Papers: all 11 ePrints from `/home/z/my-project/upload/`, extracted to text
  (`research/papers/`; SALSA via OCR). Read end-to-end; mechanism inventories
  extracted with section references.
* Code: every paper-crate read completely (`lattice-folding` 6 modules,
  `lattice-akita`, `lattice-pcs`, `lattice-lookup`, `lattice-salsa`,
  `lattice-rokoko`, `lattice-memory`, `lattice-vm`, `lattice-zkvm`,
  `lattice-zk`, `lattice-qrom`, `lattice-commitment`, `lattice-labinius`,
  `lattice-labrador`).
* Reference ground truth: `repos/akita` (~260k LOC, 16 crates), `repos/rokoko`
  (incl. `incomplete-rexl` 5.2k LOC), `repos/zem` (SALSAA tree ~5.9k LOC +
  Quasar tree ~4.3k LOC), `repos/jolt` (Twist/Shout stage pipeline),
  `repos/labinius` (upstream PCS, 20.8k LOC).
* Three evaluation dimensions per paper: **EFFICIENCY** (asymptotic prover/
  verifier/proof-size), **PERFORMANCE** (constant factors, SIMD, memory),
  **SECURITY** (soundness structure, challenge spaces, parameters, ZK, QROM).
* Every backlog item: algorithm sketch + impact + effort (S/M/L).

---

## 2. Systemic findings — the eight shared root causes

These cut across papers; fixing them once serves many papers at once.

### 2.1 Challenge-space deficit (soundness, family-wide, CRITICAL)

Every folding module samples balanced *integer* scalars:

| Module | Challenge | Bits | Paper's challenge | Paper entropy |
|---|---|---|---|---|
| pikkufold.rs:165 | int r | 8 | z ← C^k fixed-weight ternary (§5, Table 3) | ≈2^100+ |
| cyclo.rs:224 | int r | 12 | s ← D^L ternary over R_q (§3, App B) | ≈2^203 |
| latticefold_plus.rs:258 | int r | 16 | S̄ = {−1,0,1,2}^d (§4) | ≈2^148 |
| protogalattice.rs:200 | int r | 17 | δ,α,y ∈ C={−1,0,1,2}^N (Table 2) | 2^128 at T=128 |
| symphony.rs:143 | int r | 16 | β ← S^ℓ (LaBRADOR set, ‖S‖op ≤ 15) | 2^128+ |

Consequences: (i) knowledge error bounded by 1/|C| = 2^-8..2^-17 — ~90-130
bits short of λ=128; (ii) the Schwartz-Zippel-over-rings lemmas the papers
rely on (sampling sets with S−S non-zero-divisors) do not apply to scalars;
(iii) norm growth per fold is 2^7..2^15 instead of the papers' ring op-norms
(γ ≈ 8-15), so fold counts are irrecoverably capped.

**Fix (one module, M effort, serves 5 papers):**
`lattice-core::short_challenge` — `ChallengeDistribution::FixedWeightTernary`
(Fisher-Yates positions, ±B values, rejection on operator norm; for
power-of-two rings ‖c‖op = ‖σ(c)‖∞ via one NTT evaluation), with Γ_C exported
into every norm-budget computation. Instantiate per paper:
PikkuFold C^{fw}_{256,23,1}(γ=8.357); Cyclo D (biased ternary); LF+ S̄;
ProtogaLattice C={−1,0,1,2}^N with T=128; Symphony S ({0,±1,±2}, op-norm 15).

### 2.2 Missing modulus classes (blocks paper parameter sets)

| Needed | Who needs it | Status |
|---|---|---|
| q ≈ 2^50, q ≡ 129 mod 256 (e=2, incomplete NTT, quadratic slots) | Cyclo (κ_nu 2^-94 vs 2^-24 today), PikkuFold (ε_C), RoKoko (its whole §9 kernel story), SALSA (CRT slot batching) | absent; `lattice-ring` is u32 full-split only |
| q ≈ 2^128, q ≡ 5 mod 8 | HyperWolf Table 3 (Lemma 1 invertibility), LF+ (S̄ strong sampling) | absent (u128/RNS neither exists) |
| q ≈ 2^64 with 16 CRT fields | ProtogaLattice Table 2, Symphony Table 1 | absent |

The quadratic-slot NTT tree + const bound models already exist in-house
(`lattice-labinius/src/simd/ntt_quad.rs`, `scalar.rs:164`) — the *concept* is
proven; the modulus width is the gap.

### 2.3 Missing extension-field sumcheck layer

Five papers communicate sumcheck rounds as single **F_{q^e}** elements
(subfield batching via θ_a / Φ_δ / CRT slots): Cyclo (§3 "sumchecks over
R_{q^e}"), PikkuFold (RingSC, §2.5), RoKoko (Π^lin, §7), SALSA (Π^sum final
check), Symphony (K = F_{q²} tensor ring). `lattice-sumcheck` is
Goldilocks-hardcoded (`sumcheck.rs:19`). Without it: per-round errors ~d·ℓ/q ≈
2^-28 on current rings (unsound at λ=128), and the few-KB communication
profiles are unreachable.

### 2.4 Norm budgets tracked but never enforced (wraparound breaks)

`norm_budget` fields exist in cyclo.rs:249, latticefold_plus.rs:279,
protogalattice.rs:305 — all informational. With current toy q ≈ 2^31.6 and
2^15-ish challenges, **one worst-case fold wraps the balanced representative
mod q and silently destroys the SIS binding argument** (SIS needs the integer
vector, not its residue). Cyclo additionally counts folds (64) instead of
norms. Fix: a `NormBudget` type with symbolic (β, γ, β̄, δ) arithmetic and a
hard gate `β < min(q/2, β*)` per fold — S effort, family-wide.

### 2.5 Kernel arithmetic: the Ajtai hot path is ~3-6x redundant

`AjtaiPublicKey::commit` (ajtai.rs:97-117) recomputes the **forward NTT of
every matrix entry on every multiplication** (ring.rs:263-285 does
clone+2 forward+1 inverse per product). Fix: cache NTT(A) at keygen, NTT the
witness once, pointwise-accumulate, one inverse per row → ~3x at k=2, more at
higher rank; removes the full recompute in `verify_opening` (Θ(N) per
verification). Plus: `Modulus32::mul` is a hardware `%` per coefficient
(modulus.rs:46-61, every NTT butterfly) — Barrett/Montgomery discipline
(already proven in `lattice-labinius`, `fold.rs`) is absent from
`lattice-ring`, which every folding/PCS crate uses. Combined: S effort,
~5-15x on scalar commit paths before any SIMD.

### 2.6 Protocol-vs-kernel posture (verifiers missing)

`lattice-folding/src/lib.rs:14-17` claims every module exposes "a prover that
proves the fold, and a verifier that checks it" — **true for
`latticefold_plus` (prove_range/verify_range) only**; false for
protogaLattice, symphony, superneo (no verifier exists; superneo's
`verify_folded` needs the clear witness). Quasar's `verify_lookup` checks an
identity on three prover-supplied scalars — binds nothing. The zkVM's
`verify_program` re-executes the program and checks the sumcheck section by
length. The papers' verification equations (Fig 3 e*-check; Fig 4 steps 1-6;
Quasar's multicast consistency check; Twist's three sumchecks) are the
missing layer — this is *the* pattern of the whole codebase.

### 2.7 ZK/QROM wiring

The genuine ZK machinery (`lattice-zk`: blinded sumcheck with committed
masking, ABDLOP zk_linear with correct FS ordering, simulators, chi-square
KATs) is **not a dependency of `lattice-zkvm`** (Cargo.toml:10-25) and is
degree-1-single-vector-only — it cannot wrap product-structure sumchecks.
The ZK cost profile (10 committed slots per witness value) would forfeit
Twist & Shout's entire cost advantage if wired naively; needs sparse/bit-aware
masking. The QROM layer's example "zkvm composition" attestation
(review.rs:147-180) is a 4-stage fiction that matches no real protocol shape.
No production SIS profile is instantiated anywhere; `SecurityProfile`
metadata is review-gated with zero digests.

### 2.8 Missing shared infrastructure

* **8-prime RNS crate** (`lattice-rns`): needed by LaBRADOR polx
  (10-50x prove), RoKoko's CRT commitment (`commitment_crt.rs` PRIMES), and
  any 128-bit modulus strategy. Build once, use thrice.
* **SIS estimator** (ADPS16/BDGL16 Core-SVP): exists in `repos/akita`
  (`akita-sis-estimator`); LZX's `SecurityProfile::classical_bits` is an
  ungated claim field. M effort to port zero-dep.
* **Folding benchmarks**: `lattice-bench` has zero folding/lookup/salsa
  entries. PERFORMANCE.md's roadmap omits the folding family entirely.
* **Sparse/bit-packed commitment inputs**: Ajtai `commit` multiplies zero
  ring elements unconditionally — the papers' "0s are free" doctrine
  (Twist §2.9.2, Neo pay-per-bit) is unrealized; one-hot columns cost full
  matrix work.

---

## 3. Per-paper gap analyses and next implementations

### 3.1 ProtogaLattice (ePrint 2026/1317) — `lattice-folding/protogalattice.rs`

**Implemented**: two-instance Protostar-style fold with exact cross-term
extraction via finite differences on nodes 0..d (the diagonal-snapshot fix is
correct and tested at d=2,3); stacked cross-term commitment; slack-u folding;
prover-side norm budget.

**Gaps**:

* **S1 (fatal)**: nothing binds the committed cross terms to the instances —
  no verify function exists; a malicious prover folds F(w2)≠0 with E_i := 0.
  The paper (Fig 3) sends quotients K_rt *in the clear* and checks
  `e* = Σ K_rt(y)·Z_rt(y) + c_0·F(α)` — Schwartz-Zippel over rings forces true
  quotients (Thm 4).
* **S2 (fatal)**: relaxation point fixed forever (`alpha = 0x0010_0193`,
  slack_of_evals L131-143) — ker(L) is permanently forgeable; the paper
  re-randomizes β* = β + αδ every fold (round 2).
* **S3**: scalar 17-bit challenge vs ring-element C={−1,0,1,2}^N, T=128
  (§2.6-2.11); 1 RO call vs the paper's 3-round structure (δ/α/y are the
  binding mechanism).
* **S4**: no range proofs Π_rg, no bootstrapping (§5.3 Fig 4) — norm grows
  geometrically with no refresh; folding cannot iterate (the scheme's purpose).
* **E1**: k=1 only (no multi-folding / PCD accumulator folding, §4 Fig 2).
* **E2**: (d+1) relation evaluations + O(d²) inversion per fold vs the
  paper's Gröbner-quotient route (~free division by the monomial→Y_max rule,
  Prop 5); per-fold recomputation of falling-factorial/inverse tables (P1).
* **E3**: relation family is Hadamard-slot only — cannot express ring
  products or M·w terms; cannot host CCS statements.

**Next implementations** (prioritized):

1. **P0 (M)**: real protocol — `PgAccInstance{t, beta, e}` + prover/verifier
   split per Fig 3: δ ← C ring challenges, F(X) coefficients, α re-randomized
   β*, Gröbner division for K_rt, y ∈ C^k, verify `e* = Σ K_rt(y)Z_rt(y) +
   c_0 F(α)`; adversarial tests (substituted K rejected).
2. **P0 (S)**: ring-element challenge set + expansion-factor bookkeeping
   (shared module §2.1).
3. **P1 (M)**: bootstrapping (Fig 4): base-b decomposition (reuse
   `cyclo::chunk_element`), dummy zero witness, L_j interpolation, verifier
   checks Σ b^{j-1} t_j = t and Σ b^{j-1} e_j = e + Σ Z_rt(D)K_rt(D).
4. **P1 (S)**: attach range proofs (wire `latticefold_plus::prove_range` or
   the SALSAA norm sumcheck once available) + `decide`.
5. **P1 (S-M)**: cross-term extraction at the k+1 variety points (replaces
   d+1 evaluations; cache the conversion matrix in a OnceLock).
6. **P2 (M)**: multi-folding k>1 + accumulator folding for PCD; ring-product
   relation terms (NTT-based) + CCS bridge; parameter module encoding Table 2
   with `SecurityTier` markers.

### 3.2 Symphony (ePrint 2025/1905) + SuperNeo — `symphony.rs`, `superneo.rs`

**Implemented**: μ-ary degree-2 fold with explicit pairwise cross terms E_ij
(exact identity tested at μ=3,4); stacked cross commitment under a separate
key; SuperNeo relaxed-CCS fold algebra with vacuity probes.

**Gaps**:

* **S1 (fatal)**: no verification path; cross-term commitment never bound;
  `SymphonyFold` carries no claims/evaluations — nothing a verifier could hold.
* **S2**: no norm management at all (no norm fields; 16-bit integer
  challenges); Eq (50) feasibility (`B_bnd ≥ ℓ_np‖S‖op max(...)`) absent.
* **S4 (fatal, SuperNeo)**: the FS challenge is derived from *private
  witnesses* (superneo.rs:87-96 absorbs w1‖w2) — not a public-coin protocol;
  no commitments anywhere; the doc's pay-per-bit/R_q-bridge/π_CCS claims are
  unimplemented.
* **E1 (the defining gap)**: explicit O(μ²) cross-term enumeration vs the
  paper's O(μ) prover (shared-randomness merged sumchecks absorb cross terms;
  Prop 4.2). At ℓ_np = 2^10 the impl's route is ~268 MB vs the paper's <200 KB
  proof — the impl is exactly the §1.2 strawman the paper rejects.
* **E3**: the entire §2-§3 toolbox missing — tensor ring E = K⊗R_q, F_{q²},
  ts(r) eq-tensor, Π_had, Π_mon, Π_rg, R_lin/R_batchlin outputs.
* **E5**: no CP-SNARK compiler (Construction 6.1), no two-layer folding (§8),
  no memory strategy (Remark 4.1: 2+log log n passes, memory ≈ one witness).

**Next implementations**:

1. **P0 (L)**: tensor-ring substrate — K = F_{q²} arithmetic + `TensorElement`
   dual views + ts(r); template `lattice-sumcheck` past Goldilocks.
2. **P0 (M)**: Π_had (Fig 1) — degree-3 sumcheck over K with α-power column
   batching; the first real verifier in the module.
3. **P0 (M)**: replace O(μ²) cross terms with the paper's fold (Fig 4):
   shared (J, s′, α), merge 2μ sumchecks into two via α-power RLC (Eq 45),
   β ← S^μ folding of commitments/evaluations/witnesses (Eqs 48-49).
4. **P0 (S)**: LaBRADOR challenge set S + Eq-(50) feasibility check.
5. **P1 (S-M, SuperNeo)**: commit instances (`CommittedRelaxedCcsInstance`),
   stop absorbing witnesses, split `fold_public`/`fold_secret`.
6. **P1 (M-L)**: Π_rg approximate range proof (Fig 2 — reuse projection
   machinery from pikkufold/rokoko + cyclo chunking).
7. **P2 (M, SuperNeo)**: pay-per-bit sparse Ajtai commit (zero-skip in the
   MAC loop) + π_CCS decider via sumcheck + carrier bridge.
8. **P2 (L)**: CP-SNARK compiler (Construction 6.1) + instance compression.

### 3.3 Cyclo (ePrint 2026/359) — `cyclo.rs`

**Implemented**: exact signed base-2b digit chunking (the only paper-faithful
mechanism); extension commitment as chunk→pad→generic Ajtai commit; fold with
homomorphic commitment; refresh = extension-commit the accumulated witness;
additive norm bookkeeping (recomputed, not symbolic).

**Gaps**:

* **S (severe)**: 12-bit scalar challenge (2^-12 knowledge error; paper's D
  is ternary over R_q, |D| ≈ 2^203); **norm wraparound**: inputs ℓ∞ ≤ 2^20,
  |r| ≤ 2^11 → one fold can exceed q/2 ≈ 2^30.6 — the balanced representative
  wraps mod q, destroying SIS binding; FS hole: the fold transcript absorbs
  only the accumulator commitment, never the input's (cyclo.rs:213-216) —
  grinding/projection attacks on the challenge w.r.t. the input; latent
  truncation bug: after `refresh`, `witness.len() = pk_ext.m` but subsequent
  `fold` zips against `pk.m` — silent truncation, untested.
* **E**: no folding proof at all (paper Thm 3.3: La′+L R_q + (k+2)(L+1)
  R_{q^e} + sumcheck transcripts; impl emits nothing verifier-checkable);
  Π^range is a prover-local `if`, not the degree-(2b+2) sumcheck over
  F_{q^e} with the dual-basis trace lemma; Π^ext lacks the RoK constraint
  rows (Fig 2 steps 3-4) — the expensive component delivers no knowledge
  soundness; **the R1CS-over-F_q bridge (§7, θ_k, "skip Π^ext when k ≤ b") —
  the paper's raison d'être — is entirely missing**.
* **P**: per-pair NTT-per-mul commit, `%` per coefficient, no
  `partial_range_check` savings (computes the full decomposition on both
  branches), plus a 2x soundness slack in the partial branch (untested regime).

**Next implementations**: P0-1 shared short ring challenges (M); P0-2
norm-budget gate vs q/2 (S); P0-3 FS fix — absorb input commitment + params
+ fold counter (S); P0-4 length assertion after refresh (S); P1-5 Π^range as
a real protocol over F_{q^e} with the X³−X Karatsuba trick (M, needs dual
inner product + extension field); P1-6 Π^ext RoK with verifier challenge rows
(M); P1-7 homogenization sumcheck + fold with ring challenges (L); P2-8
50-bit modulus + incomplete NTT (L — port from `ntt_quad.rs` discipline);
P2-9 θ_k bridge for R1CS (L); P3 delete/fix `partial_range_check` (S).

### 3.4 LatticeFold+ (ePrint 2025/247) — `latticefold_plus.rs`

**Implemented**: algebraic range proof — but it is **LatticeFold's superseded
bit-decomposition + booleanity sumcheck**, not LF+'s monomial/ψ machinery;
"double commitment" as a literal duplicate commit (outer = commit(w)); linear
fold under a 16-bit scalar; unenforced norm budget.

**Gaps**:

* **E**: none of the paper's actual contributions exist — no monomial sets /
  `ev_a(β)² = ev_a(β²)` (Cor 4.1), no ψ/Lemma 2.2 `ct(ψ·b) = a` range core,
  no Π^mon O(n)-add evaluation trick (Remark 4.3), no real `split`/`pow`
  double commitments (Constr 4.1), no Π^rgchk, no Π^cm commitment
  transformation (Constr 4.5 — the central new protocol), no R_lin,B, no
  R1CS→R_lin reduction, no fold→decompose pipeline (Π^mlin + Π^decomp,
  Thm 5.1-5.2) — hence no unbounded folding and no path to the ≲100 KB proof.
* **S**: 16-bit challenge vs |S̄| = 5^64 ≈ 2^148 (~132 bits short); the range
  proof lives over Goldilocks, disconnected from the R_q commitment (the only
  link is a caller-supplied unauthenticated `coeff_claim_at_point`);
  vacuous double commitment (no Lemma-4.1 binding surface); wraparound
  (norm 2^22 + |r| 2^15 > q/2 after one fold).
* **P**: dense digit MLE tables (2×num_digits+1 layers materialized; the
  monomial structure would make evaluation O(n) adds); `verify_range`
  recomputes a full eq table to extract one evaluation (O(ν) Horner
  suffices).

**Next implementations**: P0-1 shared challenges (M); P0-2 norm gate (S);
P1-3 ψ/EXP/monomial layer with Lemma-2.2 iff-tests (M); P1-4 Π^mon with the
O(n)-add trick (M); P1-5 real split/pow double commitments with binding tests
(M); P2-6 R_lin,B + R1CS reduction + Π^mlin/Π^decomp fold loop (L); P2-7
Π^rgchk + Π^cm end-to-end (L); P3 constant-factor fixes (S).

### 3.5 PikkuFold (ePrint 2026/1809) — `pikkufold.rs`

**Implemented**: single-layer biased-ternary projection (entry distribution
correct); union-bound ℓ∞ "JL bound" (not the paper's certified ℓ2); fold with
an **8-bit** challenge; `fold_with_binding` — an ABDLOP LinearProof whose
response transmits **m ring elements**.

**Gaps**:

* **E (inverted headline)**: the paper's contribution is *no in-protocol
  commitments*, proofs ≈ 5.5 KB; the impl's "production" path sends an
  O(m)-sized proof — ~10^5× the paper's entire fold at the paper's scale.
  No layering (LRP: d layers, coarse ring lifts + final fine coefficient
  layer, `Tr(Mw) = P·coeff(w)`), no LMLE, no RingSC with subfield batching
  (80% of the paper's prover time and all of its communication profile), no
  evaluation claims (s, t) in the relation, no AIR folding (§7.2).
* **S**: 2^-8 knowledge error (~92 bits short); no JL failure-probability
  statement, no wrap-margin modular condition, the `‖v_tr‖ ≤ ω` gate of Fig 2
  missing entirely (nothing certifies input norms — the projection's entire
  soundness purpose); no β_out/β_bind/ρ_bind/β_sis extraction accounting.
* **P**: dense scalar projection loop; per-relation transcript hashing of
  full coefficient vectors per rejection retry (quadratic at scale); 4x XOF
  byte waste in matrix seeding.

**Next implementations**: P0-1 shared fixed-weight challenges (M); P0-2
certified JL constants (Thm 2 Table 1) + modular checks + ω-gate (S); P0-3
quarantine `fold_with_binding` as test-only (S — stop shipping an
anti-feature); P1-4 layered projection + trace/dual-basis identity (M);
P1-5 RingSC with subfield batching + the Fig 2 protocol (L); P1-6 relation
upgrade with MLE evaluation claims (M); P2-7 norm/SIS accounting + periodic
reset (M); P2-8 AIR folding (L); P3 constant factors (S).

### 3.6 Akita (ePrint 2026/1983) — `lattice-akita`

**Implemented**: packed Goldilocks→R_q commitment (3×22-bit limbs); eq·f
sumcheck evaluation proof; **full-witness opening + digit-revealing NormProof
as the "opening"**; RLC grouped openings for multiple points; schedule
catalog + security profiles as review-gated metadata. Reference repo
(~260k LOC) contains the entire protocol; LZX implements a kernel-scale
demonstrator (~720 lines).

**Gaps** (vs paper §4-§12):

* **E**: proof Θ(N) (witness + 8 digits/coefficient ≈ 512 B per 4-byte
  coefficient) vs the paper's 61-67 KB at 2^30; verifier Θ(N) (full Ajtai
  recommit + O(N) digits + O(N) MLE re-evaluation) vs Õ(N^{1/K}) — the
  headline trilemma breaker absent; no fold (no challenges c_i, no response
  z = Σ c_i s_i, no two-tier A/B/D sliced outer commitment, no digit-
  decomposed source — hence no pay-per-bit), no ring-relation checks
  (quotient lifting / transpose convolution), no digit-range sumcheck
  (symmetry halving, product trees), no exact-ℓ2 certificate, no tensor
  reduction / trace functional, no terminal, no recursion driver, no
  commitment compression (128 B payloads), no cross-commitment batching, no
  response chunking, no planner/validator.
* **S**: no SIS estimate behind `norm_bound` (the reference ships a full
  ADPS16/BDGL16 estimator); q ≡ 1 mod 8 full-splitting cannot host the
  paper's challenge invertibility regime (needs q ≡ 5 mod 8 or certified
  families); FS statement binding is the caller's duty (malleability class);
  transparent-by-reveal (no LHL hiding layer — the reference has
  `lhl_blinding.rs` capacity math for a future ZK layer).
* **P**: 3x-redundant NTT commit hot path; no SIMD (the labinius kernels are
  2.6k lines away and q-width-blocked); `Vec<Vec<i64>>` digit proofs (128x
  blowup); eq tables materialized per claim.

**Next implementations**: A1 the Hachi fold core — two-tier keys, G⁻¹ source
decomposition, sparse challenges, response digitization, fold equations
Eq 7-8, successor witness (L, *the keystone*); A2 ring-relation checks —
quotient lift with the α-after-commitment ordering (the Grand-Danois bug the
paper's App F.1 repairs) + fused relation+range sumcheck (M-L); A3 digit-
range sumcheck with symmetry halving (M); A4 tensor reduction + trace
functional (M); A5 terminal + recursion driver + signed-Rice encoding (S-M);
A6 Ajtai cached-NTT fast path + statement-absorption API (S, ~3x + kills a
malleability class); A7 schedule/planner/validator v1 (M); A8 exact-ℓ2 +
challenge families (M); A9 setup offloading + constant-root verifier (L);
A10 commitment compression F/H (M); A11 batching + chunking (M/L); A12
Goldilocks AVX-512 (L); A13 SIS estimator crate (M); A14 QROM/FS ledger (M).

### 3.7 HyperWolf (ePrint 2025/1903) — `lattice-pcs`

**Implemented**: the `PcsBackend` trait + a backend that packs, commits,
and **reveals the entire witness** with a digit norm proof; the transcript
parameter is ignored (`_transcript`, lib.rs:133) — nothing is Fiat-Shamir.

**Gaps**:

* **E**: proof Θ(N) vs the paper's 52-53 KB at 2^30 (O(log N) core +
  LaBRADOR compaction to O(log log log N)); verifier Θ(N) vs O(log N) ring
  ops; no ring mapping MR + ι-slice gadget decomposition (the witness isn't
  even in the paper's form), no leveled commitment F_{k-1,0}, no k-round
  evaluation folding (tensor-vector products, ct-checks), **no guarded IPA**
  (the split-and-fold norm constraint with the ‖s^(1)‖∞ ≤ γ smallness guard —
  the paper's standard-soundness core), no commitment folding, no challenge
  space C (fixed-weight signed, rejection to T ≤ 10), no LaBRADOR
  compaction, no three-mode batching (the trait signature itself admits one
  claim).
* **S**: "standard soundness" is realized only in the degenerate
  commit-and-reveal sense; the exact-ℓ2 extraction with (2T)^{k-1} slack and
  the wraparound guards are absent; **the parameter regime is unreachable
  today: q ≈ 2^128 with q ≡ 5 mod 8 (Lemma 1 invertibility) vs u32-only
  rings, Q_32 ≡ 1 mod 8**.
* **P**: shared substrate issues (§2.5); no σ⁻¹ conjugation automorphism in
  `lattice-ring` (needed by the ct(⟨f, σ⁻¹(g)⟩) bridge).

**Next implementations**: H1 ring mapping + gadget decomposition + leveled
commitment (M); H2 large-modulus ring layer (u128/RNS, q ≡ 5 mod 8) (M-L,
**blocker**); H3 the guarded IPA with fork-extraction tests (M); H4 k-round
evaluation folding (M); H5 challenge space C (S-M, cross-pollinate with
pikkufold); H6 LaBRADOR compaction via `lattice-labrador` (L, after
re-parameterization); H7 batching — extend the trait (M); H8
`impl PcsBackend for AkitaPcs` — unify (S).

### 3.8 RoKoko (ePrint 2026/575) — `lattice-rokoko`

**Implemented**: seed-derived ternary projection + a two-stage (coarse→fine)
*same-kind* projection with a witness-level verification shortcut (~1% of the
paper; the crate's own comments admit it).

**Gaps**:

* **E**: the recursion itself is absent — no committed superconstant-ρ
  split-and-fold (O(log_ρ mw) rounds), no gadget decomposition, no
  sumcheckify, no Π^lin with θ_a subfield batching; "coarse/fine" is
  semantically inverted (paper: ring-level vs coefficient-level trace-dual
  alternatives; impl: two composed ring projections); prover O(mw·nrp),
  verifier O(mw), proof O(mw·nrp·φ) vs the paper's O(mw λ) / polylog /
  ~200 KB.
* **P**: the load-bearing kernel story — incomplete NTT at q ≈ 2^50 with
  64 quadratic slots, Karatsuba 5→4 mults/slot, fused single-pass AVX-512
  (1.36-1.63x measured by the authors; `repos/rokoko/incomplete-rexl`
  5.2k LOC) — none of it exists here; nor the CRT commitment over eight
  16-bit primes with VNNI accumulation; nor the sparse-tiled projection
  kernels.
* **S**: **no challenge space can exist on the current fully-splitting ring**
  (e=1 ⇒ knowledge error ≥ 1/q per RoK; the reference uses e=2 with
  fixed-weight ternary TAU=22, |C| ≈ 2^103.3, op-norm 9.8); no norm-schedule
  algebra (dcmp/cmp/f̂/rad(f)), no κ composition, no parbreak SIS instances,
  no estimator; verification consumes the coarse image as an input.

**Next implementations**: 1 incomplete-NTT ring substrate (L — in-house
precedent: `simd/ntt_quad.rs` *is* an incomplete NTT); 2 faithful Π^proj-c
with committed Y_klin + constraint rows (S after 1); 3 recursive COM (Fig 1)
+ parbreak derivation (M); 4 Π^fold-split + sumcheckify over the existing
`VirtualPolynomial` engine (L); 5 Π^lin subfield-batched sumcheck (M);
6 Π^proj-f trace-dual (M/L); 7 norm schedule + FS + wire + benchmarks vs
214 KB (M); 8 PCS front end (S).

### 3.9 SALSA(A) (Kuriyama-Lai-Osadnik-Tucci) — `lattice-salsa`

**Implemented**: four standalone gadgets — field-level norm sumcheck
(Σz² = c over Goldilocks), multilinear "LDE tensor" padding-invariance check,
a plain negacyclic ring multiplication (no proof), a round-masking "zk"
sumcheck whose masks are **public** (no privacy — honestly documented).
`lattice-zkvm` declares the dependency but never calls it.

**Gaps**:

* **E**: the paper's central claim — norm checking via sumcheck (Trace
  identity ‖x‖² = Trace(⟨x,x̄⟩)) with O(τm) provers and 2-3x smaller norm
  proofs — is not implemented *where it matters*: the live norm check in the
  stack is still the pre-SALSAA digit-revealing `NormProof` + full-witness
  opening in Akita/zkVM (O(m), full disclosure). No Π^lde-⊗ zero-communication
  linearization (Lemma 2), no Π^sum with CRT-slot final check, no Π^norm+
  composition, no Π^batch* (row-count-preserving folding — Theorem 4), no
  fold/split/join/b-decomp protocol set, no SNARK/PCS/folding applications.
* **S**: the field-level norm sumcheck proves a **modular identity, not a
  norm bound** (no Lemma-4 no-wraparound condition `B'^ρ < q/2`); `verify_norm`
  with `z_at_challenge = None` performs no terminal check (footgun); no
  F_{q^e} challenges (per-round error ~2^-28 on current rings — unsound at
  λ=128); zk masks public.
* **P**: duplicated round-poly/interpolation code vs `lattice-sumcheck`;
  the engine's `sum_products` re-does factor half-binding per evaluation
  point (~2x); no SIMD/threads (the paper's 10.61 s @ 2^28 leans on
  IFMA/HEXL + parallelism).

**Next implementations**: D1 ring-level Π^norm ∘ Π^sum over R_q with F_{q^e}
challenges + CRT slots + conjugation + trace (L, *the paper's claim*); D2
Π^lde-⊗ linearization into a `LinRelation` type with row-tensor F (M); D3
Π^batch* into `lattice-sumcheck::batch` (M); D4 **swap the Akita/zkVM
response layer to the SALSAA chain** (M after D1+D2 — zkVM proofs Θ(N)→
polylog, disclosure removed; the single highest-leverage item in this
document); D5 fold/split/join/b-decomp protocol set (M-L, zem's ~5.9k-LOC
tree as blueprint — port structure, not anti-patterns); D6 engine
constant-factor pass + Goldilocks AVX-512 (S/M); D7 truth-in-advertising +
QROM registration (S).

### 3.10 Quasar (ePrint 2025/1912) — `lattice-lookup`

**Implemented**: FLI-lineage grand-product lookup containment with τ derived
by hashing **raw table+reads** (O(n) per verification); partial-evaluation
accumulation as a prover-side data transform (no union polynomial, no
commitments, **no verifier function at all**).

**Gaps**:

* **S (soundness hole, not just a gap)**: `verify_lookup` checks
  `T(τ) = R(τ)·Q(τ)` on three prover-supplied scalars — any consistent triple
  (r=q=t=1) verifies against any statement; T/R/Q are never committed.
  `accumulate_partial_evaluation` binds nothing.
* **E**: no multi-cast reduction (union polynomial w̃∪ with one commitment
  C∪ covering ℓ instances, log ℓ-round sumcheck over G(Y) = F(x̃,w̃)·eq̃,
  partial-evaluation consistency check w̃∪(τ,r_x) = w̃(r_x) — the paper's
  core), no 2-to-1 fold (Z-multilinear curves, γ-pow combined 1-round
  sumcheck, IOR_batch via commitment homomorphism), no ACC.V/ACC.D, no
  SPS/CV wrapper, no IVC loop (the shard-parallel zkVM payoff).
* **P**: O(n²) linear-scan multiset difference; k× memory blowup from
  materialized partial MLEs; scalar grand products.

**Next implementations**: Q1 commit T/R/Q + derive τ from commitments +
counting-map difference + forged-triple negative test (M, closes the soundness
hole); Q2 real `NIR_multicast` with union polynomial + log ℓ sumcheck +
verifier (L, zem's `quasar/` tree as blueprint); Q3 2-to-1 fold + decider
(L); Q4 SPS/CV wrapper mapping CCS relations into multicast-able form (M);
Q5 multi-instance IVC loop for shard-parallel proving (L, strategic); Q6
small fixes (S).

### 3.11 Twist & Shout (ePrint 2025/105) + the zkVM stack — `lattice-memory`, `lattice-vm`, `lattice-zkvm`

**Implemented**: deterministic memory-checking oracles (`twist_check`,
`shout_check` — sound ground truth); grand-product `shout_fingerprint`
(**the Lipton's-trick technique the paper eliminates**); `twist_fingerprint`
checking **final-state consistency only** — a stale read followed by a
consistent final state passes (unsound as a memory-checking statement);
dense `one_hot` helper and `counter_increments` (computed, never consumed);
a strong RV64IMAC conformance layer (decoder/executor/reference/differential
corpus); envelope discipline with strict decoding + 4000-mutation fuzzing.

**Gaps**:

* **E — the paper's protocol machinery is 0% implemented**: no one-hot
  constraint PIOP (Booleanity + Hamming-weight-one + raf-evaluation
  sumchecks, Figs 6/8), no Shout read-checking sumchecks (Figs 5/7), no
  Twist increment commitment (`Inc(k,j) = wa·(wv − Val)` — the paper's
  headline), no virtual-Val via LT (O(logT) evaluation), no d-dimensional
  one-hot (commitment-key size control), no sparse-dense sumcheck (§7,
  structured K=2^64 tables), no locality-aware binding orders (§8.2 — 3i
  mults for 2^i-local reads), no Gruen/Dao-Thaler round optimizations.
  Feeding the relation to the current dense engine would cost O(K·T·log(KT))
  — the asymptotic wall the paper exists to break.
* **E (zkVM)**: `verify_program` **re-executes the entire program**
  (prove.rs:211-215), checks the sumcheck section **by byte length only**
  (`len % 24 == 0 && len/24 == num_vars`, prove.rs:276-278 — `let _ = point;`
  at :290), recomputes the commitment from its own re-derived witness, and
  the proof's Witness section *is* the packed preimage; proof size Θ(trace);
  no instruction-semantics constraints anywhere (decode/execute correctness
  rests 100% on shared re-execution); the flat witness stream is not the
  paper's column shape (ra_i/wa_i/Inc/wv per column).
* **S**: `twist_fingerprint` (the staged production statement,
  prove.rs:104-106) is not sound for memory correctness; grand-product
  soundness ≥ (T+K)/|F| vs the paper's log(TK)/|F|; Goldilocks caps sumcheck
  soundness at ~2^-57 at scale; toy SIS parameters throughout (n=16, k=2,
  q ≈ 2^31.6); RVC funct3 mapping is codebase-local and the reference mirrors
  it (differential testing cannot catch RVC conformance bugs by construction).
* **P**: sumcheck engine re-does factor half-bindings per evaluation point
  ((d+1)x redundancy); `fix_variables` clones full arrays per round; Ajtai
  commit does not skip zeros (the paper's "free 0s" doctrine); value-oriented
  packing (one-hot bits cost 22-bit limbs — ~32-64x denser bit-packing
  possible); BTreeMap/string-keyed oracle bookkeeping.

**Next implementations**: P0-1 the real PIOPs — `shout.rs`, `onehot_check.rs`
(2^-1 point trick valid on Goldilocks), `twist.rs` with the three batched
sumchecks + `lt_extension` in `lattice-core::mle` (M for Shout, L for Twist);
P0-2 sparse sumcheck prover (index-list one-hot factors, per-register eq
arrays, Eq-46 lookup tables — port the *design* from
`repos/jolt/.../read_write_matrix/{cycle_major,address_major}`) (L); P0-3
d-dimensional one-hot + chunking policy (S); P0-4 restructure `prove_program`
around the paper's columns (M); P0-5 **delete verifier re-execution** — verify
the sumchecks + PCS openings + public-output consistency (M, unblocks after
P0-1/P0-4; add tamper tests that must fail without re-execution); P1-4
sparse-dense sumcheck for structured tables (L); P1-5 sumcheck constant-factor
pass (S-M); P1-6 zero-skipping Ajtai + bit-packed one-hot columns (S); P1-7
non-revealing Akita opening (L — the hinge for succinctness and ZK; reference
`repos/jolt/crates/jolt-akita` adapter + AKITA_ONE_HOT_K16/K256 schedules);
P1-8 instruction-semantics constraints: bytecode-fetch = Shout over the
program image, then per-family constraint systems (M-L); P2-9 production
parameters (M); P2-10 RVC standard conformance (S-M).

### 3.12 labinius port + LaBRADOR — `lattice-labinius`, `lattice-labrador`

**State**: ~90% of upstream pcs by protocol surface; AVX-512 backend
bit-exact (kernels 147-448x, full commit 37.7x, reference round 3885→222 ms
= 17.5x); LaBRADOR as a native exact-arithmetic crate with a documented
conservative encoding.

**Gaps** (verified against upstream source):

* **`Opening::Recursive` is declared but not wired** — no `residues` field
  on `CommitmentOpening`, no `T_u` left-expansion commitment, no
  `prove_opening` call site, **no test exercises Recursive mode**; this is
  the mode that delivers the compact-opening asymptotics (scheme.rs:65-69,
  :191 vs upstream commitment.rs:124-132, prover.rs:137+).
* Recursion encoding is the documented simplification (direct per-limb Ajtai
  identities, 648 constraints per limb·element block vs upstream's shared-φ
  chunked chains with carry gadgets) — constraint count (hence proof size)
  inflated; no T_R rest commitment; row evaluation sent in the clear.
* **No `wire/` module** — sizes are floors; no serialized, decodable proof
  artifact; no rANS (upstream: 664 KB → ~370 KB at ~8.8 bits/coeff, LANES=64
  interleaved streams, transmitted histogram).
* Unported kernel families: `gen_small/gen_quad/gen_large` (2,642 LOC — the
  fold/verify path; ~120 ms of the 222 ms round sits behind them),
  `bin_large` (910 LOC), `slots.rs`/`bd.rs`/`transpose32.rs` (~315 LOC).
* No cross-field machinery (`switch.rs`, `fields/` ~990 LOC) — no B128
  witnesses, no binius/flock-style front ends.
* LaBRADOR: O(N²) i128 schoolbook product vs upstream's 8-prime RNS `polx`
  (10-50x expected); hard-wired to Q = 2^48−59.
* FS hash divergence: SHAKE-256 vs upstream BLAKE3 (documented; equivalence
  argument belongs in the QROM composition review).

**Next implementations**: 1 wire Recursive mode end-to-end + tests (M,
*protocol completion before more kernels*); 2 gen_* port + vertical AuxData
fold (L — round 222→~100 ms, removes the 170 MB scatter); 3 bin_large (M);
4 wire/ bit-packing + rANS (M); 5 chunked-chain recursion encoding (L); 6
LaBRADOR RNS polx — build as the shared `lattice-rns` crate (L, serves RoKoko
CRT commitment too); 7 block-sink fusion + A-prefetch + components_of SIMD
(S/M each); 8 cross-field switch (M); 9 Goldilocks AVX-512 (L, strategic).

### 3.13 Cross-cutting security layer — `lattice-zk`, `lattice-qrom`, `lattice-commitment`

**State**: genuine and well-tested at kernel scale — Libra-style blinded
sumcheck (correct FS ordering: pre-round commitment before challenges),
ABDLOP/Lyubashevsky zk_linear with rejection sampling and the wave-3 FS-order
fix, distributional simulators with chi-square KATs, CRT-carrier relations,
type-separated OS entropy + nonce ledger, capability tokens; QROM query
ledger + domain registry + attestation + composition review with 4 blocking
manual items.

**Gaps**:

* **Not wired into the zkVM** (not even a dependency); degree-1
  single-vector expressiveness (cannot wrap product-structure sumchecks);
  10 committed slots per witness value (would forfeit Twist & Shout's cost
  profile if wired naively — needs sparse/bit-aware masking).
* ZK soundness rests on Ajtai binding at toy parameters (n=16, q ≈ 2^31.6 —
  vacuous); extraction tree asserted not proven; retry-count grinding side
  channel unbudgeted.
* QROM: the example "zkvm composition" attestation is a fiction that matches
  no real protocol shape — once the protocol layer lands, regenerate from
  the actual stage list; grinding at public-output digests unbudgeted.
* Constant-time posture unreviewed (rejection-sampling early exits on secret
  norms, infinity_norm comparisons on secrets); no timeout/CPU-quota harness
  for network verification (G6 external).
* `NormProof` reveals all digits of the full witness (the pre-SALSAA design);
  `LinearProof` (non-ZK baseline) retained alongside the fixed zk variant.

**Next implementations**: P1-7/P3-11 wire ZK into the zkVM via per-stage
blinding hooks (Jolt's Blindfold pattern — a mode of the stage pipeline, not
a bolt-on) + sparse masking (L); P3-12 real QROM composition attestation from
the prover's actual stage/query manifest + grinding bounds as machine checks
(S-M); P3-13 production parameter instantiation + estimator + digest-gated
verification (M); P3-14 CT variance tests + timeout harness (S-M); replace
`NormProof` with the SALSAA chain (D4 above).

---

## 4. The unified roadmap (Waves 6-8)

Sequenced by: soundness-critical first, then protocol completion, then
performance parity. Effort in engineer-days at the current codebase's
discipline level (S ≲ 1-2 d, M ≈ 3-7 d, L ≈ 1-3 wk).

### Wave 6 — Shared substrate + soundness-critical fixes (the prerequisites)

| # | Item | Serves | Effort |
|---|---|---|---|
| 6.1 | `lattice-core::short_challenge` — fixed-weight ternary + op-norm rejection + Γ_C accounting | PG, Sym, Cyclo, LF+, Pikku, Akita(A8), HyperWolf(H5), RoKoko | M |
| 6.2 | `NormBudget` type — symbolic (β, γ, β̄, δ) + hard gate β < min(q/2, β*) per fold | all folding modules | S |
| 6.3 | Ajtai cached-NTT commit + Barrett in `lattice-ring` + statement-absorption API | every commit/verify in the workspace | S |
| 6.4 | FS hygiene pass: absorb full statements (cyclo input commitment; superneo witnesses-out) | Cyclo, SuperNeo | S |
| 6.5 | Quasar Q1: commit T/R/Q + counting-map difference + forged-triple test | Quasar, zkVM lookups | M |
| 6.6 | F_{q^e} extension-field arithmetic + transcript sampling (in `lattice-core`) | SALSA D1, Cyclo P1-5, Pikku P1-5, RoKoko 5 | M |
| 6.7 | `Modulus50` — q ≈ 2^50, q ≡ 129 mod 256, incomplete NTT (port `ntt_quad.rs` discipline) | Cyclo, Pikku, RoKoko, SALSA | L |
| 6.8 | Zero-skipping Ajtai commit + bit-packed one-hot columns | Twist, SuperNeo pay-per-bit | S |
| 6.9 | SIS estimator crate (port ADPS16/BDGL16 from `repos/akita`) | every SecurityProfile | M |
| 6.10 | Folding/lookup/salsa benchmark entries in `lattice-bench` | measurement discipline | S |

### Wave 7 — Protocol completion (the papers' cores)

| # | Item | Paper | Effort |
|---|---|---|---|
| 7.1 | ProtogaLattice Fig-3 protocol (δ/α/y, Gröbner K_rt, e*-check) + adversarial tests | PG | M |
| 7.2 | SALSAA D1+D2: ring norm sumcheck + LDE linearization | SALSA | L |
| 7.3 | **D4: swap Akita/zkVM response layer to the SALSAA chain** (Θ(N)→polylog, disclosure removed) | SALSA+Akita+zkVM | M |
| 7.4 | ~~Twist & Shout P0-1..P0-5~~ LANDED 2026-09-29 (core): the zkVM memory argument proves/verifies without re-execution (`lattice-zkvm/{columns,ledger,memory,memproof}.rs` — digit-bit virtual one-hots, virtual-Val Fig-9, two Ajtai bundles, 552 workspace tests green). REMAINING: the arith/logic/comparison/control/routing/halted constraint families (substrate staged in `constraints.rs`) + the sparse prover (8.5) | T&S+zkVM | core done |
| 7.5 | labinius Recursive-mode wiring + end-to-end tests | labinius | M |
| 7.6 | Cyclo Π^range protocol over F_{q^e} + Π^ext RoK rows | Cyclo | M |
| 7.7 | LF+ monomial/ψ layer + Π^mon + real double commitments | LF+ | M |
| 7.8 | PikkuFold layered LRP + RingSC + certified-JL gate | PikkuFold | L |
| 7.9 | Symphony tensor ring + Π_had + O(μ) shared-randomness fold (delete pairwise E_ij) | Symphony | L |
| 7.10 | Quasar Q2+Q3: NIR_multicast + 2-to-1 fold + decider | Quasar | L |
| 7.11 | Akita A1-A5: fold core → ring checks → range sumcheck → tensor reduction → terminal/recursion | Akita | L |
| 7.12 | ~~HyperWolf H1-H5~~ LANDED 2026-09-29: full Protocols 1/2/3 in `lattice-pcs/hyperwolf.rs` (own u64 ring at q ≡ 5 mod 8, certified fixed-weight challenges; H6 compaction + H8 trait unification remain) | HyperWolf | done |
| 7.13 | ~~RoKoko 3-5~~ LANDED 2026-09-29: recursive COM + Ξ^lin + Π^fold-split + sumcheckify + Π^lin in `lattice-rokoko/{com,protocol}.rs` (items 6-8: Π^proj-f, norm schedule, PCS front end remain) | RoKoko | done |
| 7.14 | ProtogaLattice bootstrapping + range-proof attachment | PG | M |
| 7.15 | SuperNeo committed instances + pay-per-bit sparse commit | SuperNeo | M |
| 7.16 | labinius wire/ (bit-packing + rANS) | labinius | M |

### Wave 8 — Performance parity + production posture

| # | Item | Effort |
|---|---|---|
| 8.1 | gen_* transforms port + vertical AuxData (labinius round 222→~100 ms) | L |
| 8.2 | Goldilocks AVX-512 kernels for lattice-core/sumcheck/akita/zkvm (4-8x on field-bound stages) | L |
| 8.3 | `lattice-rns` 8-prime crate → LaBRADOR polx (10-50x) + RoKoko CRT commitment | L |
| 8.4 | bin_large port + block-sink fusion + A-prefetch + components_of SIMD | M |
| 8.5 | Sumcheck engine constant-factor pass (single-binding rounds, in-place fix_variables, Gruen, Dao-Thaler) | S-M |
| 8.6 | ZK wiring into the zkVM (per-stage blinding, sparse masking) + composed simulator KATs | L |
| 8.7 | Real QROM composition attestations + grinding bounds; CT + timeout harness | S-M |
| 8.8 | Production parameter instantiation + estimator-gated SecurityProfiles | M |
| 8.9 | Chunked-chain recursion encoding (labinius constraint count ↓) | L |
| 8.10 | Akita planner/validator + setup offloading + commitment compression; HyperWolf LaBRADOR compaction; Symphony CP-SNARK; Quasar IVC loop | L each |

### Dependency graph (critical path)

```
6.1 short_challenge ──┬─→ 7.1 PG protocol ──→ 7.14 bootstrap
                      ├─→ 7.6 Cyclo, 7.7 LF+, 7.8 Pikku, 7.9 Symphony
6.6 F_{q^e} ──────────┘
6.3 Ajtai fast path ─→ 7.3 SALSAA response swap ─→ 7.4 T&S/zkVM production path
6.7 Modulus50 ────────→ 7.8 Pikku at paper params, 7.13 RoKoko kernels
7.2 SALSAA D1/D2 ─────→ 7.3 (the response layer swap)
7.4 T&S P0 ───────────→ 8.6 ZK wiring
7.5 labinius Recursive → 8.1 gen_* (protocol before kernels)
8.3 lattice-rns ──────→ 8.10 HyperWolf compaction + LaBRADOR scale
```

### Calibration targets (the papers' own numbers to reach)

| Paper | Target |
|---|---|
| SALSA(A) | 2^28 witness: 10.61 s prove / 41 ms verify / 979 KB |
| RoKoko | 2^26: commit 1.57 s + prove 2.84 s / verify 8.59 ms / 214 KB |
| Akita | 2^35 bits: 61-67 KB proofs, 8.1-16.2 ms offloaded verify |
| HyperWolf | N = 2^20-2^30: 52-53 KB proofs, O(log N) verify |
| PikkuFold | 2^18-2^22: ~5.5 KB per fold, 4.6 s prover, 3.67 ms verifier |
| Cyclo | 2^20: 31.8 KB, ext-commitment 36.7 s (vs LF+ 129.4 s) |
| Symphony | ℓ_np = 2^10: < 200 KB total, prover ≈ 3·2^32 R_q-muls |
| Twist & Shout | (5 log K + 16)T mults d=1; >10x vs log-proof baselines |
| labinius | upstream ~13 ms 2-limb commit @ 2^18/256 cols; 630 cycles/element with sinks |
| ProtogaLattice | 83-123 KB proofs; 3 RO calls/fold; 4 per full iteration |

---

## 5. Honest disclosure additions for AUDIT_CHECKLIST.md (immediate)

1. `lattice-lookup::verify_lookup` binds nothing until Wave 6.5 lands
   (prover-supplied T/R/Q scalars) — the README row needs a "no verifier"
   caveat today.
2. `lattice-salsa::prove_norm` over Goldilocks proves a modular identity,
   not an integer norm bound (no Lemma-4 wraparound condition) — must not be
   cited as a norm check until D1.
3. `twist_fingerprint` is a final-state consistency check, not the paper's
   memory-correctness statement — the staged production replacement must use
   the Fig-9 sumchecks, not this identity.
4. Folding modules: cross terms / fold outputs are not verifier-bound
   (per-module gap tables in §3); the `lib.rs:14-17` "prover + verifier"
   claim holds only for `latticefold_plus::prove_range/verify_range`.
5. labinius `Opening::Recursive` is declared but unwired and untested.
