# Cyclo (ePrint 2026/359) — implementation notes

**Paper**: Garreta–Lipmaa–Luhääär–Osadnik, *Cyclo: Lightweight
Lattice-based Folding via Partial Range Checks*.

## The three landed layers

### 1. The core folding module (`cyclo.rs`)

* The exact signed base-`2b` digit chunking (`chunk_element`/
  `unchunk_elements` — the iterative-borrow algorithm, exact inverse).
* The extension commitment (chunk → pad → generic Ajtai commit) and the
  homomorphic fold; `refresh` = the extension-commit of the accumulated
  witness; the additive norm bookkeeping with the Wave-6.2 hard gate.
* The Wave-6.4 FS hygiene: the fold transcript absorbs the FULL
  statement (the input commitment, the params, the fold counter) before
  the challenge; `fold_ring_challenge` draws the short RING challenges
  through the shared `ShortChallengeSpec`.
* The P3 fix (2026-10-04): `partial_range_check` — the 2× soundness
  slack in the partial branch eliminated (the exact signed high-part
  `|high(v)| ≤ β − low_worst` with the free-prefix worst case; the
  previously-untested regime covered by the adversarial boundary test).

### 2. Π^range + Π^ext (`cyclo_protocols.rs`)

* Π^range as the degree-`(2b+2)` sumcheck over `F_{q²}` (the X³−X
  Karatsuba identity at `b = 1`; the layered digit MLE; the
  verifier-computable digit reconstruction terminal).
* Π^ext with the Fig-2 RoK constraint rows — the challenge-batched
  `⟨c, ((2b)^i ⊗ A)·v⟩ = ⟨c, t⟩` rows; the FS hole structurally closed
  (the challenge derives from `(t_input, t_ext, params, fold counter)`).

### 3. The §7 R1CS-over-F_q bridge (`cyclo_r1cs.rs` — the paper's
    raison d'être, 2026-10-04)

The reduction of knowledge from `Ξ^{R1CS}` over `F_q` to the principal
linear relation over `R_q`:

* **The θ_k digit map**: `θ_k(f) = f(k) mod q` (the F_q-module
  morphism) and `θ_k^{-1}(c)` = the base-k digit polynomial — the
  small-norm-by-construction lift (norm `< k`). The LZX digit budget is
  `⌈log_k q⌉` (the paper's `⌊log_k q⌋` under-covers the values in
  `(k^⌊log_k q⌋, q)` at non-power-of-k moduli — found by the roundtrip
  test at `c = Q/2`).
* **The bridge-local `F_{q²} = F_q[u]/(u² − 5)`** over the commitment
  ring's own modulus (5 = the smallest quadratic non-residue mod
  `3·2^30+1`) + the field-swapped sumcheck engine (the
  `fq2_sumcheck` pattern, differential-tested against the naive cube
  semantics).
* **The linearized sumcheck**: `Σ_b eq(b;r)·(Q₀(b)Q₁(b) − Q₂(b)) = 0`
  at a random `r ∈ F_{q²}^{log m}` with `Q_i = M_i·z` as dense factors
  (the HyperNova linearization blueprint) — individual degree 3.
* **The terminal**: the prover publishes `d_i = Q_i(u)`; the verifier
  checks `(d₀·d₁ − d₂)·eq(u; r) = c` with the eq value RECOMPUTED from
  its own `(u, r)` — never trusted from the proof.
* **The ring lifts** `d'_i ∈ R_q²` — the ACTUAL tensor scalar-weighted
  sums `Σ_{b'} MLE[M_i](u, b')·z'_{b'}` per component (NOT the digit
  lifts of the `d_i`: the embedding is not additive — base-k carries;
  only the projection is F_q-linear) — with the verifier-checked
  `θ_k(d'_i^{(b)}) = d_i^{(b)}` consistency; the linear claims (4) are
  DECIDED by `decide_principal_linear` (the decider model — the
  opened lift; `(D1)` the commitment binding, `(D2)` the exact ring
  equalities, `(D3)` the projected prefix check).
* **The prefix elimination**: `v ∈ F_{q²}^{log(ℓ+1)}`, `e =
  MLE[(x,1)](v)` — binds `w'`'s prefix to the public input in the
  PROJECTED form (error `≤ log(ℓ+1)/q²`).
* **The skip-Π^ext remark wired**: at `k ≤ b` the lifted witness has
  norm `< k ≤ b` — it feeds `CycloAccumulator::new` directly (the
  extension-commitment step skipped) — test-pinned.

## The honest-deviation ledger
0. **The compact-PCS terminal (2026-10-04, the staging wave)** — the
   §7 bridge's decider no longer opens the witness:
   `cyclo_terminal.rs::decide_principal_linear_compact` decides the
   ride-the-fold claims through the ring-functional width fold
   (`lattice-widthfold::ring_fold`) — (D1) rides the fold's (W0) (the
   part images sum to the commitment), (D2) the six linear claims ride
   the EXACT ring-functional layer, (D3) the two prefix claims ride
   PROJECTED functionals (θ_k of the fold's public per-part sums — the
   projected form the carry finding mandates). Deviations vs the clear
   decider: the lift's norm precondition `∥z'∥∞ < k` enters as the
   fold's β₁ (the estimator's bound accounting) instead of the
   per-element gate; the binding is the fold's estimator-gated
   `[A₂|−T]` instance at the digit gate (fail-closed at prove AND
   verify); the multi-fork extraction across the fold is the same open
   analysis the zkvm Sound profile documents. The terminal's
   communication at the bridge's scale is ~5× the opened lift (the
   fold's commitments + garbage) — the honest price of the
   witness-freedom.


0. **The carry finding** (2026-10-04, the decider wave): the digit
   embedding `θ_k^{-1}` is NOT additive (base-k carries) — the `d'_i`
   must be the actual tensor sums, not the digit lifts; the prefix
   claim rides in the projected form. The PROJECTION `θ_k` is F_q-linear
   — the load-bearing direction — and the verifier's consistency
   checks use exactly it. (The decider's end-to-end test caught this.)
1. `e = 2` (the paper's larger-e regime): the quadratic extension over
   the 31.6-bit q gives the 2^{-121}–2^{-122} Schwartz–Zippel floor —
   the ~7-bit gap from λ = 128 documented.
2. The `d'_i` ride as `R_q` PAIRS (the componentwise discipline), not
   the paper's single `R_{q^e}` tensor elements — the algebra exact,
   the wire shape differs.
3. `ℓ_k(q) = ⌈log_k q⌉` (the ceiling) — the paper's floor under-covers
   the field at non-power-of-k moduli.
4. Π^range/Π^ext run on the decider model (`opening_v` in the clear) —
   the compact-PCS terminal is the documented outer-layer gap.
5. The decider's matrix-MLE is O(m³) at kernel scale — the paper's
   amortization is the scale-up follow-up.
