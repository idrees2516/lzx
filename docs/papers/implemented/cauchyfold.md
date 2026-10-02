# CauchyFold — ePrint 2026/2011

**Paper**: Wang, *"CauchyFold: Residue-Optimal High-Arity Lattice
Folding via Scaled Cauchy Challenges"*.
**Status**: implemented in depth — `crates/lattice-cauchyfold`
(~3.8k lines, 58 tests) over the paper's own field and ring:
`q = 2^48 − 59 = 281474976710597`, `K = F_q[u]/(u^4 − 4u^2 + 2)`
(the θ-basis `(1, u, u²−2, u³−3u)`), `R_{q,64}` (the LaBRADOR ring,
reused via `lattice-labrador`).

## What was implemented

| Paper component | Status | Where |
|---|---|---|
| The Cauchy family (§4.4) | ✅ | `cauchy.rs::CauchyParams` — poles/scales, `a_i(c) = λ_i/(c−ξ_i)`, `D/P_i/P_ij`, the partial-fraction identity (eq. 18) pinned by test |
| The carrier `H_src` (§4.4, eq. 21) | ✅ | `cauchy.rs::Carrier::direct` — the pair-processing reference form |
| The carrier identity (Prop 4.4) | ✅ | `carrier_identity_holds` — verified at `3k+2` test points, arities 2–8 |
| The fast carrier (Prop 4.6 / A.3) | ✅ | `Carrier::fast` — product trees, the squared-denominator formal-derivative trick (eq. 45), multipoint + interpolation; differential-tested against `direct` |
| The fixed-carrier consistency (Lemma 4.5) | ✅ | `discrepancy_poly` — the `F(T)` of degree ≤ 2k with the root-count test (`≤ 2k/|C|`) |
| The boundary width theory (§4.1–4.3) | ✅ executable | `boundary.rs` — `Va`, the separation condition, Theorem 4.1 (`m_min = r·dim Va`), Lemma 4.2, Corollary 4.3 (`dim Va = k`) pinned by exact K-linear algebra; the power-family negative control included |
| The folded record (§4.4) | ✅ | `fold_z` / `fold_e` — `Q(z*) = E*` verified |
| The field layer (B.1) | ✅ | `field_k.rs` — `Fq48`, `K4` (exact `u^4 = 4u²−2` arithmetic), the θ↔power basis conversions, `KPoly` |
| The node protocol (§5, Fig. 1) | ✅ | `node.rs` — the transcript order (source/carrier commitments **before** `c`, the output commitment after), the fold, the field-level checks, the root reduction, the chain handoff |
| The field front end (§5.3) | ✅ | One `K`-valued sum-check (`sumcheck_k.rs`) over the 19 root objects' digit cubes carrying: the Booleanity legs (`x(x−1)=0` on the designated slots of every state in `I`), the output's linear bindings to the claimed `Az*/Bz*/Cz*` (whose quadratic combination is then public arithmetic), and the carrier-evaluation leg binding the claimed `H(c)`; the residual update `E* = E₀ + Σa_i²E_i + H(c)/D(c)` as public arithmetic |
| The root reduction (§5.3–5.4) | ✅ | The level-2 digit witness `W` (centered radix-16, `{−8..7}`), the `ΓW = Y` system with the **ring-structured commitment rows** (`A_R·Rec(W) = C_all` with the negacyclic convolution coefficients), the claim row, the `R16` range polynomial, the §5.4 fingerprint sum-check (individual degree 17) |
| The finite linear chain (§5.5) | ✅ | `reduce_chain.rs` — per layer: the per-block commitments `t_j = A·w_j` (shared columns), the `{−1,0,1}` projection with the `‖Πw‖² ≤ mS` threshold and bounded retries, the symmetric `h_ij` fixed **before** the short challenge, the certified `D46` challenges (ternary with operator norm ≤ 46), the response identities (28)–(30) with the `G = 256·S·15/14` bound and retries, the radix-64 child splits |
| The terminal (§5.6, D.7) | ✅ | `wire.rs` — the low frame (sign + 6-bit residues), the high frame, every fail-closed decoder check (lengths, negative zero, magnitude, re-encode equality); `reduce_chain.rs` — the direct terminal witness with the recomposition, norm, and commitment checks |
| The extraction (§6, C) | ✅ | `extract.rs` — Lemma 6.2 (compare before clearing) with the norm bound `‖K‖ ≤ 2B_XB_Δ`, the coordinate-replay harness (honest + cheating provers), the ring inverse via the 64×64 negacyclic linear solve, the loss accounting (`ε_i`, `Λ_i`, `κ_node`, the fixed-vector projection bound) |
| The parameters (§7, D) | ✅ | `params.rs` — both paper profiles recorded declaratively with their exact tables (127,887 / 129,002 B wires, the chain schedules, the S₀ bounds, the measured times); the executed scaled profile documented |

Benchmarks (`examples/cauchyfold_bench.rs`, release): the carrier at
`k=16` in **2.0 ms** (direct) / **1.8 ms** (fast); the boundary
analysis at `k=8` in 1.3 ms (`dim Va = 8/8` attained); the full node
`k=16` prove **1.6 s** / verify **248 ms**.

## The honest-deviation ledger

1. **Scale.** The executed profile is `k = 16` with `s = 4, y = 2`
   (~2.5K level-1 digits, ~7.5K level-2 coefficients) versus the
   paper's 57.5M-coefficient root witness. The paper's two profiles
   are recorded with their exact tables (marked not-executed: ~15 GiB,
   2.6-hour runs).
2. **The relation.** The carrier algebra requires the **homogeneous**
   quadratic `Q(z) = Az⊙Bz` (the paper's own §2–5 form); the relaxed
   R1CS embedding of Appendix A.2 (the `−uCz` linear term riding the
   extended-vector convention) is recorded in the struct docs but not
   instantiated — the relaxation is carried by the residuals `E_i`, as
   the paper's §3.1 allows.
3. **The root matrix.** One shared Ajtai root matrix for all 19 objects
   (the paper samples per-object matrices) — the γ-combination then
   rides one homomorphic system; the Γ commitment rows absorb the
   per-table γ powers (`i·key.cols` slot addressing).
4. **γ in the first coordinate.** The root γ is drawn in the first
   `K`-coordinate (a plain `F_q` scalar): the Γ rows are
   per-coefficient equations, and the ϑ-embedding for full-`K` γs would
   need ring-structured row coefficients — recorded as the deviation.
5. **The chain.** 2 nonterminal layers + terminal (a profile parameter;
   the paper's 5+1 is its scale — the layer mechanics are identical);
   the auxiliary digit commitments `u1 = B·t̃`, `u2 = D·h̃` are replaced
   by direct `h` transmission checked through identity (29) (the child
   does not carry the aux digit blocks, so the inter-layer
   recomposition is verified at the terminal).
6. **The D46 certification.** A float DFT with a conservative margin
   replaces the paper's exact rational interval test (2^-60-grid with
   the Machin π expansion); the unit property — distinct challenges
   differ by units — holds exactly via `‖Δ‖∞ ≤ 4 < √(q/2)` at
   `q ≡ 5 (mod 8)`, the Lyubashevsky–Seiler criterion.
7. **The terminal codec.** Fixed-width quotients replace the
   arithmetic-coded high frame; the frame structure and every
   fail-closed decoder check are kept. (The paper's worst-case-length
   formula is recorded in `wire.rs`'s docs.)
8. **The projection error bound (C.2, formula 48).** The paper's
   printed form `θ·(3σ²/(3σ²−1))^m` drops the `θ` inside the power and
   exceeds 1 for every `m`; the implemented bound is the correct Markov
   chain `(e^{1/(3σ²)}·θ)^m` — the typo is recorded at the
   implementation site.
9. **The digit-energy recurrence (C.3).** The implemented
   `F_t(H) = b² + F_{t−1}(⌊(H+b)/ρ⌋)` is the conservative exact
   stand-in for the paper's two-branch recurrence (the capped-interval
   endpoint analysis); u128 internals with saturation.
10. **The ring inverse.** The xgcd-over-`F_q[X]` route was replaced by
    the exact 64×64 negacyclic linear solve (`p·x ≡ 1 mod X^64+1`) —
    equivalent, easier to verify exhaustive.

## The paper's own numbers, recorded

`PAPER_MINIMUM` / `PAPER_CRS600` carry the §7 tables verbatim: the
127,887 / 129,002 B folding wires (+24,576 B fresh-input commitments =
152,463 / 153,578 B total), the initial linear stages (radix 32/128,
10/7 digits, S₀ = 2,818,920,384 / 4,491,476,992), the five-layer
chains (56→19→10→6→5→terminal blocks at 18,726→3,138→…→478 lengths),
the terminal responses (35,763 / 36,054 B), the medians (9,407 /
9,503 s pipeline, 800 / 843 s verify, 15.35 / 12.40 GiB RSS), the
`J = 22` Module-SIS roles, and `PAPER_FIELD` (the degree-4 extension,
the `2^26` root cube, the challenge support `K \ {ξᵢ}`).
