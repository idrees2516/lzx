# LatticeBlindFold — A Lattice-Based Analogue of NovaBlindFold

**Paper**: Luca Dall'Ava, *LatticeBlindFold: A Lattice-Based Analogue of
NovaBlindFold*, ePrint 2026/1857 (ICME Labs, September 2026).

**Crate**: `crates/lattice-blindfold` (~4.3k lines, 71 tests, 1 bench).

## What the paper does

Every lattice folding scheme beyond Nova itself — SuperNeo, LatticeFold(+),
Cyclo — is only *randomizing*, not *blinding*: the folding transcript leaks
information about the witnesses being folded. LatticeBlindFold is the first
lattice-based, plausibly post-quantum analogue of NovaBlindFold: it makes
SuperNeo **blinding** (honest-verifier zero-knowledge) by three devices:

1. **Libra-style polynomial masking** of the Sum-Check transcript
   (`p = a0 + Σ p̃_i(X_i)`): unlike Libra, the mask is *never opened* — the
   final evaluation check takes place entirely at the level of the ABDLOP
   commitments, so `p(r')` and `Q(r')` never appear in the verifier's view
   as field elements (§4.1.1.1 proves the masking *perfect*: the whole
   transcript, `h` included, is uniform on its consistency class).
2. **ABDLOP commit-and-prove** replacing SuperNeo's plaintext evaluation
   hints: equality checks on the hints are performed homomorphically at
   the commitment level. Since ABDLOP is only secure over the base ring
   R_F, not over R_K where the Sum-Check lives, the paper gives a
   **componentwise instantiation** (§3.3): K = F[Y]/(Y²−ν) with ν = 2
   (a non-residue since q ≡ 5 mod 8), R_K-elements committed as their
   {1, Y}-coordinates, and every R_K-relation translated into a pair of
   R_F-relations once, at parameter generation (the rank-doubling
   embedding ψ).
3. **Rejection sampling** (Rej1/Rej2 of [LNP22, Lemma 2.4/2.14]) so the
   randomized folded instance-witness pair is simulatable — forcing the
   decomposition depth k = Θ(log n_F) (k = 31 at the paper's d=128
   parameters, versus SuperNeo's native k = Θ(1); this log-factor is the
   price of blinding, Remark 4.18).

The protocol is `Π_LBF = Π'_DEC ∘ Π'_RLC ∘ Π'_R1CS` (Protocol 12): one
interactive folding step taking an *uncommitted* R1CS instance (over the
blinded layout, the unpadded relation of Definition 3.28) and outputting
`k` committed evaluation claims `CE_com(b, L, B̃, Commit, T, K)` together
with the ABDLOP openings certifying them. It is statistically complete
(≤ 9·2^−λ batched), knowledge-sound (MSIS/MLWE/Extended-MLWE over the
cyclotomic ring R_F), and blinding for K = 1 — the verifier's entire view,
aborted rejection-sampling attempts included, is simulatable from the
public instances alone (Theorem 4.13, ε_LBF-blind ≈ 2^−112 at λ = 120).
The paper honestly records ten restrictions: no decider for the output
claims, no instance-in/instance-out interface, O(1) sequential rounds of
extraction, interactive-only security figures.

## What the crate implements

Every required part of the paper, over the repo's lattices (the
Ajtai/SIS stack at q = 2^64 − 59, the paper's own Solinas prime):

| Module | Paper locus | Content |
|--------|-------------|---------|
| `fp` | §2.1 | F_q at q = 2^64−59 with cascaded Solinas reduction; the ν = 2 non-residue check (q ≡ 5 mod 8) |
| `fq2` | §3.3 | K = F[Y]/(Y²−2): the extension arithmetic, the norm form N_{K/F} (the §3.3.5 determinant), eq() over K |
| `ring` | §2.1, §2.3 | R_F = F_q[X]/(X^d+1): negacyclic arithmetic, the τ_ℓ rotation basis (Lemma 3.12), the inner-product transform (§2.4.1), σ: X↦X⁻¹, the strong sampling set C ({−1,0,1,2}^d, T = 2d), split_b with the sign-aware balanced digits |
| `rk` | §3.3 | R_K in the {1,Y}-coordinates: Eq (3.5) multiplication, degree-0 closure (Lemma 2.20), packaged rotations with K-weights, ψ/φ (Eq 3.2), the quadratic rank-doubling R̂^(a)/R̂^(b) (Eq 3.9), the isometric norm |
| `embed` | §2.4.1 | The SuperNeo coefficient embedding, the M̄_j matrix lifts (ct(M̄z) = Mz row-wise), ring-valued MLEs with K-binding, eq-arrays |
| `gauss` | §3.6 | Discrete Gaussians (box-rejection, Lemma 3.19 tails), Rej1 with the exact likelihood ratio, Rej2 (⟨s₂,z₂⟩ ≥ 0), M = exp(14/ξ+1/2ξ²), Wmax, τ_{λ,ξ}, the Eq (4.18) width calibration, the fixed-point k of Eq (4.17) |
| `ajtai` | §2.5.1, §3.1 | The compact Ajtai L (Def 2.28) with homomorphic/relaxed-binding collision extraction; the **blinded layout** (Def 3.1) with pad/truncate and the Lemma 3.29 fibre; the regime-(1)/(2) hiding feasibility check |
| `abdlop` | §2.5.2, §3.3 | ABDLOP over R_F (A₁, A₂, B uniform over R_F — genuinely R_F, not ψ(R_K)); messages in BDLOP slots; R_K messages as (a, b) slot pairs; the homomorphic combination (Protocol 7's Step-21 check) |
| `pok` | §3.2 | The full PoK stack: Π_many^(1) (Lemma 3.6) with the (w, v, c, z₁, z₂) transcript and Rej1/Rej2; **Π_many^(2)** (Lemma 3.7) with the (g₀, g₁, g₂) garbage triple committed before the challenge; **Π_many^(ct)** (Protocol 4) with the ct-zero garbage and γ-masking; **Π_anc** (Def 3.15) with the Lemma 3.10 substitution; the S_ABDLOP simulator (Protocols 1–3); the two-transcript extractor with MSIS-kernel recovery; poly_inverse via extended Euclid |
| `sumcheck` | §3.4, Protocol 5 | The masked Sum-Check over K with degree-Dmax rounds: the mask's full round bookkeeping (the suffix-multiplicity constants — §4.1.1.1's "acting by 2^{ℓ−i}"), ζ, σ = ζP+T, the 5-point interpolation engine |
| `protocol` | §4 | **Π'_R1CS (Protocol 6, all 18 steps)**: the batched opening PoK, the α/γ^(1)/γ^(2)/γ^(3)/δ challenges, the Q(X⃗) composite, the packed-coeffs commitment, the σ-anchoring, the masked Sum-Check, the new-hint commitments, the degree-0 surrogates with the π_µ batched check (Lemma 3.13), the sq/cube/prod commitments, the quadratic PoK, the cF/cN/cE/u⋆ derivations, the final anchoring; **Π'_RLC (Protocol 7)**: the Wmax-capped mask loop with the Cy,0 tuple (Remark 4.21) and Cy,j commitments, the Rej1 fold, the in-the-clear Cy,0 opening, the homomorphic output checks; **Π'_DEC (Protocol 8)**: fresh-salt decomposition with the batched y_j − Σb^{i−1}y_{i,j} identity; the samplers (Protocols 9/10/11); **Π_LBF (Protocol 12)** with ι_bl; **Π°_LBF (Corollary 4.24)**; the **folding blueprint (Protocol 13)** |
| `params` | §4.3.4, §4.3.1 | The paper's Table 2 parameter sets (d=64: k=30, B_fold=3970, κ=19, ℓ=16, m₁/m₂=15/83, n_R,bl=1668; d=128: k=31, B_fold=8194, κ=11, m₁/m₂=15/53, n_R,bl=962) and the consolidated error-budget calculator reproducing Remark 4.19's ≈116/≈121-bit verdicts and Remark 4.23's 2^−112 blinding cap |

The integration tests (`tests/blindfold.rs`) prove: the CEcom relation
roundtrip at both salt regimes; each reduction end-to-end with tamper
rejection at every layer; the full Π_LBF with the ι_bl precomposition;
the unsatisfied-circuit rejection; the norm-violation rejection; the
medium-parameter run (d=8, nf=2^8, k=20); the accumulator-free variant;
the blueprint's optional-input branches; and the simulator's accepting
transcripts.

The bench (`examples/blindfold_bench.rs`) reports the end-to-end step at
both toy sets with the per-stage timings, the Wmax/N_PoK/interactive-bits
budget and the communication accounting.

## The honest-deviation ledger

Every interpretation call, recorded in the paper's own spirit (its §1.3
"ten restrictions" and Appendix B ledgers):

1. **The quadratic PoK's garbage triple (g₀, g₁, g₂)** — the paper cites
   [LNP22, Fig 6] for Π_many^(2) at k=1, σ=id. LNP22's exact figure is
   not reproduced in the paper text; we derived the mechanism from the
   paper's own Protocol-4 pattern: f(x̃₀ + cx) = f(x̃₀) + c·cross(x̃₀, x)
   + c²·quad(x), and the c² coefficient (the *pure quadratic part of the
   secret*) does **not** vanish when the relation f(x) = 0 holds — only
   its sum with the linear/constant parts does. The implementation
   commits all three coefficients before the challenge and certifies the
   c-dependent linear row `g₀ + c·g₁ + c²·g₂ = f(m̃(c))` with the public
   right-hand side computed from the transcript. Three rewinds sharing
   the first messages pin the triple (the extraction harness implements
   the honest-case recovery; the full adversarial argument is the
   paper's, cited).
2. **Step 18's anchoring target** — the paper routes the final
   evaluation check through u⋆ and the Lemma 3.10 substitution
   (u = Σwₙ·tB_{ι(n)} − u⋆). Our u⋆-bookkeeping derivation surfaced a
   factor-2 inconsistency in the coeffs-salt term that we could not
   resolve from the text alone; we anchor the *same* relation
   `ct(Σw·m + ζR_m^{(r')}·coeffs) = h` with the target h taken directly
   (equally public, equally binding — h is bound by the Sum-Check chain).
   The u⋆ value is still computed and carried in the transcript for
   documentation. The weights follow (4.2)–(4.3) exactly, including the
   `1_{i≤K}` indicator merging the f- and NC-families on the surrogate
   slot.
3. **The Eval-family blocks** — (4.2)'s `{s2,i,j}` weights act, via the
   Lemma 3.10 substitution, on the **Step-9 hint commitments y′_{i,j}**
   (the claims at the new point r′), not on the carried hints; the
   σ-anchoring (Step 7) conversely acts on the carried hints. Both are
   implemented as such and verified by the per-family decomposition
   tests.
4. **The wrapper's inner row aggregation** — Protocol 4's Step 5 runs
   Π_many^(1) on the *single* full-ring relation Σγⱼfⱼ + g = h. With N
   relations this is ONE row whose weights are the γⱼ-scaled weights
   summed slot-wise (several relations can act on the same slot), not N
   parallel rows. The anchoring path (N = 1) degenerates correctly.
5. **The garbage tuple splitting** — the paper commits the mask
   coefficients in one commitment (its ℓ = 16 covers them at paper
   scale) and discusses tuple-splitting only for Cy,0 (Remark 4.21). At
   toy scale the quadratic garbage (3·(2(K+k)+K) R_K messages) exceeds
   any single commitment, so we apply Remark 4.21's splitting
   generically (`commit_rk_tuple`): chunks of ⌊ℓ/2⌋ messages per block,
   with the row weights mapped through the tuple layout.
6. **Rej2 as rejection** — the paper's hybrids condition z₂ on
   ⟨s₂, z₂⟩ ≥ 0 "[LNP22]'s Rej2 branch". LNP22 realizes the conditioning
   by a sign-flip trick; we implement it as plain rejection (resample),
   which realizes the same conditional distribution at a constant-factor
   cost. The honest prover restarts the whole first message (fresh
   masks) after a few failed challenge draws — with fixed masks a deeply
   negative ⟨s₂, y₂⟩ would otherwise burn the entire Wmax budget, a
   restart pattern the paper's own Wmax accounting (Remark 4.3)
   anticipates.
7. **The τ-ℓ rotation indexing** — the paper's ct(f)_ℓ is 1-indexed
   (ℓ ∈ [1, d]); the crate follows it consistently (cf(ℓ) = .0[ℓ−1],
   τ_ℓ = −X^{d−ℓ+1}), including in the packaged rotations.
8. **The toy parameter sets** — the paper's Table 2 sets (n_F = 2^21)
   are carried verbatim in `params.rs` (with the security-budget
   calculator reproducing the paper's own bit estimates) but are not
   executable at test speed; the tests run the toy sets (d = 4/8) whose
   k is the honest fixed point of Eq (4.17) (k = 17/20 — the Θ(log n_F)
   depth is inherent, not a toy artifact). The toy |C| = 4^d gives no
   challenge-space security (the budget prints −3/+5 interactive bits —
   the honest number); the blinding machinery (Rej1 flattening, perfect
   Sum-Check masking, the simulator) is exercised and tested
   statistically regardless.
9. **Fiat–Shamir** — deliberately not implemented: the paper's §4.5
   quantifies that a quadratic extension no longer suffices under FS at
   any Q of cryptographic interest (q = 2^64−59 needs K = F_{q^4}); the
   interactive figures are what the crate's budget reports.
10. **The decider** — not constructed, exactly as the paper records
    (restriction 9): Π_LBF outputs the k CE_com claims plus the N_PoK
    ABDLOP openings; consuming them (directly or by arithmetizing the
    ABDLOP verification) is future work the paper itself defers.
11. **The blinded layout's constant-one wire** — Definition 3.1.(2)
    requires z_{ι1} = 1; the tests set it explicitly and the row-check
    helper enforces the convention (without it the blinding rows do not
    vanish — the first implementation fell into exactly this trap).
12. **norm_inf on negative-dominant vectors** — a real bug found by the
    medium-parameter test: max(sym).abs() underestimates the norm when
    the largest-magnitude coefficient is negative, corrupting the
    split_b digit count (the recomposition silently drops the residual).
    Fixed to max(|sym|) with an exhaustive roundtrip test.

## Bugs found and fixed during implementation (the debugging ledger)

- `Fq::add`/`reduce_u128` u64 overflow (2Q ≥ 2^64) — the field's
  foundation; caught by the axiom tests.
- The Rej1 likelihood-ratio sign: exp((−2⟨z,v⟩ **+** ∥v∥²)/2s²), not
  −(2⟨z,v⟩ + ∥v∥²)/2s² — a wrong sign silently biases the accepted
  distribution (the flattening test caught it as a v-sign leak).
- The c·u / ρ·y ring products: embedding a ring element as the degree-0
  K-scalar (c.ct(), 0) drops its non-constant coefficients — every
  R_F-scalar fold must be the full R_K product.
- The wrapper's challenge consistency: the γ⃗ used for h, the inner rows
  and the transcript must be the same draw (the caller's, in this
  architecture).
- The mask's Sum-Check round bookkeeping: each round's message carries
  suffix_len·(a₀ + Σ_{past} p̃(r)) + (suffix_len/2)·Σ_{future} Σa and
  suffix_len·p̃_i(X) — the paper's "the coefficients of p̃_i acting on
  the non-constant coefficients of h_i by 2^{ℓ−i}".
- The sumcheck Eval term's γ^(1) instance weight (the tensored challenge
  must appear in the eval_weights, not only γ^(2)γ^(3)).
- The P-functional weights: every round's univariate contributes
  2^{ℓ−1}·Σa (x_i = 1 on half the cube), not 2^{ℓ−i}.
- The surrogate slot indexing: ŷ_{i,1} is the M1 = I surrogate
  (per_s[0]) — the NC and f-families both subtract it; the product is
  ŷ_{i,2}·ŷ_{i,3} = per_s[1]·per_s[2].
- `poly_divmod`'s constant-divisor path returned a nonzero remainder
  (infinite Euclid loop) and `poly_sub_scaled` used termwise instead of
  polynomial products (silent Bézout corruption).

## Verification

- `cargo test -p lattice-blindfold`: **71 tests, 0 failures** (60 lib +
  11 integration).
- `cargo clippy -p lattice-blindfold --all-targets -- -D warnings`:
  clean.
- The bench (release): toy (d=4, nf=2^6, k=17) — prove 159 ms, verify
  8.4 ms, 167 ABDLOP + 18 compact Ajtai commitments, ~23k ring elements
  of masked openings; medium (d=8, nf=2^8, k=20) — prove 1.1 s, verify
  27 ms, 194 + 21 commitments, ~70k elements. At the paper's own scale
  (n_F = 2^21, d = 128, k = 31) the paper reports ≈ 8.2 MB per folding
  step (§4.3.3.1's table) — the crate's communication accounting mirrors
  those rows.
