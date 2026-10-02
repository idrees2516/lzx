# LaBRADOR (ePrint 2022/1341) — IMPLEMENTED

Crate: `lattice-greyhound` (the `ring`/`challenge`/`sis`/`transcript`/`relation`/`jl`/`protocol`/`recursion`/`r1cs` modules). The sibling `lattice-labrador` crate remains the earlier Dachshund-reference port at Q = 2^48−59 with its own documented scope; this crate is the **paper-faithful engine at the papers' q ≈ 2^32 regime** (q = 2^32−99: prime, ≡ 5 mod 8, X^64+1 splits into two degree-32 factors — verified in `ring::tests::q_properties`).

## Coverage matrix (paper → code)

| Paper part | Realization | Tests |
|---|---|---|
| §2 challenge space C (32×±1 + 8×±2, op-norm ≤ T=14, invertible differences) | `challenge.rs` (partial Fisher–Yates + rejection; `poly_inv` division; LS18 Cor. 1.2 checked empirically) | 7 |
| §2 weak openings (ALS20 notion) | the CWSS module's relaxed-witness output + the extraction tests | (in `cwss`) |
| §4 JL lemma machinery (Lemma 4.1/4.2, b ≤ q/125) | `jl.rs` — ±1 matrices (reference mode) + ternary paper mode, rejection, the `q/125` bound in `sis::jl_max_norm` | 4 |
| §5.1 principal relation R (F / F′ families) | `relation.rs` — `DotCnst` with sparse terms, quadratic a-entries, `ct_only` | 5 |
| §5.2 Figure 2 (the main protocol) | `protocol.rs::prove_level` — inner/outer commitments, the digit layouts [i][j][ρ] / [pair][k], the JL projection, LIFTS=4 aggregation rounds, uniform α ∈ R_q^K / β ∈ R_q^4 F-aggregation, h-garbage = ½(⟨φᵢ,sⱼ⟩+⟨φⱼ,sᵢ⟩), u2, amortized opening z, f×b digit decomposition | via the roundtrips |
| Figure 3 (the verifier) | `protocol.rs::replay_level` + `verify_tail` — all twenty checks: lengths, challenge membership, SIS ranks at the announced norms, the JL bound, the b″ constant terms, the transcript consistency of every challenge, E1 B·[t̃;g̃]=u1, E2 D·h̃=u2, E3 Az=Σcᵢtᵢ, E4 ⟨z,z⟩=Σcᵢcⱼgᵢⱼ, E5 ⟨φ,z⟩=Σcᵢcⱼhᵢⱼ, E6 Σaᵢⱼgᵢⱼ+Σhᵢᵢ=b | 3+ |
| §5.3 recursion + splitting (the target relation, K′ = 2κ1+κ+3) | `protocol.rs::target_relation` — E1–E6 over `[z^(0..f−1), v]`; `recursion.rs` — the driver with the reference's shrinkage stopping rule | 3 |
| §5.4 norm bounds + decomposition params | `sis.rs::init_proof` (the k = 15..1 search, the varz/garbage variance models — the R·τ fold factor matched to the reference), §5.4's dynamic norm remedy implemented as the §5.4-documented "announce the measured norm" | 3 |
| §5.5 Theorem 5.1 (the rank conditions) | `replay_level`'s operational checks: κ secure for 6·T·SLACK·2^{(f−1)b}·√β′, κ1 for 2·SLACK·√β′ | via the roundtrips |
| §5.6 tail protocol (no outer commitments, 2r−1 interleaved garbage) | `prove_level(tail=true)` + `verify_tail` — h₀ before c₀, (h_{2i−1}, h_{2i}) between challenges, the c²-weighted verification identity | 2 |
| §5.7 proof size (the entropy model + the recursion optimization) | `recursion.rs` — `level_size_bits` ((u1len+u2len+LIFTS)·N·LOGQ + the JL entropy (log2‖p‖−4+2.05)·256 + 128-bit seeds) and `witness_size_bits` (log2(var)/2+2.05 per coefficient) | 1 |
| §6 binary R1CS (Figure 4) | `r1cs.rs::binary_r1cs_reduce` — (a,b,c,w,ã,b̃,c̃,w̃), the t-commitment F1, the σ^{-1}-conjugated binary checks ⟨x,x̃−1⟩, the a+b−2c Hadamard trick, λ F₂-combinations with the gᵢ responses (checked even) | 2 |
| §6 R1CS mod 2^64+1 (Figure 5) | `r1cs.rs::r1cs_mod_reduce` — NAF encodings (ternary digits, ‖Enc‖² ≤ d/2), the ϕᵢ challenge vectors, d̃ᵢ = Enc(ϕᵢ∘a), the c⁽ʲ⁾ aggregation, the X−2 divisibility checks, the encoded statement | 1 |
| §6 mixed R1CS | both reductions target the same principal relation → a single LaBRADOR execution over the concatenation (the paper's closing construction) | — |
| §7 concrete parameters (Tables 1–3) | `sizes.rs` + the bench: the level table, the analytic totals 34–53 KB across 2^26..2^30 (the paper: 46–58 KB), the near-constancy in N | 4 |
| Fiat–Shamir | `transcript.rs` — the 16-byte hash chaining with SHAKE-256 (the reference uses SHAKE128/AES-CTR) | 2 |

## The honest deviation ledger

1. **Uniform R_q α/β for the F-aggregation** (Theorem 5.1's distribution) — the
   reference pre-folds the level's verification equations E1–E6 with short
   quarternary challenges instead. We keep the K′ constraints separate and let
   the next level's α-aggregation handle them, exactly as §5.3 describes.
   Both are sound; ours follows the paper's proof.
2. **E4's quadratic structure is the full digit-cross form**
   `⟨z,z⟩ = Σ_{d,d′} 2^{(d+d′)b}⟨z^(d),z^(d′)⟩` over all part pairs —
   chunking-invariant and sound for any joining. The reference restricts
   a-entries to same-chunk-index pairs; the paper's "tridiagonal" sentence
   does not match its own equations — we implement the equations.
3. **Power-of-two decomposition bases** (b, b₁, b₂ ∈ {2^k}) — the reference's
   §6-documented deviation from the paper's general bases; adopted here.
   This costs a few percent of proof size (§5.7's analysis is base-agnostic).
4. **The JL matrices are ±1** (reference mode; the paper's Lemma 4.1 uses
   ternary {−1,0,1} with P(0)=½ — both modes are implemented in `jl.rs`,
   ±1 by default since the 53KB parameter tables are calibrated for it).
5. **Key windows A|B|C|D are disjoint** — the reference aliases some windows
   (e.g. D at offset 0); disjoint windows are strictly conservative for SIS.
6. **The conjugate equations x̃ = σ^{-1}(x) are not emitted as explicit
   constraints** — the binary quadratic checks pin the conjugates (the
   reference's dachshund does the same); the norm accounting carries them.
7. **§5.4's dynamic remedy**: when the measured output norm exceeds the
   heuristic prediction, the prover announces the measured bound (the paper:
   "restart … or increase the commitment parameters dynamically"). The
   verifier's SIS checks run against the announced bound, so binding is
   preserved. (At the paper's parameter scales the prediction matches the
   measurement within ~1% — see the bench's `[norm]` lines.)
8. **The gᵢ / gⱼ responses of the R1CS front ends are transmitted in the
   clear** (λ·32 bits each) per Figures 4/5 — not folded into the F′ lifting
   (the Greyhound PCS's evaluation claim IS folded, via the ct-only E5).
9. **SHAKE-256** replaces the reference's AES-CTR/SHAKE128 streams (same
   absorb discipline; the workspace's established port convention).

## Performance (release, single core, schoolbook i64/i128 arithmetic)

See `BENCHMARKS.md §3`. The 2^26-scale statement (34,791 ring elements =
2.2M coefficients): prove 52.5s / verify 44.3s / measured sub-proof 85.6 KB
(the analytic model at the same parameters: 31 KB — the difference is the
locally-optimized level parameters; the paper's globally-optimized parameters
are future work). The full PCS at 256–4096 ring elements: 1.9–4.7s prove,
34–45 KB total proofs — near-constant in N, the paper's headline property.
