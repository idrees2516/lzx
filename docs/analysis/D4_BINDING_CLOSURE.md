# The D4 Binding Closure — The Byte-Witness ↔ Commitment Authentication

**The formal statement of what `lattice-akita/src/salsa_binding.rs`
closes**, and of the r-column capacity split that scales it. The
measured evidence lives in BENCHMARKS.md §2l
(`cargo run -p lattice-akita --example salsa_bound_size`); the tamper
coverage is pinned in `salsa_binding.rs`'s tests.

---

## 1. The gap this closes

The open D4 response (`salsa_response.rs`) proved:

```text
claims ──RLC carrier──> f(r_sc) ──ψ-functional──> the byte-witness's MLE
                                              └─D1──> the byte-witness is short (≤ 255/coeff)
```

with the documented outer-layer gap: *nothing binds the byte-witness of
those proofs to the transmitted Ajtai commitment* — a cheating prover
could answer about any short byte-witness of its choosing. The
authenticated opening at the challenge was "the binding-complete
polylog route is the compact-mode fold."

## 2. The closure (the compact-fold composition)

`prove_grouped_salsa_bound` composes the response with the recursive
width-collapse chain over the SAME padded byte-witness:

```text
                    ┌─ (W0)  Σ p_i = t            the commitment equation, PROVEN
claims ─carrier─> f(r_sc) ─┤
                    ├─ (W0') Σ u_i = f(r_sc)      the functional thread
                    ├─ (W3)  the per-slice superposition identities
                    └─ (W2)  per-stage [A₂ | −T] ≥ 128+32 bits   THE BINDING
D1: the byte-witness's shortness certificate (the β₁ = 255 gate's justification)
```

The ψ-functional SUMCHECK of the open mode is subsumed — the chain's
`(W0')`/`(W3)` carry the identical linear functional
`Φ(v) = Σ_c eq(r_sc, x(c))·2^{8b(c)}·z(c) = f(r_sc)` (the same weights,
the same byte cube: the fold's `functional_of` and the sumcheck's
`vp2` agree term-for-term on balanced representatives).

**The statement proven** (the authenticated opening):

> the transmitted commitment `t` has a β₁-bounded (byte) preimage
> `v` under the public key `F̄` — *demonstrated*, not assumed — whose
> ψ-functional at the carrier's challenge equals `f(r_sc)`; the D1
> norm proof certifies the same witness's shortness (the Ajtai
> opening precondition); every chain stage's instance is
> estimator-gated.

The level-1 `[F̄]` MSIS hardness is never assumed (the (W0) check
proves the equation) — the binding rests on the per-stage
`[A₂ | −T]` instances, exactly the Sound-profile posture
("the level-1 response NEVER transmitted, the binding entirely the
width fold's instance"). The security narrative of the chain itself is
`docs/analysis/MULTISTAGE_EXTRACTION.md`.

**The closure is measured, not asserted**: `salsa_binding.rs`'s test
pins that a commitment to a DIFFERENT witness fails the chain's (W0)
(the binding the open mode lacked); the pipeline-level composition
test (`pipeline2.rs::v2_stage5_bound_composition_honest_and_tampered`)
pins that swapped column commitments reject at Stage 5.

## 3. The capacity law and the r-column split

**The cap.** The byte-packed D1 regime runs one norm proof per
commitment over `m` ring elements; the Lemma-4 gate
(`lattice-salsa::ring_norm::wraparound_gate`) requires

```text
m·n·B² < q/2   at B = 255 (one byte per coefficient)
```

At the reference shape (Q_32, ring dim 16): `m ≤ 1{,}547`, and with
the power-of-two slot discipline `m = 1{,}024` → **2,048 values per
commitment** (`byte_capacity`). The "~1,200 values" prose note in the
§2k ledger was this same cap, margin-rounded; the executable planner
now states it exactly and fail-closed.

**The split (the compact mode's discipline).** Beyond the cap,
`prove_grouped_salsa_split` lays the byte stream into `r` columns (the
flat domain's top log₂r value bits index the columns), each with:

* its own domain-separated Ajtai key (`column_seed(r, j)` — public
  randomness; production derives column keys from a ceremony),
* its own commitment (transmitted),
* its own D1 certificate (within the gate by construction — the
  planner sizes the columns),
* its own binding chain (the same closure as §2, per column).

The ψ-functional decomposes over the column variables — the eq
factorization `eq(r_sc, bin(x)) = μ_j · eq(r_sc_head, bin(m′))` with
`x = j·vpc + m′`:

```text
f(r_sc) = Σ_j μ_j·u_j,    μ_j = eq(r_sc[col_bits], bin(j))   (verifier-computed)
u_j     = Φ_j(v_j)        (column j's functional, bound by ITS chain)
```

Every layer is verifier-derived or checked: the μ weights from the
carrier's challenges, the decomposition as an exact field identity,
each `u_j` by its column's chain `(W0')`, each commitment by its
column's chain `(W0)`. Tamper coverage: a wrong `u_j` fails the
μ-decomposition; a wrong commitment fails its column's (W0); a tampered
D1 or fold stage fails its own layer (all pinned in tests).

**The honest scaling price.** The split response grows as
`r·(D1 + chain)` — measured 133.5 KB at 2^12 values (r=2) and
265.8 KB at 2^13 (r=4), 18.7× under the Clear mode at both scales.
The single-fold-over-columns composition (the LaBinius secondary
discipline that folds the r column-claims into ONE chain) is the
documented size-recovery follow-up; the Modulus-50 class is the
headroom route for the norm law at wider per-column gates.

## 4. What remains open (the honest ledger after this landing)

* the bound response's size grows with the stream (the chain's
  transmitted fold material) — the open mode remains the polylog-only
  path; folding the D1 certificates and the column chains into a
  single accumulator (the Quasar/PCD line) is the composition route;
* the per-column keys derive from a public domain (reference posture);
  production needs a setup ceremony;
* zero-knowledge (blinding the fold's responses) remains Wave 8.6
  (LatticeBlindFold's layer is the designated route).
