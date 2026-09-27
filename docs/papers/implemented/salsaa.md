# SALSAA (ePrint 2025/2124) — D1+D2 IMPLEMENTED

Crate: `lattice-salsa/src/ring_norm.rs` (Wave 7.2) alongside the four
Wave-6 gadgets in `lib.rs` (field-level norm sumcheck, LDE tensor,
structured matrices, zk round-masking).

## Implemented

| Paper part | Realization | Tests |
|---|---|---|
| D1 Π^norm ∘ Π^sum over R_q: trace identity ‖x‖² = Trace(⟨x, x̄⟩), conjugation, CRT-slot final check | `prove_ring_norm`/`verify_ring_norm` with the F_{q²} terminal (`challenge_fq2`, embedded norm identity) | `ring_norm_honest_roundtrip` |
| Lemma-4 no-wraparound condition (B'^ρ < q/2 family) | `wraparound_gate` — enforced at prove AND verify, fail-closed | `ring_norm_wraparound_gate_fails_closed` |
| Integer norm bound (not a modular identity) | claimed-norm reconstruction + bound envelope | `ring_norm_rejects_out_of_bound`, `ring_norm_tampered_claim_rejected` |
| Authenticated z(r) opening contract | `z_opening` parameter (caller's PCS layer) | `ring_norm_wrong_opening_rejected` |
| D2 Π^lde-⊗ linearization: row-tensor F, zero extra communication | `LinRelation` — `row_at` verifier-side eq-tensor rows; `prove_lde`/`verify_lde` | `lin_relation_lde_roundtrip`, `lin_relation_rejects_non_lde` |
| Composed response-layer chain | `prove_norm_chain`/`verify_norm_chain` (Π^norm ∘ Π^sum + LDE leg) | `norm_chain_composes` |

## Open (Wave 7/8 roadmap)

* D3 batching into `lattice-sumcheck::batch`; **D4 the Akita/zkVM
  response-layer swap** (Θ(N)→polylog, disclosure removed) — the
  composed chain API is the integration point; D5 fold/split/join
  protocol set; D6 engine constant-factor pass (Wave 8.5 overlaps);
  D7 truth-in-advertising QROM registration.
* The paper's 10.61 s @ 2^28 performance posture needs IFMA/HEXL-class
  kernels + parallelism (Wave 8.2).
