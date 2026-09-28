# Akita (ePrint 2026/1983) — PARTIAL

Crate: `lattice-akita` (~720-line kernel-scale demonstrator; the
~260k-LOC reference is the ground truth). Implemented: packed
Goldilocks→R_q commitment (3×22-bit limbs), eq·f sumcheck evaluation
proof, full-witness opening + digit-revealing NormProof (the pre-SALSAA
response layer), RLC grouped openings, schedule catalog + review-gated
security profiles.

Partially landed (Wave 7.11, `fold.rs` 1181 lines + `ring_check.rs` 830
lines): the fold core (A1-class: two-tier keys, response digitization,
fold equations) and the ring-relation checks — 26 akita tests green.

Landed (Wave 7.3 D4, `salsa_response.rs`): the response-layer swap —
`prove_evaluation_salsa`/`verify_evaluation_salsa` replace the full
packed-witness opening + digit-revealing `NormProof` with the SALSAA
`D1 ∘ D2` chain (`prove_norm_chain`): the response is polylog (the
carrier sumcheck + the two chain sumchecks + O(1) claims) and discloses
nothing but the challenge evaluations `z(r)`/`f(r_sc)` — the
Θ(N)-disclosure path remains available as the binding-complete Clear
mode. The carrier terminal is bound to the claimed `f(r_sc)` through the
verifier's own `eq(r, r_sc)` factor; D1 replays the Lemma-4 gate on both
sides with the response's declared norm bound; D2's terminal is the
verifier's own zero-communication row evaluation. **Documented gap**: the
Ajtai binding of `z(r)`/`f(r_sc)` to the commitment — SALSA(A)'s
authenticated opening at the challenge — awaits the paper's outer layer
(the kernel keeps Clear mode binding-complete alongside). 4 tests:
happy + polylog shape, tampered `f_term`/`z_r`/carrier-round rejections,
the chain-after-carrier isolation, the standalone 128-coefficient chain.

Open: the successor-witness keystone completion, the
α-after-commitment ordering test (App F.1's Grand-Danois repair), the
digit-range sumcheck with symmetry halving, tensor reduction + trace
functional, the terminal + recursion driver, planner/validator,
setup offloading, commitment compression, batching/chunking,
exact-ℓ2 certificates, the LHL hiding layer, and QROM/FS ledger
registration.
