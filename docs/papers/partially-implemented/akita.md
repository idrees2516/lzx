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

Open: the successor-witness keystone completion, the
α-after-commitment ordering test (App F.1's Grand-Danois repair), the
digit-range sumcheck with symmetry halving, tensor reduction + trace
functional, the terminal + recursion driver, planner/validator,
setup offloading, commitment compression, batching/chunking,
exact-ℓ2 certificates, the LHL hiding layer, and QROM/FS ledger
registration.
