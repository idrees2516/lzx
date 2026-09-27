# RoKoko (ePrint 2026/575) — PARTIAL

Crate: `lattice-rokoko` (313 lines — the crate's own comments admit
~1% paper coverage). Implemented: seed-derived ternary projection and
the two-stage coarse→fine same-kind projection with witness-level
verification shortcut; the modulus50 substrate for the kernel story.

Open (items 7.13, "2–5"): faithful Π^proj-c with committed Y_klin +
constraint rows (2), the recursive COM (Fig 1) with parbreak SIS
derivation (3), Π^fold-split + sumcheckify over the
`VirtualPolynomial` engine (4), Π^lin subfield-batched sumcheck (5),
Π^proj-f trace-dual (6), norm schedule + κ composition (7), the PCS
front end (8). The kernel story (incomplete NTT at q ≈ 2^50, 64
quadratic slots, Karatsuba 5→4, fused AVX-512 — 1.36–1.63x measured by
the authors) is unported.
