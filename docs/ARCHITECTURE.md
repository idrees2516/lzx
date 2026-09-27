# LZX Architecture

LZX is a lattice-based post-quantum zkVM proof stack: it executes RV64IMAC
programs, proves the execution with lattice (Module-SIS) commitments and
sumcheck-driven PIOPs, and accumulates the proof state through a family of
folding schemes realized from the recent lattice-folding literature. The
workspace is 21 crates, ~40k lines of Rust, organized in five layers.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5 — Consumers: lattice-zkvm (prove/verify/envelope),     │
│            lattice-bench, KAT manifests, fuzz corpora           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 — Protocols: lattice-folding (PGL, LF+, Cyclo,         │
│            PikkuFold, Symphony, SuperNeo), lattice-salsa        │
│            (SALSAA), lattice-lookup (Quasar), lattice-akita,    │
│            lattice-pcs (HyperWolf), lattice-rokoko,             │
│            lattice-memory (Twist & Shout PIOPs)                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — Proof engine: lattice-sumcheck (sumcheck/zerocheck/  │
│            batch/virtual polys), lattice-qrom (query ledger,    │
│            composition attestations), lattice-zk (blinded       │
│            sumcheck, ABDLOP zk-linear, CRT carriers)            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Algebra: lattice-core (Goldilocks field, Fq2,        │
│            transcripts, MLEs, short challenges, norm budgets,   │
│            gadget decomposition, Keccak), lattice-ring          │
│            (R_q, NTT, Modulus50, packing, extension)            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 — Kernels: lattice-labinius (AVX-512 PCS port,         │
│            Barrett/Montgomery, wire/), lattice-labrador         │
│            (native LaBRADOR), lattice-commitment (Ajtai with    │
│            cached-NTT + zero-skip, sparse one-hot, norm proofs) │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1 — Kernels

**lattice-labinius** (~7.5k LOC): the port of the upstream labinius PCS.
The AVX-512 backend is bit-exact against the scalar path (kernels
147–448x, full commit 37.7x, reference round 3885→222 ms = 17.5x). Wave 7
added `wire.rs` (item 7.16): LSB-first bit-packing and a static-histogram
rANS entropy coder (64-bit state, per-symbol ryg renorm thresholds,
transmitted histogram, strict decode). The declared-but-unwired
`Opening::Recursive` mode (item 7.5) remains partial — see the labinius
status file.

**lattice-labrador**: a native exact-arithmetic LaBRADOR realization
(i128 schoolbook product; the 8-prime RNS `polx` port that would give
10–50x prove speed is Wave 8.3, unimplemented).

**lattice-commitment**: Ajtai Module-SIS commitments with the Wave-6
fast path — cached NTT of the public matrix at keygen, NTT of the
witness once, pointwise accumulation, one inverse per row; zero
elements are skipped ("0s are free"). `sparse.rs` packs one-hot columns;
`norm_proof.rs` is the pre-SALSAA digit-revealing norm check (being
replaced by the SALSAA chain, item 7.2→7.3).

## Layer 2 — Algebra substrate

**lattice-core**: the Goldilocks field (2^64 − 2^32 + 1) with batch
inversion; in-house Keccak-f[1600]/SHAKE transcripts with domain
separation; dense MLEs; gadget digit decomposition. The Wave-6 soundness
substrate: `short_challenge.rs` (paper-calibrated fixed-weight ternary /
biased ternary / small-set ring challenges with certified op-norm bounds
Γ_C and rejection), `norm_budget.rs` (symbolic β with hard gates
β < min(q/2, β*)), `extension.rs` (F_{p²} arithmetic + transcript
sampling — the F_{q^e} sumcheck substrate).

**lattice-ring**: negacyclic R_q rings (u32 moduli, full-split NTT) plus
`modulus50.rs` (q ≈ 2^50, q ≡ 129 mod 256, incomplete NTT — the Wave-6
substrate for Cyclo/PikkuFold/RoKoko/SALSA parameter families) and an
extension-ring layer.

## Layer 3 — Proof engine

**lattice-sumcheck**: the shared engine (prover/verifier over
`VirtualPolynomial`s, zerocheck, batching). Constant-factor work
(single-binding rounds, in-place fix_variables) is Wave 8.5.

**lattice-qrom**: the QROM query ledger, domain registry, attestations
and composition review (granting QROM_FiatShamir); grinding bounds as
machine checks remain Wave 8.7.

**lattice-zk**: Libra-style blinded sumcheck (correct FS ordering:
pre-round commitment before challenges) and ABDLOP/Lyubashevsky
zk_linear with rejection sampling; distributional simulators with
chi-square KATs; CRT-carrier relations; capability tokens. Not yet wired
into the zkVM (Wave 8.6).

## Layer 4 — The paper protocols

Each crate realizes one (or two) papers; see `docs/papers/` for
part-by-part status. The Wave-6/7 soundness architecture shared by all
of them: ring-element challenges from strong sampling sets (never small
integers), norm budgets with wraparound gates, statement-absorbing
transcripts, and verifier-facing equations test-pinned against tampering.

The deepest Wave-7 completions:

* **ProtogaLattice** (`lattice-folding/pgl.rs`): the full PGL-Fold
  protocol — three RO rounds (δ/α/γ ring challenges from
  C = {−1,0,1,2}^N fixed-weight), F(X) with the pow-tower compression,
  β* = β + αδ re-randomization, Gröbner division of H(Y) by the chain
  ideal ⟨Y_a·Y_b − Y_a⟩ producing quotients K_ab with zero remainder
  (asserted in code), the e*-check verifier, the ghost zero-witness
  partition of unity, y₀ := 1 for linear norm growth; plus PGL-Boot
  (base-b balanced-digit refresh, the D-point identity, LF+ range
  attachments). Constraint semantics are ring-product (the paper's
  f: R^m → R_q), closing the E3 gap.
* **SALSAA** (`lattice-salsa/ring_norm.rs`): D1 the ring norm sumcheck
  with the Lemma-4 fail-closed wraparound gate and the F_{q²}
  conjugation/trace terminal; D2 the `LinRelation` LDE-tensor
  linearization (zero-communication rows); the composed norm chain
  ready for the response-layer swap.
* **labinius wire/** (`lattice-labinius/wire.rs`): bit-packing + rANS.

## Layer 5 — The zkVM

**lattice-zkvm**: proves program executions. The Wave-7 state: the
envelope discipline is strong (strict decode, 4000-mutation fuzzing),
but the proof path still carries the pre-Wave-7 response layer
(full-witness disclosure, sumcheck checked by length, verifier
re-execution) — the SALSAA swap (7.3) and the Twist & Shout PIOPs
(7.4) are the two open soundness-critical items; both have complete
designs and integration points (see status files).

**lattice-vm / lattice-memory**: the RV64IMAC conformance layer
(decoder/executor/reference + differential corpus with 131 randomized
programs and golden vectors; four real VM bugs fixed by the corpus in
Wave 3) and the memory-checking oracles (`twist_check`/`shout_check`
ground truth; the paper's PIOP machinery is item 7.4).

## Security posture

Toy parameters throughout (n = 16–64, q ≈ 2^31.6 or 2^50, k = 2) — the
protocols are exact and adversarially tested but NOT production-hardened:
`lattice-sis-estimator` (ADPS16/BDGL16 Core-SVP) exists to gate real
parameter choices (Wave 8.8 instantiates them). `SECURITY.md` carries the
capability statement, entropy rules, and four fixed-vulnerability
post-mortems; `AUDIT_CHECKLIST.md` maps evidence G1–G8.

## Where to look for what

| Question | Answer |
|---|---|
| How is a fold proven sound? | `lattice-folding/pgl.rs` (e*-check + Gröbner division) |
| How are norms bounded? | `lattice-core/norm_budget.rs` + `lattice-salsa/ring_norm.rs` |
| How do challenges get entropy? | `lattice-core/short_challenge.rs` (fixed-weight ring elements) |
| What is fast? | `lattice-labinius` AVX-512 + `PERFORMANCE.md` |
| What is honest-but-incomplete? | every `docs/papers/*-implemented` file lists it |
