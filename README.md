# LZX — Lattice-Based Post-Quantum zkVM

**28k lines of pure-`std` Rust. 21 crates. 341 tests. Zero external dependencies.**

LZX is a from-scratch, production-oriented implementation of the modern lattice-based
zero-knowledge proof stack: it implements **eleven research papers** end-to-end (prover +
verifier + exact algebraic identity tests), ports the **labinius** lattice PCS and the
**LaBRADOR** proof system as native Rust, and assembles them into a proving zkVM for
RV64IMAC programs with a bounded canonical proof envelope.

The labinius PCS path runs **upstream's AVX-512 kernel designs natively** (runtime-detected,
pure `std` intrinsics, exact scalar fallbacks): vertical batch-of-32 binary NTT kernels with
`vpermb` lookup tables and lazy reduction, the `vpmaddwd` raw-accumulation commitment MAC with
compile-time fold-back periods, `PCLMULQDQ` binary-field arithmetic — all verified bit-exact
against the scalar reference. See **`PERFORMANCE.md`** for the full efficiency analysis:
**commit 37x, fold 7x, evaluate 35x, reference round 17.5x end-to-end**.

**Wave 6 (shared substrate + soundness-critical fixes)** is in: paper-calibrated short
**ring-element** challenge distributions with certified operator-norm bounds (the family-wide
challenge-space fix), hard norm wraparound gates on every folding module, the Ajtai
cached-NTT fast path (**4.2x** on every commit/verify), the committed Quasar lookup protocol
(closing the verifier-binds-nothing hole), F_{q²} extension fields, the ~2^50
quadratic-slot incomplete NTT at RoKoko's own modulus, zero-skipping pay-per-bit commitment
inputs, and a zero-dependency SIS security estimator (ADPS16/BDGL16/LGSA). See
`NEXT_STEPS.md` for the per-paper research backlog driving Waves 6-8.

```
prove_program(RV64IMAC bytecode)  ->  Proof envelope  ->  verify_program(envelope) == Ok(())
```

## Papers implemented

| # | Paper | Crate | What is implemented |
|---|-------|-------|---------------------|
| 1 | **ProtogaLattice** (constant-round folding) | `lattice-folding` | Cross-term extraction via finite-difference Newton inversion (diagonal snapshots); exact fold identity for degree-2 and degree-3 relations |
| 2 | **Akita** (lattice PCS) | `lattice-akita` | Packed commitments, sumcheck evaluation proofs with norm-checked openings, grouped openings, schedule catalog + security profiles |
| 3 | **Cyclo** (lattice PCS) | `lattice-folding` | Extension commitment (iterative-borrow chunking, exact recomposition), partial range checks, accumulator with additive norm growth + refresh |
| 4 | **HyperWolf** (lattice PCS) | `lattice-pcs` | Standard-soundness PCS backend + the `PcsBackend` trait boundary |
| 5 | **LatticeFold+** (folding + Ajtai commitments) | `lattice-folding` | Algebraic range proof (eq-multiplied booleanity sumcheck + point reconstruction), double-commitment folding, tensor rings |
| 6 | **PikkuFold** (folding) | `lattice-folding` | Layered biased-ternary random projections with certified JL norm bounds, no in-fold commitments, linear-relation binding |
| 7 | **Quasar** (lookup arguments) | `lattice-lookup` | **Committed** grand-product lookup (Q1: Ajtai commitments to T/R/Q, τ from commitments, counting-map difference, forged-triple rejection) + partial-evaluation multi-instance accumulation |
| 8 | **RoKoko** (lattice PCS) | `lattice-rokoko` | Coarse/fine two-stage committed refinement with ternary projections; incomplete-NTT completion |
| 9 | **SALSA** (zk sumcheck) | `lattice-salsa` | Norm sumcheck, LDE tensor relation, structured (negacyclic) matrix checks, zk sumcheck with statement-derived masks |
| 10 | **Symphony** (folding + SNARK) | `lattice-folding` | High-arity (mu-ary) one-shot folding with full subset cross-term bookkeeping; exact mu-ary identity verified |
| 11 | **Twist & Shout** (small-space zkVM) | `lattice-memory`, `lattice-vm`, `lattice-zkvm` | Twist (read/write timeline) and Shout (read-only table) checks with grand-product fingerprint identities; canonical RV64IMAC decoder + deterministic executor + subword-correct sparse memory + LR/SC & AMO atomics; end-to-end prove/verify |

Plus two ports of external systems:

| System | Crate | Notes |
|--------|-------|-------|
| **labinius PCS** (osdnk/labinius `crates/pcs`) | `lattice-labinius` | Full PCS: const-evaluated ring tables (conductor-1944 split + conductor-972 quadratic), exact mixed-radix scalar NTT, GF(2^162) with carry-less multiplication, Ajtai commitment key (7 moduli), SHAKE-256 Fiat-Shamir, weight-28 bounded challenges, slot-domain fold, eval layer, bit-dropped opening (Garner digits + residual norm check), Clear + BitDropped modes, reference round |
| **LaBRADOR** (lattice-dogs vendored C) | `lattice-labrador` | Native pure-std Rust: Z_Q[X]/(X^64+1), Q=2^48-59, exact i64/i128 arithmetic, SIS-rule parameters, inner/outer commitments with digit decomposition, JL projection with rejection, amortization, full verify; simple-statement API with content digests |

## Architecture

```
lattice-core          Goldilocks field (carry-compensated), Keccak-f1600/SHA3/SHAKE,
                      Fiat-Shamir transcript, dense MLEs, gadget decomposition,
                      challenge sets (sparse ternary / uniform / small interval),
                      short ring-element challenges with certified Γ_C bounds (W6),
                      symbolic NormBudget hard gates (W6), F_{p²} extension field (W6)
lattice-ring          Negacyclic NTT (CT/GS, psi-scaling, Barrett-reduced hot path),
                      R_q arithmetic, incomplete NTT + completion, 3x22-bit split
                      packing, CRT carriers, R_q[Y]/(Y²+1) extension ring (W6),
                      Modulus50 quadratic-slot incomplete NTT (W6)
lattice-commitment    Ajtai Module-SIS commitments (seed-derived A, cached-NTT fast
                      path, zero-skipping MAC, statement-absorption API),
                      ABDLOP-style linear proofs, digit-decomposed norm proofs,
                      bit-packed one-hot column packing (pay-per-bit)
lattice-sumcheck      Generic virtual-polynomial sumcheck, Spartan-style zerocheck,
                      batched claims
lattice-relations     CCS with sparse matrices + RLC utilities
lattice-folding       ProtogaLattice, LatticeFold+, Cyclo, PikkuFold, Symphony, SuperNeo
                      (all with hard norm gates + public-coin FS hygiene)
lattice-lookup        Quasar lookups (committed Q1 protocol + accumulation)
lattice-salsa         SALSA norm/LDE/structured-matrix/zk sumchecks
lattice-rokoko        RoKoko two-stage refinement
lattice-akita         Akita PCS (full)
lattice-pcs           PcsBackend trait + HyperWolf backend
lattice-embeddings    Hachi-style slot embeddings + trace functionals
lattice-labinius      labinius PCS port
lattice-labrador      LaBRADOR native Rust port
lattice-sis-estimator Offline SIS security estimator: ADPS16/BDGL16 costs, LGSA
                      simulator, infinity + Euclidean attack paths (W6)
lattice-vm            RV64IMAC decoder (all base+M+A incl. compressed), executor,
                      trace rows, subword-correct sparse memory, LR/SC + AMO
lattice-memory        Twist & Shout grand-product memory checks
lattice-zkvm          End-to-end prove_program / verify_program, canonical envelope
lattice-zk            Zero-knowledge layer: secret entropy, HVZK simulators,
                      ABDLOP ZkLinearProof, Libra-style blinded zk-sumcheck
                      (chi-square KATs)
lattice-qrom          QROM accountability: query ledger, attestations,
                      production domain registry, composition review
lattice-bench         Pure-std reproducible benchmark matrix (35 stages + sizes)
```

## Guarantees carried in-tree

- **341 tests, 0 failures, 0 clippy warnings** — every fold identity, PCS round, and
  VM conformance class is verified exactly (algebraic identities, not statistical approximations).
- **Differential ISA conformance** — a second, independent byte-level RV64IMAC interpreter
  (`lattice-vm/reference.rs`) is compared against the traced executor over 131 randomized
  programs; golden vectors cover every instruction class.
- **Zero-knowledge with machine-checked simulators** — HVZK simulator + chi-square KAT
  (alpha = 0.001) for ABDLOP linear proofs; distributional simulator for blinded sumcheck.
- **QROM accountability** — query ledgers with worst-case rejection amplification,
  digest-bound attestations, and a composition-review protocol as the only granting path
  for the `QromFiatShamir` capability.
- **Envelope hardening** — total-bytes cap enforced before allocation; 4000-mutation
  no-panic fuzz corpus; strict version/caps/duplicate/trailing-byte rejection.
- **KAT manifest** — 29 digest-pinned known-answer vectors
  (field / transcript / NTT / packing / commitment / zk / mle).
- **Soundness-critical hard gates (Wave 6)** — norm wraparound gates `β < min(q/2, β*)`
  on every folding module (wraparound mod q silently destroys SIS binding; folds refuse
  instead); paper-calibrated ring-challenge distributions with certified operator norms;
  the committed Quasar lookup path (statement-bound τ, SIS-bound openings, counting-map
  multiset verification); a SIS security estimator pricing the toy-parameter regime
  honestly (see `lattice-sis-estimator` and `AUDIT_CHECKLIST.md`).

See `SECURITY.md` (capability statement, threat model, four fixed-vulnerability
post-mortems) and `AUDIT_CHECKLIST.md` (G1-G8 evidence map).

## Build & test

```bash
cargo test --workspace      # 341 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo run --release -p lattice-bench --bin lattice-bench   # 35-stage benchmark matrix
cargo run --release -p lattice-labinius --example round_bench       # labinius reference round
cargo run --release -p lattice-labinius --example backend_bench     # scalar vs AVX-512 backends
cargo run --release -p lattice-labrador --example cmod_bench        # LaBRADOR reduction/products
```

No external dependencies; builds with stable Rust (1.75+). Benchmarks are pure-`std`
and reproducible (median-of-runs timing harness). The AVX-512 / `PCLMULQDQ` backends are
runtime-detected (`is_x86_feature_detected!`) with the exact scalar reference paths as
fallback — the same binary runs unchanged on machines without the features.

## Performance snapshot

Reference-round of the labinius PCS at sizem (2^18 GF(2^162) elements, 128 columns,
3889+2917), pure-`std` Rust on 2 cores — **before → after** the AVX-512 backend:

| Stage | scalar | AVX-512 backend | speedup |
|-------|--------|-----------------|---------|
| commit | 3134 ms | 84 ms | 37x |
| evaluate | 327 ms | 9.4 ms | 35x |
| fold | 332 ms | 48 ms | 7x |
| challenge | 12 ms | 12.6 ms | — (hash-bound) |
| verify | 78 ms | 67 ms | 1.2x |
| **total round** | **3885 ms** | **222 ms** | **17.5x** |

Kernel-level (batch of 32 ring elements): forward NTT **270x**, commitment MAC + finish
**436x**, carry-less multiply **66x** (PCLMULQDQ). Full analysis, technique map and roadmap:
`PERFORMANCE.md`.

## Security notes

This is research-grade code implementing preprint and conference protocols. Parameter
sets are illustrative and **must be reviewed before production use** (see
`AUDIT_CHECKLIST.md`, external items). Fiat-Shamir is ROM-shaped with QROM composition
reviews recorded in `lattice-qrom`; the ZK capability is granted only behind
machine-checked simulators. Fixed vulnerabilities and their post-mortems are documented
in `SECURITY.md`.

## License

MIT.

## Wave 7 state (2026-09-29)

Protocol completion landed: ProtogaLattice PGL-Fold/PGL-Boot
(`crates/lattice-folding/src/pgl.rs`), SALSAA D1+D2 + **the A2–A5
stack** (`crates/lattice-salsa/src/{ring_sc,salsaa,air}.rs`: Π_norm+,
Π_bin, the staircase RoK, the VDF binary staircase, committed-AIR +
folding), **the full HyperWolf Protocols 1/2/3**
(`crates/lattice-pcs/src/hyperwolf.rs` — ring mapping + balanced
gadget + leveled commitment, the guarded recursive evaluation, k-round
folding, the certified challenge space, own u64 ring at q ≡ 5 mod 8),
**the RoKoko committed-refinement core**
(`crates/lattice-rokoko/src/{com,protocol}.rs` — recursive COM Fig 1,
Ξ^lin_COM, Π^fold-split, sumcheckify, Π^lin, the round driver), the
labinius wire/ + Recursive layers, **Serval** (the slack-free
split-and-fold IPA, `crates/lattice-labrador/src/serval.rs`), and the
**Hachi ring-switch** (`crates/lattice-embeddings/src/ring_switch.rs`)
— all ported from the lattice-zk-lab reference implementation to this
workspace's pure-std conventions. Full part-by-part paper coverage —
implemented / partial / unimplemented — lives in [`docs/`](docs/):
start at [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/papers/README.md`](docs/papers/README.md).

Testing: `cargo test --workspace` (519 tests at this commit);
`cargo clippy --workspace` clean. Benchmarks: `cargo run --release -p
lattice-bench --bin bench` (26 stages, reproducible matrix) — see
`PERFORMANCE.md` for the methodology.
