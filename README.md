# LZX — Lattice-Based Post-Quantum zkVM

**24.5k lines of pure-`std` Rust. 20 crates. 240 tests. Zero external dependencies.**

LZX is a from-scratch, production-oriented implementation of the modern lattice-based
zero-knowledge proof stack: it implements **eleven research papers** end-to-end (prover +
verifier + exact algebraic identity tests), ports the **labinius** lattice PCS and the
**LaBRADOR** proof system as native Rust, and assembles them into a proving zkVM for
RV64IMAC programs with a bounded canonical proof envelope.

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
| 7 | **Quasar** (lookup arguments) | `lattice-lookup` | Grand-product lookup checks + partial-evaluation multi-instance accumulation |
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
                      challenge sets (sparse ternary / uniform / small interval)
lattice-ring          Negacyclic NTT (CT/GS, psi-scaling), R_q arithmetic,
                      incomplete NTT + completion, 3x22-bit split packing, CRT carriers
lattice-commitment    Ajtai Module-SIS commitments (seed-derived A),
                      ABDLOP-style linear proofs, digit-decomposed norm proofs
lattice-sumcheck      Generic virtual-polynomial sumcheck, Spartan-style zerocheck,
                      batched claims
lattice-relations     CCS with sparse matrices + RLC utilities
lattice-folding       ProtogaLattice, LatticeFold+, Cyclo, PikkuFold, Symphony, SuperNeo
lattice-lookup        Quasar lookups
lattice-salsa         SALSA norm/LDE/structured-matrix/zk sumchecks
lattice-rokoko        RoKoko two-stage refinement
lattice-akita         Akita PCS (full)
lattice-pcs           PcsBackend trait + HyperWolf backend
lattice-embeddings    Hachi-style slot embeddings + trace functionals
lattice-labinius      labinius PCS port
lattice-labrador      LaBRADOR native Rust port
lattice-vm            RV64IMAC decoder (all base+M+A incl. compressed), executor,
                      trace rows, subword-correct sparse memory, LR/SC + AMO
lattice-memory        Twist & Shout grand-product memory checks
lattice-zkvm          End-to-end prove_program / verify_program, canonical envelope
lattice-zk            Zero-knowledge layer: secret entropy, HVZK simulators,
                      ABDLOP ZkLinearProof, Libra-style blinded zk-sumcheck
                      (chi-square KATs)
lattice-qrom          QROM accountability: query ledger, attestations,
                      production domain registry, composition review
lattice-bench         Pure-std reproducible benchmark matrix (26 stages + sizes)
```

## Guarantees carried in-tree

- **240 tests, 0 failures, 0 clippy warnings** — every fold identity, PCS round, and
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

See `SECURITY.md` (capability statement, threat model, four fixed-vulnerability
post-mortems) and `AUDIT_CHECKLIST.md` (G1-G8 evidence map).

## Build & test

```bash
cargo test --workspace      # 240 tests
cargo clippy --workspace -- -D warnings
cargo run --release -p lattice-bench          # 26-stage benchmark matrix
cargo run --release -p lattice-labinius --example round_bench   # labinius reference round
```

No external dependencies; builds with stable Rust (1.75+). Benchmarks are pure-`std`
and reproducible (median-of-runs timing harness).

## Performance snapshot

Reference-round of the labinius PCS at sizem (2^18 GF(2^162) columns, 128 columns,
3889+2917 moduli), pure-std Rust on 2 cores:

| Stage | Time |
|-------|------|
| commit | 3.13 s |
| point eval | 2 ms |
| evaluate | 327 ms |
| challenge | 12 ms |
| fold | 332 ms |
| verify | 78 ms |
| **clear-mode proof floor** | **915 KB** (commitment 249 KB + opening 664 KB + row eval 3 KB) |

Full matrix: `cargo run --release -p lattice-bench` (writes `timings.csv` + `sizes.csv`).

## Security notes

This is research-grade code implementing preprint and conference protocols. Parameter
sets are illustrative and **must be reviewed before production use** (see
`AUDIT_CHECKLIST.md`, external items). Fiat-Shamir is ROM-shaped with QROM composition
reviews recorded in `lattice-qrom`; the ZK capability is granted only behind
machine-checked simulators. Fixed vulnerabilities and their post-mortems are documented
in `SECURITY.md`.

## License

MIT.
