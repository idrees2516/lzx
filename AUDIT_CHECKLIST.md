# LZX Audit Checklist (gates G1-G8)

The engineering-side evidence map for the implementation audit's
release gates. "In-tree" items are machine-checked by the workspace's
test suite; "external" items require review beyond this codebase.

## G1-G3 — build, semantics, integration (baseline)

- [x] Single canonical workspace; every crate is a member
      (`cargo metadata` clean; zero external dependencies).
- [x] One canonical RV64IMAC semantics: decoder + executor + trace rows.
- [x] Differential conformance corpus (`lattice-vm`):
      - golden vectors for every instruction class at its edge cases,
      - an independent byte-level reference interpreter,
      - 131 randomized programs cross-checked (regs, memory bytes,
        pc, halt state).
- [x] End-to-end prove/verify over the Akita PCS with the bounded
      canonical envelope (strict decoder: version/duplicates/trailing
      rejection).
- [x] Memory checks: register/RAM Twist fingerprint identities
      (`lattice-memory`), exact algebraic identity tests.

## G4 — privacy (zero knowledge)

- [x] Privacy specification and leakage declarations
      (`lattice-zk::privacy_spec`).
- [x] Capability-typed security claims; `ZeroKnowledge` granted only
      with a machine-checked simulator.
- [x] OS-entropy separation (type-level); nonce ledger for seed reuse.
- [x] Published-mechanism masking (no ad-hoc tricks):
      - Libra-style blinded sumcheck with committed masking MLE,
      - ABDLOP/Lyubashevsky ZK linear proofs with rejection sampling.
- [x] CRT-carrier relations bridging field claims to exact mod-q
      relations (committed, norm-bounded carry limbs).
- [x] Distributional simulators for both ZK layers; statistical KATs
      (chi-square, alpha = 0.001).
- [x] Negative tests: post-hoc FS malleation, reused randomness, nonce
      collisions, tampered rounds/carries/anchors, retry-count
      witness-independence.
- [ ] **External**: production-parameter review (smoothing bounds,
      hiding distance at scale) before granting `ZeroKnowledge` for
      production-sized statements.

## G5 — QROM

- [x] Worst-case query accounting (`lattice-qrom::ledger`) with
      rejection amplification.
- [x] Workspace domain-separation registry; uniqueness test over all
      production transcript labels.
- [x] `QromAttestation`: digest-bound stage lists, budget + duplicate
      + vacuous-stage checks, verified before replay.
- [x] Composition review with machine checks (ordering, headroom,
      uniqueness, bounded rejection) and four blocking manual items.
- [x] `QromFiatShamir` capability granted **only** through a passing
      review.
- [ ] **External**: the Fiat-Shamir QROM proof for the full composition
      (the review's manual items); challenge-entropy sizing at target
      security levels.

## G6 — hardening

- [x] Verifier no-panic fuzzing: 4 000 random envelope mutations,
      truncation at every prefix, section-count/length bombs, duplicate
      tags, unknown tags, all-zero/all-0xFF buffers, commitment-decode
      fuzz — all via `catch_unwind`.
- [x] Allocation caps: per-section and **total** proof bytes, enforced
      before allocation.
- [x] Checked arithmetic on untrusted lengths/offsets; no `unwrap` on
      untrusted paths (clippy `unwrap_used` denied).
- [x] Property/mechanized checks: field identities, NTT roundtrips at
      random configurations, packing roundtrips, decomposition
      exactness, carrier identity exactness, transcript determinism.
- [ ] **External**: timeout/CPU quota harness for network-facing
      verification deployments; constant-time review for
      secret-dependent branches.

## G7 — performance

- [x] Reproducible benchmark matrix (`cargo run -p lattice-bench
      --release`): field, NTT (generic + fast), ring mul, Ajtai
      commit/open, sumcheck prove/verify, ZK sumcheck
      prove/simulate/verify, Akita PCS, zkVM end-to-end.
- [x] Proof/setup size accounting (envelope bytes, expanded key
      material, ZK proof component sizes).
- [x] NTT performance pass: contiguous per-level twiddle tables with
      exact equivalence tests (recorded effect at kernel scale: ~1% —
      the bottleneck is modular multiplication, honestly documented).
- [ ] **External**: publish the benchmark matrix per release; no
      unexplained stage regressions.

## G8 — external review

- [x] Artifacts prepared: this checklist, `SECURITY.md` (capability
      statement + fixed-vulnerability post-mortems), reproducible
      benchmark matrix, KAT seeds via `SecretSeed::from_kat_label`.
- [ ] **External**: independent cryptographic review of (a) the
      carrier-relation soundness argument, (b) the ZK simulator
      theorem at production parameters, (c) the QROM composition.
- [ ] **External**: implementation audit; release artifacts must match
      the audited commit + profile digests.

## Known limitations (honest disclosure)

1. The ZK layers are proven and KAT-tested at kernel scale only (see
   G4 external item).
2. The compressed-instruction encoding follows a codebase-local
   funct3 mapping (documented in `reference.rs`); RVC-standard
   encoding conformance is a pending P1 item.
3. The zkVM verifier in differential-reference mode re-executes the
   program (kernel-level shortcut); the production path replaces this
   with the Twist sumcheck proof — staged in `prove.rs` comments.
4. `salsa::zk_sumcheck` (statement-derived masks) provides round-value
   randomization, **not** privacy against the verifier — the genuine
   ZK path is `lattice-zk::zk_sumcheck` (documented in both modules).
