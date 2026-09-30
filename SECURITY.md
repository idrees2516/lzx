# LZX Security Policy

## Scope

The LZX workspace is a lattice-based post-quantum zkVM kernel
implementation (15+ crates, pure `std`, zero external dependencies).
This document states the security claims, the capability discipline,
and the reporting process required by the implementation audit.

## Security capability claims

A proof produced by this codebase carries **exactly** the capabilities
granted by `lattice-zk::privacy_spec::CapabilitySet` — never more:

| Capability | Status | Granting path |
|---|---|---|
| `PostQuantumBinding` | Claimed | Ajtai Module-SIS binding of commitments; classical ROM analysis in-tree |
| `ZeroKnowledge` | Claimed (kernel-scale) | `lattice-zk`: secret-entropy masking + simulator KATs; see caveats |
| `QromFiatShamir` | Review-gated | `lattice-qrom` composition review; requires the manual sign-off items |

**A proof is never "zk" merely because its PCS is lattice-based.**
The Akita evaluation path (`prove_evaluation`) opens the committed
witness in the clear; genuine zero knowledge flows only through the
`lattice-zk` layer (blinded sumcheck + ABDLOP linear proofs).

### Zero-knowledge caveats (kernel scale)

* The ZK constructions are implemented and KAT-tested at kernel-scale
  parameters (vectors ≤ 2^4 entries, m ≤ 200 commitment slots).
  Production parameters (and the associated smoothing/regression
  analysis for the Ajtai hiding argument) require the external review
  of gate G8 before the `ZeroKnowledge` capability is granted for
  production-sized statements.
* The rejection-sampling margin in `ZkLinearProof` must satisfy
  `norm_bound ≫ (n/2)·|s|`; parameter sets violating this degrade HVZK
  statistically (see `zk_linear.rs` tests for the enforced margins).

## Fixed vulnerabilities (post-mortems in git history)

1. **LinearProof Fiat-Shamir malleability** (wave 3): the challenge was
   derived from the statement only — the masking commitment `w` was not
   absorbed first. Post-hoc forgery: choose any short `z'`, set
   `w' := A·z' − c·t`. Fixed by absorbing `(w, images)` before the
   challenge; regression-tested in `lattice-commitment` and
   `lattice-zk`.
2. **Degenerate short-secret sampling** (wave 3): `sample_small_secret`
   requested `8 + 4·i` XOF bytes per ring element, silently
   zero-filling most coefficients — collapsed masking entropy across
   every Lyubashevsky-style proof. Fixed with a full rejection budget
   per element.
3. **Envelope allocation bomb** (wave 3): the section-count ×
   section-size caps allowed ~1 GiB allocation before validation; now a
   total-bytes cap (`MAX_TOTAL_SECTION_BYTES`) is enforced before any
   section allocation.
4. **Four RV64 conformance bugs** (wave 3, found by the differential
   corpus): unaligned `LD`/`SD` word-straddle semantics,
   `SRLI`/`SRAI` shamt ≥ 32 funct6 decoding, `SLLIW`/`SRLIW`/`SRAIW`
   decoding as 64-bit shifts, `LUI`/`AUIPC` missing sign extension.

## Verifier contract

* No panics on untrusted bytes: fuzz-tested entry points
  (`envelope::from_bytes`, `AjtaiCommitment::from_bytes`) with
  structured bombs and 4 000 random mutations.
* All length/offset arithmetic is checked; allocation caps are enforced
  before allocation.
* QROM attestations are verified **before** transcript replay or
  expensive arithmetic.

## Entropy rules

* Private masking randomness comes **only** from
  `lattice_zk::entropy` (OS `/dev/urandom` or explicitly-labeled test
  seeds). Transcript-derived randomness is public by construction and
  structurally cannot seed a `SecretSeed`.
* Seed reuse across proofs is a nonce collision — detected by
  `NonceLedger` at proof time.

## Reporting

Security issues follow the audit's gate G6 process:

1. Reproduce with a minimized test (the workspace's deterministic
   transcripts and KAT seeds make reproduction exact).
2. Classify against the threat model (audit §10.1: malicious prover /
   malicious proof bytes / malicious program / side channel / supply
   chain / quantum adversary).
3. Fix + regression test + capability re-review (the affected
   capability token is suspended until the fix lands).
4. Record the post-mortem in this file.

## Supply chain

* Zero external crates; `cargo metadata` must show no registry
  dependencies. Builds use a pinned stable toolchain; release
  artifacts must record the toolchain digest (gate G8 checklist).


## The compact folded opening (2026-09-30) — security notes

1. **The bits-bundle vacuous-gate fix** (a security defect, not just
   size): the old packing put 31 bits per coefficient against
   q = 3·2^30+1 (q/2 ≈ 2^30.6) — the balanced-representative norm claim
   was ambiguous mod q. The compact mode's 1-byte-per-coefficient
   packing bounds every committed coefficient by 255, restoring the
   norm gate's meaning.
2. **The two-characteristic discipline**: the legs' claims are
   Goldilocks values; the commitments live in R_q. No homomorphism
   connects the fields (verified live during implementation — the F_q
   carrier's evaluation diverges from the lifted Goldilocks claim), so
   the compact design keeps the carrier over Goldilocks and closes the
   fold with SCALAR challenges: the integer fold never wraps mod q (the
   gate bounds r·A·255 < q/2), so the Goldilocks evaluation functional
   commutes exactly through the fold. Cross-field claim lifting is
   forbidden by construction.
3. **Soundness chain**: legs → carrier (Goldilocks sumcheck) →
   w = f(r_sc) → the MLE interpolation over columns pins the ũ's → the
   commuting functional pins ũ to the response v → the Ajtai fold
   F̄·v = Σ d_j·y_j pins v to the commitments → MSIS on
   `[F̄ | −y₁..−y_r]` at the relaxed bound (2× the gate) with the
   mixed-moduli constraint lattice. Extraction needs no ring divisions
   (Q_32 splits completely — division by challenge differences is
   unsound there).
4. **Fail-closed gates**: per-coefficient norm gate (< q/2, checked on
   prove and verify); the r-alignment precondition (r ≤ every factor
   length); the response codec's strict decode; the values-only
   claims queue must drain exactly (reordered or missing claims
   rejected); commitment/width/factor-length shape checks.
5. **Challenge family**: scalar d_j ∈ [−2^12, 2^12] drawn after the
   ũ absorption; the response is a deterministic function of the
   committed columns (no prover adaptation surface). The forgery
   resistance reduces to the CVP hardness of the constraint lattice at
   the honest-gap regime — the parameter knobs (k, r, the gate) are
   documented in DESIGN_50KB.md for estimator-driven tightening.
6. **Tamper coverage** (test-pinned): wrong final registers, wrong
   program digest, wrong claim value, reordered claims, tampered
   carrier terminal, tampered ũ, tampered response bits, tampered
   commitment bytes, wrong seed, tampered widths — all rejected.
