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

## The estimator-run MSIS table (2026-09-30, Stage 5.1) — the honest verdict

`lattice-sis-estimator/examples/fold_security_table.rs` runs the ADPS16
estimator on the compact fold's actual instances
(`q = 3·2^30+1`, ring `N = 64`, `m = (n̄ + r)·64`, the relaxed `2×`
bound; scalar convention per `scalar_sis_from_ring`). The
pre-estimator assertion in note 5 above ("the lattice covering radius
at the chosen `(k, n̄)` far above the gate") **does not survive the
run**:

| shape | bound | classical bits | verdict |
|---|---|---|---|
| k=2, A=2^12, n̄ ∈ {128..1024} | gate `2·r·A·255` | **11.7** | broken |
| k=2, any n̄, any bound | any | 11.7 | the rank-2 module is broken at every response length |
| k=4, n̄=4, r=4, A=2^6 | gate | 60.2 | insufficient |
| k=4, n̄=2, r=4, A=2^8 | gate | **329.6** | sound — the second-fold regime |
| k=4, n̄=2, r=4, A=2^6 | gate | 1040.6 | sound |
| knob search: n̄ ≥ 8 at 128 bits | gate | — | needs k ≥ 16 (commitments blow the 50 KB budget) |

**The finding**: the single-level fold at the benchmark response
lengths (`n̄` in the hundreds) buys the 27 KB size, **not** the MSIS
binding — `m/n` in the tens puts every instance in the combinatorial
regime regardless of the gate. The knobs (`k↑`, `r↓`, `A↓`, the
statistical gate) do not close the gap at these lengths: reaching 128
bits at `n̄ ≥ 8` requires `k ≥ 16`, whose commitments
(`r·k·64·4` bytes) alone exceed the 50 KB budget.

**The sound posture** (the roadmap's own Stage 5.2, now estimator-
mandated): the **second-level fold** (`lattice-zkvm/src/second_fold.rs`,
landed) — the LaBRADOR-decider amortization: the level-1 responses of
`r` bundles sharing the column key fold into ONE short response bound
by a fresh Ajtai key at the estimator's sound row (`κ = 4, n̄ ∈ {2, 4},
r = 4, A = 2^8` → 329+ classical bits, re-verified FAIL-CLOSED at
construction by the estimator-gated profile). Every check is exact and
linear — the public-target fold `F̄·z = Σ γ_i·t_i`, the short-key
binding `A₂·z = Σ γ_i·T_i`, the commuting functional `Φ(z) = Σ γ_i·u_i`
— with no garbage terms.

**The LaBRADOR tail — LANDED (2026-10-04, the width fold)**: the
width-reducing fold (`lattice-zkvm/src/width_fold.rs`) implements the
quadratic-garbage construction the honest finding mandates: the wide
level-1 response splits into `r₂` parts whose link images `p_i` and
cross terms `G_ij = F̄_{(i)}·s_j` are committed BEFORE the challenges
(the symmetrized quadratic form `Σ_{i,j} γ_i γ_j G_ij` with
`G_ii = p_i`), the folded `z ∈ R^w` binds through the short instance
`[A₂ | −T]` at `(w, r₂, κ, A₂) = (8, 2, 8, 2^2–2^4)` (206–1,855
classical bits over the real level-1 gates — the `width_fold_table`
example's verdict), and the functional layer rides the same pre-
challenge discipline (the `g_ij = ψ^{(i)}(s_j)` superposition fix —
the per-slice `(W3)` identities). The **Sound compact profile**
(`compact.rs::CompactProfile::Sound` + `memproof.rs::
prove_memory_argument_sound`) consumes it live: the level-1 response is
never transmitted, so the level-1 `[F̄ | −y]` instance never arises —
the binding of the whole opening is `[A₂ | −T]`, fail-closed by the
estimator-gated profile at prove AND verify (the posture marker
re-derivation). The single-stage coverage ends at `n̄ ≤ 16`
(β₁ ≤ 2^20 at `A₁ = 2^4`); beyond it the profile refuses rather than
shipping a broken binding (the recursive staging and the Modulus-50
class are the documented follow-ups). The measured honest price at the
test scale: 129.7 KB vs the compact mode's ~60 KB (BENCHMARKS §2j) —
the quadratic garbage's cost, the estimator-mandated trade.

The table itself regenerates with:
`cargo run --release -p lattice-sis-estimator --example fold_security_table`
(and the width fold's own regime:
`cargo run --release -p lattice-sis-estimator --example width_fold_table`).
