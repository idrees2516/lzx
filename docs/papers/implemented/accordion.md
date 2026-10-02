# Accordion — ePrint 2025/1325 (lattice instantiation)

**Paper**: Eagen, Gabizon, *"Revisiting the IPA-sumcheck connection"*.
**Status**: implemented in depth — `crates/lattice-accordion`
(~2.6k lines, 23 tests) over the Ajtai module
`(R_q)^rows` with `q = 2^50 − 2687` (the workspace's 50-bit modulus
class, `lattice_ring::modulus50`).

## What was implemented

| Paper component | Status | Where |
|---|---|---|
| The module-valued sumcheck (Lemma 3.1) | ✅ verbatim | `sumcheck.rs` — degree-2 round polys as three module points, the round-recurrence verifier, the table-restriction engine |
| `gen` / `com` (§5) | ✅ | `module.rs::Srs` — the seeded `[G \| P]` matrix (columns as ring elements of `R_q`, MSIS-structured) |
| `reduce` (§5) | ✅ verbatim | `pcs.rs::reduce` — `α ←`, `P' = αP`, `A(X) = Ŵ(X)Ĝ(X) + T(X)Ŵ(X)P'`, target `cm + vP'`, the deferred terminal `(V − baP')/a` |
| `accumulate` (§6) | ✅ verbatim | `pcs.rs::accumulate` — `γ ←`, `C = ΣγⁱCᵢ`, `e(X) = Σγⁿ eq(X, rᵢ)`, the output `(r, V/e(r))` |
| `decide` (§7) | ✅ (the lattice route) | `pcs.rs::decide` / `decide_batched` — the amortized direct public evaluation `Ĝ(r)` |
| Knowledge soundness (Lemma 5.2) | ✅ executable | `extract.rs` — the two-`α` grid-tree extractor over a rewindable prover trait, with the shortness verdicts and the `[G\|P]` kernel outcomes |
| The digit-layer regime | ✅ (the lattice addition) | `module.rs::LayeredCube` — 16-bit layers, `T(X) = eq(X_D,u)·E(X_L)`, `E(ℓ) = Σ 2^{16j} eq(ℓ, e_j)` |

Benchmarks (`examples/accordion_bench.rs`, release): reduce at
`N = 4096` in **7.8 ms** with an **18.4 KB** proof (exactly
`3·m` module points + one scalar, `m = k + κ`); verify **0.14 ms**
(`O(m)`); the amortized 4-fold decide **12.9 ms** (`O(N)` once per
batch).

## The lattice re-reading of the paper

The paper's group `G` becomes the module `M = (R_q)^{rows}` — an
`F_q`-vector space, so the paper's requirement `|G| = |F| = p` forces
**one prime for both**: the challenge field and the commitment ring
live over `q = 2^50 − 2687`. The Pedersen `com(f) = Σ fᵢGᵢ` becomes
the Ajtai `cm = Σ_b w_b G_b` with the **digit-layer discipline**: an
arbitrary value vector `f` commits through `f = Σ_j 2^{16j} f^{(j)}`
on the *layered cube* `B^{k+κ}` (data variables × digit-layer
variables), so the committed vector is short (`‖w‖∞ ≤ 2^16 − 1`) and
the MSIS binding applies. Every protocol then runs verbatim on the
layered cube with the combined factor `T(X) = eq(X_D, u)·E(X_L)` —
the paper's `eq(X, z)` regeneralized. The paper's "3k G-elements and
one F-element" communication is preserved exactly: three module points
per round plus the terminal scalar `a`.

## The honest-deviation ledger

1. **The decider (contribution #2) does not port.** The paper's
   group-BaseFold decide rides FRI machinery — Merkle-committee query
   access to folded RS layers. Its naive lattice port fails on a hard
   obstruction: Ajtai commitments bind only for **short** openings, and
   FRI-folded layers have arbitrary mod-`q` entries, so layer
   commitments give no binding at all (an MSIS "kernel" with arbitrary
   coefficients is trivial to find). This crate decides in the
   Halo-amortized style instead: `accumulate` merges `t` claims and
   `decide` runs once per batch — a direct public evaluation `Ĝ(r)`
   costing `O(N)` *ring-scalar* operations. The quantitative note: a
   ring-scalar multiply is ~3 orders of magnitude cheaper than a group
   scalar multiplication, so the `O(N)` decide that motivates BaseFold
   in the group setting is ~`N/1000` MSM-equivalents here; the
   amortization (the part Halo contributes) is preserved.
2. **Knowledge soundness changes shape (DLA → MSIS).** Lemma 5.2's
   collision outcome is a direct contradiction under DLA ("no nonzero
   `f` with `⟨f,G⟩ = 0`"). Over Ajtai modules the mod-`q` kernel is
   huge; MSIS only rules out **short** kernels. The executable
   extractor (`extract.rs`) makes the outcome algebra precise:
   consistent provers yield the recovered opening with the digit-norm
   verdict (short exactly for honest/consistent behavior — the honest
   prover's witness IS its digit decomposition); inconsistent provers
   yield the two-`α` kernel relation on `[G|P]` — which is a genuine
   MSIS solution when its coefficients stay short (the double-opening
   test demonstrates exactly this on a crafted duplicated-column SRS),
   and a field-valid-but-not-short relation otherwise. The
   adversary-model closure (forcing shortness through response-norm
   checks) is the discipline of `lattice-commitment::linear_proof`,
   `lattice-labrador`, and `lattice-cauchyfold` — all in this
   workspace; the plain module-sumcheck port does not by itself
   constrain the prover to the short regime.
3. **The extraction harness shape.** The paper's arity-4 tree branches
   on field challenges with the K-valued interpolation ascending the
   tree; the executable extractor replaces the K-valued ascent with the
   two-`α` grid recovery (the multilinear through the `2^m` grid
   points, interpolated per variable) — the same information content at
   the tested scale, with the within-run recurrences covered by the
   verifier's checks and the cross-`α` consistency checked at every
   split.
4. **Hiding.** The Ajtai `com` is the clear (binding-only) baseline —
   the same posture as `lattice-commitment`'s Ajtai crate; the paper's
   Pedersen is hiding. The ZK layer (smoothing/blinding) is separately
   reviewed capability in this workspace (`lattice-zk`, `lattice-qrom`)
   and out of scope for the PCS wave.
5. **Field size.** `q ≈ 2^49.3` gives per-round soundness
   `2m/q ≈ 2^{−45}` at the tested cube sizes — Goldilocks-class margins
   matching the workspace convention; the paper's asymptotic
   super-polynomial field requirement is met in kind.
6. **The terminal edge cases** (`a = 0` in reduce, `e(r) = 0` in
   accumulate) surface as explicit errors with the completeness bounds
   (`≤ 2m/q`, `≤ tm/q`) documented at the error sites — the paper's
   implicit nonzero assumption.

## The follow-up wiring (this wave)

The Accordion engine is the short-opening PC that the
holography-pcd deviation ledger called for: `lattice-holo/pc_short.rs`
twins the module-sumcheck on the BN254 scalar field (`Fp256`) with the
same digit-layer regime, replacing the linear (long) openings at the
`open`/`verify` call sites — see the holography-pcd note's ledger
update.
