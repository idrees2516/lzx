# Ring Lookups — Lookup Arguments over Rings (ePrint 2026/471)

**Paper**: Bootle, Guskind, Patranabis, Sotiraki — *Lookup Arguments
over Rings and Applications to Batch-Verification of RAM Programs*
(March 2026).
**Crate**: `lattice-lookup-ring` (+ the `lattice-zkvm` bridge).
**Status**: implemented in depth — both PIOPs, the attack layer, the
sub-protocol toolkit, the RAM application, and the three follow-ups:
**(a)** the Greyhound-style compile onto the Ajtai/carrier stack,
**(b)** the Fp256 port of the windowed engine's binding pass, **(c)**
the wiring into the zkVM's memory arguments replacing the Twist & Shout
lookup layer.

## What the paper is

Plookup and LogUp restated over the ring `R = Z_q[X]/(X^d+1)` are
unsound: grand products lose unique factorization and log-derivatives
lose multiplicative inverses. The paper demonstrates the attacks
(Section 4) and rebuilds both lookup styles with CRT-safe index tags:
elements of the challenge space `C` (binary-coefficient ring elements,
`|C| = 2^d`) tag each table entry through the injective map
`g: [N] → C`, and every soundness argument runs through the
pairwise-invertibility of tag differences (Lemma 3.8) instead of field
cancellation.

## The implementation map

| Paper artifact | Code | Tests |
|---|---|---|
| The split ring `R ≅ F_{q^{d/2}} × F_{q^{d/2}}` (Lemma 3.8) | `ring_d.rs` — `q ≡ 5 mod 8` (4294967197), schoolbook negacyclic kernel, CRT projections/reconstruction, slot inversion via polynomial EEA, batch inversion, the challenge space `C`, `g_map`, MLE machinery | 10 |
| Section 4 attacks (CRT swapping, zero divisors) | `attacks.rs` — the Z15 Plookup/Lasso attacks at *every* grid point, the Z6 and Z12 LogUp attacks | 5 |
| Sum-check over rings (Appendix A.3) | `ring_sumcheck.rs` — the multilinear virtual-polynomial engine, Vandermonde-interpolated round messages, challenges from `C` | 4 |
| Scalar product (B.2), Hadamard (B.5), cyclic shift (B.9), entry product (B.12), integer check (B.15) | `subprotocols.rs` | 8 |
| Binary check (B.19, the LatticeFold+ range PIOP) | `subprotocols.rs` — see the deviation note below | 3 |
| Ring-Plookup (Construction 5.5, Lemma 5.4) | `ring_plookup.rs` — the merge witness, `a*/b*/w*` combination, three entry products, the `χ_w = χ_a·χ_b` check, η-consistency, shift tests, the binary check | 5 |
| Ring-LogUp (Construction 5.11, Lemma 5.8) | `ring_logup.rs` — CRT-slot batch inversion with Remark-5.10 resampling, the two sum-checks, the two zero-checks, the integer and binary checks | 5 |
| Offline memory checking (Construction 6.3, Lemma 6.4) | `ram.rs` — `memcheck` with the nine record conditions pinned in tests | 2 |
| Memory consistency (Construction 6.5) + almost identical (6.9) + the composition (Theorem 6.13) | `ram.rs` — `prove_ram_batch`/`verify_ram_batch` | 3 |
| Greyhound-style compile (§2.4's framework) | `carrier.rs` + `windowed.rs` + `compile.rs` — follow-up **(a)** | 8 |
| The Fp256 port | `fp256_port.rs` — follow-up **(b)** | 4 |
| The zkVM memory-argument wiring | `lattice-zkvm/{lookup_memory.rs, pipeline3.rs}` — follow-up **(c)** | 9 |

## Follow-up (a): the Greyhound-style compile

The PIOPs' oracle messages become Ajtai commitments; the evaluation
queries ride the windowed engine (`windowed.rs`):

* **Digit windows**: Ajtai binding needs short openings while oracle
  entries have norm up to `q/2` — each entry decomposes into 8
  4-bit windows, each window layer a slot with `‖·‖∞ ≤ 16`.
* **The grid**: the oracle vector `v ∈ R^N` parses as the
  `2^{h_r} × 2^{h_c}` matrix; the MLE evaluation factorizes as
  `v̂(r) = a^T·S·b` with the tensor pair — Greyhound's §2.4 view of
  committed multilinear polynomials. The verifier computes `(a, b)` in
  `O(log N)` ring work.
* **The binding pass**: the Lyubashevsky linear-form argument of the
  workspace's `lattice-commitment::linear_proof`, grid-structured —
  mask, challenge (from `C`), rejection-sampled response, and the
  three checks `‖z‖∞ ≤ B`, `A·z = w + c·t`, `ℓ(z) = img + c·y`.
* **The compiled driver** (`compile.rs`): `prove_logup_committed` /
  `verify_logup_committed` run Ring-LogUp end-to-end with
  commitment-absorbing transcripts — the verifier never touches the
  raw oracles, only commitments, challenge points, and binding-pass
  responses. The CF rows of the binary check are committed
  separately: coefficient extraction does not commute with the ring
  product (the convolution), so the rows cannot derive from a single
  settled `c` evaluation — the paper's own consistency block (steps
  10–12) binds them back.

Honest scope: the response layer transmits `z` in the clear
(`O(N·K_w)` — the paper's PIOP proof-length bound); Greyhound's
`√N`-transmission split-and-check compression is the documented next
optimization. The carrier runs the schoolbook kernel: the NTT
requires `2d | q−1` which forces the full split (`q ≡ 1 mod 8`),
incompatible with Lemma 5.8's two-component ring.

## Follow-up (b): the Fp256 port of the binding pass

`fp256_port.rs` instantiates the same binding pass over the BN254
scalar field on the CIOS Montgomery grid of
`lattice-projsumcheck/src/fp256.rs`:

* digit windows over the **Montgomery limbs** (16 × 16 bits — an
  exact integer identity, no canonical conversion);
* grid tensors with challenges from `sample_upper_limb` (the λ = 125
  upper-limb discipline — the "grid substrate");
* the challenge products ride `mul_upper_limb`, the CIOS short-circuit
  (18 of 36 limb products skipped), with the
  `mul == mul_upper_limb` equivalence pinned in tests;
* the structural equivalence with the ring engine (window counts,
  tensor arities, the three checks) pinned cross-engine.

Over a prime field there are no zero divisors and no norms: the
carrier is a tall random matrix and binding is the linear-algebraic
statement. The port demonstrates the engine's field-genericity and
the 256-bit carrier path for the two-characteristic discipline.

## Follow-up (c): the zkVM wiring

`lattice-zkvm/src/lookup_memory.rs` + `pipeline3.rs`:

* **fetch/input** (ROM): Ring-LogUp of the read values into the public
  table (the paper: "lookup protocols already suffice for ROM");
* **RAM/registers**: the Section-6 composition over the touched
  sub-RAM — the isolation lookups tag the touched addresses' image
  values with `g(addr)`, the memcheck record layer carries the op
  semantics, the almost-identical layer pins the untouched rest;
* u64 values pack into ring elements as `c₀ + c₁·q` (bijective);
* `prove_v3`/`verify_v3` run the demo program end-to-end — the
  verifier never re-executes; tampered final RAM (touched and
  untouched words), tampered registers, and the wrong program are all
  rejected.

The instruction-semantics AIR remains the same documented next layer
as in v2 — this pipeline swaps the lookup layer only.

## Deviations from the paper (honest list)

1. **Construction B.19's steps 7–9** (the monomial-matrix binding via
   `e_j = M̂_{f,j}(r)` checked through `ev(e_j)(β)`): polynomial
   evaluation at `β ∈ C` is **not well-defined on the quotient ring**
   (`X^d ↦ β^d ≠ −1`), so the MLE and the evaluation do not commute
   once the products wrap. The binary check enforces 0/1 coefficients
   directly — one batched `CF̂_j² − CF̂_j` sum-check per row — with the
   paper's CF-consistency block verbatim; the monomial-set check's
   `[0, d)` range is subsumed by `∈ {0, 1}`.
2. **Step 10's `v ∈ Z_q^d` typing**: with ring-valued tensor entries
   the `v_j = ⟨CF_j, ⊗r⟩` are ring elements; the paper's typing holds
   only for binary challenge *points*. The consistency identity
   `w = Σ_j v_j X^j` is implemented with ring `v_j`.
3. **Step 13 as printed** (`ct(x−1·e_j) = v_j`) does not typecheck
   under either reading of `x−1` once the tensor is ring-valued; the
   sound closing identity it gestures at — `X^f = (1−f) + f·X` for
   binary `f` — is what the `CF² = CF` check enforces.
4. **Construction 6.5's Hadamard** segment order (the printed
   `VR − V ‖ o₃ ‖ V′` gives `a∘b = −V ≠ V = c` on the initial
   segment): implemented as
   `(1^M‖o₁‖0^M) ∘ (V−V_R ‖ o₃−V_R ‖ V′−V_R) = V_W − V_R`, the
   identity the proof intends.
5. **Construction 6.9's `v̂/û̃` dance** is tightened to its sound
   core: the `(1−û)(V̂−V̂′) = 0` sum-check with `u`'s binarity and the
   touched-set membership lookup.
6. **Appendix D's `(1+β)^M` factor** in the χ check is a typo carried
   from the original Plookup normalization — the main construction
   text's `χ_w* = χ_a*·χ_b*` (with `(1+α)` absorbed in `a*`) is
   implemented.
7. **The compiled layer's response transmission** is linear, not the
   Greyhound `√N` compression (see (a) above).

## The attacks layer (why the tags matter)

`attacks.rs` reproduces Section 4 as executable demonstrations:

* **CRT swapping (Z15)**: the invalid lookup `a' = {7, 11}` (CRT
  `(1,2), (2,1)`) satisfies the original Plookup `F ≡ G` relation at
  *every* `(β, γ)` of the 225-point grid — a per-component polynomial
  identity — and the Spartan/Lasso grand product `WSt·WSh = RS·S`
  likewise; over the field `Z17` the same forgery survives only at
  ~48/289 accidental roots.
* **Zero divisors (Z6)**: the cleared-denominator sum
  `Σ_v (m_a[v]−m_b[v])·∏_{z≠v}(x−z)` is identically zero for an
  *invalid* lookup — every product contains a zero-divisor pair.
* **The relation itself (Z12)**: `P(x) ≡ 0` for any invalid lookup —
  `∏_j(x−b_j)` always contains a zero-divisor triple.

The ring-safe protocols reject the analogous forgeries in their own
test suites.

## Parameters (demonstration scale)

`q = 4294967197` (2³²−99, prime, ≡ 5 mod 8), `d ∈ {4, 8, 16, 32, 64}`,
`r = 983270775` (√−1), carrier `k = 2` rows, 8 × 4-bit windows,
norm bounds at test scale — production instantiation runs the
`lattice-sis-estimator` discipline from `SECURITY.md`.
