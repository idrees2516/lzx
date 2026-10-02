//! # lattice-accordion — Accordion (ePrint 2025/1325) over Ajtai lattices
//!
//! **Paper**: Eagen, Gabizon, *"Revisiting the IPA-sumcheck connection"*,
//! ePrint 2025/1325 ("Accordion: Efficient Batch Proving via Algebraic
//! Folding" in the user's dossier).
//!
//! ## What the paper does
//!
//! The paper revisits the Bootle–Chiesa–Sotiraki viewpoint that an inner
//! product argument is a **sumcheck protocol whose summed polynomial has
//! coefficients in a group** `G` rather than a field, and builds from it a
//! *multilinear polynomial commitment scheme with accumulation*
//! (ml-PCS, Definition 4.3):
//!
//! * `reduce` — turns an evaluation claim `(cm, u, v)` into a small instance
//!   `φ = (r, C)` of the public language `L_G = {(r, C) : Ĝ(r) = C}`, where
//!   `G = (G₀,…,G_{n−1})` is the preprocessed generator table and `Ĝ` its
//!   multilinear extension. Communication: `3k` group elements + one field
//!   element (the degree-2 sumcheck round polys over `A(X) = f̂(X)Ĝ(X) +
//!   eq(X,z) f̂(X) P'` with target `cm + vP'`).
//! * `accumulate` — merges `t` instances `φᵢ = (rᵢ, Cᵢ)` into one via a
//!   `γ`-combination and a second group-valued sumcheck over
//!   `A(X) = Ĝ(X)·e(X)` with `e(X) = Σᵢ γⁱ eq(X, rᵢ)`.
//! * `decide` — settles `φ ∈ L_G`. The paper's decider is a *group version
//!   of BaseFold* (FRI-style folding of an RS-encoded `G`, verified by
//!   Merkle queries) replacing the `O(n)` MSM.
//!
//! ## What this crate does — the lattice instantiation
//!
//! The user's directive: implement the required parts of the paper **over
//! lattices, not discrete-log groups**. The group `G` becomes the module
//! `M = (R_q)^{rows}` with `R_q` the Goldilocks negacyclic ring; the
//! preprocessed generator table is the column set of a seeded Ajtai matrix
//! `[G | P]`, and the Pedersen commitment `Σ fᵢ Gᵢ` becomes the (Module-SIS
//! bound) Ajtai commitment `cm = Σ_b w_b·G_b` — with the crucial lattice
//! discipline that **the committed vector `w` must be short**. Arbitrary
//! field-valued polynomials `f` are therefore committed through the
//! **digit-layer regime**: `f = Σ_j 2^{16j} f^{(j)}` with `f^{(j)}` binary
//! windows, and the "cube" becomes the *layered cube* `B^{k+κ}`
//! (data variables × digit-layer variables). Every protocol of the paper
//! then runs verbatim on the layered cube with the combined equality factor
//! `T(X) = eq(X_D, u)·E(X_L)`, `E(ℓ) = Σ_j 2^{16j} eq(ℓ, e_j)`.
//!
//! What ports, what does not (full ledger in
//! `docs/papers/implemented/accordion.md`):
//!
//! * **The module-valued sumcheck (Lemma 3.1) ports verbatim**: `M` is an
//!   `F_q`-vector space, so Schwartz–Zippel and the round-recurrence
//!   soundness argument apply unchanged; round messages are three module
//!   points (`3k` module elements + one field element, exactly the paper's
//!   communication).
//! * **reduce / accumulate port verbatim** on the layered cube — including
//!   the deferred-division terminals `(V − baP')/a` and `V/e(r)`.
//! * **decide**: the paper's contribution #2 (group-BaseFold) is inherently
//!   *hash-based* (Merkle-committee query access to folded RS layers). Its
//!   naive lattice port fails on a hard obstruction: Ajtai commitments are
//!   binding only for **short** openings, and FRI-folded layers have
//!   arbitrary mod-`q` entries, so layer commitments give no binding. This
//!   crate therefore decides in the *Halo amortized* style: `accumulate`
//!   merges `t` claims and `decide` runs **once** per batch — a direct
//!   public evaluation `Ĝ(r)` costing `O(N)` *ring-scalar* operations
//!   (the lattice analogue of the MSM, ~3 orders of magnitude cheaper per
//!   operation than group scalar multiplication). The obstruction and the
//!   quantitative comparison are documented in the paper note.
//! * **Knowledge soundness (Lemma 5.2) is executable here**: the 4-ary
//!   transcript-tree extraction is implemented as a harness over a
//!   rewindable prover trait, with the paper's two-`α` subtraction as the
//!   terminal step. Over lattices the outcome algebra changes: the
//!   extracted opening is field-valid by construction, **short exactly for
//!   consistent provers** (honest provers commit digit layers, and the
//!   harness verifies the `2^16` norm budget), and inconsistent provers
//!   yield *mod-`q` kernel relations on `[G|P]`* — which are
//!   Module-SIS solutions only when their coefficients stay short. That
//!   DLA→MSIS gap is the central deviation-ledger entry; the adversary-model
//!   closure (response-norm discipline) lives in this workspace in
//!   `lattice-commitment::linear_proof` (ABDLOP-style), `lattice-labrador`
//!   (amortized short openings) and `lattice-cauchyfold` (the full
//!   compare-before-clearing chain).

// The cube/grid kernels index tables by explicit bit arithmetic (MSB-first
// layered-cube layout); iterator rewrites cannot express the bit indexing —
// the same convention as lattice-labrador's ported kernels.
#![allow(clippy::needless_range_loop)]

pub mod extract;
pub mod module;
pub mod pcs;
pub mod sumcheck;

pub use module::{
    digit_layers, eq_eval_index, Fq, LayeredCube, ModulePoint, Srs,
    ACCORDION_DIGIT_BITS, ACCORDION_LAYERS, Q_50,
};
pub use pcs::{
    accumulate, accumulate_verify, decide, decide_batched, eval_claim, reduce,
    reduce_verify, AccumulateProof, AccordionPcsParams, Instance, PcsError,
    ReduceProof,
};
pub use extract::{extract, ExtractionOutcome, HonestReduceProver, ReduceProver};
