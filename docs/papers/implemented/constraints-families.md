# Twist-and-Shout constraint families (the instruction-semantics layer)

Status: **the FULL RV64IM constraint set is implemented** (the
family-completion session, 2026-10-03): `lattice-zkvm/src/constraints.rs`
(~5.4k lines) + the integration suite
`crates/lattice-zkvm/tests/constraints.rs` (14 tests) +
`crates/lattice-zkvm/tests/semantics.rs` (9 end-to-end tests through the
bundle-commitment layer) + the pipeline integration
(`lattice-zkvm/src/semantics.rs` + the pipeline2 Stage 4.5 + 4
full-pipeline tests). The shifts, MUL, and DIV/REM families landed with
range-linked limbs (every limb/carry column decomposed into boolean bits —
the integer-semantics anchor), the DECODE leg (the instruction tensor bound
to the fetched word + the coverage partition identities — the verifier-side
coverage gate the pipeline needs), and the RANGE-LINKS leg.

## What landed

Ten legs in fixed protocol order (`prove_constraints` / `verify_constraints`):

| Leg | Family | Content |
|---|---|---|
| `bool-tensors` | booleanity | per-row booleanity of the six value bit-tensors (64 rows each): `eq(r)·B_row·(1−B_row) = 0`, one α-batched leg over logT — never the 2^(64+logT) tensor space |
| `bool-instr` | booleanity | the same per-row construction over the 32 instruction-bit rows |
| `bool-cols` | booleanity | every auxiliary bit column (selectors, flags, carries, full-width eq-prefixes, comparisons) |
| `sel` | decode | every selector column equals the product of polarized instruction-bit indicators from the public mask (opcode 7 bits + funct3 3 + funct7 7 when masked); the memoized factor pool shares bit rows across selectors |
| `flags` | activity | `mem_re/mem_we` vs the load/store selectors; `taken` vs the branch selectors composed with the comparison columns (`eq` full-equality, `lt`, `ltu`); `rd_we = W·(1−Π(1−rd_i))` over instr bits 7..11 |
| `arith` | ALU | ADD/ADDI/SUB (4-limb carry/borrow recurrences, selector-masked), ADDW/ADDIW/SUBW (2 limbs + the bit-31 sign extension), LUI (rd = imm), AUIPC (rd = pc+imm via the ctrl chain), JAL/JALR (rd = pc+4 via the pc4 chain) |
| `cmp` | comparisons | full-width eq-prefix recurrence (65 columns per comparison: `eqp[i+1] = eqp[i]·(1−a−b+2ab)`), the unsigned `ltu = Σ eqp·(1−a)b` and the signed head-term form |
| `ctrl` | control | `pc = 4·fetch_word`; the next-pc MUX `np = A + b·t·(T−A) + j·(T−A) + r·(J−A)` with A = pc+4 (carry_pc4), T = pc+imm (carry_ctrl), J = rs1+imm (carry_jalr); post-halt inactivity `h·(rd_we+mem_we+mem_re) = 0` |
| `route` | routing | effective address `mem_addr = rs1+imm` (limb-wise over the jalr chain, committed addr-limb columns); word addressing `addr = 8·word + 4·half` (word accesses) / `8·word` (double); SD `mem_new = rs2`; LD `rd = mem_old`; SW/LW half-MUX routing with the LW sign extension via committed bit rows; the bitwise AND/OR/XOR per-bit identities (register + immediate forms); the comparison rd routing |
| `decode` | decode | the instruction tensor bound to the fetched word (`Σ 2^i·bit_i = iw`); the class partition (`Σ 12 class selectors = 1`); the sub-class partitions per class; the SYSTEM discipline (f3 = 0, funct12 ∈ {0,1}) — **the verifier-side coverage gate** |
| `shift` | shifts | the shamt one-hots (register 6-bit / W 5-bit from rs2's low bits, immediate from the instruction's shamt field — each = the product of polarized source-bit rows); the per-bit shift MUXes (left: `rd_i = Σ_{s≤i} oh_s·rs1_{i−s}`; right-logical: the in-range sum; right-arithmetic: the in-range sum + the sign fill via the one-hot complement); the W sign extension (`rd_i = rd_31` for i ≥ 32) |
| `mul` | multiply | the limb recurrence `S_k + c_k − p_k − 2^16·c_{k+1} = 0` over the range-linked limbs/carries (k = 0..7 for MULH/MULHU with the 128-bit closure; k = 0..3 for MUL with the free c_4 wrap; k = 0..1 for MULW); the MULHU rd routing (rd = hi); the MULH rd composition (`rd ≡ hi_u − a₆₃·b − b₆₃·a mod 2^64` via the borrow chain); the W sign extension |
| `div` | divide | the eq-to-zero prefixes (bz for the 64- and 32-bit divisor checks); the magnitude definitions (the class-dependent sign sources: unsigned raw, signed \|x\| via the negation identity); the Euclidean recurrence `\|a\| = \|q\|·\|b\| + \|r\|` with the closure carry; the remainder bound (`\|r\| < \|b\|` via the subtraction borrow chain with the range-linked out limbs); the rd routing (the sign compositions `sq = sa ⊕ sb` expanded, the W extensions); the divide-by-zero specials (rd = −1 / rd = rs1) |
| `rangelinks` | range | every mul/div limb and carry column composed from boolean bit columns (`limb_l = Σ_j 2^j·bit_{l,j}`) — **the integer-semantics anchor**: without the range, the limb recurrences only constrain field combinations |
| `halt-end` | termination | `Σ e_last·halted = halted[T−1] = 1` — the indicator-MLE point check forcing termination |

## The coverage set (fail-closed)

Covered: **the full RV64IM subset** — ADD/ADDI/ADDW/ADDIW/SUB/SUBW,
AND/OR/XOR (+ immediates), SLT/SLTU/SLTI/SLTIU, LUI, AUIPC, JAL/JALR, all
six branches, LD/LW/LWU, SD/SW, ECALL/EBREAK, **all twelve shift classes**
(SLL/SRL/SRA + W + I forms with the shamt one-hots — register 6-bit, W
5-bit, immediate from the instruction bits), **the MUL family** (MUL,
MULH, MULHU, MULW — the limb recurrence with the MULH sign-corrected rd
composition), and **the DIV/REM family** (DIV/DIVU/REM/REMU + the W forms
— the Euclidean quotient/remainder discipline over range-linked
magnitudes with the sign compositions and the divide-by-zero specials).
`prove_constraints`/`verify_constraints` reject anything else (the
atomics, the CSR space) with `UncoveredInstruction`.

## Structural fixes forced by execution (the staged substrate had never run)

1. **`bit_tensor` was a structurally inconsistent MLE**: `num_vars =
   nbits + log_t` (67) with only `nbits·T` evaluations. Fixed to
   `num_vars = log2(nbits) + log_t` — the row-major bit×cycle grid is a
   consistent MLE whose row block is `log2(nbits)` MSB-first variables.
   The ledger's `bit_row_claim`/`TensorRow` views follow the same
   convention (`idx_point(log2(nbits), row)`).
2. **`Ledger::limb/value_combo/instr_word` mapped bit rows LSB-first**
   against the MSB-first packing (value-bit `b` lives at row `63−b`);
   fixed via the `value_bit_row` mapping.
3. **The booleanity legs were written against a "(6+logT)" universe** and
   materialized eq tables over 2^67 points; replaced by the per-row
   construction above.
4. **`stage()`'s view/factor-claim pairing was misaligned** (views zipped
   against the raw factor-claim list including synthetic `(1−b)` factors);
   replaced with explicit `(factor index, view)` pairs.

## Honest deviation ledger

* Field negatives: every negative constant is `fe(k).neg()` — the
  `fe(0u64.wrapping_sub(k))` idiom is WRONG in Goldilocks (`from_u64`
  reduces mod p and `2^64−k mod p ≠ p−k`); the initial implementation
  used it and the affected families passed only vacuously (their
  selector-masked terms were zero on the test programs). Found by the
  per-i VP-contribution probe; fixed globally.
* `mem_half` is the half-SELECTOR (which 32-bit half a word access
  touches), not an access indicator — no flags-level identity; it is
  constrained by the routing family's word-addressing identity.
* The halted family proves booleanity + termination + post-halt
  inactivity (rd_we/mem_we/mem_re frozen), and the executor's post-halt
  pc evolution is unconstrained (fetch consistency is authenticated by
  the memory layer's Shout). The monotone propagation
  `halted(c) ≤ halted(c+1)` is enforced at witness build; its
  shifted-row leg is a documented follow-up (the covered statement stays
  sound: raising `halted` early only freezes the machine sooner, and
  every covered class is fully constrained, so the ECALL padding cycles
  are forced no-ops).
* `build_aux`'s contract is the per-cycle EXECUTED instructions (the
  trace's instr column) — the kernel-scale tests initially passed the
  program listing, which only coincides for straight-line code (found by
  the jal test).
* Comparisons use full-width eq-prefixes (65 columns per comparison) —
  the staged 6-prefix design could not express 64-bit comparisons.
* The prover/verifier legs at kernel scale run against a table-backed
  ledger (the PCS-authenticated openings stand in via the witness table);
  the production wiring goes through the bundle-opening layer as in
  `memproof.rs`.

## What this buys

The zkVM prove path now covers decode, ALU, comparisons, control flow,
memory routing, and termination for the covered subset with the verifier
recomputing only public material — the P0-5 "delete verifier re-execution"
posture extended from the memory argument (W7.4) to instruction
semantics. Every leg lands with negative tests in the same commit.
