# labinius (upstream PCS port) — wire/ IMPLEMENTED

Crate: `lattice-labinius` (~90% of upstream pcs by protocol surface;
AVX-512 backend bit-exact, reference round 17.5x — see PERFORMANCE.md).

## Wave 7 additions

| Part | Realization | Tests |
|---|---|---|
| 7.16 bit-packing | `wire::BitWriter/BitReader` — LSB-first, exact widths, `encoded_len` | `bit_writer_reader_roundtrip`, `bit_reader_strict_on_truncation`, `bit_packing_beats_padded_lanes` |
| 7.16 rANS entropy coding | `wire::RansCoder` — 64-bit state, static transmitted histogram, per-symbol ryg renorm threshold `x_max = ((L>>M)<<8)·f`, byte-wise LIFO renorm stack, exact roundtrip | `rans_roundtrip_exact`, `rans_uniform_roundtrip`, `rans_bad_histogram_rejected`, `rans_symbol_out_of_range_rejected` |
| 7.16 framed artifact | `wire::WireArtifact` — magic/version/params-digest header, blob table, strict decode | `wire_artifact_roundtrip`, `wire_artifact_strict_decode` |

## Partial / open

* **7.5 `Opening::Recursive`** remains declared-but-unwired (no residues
  field, no T_u left-expansion commitment, no prove_opening call site,
  no test) — the compact-opening asymptotics are the next protocol
  completion here.
* gen_*/bin_large kernel families (2.6k + 910 upstream LOC; Wave 8.1/8.4),
  chunked-chain recursion encoding (8.9), cross-field switch machinery,
  LANES=64 interleaved rANS streams (constant-factor throughput).


## Wave 7.5: the Recursive opening mode, wired

`scheme.rs` now exposes `Prover::prove_recursive` and
`Verifier::verify_opening_recursive`; `tests/recursive.rs` runs the
end-to-end round (two suites): commit → point → row → challenges → the
LaBRADOR proof of the folded-opening relation → the claim identity +
statement-rebuild verification. The prover encodes real A-row
constraints over the per-limb transforms (LaBRADOR rejects a wrong
witness at prove time — the round-trip test pins this), the b-vectors
are transmitted and digest-pinned (`statement_digest_with_b`), and the
verifier rebuilds the statement (specs from the announced caps, phis
from the public key, b from the transmitted values) before running the
LaBRADOR verifier. Two honest gaps are documented in the module header:
the b-vectors' alignment with the public folded commitment (the
`components_of` recomposition is not an inverse on the quadratic limbs;
upstream's chunked S-chains unported), and the port's LaBRADOR
`verify` being structural (the amortized relation check is prove-side
only). Both are on the Wave-8 ledger.
