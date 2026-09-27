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
