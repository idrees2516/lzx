# HyperWolf (ePrint 2025/1903) — PARTIAL

Crate: `lattice-pcs` (291 lines). Implemented: the `PcsBackend` trait +
a transparent (commit-and-reveal) backend with digit norm proofs; the
trait signature admits a single claim.

Open (item 7.12, H1–H5): the ring mapping MR + ι-slice gadget
decomposition + leveled commitment F_{k−1,0}; the guarded IPA (the
split-and-fold norm constraint with the ‖s^(1)‖∞ ≤ γ guard — the
standard-soundness core); k-round evaluation folding (tensor-vector
products, ct-checks); the fixed-weight signed challenge space C with
rejection to T ≤ 10; LaBRADOR compaction (blocked on the u128/RNS ring
layer, Wave 8.3); three-mode batching (trait extension). The paper's
parameter regime (q ≈ 2^128, q ≡ 5 mod 8) is unreachable until the
128-bit ring layer exists.
