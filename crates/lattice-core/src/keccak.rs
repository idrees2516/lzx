//! Keccak-f\[1600\] permutation and the SHA3 / SHAKE sponge built on it.
//!
//! Zero-dependency implementation so the whole workspace has no external
//! crates on the prover/verifier path. Verified against FIPS 202 KATs in the
//! test module. The transcript uses SHAKE-256 for arbitrary-length challenge
//! absorption and squeeze.

/// Keccak-f\[1600\] round constants (24 rounds).
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets for the rho step, indexed by lane (x, y) -> y*5 + x.
const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56,
    14,
];

/// Keccak-f\[1600\] permutation, in place over 25 lanes.
pub fn keccak_f1600(state: &mut [u64; 25]) {
    for &rc in RC.iter() {
        // theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= d;
            }
        }
        // rho + pi
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                let idx = x + 5 * y;
                let new_idx = y + 5 * ((2 * x + 3 * y) % 5);
                b[new_idx] = state[idx].rotate_left(RHO[idx]);
            }
        }
        // chi
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
            }
        }
        // iota
        state[0] ^= rc;
    }
}

/// Byte-order mapping: Keccak state is column-major with lanes in LE.
fn bytes_to_state(bytes: &[u8], state: &mut [u64; 25], offset: usize) {
    for i in 0..25 {
        let base = offset + i * 8;
        let mut lane = [0u8; 8];
        lane.copy_from_slice(&bytes[base..base + 8]);
        state[i] = u64::from_le_bytes(lane);
    }
}

/// Generic Keccak sponge (original padding, not SHA3's 0x06 domain byte —
/// the domain byte is supplied by the caller for flexibility).
pub struct KeccakSponge {
    state: [u64; 25],
    rate: usize,
    /// Position within the current rate block.
    pos: usize,
    suffix: u8,
}

impl KeccakSponge {
    /// SHA3 (FIPS 202): suffix 0x06, rate 136 for 256-bit output.
    pub fn new_sha3_256() -> Self {
        Self::new(136, 0x06)
    }

    /// SHAKE256: suffix 0x1F, rate 136, arbitrary output length.
    pub fn new_shake256() -> Self {
        Self::new(136, 0x1f)
    }

    fn new(rate: usize, suffix: u8) -> Self {
        KeccakSponge {
            state: [0u64; 25],
            rate,
            pos: 0,
            suffix,
        }
    }

    /// Snapshot of the internal state (for resumable squeeze in transcripts).
    pub fn raw_state(&self) -> [u64; 25] {
        self.state
    }

    /// Absorb bytes.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.state[self.pos / 8] ^= (byte as u64) << (8 * (self.pos % 8));
            self.pos += 1;
            if self.pos == self.rate {
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
        }
    }

    /// Finish absorbing with pad10*1 + suffix, then squeeze `out_len` bytes.
    pub fn finalize(mut self, out_len: usize) -> Vec<u8> {
        // XOR the domain suffix and the final 0x80 bit into the block.
        let pos = self.pos;
        self.state[pos / 8] ^= (self.suffix as u64) << (8 * (pos % 8));
        let last = self.rate - 1;
        self.state[last / 8] ^= 0x80u64 << (8 * (last % 8));
        keccak_f1600(&mut self.state);

        let mut out = Vec::with_capacity(out_len);
        while out.len() < out_len {
            let mut block = [0u8; 200];
            for i in 0..25 {
                block[i * 8..i * 8 + 8].copy_from_slice(&self.state[i].to_le_bytes());
            }
            let take = self.rate.min(out_len - out.len());
            out.extend_from_slice(&block[..take]);
            if out.len() < out_len {
                keccak_f1600(&mut self.state);
            }
        }
        out
    }
}

/// SHA3-256 one-shot.
pub fn sha3_256(input: &[u8]) -> [u8; 32] {
    let mut s = KeccakSponge::new_sha3_256();
    s.update(input);
    let out = s.finalize(32);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&out);
    digest
}

/// SHAKE-256 one-shot XOF.
pub fn shake256(input: &[u8], out_len: usize) -> Vec<u8> {
    let mut s = KeccakSponge::new_shake256();
    s.update(input);
    s.finalize(out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha3_256_kat() {
        // FIPS 202 KAT: SHA3-256("") and SHA3-256("abc").
        let empty = sha3_256(b"");
        assert_eq!(
            hex(&empty),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
        let abc = sha3_256(b"abc");
        assert_eq!(
            hex(&abc),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
        // Longer input crossing the rate boundary (136 bytes).
        let long = sha3_256(&[0u8; 200]);
        assert_eq!(hex(&long), "2b43036c229ba512995f91fdb46fcd5327a4dc834d86d6e0f58a08053346dc2e");
    }

    #[test]
    fn shake256_kat() {
        // FIPS 202 KAT: SHAKE256("", 32).
        let out = shake256(b"", 32);
        assert_eq!(
            hex(&out),
            "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
        );
        // Determinism and length extension consistency.
        let a = shake256(b"hello", 64);
        let b = shake256(b"hello", 32);
        assert_eq!(&a[..32], &b[..]);
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
