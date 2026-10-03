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

/// Rotation offsets for the rho step, indexed by lane (x, y) -> y*5 + x
/// (used by the cfg(test) reference permutation form).
#[cfg_attr(not(test), allow(dead_code))]
const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Keccak-f\[1600\] permutation, in place over 25 lanes.
///
/// Fully unrolled over the 25 lanes (the tiny-keccak discipline: SSA
/// lane variables, constant rotations, no inner-loop indexing or `%`
/// arithmetic) — the zero-dependency loop form was the dominant cost of
/// every XOF path (the verifier's seeded matrix regeneration measured
/// ~22 us/element, ~4 us per permutation). Bit-identical output to the
/// reference loop form below (test-pinned) and the FIPS 202 KATs.
#[inline(always)]
pub fn keccak_f1600(state: &mut [u64; 25]) {
    let mut s0 = state[0];
    let mut s1 = state[1];
    let mut s2 = state[2];
    let mut s3 = state[3];
    let mut s4 = state[4];
    let mut s5 = state[5];
    let mut s6 = state[6];
    let mut s7 = state[7];
    let mut s8 = state[8];
    let mut s9 = state[9];
    let mut s10 = state[10];
    let mut s11 = state[11];
    let mut s12 = state[12];
    let mut s13 = state[13];
    let mut s14 = state[14];
    let mut s15 = state[15];
    let mut s16 = state[16];
    let mut s17 = state[17];
    let mut s18 = state[18];
    let mut s19 = state[19];
    let mut s20 = state[20];
    let mut s21 = state[21];
    let mut s22 = state[22];
    let mut s23 = state[23];
    let mut s24 = state[24];

    for &rc in RC.iter() {
        // theta
        let c0 = s0 ^ s5 ^ s10 ^ s15 ^ s20;
        let c1 = s1 ^ s6 ^ s11 ^ s16 ^ s21;
        let c2 = s2 ^ s7 ^ s12 ^ s17 ^ s22;
        let c3 = s3 ^ s8 ^ s13 ^ s18 ^ s23;
        let c4 = s4 ^ s9 ^ s14 ^ s19 ^ s24;
        let d0 = c4 ^ c1.rotate_left(1);
        let d1 = c0 ^ c2.rotate_left(1);
        let d2 = c1 ^ c3.rotate_left(1);
        let d3 = c2 ^ c4.rotate_left(1);
        let d4 = c3 ^ c0.rotate_left(1);
        s0 ^= d0;
        s5 ^= d0;
        s10 ^= d0;
        s15 ^= d0;
        s20 ^= d0;
        s1 ^= d1;
        s6 ^= d1;
        s11 ^= d1;
        s16 ^= d1;
        s21 ^= d1;
        s2 ^= d2;
        s7 ^= d2;
        s12 ^= d2;
        s17 ^= d2;
        s22 ^= d2;
        s3 ^= d3;
        s8 ^= d3;
        s13 ^= d3;
        s18 ^= d3;
        s23 ^= d3;
        s4 ^= d4;
        s9 ^= d4;
        s14 ^= d4;
        s19 ^= d4;
        s24 ^= d4;

        // rho + pi (b[y + 5*((2x+3y)%5)] = rot(s[x+5y], RHO[x+5y]))
        let b0 = s0.rotate_left(0);
        let b10 = s1.rotate_left(1);
        let b20 = s2.rotate_left(62);
        let b5 = s3.rotate_left(28);
        let b15 = s4.rotate_left(27);
        let b16 = s5.rotate_left(36);
        let b1 = s6.rotate_left(44);
        let b11 = s7.rotate_left(6);
        let b21 = s8.rotate_left(55);
        let b6 = s9.rotate_left(20);
        let b7 = s10.rotate_left(3);
        let b17 = s11.rotate_left(10);
        let b2 = s12.rotate_left(43);
        let b12 = s13.rotate_left(25);
        let b22 = s14.rotate_left(39);
        let b23 = s15.rotate_left(41);
        let b8 = s16.rotate_left(45);
        let b18 = s17.rotate_left(15);
        let b3 = s18.rotate_left(21);
        let b13 = s19.rotate_left(8);
        let b14 = s20.rotate_left(18);
        let b24 = s21.rotate_left(2);
        let b9 = s22.rotate_left(61);
        let b19 = s23.rotate_left(56);
        let b4 = s24.rotate_left(14);

        // chi
        s0 = b0 ^ (!b1 & b2);
        s1 = b1 ^ (!b2 & b3);
        s2 = b2 ^ (!b3 & b4);
        s3 = b3 ^ (!b4 & b0);
        s4 = b4 ^ (!b0 & b1);
        s5 = b5 ^ (!b6 & b7);
        s6 = b6 ^ (!b7 & b8);
        s7 = b7 ^ (!b8 & b9);
        s8 = b8 ^ (!b9 & b5);
        s9 = b9 ^ (!b5 & b6);
        s10 = b10 ^ (!b11 & b12);
        s11 = b11 ^ (!b12 & b13);
        s12 = b12 ^ (!b13 & b14);
        s13 = b13 ^ (!b14 & b10);
        s14 = b14 ^ (!b10 & b11);
        s15 = b15 ^ (!b16 & b17);
        s16 = b16 ^ (!b17 & b18);
        s17 = b17 ^ (!b18 & b19);
        s18 = b18 ^ (!b19 & b15);
        s19 = b19 ^ (!b15 & b16);
        s20 = b20 ^ (!b21 & b22);
        s21 = b21 ^ (!b22 & b23);
        s22 = b22 ^ (!b23 & b24);
        s23 = b23 ^ (!b24 & b20);
        s24 = b24 ^ (!b20 & b21);

        // iota
        s0 ^= rc;
    }

    state[0] = s0;
    state[1] = s1;
    state[2] = s2;
    state[3] = s3;
    state[4] = s4;
    state[5] = s5;
    state[6] = s6;
    state[7] = s7;
    state[8] = s8;
    state[9] = s9;
    state[10] = s10;
    state[11] = s11;
    state[12] = s12;
    state[13] = s13;
    state[14] = s14;
    state[15] = s15;
    state[16] = s16;
    state[17] = s17;
    state[18] = s18;
    state[19] = s19;
    state[20] = s20;
    state[21] = s21;
    state[22] = s22;
    state[23] = s23;
    state[24] = s24;
}

/// The pre-unrolling reference form — retained for the differential
/// test against [`keccak_f1600`].
#[cfg(test)]
pub(crate) fn keccak_f1600_reference(state: &mut [u64; 25]) {
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

/// Generic Keccak sponge (original padding, not SHA3's 0x06 domain byte —
/// the domain byte is supplied by the caller for flexibility).
pub struct KeccakSponge {
    state: [u64; 25],
    rate: usize,
    /// Position within the current rate block.
    pos: usize,
    suffix: u8,
}

impl Clone for KeccakSponge {
    fn clone(&self) -> Self {
        KeccakSponge {
            state: self.state,
            rate: self.rate,
            pos: self.pos,
            suffix: self.suffix,
        }
    }
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

    /// Finish absorbing (apply padding and permute once), transitioning the sponge to squeeze
    /// mode without consuming it. After this, `update` must not be called; use `squeeze`.
    pub fn finalize_in_place(&mut self) {
        let pos = self.pos;
        self.state[pos / 8] ^= (self.suffix as u64) << (8 * (pos % 8));
        let last = self.rate - 1;
        self.state[last / 8] ^= 0x80u64 << (8 * (last % 8));
        keccak_f1600(&mut self.state);
        self.pos = 0;
    }

    /// Squeeze `out.len()` bytes from a finalized sponge (`finalize_in_place` already called),
    /// permuting between rate blocks. This is the raw SHAKE squeeze step.
    pub fn squeeze(&mut self, out: &mut [u8]) {
        let mut done = 0;
        while done < out.len() {
            if self.pos == self.rate {
                keccak_f1600(&mut self.state);
                self.pos = 0;
            }
            let avail = self.rate - self.pos;
            let take = avail.min(out.len() - done);
            let mut i = 0;
            while i < take {
                let lane = (self.pos + i) / 8;
                let off = (self.pos + i) % 8;
                let bytes = self.state[lane].to_le_bytes();
                let chunk = (8 - off).min(take - i);
                out[done + i..done + i + chunk].copy_from_slice(&bytes[off..off + chunk]);
                i += chunk;
            }
            self.pos += take;
            done += take;
        }
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
            for (i, lane) in self.state.iter().enumerate() {
                block[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
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
    fn unrolled_permutation_matches_reference() {
        // The unrolled keccak_f1600 must be bit-identical to the loop
        // reference across random and structured states.
        let mut rng: u64 = 0xdead_beef_cafe_f00d;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng
        };
        for case in 0..200 {
            let mut a = [0u64; 25];
            let mut b = [0u64; 25];
            if case == 0 {
                // all-zero and single-lane states exercise the extremes
            } else if case == 1 {
                a[0] = u64::MAX;
                b[0] = u64::MAX;
            } else if case == 2 {
                for i in 0..25 {
                    a[i] = u64::MAX;
                    b[i] = u64::MAX;
                }
            } else {
                for lane in a.iter_mut() {
                    *lane = next();
                }
                b.copy_from_slice(&a);
            }
            super::keccak_f1600(&mut a);
            super::keccak_f1600_reference(&mut b);
            assert_eq!(a, b, "case {case}");
        }
    }

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
        assert_eq!(
            hex(&long),
            "2b43036c229ba512995f91fdb46fcd5327a4dc834d86d6e0f58a08053346dc2e"
        );
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
