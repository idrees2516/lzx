//! Machine state: 32 registers, PC, sparse word memory. The zkVM layer
//! owns the paged/committed layout; this kernel keeps a deterministic
//! sparse map keyed by word address.

use std::collections::BTreeMap;

/// Word-granular machine memory (deterministic ordering for transcript
/// stability).
#[derive(Clone, Debug, Default)]
pub struct Memory {
    /// Word address -> 64-bit value (address is byte address, 8-aligned).
    words: BTreeMap<u64, u64>,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            words: BTreeMap::new(),
        }
    }

    /// Load a 64-bit word; missing addresses read zero (sparse-default).
    pub fn load(&self, addr: u64) -> u64 {
        self.words.get(&addr).copied().unwrap_or(0)
    }

    /// Store a 64-bit word.
    pub fn store(&mut self, addr: u64, value: u64) {
        self.words.insert(addr, value);
    }

    /// Load a 32-bit word at any alignment (unaligned subword loads span
    /// two 64-bit words).
    pub fn load_word32(&self, addr: u64) -> u64 {
        let base = addr & !0x7;
        let off = (addr & 0x7) as u32;
        let w = self.load(base);
        if off <= 4 {
            (w >> (off * 8)) & 0xFFFF_FFFF
        } else {
            let next = self.load(base + 8);
            let lo_bits = 64 - off * 8;
            ((w >> (off * 8)) | (next << lo_bits)) & 0xFFFF_FFFF
        }
    }

    /// Store a 32-bit word at any alignment (subword expansion: the value
    /// straddles two 64-bit words when offset > 4).
    pub fn store_word32(&mut self, addr: u64, value: u32) {
        let base = addr & !0x7;
        let off = (addr & 0x7) as u32;
        let mut w = self.load(base);
        if off <= 4 {
            let mask = 0xFFFF_FFFFu64 << (off * 8);
            w = (w & !mask) | ((value as u64) << (off * 8));
            self.store(base, w);
        } else {
            // Bits [off*8, 64) of this word hold the value's low part.
            let lo_bits = 64 - off * 8;
            let lo_mask: u64 = (1u64 << lo_bits) - 1;
            let pos_mask = lo_mask << (off * 8);
            w = (w & !pos_mask) | (((value as u64) & lo_mask) << (off * 8));
            self.store(base, w);
            // The value's high part lands in the next word's low bits.
            let hi = self.load(base + 8);
            let hi_mask: u64 = u32::MAX as u64 >> lo_bits;
            let hi = (hi & !hi_mask) | ((value as u64) >> lo_bits);
            self.store(base + 8, hi);
        }
    }

    /// Deterministic (address, value) pairs (the Twist final state).
    pub fn snapshot_pairs(&self) -> Vec<(u64, u64)> {
        self.words.iter().map(|(a, v)| (*a, *v)).collect()
    }

    /// Memory footprint in words.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Deterministic memory digest (initial/final state commitments).
    pub fn digest(&self) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(self.words.len() * 16);
        for (addr, value) in &self.words {
            bytes.extend_from_slice(&addr.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        lattice_core::transcript::Transcript::hash_domain(b"vm-memory", &bytes)
    }
}

/// Full machine state.
#[derive(Clone, Debug)]
pub struct MachineState {
    pub pc: u64,
    /// x0 is hardwired zero (enforced on every write).
    pub regs: [u64; 32],
    pub memory: Memory,
    /// LR/SC reservation address (None = no reservation).
    pub reservation: Option<u64>,
    /// Execution halted (ecall/ebreak).
    pub halted: bool,
}

impl Default for MachineState {
    fn default() -> Self {
        Self::new()
    }
}

impl MachineState {
    pub fn new() -> Self {
        MachineState {
            pc: 0,
            regs: [0u64; 32],
            memory: Memory::new(),
            reservation: None,
            halted: false,
        }
    }

    /// Register read (x0 hardwired to zero).
    pub fn reg(&self, idx: u8) -> u64 {
        if idx == 0 {
            0
        } else {
            self.regs[idx as usize]
        }
    }

    /// Register write (x0 writes are discarded).
    pub fn set_reg(&mut self, idx: u8, value: u64) {
        if idx != 0 {
            self.regs[idx as usize] = value;
        }
    }

    /// Load a program image at an address.
    pub fn load_program(&mut self, base: u64, program: &[u8]) {
        for (i, chunk) in program.chunks(8).enumerate() {
            let mut word = 0u64;
            for (b, byte) in chunk.iter().enumerate() {
                word |= (*byte as u64) << (b * 8);
            }
            self.memory.store(base + (i as u64) * 8, word);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x0_hardwired() {
        let mut s = MachineState::new();
        s.set_reg(0, 42);
        assert_eq!(s.reg(0), 0);
        s.set_reg(5, 7);
        assert_eq!(s.reg(5), 7);
    }

    #[test]
    fn word32_subword_roundtrip() {
        let mut m = Memory::new();
        for off in [0u64, 4, 1, 3, 5, 7] {
            let addr = 0x100 + off;
            m.store_word32(addr, 0xDEAD_BEEF);
            assert_eq!(m.load_word32(addr), 0xDEAD_BEEF, "offset {off}");
        }
    }

    #[test]
    fn sparse_default_zero() {
        let m = Memory::new();
        assert_eq!(m.load(0x9999), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn program_load_and_digest_deterministic() {
        let mut s1 = MachineState::new();
        let mut s2 = MachineState::new();
        let program = [0x13u8, 0x05, 0x50, 0x01, 0x93, 0x05, 0x50, 0x02, 0x73, 0x00, 0x00, 0x00];
        s1.load_program(0x1000, &program);
        s2.load_program(0x1000, &program);
        assert_eq!(s1.memory.digest(), s2.memory.digest());
        assert_eq!(s1.memory.load(0x1000) & 0xFFFF_FFFF, 0x0150_0513);
    }
}
