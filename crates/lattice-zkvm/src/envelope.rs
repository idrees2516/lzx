//! The bounded canonical proof envelope: digests bound before the first
//! challenge, sectioned payload, strict decoder.

use lattice_core::transcript::Transcript;

/// Proof envelope sections in canonical order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Section {
    /// Commitment bytes (Akita commitment rows).
    Commitment(Vec<u8>),
    /// Sumcheck proof bytes (round evaluations, 8 bytes each).
    Sumcheck(Vec<u8>),
    /// Opened witness bytes (ring elements, 4 bytes per coefficient).
    Witness(Vec<u8>),
    /// Norm-proof digits (i64 LE per digit).
    Norm(Vec<u8>),
}

/// The envelope: versioned, digest-bound, length-capped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofEnvelope {
    pub version: u32,
    pub program_digest: [u8; 32],
    pub public_input_digest: [u8; 32],
    pub public_output_digest: [u8; 32],
    pub sections: Vec<Section>,
}

/// Hard caps (allocation-bomb defense).
pub const MAX_SECTIONS: usize = 64;
pub const MAX_SECTION_BYTES: usize = 1 << 24; // 16 MiB
/// Total proof-size cap: the *sum* of section payloads. Without this,
/// a 64 x 16 MiB section-count bomb allocates 1 GiB before any check
/// runs (audit SS10.2: no unbounded allocations on untrusted paths).
pub const MAX_TOTAL_SECTION_BYTES: usize = 1 << 25; // 32 MiB

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    VersionUnsupported { got: u32 },
    TooManySections { got: usize },
    SectionTooLarge { tag: u8, got: usize },
    TotalTooLarge { got: usize },
    TrailingBytes { got: usize },
    MissingSection { tag: u8 },
    DuplicateSection { tag: u8 },
}

impl ProofEnvelope {
    pub const VERSION: u32 = 1;

    /// Build an envelope (validating caps).
    pub fn new(
        program_digest: [u8; 32],
        public_input_digest: [u8; 32],
        public_output_digest: [u8; 32],
        sections: Vec<Section>,
    ) -> Result<Self, EnvelopeError> {
        if sections.len() > MAX_SECTIONS {
            return Err(EnvelopeError::TooManySections {
                got: sections.len(),
            });
        }
        let mut seen = [false; 5];
        let mut total = 0usize;
        for s in &sections {
            let (tag, len) = match s {
                Section::Commitment(b) => (1u8, b.len()),
                Section::Sumcheck(b) => (2u8, b.len()),
                Section::Witness(b) => (3u8, b.len()),
                Section::Norm(b) => (4u8, b.len()),
            };
            if len > MAX_SECTION_BYTES {
                return Err(EnvelopeError::SectionTooLarge { tag, got: len });
            }
            total = total.saturating_add(len);
            if total > MAX_TOTAL_SECTION_BYTES {
                return Err(EnvelopeError::TotalTooLarge { got: total });
            }
            let idx = tag as usize;
            if idx < seen.len() {
                if seen[idx] {
                    return Err(EnvelopeError::DuplicateSection { tag });
                }
                seen[idx] = true;
            }
        }
        Ok(ProofEnvelope {
            version: Self::VERSION,
            program_digest,
            public_input_digest,
            public_output_digest,
            sections,
        })
    }

    /// Canonical serialization: header + length-prefixed sections.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.program_digest);
        out.extend_from_slice(&self.public_input_digest);
        out.extend_from_slice(&self.public_output_digest);
        out.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for s in &self.sections {
            let (tag, bytes) = match s {
                Section::Commitment(b) => (1u8, b),
                Section::Sumcheck(b) => (2u8, b),
                Section::Witness(b) => (3u8, b),
                Section::Norm(b) => (4u8, b),
            };
            out.push(tag);
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        out
    }

    /// Strict decode: rejects bad versions, oversize sections, duplicate
    /// tags, and trailing bytes (the verifier's no-panic contract).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EnvelopeError> {
        if bytes.len() < 4 + 32 * 3 + 4 {
            return Err(EnvelopeError::TrailingBytes { got: bytes.len() });
        }
        let version = u32::from_le_bytes(
            bytes[..4].try_into().unwrap_or([0u8; 4]),
        );
        if version != Self::VERSION {
            return Err(EnvelopeError::VersionUnsupported { got: version });
        }
        let mut off = 4;
        let mut digests = [[0u8; 32]; 3];
        for d in digests.iter_mut() {
            d.copy_from_slice(
                bytes.get(off..off + 32).ok_or(EnvelopeError::TrailingBytes {
                    got: bytes.len(),
                })?,
            );
            off += 32;
        }
        let num_sections = u32::from_le_bytes(
            bytes.get(off..off + 4)
                .ok_or(EnvelopeError::TrailingBytes { got: bytes.len() })?
                .try_into()
                .unwrap_or([0u8; 4]),
        ) as usize;
        off += 4;
        if num_sections > MAX_SECTIONS {
            return Err(EnvelopeError::TooManySections {
                got: num_sections,
            });
        }
        let mut sections = Vec::with_capacity(num_sections);
        let mut seen = [false; 5];
        let mut total = 0usize;
        for _ in 0..num_sections {
            let tag = *bytes.get(off).ok_or(EnvelopeError::TrailingBytes {
                got: bytes.len(),
            })?;
            off += 1;
            let len = u32::from_le_bytes(
                bytes.get(off..off + 4)
                    .ok_or(EnvelopeError::TrailingBytes { got: bytes.len() })?
                    .try_into()
                    .unwrap_or([0u8; 4]),
            ) as usize;
            off += 4;
            if len > MAX_SECTION_BYTES {
                return Err(EnvelopeError::SectionTooLarge { tag, got: len });
            }
            // Sum cap enforced BEFORE the section allocation.
            total = total.saturating_add(len);
            if total > MAX_TOTAL_SECTION_BYTES {
                return Err(EnvelopeError::TotalTooLarge { got: total });
            }
            let data = bytes
                .get(off..off + len)
                .ok_or(EnvelopeError::TrailingBytes { got: bytes.len() })?;
            off += len;
            let section = match tag {
                1 => Section::Commitment(data.to_vec()),
                2 => Section::Sumcheck(data.to_vec()),
                3 => Section::Witness(data.to_vec()),
                4 => Section::Norm(data.to_vec()),
                _ => {
                    return Err(EnvelopeError::MissingSection { tag });
                }
            };
            let idx = tag as usize;
            if idx < seen.len() {
                if seen[idx] {
                    return Err(EnvelopeError::DuplicateSection { tag });
                }
                seen[idx] = true;
            }
            sections.push(section);
        }
        if off != bytes.len() {
            return Err(EnvelopeError::TrailingBytes {
                got: bytes.len() - off,
            });
        }
        Ok(ProofEnvelope {
            version,
            program_digest: digests[0],
            public_input_digest: digests[1],
            public_output_digest: digests[2],
            sections,
        })
    }

    /// Absorb the envelope header into a transcript (preamble before any
    /// challenge).
    pub fn absorb_header(&self, transcript: &mut Transcript, label: &[u8]) {
        let _ = transcript.append_bytes(label, &self.to_bytes()[..4 + 96]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip() {
        let env = ProofEnvelope::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            vec![
                Section::Commitment(vec![9u8; 100]),
                Section::Sumcheck(vec![1, 2, 3]),
                Section::Witness(vec![0xab; 64]),
                Section::Norm(vec![0; 16]),
            ],
        )
        .ok()
        .unwrap();
        let bytes = env.to_bytes();
        let back = ProofEnvelope::from_bytes(&bytes).ok().unwrap();
        assert_eq!(back, env);
    }

    #[test]
    fn envelope_rejects_trailing_bytes() {
        let env = ProofEnvelope::new([1u8; 32], [2u8; 32], [3u8; 32], vec![])
            .ok()
            .unwrap();
        let mut bytes = env.to_bytes();
        bytes.push(0xff);
        assert!(matches!(
            ProofEnvelope::from_bytes(&bytes),
            Err(EnvelopeError::TrailingBytes { .. })
        ));
    }

    #[test]
    fn envelope_rejects_bad_version() {
        let env = ProofEnvelope::new([1u8; 32], [2u8; 32], [3u8; 32], vec![])
            .ok()
            .unwrap();
        let mut bytes = env.to_bytes();
        bytes[0] = 0x63; // version 99 (LE first byte)
        bytes[1] = 0;
        bytes[2] = 0;
        bytes[3] = 0;
        assert!(matches!(
            ProofEnvelope::from_bytes(&bytes),
            Err(EnvelopeError::VersionUnsupported { got: 99 })
        ));
    }

    #[test]
    fn envelope_rejects_duplicates_and_oversize() {
        let dup = ProofEnvelope::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            vec![Section::Sumcheck(vec![1]), Section::Sumcheck(vec![2])],
        );
        assert!(matches!(
            dup,
            Err(EnvelopeError::DuplicateSection { tag: 2 })
        ));
        let big = ProofEnvelope::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            vec![Section::Witness(vec![0u8; MAX_SECTION_BYTES + 1])],
        );
        assert!(matches!(
            big,
            Err(EnvelopeError::SectionTooLarge { tag: 3, .. })
        ));
    }
}
