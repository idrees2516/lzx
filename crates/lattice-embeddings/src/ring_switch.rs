//! Hachi (Nguyen, O'Rourke, Zhang; ePrint 2025/1055-lineage): the HMZ
//! ring-switch norm proof — ported from the lattice-zk-lab reference
//! implementation onto this workspace's ring sumcheck engine.
//!
//! Hachi lifts a norm proof for a witness over one ring to a sumcheck
//! over the extension slot: the prover commits `w`, sends
//! `t = ⟨w, w̄⟩` (the conjugate inner product), and proves it with the
//! degree-2 ring sumcheck `Σ_z MLE[w̄](z)·MLE[w](z) = t`; the verifier's
//! balanced-trace gate `Tr(t) = n·‖cf(w)‖² ≤ n·β²` reads the integer
//! norm exactly in the wraparound-free regime (the constant-term
//! identity making the verifier's cyclotomic multiplications free).
//! The switched evaluation claim (`MLE[w](r)` lifted through the slot
//! embedding) is the caller's opening layer — the switch itself is the
//! norm proof plus the trace gate.
//!
//! Deviation (kernel scale): the extension-slot challenge evaluation is
//! realized through this workspace's `R_q` sumcheck with `Z_q` round
//! challenges (the paper's F_{q^k} slot structure is a packing
//! optimisation on top of the same algebraic identity).

use lattice_core::transcript::{Transcript, TranscriptError};
use lattice_ring::{RingConfig, RingElement};
use lattice_salsa::ring_sc::{
    norm_conjugate_inner, ring_sc_prove, ring_sc_verify, trace_balanced, ProductClaim, RingScError,
    RingScProof,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RingSwitchError {
    RingSc(RingScError),
    Transcript(TranscriptError),
    /// The balanced-trace gate failed: Tr(t) < 0 or > n·β².
    TraceGateFailed { trace: i64 },
    /// The terminal identity failed.
    TerminalFailed,
    /// The revealed norm does not match the claim.
    NormMismatch,
    Shape { expected: usize, got: usize },
}

impl From<RingScError> for RingSwitchError {
    fn from(e: RingScError) -> Self {
        RingSwitchError::RingSc(e)
    }
}
impl From<TranscriptError> for RingSwitchError {
    fn from(e: TranscriptError) -> Self {
        RingSwitchError::Transcript(e)
    }
}

/// The ring-switch proof: the conjugate inner product, the degree-2
/// sumcheck, and the two final openings.
#[derive(Clone, Debug)]
pub struct RingSwitchProof {
    pub t: RingElement,
    pub sumcheck: RingScProof,
    pub v0: RingElement,
    pub v1: RingElement,
    pub point: Vec<u32>,
}

/// The HMZ ring-switch prover: `t = ⟨w, w̄⟩` proven directly with the
/// degree-2 sumcheck; the openings v0 = MLE[w](r), v1 = MLE[w̄](r).
pub fn ring_switch_prove(
    ring: &RingConfig,
    w: &[RingElement],
    transcript: &mut Transcript,
) -> Result<RingSwitchProof, RingSwitchError> {
    if w.is_empty() || !w.len().is_power_of_two() {
        return Err(RingSwitchError::Shape {
            expected: 0,
            got: w.len(),
        });
    }
    let wbar: Vec<RingElement> = w
        .iter()
        .map(lattice_salsa::ring_sc::conj)
        .collect();
    let t = norm_conjugate_inner(w)?;
    let claim = ProductClaim {
        tables: vec![wbar.clone(), w.to_vec()],
        value: t.clone(),
    };
    let sumcheck = ring_sc_prove(ring, &[claim], &[ring.one()], transcript)?;
    let point = sumcheck.point.clone();
    let v0 = lattice_salsa::ring_sc::mle_eval_ring(w, &point)?;
    let v1 = lattice_salsa::ring_sc::mle_eval_ring(&wbar, &point)?;
    Ok(RingSwitchProof {
        t,
        sumcheck,
        v0,
        v1,
        point,
    })
}

/// The HMZ ring-switch verifier: the balanced-trace gate, the round
/// checks, and the terminal `v1·v0 == last`.
pub fn ring_switch_verify(
    ring: &RingConfig,
    beta: u64,
    proof: &RingSwitchProof,
    transcript: &mut Transcript,
) -> Result<(), RingSwitchError> {
    // trace gate: Tr(t) = n·||cf(w)||^2 in [0, n·beta^2]
    let tr = trace_balanced(&proof.t);
    if tr < 0 || tr > (ring.n() as i64) * (beta as i64) * (beta as i64) {
        return Err(RingSwitchError::TraceGateFailed { trace: tr });
    }
    let mu = proof.sumcheck.rounds.len();
    let last = ring_sc_verify(ring, 2, mu, &proof.t, &proof.sumcheck, transcript)?;
    if proof.v0.mul(&proof.v1).map_err(RingScError::Ring)? != last {
        return Err(RingSwitchError::TerminalFailed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> RingConfig {
        lattice_ring::RingConfig::new(lattice_ring::Modulus32::Q_32, 4)
            .ok()
            .unwrap()
    }

    fn small_vec(ring: &RingConfig, m: usize, tag: &[u8], span: u32) -> Vec<RingElement> {
        (0..m)
            .map(|i| {
                let bytes = Transcript::xof(
                    b"hachi-test",
                    &[tag, &(i as u32).to_le_bytes()].concat(),
                    4 * ring.n(),
                );
                let coeffs: Vec<u32> = bytes
                    .chunks(4)
                    .take(ring.n())
                    .map(|c| {
                        let mut a = [0u8; 4];
                        a.copy_from_slice(&c[..4]);
                        u32::from_le_bytes(a) % (2 * span + 1)
                    })
                    .collect();
                RingElement::from_coeffs(ring, coeffs)
            })
            .collect()
    }

    #[test]
    fn ring_switch_honest_and_tampered() {
        let ring = ring();
        let w = small_vec(&ring, 4, b"hw", 8);
        let mut t = Transcript::new_default(b"lzx-hachi");
        let proof = ring_switch_prove(&ring, &w, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-hachi");
        assert!(ring_switch_verify(&ring, 1024, &proof, &mut vt).is_ok());
        // tampered t fails the trace gate
        let mut bad = proof.clone();
        bad.t = bad.t.add(&ring.one()).ok().unwrap();
        let mut vt2 = Transcript::new_default(b"lzx-hachi");
        assert!(ring_switch_verify(&ring, 1024, &bad, &mut vt2).is_err());
        // tampered v0 fails the terminal
        let mut bad2 = proof.clone();
        bad2.v0 = bad2.v0.add(&ring.one()).ok().unwrap();
        let mut vt3 = Transcript::new_default(b"lzx-hachi");
        assert!(ring_switch_verify(&ring, 1024, &bad2, &mut vt3).is_err());
        // tampered sumcheck rounds rejected
        let mut bad3 = proof.clone();
        let r0 = bad3.sumcheck.rounds[0][0].clone();
        bad3.sumcheck.rounds[0][0] = r0.add(&ring.one()).ok().unwrap();
        let mut vt4 = Transcript::new_default(b"lzx-hachi");
        assert!(ring_switch_verify(&ring, 1024, &bad3, &mut vt4).is_err());
    }

    #[test]
    fn trace_gate_rejects_inflated_beta() {
        let ring = ring();
        // witness with 16 coefficients up to 16: norm² up to 4096;
        // beta = 4 bounds the flattened norm at 16 → the honest proof
        // must be rejected by the tight gate
        let w = small_vec(&ring, 4, b"tb", 8);
        let mut t = Transcript::new_default(b"lzx-hachi");
        let proof = ring_switch_prove(&ring, &w, &mut t).ok().unwrap();
        let mut vt = Transcript::new_default(b"lzx-hachi");
        assert!(ring_switch_verify(&ring, 4, &proof, &mut vt).is_err());
    }

    #[test]
    fn non_power_of_two_rejected() {
        let ring = ring();
        let w = small_vec(&ring, 3, b"np", 8);
        let mut t = Transcript::new_default(b"lzx-hachi");
        assert!(matches!(
            ring_switch_prove(&ring, &w, &mut t),
            Err(RingSwitchError::Shape { .. })
        ));
    }
}
