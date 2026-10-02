//! The ABDLOP commitment scheme (§2.5.2) with the componentwise
//! instantiation over R_K (§3.3).
//!
//! Public matrices A₁ ∈ R_F^{κ×m₁}, A₂ ∈ R_F^{κ×m₂}, B ∈ R_F^{ℓ×m₂}
//! (uniform over R_F — genuinely R_F, NOT confined to ψ(R_K), §3.3.1).
//! Commitment of message m ∈ R_F^ℓ (in the BDLOP slot) with salts
//! s₁ ∈ R_F^{m₁}, s₂ ∈ R_F^{m₂}:
//!
//! ```text
//! t_A = A₁·s₁ + A₂·s₂           (mod q)      — the Ajtai part
//! t_B = B·s₂ + m                (mod q)      — the BDLOP part
//! ```
//!
//! Every message of this paper rides in the BDLOP slot (ℓ ≥ 1 — §2.1's
//! standing convention); both s₁ and s₂ are salts. R_K-valued messages
//! are committed **componentwise**: an R_K element x = a + bY occupies
//! TWO BDLOP slots (a, b) — the rank-doubling φ embedding; nothing in the
//! protocol requires MLWE/MSIS over R_K.
//!
//! The message layout per commitment block is explicit:
//! [`MsgLayout`] interleaves R_K messages (slot pairs) and R_F messages
//! (single slots), mirroring the packed-coefficients commitments, the
//! hint commitments, the surrogate commitments and the (c_y, x_y)
//! mask-block tuple of Remark 4.21.

use crate::fp::Fq;
use crate::gauss::Rng;
use crate::ring::Poly;
use crate::rk::PolyK;

/// Where a BDLOP slot's message came from.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotKind {
    /// The a-part (or b-part) of the ℓ-th R_K message of the block.
    RkPart { msg: usize, part: u8 },
    /// A native R_F message.
    Rf { msg: usize },
    /// The garbage slot appended by the Π_many^(ct) wrapper (Protocol 4).
    Garbage,
}

/// The per-block message layout: which BDLOP slots hold which messages.
#[derive(Clone, Debug)]
pub struct MsgLayout {
    pub kinds: Vec<SlotKind>,
    /// Number of R_K messages.
    pub n_rk: usize,
    /// Number of native R_F messages.
    pub n_rf: usize,
    /// Whether a garbage slot is appended (Protocol 4).
    pub has_garbage: bool,
}

impl MsgLayout {
    /// Layout for `n_rk` R_K messages: slots (a₀, b₀, a₁, b₁, ...).
    pub fn rk_only(n_rk: usize) -> MsgLayout {
        let mut kinds = Vec::with_capacity(2 * n_rk);
        for i in 0..n_rk {
            kinds.push(SlotKind::RkPart { msg: i, part: 0 });
            kinds.push(SlotKind::RkPart { msg: i, part: 1 });
        }
        MsgLayout {
            kinds,
            n_rk,
            n_rf: 0,
            has_garbage: false,
        }
    }

    /// Layout mixing R_K and R_F messages: R_K pairs first, then R_F.
    pub fn mixed(n_rk: usize, n_rf: usize) -> MsgLayout {
        let mut base = MsgLayout::rk_only(n_rk);
        for i in 0..n_rf {
            base.kinds.push(SlotKind::Rf { msg: i });
        }
        base.n_rf = n_rf;
        base
    }

    pub fn with_garbage(mut self) -> MsgLayout {
        self.has_garbage = true;
        self.kinds.push(SlotKind::Garbage);
        self
    }

    pub fn slots(&self) -> usize {
        self.kinds.len()
    }
}

/// The ABDLOP public parameters (shared by every commitment).
#[derive(Clone, Debug)]
pub struct AbdlopPp {
    pub kappa: usize,
    pub ell: usize, // = κ′ = number of BDLOP message slots
    pub m1: usize,
    pub m2: usize,
    pub d: usize,
    pub a1: Vec<Vec<Poly>>, // κ × m1
    pub a2: Vec<Vec<Poly>>, // κ × m2
    pub b: Vec<Vec<Poly>>,  // ℓ × m2
}

impl AbdlopPp {
    pub fn setup(kappa: usize, ell: usize, m1: usize, m2: usize, d: usize, rng: &mut Rng) -> AbdlopPp {
        let uniform_poly = |rng: &mut Rng| -> Poly {
            let c: Vec<Fq> = (0..d).map(|_| Fq(rng.next_u64() % crate::fp::Q)).collect();
            Poly(c)
        };
        let a1: Vec<Vec<Poly>> = (0..kappa)
            .map(|_| (0..m1).map(|_| uniform_poly(rng)).collect())
            .collect();
        let a2: Vec<Vec<Poly>> = (0..kappa)
            .map(|_| (0..m2).map(|_| uniform_poly(rng)).collect())
            .collect();
        let b: Vec<Vec<Poly>> = (0..ell)
            .map(|_| (0..m2).map(|_| uniform_poly(rng)).collect())
            .collect();
        AbdlopPp {
            kappa,
            ell,
            m1,
            m2,
            d,
            a1,
            a2,
            b,
        }
    }

    /// t_A = A₁·s₁ + A₂·s₂.
    pub fn t_a(&self, s1: &[Poly], s2: &[Poly]) -> Vec<Poly> {
        debug_assert_eq!(s1.len(), self.m1);
        debug_assert_eq!(s2.len(), self.m2);
        let mut out = Vec::with_capacity(self.kappa);
        for r in 0..self.kappa {
            let mut acc = Poly::zero(self.d);
            for (m, s) in self.a1[r].iter().zip(s1.iter()) {
                acc.add_assign(&m.mul(s));
            }
            for (m, s) in self.a2[r].iter().zip(s2.iter()) {
                acc.add_assign(&m.mul(s));
            }
            out.push(acc);
        }
        out
    }

    /// t_B = B·s₂ + m.
    pub fn t_b(&self, s2: &[Poly], msgs: &[Poly]) -> Vec<Poly> {
        debug_assert_eq!(msgs.len(), self.ell);
        let mut out = Vec::with_capacity(self.ell);
        for r in 0..self.ell {
            let mut acc = msgs[r].clone();
            for (m, s) in self.b[r].iter().zip(s2.iter()) {
                acc.add_assign(&m.mul(s));
            }
            out.push(acc);
        }
        out
    }

    /// B·s₂ (the salt aggregate of the BDLOP part).
    pub fn b_s2(&self, s2: &[Poly]) -> Vec<Poly> {
        let mut out = Vec::with_capacity(self.ell);
        for r in 0..self.ell {
            let mut acc = Poly::zero(self.d);
            for (m, s) in self.b[r].iter().zip(s2.iter()) {
                acc.add_assign(&m.mul(s));
            }
            out.push(acc);
        }
        out
    }
}

/// One ABDLOP commitment block.
#[derive(Clone, Debug)]
pub struct AbdlopCommitment {
    pub t_a: Vec<Poly>,
    pub t_b: Vec<Poly>,
    pub layout: MsgLayout,
}

/// The secret side: salts + messages in slot order.
#[derive(Clone, Debug)]
pub struct AbdlopOpening {
    pub s1: Vec<Poly>,
    pub s2: Vec<Poly>,
    /// The BDLOP slots in order (length ℓ), as R_F elements.
    pub slots: Vec<Poly>,
}

impl AbdlopOpening {
    /// Commit R_K messages (+ optional R_F messages) with ternary salts.
    pub fn commit_rk(
        pp: &AbdlopPp,
        rk_msgs: &[PolyK],
        rf_msgs: &[Poly],
        rng: &mut Rng,
    ) -> (AbdlopCommitment, AbdlopOpening) {
        let mut slots: Vec<Poly> = Vec::with_capacity(pp.ell);
        for x in rk_msgs {
            slots.push(x.a.clone());
            slots.push(x.b.clone());
        }
        for m in rf_msgs {
            slots.push(m.clone());
        }
        while slots.len() < pp.ell {
            slots.push(Poly::zero(pp.d));
        }
        debug_assert_eq!(slots.len(), pp.ell);
        let s1: Vec<Poly> = (0..pp.m1).map(|_| ternary_poly(pp.d, rng)).collect();
        let s2: Vec<Poly> = (0..pp.m2).map(|_| ternary_poly(pp.d, rng)).collect();
        let t_a = pp.t_a(&s1, &s2);
        let t_b = pp.t_b(&s2, &slots);
        let layout = MsgLayout::mixed(rk_msgs.len(), rf_msgs.len());
        (
            AbdlopCommitment {
                t_a,
                t_b,
                layout,
            },
            AbdlopOpening { s1, s2, slots },
        )
    }

    /// Verify an opening against a commitment.
    pub fn verify(&self, pp: &AbdlopPp, com: &AbdlopCommitment) -> bool {
        pp.t_a(&self.s1, &self.s2) == com.t_a && pp.t_b(&self.s2, &self.slots) == com.t_b
    }
}

fn ternary_poly(d: usize, rng: &mut Rng) -> Poly {
    Poly((0..d).map(|_| Fq::from_i64(rng.ternary())).collect())
}

/// Homomorphic combination of commitments:
/// (t_A, t_B) = Σ ρ_i·(t_A^{(i)}, t_B^{(i)}) + (t_A^{(y)}, t_B^{(y)}) with
/// R_F challenges ρ (the verifier's Step-21 check of Protocol 7).
pub fn homomorphic_comb(
    coms: &[&AbdlopCommitment],
    rhos: &[Poly],
    extra: Option<&AbdlopCommitment>,
) -> AbdlopCommitment {
    let kappa = coms.first().map(|c| c.t_a.len()).unwrap_or(0);
    let ell = coms.first().map(|c| c.t_b.len()).unwrap_or(0);
    let d = coms
        .first()
        .and_then(|c| c.t_a.first())
        .map(|p| p.d())
        .unwrap_or(0);
    let mut ta = vec![Poly::zero(d); kappa];
    let mut tb = vec![Poly::zero(d); ell];
    for (com, rho) in coms.iter().zip(rhos.iter()) {
        for i in 0..kappa {
            ta[i].add_assign(&rho.mul(&com.t_a[i]));
        }
        for i in 0..ell {
            tb[i].add_assign(&rho.mul(&com.t_b[i]));
        }
    }
    if let Some(e) = extra {
        for i in 0..kappa {
            ta[i].add_assign(&e.t_a[i]);
        }
        for i in 0..ell {
            tb[i].add_assign(&e.t_b[i]);
        }
    }
    let layout = coms
        .first()
        .map(|c| c.layout.clone())
        .unwrap_or_else(|| MsgLayout::rk_only(0));
    AbdlopCommitment {
        t_a: ta,
        t_b: tb,
        layout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abdlop_commit_open_roundtrip() {
        let mut rng = Rng::new(b"abdlop");
        let pp = AbdlopPp::setup(4, 8, 6, 10, 4, &mut rng);
        let mut ctr = 0u64;
        let msgs: Vec<PolyK> = (0..4).map(|_| PolyK::uniform(4, b"msg", &mut ctr)).collect();
        let (com, op) = AbdlopOpening::commit_rk(&pp, &msgs, &[], &mut rng);
        assert!(op.verify(&pp, &com));
        // Tamper: flip a message slot.
        let mut bad = op.clone();
        bad.slots[0] = bad.slots[0].add(&Poly::one(4));
        assert!(!bad.verify(&pp, &com));
        // Tamper: change salts.
        let mut bad2 = op.clone();
        bad2.s1[0] = bad2.s1[0].add(&Poly::one(4));
        assert!(!bad2.verify(&pp, &com));
    }

    #[test]
    fn homomorphic_combination() {
        let mut rng = Rng::new(b"hom");
        let pp = AbdlopPp::setup(3, 6, 5, 8, 4, &mut rng);
        let mut ctr = 0u64;
        let m1: Vec<PolyK> = (0..3).map(|_| PolyK::uniform(4, b"hm1", &mut ctr)).collect();
        let m2: Vec<PolyK> = (0..3).map(|_| PolyK::uniform(4, b"hm2", &mut ctr)).collect();
        let (c1, o1) = AbdlopOpening::commit_rk(&pp, &m1, &[], &mut rng);
        let (c2, o2) = AbdlopOpening::commit_rk(&pp, &m2, &[], &mut rng);
        let rho = Poly::small_b(4, 2, b"hrho", &mut ctr);
        // Folded: message m1·... the RLC of messages with K-degree-0
        // challenge — model the fold at the opening level:
        // slots fold with rho (ring mult), i.e. the folded message is
        // m1 + rho·m2 read componentwise.
        let comb = homomorphic_comb(&[&c1, &c2], &[Poly::one(4), rho.clone()], None);
        // Openings fold identically:
        let fs1: Vec<Poly> = o1
            .s1
            .iter()
            .zip(o2.s1.iter())
            .map(|(a, b)| a.add(&rho.mul(b)))
            .collect();
        let fs2: Vec<Poly> = o1
            .s2
            .iter()
            .zip(o2.s2.iter())
            .map(|(a, b)| a.add(&rho.mul(b)))
            .collect();
        let fslots: Vec<Poly> = o1
            .slots
            .iter()
            .zip(o2.slots.iter())
            .map(|(a, b)| a.add(&rho.mul(b)))
            .collect();
        let folded = AbdlopOpening {
            s1: fs1,
            s2: fs2,
            slots: fslots,
        };
        assert!(folded.verify(&pp, &comb), "ABDLOP is homomorphic");
    }

    #[test]
    fn componentwise_linear_relation() {
        // §3.3.2: A_K·x = t_K expands to ψ(A_K)(a;b) = (t_a; t_b).
        let mut ctr = 0u64;
        let d = 4;
        let ak = PolyK::uniform(d, b"ak", &mut ctr);
        let x = PolyK::uniform(d, b"xk", &mut ctr);
        let tk = ak.mul(&x);
        // ψ(A_K)·φ(x) = φ(A_K·x)
        let (xu, xv) = x.phi();
        let (top, bot) = crate::rk::psi_mul(&ak, &xu, &xv);
        assert_eq!((top.clone(), bot.clone()), tk.phi());
        // The componentwise R_F rows: the a-row and b-row.
        let nu = Fq::new(crate::fp::NU);
        assert_eq!(top, ak.a.mul(&xu).add(&ak.b.mul(&xv).scale(&nu)));
        assert_eq!(bot, ak.b.mul(&xu).add(&ak.a.mul(&xv)));
    }

}
