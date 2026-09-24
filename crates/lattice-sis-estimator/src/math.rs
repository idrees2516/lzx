//! Log-space helpers and `erf`/`erfc` (pure-std, double precision).
//!
//! `erf`/`erfc` via the incomplete-gamma route (`erf(x) = P(½, x²)`,
//! `erfc(x) = Q(½, x²)`): series for `x² < 1.5`, modified-Lentz continued
//! fraction above — ~1e-15 accuracy without external crates. This is the
//! only special-function dependency of the LGSA small-box probability.

/// log2(x) for positive x (non-finite/non-positive → -inf).
pub fn log2_positive(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        x.log2()
    }
}

/// log2 of a u128 modulus (exact enough: top-64-bit decomposition).
pub fn log2_u128(q: u128) -> f64 {
    if q == 0 {
        return f64::NEG_INFINITY;
    }
    let bits = 128 - q.leading_zeros();
    if bits <= 64 {
        (q as f64).log2()
    } else {
        let shift = bits - 64;
        let top = q >> shift;
        shift as f64 + (top as f64).log2()
    }
}

/// `(q - 1) / 2` as f64.
pub fn half_q(q: u128) -> f64 {
    (q - 1) as f64 / 2.0
}

/// Γ(½) = √π (f64 nearest).
const SQRT_PI: f64 = 1.772_453_850_905_516;
/// ln Γ(½) = ½ ln π (f64 nearest).
const LN_GAMMA_HALF: f64 = 0.572_364_942_924_700;

/// Error function (double precision, incomplete-gamma route).
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = x.signum();
    let ax = x.abs();
    if ax >= 6.0 {
        return sign; // erf saturates to ±1 well before 6.
    }
    let t = ax * ax;
    if t < 1.5 {
        // P(½, t) via the NR gser series:
        // P = e^{-t} t^{1/2}/Γ(½) · Σ_{n≥0} t^n/(½·(3/2)···(½+n)).
        let mut ap = 0.5f64;
        let mut sum = 1.0 / ap;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= t / ap;
            sum += del;
            if del <= sum * 1e-17 {
                break;
            }
        }
        let p = (-t).exp() * t.sqrt() / SQRT_PI * sum;
        sign * p.min(1.0)
    } else {
        let one_minus = 1.0 - erfc(ax);
        sign * one_minus.clamp(-1.0, 1.0)
    }
}

/// Complementary error function (double precision).
pub fn erfc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 1.224_744_871_391_589 {
        // t < 1.5: via erf.
        return 1.0 - erf(x);
    }
    if x >= 27.0 {
        // e^{-x²} underflows; erfc is below representable tails.
        return 0.0;
    }
    let t = x * x;
    // Q(½, t) via the modified-Lentz continued fraction.
    let a = 0.5f64;
    let tiny = 1e-300;
    let mut b = t + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    // Q = e^{-t + a ln t - ln Γ(a)} · h.
    (-t + a * t.ln() - LN_GAMMA_HALF).exp() * h
}

/// Upper numerical estimate of `log2(erf(2^log2_arg))` (ports the
/// lattice-estimator discipline: round the tail toward larger attack
/// probability before taking logarithms; the tiny-argument branch uses the
/// analytic bound `erf(x) ≤ 2x/√π` and never materializes underflow).
pub fn log2_erf_from_log2_arg(log2_arg: f64) -> f64 {
    // log2(2/√π) rounded upward:
    const LOG2_TWO_OVER_SQRT_PI_UPPER: f64 = 0.174_251_935_263_840_6;
    if log2_arg.is_nan() || log2_arg == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if log2_arg < -20.0 {
        // next_up equivalent at MSRV 1.75: bump by one ulp via the
        // successor through the next representable float.
        let v = log2_arg + LOG2_TWO_OVER_SQRT_PI_UPPER;
        return if v == 0.0 {
            f64::from_bits(1)
        } else {
            f64::from_bits(v.to_bits() + 1)
        };
    }
    let x = (log2_arg).exp2();
    let log_probability = if x < 1.0 {
        let mass = erf(x);
        log2_positive(mass.clamp(0.0, 1.0))
    } else {
        let tail = erfc(x).max(0.0);
        if tail <= 0.0 {
            0.0
        } else {
            (-tail).ln_1p() * std::f64::consts::LOG2_E
        }
    };
    {
        // next_up equivalent at MSRV 1.75 (see above).
        let v = log_probability.min(0.0);
        let bumped = if v == 0.0 {
            f64::from_bits(1)
        } else if v.is_sign_negative() {
            f64::from_bits(v.to_bits() - 1)
        } else {
            f64::from_bits(v.to_bits() + 1)
        };
        bumped.min(0.0)
    }
}

/// `log(1 - 2^log_x)` from `log_x = log2(x)` with `x ≤ 1`.
pub fn log1mexp2(log_x: f64) -> f64 {
    if log_x > 0.0 {
        return f64::NAN;
    }
    if log_x == 0.0 {
        return f64::NEG_INFINITY;
    }
    let x = log_x.exp2();
    (-x).ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        // Reference values (high-precision tables):
        // erf(0.1) ≈ 0.1124629160182849, erf(0.5) ≈ 0.5204998778130465,
        // erf(1.0) ≈ 0.8427007929497149, erf(1.5) ≈ 0.9661051464753108,
        // erf(2.0) ≈ 0.9953222650189527, erf(3.0) ≈ 0.9999779095030014.
        for (x, want) in [
            (0.1, 0.1124629160182849),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (1.5, 0.9661051464753108),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
        ] {
            let got = erf(x);
            assert!((got - want).abs() < 1e-14, "erf({x}) = {got}, want {want}");
        }
        // Odd symmetry and saturation.
        assert_eq!(erf(-0.5), -erf(0.5));
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(9.0), 1.0);
        assert_eq!(erf(-9.0), -1.0);
    }

    #[test]
    fn erfc_matches_known_values_and_complementarity() {
        for (x, want) in [
            (0.5, 0.4795001221869535),
            (1.0, 0.1572992070502851),
            (1.5, 0.03389485352468921),
            (2.0, 0.004677734981047266),
            (3.0, 2.209049699858544e-5),
            (5.0, 1.537459794428035e-12),
        ] {
            let got = erfc(x);
            assert!(
                (got - want).abs() < 1e-15 * want.max(1e-12) || (got - want).abs() / want < 1e-12,
                "erfc({x}) = {got}, want {want}"
            );
        }
        // Complementarity across the branch boundary.
        for x in [0.3, 0.9, 1.224744871391589, 1.3, 2.0, 4.0] {
            let sum = erf(x) + erfc(x);
            assert!((sum - 1.0).abs() < 1e-14, "x = {x}, sum = {sum}");
        }
        assert_eq!(erfc(0.0), 1.0);
        assert!(erfc(30.0) <= f64::MIN_POSITIVE * 10.0);
    }

    #[test]
    fn log2_erf_extremes() {
        // Preserves tails and extreme arguments (ports akita's tests).
        assert!(log2_erf_from_log2_arg(3.0) < 0.0);
        assert!(log2_erf_from_log2_arg(-10_000.0).is_finite());
        assert_eq!(log2_erf_from_log2_arg(1_024.0), 0.0);
        // Monotone in the small regime.
        let a = log2_erf_from_log2_arg(-5.0);
        let b = log2_erf_from_log2_arg(-4.0);
        assert!(a < b);
    }

    #[test]
    fn log_helpers() {
        assert_eq!(log2_positive(0.0), f64::NEG_INFINITY);
        assert_eq!(log2_positive(-1.0), f64::NEG_INFINITY);
        assert!((log2_positive(8.0) - 3.0).abs() < 1e-12);
        // u128 moduli.
        let lq = log2_u128(4_294_967_197u128); // 2^32 - 99
        assert!(lq < 32.0 && lq > 31.999_999, "log2(2^32-99) = {lq}");
        assert!(log2_u128(u128::MAX) > 127.9);
        assert_eq!(log2_u128(0), f64::NEG_INFINITY);
        // log1mexp2 sanity.
        assert!((log1mexp2(-1.0) - 0.5f64.ln()).abs() < 1e-12);
        assert!(log1mexp2(0.0).is_infinite());
    }
}
