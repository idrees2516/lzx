//! Guest A — perp CLOB kernel for the lzx zkVM.
//!
//! A dependency-free (`core` only), integer-only port of the
//! `perp-options-clob` matching / positions / fees / margin / funding core,
//! following `lzx/docs/CLOB_WORKLOAD_RESEARCH.md` (§2 matching, §3.1-3.3
//! positions & funding, §5 margin scan, §8 port plan, staged Guest A).
//!
//! Proving-profile constraints baked into the design (see the task spec):
//! * every in-memory field is `u64`/`i64` (R1: only `ld`/`sd` ever emitted);
//! * no `u128`, no floats, no atomics, no inline asm (R2);
//! * only unsigned `/` and `%`; signed semantics via manual sign handling (R3);
//! * all loops bounded by const caps; no dynamic memory (R4/R5);
//! * every mul-then-div site is guarded by validated input caps so the true
//!   (un-wrapped) product stays < 2^64 (worst composite bound documented at
//!   each cap below).
//!
//! The kernel is deterministic and pure w.r.t. its arguments, so the host
//! test harness and the RISC-V guest run *bit-identical* code.

#![cfg_attr(not(test), no_std)]

// ---------------------------------------------------------------------------
// Constants and capacities
// ---------------------------------------------------------------------------

/// INPUT region size in u64 words (0x1C00-0x1DFF).
pub const INPUT_WORDS: usize = 64;
/// OUTPUT region size in u64 words (0x1E00-0x1FFF).
pub const OUTPUT_WORDS: usize = 64;
/// Input magic (task spec literal `0x4C4F4243`).
pub const INPUT_MAGIC: u64 = 0x4C4F4243;
/// Output magic (task spec literal `0x4F555455`).
pub const OUTPUT_MAGIC: u64 = 0x4F555455;
/// Fixed-point scale for prices / margins / ratios (1e9).
pub const FP_SCALE: u64 = 1_000_000_000;

/// Max batch orders. Spec says "<= 16", but the 64-word INPUT region caps the
/// sequential layout (2 + 4*n + 6 + 6 + 4 <= 64) at n = 11. Validated.
pub const MAX_ORDERS: usize = 11;
/// Max resting orders per side (spec: "max 32 resting" total -> 16/side).
pub const MAX_RESTING: usize = 16;
/// Max trades out. Spec says "<= 16", but the 64-word OUTPUT region caps the
/// layout (2 + 4*t + 16 + 2 <= 64) at t = 11. Validated; matching stops early.
pub const MAX_TRADES: usize = 11;
/// Traders (spec: 4).
pub const NUM_TRADERS: usize = 4;
/// Margin scenarios (spec: 6 spots).
pub const NUM_SCENARIOS: usize = 6;

/// Taker fee in basis points of notional (house rounds UP, `num.rs` house-up).
pub const TAKER_FEE_BPS: u64 = 5;
/// Maker fee in basis points of notional (user rounds DOWN, `num.rs`
/// user-down). Fee model: both sides pay; the taker pays more.
pub const MAKER_FEE_BPS: u64 = 2;

// Input validation caps: chosen so every mul-then-div intermediate in the
// kernel stays < 2^64 (worst-case composite bounds):
//   notional (fees)        qty*lot*price*bps+1e4   <= 1000*100*1e12*5+1e4      ~ 2^59
//   VWAP numerator         |old|*entry + qty*price <= 11000*1e12 + 1000*1e12   ~ 2^54
//   realized PnL (i64)     diff*reduced*lot        <= 1.2e12*11000*100        ~ 2^60
//   scenario PnL (i64)     diff*lot*size           <= 1.2e12*100*11000        ~ 2^57
//   requirement            (maint*spot/1e9)*base   <= 6e9 * 1.1e6             ~ 2^53
//   funding pre-product    |rate|*mark*lot + 1e9   <= 1e6*1.4e11*100 + 1e9    ~ 2^63.6 (< 2^64)
pub const CAP_PRICE: u64 = 1_000_000_000_000; // <= $1000 in 1e9 fp
pub const CAP_QTY: u64 = 1_000; // lots per order
pub const CAP_LOT: u64 = 100;
pub const CAP_TICK: u64 = 1_000_000_000;
pub const CAP_INIT_FP: u64 = 100_000_000; // <= 10% in 1e9 fp
pub const CAP_MAINT_FP: u64 = 40_000_000; // <= 4% in 1e9 fp
pub const CAP_RATE_FP: i64 = 1_000_000; // |funding rate| <= 0.001 in 1e9 fp
pub const CAP_MARK: u64 = 140_000_000_000; // <= $140 in 1e9 fp
pub const CAP_SPOT: u64 = 150_000_000_000; // <= $150 in 1e9 fp
pub const CAP_MARGIN: u64 = 1_000_000_000_000_000; // <= $1e6 in 1e9 fp

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// One resting order (price-time priority slot). A fully-filled slot stays in
/// place as a zero-qty tombstone — the flat analogue of zkLighter's channel
/// tombstones (`orderbook/src/lib.rs:93-97`), preserving arrival order.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Resting {
    pub price_fp: u64,
    pub qty: u64,
    pub trader: u64,
    pub seq: u64,
}

/// Per-trader perp position + margin, ported from `margin/src/account.rs:17-28`
/// (signed lots, VWAP entry, realized PnL credited to margin/cash).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Trader {
    pub size: i64,
    pub entry_fp: u64,
    pub margin_fp: u64,
    pub realized_fp: i64,
    pub health: u64,
}

/// One emitted trade (fill) — always executed at the maker's price.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TradeOut {
    pub price_fp: u64,
    pub qty: u64,
    pub maker: u64,
    pub taker: u64,
}

/// The whole guest state (all `u64`/`i64` fields; lives in .bss, zero-init).
///
/// The resting book is one flat array: bids occupy `[0..MAX_RESTING)`, asks
/// `[MAX_RESTING..2*MAX_RESTING)` — a single code path serves both sides
/// (the side offset replaces the two duplicated bid/ask code paths).
#[repr(C)]
pub struct ClobState {
    pub resting: [Resting; 2 * MAX_RESTING],
    pub n_bids: u64,
    pub n_asks: u64,
    pub traders: [Trader; NUM_TRADERS],
    pub trades: [TradeOut; MAX_TRADES],
    pub n_trades: u64,
    pub tick_size: u64,
    pub lot_size: u64,
    pub maint_fp: u64,
    pub mark_fp: u64,
    pub rate_fp: i64,
    pub fee_sum: u64,
    pub funding_sum: u64,
    pub digest: u64,
    pub conservation: u64,
}

impl ClobState {
    /// All-zero state (matches the .bss zero-fill the flat loader provides).
    pub const fn zeros() -> Self {
        ClobState {
            resting: [Resting { price_fp: 0, qty: 0, trader: 0, seq: 0 }; 2 * MAX_RESTING],
            n_bids: 0,
            n_asks: 0,
            traders: [Trader { size: 0, entry_fp: 0, margin_fp: 0, realized_fp: 0, health: 0 };
                NUM_TRADERS],
            trades: [TradeOut { price_fp: 0, qty: 0, maker: 0, taker: 0 }; MAX_TRADES],
            n_trades: 0,
            tick_size: 0,
            lot_size: 0,
            maint_fp: 0,
            mark_fp: 0,
            rate_fp: 0,
            fee_sum: 0,
            funding_sum: 0,
            digest: 0,
            conservation: 0,
        }
    }
}

/// Rough instrumentation counters (also incremented by the guest so the
/// executed image self-reports its loop trip counts).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Stats {
    pub orders: u64,
    pub scans: u64,
    pub fills: u64,
    pub rest: u64,
    pub scen: u64,
}

impl Stats {
    pub const fn zeros() -> Self {
        Stats { orders: 0, scans: 0, fills: 0, rest: 0, scen: 0 }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic helpers (all unsigned; explicit rounding — `core/src/num.rs`
// house-up / user-down discipline)
// ---------------------------------------------------------------------------

/// `(a * b) / d` on u64 with explicit rounding. `round_up` = ceil (house
/// receives), else floor (user receives). Operands are capped by input
/// validation so the true product is < 2^64 (no wrap, exact division).
#[inline(never)]
fn mul_div(a: u64, b: u64, d: u64, round_up: bool) -> u64 {
    let p = a.wrapping_mul(b);
    if round_up {
        p.wrapping_add(d - 1) / d
    } else {
        p / d
    }
}

// ---------------------------------------------------------------------------
// Positions (port of `margin/src/account.rs:59-119` apply_fill)
// ---------------------------------------------------------------------------

/// Apply one fill to a trader's position.
///
/// * flat -> open at fill price (no cash movement);
/// * same direction -> VWAP extension `avg = (|old|*avg + q*p)/(|old|+q)`
///   (integer division, as in the source);
/// * opposite -> reduction (possibly with flip): realized PnL
///   `(fill - entry) * reduced * lot` for longs, negated for shorts, credited
///   to margin; flat zeroes the entry, a flip re-marks the entry at the fill
///   price for the new-side remainder.
#[inline(never)]
fn apply_fill(t: &mut Trader, is_buy: bool, qty: u64, price_fp: u64, lot: u64) {
    let old = t.size;
    let delta: i64 = if is_buy { qty as i64 } else { -(qty as i64) };
    if old == 0 {
        t.size = delta;
        t.entry_fp = price_fp;
    } else if (old > 0) == (delta > 0) {
        // extension: VWAP-on-extension, integer div
        let a = old
            .unsigned_abs()
            .wrapping_mul(t.entry_fp)
            .wrapping_add(qty.wrapping_mul(price_fp));
        let b = old.unsigned_abs() + qty;
        t.entry_fp = a / b;
        t.size = old.wrapping_add(delta);
    } else {
        // reduction, possibly with flip
        let old_abs = old.unsigned_abs();
        let reduced = if old_abs < qty { old_abs } else { qty };
        let diff = (price_fp as i64).wrapping_sub(t.entry_fp as i64);
        let mut real = diff.wrapping_mul(reduced as i64).wrapping_mul(lot as i64);
        if old < 0 {
            real = real.wrapping_neg(); // short: (entry - fill) * reduced
        }
        t.realized_fp = t.realized_fp.wrapping_add(real);
        t.margin_fp = t.margin_fp.wrapping_add(real as u64);
        let new = old.wrapping_add(delta);
        if new == 0 {
            t.entry_fp = 0;
        } else if qty > old_abs {
            // flipped through zero: new-side portion carried at fill price
            t.entry_fp = price_fp;
        }
        t.size = new;
    }
}

// ---------------------------------------------------------------------------
// Matching (port of `orderbook/src/lib.rs:802-921` match_taker walk)
// ---------------------------------------------------------------------------

/// Find the best crossing resting order on the opposite side.
///
/// Returns the array index of the best maker, or -1 if none. "Best" is price
/// priority (highest bid / lowest ask) with time priority (arrival order)
/// among equal prices — scanning the arrival-ordered array and only replacing
/// on a *strictly* better price keeps the earliest equal-price maker, which is
/// exactly the level/FIFO walk of `match_taker`. Fills always happen at the
/// maker's price (`lib.rs:49-50`). Self-trades (maker == taker) are skipped —
/// the simplified STP port (deviation documented in the port notes).
#[inline(never)]
fn find_best(st: &ClobState, stats: &mut Stats, taker_is_ask: bool, limit: u64, taker: u64) -> i64 {
    // an ask taker walks the resting bids (offset 0); a bid taker walks the
    // resting asks (offset MAX_RESTING)
    let (off, n) = if taker_is_ask {
        (0, st.n_bids)
    } else {
        (MAX_RESTING as u64, st.n_asks)
    };
    let mut best: i64 = -1;
    let mut best_p: u64 = 0;
    let mut i: u64 = 0;
    while i < n {
        stats.scans += 1;
        let r = st.resting[(off + i) as usize];
        if r.qty != 0 && r.trader != taker {
            // crossing test: taker sell takes bids >= limit, buy takes asks <= limit
            let cross = if taker_is_ask { r.price_fp >= limit } else { r.price_fp <= limit };
            if cross {
                let better = if best < 0 {
                    true
                } else if taker_is_ask {
                    r.price_fp > best_p
                } else {
                    r.price_fp < best_p
                };
                if better {
                    best = i as i64;
                    best_p = r.price_fp;
                }
            }
        }
        i += 1;
    }
    best
}

/// Apply one fill: trade record, maker reduction (tombstone at zero),
/// both position updates, taker fee (ceil) and maker fee (floor).
#[inline(never)]
fn do_fill(
    st: &mut ClobState,
    stats: &mut Stats,
    taker_is_ask: bool,
    fq: u64,
    idx: usize,
    taker: u64,
) {
    // maker order (opposite side of the taker) — by-value copy of the slot
    let off = if taker_is_ask { 0 } else { MAX_RESTING };
    let m = st.resting[off + idx];
    let mprice = m.price_fp;
    let mtr = m.trader;
    // 1) trade record (always at the maker's price)
    if st.n_trades < MAX_TRADES as u64 {
        st.trades[st.n_trades as usize] =
            TradeOut { price_fp: mprice, qty: fq, maker: mtr, taker };
        st.n_trades += 1;
    }
    // 2) reduce the maker's resting qty (zero qty = tombstone)
    st.resting[off + idx].qty = m.qty - fq;
    // 3) positions: maker takes the opposite side of the taker (a bid taker
    // is buying, so the maker is selling; an ask taker is selling, so the
    // maker is buying)
    let lot = st.lot_size;
    apply_fill(&mut st.traders[mtr as usize], taker_is_ask, fq, mprice, lot);
    apply_fill(&mut st.traders[taker as usize], !taker_is_ask, fq, mprice, lot);
    // 4) fees: both sides pay — taker ceil(bps * notional), maker floor
    let notional = fq.wrapping_mul(lot).wrapping_mul(mprice);
    let tfee = mul_div(notional, TAKER_FEE_BPS, 10_000, true);
    let mfee = mul_div(notional, MAKER_FEE_BPS, 10_000, false);
    st.traders[taker as usize].margin_fp =
        st.traders[taker as usize].margin_fp.wrapping_sub(tfee);
    st.traders[mtr as usize].margin_fp =
        st.traders[mtr as usize].margin_fp.wrapping_sub(mfee);
    st.fee_sum = st.fee_sum.wrapping_add(tfee).wrapping_add(mfee);
    stats.fills += 1;
}

/// Process one batch order: validate, clamp (reduce-only), match to the
/// opposite side (IOC for market orders; GTC limit orders rest the remainder),
/// honour post-only (skip if it would cross, rest otherwise) and self-trade
/// skipping.
#[inline(never)]
fn process_order(st: &mut ClobState, stats: &mut Stats, input: &[u64], k: usize) {
    let base = 2 + 4 * k;
    let flags = input[base];
    let price = input[base + 1];
    let qty_in = input[base + 2];
    let trader = input[base + 3];
    let is_ask = flags & 1 != 0;
    let is_market = flags & 2 != 0;
    let post_only = flags & 4 != 0;
    let reduce_only = flags & 8 != 0;
    let seq = flags >> 8;

    stats.orders += 1;
    if trader >= NUM_TRADERS as u64 || qty_in == 0 || qty_in > CAP_QTY {
        return;
    }
    // limit price for the crossing test: market orders cross everything
    let limit = if is_market {
        if is_ask {
            0
        } else {
            u64::MAX
        }
    } else {
        if price == 0 || price > CAP_PRICE {
            return;
        }
        // tick-grid validation (limit orders only)
        if price % st.tick_size != 0 {
            return;
        }
        price
    };

    // reduce-only clamps to the reducible size (`risk/src/pretrade.rs:315-325`);
    // clamped-to-zero orders are skipped, and reduce-only never rests
    // (deviation: simplified to IOC semantics — documented).
    let mut qty = qty_in;
    if reduce_only {
        let size = st.traders[trader as usize].size;
        let cap = if is_ask {
            if size > 0 { size as u64 } else { 0 }
        } else if size < 0 {
            size.unsigned_abs()
        } else {
            0
        };
        if qty > cap {
            qty = cap;
        }
    }
    if qty == 0 {
        return;
    }

    // post-only: reject (skip entirely) if the order would cross
    if post_only && find_best(st, stats, is_ask, limit, trader) >= 0 {
        return;
    }

    // taker walk: fill against best crossing makers until done / book drained
    let mut remaining = qty;
    let mk_off = if is_ask { 0 } else { MAX_RESTING };
    while remaining > 0 {
        let idx = find_best(st, stats, is_ask, limit, trader);
        if idx < 0 {
            break;
        }
        let i = idx as usize;
        let maker_qty = st.resting[mk_off + i].qty;
        let fq = if remaining < maker_qty { remaining } else { maker_qty };
        if fq == 0 {
            break;
        }
        do_fill(st, stats, is_ask, fq, i, trader);
        remaining -= fq;
        // trade buffer full: stop matching (remainder handled normally)
        if st.n_trades >= MAX_TRADES as u64 {
            break;
        }
    }

    // GTC remainder rests on its own side (arrival-order append). Market and
    // reduce-only remainders are discarded (IOC). Post-only orders that did
    // not cross rest here too — that is the point of post-only.
    if remaining > 0 && !is_market && !reduce_only {
        let r = Resting { price_fp: limit, qty: remaining, trader, seq };
        if is_ask {
            if st.n_asks < MAX_RESTING as u64 {
                st.resting[MAX_RESTING + st.n_asks as usize] = r;
                st.n_asks += 1;
                stats.rest += 1;
            }
        } else if st.n_bids < MAX_RESTING as u64 {
            st.resting[st.n_bids as usize] = r;
            st.n_bids += 1;
            stats.rest += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Margin scenarios + funding + commitments
// ---------------------------------------------------------------------------

/// Six-scenario maintenance margin scan (the perp leg of the SFPM grid,
/// `margin/src/portfolio.rs:319-346`, spot-shock scan without options legs).
/// `pnl = size * lot * (spot - entry)` (i64 exact), requirement =
/// `floor(maint_fp * spot / 1e9) * |size| * lot`; the trader is flagged
/// unhealthy if `margin + pnl < requirement` in ANY scenario.
#[inline(never)]
fn margin_scan(st: &mut ClobState, stats: &mut Stats, spots: &[u64]) {
    for t in 0..NUM_TRADERS {
        let size = st.traders[t].size;
        let entry = st.traders[t].entry_fp;
        let margin = st.traders[t].margin_fp;
        let base = size.unsigned_abs().wrapping_mul(st.lot_size);
        let mut unhealthy: u64 = 0;
        for s in 0..NUM_SCENARIOS {
            let spot = spots[s];
            let diff = (spot as i64).wrapping_sub(entry as i64);
            let pnl = diff.wrapping_mul(st.lot_size as i64).wrapping_mul(size);
            let equity = (margin as i64).wrapping_add(pnl);
            let per_base = mul_div(st.maint_fp, spot, FP_SCALE, false);
            let req = per_base.wrapping_mul(base);
            if equity < req as i64 {
                unhealthy = 1;
            }
            stats.scen += 1;
        }
        st.traders[t].health = unhealthy;
    }
}

/// One funding step (`economics/src/funding.rs:108-129`): per-lot payment
/// magnitude `ceil(|rate_fp| * mark_fp * lot / 1e9)` (payer's expense —
/// magnitude rounds up, matching the source's per-lot ceil discipline),
/// `delta = -(size * sign(rate) * per_lot)` applied to margins. Zero-sum
/// whenever positions net to zero (`sweep.rs:526-538`).
#[inline(never)]
fn funding(st: &mut ClobState) {
    let rate = st.rate_fp;
    if rate == 0 {
        return;
    }
    // raw pre-product |rate| * mark (<= 1e6 * 1.4e11 = 1.4e17), then scale by
    // lot in the mul-div (<= 1.4e17 * 100 + 1e9 < 2^64 — no wrap under caps)
    let x = rate.unsigned_abs().wrapping_mul(st.mark_fp);
    let per_lot = mul_div(x, st.lot_size, FP_SCALE, true);
    if per_lot == 0 {
        return;
    }
    for t in 0..NUM_TRADERS {
        let size = st.traders[t].size;
        if size != 0 {
            let mut pay = (size as i64).wrapping_mul(per_lot as i64);
            if rate < 0 {
                pay = pay.wrapping_neg();
            }
            let delta = pay.wrapping_neg();
            st.traders[t].margin_fp = st.traders[t].margin_fp.wrapping_add(delta as u64);
            st.funding_sum = st.funding_sum.wrapping_add(delta as u64);
        }
    }
}

/// Book digest: `sum_{i} (i+1) * price * qty` (u64 wrapping, linear sum) over
/// every resting slot in array order — bids first, then asks, index continuing
/// across sides; zero-qty tombstone slots contribute 0 but consume an index.
#[inline(never)]
fn book_digest(st: &mut ClobState) {
    let nb = st.n_bids;
    let total = st.n_bids + st.n_asks;
    let mut sum: u64 = 0;
    let mut g: u64 = 0;
    while g < total {
        // bids occupy slots [0, nb) in arrival order; asks occupy
        // [MAX_RESTING, MAX_RESTING + n_asks) — the global digest index runs
        // over bids first, then asks
        let slot = if g < nb { g } else { MAX_RESTING as u64 + g - nb };
        let r = st.resting[slot as usize];
        sum = sum
            .wrapping_add((g + 1).wrapping_mul(r.price_fp).wrapping_mul(r.qty));
        g += 1;
    }
    st.digest = sum;
}

/// Conservation invariant. With both fees deducted from margins, realized PnL
/// credited to margins (doc `account.rs` variation-margin convention) and
/// funding zero-sum, the exact invariant is
/// `sum(margins) + fees - realized - funding == sum(initial margins)`,
/// all mod 2^64 — the verifier compares this word against the input-region
/// margin sum. (Deviation from the task's literal `sum(margins) + fees`:
/// that expression is not invariant once realized PnL lands in margins.)
#[inline(never)]
fn conservation(st: &mut ClobState) {
    let mut m: u64 = 0;
    let mut real: u64 = 0;
    for t in 0..NUM_TRADERS {
        m = m.wrapping_add(st.traders[t].margin_fp);
        real = real.wrapping_add(st.traders[t].realized_fp as u64);
    }
    st.conservation = m
        .wrapping_add(st.fee_sum)
        .wrapping_sub(real)
        .wrapping_sub(st.funding_sum);
}

// ---------------------------------------------------------------------------
// Output marshalling
// ---------------------------------------------------------------------------

/// Write the OUTPUT region words: magic, num_trades, trades (4 words each),
/// traders (4 words each), book_digest, conservation; tail stays zero.
#[inline(never)]
fn write_output(st: &ClobState, output: &mut [u64]) {
    output[0] = OUTPUT_MAGIC;
    output[1] = st.n_trades;
    let mut w = 2;
    let mut i: u64 = 0;
    while i < st.n_trades {
        let tr = st.trades[i as usize];
        output[w] = tr.price_fp;
        output[w + 1] = tr.qty;
        output[w + 2] = tr.maker;
        output[w + 3] = tr.taker;
        w += 4;
        i += 1;
    }
    for t in 0..NUM_TRADERS {
        output[w] = st.traders[t].size as u64;
        output[w + 1] = st.traders[t].margin_fp;
        output[w + 2] = st.traders[t].health;
        output[w + 3] = st.traders[t].realized_fp as u64;
        w += 4;
    }
    output[w] = st.digest;
    output[w + 1] = st.conservation;
}

// ---------------------------------------------------------------------------
// Kernel entry
// ---------------------------------------------------------------------------

/// Run the whole Guest-A batch:
/// parse + validate INPUT -> match all orders -> margin scan -> funding ->
/// book digest -> conservation -> OUTPUT.
///
/// On any invalid input (bad magic / out-of-cap instrument words) the output
/// stays all-zero (documented error behaviour).
pub fn run_kernel(input: &[u64], output: &mut [u64], st: &mut ClobState, stats: &mut Stats) {
    // zero the output region (fill sequentially, then pad)
    for i in 0..OUTPUT_WORDS {
        output[i] = 0;
    }
    if input.len() < INPUT_WORDS || input[0] != INPUT_MAGIC {
        return;
    }
    let n = if input[1] > MAX_ORDERS as u64 { MAX_ORDERS as u64 } else { input[1] };
    let c = (2 + 4 * n) as usize;

    let tick = input[c];
    let lot = input[c + 1];
    let init = input[c + 2];
    let maint = input[c + 3];
    let rate = input[c + 4] as i64;
    let mark = input[c + 5];
    let spots = [
        input[c + 6],
        input[c + 7],
        input[c + 8],
        input[c + 9],
        input[c + 10],
        input[c + 11],
    ];
    for t in 0..NUM_TRADERS {
        st.traders[t].margin_fp = input[c + 12 + t];
    }

    // instrument validation (caps keep every product site < 2^64)
    if lot == 0
        || lot > CAP_LOT
        || tick == 0
        || tick > CAP_TICK
        || init > CAP_INIT_FP
        || maint > CAP_MAINT_FP
        || mark > CAP_MARK
        || rate > CAP_RATE_FP
        || rate < -CAP_RATE_FP
    {
        return;
    }
    for i in 0..NUM_SCENARIOS {
        if spots[i] > CAP_SPOT {
            return;
        }
    }
    for t in 0..NUM_TRADERS {
        if st.traders[t].margin_fp > CAP_MARGIN {
            return;
        }
    }

    st.tick_size = tick;
    st.lot_size = lot;
    st.maint_fp = maint;
    st.mark_fp = mark;
    st.rate_fp = rate;

    // order batch
    for k in 0..n as usize {
        process_order(st, stats, input, k);
    }

    // margin health (pre-funding, per the task spec ordering), then funding
    margin_scan(st, stats, &spots);
    funding(st);

    // commitments
    book_digest(st);
    conservation(st);

    write_output(st, output);
}

// ---------------------------------------------------------------------------
// Host-side unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_div_rounding() {
        assert_eq!(mul_div(1, 1, 3, true), 1); // ceil(1/3)
        assert_eq!(mul_div(1, 1, 3, false), 0); // floor(1/3)
        assert_eq!(mul_div(3, 3, 3, true), 3);
        assert_eq!(mul_div(7, 1, 2, true), 4);
        assert_eq!(mul_div(7, 1, 2, false), 3);
        // large operands stay exact (product < 2^63): 4e6 * 1.5e11 = 6e17
        assert_eq!(mul_div(4_000_000, 150_000_000_000, 1_000_000_000, false), 600_000_000);
    }

    #[test]
    fn open_and_vwap_extension() {
        let mut t = Trader { size: 0, entry_fp: 0, margin_fp: 1_000_000_000, realized_fp: 0, health: 0 };
        apply_fill(&mut t, true, 4, 101_000_000_000, 1);
        assert_eq!(t.size, 4);
        assert_eq!(t.entry_fp, 101_000_000_000);
        assert_eq!(t.realized_fp, 0);
        apply_fill(&mut t, true, 2, 98_000_000_000, 1);
        // VWAP = (4*101e9 + 2*98e9)/6 = 600e9/6 = 100e9
        assert_eq!(t.size, 6);
        assert_eq!(t.entry_fp, 100_000_000_000);
        assert_eq!(t.realized_fp, 0);
        assert_eq!(t.margin_fp, 1_000_000_000);
    }

    #[test]
    fn reduction_long_and_short() {
        let mut t = Trader { size: 6, entry_fp: 100_000_000_000, margin_fp: 1_000_000_000, realized_fp: 0, health: 0 };
        apply_fill(&mut t, false, 2, 102_000_000_000, 1);
        assert_eq!(t.size, 4);
        assert_eq!(t.realized_fp, 4_000_000_000); // (102-100)*2
        assert_eq!(t.margin_fp, 5_000_000_000); // 1e9 seed + realized
        assert_eq!(t.entry_fp, 100_000_000_000);
        // short side: entry 100, buy back at 99 -> +1 per lot
        let mut s = Trader { size: -3, entry_fp: 100_000_000_000, margin_fp: 1_000_000_000, realized_fp: 0, health: 0 };
        apply_fill(&mut s, true, 3, 99_000_000_000, 1);
        assert_eq!(s.size, 0);
        assert_eq!(s.entry_fp, 0);
        assert_eq!(s.realized_fp, 3_000_000_000);
        assert_eq!(s.margin_fp, 4_000_000_000); // 1e9 seed + realized
    }

    #[test]
    fn reduction_with_flip() {
        // long 2 @100, sell 5 @110 -> short 3 @110, realized (110-100)*2
        let mut t = Trader { size: 2, entry_fp: 100_000_000_000, margin_fp: 0, realized_fp: 0, health: 0 };
        apply_fill(&mut t, false, 5, 110_000_000_000, 1);
        assert_eq!(t.size, -3);
        assert_eq!(t.realized_fp, 20_000_000_000);
        assert_eq!(t.entry_fp, 110_000_000_000); // flip re-marks entry
    }

    #[test]
    fn lot_size_scales_realized() {
        let mut t = Trader { size: -10, entry_fp: 100_000_000_000, margin_fp: 0, realized_fp: 0, health: 0 };
        apply_fill(&mut t, true, 4, 90_000_000_000, 10);
        // (100-90)*4 lots * 10 base/lot = 400
        assert_eq!(t.realized_fp, 400_000_000_000);
        assert_eq!(t.size, -6);
    }

    #[test]
    fn funding_is_zero_sum_and_ceil() {
        let mut st = ClobState::zeros();
        st.rate_fp = 100_000; // 0.0001
        st.mark_fp = 100_000_000_000;
        st.lot_size = 1;
        st.traders[0].size = 4;
        st.traders[1].size = -9;
        st.traders[2].size = 5;
        funding(&mut st);
        let per_lot = 10_000_000u64; // ceil(1e5*1e11*1/1e9) exact
        assert_eq!(st.traders[0].margin_fp, (-(4 * per_lot as i64)) as u64);
        assert_eq!(st.traders[1].margin_fp, 9 * per_lot);
        assert_eq!(st.traders[2].margin_fp, (-(5 * per_lot as i64)) as u64);
        // zero-sum: sum of deltas == 0
        assert_eq!(st.funding_sum, 0);
        // ceil: rate = 1 (1e-9), mark = 1e9 -> per_lot = ceil(1e9/1e9) = 1
        let mut st2 = ClobState::zeros();
        st2.rate_fp = 1;
        st2.mark_fp = 1_000_000_000;
        st2.lot_size = 1;
        st2.traders[0].size = 1;
        funding(&mut st2);
        assert_eq!(st2.traders[0].margin_fp.wrapping_neg(), 1);
        // floor would give 0: rate = 1, mark = 999_999_999 -> ceil(0.999..) = 1
        let mut st3 = ClobState::zeros();
        st3.rate_fp = 1;
        st3.mark_fp = 999_999_999;
        st3.lot_size = 1;
        st3.traders[0].size = 1;
        funding(&mut st3);
        assert_eq!(st3.traders[0].margin_fp.wrapping_neg(), 1);
    }

    #[test]
    fn funding_scales_with_lot() {
        // per-lot payment must include the lot multiplier: rate 1e-4,
        // mark $100, lot 10 -> $0.01 * 10 = $0.10 per lot
        let mut st = ClobState::zeros();
        st.rate_fp = 100_000;
        st.mark_fp = 100_000_000_000;
        st.lot_size = 10;
        st.traders[0].size = 1;
        st.traders[1].size = -1;
        funding(&mut st);
        assert_eq!(st.traders[0].margin_fp.wrapping_neg(), 100_000_000);
        assert_eq!(st.traders[1].margin_fp, 100_000_000);
        assert_eq!(st.funding_sum, 0);
    }

    #[test]
    fn post_only_order_rests_when_not_crossing() {
        // a post-only ask above the best bid must rest on the book
        let mut st = ClobState::zeros();
        st.tick_size = 500_000_000;
        st.lot_size = 1;
        let mut stats = Stats::zeros();
        // seed: T1 bid 99 x4
        let mut input = [0u64; INPUT_WORDS];
        input[0] = INPUT_MAGIC;
        input[1] = 2;
        // o1: T0 ASK L 102.5 x15 post_only (no cross -> must rest)
        input[2] = 1 | 4 | (1 << 8);
        input[3] = 102_500_000_000;
        input[4] = 15;
        input[5] = 0;
        // o2: T0 ASK L 98.5 x2 post_only (crosses the o1 ask? no bids exist ->
        // best bid none; asks never cross asks) -> rests too
        input[6] = 1 | 4 | (2 << 8);
        input[7] = 98_500_000_000;
        input[8] = 2;
        input[9] = 0;
        let c = 2 + 4 * 2;
        input[c] = 500_000_000; // tick
        input[c + 1] = 1; // lot
        input[c + 2] = 50_000_000;
        input[c + 3] = 37_500_000;
        input[c + 4] = 0;
        input[c + 5] = 100_000_000_000;
        let mut output = [0u64; OUTPUT_WORDS];
        run_kernel(&input, &mut output, &mut st, &mut stats);
        // both post-only asks rested (nothing to cross against)
        assert_eq!(st.n_asks, 2);
        assert_eq!(st.resting[MAX_RESTING].qty, 15);
        assert_eq!(st.resting[MAX_RESTING + 1].qty, 2);
        assert_eq!(stats.rest, 2);
    }

    #[test]
    fn self_trade_is_skipped() {
        // T0's market buy must skip T0's own resting ask and fill the other
        // maker instead (simplified STP)
        let mut st = ClobState::zeros();
        st.tick_size = 500_000_000;
        st.lot_size = 1;
        let mut stats = Stats::zeros();
        let mut input = [0u64; INPUT_WORDS];
        input[0] = INPUT_MAGIC;
        input[1] = 3;
        // o1: T0 ASK L 101 x5 (own)
        input[2] = 1 | (1 << 8);
        input[3] = 101_000_000_000;
        input[4] = 5;
        input[5] = 0;
        // o2: T1 ASK L 102 x5 (other)
        input[6] = 1 | (2 << 8);
        input[7] = 102_000_000_000;
        input[8] = 5;
        input[9] = 1;
        // o3: T0 BID MKT x3 -> skips own 101-ask, fills 3 @102 from T1
        input[10] = 2 | (3 << 8);
        input[11] = 0;
        input[12] = 3;
        input[13] = 0;
        let c = 2 + 4 * 3;
        input[c] = 500_000_000;
        input[c + 1] = 1;
        input[c + 2] = 50_000_000;
        input[c + 3] = 37_500_000;
        input[c + 4] = 0;
        input[c + 5] = 100_000_000_000;
        let mut output = [0u64; OUTPUT_WORDS];
        run_kernel(&input, &mut output, &mut st, &mut stats);
        assert_eq!(st.n_trades, 1);
        assert_eq!(st.trades[0].price_fp, 102_000_000_000);
        assert_eq!(st.trades[0].qty, 3);
        assert_eq!(st.trades[0].maker, 1);
        assert_eq!(st.trades[0].taker, 0);
        // own ask untouched
        assert_eq!(st.resting[MAX_RESTING].qty, 5);
        assert_eq!(st.resting[MAX_RESTING + 1].qty, 2);
        assert_eq!(st.traders[0].size, 3);
        assert_eq!(st.traders[1].size, -3);
    }

    #[test]
    fn reduce_only_clamped_to_zero_is_skipped() {
        // flat trader sends a reduce-only order -> clamped to 0 -> skipped
        let mut st = ClobState::zeros();
        st.tick_size = 500_000_000;
        st.lot_size = 1;
        let mut stats = Stats::zeros();
        let mut input = [0u64; INPUT_WORDS];
        input[0] = INPUT_MAGIC;
        input[1] = 2;
        // o1: T1 ASK L 101 x5
        input[2] = 1 | (1 << 8);
        input[3] = 101_000_000_000;
        input[4] = 5;
        input[5] = 1;
        // o2: T0 BID L 101 x5 reduce_only (T0 flat -> skipped entirely)
        input[6] = 8 | (2 << 8);
        input[7] = 101_000_000_000;
        input[8] = 5;
        input[9] = 0;
        let c = 2 + 4 * 2;
        input[c] = 500_000_000;
        input[c + 1] = 1;
        input[c + 2] = 50_000_000;
        input[c + 3] = 37_500_000;
        input[c + 4] = 0;
        input[c + 5] = 100_000_000_000;
        let mut output = [0u64; OUTPUT_WORDS];
        run_kernel(&input, &mut output, &mut st, &mut stats);
        assert_eq!(st.n_trades, 0);
        assert_eq!(st.traders[0].size, 0);
        // reduce-only never rests
        assert_eq!(st.n_bids, 0);
        assert_eq!(st.n_asks, 1);
    }

    #[test]
    fn invalid_magic_leaves_output_zero() {
        let mut input = [0u64; 64];
        let mut output = [7u64; 64];
        let mut st = ClobState::zeros();
        let mut stats = Stats::zeros();
        input[0] = 0xDEAD;
        run_kernel(&input, &mut output, &mut st, &mut stats);
        assert!(output.iter().all(|&w| w == 0));
    }

    #[test]
    fn out_of_cap_instrument_rejected() {
        let mut input = [0u64; 64];
        input[0] = INPUT_MAGIC;
        input[1] = 0;
        input[2] = 1; // tick
        input[3] = 101; // lot > CAP_LOT -> reject
        let mut output = [0u64; 64];
        let mut st = ClobState::zeros();
        let mut stats = Stats::zeros();
        run_kernel(&input, &mut output, &mut st, &mut stats);
        assert!(output.iter().all(|&w| w == 0));
    }
}
