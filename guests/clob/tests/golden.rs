//! Host golden-vector test: runs the SAME lib logic the RISC-V guest runs on
//! the fixed sample batch, asserts the hand-computed expected output, and
//! writes `input_sample.bin` / `expected_output.bin` (64 u64 words, LE) at the
//! crate root for the zkVM loader.

use clob_guest::{
    run_kernel, ClobState, Stats, INPUT_MAGIC, INPUT_WORDS, MAX_ORDERS, OUTPUT_MAGIC,
    OUTPUT_WORDS,
};
use std::fs;
use std::path::PathBuf;

fn fp(x: u64) -> u64 {
    x * 1_000_000_000
}

/// The deterministic sample batch (11 orders — exactly MAX_ORDERS).
///
/// flags: bit0 side (1 = ask/sell), bit1 is_market, bit2 post_only,
/// bit3 reduce_only, seq in bits 8+.
///
/// Hand-traced narrative (tick $0.50, lot 1 base, mark $100):
///  o1  T0 ASK L 101.0 x5      -> rests
///  o2  T1 BID L 99.0  x4      -> rests
///  o3  T3 ASK L 101.0 x3      -> rests (FIFO behind o1 at 101)
///  o4  T2 BID L 99.0  x2      -> rests (FIFO behind o2 at 99)
///  o5  T1 BID L 101.5 x6      -> crosses: fills o1 5@101, o3 1@101 (2 trades)
///  o6  T3 ASK L 98.5  x2 PO   -> would cross best bid 99 -> post-only reject
///  o7  T0 ASK L 100.5 x8      -> no cross -> rests
///  o8  T1 ASK MKT     x2      -> walks bids: skips own o2 (STP), fills o4
///                                2@99 (1 trade, o4 tombstoned)
///  o9  T1 BID L 102.0 x12     -> fills o7 8@100.5, o3 2@101; remainder 2
///                                rests as bid 102.0 (2 trades)
///  o10 T3 ASK L 102.5 x15 PO  -> 102.5 > best bid 102 -> no cross -> RESTS
///                                (post-only resting path)
///  o11 T0 BID L 103.0 x20 RO  -> clamped to |short 13| = 13, fills o10
///                                13@102.5, T0 back to flat; remainder not
///                                rested (reduce-only => IOC) (1 trade)
fn sample_input() -> [u64; INPUT_WORDS] {
    let mut w = [0u64; INPUT_WORDS];
    w[0] = INPUT_MAGIC;
    w[1] = 11;

    let orders: [(u64, u64, u64, u64); 11] = [
        (1 | (1 << 8), fp(101), 5, 0),                    // o1
        (0 | (2 << 8), fp(99), 4, 1),                     // o2
        (1 | (3 << 8), fp(101), 3, 3),                    // o3
        (0 | (4 << 8), fp(99), 2, 2),                     // o4
        (0 | (5 << 8), fp(101) + 500_000_000, 6, 1),      // o5 (crossing)
        (1 | 4 | (6 << 8), fp(98) + 500_000_000, 2, 3),   // o6 post_only (rejected)
        (1 | (7 << 8), fp(100) + 500_000_000, 8, 0),      // o7 (rests)
        (1 | 2 | (8 << 8), 0, 2, 1),                      // o8 ask market (self-skip)
        (0 | (9 << 8), fp(102), 12, 1),                   // o9 (GTC remainder rests)
        (1 | 4 | (10 << 8), fp(102) + 500_000_000, 15, 3),// o10 post_only (rests)
        (8 | (11 << 8), fp(103), 20, 0),                  // o11 reduce_only (clamped 20->13)
    ];
    for k in 0..11 {
        w[2 + 4 * k] = orders[k].0;
        w[3 + 4 * k] = orders[k].1;
        w[4 + 4 * k] = orders[k].2;
        w[5 + 4 * k] = orders[k].3;
    }
    let c = 2 + 4 * 11;
    w[c] = 500_000_000; // tick_size_fp  ($0.50)
    w[c + 1] = 1; // lot_size (1 base per lot)
    w[c + 2] = 50_000_000; // init_margin_fp  (5%, validated/reserved)
    w[c + 3] = 37_500_000; // maint_margin_fp (3.75%)
    w[c + 4] = 100_000; // funding_rate_fp  (0.0001)
    w[c + 5] = fp(100); // mark_price_fp    ($100)
    let spots = [fp(60), fp(80), fp(90), fp(110), fp(120), fp(140)];
    for i in 0..6 {
        w[c + 6 + i] = spots[i];
    }
    let margins = [2_000_000_000_000, 1_500_000_000_000, 1_000_000_000_000, 15_000_000_000];
    for t in 0..4 {
        w[c + 12 + t] = margins[t];
    }
    w
}

/// Hand-computed expected output for the sample (derivation in the task
/// report / port notes). Layout: magic, num_trades, 4 words/trade, 4
/// words/trader (size, margin_fp, health, realized_fp), book_digest,
/// conservation, pad.
///
/// Positions: T0 flat (realized -23.500000004), T1 +14 @ 100.714285713
/// (VWAP, realized -4.0), T2 +2 @ 99, T3 -16 @ 102.21875.
/// Fees (taker 5bps ceil / maker 2bps floor): T0 928.05e6, T1 905e6,
/// T2 39.6e6, T3 327.1e6 (total 2,199.75e6 fp).
/// Funding: per-lot ceil(1e-4 * $100 * 1) = $0.01; T1 pays 0.14, T2 0.02,
/// T3 receives 0.16 (zero-sum).
/// T3 unhealthy in the 110/120/140 spot scenarios (short 16 into a rising
/// spot with only $14.67e9 margin).
fn expected_output() -> [u64; OUTPUT_WORDS] {
    let mut e = [0u64; OUTPUT_WORDS];
    e[0] = OUTPUT_MAGIC;
    e[1] = 6;
    let trades: [(u64, u64, u64, u64); 6] = [
        (fp(101), 5, 0, 1),               // o5 t1: T1 takes 5 @101 from T0
        (fp(101), 1, 3, 1),               // o5 t2: T1 takes 1 @101 from T3
        (fp(99), 2, 2, 1),                // o8 t3: T1 sells 2 @99 to T2 (own bid skipped)
        (fp(100) + 500_000_000, 8, 0, 1), // o9 t4: T1 takes 8 @100.5 from T0
        (fp(101), 2, 3, 1),               // o9 t5: T1 takes 2 @101 from T3
        (fp(102) + 500_000_000, 13, 3, 0),// o11 t6: T0 buys 13 @102.5 from T3
    ];
    for (i, t) in trades.iter().enumerate() {
        e[2 + 4 * i] = t.0;
        e[3 + 4 * i] = t.1;
        e[4 + 4 * i] = t.2;
        e[5 + 4 * i] = t.3;
    }
    let w = 2 + 4 * 6;
    // T0: flat, realized -23.500000004, margin 2e12 - 928.05e6 - 23.5e9 = 1_975_571_949_996
    e[w] = 0;
    e[w + 1] = 1_975_571_949_996;
    e[w + 2] = 0;
    e[w + 3] = (-23_500_000_004i64) as u64;
    // T1: +14, realized -4.0, margin 1.5e12 - 905e6 - 4e9 - funding 140e6
    e[w + 4] = 14;
    e[w + 5] = 1_494_955_000_000;
    e[w + 6] = 0;
    e[w + 7] = (-4_000_000_000i64) as u64;
    // T2: +2 @99, margin 1e12 - 39.6e6 - funding 20e6
    e[w + 8] = 2;
    e[w + 9] = 999_940_400_000;
    e[w + 10] = 0;
    e[w + 11] = 0;
    // T3: -16 @102.21875, margin 15e9 - 327.1e6 + funding 160e6, unhealthy
    e[w + 12] = (-16i64) as u64;
    e[w + 13] = 14_832_900_000;
    e[w + 14] = 1;
    e[w + 15] = 0;
    // book digest: bids [o2 99x4, o4 99x0(tomb), o9rem 102x2],
    //               asks [o1 101x0, o3 101x0, o7 100.5x0, o10 102.5x2]
    //   = 1*99e9*4 + 3*102e9*2 + 7*102.5e9*2 = 2_443_000_000_000
    e[w + 16] = 2_443_000_000_000;
    // conservation: sum(margins) + fees - realized - funding == seed total
    e[w + 17] = 4_515_000_000_000;
    e
}

fn write_words_le(path: &PathBuf, words: &[u64]) {
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    fs::write(path, &bytes).unwrap();
}

#[test]
fn golden_sample_matches_hand_computed_output() {
    let input = sample_input();
    let mut output = [0u64; OUTPUT_WORDS];
    let mut st = ClobState::zeros();
    let mut stats = Stats::zeros();
    run_kernel(&input, &mut output, &mut st, &mut stats);

    let expected = expected_output();
    assert_eq!(output.len(), expected.len());
    for i in 0..OUTPUT_WORDS {
        assert_eq!(
            output[i], expected[i],
            "output word {} mismatch: got {:?} (hex {:#x}) want {:#x}",
            i, output[i] as i64, output[i], expected[i]
        );
    }

    // ---- semantic invariants (independent of the golden numbers) ----
    // positions net to zero in the closed system
    let net: i64 = st.traders.iter().map(|t| t.size).sum();
    assert_eq!(net, 0);
    // funding is zero-sum
    assert_eq!(st.funding_sum, 0);
    // conservation == sum of the seeded input margins
    let seed: u64 = 2_000_000_000_000 + 1_500_000_000_000 + 1_000_000_000_000 + 15_000_000_000;
    assert_eq!(st.conservation, seed);
    // book state sanity: 3 bid slots (1 live + tomb + live), 4 ask slots
    assert_eq!(st.n_bids, 3);
    assert_eq!(st.n_asks, 4);

    // ---- persist golden artifacts (64 u64 words, LE) ----
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    write_words_le(&dir.join("input_sample.bin"), &input);
    write_words_le(&dir.join("expected_output.bin"), &output);

    // ---- report ----
    println!("stats: orders={} scans={} fills={} rest={} scen={}",
             stats.orders, stats.scans, stats.fills, stats.rest, stats.scen);
    // Rough dynamic-instruction estimate for the RV64IMAC guest:
    //   64 input volatile reads + 64 output volatile writes ~ 6 instr each,
    //   per scan slot ~ 12, per fill ~ 90 (2 apply_fill + fees + book ops),
    //   per scenario eval ~ 25, per resting insert ~ 12, per order ~ 40,
    //   plus ~ 700 for validation/marshalling/loops overhead.
    let est = 128 * 6
        + (stats.scans as u64) * 12
        + (stats.fills as u64) * 90
        + (stats.scen as u64) * 25
        + (stats.rest as u64) * 12
        + (stats.orders as u64) * 40
        + 700;
    println!("estimated dynamic instructions (sample): ~{}", est);
    // Rough data-access estimate: 64 in + 64 out + ~5 per scan slot + ~14 per
    // fill + ~6 per scenario + 3 per rest + ~30 per order validation.
    let acc = 128
        + (stats.scans as u64) * 5
        + (stats.fills as u64) * 14
        + (stats.scen as u64) * 6
        + (stats.rest as u64) * 3
        + (stats.orders as u64) * 12;
    println!("estimated data memory accesses (loads+stores): ~{}", acc);

    print!("expected_output.bin words 0..15:");
    for i in 0..16 {
        print!(" {:>6}:{:#x}", i, output[i]);
    }
    println!();
    // full word dump for the report
    let mut line = String::new();
    for i in 0..OUTPUT_WORDS {
        if i % 4 == 0 {
            if !line.is_empty() {
                println!("{}", line);
            }
            line = format!("  [{:>2}]", i);
        }
        line.push_str(&format!(" {:>22}", output[i]));
    }
    println!("{}", line);
}

#[test]
fn sample_orders_fit_input_region() {
    let input = sample_input();
    assert_eq!(input[1] as usize, MAX_ORDERS);
    // 2 + 4*11 + 6 + 6 + 4 = 62 <= 64 words used, 2 pad words
    assert_eq!(2 + 4 * 11 + 16, 62);
    // every limit price is on the $0.50 tick grid
    for k in 0..11 {
        let flags = input[2 + 4 * k];
        if flags & 2 == 0 {
            // limit order: price word must be a tick multiple
            assert_eq!(input[3 + 4 * k] % 500_000_000, 0, "order {}", k + 1);
        }
    }
}
