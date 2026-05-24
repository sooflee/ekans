"""
AA3 MSTR premium to NAV (BTC holdings).
MicroStrategy (now Strategy) accumulates BTC. Hypothesis: when MSTR market cap
exceeds 2.5x its BTC holding value, the premium is rich → short MSTR / long BTC.
Unwind below 1.5x.

BTC holdings curated from MicroStrategy 8-K filings + quarterly disclosures.
~quarterly granularity (forward-filled between purchase dates).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

# (date, cumulative BTC held). Curated from MSTR 8-K filings.
# Source: https://www.strategy.com/purchases (Strategy.com BTC purchases list)
BTC_HOLDINGS = [
    ("2020-08-11",   21454),  # initial 21,454 BTC
    ("2020-09-14",   38250),
    ("2020-12-04",   40824),
    ("2020-12-21",   70470),
    ("2021-02-02",   71079),
    ("2021-02-24",   90531),
    ("2021-03-12",   91326),
    ("2021-04-05",   91579),
    ("2021-05-13",   92079),
    ("2021-06-21",  105085),
    ("2021-08-24",  108992),
    ("2021-09-13",  114042),
    ("2021-11-29",  121044),
    ("2021-12-09",  122478),
    ("2021-12-30",  124391),
    ("2022-02-01",  125051),
    ("2022-04-05",  129218),
    ("2022-06-29",  129699),
    ("2022-09-20",  130000),
    ("2022-12-28",  132500),
    ("2023-03-23",  140000),
    ("2023-04-05",  140000),
    ("2023-06-28",  152333),
    ("2023-08-01",  152800),
    ("2023-09-25",  158245),
    ("2023-11-30",  174530),
    ("2024-02-26",  193000),
    ("2024-03-19",  214246),
    ("2024-04-01",  214400),
    ("2024-06-20",  226331),
    ("2024-08-01",  226500),
    ("2024-09-20",  252220),
    ("2024-10-31",  252220),
    ("2024-11-25",  386700),
    ("2024-12-09",  423650),
    ("2024-12-23",  444262),
    ("2025-01-13",  450000),
    ("2025-02-24",  478740),
    ("2025-03-31",  528185),
    ("2025-05-12",  568840),
    ("2025-06-30",  597325),
    ("2025-08-15",  629376),
    ("2025-10-01",  640250),
    ("2025-12-15",  649870),
    ("2026-02-15",  659000),
    ("2026-04-15",  668500),
]


def main():
    try:
        px = load_prices(["MSTR", "BTC-USD", "IBIT"], start="2020-08-01")
    except Exception as e:
        return mark_failed("AA3_mstr_premium", f"data load failed: {e}")

    mstr = px.get("MSTR")
    btc = px.get("BTC-USD")
    if mstr is None or btc is None or mstr.dropna().empty or btc.dropna().empty:
        return mark_failed("AA3_mstr_premium", "missing MSTR or BTC-USD price")

    # Use BTC-USD as the BTC proxy throughout (24/7 data, full history back to 2020).
    # IBIT only exists Jan 2024+ so it can't backtest 2020-2023.
    btc_proxy = btc
    # Reindex everything onto MSTR's index (NYSE calendar).
    idx = mstr.dropna().index
    btc_d = btc.reindex(idx).ffill()
    btc_proxy_d = btc_proxy.reindex(idx).ffill()

    # Build BTC holdings step series, forward filled.
    hold_dates = [pd.Timestamp(d) for d, _ in BTC_HOLDINGS]
    hold_qty = [q for _, q in BTC_HOLDINGS]
    holdings = pd.Series(hold_qty, index=hold_dates).reindex(idx, method="ffill").bfill()

    # Shares outstanding — load via yfinance .fast_info or curated.
    # Use approximate constant shares (curated) per period — MSTR has done splits and ATMs.
    # Effective shares outstanding (split-adjusted to current, in millions):
    # We'll approximate using yfinance's market cap fast_info today scaled by price ratio.
    # Simpler: use BTC-per-share notion. Construct mNAV = MSTR_price / (BTC_per_share * BTC_price).
    # BTC-per-share = holdings / shares. Curate shares outstanding (split-adjusted):
    # yfinance auto_adjust=True returns split-adjusted prices (10:1 split Aug 2024
    # applied retroactively). To make market_cap continuous, shares-outstanding
    # must also be on a split-adjusted ("today-equivalent") basis throughout.
    # Pre-split actual shares × 10 = split-adjusted shares.
    SHARES_OUT = [
        ("2020-08-01",   97e6),     # 9.7M actual × 10 split-adj
        ("2021-03-31",   97e6),
        ("2022-03-31",  113e6),
        ("2023-03-31",  137e6),
        ("2024-03-31",  176e6),
        ("2024-08-01",  183e6),
        ("2024-10-31",  196e6),
        ("2024-12-01",  228e6),
        ("2025-02-28",  270e6),
        ("2025-06-30",  285e6),
        ("2025-12-31",  297e6),
        ("2026-04-30",  305e6),
    ]
    so_idx = [pd.Timestamp(d) for d, _ in SHARES_OUT]
    so_val = [v for _, v in SHARES_OUT]
    shares = pd.Series(so_val, index=so_idx).reindex(idx, method="ffill").bfill()

    # NOTE: yfinance auto_adjust=True returns split-adjusted prices.
    # Pre-split MSTR was ~$1300; post-split ~$130. The shares curve above is set so that
    # market_cap = price * shares is continuous across the split.

    market_cap = mstr * shares
    btc_value = holdings * btc_d
    mnav = market_cap / btc_value

    # Trading rule: entry when mnav > 2.5 → short MSTR / long BTC proxy.
    # Exit (flat) when mnav drops below 1.5. Hysteresis.
    pos_mstr = pd.Series(0.0, index=idx)
    in_trade = False
    for t in idx:
        v = mnav.loc[t]
        if not in_trade and v > 2.5:
            in_trade = True
        elif in_trade and v < 1.5:
            in_trade = False
        pos_mstr.loc[t] = -1.0 if in_trade else 0.0
    pos_btc = -pos_mstr  # long BTC when short MSTR

    mstr_r = mstr.pct_change()
    btc_proxy_r = btc_proxy_d.pct_change()
    pnl = (pos_mstr.shift(1) * mstr_r + pos_btc.shift(1) * btc_proxy_r).dropna()

    if (pos_mstr != 0).sum() < 30:
        return mark_failed("AA3_mstr_premium", "too few days in-trade with mNAV thresholds")

    bench = mstr_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA3 MSTR mNAV premium short")
    print_metrics(m)
    save_result("AA3_mstr_premium", m, extra={
        "status": "ok",
        "rule": "Short MSTR / long BTC when mNAV > 2.5; flat when < 1.5.",
        "universe": "MSTR vs BTC-USD (or IBIT)",
        "source": "Strategy.com purchases list + 8-K filings",
        "n_in_trade_days": int((pos_mstr != 0).sum()),
    })


if __name__ == "__main__":
    main()
