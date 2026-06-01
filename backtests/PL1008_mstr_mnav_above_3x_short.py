"""PL1008_mstr_mnav_above_3x_short — MSTR mNAV Premium >3x Extreme Threshold — Short MSTR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1008_mstr_mnav_above_3x_short"

    # MSTR BTC holdings by quarter from 8-K filings (date: BTC_count)
    # Source: MSTR investor relations / SEC EDGAR 8-K filings
    btc_holdings_schedule = [
        ("2020-08-11", 21454),
        ("2020-12-31", 70470),
        ("2021-03-31", 91579),
        ("2021-06-30", 105085),
        ("2021-09-30", 114042),
        ("2021-12-31", 124391),
        ("2022-03-31", 129218),
        ("2022-06-30", 129699),
        ("2022-09-30", 130000),
        ("2022-12-31", 130000),
        ("2023-03-31", 140000),
        ("2023-06-30", 152333),
        ("2023-09-30", 158245),
        ("2023-12-31", 189150),
        ("2024-03-31", 214400),
        ("2024-06-30", 226500),
        ("2024-09-30", 252220),
        ("2024-12-31", 446400),
        ("2025-03-31", 528185),
    ]

    try:
        px = load_prices(["MSTR", "BTC-USD", "SPY"], start="2020-08-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Build daily BTC holdings time series via forward-fill from quarterly 8-K dates
    holdings_series = pd.Series(
        {pd.Timestamp(d): v for d, v in btc_holdings_schedule}
    )
    full_idx = px.index
    holdings_daily = holdings_series.reindex(full_idx).ffill().bfill()

    # MSTR shares outstanding (split-adjusted, diluted).
    # MSTR did a 10:1 forward split on 2024-08-01.
    # yfinance prices are split-adjusted throughout, so pre-split prices ÷10.
    # Therefore shares must be multiplied by 10 for pre-split periods to keep
    # market-cap = price * shares consistent.
    # Pre-split diluted shares ~9.7-13.5M → split-adjusted = 97M-135M.
    # Post-split shares: ~150M-220M as MSTR issued new shares for BTC purchases.
    shares_schedule = [
        ("2020-08-01", 97e6),
        ("2021-01-01", 98e6),
        ("2021-06-01", 101e6),
        ("2021-12-01", 105e6),
        ("2022-06-01", 110e6),
        ("2023-01-01", 115e6),
        ("2023-12-01", 135e6),
        ("2024-06-01", 150e6),
        ("2024-08-01", 152e6),
        ("2024-12-01", 180e6),
        ("2025-03-31", 220e6),
    ]
    shares_s = pd.Series(
        {pd.Timestamp(d): v for d, v in shares_schedule}
    )
    shares_daily = shares_s.reindex(full_idx).ffill().bfill()

    mstr_px = px["MSTR"]
    btc_px = px["BTC-USD"]

    # Compute market cap and mNAV
    mstr_mcap = mstr_px * shares_daily
    btc_value = holdings_daily * btc_px
    mnav = mstr_mcap / btc_value

    # Signal: mNAV > 3.0 for 5 consecutive trading days
    above_3x = (mnav > 3.0).astype(int)
    consec = above_3x.copy()
    for i in range(1, 5):
        consec = consec & above_3x.shift(i).fillna(0).astype(int)
    signal = consec.astype(bool)

    print(f"Signal days (mNAV>3x for 5d): {signal.sum()}")
    print(f"mNAV stats: min={mnav.min():.2f}, max={mnav.max():.2f}, mean={mnav.mean():.2f}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    mstr_r = ret["MSTR"]
    btc_r = ret["BTC-USD"]

    # Trade: short MSTR on signal. Exit when:
    # 1. mNAV drops below 2.0
    # 2. BTC-USD 1-day return < -15% (sharp BTC drop → premium already collapsed)
    # 3. 30 trading days elapsed
    hold_max = 30
    pnl = pd.Series(0.0, index=full_idx)
    in_trade = False
    entry_date = None
    days_held = 0
    event_records = []

    for i, date in enumerate(full_idx):
        if in_trade:
            days_held += 1
            if date in mstr_r.index:
                pnl[date] = -mstr_r[date]  # short MSTR

            # Check exit conditions
            mnav_val = mnav.get(date, np.nan)
            btc_ret = btc_r.get(date, 0.0)
            exit_reason = None

            if mnav_val < 2.0:
                exit_reason = "mnav_below_2x"
            elif btc_ret < -0.15:
                exit_reason = "btc_crash"
            elif days_held >= hold_max:
                exit_reason = "time"

            if exit_reason:
                # Compute total trade return
                trade_pnl = pnl[entry_date:date].sum()
                event_records.append({
                    "entry": str(entry_date.date()),
                    "exit": str(date.date()),
                    "days": days_held,
                    "trade_pnl": round(float(trade_pnl), 4),
                    "exit_reason": exit_reason,
                })
                in_trade = False
                entry_date = None
                days_held = 0

        elif signal.get(date, False) and not in_trade:
            in_trade = True
            entry_date = date
            days_held = 0

    # If still in trade at end
    if in_trade and entry_date is not None:
        last_date = full_idx[-1]
        trade_pnl = pnl[entry_date:last_date].sum()
        event_records.append({
            "entry": str(entry_date.date()),
            "exit": str(last_date.date()),
            "days": days_held,
            "trade_pnl": round(float(trade_pnl), 4),
            "exit_reason": "still_open",
        })

    print(f"\nTrade events ({len(event_records)}):")
    for e in event_records:
        print(f"  {e['entry']} -> {e['exit']} ({e['days']}d, {e['exit_reason']}): {e['trade_pnl']:.2%}")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="MSTR mNAV>3x Short")

    save_result(sid, m, extra={
        "rule": "Short MSTR when mNAV (market_cap / BTC_holdings_value) > 3.0 for 5 consecutive days. Cover at mNAV<2.0, BTC crash >15%, or 30 days elapsed.",
        "mechanism": "MSTR trades at extreme premium to BTC when sentiment is euphoric; mNAV >3x is unsustainable — premium historically compresses back toward 1-2x, creating short-alpha versus BTC.",
        "source": "MSTR and BTC-USD via yfinance; MSTR BTC holdings from SEC EDGAR 8-K filings (saylor.org tracker); shares outstanding from MSTR quarterly filings.",
        "n_events": len(event_records),
        "events": event_records,
    })


if __name__ == "__main__":
    main()
