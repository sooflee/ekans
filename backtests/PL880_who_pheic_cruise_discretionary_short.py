"""PL880_who_pheic_cruise_discretionary_short - WHO PHEIC Declaration -> Short Cruise + Big-Ticket Discretionary

Event-study backtest on all WHO PHEIC declarations since 2009. Short equal-weight
basket of CCL, NCLH, RCL, HOG vs SPY benchmark. Hold ~10 trading days.
Pre-2013 events use CCL+HOG+RCL only (NCLH IPO 2013).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# WHO PHEIC declaration dates
EVENTS = [
    "2009-04-25",   # H1N1 swine flu
    "2014-05-05",   # Polio
    "2014-08-08",   # Ebola West Africa
    "2016-02-01",   # Zika
    "2019-07-17",   # Ebola DRC
    "2020-01-30",   # COVID-19
    "2022-07-23",   # Mpox
    "2024-08-14",   # Mpox 2024
]
HOLD_DAYS = 10  # ~14 calendar days


def main():
    sid = "PL880_who_pheic_cruise_discretionary_short"
    try:
        px = load_prices(["CCL", "NCLH", "RCL", "HOG", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    if "SPY" not in ret.columns:
        return mark_failed(sid, "SPY missing")
    spy_r = ret["SPY"]

    pnl_parts = []
    events_detail = []

    for d in EVENTS:
        td = pd.Timestamp(d)
        # Entry: T+1 close after PHEIC declaration
        mask = ret.index > td
        if mask.sum() < 2:
            continue
        idxs = ret.index[mask]
        entry = idxs[0]  # first trading day after declaration
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 5:
            continue

        window = ret.iloc[loc:end]

        # Determine which tickers available (NCLH only after 2013)
        short_names = ["CCL", "RCL", "HOG"]
        if td >= pd.Timestamp("2013-01-01") and "NCLH" in window.columns:
            if window["NCLH"].notna().sum() > 3:
                short_names.append("NCLH")

        short_legs = []
        used_names = []
        for ticker in short_names:
            if ticker in window.columns and window[ticker].notna().sum() > 3:
                short_legs.append(-window[ticker])
                used_names.append(ticker)

        if not short_legs:
            continue

        n_short = len(short_legs)
        # PnL = short basket minus SPY (measuring outperformance of short vs index)
        short_basket = sum(s / n_short for s in short_legs)
        # Raw short basket PnL (positive = shorts fall, good for us)
        pnl_window = short_basket - window["SPY"]

        pnl_parts.append(pnl_window)
        cumret = float((1 + pnl_window).prod() - 1)
        events_detail.append({
            "trigger_date": d,
            "entry": str(entry.date()),
            "basket": used_names,
            "n_days": len(pnl_window),
            "net_return": round(cumret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no valid events after data availability check")

    # Build daily pnl: when not in an event, flat (0)
    # Align all windows on a common index
    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="WHO PHEIC Cruise Discretionary Short")
    save_result(sid, m, extra={
        "rule": "T+1 close: short equal-weight CCL+NCLH+RCL+HOG vs SPY, hold ~10 trading days after WHO PHEIC declaration.",
        "mechanism": "PHEIC triggers travel fear, forcing cruise lines and big-ticket discretionary stocks to collapse disproportionately vs broad market.",
        "source": "WHO IHR PHEIC declarations (who.int); yfinance prices",
        "n_events": len(events_detail),
        "events": events_detail,
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events_detail])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events_detail])), 4),
        "note": "COVID-19 2020-01-30 event dominates; check Sharpe without that outlier.",
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events_detail)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")
    for e in events_detail:
        print(f"  {e['trigger_date']}: {e['basket']} -> {e['net_return']:+.3f}")


if __name__ == "__main__":
    main()
