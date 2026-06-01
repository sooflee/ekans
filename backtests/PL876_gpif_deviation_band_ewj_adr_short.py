"""PL876_gpif_deviation_band_ewj_adr_short - GPIF Domestic-Equity Deviation Band Breach -> Short EWJ + Japan ADRs

Event-study backtest: GPIF quarterly disclosure dates where domestic-equity weight
was confirmed >= 31% (upper deviation band breach). Short EWJ + TM + MUFG + SFTBY
vs long URTH, hold 8 weeks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# GPIF quarterly disclosure dates where domestic-equity weight >= 31%
# Based on known events per strategy specification
EVENTS = [
    "2015-08-28",   # Q1 FY2015 disclosure
    "2017-11-30",   # Q2 FY2017 disclosure
    "2019-11-29",   # Q2 FY2019 disclosure
    "2021-02-26",   # Q3 FY2020 disclosure
    "2023-11-30",   # Q2 FY2023 disclosure
]
HOLD_DAYS = 40  # ~8 weeks of trading days


def main():
    sid = "PL876_gpif_deviation_band_ewj_adr_short"
    try:
        tickers = ["EWJ", "TM", "MUFG", "SFTBY", "URTH", "SPY"]
        px = load_prices(tickers, start="2013-01-01")
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
        # Find next trading day after event date
        mask = ret.index > td
        if mask.sum() < 2:
            continue
        idxs = ret.index[mask]
        entry = idxs[0]  # T+1 entry
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 15:
            continue

        window = ret.iloc[loc:end]

        # Build short basket: equal weight EWJ, TM, MUFG, SFTBY
        # and long URTH as benchmark
        short_legs = []
        available_shorts = []
        for ticker in ["EWJ", "TM", "MUFG", "SFTBY"]:
            if ticker in window.columns and window[ticker].notna().sum() > 10:
                short_legs.append(-window[ticker])
                available_shorts.append(ticker)

        if not short_legs:
            continue

        long_leg = window["URTH"] if "URTH" in window.columns else window["SPY"] * 0

        # Equal-weight short basket normalized to 1 gross, minus long URTH
        n_short = len(short_legs)
        short_basket = sum(s / n_short for s in short_legs)

        # PnL = short basket - long URTH (long/short pair, net zero dollar)
        # Scale to 50/50 gross: 0.5 short basket + 0.5 long URTH
        pnl_window = 0.5 * short_basket + 0.5 * long_leg

        pnl_parts.append(pnl_window)
        cumret = float((1 + pnl_window).prod() - 1)
        events_detail.append({
            "trigger_date": d,
            "entry": str(entry.date()),
            "available_shorts": available_shorts,
            "n_days": len(pnl_window),
            "net_return": round(cumret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no valid events after data availability check")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="GPIF Deviation Band EWJ ADR Short")
    save_result(sid, m, extra={
        "rule": "Short EWJ + TM + MUFG + SFTBY, long URTH, on GPIF domestic-equity weight >= 31% (deviation band breach), hold 8 weeks.",
        "mechanism": "GPIF rebalancing forces mechanical selling of domestic Japanese equities when portfolio weight breaches +6pp upper band, creating persistent selling pressure on EWJ and Japan ADRs.",
        "source": "GPIF quarterly portfolio composition disclosures (gpif.go.jp/en/performance); yfinance prices",
        "n_events": len(events_detail),
        "events": events_detail,
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events_detail])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events_detail])), 4),
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events_detail)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")


if __name__ == "__main__":
    main()
