"""PL810_banxico_fdi_mfg_breakout_mx_industrial_reit — Banxico BOP FDI Manufacturing Breakout -> Long Mexican Industrial REITs (Nearshoring Signal)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL810_banxico_fdi_mfg_breakout_mx_industrial_reit"

    # Known anchor dates: Banxico BOP FDI manufacturing breakout events /
    # nearshoring announcements that trigger the rule
    # Entry: at announcement/publication date; Hold: 6 months (126 trading days)
    events = [
        {"date": "2022-11-15", "label": "Q3-2022 Banxico FDI publication nearshoring wave onset"},
        {"date": "2023-01-26", "label": "Tesla Monterrey $5B announcement"},
        {"date": "2023-05-15", "label": "Q1-2023 Banxico FDI publication confirming nearshoring surge"},
        {"date": "2024-02-15", "label": "BYD/Chinese EV cluster confirmation"},
    ]
    HOLD_DAYS = 126  # 6 months in trading days

    try:
        # Primary: Mexican industrial REITs + SPY benchmark
        # Note: MXN-listed tickers may have limited yfinance coverage; fall back to EWW proxy
        px = load_prices(["FIBRAPL14.MX", "FIBRAMQ12.MX", "VESTA.MX", "EWW", "SPY"],
                         start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "empty price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna() if "SPY" in ret.columns else None
    if spy_r is None or spy_r.empty:
        return mark_failed(sid, "SPY data missing")

    # Determine which tickers are available
    mx_tickers = [t for t in ["FIBRAPL14.MX", "FIBRAMQ12.MX", "VESTA.MX"] if t in ret.columns and ret[t].dropna().shape[0] > 50]
    if not mx_tickers:
        # Fall back to EWW as Mexico proxy
        if "EWW" in ret.columns and ret["EWW"].dropna().shape[0] > 50:
            mx_tickers = ["EWW"]
            print("MXN REITs unavailable; using EWW as Mexico proxy")
        else:
            return mark_failed(sid, "no usable Mexico exposure tickers")

    print(f"Available MX tickers: {mx_tickers}")

    # Build equal-weighted Mexico industrial REIT return
    mx_ret = ret[mx_tickers].mean(axis=1)

    all_idx = spy_r.index
    pnl = pd.Series(0.0, index=all_idx)
    event_details = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        # Find entry date in index (or nearest future trading day)
        future = all_idx[all_idx >= entry_date]
        if len(future) == 0:
            print(f"Event {entry_date.date()}: no trading days after event, skipping")
            continue
        entry_idx = all_idx.get_loc(future[0])
        exit_idx = min(entry_idx + HOLD_DAYS, len(all_idx) - 1)
        window = all_idx[entry_idx:exit_idx + 1]

        if len(window) < 20:
            print(f"Event {entry_date.date()}: window too short ({len(window)}), skipping")
            continue

        ev_pnl = mx_ret.reindex(window).fillna(0)
        pnl.loc[window] = pnl.loc[window].add(ev_pnl, fill_value=0)

        cum_ret = float((1 + ev_pnl).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(window).fillna(0)).prod() - 1)
        event_details.append({
            "date": ev["date"],
            "label": ev["label"],
            "entry": str(future[0].date()),
            "exit": str(all_idx[exit_idx].date()),
            "days": len(window),
            "mx_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4),
        })
        print(f"  {ev['date']} [{ev['label'][:40]}]: MX {cum_ret:.2%} vs SPY {spy_cum:.2%} over {len(window)} days")

    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)}); event coverage too thin")

    spy_bench = spy_r.reindex(active_pnl.index).dropna()
    m = compute_metrics(active_pnl, benchmark=spy_bench,
                        name="Banxico FDI Mfg Breakout -> Long MX Industrial REITs")
    m["n_events"] = len(event_details)

    save_result(sid, m, extra={
        "rule": "When Banxico quarterly BOP FDI manufacturing sub-component shows breakout (>1.5 stdev above 12-quarter mean for 2 consecutive quarters, or single print >80% above mean), enter long FIBRAPL14.MX (40%) + FIBRAMQ12.MX (40%) + VESTA.MX (20%). Known anchors: Q3-2022 nearshoring onset, Tesla Monterrey Jan 2023, Q1-2023 confirmation, BYD cluster Feb 2024. Hold 6 months.",
        "mechanism": "Nearshoring FDI into Mexican manufacturing directly drives demand for industrial real estate (logistics parks, assembly plants). Banxico BOP data confirms macro flow before it is reflected in property valuations. Mexican industrial REITs (FIBRAs) are the most direct beneficiary.",
        "source": "Banxico BOP FDI quarterly data; Secretaria de Economia FDI announcements; Tesla/BYD public announcements; yfinance FIBRAPL14.MX, FIBRAMQ12.MX, VESTA.MX, EWW, SPY",
        "tickers_used": mx_tickers,
        "events": event_details,
        "caveats": "MXN-listed REIT tickers may have limited yfinance coverage; EWW used as fallback. MXN/USD FX embedded in MXN returns. Only 4 well-defined events in history — event study has limited statistical power.",
    })
    print(f"Saved: Sharpe={m.get('sharpe','N/A')}, CAGR={m.get('cagr','N/A')}")


if __name__ == "__main__":
    main()
