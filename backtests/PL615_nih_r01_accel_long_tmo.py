"""
PL615 — NIH R01 Obligation YoY Acceleration -> Long TMO

Annual event on first trading day on/after Oct 15 of each calendar year. Look
up the prior-FY R01 obligation YoY. If >= 8%, go long TMO; hold 60 trading
days; exit on +10% (profit) or -8% (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL615_nih_r01_accel_long_tmo"
NAME = "NIH R01 YoY >=8% -> Long TMO (60d)"

# Hardcoded NIH R01 new-award obligated $B by fiscal year (per spec)
NIH_R01_B = {
    2015: 5.4,
    2016: 5.9,
    2017: 6.3,
    2018: 6.9,
    2019: 7.3,
    2020: 7.8,
    2021: 8.5,
    2022: 8.7,
    2023: 8.9,
    2024: 9.2,
}


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["TMO", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "TMO" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    tmo_r = ret["TMO"]
    spy_r = ret["SPY"]
    tmo = px["TMO"]
    trading_idx = ret.index

    # Annual anchors: first trading day on/after Oct 15
    hold_days = 60
    take_profit = 0.10
    stop_loss = -0.08

    fire_events = []
    for year in range(2015, 2026):
        # Calendar year: anchor first trading day >= Oct 15
        anchor = pd.Timestamp(year, 10, 15)
        # prior FY obligation YoY = NIH_R01_B[year] / NIH_R01_B[year-1] - 1
        # spec: "uses prior-FY's actual obligation level" — so for anchor in calendar
        # year Y (post fiscal-year end Sep 30), the latest available FY is Y, hence
        # YoY = NIH_R01_B[Y] / NIH_R01_B[Y-1] - 1.
        if year not in NIH_R01_B or (year - 1) not in NIH_R01_B:
            continue
        yoy = NIH_R01_B[year] / NIH_R01_B[year - 1] - 1.0
        if yoy < 0.08:
            continue
        fire_events.append({"year": year, "yoy": yoy, "anchor": anchor})

    if not fire_events:
        return mark_failed(
            sid,
            "no FY-YoY >=8% events in hardcoded table",
            extra={
                "rule": "Long TMO 60d when NIH R01 YoY >=8% as of Oct 15",
                "mechanism": "NIH R01 acceleration drives life-science tools spend",
                "source": "PL615 catalog; hardcoded NIH RePORTER aggregates",
            },
        )

    positions = pd.Series(0.0, index=trading_idx)
    events_out = []
    last_exit_loc = -1

    for fe in fire_events:
        loc = trading_idx.searchsorted(fe["anchor"])
        if loc >= len(trading_idx) - 1:
            continue
        entry_loc = loc
        if entry_loc <= last_exit_loc:
            continue
        if pd.isna(tmo.iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = tmo.iloc[entry_loc]
        tmo_win = tmo.iloc[entry_loc:end_loc + 1]
        cumret = tmo_win / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold"
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val >= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            if pd.notna(val) and val <= stop_loss:
                exit_loc = cur_i
                exit_reason = "stop_loss"
                break
        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events_out.append({
            "year": fe["year"],
            "yoy_pct": round(fe["yoy"] * 100, 2),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "tmo_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
        })

    if not events_out:
        return mark_failed(sid, "fire events outside TMO price coverage")

    pnl = positions.shift(1).fillna(0.0) * tmo_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()
    first_entry = pd.Timestamp(events_out[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos,
        cost_bps=10,
    )
    m["n_events"] = len(events_out)
    m["pct_in_market"] = float(active_pos.mean())
    m["events"] = events_out
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "On first trading day on/after Oct 15 each year, if NIH R01 new-award obligated $ (per hardcoded annual table from NIH RePORTER ExPORTER) increased YoY by >=8%, go long TMO at entry day's close for up to 60 trading days. Exit on +10% (profit target) or -8% (stop).",
            "mechanism": "NIH R01 obligations directly fund principal investigator labs which buy life-science tools (mass spec, sequencers, reagents) supplied by Thermo Fisher. Acceleration in YoY R01 funding ($B level) leads life-science-tools revenue by 1-3 quarters as labs deploy budgets.",
            "source": "PL615 idea catalog; NIH RePORTER ExPORTER bulk-download aggregates (hardcoded)",
            "caveats": "Small N (~5 events 2015-2024). NIH RePORTER ExPORTER values approximated from public summaries. TMO single-name idiosyncratic risk dominates over 60-day windows.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events_out)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
