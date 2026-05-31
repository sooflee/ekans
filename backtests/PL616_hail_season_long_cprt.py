"""
PL616 — Hail-Season + Auto-MSA Cluster Proxy -> Long CPRT

Annual event on the first trading day of June. Fire if:
  - CPRT 60d return < +5% (not chased)
  - PGR 90d return < -5% (insurance equity stress)
Hold 60 days; exit on +12% (profit) or -8% (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL616_hail_season_long_cprt"
NAME = "Hail-Season + PGR Stress -> Long CPRT (60d)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["CPRT", "PGR", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["CPRT", "PGR", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    cprt = px["CPRT"]
    pgr = px["PGR"]
    cprt_r = ret["CPRT"]
    spy_r = ret["SPY"]
    trading_idx = ret.index

    cprt_60 = cprt / cprt.shift(60) - 1.0
    pgr_90 = pgr / pgr.shift(90) - 1.0

    hold_days = 60
    take_profit = 0.12
    stop_loss = -0.08

    positions = pd.Series(0.0, index=trading_idx)
    events = []
    last_exit_loc = -1

    for year in range(2010, 2027):
        june_first = pd.Timestamp(year, 6, 1)
        loc = trading_idx.searchsorted(june_first)
        if loc >= len(trading_idx) - 1:
            continue
        entry_loc = loc
        if entry_loc <= last_exit_loc:
            continue
        cprt60 = cprt_60.iloc[entry_loc] if entry_loc < len(cprt_60) else np.nan
        pgr90 = pgr_90.iloc[entry_loc] if entry_loc < len(pgr_90) else np.nan
        if pd.isna(cprt60) or pd.isna(pgr90):
            continue
        if not (cprt60 < 0.05 and pgr90 < -0.05):
            continue
        if pd.isna(cprt.iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = cprt.iloc[entry_loc]
        cprt_win = cprt.iloc[entry_loc:end_loc + 1]
        cumret = cprt_win / entry_price - 1.0
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
        events.append({
            "year": year,
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "cprt_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "cprt_60d_at_trigger": float(cprt60),
            "pgr_90d_at_trigger": float(pgr90),
        })

    if not events:
        return mark_failed(
            sid,
            "no joint hail-season + PGR-stress events fired",
            extra={
                "rule": "Long CPRT 60d on June anchor + CPRT 60d <+5% + PGR 90d <-5%",
                "mechanism": "Peak hail season + auto-insurance stress -> salvage volume surge",
                "source": "PL616 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * cprt_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()
    first_entry = pd.Timestamp(events[0]["entry_date"])
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
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.mean())
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "On the first trading day of June each year, if CPRT trailing 60d return < +5% AND PGR trailing 90d return < -5%, go long CPRT at entry close for up to 60 trading days. Exit on +12% (profit) or -8% (stop).",
            "mechanism": "Peak SPC large-hail activity April-June across the Plains/Midwest auto-density corridor drives spike in auto total-losses; Copart (CPRT) is the dominant salvage-auction operator and earnings lever directly on total-loss volume. PGR 90d-stress filter captures regimes where insurance equity is already pricing elevated claims activity, raising the probability that elevated hail catastrophe is the driver.",
            "source": "PL616 idea catalog; yfinance CPRT/PGR; SPC seasonal hail climatology",
            "caveats": "Coarse proxy for SPC hail clusters (SPC Storm Reports CSV not auto-ingested). PGR trailing-90d stress is non-specific. Small N (annual). CPRT is high-multiple growth name with idiosyncratic risk dominant over 60-day windows.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
