"""
PL631 — FAA AD Cluster Proxy (Boeing Stress) -> Long HEI/TDG Aftermarket

Entry:
  - HEI 60d - BA 60d >= +0.10
  - TDG 60d - BA 60d >= +0.10
  - HEI close / HEI 50d SMA <= 1.15
Position: long 0.5*HEI + 0.5*TDG basket.
Exit (earliest of):
  - 80 trading days
  - basket cumret >= +12% (profit)
  - BA 20d - avg(HEI 20d, TDG 20d) >= +0.05 (regime exit)
  - basket cumret <= -8% (stop)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL631_faa_ad_cluster_long_hei_tdg"
NAME = "BA underperform vs HEI/TDG -> Long HEI/TDG (80d basket)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["HEI", "TDG", "BA", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    for t in ["HEI", "TDG", "BA", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    trading_idx = ret.index

    hei = px["HEI"]
    tdg = px["TDG"]
    ba = px["BA"]
    hei_60 = hei / hei.shift(60) - 1.0
    tdg_60 = tdg / tdg.shift(60) - 1.0
    ba_60 = ba / ba.shift(60) - 1.0
    hei_sma50 = hei.rolling(50, min_periods=20).mean()
    hei_over_sma = hei / hei_sma50

    cond_hei = (hei_60 - ba_60) >= 0.10
    cond_tdg = (tdg_60 - ba_60) >= 0.10
    cond_chase = hei_over_sma <= 1.15
    entry_signal = (cond_hei & cond_tdg & cond_chase).fillna(False)

    basket_r = (0.5 * ret["HEI"] + 0.5 * ret["TDG"]).dropna()
    basket_r_d = basket_r.reindex(trading_idx).fillna(0.0)

    hold_days = 80
    take_profit = 0.12
    stop_loss = -0.08

    hei_20 = hei / hei.shift(20) - 1.0
    tdg_20 = tdg / tdg.shift(20) - 1.0
    ba_20 = ba / ba.shift(20) - 1.0

    positions = pd.Series(0.0, index=trading_idx)
    events = []
    last_exit_loc = -1

    for i, dt in enumerate(trading_idx):
        if i <= last_exit_loc:
            continue
        if not entry_signal.iloc[i]:
            continue
        entry_loc = i + 1
        if entry_loc >= len(trading_idx) - 1:
            continue
        if pd.isna(px["HEI"].iloc[entry_loc]) or pd.isna(px["TDG"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        bw = basket_r_d.iloc[entry_loc:end_loc + 1]
        cumret = (1 + bw).cumprod() - 1
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
            if cur_i < len(ba_20) and pd.notna(ba_20.iloc[cur_i]) and pd.notna(hei_20.iloc[cur_i]) and pd.notna(tdg_20.iloc[cur_i]):
                regime_diff = ba_20.iloc[cur_i] - 0.5 * (hei_20.iloc[cur_i] + tdg_20.iloc[cur_i])
                if regime_diff >= 0.05:
                    exit_loc = cur_i
                    exit_reason = "regime_exit"
                    break
        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "basket_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
        })

    if not events:
        return mark_failed(
            sid,
            "no BA-underperformance + non-chase events fired",
            extra={
                "rule": "Long HEI/TDG 80d on BA-underperformance + non-chase",
                "mechanism": "FAA AD cluster regime drives unscheduled MRO benefiting aftermarket",
                "source": "PL631 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * basket_r_d
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
    m["events"] = events[:30]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Go long equal-weight (0.5*HEI + 0.5*TDG) basket at next-day close when HEI 60d return - BA 60d return >= +10% AND TDG 60d return - BA 60d return >= +10% AND HEI close <= 1.15 * HEI 50d SMA (not chasing). Hold up to 80 trading days; exit on +12% basket (profit), -8% basket (stop), or BA 20d return - avg(HEI 20d, TDG 20d) >= +5% (regime exit).",
            "mechanism": "Sustained Boeing relative underperformance vs HEI/TDG over 60d signals an active FAA AD cluster / aircraft-grounding regime (e.g., 2019 MAX grounding, 2024 door-plug AD). HEI (PMA spares) and TDG (proprietary components) benefit from the resulting unscheduled MRO surge. The HEI-not-chasing condition avoids late-cycle entries.",
            "source": "PL631 idea catalog; yfinance",
            "caveats": "FAA AD database not in FRED — BA relative underperformance is a coarse proxy. Three known major regimes (2019, 2024) limit sample size of clean regimes.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
