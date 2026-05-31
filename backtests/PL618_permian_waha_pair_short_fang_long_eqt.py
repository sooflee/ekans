"""
PL618 — Permian Associated-Gas Oversupply Proxy -> Short FANG/OXY vs Long EQT/AR

Pair: short 0.5*FANG + 0.5*OXY, long 0.5*EQT + 0.5*AR
Entry: NG=F percentile <= 0.25 over trailing 252d AND NG=F < 200d SMA for prior 60d
Exit (earliest of):
  - 30 trading days
  - pair cumulative return >= +8% (profit)
  - NG=F percentile rises above 0.50 (regime exit)
  - pair cumulative return <= -6% (stop)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL618_permian_waha_pair_short_fang_long_eqt"
NAME = "NG=F Bot25% + sub-200d-SMA -> Short FANG/OXY Long EQT/AR (30d pair)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["FANG", "OXY", "EQT", "AR", "NG=F", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["FANG", "OXY", "EQT", "AR", "NG=F", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    trading_idx = ret.index

    # Pair return (long EQT/AR - short FANG/OXY), equal weights 0.5 each
    pair_r = (0.5 * ret["EQT"] + 0.5 * ret["AR"] -
              0.5 * ret["FANG"] - 0.5 * ret["OXY"]).dropna()

    ng = px["NG=F"]
    ng_252_pct = ng.rolling(252).apply(lambda x: (x <= x[-1]).sum() / len(x), raw=True)
    ng_sma200 = ng.rolling(200, min_periods=100).mean()
    ng_below = (ng < ng_sma200).astype(int)
    ng_below_60 = ng_below.rolling(60).sum()  # 60 if entire 60d below

    cond_pct = ng_252_pct <= 0.25
    cond_streak = ng_below_60 >= 60
    entry_signal = (cond_pct & cond_streak).fillna(False)

    hold_days = 30
    take_profit = 0.08
    stop_loss = -0.06

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
        # need all 4 legs valid
        if any(pd.isna(px[t].iloc[entry_loc]) for t in ["FANG", "OXY", "EQT", "AR"]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        # walk
        pair_win = pair_r.reindex(trading_idx).iloc[entry_loc:end_loc + 1].fillna(0.0)
        cumret = (1 + pair_win).cumprod() - 1
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
            # regime exit
            if cur_i < len(ng_252_pct) and pd.notna(ng_252_pct.iloc[cur_i]) and ng_252_pct.iloc[cur_i] > 0.50:
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
            "pair_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "ng_pct_at_trigger": float(ng_252_pct.iloc[i]),
        })

    if not events:
        return mark_failed(
            sid,
            "no NG=F oversupply regime events fired",
            extra={
                "rule": "Pair short FANG/OXY long EQT/AR on NG=F bot-25% + sub-200d-SMA 60d streak",
                "mechanism": "Sustained NG oversupply hits Permian (worse Waha basis) > Appalachia",
                "source": "PL618 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * pair_r.reindex(positions.index).fillna(0.0)
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
        cost_bps=20,  # pair of 4 legs, higher round-trip
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.mean())
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Pair: short 0.5*FANG + 0.5*OXY, long 0.5*EQT + 0.5*AR. Enter at next-day close when NG=F front-month percentile <= 0.25 over trailing 252d AND NG=F was below 200d SMA for the prior 60 consecutive trading days. Hold up to 30 trading days; exit on +8% pair (profit), -6% pair (stop), or NG=F percentile > 0.50 (regime exit).",
            "mechanism": "Sustained Henry Hub oversupply (low percentile + sub-200d streak) historically coincides with the worst Waha basis blowouts (2019, 2023, 2024). Permian producers (FANG pure-Permian, OXY heavy Permian post-Anadarko) suffer disproportionately from negative associated-gas basis vs Appalachian dry-gas producers (EQT, AR) that price closer to Henry Hub. Pair isolates the basin-basis divergence.",
            "source": "PL618 idea catalog; yfinance NG=F",
            "caveats": "PXD delisted (XOM acquisition May 2024) — OXY substituted as Permian leg #2. Waha basis time-series not in FRED so NG=F regime is a coarse proxy. Pair has 4 legs and higher friction (20bps round-trip used).",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
