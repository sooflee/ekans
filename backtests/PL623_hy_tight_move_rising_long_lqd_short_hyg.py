"""
PL623 — HY OAS < 300bps + MOVE Rising -> Long LQD / Short HYG Pair

Entry:
  - BAMLH0A0HYM2 < 3.00 (HY OAS < 300bps)
  - IG-HY spread (HY - IG) <= 10th percentile of trailing 2520d distribution
  - ^MOVE 20d MA > 110
  - ^MOVE rose in 3+ of last 5 weekly closes
Position: long LQD, short HYG (notional matched)
Exit (earliest of):
  - 45 trading days
  - pair PnL >= +5% (profit)
  - HY OAS > 4.50 (regime confirmed)
  - ^MOVE > 140
  - pair PnL <= -3% (stop)

Counter-signal vs long_SPY / long_credit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL623_hy_tight_move_rising_long_lqd_short_hyg"
NAME = "HY OAS<300 + MOVE Rising -> Long LQD / Short HYG (45d pair)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["LQD", "HYG", "^MOVE", "SPY"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")

    for t in ["LQD", "HYG", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    try:
        fr = load_fred(["BAMLH0A0HYM2", "BAMLC0A0CM"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (FRED): {e}")

    for s in ["BAMLH0A0HYM2", "BAMLC0A0CM"]:
        if s not in fr.columns:
            return mark_failed(sid, f"missing FRED series {s}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    if "^MOVE" not in px.columns:
        return mark_failed(sid, "^MOVE not returned from yfinance")

    move = px["^MOVE"].dropna()
    move_20 = move.rolling(20, min_periods=10).mean()
    # weekly closes: take last close per week, compare with previous 5 weekly closes
    move_weekly = move.resample("W").last().dropna()
    weekly_up = (move_weekly.diff() > 0).astype(int)
    weekly_up_5 = weekly_up.rolling(5, min_periods=3).sum()

    hy = fr["BAMLH0A0HYM2"].dropna()
    ig = fr["BAMLC0A0CM"].dropna()
    spread = hy - ig
    # rolling 2520-day percentile of the spread
    spread_pct = spread.rolling(2520, min_periods=504).apply(
        lambda x: (x <= x[-1]).sum() / len(x), raw=True
    )

    trading_idx = ret.index
    # forward fill series onto trading_idx
    hy_d = hy.reindex(trading_idx, method="ffill")
    spread_pct_d = spread_pct.reindex(trading_idx, method="ffill")
    move_d = move.reindex(trading_idx, method="ffill")
    move_20_d = move_20.reindex(trading_idx, method="ffill")
    weekly_up_5_d = weekly_up_5.reindex(trading_idx, method="ffill")

    cond_hy = hy_d < 3.00
    cond_spread_pct = spread_pct_d <= 0.10
    cond_move_ma = move_20_d > 110
    cond_move_up = weekly_up_5_d >= 3
    entry_signal = (cond_hy & cond_spread_pct & cond_move_ma & cond_move_up).fillna(False)

    pair_r = (ret["LQD"] - ret["HYG"]).dropna()
    pair_r_d = pair_r.reindex(trading_idx).fillna(0.0)

    hold_days = 45
    take_profit = 0.05
    stop_loss = -0.03

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
        if pd.isna(px["LQD"].iloc[entry_loc]) or pd.isna(px["HYG"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        bw = pair_r_d.iloc[entry_loc:end_loc + 1]
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
            if cur_i < len(hy_d) and pd.notna(hy_d.iloc[cur_i]) and hy_d.iloc[cur_i] > 4.50:
                exit_loc = cur_i
                exit_reason = "regime_confirm"
                break
            if cur_i < len(move_d) and pd.notna(move_d.iloc[cur_i]) and move_d.iloc[cur_i] > 140:
                exit_loc = cur_i
                exit_reason = "move_spike"
                break
        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "pair_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "hy_oas_at_trigger": float(hy_d.iloc[i]) if pd.notna(hy_d.iloc[i]) else None,
            "spread_pct_at_trigger": float(spread_pct_d.iloc[i]) if pd.notna(spread_pct_d.iloc[i]) else None,
            "move_20_at_trigger": float(move_20_d.iloc[i]) if pd.notna(move_20_d.iloc[i]) else None,
        })

    if not events:
        return mark_failed(
            sid,
            "no joint HY<300 + tight IG-HY + MOVE rising events fired",
            extra={
                "rule": "Pair long LQD / short HYG 45d on credit-cycle top proxy",
                "mechanism": "Tight HY at cycle top + rising rates vol presages credit underperformance",
                "source": "PL623 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * pair_r_d
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
            "rule": "Open long LQD / short HYG pair at next-day close when (a) BAMLH0A0HYM2 < 3.00, (b) IG-HY spread (HY - IG) in bottom 10th percentile of trailing 2520d (10y), (c) ^MOVE 20d MA > 110, and (d) ^MOVE rose in 3+ of last 5 weekly closes. Hold 45 trading days; exit on +5% pair (profit), -3% pair (stop), HY OAS > 4.50 (regime confirmed), or ^MOVE > 140 (vol spike).",
            "mechanism": "Combination of historically-tight HY spreads, even-tighter compression vs IG, and rising rate volatility marks credit-cycle tops (e.g., 2007 Q3, late 2019/early 2020, Jan 2022). At those junctions, HY underperforms IG as rate-vol passes through to spread widening with leverage. Pair captures relative HY underperformance with controlled duration mismatch (LQD ~9y vs HYG ~4y).",
            "source": "PL623 idea catalog; FRED BAMLH0A0HYM2 + BAMLC0A0CM; yfinance ^MOVE",
            "caveats": "HY tight + MOVE rising is a multi-condition gating signal so n_events is small. Counter-signal against long_SPY / long_credit defaults. LQD-HYG duration mismatch leaves some unhedged rate exposure.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
