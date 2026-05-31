"""
PL614 — Cotton Supply-Cut Regime Proxy -> Long CT=F

Price-based regime detector substituting for PDF-only WASDE production cuts.
Entry:
  - CT=F 90d return <= -20% (oversold / over-supplied context)
  - CT=F close > 50d SMA on entry day (regime change)
  - CT=F close was <= 50d SMA for at least 30 of the prior 60 trading days (fresh breakout)
Exit (earliest of):
  - 60 trading days
  - CT=F +15% from entry close (profit target)
  - CT=F closes < 50d SMA for 5 consecutive days
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL614_cotton_wasde_cut_long_ct"
NAME = "Cotton 90d -20% + 50d SMA Breakout -> Long CT=F (60d)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["CT=F", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "CT=F" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    ct_r = ret["CT=F"]
    spy_r = ret["SPY"]

    ct = px["CT=F"].dropna()
    if len(ct) < 1000:
        return mark_failed(sid, f"insufficient CT=F coverage: {len(ct)}")

    trading_idx = ct.index
    ret_90 = ct / ct.shift(90) - 1.0
    sma50 = ct.rolling(50, min_periods=20).mean()
    above_sma = ct > sma50
    below_sma = (ct <= sma50).astype(int)
    below_60 = below_sma.rolling(60, min_periods=30).sum()

    cond_oversold = ret_90 <= -0.20
    cond_breakout = above_sma & (below_60 >= 30)
    entry_signal = (cond_oversold & cond_breakout).fillna(False)

    hold_days = 60
    take_profit = 0.15
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
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = ct.iloc[entry_loc]
        if pd.isna(entry_price):
            continue
        ct_win = ct.iloc[entry_loc:end_loc + 1]
        cumret = ct_win / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold"
        below_count = 0
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val >= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            # 5 consecutive days below SMA
            if cur_i < len(trading_idx):
                if pd.notna(ct.iloc[cur_i]) and pd.notna(sma50.iloc[cur_i]):
                    if ct.iloc[cur_i] < sma50.iloc[cur_i]:
                        below_count += 1
                        if below_count >= 5:
                            exit_loc = cur_i
                            exit_reason = "sma_break"
                            break
                    else:
                        below_count = 0

        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "ct_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "ret_90d_at_trigger": float(ret_90.iloc[i]),
        })

    if not events:
        return mark_failed(
            sid,
            "no joint oversold+breakout events fired",
            extra={
                "rule": "Long CT=F 60d on -20% 90d + 50d SMA breakout",
                "mechanism": "Oversold cotton + breakout = supply-cut regime change",
                "source": "PL614 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * ct_r.reindex(positions.index).fillna(0.0)
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
    m["pct_in_market"] = float(active_pos.mean()) if len(active_pos) else 0.0
    m["events"] = events[:20]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Go long CT=F at next-day close when CT=F 90-day return <= -20% AND CT=F closes above its 50d SMA AND CT=F closed at/below 50d SMA on at least 30 of the prior 60 trading days. Hold up to 60 trading days; exit on +15% (profit target) or 5 consecutive closes below 50d SMA.",
            "mechanism": "20%+ 90d drawdown reflects an over-supplied cotton market where WASDE country-level cuts (Pakistan/India) historically follow. SMA breakout marks the regime change as the market begins pricing in tighter forward S&D. Strategy substitutes price-regime detection for PDF-only WASDE production tables.",
            "source": "PL614 idea catalog; yfinance CT=F",
            "caveats": "Pure-price proxy for WASDE country production cuts; will fire on non-fundamental drawdowns (e.g. macro-driven commodity selloffs) too. BAL ETF delisted so CT=F futures are the only backtestable target. Cotton is volatile and historically subject to large gappy moves.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
