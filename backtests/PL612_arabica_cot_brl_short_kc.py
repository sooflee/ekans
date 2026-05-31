"""
PL612 — Arabica COT proxy (KC=F price percentile) + BRL strength -> Short KC=F

Entry:
  - KC=F price percentile over trailing 756d >= 0.90
  - DEXBZUS down >2% over 20 trading days (USD/BRL fell -> BRL strengthened)
Exit (earliest of):
  - 30 trading days
  - KC=F closes <= -10% from entry (profit target on short)
  - KC=F closes at trailing 252d high (signal failure stop)

Counter-signal against long_commodity_complex.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL612_arabica_cot_brl_short_kc"
NAME = "KC=F 3y-90th + BRL +2% -> Short KC=F (30d)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["KC=F", "SPY"], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")

    if "KC=F" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    try:
        fx = load_fred(["DEXBZUS"], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (DEXBZUS): {e}")

    if "DEXBZUS" not in fx.columns:
        return mark_failed(sid, "DEXBZUS not returned from FRED")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    kc_r = ret["KC=F"]
    spy_r = ret["SPY"]

    kc = px["KC=F"].dropna()
    spy = px["SPY"].dropna()
    brl = fx["DEXBZUS"].dropna()

    # Align: use trading-day index of KC=F
    trading_idx = kc.index.intersection(spy.index)
    kc = kc.reindex(trading_idx).ffill(limit=3)
    brl = brl.reindex(trading_idx).ffill(limit=5)

    if len(kc) < 1000:
        return mark_failed(sid, f"insufficient KC=F coverage: {len(kc)}")

    # rolling 756d percentile rank of KC=F (rank of last value within window / window)
    win = 756
    def rolling_pct_rank(s, window):
        # rank(method='max') of last value
        def fn(x):
            return (x <= x[-1]).sum() / len(x)
        return s.rolling(window).apply(fn, raw=True)

    pct = rolling_pct_rank(kc, win)

    # 20-day BRL change
    brl_20 = brl / brl.shift(20) - 1.0

    # 252d high check (stop)
    kc_252_max = kc.rolling(252, min_periods=60).max()

    cond_pct = pct >= 0.90
    cond_brl = brl_20 <= -0.02
    entry_signal = (cond_pct & cond_brl).fillna(False)

    hold_days = 30
    take_profit = -0.10  # short profit target = KC down 10%
    in_pos = False
    positions = pd.Series(0.0, index=trading_idx)
    events = []
    last_exit_loc = -1

    for i, dt in enumerate(trading_idx):
        if in_pos:
            continue
        if i <= last_exit_loc:
            continue
        if not entry_signal.iloc[i]:
            continue
        # entry next trading day
        entry_loc = i + 1
        if entry_loc >= len(trading_idx) - 1:
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = kc.iloc[entry_loc]
        if pd.isna(entry_price):
            continue
        kc_window = kc.iloc[entry_loc:end_loc + 1]
        cumret = kc_window / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold"
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val <= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            # signal-failure stop: KC=F closes at 252-day high
            if cur_i < len(trading_idx):
                if pd.notna(kc.iloc[cur_i]) and pd.notna(kc_252_max.iloc[cur_i]):
                    if kc.iloc[cur_i] >= kc_252_max.iloc[cur_i]:
                        exit_loc = cur_i
                        exit_reason = "fail_stop_252high"
                        break

        positions.iloc[entry_loc:exit_loc + 1] = -1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "kc_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "kc_percentile": float(pct.iloc[i]),
            "brl_20d_change": float(brl_20.iloc[i]),
        })

    if not events:
        return mark_failed(
            sid,
            "no joint KC-percentile + BRL signal events fired",
            extra={
                "rule": "Short KC=F 30d on KC 3y-90th + BRL down 2% in 20d",
                "mechanism": "Crowded spec long + currency tailwind for producer = mean-reversion in arabica",
                "source": "PL612 catalog; FRED DEXBZUS",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * kc_r.reindex(positions.index).fillna(0.0)
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
        positions=active_pos.abs(),
        cost_bps=10,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.abs().mean()) if len(active_pos) else 0.0
    m["events"] = events[:20]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Short KC=F at next-day close when KC=F front-month price percentile over trailing 756 trading days >= 0.90 AND DEXBZUS down >2% over trailing 20 trading days (BRL strengthening). Hold 30 trading days; exit on -10% KC close (profit target) or KC at 252d high (signal failure).",
            "mechanism": "Spec-long managed-money positioning in arabica builds with rising prices; price-percentile is a high-correlation proxy for CFTC COT crowding (since data not in FRED). BRL strength acts as supplementary fundamental: Brazilian-producer currency strength dampens producer-side selling, but combined with extreme spec longs creates mean-reversion risk. Counter to long_commodity_complex default.",
            "source": "PL612 idea catalog; FRED DEXBZUS; yfinance KC=F",
            "caveats": "Price-percentile substitutes for CFTC COT managed-money net-long. Coffee futures highly volatile around Brazilian frost / drought events. Counter-signal against broad long commodity bias.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
