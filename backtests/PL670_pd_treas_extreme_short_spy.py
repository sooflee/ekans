"""PL670 — Primary Dealer Net Treasury Extreme - Counter-Signal Short SPY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL670_pd_treas_extreme_short_spy"

    # The NY Fed Primary Dealer data isn't directly available in FRED or yfinance.
    # We proxy with a spread between 2Y and 10Y Treasury yields (FRED):
    # When primary dealers pile into long Treasuries it typically coincides with
    # yield curve inversion deepening or flight-to-safety events.
    # Proxy signal: 10Y-2Y spread goes very negative (extreme flattening/inversion)
    # Coupled with a 3-month z-score of the spread being below -2, short SPY.
    try:
        t10 = load_fred("GS10", start="2015-01-01").squeeze()
        t2 = load_fred("GS2", start="2015-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED load error: {e}")

    spread = (t10 - t2).dropna()
    spread = spread.reindex(pd.date_range(spread.index.min(), spread.index.max(), freq="B")).ffill()

    # Compute rolling 1-year (252 day) z-score of the spread
    roll_mean = spread.rolling(252).mean()
    roll_std = spread.rolling(252).std()
    z = (spread - roll_mean) / roll_std.replace(0, np.nan)

    # Trigger: z-score < -2 (primary dealers typically max-long Treasuries in extreme inversion)
    # Enter on that day; hold 30 trading days. Short SPY, hedge 50% long TLT.
    try:
        px = load_prices(["SPY", "TLT"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    tlt_r = ret["TLT"]

    # Net PnL per day: short SPY + 50% long TLT = -spy_ret + 0.5*tlt_ret
    ls_r = -spy_r + 0.5 * tlt_r

    # Align z to trading dates
    z_trade = z.reindex(spy_r.index, method="ffill")
    trigger = (z_trade < -2.0)

    hold = 30
    pnl = pd.Series(0.0, index=spy_r.index)
    evts = []
    last_entry = None

    for i in range(len(trigger)):
        if not trigger.iloc[i]:
            continue
        dt = trigger.index[i]
        if last_entry is not None and (dt.to_pydatetime() - last_entry).days < hold * 1.5:
            continue  # avoid overlapping windows
        last_entry = dt.to_pydatetime()
        p = spy_r.index.get_loc(dt)
        ep = min(p + hold, len(spy_r))
        window = ls_r.iloc[p:ep]
        pnl.iloc[p:ep] += window.values[:ep - p]

        cum_ret = float((1 + window).prod() - 1)
        spy_c = float((1 + spy_r.iloc[p:ep]).prod() - 1)
        evts.append({
            "trigger_date": str(dt.date()),
            "spread_z": round(float(z_trade.iloc[i]), 3),
            "ls_return": round(cum_ret, 4),
            "spy_return": round(spy_c, 4),
        })

    print(f"Events triggered: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no trigger events")

    active = pnl[pnl != 0]
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="Primary Dealer Treasury Extreme → Short SPY")
    avg_ret = float(np.mean([e["ls_return"] for e in evts]))
    win_rate = float(np.mean([e["ls_return"] > 0 for e in evts]))

    save_result(sid, m, extra={
        "rule": "When 10Y-2Y yield spread z-score (252d window) < -2, short SPY 50% + long TLT 50% for 30 trading days",
        "mechanism": "Primary dealers accumulate long Treasuries at extremes of yield curve inversion; crowded positions unwind when growth fears peak, causing both SPY underperformance and TLT strength",
        "source": "FRED GS10/GS2 proxy for NY Fed primary dealer positions; yfinance SPY/TLT",
        "n_events": len(evts),
        "avg_event_return": round(avg_ret, 4),
        "event_win_rate": round(win_rate, 4),
        "events": evts,
    })
    print(f"Done: Sharpe={m.get('sharpe'):.2f}, CAGR={m.get('cagr'):.2%}, Events={len(evts)}")


if __name__ == "__main__":
    main()
