"""PL589_0dte_share_up_tape_spy_next_day_short - 0DTE SPX Share >55% on Up-Tape -> Short SPY Next Day

Event study: on days where 0DTE SPX share >55% AND SPX same-day return >+0.4%,
short SPY at the close of D for one trading day (close-to-close approximation
of next-day open-to-mid-morning short).

Since granular 0DTE share data isn't readily available daily through 1990,
we proxy with a heuristic: any day in 2023-2024 with daily SPY return > +0.4%
during periods when 0DTE share regularly exceeded 55% (per CBOE reports,
roughly mid-2023 through 2024). This is a representative-window proxy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL589_0dte_share_up_tape_spy_next_day_short"
    try:
        px = load_prices(["SPY"], start="2022-06-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    spy = px["SPY"].dropna()
    ret = spy.pct_change().dropna()

    # Approximation: 0DTE share regularly >55% from June 2023 onward per CBOE.
    # Use ret > +0.4% as the up-tape filter within that window.
    window_start = pd.Timestamp("2023-06-01")
    window_end = pd.Timestamp("2024-12-31")
    in_window = (ret.index >= window_start) & (ret.index <= window_end)
    up_tape = ret > 0.004
    trigger = in_window & up_tape

    trigger_dates = ret.index[trigger]
    if len(trigger_dates) < 10:
        return mark_failed(sid, f"too few triggers ({len(trigger_dates)})")

    # Short next day's SPY close-to-close (approximation of open-to-mid-morning)
    pnl = pd.Series(0.0, index=ret.index)
    events = []
    for td in trigger_dates:
        loc = ret.index.get_loc(td)
        if loc + 1 >= len(ret):
            continue
        next_ret = ret.iloc[loc + 1]
        short_ret = -float(next_ret)
        pnl.iloc[loc + 1] = short_ret
        events.append({
            "trigger_date": str(td.date()),
            "trigger_ret": round(float(ret.iloc[loc]), 4),
            "short_pnl": round(short_ret, 4),
        })

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=ret, name="0DTE Up-Tape Short SPY Next Day")
    save_result(sid, m, extra={
        "rule": "Short SPY at next-day close (proxy for open-to-mid-morning) when 0DTE share >55% AND SPX day return > +0.4%.",
        "mechanism": "Dealer gamma flip + 0DTE-induced pin pressure; up-tape exhaustion next morning.",
        "source": "CBOE daily 0DTE share (proxied by mid-2023 onward window) + yfinance SPY",
        "n_events": len(events),
        "events": events[:50],
        "avg_short_pnl": round(float(np.mean([e["short_pnl"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["short_pnl"] > 0 for e in events])), 4),
        "caveat": "0DTE share approximated by mid-2023 onward window; CBOE published share-level data would refine this.",
    })
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(in_pos)}")


if __name__ == "__main__":
    main()
