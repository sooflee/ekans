"""
AA6 Quad-witch Monday reversal.
Quad-witch = 3rd Friday of Mar/Jun/Sep/Dec (simultaneous expiry of
index futures, index options, single-stock futures, stock options).

Spec: SPY conditional Monday reversal — if SPY *down* on the quad-witch week
→ long SPY on the following Monday; if *up* → short SPY on the Monday.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calendar
import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def third_friday(year, month):
    cal = calendar.Calendar(firstweekday=0)
    fridays = [d for d in cal.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    return fridays[2] if len(fridays) >= 3 else None


def main():
    try:
        px = load_prices(["SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed("AA6_quad_witch_reversal", f"data load failed: {e}")
    spy = px["SPY"].dropna()
    rets = spy.pct_change()
    idx = spy.index

    quad_months = [3, 6, 9, 12]
    pos = pd.Series(0.0, index=idx)
    n_events = 0
    for y in range(idx[0].year, idx[-1].year + 1):
        for m in quad_months:
            tf = third_friday(y, m)
            if tf is None:
                continue
            qw = pd.Timestamp(tf)
            i_qw = idx.searchsorted(qw, side="right") - 1
            if i_qw < 5 or i_qw >= len(idx) - 1:
                continue
            # Week return: Mon→Fri of quad-witch week. The Monday is i_qw - 4 (approx),
            # but use the latest Monday at or before qw - 4 days for safety.
            # Simpler: compare SPY close on quad-witch Friday vs SPY close on prior Friday (5 trading days back).
            wk_start_close = spy.iloc[i_qw - 5]
            wk_end_close = spy.iloc[i_qw]
            wk_ret = wk_end_close / wk_start_close - 1
            # Monday following = next trading day after qw.
            i_mon = i_qw + 1
            if i_mon >= len(idx):
                continue
            # Position is set at close of i_qw, earns return on i_mon.
            if wk_ret < 0:
                pos.iloc[i_qw] = 1.0   # long Monday
            elif wk_ret > 0:
                pos.iloc[i_qw] = -1.0  # short Monday
            n_events += 1

    if n_events < 30:
        return mark_failed("AA6_quad_witch_reversal", f"too few events ({n_events})")

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AA6 quad-witch Mon reversal")
    print_metrics(m)
    save_result("AA6_quad_witch_reversal", m, extra={
        "status": "ok",
        "rule": "On Mon after quad-witch Fri: long SPY if quad-witch-week ret < 0, short if > 0.",
        "universe": "SPY",
        "source": "Quad-witch options-expiry literature (3rd Fri Mar/Jun/Sep/Dec).",
        "n_events": n_events,
    })


if __name__ == "__main__":
    main()
