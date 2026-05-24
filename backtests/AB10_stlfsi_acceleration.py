"""
AB10 STLFSI acceleration
FRED STLFSI4 (weekly St. Louis Fed Financial Stress Index). Compute 4-week change.
When 4w_change > +0.30 AND level < +1, short SPY (rotate to TLT) for 6 weeks (30 trading days).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB10_stlfsi_acceleration"
    try:
        stl = load_fred(["STLFSI4"], start="2000-01-01").iloc[:, 0].dropna().rename("STLFSI")
        spy = load_prices(["SPY"], start="2000-01-01").iloc[:, 0].rename("SPY")
        tlt = load_prices(["TLT"], start="2002-07-01").iloc[:, 0].rename("TLT")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    # weekly delta
    delta = (stl - stl.shift(4)).rename("d4w")
    # daily forward-filled (weekly series)
    spy = spy.dropna()
    tlt = tlt.dropna()
    idx = spy.index.union(tlt.index)
    stl_d = stl.reindex(idx, method="ffill")
    delta_d = delta.reindex(idx, method="ffill")

    cond_d = (delta_d > 0.30) & (stl_d < 1.0)
    cond_d = cond_d.fillna(False)

    rets_spy = spy.pct_change()
    rets_tlt = tlt.pct_change()

    pos_spy = pd.Series(0.0, index=idx)
    pos_tlt = pd.Series(0.0, index=idx)
    rem = 0
    last_trigger_value = None
    for i in range(len(idx)):
        # trigger fires on new STLFSI release date (i.e., when stl_d changes)
        # but cond_d is daily true throughout the lookback window unless we detect releases.
        # Simpler: re-arm rem=30 whenever cond_d is True; that's persistent (~6 weeks=30 td).
        if bool(cond_d.iloc[i]):
            rem = 30
        if rem > 0:
            pos_spy.iloc[i] = -1.0
            pos_tlt.iloc[i] = 1.0  # rotate to TLT
            rem -= 1

    # We backtest the "rotate to TLT" version: -SPY + TLT, net 0 gross 2
    # Net daily PnL = -1 * spy_ret + 1 * tlt_ret (half-leverage interpretation = / 2 to keep gross 1)
    spy_part = (pos_spy.shift(1) * rets_spy).fillna(0.0)
    tlt_part = (pos_tlt.shift(1) * rets_tlt).fillna(0.0)
    pnl = ((spy_part + tlt_part) / 2.0).reindex(spy.index).dropna()

    bench = rets_spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AB10 STLFSI acceleration (rotate SPY->TLT)")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "STLFSI4 4-week change > +0.30 AND level < +1 -> short SPY + long TLT (gross 1, half-each) 6 weeks (30 td).",
        "data_source": "FRED STLFSI4 (weekly)",
        "n_active_days": int(cond_d.sum()),
    })


if __name__ == "__main__":
    main()
