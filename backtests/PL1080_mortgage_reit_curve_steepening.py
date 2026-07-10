"""PL1080_mortgage_reit_curve_steepening Mortgage REITs vs curve steepening (REM).

Mortgage REITs earn the spread between long mortgage assets and short funding; a steepening curve widens it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1080_mortgage_reit_curve_steepening"
    try:
        px = load_prices(['REM', 'SPY'], start="2005-01-01")
        fr = load_fred(['T10Y2Y'], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    c = fr['T10Y2Y'].reindex(px.index).ffill()
    regime = c.diff(21) > 0

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['REM']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[30:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Mortgage REITs vs curve steepening (REM)', positions=pos)
    save_result(sid, m, extra={
        "rule": '2s10s (T10Y2Y) steepening (21d change > 0) -> long REM, else flat.',
        "mechanism": 'Mortgage REITs earn the spread between long mortgage assets and short funding; a steepening curve widens it.',
        "source": 'mREIT spread / yield-curve literature; FRED T10Y2Y.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
