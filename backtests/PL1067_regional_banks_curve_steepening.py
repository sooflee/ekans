"""PL1067_regional_banks_curve_steepening Regional banks vs curve steepening (KRE).

Regional-bank net interest margins expand when the curve steepens from positive territory.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1067_regional_banks_curve_steepening"
    try:
        px = load_prices(['KRE', 'SPY'], start="2005-01-01")
        fr = load_fred(['T10Y2Y'], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    c = fr['T10Y2Y'].reindex(px.index).ffill()
    regime = (c.diff(21) > 0) & (c > 0)

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['KRE']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[30:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Regional banks vs curve steepening (KRE)', positions=pos)
    save_result(sid, m, extra={
        "rule": '2s10s (T10Y2Y) steepening (21d change > 0) AND positive -> long KRE, else flat.',
        "mechanism": 'Regional-bank net interest margins expand when the curve steepens from positive territory.',
        "source": 'Bank-NIM / yield-curve literature; FRED T10Y2Y.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
