"""PL1065_reit_vs_real_yield REIT vs real-yield regime (VNQ).

REITs are long-duration cash-flow assets; a falling real yield lowers the discount rate and drives VNQ outperformance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1065_reit_vs_real_yield"
    try:
        px = load_prices(['VNQ', 'SPY'], start="2005-01-01")
        fr = load_fred(['DFII10'], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    ry = fr['DFII10'].reindex(px.index).ffill()
    regime = (ry.diff(21) < 0) & (ry < ry.rolling(200).mean())

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['VNQ']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[210:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='REIT vs real-yield regime (VNQ)', positions=pos)
    save_result(sid, m, extra={
        "rule": '10y real yield (DFII10) falling (21d) AND below its 200d avg -> long VNQ, else flat.',
        "mechanism": 'REITs are long-duration cash-flow assets; a falling real yield lowers the discount rate and drives VNQ outperformance.',
        "source": 'Duration / discount-rate REIT literature; FRED DFII10.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
