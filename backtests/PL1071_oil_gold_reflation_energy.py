"""PL1071_oil_gold_reflation_energy Oil/gold reflation ratio -> energy (XLE).

A rising oil/gold ratio marks reflationary, growth-positive regimes that favor energy equities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1071_oil_gold_reflation_energy"
    try:
        px = load_prices(['XLE', 'USO', 'GLD', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    ratio = px['USO'] / px['GLD']
    regime = (ratio.pct_change(21) > 0) & (ratio > ratio.rolling(200).mean())

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['XLE']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[210:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Oil/gold reflation ratio -> energy (XLE)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'USO/GLD rising (21d) AND above its 200d SMA -> long XLE, else flat.',
        "mechanism": 'A rising oil/gold ratio marks reflationary, growth-positive regimes that favor energy equities.',
        "source": 'Commodity-ratio reflation literature.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
