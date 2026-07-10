"""PL1077_copper_miner_operating_leverage Copper-miner operating leverage (COPX/CPER).

Copper miners re-rate non-linearly to the copper price via operating leverage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1077_copper_miner_operating_leverage"
    try:
        px = load_prices(['COPX', 'CPER', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    ratio = px['COPX'] / px['CPER']
    regime = (px['CPER'] > px['CPER'].rolling(200).mean()) & (ratio.pct_change(63) > 0)

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['COPX']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[210:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Copper-miner operating leverage (COPX/CPER)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'Copper (CPER) > 200d SMA AND COPX/CPER 63d momentum > 0 -> long COPX, else flat.',
        "mechanism": 'Copper miners re-rate non-linearly to the copper price via operating leverage.',
        "source": 'Copper-miner leverage literature.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
