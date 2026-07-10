"""PL1084_japan_equity_yen_weakness Japan equity on yen weakness (EWJ/FXY).

A weakening yen lifts Japanese exporters' translated earnings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1084_japan_equity_yen_weakness"
    try:
        px = load_prices(['EWJ', 'FXY', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    regime = px['FXY'] < px['FXY'].rolling(200).mean()

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['EWJ']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[210:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Japan equity on yen weakness (EWJ/FXY)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'Yen proxy FXY below its 200d SMA -> long EWJ, else flat.',
        "mechanism": "A weakening yen lifts Japanese exporters' translated earnings.",
        "source": 'Japan exporter / yen literature.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
