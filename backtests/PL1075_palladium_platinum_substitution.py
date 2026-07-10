"""PL1075_palladium_platinum_substitution Palladium/platinum substitution reversion.

A stretched-high palladium/platinum ratio triggers autocatalyst substitution toward cheaper platinum, pulling the ratio back.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1075_palladium_platinum_substitution"
    try:
        px = load_prices(['PALL', 'PPLT', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    ratio = px['PALL'] / px['PPLT']
    q80 = ratio.rolling(756, min_periods=252).quantile(0.80)
    regime = ratio >= q80

    pos = regime.shift(1).fillna(False).astype(float)
    long_r = rets['PPLT']
    short_r = rets['PALL']
    pair_r = (long_r - short_r) / 2.0  # dollar-neutral
    pnl = (pos * pair_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[756:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Palladium/platinum substitution reversion', positions=pos)
    save_result(sid, m, extra={
        "rule": 'PALL/PPLT in trailing-3y top 20th pct -> long PPLT / short PALL (dollar-neutral), else flat.',
        "mechanism": 'A stretched-high palladium/platinum ratio triggers autocatalyst substitution toward cheaper platinum, pulling the ratio back.',
        "source": 'PGM substitution literature.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
