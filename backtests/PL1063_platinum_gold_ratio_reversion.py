"""PL1063_platinum_gold_ratio_reversion Platinum/gold ratio bottom-decile reversion.

A stretched-cheap platinum/gold ratio that is turning up marks a mean-reverting entry; platinum re-rates toward gold as the discount closes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1063_platinum_gold_ratio_reversion"
    try:
        px = load_prices(['PPLT', 'GLD', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    ratio = px['PPLT'] / px['GLD']
    q20 = ratio.rolling(756, min_periods=252).quantile(0.20)
    regime = (ratio <= q20) & (ratio.pct_change(21) > 0)

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['PPLT']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[756:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Platinum/gold ratio bottom-decile reversion', positions=pos)
    save_result(sid, m, extra={
        "rule": 'PPLT/GLD in trailing-3y bottom 20th pct AND 21d momentum > 0 -> long PPLT, else flat.',
        "mechanism": 'A stretched-cheap platinum/gold ratio that is turning up marks a mean-reverting entry; platinum re-rates toward gold as the discount closes.',
        "source": 'Precious-metals substitution / ratio-reversion literature.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
