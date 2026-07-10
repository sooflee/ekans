"""PL1072_orange_juice_supply_squeeze Orange juice supply-squeeze momentum (OJ=F).

Thin inventories and inelastic demand make OJ supply-shock breakouts persist.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1072_orange_juice_supply_squeeze"
    try:
        px = load_prices(['OJ=F', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    regime = (px['OJ=F'].pct_change(63) > 0) & (px['OJ=F'] > px['OJ=F'].rolling(200).mean())

    pos = regime.shift(1).fillna(False).astype(float)
    asset_r = rets['OJ=F']
    pnl = (pos * asset_r).reindex(px.index).fillna(0.0)

    pnl = pnl.iloc[210:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl.loc[lambda s: s != 0]) < 30 or len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup/active data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Orange juice supply-squeeze momentum (OJ=F)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'OJ=F 63d momentum > 0 AND price > 200d SMA -> long OJ futures, else flat.',
        "mechanism": 'Thin inventories and inelastic demand make OJ supply-shock breakouts persist.',
        "source": 'Soft-commodity supply-shock momentum.',
        "active_frac_check": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
