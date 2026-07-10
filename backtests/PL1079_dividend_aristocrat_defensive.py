"""PL1079_dividend_aristocrat_defensive Dividend-aristocrat defensive rotation (NOBL).

Dividend-aristocrat leadership is a late-cycle quality/defensive tell against broad long-SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1079_dividend_aristocrat_defensive"
    try:
        px = load_prices(['NOBL', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    defensive = px['NOBL'].pct_change(63) > px['SPY'].pct_change(63)

    defensive = defensive.shift(1).reindex(px.index).fillna(False)
    risk_r = rets['SPY'].reindex(px.index).fillna(0.0)
    hedge_r = rets['NOBL'].reindex(px.index).fillna(0.0).to_numpy()
    pnl = pd.Series(np.where(defensive.to_numpy(), hedge_r, risk_r.to_numpy()),
                    index=px.index)
    # position proxy: 0/1 "in the defensive leg" — one unit of turnover per switch.
    pos = defensive.astype(float)

    pnl = pnl.iloc[70:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Dividend-aristocrat defensive rotation (NOBL)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'NOBL 63d return > SPY 63d return -> hold NOBL (defensive), else hold SPY.',
        "mechanism": 'Dividend-aristocrat leadership is a late-cycle quality/defensive tell against broad long-SPY.',
        "source": 'Quality / dividend-growth factor literature.',
        "counter_signal": True,
        "counters": 'long_SPY',
        "frac_days_defensive": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
