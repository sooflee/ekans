"""PL1069_defensive_sector_leadership_riskoff Defensive-sector leadership = risk-off (XLP+XLU+XLV).

Defensive-sector leadership signals a risk-off regime that blind long-SPY exposure misses; step aside to cash.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1069_defensive_sector_leadership_riskoff"
    try:
        px = load_prices(['XLP', 'XLU', 'XLV', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    defbasket = (px['XLP'].pct_change(21) + px['XLU'].pct_change(21) + px['XLV'].pct_change(21)) / 3
    defensive = defbasket > px['SPY'].pct_change(21)

    defensive = defensive.shift(1).reindex(px.index).fillna(False)
    risk_r = rets['SPY'].reindex(px.index).fillna(0.0)
    hedge_r = 0.0
    pnl = pd.Series(np.where(defensive.to_numpy(), hedge_r, risk_r.to_numpy()),
                    index=px.index)
    # position proxy: 0/1 "in the defensive leg" — one unit of turnover per switch.
    pos = defensive.astype(float)

    pnl = pnl.iloc[30:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Defensive-sector leadership = risk-off (XLP+XLU+XLV)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'mean 21d return of (XLP,XLU,XLV) > SPY 21d return -> go to cash, else hold SPY.',
        "mechanism": 'Defensive-sector leadership signals a risk-off regime that blind long-SPY exposure misses; step aside to cash.',
        "source": 'Defensive-rotation / risk-off literature.',
        "counter_signal": True,
        "counters": 'long_SPY',
        "frac_days_defensive": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
