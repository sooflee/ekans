"""PL1083_semis_relative_weakness_riskoff Semis relative weakness -> risk-off (SOXX).

Semis lead the tech cycle; relative weakness marks the growth engine rolling over. Counter to the crowded long-semis trade.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1083_semis_relative_weakness_riskoff"
    try:
        px = load_prices(['SOXX', 'SPY'], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    rets = daily_returns(px)
    spy_r = rets["SPY"]

    # --- regime ---
    ratio = px['SOXX'] / px['SPY']
    defensive = ratio.pct_change(63) < 0

    defensive = defensive.shift(1).reindex(px.index).fillna(False)
    risk_r = rets['SOXX'].reindex(px.index).fillna(0.0)
    hedge_r = 0.0
    pnl = pd.Series(np.where(defensive.to_numpy(), hedge_r, risk_r.to_numpy()),
                    index=px.index)
    # position proxy: 0/1 "in the defensive leg" — one unit of turnover per switch.
    pos = defensive.astype(float)

    pnl = pnl.iloc[70:]
    pos = pos.reindex(pnl.index).fillna(0.0)
    if len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench, name='Semis relative weakness -> risk-off (SOXX)', positions=pos)
    save_result(sid, m, extra={
        "rule": 'SOXX/SPY 63d momentum < 0 -> go to cash, else hold SOXX.',
        "mechanism": 'Semis lead the tech cycle; relative weakness marks the growth engine rolling over. Counter to the crowded long-semis trade.',
        "source": 'Semiconductor-leadership / tech-cycle literature.',
        "counter_signal": True,
        "counters": 'long_semis',
        "frac_days_defensive": float((pos != 0).mean()),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
