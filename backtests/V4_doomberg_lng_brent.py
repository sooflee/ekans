"""
V4 Doomberg LNG/Brent regime (UNG vs BNO proxy)
Original signal references JKM LNG vs Brent, which is not on yfinance.
Substitution: use UNG (US natgas ETF) / BNO (Brent ETF) ratio as a tradeable
proxy for natgas-vs-oil dislocations.
Rule:
  Compute UNG/BNO ratio. Compute its 3y (756 trading day) rolling mean and std.
  When ratio z-score < -2.0 (UNG is 2 sigma cheap vs BNO), go long UNG / short
  BNO until z-score reverts back through 0.
Mechanism (Doomberg): cross-fuel arbitrage. LNG/natgas vs oil/Brent moves apart
during regime-specific shocks (e.g. mild winters, glut, geopolitical pipe cuts);
mean-reverts as substitution kicks in. UNG/BNO is the cleanest US-listed proxy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, rolling_zscore,
)


def main():
    try:
        px = load_prices(["UNG", "BNO"], start="2010-06-01")
    except Exception as e:
        return mark_failed("V4_doomberg_lng_brent", f"data load: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("V4_doomberg_lng_brent", "no UNG/BNO overlap")

    ratio = px["UNG"] / px["BNO"]
    z = rolling_zscore(ratio, 756)  # 3y window

    rets = px.pct_change()
    # State 1 = pair on (long UNG, short BNO equal-notional);
    # State 0 = flat.
    state = 0
    pos = pd.DataFrame(0.0, index=px.index, columns=["UNG", "BNO"])
    for i in range(len(px)):
        zi = z.iloc[i]
        if not np.isnan(zi):
            if state == 0 and zi < -2.0:
                state = 1
            elif state == 1 and zi >= 0.0:
                state = 0
        if state == 1:
            pos.iloc[i] = [1.0, -1.0]
    # 0.5x each leg keeps gross exposure ~1
    pos *= 0.5

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    # benchmark = SPY
    try:
        spy = load_prices(["SPY"], start=str(pnl.index.min().date())).iloc[:, 0].pct_change()
    except Exception:
        spy = None
    m = compute_metrics(pnl, benchmark=spy.reindex(pnl.index) if spy is not None else None,
                        name="V4 UNG/BNO mean-reversion")
    print_metrics(m)
    n_trigger = int((z < -2.0).sum())
    save_result("V4_doomberg_lng_brent", m, extra={
        "status": "ok",
        "rule": "Long UNG / short BNO (0.5x each) when 3y z-score of UNG/BNO < -2.0; exit when z >= 0.",
        "mechanism": "Cross-fuel mean reversion: natgas vs oil dislocate during regime shocks, then revert as substitution kicks in.",
        "source": "Doomberg, YouTube interview round 2 (Phase 1V). Original signal used JKM/Brent; substituted UNG/BNO as accessible proxy.",
        "substitution": "JKM LNG futures not on yfinance; used UNG (US natgas ETF) / BNO (Brent ETF) ratio instead.",
        "n_trigger_days": n_trigger,
    })


if __name__ == "__main__":
    main()
