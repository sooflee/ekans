"""
G-14 Pre-FOMC drift, regime-conditional on DGS2 dovishness.

Rule:
- Long SPY from close of T-1 to close of T (FOMC decision day).
- CONDITIONAL: only enter when 1-month change in DGS2 (FRED) is below trailing-1y median (dovish
  regime, rates have been falling).
- Compare to unconditional version.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed
from _fomc_dates import fomc_dates_ts


def main():
    px = load_prices(["SPY"], start="2000-01-01")["SPY"]
    rets = px.pct_change()
    idx = rets.index

    try:
        dgs2 = load_fred(["DGS2"], start="1998-01-01")["DGS2"].dropna()
    except Exception as e:
        return mark_failed("G14_prefomc_regime", f"FRED DGS2 load failed: {e}")
    # 1-month change in DGS2 (21 trading-day approximation on FRED daily)
    dgs2 = dgs2.reindex(idx, method="ffill")
    dgs2_chg = dgs2 - dgs2.shift(21)
    rolling_median = dgs2_chg.rolling(252, min_periods=120).median()
    dovish = (dgs2_chg < rolling_median)

    fomc = sorted([d for d in fomc_dates_ts() if idx[0] <= d <= idx[-1]])
    pos_u = pd.Series(0.0, index=idx)
    pos_c = pd.Series(0.0, index=idx)
    n_u = 0
    n_c = 0
    for d in fomc:
        loc = idx.searchsorted(d, side="right") - 1
        if loc <= 0:
            continue
        t_day = idx[loc]
        if t_day != d:
            continue
        t_minus_1 = idx[loc - 1]
        pos_u.loc[t_minus_1] = 1.0
        n_u += 1
        if bool(dovish.loc[t_minus_1]) if t_minus_1 in dovish.index else False:
            pos_c.loc[t_minus_1] = 1.0
            n_c += 1

    pnl_u = (pos_u.shift(1).fillna(0) * rets).dropna()
    pnl_c = (pos_c.shift(1).fillna(0) * rets).dropna()
    m_u = compute_metrics(pnl_u, benchmark=rets.dropna(), name="G14 Pre-FOMC unconditional")
    m_c = compute_metrics(pnl_c, benchmark=rets.dropna(), name="G14 Pre-FOMC dovish only")
    print_metrics(m_u)
    print_metrics(m_c)

    save_result("G14_prefomc_regime", m_c, extra={
        "status": "ok",
        "rule": "Long SPY close-of-T-1 to close-of-T at FOMC. Conditional: only when 1m change in "
                "DGS2 (FRED) is below trailing-1y median (dovish regime).",
        "universe": "SPY",
        "n_events_unconditional": n_u,
        "n_events_dovish": n_c,
        "unconditional_metrics": m_u,
        "source": "Lucca-Moench 2015; regime-cond variant",
    })


if __name__ == "__main__":
    main()
