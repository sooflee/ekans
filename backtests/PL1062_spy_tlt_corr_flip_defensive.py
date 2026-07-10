"""PL1062 Stock-bond correlation flip -> defensive gold.

Counter-signal to "long SPY". When the rolling 60d correlation of SPY and TLT
daily returns turns positive, bonds have stopped hedging equities (the 2022 /
inflation-vol regime). On those days hold GLD instead of SPY; otherwise hold SPY.
The thesis is that switching into gold exactly when the 60/40 hedge breaks avoids
the rate-driven drawdowns pure trend-following long signals ride straight down.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed,
                     daily_returns)


def main():
    sid = "PL1062_spy_tlt_corr_flip_defensive"
    try:
        px = load_prices(["SPY", "TLT", "GLD"], start="2010-01-01")
        rets = daily_returns(px).dropna()
        spy_r, tlt_r, gld_r = rets["SPY"], rets["TLT"], rets["GLD"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # 60d rolling correlation of SPY vs TLT daily returns.
    corr = spy_r.rolling(60).corr(tlt_r)

    # Regime: corr > 0 -> defensive (GLD); else risk-on (SPY).
    # Shift the regime by 1 day so today's position uses only past data.
    defensive = (corr > 0).shift(1).fillna(False)

    pnl = pd.Series(np.where(defensive, gld_r, spy_r), index=rets.index)
    # Positions for turnover/cost accounting: +1 SPY-leg vs the GLD-leg switch.
    # Use a single 0/1 "in GLD" indicator as the position proxy (each flip = 1 unit turnover).
    positions = defensive.astype(float)

    pnl = pnl.loc[corr.dropna().index[0]:]  # drop the 60d warmup
    if len(pnl) < 252:
        return mark_failed(sid, f"insufficient post-warmup data: {len(pnl)} days")

    spy_bench = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_bench,
                        name="Stock-bond corr flip defensive", positions=positions)
    frac_def = float(defensive.reindex(pnl.index).mean())
    save_result(sid, m, extra={
        "rule": "60d corr(SPY,TLT) > 0 -> hold GLD, else hold SPY (regime shifted 1d).",
        "mechanism": "Positive stock-bond correlation marks the rate/inflation-vol "
                     "regime where bonds stop hedging equities and 60/40 de-grosses; "
                     "rotating to gold sidesteps the drawdown trend-long signals miss.",
        "source": "Diversification-breakdown literature; 2022 60/40 drawdown.",
        "counter_signal": True, "counters": "long_SPY",
        "frac_days_defensive": frac_def,
    })


if __name__ == "__main__":
    main()
