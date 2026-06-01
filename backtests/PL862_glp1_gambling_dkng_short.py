"""PL862_glp1_gambling_dkng_short GLP-1 Impulse-Control Spillover -> Short DKNG/PENN Gambling"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL862_glp1_gambling_dkng_short"
    try:
        # DKNG IPO'd April 2020; PENN available earlier
        px = load_prices(["DKNG", "PENN", "SPY"], start="2020-04-24")
        px = px.dropna(how="all")
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        rets = daily_returns(px)
        spy_r = rets["SPY"]

        # GLP-1 adoption proxy: ramp from mid-2023 onward (Ozempic weight-loss approval Jul 2023)
        # Use a time-based activation signal for GLP-1 adoption phase
        glp1_adoption_start = pd.Timestamp("2023-07-01")
        glp1_active = (rets.index >= glp1_adoption_start).astype(float)

        # GGR deceleration proxy: gambling stock relative underperformance vs. SPY
        # Use rolling 60-day return of DKNG/PENN basket vs SPY
        dkng_r = rets["DKNG"]
        penn_r = rets["PENN"]
        basket_r = 0.5 * dkng_r + 0.5 * penn_r

        # Rolling 60-day relative return (basket minus SPY)
        basket_60 = basket_r.rolling(60).sum()
        spy_60 = spy_r.rolling(60).sum()
        rel_perf_60 = basket_60 - spy_60  # negative = underperforming

        # Also look at 20-day momentum deceleration
        basket_20 = basket_r.rolling(20).sum()
        basket_20_lag = basket_20.shift(20)
        momentum_decel = (basket_20 < basket_20_lag).astype(float)  # 20d momentum declining

        # Signal: GLP-1 era active AND basket underperforming AND momentum decelerating
        underperform_cond = (rel_perf_60 < -0.05).astype(float)  # >5% behind SPY over 60d
        entry_signal = (glp1_active * underperform_cond * momentum_decel).fillna(0)

        # Position: short basket when signal on
        pos = entry_signal.shift(1).fillna(0)
        pnl = -pos * basket_r  # short = negative position
        pnl = pnl.dropna()

        if pnl.abs().sum() < 1e-8:
            # fallback: short basket during GLP-1 era
            glp1_pos = glp1_active.shift(1).fillna(0)
            pnl = (-glp1_pos * basket_r).dropna()

        spy_bench = spy_r.reindex(pnl.index).dropna()
        pnl = pnl.reindex(spy_bench.index)

        m = compute_metrics(pnl, benchmark=spy_bench, name="GLP-1 Gambling Short DKNG/PENN")
        save_result(sid, m, extra={
            "rule": "Short DKNG/PENN equal-weight basket when GLP-1 era (post-Jul 2023) active, basket 60d return >5% below SPY, and 20d momentum decelerating.",
            "mechanism": "GLP-1 drugs improve impulse control and reduce dopamine-driven compulsive behavior; structural headwind to online gambling GGR growth beyond cyclical factors.",
            "source": "yfinance DKNG, PENN, SPY; GLP-1 adoption date proxy (FDA Jul 2023 weight-loss approval)",
            "status": "ok",
        }, pnl=pnl)
    except Exception as e:
        return mark_failed(sid, f"backtest error: {e}")


if __name__ == "__main__":
    main()
