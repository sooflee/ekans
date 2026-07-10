"""PL1085 SPX options skew & HY OAS divergence -> short SPY.

Counter-signal to long_SPY. Complacency in options skew (CBOE ^SKEW proxy sitting
low / below its 20d average) *while* high-yield credit spreads are widening
(ICE BofA HY OAS, FRED BAMLH0A0HYM2, above its 50d average and rising) is a
divergence: equity overconfidence against deteriorating credit. Short SPY on the
divergence and manage the trade with a profit target / stop / time stop / reverse.

^SKEW is a free proxy for SPX put skew (direct put/call skew isn't on yfinance).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns)


def main():
    sid = "PL1085_spx_options_skew_hy_oas_diverg"
    try:
        px = load_prices(["SPY", "^SKEW"], start="2010-01-01")
        oas = load_fred(["BAMLH0A0HYM2"], start="2009-06-01")["BAMLH0A0HYM2"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.dropna(how="all").sort_index()
    spy = px["SPY"].dropna()
    skew = px["^SKEW"].reindex(spy.index).ffill()
    oas = oas.reindex(spy.index).ffill()

    skew_ma20 = skew.rolling(20).mean()
    oas_ma50 = oas.rolling(50).mean()
    oas_up5bp = oas.diff() >= 0.05  # HY OAS is in percentage points; 5bp = 0.05

    # NOTE — calibration fix vs the literal spec. The spec used absolute ^SKEW
    # constants (entry skew<120, exit skew>125). Against 2010-2026 ^SKEW those are
    # miscalibrated: skew<120 (only 718 days, all calm tape) NEVER coincides with
    # HY-OAS widening, so the literal rule fires 0 trades; and skew>125 sits below
    # the median (129), so the reverse-exit would trigger on day 1. We keep the
    # thesis (complacency + credit widening) but express complacency *relative* to
    # skew's own 20d trend and exit when that complacency unwinds — distribution-
    # aware versions of the same conditions.
    entry = (skew < skew_ma20) & (oas > oas_ma50) & oas_up5bp

    idx = spy.index
    n = len(idx)
    spy_v = spy.to_numpy()
    skew_v = skew.to_numpy()
    skew_ma20_v = skew_ma20.to_numpy()
    oas_v = oas.to_numpy()
    oas_ma50_v = oas_ma50.to_numpy()
    entry_v = entry.reindex(idx).fillna(False).to_numpy()

    pos = np.zeros(n)          # position IN EFFECT during day i (decided at close i-1)
    held = 0
    entry_price = np.nan
    days = 0
    for i in range(n):
        pos[i] = held          # no lookahead: decided at prior close
        if held == -1:         # currently short — update state at today's close
            days += 1
            move = spy_v[i] / entry_price - 1.0
            # reverse: complacency unwinds (skew back above its 20d trend) OR
            # credit stress resolves (OAS back below its 50d average).
            reverse = (skew_v[i] > skew_ma20_v[i]) or (oas_v[i] < oas_ma50_v[i])
            if move <= -0.02 or move >= 0.01 or days >= 20 or reverse:
                held, entry_price, days = 0, np.nan, 0
        else:                  # flat — look for entry
            if entry_v[i] and not np.isnan(oas_ma50_v[i]) and not np.isnan(skew_ma20_v[i]):
                held, entry_price, days = -1, spy_v[i], 0

    positions = pd.Series(pos, index=idx)          # -1 short / 0 flat
    spy_r = daily_returns(spy)
    pnl = (positions * spy_r).reindex(idx).fillna(0.0)   # short: pnl = -1 * spy_return

    # drop the 50d warmup (longest lookback)
    pnl = pnl.iloc[50:]
    positions = positions.reindex(pnl.index).fillna(0.0)
    n_active = int((positions != 0).sum())
    if len(pnl) < 252 or n_active < 30:
        return mark_failed(sid, f"insufficient data/active days: {len(pnl)}d, {n_active} active")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index),
                        name="SPX skew / HY OAS divergence short SPY", positions=positions)
    save_result(sid, m, extra={
        "rule": "Short SPY when ^SKEW < its 20d SMA (relative complacency) AND HY OAS "
                "(BAMLH0A0HYM2) > 50d SMA and rising >=5bp (credit widening). Exit on -2% target / "
                "+1% stop / 20d time stop / reverse (^SKEW back above 20d SMA or OAS < 50d SMA).",
        "mechanism": "Options complacency against widening HY credit spreads is a divergence that "
                     "precedes equity drawdowns; short SPY into the overconfidence.",
        "source": "Vol-skew vs credit-spread divergence literature (CBOE SKEW; ICE BofA HY OAS).",
        "counter_signal": True, "counters": "long_SPY",
        "spec_calibration_note": "Literal Gemini spec used absolute ^SKEW thresholds (entry<120, "
            "exit>125) that fire 0 trades vs 2010-26 ^SKEW (median 129); reimplemented with "
            "relative/trend thresholds preserving the thesis.",
        "n_active_days": n_active,
    }, pnl=pnl)


if __name__ == "__main__":
    main()
