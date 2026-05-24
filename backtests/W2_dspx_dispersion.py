"""
W2 CBOE DSPX (S&P 500 Dispersion Index).

Mechanism: DSPX measures expected dispersion of S&P 500 component returns
(option-implied, similar in spirit to VIX but for cross-sectional dispersion).
When dispersion expectations are low (bottom quartile of trailing 252d), the
market is more index-like, so equal-weight (RSP) tends to underperform cap-
weighted (SPY). When dispersion expectations are HIGH, stock-picking and
equal-weight tilts pay more.

Rule (per spec): when DSPX in trailing-252d BOTTOM quartile, overweight RSP /
underweight SPY (pair PnL = RSP - SPY).

Note from research (Trupti Patel et al., CBOE 2023): the original framing was
"dispersion is a regime variable that predicts equal-weight relative returns".
We implement the spec rule literally and let the data speak.

Source: CBOE DSPX index (^DSPX on Yahoo Finance).
Period: DSPX live since 2018 (yfinance data ~2024 onward; ~500 days available).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        dspx = load_prices(["^DSPX"], start="2018-01-01")
    except Exception as e:
        return mark_failed("W2_dspx_dispersion", f"DSPX fetch failed: {e}")

    if dspx.empty or len(dspx) < 60:
        return mark_failed("W2_dspx_dispersion",
                           f"DSPX series too short: {len(dspx)} rows")

    try:
        px = load_prices(["SPY", "RSP"], start="2018-01-01")
    except Exception as e:
        return mark_failed("W2_dspx_dispersion", f"SPY/RSP fetch failed: {e}")

    df = pd.concat([dspx["^DSPX"].rename("dspx"),
                    px["SPY"].rename("SPY"),
                    px["RSP"].rename("RSP")], axis=1).dropna()
    if len(df) < 100:
        return mark_failed("W2_dspx_dispersion",
                           f"After joining DSPX/SPY/RSP: {len(df)} rows")

    # Trailing-252d quartile rank of DSPX (use 126 if window is short)
    win = 252 if len(df) >= 300 else 126
    rank = df["dspx"].rolling(win).rank(pct=True)
    # Bottom quartile = pct <= 0.25
    low_disp = (rank <= 0.25).astype(float)

    # Pair: long RSP, short SPY (overweight RSP / underweight SPY)
    rsp_ret = df["RSP"].pct_change()
    spy_ret = df["SPY"].pct_change()
    pair_ret = rsp_ret - spy_ret  # daily long-RSP-short-SPY excess

    # Apply signal at t-1, earn at t
    pos = low_disp.shift(1).fillna(0)
    pnl = (pos * pair_ret).dropna()
    pnl = pnl[pnl.index >= df.index[win]]  # warm-up

    if len(pnl) < 30:
        return mark_failed("W2_dspx_dispersion",
                           f"Too few post-warmup days: {len(pnl)}")

    n_active = int((pos.shift(-1) > 0).sum())

    bench = spy_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="W2 DSPX dispersion (RSP-SPY)")
    m["active_days"] = n_active
    m["active_pct"] = float(n_active / max(len(pnl), 1))
    print_metrics(m)

    save_result("W2_dspx_dispersion", m, extra={
        "status": "ok",
        "rule": ("When CBOE DSPX in trailing-252d bottom quartile (low expected "
                 "dispersion), go long RSP and short SPY (pair). Otherwise flat."),
        "mechanism": ("Low option-implied dispersion → equity returns are more "
                       "correlated / index-like → equal-weight (RSP) tilts vs "
                       "cap-weight (SPY) have lower expected payoff. Spec rule "
                       "tests the contrarian direction: take RSP-overweight "
                       "when dispersion expectations are low."),
        "source": ("CBOE DSPX dispersion index (^DSPX on yfinance). "
                    "See CBOE 'Introducing DSPX' (2023) and Driessen-Maenhout-"
                    "Vilkov (RFS 2009) for the dispersion-vs-index-vol theory."),
        "universe": "SPY (cap-weight) vs RSP (equal-weight) pair.",
        "window_days": win,
        "data_points": int(len(df)),
    })


if __name__ == "__main__":
    main()
