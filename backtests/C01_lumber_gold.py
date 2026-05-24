"""
C01 Lumber/Gold (Gayed)
13-week return lumber vs gold. If lumber > gold, long SPY for next week; else TLT.
Weekly rebalance, daily PnL.

Lumber: chain LBS=F (legacy, 2005-2023) with LBR=F (new, 2022+).
Gold:   GC=F (fall back to GLD if needed).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def build_lumber():
    legacy = load_prices(["LBS=F"], start="2005-01-01").iloc[:, 0]
    legacy.name = "LUMBER"
    try:
        new = load_prices(["LBR=F"], start="2022-08-01").iloc[:, 0]
        new.name = "LUMBER"
    except Exception:
        new = None
    if new is not None and len(new):
        # Scale new contract to legacy's last price on overlap if any, else just concat
        overlap = legacy.index.intersection(new.index)
        if len(overlap):
            scale = legacy.loc[overlap].iloc[-1] / new.loc[overlap].iloc[-1]
            new_scaled = new * scale
            # use legacy until its end then switch
            last_legacy = legacy.dropna().index.max()
            extension = new_scaled.loc[new_scaled.index > last_legacy]
            return pd.concat([legacy.dropna(), extension])
        else:
            last_legacy_val = legacy.dropna().iloc[-1]
            first_new_val = new.dropna().iloc[0]
            scale = last_legacy_val / first_new_val if first_new_val != 0 else 1.0
            extension = (new * scale).dropna()
            extension = extension.loc[extension.index > legacy.dropna().index.max()]
            return pd.concat([legacy.dropna(), extension])
    return legacy.dropna()


def main():
    try:
        lumber = build_lumber()
        gold = load_prices(["GC=F"], start="2005-01-01").iloc[:, 0]
        gold.name = "GOLD"
        spy = load_prices(["SPY"], start="2005-01-01").iloc[:, 0]
        spy.name = "SPY"
        tlt = load_prices(["TLT"], start="2005-01-01").iloc[:, 0]
        tlt.name = "TLT"
    except Exception as e:
        return mark_failed("C01_lumber_gold", f"data load failed: {e}")

    df = pd.concat([lumber.rename("LUMBER"), gold.rename("GOLD"),
                    spy.rename("SPY"), tlt.rename("TLT")], axis=1).dropna()
    if df.empty:
        return mark_failed("C01_lumber_gold", "no overlap in price frame")

    # 13-week return = ~65 trading days
    look = 65
    r_lumber = df["LUMBER"].pct_change(look)
    r_gold = df["GOLD"].pct_change(look)
    signal = (r_lumber > r_gold).astype(float)  # 1 = long SPY, 0 = TLT

    # Weekly rebalance: take signal at last day of each week
    week = signal.resample("W-FRI").last().reindex(df.index, method="ffill")

    spy_ret = df["SPY"].pct_change()
    tlt_ret = df["TLT"].pct_change()

    pos_spy = week.shift(1)  # apply at next day's open after Friday signal
    pos_tlt = 1.0 - pos_spy
    pnl = pos_spy * spy_ret + pos_tlt * tlt_ret
    pnl = pnl.dropna()

    bench = spy_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C01 Lumber/Gold (Gayed)")
    print_metrics(m)
    save_result("C01_lumber_gold", m, extra={
        "status": "ok",
        "rule": "13-wk return lumber vs gold; if lumber > gold long SPY for next week, else long TLT. Weekly rebalance.",
        "universe": "SPY, TLT (gated by lumber/gold)",
        "source": "Gayed (2014) — Lumber:Gold ratio",
    })


if __name__ == "__main__":
    main()
