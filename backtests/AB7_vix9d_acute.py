"""
AB7 VIX9D acute fear
yfinance ^VIX9D and ^VIX. When VIX9D/VIX > 1.20 close AND VIX < 1y 70th pct,
long SPY at next open for 5 trading days. Approx with adjusted closes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB7_vix9d_acute"
    try:
        df = load_prices(["^VIX9D", "^VIX", "SPY"], start="2011-01-01").dropna()
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if len(df) < 400:
        return mark_failed(sid, "insufficient overlap")

    ratio = df["^VIX9D"] / df["^VIX"]
    vix_pct70 = df["^VIX"].rolling(252).quantile(0.70)

    trig = (ratio > 1.20) & (df["^VIX"] < vix_pct70)
    rets = df["SPY"].pct_change()

    pos = pd.Series(0.0, index=df.index)
    rem = 0
    for i in range(len(df)):
        if trig.iloc[i]:
            rem = 5
        if rem > 0:
            pos.iloc[i] = 1.0
            rem -= 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AB7 VIX9D acute fear")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "VIX9D/VIX > 1.20 close AND VIX < 1y 70th pct -> long SPY 5 days (entry next open, approx with close-to-close).",
        "data_source": "yfinance ^VIX9D ^VIX SPY",
        "n_triggers": int(trig.sum()),
    })


if __name__ == "__main__":
    main()
