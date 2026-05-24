"""
M3 stETH discount.
Long ETH when stETH/ETH ratio (proxy: STETH-USD / ETH-USD) closes < 0.985.
Hold 90 days.
Mechanism: Forced sellers / liquidations pushing stETH below peg historically marked ETH bottoms (Jun 2022).
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
        px = load_prices(["STETH-USD", "ETH-USD"], start="2020-12-01")
    except Exception as e:
        return mark_failed("M3_steth_discount", f"data load failed: {e}")

    px = px.dropna()
    if len(px) < 200:
        return mark_failed("M3_steth_discount", f"insufficient data: {len(px)} rows")

    ratio = px["STETH-USD"] / px["ETH-USD"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    df = pd.concat([px["ETH-USD"].rename("ETH"), ratio.rename("R")], axis=1).dropna()

    trigger = (df["R"] < 0.985)
    # First crossing (avoid stacking)
    prev = trigger.shift(1).fillna(False)
    first_cross = trigger & (~prev)
    event_dates = df.index[first_cross]

    rets = df["ETH"].pct_change()
    pos = pd.Series(0.0, index=df.index)
    n_events = 0
    cooldown_until = None
    for d in event_dates:
        if cooldown_until is not None and d <= cooldown_until:
            continue
        ix = df.index.searchsorted(d)
        if ix >= len(df.index):
            continue
        end_ix = min(ix + 90, len(df.index))
        pos.iloc[ix:end_ix] = 1.0
        n_events += 1
        cooldown_until = d + pd.Timedelta(days=90)

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M3 stETH/ETH < 0.985 → long ETH 90d")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events: {n_events}")
    save_result("M3_steth_discount", m, extra={
        "status": "ok",
        "rule": "Long ETH-USD when STETH-USD / ETH-USD closes < 0.985 (first crossing, 90d cooldown). Hold 90 days.",
        "mechanism": "stETH depeg events historically mark forced-seller capitulation in ETH (Jun 2022 Celsius/3AC).",
        "source": "yfinance STETH-USD, ETH-USD.",
    })


if __name__ == "__main__":
    main()
