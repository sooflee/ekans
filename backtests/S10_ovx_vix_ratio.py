"""
S10 OVX/VIX ratio -> short USO.

Rule: When CBOE OVX (oil VIX) / VIX > 2.0 for 5 consecutive trading days,
short USO for 30 trading days. Non-overlapping events.

Mechanism: Oil-vol decoupling well above equity-vol signals an idiosyncratic
oil supply shock or capitulation; historically followed by mean reversion
in OVX and weakness in USO as the implied-vol risk premium unwinds.

Source: yfinance ^OVX, ^VIX, USO.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    try:
        ovx = load_prices(["^OVX"], start="2007-05-01").iloc[:, 0]
        vix = load_prices(["^VIX"], start="2007-05-01").iloc[:, 0]
        uso = load_prices(["USO"], start="2007-05-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S10_ovx_vix_ratio", f"yfinance load failed: {e}")

    df = pd.concat({"ovx": ovx, "vix": vix, "uso": uso}, axis=1).dropna()
    if len(df) < 200:
        return mark_failed("S10_ovx_vix_ratio", f"insufficient overlap ({len(df)} obs)")

    df["ratio"] = df["ovx"] / df["vix"]
    df["ret"] = df["uso"].pct_change()

    # 5-consecutive-day OVX/VIX > 2.0
    high = (df["ratio"] > 2.0).astype(int)
    streak = high.groupby((high == 0).cumsum()).cumcount() + 1
    df["streak5"] = (high == 1) & (streak >= 5)

    triggers = df.index[df["streak5"]].tolist()

    pos = pd.Series(0.0, index=df.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d in triggers:
        idx = df.index.get_loc(d)
        # Enter day after trigger
        start_idx = idx + 1
        if start_idx >= len(df.index):
            continue
        start = df.index[start_idx]
        if last_end is not None and start <= last_end:
            continue
        end_idx = min(start_idx + 30, len(df.index))
        for j in range(start_idx, end_idx):
            pos.iloc[j] = -1.0
        last_end = df.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S10_ovx_vix_ratio",
                           f"no qualifying triggers (max OVX/VIX={df['ratio'].max():.2f})")

    pnl = (pos.shift(1) * df["ret"]).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = df["ret"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S10 OVX/VIX>2 5d -> short USO 30d")
    m["n_events"] = n_events
    print(f"OVX/VIX max: {df['ratio'].max():.2f}; days >2: {(df['ratio']>2).sum()}")
    print(f"5d-streak triggers: {len(triggers)}; non-overlap events: {n_events}")
    print(f"First events: {event_dates[:5]}")
    print_metrics(m)

    save_result("S10_ovx_vix_ratio", m, extra={
        "status": "ok",
        "rule": "When CBOE OVX/VIX > 2.0 for 5 consecutive trading days, short USO for 30 trading days; non-overlapping events.",
        "mechanism": "Sustained dislocation of oil vol vs equity vol signals capitulation / supply-shock vol premium that historically mean-reverts, with USO underperforming as the implied-vol crush unwinds.",
        "source": "yfinance ^OVX, ^VIX, USO (CBOE OVX/VIX official indices).",
        "n_events": n_events,
        "first_events": event_dates[:5],
    })


if __name__ == "__main__":
    main()
