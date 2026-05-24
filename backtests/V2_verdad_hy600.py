"""
V2 Verdad HY 600bps regime
Rule:
  When BAML US HY OAS (FRED BAMLH0A0HYM2) closes > 600bps, rotate to
  50/50 HYG + IJS. Hold until OAS closes < 400bps, then back to SPY (default).
  Hysteresis prevents whipsaws.
Mechanism (Verdad / Dan Rasmussen): historically the highest forward-12m
returns to credit and small-cap value are concentrated in the periods just
after a HY spread blowout (>600bps). Spreads >600bps are rare (~5-10% of time)
and have historically marked exceptional vintages for HY and small-cap value.
NOTE: FRED's free CSV endpoint truncates BAMLH0A0HYM2 to ~3y; full series
needs a FRED API key. Backtest limited to 2023-present.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        hy = load_fred("BAMLH0A0HYM2", start="1996-12-01").iloc[:, 0].dropna().astype(float)
    except Exception as e:
        return mark_failed("V2_verdad_hy600", f"FRED load failed: {e}")

    start_overlap = max(hy.index.min(), pd.Timestamp("2007-01-01"))
    try:
        px = load_prices(["SPY", "HYG", "IJS"], start=str(start_overlap.date()))
    except Exception as e:
        return mark_failed("V2_verdad_hy600", f"price load: {e}")

    df = pd.concat([hy.rename("HY"), px], axis=1).dropna()
    if df.empty:
        return mark_failed("V2_verdad_hy600", "no overlap")
    rets = df[["SPY", "HYG", "IJS"]].pct_change()

    # State machine: 0 = default SPY, 1 = HYG/IJS 50/50
    state = 0
    pos = pd.DataFrame(0.0, index=df.index, columns=["SPY", "HYG", "IJS"])
    for i, dt in enumerate(df.index):
        oas = df["HY"].iloc[i]
        if state == 0 and oas > 6.0:
            state = 1
        elif state == 1 and oas < 4.0:
            state = 0
        if state == 0:
            pos.iloc[i] = [1.0, 0.0, 0.0]
        else:
            pos.iloc[i] = [0.0, 0.5, 0.5]

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V2 Verdad HY > 600bps regime")
    print_metrics(m)
    save_result("V2_verdad_hy600", m, extra={
        "status": "ok",
        "rule": "Default 100% SPY. When HY OAS > 600bps -> 50/50 HYG+IJS. Exit when OAS < 400bps (hysteresis).",
        "mechanism": "Distressed credit + small-cap value have outsized forward 12m returns post HY-spread blowouts (Verdad research).",
        "source": "Verdad / Dan Rasmussen, YouTube interview round 2 (Phase 1V).",
        "data_note": "FRED public CSV truncates BAMLH0A0HYM2 to ~3y; backtest 2023-present unless API key supplied.",
    })


if __name__ == "__main__":
    main()
