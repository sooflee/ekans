"""
M5 TGA drain. FRED WTREGEN (weekly Treasury General Account balance, $B).
When weekly change < -$100B, long SPY for 20 trading days.
Mechanism: TGA drawdown injects reserves into the banking system (de facto QE).
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
        # WTREGEN is in $M (FRED units), weekly (Wednesday). Convert to $B.
        tga_raw = load_fred("WTREGEN", start="2005-01-01").iloc[:, 0]
        tga = (tga_raw / 1_000.0).rename("TGA")  # $B
        spy = load_prices(["SPY"], start="2005-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("M5_tga_drain", f"data load failed: {e}")

    tga = tga.dropna()
    weekly_chg = tga.diff()  # $B WoW change

    # Drain events
    drain_dates = weekly_chg.index[weekly_chg < -100]
    n_events = len(drain_dates)

    # Build daily position series: 1.0 if within 20 trading days after a drain event, else 0
    rets = spy.pct_change()
    pos = pd.Series(0.0, index=spy.index)

    for d in drain_dates:
        # Align to next available SPY trading day
        ix = spy.index.searchsorted(d)
        if ix >= len(spy.index):
            continue
        end_ix = min(ix + 20, len(spy.index))
        pos.iloc[ix:end_ix] = 1.0

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="M5 TGA drain → long SPY 20d")
    m["n_events"] = int(n_events)
    print_metrics(m)
    print(f"  n_events: {n_events}")
    save_result("M5_tga_drain", m, extra={
        "status": "ok",
        "rule": "When WTREGEN WoW change < -$100B, long SPY for 20 trading days.",
        "mechanism": "TGA drawdown adds reserves to bank system; de facto liquidity injection.",
        "source": "FRED WTREGEN; Strategas/Hedgeye TGA pulse research.",
    })


if __name__ == "__main__":
    main()
