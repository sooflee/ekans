"""
N6 SOFR-IORB stress.
When (SOFR - IORB) > +10bp for 2 consecutive days, short SPY for 5 days.
Pre-IORB launch (Jul 2021), use IOER as substitute.
Sample only since SOFR start (Apr 2018).
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
        sofr = load_fred("SOFR", start="2018-04-01").iloc[:, 0].rename("SOFR")
        iorb = load_fred("IORB", start="2018-04-01").iloc[:, 0].rename("IORB")
        ioer = load_fred("IOER", start="2018-04-01").iloc[:, 0].rename("IOER")
        spy = load_prices(["SPY"], start="2018-04-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("N6_sofr_iorb", f"data load failed: {e}")

    # Splice IOER -> IORB (IORB starts Jul 2021)
    ior = iorb.combine_first(ioer).rename("IOR")

    df = pd.concat([sofr, ior], axis=1).ffill().dropna()
    df["spread_bp"] = (df["SOFR"] - df["IOR"]) * 100.0

    # Signal: spread > 10bp for 2 consecutive days
    stress = df["spread_bp"] > 10.0
    sig = stress & stress.shift(1).fillna(False)
    trig_dates = df.index[sig]

    spy_rets = spy.pct_change()
    # Build a position series: short -1 for next 5 sessions after trigger
    pos = pd.Series(0.0, index=spy_rets.index)
    for d in trig_dates:
        loc = spy_rets.index.searchsorted(d)
        # apply from next session for 5 days
        for k in range(1, 6):
            if loc + k < len(pos):
                pos.iloc[loc + k] = -1.0

    pnl = (pos * spy_rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]  # trim leading zeros

    if pnl.empty or len(pnl) < 30:
        return mark_failed(
            "N6_sofr_iorb",
            f"insufficient trigger days; n_triggers={len(trig_dates)}",
            extra={"rule": "Short SPY 5d when SOFR-IORB>+10bp for 2 days",
                   "source": "Anbil/Anderson/Senyuz Fed 2020"},
        )

    bench = spy_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="N6 SOFR-IORB stress short SPY")
    m["n_triggers"] = int(len(trig_dates))
    print_metrics(m)
    save_result("N6_sofr_iorb", m, extra={
        "status": "ok",
        "rule": "When (SOFR-IORB)>+10bp for 2 consecutive days, short SPY 5 trading days.",
        "mechanism": "Repo/money-market stress -> funding squeeze -> equity drawdown",
        "universe": "SPY",
        "source": "Fed FEDS Notes (Anbil et al 2020); Sep 2019 repo spike",
        "data": "FRED SOFR + IORB (Jul 2021+) spliced with IOER (pre-Jul 2021)",
    })


if __name__ == "__main__":
    main()
