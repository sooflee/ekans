"""
V12 Kuppy uranium pair (URNM vs SRUUF)
Universe:
  URNM  = North-Shore Uranium Miners ETF
  SRUUF = Sprott Physical Uranium Trust (OTC; trades U3O8 spot exposure)
Rule:
  Compute 6-month (126 trading day) relative return: r6_urnm - r6_sruuf.
  When URNM has UNDERPERFORMED SRUUF by > 20% over 6m, go long URNM / short
  SRUUF (50/50 notional). Close when the trailing 6m gap closes to within 5%.
  Default = flat.
Mechanism (Kuppy / Praetorian Capital): uranium miners are leveraged plays on
the spot uranium price; large miner-underperformance vs the physical typically
mean-reverts via miner re-rating (operating leverage to spot price).
Note: SRUUF is OTC pink; yfinance returns it but liquidity is thin. We treat
this as a research backtest, not an execution-ready strategy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    try:
        px = load_prices(["URNM", "SRUUF", "SPY"], start="2021-07-22")
    except Exception as e:
        return mark_failed("V12_kuppy_uranium_pair", f"data load: {e}")

    px = px.dropna()
    if px.empty:
        return mark_failed("V12_kuppy_uranium_pair", "no URNM/SRUUF overlap")

    urnm_r6 = px["URNM"].pct_change(126)
    sruuf_r6 = px["SRUUF"].pct_change(126)
    gap = urnm_r6 - sruuf_r6  # negative = URNM underperformed

    # State machine: 0 flat, 1 long URNM / short SRUUF
    state = 0
    pos = pd.DataFrame(0.0, index=px.index, columns=["URNM","SRUUF"])
    for i in range(len(px)):
        g = gap.iloc[i]
        if not np.isnan(g):
            if state == 0 and g < -0.20:
                state = 1
            elif state == 1 and g > -0.05:
                state = 0
        if state == 1:
            pos.iloc[i] = [0.5, -0.5]

    rets = px[["URNM","SRUUF"]].pct_change()
    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    bench = px["SPY"].pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V12 Kuppy URNM/SRUUF pair")
    print_metrics(m)
    save_result("V12_kuppy_uranium_pair", m, extra={
        "status": "ok",
        "rule": "Long URNM / short SRUUF (50/50) when URNM lags SRUUF by >20% over 6m; close when gap closes to within 5%.",
        "mechanism": "Uranium miners are operating-leveraged to spot U3O8; large miner-underperformance vs physical mean-reverts as miners re-rate.",
        "source": "Harris Kuppy (Kuppy / Praetorian Capital), YouTube interview round 2 (Phase 1V).",
        "data_note": "SRUUF is OTC; thin liquidity. Research-grade only.",
    })


if __name__ == "__main__":
    main()
