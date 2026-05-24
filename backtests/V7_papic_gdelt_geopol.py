"""
V7 Papic geopolitical sell-off buy
Original signal: when GDELT global conflict index > 2 sigma AND SPY drops 5%
in 5 days, go long SPY 12 months.
GDELT BigQuery streaming is heavy; substitute Caldara-Iacoviello daily
Geopolitical Risk index (GPRD) which is already used in this catalog.
Rule:
  z = z-score of GPRD vs trailing 252d.
  trig = z > 2 AND SPY 5d return < -5%.
  When trig: go long SPY 252 trading days (12 months). Position resets only
  after the holding ends; overlapping triggers extend the hold by reusing the
  most recent trigger as the start.
Mechanism (Marko Papic / Geopolitical Alpha): geopolitical shocks paired with
acute equity sell-offs are systematically over-priced; mean reversion of
risk premium plus dovish policy response drives outsized 12m forward returns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics,
    save_result, mark_failed, rolling_zscore,
)
from _gpr import load_gpr


def main():
    try:
        gpr = load_gpr()
    except Exception as e:
        return mark_failed("V7_papic_gdelt_geopol", f"GPR load: {e}")

    gprd = gpr["GPRD"].dropna()
    try:
        spy = load_prices(["SPY"], start=str(max(gprd.index.min(), pd.Timestamp("1993-01-29")).date())).iloc[:, 0]
    except Exception as e:
        return mark_failed("V7_papic_gdelt_geopol", f"SPY load: {e}")

    df = pd.concat([gprd.rename("GPRD"), spy.rename("SPY")], axis=1).dropna()
    df["z"] = rolling_zscore(df["GPRD"], 252)
    df["r5"] = df["SPY"].pct_change(5)
    trig = (df["z"] > 2.0) & (df["r5"] < -0.05)

    # build position: 1 for 252 trading days following each trigger
    pos = pd.Series(0.0, index=df.index)
    last_trig_end = -1
    for i, dt in enumerate(df.index):
        if bool(trig.iloc[i]):
            last_trig_end = i + 252
        if i < last_trig_end:
            pos.iloc[i] = 1.0

    rets = df["SPY"].pct_change()
    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V7 Papic GPR+SPY shock buy")
    print_metrics(m)
    save_result("V7_papic_gdelt_geopol", m, extra={
        "status": "ok",
        "rule": "When GPRD z(252) > 2 AND SPY 5d return < -5%, go long SPY for 252 trading days (12 months).",
        "mechanism": "Geopolitical shocks + equity sell-offs systematically over-discount risk; recovery is fuelled by dovish policy response.",
        "source": "Marko Papic (Clocktower / BCA), YouTube interview round 2 (Phase 1V).",
        "substitution": "GDELT BigQuery substituted with Caldara-Iacoviello daily Geopolitical Risk (GPRD) index, already in catalog (consistent with brief).",
        "n_triggers": int(trig.sum()),
        "pct_time_long": float(pos.mean()),
    })


if __name__ == "__main__":
    main()
