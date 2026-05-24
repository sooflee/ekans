"""
V8 Crittenden 12-asset trend (STRONG CANDIDATE)
Universe (12 ETFs):
  Equities:    SPY, EFA, EEM
  Bonds:       TLT
  USD:         UUP
  Commodities: GLD, USO, CPER, DBA, UNG, SOYB, WEAT
Rule:
  Compute 200-day SMA on each asset. Equal-weight all assets currently
  above their SMA; cash for any that are below. Rebalance monthly.
Mechanism (Cole Crittenden / Faber-style absolute momentum at the asset
level): assets above their 10-month/200-day SMA have positive expected
risk premium; staying off the boat for assets below SMA cuts drawdowns
dramatically with only modest CAGR cost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    tickers = ["SPY","EFA","EEM","TLT","UUP","GLD","USO","CPER","DBA","UNG","SOYB","WEAT"]
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed("V8_crittenden_12asset_trend", f"data load: {e}")

    # Drop tickers that never start (e.g. CPER 2011, SOYB 2011)
    px = px.dropna(how="all")
    # We'll use only assets with > 200 valid days; for assets not yet listed
    # treat them as "below SMA" until they have data + 200 day history.
    sma = px.rolling(200).mean()
    above = (px > sma).astype(float)

    # Monthly rebalance: take last business day of each month signal,
    # forward-fill for that month.
    monthly_idx = above.resample("ME").last().index
    above_m = above.resample("ME").last().fillna(0.0)
    # weight = 1/N_above (or 0 if none above)
    row_sum = above_m.sum(axis=1).replace(0, np.nan)
    weights_m = above_m.div(row_sum, axis=0).fillna(0.0)

    # daily positions, lagged 1 day
    weights_d = weights_m.reindex(px.index, method="ffill").fillna(0.0)
    rets = px.pct_change()
    pnl = (weights_d.shift(1) * rets).sum(axis=1).dropna()
    # focus on era where most assets exist
    pnl = pnl.loc[pnl.index >= "2012-01-01"]

    try:
        spy = load_prices(["SPY"], start="2011-12-01").iloc[:, 0].pct_change()
    except Exception:
        spy = None
    m = compute_metrics(pnl, benchmark=spy.reindex(pnl.index) if spy is not None else None,
                        name="V8 Crittenden 12-asset 200d-SMA trend")
    print_metrics(m)
    save_result("V8_crittenden_12asset_trend", m, extra={
        "status": "ok",
        "rule": "12 ETFs (SPY EFA EEM TLT UUP GLD USO CPER DBA UNG SOYB WEAT). Equal-weight all assets above their 200d SMA, cash for the rest. Monthly rebal.",
        "mechanism": "Asset-level absolute momentum (Faber-style): being out of assets below trend keeps drawdowns shallow at modest CAGR cost.",
        "source": "Cole Crittenden, YouTube interview round 2 (Phase 1V).",
        "universe": "SPY,EFA,EEM,TLT,UUP,GLD,USO,CPER,DBA,UNG,SOYB,WEAT",
    })


if __name__ == "__main__":
    main()
