"""
C03 Copper/Gold -> 10Y yield
HG=F vs GC=F. 60-day change in ratio. When ratio rises >10% over 60d while DGS10 flat-to-falling,
short TLT. Reverse case for long TLT.
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
        hg = load_prices(["HG=F"], start="2005-01-01").iloc[:, 0].rename("HG")
        gc = load_prices(["GC=F"], start="2005-01-01").iloc[:, 0].rename("GC")
        tlt = load_prices(["TLT"], start="2005-01-01").iloc[:, 0].rename("TLT")
        dgs10 = load_fred("DGS10", start="2005-01-01").iloc[:, 0].rename("DGS10")
    except Exception as e:
        return mark_failed("C03_copper_gold", f"data load failed: {e}")

    df = pd.concat([hg, gc, tlt, dgs10], axis=1)
    df["DGS10"] = df["DGS10"].ffill()
    df = df.dropna()

    ratio = df["HG"] / df["GC"]
    ratio_chg = ratio.pct_change(60)
    yld_chg = df["DGS10"] - df["DGS10"].shift(60)

    tlt_ret = df["TLT"].pct_change()

    # Signal:
    #  +1 ratio rises >10% and yield flat/falling => short TLT
    #  -1 ratio falls < -10% and yield flat/rising => long TLT
    pos = pd.Series(0.0, index=df.index)
    short_sig = (ratio_chg > 0.10) & (yld_chg <= 0)
    long_sig = (ratio_chg < -0.10) & (yld_chg >= 0)
    pos[short_sig] = -1.0
    pos[long_sig] = 1.0

    pnl = (pos.shift(1) * tlt_ret).dropna()
    bench = tlt_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C03 Copper/Gold -> TLT")
    print_metrics(m)
    save_result("C03_copper_gold", m, extra={
        "status": "ok",
        "rule": "60d copper/gold ratio change vs DGS10 change. >+10% & yield flat/down: short TLT. <-10% & yield flat/up: long TLT.",
        "universe": "TLT",
        "source": "Copper/Gold as bond proxy (Jeff Gundlach popularized)",
    })


if __name__ == "__main__":
    main()
