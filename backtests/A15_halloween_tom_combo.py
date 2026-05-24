"""
A15 Halloween + ToM combo.
Long SPY only when BOTH Halloween (Nov-Apr) AND Turn-of-Month windows are active.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result,
)


def main():
    px = load_prices(["SPY"], start="2000-01-01")
    rets = daily_returns(px)["SPY"]
    idx = rets.index

    # Halloween mask
    months = idx.month
    halloween = pd.Series(((months >= 11) | (months <= 4)).astype(float), index=idx)

    # ToM mask: last trading day of month OR first 3 trading days of month
    df = pd.DataFrame({"ret": rets})
    df["year"] = df.index.year
    df["month"] = df.index.month
    last_of_month = df.groupby(["year", "month"]).apply(lambda g: g.index[-1])
    last_set = set(last_of_month.values)
    first_n = df.groupby(["year", "month"]).apply(lambda g: list(g.index[:3]))
    first_set = set()
    for arr in first_n:
        for d in arr:
            first_set.add(d)

    tom = pd.Series(0.0, index=idx)
    tom[tom.index.isin(last_set)] = 1.0
    tom[tom.index.isin(first_set)] = 1.0

    pos = (halloween > 0) & (tom > 0)
    pos = pos.astype(float)

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="A15 Halloween x ToM")
    print_metrics(m)
    save_result("A15_halloween_tom_combo", m, extra={
        "status": "ok",
        "rule": "Long SPY only when BOTH Halloween (Nov-Apr) AND ToM windows active. AND of A01 and A02.",
        "universe": "SPY",
        "source": "Combo of A01 (Bouman-Jacobsen 2002) and A02 (Ariel 1987)",
    })


if __name__ == "__main__":
    main()
