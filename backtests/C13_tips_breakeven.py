"""
C13 TIPS breakeven
FRED T10YIE. Long DBC (commodities ETF) when T10YIE > 60-day MA AND change >+20bps over 30d.
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
        bei = load_fred("T10YIE", start="2003-01-02").iloc[:, 0].rename("T10YIE")
        dbc = load_prices(["DBC"], start="2006-02-06").iloc[:, 0].rename("DBC")
    except Exception as e:
        return mark_failed("C13_tips_breakeven", f"data load failed: {e}")

    df = pd.DataFrame({"T10YIE": bei}).reindex(dbc.index, method="ffill")
    df["DBC"] = dbc
    df = df.dropna()

    bei = df["T10YIE"]
    ma60 = bei.rolling(60).mean()
    chg30 = bei - bei.shift(30)

    sig = ((bei > ma60) & (chg30 > 0.20)).astype(float)

    dbc_ret = df["DBC"].pct_change()
    pnl = (sig.shift(1) * dbc_ret).dropna()
    bench = dbc_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="C13 TIPS breakeven -> DBC")
    print_metrics(m)
    save_result("C13_tips_breakeven", m, extra={
        "status": "ok",
        "rule": "Long DBC when T10YIE > 60d MA AND 30d change > +20bps.",
        "universe": "DBC (gated by T10YIE)",
        "source": "TIPS breakeven inflation literature",
    })


if __name__ == "__main__":
    main()
