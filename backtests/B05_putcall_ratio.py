"""
B05 Put/call ratio extremes
Try FRED for CBOE total P/C ratio. If unavailable, try scraping CBOE.
Rule: 10-day MA: > 0.75 -> long SPY 20 sessions; < 0.50 -> flat.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, long_short_pnl,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    pc = None
    note = None
    # Try FRED first (no canonical free P/C series exists)
    candidates = ["CBOE_EQUITY_PC", "PUTCALL", "PCRATIO"]
    for s in candidates:
        try:
            from pandas_datareader import data as pdr
            df = pdr.DataReader(s, "fred", start="1995-01-01")
            pc = df.iloc[:, 0]
            note = f"FRED series {s}"
            break
        except Exception:
            continue

    # Try CBOE direct download
    if pc is None:
        urls = [
            "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_TOTAL_PC.csv",
            "https://cdn.cboe.com/resources/options/total_pc.csv",
        ]
        for url in urls:
            try:
                df = pd.read_csv(url)
                # try heuristic
                if "DATE" in df.columns.str.upper().tolist():
                    pass
                note = f"CBOE direct: {url}"
                break
            except Exception:
                continue

    if pc is None:
        return mark_failed(
            "B05_putcall_ratio",
            "CBOE P/C history not free-API accessible (FRED has no current P/C series; CBOE CSVs are gated).",
        )

    spy = load_prices(["SPY"], start="2000-01-01").iloc[:, 0].rename("SPY")
    df = pd.concat([pc.rename("PC"), spy], axis=1).dropna()
    rets = df["SPY"].pct_change()
    pc_ma = df["PC"].rolling(10).mean()
    long_trig = pc_ma > 0.75
    flat_trig = pc_ma < 0.50

    pos = pd.Series(0.0, index=df.index)
    remaining_long = 0
    remaining_flat = 0
    for i in range(len(df)):
        if long_trig.iloc[i]:
            remaining_long = 20
        if flat_trig.iloc[i]:
            remaining_flat = 20
        if remaining_flat > 0:
            pos.iloc[i] = 0.0
            remaining_flat -= 1
            if remaining_long > 0:
                remaining_long -= 1
        elif remaining_long > 0:
            pos.iloc[i] = 1.0
            remaining_long -= 1

    pnl = long_short_pnl(pos, rets)
    m = compute_metrics(pnl, benchmark=rets, name="B05 P/C ratio")
    print_metrics(m)
    save_result("B05_putcall_ratio", m, extra={
        "status": "ok",
        "rule": "10d MA P/C > 0.75 -> long SPY 20d; < 0.50 -> flat.",
        "data_source": note,
    })


if __name__ == "__main__":
    main()
