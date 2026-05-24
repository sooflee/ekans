"""
AB6 MOVE/DTB3 funding stress
yfinance ^MOVE + FRED DTB3. Compute MOVE / (DTB3*100). When > 1y 90th pct for 3 consec days,
short SPY for 20 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB6_move_dtb3_funding"
    try:
        move = load_prices(["^MOVE"], start="2003-01-01").iloc[:, 0].rename("MOVE")
        dtb3 = load_fred(["DTB3"], start="2003-01-01").iloc[:, 0].rename("DTB3")
        spy = load_prices(["SPY"], start="2003-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    df = pd.concat([move, dtb3, spy], axis=1).sort_index().ffill().dropna()
    if df.empty:
        return mark_failed(sid, "no overlap MOVE/DTB3/SPY")

    # avoid blow-up when DTB3 ~ 0: floor at 10bps (0.10)
    floor = df["DTB3"].clip(lower=0.10)
    ratio = df["MOVE"] / (floor * 100.0)

    win = 252
    pct90 = ratio.rolling(win).quantile(0.90)
    trig = (ratio > pct90)
    # 3 consec days
    trig3 = trig.rolling(3).sum() == 3

    rets = df["SPY"].pct_change()
    pos = pd.Series(0.0, index=df.index)
    rem = 0
    for i in range(len(df)):
        if trig3.iloc[i]:
            rem = 20
        if rem > 0:
            pos.iloc[i] = -1.0
            rem -= 1

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="AB6 MOVE/DTB3 funding")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "ratio = ^MOVE / max(DTB3*100, 10bps*100). If ratio > 1y 90th pct for 3 consec days -> short SPY 20d.",
        "data_source": "yfinance ^MOVE; FRED DTB3",
        "n_triggers": int(trig3.sum()),
    })


if __name__ == "__main__":
    main()
