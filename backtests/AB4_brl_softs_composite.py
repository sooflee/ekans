"""
AB4 BRL softs-composite (gap signal)
FRED DEXBZUS + yfinance VALE (iron ore proxy) + SB=F + KC=F + ZS=F. Compute composite
60d-return z-score across commodity basket minus BRL z-score. When gap > 1.5 -> long BRL
(short USDBRL). Hold while gap > 0.5; exit when gap < 0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def zscore(s, win=252):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


def main():
    sid = "AB4_brl_softs_composite"
    try:
        fx = load_fred(["DEXBZUS"], start="2010-01-01").dropna()
        comms = load_prices(["VALE", "SB=F", "KC=F", "ZS=F"], start="2010-01-01").dropna(how="all")
        spy = load_prices(["SPY"], start="2010-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if len(fx) < 500 or comms.empty:
        return mark_failed(sid, "insufficient data")

    fx = fx.copy()
    fx.columns = ["USDBRL"]
    fx["dUSDBRL"] = np.log(fx["USDBRL"]).diff()
    fx["brl_ret"] = -fx["dUSDBRL"]

    # join with comm prices (use union daily index from fx)
    px = comms.reindex(fx.index, method="ffill")
    comm_ret = np.log(px).diff()

    # 60d cumulative returns
    win = 60
    brl_60 = fx["brl_ret"].rolling(win).sum()
    comm_60 = comm_ret.rolling(win).sum()

    # z-scores (252d)
    brl_z = zscore(brl_60)
    comm_z = comm_60.apply(zscore, axis=0)
    comm_z_avg = comm_z.mean(axis=1)  # composite z

    gap = (comm_z_avg - brl_z).dropna()
    # Long BRL position when gap > 1.5; hold while gap > 0.5; flat below.

    pos = pd.Series(0.0, index=gap.index)
    state = 0
    for i, g in enumerate(gap.values):
        if state == 0 and g > 1.5:
            state = 1
        elif state == 1 and g < 0.0:
            state = 0
        elif state == 0 and g < -1.5:
            state = -1  # short BRL (long USDBRL)
        elif state == -1 and g > 0.0:
            state = 0
        pos.iloc[i] = state

    # signal interpretation: state==1 -> long BRL = short USDBRL = position -1 on USDBRL
    # state==-1 -> long USDBRL = position +1 on USDBRL
    # pnl on USDBRL with daily dUSDBRL: usdbrl_pos * dUSDBRL
    usdbrl_pos = -pos  # invert
    pnl = (usdbrl_pos.shift(1) * fx["dUSDBRL"]).reindex(gap.index).dropna()

    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB4 BRL softs composite")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "60d comm composite z (VALE, SB, KC, ZS) minus BRL z. gap>+1.5 -> long BRL; hold while >0. gap<-1.5 -> short BRL; hold while <0.",
        "data_source": "FRED DEXBZUS, yfinance VALE/SB=F/KC=F/ZS=F",
    })


if __name__ == "__main__":
    main()
