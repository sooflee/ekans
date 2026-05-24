"""
AB5 CLP leads copper
yfinance CLP=X (USDCLP) + HG=F. When CLP 10d return (CLP appreciation = -d log USDCLP) MINUS
HG=F 10d return > +2sigma, CLP outperforming copper -> long HG=F (mean-reversion catch-up).
Exit after 10 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB5_clp_leads_copper"
    try:
        px = load_prices(["CLP=X", "HG=F"], start="2010-01-01").dropna(how="all")
        spy = load_prices(["SPY"], start="2010-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    px = px.dropna()
    if len(px) < 400:
        return mark_failed(sid, "insufficient overlap CLP=X / HG=F")

    # USDCLP = px["CLP=X"]; CLP appreciation = -d log USDCLP
    dUSDCLP = np.log(px["CLP=X"]).diff()
    clp_ret = -dUSDCLP
    hg_ret = np.log(px["HG=F"]).diff()

    win = 10
    clp10 = clp_ret.rolling(win).sum()
    hg10 = hg_ret.rolling(win).sum()
    diff = clp10 - hg10
    z = (diff - diff.rolling(252).mean()) / diff.rolling(252).std()

    # signal: z > 2 -> long HG (catch-up). z < -2 -> short HG (HG ahead, expect reversion).
    pos = pd.Series(0.0, index=px.index)
    rem = 0
    side = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if rem == 0:
            if pd.notna(zi) and zi > 2.0:
                side = 1
                rem = win
            elif pd.notna(zi) and zi < -2.0:
                side = -1
                rem = win
        if rem > 0:
            pos.iloc[i] = side
            rem -= 1

    pnl = (pos.shift(1) * hg_ret).dropna()

    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB5 CLP leads copper")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "10d CLP return minus 10d HG=F return. z>2 -> long HG 10d; z<-2 -> short HG 10d.",
        "data_source": "yfinance CLP=X, HG=F",
    })


if __name__ == "__main__":
    main()
