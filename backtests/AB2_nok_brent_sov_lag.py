"""
AB2 NOK Brent sovereign-lag
FRED DEXNOUS (USDNOK) + DCOILBRENTEU. When 20d Brent return > +8% AND
20d NOK return (NOK appreciation = USDNOK down) is less than 50% of (beta * Brent_return),
long NOK = short USDNOK. Hold 20d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB2_nok_brent_sov_lag"
    try:
        fx = load_fred(["DEXNOUS", "DCOILBRENTEU"], start="2003-01-01").dropna()
        spy = load_prices(["SPY"], start="2003-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if len(fx) < 500:
        return mark_failed(sid, "insufficient FRED data")

    fx = fx.copy()
    fx.columns = ["USDNOK", "Brent"]
    # NOK appreciation = USDNOK falling. We'll define nok_ret as -d log(USDNOK)
    fx["dUSDNOK"] = np.log(fx["USDNOK"]).diff()
    fx["nok_ret"] = -fx["dUSDNOK"]
    fx["brent_ret"] = np.log(fx["Brent"]).diff()
    fx = fx.dropna()

    win = 20
    nok20 = fx["nok_ret"].rolling(win).sum()
    brent20 = fx["brent_ret"].rolling(win).sum()

    # estimate beta over 252d window of brent_ret -> nok_ret
    cov = fx["nok_ret"].rolling(252).cov(fx["brent_ret"])
    var = fx["brent_ret"].rolling(252).var()
    beta = (cov / var).rename("beta").clip(0.05, 1.0)  # NOK typically beta ~ 0.2-0.4 to Brent

    cond = (brent20 > 0.08) & (nok20 < 0.5 * beta * brent20) & beta.notna()

    # signal: long NOK (short USDNOK) = position -1 on USDNOK for 20d
    pos = pd.Series(0.0, index=fx.index)
    rem = 0
    for i in range(len(fx)):
        if cond.iloc[i]:
            rem = 20
        if rem > 0:
            pos.iloc[i] = -1.0  # short USDNOK
            rem -= 1

    pnl = (pos.shift(1) * fx["dUSDNOK"]).dropna()
    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB2 NOK Brent sov-lag")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "If 20d Brent > +8% AND 20d NOK appreciation < 0.5 * beta_252d * Brent: short USDNOK 20d.",
        "data_source": "FRED DEXNOUS, DCOILBRENTEU",
        "n_triggers": int(cond.sum()),
    })


if __name__ == "__main__":
    main()
