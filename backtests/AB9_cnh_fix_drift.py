"""
AB9 CNH fix drift
FRED DEXCHUS (PBoC reference rate proxy via daily fix). When 5d cumulative move > +0.5%
(USDCNY up; CNY weakening) AND spot is near upper 2% band (i.e., USDCNY z-score > 1 over 252d),
long USDCNH proxy. Since yfinance USDCNH=X data is broken (1 row), short FXI (China large cap)
as a CNY-weakness proxy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB9_cnh_fix_drift"
    try:
        fx = load_fred(["DEXCHUS"], start="2005-01-01").iloc[:, 0].dropna().rename("USDCNY")
        fxi = load_prices(["FXI"], start="2005-01-01").iloc[:, 0].rename("FXI")
        spy = load_prices(["SPY"], start="2005-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if len(fx) < 500:
        return mark_failed(sid, "insufficient FRED data")

    df = pd.concat([fx, fxi], axis=1).sort_index().ffill().dropna()
    df["dUSDCNY"] = np.log(df["USDCNY"]).diff()
    df["fxi_ret"] = df["FXI"].pct_change()

    cum5 = df["dUSDCNY"].rolling(5).sum()
    z252 = (df["USDCNY"] - df["USDCNY"].rolling(252).mean()) / df["USDCNY"].rolling(252).std()

    cond = (cum5 > 0.005) & (z252 > 1.0)

    # short FXI for 20 days as proxy for short CNH
    pos = pd.Series(0.0, index=df.index)
    rem = 0
    for i in range(len(df)):
        if cond.iloc[i]:
            rem = 20
        if rem > 0:
            pos.iloc[i] = -1.0
            rem -= 1

    pnl = (pos.shift(1) * df["fxi_ret"]).dropna()
    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB9 CNH fix drift")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "5d log USDCNY > +0.5% AND 252d z-score(USDCNY) > 1 -> short FXI 20d as CNH proxy.",
        "data_source": "FRED DEXCHUS (USDCNY daily fix), yfinance FXI proxy (USDCNH=X is broken on yfinance).",
        "n_triggers": int(cond.sum()),
    })


if __name__ == "__main__":
    main()
