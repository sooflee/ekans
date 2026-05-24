"""
AB1 CAD-WTI residual (CAD overshoots oil)
FRED DEXCAUS (USDCAD) + DCOILWTICO. 60d OLS regression of CAD daily returns on WTI daily returns.
5d rolling residual = sum of last 5d (actual - predicted) CAD-return.
When residual z-score (1y window) < -2 -> CAD underperformed oil by a lot -> long USDCAD until z back to 0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB1_cad_wti_residual"
    try:
        fx = load_fred(["DEXCAUS", "DCOILWTICO"], start="2003-01-01").dropna()
        spy = load_prices(["SPY"], start="2003-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if len(fx) < 500:
        return mark_failed(sid, "insufficient FRED data")

    # daily log returns (CAD-positive return = CAD weakening, USDCAD rising)
    r = np.log(fx).diff().dropna()
    r.columns = ["dCAD", "dWTI"]

    win = 60
    # rolling OLS slope/intercept of CAD on WTI
    cov = r["dCAD"].rolling(win).cov(r["dWTI"])
    var = r["dWTI"].rolling(win).var()
    beta = (cov / var).rename("beta")
    # intercept
    mc = r["dCAD"].rolling(win).mean()
    mw = r["dWTI"].rolling(win).mean()
    alpha = (mc - beta * mw).rename("alpha")
    pred = alpha + beta * r["dWTI"]
    resid = (r["dCAD"] - pred).rename("resid")

    # 5d sum of residuals
    r5 = resid.rolling(5).sum()
    # z-score over 252d window
    z = (r5 - r5.rolling(252).mean()) / r5.rolling(252).std()
    z = z.dropna()

    # Signal: when z < -2 -> CAD has *underperformed* oil (too strong vs what oil implies);
    # actually negative residual = CAD weaker than implied by WTI (USDCAD higher than expected).
    # In a mean-reversion view: USDCAD should revert lower -> short USDCAD.
    # Strategy convention here: long USDCAD position when expecting USDCAD to *rise*.
    # Convention: dCAD = d log(USDCAD). Negative residual = USDCAD rose less than oil-implied.
    # If oil rose and USDCAD didn't fall as much as expected (i.e., CAD weaker than predicted),
    # CAD is overpriced weak -> USDCAD should come down -> short USDCAD (position = -1).
    # When z < -2 -> short USDCAD, exit when z >= 0.
    # PnL on USDCAD long = dCAD (since dCAD positive = USDCAD up).

    pos = pd.Series(0.0, index=z.index)
    state = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if state == 0 and zi < -2:
            state = -1  # short USDCAD
        elif state == 0 and zi > 2:
            state = 1   # long USDCAD (symmetric)
        elif state == -1 and zi >= 0:
            state = 0
        elif state == 1 and zi <= 0:
            state = 0
        pos.iloc[i] = state

    # pnl on USDCAD = dCAD * position (lagged)
    cad_ret = r["dCAD"].reindex(pos.index)
    pnl = (pos.shift(1) * cad_ret).dropna()

    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB1 CAD-WTI residual")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "60d OLS CAD~WTI rolling residual. 5d sum z-score (252d). z<-2 -> short USDCAD until z>=0; z>+2 -> long USDCAD until z<=0.",
        "data_source": "FRED DEXCAUS, DCOILWTICO",
        "caveat": "FRED FX is mid-rate, not transactable; bid/ask omitted.",
    })


if __name__ == "__main__":
    main()
