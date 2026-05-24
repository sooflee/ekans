"""
AB8 US-JP real rate gap
FRED DFII10 (US 10Y real, daily) + IRLTLT01JPM156N (JP 10Y nominal, monthly) +
JPNCPIALLMINMEI (JP CPI level, monthly) -> JP 10Y real ~ JP 10Y nominal - JP CPI YoY.
Compute US-JP real differential. When 20d change > +25bps and USDJPY (DEXJPUS) hasn't
moved correspondingly (USDJPY 20d return < 0.5 * avg historical sensitivity), long USDJPY (20d hold).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB8_us_jp_real_rate"
    try:
        us_real = load_fred(["DFII10"], start="2003-01-01").iloc[:, 0].rename("US_real")
        jp_nom = load_fred(["IRLTLT01JPM156N"], start="2003-01-01").iloc[:, 0].rename("JP_nom")
        jp_cpi = load_fred(["JPNCPIALLMINMEI"], start="2000-01-01").iloc[:, 0].rename("JP_CPI")
        fx = load_fred(["DEXJPUS"], start="2003-01-01").iloc[:, 0].rename("USDJPY")
        spy = load_prices(["SPY"], start="2003-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    # JP YoY CPI (annual %)
    jp_yoy = (jp_cpi / jp_cpi.shift(12) - 1.0) * 100.0
    jp_real = (jp_nom - jp_yoy).dropna().rename("JP_real")

    # Build daily series (ffill monthly)
    idx = fx.index
    us_d = us_real.reindex(idx, method="ffill")
    jp_d = jp_real.reindex(idx, method="ffill")
    diff = (us_d - jp_d).dropna()
    fx = fx.reindex(diff.index, method="ffill")

    d20_diff = diff - diff.shift(20)  # in pct points (bps/100)
    fx_ret20 = np.log(fx).diff(20)

    # historical sensitivity of fx_ret_20d to diff change (bps in pct points)
    # avg slope: use 5y rolling regression for stability
    cov = fx_ret20.rolling(252 * 5).cov(d20_diff)
    var = d20_diff.rolling(252 * 5).var()
    beta = (cov / var)

    expected_move = beta * d20_diff
    actual_move = fx_ret20
    # Trigger: diff change > +25bps (= +0.25%), and actual_move < 0.5 * expected_move
    cond = (d20_diff > 0.25) & (actual_move < 0.5 * expected_move) & beta.notna() & (expected_move > 0)

    # long USDJPY 20d
    dUSDJPY = np.log(fx).diff()
    pos = pd.Series(0.0, index=fx.index)
    rem = 0
    for i in range(len(fx)):
        if i < len(cond) and bool(cond.iloc[i]):
            rem = 20
        if rem > 0:
            pos.iloc[i] = 1.0
            rem -= 1

    pnl = (pos.shift(1) * dUSDJPY).dropna()
    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB8 US-JP real rate gap")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "US10Y_real - (JP10Y_nominal - JP_CPI_YoY). When 20d change > +25bps AND actual USDJPY 20d return < 0.5 * 5y-beta * diff -> long USDJPY 20d.",
        "data_source": "FRED DFII10, IRLTLT01JPM156N, JPNCPIALLMINMEI (monthly, ffilled), DEXJPUS",
        "caveat": "JPNCPIALLMINMEI series ends ~2021 on FRED; lookback may be limited near present.",
        "n_triggers": int(cond.sum()),
    })


if __name__ == "__main__":
    main()
