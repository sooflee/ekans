"""
AB3 MXN oil-decoupling carry
FRED DEXMXUS (USDMXN) + DCOILWTICO + IR3TIB01MXM156N (MX 3M, monthly) + DGS1 (US 1Y).
When 90d corr(MXN_return, WTI_return) < 0.15 (MXN behaving idiosyncratically) AND
MX_short_rate - US_short_rate > +500bps (carry pickup), short USDMXN (long MXN) to harvest carry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_fred, load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "AB3_mxn_oil_decoupling_carry"
    try:
        fx = load_fred(["DEXMXUS", "DCOILWTICO"], start="2003-01-01").dropna()
        mx_rate = load_fred(["IR3TIB01MXM156N"], start="2003-01-01").iloc[:, 0]  # monthly
        us_rate = load_fred(["DGS1"], start="2003-01-01").iloc[:, 0]
        spy = load_prices(["SPY"], start="2003-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if len(fx) < 500:
        return mark_failed(sid, "insufficient FRED data")

    fx = fx.copy()
    fx.columns = ["USDMXN", "WTI"]
    fx["dUSDMXN"] = np.log(fx["USDMXN"]).diff()
    fx["mxn_ret"] = -fx["dUSDMXN"]
    fx["wti_ret"] = np.log(fx["WTI"]).diff()
    fx = fx.dropna()

    # align rates to daily index, forward fill
    rates = pd.concat([mx_rate.rename("MX"), us_rate.rename("US")], axis=1).sort_index().ffill()
    rates = rates.reindex(fx.index, method="ffill")

    # carry spread in percentage points
    spread = (rates["MX"] - rates["US"])  # in %
    corr = fx["mxn_ret"].rolling(90).corr(fx["wti_ret"])

    cond = (corr.abs() < 0.15) & (spread > 5.0)

    # Trade: short USDMXN (long MXN). hold while condition true (no fixed exit; check daily).
    # We also harvest carry: daily carry pickup ~= spread / 252 / 100.
    pos = (-1.0) * cond.astype(float)  # -1 on USDMXN

    # daily PnL = position * (-d log(USDMXN)) when shorting USDMXN = position * mxn_ret
    # plus carry accrual: daily_carry = spread/100/252 when long MXN (position == -1 USDMXN -> + carry)
    daily_carry = (spread / 100.0 / 252.0) * (-pos)  # if pos==-1 (long MXN), carry positive
    spot_pnl = pos.shift(1) * fx["dUSDMXN"]
    carry_pnl = daily_carry.shift(1).fillna(0.0)
    pnl = (spot_pnl + carry_pnl).dropna()

    spy_r = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_r, name="AB3 MXN oil-decoupling carry")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "If |corr_90d(MXN, WTI)| < 0.15 AND (MX 3M - US 1Y) > +5%pts -> short USDMXN (long MXN) + carry.",
        "data_source": "FRED DEXMXUS, DCOILWTICO, IR3TIB01MXM156N (monthly), DGS1",
        "n_active_days": int(cond.sum()),
    })


if __name__ == "__main__":
    main()
