"""
P9 Benjamin Cowen BTC "risk metric".

Compute custom: rolling 4-year regression of log(BTC price) vs log(days since
inception), then take the z-score of the residual (over the same window).
Interpret this z-score as the "risk metric" proxy.

Rule (continuous): when risk-metric <= 0.4 (interpreted relative to a [0,1]
normalized scale of z-score percentile), long BTC; when >= 0.75, exit.

To make this concrete, we normalize the residual to a [0,1] percentile within
the rolling 4-year window — that's the closest replica of Cowen's "risk band".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result,
)


def main():
    btc = load_prices(["BTC-USD"], start="2014-01-01")["BTC-USD"]
    btc = btc.dropna()
    log_price = np.log(btc)
    # days since first observation
    days = np.asarray((btc.index - btc.index[0]).days, dtype=float)
    days[days < 1] = 1.0
    log_days = np.log(days)

    df = pd.DataFrame({"log_p": log_price.values, "log_d": log_days}, index=btc.index)
    win = 252 * 4  # 4 years

    risk = pd.Series(index=df.index, dtype=float)
    for i in range(win, len(df)):
        x = df["log_d"].iloc[i - win:i].values
        y = df["log_p"].iloc[i - win:i].values
        # linear regression
        b = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        a = y.mean() - b * x.mean()
        resid = y - (a + b * x)
        # residual today vs window distribution
        today_resid = df["log_p"].iloc[i] - (a + b * df["log_d"].iloc[i])
        # Percentile rank of today's residual within the window
        pct = (resid < today_resid).mean()
        risk.iloc[i] = pct

    # Strategy: regime-state machine. Enter long when risk<=0.4; exit when risk>=0.75.
    state = pd.Series(False, index=btc.index)
    on = False
    for d in btc.index:
        r = risk.loc[d] if d in risk.index else np.nan
        if not np.isnan(r):
            if not on and r <= 0.4:
                on = True
            elif on and r >= 0.75:
                on = False
        state.loc[d] = on

    rets = btc.pct_change()
    pnl = (state.shift(1).fillna(False).astype(float) * rets).dropna()
    spy = load_prices(["SPY"], start="2014-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy.dropna(), name="P9 Cowen BTC Risk")
    print_metrics(m)
    n_long_days = int(state.sum())
    save_result("P9_cowen_btc_risk", m, extra={
        "status": "ok",
        "rule": "Rolling 4y log-log regression of BTC; residual percentile (window-rank) is 'risk metric'. Long when risk<=0.4, exit when >=0.75.",
        "mechanism": "Cowen: BTC log-trend channel; price below trend = accumulation, above = distribution.",
        "universe": "BTC-USD daily.",
        "n_long_days": n_long_days,
        "source": "Benjamin Cowen (YouTube, Phase 1P)",
    })


if __name__ == "__main__":
    main()
