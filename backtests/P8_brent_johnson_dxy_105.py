"""
P8 Brent Johnson "Dollar Milkshake" / DXY 105 trigger.

Rule: When DXY closes > 105 for 5 consecutive sessions, short EEM + long UUP
(equal weight, +1 UUP / -1 EEM as a $-neutral pair) until DXY closes < 105.

Implementation:
- Use DX-Y.NYB (ICE Dollar Index) from yfinance for DXY.
- Track regime: signal goes on when rolling-5d-min(DXY close) > 105;
  goes off when DXY close < 105 on any day.
- PnL = mean(UUP return, -EEM return) on regime days (shifted +1 day).
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
    # DXY: yfinance ticker is DX-Y.NYB. UUP and EEM are ETFs.
    dxy = load_prices(["DX-Y.NYB"], start="2007-03-01")["DX-Y.NYB"]
    etfs = load_prices(["UUP", "EEM"], start="2007-03-01")
    px = etfs.join(dxy.rename("DXY"), how="inner").dropna()
    rets = px.pct_change()

    # 5-consecutive-day rule: regime ON if rolling 5d min of DXY > 105
    # regime OFF on any day DXY close drops below 105.
    regime_on = px["DXY"].rolling(5).min() > 105

    # Build a stateful regime that turns on at the moment the 5-day rule holds,
    # and turns off the next day DXY < 105.
    state = pd.Series(False, index=px.index)
    on = False
    for i, d in enumerate(px.index):
        if on:
            if px["DXY"].iloc[i] < 105:
                on = False
        else:
            if bool(regime_on.iloc[i]):
                on = True
        state.iloc[i] = on

    # Pair PnL: +1 UUP, -1 EEM (each half-weight so notional is 1)
    # PnL applied with 1-day lag.
    pair_ret = 0.5 * rets["UUP"] - 0.5 * rets["EEM"]
    pnl = (state.shift(1).fillna(False).astype(float) * pair_ret).dropna()

    spy = load_prices(["SPY"], start="2007-03-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="P8 Brent Johnson DXY>105 (UUP-EEM)")
    print_metrics(m)
    in_regime_days = int(state.sum())
    save_result("P8_brent_johnson_dxy_105", m, extra={
        "status": "ok",
        "rule": "When DXY > 105 for 5 consecutive sessions, +UUP / -EEM (half-weight each); exit when DXY < 105.",
        "mechanism": "Brent Johnson 'Dollar Milkshake': strong USD drains EM liquidity; EM equities underperform.",
        "universe": "UUP long / EEM short pair; signal: DX-Y.NYB.",
        "in_regime_days": in_regime_days,
        "source": "Brent Johnson, Santiago Capital (Phase 1P, YouTube)",
    })


if __name__ == "__main__":
    main()
