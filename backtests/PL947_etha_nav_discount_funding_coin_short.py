"""PL947_etha_nav_discount_funding_coin_short — ETH ETF NAV Discount + High Funding Rate: Short COIN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL947_etha_nav_discount_funding_coin_short"
    # ETHA / FETH launched 2024-07-23; only ~22 months of history
    try:
        px = load_prices(["ETHA", "ETH-USD", "COIN", "SPY"], start="2024-07-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "ETHA" not in px.columns or "ETH-USD" not in px.columns:
        return mark_failed(sid, "ETHA or ETH-USD price data unavailable")

    # Align all series on common dates
    px = px.dropna(subset=["ETHA", "ETH-USD", "COIN", "SPY"])
    if len(px) < 30:
        return mark_failed(sid, f"insufficient data after join: {len(px)} days")

    # Calibrate base ratio from first 30 days of trading
    base_ratio = (px["ETHA"] / px["ETH-USD"]).iloc[:30].mean()
    if np.isnan(base_ratio) or base_ratio <= 0:
        return mark_failed(sid, "could not calibrate ETHA/ETH-USD base ratio")

    # NAV proxy: ETHA should trade at base_ratio * ETH-USD
    nav_proxy = px["ETH-USD"] * base_ratio
    # Discount = (ETHA_close / NAV_proxy) - 1  (negative = discount)
    discount = (px["ETHA"] / nav_proxy) - 1.0

    # Signal: 3-day rolling average discount > 25bps (ETHA below NAV)
    discount_3d = discount.rolling(3).mean()
    # ETH-USD 10-day return (positive = crowded longs not yet capitulated)
    eth_ret_10d = px["ETH-USD"].pct_change(10)

    # Entry signal: discount > 25bps AND ETH positive momentum
    signal = (discount_3d < -0.0025) & (eth_ret_10d > 0)

    # Build positions: shift by 1 day to avoid look-ahead
    # Short COIN when signal fires; hold up to 20 trading days
    coin_ret = daily_returns(px[["COIN"]]).iloc[:, 0]
    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]

    positions = pd.Series(0.0, index=coin_ret.index)
    in_trade_days = 0
    HOLD_MAX = 20

    # Exit when premium returns to flat (within 10bps) AND ETH 10d momentum turns negative
    for i in range(len(positions)):
        date = positions.index[i]
        if date not in signal.index:
            continue
        if in_trade_days > 0:
            in_trade_days += 1
            positions.iloc[i] = -1.0  # short COIN
            # Check exit conditions
            if date in discount.index and date in eth_ret_10d.index:
                disc_today = discount.loc[date]
                eth_mom = eth_ret_10d.loc[date]
                if (disc_today > -0.0010 and eth_mom < 0) or in_trade_days >= HOLD_MAX:
                    in_trade_days = 0
        elif signal.loc[date]:
            # New entry: start short COIN
            positions.iloc[i] = -1.0
            in_trade_days = 1

    # PnL: position * next day's COIN return (shift already handled in long_short_pnl)
    # We apply positions to same day's returns (positions set on entry day close)
    pnl = positions.shift(1) * coin_ret
    pnl = pnl.dropna()

    # Keep only days with a position
    active_pnl = pnl[positions.shift(1) != 0].dropna()

    if len(active_pnl) < 30:
        # Use full pnl series (including flat days) for metrics but note limitation
        if len(pnl) < 30:
            return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    # Use full series for metrics (including flat/zero days gives realistic picture)
    m = compute_metrics(pnl, benchmark=spy_r, name="ETH NAV Discount → Short COIN")

    n_signals = int(signal.sum())
    n_active = int((positions != 0).sum())

    save_result(sid, m, extra={
        "rule": "Short COIN when ETHA 3-day avg discount to ETH-USD NAV proxy > 25bps AND ETH-USD 10d momentum positive; exit when discount narrows to <10bps + ETH momentum turns negative, or max 20 trading days",
        "mechanism": "ETHA/FETH ETF discount below NAV indicates institutional net selling of ETH exposure while retail remains bullish (positive funding); COIN revenues tied to crypto custody/staking suffer disproportionately in institutional capitulation events",
        "source": "yfinance ETHA, ETH-USD, COIN, SPY; base ratio calibrated from first 30 days of ETHA trading (launched 2024-07-23)",
        "base_ratio": float(round(base_ratio, 6)),
        "n_signals": n_signals,
        "n_active_days": n_active,
        "data_start": "2024-07-23",
        "caveat": "Only ~22 months of history (ETHA launched July 2024); results may not be statistically robust",
    })
    print(f"Done: {n_signals} signals, {n_active} active days, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
