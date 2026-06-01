"""PL938_syf_bfh_nco_spread_retailer_short Synchrony/Bread Financial NCO Spread Widening: Short KSS/M/ANF"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL938_syf_bfh_nco_spread_retailer_short"
    hold_days = 90

    try:
        px = load_prices(["SYF", "COF", "KSS", "M", "ANF", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    rets = daily_returns(px)
    spy_r = rets["SPY"].dropna()

    # Build rolling signal:
    # SYF 63-day cumulative return minus COF 63-day cumulative return
    syf_cum63 = (1 + rets["SYF"].fillna(0)).rolling(63).apply(lambda x: x.prod()) - 1
    cof_cum63 = (1 + rets["COF"].fillna(0)).rolling(63).apply(lambda x: x.prod()) - 1
    spread = syf_cum63 - cof_cum63

    # Signal: spread < -0.12 (SYF underperforming COF by >12pp on 63-day basis)
    raw_signal = (spread < -0.12).astype(float)

    # Avoid entering mid-cluster: only enter on first day of a new signal episode
    # Use edge detection: signal transitions from 0 to 1
    signal_entry = ((raw_signal == 1) & (raw_signal.shift(1) == 0)).astype(float)

    # Build position series: hold for hold_days after each entry
    # Also include VIX proxy filter: don't enter when market is in risk-off
    # Proxy: SPY 5-day return < -5% (avoid VIX spike >35 days)
    spy_5d = (1 + spy_r.fillna(0)).rolling(5).apply(lambda x: x.prod()) - 1
    risk_off = (spy_5d < -0.05).astype(float)

    all_dates = px.index
    position = pd.Series(0.0, index=all_dates)

    in_trade = False
    trade_end = None
    for date in all_dates:
        if in_trade and date > trade_end:
            in_trade = False

        if not in_trade:
            if signal_entry.get(date, 0) == 1 and risk_off.get(date, 0) == 0:
                # Enter trade
                idx_loc = all_dates.get_loc(date)
                end_loc = min(idx_loc + hold_days - 1, len(all_dates) - 1)
                trade_end = all_dates[end_loc]
                in_trade = True
                position[date] = -1.0
        else:
            position[date] = -1.0

    # Position is -1 when short the basket (KSS 33%, M 33%, ANF 34%)
    # PnL = -1 * (0.33*KSS + 0.33*M + 0.34*ANF)
    basket_r = (0.33 * rets["KSS"].fillna(0) +
                0.33 * rets["M"].fillna(0) +
                0.34 * rets["ANF"].fillna(0))

    # Shift position by 1 for no look-ahead
    pos_shifted = position.shift(1).fillna(0)
    pnl = (pos_shifted * basket_r).dropna()

    # Filter to only periods when we actually had a position
    active = pnl[pos_shifted.reindex(pnl.index).fillna(0) != 0]

    if len(active) < 60:
        # Not enough active days; use full series with zeros
        if len(pnl) < 60:
            return mark_failed(sid, "insufficient signal-active days")
        # Use full series
        pass
    else:
        pnl = active

    spy_aligned = spy_r.reindex(pnl.index).dropna()

    m = compute_metrics(pnl, benchmark=spy_aligned,
                        name="SYF/COF NCO Spread Short KSS/M/ANF")
    save_result(sid, m, extra={
        "rule": "Short KSS+M+ANF basket when SYF 63-day return underperforms COF by >12pp "
                "(proxy for SYF NCO rate widening vs general-purpose issuers). Hold 90 days.",
        "mechanism": "SYF/BFH NCO spread widening signals deteriorating subprime/near-prime consumer credit; "
                     "private-label anchor retailers (KSS, M, ANF) experience sharper revenue declines "
                     "than peers as their credit-dependent customer base pulls back.",
        "source": "SYF, COF, KSS, M, ANF price data via yfinance; NCO proxy via SYF/COF relative return",
        "status": "ok",
    })


if __name__ == "__main__":
    main()
