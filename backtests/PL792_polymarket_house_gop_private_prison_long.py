"""PL792_polymarket_house_gop_private_prison_long
Polymarket GOP House 2026 Probability Surge -> Long CXW / GEO Private Prison Detention Bet

Event-study backtest using known US election/policy inflection points for
private prison operators (CXW, GEO). Enters long on GOP-favorable policy
events, exits after 15 trading days. Benchmarked vs SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL792_polymarket_house_gop_private_prison_long"
    tickers = ["CXW", "GEO", "SPY"]

    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        # GEO or CXW may not load — try alternate
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Known GOP-favorable policy inflection events for private prisons
    # Entry is at next open (shift by 1 day in our daily model)
    # Positive events (long): Trump win, GOP House 2022 win
    # Negative events (short): Biden win/exec-order - but strategy only longs
    # We model LONG entries on the 2 clear GOP-bullish events
    # and also 2024 Trump win as an event.
    long_entry_dates = [
        "2016-11-09",   # Trump presidential win → private prison policy reversal
        "2022-11-09",   # GOP narrow House win → policy tailwind
        "2024-11-07",   # Trump 2024 win → strong private prison rally
    ]

    hold_days = 15
    basket = [t for t in ["CXW", "GEO"] if t in ret.columns]
    if not basket:
        return mark_failed(sid, "CXW and GEO both missing from price data")

    basket_ret = ret[basket].mean(axis=1).fillna(0)

    positions = pd.Series(0.0, index=ret.index)

    for entry_date_str in long_entry_dates:
        # Find next available trading day on or after entry date
        try:
            entry_dt = pd.Timestamp(entry_date_str)
            future = ret.index[ret.index >= entry_dt]
            if len(future) == 0:
                continue
            # Use the day AFTER entry date (next open model)
            start_idx = ret.index.get_loc(future[0])
            entry_idx = start_idx + 1  # enter at next day open
            exit_idx = entry_idx + hold_days
            if entry_idx >= len(ret.index):
                continue
            exit_idx = min(exit_idx, len(ret.index))
            # Set position = 1 for the holding period
            hold_dates = ret.index[entry_idx:exit_idx]
            positions.loc[hold_dates] = 1.0
        except Exception:
            continue

    pnl = positions.shift(1).fillna(0) * basket_ret
    pnl = pnl.reindex(spy_r.index).fillna(0)

    m = compute_metrics(pnl, benchmark=spy_r, name="Polymarket GOP House CXW/GEO Long")
    save_result(sid, m, extra={
        "rule": "Enter long CXW+GEO 50/50 on GOP-favorable election/policy events; exit after 15 trading days.",
        "mechanism": "GOP administrations expand private prison/ICE detention contracts (CXW and GEO derive ~65% revenue from federal/ICE). Policy-beta play on political inflection.",
        "source": "Known election dates: 2016-11-09 Trump win, 2022-11-09 GOP House win, 2024-11-07 Trump win. Proxy for Polymarket GOP House 2026 probability surge.",
    })


if __name__ == "__main__":
    main()
