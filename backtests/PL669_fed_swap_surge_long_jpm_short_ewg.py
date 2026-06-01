"""PL669 — Fed Swap-Line Drawdown Surge - Long JPM Short EWG"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL669_fed_swap_surge_long_jpm_short_ewg"

    try:
        # SWPT = Central Bank Liquidity Swaps (billions USD, weekly)
        fred_df = load_fred("SWPT", start="2008-01-01")
        swpt = fred_df.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if swpt.empty:
        return mark_failed(sid, "no SWPT data")

    try:
        px = load_prices(["JPM", "EWG", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty:
        return mark_failed(sid, "no price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Identify surge events: swap-line outstanding rises >$5B WoW from baseline <$1B
    events = []
    baseline = 1.0  # billion
    surge_threshold = 5.0  # billion

    # Weekly SWPT series - compute WoW change
    swpt_wow = swpt.diff()

    for i in range(1, len(swpt)):
        prev_val = float(swpt.iloc[i - 1])
        curr_val = float(swpt.iloc[i])
        wow_change = curr_val - prev_val

        # Baseline sub-$1B, surge >$5B WoW
        if prev_val < baseline and wow_change > surge_threshold:
            event_date = swpt.index[i]
            # Avoid clustering: skip if within 30 days of prior event
            if events and (event_date - events[-1]).days < 30:
                continue
            events.append(event_date)

    print(f"Swap surge events identified: {len(events)}")
    for ev in events:
        print(f"  {ev.date()} (SWPT level: {swpt.loc[ev]:.1f}B, WoW change: {swpt_wow.loc[ev]:.1f}B)")

    if len(events) == 0:
        return mark_failed(sid, "no swap surge events found")

    hold = 20  # trading days
    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for event_date in events:
        mask = ret.index >= event_date
        if mask.sum() == 0:
            continue
        start_idx = ret.index[mask][0]
        p = ret.index.get_loc(start_idx)
        end_idx = min(p + hold, len(ret))

        window_ret = ret.iloc[p:end_idx]

        if "JPM" not in window_ret.columns or "EWG" not in window_ret.columns:
            continue

        jpm_win = window_ret["JPM"].fillna(0)
        ewg_win = window_ret["EWG"].fillna(0)

        # Long JPM, short EWG (1:1)
        event_pnl = jpm_win - ewg_win

        already_in = (pnl.iloc[p:end_idx] != 0).any()
        if not already_in:
            pnl.iloc[p:end_idx] = event_pnl.values

        total_return = float((1 + event_pnl).prod() - 1)
        spy_win = spy_r.iloc[p:end_idx]
        spy_total = float((1 + spy_win).prod() - 1) if len(spy_win) > 0 else None

        event_records.append({
            "event_date": str(event_date.date()),
            "pnl_return": round(total_return, 4),
            "spy_return": round(spy_total, 4) if spy_total is not None else None,
        })

    print(f"Events with valid data: {len(event_records)}")
    active_pnl = pnl[pnl != 0]
    print(f"Active days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Fed Swap Surge Long JPM Short EWG")
    save_result(sid, m, extra={
        "rule": "Long JPM, short EWG (1:1) for 20 days when FRBNY swap-line outstanding rises >$5B WoW from sub-$1B baseline",
        "mechanism": "Central bank swap-line surge signals global USD funding stress; US bank (JPM) benefits as safe-haven USD provider vs European equities (EWG) which face capital outflows",
        "source": "FRED SWPT (Central Bank Liquidity Swaps); yfinance",
        "n_events": len(event_records),
        "events": event_records,
        "status": "ok",
    })
    print(f"Done: {len(event_records)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
