"""PL834_bor_tier2a_water_utility_long — Bureau of Reclamation Tier-2a Shortage Declaration -> Long Water Utilities vs Short XLU"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Bureau of Reclamation August 24-Month Study shortage declarations
# Entry = day after publication (~Aug 15 each year)
# Hold = 60 trading days (~3 months)
EVENTS = [
    "2021-08-17",  # Tier-1 first-ever shortage declaration
    "2022-08-15",  # Tier-2a shortage
    "2023-08-16",  # Tier-2a continued
]

HOLD_DAYS = 60   # trading days


def main():
    sid = "PL834_bor_tier2a_water_utility_long"
    try:
        px = load_prices(["AWK", "WTRG", "MSEX", "XLU", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Check available columns
    for col in ["AWK", "WTRG", "MSEX", "XLU"]:
        if col not in ret.columns:
            return mark_failed(sid, f"missing column {col}")

    events = []
    pnl = pd.Series(0.0, index=spy_r.index)

    for ev_str in EVENTS:
        ev_dt = pd.Timestamp(ev_str)

        # Find entry date: first trading day on or after event date
        mask = spy_r.index >= ev_dt
        if mask.sum() == 0:
            print(f"  Event {ev_str}: no trading days, skipping")
            continue

        entry_date = spy_r.index[mask][0]
        entry_loc  = spy_r.index.get_loc(entry_date)
        exit_loc   = min(entry_loc + HOLD_DAYS, len(spy_r) - 1)

        if exit_loc <= entry_loc:
            print(f"  Event {ev_str}: window too short, skipping")
            continue

        # Long basket: equal-weight AWK + WTRG + MSEX
        # Short leg: XLU
        w_long  = {"AWK": 1/3, "WTRG": 1/3, "MSEX": 1/3}
        w_short = {"XLU": -1.0}

        # Build window returns
        long_tickers  = ["AWK", "WTRG", "MSEX"]
        short_tickers = ["XLU"]

        long_r_window  = ret[long_tickers].iloc[entry_loc:exit_loc + 1].fillna(0)
        short_r_window = ret[short_tickers].iloc[entry_loc:exit_loc + 1].fillna(0)
        spy_window     = spy_r.iloc[entry_loc:exit_loc + 1].fillna(0)

        # Equal-weight long basket
        basket_r = long_r_window.mean(axis=1)

        # Long-short spread: long basket minus XLU
        spread_r = basket_r - short_r_window["XLU"]

        # Accumulate PnL (half-weight each leg: 0.5 long, 0.5 short for dollar neutral)
        spread_r_scaled = spread_r * 0.5

        idx = spy_r.index[entry_loc:exit_loc + 1]
        pnl.loc[idx] += spread_r_scaled.values[:len(idx)]

        # Cumulative returns for reporting
        long_cum = float((1 + basket_r).prod() - 1)
        xlu_cum  = float((1 + short_r_window["XLU"]).prod() - 1)
        spy_cum  = float((1 + spy_window).prod() - 1)
        spread_cum = long_cum - xlu_cum

        exit_date = spy_r.index[exit_loc]
        events.append({
            "event_date":    ev_str,
            "entry_date":    str(entry_date.date()),
            "exit_date":     str(exit_date.date()),
            "basket_return": round(long_cum, 4),
            "xlu_return":    round(xlu_cum, 4),
            "spread_return": round(spread_cum, 4),
            "spy_return":    round(spy_cum, 4),
        })
        print(f"  Event {ev_str}: basket {long_cum*100:.1f}%  XLU {xlu_cum*100:.1f}%  "
              f"spread {spread_cum*100:.1f}%  SPY {spy_cum*100:.1f}%")

    if not events:
        return mark_failed(sid, "no valid events found")

    active = pnl[pnl != 0]
    print(f"\nActive trading days: {len(active)}")
    print(f"Events: {len(events)}")

    if len(active) < 10:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="BOR Tier-2a → Long Water Utilities vs Short XLU")

    spread_returns = [e["spread_return"] for e in events]
    save_result(sid, m, extra={
        "rule": (
            "Enter LONG AWK/WTRG/MSEX equal-weight + SHORT XLU when Bureau of Reclamation "
            "August 24-Month Study projects Lake Mead below shortage tier threshold; "
            "hold 60 trading days (~3 months)."
        ),
        "mechanism": (
            "Water utilities serving AZ/NV/SoCal file emergency rate riders with state PUCs "
            "that historically clear within 60-90 days at +6-9% revenue uplift; "
            "XLU does not benefit from drought premium."
        ),
        "source": "USBR 24-Month Study (usbr.gov/lc/region/g4000/24mo); yfinance AWK/WTRG/MSEX/XLU/SPY",
        "n_events": len(events),
        "avg_spread_return": round(float(np.mean(spread_returns)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in spread_returns])), 4),
        "events": events,
    })
    print(f"\nDone: {len(events)} BOR shortage events")
    from harness import print_metrics
    print_metrics(m)


if __name__ == "__main__":
    main()
