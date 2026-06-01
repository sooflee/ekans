"""PL828_nato_summit_eu_defense_long — NATO Summit GDP-Floor Ratchet -> Long EU Defense Primes (ITA/RHM.DE)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known NATO summit open dates where communique produced a GDP-floor ratchet.
# Entry = T-15 trading days before summit open; exit = T+5 trading days post-summit.
SUMMIT_DATES = [
    "2014-09-04",  # Wales: 2% pledge
    "2018-07-11",  # Brussels: reaffirm
    "2023-07-11",  # Vilnius: 'minimum 2%'
    "2024-07-09",  # Washington: 3% trajectory
    # 2025-06-24 (The Hague) — future date, skip
]

ENTRY_OFFSET = -15   # trading days before summit open
EXIT_OFFSET  = +5    # trading days after summit (assume day 2 of summit)
HOLD_WINDOW  = 20    # approximate total trading days in window (15 pre + 5 post)


def main():
    sid = "PL828_nato_summit_eu_defense_long"
    try:
        # ITA available since 2001; RHM.DE available since ~2000
        px = load_prices(["ITA", "SPY"], start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load ITA/SPY: {e}")

    # RHM.DE may not be available via yfinance reliably; use ITA as primary
    try:
        rhm = load_prices(["RHM.DE"], start="2013-01-01")
        has_rhm = not rhm.empty and "RHM.DE" in rhm.columns
    except Exception:
        has_rhm = False

    ita_r = daily_returns(px[["ITA"]]).iloc[:, 0].dropna()
    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()

    summit_dates = [pd.Timestamp(d) for d in SUMMIT_DATES]

    # Build a combined trading calendar
    all_dates = ita_r.index

    events = []
    pnl_ita = pd.Series(0.0, index=all_dates)

    for summit_dt in summit_dates:
        # Find closest trading day on or after summit date
        future_mask = all_dates >= summit_dt
        if future_mask.sum() == 0:
            print(f"  Summit {summit_dt.date()}: no trading days found, skipping")
            continue

        summit_idx = all_dates[future_mask][0]
        loc = all_dates.get_loc(summit_idx)

        # Entry = T-15 trading days before summit
        entry_loc = max(0, loc + ENTRY_OFFSET)
        # Exit = T+5 trading days after summit
        exit_loc  = min(len(all_dates) - 1, loc + EXIT_OFFSET)

        if exit_loc <= entry_loc:
            print(f"  Summit {summit_dt.date()}: window too short, skipping")
            continue

        entry_date = all_dates[entry_loc]
        exit_date  = all_dates[exit_loc]

        # ITA window return
        window_r = ita_r.iloc[entry_loc:exit_loc + 1]
        pnl_ita.iloc[entry_loc:exit_loc + 1] += window_r.values

        # SPY window return for excess calc
        spy_window = spy_r.reindex(window_r.index).fillna(0)
        ita_cum  = float((1 + window_r).prod() - 1)
        spy_cum  = float((1 + spy_window).prod() - 1)

        events.append({
            "summit_date":  str(summit_dt.date()),
            "entry_date":   str(entry_date.date()),
            "exit_date":    str(exit_date.date()),
            "ita_return":   round(ita_cum, 4),
            "spy_return":   round(spy_cum, 4),
            "excess_return": round(ita_cum - spy_cum, 4),
        })
        print(f"  Summit {summit_dt.date()}: ITA {ita_cum*100:.1f}%  SPY {spy_cum*100:.1f}%  "
              f"Excess {(ita_cum-spy_cum)*100:.1f}%")

    if not events:
        return mark_failed(sid, "no valid summit events found")

    # Trim pnl to active periods only
    active = pnl_ita[pnl_ita != 0]
    print(f"\nTotal active trading days: {len(active)}")
    print(f"Events: {len(events)}")

    if len(active) < 30:
        # Not enough days for full metrics — use what we have but note it
        print("Warning: fewer than 30 active days; metrics may be unreliable")
        if len(active) < 10:
            return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="NATO Summit GDP-Floor Ratchet → Long ITA")

    returns_list = [e["ita_return"] for e in events]
    excess_list  = [e["excess_return"] for e in events]

    save_result(sid, m, extra={
        "rule": (
            "Enter LONG ITA 15 trading days before NATO summit open date when "
            "5+ NATO capitals publicly support a higher GDP-share floor; "
            "exit 5 trading days post-communique."
        ),
        "mechanism": (
            "NATO summit communique GDP-floor ratchets drive European defense "
            "budget increases, benefiting US-listed defense primes (ITA) "
            "through order-book re-rating."
        ),
        "source": "nato.int summit calendar; yfinance ITA/SPY",
        "n_events": len(events),
        "avg_ita_return":    round(float(np.mean(returns_list)), 4),
        "avg_excess_return": round(float(np.mean(excess_list)), 4),
        "win_rate":          round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events,
    })
    print(f"\nDone: {len(events)} summit events")
    from harness import print_metrics
    print_metrics(m)


if __name__ == "__main__":
    main()
