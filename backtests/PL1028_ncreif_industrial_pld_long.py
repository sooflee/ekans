"""PL1028_ncreif_industrial_pld_long — Retail-Demand Surge → Industrial REIT Tightening → Long PLD vs IYR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1028_ncreif_industrial_pld_long"
    try:
        fred = load_fred("RETAILSMSA", start="2010-01-01")
        retail = fred.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if retail.empty:
        return mark_failed(sid, "no RETAILSMSA data")

    # Compute monthly YoY growth
    retail_monthly = retail.resample("MS").last()
    yoy = retail_monthly.pct_change(12)  # 12-month YoY
    # 3-month rolling average of YoY
    roll3 = yoy.rolling(3).mean()

    # Find signal dates: 3M avg YoY crosses above 5%
    # Cross = was below 5% last month, now above 5%
    triggers = []
    for i in range(1, len(roll3)):
        if pd.isna(roll3.iloc[i]) or pd.isna(roll3.iloc[i-1]):
            continue
        if roll3.iloc[i] > 0.05 and roll3.iloc[i-1] <= 0.05:
            # Use the date of the data release (approx ~45 days after month end)
            trigger_date = roll3.index[i] + pd.DateOffset(days=45)
            triggers.append(trigger_date)

    print(f"Raw signal events: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no signal events found")

    try:
        px = load_prices(["PLD", "IYR", "SPY"], start="2010-01-01")
    except Exception as e:
        try:
            px = load_prices(["PLD", "IYR", "SPY"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    ret = daily_returns(px)
    pld_r = ret["PLD"]
    spy_r = ret["SPY"]

    hold_days = 60  # calendar days hold period, use trading days approximation (~42 trading days)
    hold_td = 42   # ~60 calendar days in trading days

    pnl = pd.Series(0.0, index=pld_r.index)
    events = []
    last_exit_date = None

    for td in triggers:
        # 30-day re-entry lock from prior exit
        if last_exit_date is not None and (td - last_exit_date).days < 30:
            continue

        # Find next available trading date on or after trigger
        mask = pld_r.index >= td
        if mask.sum() < hold_td:
            continue

        entry_idx = pld_r.index[mask][0]
        p = pld_r.index.get_loc(entry_idx)
        ep = min(p + hold_td, len(pld_r))

        # Check exit condition: RETAILSMSA 3M avg YoY falls below 3%
        # Find the roll3 value during the holding period; if below 3%, early exit
        exit_p = ep
        for j in range(p, ep):
            hold_date = pld_r.index[j]
            # Find most recent roll3 value <= hold_date
            r3_avail = roll3[roll3.index <= hold_date]
            if len(r3_avail) > 0 and r3_avail.iloc[-1] < 0.03:
                exit_p = j
                break

        seg = pld_r.iloc[p:exit_p]
        if len(seg) == 0:
            continue

        pnl.iloc[p:exit_p] = seg.values
        last_exit_date = pld_r.index[exit_p - 1]

        # Compute event return
        pld_ret = float((1 + seg).prod() - 1)
        spy_seg = spy_r.iloc[p:exit_p]
        spy_ret = float((1 + spy_seg).prod() - 1) if len(spy_seg) > 0 else None

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": len(seg),
            "pld_return": round(pld_ret, 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
        })

    print(f"Events executed: {len(events)}")
    if not events:
        return mark_failed(sid, "no executable events")

    # Build non-zero PnL for metrics
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Retail Demand Surge → Long PLD vs IYR")
    returns_list = [e["pld_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Long PLD 60-days when RETAILSMSA 3M avg YoY crosses above 5%; exit when YoY falls below 3% or 60-day stop",
        "mechanism": "Strong retail sales → e-commerce/logistics demand → industrial REIT vacancy tightening → PLD rent growth outperforms",
        "source": "FRED RETAILSMSA; yfinance PLD/IYR/SPY",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, metrics saved.")


if __name__ == "__main__":
    main()
