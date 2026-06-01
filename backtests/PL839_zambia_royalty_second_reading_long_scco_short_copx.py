"""PL839_zambia_royalty_second_reading_long_scco_short_copx — Zambia Mineral Royalty Bill Second Reading -> Long SCCO / Short COPX Spread"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known analog events for the Zambia royalty bill spread trade
# 2019-01-30: Zambia royalty hike announcement (Jan 2019 Mineral Royalty Act amendment)
#   -> Long SCCO (Peru/Mexico, zero Zambia exposure), Short COPX (contains ~8% FQM/Zambia miners)
# 2022-10-14: Hichilema partial reversal of 2019 hike (mining concessions announced)
#   -> Inverted direction: Long COPX, Short SCCO (thesis reversal)
EVENTS = [
    {"date": "2019-01-30", "direction": 1,  "desc": "Zambia royalty hike (Long SCCO, Short COPX)"},
    {"date": "2022-10-14", "direction": -1, "desc": "Zambia royalty reversal (Long COPX, Short SCCO)"},
]

# Hold T+2 to T+30 business days after the event (2-day announcement delay included)
PRE_DAYS  = 2   # enter at T+2 to avoid immediate reaction noise
HOLD_DAYS = 30  # hold 30 business days (~6 weeks)


def main():
    sid = "PL839_zambia_royalty_second_reading_long_scco_short_copx"
    try:
        px = load_prices(["SCCO", "COPX", "BHP", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    for col in ["SCCO", "COPX"]:
        if col not in ret.columns:
            return mark_failed(sid, f"missing column {col}")

    events_out = []
    pnl = pd.Series(0.0, index=spy_r.index)

    for ev in EVENTS:
        ev_dt = pd.Timestamp(ev["date"])
        direction = ev["direction"]  # +1 = Long SCCO / Short COPX; -1 = reversed

        # Find entry: T+PRE_DAYS trading days after event
        future_mask = spy_r.index >= ev_dt
        if future_mask.sum() < PRE_DAYS + 1:
            print(f"  Event {ev['date']}: insufficient data post-event, skipping")
            continue

        post_dates   = spy_r.index[future_mask]
        entry_loc_in_post = min(PRE_DAYS, len(post_dates) - 1)
        entry_date   = post_dates[entry_loc_in_post]
        entry_loc    = spy_r.index.get_loc(entry_date)
        exit_loc     = min(entry_loc + HOLD_DAYS, len(spy_r) - 1)

        if exit_loc <= entry_loc:
            print(f"  Event {ev['date']}: window too short, skipping")
            continue

        # Spread return: direction * (SCCO - COPX)
        scco_w  = spy_r.iloc[entry_loc:exit_loc + 1].index
        scco_r  = ret["SCCO"].reindex(scco_w).fillna(0)
        copx_r  = ret["COPX"].reindex(scco_w).fillna(0)
        spy_w   = spy_r.reindex(scco_w).fillna(0)

        # Spread = Long SCCO, Short COPX (or reversed per direction)
        spread_r = direction * (scco_r - copx_r) * 0.5  # 50% allocation each leg

        # Also track BHP for reference
        bhp_r = ret.get("BHP", pd.Series(dtype=float)).reindex(scco_w).fillna(0)

        pnl.loc[scco_w] += spread_r.values[:len(scco_w)]

        scco_cum  = float((1 + scco_r).prod() - 1)
        copx_cum  = float((1 + copx_r).prod() - 1)
        spy_cum   = float((1 + spy_w).prod() - 1)
        spread_cum = direction * (scco_cum - copx_cum)

        exit_date = spy_r.index[exit_loc]
        events_out.append({
            "event_date":    ev["date"],
            "direction":     direction,
            "desc":          ev["desc"],
            "entry_date":    str(entry_date.date()),
            "exit_date":     str(exit_date.date()),
            "scco_return":   round(scco_cum, 4),
            "copx_return":   round(copx_cum, 4),
            "spread_return": round(spread_cum, 4),
            "spy_return":    round(spy_cum, 4),
        })
        print(f"  {ev['desc']} ({ev['date']}): "
              f"SCCO {scco_cum*100:.1f}%  COPX {copx_cum*100:.1f}%  "
              f"spread {spread_cum*100:.1f}%  SPY {spy_cum*100:.1f}%")

    if not events_out:
        return mark_failed(sid, "no valid events found")

    active = pnl[pnl != 0]
    print(f"\nActive trading days: {len(active)}")
    print(f"Events: {len(events_out)}")

    if len(active) < 10:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r,
                        name="Zambia Royalty Bill 2nd Reading → Long SCCO / Short COPX")

    spread_returns = [e["spread_return"] for e in events_out]
    save_result(sid, m, extra={
        "rule": (
            "Enter Long SCCO (60%) / Short COPX (40%) when Zambian National Assembly "
            "lists a mineral royalty/windfall tax bill for Second Reading with a rate increase; "
            "hold through T+6 weeks or bill resolution. Reverse spread for royalty reversals."
        ),
        "mechanism": (
            "SCCO (Peru/Mexico operations, zero Zambia exposure) outperforms COPX "
            "(contains ~8% FQM + other Zambia-exposed names) when Zambia raises mining royalties, "
            "creating a spread alpha on geopolitical risk segmentation."
        ),
        "source": "parliament.gov.zm Order Papers; mineszambia.com; yfinance SCCO/COPX/BHP/SPY",
        "n_events": len(events_out),
        "avg_spread_return": round(float(np.mean(spread_returns)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in spread_returns])), 4),
        "events": events_out,
    })
    print(f"\nDone: {len(events_out)} Zambia royalty events")
    from harness import print_metrics
    print_metrics(m)


if __name__ == "__main__":
    main()
