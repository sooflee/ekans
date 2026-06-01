"""PL932_fcc_space_bureau_asts_short — FCC Space Bureau Orbital Debris NPRM → Short ASTS BlueBird Capex"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL932_fcc_space_bureau_asts_short"

    # ASTS went public via SPAC merger ~Aug 2022; price history starts around then
    try:
        px = load_prices(["ASTS", "SPY", "RKLB"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "ASTS" not in px.columns:
        return mark_failed(sid, "ASTS price data unavailable")

    ret = daily_returns(px)
    asts_r = ret["ASTS"]
    spy_r = ret["SPY"]

    # Known FCC Space Bureau regulatory trigger dates from strategy specification
    # FCC Order tightening LEO orbital debris/disposal rules → increases capex for ASTS BlueBird
    trigger_dates_str = [
        "2022-09-30",  # FCC Order 22-74 final 5-year PMD rule effective (~Sep 29, 2022)
        "2022-12-06",  # FCC Space Bureau IB Docket 22-271 NPRM comment period
        "2023-08-15",  # ASTS BlueBird FM1-FM2 launch and FCC modification filing
        "2024-04-05",  # FCC Space Bureau follow-on constellation NPRM (active maneuverability)
    ]

    trigger_dates = []
    for ds in trigger_dates_str:
        try:
            td = pd.Timestamp(ds)
            mask = asts_r.index >= td
            if mask.sum() > 0:
                trigger_dates.append(asts_r.index[mask][0])
        except Exception:
            continue

    print(f"Trigger dates found: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no trigger dates found in price history")

    hold_days = 45  # ~60 trading days max, use 45 per strategy spec midpoint

    pnl = pd.Series(0.0, index=asts_r.index)
    events = []

    for td in trigger_dates:
        mask = asts_r.index >= td
        if mask.sum() < hold_days:
            continue
        ei = asts_r.index[mask][0]
        p = asts_r.index.get_loc(ei)
        ep = min(p + hold_days, len(asts_r))

        # SHORT ASTS: negate the returns
        er = -asts_r.iloc[p:ep]
        pnl.iloc[p:ep] += er.values[:ep - p]

        asts_short_ret = float((1 + er).prod() - 1)
        spy_slice = spy_r.iloc[p:ep]
        spy_ret = float((1 + spy_slice).prod() - 1) if len(spy_slice) > 0 else None
        events.append({
            "trigger_date": str(td.date()),
            "asts_short_return": round(asts_short_ret, 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
        })

    print(f"Events processed: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: short_ret={e['asts_short_return']:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active PnL days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FCC Debris NPRM → Short ASTS")
    save_result(sid, m, extra={
        "rule": "Short ASTS 45d when FCC Space Bureau publishes NPRM/Order tightening LEO disposal requirements",
        "mechanism": "Tighter debris rules increase BlueBird constellation capex and re-licensing burden; ASTS cash burn accelerates",
        "source": "FCC IB Docket filings (fcc.gov); yfinance ASTS",
        "n_events": len(events),
        "avg_short_return": round(float(np.mean([e["asts_short_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["asts_short_return"] > 0 for e in events])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events")


if __name__ == "__main__":
    main()
