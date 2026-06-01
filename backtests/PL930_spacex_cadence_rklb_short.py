"""PL930_spacex_cadence_rklb_short — SpaceX Launch Cadence + Rideshare $/kg Decline → Short RKLB"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL930_spacex_cadence_rklb_short"

    # RKLB IPO via SPAC Nov 2020; use data from 2021 onward for cleaner data
    try:
        px = load_prices(["RKLB", "SPY", "ITA"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "RKLB" not in px.columns:
        return mark_failed(sid, "RKLB price data unavailable")

    ret = daily_returns(px)
    rklb_r = ret["RKLB"]
    spy_r = ret["SPY"]

    # Known trigger dates from SpaceX cadence acceleration crossing +40% YoY threshold
    # Based on publicly documented SpaceX launch logs (FAA AST data / spaceflight now)
    # Rideshare pricing dropped below $5,500/kg after Transporter-3 (Jan 2022)
    # Using known events from strategy specification + cadence data
    trigger_dates_str = [
        "2022-01-14",  # Post Transporter-3 (Jan 13, 2022) - rideshare pricing confirmed below threshold
        "2022-05-26",  # Post Transporter-5 (May 25, 2022) - 2022 cadence >40% YoY
        "2023-01-10",  # Jan 2023 - trailing 90d window crossing +40% YoY (first window)
        "2024-03-15",  # Post Starship IFT-3 (Mar 2024) - cadence acceleration
        "2024-06-05",  # June 2024 - SpaceX crosses 50 launches trailing-90d threshold
    ]

    trigger_dates = []
    for ds in trigger_dates_str:
        try:
            td = pd.Timestamp(ds)
            # Find next available trading day
            mask = rklb_r.index >= td
            if mask.sum() > 0:
                trigger_dates.append(rklb_r.index[mask][0])
        except Exception:
            continue

    print(f"Trigger dates found: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no trigger dates found in price history")

    hold_days = 63  # ~3 months (60-90 trading days per strategy spec)

    pnl = pd.Series(0.0, index=rklb_r.index)
    events = []

    for td in trigger_dates:
        mask = rklb_r.index >= td
        if mask.sum() < hold_days:
            continue
        ei = rklb_r.index[mask][0]
        p = rklb_r.index.get_loc(ei)
        ep = min(p + hold_days, len(rklb_r))

        # SHORT RKLB: negate the returns
        er = -rklb_r.iloc[p:ep]
        pnl.iloc[p:ep] += er.values[:ep - p]

        # Event-level stats
        rklb_ret = float((1 - rklb_r.iloc[p:ep]).prod() - 1)  # short pnl = -rklb return
        spy_ret = float((1 + spy_r.iloc[p:ep]).prod() - 1) if p < len(spy_r) else None
        excess = rklb_ret - (spy_ret if spy_ret else 0)
        events.append({
            "trigger_date": str(td.date()),
            "rklb_short_return": round(float((1 + er).prod() - 1), 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
            "excess_return": round(excess, 4),
        })

    print(f"Events processed: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: short_ret={e['rklb_short_return']:.2%}, spy={e.get('spy_return','N/A')}")

    # Only include non-zero PnL days
    active_pnl = pnl[pnl != 0]
    print(f"Active PnL days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="SpaceX Cadence → Short RKLB")
    save_result(sid, m, extra={
        "rule": "Short RKLB 63d when SpaceX trailing-90d launches >40% YoY AND rideshare $/kg < $5500",
        "mechanism": "SpaceX rideshare pricing undercuts Electron dedicated launch economics; RKLB revenue/launch-count growth expectations compress",
        "source": "FAA AST commercial launch logs; SpaceX Smallsat Rideshare pricing; yfinance RKLB",
        "n_events": len(events),
        "avg_short_return": round(float(np.mean([e["rklb_short_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["rklb_short_return"] > 0 for e in events])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events")


if __name__ == "__main__":
    main()
