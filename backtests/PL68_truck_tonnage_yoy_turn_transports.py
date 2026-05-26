"""PL68_truck_tonnage_yoy_turn_transports — ATA Truck Tonnage YoY Positive Turn → Long Transports
When FRED TRUCKD11 YoY turns positive after 6+ months negative, long IYT 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL68_truck_tonnage_yoy_turn_transports"

    # Try FRED series for truck tonnage
    tonnage = None
    series_used = None
    for series_id in ["TRUCKD11", "TSIFRGHT"]:
        try:
            tonnage = load_fred(series_id, start="1998-01-01").squeeze()
            if len(tonnage.dropna()) > 24:
                series_used = series_id
                break
        except Exception:
            continue

    try:
        px = load_prices(["IYT", "SPY"], start="2002-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    ret = daily_returns(px)
    iyt_r = ret["IYT"]
    spy_r = ret["SPY"]

    if tonnage is not None and series_used is not None:
        # Programmatic: compute YoY, find positive turn after 6+ months negative
        ton_m = tonnage.resample("ME").last().dropna()
        ton_yoy = ton_m.pct_change(12)

        neg_count = 0
        triggers = []
        last_trigger = None
        for i in range(len(ton_yoy)):
            val = float(ton_yoy.iloc[i])
            if np.isnan(val):
                neg_count = 0
                continue
            if val < 0:
                neg_count += 1
            elif val >= 0 and neg_count >= 6:
                trig = ton_yoy.index[i]
                if last_trigger is None or (trig - last_trigger).days > 300:
                    triggers.append(trig)
                    last_trigger = trig
                neg_count = 0
            else:
                neg_count = 0
        print(f"Using FRED {series_used}: found {len(triggers)} tonnage YoY positive turn events")
    else:
        # Fall back to hand-coded dates
        print("No FRED truck tonnage series available — using hand-coded turn dates")
        series_used = "hand-coded"
        triggers = pd.to_datetime(["2002-01-01", "2009-09-01", "2016-06-01", "2020-07-01"])

    events = []
    pnl_parts = []
    hold_days = 126

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        iyt_window = iyt_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(iyt_window)

        iyt_cum = float((1 + iyt_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()) if hasattr(trig_date, 'date') else str(trig_date),
            "iyt_6m_return": round(iyt_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(iyt_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No truck tonnage positive-turn events found with enough data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Truck Tonnage YoY Turn → Long IYT")

    avg_iyt = np.mean([e["iyt_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["iyt_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": f"FRED {series_used} YoY turns positive after 6+ months negative → long IYT 6mo",
        "mechanism": "Freight recovery = early-cycle industrial inflection → transports re-rate",
        "source": f"FRED {series_used} + yfinance",
        "n_events": len(events),
        "avg_iyt_return": round(avg_iyt, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg IYT: {avg_iyt*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["iyt_6m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: IYT {e['iyt_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
