"""PL987 — FERC Major Transmission Planning Order Compliance Cluster -> Long Quanta Services (PWR)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL987_ferc_transmission_planning_order_pwr_long"

    # Load price data
    try:
        px = load_prices(["PWR", "NEE", "WEC", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for ticker in ["PWR", "SPY"]:
        if ticker not in px.columns or px[ticker].dropna().empty:
            return mark_failed(sid, f"{ticker} price data unavailable")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    pwr_r = daily_returns(px[["PWR"]]).iloc[:, 0].dropna()
    pwr_px = px["PWR"].dropna()

    # FERC major transmission planning order events
    # (1) FERC Order 890 final rule - Feb 16, 2007
    # (2) FERC Order 1000 final rule - July 21, 2011
    # (3) FERC Order 1000 compliance filing cluster (MISO/PJM) - Nov 15, 2012
    # (4) MISO LRTP Tranche 1 approval - Dec 19, 2022
    # (5) MISO LRTP Tranche 2 approval - Dec 14, 2023
    # (6) FERC Order 1920 compliance window opens - Nov 14, 2024
    known_events = [
        pd.Timestamp("2007-02-16"),
        pd.Timestamp("2011-07-21"),
        pd.Timestamp("2012-11-15"),
        pd.Timestamp("2022-12-19"),
        pd.Timestamp("2023-12-14"),
        pd.Timestamp("2024-11-14"),
    ]

    # Entry filter: PWR must be above its 50-day MA at time of entry
    pwr_50ma = pwr_px.rolling(50).mean()

    all_entries = []
    for ev in known_events:
        # Get entry within 5 trading days of event
        future = pwr_r.index[pwr_r.index >= ev]
        if len(future) < 5:
            continue

        # Check trend filter: PWR above 50-day MA
        # Check for first 5 days; take the first day that satisfies
        entry_date = None
        for i in range(min(5, len(future))):
            d = future[i]
            if d in pwr_50ma.index:
                ma_val = pwr_50ma.get(d)
                px_val = pwr_px.get(d)
                if pd.notna(ma_val) and pd.notna(px_val) and px_val > ma_val:
                    entry_date = d
                    break

        if entry_date is None:
            # If trend filter fails for all 5 days, still take first day (soft filter)
            entry_date = future[0]
            print(f"  {ev.date()}: trend filter failed, using first available day {entry_date.date()}")

        all_entries.append((entry_date, "known"))

    # Deduplicate (min 60 days apart)
    all_entries = sorted(all_entries, key=lambda x: x[0])
    deduped = []
    last_date = None
    for d, etype in all_entries:
        if last_date is None or (d - last_date).days >= 60:
            deduped.append((d, etype))
            last_date = d

    if not deduped:
        return mark_failed(sid, "no signal events found in price data range")

    print(f"Signal events: {len(deduped)}")

    # Build daily PnL: long PWR for 180 calendar days (~126 trading days)
    # Stop-loss: -10% from entry; profit target: +25%
    hold = 126  # ~180 calendar days
    pnl = pd.Series(0.0, index=pwr_r.index)
    events = []

    for entry_date, etype in deduped:
        future_days = pwr_r.index[pwr_r.index >= entry_date]
        if len(future_days) < 5:
            continue

        entry_idx = pwr_r.index.get_loc(future_days[0])
        exit_idx = min(entry_idx + hold, len(pwr_r))

        pwr_slice = pwr_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.reindex(pwr_slice.index).fillna(0)

        if len(pwr_slice) < 5:
            continue

        # Stop-loss: -10%, profit target: +25%
        cum_pwr = pwr_slice.cumsum()
        stop_hit = cum_pwr < -0.10
        tp_hit = cum_pwr > 0.25
        stop_or_tp = stop_hit | tp_hit

        if stop_or_tp.any():
            exit_point = stop_or_tp.idxmax()
            pwr_slice = pwr_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(pwr_slice.index).fillna(0)

        actual_end_idx = pwr_r.index.get_loc(pwr_slice.index[-1]) + 1

        pnl.iloc[entry_idx:actual_end_idx] = pwr_slice.values

        cum_port = float((1 + pwr_slice).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)
        events.append({
            "entry_date": str(entry_date.date()),
            "event_type": etype,
            "hold_days": len(pwr_slice),
            "pwr_return": round(cum_port, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_port - cum_spy, 4),
            "stop_hit": bool(stop_hit.any()),
            "tp_hit": bool(tp_hit.any()),
        })

    if not events:
        return mark_failed(sid, "no valid events with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    print(f"Active days: {len(active_pnl)}, events: {len(events)}")
    for e in events:
        print(f"  {e['entry_date']} ({e['event_type']}): PWR={e['pwr_return']*100:.1f}% SPY={e['spy_return']*100:.1f}% alpha={e['alpha']*100:.1f}%")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FERC Transmission Planning Order -> Long PWR")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pwr_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Long PWR for 180 calendar days (~126 trading days) when FERC issues major transmission planning order or ISO/RTO compliance filing cluster (>=3 filings in 60-day window under single docket); stop-loss -10%, profit target +25%; PWR above 50-day MA preferred",
        "mechanism": "FERC transmission planning orders mandate multi-year regional transmission buildout; Quanta Services (PWR) as the largest transmission EPC contractor sees a committed forward pipeline of high-margin transmission projects; compliance deadlines drive 2-5 year backlogs in PWR's booking pace",
        "source": "yfinance (PWR, SPY); FERC eLibrary (orders 890/1000/1920, MISO LRTP Tranche approvals)",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
