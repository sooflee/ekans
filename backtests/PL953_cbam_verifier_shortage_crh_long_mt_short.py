"""PL953_cbam_verifier_shortage_crh_long_mt_short
CBAM Definitive Regime Verifier Shortage: Long CRH, Short MT (ArcelorMittal)

Event-study: on each EU Commission CBAM regulatory milestone, long CRH / short MT
equal-dollar pair for 20 trading days. CRH = domestic EU cement/building materials,
net beneficiary of competitor cost uplift; MT = ArcelorMittal ADR, exposed to
Turkish/Indian slab feedstock with embedded CBAM emissions surcharges.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL953_cbam_verifier_shortage_crh_long_mt_short"
    tickers = ["CRH", "MT", "EWG", "SPY"]

    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # CBAM regulatory milestone event dates
    event_dates_str = [
        "2023-05-10",  # CBAM Regulation published in Official Journal
        "2023-10-01",  # Transitional phase start
        "2024-01-31",  # First Q3-2023 report deadline
        "2024-04-30",  # Q4-2023 declaration deadline
        "2026-01-01",  # Definitive regime enforcement
        "2026-04-15",  # Q1-2026 quarterly stats release (proxy)
    ]
    event_dates = [pd.Timestamp(d) for d in event_dates_str]

    # For each event: long CRH / short MT equal notional for 20 trading days
    hold_days = 20
    crh_r = ret["CRH"].fillna(0)
    mt_r = ret["MT"].fillna(0)
    pair_r = crh_r - mt_r  # daily long CRH / short MT return (dollar-neutral pair)

    idx_list = list(pair_r.index)
    n = len(idx_list)

    # Build daily pnl series (allow overlapping events — average position)
    daily_positions = pd.Series(0.0, index=pair_r.index)

    events = []
    for evt_date in event_dates:
        # Find first trading day on or after event date
        eligible = [d for d in idx_list if d >= evt_date]
        if not eligible:
            continue
        entry_date = eligible[0]
        entry_idx = idx_list.index(entry_date)
        exit_idx = min(entry_idx + hold_days, n)
        hold_slice = idx_list[entry_idx:exit_idx]
        for d in hold_slice:
            daily_positions[d] += 1.0  # accumulate exposure count
        event_return = float((1 + pair_r.loc[hold_slice]).prod() - 1)
        events.append({
            "event_date": str(evt_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(idx_list[exit_idx - 1].date()),
            "hold_days": len(hold_slice),
            "event_return": round(event_return, 4),
        })

    # Normalize positions: if multiple events overlap, cap at 1 unit
    # (use max-of-1 clamp to avoid leverage amplification)
    daily_positions = daily_positions.clip(upper=1.0)

    # Daily pnl = positions * pair return
    pnl = daily_positions * pair_r
    pnl = pnl.dropna()

    n_active_days = int((pnl != 0).sum())
    n_events = len(events)

    if n_active_days < 10:
        return mark_failed(
            sid,
            f"insufficient in-position days: {n_active_days} (n_events={n_events})"
        )

    # Restrict spy_r to matching index
    spy_r = spy_r.reindex(pnl.index).fillna(0)

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="CBAM CRH Long / MT Short",
        positions=daily_positions.reindex(pnl.index).fillna(0),
        cost_bps=15,
    )

    # Event-level summary stats
    ev_returns = [e["event_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each EU Commission CBAM regulatory milestone, enter long CRH / "
                "short MT equal-dollar pair at open on event date; hold 20 trading days."
            ),
            "mechanism": (
                "CBAM imposes carbon border adjustment on non-EU steel/cement imports. "
                "ArcelorMittal (MT) uses Turkish/Indian slab feedstock subject to "
                "default verifier-shortage penalty pricing (25-40% above actual for "
                "non-EU steel). CRH as domestic EU incumbent benefits from higher "
                "competitor costs. Each CBAM milestone heightens enforcement attention."
            ),
            "source": (
                "yfinance auto-adjusted close; EU Commission CBAM Regulation tracker; "
                "EU Official Journal May 2023"
            ),
            "tickers": ["CRH", "MT"],
            "event_dates": event_dates_str,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Only 6 event dates — very small sample; any single event dominates "
                "statistics. CRH transferred primary listing from Dublin to NYSE "
                "September 2023; pre-2023 data is ADR. MT is US-listed ADR (not MT.PA). "
                "Two events (Jan 2026, Apr 2026) are very recent; post-2025 data "
                "quality may be limited."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, n_active_days: {n_active_days}")
    print(f"  win_rate: {win_rate}, avg_event_return: {avg_event}")
    for e in events:
        print(f"    {e['event_date']} -> {e['entry_date']} exit {e['exit_date']}: {e['event_return']:+.2%}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "net_sharpe" in m:
        print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
