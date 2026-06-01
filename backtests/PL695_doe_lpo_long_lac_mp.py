"""PL695_doe_lpo_long_lac_mp
DOE LPO Critical Minerals Loan Cluster - Long LAC/MP

When DOE LPO/ATVM publishes 2+ critical-minerals conditional commitment
announcements within a 30-day window, enter long an equal-weight basket
of LAC and MP, hedged 50% short LIT, for 60 trading days.

Known events: 2024-02-15, 2024-09-20 (from strategy spec).
LAC was spun off from LAAC in 2023; we restrict to post-spin data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known DOE LPO/ATVM critical-minerals conditional commitment cluster events
# Each date is when 2+ commitments were published within the preceding 30-day window.
# These are hardcoded from the strategy spec (known_events) and public DOE LPO records.
EVENTS = [
    {"event_date": "2024-02-15", "description": "DOE LPO CM cluster Feb 2024"},
    {"event_date": "2024-09-20", "description": "DOE LPO CM cluster Sep 2024"},
]

HOLD_DAYS = 60
BASKET = ["LAC", "MP"]
HEDGE = "LIT"
HEDGE_WEIGHT = -0.5  # 50% short LIT as hedge


def run_event_study(events, ret, basket, hedge, hedge_weight, hold_days):
    """Simulate entry into basket long + hedge short on event day +1 open."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    basket_ret = ret[basket].fillna(0).mean(axis=1)
    hedge_ret = ret[hedge].fillna(0)

    for ev in events:
        ev_dt = pd.Timestamp(ev["event_date"])
        future = idx[idx > ev_dt]
        if len(future) == 0:
            event_log.append({**ev, "status": "no_data_after_event"})
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Combined daily PnL: long basket + hedge_weight * hedge
        slice_basket = basket_ret.iloc[entry_pos:exit_pos]
        slice_hedge = hedge_ret.iloc[entry_pos:exit_pos]
        combined = slice_basket + hedge_weight * slice_hedge

        cum_ret = float((1 + combined).prod() - 1) if len(combined) else None
        ev_record = {
            **ev,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return": round(cum_ret, 4) if cum_ret is not None else None,
        }
        event_log.append(ev_record)

        # Fill PnL (no double-count on overlap)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = combined.iloc[j - entry_pos]

    return pnl, positions, event_log


def main():
    sid = "PL695_doe_lpo_long_lac_mp"
    tickers = BASKET + [HEDGE, "SPY"]

    # LAC spun from LAAC in 2023 — use post-spin start date
    try:
        px = load_prices(tickers, start="2023-06-01")
    except Exception as e:
        # Retry once
        try:
            px = load_prices(tickers, start="2023-06-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        EVENTS, ret, BASKET, HEDGE, HEDGE_WEIGHT, HOLD_DAYS
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))

    # Use held-days-only for metrics so Sharpe is not diluted by flat stretches
    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        # Minimal data — store what we have and mark failed
        return mark_failed(
            sid,
            f"insufficient held days (n_events={n_events}, held_days={len(held_pnl)})",
            extra={
                "events": event_log,
                "n_events": n_events,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="DOE LPO CM Cluster Long LAC/MP hedge LIT (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level summary
    rets = [e["event_return"] for e in event_log if e.get("event_return") is not None]
    summary = {
        "n_events": len(rets),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "best": round(float(np.max(rets)), 4) if rets else None,
        "worst": round(float(np.min(rets)), 4) if rets else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When DOE LPO/ATVM publishes 2+ critical-minerals conditional commitment "
                "announcements within a 30-day window, enter long an equal-weight LAC + MP "
                "basket at the next session's open, hedged 50% short LIT; hold 60 trading "
                "days; exit at the close."
            ),
            "mechanism": (
                "DOE LPO conditional commitments provide near-certain project financing for "
                "critical mineral miners/processors. A cluster of 2+ announcements signals a "
                "positive policy environment that tends to be sustained, triggering re-rating "
                "of direct beneficiaries (LAC lithium, MP rare-earths). The LIT hedge removes "
                "broad EV/battery-metals beta, isolating the policy catalyst."
            ),
            "source": (
                "DOE LPO loan announcements (lpo.energy.gov); "
                "prices via yfinance (auto_adjust=True)."
            ),
            "tickers": BASKET + [HEDGE],
            "events": event_log,
            "summary": summary,
            "n_events": n_events,
            "caveats": (
                "Only 2 known events in the backtest window (2024-02-15, 2024-09-20). "
                "Sample size is extremely limited and statistical significance is low. "
                "LAC had limited trading history pre-spin (Jun 2023). "
                "DOE LPO announcement cadence is policy-driven and may not repeat. "
                "LIT hedge introduces basis risk."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, held_days={len(held_pnl)}")
    print(f"  summary: {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )


if __name__ == "__main__":
    main()
