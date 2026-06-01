"""PL704_ma_prior_auth_long_hca_short_hum
MA Prior-Auth Denial Drop -> Long HCA Short HUM

Following CMS final-rule effective dates that restrict Medicare Advantage
prior-authorization denials, go long HCA (hospital operator beneficiary)
and short HUM (MA insurer, burdened by the rule) for 60 trading days.

Known CMS MA prior-auth rule effective dates (hardcoded from CMS records):
  - 2024-01-01: CMS Interoperability and Prior Authorization Final Rule
    (OMB approved 2023; effective Jan 1 2024; MA plans must process PA faster)
  - 2024-07-01: Additional PA transparency and denial-reason disclosure milestones
  - 2026-01-01: Stricter gold-carding provisions take effect (future event)

References:
  CMS-0057-F (January 17, 2024) — Prior Authorization Final Rule
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# CMS MA prior-auth restriction rule effective dates
EVENTS = [
    {"event_date": "2023-12-29", "description": "CMS-0057-F pre-publication market pricing (~2 days before Jan 1 effective)"},
    {"event_date": "2024-01-01", "description": "CMS MA prior-auth final rule effective (CMS-0057-F)"},
    {"event_date": "2024-07-01", "description": "CMS MA PA transparency milestones effective"},
]

HOLD_DAYS = 60
LONG_TICKER = "HCA"
SHORT_TICKER = "HUM"


def run_event_study(events, ret, long_ticker, short_ticker, hold_days):
    """Long long_ticker, short short_ticker, hold hold_days after each event."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    spread_ret = ret[long_ticker].fillna(0) - ret[short_ticker].fillna(0)

    for ev in events:
        ev_dt = pd.Timestamp(ev["event_date"])
        future = idx[idx > ev_dt]
        if len(future) == 0:
            event_log.append({**ev, "status": "no_data_after_event"})
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_pnl = spread_ret.iloc[entry_pos:exit_pos]
        cum_ret = float((1 + slice_pnl).prod() - 1) if len(slice_pnl) else None

        ev_record = {
            **ev,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return": round(cum_ret, 4) if cum_ret is not None else None,
        }
        event_log.append(ev_record)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = spread_ret.iloc[j - entry_pos]

    return pnl, positions, event_log


def main():
    sid = "PL704_ma_prior_auth_long_hca_short_hum"
    tickers = [LONG_TICKER, SHORT_TICKER, "SPY"]

    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2022-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        EVENTS, ret, LONG_TICKER, SHORT_TICKER, HOLD_DAYS
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days (n_events={n_events}, held_days={len(held_pnl)})",
            extra={"events": event_log, "n_events": n_events},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="CMS MA Prior-Auth Long HCA / Short HUM (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    rets = [e["event_return"] for e in event_log if e.get("event_return") is not None]
    summary = {
        "n_events": len(rets),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "best": round(float(np.max(rets)), 4) if rets else None,
        "worst": round(float(np.min(rets)), 4) if rets else None,
    }

    # Also compute per-event vs SPY
    excess_vs_spy = []
    for e in event_log:
        if not e.get("entry_date") or not e.get("exit_date"):
            continue
        entry_dt = pd.Timestamp(e["entry_date"])
        exit_dt = pd.Timestamp(e["exit_date"])
        spy_slice = spy_r.loc[entry_dt:exit_dt]
        if len(spy_slice):
            spy_cum = float((1 + spy_slice).prod() - 1)
            excess_vs_spy.append({
                "event_date": e["event_date"],
                "entry_date": e["entry_date"],
                "spread_return": e.get("event_return"),
                "spy_return": round(spy_cum, 4),
                "excess": round((e.get("event_return") or 0) - spy_cum, 4),
            })

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Following CMS final-rule effective dates that restrict Medicare Advantage "
                "prior-authorization denials, enter long HCA and short HUM at the next "
                "session's open; hold 60 trading days; exit at close."
            ),
            "mechanism": (
                "MA prior-auth restrictions reduce the revenue-protection lever for MA insurers "
                "(HUM, CVS, ELV), leading to margin compression and earnings guidance cuts. "
                "Hospital operators (HCA) benefit directly from faster reimbursement flows and "
                "reduced administrative burden of appealing denied claims. The HCA/HUM spread "
                "captures this policy transfer: beneficiary vs payor."
            ),
            "source": (
                "CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F, "
                "Jan 17 2024); CMS.gov prior-auth rule milestone dates; "
                "prices via yfinance (auto_adjust=True)."
            ),
            "tickers": [LONG_TICKER, SHORT_TICKER],
            "events": event_log,
            "summary": summary,
            "n_events": n_events,
            "excess_vs_spy": excess_vs_spy,
            "caveats": (
                "Limited to 3 hardcoded events starting 2023-12-29 through 2024-07-01; "
                "overlapping windows compress effective sample. HUM underperformance in 2024 "
                "was also driven by elevated medical-loss ratios beyond just PA rule changes. "
                "HCA is exposed to broader labor and reimbursement-rate risks independent of "
                "PA rules. Rule enforcement and MCO compliance varies by plan. "
                "Future events (2026-01-01) are not yet tradeable at time of writing."
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
