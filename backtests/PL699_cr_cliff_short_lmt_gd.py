"""PL699_cr_cliff_short_lmt_gd
CR Funding Cliff <14d No Conference - Short LMT/GD Long RTX

When a Continuing Resolution expiration is within 14 calendar days with no
House-Senate conference report filed and a topline budget gap >$30B, short
LMT/GD/NOC/HII equal-weight and long RTX 1:1 notional for 30 trading days,
with a 25% SPY hedge to reduce market beta.

Known CR funding-cliff events: 2018-12-22, 2023-09-30, 2024-09-30.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hardcoded CR funding-cliff events (days when government shutdown began /
# CR expired without conference report with large topline gap)
EVENTS = [
    {"event_date": "2018-12-22", "description": "Dec 2018 partial government shutdown begins"},
    {"event_date": "2023-09-30", "description": "Sep 2023 CR expiration cliff (shutdown avoided last minute)"},
    {"event_date": "2024-09-30", "description": "Sep 2024 CR expiration cliff"},
]

SHORT_BASKET = ["LMT", "GD", "NOC", "HII"]
LONG_TICKER = "RTX"
SPY_HEDGE_WEIGHT = -0.25  # 25% short SPY as hedge
HOLD_DAYS = 30


def run_event_study(events, ret, short_basket, long_ticker, spy_hedge_wt, hold_days):
    """Short short_basket, long long_ticker, small SPY hedge, for hold_days post event."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    # Short basket = -1 * mean of short basket returns
    short_ret = ret[short_basket].fillna(0).mean(axis=1) * (-1)
    long_ret = ret[long_ticker].fillna(0)
    spy_ret = ret["SPY"].fillna(0) * spy_hedge_wt

    # Combined daily PnL: short basket + long RTX + SPY hedge
    combined_base = short_ret + long_ret + spy_ret

    for ev in events:
        ev_dt = pd.Timestamp(ev["event_date"])
        future = idx[idx > ev_dt]
        if len(future) == 0:
            event_log.append({**ev, "status": "no_data_after_event"})
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_pnl = combined_base.iloc[entry_pos:exit_pos]
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
                pnl.iloc[j] = combined_base.iloc[j - entry_pos]

    return pnl, positions, event_log


def main():
    sid = "PL699_cr_cliff_short_lmt_gd"
    tickers = SHORT_BASKET + [LONG_TICKER, "SPY"]

    try:
        px = load_prices(tickers, start="2016-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2016-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        EVENTS, ret, SHORT_BASKET, LONG_TICKER, SPY_HEDGE_WEIGHT, HOLD_DAYS
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
        name="CR Cliff Short Defense Primes Long RTX (held-days)",
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

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When CR expiration is <14 calendar days away with no House-Senate conference "
                "report filed and topline budget gap >$30B: short LMT/GD/NOC/HII equal-weight, "
                "long RTX 1:1 notional, hedge 25% short SPY; hold 30 trading days; exit."
            ),
            "mechanism": (
                "CR funding cliffs create near-term program-delay uncertainty for long-lead "
                "defense programs (F-35 at LMT, submarine/destroyer at HII/GD, ground-system "
                "integrators at NOC). RTX has relatively more aftermarket/services revenue "
                "insulated from procurement delays. The spread trade isolates cliff-specific "
                "program risk while the SPY hedge dampens broad equity-market beta."
            ),
            "source": (
                "CR event dates from public Congressional records (congress.gov); "
                "prices via yfinance (auto_adjust=True)."
            ),
            "tickers": SHORT_BASKET + [LONG_TICKER],
            "events": event_log,
            "summary": summary,
            "n_events": n_events,
            "caveats": (
                "Only 3 hardcoded events (2018-12-22, 2023-09-30, 2024-09-30); sample is tiny. "
                "Defense primes have complex exposure profiles; individual events driven by "
                "factors beyond CR uncertainty. RTX long may co-move with short basket. "
                "The CR-cliff vs conference-report detection is manual/hardcoded here; "
                "live implementation requires real-time Congress.gov monitoring. "
                "The 2018 shutdown was partisan; defense appropriations dynamics differ "
                "by political regime."
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
