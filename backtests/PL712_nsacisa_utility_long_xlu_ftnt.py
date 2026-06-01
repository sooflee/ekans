"""PL712_nsacisa_utility_long_xlu_ftnt
NSA/CISA Utility Advisory -> Long XLU + FTNT

When NSA and CISA jointly publish an advisory naming nation-state targeting
of US utilities, go long equal-weight XLU + FTNT for 45 trading days.

Logic: advisories trigger accelerated utility-sector OT/ICS security spending,
benefiting both the sector (XLU, as capital-raise announcements support NAV)
and the leading utility-OT cybersecurity vendor (FTNT).

Known NSA/CISA joint advisories citing utility sector (from cisa.gov):
  - 2021-07-20: AA21-201A — Chinese state-sponsored cyber operations including
    energy sector
  - 2022-03-24: AA22-083A — Destructive malware targeting Ukraine; US utilities
    alert (ICS-CERT scope)
  - 2022-04-13: AA22-103A — APT targeting ICS / SCADA in energy sector (CISA
    emergency directive adjacent)
  - 2023-06-01: AA23-144A — Volt Typhoon targeting US critical infrastructure
    including utilities
  - 2024-02-07: CISA/NSA/FBI joint advisory on Volt Typhoon living-off-the-land
    in US utility OT environments (the 'known_event' in the queue spec)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hardcoded NSA/CISA joint utility-sector advisory dates from cisa.gov
EVENTS = [
    {"event_date": "2021-07-20", "advisory": "AA21-201A — Chinese APT energy-sector OT ops"},
    {"event_date": "2022-03-24", "advisory": "AA22-083A — destructive malware US ICS energy alert"},
    {"event_date": "2022-04-13", "advisory": "AA22-103A — APT targeting energy/ICS SCADA"},
    {"event_date": "2023-06-01", "advisory": "AA23-144A — Volt Typhoon US critical infra"},
    {"event_date": "2024-02-07", "advisory": "CISA/NSA/FBI — Volt Typhoon utility OT advisory"},
]

BASKET = ["XLU", "FTNT"]
HOLD_DAYS = 45


def run_event_study(events, ret, basket, spy_r, hold_days):
    """Long equal-weight basket for hold_days after each event."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    basket_ret = ret[basket].fillna(0).mean(axis=1)

    for ev in events:
        ev_dt = pd.Timestamp(ev["event_date"])
        future = idx[idx > ev_dt]
        if len(future) == 0:
            event_log.append({**ev, "status": "no_data_after_event"})
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_pnl = basket_ret.iloc[entry_pos:exit_pos]
        cum_ret = float((1 + slice_pnl).prod() - 1) if len(slice_pnl) else None

        spy_slice = spy_r.iloc[entry_pos:min(exit_pos, len(spy_r))]
        spy_cum = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        ev_record = {
            **ev,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return": round(cum_ret, 4) if cum_ret is not None else None,
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
            "excess": round(cum_ret - spy_cum, 4) if (cum_ret is not None and spy_cum is not None) else None,
        }
        event_log.append(ev_record)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = basket_ret.iloc[j - entry_pos]

    return pnl, positions, event_log


def main():
    sid = "PL712_nsacisa_utility_long_xlu_ftnt"
    tickers = BASKET + ["SPY"]

    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2019-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(EVENTS, ret, BASKET, spy_r, HOLD_DAYS)

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
        name="NSA/CISA Utility Advisory Long XLU+FTNT (held-days)",
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
                "When NSA and CISA jointly publish an advisory naming nation-state targeting "
                "of US utilities, enter long equal-weight XLU + FTNT at the next session's "
                "open; hold 45 trading days; exit at the close."
            ),
            "mechanism": (
                "Joint NSA/CISA advisories on utility-sector nation-state threats trigger "
                "accelerated OT/ICS security spending mandates. Utilities must allocate "
                "incremental capex (supporting sector NAV/rate-base), and FTNT is the "
                "leading OT/ICS platform vendor with FortiGate deployments in energy sector. "
                "The XLU long captures the sector's hardening tailwind; FTNT captures the "
                "vendor benefit."
            ),
            "source": (
                "CISA.gov advisory archive (cisa.gov/news-events/cybersecurity-advisories); "
                "NSA/CISA/FBI joint advisories 2021-2024; "
                "prices via yfinance (auto_adjust=True)."
            ),
            "tickers": BASKET,
            "events": event_log,
            "summary": summary,
            "n_events": n_events,
            "caveats": (
                "Only 5 hardcoded events; advisory dates are approximately correct but may "
                "not align exactly with market open timing. XLU is driven primarily by "
                "interest-rate dynamics, which may dominate the advisory signal. FTNT "
                "performance is driven by overall enterprise security spending, not purely "
                "utility-OT advisories. No live CISA feed is used; event selection may "
                "have survivorship bias toward 'notable' advisories."
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
