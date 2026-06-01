"""PL827_scotus_cert_ftc_hsr_boutique_ma_long
SCOTUS Cert Grant on FTC HSR/Section 7 Merger-Challenge Authority ->
Long Boutique M&A Advisors (EVR/HLI/PJT) vs Short GS/MS

Event-study: on each court ruling that materially limits FTC merger-challenge
authority (cert grant or major injunction-denial), enter LONG equal-weight
EVR/HLI/PJT and SHORT equal-weight GS/MS for 40 trading days.

Known events:
  2023-07-11 - 9th Circuit upholds denial of FTC v. Microsoft/Activision PI
  2023-01-31 - FTC v. Meta/Within preliminary injunction denied
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Event definitions
EVENTS = [
    {
        "event_date": "2023-07-11",
        "description": "9th Circuit upholds denial of FTC v. Microsoft/Activision PI",
        "long_tickers": ["EVR", "HLI", "PJT"],
        "short_tickers": ["GS", "MS"],
    },
    {
        "event_date": "2023-01-31",
        "description": "FTC v. Meta/Within preliminary injunction denied",
        "long_tickers": ["EVR", "HLI", "PJT"],
        "short_tickers": ["GS", "MS"],
    },
]

HOLD_DAYS = 40


def main():
    sid = "PL827_scotus_cert_ftc_hsr_boutique_ma_long"
    all_tickers = ["EVR", "HLI", "PJT", "LAZ", "GS", "MS", "SPY"]

    try:
        px = load_prices(all_tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    if "SPY" not in px.columns:
        return mark_failed(sid, "SPY missing from price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    combined_pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in EVENTS:
        ev_date = pd.Timestamp(ev["event_date"])
        long_tks = ev["long_tickers"]
        short_tks = ev["short_tickers"]

        needed = long_tks + short_tks
        missing = [t for t in needed if t not in ret.columns]
        if missing:
            ev_record = dict(ev)
            ev_record["status"] = f"missing_tickers: {missing}"
            event_log.append(ev_record)
            continue

        future_sessions = idx[idx > ev_date]
        if len(future_sessions) == 0:
            ev_record = dict(ev)
            ev_record["status"] = "no_future_sessions"
            event_log.append(ev_record)
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + HOLD_DAYS, len(idx))

        long_ret = ret[long_tks].fillna(0).mean(axis=1)
        short_ret = ret[short_tks].fillna(0).mean(axis=1)
        ls_ret = long_ret - short_ret

        slice_r = ls_ret.iloc[entry_pos:exit_pos]
        ev_cum = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        spy_slice = spy_r.reindex(ls_ret.iloc[entry_pos:exit_pos].index).fillna(0)
        spy_cum = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        # Also compute long-only and short-only legs for diagnostics
        long_slice = long_ret.iloc[entry_pos:exit_pos]
        short_slice = short_ret.iloc[entry_pos:exit_pos]
        long_cum = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        short_cum = float((1 + short_slice).prod() - 1) if len(short_slice) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["long_short_return"] = round(ev_cum, 4) if ev_cum is not None else None
        ev_record["long_leg_return"] = round(long_cum, 4) if long_cum is not None else None
        ev_record["short_leg_return"] = round(short_cum, 4) if short_cum is not None else None
        ev_record["spy_return"] = round(spy_cum, 4) if spy_cum is not None else None
        ev_record["excess_vs_spy"] = (
            round(ev_cum - spy_cum, 4) if (ev_cum is not None and spy_cum is not None) else None
        )
        event_log.append(ev_record)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                combined_pnl.iloc[j] = ls_ret.iloc[j]

    held_pnl = combined_pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    n_events_ok = sum(1 for e in event_log if e.get("entry_date"))

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_events_ok} events",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="FTC Merger-Challenge Limit: Long Boutique MA Short Bulge Bracket (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    ls_returns = [e["long_short_return"] for e in event_log if e.get("long_short_return") is not None]
    excess_returns = [e["excess_vs_spy"] for e in event_log if e.get("excess_vs_spy") is not None]

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On a court ruling materially limiting FTC merger-challenge authority "
                "(9th Circuit injunction denial or SCOTUS cert grant), enter LONG "
                "equal-weight EVR/HLI/PJT and SHORT equal-weight GS/MS; hold 40 "
                "trading days."
            ),
            "mechanism": (
                "Boutique M&A advisors (EVR, HLI, PJT) earn 80-100% of revenue from "
                "advisory fees on completed deals; any reduction in FTC blocking power "
                "accelerates deal closings and fee recognition. Large-cap banks (GS, MS) "
                "have diverse revenue streams (~10-15% advisory) so react less sharply. "
                "The pair trade isolates M&A deal-flow beta from broad financial sector moves."
            ),
            "source": (
                "SCOTUSblog cert grant orders; federal court dockets; "
                "prices via yfinance (auto_adjust=True)."
            ),
            "tickers_long": ["EVR", "HLI", "PJT"],
            "tickers_short": ["GS", "MS"],
            "n_events": n_events_ok,
            "events": event_log,
            "ls_returns": ls_returns,
            "avg_ls_return": round(float(np.mean(ls_returns)), 4) if ls_returns else None,
            "avg_excess_vs_spy": round(float(np.mean(excess_returns)), 4) if excess_returns else None,
            "caveats": (
                "Only 2 analog events in the backtest window (both in 2023). "
                "Statistical significance is extremely limited with n=2. The 2023-07-11 "
                "MSFT/ACTI ruling is the primary analog; the 2023-01-31 Meta/Within "
                "ruling is a weaker signal. Boutique M&A revenue is highly lumpy and "
                "depends on deal completion, not just court rulings. GS/MS hedge also "
                "introduces financial-sector factor risk. Metrics computed on held-day "
                "returns only; cost drag applied at 10bps round-trip."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events_ok}")
    for ev in event_log:
        print(f"  {ev.get('event_date')} ({ev.get('description','')[:50]}): "
              f"LS={ev.get('long_short_return')}, long={ev.get('long_leg_return')}, excess={ev.get('excess_vs_spy')}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
