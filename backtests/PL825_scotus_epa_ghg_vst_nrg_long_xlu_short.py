"""PL825_scotus_epa_ghg_vst_nrg_long_xlu_short
SCOTUS Cert Grant on EPA Section 111 GHG Rule -> Long Merchant IPP (VST/NRG) Short XLU

Event-study on SCOTUS cert grant targeting EPA Section 111 / Clean Air Act GHG
power-plant rules. When SCOTUS grants cert on a pending CAA Section 111 challenge
(or issues an emergency stay of an EPA GHG rule), enter LONG equal-weight VST+NRG
and SHORT XLU, hold 30 trading days.

Known events:
  2016-02-09 - SCOTUS emergency stay of EPA Clean Power Plan (use NRG only; VST not yet public)
  2021-10-29 - SCOTUS cert granted West Virginia v. EPA (use VST+NRG for long, XLU short)
  2022-06-30 - West Virginia v. EPA decided for petitioners (use VST+NRG+CEG, XLU short)

Implementation notes:
  - VST (Vistra Energy) starts ~Oct 2016; excluded from 2016-02-09 event
  - TLN (Talen Energy) starts June 2023; excluded from all events
  - CEG (Constellation Energy) became independent Feb 2022; only included in 2022-06-30 event
  - The pair trade isolates fossil-fleet regulatory beta from broad utility duration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Event definitions: each entry specifies the event date, the long basket and short basket
# Note: "short" tickers will have their returns negated for the long-short pnl
EVENTS = [
    {
        "event_date": "2016-02-09",
        "description": "SCOTUS emergency stay of EPA Clean Power Plan",
        "long_tickers": ["NRG"],           # VST not yet public
        "short_tickers": ["XLU"],
    },
    {
        "event_date": "2021-10-29",
        "description": "SCOTUS cert granted West Virginia v. EPA",
        "long_tickers": ["VST", "NRG"],
        "short_tickers": ["XLU"],
    },
    {
        "event_date": "2022-06-30",
        "description": "West Virginia v. EPA decided for petitioners",
        "long_tickers": ["VST", "NRG", "CEG"],
        "short_tickers": ["XLU"],
    },
]

HOLD_DAYS = 30


def main():
    sid = "PL825_scotus_epa_ghg_vst_nrg_long_xlu_short"
    all_tickers = ["VST", "NRG", "XLU", "CEG", "SPY"]

    try:
        px = load_prices(all_tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    # SPY must be present
    if "SPY" not in px.columns:
        return mark_failed(sid, "SPY missing from price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # Build a combined long-short pnl series across all events
    # All event windows may overlap conceptually but given they are 3+ years apart, they don't
    combined_pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)  # net position indicator (1=in trade, 0=flat)

    event_log = []
    for ev in EVENTS:
        ev_date = pd.Timestamp(ev["event_date"])
        long_tks = ev["long_tickers"]
        short_tks = ev["short_tickers"]

        # Verify required tickers are present
        needed = long_tks + short_tks
        missing = [t for t in needed if t not in ret.columns]
        if missing:
            # Skip this event if tickers not available (shouldn't happen given our data range)
            ev_record = dict(ev)
            ev_record["status"] = f"missing_tickers: {missing}"
            event_log.append(ev_record)
            continue

        # Entry on the next trading session after the event date
        future_sessions = idx[idx > ev_date]
        if len(future_sessions) == 0:
            ev_record = dict(ev)
            ev_record["status"] = "no_future_sessions"
            event_log.append(ev_record)
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + HOLD_DAYS, len(idx))

        # Compute the long-short return series for the hold window
        # Long leg: equal-weight long_tickers
        long_ret = ret[long_tks].fillna(0).mean(axis=1)
        # Short leg: equal-weight short_tickers (negated)
        short_ret = ret[short_tks].fillna(0).mean(axis=1)
        # Long-short spread: long - short
        ls_ret = long_ret - short_ret

        # Compute event cumulative return
        slice_r = ls_ret.iloc[entry_pos:exit_pos]
        ev_cum = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        # SPY same-window return for excess calculation
        spy_slice = spy_r.iloc[entry_pos:exit_pos] if entry_pos < len(spy_r) else pd.Series(dtype=float)
        spy_slice = spy_slice.reindex(ls_ret.iloc[entry_pos:exit_pos].index).fillna(0)
        spy_cum = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["long_short_return"] = round(ev_cum, 4) if ev_cum is not None else None
        ev_record["spy_return"] = round(spy_cum, 4) if spy_cum is not None else None
        ev_record["excess_vs_spy"] = (
            round(ev_cum - spy_cum, 4) if (ev_cum is not None and spy_cum is not None) else None
        )
        event_log.append(ev_record)

        # Fill pnl + positions (no overlap expected, but guard anyway)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                combined_pnl.iloc[j] = ls_ret.iloc[j]

    # Extract held-day pnl for metrics
    held_pnl = combined_pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    n_events_ok = sum(1 for e in event_log if e.get("entry_date"))

    if len(held_pnl) < 20:
        # Too few held days — report what we can
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_events_ok} events",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="SCOTUS EPA GHG Rule: Long IPP Short XLU (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level summaries
    ls_returns = [e["long_short_return"] for e in event_log if e.get("long_short_return") is not None]
    excess_returns = [e["excess_vs_spy"] for e in event_log if e.get("excess_vs_spy") is not None]

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On SCOTUS cert grant or emergency stay of an EPA Section 111 / "
                "Clean Air Act GHG power-plant rule, enter LONG equal-weight "
                "merchant IPP basket (VST/NRG/CEG where available) and SHORT XLU "
                "at the next session's open; hold 30 trading days."
            ),
            "mechanism": (
                "SCOTUS action limiting EPA's power-plant GHG authority removes a "
                "regulatory ceiling on fossil-fuel merchant generators (VST, NRG) "
                "while reducing the 'green premium' embedded in regulated utility "
                "valuations (XLU). The long-short spread captures the repricing of "
                "marginal cost-of-electricity risk from regulatory to market-based "
                "regime. West Virginia v. EPA (2022) established the 'major questions "
                "doctrine' as the key precedent."
            ),
            "source": (
                "SCOTUSblog docket; EPA rulemaking docket (regulations.gov); "
                "prices via yfinance (auto_adjust=True)."
            ),
            "tickers_long": ["VST", "NRG", "CEG"],
            "tickers_short": ["XLU"],
            "n_events": n_events_ok,
            "events": event_log,
            "ls_returns": ls_returns,
            "avg_ls_return": round(float(np.mean(ls_returns)), 4) if ls_returns else None,
            "avg_excess_vs_spy": round(float(np.mean(excess_returns)), 4) if excess_returns else None,
            "caveats": (
                "Only 3 analog events in the backtest window (2016, 2021, 2022). "
                "Statistical significance is extremely limited with n=3. The 2016 "
                "event uses NRG only (VST not yet public); the 2022 event adds CEG "
                "post-Exelon spin-off. Future SCOTUS EPA actions depend on docket "
                "composition and political climate, which are not predictable. "
                "Metrics computed on held-day returns only; cost drag applied at "
                "10bps round-trip."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events_ok}")
    for ev in event_log:
        print(f"  {ev.get('event_date')} ({ev.get('description','')[:40]}): "
              f"LS={ev.get('long_short_return')}, excess={ev.get('excess_vs_spy')}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
