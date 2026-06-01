"""PL956_mas_mps_sneer_surprise_ews_sreit_short
MAS Semi-Annual MPS Hawkish Surprise: Short EWS + Short Singapore REITs

On each MAS Monetary Policy Statement release date (semi-annual: April and October),
enter short EWS (iShares MSCI Singapore ETF) at the open on MPS day.
Hold for 10 trading days, exit at close on day +10.

Simplified rule: use all MPS dates as event study (no OIS data available for
hawkish-surprise filtering). EWS is ~40% SREITs/banks, making it the best
single-instrument proxy for the full mechanism.

SGX-listed SREIT tickers (C38U.SI, A17U.SI, ME8U.SI) have limited history on
yfinance — we use them as a secondary check on available 2018+ data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# MAS MPS release dates (semi-annual: April + October)
# Source: mas.gov.sg monetary policy statements
MAS_MPS_DATES = [
    "2018-04-13",
    "2018-10-12",
    "2019-04-12",
    "2019-10-14",
    "2020-03-30",   # emergency MPS (COVID), March instead of April
    "2020-10-13",
    "2021-04-14",
    "2021-10-14",
    "2022-04-14",
    "2022-10-14",
    "2023-04-14",
    "2023-10-13",
    "2024-04-26",
    "2024-10-14",
]

HOLD_DAYS = 10


def run_event_study(mps_dates, ret, tickers, hold_days=HOLD_DAYS, short=True):
    """Event study: on each MPS date, enter short (or long) on `tickers` basket.
    Entry at open on MPS day (using day's return as proxy), hold `hold_days` trading days.
    Returns (pnl_series, positions_series, event_log).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    # Equal-weight basket return across available tickers
    available = [t for t in tickers if t in ret.columns]
    if not available:
        return pnl, positions, []

    basket_ret = ret[available].fillna(0).mean(axis=1)
    direction = -1.0 if short else 1.0

    event_log = []
    for date_str in mps_dates:
        mps_dt = pd.Timestamp(date_str)

        # Find the MPS day or the first trading day on/after MPS date
        future = idx[idx >= mps_dt]
        if len(future) == 0:
            event_log.append({"date": date_str, "status": "no_data_after_date"})
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Compute event-level cumulative short return
        slice_r = basket_ret.iloc[entry_pos:exit_pos]
        event_ret = float((1 + direction * slice_r).prod() - 1) if len(slice_r) else None

        ev_record = {
            "mps_date": date_str,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return_short": round(event_ret, 4) if event_ret is not None else None,
            "tickers_used": available,
        }
        event_log.append(ev_record)

        # Build daily PnL
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = direction
                pnl.iloc[j] = direction * basket_ret.iloc[j]

    return pnl, positions, event_log


def main():
    sid = "PL956_mas_mps_sneer_surprise_ews_sreit_short"

    # Primary: EWS full history; secondary: SGX SREIT tickers (limited history)
    primary_tickers = ["EWS", "SPY"]
    sreit_tickers = ["C38U.SI", "A17U.SI", "ME8U.SI"]

    # Load primary tickers
    try:
        px_primary = load_prices(primary_tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load primary: {e}")

    missing_primary = [t for t in primary_tickers if t not in px_primary.columns]
    if missing_primary:
        return mark_failed(sid, f"missing primary tickers: {missing_primary}")

    px_primary = px_primary.sort_index().ffill(limit=2)
    ret_primary = daily_returns(px_primary)
    spy_r = ret_primary["SPY"].dropna()

    # Try loading SGX-listed SREITs (optional — may not be available)
    sreit_available = []
    try:
        px_sreit = load_prices(sreit_tickers, start="2021-01-01")
        px_sreit = px_sreit.sort_index().ffill(limit=2)
        sreit_available = [t for t in sreit_tickers if t in px_sreit.columns]
    except Exception:
        px_sreit = None

    # ---------- Primary event study: short EWS on all MPS dates ----------
    pnl_ews, pos_ews, log_ews = run_event_study(
        MAS_MPS_DATES, ret_primary, ["EWS"], hold_days=HOLD_DAYS, short=True
    )

    # Held-day PnL (non-zero positions)
    held_mask = pos_ews != 0.0
    held_pnl = pnl_ews[held_mask]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    held_pnl_aligned = held_pnl.reindex(held_spy.index).dropna()

    n_events = sum(1 for e in log_ews if e.get("entry_date"))
    if len(held_pnl_aligned) < 20:
        return mark_failed(
            sid,
            f"insufficient held days for metrics: held={len(held_pnl_aligned)}, events={n_events}",
            extra={"events": log_ews},
        )

    m = compute_metrics(
        held_pnl_aligned,
        benchmark=held_spy.reindex(held_pnl_aligned.index).fillna(0),
        name="MAS MPS Short EWS Event Study (held-days only)",
        positions=pos_ews.reindex(held_pnl_aligned.index).fillna(0),
        cost_bps=8,
    )

    # ---------- Secondary: short SREIT basket 2021+ (if available) ----------
    sreit_log = []
    sreit_summary = None
    if sreit_available and px_sreit is not None:
        ret_sreit = daily_returns(px_sreit)
        # Only MPS dates within SREIT data window
        sreit_mps = [d for d in MAS_MPS_DATES if pd.Timestamp(d) >= pd.Timestamp("2021-01-01")]
        pnl_sreit, pos_sreit, sreit_log = run_event_study(
            sreit_mps, ret_sreit, sreit_available, hold_days=HOLD_DAYS, short=True
        )
        rets_sreit = [e["event_return_short"] for e in sreit_log if e.get("event_return_short") is not None]
        if rets_sreit:
            sreit_summary = {
                "n_events": len(rets_sreit),
                "avg_event_return": round(float(np.mean(rets_sreit)), 4),
                "win_rate": round(float(np.mean([r > 0 for r in rets_sreit])), 4),
                "best": round(float(np.max(rets_sreit)), 4),
                "worst": round(float(np.min(rets_sreit)), 4),
            }

    # ---------- EWS event-level summary ----------
    ews_rets = [e.get("event_return_short") for e in log_ews if e.get("event_return_short") is not None]
    ews_summary = {
        "n_events": len(ews_rets),
        "avg_event_return": round(float(np.mean(ews_rets)), 4) if ews_rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in ews_rets])), 4) if ews_rets else None,
        "best": round(float(np.max(ews_rets)), 4) if ews_rets else None,
        "worst": round(float(np.min(ews_rets)), 4) if ews_rets else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each MAS Monetary Policy Statement release date (semi-annual, April "
                "and October), enter short EWS (iShares MSCI Singapore ETF) at the open "
                "on MPS day. Hold 10 trading days; exit at close on day +10. "
                "Simplified rule: all MPS events (no OIS hawkish-surprise filter). "
                "EWS is ~40% SREITs/banks and is the primary instrument."
            ),
            "mechanism": (
                "Singapore's MAS manages monetary policy via the S$NEER slope/midpoint. "
                "When MAS surprises hawkishly (holds/steepens when market expects easing), "
                "it signals higher SGD rates, compressing REIT cap-rate spreads and "
                "hurting carry-funded EM positions. EWS, as a proxy for SGX-listed "
                "REITs and financials, is the most liquid instrument capturing this. "
                "The 10-day hold targets the immediate post-MPS adjustment period."
            ),
            "source": (
                "MAS MPS dates: mas.gov.sg monetary policy statements; "
                "prices via yfinance (auto_adjust=True); EWS as SGX proxy."
            ),
            "tickers_primary": ["EWS"],
            "tickers_sreit": sreit_available,
            "events_ews": log_ews,
            "summary_ews": ews_summary,
            "events_sreit": sreit_log,
            "summary_sreit": sreit_summary,
            "hold_days": HOLD_DAYS,
            "n_mps_events": n_events,
            "caveats": (
                "No OIS/SGD forward data available to filter for 'hawkish surprise' events; "
                "simplified event study uses all MPS dates. Sample size is ~14 events "
                "(2018-2024). SGX-listed SREIT tickers have limited yfinance history. "
                "EWS includes banks/industrials beyond REITs, diluting the mechanism. "
                "Metrics computed on held-days only to avoid dilution from flat periods."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, hold_days={HOLD_DAYS}")
    print(f"  EWS event summary: {ews_summary}")
    if sreit_summary:
        print(f"  SREIT event summary (2021+): {sreit_summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )


if __name__ == "__main__":
    main()
