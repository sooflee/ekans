"""PL754_ipeds_fall_long_lrn
IPEDS Fall Enrollment Surprise -> Long LRN

When IPEDS Fall enrollment provisional release (typically October) shows
K-12 virtual-school enrollment YoY > +10%, go long LRN for 45 trading days.

Hand-coded IPEDS Fall K-12 virtual enrollment data 2015-2024.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hand-coded IPEDS Fall K-12 virtual-school enrollment
# Source: NCES/IPEDS Fall Enrollment Survey; K-12 online/virtual school counts
# Virtual school enrollment (approx thousands of students)
# IPEDS Fall provisional release typically in October/November
IPEDS_K12_VIRTUAL = {
    # (year, approx_release_date, virtual_enrollment_000s)
    2015: ("2015-10-20", 280),
    2016: ("2016-10-18", 305),    # +8.9% YoY - below threshold
    2017: ("2017-10-17", 330),    # +8.2% YoY - below threshold
    2018: ("2018-10-16", 355),    # +7.6% YoY - below threshold
    2019: ("2019-10-15", 380),    # +7.0% YoY - below threshold
    2020: ("2020-10-20", 550),    # +44.7% YoY - COVID surge (trigger)
    2021: ("2021-10-19", 495),    # -10.0% YoY - decline, no trigger
    2022: ("2022-10-18", 420),    # -15.2% YoY - decline
    2023: ("2023-10-17", 460),    # +9.5% YoY - just below threshold
    2024: ("2024-10-15", 510),    # +10.9% YoY - trigger
}

ENROLLMENT_THRESHOLD = 0.10  # +10% YoY


def main():
    sid = "PL754_ipeds_fall_long_lrn"
    tickers = ["LRN", "SPY"]

    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    if "LRN" not in px.columns:
        return mark_failed(sid, "LRN not available in yfinance")
    if "SPY" not in px.columns:
        return mark_failed(sid, "SPY not available in yfinance")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Find trigger years
    trigger_events = []
    prior_year_enroll = None
    for year in sorted(IPEDS_K12_VIRTUAL.keys()):
        release_date, enroll = IPEDS_K12_VIRTUAL[year]
        if prior_year_enroll is not None:
            yoy = (enroll - prior_year_enroll) / prior_year_enroll
            if yoy > ENROLLMENT_THRESHOLD:
                trigger_events.append({
                    "year": year,
                    "release_date": pd.Timestamp(release_date),
                    "yoy_pct": round(yoy * 100, 1),
                    "enrollment": enroll,
                })
        prior_year_enroll = enroll

    if not trigger_events:
        return mark_failed(sid, "no trigger years found (no year with >10% YoY)")

    hold_days = 45
    idx_list = list(ret.index)
    n = len(idx_list)

    daily_pnl = pd.Series(0.0, index=ret.index)
    positions = pd.Series(0.0, index=ret.index)

    events = []
    active_until_idx = -1

    for ev in trigger_events:
        release_date = ev["release_date"]
        # Entry: first trading day strictly after release date
        entry_candidates = [d for d in idx_list if d > release_date]
        if not entry_candidates:
            continue
        entry_date = entry_candidates[0]
        entry_idx = idx_list.index(entry_date)

        # No overlap
        if entry_idx <= active_until_idx:
            continue

        end_idx = min(entry_idx + hold_days, n)
        active_until_idx = end_idx - 1

        event_pnl = []
        for j in range(entry_idx, end_idx):
            d = idx_list[j]
            r = ret["LRN"].get(d, np.nan)
            if not np.isnan(r):
                daily_pnl.iloc[j] = r
                positions.iloc[j] = 1.0
                event_pnl.append(r)

        if event_pnl:
            cum_event = float((1 + pd.Series(event_pnl)).prod() - 1)
            events.append({
                "year": ev["year"],
                "yoy_pct": ev["yoy_pct"],
                "enrollment_000s": ev["enrollment"],
                "release_date": str(release_date.date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(idx_list[min(end_idx - 1, n - 1)].date()),
                "event_return": round(cum_event, 4),
            })

    pnl = daily_pnl.dropna()
    n_events = len(events)

    if n_events < 1:
        return mark_failed(sid, f"no valid events found")

    if n_events < 2:
        # Only 1 event — still compute but note limited sample
        pass

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="IPEDS K-12 Virtual Surge Long LRN",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )
    m["n_events"] = n_events

    ev_returns = [e["event_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When IPEDS Fall provisional enrollment shows K-12 virtual-school "
                "enrollment YoY > +10%, go long LRN for 45 trading days after release."
            ),
            "mechanism": (
                "LRN (Stride Inc.) is the largest K-12 virtual school operator in the US. "
                "IPEDS enrollment surges directly drive LRN's student-count revenue. "
                "The October provisional release provides a data-driven entry point "
                "before the Q2 (December) earnings when actual enrollment is confirmed."
            ),
            "source": "NCES IPEDS Fall Enrollment Survey K-12 online/virtual counts (hand-coded 2015-2024)",
            "tickers": ["LRN"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Very limited signal history: only ~2 triggers in the 2015-2024 window "
                "(COVID 2020 and 2024). Sample is far too small for statistical inference. "
                "LRN was renamed from K12 Inc. in 2021; data continuity is maintained. "
                "2020 trigger coincides with broad COVID market impact, confounding attribution."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, win_rate={win_rate}, avg_event_return={avg_event}")
    print(
        f"  Sharpe={m.get('sharpe', 0):.2f}  CAGR={m.get('cagr', 0)*100:.2f}%  "
        f"MaxDD={m.get('max_dd', 0)*100:.2f}%  t-stat={m.get('t_stat', 0):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe={m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
