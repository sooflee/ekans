"""PL611_carb_zev_credit_collapse_short_tsla
CARB ZEV Credit Price YoY Collapse Proxy -> Short TSLA into Earnings.

Uses hardcoded table of TSLA quarterly regulatory-credit revenue from 10-Q filings
(SEC EDGAR public). Compute trailing-4Q (TTM) sum and trigger when latest TTM
is >=25% below TTM 4 quarters prior.

Entry: short TSLA at the next TSLA earnings-day open following the qualifying
quarterly print. Earnings dates approximated as Apr 22, Jul 22, Oct 22, Jan 28.

Exit: earliest of (i) 50 trading days, (ii) next earnings date, (iii) TSLA closes
>+12% above entry (stop).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


# Hardcoded TSLA quarterly regulatory-credit revenue ($M) per 10-Q filings
TSLA_RC = {
    "2020Q1": 354, "2020Q2": 428, "2020Q3": 397, "2020Q4": 401,
    "2021Q1": 518, "2021Q2": 354, "2021Q3": 279, "2021Q4": 314,
    "2022Q1": 679, "2022Q2": 344, "2022Q3": 286, "2022Q4": 467,
    "2023Q1": 521, "2023Q2": 282, "2023Q3": 554, "2023Q4": 433,
    "2024Q1": 442, "2024Q2": 890, "2024Q3": 739, "2024Q4": 692,
    "2025Q1": 595,
}

# Earnings-day approx anchors (M, D) per quarter
EARNINGS_ANCHORS = {
    "Q1": (4, 22),
    "Q2": (7, 22),
    "Q3": (10, 22),
    "Q4": (1, 28),  # rolls to next year
}


def quarter_to_earnings_date(qkey):
    """Return the earnings-release date that REPORTS the given quarter."""
    year = int(qkey[:4])
    qnum = qkey[-1]
    m, d = EARNINGS_ANCHORS["Q" + qnum]
    if qnum == "4":
        year += 1
    return pd.Timestamp(year=year, month=m, day=d)


def main():
    sid = "PL611_carb_zev_credit_collapse_short_tsla"
    try:
        px = load_prices(["TSLA", "SPY"], start="2019-06-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty or "TSLA" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "missing TSLA/SPY")

    tsla = px["TSLA"].dropna()
    spy = px["SPY"].dropna()

    df = pd.concat({"TSLA": tsla, "SPY": spy}, axis=1).dropna()
    df["TSLA_r"] = df["TSLA"].pct_change()
    df["SPY_r"] = df["SPY"].pct_change()

    # Build TTM series by quarter
    qkeys = sorted(TSLA_RC.keys())
    ttm = {}
    for i in range(len(qkeys)):
        if i < 3:
            continue
        window = qkeys[i - 3:i + 1]
        ttm[qkeys[i]] = sum(TSLA_RC[q] for q in window)

    # YoY of TTM: trigger when ttm[q] / ttm[q-4] - 1 <= -0.25
    trigger_quarters = []
    sorted_qkeys = sorted(ttm.keys())
    ttm_qkey_idx = {q: i for i, q in enumerate(sorted_qkeys)}
    for q in sorted_qkeys:
        idx = ttm_qkey_idx[q]
        if idx < 4:
            continue
        prev = sorted_qkeys[idx - 4]
        if ttm[prev] <= 0:
            continue
        yoy = ttm[q] / ttm[prev] - 1
        if yoy <= -0.25:
            trigger_quarters.append((q, yoy))

    if not trigger_quarters:
        return mark_failed(sid, "no signal firings in hardcoded credit table")

    # For each trigger quarter, entry = earnings date *reporting* that quarter
    # exit = earliest of (50 trading days, next quarter's earnings date, -12% stop... wait, stop is +12% UP since we are SHORT)
    # Re-reading: "TSLA closes >+12% above entry (stop)" — yes, short stop is +12% upside.

    dates = list(df.index)
    pos_lookup = {d: i for i, d in enumerate(dates)}

    def find_trading_idx_on_or_after(target):
        for i, d in enumerate(dates):
            if d >= target:
                return i
        return None

    legs = []
    events = []
    open_until = None

    for q, yoy in trigger_quarters:
        earn = quarter_to_earnings_date(q)
        # Entry = open of the next trading day on/after earnings date
        entry_i = find_trading_idx_on_or_after(earn)
        if entry_i is None or entry_i + 1 >= len(dates):
            continue
        # actually the entry day itself (close), if earnings day is trading day
        # use entry_i (the earnings-day candle / next trading day)
        entry_date = dates[entry_i]
        if open_until is not None and entry_date <= open_until:
            continue
        entry_px = df["TSLA"].iloc[entry_i]
        # Next earnings ~ next quarter's anchor
        # roll qkey
        year = int(q[:4]); qnum = int(q[-1])
        next_qnum = qnum + 1
        next_year = year
        if next_qnum > 4:
            next_qnum = 1
            next_year = year + 1
        next_q = f"{next_year}Q{next_qnum}"
        next_earn = quarter_to_earnings_date(next_q)
        next_earn_i = find_trading_idx_on_or_after(next_earn)

        max_hold = 50
        end_i = min(entry_i + max_hold, len(dates))
        exit_i = None
        exit_reason = "time"
        for j in range(entry_i, end_i):
            cur_px = df["TSLA"].iloc[j]
            if cur_px / entry_px - 1 > 0.12:
                exit_i = j
                exit_reason = "stop_up12"
                break
            if next_earn_i is not None and j >= next_earn_i and j > entry_i:
                exit_i = j
                exit_reason = "next_earnings"
                break
        if exit_i is None:
            exit_i = min(entry_i + max_hold - 1, len(dates) - 1)
        if exit_i < entry_i:
            continue
        leg = -1.0 * df["TSLA_r"].iloc[entry_i:exit_i + 1]
        if leg.empty:
            continue
        legs.append(leg)
        events.append({
            "trigger_quarter": q,
            "ttm_yoy": round(float(yoy), 4),
            "entry_date": str(entry_date.date()),
            "exit_date": str(dates[exit_i].date()),
            "exit_reason": exit_reason,
            "days_held": exit_i - entry_i + 1,
            "ret": round(float((1 + leg).prod() - 1), 4),
        })
        open_until = dates[exit_i]

    if not legs:
        return mark_failed(sid, "no valid events in price window")

    pnl = pd.Series(0.0, index=df.index)
    for leg in legs:
        pnl.loc[leg.index] = pnl.loc[leg.index] + leg.values
    pnl = pnl.loc[legs[0].index[0]:]

    if len(pnl) < 60:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    bench = df["SPY_r"].reindex(pnl.index).fillna(0)
    m = compute_metrics(pnl, benchmark=bench,
                        name="TSLA Reg-Credit TTM YoY Collapse -> Short TSLA")

    save_result(sid, m, extra={
        "rule": ("Short TSLA at next earnings open when TSLA TTM regulatory-credit revenue "
                 "(from hardcoded 10-Q table) is >=25% below TTM 4Q prior. Hold up to 50d "
                 "or to next earnings, +12% stop."),
        "mechanism": ("ZEV / CARB credit revenue collapse signals competitor EV supply catching "
                      "up — high-margin credit income evaporates and TSLA earnings beat is at risk."),
        "source": ("Hardcoded TSLA 10-Q regulatory-credit revenue per quarter (SEC EDGAR). "
                   "yfinance TSLA, SPY."),
        "n_events": len(events),
        "events_sample": events[:15],
        "horizon": "8-12 weeks",
        "counter_signal": True,
    })
    print(f"Done {sid}: events={len(events)} pnl_days={len(pnl)} "
          f"Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
