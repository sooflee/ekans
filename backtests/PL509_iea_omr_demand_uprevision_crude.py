"""PL509_iea_omr_demand_uprevision_crude
IEA Oil Market Report (OMR) Demand Upward-Revision Event-Study -> Long USO + XOP

Rule (strict):
  Each month around mid-month, the IEA publishes the Oil Market Report (OMR)
  with a headline current-year global oil demand forecast (mb/d).  On each
  historical OMR release date where the headline current-year demand forecast
  was revised UPWARD by >= +0.3 mb/d versus the prior month's OMR (same
  calendar-year baseline), enter long an equal-weight basket of USO (50%) and
  XOP (50%) at the next session's open and hold for 15 trading days (~3 weeks).
  SPY is the benchmark.

Relaxed sensitivity: same rule with a >= +0.2 mb/d threshold.

Per-event PnL is computed by holding the 50/50 basket for 15 trading days from
the next session after the release date.  A daily aggregate position series is
also built (1 during in-trade days, 0 otherwise) to support full Sharpe / CAGR
calculation on held-days-only.

Note: the OMR demand forecast series is hard-coded from press releases /
Reuters coverage; the queue spec acknowledges these are approximate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# ---------- IEA OMR headline current-year global oil demand forecast (mb/d) ----------
# Each row: (release_date, current-year demand forecast at this release in mb/d)
# Forecasts are for the *current calendar year* — when the calendar year flips
# (December -> January) the baseline changes, so the MoM delta in January is
# NOT comparable to December (different reference year).  The flagging routine
# below resets the baseline at the start of each calendar year.
OMR_TABLE = [
    # 2021 (forecast for 2021 demand)
    ("2021-01-19", 96.6), ("2021-02-11", 96.4), ("2021-03-17", 96.5),
    ("2021-04-14", 96.7), ("2021-05-12", 96.4), ("2021-06-11", 96.5),
    ("2021-07-13", 96.2), ("2021-08-12", 96.2), ("2021-09-14", 96.1),
    ("2021-10-14", 96.3), ("2021-11-16", 96.2), ("2021-12-14", 96.2),
    # 2022 (forecast for 2022 demand)
    ("2022-01-19", 99.7), ("2022-02-11", 100.6), ("2022-03-16", 99.7),
    ("2022-04-13", 99.4), ("2022-05-12", 99.4), ("2022-06-15", 99.2),
    ("2022-07-13", 99.2), ("2022-08-11", 99.7), ("2022-09-14", 99.7),
    ("2022-10-13", 99.6), ("2022-11-15", 99.6), ("2022-12-14", 99.9),
    # 2023 (forecast for 2023 demand)
    ("2023-01-18", 101.7), ("2023-02-15", 101.9), ("2023-03-15", 101.9),
    ("2023-04-14", 101.9), ("2023-05-16", 102.0), ("2023-06-14", 102.3),
    ("2023-07-13", 102.1), ("2023-08-11", 102.2), ("2023-09-13", 101.8),
    ("2023-10-12", 101.9), ("2023-11-14", 102.0), ("2023-12-14", 101.7),
    # 2024 (forecast for 2024 demand)
    ("2024-01-18", 102.9), ("2024-02-15", 102.9), ("2024-03-14", 103.2),
    ("2024-04-12", 103.0), ("2024-05-15", 103.2), ("2024-06-12", 103.1),
    ("2024-07-11", 103.0), ("2024-08-13", 102.9), ("2024-09-12", 102.8),
    ("2024-10-15", 102.8), ("2024-11-14", 102.8), ("2024-12-12", 102.8),
    # 2025 (forecast for 2025 demand)
    ("2025-01-15", 103.9), ("2025-02-13", 103.9), ("2025-03-13", 103.9),
    ("2025-04-15", 103.9), ("2025-05-15", 103.7),
]


def flag_events(table, threshold_mbd=0.3):
    """Return list of dicts for each release date where the same-year MoM delta
    is >= threshold_mbd.  Baseline resets at the start of each calendar year
    (the demand forecast tracks the current year, not a rolling forward).
    """
    events = []
    prev_value = None
    prev_year = None
    for date_str, value in table:
        year = int(date_str[:4])
        if prev_year is not None and year == prev_year and prev_value is not None:
            delta = value - prev_value
            if delta >= threshold_mbd:
                events.append({
                    "release_date": date_str,
                    "year": year,
                    "forecast_mbd": value,
                    "prev_forecast_mbd": prev_value,
                    "delta_mbd": round(delta, 2),
                })
        prev_value = value
        prev_year = year
    return events


def run_event_study(events, ret, basket, hold_days=15):
    """Enter equal-weight basket at next session after release date; hold for
    `hold_days` trading days.  Returns (pnl_series, positions_series, event_log).
    On days inside more than one event's hold window, the position stays at 1.0
    (no double counting); pnl uses the basket return on each in-trade day.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    basket_ret = ret[basket].fillna(0).mean(axis=1)  # equal-weight 50/50

    event_log = []
    for ev in events:
        rel = pd.Timestamp(ev["release_date"])
        future = idx[idx > rel]
        if len(future) == 0:
            ev_record = dict(ev); ev_record["status"] = "no_data_after_release"
            event_log.append(ev_record)
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_r = basket_ret.iloc[entry_pos:exit_pos]
        ev_ret = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["event_return"] = round(ev_ret, 4) if ev_ret is not None else None
        event_log.append(ev_record)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = basket_ret.iloc[j]
    return pnl, positions, event_log


def event_summary(log):
    rets = [e["event_return"] for e in log if e.get("event_return") is not None]
    if not rets:
        return None
    return {
        "n_events": len(rets),
        "avg_event_return": round(float(np.mean(rets)), 4),
        "median_event_return": round(float(np.median(rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
        "best": round(float(np.max(rets)), 4),
        "worst": round(float(np.min(rets)), 4),
    }


def event_vs_spy(log, spy_ret):
    rows = []
    for e in log:
        entry = e.get("entry_date"); exit_d = e.get("exit_date")
        if not entry or not exit_d:
            continue
        spy_slice = spy_ret.loc[pd.Timestamp(entry):pd.Timestamp(exit_d)]
        if len(spy_slice):
            spy_cum = float((1 + spy_slice).prod() - 1)
            rows.append({
                "release_date": e["release_date"],
                "entry_date": entry,
                "basket_return": e.get("event_return"),
                "spy_return": round(spy_cum, 4),
                "excess": round((e.get("event_return") or 0) - spy_cum, 4),
            })
    return rows


def main():
    sid = "PL509_iea_omr_demand_uprevision_crude"
    basket = ["USO", "XOP"]
    tickers = basket + ["SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Strict (>= +0.3 mb/d) and relaxed (>= +0.2 mb/d) thresholds
    strict_events = flag_events(OMR_TABLE, threshold_mbd=0.3)
    relaxed_events = flag_events(OMR_TABLE, threshold_mbd=0.2)

    pnl_strict, pos_strict, log_strict = run_event_study(
        strict_events, ret, basket, hold_days=15,
    )
    pnl_relaxed, pos_relaxed, log_relaxed = run_event_study(
        relaxed_events, ret, basket, hold_days=15,
    )

    n_strict = sum(1 for e in log_strict if e.get("entry_date"))
    n_relaxed = sum(1 for e in log_relaxed if e.get("entry_date"))

    # Primary = strict.  If strict has fewer than ~30 held days, fall back to relaxed.
    held_strict = pnl_strict[pos_strict > 0]
    held_relaxed = pnl_relaxed[pos_relaxed > 0]

    if len(held_strict) >= 30:
        primary_pnl, primary_pos, primary_log = pnl_strict, pos_strict, log_strict
        primary_label = "strict_>=+0.3mbd"
    elif len(held_relaxed) >= 30:
        primary_pnl, primary_pos, primary_log = pnl_relaxed, pos_relaxed, log_relaxed
        primary_label = "relaxed_>=+0.2mbd"
    else:
        return mark_failed(
            sid,
            f"insufficient held days (strict={len(held_strict)}, relaxed={len(held_relaxed)})",
            extra={
                "events_strict": log_strict,
                "events_relaxed": log_relaxed,
                "summary_strict": event_summary(log_strict),
                "summary_relaxed": event_summary(log_relaxed),
            },
        )

    held_pnl = primary_pnl[primary_pos > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="IEA OMR Demand Up-Revision Long USO+XOP (held-days only)",
        positions=primary_pos.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On an IEA OMR release date where the current-year global oil "
                "demand forecast was revised UPWARD by >= +0.3 mb/d vs the "
                "prior month's OMR (same calendar-year baseline), enter long "
                "equal-weight USO (50%) + XOP (50%) at the next session's "
                "open; hold 15 trading days; exit at the close."
            ),
            "mechanism": (
                "An IEA upward revision to current-year global oil demand is "
                "widely tracked by physical and financial crude participants and "
                "is read as a bullish signal on the supply/demand balance.  WTI "
                "and Brent typically rally on/after the release as traders "
                "re-price the call on OPEC+ supply; producer equities (XOP) "
                "respond with operational leverage as cash-flow forecasts are "
                "marked up.  The 15-day window captures the post-release drift "
                "and any follow-through into OPEC+ commentary."
            ),
            "source": (
                "IEA Oil Market Report monthly press releases (iea.org/reports/"
                "oil-market-report) and Reuters/Bloomberg coverage on release "
                "day; prices via yfinance (auto_adjust=True)."
            ),
            "tickers": basket,
            "primary_rule_label": primary_label,
            "events_strict": log_strict,
            "events_relaxed": log_relaxed,
            "summary_strict": event_summary(log_strict),
            "summary_relaxed": event_summary(log_relaxed),
            "excess_vs_spy_strict": event_vs_spy(log_strict, spy_r),
            "excess_vs_spy_relaxed": event_vs_spy(log_relaxed, spy_r),
            "n_events_strict": n_strict,
            "n_events_relaxed": n_relaxed,
            "caveats": (
                "The OMR headline demand forecasts are hard-coded from press "
                "releases (approximate to 0.1 mb/d); IEA historical revisions "
                "could re-class which months trigger.  Event-study sample is "
                "modest (single-digit strict events).  USO suffers from contango "
                "roll yield, which depresses long-WTI exposure mechanically "
                "during pre-2022 contango regimes — this is a known structural "
                "drag that compounds over the 15-day hold.  Calendar-year baseline "
                "resets in January so the Jan-vs-Dec print is intentionally "
                "skipped.  Period: 2021-01 through 2026-05."
            ),
        },
        pnl=primary_pnl[primary_pos > 0],
    )

    print(f"Done: {sid}")
    print(f"  primary rule: {primary_label}")
    print(f"  events strict (>=+0.3): {n_strict}; events relaxed (>=+0.2): {n_relaxed}")
    print(f"  summary_strict: {event_summary(log_strict)}")
    print(f"  summary_relaxed: {event_summary(log_relaxed)}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
