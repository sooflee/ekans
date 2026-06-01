"""PL766_jgb_no_rinban_fxy_long_ewj_short
JGB 10Y Yield Breach + No BoJ Rinban Defense -> Long FXY / Short EWJ

When JGB 10Y yield closes above a threshold AND no BoJ rinban defense is
confirmed within 48 trading hours, enter LONG FXY / SHORT EWJ (1:1 notional)
at T+2 close. Hold 15 trading days.

DATA LIMITATION NOTE: The 1.5% threshold was first crossed in modern era only
in mid-2025, yielding <3 observations. Per implementation_notes, we lower the
trigger threshold to 1.0% (then also test 1.2%) to build an analog event set
from the 2022-2025 JGB yield normalization cycle. The "no-rinban-defense"
condition is approximated by excluding any breach dates that fall within 2 days
of a known BoJ emergency/unscheduled operation date (hard-coded from public
BoJ records for the post-YCC-abolition normalization era).

Benchmark: SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -----------------------------------------------------------------------
# Known BoJ emergency / unscheduled rinban operation dates
# (post-YCC: sourced from BoJ Financial Markets Dept daily ops calendar
#  and Bloomberg/Reuters news for publicly documented emergency ops).
# These are dates where BoJ intervened with an UNscheduled JGB purchase
# or emergency fixed-rate op — if a yield breach falls within +/-2 trading
# days of these dates, it is excluded (BoJ DID defend → thesis fails).
# -----------------------------------------------------------------------
BOJ_EMERGENCY_OPS = [
    "2022-06-13",  # BoJ emergency rinban after JGB yield hit YCC band
    "2022-06-14",
    "2022-06-16",
    "2022-06-17",
    "2022-09-07",  # BoJ emergency unlimited fixed-rate op Sep 2022
    "2022-09-08",
    "2022-10-20",  # BoJ emergency unlimited fixed-rate op Oct 2022
    "2022-10-21",
    "2023-01-13",  # BoJ shock YCC band widening + emergency ops
    "2023-07-28",  # BoJ YCC policy adjustment (band to +/-1.0%)
    "2023-10-31",  # BoJ further YCC flexibility (+/-1.0% soft cap)
    "2024-03-19",  # BoJ abolishes YCC, ends NIRP
    "2025-01-24",  # BoJ rate hike 0.25% -> 0.5%
]

BOJ_EMERGENCY_DATES = {pd.Timestamp(d) for d in BOJ_EMERGENCY_OPS}


def is_defended(breach_date: pd.Timestamp, trading_index: pd.Index, window: int = 2) -> bool:
    """Return True if any BoJ emergency op falls within `window` trading days of the breach."""
    pos = trading_index.get_loc(breach_date) if breach_date in trading_index else None
    if pos is None:
        return False
    lo = max(0, pos - window)
    hi = min(len(trading_index), pos + window + 1)
    window_dates = set(trading_index[lo:hi])
    return bool(window_dates & BOJ_EMERGENCY_DATES)


def find_breach_events(yield_series: pd.Series, threshold: float,
                       trading_index: pd.Index,
                       min_days_between: int = 60) -> list[dict]:
    """
    Find dates where yield first closes ABOVE `threshold` (a new breach),
    where the prior close was at or below threshold, AND BoJ did NOT
    defend within 2 trading days.

    Enforce at least `min_days_between` calendar days between events.
    """
    events = []
    last_event_date = None

    prev_val = None
    for dt, val in yield_series.items():
        if np.isnan(val):
            prev_val = None
            continue
        if prev_val is not None and prev_val <= threshold < val:
            # New breach detected (first close above threshold)
            if last_event_date is not None:
                gap = (dt - last_event_date).days
                if gap < min_days_between:
                    prev_val = val
                    continue
            # Check BoJ defense
            defended = is_defended(dt, trading_index, window=2)
            events.append({
                "breach_date": str(dt.date()),
                "yield_pct": round(val, 3),
                "threshold": threshold,
                "boj_defended": defended,
                "included": not defended,
            })
            if not defended:
                last_event_date = dt
        prev_val = val

    return events


def run_long_short_event_study(events: list[dict], ret: pd.DataFrame,
                               hold_days: int = 15,
                               entry_lag: int = 2) -> tuple:
    """
    For each included event (breach + no defense), enter LONG FXY / SHORT EWJ
    `entry_lag` trading days after the breach date. Hold `hold_days` trading days.

    Returns (pnl_series, positions_series, event_log).
    pnl = FXY return - EWJ return (long FXY, short EWJ, 1:1 notional)
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in events:
        if not ev.get("included"):
            continue
        breach_dt = pd.Timestamp(ev["breach_date"])
        # Find entry: `entry_lag` trading sessions after breach_date
        future = idx[idx > breach_dt]
        if len(future) < entry_lag:
            ev_record = dict(ev)
            ev_record["status"] = "no_data_for_entry"
            event_log.append(ev_record)
            continue
        entry_dt = future[entry_lag - 1]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Long-short pnl = FXY return - EWJ return
        fxy_r = ret["FXY"].iloc[entry_pos:exit_pos]
        ewj_r = ret["EWJ"].iloc[entry_pos:exit_pos]
        ls_r = fxy_r.values - ewj_r.values  # yen appreciation vs short Japan equities

        ev_cum = float((1 + fxy_r).prod() - 1) - float((1 + ewj_r).prod() - 1)

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["event_return"] = round(ev_cum, 4)
        ev_record["fxy_return"] = round(float((1 + fxy_r).prod() - 1), 4)
        ev_record["ewj_return"] = round(float((1 + ewj_r).prod() - 1), 4)
        event_log.append(ev_record)

        for j, (f, e) in enumerate(zip(fxy_r, ewj_r)):
            gpos = entry_pos + j
            if positions.iloc[gpos] == 0.0:  # no double-counting
                pnl.iloc[gpos] = float(f) - float(e)
                positions.iloc[gpos] = 1.0

    return pnl, positions, event_log


def main():
    sid = "PL766_jgb_no_rinban_fxy_long_ewj_short"

    # ---- Load prices ----
    tickers = ["FXY", "EWJ", "SPY"]
    try:
        px = load_prices(tickers, start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    trading_index = ret.index

    # ---- Load JGB 10Y yield from FRED ----
    # IRLTLT01JPM156N = Japan Long-Term Government Bond Yields (monthly, %)
    # We supplement with DGS10 analog. But we primarily need daily JGB data.
    # FRED has JPNBON10Y (daily) for Japan 10-year government bond yield.
    # Try IRLTLT01JPM156N first (monthly), then forward-fill to daily.
    jgb_series = None
    try:
        jgb_monthly = load_fred("IRLTLT01JPM156N", start="2007-01-01")
        jgb_series = jgb_monthly.squeeze()
        # Forward-fill monthly to daily aligned with trading index
        jgb_daily = jgb_series.reindex(trading_index, method="ffill")
        jgb_daily.name = "jgb_10y"
    except Exception as e:
        return mark_failed(sid, f"FRED IRLTLT01JPM156N: {e}")

    # ---- Also try daily FRED series ----
    try:
        jgb_daily2 = load_fred("IRLTLT01JPM156N", start="2007-01-01")
        jgb_daily2 = jgb_daily2.squeeze().reindex(trading_index, method="ffill")
    except Exception:
        jgb_daily2 = None

    # Use the daily forward-filled version
    yield_series = jgb_daily.dropna()
    if len(yield_series) < 252:
        return mark_failed(sid, f"insufficient yield data: {len(yield_series)} obs")

    print(f"JGB 10Y yield range: {yield_series.min():.3f}% – {yield_series.max():.3f}%")
    print(f"Latest value: {yield_series.iloc[-1]:.3f}% on {yield_series.index[-1].date()}")

    # ---- Run event study at 1.0% threshold (primary) ----
    threshold_primary = 1.0
    events_1p0 = find_breach_events(
        yield_series, threshold_primary, trading_index, min_days_between=60
    )
    included_1p0 = [e for e in events_1p0 if e["included"]]
    print(f"\nThreshold {threshold_primary}%: {len(events_1p0)} breaches, "
          f"{len(included_1p0)} not-defended (included)")

    # ---- Run event study at 1.2% threshold (sensitivity) ----
    threshold_sensitivity = 1.2
    events_1p2 = find_breach_events(
        yield_series, threshold_sensitivity, trading_index, min_days_between=60
    )
    included_1p2 = [e for e in events_1p2 if e["included"]]
    print(f"Threshold {threshold_sensitivity}%: {len(events_1p2)} breaches, "
          f"{len(included_1p2)} not-defended (included)")

    # ---- Decide primary dataset ----
    if len(included_1p0) >= 3:
        primary_events = events_1p0
        primary_threshold = threshold_primary
        primary_label = f"1.0%_threshold"
    elif len(included_1p2) >= 3:
        primary_events = events_1p2
        primary_threshold = threshold_sensitivity
        primary_label = f"1.2%_threshold"
    else:
        # Fall back to any events we have
        primary_events = events_1p0 if len(events_1p0) >= len(events_1p2) else events_1p2
        primary_threshold = threshold_primary if len(events_1p0) >= len(events_1p2) else threshold_sensitivity
        primary_label = f"fallback_threshold={primary_threshold}%"

    pnl, positions, event_log = run_long_short_event_study(
        primary_events, ret, hold_days=15, entry_lag=2
    )

    # ---- Run sensitivity for 1.2% ----
    pnl_1p2, positions_1p2, event_log_1p2 = run_long_short_event_study(
        events_1p2, ret, hold_days=15, entry_lag=2
    )

    n_events_primary = len([e for e in event_log if e.get("entry_date")])
    n_events_1p2 = len([e for e in event_log_1p2 if e.get("entry_date")])

    print(f"\nPrimary ({primary_label}): {n_events_primary} traded events")
    print(f"Sensitivity (1.2%): {n_events_1p2} traded events")

    # ---- Compute metrics on held-days only ----
    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        # Too few held days — record event details but mark as failed
        event_returns = [e["event_return"] for e in event_log if e.get("event_return") is not None]
        avg_ret = float(np.mean(event_returns)) if event_returns else None
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events_primary}); "
            f"JGB yield only reached {yield_series.max():.2f}% in data window; "
            f"threshold substitution to {primary_threshold}% generated too few events",
            extra={
                "threshold_primary": primary_threshold,
                "events_primary": event_log,
                "events_sensitivity_1p2": event_log_1p2,
                "n_events_primary": n_events_primary,
                "n_events_1p2": n_events_1p2,
                "avg_event_return": avg_ret,
                "yield_max": round(float(yield_series.max()), 3),
                "yield_min": round(float(yield_series.min()), 3),
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="JGB No-Rinban Defense: Long FXY / Short EWJ",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level stats
    event_returns = [e["event_return"] for e in event_log if e.get("event_return") is not None]
    win_rate = float(np.mean([r > 0 for r in event_returns])) if event_returns else None
    avg_ret = float(np.mean(event_returns)) if event_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                f"When Japan JGB 10Y yield (FRED IRLTLT01JPM156N, monthly forward-filled) "
                f"crosses above {primary_threshold}% AND no BoJ emergency/unscheduled "
                f"rinban operation is detected within 2 trading days of the breach, "
                "enter LONG FXY + SHORT EWJ (1:1 notional) at T+2 close; "
                "hold 15 trading days. Min 60 calendar days between signals. "
                "Threshold substituted from 1.5% (original spec) due to data scarcity."
            ),
            "mechanism": (
                "When JGB yields rise past BoJ's de facto tolerance ceiling without "
                "an emergency defense, the market interprets it as BoJ capitulation "
                "on yield control -> JPY appreciation (FXY long) and Japan export "
                "stock de-rating (EWJ short). The 48h no-rinban window is proxied by "
                "excluding known emergency BoJ operation dates."
            ),
            "source": (
                "FRED IRLTLT01JPM156N (Japan LT Govt Bond Yield, monthly); "
                "BoJ Financial Markets Dept public ops calendar; "
                "prices via yfinance (auto_adjust=True)"
            ),
            "tickers": ["FXY", "EWJ"],
            "threshold_primary": primary_threshold,
            "threshold_original": 1.5,
            "primary_label": primary_label,
            "n_events_primary": n_events_primary,
            "n_events_sensitivity_1p2": n_events_1p2,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_ret, 4) if avg_ret is not None else None,
            "events_primary": event_log,
            "events_sensitivity_1p2": event_log_1p2,
            "caveats": (
                "CRITICAL: JGB 10Y yield series from FRED is monthly (IRLTLT01JPM156N) "
                "and forward-filled to daily; daily resolution is approximate. "
                "The 1.5% threshold (original spec) was only crossed in mid-2025, "
                "so threshold lowered to 1.0% for historical analog set. "
                "BoJ defense detection is via hard-coded emergency-op dates — "
                "not the full daily ops-calendar automated check specified. "
                "FXY data begins Feb 2007. Monthly yield granularity means "
                "breach-timing noise can be ~1 month. Sample is limited to "
                "post-2022 normalization cycle for high thresholds."
            ),
            "boj_emergency_dates_used": BOJ_EMERGENCY_OPS,
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  primary_label: {primary_label}, n_events: {n_events_primary}")
    print(f"  win_rate: {win_rate}, avg_event_return: {avg_ret}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
