"""PL682_gasc_cancel_short_weat
GASC Wheat Tender Cancellation Cluster - Short WEAT

When the Egyptian General Authority for Supply Commodities (GASC) cancels
3+ consecutive wheat tenders or buys minimal volume within a 60-day window,
short WEAT for 25 trading days with a 50% CORN hedge (long).

Two-pass approach:
1. Hard-coded known GASC cluster dates from public Reuters/FAS USDA records.
2. Price-proxy extension: WEAT rolling 20-day return < -10% triggers a
   cluster signal (captures episodes not in the hard-coded list).

Primary series: held-day returns during short-WEAT positions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# ---- Hard-coded GASC cancellation cluster dates (Reuters/FAS USDA records) ----
# These are the dates of the 3rd cancellation within the 60-day window
GASC_CLUSTER_EVENTS = [
    {"event_date": "2022-07-15", "source": "GASC tender cancellations Jul 2022 (Reuters)"},
    {"event_date": "2023-10-12", "source": "GASC minimal-buy cluster Oct 2023 (FAS USDA)"},
]


def run_event_study(events, ret_weat, ret_corn, spy_ret, hold_days=25, hedge_ratio=0.5):
    """Simulate short WEAT + long CORN hedge for each event.

    Short WEAT = -1 position; hedge = +hedge_ratio in CORN.
    Net pnl per day = -ret_weat + hedge_ratio * ret_corn.
    Entry at next session open after event_date, hold for hold_days trading days.
    """
    idx = ret_weat.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    for ev in events:
        rel = pd.Timestamp(ev["event_date"])
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            continue
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_weat = ret_weat.iloc[entry_pos:exit_pos]
        slice_corn = ret_corn.reindex(slice_weat.index).fillna(0)
        net_ret = -slice_weat + hedge_ratio * slice_corn
        ev_cum = float((1 + net_ret).prod() - 1) if len(net_ret) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["event_return"] = round(ev_cum, 4) if ev_cum is not None else None

        # Fill in pnl for the hold window (no double-counting)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = -ret_weat.iloc[j] + hedge_ratio * ret_corn.reindex([ret_weat.index[j]]).fillna(0).iloc[0]

        event_log.append(ev_record)

    return pnl, positions, event_log


def generate_proxy_events(ret_weat, px_weat, roll_window=20, threshold=-0.10, min_gap=30):
    """Generate price-proxy events when WEAT 20-day rolling return drops below -10%.

    Enforces minimum gap of min_gap days between signals to avoid clustering from
    the same drawdown period.
    """
    roll_ret = (px_weat / px_weat.shift(roll_window) - 1).dropna()
    triggered = roll_ret[roll_ret < threshold].index

    events = []
    last_event = pd.Timestamp("2000-01-01")
    for dt in triggered:
        if (dt - last_event).days >= min_gap:
            events.append({"event_date": str(dt.date()), "source": "price_proxy_20d_roll_lt_-10pct"})
            last_event = dt
    return events


def main():
    sid = "PL682_gasc_cancel_short_weat"
    tickers = ["WEAT", "CORN", "SPY"]

    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    ret_weat = ret["WEAT"].dropna()
    ret_corn = ret["CORN"].dropna()
    spy_r = ret["SPY"].dropna()

    # --- Pass 1: Hard-coded GASC cluster events (filtered to available data) ---
    avail_start = ret_weat.index[0]
    hard_events = [e for e in GASC_CLUSTER_EVENTS if pd.Timestamp(e["event_date"]) >= avail_start]

    # --- Pass 2: Price-proxy events (exclude the hard-coded cluster dates +-30 days) ---
    proxy_events_raw = generate_proxy_events(ret_weat, px["WEAT"])
    hard_dates = [pd.Timestamp(e["event_date"]) for e in hard_events]
    proxy_events = []
    for ev in proxy_events_raw:
        dt = pd.Timestamp(ev["event_date"])
        # Skip if within 30 days of a hard-coded event
        too_close = any(abs((dt - hd).days) <= 30 for hd in hard_dates)
        if not too_close:
            proxy_events.append(ev)

    # Combine and sort by event date
    all_events = hard_events + proxy_events
    all_events.sort(key=lambda x: x["event_date"])

    if len(all_events) == 0:
        return mark_failed(sid, "no events found in data window")

    # --- Run event study ---
    pnl, positions, event_log = run_event_study(
        all_events, ret_weat, ret_corn, spy_r, hold_days=25, hedge_ratio=0.5
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    n_hard = sum(1 for e in event_log if "source" in e and "price_proxy" not in e.get("source", ""))
    n_proxy = n_events - n_hard

    # Use held-day returns only for metrics
    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={"events": event_log, "n_events": n_events},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="GASC Cancellation Cluster Short WEAT (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )
    m["n_events"] = n_events

    # Event-level summary
    rets = [e["event_return"] for e in event_log if e.get("event_return") is not None]
    event_summary = {
        "n_events": len(rets),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "median_event_return": round(float(np.median(rets)), 4) if rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "best": round(float(max(rets)), 4) if rets else None,
        "worst": round(float(min(rets)), 4) if rets else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When 3+ consecutive GASC wheat tenders are cancelled or buy minimal "
                "volume within a 60-day window, short WEAT for 25 trading days with "
                "a 50% long CORN hedge. Entry at the next session's open after the "
                "3rd cancellation date."
            ),
            "mechanism": (
                "GASC (Egyptian General Authority for Supply Commodities) is the "
                "world's largest single buyer of wheat. Cancellation clusters signal "
                "either demand destruction (high prices/logistics issues) or a supply "
                "pipeline surprise. When Egypt repeatedly cancels, it removes a "
                "significant demand support for Black Sea / French wheat prices, "
                "historically dragging WEAT lower over the following 3-5 weeks. "
                "The CORN hedge mitigates broad-grain beta while isolating the "
                "wheat-specific demand shock."
            ),
            "source": (
                "GASC tender results via Reuters commodity desk and USDA FAS "
                "Global Commodity News; prices via yfinance (WEAT, CORN, SPY)."
            ),
            "tickers": ["WEAT", "CORN"],
            "n_hard_events": n_hard,
            "n_proxy_events": n_proxy,
            "event_log": event_log,
            "event_summary": event_summary,
            "caveats": (
                "Hard-coded GASC event dates are limited (2 known clusters); the "
                "price-proxy extension (WEAT 20-day roll < -10%) approximates "
                "additional episodes but may include non-GASC-driven drawdowns. "
                "BT feasibility is low (3/10) due to limited GASC data availability. "
                "WEAT and CORN ETFs have meaningful tracking error vs physical futures. "
                "Held-day Sharpe metrics on short commodity ETFs are highly path-dependent."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events} (hard={n_hard}, proxy={n_proxy})")
    print(f"  event_summary: {event_summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )


if __name__ == "__main__":
    main()
