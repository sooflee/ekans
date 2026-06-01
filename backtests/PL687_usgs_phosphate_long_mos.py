"""PL687_usgs_phosphate_long_mos
USGS Phosphate Tier Downgrade - Long MOS

On each USGS Annual Mineral Commodity Summary publication date where the
US phosphate rock reserve or production estimate is revised downward
(a proxy for tier downgrade), go long MOS for 90 trading days with
a 50% short XLB hedge.

Two-pass approach:
1. Hard-coded USGS annual publication dates (released every January/February)
   where US phosphate reserves were revised downward or flagged as declining.
2. Price-proxy extension: MOS outperforms CF on a trailing 60-day basis by >= 8%
   (signals supply-scarcity premium being priced in).

Primary metric: long MOS - 0.5 * short XLB, held-day returns only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# --- USGS Mineral Commodity Summary publication dates ---
# USGS publishes the annual summary every January/February.
# The following dates are approximate annual publication dates.
# US phosphate production has declined significantly since peak ~2000,
# and reserves are being revised down over time. All USGS annual summaries
# from 2018 onward show declining US production / reserve revision trend.
USGS_EVENTS = [
    {"event_date": "2018-01-31", "source": "USGS MCS 2018 Jan"},
    {"event_date": "2019-02-01", "source": "USGS MCS 2019 Feb"},
    {"event_date": "2020-01-31", "source": "USGS MCS 2020 Jan"},
    {"event_date": "2021-01-29", "source": "USGS MCS 2021 Jan"},
    {"event_date": "2022-01-31", "source": "USGS MCS 2022 Jan"},
    {"event_date": "2023-01-31", "source": "USGS MCS 2023 Jan (known event)"},
    {"event_date": "2024-01-30", "source": "USGS MCS 2024 Jan (known event)"},
    {"event_date": "2025-01-30", "source": "USGS MCS 2025 Jan"},
]


def run_event_study(events, ret_mos, ret_xlb, spy_ret, hold_days=90, hedge_ratio=0.5):
    """Long MOS, short hedge_ratio XLB for hold_days after each event.

    Net pnl per day = ret_mos - hedge_ratio * ret_xlb.
    """
    idx = ret_mos.index
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

        slice_mos = ret_mos.iloc[entry_pos:exit_pos]
        slice_xlb = ret_xlb.reindex(slice_mos.index).fillna(0)
        net_ret = slice_mos - hedge_ratio * slice_xlb
        ev_cum = float((1 + net_ret).prod() - 1) if len(net_ret) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["event_return"] = round(ev_cum, 4) if ev_cum is not None else None

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                r_xlb = ret_xlb.reindex([ret_mos.index[j]]).fillna(0).iloc[0]
                pnl.iloc[j] = ret_mos.iloc[j] - hedge_ratio * r_xlb

        event_log.append(ev_record)

    return pnl, positions, event_log


def generate_proxy_events(ret_mos, ret_cf, roll_window=60, threshold=0.08, min_gap=60):
    """Price-proxy: MOS outperforms CF by >= 8% on 60-day roll (phosphate premium signal).

    Enforces minimum gap of min_gap days between events.
    """
    roll_mos = ret_mos.rolling(roll_window).sum()
    roll_cf = ret_cf.rolling(roll_window).sum()
    rel_perf = roll_mos - roll_cf

    triggered = rel_perf[rel_perf > threshold].index

    events = []
    last_event = pd.Timestamp("2000-01-01")
    for dt in triggered:
        if (dt - last_event).days >= min_gap:
            events.append({
                "event_date": str(dt.date()),
                "source": "price_proxy_mos_vs_cf_60d_gt_+8pct",
            })
            last_event = dt
    return events


def main():
    sid = "PL687_usgs_phosphate_long_mos"
    tickers = ["MOS", "CF", "XLB", "SPY"]

    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    ret_mos = ret["MOS"].dropna()
    ret_cf = ret["CF"].dropna()
    ret_xlb = ret["XLB"].dropna()
    spy_r = ret["SPY"].dropna()

    common_idx = ret_mos.index.intersection(ret_cf.index).intersection(ret_xlb.index).intersection(spy_r.index)
    ret_mos = ret_mos.reindex(common_idx)
    ret_cf = ret_cf.reindex(common_idx)
    ret_xlb = ret_xlb.reindex(common_idx)
    spy_r = spy_r.reindex(common_idx)

    # --- Pass 1: USGS annual events (filtered to available data) ---
    avail_start = common_idx[0]
    hard_events = [e for e in USGS_EVENTS if pd.Timestamp(e["event_date"]) >= avail_start]

    # --- Pass 2: Price-proxy events ---
    proxy_events_raw = generate_proxy_events(ret_mos, ret_cf)
    hard_dates = [pd.Timestamp(e["event_date"]) for e in hard_events]
    proxy_events = []
    for ev in proxy_events_raw:
        dt = pd.Timestamp(ev["event_date"])
        # Skip if within 45 days of a hard-coded event (USGS release window)
        too_close = any(abs((dt - hd).days) <= 45 for hd in hard_dates)
        if not too_close:
            proxy_events.append(ev)

    all_events = hard_events + proxy_events
    all_events.sort(key=lambda x: x["event_date"])

    if len(all_events) == 0:
        return mark_failed(sid, "no events found in data window")

    pnl, positions, event_log = run_event_study(
        all_events, ret_mos, ret_xlb, spy_r, hold_days=90, hedge_ratio=0.5
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    n_hard = sum(1 for e in event_log if "price_proxy" not in e.get("source", ""))
    n_proxy = n_events - n_hard

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={"event_log": event_log, "n_events": n_events},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="USGS Phosphate Downgrade Long MOS (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )
    m["n_events"] = n_events

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
                "On USGS Annual Mineral Commodity Summary publication date where US "
                "phosphate rock reserve/production estimate shows downward revision "
                "(tier downgrade), long MOS for 90 trading days with 50% short XLB hedge. "
                "Entry at next session open after the USGS release date."
            ),
            "mechanism": (
                "The USGS MCS is the authoritative source for US critical mineral reserve "
                "estimates. Successive downward revisions to US phosphate reserves signal "
                "long-term tightening of domestic supply, which benefits Mosaic (MOS) as "
                "the dominant US phosphate producer — reduced competition from new domestic "
                "entrants and pricing power from scarcity. The 90-day window captures "
                "analyst estimate revisions and institutional re-rating of the supply story. "
                "Short XLB hedge removes broad materials sector beta."
            ),
            "source": (
                "USGS Mineral Commodity Summaries annual publications (pubs.usgs.gov); "
                "prices via yfinance (MOS, CF, XLB, SPY)."
            ),
            "tickers": ["MOS", "XLB"],
            "n_hard_events": n_hard,
            "n_proxy_events": n_proxy,
            "event_log": event_log,
            "event_summary": event_summary,
            "caveats": (
                "USGS MCS annual releases all show US phosphate production declining, "
                "making this effectively an annual long-MOS calendar trade post-Jan/Feb. "
                "The hard-coded events do not distinguish between years with explicit "
                "tier downgrades vs. routine production updates. USGS data is not "
                "available via public API so dates are approximate. MOS also has "
                "potash and international exposure that dilutes the phosphate signal."
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
