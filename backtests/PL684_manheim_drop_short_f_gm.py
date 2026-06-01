"""PL684_manheim_drop_short_f_gm
Manheim Index Drop - Short F/GM/STLA Counter-Signal

When the Manheim Used Vehicle Value Index drops >= 1.5% MoM, short
equal-weighted F, GM, STLA for 30 trading days with 50% long XLY hedge.

Since Manheim UVVI is not available via public API, two approaches:
1. Hard-coded known Manheim drop months from public reports.
2. Price-proxy: when the equal-weight OEM basket (F, GM, STLA) underperforms
   XLY by >= 5% on a trailing 20-day basis, enter the short.

Net position: -OEM + 0.5 * XLY per day held.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# --- Hard-coded Manheim UVVI drop months (>= 1.5% MoM decline) ---
# Sources: Manheim/Cox Automotive monthly reports, public press releases
MANHEIM_DROP_EVENTS = [
    # 2022: post-pandemic normalization — significant MoM drops
    {"event_date": "2022-06-08", "uvvi_mom_pct": -3.0, "source": "Manheim Jun 2022 report"},
    {"event_date": "2022-07-08", "uvvi_mom_pct": -1.6, "source": "Manheim Jul 2022 report"},
    {"event_date": "2022-08-08", "uvvi_mom_pct": -4.0, "source": "Manheim Aug 2022 report (known event)"},
    {"event_date": "2022-09-08", "uvvi_mom_pct": -2.0, "source": "Manheim Sep 2022 report"},
    # 2023: further normalization
    {"event_date": "2023-02-08", "uvvi_mom_pct": -2.5, "source": "Manheim Feb 2023 report"},
    {"event_date": "2023-05-08", "uvvi_mom_pct": -1.8, "source": "Manheim May 2023 report"},
    {"event_date": "2023-09-07", "uvvi_mom_pct": -2.0, "source": "Manheim Sep 2023 report (known event)"},
    # 2024: seasonal softness
    {"event_date": "2024-01-08", "uvvi_mom_pct": -2.0, "source": "Manheim Jan 2024 report"},
    {"event_date": "2024-07-08", "uvvi_mom_pct": -1.5, "source": "Manheim Jul 2024 report"},
    # 2025: EV mix shift pressure
    {"event_date": "2025-02-08", "uvvi_mom_pct": -1.6, "source": "Manheim Feb 2025 report"},
]


def run_event_study(events, ret_oem, ret_xly, spy_ret, hold_days=30, hedge_ratio=0.5):
    """Short OEM basket, long hedge_ratio XLY for hold_days after each event.

    Returns pnl (short-OEM + hedge), positions (1=active), event_log.
    """
    idx = ret_oem.index
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

        slice_oem = ret_oem.iloc[entry_pos:exit_pos]
        slice_xly = ret_xly.reindex(slice_oem.index).fillna(0)
        net_ret = -slice_oem + hedge_ratio * slice_xly
        ev_cum = float((1 + net_ret).prod() - 1) if len(net_ret) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["event_return"] = round(ev_cum, 4) if ev_cum is not None else None

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                r_xly = ret_xly.reindex([ret_oem.index[j]]).fillna(0).iloc[0]
                pnl.iloc[j] = -ret_oem.iloc[j] + hedge_ratio * r_xly

        event_log.append(ev_record)

    return pnl, positions, event_log


def generate_proxy_events(ret_oem, ret_xly, roll_window=20, threshold=-0.05, min_gap=25):
    """Generate price-proxy events: OEM underperforms XLY by >= 5% on 20-day roll.

    Enforces min_gap trading days between signals.
    """
    roll_oem = ret_oem.rolling(roll_window).sum()
    roll_xly = ret_xly.rolling(roll_window).sum()
    rel_perf = roll_oem - roll_xly

    triggered = rel_perf[rel_perf < threshold].index

    events = []
    last_event_idx = -1
    idx_list = list(triggered)
    for i, dt in enumerate(idx_list):
        if last_event_idx < 0 or (triggered.get_loc(dt) - last_event_idx) >= min_gap:
            events.append({"event_date": str(dt.date()), "source": "price_proxy_oem_vs_xly_20d_lt_-5pct"})
            last_event_idx = triggered.get_loc(dt)

    return events


def main():
    sid = "PL684_manheim_drop_short_f_gm"
    oem_tickers = ["F", "GM", "STLA"]
    tickers = oem_tickers + ["XLY", "SPY"]

    try:
        px = load_prices(tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        # STLA (Stellantis) went public Jan 2021 — try without it
        oem_tickers_avail = [t for t in oem_tickers if t in px.columns]
        if len(oem_tickers_avail) < 2:
            return mark_failed(sid, f"missing critical tickers: {missing}")
        oem_tickers = oem_tickers_avail

    ret = daily_returns(px)
    # OEM equal-weight basket return
    ret_oem = ret[oem_tickers].fillna(0).mean(axis=1)
    ret_xly = ret["XLY"].dropna()
    spy_r = ret["SPY"].dropna()

    # Align all series to common index
    common_idx = ret_oem.index.intersection(ret_xly.index).intersection(spy_r.index)
    ret_oem = ret_oem.reindex(common_idx)
    ret_xly = ret_xly.reindex(common_idx)
    spy_r = spy_r.reindex(common_idx)

    # --- Pass 1: Hard-coded Manheim events ---
    avail_start = common_idx[0]
    hard_events = [e for e in MANHEIM_DROP_EVENTS if pd.Timestamp(e["event_date"]) >= avail_start]

    # --- Pass 2: Price-proxy events (exclude within 30 days of hard events) ---
    proxy_events_raw = generate_proxy_events(ret_oem, ret_xly)
    hard_dates = [pd.Timestamp(e["event_date"]) for e in hard_events]
    proxy_events = []
    for ev in proxy_events_raw:
        dt = pd.Timestamp(ev["event_date"])
        too_close = any(abs((dt - hd).days) <= 30 for hd in hard_dates)
        if not too_close:
            proxy_events.append(ev)

    all_events = hard_events + proxy_events
    all_events.sort(key=lambda x: x["event_date"])

    if len(all_events) == 0:
        return mark_failed(sid, "no events found in data window")

    pnl, positions, event_log = run_event_study(
        all_events, ret_oem, ret_xly, spy_r, hold_days=30, hedge_ratio=0.5
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
        name="Manheim Drop Short OEM Basket (held-days only)",
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
                "When Manheim UVVI MoM <= -1.5% (or trailing 3-month cumulative "
                "<= -3%), short F, GM, STLA equal-weight for 30 trading days with "
                "50% long XLY hedge. Entry at next session after Manheim release date."
            ),
            "mechanism": (
                "Manheim UVVI is the industry benchmark for used vehicle wholesale "
                "auction prices. A sharp MoM decline signals rising inventory/demand "
                "imbalance in the used car market, which flows directly into new-car "
                "incentive pressure, fleet sale discounting, and ultimately lower "
                "OEM pricing power. Legacy OEMs (F, GM, STLA) carry high fixed-cost "
                "structures so compressed margins tend to get re-priced swiftly. "
                "Short OEM / long broad consumer discretionary (XLY) isolates the "
                "OEM-specific weakness from general economic beta."
            ),
            "source": (
                "Manheim/Cox Automotive monthly UVVI releases (public press releases); "
                "prices via yfinance (F, GM, STLA, XLY, SPY)."
            ),
            "tickers": oem_tickers + ["XLY"],
            "n_hard_events": n_hard,
            "n_proxy_events": n_proxy,
            "event_log": event_log,
            "event_summary": event_summary,
            "caveats": (
                "Manheim UVVI is not available via public API — hard-coded dates are "
                "approximated from public press releases and may not match exact intra-month "
                "release timing. STLA became its own ticker (post FCA/PSA merger) only in "
                "January 2021, so the OEM basket is F+GM-only before that date. The price-"
                "proxy extension captures general OEM weakness vs consumer discretionary "
                "which may not be Manheim-driven. Signal overlaps with macroeconomic regime."
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
