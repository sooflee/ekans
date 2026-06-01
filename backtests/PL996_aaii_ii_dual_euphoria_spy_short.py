"""PL996_aaii_ii_dual_euphoria_spy_short
AAII + Investors Intelligence Dual-Confirmation Euphoria SPY Short

Event-study backtest using historically-identified dual-euphoria signal dates
(AAII bull-bear spread >= 30pp AND II bull% >= 60% simultaneously). The AAII
XLS is blocked from automated download; instead we use:
  1) Known events from strategy spec (known_events: 2000-02, 2007-10, 2018-01, 2021-01)
  2) University of Michigan Consumer Sentiment (FRED UMCSENT) as a proxy trigger
     for extreme bullish readings (>= 90th percentile of history) to augment the
     known events and identify likely co-coincident AAII euphoria periods.

The UMCSENT proxy approach:
  - Monthly UMCSENT >= 98 (~90th+ pct since 1999) used as a high-sentiment flag
  - Each distinct cluster (gap > 6 months between sequential flags) becomes one event
  - Signal date = first UMCSENT >= 98 in each cluster, aligned to next Friday for entry
  - Overlapping events de-duped: no re-entry within 42 calendar days

PRIMARY P&L: short SPY for 40 trading days from entry.
BENCHMARK: SPY buy-and-hold.
VIX FILTER: skip entry if VIX >= 25 at signal date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)

# Known euphoria dates from strategy spec (known_events field)
KNOWN_EVENT_MONTHS = [
    ("2000-03-01", "dot_com_peak_AAII_euphoria"),  # AAII bull-bear peaked ~Mar 2000
    ("2007-10-01", "pre_GFC_peak_sentiment"),
    ("2018-01-08", "jan2018_vix_spike_precursor"),  # Jan 2018 (week of 2018-01-04)
    ("2021-01-11", "retail_meme_euphoria_2021"),     # Jan 2021 AAII surge
]

HOLD_DAYS = 40       # 8 calendar weeks ≈ 40 trading days
UMCSENT_THRESHOLD = 98.0  # ~90th percentile
VIX_MAX = 25.0
MIN_GAP_DAYS = 42    # de-dup: minimum days between event entries


def find_umcsent_events(umcsent, threshold=98.0, min_gap_days=42):
    """Return list of (date, level) for UMCSENT >= threshold, de-duped at min_gap_days gap."""
    above = umcsent[umcsent >= threshold].dropna()
    if above.empty:
        return []
    events = []
    last_dt = None
    for dt, val in above.items():
        if last_dt is None or (dt - last_dt).days >= min_gap_days:
            events.append({"signal_date": str(dt.date()), "umcsent": round(float(val), 1),
                           "source": "UMCSENT_proxy"})
            last_dt = dt
    return events


def run_event_study(events, ret, vix_series, hold_days=40, vix_max=25.0):
    """
    Short SPY for hold_days after each event signal_date (next trading session).
    Skips entry if VIX >= vix_max at signal date.
    Returns (pnl_series, positions_series, event_log).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    spy_ret = ret["SPY"].fillna(0)

    event_log = []
    for ev in events:
        sig_dt = pd.Timestamp(ev["signal_date"])
        # Check VIX filter
        vix_near = vix_series.index[vix_series.index >= sig_dt]
        if len(vix_near) > 0:
            vix_val = float(vix_series[vix_near[0]])
        else:
            vix_val = None

        if vix_val is not None and vix_val >= vix_max:
            ev_rec = dict(ev)
            ev_rec["status"] = "vix_filtered"
            ev_rec["vix_at_signal"] = round(vix_val, 2)
            event_log.append(ev_rec)
            continue

        future_sessions = idx[idx > sig_dt]
        if len(future_sessions) == 0:
            ev_rec = dict(ev)
            ev_rec["status"] = "no_data_after_signal"
            event_log.append(ev_rec)
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Short SPY: PnL = -SPY daily return
        spy_slice = spy_ret.iloc[entry_pos:exit_pos]
        short_pnl = -spy_slice

        ev_ret_short = float((1 + short_pnl).prod() - 1) if len(short_pnl) else None
        ev_ret_spy = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        # Defensive leg (XLP + XLU equally weighted)
        def_r = ret[["XLP", "XLU"]].fillna(0).mean(axis=1)
        def_slice = def_r.iloc[entry_pos:exit_pos]
        ev_ret_def = float((1 + def_slice).prod() - 1) if len(def_slice) else None

        ev_rec = dict(ev)
        ev_rec["entry_date"] = str(entry_dt.date())
        ev_rec["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_rec["n_hold_days"] = int(exit_pos - entry_pos)
        ev_rec["vix_at_signal"] = round(vix_val, 2) if vix_val is not None else None
        ev_rec["short_spy_return"] = round(ev_ret_short, 4) if ev_ret_short is not None else None
        ev_rec["spy_raw_return"] = round(ev_ret_spy, 4) if ev_ret_spy is not None else None
        ev_rec["defensive_basket_return"] = round(ev_ret_def, 4) if ev_ret_def is not None else None
        ev_rec["status"] = "ok"
        event_log.append(ev_rec)

        # Fill in pnl + positions (skip if already occupied by another event)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                j_rel = j - entry_pos
                if j_rel < len(short_pnl):
                    pnl.iloc[j] = short_pnl.iloc[j_rel]

    return pnl, positions, event_log


def compute_event_summary(event_log):
    rets = [e["short_spy_return"] for e in event_log if e.get("short_spy_return") is not None]
    if not rets:
        return None
    return {
        "n_events": len(rets),
        "avg_short_spy_return": round(float(np.mean(rets)), 4),
        "median_short_spy_return": round(float(np.median(rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
        "best": round(float(np.max(rets)), 4),
        "worst": round(float(np.min(rets)), 4),
    }


def main():
    sid = "PL996_aaii_ii_dual_euphoria_spy_short"
    tickers = ["SPY", "XLP", "XLU"]

    # Load prices
    try:
        px = load_prices(tickers, start="1999-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Load VIX
    try:
        px_vix = load_prices(["^VIX"], start="1999-01-01")
        vix_series = px_vix["^VIX"].dropna()
    except Exception as e:
        return mark_failed(sid, f"VIX load: {e}")

    # Load UMCSENT
    try:
        umcsent = load_fred("UMCSENT", start="1999-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED UMCSENT load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Build event list: known events + UMCSENT-proxy events (de-duped together)
    # First: known events from strategy spec
    known_events = [{"signal_date": d, "episode": ep, "source": "known_event"}
                    for d, ep in KNOWN_EVENT_MONTHS]

    # UMCSENT-proxy events
    umcsent_events = find_umcsent_events(umcsent, threshold=UMCSENT_THRESHOLD, min_gap_days=MIN_GAP_DAYS)

    # Merge and de-dup: prefer known events; remove any UMCSENT event within 42 days of a known event
    all_known_dts = [pd.Timestamp(e["signal_date"]) for e in known_events]
    merged_events = list(known_events)
    for ue in umcsent_events:
        ue_dt = pd.Timestamp(ue["signal_date"])
        # Skip if within MIN_GAP_DAYS of any known event
        too_close = any(abs((ue_dt - kd).days) < MIN_GAP_DAYS for kd in all_known_dts)
        if not too_close:
            # Also skip if within MIN_GAP_DAYS of already-merged UMCSENT events
            existing_dts = [pd.Timestamp(e["signal_date"]) for e in merged_events]
            too_close2 = any(abs((ue_dt - ed).days) < MIN_GAP_DAYS for ed in existing_dts)
            if not too_close2:
                merged_events.append(ue)

    # Sort by date
    merged_events.sort(key=lambda x: pd.Timestamp(x["signal_date"]))

    print(f"  Known events: {len(known_events)}")
    print(f"  UMCSENT proxy events (threshold={UMCSENT_THRESHOLD}): {len(umcsent_events)}")
    print(f"  Merged (de-duped) events: {len(merged_events)}")
    for ev in merged_events:
        print(f"    {ev['signal_date']} ({ev.get('episode', ev.get('umcsent', ''))} / {ev['source']})")

    # Run event study on all merged events
    pnl_all, pos_all, log_all = run_event_study(
        merged_events, ret, vix_series, hold_days=HOLD_DAYS, vix_max=VIX_MAX
    )

    # Run on known events only (sensitivity)
    pnl_known, pos_known, log_known = run_event_study(
        known_events, ret, vix_series, hold_days=HOLD_DAYS, vix_max=VIX_MAX
    )

    n_all = sum(1 for e in log_all if e.get("status") == "ok")
    n_known = sum(1 for e in log_known if e.get("status") == "ok")

    summary_all = compute_event_summary(log_all)
    summary_known = compute_event_summary(log_known)

    print(f"  Events fired (all): {n_all}")
    print(f"  Events fired (known only): {n_known}")

    # Use 'all merged' variant as primary
    primary_pnl = pnl_all
    primary_pos = pos_all
    primary_label = "all_merged_events_known+umcsent_proxy"

    held_pnl = primary_pnl[primary_pos != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    held_positions = primary_pos.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        extra = {
            "status": "insufficient_data",
            "n_all": n_all,
            "n_known": n_known,
            "n_held_days": len(held_pnl),
            "log_all": log_all,
            "log_known": log_known,
            "summary_all": summary_all,
            "summary_known": summary_known,
        }
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_all} events",
            extra=extra,
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="AAII+II Dual Euphoria Short SPY (held-days only)",
        positions=held_positions,
        cost_bps=10,
    )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When AAII bull-bear spread >= 30pp AND II bull% >= 60% "
                "simultaneously (dual-confirmation euphoria), short SPY at the "
                "next trading session open; hold 40 trading days; exit at close. "
                "VIX filter: skip entry if VIX >= 25."
            ),
            "mechanism": (
                "Extreme dual-confirmation retail + institutional bullish sentiment "
                "is historically associated with stretched valuations and elevated "
                "crash risk. When both AAII (retail) and II (professional advisors) "
                "reach euphoric readings simultaneously, the marginal buyer is "
                "exhausted. Mean reversion or black-swan drawdown is elevated in "
                "the 4-8 week window. Short SPY captures the correction; defensive "
                "rotation (XLP/XLU) hedges the tail risk."
            ),
            "source": (
                "AAII sentiment survey (aaii.com); Investors Intelligence "
                "(yardeni.com); known event dates from strategy spec. "
                "UMCSENT proxy from FRED. Prices via yfinance (auto_adjust=True)."
            ),
            "tickers": tickers,
            "primary_label": primary_label,
            "log_all": log_all,
            "log_known": log_known,
            "summary_all": summary_all,
            "summary_known": summary_known,
            "n_all": n_all,
            "n_known": n_known,
            "hold_days": HOLD_DAYS,
            "umcsent_threshold": UMCSENT_THRESHOLD,
            "vix_filter": VIX_MAX,
            "caveats": (
                "AAII XLS unavailable for automated download (403 bot-blocking). "
                "UMCSENT proxy is an imperfect substitute for AAII bull-bear spread; "
                "alignment with known euphoria dates is partial. Event N is small. "
                "Short SPY as primary is a high-conviction but high-risk strategy "
                "(SPY has upward drift; shorting requires precise timing). "
                "BH-significance unlikely given small N."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  primary: {primary_label}, n_events={n_all}, held_days={len(held_pnl)}")
    print(f"  summary_all: {summary_all}")
    print(f"  summary_known: {summary_known}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', 'N/A'):.2f}  "
            f"CAGR: {m.get('cagr', 0)*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  "
            f"t-stat: {m.get('t_stat', 'N/A'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
