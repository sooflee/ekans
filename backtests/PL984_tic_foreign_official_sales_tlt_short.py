"""PL984_tic_foreign_official_sales_tlt_short
TIC Foreign Official Net Treasury Sales -> Short TLT (Duration Headwind)

Event-study backtest using known TIC threshold-crossing dates (trailing-12-month
foreign official net long-term Treasury purchases < -150B USD). Signal dates
from strategy spec known_events:
  1) 2016-07-01 -- 2016 EM FX defense selling episode
  2) 2017-03-01 -- continuation of EM outflow cycle
  3) 2022-06-01 -- 2022 de-dollarization / rate-hike balance sheet reduction
  4) 2023-03-01 -- continuation de-dollarization episode

ENTRY: Short TLT at next trading session open after TIC release date.
EXIT: 8-week holding period (40 trading days).
REGIME FILTER (primary): DGS10 >= 3.5% (strict) vs. DGS10 >= 2.0% (relaxed).
BENCHMARK: SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known TIC threshold-crossing signal dates from strategy spec
KNOWN_EVENTS = [
    {"signal_date": "2016-07-01", "episode": "EM_FX_defense_2016"},
    {"signal_date": "2017-03-01", "episode": "EM_FX_defense_2017_continuation"},
    {"signal_date": "2022-06-01", "episode": "de_dollarization_2022"},
    {"signal_date": "2023-03-01", "episode": "de_dollarization_2023_continuation"},
]

HOLD_DAYS = 40    # 8-week holding period in trading days


def apply_regime_filter(events, dgs10, min_dgs10=3.5):
    """Keep only events where 10y yield >= min_dgs10 at signal date."""
    out = []
    for ev in events:
        dt = pd.Timestamp(ev["signal_date"])
        future = dgs10.index[dgs10.index >= dt]
        if len(future) == 0:
            ev_rec = dict(ev)
            ev_rec["dgs10_at_signal"] = None
            ev_rec["regime_filter_pass"] = False
            out.append(ev_rec)
            continue
        rate = float(dgs10[future[0]])
        ev_rec = dict(ev)
        ev_rec["dgs10_at_signal"] = round(rate, 2)
        ev_rec["regime_filter_pass"] = (rate >= min_dgs10)
        out.append(ev_rec)
    return out


def run_event_study(events, ret, hold_days=40, require_pass=True):
    """
    Short TLT for hold_days after signal_date (next trading session).
    Returns (pnl_series, positions_series, event_log).
    pnl = -TLT_return each day (short position).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    tlt_ret = ret["TLT"].fillna(0)

    event_log = []
    for ev in events:
        if require_pass and not ev.get("regime_filter_pass", True):
            ev_rec = dict(ev)
            ev_rec["status"] = "regime_filtered"
            event_log.append(ev_rec)
            continue

        sig_dt = pd.Timestamp(ev["signal_date"])
        future_sessions = idx[idx > sig_dt]
        if len(future_sessions) == 0:
            ev_rec = dict(ev)
            ev_rec["status"] = "no_data_after_signal"
            event_log.append(ev_rec)
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Short TLT: PnL = -TLT daily return
        tlt_slice = tlt_ret.iloc[entry_pos:exit_pos]
        short_pnl = -tlt_slice

        # Cumulative returns over hold window
        ev_ret_short = float((1 + short_pnl).prod() - 1) if len(short_pnl) else None
        ev_ret_tlt = float((1 + tlt_slice).prod() - 1) if len(tlt_slice) else None

        spy_slice = ret["SPY"].fillna(0).iloc[entry_pos:exit_pos]
        ev_ret_spy = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        ief_slice = ret["IEF"].fillna(0).iloc[entry_pos:exit_pos]
        ev_ret_ief = float((1 + ief_slice).prod() - 1) if len(ief_slice) else None

        ev_rec = dict(ev)
        ev_rec["entry_date"] = str(entry_dt.date())
        ev_rec["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_rec["n_hold_days"] = int(exit_pos - entry_pos)
        ev_rec["short_tlt_return"] = round(ev_ret_short, 4) if ev_ret_short is not None else None
        ev_rec["tlt_raw_return"] = round(ev_ret_tlt, 4) if ev_ret_tlt is not None else None
        ev_rec["spy_return"] = round(ev_ret_spy, 4) if ev_ret_spy is not None else None
        ev_rec["ief_return"] = round(ev_ret_ief, 4) if ev_ret_ief is not None else None
        ev_rec["excess_vs_spy"] = round(ev_ret_short - ev_ret_spy, 4) if (ev_ret_short is not None and ev_ret_spy is not None) else None
        ev_rec["status"] = "ok"
        event_log.append(ev_rec)

        # Fill in pnl + positions (no overlapping events expected)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0   # short position
                j_rel = j - entry_pos
                if j_rel < len(short_pnl):
                    pnl.iloc[j] = short_pnl.iloc[j_rel]

    return pnl, positions, event_log


def compute_event_summary(event_log):
    rets = [e["short_tlt_return"] for e in event_log if e.get("short_tlt_return") is not None]
    if not rets:
        return None
    return {
        "n_events": len(rets),
        "avg_short_tlt_return": round(float(np.mean(rets)), 4),
        "median_short_tlt_return": round(float(np.median(rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
        "best": round(float(np.max(rets)), 4),
        "worst": round(float(np.min(rets)), 4),
    }


def main():
    sid = "PL984_tic_foreign_official_sales_tlt_short"
    tickers = ["TLT", "IEF", "TBT", "SPY"]

    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Load DGS10 for regime filter
    try:
        dgs10 = load_fred("DGS10", start="2015-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED DGS10 load: {e}")

    # Apply strict regime filter (DGS10 >= 3.5%)
    events_strict = apply_regime_filter(KNOWN_EVENTS, dgs10, min_dgs10=3.5)
    # Apply relaxed regime filter (DGS10 >= 2.0%) for sensitivity
    events_relaxed = apply_regime_filter(KNOWN_EVENTS, dgs10, min_dgs10=2.0)
    # No filter (all 4 events)
    events_all = apply_regime_filter(KNOWN_EVENTS, dgs10, min_dgs10=0.0)

    pnl_strict, pos_strict, log_strict = run_event_study(events_strict, ret, HOLD_DAYS, require_pass=True)
    pnl_relaxed, pos_relaxed, log_relaxed = run_event_study(events_relaxed, ret, HOLD_DAYS, require_pass=True)
    pnl_all, pos_all, log_all = run_event_study(events_all, ret, HOLD_DAYS, require_pass=True)

    n_strict = sum(1 for e in log_strict if e.get("status") == "ok")
    n_relaxed = sum(1 for e in log_relaxed if e.get("status") == "ok")
    n_all = sum(1 for e in log_all if e.get("status") == "ok")

    summary_strict = compute_event_summary(log_strict)
    summary_relaxed = compute_event_summary(log_relaxed)
    summary_all = compute_event_summary(log_all)

    print(f"  n_strict (DGS10>=3.5%): {n_strict}")
    print(f"  n_relaxed (DGS10>=2.0%): {n_relaxed}")
    print(f"  n_all (no filter): {n_all}")

    # Use the 'all events' variant as primary (most data for Sharpe estimation)
    primary_pnl = pnl_all
    primary_pos = pos_all
    primary_log = log_all
    primary_label = "all_4_events_no_regime_filter"

    held_pnl = primary_pnl[primary_pos != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    held_positions = primary_pos.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        extra = {
            "status": "insufficient_data",
            "n_all": n_all,
            "n_held_days": len(held_pnl),
            "log_all": log_all,
            "summary_all": summary_all,
            "summary_strict": summary_strict,
            "summary_relaxed": summary_relaxed,
        }
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_all} events",
            extra=extra,
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="TIC Foreign Official Sales Short TLT (held-days only)",
        positions=held_positions,
        cost_bps=10,
    )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When trailing-12-month foreign official net long-term Treasury "
                "purchases (TIC Table 5) cross below -150B USD: short TLT at the "
                "next session open; hold 40 trading days (~8 weeks); exit at close."
            ),
            "mechanism": (
                "Foreign official institutions (central banks, SWFs) are major holders "
                "of long-duration USTs. Sustained net selling creates persistent "
                "duration supply that pressures yields upward and TLT downward. During "
                "EM FX-defense episodes and de-dollarization, selling is correlated "
                "and sustained (months-long), making the signal persistent enough for "
                "a short-TLT trade over the 8-week window."
            ),
            "source": (
                "US Treasury TIC data (ticdata.treasury.gov, Table 5); signal dates "
                "embedded from strategy spec known_events. DGS10 via FRED. "
                "Prices via yfinance (auto_adjust=True)."
            ),
            "tickers": tickers,
            "primary_label": primary_label,
            "log_all": log_all,
            "log_strict": log_strict,
            "log_relaxed": log_relaxed,
            "summary_all": summary_all,
            "summary_strict": summary_strict,
            "summary_relaxed": summary_relaxed,
            "n_all": n_all,
            "n_strict": n_strict,
            "n_relaxed": n_relaxed,
            "hold_days": HOLD_DAYS,
            "caveats": (
                f"Event study with N={n_all} events total (strict DGS10>=3.5% leaves "
                f"only N={n_strict}). Insufficient for BH-significance or robust "
                "Sharpe estimation. Known_events are approximate (based on narrative "
                "TIC reporting, not automated threshold detection from raw TIC data). "
                "TLT short is exposed to Fed QE surprise risk. 2016 and 2022 events "
                "occurred in low-yield environments where the short-TLT thesis is "
                "weakest (yields already near/below term-premium floor)."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  primary: {primary_label}, n_events={n_all}, held_days={len(held_pnl)}")
    print(f"  summary_all: {summary_all}")
    print(f"  summary_strict: {summary_strict}")
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
