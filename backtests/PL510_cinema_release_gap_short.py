"""PL510_cinema_release_gap_short
MPAA Wide-Release Calendar Gap -> Short Cinema Operators, Long XLY

Rule (strict):
  On each event date when Box Office Mojo's forward 8-week wide-release
  calendar (films opening on >= 2,000 screens) shows fewer than 6 scheduled
  wide releases, enter a market-neutral basket at the next session's open:
      short CNK  (25% notional)
      short IMAX (25% notional)
      long  XLY  (50% notional)
  Hold 40 trading days (~8 weeks) and exit at the close.  SPY is the
  benchmark.  AMC excluded due to 2021 meme-stock idiosyncratic noise.

Relaxed sensitivity: same rule but with the threshold widened to forward
8-week count below 7 (used only if the strict events yield fewer than ~30
held days of in-trade data).

Per-event basket return uses compound returns over the 40-day hold.  The
daily aggregate position series is built (=1 during in-trade days, 0
otherwise) to support held-days-only Sharpe / CAGR.

Event dates are hard-coded from public Box Office Mojo / The Numbers
calendars and Deadline / Variety trade-press coverage of sparse-slate
windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# ---------- Hard-coded gap event dates (strict: forward 8-wk wide-release count < 6) ----------
STRICT_EVENTS = [
    "2020-03-16",  # COVID closures and pulls; forward 8-wk count ~0
    "2020-09-08",  # ongoing COVID gap; Tenet underperformance; Mulan to PVOD
    "2023-09-05",  # post WGA/SAG strike slate thinning; late-2023 releases pulled
    "2024-01-08",  # post-holiday early-2024 sparse window after strikes
    "2024-08-12",  # post Deadpool/Wolverine summer-end gap; sparse late-Aug/Sept slate
]

# ---------- Relaxed sensitivity (threshold < 7) ----------
# Adds widely-noted near-misses (sparse-but-not-empty 8-week forward windows).
RELAXED_EXTRA_EVENTS = [
    "2021-08-30",  # Delta variant; some studios pulling fall titles
    "2022-01-10",  # Omicron-driven post-holiday slate sparseness
    "2023-01-09",  # post-Avatar2 winter slate (Jan/Feb 2023 sparse)
    "2025-01-13",  # post-holiday 2025 sparse window per trade press
]


def run_event_study(event_dates, ret, weights, idx, hold_days=40):
    """Enter weighted basket at next session after each event date; hold for
    `hold_days` trading days.  weights: dict ticker -> signed weight (sum to 0
    net here).  Returns (pnl_series, positions_series, event_log).

    On days inside more than one event's hold window, the position stays at 1.0
    (no double counting); pnl uses the basket return on each in-trade day.
    """
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    # basket_ret on each trading day = sum_w(w * ret[ticker])
    basket_ret = pd.Series(0.0, index=idx)
    for tkr, w in weights.items():
        basket_ret = basket_ret + w * ret[tkr].fillna(0.0)

    event_log = []
    for ev_str in event_dates:
        rel = pd.Timestamp(ev_str)
        future = idx[idx > rel]
        if len(future) == 0:
            event_log.append({"event_date": ev_str, "status": "no_data_after_release"})
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_r = basket_ret.iloc[entry_pos:exit_pos]
        ev_ret = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        event_log.append({
            "event_date": ev_str,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return": round(ev_ret, 4) if ev_ret is not None else None,
        })

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
                "event_date": e["event_date"],
                "entry_date": entry,
                "basket_return": e.get("event_return"),
                "spy_return": round(spy_cum, 4),
                "excess": round((e.get("event_return") or 0) - spy_cum, 4),
            })
    return rows


def main():
    sid = "PL510_cinema_release_gap_short"
    basket_tickers = ["CNK", "IMAX", "XLY"]
    tickers = basket_tickers + ["SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01", end="2026-05-28")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Trim to series with data
    px = px.dropna(how="all")
    ret = daily_returns(px).dropna(how="all")
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # Basket weights: -25% CNK, -25% IMAX, +50% XLY  (net 0, gross 100%)
    weights = {"CNK": -0.25, "IMAX": -0.25, "XLY": 0.50}

    # --- Strict events ---
    pnl_strict, pos_strict, log_strict = run_event_study(
        STRICT_EVENTS, ret, weights, idx, hold_days=40,
    )
    # --- Relaxed events (strict + extras) ---
    relaxed_events = sorted(set(STRICT_EVENTS) | set(RELAXED_EXTRA_EVENTS))
    pnl_relaxed, pos_relaxed, log_relaxed = run_event_study(
        relaxed_events, ret, weights, idx, hold_days=40,
    )

    held_strict = pnl_strict[pos_strict > 0]
    held_relaxed = pnl_relaxed[pos_relaxed > 0]

    if len(held_strict) >= 30:
        primary_pnl, primary_pos, primary_log = pnl_strict, pos_strict, log_strict
        primary_label = "strict_<6_wide_releases_fwd_8wk"
    elif len(held_relaxed) >= 30:
        primary_pnl, primary_pos, primary_log = pnl_relaxed, pos_relaxed, log_relaxed
        primary_label = "relaxed_<7_wide_releases_fwd_8wk"
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
        name="MPAA Wide-Release Gap -> Short CNK/IMAX + Long XLY (held-days only)",
        positions=primary_pos.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When the Box Office Mojo forward 8-week wide-release calendar "
                "(>=2000-screen openings) shows fewer than 6 scheduled wide "
                "releases, enter at next session's open: short CNK 25%, short "
                "IMAX 25%, long XLY 50% (gross 100%, net 0).  Hold 40 trading "
                "days (~8 weeks) and exit at the close.  Benchmark: SPY."
            ),
            "mechanism": (
                "Cinema operators (CNK, IMAX) earn revenue almost entirely from "
                "ticketed admissions to wide theatrical releases; a sparse "
                "forward slate translates into mechanically lower attendance, "
                "concession spend, and operating leverage 6-12 weeks out.  "
                "Wide-release calendars are published 2-3 months in advance, so "
                "the signal is observable without lookahead.  The XLY long is a "
                "consumer-discretionary hedge that neutralizes broad consumer-"
                "spending beta, isolating the cinema-specific demand shortfall.  "
                "AMC is excluded because 2021 meme-stock dynamics decoupled it "
                "from theatrical fundamentals."
            ),
            "source": (
                "Box Office Mojo wide-release calendar (boxofficemojo.com/calendar), "
                "The Numbers (the-numbers.com) historical wide-release archives, "
                "Deadline / Variety trade-press coverage of sparse-slate windows "
                "(COVID 2020 closures; post-WGA/SAG strike late-2023 / early-2024; "
                "post-Deadpool & Wolverine summer-2024 gap).  Prices via yfinance "
                "(auto_adjust=True)."
            ),
            "tickers": basket_tickers,
            "basket_weights": {k: float(v) for k, v in weights.items()},
            "primary_rule_label": primary_label,
            "events_strict": log_strict,
            "events_relaxed": log_relaxed,
            "summary_strict": event_summary(log_strict),
            "summary_relaxed": event_summary(log_relaxed),
            "excess_vs_spy_strict": event_vs_spy(log_strict, spy_r),
            "excess_vs_spy_relaxed": event_vs_spy(log_relaxed, spy_r),
            "caveats": (
                "Event dates are hard-coded approximations of forward-calendar "
                "transitions reported by Box Office Mojo / Variety / Deadline.  "
                "Small event sample (5 strict / 9 relaxed); event-study sample "
                "size is the dominant statistical limitation.  Held-days-only "
                "Sharpe annualizes a small in-trade window so it should be "
                "interpreted cautiously.  COVID 2020 events sit on top of an "
                "unprecedented exogenous shock and drive much of the basket "
                "PnL; sensitivity excluding 2020 should be run before sizing.  "
                "Both AMC (meme dynamics) and Marcus Corp (regional thin float) "
                "were excluded; basket reflects only the two large-cap cinema "
                "names with relatively clean equity dynamics over 2018-2026.  "
                "Period: 2018-01 through 2026-05."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  primary rule: {primary_label}")
    print(f"  events strict: {sum(1 for e in log_strict if e.get('entry_date'))}; "
          f"events relaxed: {sum(1 for e in log_relaxed if e.get('entry_date'))}")
    print(f"  summary_strict: {event_summary(log_strict)}")
    print(f"  summary_relaxed: {event_summary(log_relaxed)}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}  Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
