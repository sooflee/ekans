"""PL511_candida_auris_antifungal_event
CDC Candida Auris Surge Announcements -> Long Antifungal/Diagnostics vs XLV

On each known CDC C. auris escalation announcement date, enter at the next
session open a long basket (50% SCYX + 25% DHR + 25% TMO) and short an
equal-notional XLV (dollar-neutral). Hold 60 trading days; exit at close.

Daily PnL = weighted long basket return MINUS XLV return on hold days.
Aggregate metrics computed on held-day returns only; SPY is reference
benchmark (excess return reporting). 25 bps slippage applied to SCYX.

Event dates curated from CDC HAN/MMWR archive (per queue spec):
  2023-03-20  MMWR on rapid US increase in C. auris cases
  2023-07-13  CDC update on healthcare facility outbreaks
  2024-04-29  Spring 2024 update
  2024-09-10  Fall 2024 update
  2025-03-25  CDC AR Threats annual update (tier-1 urgent threat)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2023-03-20",
    "2023-07-13",
    "2024-04-29",
    "2024-09-10",
    "2025-03-25",
]

# Basket weights (longs)
BASKET = {"SCYX": 0.50, "DHR": 0.25, "TMO": 0.25}
SHORT_TICKER = "XLV"  # equal-notional short (weight 1.0)

# Per-leg one-way slippage (decimal). SCYX is small-cap & illiquid.
SLIPPAGE = {"SCYX": 0.0025, "DHR": 0.0001, "TMO": 0.0001, "XLV": 0.0001}

HOLD_DAYS = 60


def run_event_study(events, ret, basket_weights, short_ticker, hold_days):
    """Build daily PnL & positions for a long-short event study.

    Position is applied to the day's return on each held session (entry day's
    open->close return is captured by the adj-close return on the entry day,
    since adj_close-to-adj_close returns approximate full-day P&L of an
    open-at-entry position when entry is at the open of that session).

    Long basket cumulates with weights; XLV short subtracts.
    When events overlap, positions are NOT stacked (capped at 1 unit of book).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)  # 1.0 when in a position (book size)
    event_log = []

    long_tickers = list(basket_weights.keys())
    weights = np.array([basket_weights[t] for t in long_tickers], dtype=float)

    # Daily long basket return
    long_ret_df = ret[long_tickers].fillna(0.0)
    long_basket_ret = (long_ret_df * weights).sum(axis=1)
    short_ret = ret[short_ticker].fillna(0.0)

    # Spread = long - short (dollar-neutral)
    spread_ret = long_basket_ret - short_ret

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({
                "event_date": ev_date,
                "status": "no_data_after_event",
            })
            continue
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))
        exit_dt = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        # Slippage: one-way entry cost + one-way exit cost on each leg
        # Cost as % of book: sum over legs of weight * slip_one_way * 2
        slip = (
            sum(basket_weights[t] * SLIPPAGE[t] for t in long_tickers)
            + SLIPPAGE[short_ticker]  # short leg weight 1.0
        ) * 2.0  # round trip

        # Per-event cumulative spread return (gross of slippage)
        slice_r = spread_ret.iloc[entry_pos:exit_pos]
        long_slice = long_basket_ret.iloc[entry_pos:exit_pos]
        short_slice = short_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_long = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        gross_short = float((1 + short_slice).prod() - 1) if len(short_slice) else None
        net_ev = gross_ev - slip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "long_basket_return": round(gross_long, 4) if gross_long is not None else None,
            "xlv_return": round(gross_short, 4) if gross_short is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        # Fill daily PnL on held window; cap at 1 book unit (no double-stacking)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                # Day-1: subtract slip/hold_days (amortize entry/exit equally)
                day_pnl = spread_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip / 2.0  # entry slip
                if j == exit_pos - 1:
                    day_pnl -= slip / 2.0  # exit slip
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL511_candida_auris_antifungal_event"
    long_tickers = list(BASKET.keys())
    tickers = long_tickers + [SHORT_TICKER, "SPY"]

    try:
        # SCYX listed 2014-05-02; events start 2023, so 2022 start is safe.
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Drop rows where ANY of the trading tickers is missing
    px_use = px[tickers].dropna(how="any", subset=long_tickers + [SHORT_TICKER])
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, BASKET, SHORT_TICKER, HOLD_DAYS,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    # Restrict metrics to held days only (avoid flat-day Sharpe dilution).
    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)}",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="C. auris Event Long Antifungal/Diagnostics vs XLV (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level summary
    def event_summary(log):
        rets_gross = [e["gross_event_return"] for e in log if e.get("gross_event_return") is not None]
        rets_net = [e["net_event_return"] for e in log if e.get("net_event_return") is not None]
        if not rets_gross:
            return None
        return {
            "n_events": len(rets_gross),
            "avg_gross_event_return": round(float(np.mean(rets_gross)), 4),
            "median_gross_event_return": round(float(np.median(rets_gross)), 4),
            "avg_net_event_return": round(float(np.mean(rets_net)), 4) if rets_net else None,
            "win_rate_gross": round(float(np.mean([r > 0 for r in rets_gross])), 4),
            "win_rate_net": round(float(np.mean([r > 0 for r in rets_net])), 4) if rets_net else None,
            "best": round(float(np.max(rets_gross)), 4),
            "worst": round(float(np.min(rets_gross)), 4),
        }

    summary = event_summary(event_log)

    # Excess vs SPY per event (book = spread; SPY = same window buy-and-hold)
    def event_vs_spy(log, spy_ret):
        rows = []
        for e in log:
            entry = e.get("entry_date"); exit_d = e.get("exit_date")
            if not entry or not exit_d:
                continue
            entry_dt = pd.Timestamp(entry); exit_dt = pd.Timestamp(exit_d)
            spy_slice = spy_ret.loc[entry_dt:exit_dt]
            if len(spy_slice):
                spy_cum = float((1 + spy_slice).prod() - 1)
                rows.append({
                    "event_date": e["event_date"],
                    "entry_date": entry,
                    "spread_return": e.get("net_event_return"),
                    "spy_return": round(spy_cum, 4),
                    "excess_vs_spy": round((e.get("net_event_return") or 0) - spy_cum, 4),
                })
        return rows

    excess_vs_spy = event_vs_spy(event_log, spy_r)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each CDC C. auris escalation announcement date (CDC HAN advisory, "
                "MMWR report on C. auris case acceleration, or annual CDC AR Threats "
                "update flagging C. auris as Tier 1 urgent threat), enter at next "
                "session open a long basket (50% SCYX + 25% DHR + 25% TMO) and short "
                "equal-notional XLV. Hold 60 trading days; exit at close."
            ),
            "mechanism": (
                "C. auris escalation announcements drive hospital and public-health "
                "demand for novel antifungal therapeutics (SCYX is the pure-play "
                "antifungal developer with ibrexafungerp) and for clinical-diagnostics "
                "platforms able to rapidly identify resistant fungal isolates (DHR/Cepheid "
                "and TMO). Shorting XLV neutralizes broad health-care sector beta so the "
                "spread isolates the C. auris-specific demand shock. The 60-day window "
                "captures the news -> guidance -> contract-flow propagation."
            ),
            "source": (
                "CDC Health Alert Network (HAN) advisories, MMWR archive, and CDC "
                "Antibiotic Resistance (AR) Threats Report. Prices via yfinance "
                "(auto_adjust=True). SCYX one-way slippage 25 bps; majors 1 bp."
            ),
            "tickers": long_tickers + [SHORT_TICKER],
            "basket_weights": BASKET,
            "short_ticker": SHORT_TICKER,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "excess_vs_spy": excess_vs_spy,
            "caveats": (
                "Small event sample (n=5 curated dates). Event dates hand-curated from "
                "CDC HAN/MMWR archive — different curators might select different dates "
                "and add look-ahead risk. SCYX is small-cap and illiquid; 25 bps slip is "
                "a conservative estimate but realized cost could be higher. MNK from the "
                "original idea was dropped (delisted). Overlapping events are not "
                "double-stacked. Sharpe / CAGR are computed on held days only, which "
                "amplifies the realized exposure but is sensitive to small n. Statistical "
                "significance is weak (t-stat caveat: 5 events x 60 days = 300 daily "
                "observations, but they are auto-correlated within events)."
            ),
        },
        pnl=pnl[positions > 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary: {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}  Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
