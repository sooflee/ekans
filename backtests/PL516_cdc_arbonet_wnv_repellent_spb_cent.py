"""PL516_cdc_arbonet_wnv_repellent_spb_cent
CDC ArboNET West Nile Case Surge -> Long SPB+CENT vs XLP (Repellent/Adulticide Demand Spike)

On each curated CDC ArboNET West Nile Virus surge acceleration date (Tuesday
weekly bulletin showing US WNV human case count breaching trailing-3yr
same-week 90th percentile for two consecutive weeks), enter at the next
session's open a dollar-neutral pair:
    long 50% SPB + 50% CENT, short equal-notional XLP.
Hold 42 trading days, then exit at close.

Daily PnL on held days = 0.5*SPB return + 0.5*CENT return MINUS XLP return.

Event dates curated from CDC ArboNET WNV final-annual archives and weekly
bulletin peak-week timing 2017-2024, focused on outbreak years (TX 2012,
CA 2017, AZ 2019/2021, US 2018/2021/2022). Peak WNV transmission is mid-Aug
through mid-Sep so events cluster Aug-Sep.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2017-08-08",
    "2018-08-21",
    "2018-09-04",
    "2019-08-13",
    "2020-08-11",
    "2021-08-17",
    "2021-09-07",
    "2022-08-09",
    "2022-08-30",
    "2023-08-15",
    "2024-08-13",
]

# Basket weights (longs)
BASKET = {"SPB": 0.50, "CENT": 0.50}
SHORT_TICKER = "XLP"  # equal-notional short (weight 1.0)

# Per-leg one-way slippage (decimal). SPB & CENT are mid-cap (~$2-3B).
SLIPPAGE = {"SPB": 0.0010, "CENT": 0.0010, "XLP": 0.0001}

HOLD_DAYS = 42  # primary horizon (~8 weeks)
# Sensitivity horizons reported as secondary results
SENS_HOLD_DAYS = [28, 56]


def run_event_study(events, ret, basket_weights, short_ticker, hold_days,
                    slippage):
    """Build daily PnL & positions for a long-short event study.

    Adj-close-to-adj-close returns approximate full-day P&L of an
    open-at-entry position when entry is at the open of that session.
    When events overlap, positions are NOT stacked (capped at 1 book unit).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    long_tickers = list(basket_weights.keys())
    weights = np.array([basket_weights[t] for t in long_tickers], dtype=float)

    long_ret_df = ret[long_tickers].fillna(0.0)
    long_basket_ret = (long_ret_df * weights).sum(axis=1)
    short_ret = ret[short_ticker].fillna(0.0)

    spread_ret = long_basket_ret - short_ret

    # Round-trip slippage (book units): sum over legs of weight*slip*2
    slip_total = (
        sum(basket_weights[t] * slippage[t] for t in long_tickers)
        + slippage[short_ticker]  # short leg weight 1.0
    ) * 2.0

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

        slice_r = spread_ret.iloc[entry_pos:exit_pos]
        long_slice = long_basket_ret.iloc[entry_pos:exit_pos]
        short_slice = short_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_long = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        gross_short = float((1 + short_slice).prod() - 1) if len(short_slice) else None
        net_ev = gross_ev - slip_total if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "long_basket_return": round(gross_long, 4) if gross_long is not None else None,
            "xlp_return": round(gross_short, 4) if gross_short is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip_total, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = spread_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip_total / 2.0
                if j == exit_pos - 1:
                    day_pnl -= slip_total / 2.0
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


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


def main():
    sid = "PL516_cdc_arbonet_wnv_repellent_spb_cent"
    long_tickers = list(BASKET.keys())
    # CLX reference only (not traded); SPY for benchmark
    tickers = long_tickers + [SHORT_TICKER, "CLX", "SPY"]
    trade_tickers = long_tickers + [SHORT_TICKER]

    try:
        # SPB IPO July 2010; CENT older. Start 2016 to bracket first event 2017.
        px = load_prices(tickers, start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in trade_tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[[c for c in tickers if c in px.columns]].dropna(
        how="any", subset=trade_tickers
    )
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna() if "SPY" in ret.columns else None

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, BASKET, SHORT_TICKER, HOLD_DAYS, SLIPPAGE,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna() if spy_r is not None else None

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)}",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="CDC ArboNET WNV Surge Long SPB+CENT vs XLP (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    summary = event_summary(event_log)
    excess = event_vs_spy(event_log, spy_r) if spy_r is not None else []

    # Sensitivity: alternate hold horizons (28d, 56d) — event-level summary only
    sens = {}
    for h in SENS_HOLD_DAYS:
        _pnl, _pos, _log = run_event_study(
            KNOWN_EVENTS, ret, BASKET, SHORT_TICKER, h, SLIPPAGE,
        )
        _held = _pnl[_pos > 0]
        if len(_held) >= 30:
            _m = compute_metrics(
                _held, benchmark=spy_r.reindex(_held.index).dropna() if spy_r is not None else None,
                name=f"sens_{h}d", positions=_pos.reindex(_held.index).fillna(0),
                cost_bps=10,
            )
            sens[f"hold_{h}d"] = {
                "sharpe": _m.get("sharpe"),
                "cagr": _m.get("cagr"),
                "max_dd": _m.get("max_dd"),
                "t_stat": _m.get("t_stat"),
                "n_days": _m.get("n_days"),
                "summary": event_summary(_log),
            }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each curated CDC ArboNET West Nile Virus surge acceleration date "
                "(Tuesday weekly bulletin showing US WNV human case count breaching "
                "trailing-3yr same-week 90th percentile for two consecutive weeks), "
                "enter at the next session's open a dollar-neutral pair: long 50% SPB + "
                "50% CENT, short equal-notional XLP. Hold 42 trading days, then exit at "
                "the close."
            ),
            "mechanism": (
                "CDC ArboNET WNV surge weeks drive consumer demand for personal "
                "repellents (DEET, picaridin, OLE) and municipal/community demand for "
                "adulticide and larvicide products. SPB (Spectrum Brands - Hot Shot, "
                "Cutter, Repel) and CENT (Central Garden & Pet - Adams, Sergeant's, "
                "lawn-care insecticides) are the two listed US pure-plays. XLP (Consumer "
                "Staples Select) neutralizes broad staples-sector beta so the spread "
                "isolates the WNV-specific demand pulse. The 42-day window captures "
                "panic-pull-through into 8-K guidance / Q3 sell-through commentary. "
                "CLX is a reference household-care proxy (not traded in legs)."
            ),
            "source": (
                "CDC ArboNET WNV final annual data archives and Tuesday weekly bulletin "
                "peak-week timing 2017-2024. Prices via yfinance (auto_adjust=True). "
                "Mid-cap one-way slippage 10 bps for SPB & CENT; 1 bp for XLP."
            ),
            "tickers": trade_tickers,
            "basket_weights": BASKET,
            "short_ticker": SHORT_TICKER,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "excess_vs_spy": excess,
            "sensitivity": sens,
            "caveats": (
                "Event dates curated post-hoc from CDC ArboNET archives - look-ahead "
                "risk in date selection (the 'two consecutive 90th-percentile weeks' "
                "rule is applied retrospectively; a real-time implementer would need to "
                "compute the trailing-3yr 90th percentile fresh each Tuesday). Sample "
                "n=11 events across 2017-2024. SPB and CENT are mid-cap "
                "($2-3B); 10 bps one-way slippage assumed but realized cost in a panic "
                "buy could be higher. Overlapping events not double-stacked. Sharpe / "
                "CAGR computed on held days only - amplifies realized exposure but "
                "sensitive to small n. WNV human-case surveillance has a multi-week "
                "reporting lag in ArboNET, so 'breach' weeks lag actual transmission; "
                "this is by design (trade off the *report*, not the *outbreak*)."
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
    if sens:
        print(f"  Sensitivity:")
        for k, v in sens.items():
            print(f"    {k}: Sharpe {v['sharpe']:.2f}  CAGR {v['cagr']*100:.2f}%  n_days {v['n_days']}")


if __name__ == "__main__":
    main()
