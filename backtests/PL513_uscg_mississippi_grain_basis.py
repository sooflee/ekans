"""PL513_uscg_mississippi_grain_basis
USCG Lower Mississippi Marine Casualty / Low-Water Cluster
-> Short ADM/BG vs Long EWZ (Brazil grain substitution play)

On each curated 'Lower Mississippi disruption' event date (USCG MISLE
marine-casualty cluster of >=3 tow/barge allisions, groundings, or sinkings
within 30 days on Mississippi mile 0-300, plus USCG Sector New Orleans
navigation safety advisory or low-water restriction), enter at next session
open a dollar-neutral pair:
  - Short basket 50% ADM + 50% BG
  - Long equal-notional EWZ
Hold 50 trading days; exit at close.

Daily PnL on held sessions = EWZ return - 0.5*ADM return - 0.5*BG return,
less slippage amortized over entry/exit days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2014-11-20",
    "2015-06-10",
    "2016-01-15",
    "2017-08-30",
    "2018-09-05",
    "2019-05-15",
    "2019-09-13",
    "2020-08-27",
    "2021-08-30",
    "2022-09-23",
    "2022-10-14",
    "2023-09-22",
    "2023-10-20",
    "2024-09-12",
    "2024-10-25",
]

# Short basket weights (each ticker is shorted with these weights, summing to 1.0)
SHORT_BASKET = {"ADM": 0.50, "BG": 0.50}
LONG_TICKER = "EWZ"  # dollar-neutral long leg, weight 1.0

# Per-leg one-way slippage (decimal). Large-cap names; modest costs.
SLIPPAGE = {"ADM": 0.0005, "BG": 0.0005, "EWZ": 0.0005}

HOLD_DAYS = 50


def run_event_study(events, ret, short_basket, long_ticker, hold_days):
    """Build daily PnL & positions for the dollar-neutral pair event study.

    Spread return = long_ticker return - sum(short_basket[t] * ret[t]).
    Positions are NOT stacked (capped at 1 book unit) when events overlap.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    short_tickers = list(short_basket.keys())
    short_w = np.array([short_basket[t] for t in short_tickers], dtype=float)

    short_ret_df = ret[short_tickers].fillna(0.0)
    short_basket_ret = (short_ret_df * short_w).sum(axis=1)
    long_ret = ret[long_ticker].fillna(0.0)

    # Spread = long - short_basket (dollar-neutral pair)
    spread_ret = long_ret - short_basket_ret

    # One round-trip slippage cost in decimal:
    # long leg (weight 1.0) + sum of short basket weights * slippage per leg, * 2 (round trip)
    slip = (
        SLIPPAGE[long_ticker]
        + sum(short_basket[t] * SLIPPAGE[t] for t in short_tickers)
    ) * 2.0

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date, "status": "no_data_after_event"})
            continue
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))
        exit_dt = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        slice_r = spread_ret.iloc[entry_pos:exit_pos]
        long_slice = long_ret.iloc[entry_pos:exit_pos]
        short_slice = short_basket_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_long = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        gross_short = float((1 + short_slice).prod() - 1) if len(short_slice) else None
        net_ev = gross_ev - slip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "long_ewz_return": round(gross_long, 4) if gross_long is not None else None,
            "short_basket_return": round(gross_short, 4) if gross_short is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        # Fill daily PnL on held window; cap at 1 book unit (no double-stacking)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = spread_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip / 2.0
                if j == exit_pos - 1:
                    day_pnl -= slip / 2.0
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL513_uscg_mississippi_grain_basis"
    short_tickers = list(SHORT_BASKET.keys())
    trade_tickers = short_tickers + [LONG_TICKER]
    tickers = trade_tickers + ["SPY"]

    try:
        # Events start 2014; pull 2013 to ensure full first-event window.
        px = load_prices(tickers, start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any", subset=trade_tickers)
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, SHORT_BASKET, LONG_TICKER, HOLD_DAYS,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

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
        name="USCG Lower Mississippi Disruption: Long EWZ / Short (0.5 ADM + 0.5 BG) (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

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

    # Robustness: also report 30-day holding window event summary
    pnl_30, pos_30, log_30 = run_event_study(
        KNOWN_EVENTS, ret, SHORT_BASKET, LONG_TICKER, 30,
    )
    summary_30 = event_summary(log_30)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each curated 'Lower Mississippi disruption' event date (USCG MISLE "
                "marine-casualty cluster of >=3 tow/barge allisions, groundings, or "
                "sinkings within 30 days on Mississippi mile 0-300, plus USCG Sector "
                "New Orleans navigation safety advisory or low-water restriction), "
                "enter at next session open a dollar-neutral pair: short basket 50% "
                "ADM + 50% BG, long equal-notional EWZ. Hold 50 trading days then "
                "exit at close."
            ),
            "mechanism": (
                "Lower Mississippi navigation disruptions (low-water draft restrictions, "
                "post-hurricane closures, ice/fog allision clusters) widen Gulf grain "
                "basis vs interior, raising freight cost and depressing US grain export "
                "margins. ADM and BG dominate US Gulf grain handling/origination, so "
                "their earnings power is squeezed by widened basis and reduced barge "
                "throughput. Brazilian soybean and corn exporters benefit from the "
                "substitution flow, boosting Brazil's macro/FX/equity complex captured "
                "by EWZ. The dollar-neutral pair isolates the US-grain-handling vs "
                "Brazil-export-substitution differential. 50 trading days (~10 weeks) "
                "captures the harvest-to-export-quarter propagation cycle."
            ),
            "source": (
                "USCG Sector New Orleans Marine Safety Information Bulletins, USCG MISLE "
                "marine casualty database, USDA AMS Grain Transportation Report archives, "
                "Soybean Transportation Coalition disruption reports (2014-2024). "
                "Prices via yfinance auto_adjust=True. Per-leg one-way slippage 5 bps."
            ),
            "tickers": trade_tickers,
            "short_basket_weights": SHORT_BASKET,
            "long_ticker": LONG_TICKER,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "summary_30d": summary_30,
            "excess_vs_spy": excess_vs_spy,
            "caveats": (
                "Curated event sample (n=15) of Lower Mississippi navigation disruptions "
                "spanning 2014-2024. Event dates are approximations of USCG-imposed "
                "restrictions (low water 2022/2023/2024, fog/casualty clusters 2014/2016, "
                "post-hurricane closures 2017 Harvey, 2020 Laura, 2021 Ida) and carry "
                "look-ahead curation risk; a different curator might pick different "
                "dates. EWZ is sensitive to broader Brazil/EM macro/FX shocks unrelated "
                "to grain substitution. ADM/BG also have non-Mississippi exposures. "
                "Overlapping events not stacked; Sharpe/CAGR on held days only "
                "(amplifies realized exposure but small effective sample). Statistical "
                "significance modest: 15 events x 50 days = 750 daily obs but heavily "
                "auto-correlated within events. 30-day robustness window reported alongside."
            ),
        },
        pnl=pnl[positions > 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary 50d: {summary}")
    print(f"  summary 30d: {summary_30}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}  Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
