"""PL523_uscis_h1b_denial_spike_india_it
USCIS H-1B Quarterly Denial-Rate Spike -> Long INFY+WIT vs ACN
(India IT Onshore Repricing Margin Expansion)

On each curated USCIS H-1B Employer Data Hub quarterly release date where the
headline H-1B initial-petition denial rate jumped >= 5pp QoQ, enter at the
next session open a dollar-neutral pair: long basket (50% INFY + 50% WIT),
short equal-notional ACN. Hold 63 trading days; exit at close.

Daily PnL = 0.5*INFY_ret + 0.5*WIT_ret - ACN_ret on held days.

KFRC and MHH are reference tickers (loaded but not in trade legs).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated USCIS H-1B denial-rate spike quarterly release dates.
KNOWN_EVENTS = [
    "2017-04-15",
    "2017-10-15",
    "2018-04-15",
    "2018-10-15",
    "2019-01-15",
    "2019-07-15",
    "2020-06-22",
    "2020-10-15",
]

# Long basket
BASKET = {"INFY": 0.50, "WIT": 0.50}
SHORT_TICKER = "ACN"  # equal-notional short (weight 1.0)

# One-way slippage estimates. ADRs (INFY/WIT) and large-cap ACN are liquid.
SLIPPAGE = {"INFY": 0.0005, "WIT": 0.0005, "ACN": 0.0003}

HOLD_DAYS = 63


def run_event_study(events, ret, basket_weights, short_ticker, hold_days):
    """Build daily PnL & positions for a long-short event study.

    No double-stacking: when events overlap, book stays at 1 unit.
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

        slip = (
            sum(basket_weights[t] * SLIPPAGE[t] for t in long_tickers)
            + SLIPPAGE[short_ticker]  # short leg weight 1.0
        ) * 2.0  # round trip

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
            "acn_return": round(gross_short, 4) if gross_short is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

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
    sid = "PL523_uscis_h1b_denial_spike_india_it"
    long_tickers = list(BASKET.keys())
    trade_tickers = long_tickers + [SHORT_TICKER]
    # Include SPY benchmark; KFRC/MHH optional reference (loaded if available)
    all_tickers = trade_tickers + ["SPY", "KFRC", "MHH"]

    try:
        px = load_prices(all_tickers, start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing_trade = [t for t in trade_tickers + ["SPY"] if t not in px.columns]
    if missing_trade:
        return mark_failed(sid, f"missing required tickers: {missing_trade}")

    px_use = px.dropna(how="any", subset=trade_tickers + ["SPY"])
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
        name="USCIS H-1B Denial Spike Long INFY+WIT vs ACN (held-days)",
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

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each curated USCIS H-1B Employer Data Hub quarterly release date "
                "where the headline H-1B initial-petition denial rate jumped >= 5 "
                "percentage points QoQ (2017-2020), enter at the next session open a "
                "dollar-neutral pair: long basket 50% INFY + 50% WIT, short equal-"
                "notional ACN. Hold 63 trading days then exit at close."
            ),
            "mechanism": (
                "Rising H-1B denial rates raise the marginal cost and reduce the "
                "availability of onshore Indian-IT staffing in the US. India-headquartered "
                "IT services firms (INFY, WIT) have established offshore delivery "
                "infrastructure and can re-route work to lower-cost India centers, "
                "expanding margins. Pure onshore competitors like ACN, with lower "
                "offshore mix, face higher labor costs and bill-rate pressure. The "
                "long INFY+WIT / short ACN spread should capture the relative "
                "margin/repricing benefit. 63 trading days (~1 quarter) is the right "
                "edge of the news -> guidance -> bookings propagation window."
            ),
            "source": (
                "Event dates curated from USCIS H-1B Employer Data Hub quarterly "
                "releases and NFAP / USCIS-OPQ policy-shift quarterly reports 2017-"
                "2020. Prices via yfinance (auto_adjust=True). One-way slippage 5 bps "
                "ADR / 3 bps large-cap."
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
                "Small curated event sample (n=8). Event dates are approximate to "
                "USCIS quarterly release windows; different curators could pick "
                "different dates, raising selection-bias / look-ahead concerns. "
                "Events cluster in 2017-2020 covering one full Trump-era enforcement "
                "regime, so results may not generalize. ACN has rapidly grown its "
                "India workforce post-2020 which erodes the long-short thesis going "
                "forward. Overlapping events not double-stacked. Sharpe/CAGR computed "
                "on held days only -- amplifies exposure but small n caveats apply. "
                "INFY/WIT are ADRs and may exhibit nontrivial overnight gap risk vs "
                "Indian listings."
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
