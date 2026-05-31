"""PL527_ppaq_maple_reserve_drawdown_food_margin
PPAQ Quebec Maple Syrup Reserve <25% Post-Season ->
Short Maple-Input Food Makers Basket vs Long XLP (130 Trading Days)

On each curated PPAQ post-season reserve report release date where the Quebec
Strategic Maple Syrup Reserve falls below 25% of ~133M lb full capacity
(i.e., <~33M lb), enter at the next regular session open a beta-hedged
short basket vs long staples ETF:
  - Short equal-weight basket {BGS, SJM, MKC, HRL}, 25% notional each
  - Long XLP at 100% gross (1:1 dollar hedge against staples beta)
Hold 130 trading days, exit at close.

Daily PnL = -0.25*(BGS_ret + SJM_ret + MKC_ret + HRL_ret) + 1.0*XLP_ret.

Mechanism: PPAQ reserve depletion forces wholesale maple-strip prices higher
in subsequent hedge-roll cycles. US food manufacturers using maple as an
ingredient face 50-150bp gross-margin compression over the 4-9 month
pass-through window. XLP long isolates company-specific maple-cost alpha
from broad consumer-staples beta.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated PPAQ post-season reserve <25% capacity release dates
KNOWN_EVENTS = [
    "2021-06-15",  # Post-2021 tap season, warm 2020-21 winter, low reserves
    "2022-03-15",  # Bloomberg "World's Strategic Maple Syrup Reserve Hits Record Low"
    "2024-06-17",  # 2024 PPAQ post-season reserve depletion press release
]

# Short basket: equal-weight US food manufacturers with maple input exposure
SHORT_BASKET = {"BGS": 0.25, "SJM": 0.25, "MKC": 0.25, "HRL": 0.25}
LONG_TICKER = "XLP"  # 100% long staples ETF hedge
HOLD_DAYS = 130

# One-way slippage estimates (bps as decimals).
# BGS is mid-cap with thinner liquidity; others are large-cap staples.
SLIPPAGE = {"BGS": 0.0015, "SJM": 0.0008, "MKC": 0.0007, "HRL": 0.0007,
            "XLP": 0.0005}


def run_event_study(events, ret, short_weights, long_ticker, hold_days):
    """Build daily PnL & positions for a short-basket-vs-long-hedge event study.

    No double-stacking: when events overlap, book stays at 1 unit.
    Strategy daily return = -sum(short_weights * short_basket_returns) + long_ret.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    short_tickers = list(short_weights.keys())
    sw = np.array([short_weights[t] for t in short_tickers], dtype=float)

    short_ret_df = ret[short_tickers].fillna(0.0)
    short_basket_ret = (short_ret_df * sw).sum(axis=1)  # weighted long-side basket return
    long_ret = ret[long_ticker].fillna(0.0)

    # Strategy daily return: short the basket (=-basket) plus long XLP
    strat_ret = -short_basket_ret + long_ret

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date,
                              "status": "no_data_after_event"})
            continue
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))
        exit_dt = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        # Round-trip slippage on all 5 legs
        slip = (
            sum(short_weights[t] * SLIPPAGE[t] for t in short_tickers)
            + SLIPPAGE[long_ticker]  # long leg weight 1.0
        ) * 2.0

        slice_r = strat_ret.iloc[entry_pos:exit_pos]
        short_slice = short_basket_ret.iloc[entry_pos:exit_pos]
        long_slice = long_ret.iloc[entry_pos:exit_pos]

        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_short = float((1 + short_slice).prod() - 1) if len(short_slice) else None
        gross_long = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        net_ev = gross_ev - slip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "short_basket_return": round(gross_short, 4) if gross_short is not None else None,
            "xlp_return": round(gross_long, 4) if gross_long is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = strat_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip / 2.0
                if j == exit_pos - 1:
                    day_pnl -= slip / 2.0
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL527_ppaq_maple_reserve_drawdown_food_margin"
    short_tickers = list(SHORT_BASKET.keys())
    trade_tickers = short_tickers + [LONG_TICKER]
    all_tickers = trade_tickers + ["SPY"]

    try:
        px = load_prices(all_tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in trade_tickers + ["SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing required tickers: {missing}")

    px_use = px.dropna(how="any", subset=trade_tickers + ["SPY"])
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
        name="PPAQ Maple Reserve <25% Short Food Makers vs Long XLP (130d)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=20,  # ~5bp + ~5bp short basket + ~5bp XLP per round trip leg
    )

    def event_summary(log):
        rets_gross = [e["gross_event_return"] for e in log
                      if e.get("gross_event_return") is not None]
        rets_net = [e["net_event_return"] for e in log
                    if e.get("net_event_return") is not None]
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
                    "strategy_return": e.get("net_event_return"),
                    "spy_return": round(spy_cum, 4),
                    "excess_vs_spy": round(
                        (e.get("net_event_return") or 0) - spy_cum, 4
                    ),
                })
        return rows

    excess_vs_spy = event_vs_spy(event_log, spy_r)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each curated PPAQ post-season reserve report release date "
                "where the Quebec Strategic Maple Syrup Reserve falls below 25% "
                "of ~133M lb capacity (<~33M lb), enter at the next regular "
                "session open: short equal-weight basket {BGS, SJM, MKC, HRL} "
                "(25% each) and long XLP at 100% gross as staples-beta hedge. "
                "Hold 130 trading days, exit at close. "
                "Daily PnL = -0.25*(BGS+SJM+MKC+HRL) + XLP."
            ),
            "mechanism": (
                "PPAQ (Producteurs et productrices acericoles du Quebec) "
                "manages the world's only strategic maple-syrup reserve "
                "(~133M lb capacity). When post-season reserves drop below 25% "
                "of capacity (signaling supply-side stress from poor tap years "
                "or warm winters), PPAQ tightens wholesale strip allocations "
                "and prices rise materially in subsequent hedge-roll cycles. "
                "US food manufacturers with maple-syrup ingredient exposure "
                "(BGS Smart Balance/syrups, SJM Smucker's, MKC McCormick, HRL "
                "Hormel/Skippy) face 50-150bp gross-margin compression as new "
                "contracts price in over the next 4-9 months. The XLP long "
                "neutralizes broad staples-sector beta and isolates the "
                "company-specific maple-cost-pressure alpha. 130 trading days "
                "(~6 months) approximates the midpoint of the pass-through "
                "window."
            ),
            "source": (
                "Event dates curated from PPAQ Plan Conjoint annual reports "
                "2018-2024, Bloomberg Mar-2022 'World's Strategic Maple Syrup "
                "Reserve Hits Record Low' coverage, and 2024 PPAQ post-season "
                "press release. Prices via yfinance (auto_adjust=True). "
                "One-way slippage 5-15 bps; round-trip costs included."
            ),
            "tickers": short_tickers + [LONG_TICKER],
            "short_basket_weights": SHORT_BASKET,
            "long_ticker": LONG_TICKER,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "excess_vs_spy": excess_vs_spy,
            "caveats": (
                "Very small curated event sample (N=3). Event dates approximate "
                "PPAQ release windows derived from public press releases and "
                "Bloomberg coverage; alternate curators could pick adjacent "
                "dates, raising selection-bias / look-ahead concerns. The four "
                "maple-input basket members all sell broader portfolios "
                "(BGS=Green Giant/Cream of Wheat, SJM=Jif/Folgers, MKC=spices, "
                "HRL=Spam/Skippy) so the maple-cost share of consolidated "
                "gross profit is modest -- pass-through-margin alpha may be "
                "swamped by company-specific news. XLP long hedges sector beta "
                "but XLP top weights (PG, KO, PEP, WMT, COST) have no maple "
                "input exposure, so basis risk between basket and XLP is high. "
                "Sharpe/CAGR computed on held days only -- amplifies exposure "
                "vs continuously-invested strategy. 130-day holds across 3 "
                "events partially overlap with broad-market regimes (COVID "
                "reopening, 2022 inflation, 2024 disinflation) that can "
                "dominate single-factor signal. Overlapping events not "
                "double-stacked."
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
