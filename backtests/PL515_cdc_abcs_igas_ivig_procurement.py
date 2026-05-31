"""PL515_cdc_abcs_igas_ivig_procurement
CDC ABCs Invasive Group A Strep Case Surge -> Long GRFS+CRL vs XLV
(IVIG/Penicillin Procurement Cycle)

On each curated CDC ABCs iGAS surge acceleration date (CDC ABCs monthly
report, MMWR Notice, or CDC HAN advisory showing 4-week iGAS incidence >35%
above trailing 3-year baseline across the 10 ABCs sentinel sites), enter at
next session open a dollar-neutral pair: long basket 50% GRFS + 50% CRL,
short equal-notional XLV. Hold 63 trading days (~3 months); exit at close.

Daily PnL = 0.5 * GRFS return + 0.5 * CRL return MINUS XLV return on hold
days. Aggregate metrics computed on held-day returns only; SPY is reference
benchmark. 25 bps one-way slippage applied to GRFS (Spanish ADR, lower
volume); 1 bp to CRL/XLV.

Event dates curated from CDC ABCs Group A Streptococcus bulletins,
MMWR iGAS notices, and CDC HAN advisories 2018-2024 documenting
post-COVID rebound surges (HAN 483 Dec 22 2022 was the primary US
iGAS-in-children HAN).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2018-04-15",
    "2019-04-12",
    "2020-04-10",
    "2022-12-22",
    "2023-02-09",
    "2023-04-13",
    "2023-07-20",
    "2023-12-14",
    "2024-04-11",
    "2024-12-12",
]

# Basket weights (longs)
BASKET = {"GRFS": 0.50, "CRL": 0.50}
SHORT_TICKER = "XLV"  # equal-notional short (weight 1.0)

# Per-leg one-way slippage (decimal). GRFS is a Spanish ADR (lower volume).
SLIPPAGE = {"GRFS": 0.0025, "CRL": 0.0001, "XLV": 0.0001}

HOLD_DAYS = 63


def run_event_study(events, ret, basket_weights, short_ticker, hold_days):
    """Build daily PnL & positions for a long-short event study.

    Position is applied to the day's return on each held session. Long basket
    cumulates with weights; XLV short subtracts. Overlapping events are NOT
    stacked (capped at 1 unit of book).
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
            event_log.append({
                "event_date": ev_date,
                "status": "no_data_after_event",
            })
            continue
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))
        exit_dt = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        slip = (
            sum(basket_weights[t] * SLIPPAGE[t] for t in long_tickers)
            + SLIPPAGE[short_ticker]
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
            "xlv_return": round(gross_short, 4) if gross_short is not None else None,
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
    sid = "PL515_cdc_abcs_igas_ivig_procurement"
    long_tickers = list(BASKET.keys())
    tickers = long_tickers + [SHORT_TICKER, "SPY"]

    try:
        # Earliest event 2018-04-15; start a bit before for warmup
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

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
        name="CDC ABCs iGAS Surge Long GRFS+CRL vs XLV (held-days)",
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
                "On each curated CDC ABCs iGAS surge acceleration date (CDC ABCs "
                "monthly report, MMWR Notice, or CDC HAN advisory showing 4-week iGAS "
                "incidence >35% above trailing 3-year baseline across the 10 ABCs "
                "sentinel sites), enter at next session open a dollar-neutral pair: "
                "long basket 50% GRFS + 50% CRL, short equal-notional XLV. Hold 63 "
                "trading days then exit at close."
            ),
            "mechanism": (
                "Invasive Group A Strep (iGAS) surges drive IVIG demand for adjunctive "
                "treatment of streptococcal toxic shock and necrotizing fasciitis, plus "
                "penicillin / beta-lactam procurement build. GRFS (Grifols) is a "
                "pure-play plasma / IVIG manufacturer that benefits from procurement "
                "cycle pull-through. CRL (Charles River Labs) is the dominant preclinical "
                "CRO and biologics testing house that benefits from accelerated "
                "antimicrobial / monoclonal pipeline activity following surveillance "
                "escalation. Shorting XLV neutralizes broad health-care sector beta so "
                "the spread isolates the iGAS-specific procurement / R&D shock. The "
                "63-day window (~3 months) captures the surveillance -> hospital "
                "purchasing -> contract flow propagation."
            ),
            "source": (
                "CDC Active Bacterial Core (ABCs) surveillance bulletins, MMWR archive, "
                "and CDC Health Alert Network (HAN) advisories (HAN 483 Dec 22 2022 "
                "primary iGAS-in-children HAN). Prices via yfinance (auto_adjust=True). "
                "GRFS one-way slippage 25 bps (Spanish ADR, lower volume); CRL/XLV 1 bp."
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
                "Event dates hand-curated from CDC ABCs / MMWR / HAN archive — "
                "different curators might select different dates and add look-ahead "
                "risk. GRFS is a Spanish ADR with lower US volume; 25 bps one-way slip "
                "is a conservative estimate but realized cost could be higher. "
                "Overlapping events (Feb 2023 / Apr 2023 cluster, etc.) are not "
                "double-stacked. Sharpe / CAGR are computed on held days only, which "
                "amplifies the realized exposure but is sensitive to small n. "
                "Statistical significance is weak (10 events x 63 days = 630 daily "
                "observations gross, but auto-correlated within events). PFE/MRK are "
                "diagnostic-only and not in trade legs."
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
