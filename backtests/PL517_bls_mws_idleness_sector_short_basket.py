"""PL517_bls_mws_idleness_sector_short_basket
BLS Major Work Stoppages Monthly Idleness Surge -> Short Affected-NAICS
Equity Basket vs Long SPY (35 Trading Days).

Event-study short-basket vs SPY-hedge trade. For each curated BLS MWS
release date where a single 3-digit NAICS subsector's days-idle exceeded
the trailing 24-month 95th percentile, enter at the next regular session
open: short an equal-weight basket of the 3 largest publicly traded
operators mapped to the affected NAICS subsector at 100% gross, plus long
SPY at 100% as beta hedge. Hold 35 trading days then exit at the close.

Daily strategy PnL = -mean(basket_rets) + 1.0 * SPY_ret  on event days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated BLS Major Work Stoppages events
KNOWN_EVENTS = [
    {
        "date": "2019-10-17",
        "naics": "3361",
        "basket": ["GM", "F", "STLA"],
        "note": "BLS MWS Oct-2019 release: Sep days-idle surge from GM/UAW strike (Sep 16 - Oct 25, 2019)",
    },
    {
        "date": "2023-08-17",
        "naics": "5121",
        "basket": ["NFLX", "WBD", "DIS"],
        "note": "BLS MWS Aug-2023 release: Jul days-idle from SAG-AFTRA strike start (Jul 14, 2023) overlapping WGA",
    },
    {
        "date": "2023-11-17",
        "naics": "3361",
        "basket": ["GM", "F", "STLA"],
        "note": "BLS MWS Nov-2023 release: Oct days-idle peak from UAW Big-3 strike (Sep 15 - Oct 30, 2023)",
    },
]

ALL_TICKERS = sorted({t for e in KNOWN_EVENTS for t in e["basket"]} | {"SPY"})
HOLD_DAYS = 35


def run_event_study(events, ret, spy_r, hold_days=HOLD_DAYS):
    """Run the event-study trade.
    Strategy daily return on held days = -mean(basket_rets) + SPY_ret.
    Returns (pnl, positions, event_log).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in events:
        rel = pd.Timestamp(ev["date"])
        future = idx[idx > rel]
        if len(future) == 0:
            log = dict(ev)
            log["status"] = "no_data_after_release"
            event_log.append(log)
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        basket = ev["basket"]
        missing = [t for t in basket if t not in ret.columns]
        if missing:
            log = dict(ev)
            log["status"] = f"missing tickers: {missing}"
            event_log.append(log)
            continue

        basket_ret = ret[basket].mean(axis=1)
        # Strategy daily return = -basket_ret + spy_r
        slice_basket = basket_ret.iloc[entry_pos:exit_pos]
        slice_spy = spy_r.reindex(slice_basket.index)
        strat_slice = -slice_basket + slice_spy

        # Cumulative event returns
        basket_cum = float((1 + slice_basket).prod() - 1)
        spy_cum = float((1 + slice_spy).prod() - 1)
        strat_cum = float((1 + strat_slice).prod() - 1)

        log = dict(ev)
        log["entry_date"] = str(entry_dt.date())
        log["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        log["n_hold_days"] = int(exit_pos - entry_pos)
        log["basket_return"] = round(basket_cum, 4)
        log["spy_return"] = round(spy_cum, 4)
        log["strategy_return"] = round(strat_cum, 4)
        event_log.append(log)

        for j in range(entry_pos, exit_pos):
            # don't double count overlapping events (none expected at hold=35d)
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = strat_slice.iloc[j - entry_pos]

    return pnl, positions, event_log


def main():
    sid = "PL517_bls_mws_idleness_sector_short_basket"

    try:
        px = load_prices(ALL_TICKERS, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ALL_TICKERS if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(KNOWN_EVENTS, ret, spy_r,
                                                hold_days=HOLD_DAYS)

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days (n_events={len(KNOWN_EVENTS)}, held_days={len(held_pnl)})",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="BLS MWS Idleness Surge Short-NAICS-Basket vs Long SPY (35d hold)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Per-event summary
    event_returns = [e["strategy_return"] for e in event_log if "strategy_return" in e]
    excess_returns = [
        e["strategy_return"] for e in event_log if "strategy_return" in e
    ]
    if event_returns:
        summary = {
            "n_events": len(event_returns),
            "avg_strategy_return": round(float(np.mean(event_returns)), 4),
            "median_strategy_return": round(float(np.median(event_returns)), 4),
            "win_rate": round(float(np.mean([r > 0 for r in event_returns])), 4),
            "best": round(float(np.max(event_returns)), 4),
            "worst": round(float(np.min(event_returns)), 4),
        }
    else:
        summary = None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each curated BLS Major Work Stoppages monthly release date "
                "where a 3-digit NAICS subsector's days-idle exceeded its trailing "
                "24-month 95th percentile, enter at the next regular session open: "
                "short an equal-weight basket of the 3 largest publicly traded "
                "operators mapped to the affected NAICS subsector at 100% gross, "
                "plus long SPY at 100% as beta hedge. Hold 35 trading days; exit "
                "at the close. Daily strategy PnL = -mean(basket_rets) + SPY_ret."
            ),
            "mechanism": (
                "A major strike removes a sustained slug of operating capacity at "
                "directly-affected operators; the BLS MWS monthly release is the "
                "first formal national tally of days-idle and tends to crystallize "
                "sell-side EPS cuts and guidance revisions over the following 3-8 "
                "weeks. Going short the directly-named NAICS basket while long SPY "
                "isolates sector-alpha from broad-market beta. Historical anchors: "
                "2019 GM strike (~$3.6B EBIT hit), 2023 SAG-AFTRA/WGA "
                "(NFLX/DIS/WBD guidance cuts), 2023 UAW Big-3 (F/STLA guidance cuts)."
            ),
            "source": (
                "BLS Major Work Stoppages monthly release historical archive 2019-2024; "
                "NAICS-to-basket map hard-coded per implementation_notes. Prices via "
                "yfinance (auto_adjust=True)."
            ),
            "tickers": ALL_TICKERS,
            "hold_days": HOLD_DAYS,
            "events": event_log,
            "summary": summary,
            "caveats": (
                "Small-sample event study (N=3 curated events 2019-2023). Statistical "
                "significance is extremely limited; Sharpe/t-stat figures are sensitive "
                "to single events and benefit from bootstrap. STLA traded as FCAU "
                "prior to Jan 2021 but yfinance returns continuous merged history. "
                "SPY 1:1 long is a dollar-beta hedge, not a CAPM-beta hedge - if basket "
                "beta != 1.0 the residual exposure is non-zero. Event dates were "
                "hand-curated from major historical strike clusters; out-of-sample "
                "performance on future BLS MWS surge releases is unknown."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events_held: {sum(1 for e in event_log if 'strategy_return' in e)}")
    print(f"  summary: {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
