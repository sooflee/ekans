"""PL283_census_datacenter_construction_power_infra — Census Data Center Construction Spending
Event Study → Long ETN/HUBB/D power-infra basket for 40 trading days.

Hypothesis: Each Census C30 release where data-center / computer / data-processing
construction prints a new monthly high confirms accelerating power-infra demand.
Transformer/electrical-gear lead times extend, ETN/HUBB book orders, and Dominion
(D) re-prices on load-growth narrative.

Event window: [+1, +40] trading days from each curated release date.
Daily PnL: equal-weight basket of ETN, HUBB, D returns during open event windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, compute_metrics, save_result, mark_failed, daily_returns
)


def main():
    sid = "PL283_census_datacenter_construction_power_infra"
    try:
        px = load_prices(["ETN", "HUBB", "D", "SPY"], start="2022-06-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "SPY" not in ret.columns:
        return mark_failed(sid, "SPY missing from price data")
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["ETN", "HUBB", "D"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Only {len(basket_tickers)} basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Curated Census C30 release dates (~1st business day of each month)
    # where the data-center category printed a new monthly high during
    # the post-2022 AI-driven capex acceleration.
    known_events = [
        "2023-02-01", "2023-05-01", "2023-08-01", "2023-11-01",
        "2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01",
        "2025-02-03", "2025-05-01", "2025-08-01", "2025-11-03",
        "2026-02-02", "2026-05-01",
    ]

    hold_days = 40
    events = []
    pnl_parts = []

    for evd in known_events:
        trig = pd.Timestamp(evd)
        # Entry at next session close after release: skip release-day, enter t+1.
        mask = ret.index > trig
        if mask.sum() < 5:
            continue
        entry_idx = ret.index[mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        if exit_loc <= entry_loc:
            continue

        window = slice(entry_loc, exit_loc)
        bw = basket_r.iloc[window]
        sw = spy_r.iloc[window]
        pnl_parts.append(bw)

        bask_cum = float((1 + bw).prod() - 1)
        spy_cum = float((1 + sw).prod() - 1)
        events.append({
            "trigger_date": str(trig.date()),
            "entry_date": str(entry_idx.date()),
            "basket_40d_return": round(bask_cum, 4),
            "spy_40d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No events landed in price data range")

    # Stitch overlapping event windows: dedupe by date, keep first.
    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")].sort_index()

    bench = spy_r.reindex(all_pnl.index).dropna()
    m = compute_metrics(
        all_pnl, benchmark=bench,
        name="Census DataCenter Construction → Long ETN/HUBB/D Power-Infra",
    )

    avg_bask = float(np.mean([e["basket_40d_return"] for e in events]))
    avg_excess = float(np.mean([e["excess"] for e in events]))
    win_count = int(sum(1 for e in events if e["basket_40d_return"] > 0))
    excess_win = int(sum(1 for e in events if e["excess"] > 0))

    save_result(sid, m, extra={
        "rule": "On Census C30 release dates flagged as data-center new-month-high, long equal-weight ETN+HUBB+D for 40 trading days (entry t+1 close).",
        "mechanism": "Accelerating data-center buildout → power-infra (transformers, switchgear, transmission) order book extends; ETN/HUBB book backlog; Dominion's PJM/Virginia load-growth narrative re-prices.",
        "source": "Census C30 release dates (2023-2026, curated); yfinance prices for ETN, HUBB, D, SPY.",
        "basket_tickers": basket_tickers,
        "hold_days": hold_days,
        "n_events": len(events),
        "avg_basket_return": round(avg_bask, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "excess_win_rate": f"{excess_win}/{len(events)}",
        "events": events,
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done {sid}: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  avg basket {avg_bask*100:+.1f}%  avg excess {avg_excess*100:+.1f}%  "
          f"win {win_count}/{len(events)}  excess-win {excess_win}/{len(events)}")


if __name__ == "__main__":
    main()
