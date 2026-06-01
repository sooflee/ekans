"""PL772_ukraine_recon_uri_mtz -- EU Frozen-Asset Transfer + USASpending Ukraine Surge
Long URI/MTZ reconstruction pair vs short XLI on major Ukraine aid authorizations.

Event study: 4 known aid-authorization trigger dates (2022-2025).
Long equal-weight URI+MTZ, short XLI, hold up to 60 trading days.
Exit on 12% drawdown vs XLI, 20% outperformance, or 60 days elapsed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known Ukraine aid trigger events with their authorization dates.
# Trigger = date when major aid package was signed/activated.
UKRAINE_EVENTS = [
    {
        "date": "2022-04-28",
        "note": "US Ukraine Democracy Defense Lend-Lease Act + $40B supplemental aid signed",
    },
    {
        "date": "2022-09-30",
        "note": "EU Council Decision 2022/1784: $19B macro-financial assistance to Ukraine",
    },
    {
        "date": "2024-07-01",
        "note": "EU Reg 2024/576 windfall profits mechanism activated, $3B tranche to Ukraine",
    },
    {
        "date": "2025-01-22",
        "note": "US Ukraine Supplemental $61B final disbursement milestone",
    },
]


def main():
    sid = "PL772_ukraine_recon_uri_mtz"

    # Load prices
    try:
        px = load_prices(["URI", "MTZ", "XLI", "SPY"], start="2020-01-01")
    except Exception as e:
        try:
            px = load_prices(["URI", "MTZ", "XLI", "SPY"], start="2020-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    if px is None or px.empty:
        return mark_failed(sid, "price data empty")

    ret = daily_returns(px)

    long_tickers = [t for t in ["URI", "MTZ"] if t in ret.columns]
    if not long_tickers:
        return mark_failed(sid, "neither URI nor MTZ in price data")
    if "XLI" not in ret.columns:
        return mark_failed(sid, "XLI not in price data")

    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    hold_days = 60   # maximum hold 60 trading days (12 weeks ~= 3 months)
    stop_loss_pct = -0.12   # exit if basket underperforms XLI by >12%
    take_profit_pct = 0.20  # exit if basket outperforms XLI by >20%

    pnl_parts = []
    event_results = []

    for event in UKRAINE_EVENTS:
        trigger_date = pd.Timestamp(event["date"])

        # Find entry: first trading day on or after trigger date
        entry_mask = ret.index >= trigger_date
        if entry_mask.sum() < 10:
            print(f"  Skipping {trigger_date.date()}: insufficient price data ahead")
            continue

        entry_loc = ret.index.get_loc(ret.index[entry_mask][0])
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        if exit_loc - entry_loc < 5:
            print(f"  Skipping {trigger_date.date()}: window too short")
            continue

        # Build daily PnL for the window with stop-loss/take-profit check
        basket_pnl = []
        hedge_pnl = []
        spread_pnl = []
        actual_exit_loc = exit_loc
        cum_spread = 0.0

        for i in range(entry_loc, exit_loc):
            long_r = ret[long_tickers].iloc[i].mean()
            short_r = ret["XLI"].iloc[i]
            spread_day = long_r - short_r
            cum_spread = (1 + cum_spread) * (1 + spread_day) - 1
            basket_pnl.append(long_r)
            hedge_pnl.append(short_r)
            spread_pnl.append(spread_day)

            # Check stop-loss / take-profit
            if cum_spread <= stop_loss_pct:
                print(f"    {trigger_date.date()}: stop-loss at day {i - entry_loc + 1} (spread={cum_spread*100:.1f}%)")
                actual_exit_loc = i + 1
                break
            if cum_spread >= take_profit_pct:
                print(f"    {trigger_date.date()}: take-profit at day {i - entry_loc + 1} (spread={cum_spread*100:.1f}%)")
                actual_exit_loc = i + 1
                break

        if not spread_pnl:
            continue

        spread_series = pd.Series(
            spread_pnl,
            index=ret.index[entry_loc:entry_loc + len(spread_pnl)]
        )
        pnl_parts.append(spread_series)

        basket_cum = float((1 + pd.Series(basket_pnl)).prod() - 1)
        xli_cum = float((1 + pd.Series(hedge_pnl)).prod() - 1)
        spread_cum = basket_cum - xli_cum

        spy_cum = None
        if spy_r is not None:
            spy_window = spy_r.iloc[entry_loc:actual_exit_loc]
            spy_cum = float((1 + spy_window).prod() - 1)

        event_results.append({
            "trigger_date": str(trigger_date.date()),
            "entry_date": str(ret.index[entry_loc].date()),
            "exit_date": str(ret.index[entry_loc + len(spread_pnl) - 1].date()),
            "hold_days": len(spread_pnl),
            "note": event["note"],
            "basket_return": round(basket_cum, 4),
            "xli_return": round(xli_cum, 4),
            "spread_return": round(spread_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    if not event_results:
        return mark_failed(sid, "no valid Ukraine aid events found in price data")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")].dropna()

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(all_pnl)})")

    bench = spy_r.reindex(all_pnl.index).dropna() if spy_r is not None else None
    m = compute_metrics(all_pnl, benchmark=bench,
                        name="Ukraine Reconstruction: Long URI+MTZ / Short XLI")

    spreads = [e["spread_return"] for e in event_results]
    win_rate = sum(1 for s in spreads if s > 0) / len(spreads)

    save_result(sid, m, extra={
        "rule": "Long URI+MTZ, short XLI on major Ukraine aid authorization dates; hold 60d or until stop/profit target",
        "mechanism": "Large Ukraine infrastructure aid packages signal medium-term demand for specialty construction equipment (URI) and infrastructure contractor services (MTZ), outperforming broad industrials (XLI) due to reconstruction-specific revenue exposure",
        "source": "Known Ukraine aid events (USASpending.gov + EUR-Lex); yfinance URI/MTZ/XLI/SPY prices",
        "n_events": len(event_results),
        "avg_spread_return": round(float(np.mean(spreads)), 4),
        "win_rate": round(win_rate, 3),
        "events": event_results,
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(event_results)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg spread: {np.mean(spreads)*100:.1f}%  Win rate: {win_rate*100:.0f}%")
    for e in event_results:
        flag = "+" if e["spread_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: spread {e['spread_return']*100:+.1f}%  "
              f"basket={e['basket_return']*100:+.1f}%  XLI={e['xli_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
