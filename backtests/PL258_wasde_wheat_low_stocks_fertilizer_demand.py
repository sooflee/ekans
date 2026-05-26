"""PL258_wasde_wheat_low_stocks_fertilizer_demand — WASDE Global Wheat Stocks-to-Use < 30% → Long Fertilizers
When USDA WASDE report shows global wheat stocks-to-use ratio below 30%, tight supply
incentivizes maximum-yield planting → fertilizer demand surges. Long NTR+MOS+CF for 42 trading days.
Uses hand-coded WASDE report dates when global wheat S/U was low.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL258_wasde_wheat_low_stocks_fertilizer_demand"
    try:
        px = load_prices(["NTR", "MOS", "CF", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["NTR", "MOS", "CF"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough fertilizer tickers available: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    # WASDE report dates when global wheat stocks-to-use was <30%
    # Source: USDA WASDE historical reports (public, free)
    # Global wheat S/U was notably tight in: 2007-08, 2010-11, 2012-13, 2021-22, 2022-23
    wasde_low_stocks_dates = [
        "2007-11-09",  # Nov 2007 WASDE — wheat S/U ~23% (historic low)
        "2008-02-08",  # Feb 2008 WASDE — continued tight stocks
        "2010-08-12",  # Aug 2010 WASDE — Russian drought → wheat spike
        "2011-01-12",  # Jan 2011 WASDE — global wheat stocks tight
        "2012-07-11",  # Jul 2012 WASDE — US drought → wheat S/U sub-30%
        "2013-01-11",  # Jan 2013 WASDE — continued tight global wheat
        "2021-05-12",  # May 2021 WASDE — post-covid supply concerns
        "2022-03-09",  # Mar 2022 WASDE — Russia-Ukraine war → wheat crisis
        "2022-06-10",  # Jun 2022 WASDE — continued extreme tightness
        "2023-01-12",  # Jan 2023 WASDE — still below 30%
    ]

    events = []
    pnl_parts = []
    hold_days = 42

    for date_str in wasde_low_stocks_dates:
        trig_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": date_str,
            "basket_42d_return": round(bask_cum, 4),
            "spy_42d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No WASDE low-wheat-stocks events in price data range")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="WASDE Wheat Low S/U → Long Fertilizers")

    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long NTR+MOS+CF equal-weight 42d when WASDE global wheat stocks-to-use < 30%",
        "mechanism": "Low global wheat stocks → incentive for max-yield planting → fertilizer demand and pricing surge",
        "source": "USDA WASDE + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
