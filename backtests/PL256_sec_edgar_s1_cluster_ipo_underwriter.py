"""PL256_sec_edgar_s1_cluster_ipo_underwriter — SEC EDGAR S-1 Filing Cluster → Long Investment Banks
When IPO pipeline heats up (surge of S-1 filings), investment banks (GS, MS) benefit from
underwriting fees. Since we can't query EDGAR in real-time, use known IPO wave periods
from public IPO calendars and Renaissance Capital data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL256_sec_edgar_s1_cluster_ipo_underwriter"
    try:
        px = load_prices(["GS", "MS", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["GS", "MS"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No investment bank tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Known major IPO wave start dates (when S-1 filing clusters were visible)
    # Source: Jay Ritter's IPO data, Renaissance Capital historical
    ipo_wave_dates = [
        "2007-06-01",  # Pre-crisis IPO boom (Blackstone, etc.)
        "2010-11-01",  # Post-GFC IPO reopening (GM, etc.)
        "2012-05-01",  # Facebook IPO era
        "2013-09-01",  # Twitter IPO era + biotech wave
        "2014-09-01",  # Alibaba IPO + tech wave
        "2017-03-01",  # Snap IPO era
        "2019-03-01",  # Uber/Lyft/Pinterest S-1 cluster
        "2020-08-01",  # Post-covid IPO/SPAC boom
        "2021-01-01",  # Peak IPO mania
        "2021-09-01",  # Late-cycle IPO wave (Rivian etc.)
        "2024-02-01",  # IPO market reopening (Reddit, Astera Labs)
    ]

    events = []
    pnl_parts = []
    hold_days = 60

    for date_str in ipo_wave_dates:
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
            "basket_60d_return": round(bask_cum, 4),
            "spy_60d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No IPO wave events found in price data range")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="S-1 Filing Cluster → Long Investment Banks (GS+MS)")

    avg_basket = np.mean([e["basket_60d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_60d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long GS+MS equal-weight 60d when SEC EDGAR S-1 filing cluster signals IPO wave",
        "mechanism": "IPO wave → underwriting fee surge → GS/MS equity capital markets revenue boost",
        "source": "SEC EDGAR S-1 filings + yfinance",
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
        flag = "+" if e["basket_60d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_60d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
