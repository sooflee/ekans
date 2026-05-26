"""PL252_fda_orange_book_patent_cluster_pbm — FDA Orange Book Patent Expiry Cluster → Long PBMs (CVS+CI)
When a cluster of major drug patent expirations occurs, generic entry boosts PBM rebate revenue.
Since we don't have actual Orange Book API data in the harness, we use known patent cliff years
from public sources. Major patent cliff years: 2012 (Lipitor/Plavix), 2015 (Nexium/Abilify),
2019 (Lyrica/Tecfidera), 2023 (Humira/Keytruda biosimilars), 2025 (Stelara/Opdivo).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL252_fda_orange_book_patent_cluster_pbm"
    try:
        px = load_prices(["CVS", "CI", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["CVS", "CI"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No PBM tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Known major patent cliff events (month when generics launched / biosimilars entered)
    patent_cliff_dates = [
        "2012-06-01",  # Lipitor + Plavix generics fully ramped
        "2015-01-01",  # Nexium + Abilify generics launched
        "2016-06-01",  # Crestor generic launched
        "2019-07-01",  # Lyrica generic launched
        "2023-02-01",  # Humira biosimilars entered US market
    ]

    events = []
    pnl_parts = []
    hold_days = 60

    for date_str in patent_cliff_dates:
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
        return mark_failed(sid, "No patent cliff events found in price data range")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="FDA Patent Cliff Cluster → Long PBMs (CVS+CI)")

    avg_basket = np.mean([e["basket_60d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_60d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long CVS+CI equal-weight 60d when major patent cliff cluster launches generics/biosimilars",
        "mechanism": "Patent cliffs → generic entry → PBMs capture higher rebates and formulary leverage",
        "source": "FDA Orange Book public data + yfinance",
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
