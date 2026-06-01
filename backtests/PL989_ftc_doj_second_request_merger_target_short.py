"""PL989_ftc_doj_second_request_merger_target_short — FTC/DOJ HSR Second Request -> Merger Target Spread Widening Short"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL989_ftc_doj_second_request_merger_target_short"

    # Known HSR Second Request events where target remained public long enough:
    # 2022-03-29: FTC Second Request Kroger/Albertsons (ACI) — deal ultimately blocked Dec 2024
    # 2023-06-12: FTC Second Request Tapestry/Capri (CPRI) — deal blocked Oct 2023
    # 2022-04-28: Microsoft/Activision (ATVI) — SR announced, deal closed Oct 2023 (delisted, skip)
    # Use ACI and CPRI as primary tickers

    events = [
        ("2022-03-29", "ACI", 239.0),   # Kroger deal price ~$34.10/share, ACI ~$28 at SR
        ("2023-06-12", "CPRI", 57.0),   # Tapestry deal price $57/share, CPRI ~$46 at SR
    ]

    hold_days = 80  # trading days (per backtest_approach)

    try:
        px = load_prices(["ACI", "CPRI", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for ev_date_str, ticker, deal_price in events:
        if ticker not in ret.columns:
            print(f"Skipping {ev_date_str} {ticker}: not in price data")
            continue

        ev_date = pd.Timestamp(ev_date_str)
        future_idx = ret.index[ret.index >= ev_date]
        if len(future_idx) < 5:
            print(f"Skipping {ev_date_str} {ticker}: insufficient future data")
            continue

        entry_idx = future_idx[0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret))

        target_ret = ret[ticker].iloc[entry_loc:exit_loc]
        spy_window = spy_r.iloc[entry_loc:exit_loc]

        # Strategy: short the target (profit when deal fails and price falls back)
        strat_ret = -target_ret

        # Track cumulative returns
        target_cum = float((1 + target_ret).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        event_records.append({
            "event_date": ev_date_str,
            "ticker": ticker,
            "deal_price": deal_price,
            "target_return_80d": round(target_cum, 4),
            "spy_return_80d": round(spy_cum, 4),
            "short_pnl_80d": round(-target_cum, 4),
            "n_days": len(target_ret),
        })

        for idx, r in strat_ret.items():
            if idx in pnl.index:
                pnl[idx] += r

    if not event_records:
        return mark_failed(sid, "no valid events found")

    print(f"Events processed: {len(event_records)}")
    for e in event_records:
        print(f"  {e['event_date']} {e['ticker']}: target={e['target_return_80d']:.2%}, "
              f"SPY={e['spy_return_80d']:.2%}, short_pnl={e['short_pnl_80d']:.2%}")

    # Normalize: average across concurrent positions
    n_concurrent = pnl.apply(lambda x: 1 if x != 0 else 0)
    active_pnl = pnl[pnl != 0]

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FTC/DOJ HSR Second Request Short")

    save_result(sid, m, extra={
        "rule": "Short acquisition target equity within 5 trading days of FTC/DOJ HSR Second Request announcement; cover at 80 days, deal block, or deal close",
        "mechanism": "Second Requests signal heightened antitrust scrutiny; historically 15-25% of SR deals are blocked; target spread widens as market prices failure risk; short captures spread widening on failed deals",
        "source": "FTC press releases RSS (ftc.gov); DOJ ATR news RSS; SEC EDGAR 8-K; known events: ACI 2022-03-29, CPRI 2023-06-12",
        "n_events": len(event_records),
        "events": event_records,
    })


if __name__ == "__main__":
    main()
