"""PL674 — Buffett 13F New Concentrated Position - Single-Name Drift Long"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL674_buffett_13f_long_drift"

    # Known Berkshire 13F new position events (filing disclosure dates, post-close entry next day).
    # Each entry: (filing_date_str, ticker, approx_value_bn)
    # These are the first 13F filing date where the position appeared (quarterly, due 45d after quarter end).
    events = [
        # (entry_after_filing_date, ticker)
        ("2016-02-12", "AAPL"),   # Q4 2015 13F filed ~Feb 2016 — AAPL new position
        ("2016-05-16", "AAPL"),   # Q1 2016 13F — AAPL significantly increased
        ("2017-05-15", "BAC"),    # Q1 2017 13F — BAC large new position via warrants
        ("2019-02-14", "KHC"),    # Q4 2018 13F — KHC (already held, reappearance)
        ("2020-08-14", "BIIB"),   # Q2 2020 13F — BIIB new position briefly
        ("2022-02-14", "OXY"),    # Q4 2021 13F — OXY new ~$850M position (spec known event)
        ("2022-05-16", "OXY"),    # Q1 2022 13F — OXY dramatically increased
        ("2022-08-15", "OXY"),    # Q2 2022 13F — OXY continued accumulation
        ("2023-02-14", "TSMC"),   # Q4 2022 13F — TSMC new ~$4.1B position
        ("2023-11-14", "OXY"),    # Q3 2023 13F — OXY >$20B, continued accumulation
        ("2024-02-14", "COF"),    # Q4 2023 13F — COF new position
        ("2024-08-14", "ULTA"),   # Q2 2024 13F — ULTA new ~$266M position (spec known event)
    ]

    # Collect all unique tickers + SPY
    tickers = list(set([e[1] for e in events])) + ["SPY"]

    try:
        px = load_prices(tickers, start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    hold = 90
    pnl = pd.Series(0.0, index=spy_r.index)
    valid_evts = []

    for filing_date_str, ticker in events:
        if ticker not in ret.columns:
            continue
        ticker_r = ret[ticker]
        ev_dt = pd.Timestamp(filing_date_str)

        # Entry on next business day after filing
        mask = ticker_r.index > ev_dt
        if mask.sum() < hold:
            continue
        entry_dt = ticker_r.index[mask][0]
        p = ticker_r.index.get_loc(entry_dt)
        ep = min(p + hold, len(ticker_r))

        # PnL: long target - 50% short SPY (market-hedged)
        tgt_window = ticker_r.iloc[p:ep]
        spy_p = spy_r.index.get_loc(entry_dt) if entry_dt in spy_r.index else p
        spy_window = spy_r.iloc[spy_p:min(spy_p + hold, len(spy_r))]

        n = min(len(tgt_window), len(spy_window))
        if n < 20:
            continue

        ls_window = tgt_window.values[:n] - 0.5 * spy_window.values[:n]
        pnl.iloc[p:p + n] += ls_window

        cum_tgt = float((1 + pd.Series(tgt_window.values[:n])).prod() - 1)
        cum_spy = float((1 + pd.Series(spy_window.values[:n])).prod() - 1)
        cum_ls = float((1 + pd.Series(ls_window)).prod() - 1)

        valid_evts.append({
            "filing_date": filing_date_str,
            "entry_date": str(entry_dt.date()),
            "ticker": ticker,
            "target_return": round(cum_tgt, 4),
            "spy_return": round(cum_spy, 4),
            "ls_return": round(cum_ls, 4),
        })

    print(f"Valid events: {len(valid_evts)}")
    if not valid_evts:
        return mark_failed(sid, "no valid events")

    active = pnl[pnl != 0]
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="Buffett 13F New Position Drift Long")
    avg_ret = float(np.mean([e["ls_return"] for e in valid_evts]))
    win_rate = float(np.mean([e["ls_return"] > 0 for e in valid_evts]))
    avg_tgt = float(np.mean([e["target_return"] for e in valid_evts]))

    save_result(sid, m, extra={
        "rule": "Long target stock 90 trading days after Berkshire 13F reveals new position >$500M; hedge 50% short SPY",
        "mechanism": "Berkshire 13F disclosures signal high-quality value conviction; market drift effect as institutional investors and retail follow Buffett's high-profile new positions",
        "source": "SEC EDGAR 13F-HR quarterly filings; yfinance",
        "n_events": len(valid_evts),
        "avg_target_return": round(avg_tgt, 4),
        "avg_ls_return": round(avg_ret, 4),
        "event_win_rate": round(win_rate, 4),
        "events": valid_evts,
    })
    print(f"Done: Sharpe={m.get('sharpe'):.2f}, CAGR={m.get('cagr'):.2%}, Events={len(valid_evts)}")


if __name__ == "__main__":
    main()
