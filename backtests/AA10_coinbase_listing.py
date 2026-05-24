"""
AA10 Coinbase listing pop.
Coinbase blog announces upcoming listings (roadmap blog) typically ~24h
before the asset becomes tradeable. Hypothesis: announcement → 24h drift up
as users front-run liquidity.

Curated listings of tokens that have yfinance-USD tickers (BTC, ETH, LTC,
BCH, SOL, AVAX, DOGE, MATIC, LINK, XLM, ADA, DOT, ATOM, ALGO, FIL, UNI, AAVE,
COMP, MKR, BAT, ZEC, XTZ, EOS, ETC, REP, SHIB).

Note: Coinbase already had BTC/ETH/LTC at launch — those are NOT listings.
This list is *roadmap* / *new add* announcements only for the named tokens.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


# (announcement date, yfinance ticker). Curated from Coinbase blog.
# For names already on Coinbase Pro pre-2020, we use the Coinbase consumer
# (retail) add date. Only tokens with reliable yfinance USD pairs included.
COINBASE_LISTINGS = [
    # 2018
    ("2018-07-13", "ETC-USD"),  # ETC on Coinbase
    ("2018-12-07", "BAT-USD"),
    ("2018-12-17", "ZRX-USD"),
    # 2019
    ("2019-02-26", "BAT-USD"),  # general listing
    ("2019-06-26", "XTZ-USD"),
    ("2019-08-08", "LINK-USD"),
    ("2019-12-13", "XLM-USD"),
    # 2020
    ("2020-06-10", "OXT-USD"),
    ("2020-06-18", "COMP-USD"),
    ("2020-09-23", "BNT-USD"),
    ("2020-10-22", "FIL-USD"),
    ("2020-11-25", "MKR-USD"),
    # 2021
    ("2021-04-29", "ICP-USD"),
    ("2021-06-08", "DOGE-USD"),
    ("2021-07-08", "SHIB-USD"),
    ("2021-09-13", "AVAX-USD"),
    ("2021-10-08", "SOL-USD"),
    ("2021-11-09", "ATOM-USD"),
    # 2022
    ("2022-03-17", "APE-USD"),
    ("2022-04-21", "GMT-USD"),
    ("2022-09-22", "ALGO-USD"),
    # 2023
    ("2023-03-15", "ARB-USD"),
    ("2023-07-13", "OP-USD"),
    # 2024
    ("2024-02-21", "JTO-USD"),
    ("2024-04-30", "WIF-USD"),
    ("2024-09-25", "POPCAT-USD"),
    # 2025
    ("2025-03-12", "PYTH-USD"),
]


def main():
    # Pull all unique tickers
    tickers = sorted({t for _, t in COINBASE_LISTINGS})
    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed("AA10_coinbase_listing", f"data load failed: {e}")

    if px.empty:
        return mark_failed("AA10_coinbase_listing", "no token data loaded")

    # For each (date, ticker), compute t+0 close → t+1 close return.
    events = []
    n_tested = 0
    n_missing = 0
    for d_str, tk in COINBASE_LISTINGS:
        if tk not in px.columns:
            n_missing += 1
            continue
        s = px[tk].dropna()
        if s.empty:
            n_missing += 1
            continue
        d = pd.Timestamp(d_str)
        # Find first trading day >= announcement
        if d < s.index[0] or d > s.index[-1]:
            n_missing += 1
            continue
        i = s.index.searchsorted(d, side="left")
        if i + 1 >= len(s):
            n_missing += 1
            continue
        entry = s.iloc[i]
        exit_ = s.iloc[i + 1]
        r = exit_ / entry - 1
        events.append({"date": s.index[i], "ticker": tk, "ret": float(r)})
        n_tested += 1

    if n_tested < 10:
        return mark_failed("AA10_coinbase_listing",
                           f"too few testable events ({n_tested}, missing={n_missing})")

    ev_df = pd.DataFrame(events).set_index("date").sort_index()
    # Daily PnL series with one event per day (or sum if multiple).
    pnl = ev_df.groupby(ev_df.index)["ret"].sum()
    # For metrics, treat each event-day as one day of returns.
    # Avg per-event return:
    avg_ret = float(ev_df["ret"].mean())
    hit = float((ev_df["ret"] > 0).mean())
    # t-stat across events:
    se = ev_df["ret"].std() / np.sqrt(len(ev_df))
    t_stat = avg_ret / se if se > 0 else 0.0

    # Compose a sparse pnl series indexed by trading days (zeros on non-event days)
    # for compute_metrics. Use BTC-USD calendar as the trading calendar (24/7 crypto).
    cal = px.index
    daily = pd.Series(0.0, index=cal)
    for t, r in zip(ev_df.index, ev_df["ret"].values):
        if t in daily.index:
            daily.loc[t] += r
        else:
            # nearest
            i = cal.searchsorted(t, side="left")
            if 0 <= i < len(cal):
                daily.iloc[i] += r
    m = compute_metrics(daily, name="AA10 Coinbase listing 24h")
    # Override hit_rate / t_stat with per-event stats (more meaningful here)
    m["hit_rate"] = hit
    m["t_stat"] = float(t_stat)
    m["n_events"] = int(n_tested)
    m["avg_event_ret"] = avg_ret
    print_metrics(m)
    save_result("AA10_coinbase_listing", m, extra={
        "status": "ok",
        "rule": "Long token at announcement close, exit T+1 close (24h hold).",
        "universe": "Coinbase listings with yfinance USD pairs",
        "source": "Coinbase blog roadmap announcements (curated)",
        "n_events": n_tested,
        "n_missing_data": n_missing,
        "avg_event_ret_pct": avg_ret * 100,
    })


if __name__ == "__main__":
    main()
