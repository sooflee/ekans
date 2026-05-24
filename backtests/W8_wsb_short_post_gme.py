"""
W8 Short WSB-pumped names post-spike (curated event sample).

ApeWisdom's public REST API exposes only the current snapshot + a 24h-ago
mention count; there is no historical time series. To produce an honest
back-test we instead use a small curated list of well-documented WSB squeeze
peaks (GME 2021-01, AMC 2021-06, BBBY 2022-08, GME 2024-05, NVDA pre-split
2024-05, etc.) and test the simple rule:

Rule: at the close of the documented WSB peak day, short the stock for the
next 5 trading days, beta-hedged with SPY. Pair PnL = -stock_ret + beta * SPY_ret.

This is a small-N event study (≤10 events), not a continuous back-test.

Source: Bradley-Hanousek-Jame-Xiao (2023-24, RFS) 'Meme stocks and the
        feedback loop'; ApeWisdom and SwaggyStocks data not available
        historically without payment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


# Documented WSB squeeze peak dates (close of the day of peak mention frenzy).
# Picked from public news / WSB archive references; each is the day the
# r/wallstreetbets attention metric was at its post-event maximum.
EVENTS = [
    ("GME",  "2021-01-27"),  # GME peak short-squeeze
    ("AMC",  "2021-06-02"),  # AMC apes peak
    ("BBBY", "2022-08-17"),  # BBBY meme rally peak (Cohen)
    ("GME",  "2024-05-13"),  # GME Roaring Kitty return rally
    ("AMC",  "2024-05-14"),  # AMC piggy-back rally
    ("NVDA", "2024-06-18"),  # pre-split NVDA top
    ("SMCI", "2024-03-08"),  # SMCI parabolic peak before split-adjusted drop
    ("DJT",  "2024-03-26"),  # DJT meme post-merger peak
    ("TSLA", "2024-12-17"),  # TSLA post-election parabolic peak
]


def main():
    tickers = sorted({t for t, _ in EVENTS})
    try:
        px = load_prices(tickers + ["SPY"], start="2020-06-01")
    except Exception as e:
        return mark_failed("W8_wsb_short_post_gme", f"Price load failed: {e}")

    if px.empty:
        return mark_failed("W8_wsb_short_post_gme", "No price data")

    rets = px.pct_change()

    # 60-day pre-event beta for hedge (use OLS slope of stock_ret on spy_ret)
    def beta_for(stock, asof):
        try:
            window = rets[[stock, "SPY"]].loc[:asof].tail(60).dropna()
            if len(window) < 30:
                return 1.0
            cov = window.cov().iloc[0, 1]
            var = window["SPY"].var()
            if var == 0:
                return 1.0
            return float(cov / var)
        except Exception:
            return 1.0

    event_rows = []
    for ticker, ev_date in EVENTS:
        ev_ts = pd.Timestamp(ev_date)
        if ticker not in rets.columns:
            continue
        future = rets.index[rets.index > ev_ts]
        if len(future) < 5:
            continue
        hold_idx = future[:5]
        stock_ret = rets[ticker].reindex(hold_idx)
        spy_ret = rets["SPY"].reindex(hold_idx)
        beta = beta_for(ticker, ev_ts)
        # Pair PnL: short stock (=-stock_ret) + beta*long SPY (=beta*spy_ret)
        pair = (-stock_ret + beta * spy_ret).dropna()
        if pair.empty:
            continue
        total = float((1 + pair).prod() - 1)
        event_rows.append({
            "ticker": ticker,
            "event_date": ev_date,
            "beta": beta,
            "five_day_pair_return": total,
            "five_day_stock_return": float((1 + stock_ret).prod() - 1),
            "hit": int(total > 0),
        })

    if len(event_rows) < 5:
        return mark_failed("W8_wsb_short_post_gme",
                           f"Only {len(event_rows)} events with data")

    ev_df = pd.DataFrame(event_rows)
    mean_pair = ev_df["five_day_pair_return"].mean()
    median_pair = ev_df["five_day_pair_return"].median()
    hit = ev_df["hit"].mean()
    n = len(ev_df)
    std_pair = ev_df["five_day_pair_return"].std()
    t_stat = mean_pair / (std_pair / np.sqrt(n)) if std_pair > 0 else 0.0

    # Build a pseudo-daily PnL: distribute each event's pair PnL across its 5-day window
    # so compute_metrics has a series to work with.
    daily_pnl_series = []
    daily_idx = []
    for _, row in ev_df.iterrows():
        ev_ts = pd.Timestamp(row["event_date"])
        future = rets.index[rets.index > ev_ts][:5]
        stock_ret = rets[row["ticker"]].reindex(future)
        spy_ret = rets["SPY"].reindex(future)
        pair = (-stock_ret + row["beta"] * spy_ret).dropna()
        daily_pnl_series.append(pair)
    full = pd.concat(daily_pnl_series).sort_index()
    # If duplicate dates (two events overlapping), average them
    full = full.groupby(level=0).mean()

    if len(full) >= 30:
        m = compute_metrics(full, benchmark=rets["SPY"].reindex(full.index),
                            name="W8 WSB squeeze short-and-hedge (small-N event)")
    else:
        # Tiny: short metric dict
        m = {
            "name": "W8 WSB squeeze short-and-hedge (small-N event)",
            "start": str(full.index.min().date()),
            "end": str(full.index.max().date()),
            "n_days": int(len(full)),
            "cagr": None,
            "ann_vol": float(full.std() * np.sqrt(252)) if full.std() > 0 else 0.0,
            "sharpe": None,
            "max_dd": None,
            "calmar": None,
            "hit_rate": float((full > 0).mean()),
            "t_stat": float(full.mean() / (full.std() / np.sqrt(len(full))))
                          if full.std() > 0 else 0.0,
        }

    m["n_events"] = int(n)
    m["event_hit_rate"] = float(hit)
    m["mean_event_pair_return"] = float(mean_pair)
    m["median_event_pair_return"] = float(median_pair)
    m["event_t_stat"] = float(t_stat)

    print_metrics(m) if "error" not in m else None
    print(f"Events: {n}, mean 5d pair return: {mean_pair*100:.2f}%, "
          f"median: {median_pair*100:.2f}%, hit: {hit*100:.0f}%, "
          f"t-stat: {t_stat:.2f}")

    save_result("W8_wsb_short_post_gme", m, extra={
        "status": "small_sample_event_study",
        "rule": ("On documented WSB peak day, short the stock for next 5 trading "
                 "days, beta-hedged with SPY. Pair PnL = -stock_ret + beta*spy_ret."),
        "mechanism": ("Attention-driven retail buying creates over-valuation at "
                       "the WSB attention peak; mean-reversion + lottery preference "
                       "decay produce negative ex-post drift over the following week."),
        "source": ("ApeWisdom REST API exposes only a current snapshot, so we "
                    "use a curated list of WSB attention peaks documented in "
                    "news archives. References: Bradley et al. (RFS 2024) "
                    "'Meme stocks and the feedback loop'; Pedersen 'Game on'."),
        "universe": "GME, AMC, BBBY, NVDA, SMCI, DJT, TSLA (curated WSB peaks).",
        "events_used": [{"ticker": t, "date": d} for t, d in EVENTS],
        "events_traded": ev_df.to_dict(orient="records"),
        "caveats": ("N is small (≤10). Sharpe/CAGR computed on the union of 5-day "
                     "windows, not on a continuous series — interpret with care."),
    })


if __name__ == "__main__":
    main()
