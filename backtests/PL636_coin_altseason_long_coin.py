"""
PL636 — Altcoin Breadth + Retail Crypto Search Surge -> Long COIN Into Earnings

Anchors: Feb 15, May 9, Aug 8, Nov 7 each year.
On each anchor, look back 10 trading days for entry candidates:
  - ETH 30d return - BTC 30d return >= +0.10 (altseason)
  - Google Trends 'buy crypto' (US) 4w MA > trailing 26w 75th percentile
If both true, enter long COIN at anchor - 10 trading days, hold 14 days.
Exit: +15% (profit), -10% (stop), or 14d max.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns, ROOT)


SIGNAL_ID = "PL636_coin_altseason_long_coin"
NAME = "ETH>BTC + Crypto-search hot -> Long COIN pre-earnings (14d)"

EARNINGS_ANCHORS = [(2, 15), (5, 9), (8, 8), (11, 7)]
CACHE = ROOT / "data" / "pytrends_buy_crypto.parquet"


def fetch_trends():
    if CACHE.exists():
        try:
            return pd.read_parquet(CACHE)
        except Exception:
            pass
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return None
    try:
        pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        pt.build_payload(["buy crypto"], cat=0, timeframe="today 5-y", geo="US", gprop="")
        df = pt.interest_over_time()
        if df.empty:
            return None
        df = df[["buy crypto"]].astype(float)
        df.index = pd.to_datetime(df.index)
        df.to_parquet(CACHE)
        return df
    except Exception as e:
        print(f"pytrends fetch failed: {e}")
        return None


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["COIN", "BTC-USD", "ETH-USD", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["COIN", "BTC-USD", "ETH-USD", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    coin_r = ret["COIN"]
    spy_r = ret["SPY"]
    trading_idx = ret.index

    btc = px["BTC-USD"]
    eth = px["ETH-USD"]
    btc_30 = btc / btc.shift(30) - 1.0
    eth_30 = eth / eth.shift(30) - 1.0
    alt_diff = eth_30 - btc_30

    trends = fetch_trends()
    if trends is None or trends.empty:
        return mark_failed(
            sid,
            "pytrends failed to fetch 'buy crypto' index",
            extra={
                "rule": "Long COIN 14d pre-earnings on altseason + retail surge",
                "mechanism": "Retail crypto re-engagement + ETH-led altseason drives COIN volumes / earnings",
                "source": "PL636 catalog",
            },
        )
    s = trends["buy crypto"]
    s_4w = s.rolling(4, min_periods=2).mean()
    s_26w_75 = s_4w.rolling(26, min_periods=10).quantile(0.75)
    trends_hot = (s_4w > s_26w_75).reindex(trading_idx, method="ffill").fillna(False)

    hold_days = 14
    take_profit = 0.15
    stop_loss = -0.10

    positions = pd.Series(0.0, index=trading_idx)
    events = []
    last_exit_loc = -1

    # generate all earnings anchors in COIN's history
    anchors = []
    for year in range(2021, 2027):
        for (mm, dd) in EARNINGS_ANCHORS:
            anchors.append(pd.Timestamp(year, mm, dd))

    for anchor in anchors:
        loc = trading_idx.searchsorted(anchor)
        if loc >= len(trading_idx):
            continue
        # entry 10 trading days before anchor
        entry_loc = max(0, loc - 10)
        if entry_loc <= last_exit_loc:
            continue
        if entry_loc >= len(trading_idx) - 1:
            continue
        if pd.isna(px["COIN"].iloc[entry_loc]):
            continue
        # check conditions on entry date
        if entry_loc >= len(alt_diff) or pd.isna(alt_diff.iloc[entry_loc]):
            continue
        if alt_diff.iloc[entry_loc] < 0.10:
            continue
        if not bool(trends_hot.iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = px["COIN"].iloc[entry_loc]
        coin_win = px["COIN"].iloc[entry_loc:end_loc + 1]
        cumret = coin_win / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold"
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val >= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            if pd.notna(val) and val <= stop_loss:
                exit_loc = cur_i
                exit_reason = "stop_loss"
                break
        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events.append({
            "anchor": str(anchor.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "coin_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "alt_diff_at_trigger": float(alt_diff.iloc[entry_loc]),
        })

    if not events:
        return mark_failed(
            sid,
            "no joint altseason + trends-hot anchors fired in COIN history",
            extra={
                "rule": "Long COIN 14d pre-earnings on altseason + retail surge",
                "mechanism": "Retail crypto re-engagement + ETH-led altseason",
                "source": "PL636 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * coin_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()
    first_entry = pd.Timestamp(events[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos,
        cost_bps=15,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.mean())
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Anchor dates: Feb 15, May 9, Aug 8, Nov 7 each year (COIN earnings proxy). Enter long COIN 10 trading days before each anchor if (a) ETH-USD 30d return - BTC-USD 30d return >= +10% (altseason) AND (b) Google Trends US 'buy crypto' 4w MA > trailing 26w 75th percentile (retail re-engagement). Hold 14 trading days; exit on +15% (profit) or -10% (stop).",
            "mechanism": "ETH outperformance vs BTC signals altseason regime; rising consumer 'buy crypto' searches signal retail re-engagement. Both drive Coinbase trading volumes which dominate COIN earnings beats. Entering 10 trading days pre-earnings captures the pre-print drift.",
            "source": "PL636 idea catalog; pytrends Google Trends US; yfinance",
            "caveats": "COIN IPO Apr 2021 — very short sample. Earnings anchors are calendar approximations of actual variable earnings dates. Crypto-equity is high-vol; the +15%/-10% asymmetric bounds keep R/R favorable but the strategy is event-driven with small N.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
