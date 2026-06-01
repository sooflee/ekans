"""PL686 — BYD Price-Cut Announcements → Short LI/NIO/XPEV, Long ALB/MP"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# BYD major price cut events (>=5% cut on a popular trim)
# Sourced from public Chinese EV news (Reuters, Bloomberg, Sina Finance)
BYD_PRICE_CUT_EVENTS = [
    "2022-09-26",   # BYD Han/Tang price cuts following Model 3 price war
    "2023-02-10",   # BYD February price adjustments post-Tesla cuts
    "2023-05-15",   # BYD Sea Lion / Seal pricing pressure
    "2023-09-19",   # BYD "Golden September" promotional pricing
    "2024-01-22",   # BYD 2024 model year repricing
    "2024-03-01",   # BYD spring price adjustments, widespread
    "2024-05-07",   # BYD Dynasty series cuts ahead of 618 festival
    "2024-07-15",   # BYD midyear pricing reset
    "2024-10-08",   # BYD Q4 promotional pricing
    "2025-01-10",   # BYD 2025 model year launch pricing
]


def main():
    sid = "PL686_byd_pricecut_short_li_nio"
    try:
        px = load_prices(["LI", "NIO", "XPEV", "ALB", "MP", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "price data empty")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Short basket: LI, NIO, XPEV (equal weight)
    short_tickers = [t for t in ["LI", "NIO", "XPEV"] if t in ret.columns]
    # Long basket: ALB, MP (equal weight)
    long_tickers = [t for t in ["ALB", "MP"] if t in ret.columns]

    if not short_tickers or not long_tickers:
        return mark_failed(sid, f"missing tickers: short={short_tickers}, long={long_tickers}")

    hold = 25
    pnl = pd.Series(0.0, index=spy_r.index)
    events_detail = []

    for date_str in BYD_PRICE_CUT_EVENTS:
        td = pd.Timestamp(date_str)

        # Find next available trading day
        mask = spy_r.index >= td
        if mask.sum() < hold:
            continue
        ei = spy_r.index[mask][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + hold, len(spy_r))
        window_idx = spy_r.index[p:ep]

        # Avoid overlap
        already_active = pnl.loc[window_idx] != 0.0
        if already_active.any():
            continue

        # Short basket return (equal-weight average)
        short_r = ret[short_tickers].reindex(window_idx).fillna(0.0)
        short_basket = short_r.mean(axis=1)

        # Long basket return (equal-weight average)
        long_r = ret[long_tickers].reindex(window_idx).fillna(0.0)
        long_basket = long_r.mean(axis=1)

        # PnL: long - short
        period_pnl = long_basket.values - short_basket.values
        pnl.loc[window_idx] = period_pnl

        long_ret = float((1 + long_basket).prod() - 1)
        short_ret = float((1 + short_basket).prod() - 1)
        pair_ret = long_ret - short_ret
        sp_ret = float((1 + spy_r.reindex(window_idx).fillna(0.0)).prod() - 1)

        events_detail.append({
            "event_date": date_str,
            "entry_date": str(ei.date()),
            "long_basket_return": round(long_ret, 4),
            "short_basket_return": round(short_ret, 4),
            "pair_pnl": round(pair_ret, 4),
            "spy_return": round(sp_ret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no non-overlapping events found")

    active_pnl = pnl[pnl != 0.0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="BYD Price Cut → Short CN EV, Long Li-Miners")
    avg_ret = np.mean([e["pair_pnl"] for e in events_detail])
    win_rate = np.mean([e["pair_pnl"] > 0 for e in events_detail])

    save_result(sid, m, extra={
        "rule": "On BYD list-price cut >=5%, short LI/NIO/XPEV equal-weight, long ALB/MP equal-weight for 25 trading days",
        "mechanism": "BYD cuts pressure Chinese EV peers on margin expectations; price war accelerates EV adoption (bullish for Li-miners long-term)",
        "source": "BYD official announcements; Reuters; yfinance",
        "n_events": len(events_detail),
        "avg_event_return": round(float(avg_ret), 4),
        "event_win_rate": round(float(win_rate), 4),
        "events": events_detail,
    })
    print(f"Done: {len(events_detail)} events, avg_ret={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
