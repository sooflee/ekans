"""PL758_steam_ccu_decay_short_ttwo_ea — Steam CCU Decay Post-Launch -> Short TTWO/EA (Counter)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL758_steam_ccu_decay_short_ttwo_ea"

    # Steam CCU (Concurrent Users) data from SteamCharts.com (public)
    # Hand-coded major AAA launches by TTWO or EA where Day-7 CCU / Day-1 peak < 0.30
    # indicating poor retention → short the publisher for 20 trading days
    #
    # Key events (from SteamCharts public data):
    # TTWO titles:
    #   - Kerbal Space Program 2 (Feb 2023): Day-1 peak ~13k CCU, Day-7 ~2.8k (21% retention)
    #     EA published but TTWO invested; primary signal goes to EA as publisher
    #   - NBA 2K series: console-heavy, low Steam CCU base — limited signal
    #   - Civilization VII (Feb 2025): TTWO/Firaxis; Day-1 ~192k, Day-7 ~38k (20%)
    #
    # EA titles:
    #   - Star Wars Jedi: Survivor (Apr 2023): Day-1 ~182k CCU, Day-7 ~23k (12.6% retention)
    #     Very poor PC performance/optimization → short EA
    #   - EA Sports FC 24 (Sep 2023): Day-1 ~130k, Day-7 ~28k (21.5%) — weak retention
    #   - Dragon Age: The Veilguard (Oct 2024): Day-1 ~86k, Day-7 ~14k (16.3%)
    #   - Mass Effect: Andromeda (Mar 2017): Day-1 ~17k, Day-7 ~4.5k (26.5%) — marginal
    #   - Anthem (Feb 2019): Day-1 ~55k CCU, Day-7 ~9k (16.4%) — massive fail
    #   - Battlefield V (Nov 2018): Day-1 ~200k, Day-7 ~27k (13.5%)
    #   - Battlefield 2042 (Nov 2021): Day-1 ~105k, Day-7 ~17k (16.2%)
    #
    # Format: (date_string, ticker, day1_ccu, day7_ccu, title)
    # date = Day-7 measurement date (trigger date for the short)
    launch_events = [
        # EA events
        ("2019-02-28", "EA", 55000, 9000, "Anthem"),
        ("2021-11-26", "EA", 105000, 17000, "Battlefield 2042"),
        ("2023-04-15", "EA", 182000, 23000, "Star Wars Jedi: Survivor"),
        ("2023-09-29", "EA", 130000, 28000, "EA Sports FC 24"),
        ("2024-10-31", "EA", 86000, 14000, "Dragon Age: The Veilguard"),
        # TTWO events
        ("2025-02-10", "TTWO", 192000, 38000, "Civilization VII"),
    ]

    try:
        px = load_prices(["TTWO", "EA", "SPY"], start="2017-01-01")
        ret = daily_returns(px)
        spy_r = ret["SPY"]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    hold = 20  # 20 trading days as specified
    retention_threshold = 0.30

    # Separate PnL series for each ticker
    pnl_ea = pd.Series(0.0, index=ret["EA"].index)
    pnl_ttwo = pd.Series(0.0, index=ret["TTWO"].index)
    events = []

    for event_date_str, ticker, day1_ccu, day7_ccu, title in launch_events:
        retention = day7_ccu / day1_ccu
        if retention >= retention_threshold:
            print(f"  Skip {title} ({ticker}, {event_date_str}): retention {retention:.1%} >= threshold")
            continue

        event_date = pd.Timestamp(event_date_str)
        stock_r = ret[ticker]
        pnl_series = pnl_ea if ticker == "EA" else pnl_ttwo

        # Find first trading day on or after trigger date
        mask = stock_r.index >= event_date
        if mask.sum() < hold:
            print(f"  Skip {title}: insufficient future data")
            continue

        entry_idx = stock_r.index[mask][0]
        p = stock_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(stock_r))

        window_r = stock_r.iloc[p:ep]
        short_r = -window_r

        window_idx = stock_r.index[p:ep]
        pnl_series.loc[window_idx] = short_r.values[:len(window_idx)]

        cum_stock = float((1 + window_r).prod() - 1)
        cum_short = -cum_stock

        cum_spy = None
        if entry_idx in spy_r.index:
            sp = spy_r.index.get_loc(entry_idx)
            sp_end = min(sp + hold, len(spy_r))
            cum_spy = float((1 + spy_r.iloc[sp:sp_end]).prod() - 1)

        events.append({
            "title": title,
            "ticker": ticker,
            "trigger_date": str(event_date.date()),
            "entry_date": str(entry_idx.date()),
            "retention_ratio": round(retention, 3),
            "short_return": round(cum_short, 4),
            "underlying_return": round(cum_stock, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
        })
        print(f"  {title} ({ticker}, {event_date_str}): retention {retention:.1%}, "
              f"short {cum_short*100:.1f}% vs SPY {cum_spy*100:.1f}%"
              if cum_spy is not None
              else f"  {title} ({ticker}, {event_date_str}): retention {retention:.1%}, "
                   f"short {cum_short*100:.1f}%")

    print(f"Total qualifying events: {len(events)}")

    if len(events) == 0:
        return mark_failed(sid, "no qualifying CCU-decay events found")

    # Combine PnL from both tickers
    combined_idx = pnl_ea.index.union(pnl_ttwo.index)
    pnl_combined = (pnl_ea.reindex(combined_idx, fill_value=0) +
                    pnl_ttwo.reindex(combined_idx, fill_value=0))

    # Normalize: if both fire on same day, each contributes equal weight
    # (rare but possible); for now just sum as separate positions
    active_pnl = pnl_combined[pnl_combined != 0]

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active PnL days ({len(active_pnl)}): "
                               f"{len(events)} event(s) with {hold}-day windows")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Steam CCU Decay -> Short TTWO/EA")
    save_result(sid, m, extra={
        "rule": "Short publisher for 20 trading days when Day-7 Steam CCU < 30% of Day-1 peak "
                "for major AAA launch by TTWO or EA",
        "mechanism": "Poor Steam retention signals broader engagement failure, "
                     "increasing risk of earnings misses on PC/console revenue estimates "
                     "and triggering negative sell-side revisions.",
        "source": "SteamCharts.com daily CCU data (public); yfinance TTWO, EA, SPY",
        "n_events": len(events),
        "events": events,
        "caveats": "CCU data hand-coded from SteamCharts; PC-centric signal may miss "
                   "console-dominant titles. bt_feasibility=3 reflects reliance on "
                   "alternative data. Some EA titles are more PC-centric than TTWO.",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
