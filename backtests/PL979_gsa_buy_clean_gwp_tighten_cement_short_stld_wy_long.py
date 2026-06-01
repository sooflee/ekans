"""PL979_gsa_buy_clean_gwp_tighten_cement_short_stld_wy_long — GSA Buy Clean GWP Threshold Tightening: Short CX+EXP+MLM / Long STLD+WY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL979_gsa_buy_clean_gwp_tighten_cement_short_stld_wy_long"
    # Known GSA Buy Clean GWP threshold tightening events
    # 2023-09-15: GSA Buy Clean program launch with initial GWP concrete limits
    # 2024-10-01: GSA Buy Clean GWP threshold tightening update
    trigger_dates = pd.to_datetime(["2023-09-15", "2024-10-01"])

    try:
        # Short basket: CX (Cemex), EXP (Eagle Materials), MLM (Martin Marietta)
        # Long basket: STLD (Steel Dynamics), WY (Weyerhaeuser)
        tickers = ["CX", "EXP", "MLM", "STLD", "WY", "SPY"]
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Short basket: equal-weight CX + EXP + MLM
    short_basket = ret[["CX", "EXP", "MLM"]].mean(axis=1)
    # Long basket: equal-weight STLD + WY
    long_basket = ret[["STLD", "WY"]].mean(axis=1)
    # Pair PnL: long - short
    pair_r = long_basket - short_basket

    hold = 126  # 126 trading days (~6 months)
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for td in trigger_dates:
        future_mask = spy_r.index >= td
        if future_mask.sum() < hold:
            continue
        entry_idx = spy_r.index[future_mask][0]
        p = spy_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(spy_r))

        pair_future = pair_r[pair_r.index >= entry_idx]
        if len(pair_future) < 5:
            continue

        pair_window = pair_future.iloc[:hold]
        spy_window = spy_r.iloc[p:ep]

        pair_total = float((1 + pair_window).prod() - 1)
        spy_total = float((1 + spy_window).prod() - 1)

        short_window = short_basket[short_basket.index >= entry_idx].iloc[:hold]
        long_window = long_basket[long_basket.index >= entry_idx].iloc[:hold]
        short_total = float((1 + short_window).prod() - 1)
        long_total = float((1 + long_window).prod() - 1)

        # Accumulate PnL
        for dt, val in pair_window.items():
            if dt in pnl.index:
                pnl.loc[dt] += val

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "pair_return": round(pair_total, 4),
            "long_return": round(long_total, 4),
            "short_return": round(short_total, 4),
            "spy_return": round(spy_total, 4),
            "n_days": len(pair_window),
        })

    print(f"Events: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: pair={e['pair_return']:.2%}, "
              f"long={e['long_return']:.2%}, short={e['short_return']:.2%}, "
              f"SPY={e['spy_return']:.2%}")

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r,
                        name="GSA Buy Clean: Short Cement (CX+EXP+MLM) / Long Steel+Timber (STLD+WY)")

    pair_returns = [e["pair_return"] for e in events]
    mean_pair = float(np.mean(pair_returns))
    win_rate = float(np.mean([r > 0 for r in pair_returns]))

    save_result(sid, m, extra={
        "rule": "Short equal-weight CX+EXP+MLM / Long equal-weight STLD+WY 126 trading days when GSA Buy Clean GWP limit for portland cement tightens >10%",
        "mechanism": "High-carbon cement producers lose federal procurement share; low-carbon EAF steel (STLD) and mass timber (WY) benefit from carbon-preference procurement rules",
        "source": "GSA Buy Clean program bulletins: 2023-09-15 launch, 2024-10-01 update",
        "n_events": len(events),
        "mean_pair_return": round(mean_pair, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. Mean pair return: {mean_pair:.2%}, Win rate: {win_rate:.0%}")


if __name__ == "__main__":
    main()
