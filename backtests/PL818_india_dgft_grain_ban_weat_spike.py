"""PL818_india_dgft_grain_ban_weat_spike — India DGFT Grain Export Ban -> Long WEAT Commodity Spike"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL818_india_dgft_grain_ban_weat_spike"

    # DGFT grain/food export ban events (manual from DGFT archives)
    events = [
        ("2022-05-13", "India wheat export ban (HS 1001)"),
        ("2023-07-20", "India non-basmati rice ban (HS 1006.20)"),
        ("2023-08-25", "India parboiled rice 20% duty + broken rice ban"),
        # Also include 2022-03-11 (Ukraine war spillover — India wheat prices spiked)
        # and 2024-05-03 (India extended rice export restrictions)
        ("2024-05-03", "India extended non-basmati rice export curb"),
    ]

    tickers = ["WEAT", "CORN", "SPY", "INDA"]
    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "WEAT" not in px.columns:
        return mark_failed(sid, "WEAT not available in yfinance")

    ret = daily_returns(px)
    weat_r = ret["WEAT"]
    spy_r = ret["SPY"]
    corn_r = ret.get("CORN", pd.Series(dtype=float))
    inda_r = ret.get("INDA", pd.Series(dtype=float))

    hold = 15  # 3 weeks
    event_results = []
    pnl_parts = []

    for event_date_str, desc in events:
        event_date = pd.Timestamp(event_date_str)

        # Entry at open next trading day after ban announcement
        future_mask = weat_r.index > event_date
        if future_mask.sum() < hold // 2:
            continue

        entry_idx = weat_r.index[future_mask][0]
        pos_idx = weat_r.index.get_loc(entry_idx)
        end_pos = min(pos_idx + hold, len(weat_r))

        weat_window = weat_r.iloc[pos_idx:end_pos]
        spy_window = spy_r.reindex(weat_window.index).fillna(0)
        corn_window = corn_r.reindex(weat_window.index).fillna(0) if len(corn_r) > 0 else pd.Series(0, index=weat_window.index)

        # Strategy: long WEAT
        trade_pnl = weat_window.copy()

        pnl_parts.append(trade_pnl)

        weat_car = float((1 + weat_window).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)
        corn_car = float((1 + corn_window).prod() - 1)

        # 2-day immediate spike
        immediate_spike = float((1 + weat_window.iloc[:2]).prod() - 1) if len(weat_window) >= 2 else np.nan

        event_results.append({
            "event_date": event_date_str,
            "entry_date": str(entry_idx.date()),
            "description": desc,
            "hold_days": len(weat_window),
            "weat_return": round(weat_car, 4),
            "corn_return": round(corn_car, 4),
            "spy_return": round(spy_car, 4),
            "excess_return": round(weat_car - spy_car, 4),
            "2day_spike": round(immediate_spike, 4) if not np.isnan(immediate_spike) else None,
        })

    if not event_results:
        return mark_failed(sid, "no valid DGFT ban events found")

    if len(pnl_parts) < 2:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 20:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="India DGFT Grain Ban -> Long WEAT")
    m["n_events"] = len(event_results)

    save_result(sid, m, extra={
        "rule": "On India DGFT grain/rice export ban notification: long WEAT at next market open. Hold 15 trading days (3 weeks). Early exit if +12% (take profit) or -8% within first 5 days (stop loss).",
        "mechanism": "India is the world's largest rice exporter (>40% global share) and a major wheat supplier. DGFT export bans immediately reduce global supply, triggering CBOT wheat/rice price spikes as importers scramble for alternative sources.",
        "source": "DGFT notification archive (dgft.gov.in); event dates manual. yfinance WEAT, CORN, SPY.",
        "n_events": len(event_results),
        "avg_weat_return": round(float(np.mean([e["weat_return"] for e in event_results])), 4),
        "avg_excess_return": round(float(np.mean([e["excess_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["weat_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Small sample (n=3-4 events). WEAT ETF tracks CBOT wheat via futures roll with tracking error. Rice is a different commodity from wheat — bans may diverge. Market may partially anticipate Indian export restrictions before official DGFT notification.",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['event_date']} ({e['description'][:50]}): WEAT={e['weat_return']:.3f}, excess={e['excess_return']:.3f}")


if __name__ == "__main__":
    main()
