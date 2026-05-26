"""PL16_great_lakes_ice_steel_pair — Great Lakes Heavy Ice → Long EAF / Short BF Steel
In heavy-ice years (GLERL peak >60%), Soo Locks delays strand iron ore pellets.
Long NUE+STLD (EAF, scrap-fed) / short CLF (blast furnace, pellet-dependent) Mar-May.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL16_great_lakes_ice_steel_pair"
    try:
        px = load_prices(["NUE", "STLD", "CLF", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    eaf_basket = (ret["NUE"] + ret["STLD"]) / 2
    bf_leg = ret["CLF"]
    spy_r = ret["SPY"]

    # Pair: long EAF, short BF
    pair_daily = eaf_basket - bf_leg

    # Heavy ice years (NOAA GLERL peak ice cover >60%)
    heavy_ice_years = [2014, 2015, 2018, 2019]
    # Light ice control years
    light_ice_years = [2012, 2013, 2016, 2017, 2020, 2021, 2022, 2023, 2024, 2025]

    def season_stats(year, label):
        """Mar 1 - May 15 pair return."""
        mask = (pair_daily.index.year == year) & \
               (pair_daily.index.month >= 3) & \
               ((pair_daily.index.month < 5) | ((pair_daily.index.month == 5) & (pair_daily.index.day <= 15)))
        if mask.sum() < 20:
            return None
        window = pair_daily[mask]
        eaf_w = eaf_basket[mask]
        bf_w = bf_leg[mask]
        spy_w = spy_r[mask]
        return {
            "year": year,
            "type": label,
            "pair_return": round(float((1 + window).prod() - 1), 4),
            "eaf_return": round(float((1 + eaf_w).prod() - 1), 4),
            "clf_return": round(float((1 + bf_w).prod() - 1), 4),
            "spy_return": round(float((1 + spy_w).prod() - 1), 4),
            "n_days": int(mask.sum()),
        }, window

    events = []
    pnl_parts = []
    control_events = []

    for yr in heavy_ice_years:
        result = season_stats(yr, "heavy_ice")
        if result is None:
            continue
        stats, window_pnl = result
        events.append(stats)
        pnl_parts.append(window_pnl)

    for yr in light_ice_years:
        result = season_stats(yr, "light_ice")
        if result is None:
            continue
        stats, _ = result
        control_events.append(stats)

    if not events:
        return mark_failed(sid, "No heavy-ice events with data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Great Lakes Ice Steel Pair (Mar-May)")

    heavy_avg = np.mean([e["pair_return"] for e in events])
    light_avg = np.mean([e["pair_return"] for e in control_events]) if control_events else 0

    save_result(sid, m, extra={
        "rule": "Long NUE+STLD / short CLF Mar 1 - May 15 in years with GLERL peak ice >60%",
        "mechanism": "Heavy ice delays Soo Locks → iron ore pellet shortage for blast furnace mills → EAF mills (scrap-fed) gain relative advantage",
        "source": "NOAA GLERL ice data; yfinance",
        "n_heavy_ice_events": len(events),
        "heavy_ice_avg_pair_return": round(heavy_avg, 4),
        "light_ice_avg_pair_return": round(light_avg, 4),
        "edge_vs_control": round(heavy_avg - light_avg, 4),
        "events": events,
        "control_events": control_events[:5],
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} heavy-ice years, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Heavy avg: {heavy_avg*100:.1f}%  Light avg: {light_avg*100:.1f}%  Edge: {(heavy_avg-light_avg)*100:.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
