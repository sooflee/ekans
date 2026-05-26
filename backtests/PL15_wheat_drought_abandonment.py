"""PL15_wheat_drought_abandonment — Winter Wheat Drought → Long KC Wheat (Mar-May)
In years with severe winter wheat drought (poor/very-poor >35% by March),
long KE=F from Mar 1 through May 31 to capture the WASDE production cut.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL15_wheat_drought_abandonment"
    try:
        px = load_prices(["KE=F", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    wheat_r = ret["KE=F"]
    spy_r = ret["SPY"]

    # Drought-abandonment years (USDA Crop Progress poor/very-poor >35% by March)
    drought_years = [2011, 2013, 2014, 2018, 2022, 2023]
    # Control: non-drought years
    all_years = list(range(2010, 2027))
    non_drought_years = [y for y in all_years if y not in drought_years]

    def season_return(year, series):
        """Mar 1 - May 31 return for a given year."""
        mask = (series.index.year == year) & (series.index.month >= 3) & (series.index.month <= 5)
        if mask.sum() < 20:
            return None, 0
        window = series[mask]
        return float((1 + window).prod() - 1), int(mask.sum())

    events = []
    pnl_parts = []

    for yr in drought_years:
        wheat_ret, n_days = season_return(yr, wheat_r)
        spy_ret, _ = season_return(yr, spy_r)
        if wheat_ret is None:
            continue

        mask = (wheat_r.index.year == yr) & (wheat_r.index.month >= 3) & (wheat_r.index.month <= 5)
        pnl_parts.append(wheat_r[mask])

        events.append({
            "year": yr,
            "type": "drought",
            "wheat_mar_may_return": round(wheat_ret, 4),
            "spy_mar_may_return": round(spy_ret, 4),
            "excess": round(wheat_ret - spy_ret, 4),
            "n_days": n_days,
        })

    # Also compute control years for comparison
    control_events = []
    for yr in non_drought_years:
        wheat_ret, n_days = season_return(yr, wheat_r)
        spy_ret, _ = season_return(yr, spy_r)
        if wheat_ret is None:
            continue
        control_events.append({
            "year": yr,
            "type": "control",
            "wheat_mar_may_return": round(wheat_ret, 4),
            "spy_mar_may_return": round(spy_ret, 4),
        })

    if not events:
        return mark_failed(sid, "No drought events with KE=F data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Wheat Drought Abandonment (Mar-May)")

    # Compute control average for comparison
    ctrl_avg = np.mean([e["wheat_mar_may_return"] for e in control_events]) if control_events else 0
    drought_avg = np.mean([e["wheat_mar_may_return"] for e in events])

    save_result(sid, m, extra={
        "rule": "Long KE=F Mar 1 - May 31 in years with USDA winter wheat poor/very-poor >35% by March",
        "mechanism": "Drought during dormancy → high abandonment (20-35% vs 10% normal) → WASDE cuts production → KC wheat rallies",
        "source": "USDA Crop Progress winter wheat conditions; yfinance KE=F",
        "n_drought_events": len(events),
        "drought_avg_return": round(drought_avg, 4),
        "control_avg_return": round(ctrl_avg, 4),
        "drought_vs_control": round(drought_avg - ctrl_avg, 4),
        "events": events,
        "control_events": control_events[:5],
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} drought years, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Drought avg: {drought_avg*100:.1f}%  Control avg: {ctrl_avg*100:.1f}%  Edge: {(drought_avg-ctrl_avg)*100:.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
