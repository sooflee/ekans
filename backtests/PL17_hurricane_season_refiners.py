"""PL17_hurricane_season_refiners — Gulf Hurricane Season → Long Refiners (Jul-Oct)
In hyperactive hurricane years (CSU >17 named storms), long refiner basket Jul-Oct.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL17_hurricane_season_refiners"
    try:
        px = load_prices(["VLO", "PSX", "MPC", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hyperactive hurricane years with major Gulf landfall
    hyperactive = [2005, 2008, 2017, 2020, 2021]
    # Control: non-hyperactive
    all_years = list(range(2005, 2026))
    control_years = [y for y in all_years if y not in hyperactive]

    def season_return(year):
        """Jul 1 - Oct 31 return for refiner basket."""
        mask = (ret.index.year == year) & (ret.index.month >= 7) & (ret.index.month <= 10)
        if mask.sum() < 40:
            return None, None, None

        # Build basket — PSX only from 2012, MPC from 2011
        available = ["VLO"]
        if year >= 2012 and "PSX" in ret.columns:
            available.append("PSX")
        if year >= 2011 and "MPC" in ret.columns:
            available.append("MPC")

        basket_r = ret[available][mask].mean(axis=1)
        spy_window = spy_r[mask]

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        return basket_cum, spy_cum, basket_r

    events = []
    pnl_parts = []

    for yr in hyperactive:
        b_ret, s_ret, daily_pnl = season_return(yr)
        if b_ret is None:
            continue
        events.append({
            "year": yr,
            "type": "hyperactive",
            "refiner_jul_oct_return": round(b_ret, 4),
            "spy_jul_oct_return": round(s_ret, 4),
            "excess": round(b_ret - s_ret, 4),
        })
        pnl_parts.append(daily_pnl)

    control_events = []
    for yr in control_years:
        b_ret, s_ret, _ = season_return(yr)
        if b_ret is None:
            continue
        control_events.append({
            "year": yr,
            "type": "control",
            "refiner_jul_oct_return": round(b_ret, 4),
            "spy_jul_oct_return": round(s_ret, 4),
        })

    if not events:
        return mark_failed(sid, "No hyperactive hurricane events with data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Hurricane Season Refiners (Jul-Oct)")

    hyper_avg = np.mean([e["refiner_jul_oct_return"] for e in events])
    ctrl_avg = np.mean([e["refiner_jul_oct_return"] for e in control_events]) if control_events else 0

    save_result(sid, m, extra={
        "rule": "Long VLO+PSX+MPC Jul 1 - Oct 31 in hyperactive hurricane years (CSU >17 named storms)",
        "mechanism": "Gulf hurricanes shut 1-2M bpd offshore crude → crack spreads spike → refiners capture windfall margins on pipeline-delivered crude",
        "source": "CSU hurricane forecasts; BSEE shut-in data; yfinance",
        "n_hyperactive_events": len(events),
        "hyperactive_avg_return": round(hyper_avg, 4),
        "control_avg_return": round(ctrl_avg, 4),
        "edge_vs_control": round(hyper_avg - ctrl_avg, 4),
        "events": events,
        "control_events": control_events[:5],
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} hyperactive years, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Hyperactive avg: {hyper_avg*100:.1f}%  Control avg: {ctrl_avg*100:.1f}%  Edge: {(hyper_avg-ctrl_avg)*100:.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
