"""PL1_heatwave_dc_cooling — US Heatwave CDD → Data Center REIT Margin Compression
Short DLR+EQIX equal-weight Jun 1 - Aug 31 each year, benchmark vs SPY same period.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1_heatwave_dc_cooling"
    try:
        px = load_prices(["DLR", "EQIX", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)

    dc_basket = (ret["DLR"] + ret["EQIX"]) / 2
    spy_r = ret["SPY"]

    # Short DC REITs = negative of their return; pair PnL = SPY - DC basket
    pair_pnl = spy_r - dc_basket

    # Only hold Jun-Aug each year
    summer_mask = pair_pnl.index.month.isin([6, 7, 8])
    pnl = pair_pnl.where(summer_mask, 0.0)

    # Also compute per-year summer stats
    events = []
    for yr in range(2015, 2027):
        mask_yr = (pnl.index.year == yr) & summer_mask
        if mask_yr.sum() == 0:
            continue
        yr_pnl = pnl[mask_yr]
        yr_ret = (1 + yr_pnl).prod() - 1
        dc_ret = ((1 + dc_basket[mask_yr]).prod() - 1)
        spy_ret_yr = ((1 + spy_r[mask_yr]).prod() - 1)
        events.append({
            "year": yr,
            "dc_reit_summer_return": round(float(dc_ret), 4),
            "spy_summer_return": round(float(spy_ret_yr), 4),
            "pair_return_spy_minus_dc": round(float(yr_ret), 4),
        })

    # Compute metrics on the daily PnL series (in-position only = summer days)
    in_pos = pnl[summer_mask].dropna()
    if len(in_pos) < 30:
        return mark_failed(sid, f"only {len(in_pos)} summer trading days")

    m = compute_metrics(in_pos, benchmark=spy_r[summer_mask].dropna(),
                        name="Heatwave DC REIT Short (Jun-Aug)")

    save_result(sid, m, extra={
        "rule": "Short equal-weight DLR+EQIX Jun 1 - Aug 31 each year vs long SPY",
        "mechanism": "Summer heat spikes data center cooling costs (40% of DC energy); colo REITs face margin compression from electricity costs",
        "source": "Uptime Institute PUE surveys; yfinance",
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
