"""PL14_soybean_export_season — Soybean Export Season Momentum (Oct-Jan)
Long ZS=F from Oct 1 through Jan 31 each year to capture peak US export demand.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL14_soybean_export_season"
    try:
        px = load_prices(["ZS=F", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    soy_r = ret["ZS=F"]
    spy_r = ret["SPY"]

    # Only hold Oct-Jan each year
    export_mask = soy_r.index.month.isin([10, 11, 12, 1])
    pnl = soy_r.where(export_mask, 0.0)

    # Per-year event stats
    events = []
    for yr in range(2010, 2026):
        # Season runs Oct of yr through Jan of yr+1
        mask = ((soy_r.index.year == yr) & (soy_r.index.month >= 10)) | \
               ((soy_r.index.year == yr + 1) & (soy_r.index.month == 1))
        if mask.sum() < 20:
            continue
        season_soy = soy_r[mask]
        season_spy = spy_r[mask]
        soy_ret = float((1 + season_soy).prod() - 1)
        spy_ret = float((1 + season_spy).prod() - 1)
        events.append({
            "season": f"Oct {yr} - Jan {yr+1}",
            "soy_return": round(soy_ret, 4),
            "spy_return": round(spy_ret, 4),
            "excess": round(soy_ret - spy_ret, 4),
        })

    if not events:
        return mark_failed(sid, "No complete export seasons found")

    in_pos = pnl[export_mask].dropna()
    if len(in_pos) < 30:
        return mark_failed(sid, f"Only {len(in_pos)} in-position days")

    m = compute_metrics(in_pos, benchmark=spy_r[export_mask].dropna(),
                        name="Soybean Export Season (Oct-Jan)")
    save_result(sid, m, extra={
        "rule": "Long ZS=F Oct 1 through Jan 31 each year",
        "mechanism": "Peak US soybean export season — China/global buyers take new-crop delivery",
        "source": "yfinance ZS=F; USDA export calendar",
        "n_events": len(events),
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} seasons, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
