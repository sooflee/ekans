"""PL9_scrap_steel_hrc_spread — Scrap/HRC Spread → Long EAF Steel Mills
When steel PPI rises while NUE/STLD are flat, margin expansion is coming.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL9_scrap_steel_hrc_spread"
    try:
        px = load_prices(["NUE", "STLD", "SPY"], start="2010-01-01")
        steel_ppi = load_fred("WPUSI019011", start="2009-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    mill_basket = (ret["NUE"] + ret["STLD"]) / 2

    # Monthly steel PPI 6-month return
    steel_monthly = steel_ppi.resample("M").last().dropna()
    steel_6m_ret = steel_monthly.pct_change(6)

    # Signal: steel PPI 6mo return > 10%
    events = []
    pnl_all = pd.Series(0.0, index=ret.index)
    hold_days = 120

    for date, ppi_ret in steel_6m_ret.dropna().items():
        ppi_val = float(ppi_ret)
        if np.isnan(ppi_val) or ppi_val <= 0.10:
            continue

        entry_mask = ret.index >= date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        daily_pnl = mill_basket.iloc[window]
        pnl_all.iloc[window] += daily_pnl.values[:exit_loc - entry_loc]

        cum_ret = float((1 + daily_pnl).prod() - 1)
        spy_cum = float((1 + spy_r.iloc[window]).prod() - 1)
        events.append({
            "trigger_date": str(date.date()),
            "steel_ppi_6m_return": round(ppi_val, 4),
            "mill_basket_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4),
            "excess": round(cum_ret - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No trigger events (steel PPI 6mo > 10%)")

    in_pos = pnl_all[pnl_all != 0].dropna()
    if len(in_pos) < 30:
        return mark_failed(sid, f"Only {len(in_pos)} in-position days")

    m = compute_metrics(in_pos, benchmark=spy_r.reindex(in_pos.index).dropna(),
                        name="Steel Spread → Long EAF Mills")
    save_result(sid, m, extra={
        "rule": "When FRED steel PPI 6mo return > 10%: long NUE+STLD equal-weight 120 days",
        "mechanism": "Rising HRC prices with lagging scrap costs → EAF mill margin expansion",
        "source": "FRED WPUSI019011 + yfinance",
        "n_events": len(events),
        "events": events[:10],
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
