"""PL146_eia_summer_injection_shortfall_natgas — Aug 1 Storage < 5yr Avg by >10% → Long NG Aug-Nov
When FRED NGTRSTUS shows Aug 1 storage >10% below 5-year average, long natural gas Aug-Nov.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL146_eia_summer_injection_shortfall_natgas"
    try:
        px = load_prices(["UNG", "SPY"], start="2007-01-01")
        ngs = load_fred("NGTRSTUS", start="2002-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "UNG" not in ret.columns:
        return mark_failed(sid, "UNG ETF data not available")

    ung_r = ret["UNG"]
    spy_r = ret["SPY"]

    ngs = ngs.dropna()

    events = []
    pnl_parts = []

    for year in range(2007, 2026):
        # Find storage value closest to Aug 1
        aug1 = pd.Timestamp(f"{year}-08-01")
        mask = (ngs.index >= aug1 - pd.Timedelta(days=7)) & (ngs.index <= aug1 + pd.Timedelta(days=7))
        aug_vals = ngs[mask]
        if len(aug_vals) == 0:
            continue
        current_storage = float(aug_vals.iloc[-1])

        # Compute 5-year average for same week
        five_yr_vals = []
        for prev_year in range(year - 5, year):
            prev_aug1 = pd.Timestamp(f"{prev_year}-08-01")
            prev_mask = (ngs.index >= prev_aug1 - pd.Timedelta(days=7)) & (ngs.index <= prev_aug1 + pd.Timedelta(days=7))
            prev_vals = ngs[prev_mask]
            if len(prev_vals) > 0:
                five_yr_vals.append(float(prev_vals.iloc[-1]))
        if len(five_yr_vals) < 3:
            continue
        avg_storage = np.mean(five_yr_vals)

        deficit_pct = (current_storage - avg_storage) / avg_storage

        if deficit_pct < -0.10:
            # Long UNG from Aug 1 to Nov 30
            nov30 = pd.Timestamp(f"{year}-11-30")
            trade_mask = (ret.index >= aug1) & (ret.index <= nov30)
            ung_window = ung_r[trade_mask]
            spy_window = spy_r[trade_mask]

            if len(ung_window) < 20:
                continue

            pnl_parts.append(ung_window)
            ung_cum = float((1 + ung_window).prod() - 1)
            spy_cum = float((1 + spy_window).prod() - 1)

            events.append({
                "year": year,
                "storage_bcf": round(current_storage, 0),
                "avg_5yr_bcf": round(avg_storage, 0),
                "deficit_pct": round(deficit_pct * 100, 1),
                "ung_aug_nov_return": round(ung_cum, 4),
                "spy_aug_nov_return": round(spy_cum, 4),
                "excess": round(ung_cum - spy_cum, 4),
            })

    if not events:
        return mark_failed(sid, "No storage deficit events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="NatGas Storage Deficit Aug → Long NG")

    avg_ung = np.mean([e["ung_aug_nov_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["ung_aug_nov_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Aug 1 NGTRSTUS >10% below 5yr avg → long UNG Aug-Nov",
        "mechanism": "Summer injection shortfall leaves winter supply at risk — front-month NG rises on storage concern",
        "source": "FRED NGTRSTUS + yfinance UNG",
        "n_events": len(events),
        "avg_ung_return": round(avg_ung, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg UNG: {avg_ung*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["ung_aug_nov_return"] > 0 else "-"
        print(f"  {flag} {e['year']} (deficit={e['deficit_pct']:.0f}%): UNG {e['ung_aug_nov_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
