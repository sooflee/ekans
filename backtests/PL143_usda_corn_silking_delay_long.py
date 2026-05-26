"""PL143_usda_corn_silking_delay_long — Corn Jul 15-Aug 15 Seasonal Trade
Long CORN ETF (or ZC=F) from Jul 15 to Aug 15 each year during peak pollination stress window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL143_usda_corn_silking_delay_long"
    try:
        px = load_prices(["CORN", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "CORN" not in ret.columns:
        return mark_failed(sid, "CORN ETF data not available")

    corn_r = ret["CORN"]
    spy_r = ret["SPY"]

    events = []
    pnl_parts = []

    for year in range(2010, 2026):
        try:
            jul15 = pd.Timestamp(f"{year}-07-15")
            aug15 = pd.Timestamp(f"{year}-08-15")
        except:
            continue

        mask = (ret.index >= jul15) & (ret.index <= aug15)
        window_r = corn_r[mask]
        spy_window = spy_r[mask]

        if len(window_r) < 10:
            continue

        pnl_parts.append(window_r)
        corn_cum = float((1 + window_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "year": year,
            "corn_return": round(corn_cum, 4),
            "spy_return": round(spy_cum, 4),
            "excess": round(corn_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No corn seasonal events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Corn Jul 15-Aug 15 Seasonal")

    avg_corn = np.mean([e["corn_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["corn_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long CORN ETF Jul 15 to Aug 15 each year (peak pollination risk window)",
        "mechanism": "Weather risk premium during corn pollination — heat/drought stress risk peaks mid-July to mid-August",
        "source": "yfinance CORN ETF",
        "n_events": len(events),
        "avg_corn_return": round(avg_corn, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg CORN: {avg_corn*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["corn_return"] > 0 else "-"
        print(f"  {flag} {e['year']}: CORN {e['corn_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
