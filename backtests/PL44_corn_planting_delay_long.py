"""PL44_corn_planting_delay_long — Corn Planting >15pp Behind Avg -> Long ZC=F
Hand-code years when USDA Crop Progress showed corn planting >15pp behind 5yr avg by May 15.
Long ZC=F from May 15 to Jun 30. Events: 2013, 2019, 2022.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL44_corn_planting_delay_long"
    try:
        px = load_prices(["ZC=F", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded late planting years (>15pp behind 5yr avg by May 15)
    delay_events = [
        ("2013-05-15", "2013: 36% planted vs 65% avg — 29pp behind, wet spring"),
        ("2019-05-15", "2019: 49% planted vs 80% avg — 31pp behind, historic floods"),
        ("2022-05-15", "2022: 22% planted vs 49% avg — 27pp behind, cold/wet spring"),
    ]

    events = []
    pnl_parts = []

    for date_str, desc in delay_events:
        entry_date = pd.Timestamp(date_str)
        # Exit on June 30 of same year
        year = entry_date.year
        exit_date = pd.Timestamp(f"{year}-06-30")

        entry_mask = ret.index >= entry_date
        exit_mask = ret.index <= exit_date
        combined_mask = entry_mask & exit_mask

        if combined_mask.sum() < 10:
            continue

        zc_col = "ZC=F" if "ZC=F" in ret.columns else None
        if zc_col is None or ret[zc_col].loc[combined_mask].isna().all():
            continue

        zc_window = ret[zc_col].loc[combined_mask].dropna()
        spy_window = spy_r.loc[combined_mask].dropna()
        pnl_parts.append(zc_window)

        zc_cum = float((1 + zc_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1) if len(spy_window) > 0 else 0

        events.append({
            "entry_date": date_str,
            "exit_date": f"{year}-06-30",
            "description": desc,
            "zc_return": round(zc_cum, 4),
            "spy_return": round(spy_cum, 4),
            "excess": round(zc_cum - spy_cum, 4),
            "n_days": len(zc_window),
        })

    if not events:
        return mark_failed(sid, "No planting delay events with corn futures data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Corn Planting Delay -> Long ZC=F")

    avg_zc = np.mean([e["zc_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["zc_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long ZC=F May 15 - Jun 30 in years when planting >15pp behind avg by May 15",
        "mechanism": "Late planting -> reduced acreage + lower yields -> USDA cuts production estimates -> corn prices rally into WASDE",
        "source": "USDA Crop Progress (hand-coded delay years); yfinance ZC=F",
        "n_events": len(events),
        "avg_corn_return": round(avg_zc, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg corn: {avg_zc*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["zc_return"] > 0 else "-"
        print(f"  {flag} {e['entry_date']}-{e['exit_date']}: ZC {e['zc_return']*100:+.1f}%, SPY {e['spy_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
