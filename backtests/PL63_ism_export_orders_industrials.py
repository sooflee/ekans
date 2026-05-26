"""PL63_ism_export_orders_industrials — ISM Export Orders > 55 from Below 50 → Long Industrial Exporters
When NAPMEI crosses above 55 after 4+ months below 50, foreign demand is recovering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL63_ism_export_orders_industrials"
    try:
        px = load_prices(["HON", "EMR", "ITW", "PH", "SPY"], start="1999-01-01")
        napmei = load_fred("NAPMEI", start="1988-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Find months where NAPMEI crosses above 55 after 4+ months below 50
    napmei_m = napmei.resample("M").last().dropna()
    below_50_count = 0
    triggers = []

    for i in range(1, len(napmei_m)):
        val = float(napmei_m.iloc[i])
        prev = float(napmei_m.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue
        if val < 50:
            below_50_count += 1
        elif val >= 55 and prev < 55 and below_50_count >= 4:
            triggers.append(napmei_m.index[i])
            below_50_count = 0
        elif val >= 50:
            below_50_count = 0

    events = []
    pnl_parts = []
    hold_days = 126

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        available = [t for t in ["HON", "EMR", "ITW", "PH"] if t in ret.columns and not ret[t].iloc[window].isna().all()]
        if not available:
            continue

        basket_r = ret[available].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "napmei_value": round(float(napmei_m.loc[trig_date]), 1),
            "basket_6m_return": round(basket_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No NAPMEI crossing events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="ISM Export Orders → Industrial Exporters")

    avg_basket = np.mean([e["basket_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "NAPMEI crosses above 55 after 4+ months below 50 → long HON+EMR+ITW+PH 6mo",
        "mechanism": "ISM export orders recovery = foreign demand returning → US industrial exporter order books fill → earnings revisions positive",
        "source": "FRED NAPMEI + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_6m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (NAPMEI={e['napmei_value']}): basket {e['basket_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
