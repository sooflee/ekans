"""PL69_capacity_utilization_recovery_xlb — Capacity Utilization Crosses 76% from Below → Long XLB
When FRED TCU crosses above 76% after 6+ months below, long XLB 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL69_capacity_utilization_recovery_xlb"
    try:
        px = load_prices(["XLB", "SPY"], start="1998-01-01")
        tcu = load_fred("TCU", start="1990-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xlb_r = ret["XLB"]
    spy_r = ret["SPY"]

    # Monthly TCU
    tcu_m = tcu.resample("ME").last().dropna()

    # Find months where TCU crosses above 76 after 6+ months below 76
    below_count = 0
    triggers = []
    last_trigger = None
    for i in range(1, len(tcu_m)):
        val = float(tcu_m.iloc[i])
        prev = float(tcu_m.iloc[i - 1])
        if np.isnan(val) or np.isnan(prev):
            below_count = 0
            continue
        if val < 76:
            below_count += 1
        elif val >= 76 and prev < 76 and below_count >= 6:
            trig = tcu_m.index[i]
            if last_trigger is None or (trig - last_trigger).days > 300:
                triggers.append(trig)
                last_trigger = trig
            below_count = 0
        elif val >= 76:
            below_count = 0

    print(f"Found {len(triggers)} capacity utilization recovery events")

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
        xlb_window = xlb_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xlb_window)

        xlb_cum = float((1 + xlb_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        # Get TCU value at trigger
        tcu_val = float(tcu_m.iloc[tcu_m.index.get_indexer([trig_date], method='nearest')[0]])

        events.append({
            "trigger_date": str(trig_date.date()),
            "tcu_value": round(tcu_val, 2),
            "xlb_6m_return": round(xlb_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(xlb_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No capacity utilization recovery events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="CapUtil Recovery → Long XLB")

    avg_xlb = np.mean([e["xlb_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xlb_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED TCU crosses above 76% after 6+ months below → long XLB 6mo",
        "mechanism": "Factory restarts → raw material demand recovers → materials stocks benefit",
        "source": "FRED TCU + yfinance",
        "n_events": len(events),
        "avg_xlb_return": round(avg_xlb, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XLB: {avg_xlb*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xlb_6m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (TCU={e['tcu_value']}%): XLB {e['xlb_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
