"""PL408_usda_pnw_grain_export_unp_long — USDA PNW Grain Export Inspection Surge -> Long Union Pacific
Proxy: when UNP outperforms CSX by > 5% over 20 trading days during grain export season (Sep-Feb),
long UNP for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL408_usda_pnw_grain_export_unp_long"
    try:
        px = load_prices(["UNP", "CSX", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    if "UNP" not in ret.columns or "CSX" not in ret.columns:
        return mark_failed(sid, "UNP or CSX data not available")

    unp_r = ret["UNP"]
    csx_r = ret["CSX"]

    # Rolling 20-day cumulative return
    unp_cum20 = (1 + unp_r).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    csx_cum20 = (1 + csx_r).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    excess_20d = unp_cum20 - csx_cum20

    # Find dates where UNP outperforms CSX by > 5% during grain season (Sep-Feb)
    triggers = []
    last_trigger = None

    for i in range(1, len(excess_20d)):
        dt = excess_20d.index[i]
        if dt.month not in (9, 10, 11, 12, 1, 2):
            continue
        val = float(excess_20d.iloc[i])
        prev = float(excess_20d.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue
        # Cross above 5%
        if val > 0.05 and prev <= 0.05:
            if last_trigger is None or (dt - last_trigger).days >= 90:
                triggers.append(dt)
                last_trigger = dt

    if not triggers:
        return mark_failed(sid, "No UNP vs CSX outperformance > 5% events in grain season")

    events = []
    pnl_parts = []
    hold_days = 42

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        unp_window = unp_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(unp_window)

        unp_cum = float((1 + unp_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "unp_vs_csx_20d": round(float(excess_20d.loc[trig_date]), 4),
            "unp_42d_return": round(unp_cum, 4),
            "spy_42d_return": round(spy_cum, 4),
            "excess": round(unp_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable grain season UNP events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="UNP PNW Grain Season Outperformance -> Long UNP")

    avg_basket = np.mean([e["unp_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["unp_42d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "UNP outperforms CSX by > 5% over 20d during Sep-Feb grain season -> long UNP 42 days",
        "mechanism": "UNP relative strength vs CSX during grain season proxies PNW grain export surge benefiting UNP's franchise",
        "source": "USDA PNW grain (proxy via UNP/CSX relative strength) + yfinance",
        "n_events": len(events),
        "avg_unp_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg UNP: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["unp_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: UNP_ex={e['unp_vs_csx_20d']*100:.1f}%, UNP {e['unp_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
