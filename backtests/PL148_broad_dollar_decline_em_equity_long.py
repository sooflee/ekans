"""PL148_broad_dollar_decline_em_equity_long — Dollar 6mo Return < -5% → Long EEM 63d
When FRED DTWEXBGS 6-month return crosses below -5%, long EEM for 63 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL148_broad_dollar_decline_em_equity_long"
    try:
        px = load_prices(["EEM", "SPY"], start="2003-01-01")
        dxy = load_fred("DTWEXBGS", start="2002-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "EEM" not in ret.columns:
        return mark_failed(sid, "EEM data not available")
    eem_r = ret["EEM"]
    spy_r = ret["SPY"]

    dxy = dxy.dropna()
    # Compute rolling 126-day (6-month) return
    dxy_6m_ret = dxy / dxy.shift(126) - 1

    # Find dates where 6-month return crosses below -5%
    triggers = []
    cooldown = 0
    for i in range(127, len(dxy_6m_ret)):
        if cooldown > 0:
            cooldown -= 1
            continue
        val = float(dxy_6m_ret.iloc[i])
        prev = float(dxy_6m_ret.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue
        if prev >= -0.05 and val < -0.05:
            triggers.append(dxy_6m_ret.index[i])
            cooldown = 63  # skip 63 trading days

    events = []
    pnl_parts = []
    hold_days = 63

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        eem_window = eem_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(eem_window)

        eem_cum = float((1 + eem_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "dxy_6m_return": round(float(dxy_6m_ret.iloc[dxy_6m_ret.index.get_loc(trig_date)]) * 100, 1),
            "eem_63d_return": round(eem_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(eem_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No broad dollar decline events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Broad Dollar Decline → Long EEM")

    avg_eem = np.mean([e["eem_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["eem_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED DTWEXBGS 6-month return crosses below -5% → long EEM 63 days",
        "mechanism": "Dollar weakness improves EM competitiveness, eases USD-denominated debt burden, and drives capital flows into EM equities",
        "source": "FRED DTWEXBGS + yfinance",
        "n_events": len(events),
        "avg_eem_return": round(avg_eem, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg EEM: {avg_eem*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["eem_63d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (DXY 6m={e['dxy_6m_return']:.1f}%): EEM {e['eem_63d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
