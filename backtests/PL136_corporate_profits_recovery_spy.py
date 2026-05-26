"""PL136_corporate_profits_recovery_spy — Corporate Profits YoY Turns Positive After 2Q Negative → Long SPY 12mo
When FRED CP YoY turns positive after 2+ quarters of negative YoY, long SPY for 252 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL136_corporate_profits_recovery_spy"
    try:
        px = load_prices(["SPY"], start="1993-01-01")
        cp = load_fred("CP", start="1990-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # CP is quarterly — resample to quarter end, compute YoY
    cp_q = cp.resample("QE").last().dropna()
    cp_yoy = cp_q.pct_change(4)  # 4 quarters = YoY

    # Find quarters where YoY turns positive after 2+ consecutive negative quarters
    neg_count = 0
    triggers = []
    cooldown = 0
    for i in range(1, len(cp_yoy)):
        val = float(cp_yoy.iloc[i])
        if np.isnan(val):
            continue
        if cooldown > 0:
            cooldown -= 1
            if val < 0:
                neg_count += 1
            else:
                neg_count = 0
            continue
        if val < 0:
            neg_count += 1
        elif val >= 0 and neg_count >= 2:
            triggers.append(cp_yoy.index[i])
            neg_count = 0
            cooldown = 4  # wait 4 quarters before next trigger
        else:
            neg_count = 0

    events = []
    pnl_parts = []
    hold_days = 252

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(spy_window)

        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "cp_yoy": round(float(cp_yoy.loc[trig_date]), 4),
            "spy_12m_return": round(spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No corporate profits recovery events found")

    all_pnl = pd.concat(pnl_parts)
    # Benchmark is SPY buy-and-hold over same windows
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Corporate Profits Recovery → Long SPY")

    avg_spy = np.mean([e["spy_12m_return"] for e in events])
    win_count = sum(1 for e in events if e["spy_12m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED CP YoY turns positive after 2+ quarters negative → long SPY 252 days",
        "mechanism": "Corporate profit recovery signals cyclical expansion — earnings growth drives equity re-rating",
        "source": "FRED CP (corporate profits) + yfinance",
        "n_events": len(events),
        "avg_spy_return": round(avg_spy, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg SPY: {avg_spy*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["spy_12m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (YoY={e['cp_yoy']*100:.1f}%): SPY {e['spy_12m_return']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
