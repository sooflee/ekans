"""PL64_bank_credit_reacceleration_xlf — Bank Credit YoY Reaccelerates → Long XLF
When FRED TOTBKCR YoY crosses above +2% after 6+ months below, long XLF 12mo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL64_bank_credit_reacceleration_xlf"
    try:
        px = load_prices(["XLF", "SPY"], start="1999-01-01")
        bkcr = load_fred("TOTBKCR", start="1997-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xlf_r = ret["XLF"]
    spy_r = ret["SPY"]

    # Resample to monthly, compute YoY growth
    bkcr_m = bkcr.resample("M").last().dropna()
    bkcr_yoy = bkcr_m.pct_change(12)

    # Find months where YoY crosses above +2% after 6+ months below +2%
    below_count = 0
    triggers = []
    for i in range(1, len(bkcr_yoy)):
        val = float(bkcr_yoy.iloc[i])
        if np.isnan(val):
            continue
        if val < 0.02:
            below_count += 1
        elif val >= 0.02 and below_count >= 6:
            triggers.append(bkcr_yoy.index[i])
            below_count = 0
        else:
            below_count = 0

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
        xlf_window = xlf_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xlf_window)

        xlf_cum = float((1 + xlf_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "bkcr_yoy": round(float(bkcr_yoy.loc[trig_date]), 4),
            "xlf_12m_return": round(xlf_cum, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(xlf_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No bank credit reacceleration events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Bank Credit Reacceleration → Long XLF")

    avg_xlf = np.mean([e["xlf_12m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xlf_12m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED TOTBKCR YoY crosses above +2% after 6+ months below → long XLF 12 months",
        "mechanism": "Credit expansion = banks lending again → NIM improves → loan growth drives net interest income → banks re-rate",
        "source": "FRED TOTBKCR (H.8 bank credit) + yfinance",
        "n_events": len(events),
        "avg_xlf_return": round(avg_xlf, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XLF: {avg_xlf*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xlf_12m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (YoY={e['bkcr_yoy']*100:.1f}%): XLF {e['xlf_12m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
