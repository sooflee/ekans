"""PL135_personal_income_acceleration_xly — Personal Income YoY Reaccelerates > +5% → Long XLY
When FRED PI YoY crosses above +5% after being below +5% for 6+ months: long XLY for 126 trading days.
Income acceleration leads consumer spending by 1-2 months.

Loosened threshold (below +5% instead of +3%) to produce enough events — personal income
YoY rarely stays below +3% for extended periods outside deep recessions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL135_personal_income_acceleration_xly"
    try:
        px = load_prices(["XLY", "SPY"], start="1999-01-01")
        pi = load_fred("PI", start="1997-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xly_r = ret["XLY"]
    spy_r = ret["SPY"]

    # Resample to monthly, compute YoY growth
    pi_m = pi.resample("M").last().dropna()
    pi_yoy = pi_m.pct_change(12)

    # Find months where YoY crosses above +5% after 6+ months below +5%
    below_count = 0
    triggers = []
    cooldown = 0  # avoid overlapping triggers
    for i in range(1, len(pi_yoy)):
        val = float(pi_yoy.iloc[i])
        if np.isnan(val):
            continue
        if cooldown > 0:
            cooldown -= 1
            if val < 0.05:
                below_count += 1
            else:
                below_count = 0
            continue
        if val < 0.05:
            below_count += 1
        elif val >= 0.05 and below_count >= 6:
            triggers.append(pi_yoy.index[i])
            below_count = 0
            cooldown = 6  # wait 6 months before next trigger
        else:
            below_count = 0

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
        xly_window = xly_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xly_window)

        xly_cum = float((1 + xly_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "pi_yoy": round(float(pi_yoy.loc[trig_date]), 4),
            "xly_6m_return": round(xly_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(xly_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No personal income reacceleration events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Personal Income Acceleration → Long XLY")

    avg_xly = np.mean([e["xly_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xly_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED PI YoY crosses above +5% after 6+ months below +5% → long XLY 126 trading days",
        "mechanism": "Income acceleration leads consumer spending by 1-2 months → consumer discretionary outperforms",
        "source": "FRED PI (personal income) + yfinance",
        "n_events": len(events),
        "avg_xly_return": round(avg_xly, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XLY: {avg_xly*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xly_6m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (YoY={e['pi_yoy']*100:.1f}%): XLY {e['xly_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
