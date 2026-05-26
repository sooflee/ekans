"""PL71_consumer_credit_acceleration_xly — Consumer Credit YoY Reaccelerates > +6% → Long XLY
When FRED TOTALSL YoY crosses above +6% after 6+ months below +3%, long XLY 126 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL71_consumer_credit_acceleration_xly"
    try:
        px = load_prices(["XLY", "SPY"], start="1998-01-01")
        credit = load_fred("TOTALSL", start="1996-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xly_r = ret["XLY"]
    spy_r = ret["SPY"]

    # Monthly consumer credit, compute YoY growth
    credit_m = credit.resample("ME").last().dropna()
    credit_yoy = credit_m.pct_change(12)

    # Find months where YoY crosses above +6% after 6+ months below +3%
    # Track consecutive months below 3%, and once we've had 6+, wait for the 6% cross
    below_count = 0
    had_six_below = False
    triggers = []
    last_trigger = None
    for i in range(len(credit_yoy)):
        val = float(credit_yoy.iloc[i])
        if np.isnan(val):
            continue
        if val < 0.03:
            below_count += 1
            if below_count >= 6:
                had_six_below = True
        elif had_six_below and val >= 0.06:
            trig = credit_yoy.index[i]
            if last_trigger is None or (trig - last_trigger).days > 300:
                triggers.append(trig)
                last_trigger = trig
            had_six_below = False
            below_count = 0
        elif val >= 0.03:
            # Above 3% but below 6% — don't reset the "had 6 below" flag,
            # just stop counting
            pass

    print(f"Found {len(triggers)} consumer credit acceleration events")

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
            "credit_yoy": round(float(credit_yoy.loc[trig_date]), 4),
            "xly_6m_return": round(xly_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(xly_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No consumer credit acceleration events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Consumer Credit Accel → Long XLY")

    avg_xly = np.mean([e["xly_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xly_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED TOTALSL YoY crosses above +6% after 6+ months below +3% → long XLY 6mo",
        "mechanism": "Consumers borrowing again at above-trend rates → discretionary spending recovery",
        "source": "FRED TOTALSL + yfinance",
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
        print(f"  {flag} {e['trigger_date']} (YoY={e['credit_yoy']*100:.1f}%): XLY {e['xly_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
