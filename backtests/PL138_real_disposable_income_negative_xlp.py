"""PL138_real_disposable_income_negative_xlp — Real Disposable Income YoY Turns Negative → Long XLP 12mo
When FRED DSPIC96 YoY turns negative after 12+ months positive, long XLP for 252 days.
Thesis: real income squeeze → trade-down → staples outperform.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL138_real_disposable_income_negative_xlp"
    try:
        px = load_prices(["XLP", "SPY"], start="1998-01-01")
        dpi = load_fred("DSPIC96", start="1996-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xlp_r = ret["XLP"]
    spy_r = ret["SPY"]

    # Monthly real disposable income, YoY
    dpi_m = dpi.resample("M").last().dropna()
    dpi_yoy = dpi_m.pct_change(12)

    # Find months where YoY turns negative after 12+ months positive
    pos_count = 0
    triggers = []
    cooldown = 0
    for i in range(1, len(dpi_yoy)):
        val = float(dpi_yoy.iloc[i])
        if np.isnan(val):
            continue
        if cooldown > 0:
            cooldown -= 1
            if val >= 0:
                pos_count += 1
            else:
                pos_count = 0
            continue
        if val >= 0:
            pos_count += 1
        elif val < 0 and pos_count >= 12:
            triggers.append(dpi_yoy.index[i])
            pos_count = 0
            cooldown = 12  # wait 12 months before next trigger
        else:
            pos_count = 0

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
        xlp_window = xlp_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xlp_window)

        xlp_cum = float((1 + xlp_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "dpi_yoy": round(float(dpi_yoy.loc[trig_date]), 4),
            "xlp_12m_return": round(xlp_cum, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(xlp_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No real disposable income negative events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Real Disposable Income Turns Negative → Long XLP")

    avg_xlp = np.mean([e["xlp_12m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xlp_12m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED DSPIC96 YoY turns negative after 12+ months positive → long XLP 252 days",
        "mechanism": "Real income squeeze leads to trade-down behavior → consumer staples outperform discretionary",
        "source": "FRED DSPIC96 (real disposable income) + yfinance",
        "n_events": len(events),
        "avg_xlp_return": round(avg_xlp, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XLP: {avg_xlp*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xlp_12m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (YoY={e['dpi_yoy']*100:.1f}%): XLP {e['xlp_12m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
