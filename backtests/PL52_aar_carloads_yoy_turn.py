"""PL52_aar_carloads_yoy_turn — AAR Carloads YoY Positive Turn -> Long XLI
Since AAR weekly data not on FRED, proxy with INDPRO monthly.
When FRED INDPRO YoY turns positive after 6+ months negative, long XLI 6 months.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL52_aar_carloads_yoy_turn"
    try:
        indpro = load_fred("INDPRO", start="1990-01-01")
        px = load_prices(["XLI", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xli_r = ret["XLI"]
    spy_r = ret["SPY"]

    # Compute YoY change in INDPRO
    indpro_s = indpro["INDPRO"].dropna()
    yoy = indpro_s.pct_change(12)  # 12-month (YoY) change
    yoy = yoy.dropna()

    # Find months where YoY turns positive after 6+ months negative
    crossing_dates = []
    consecutive_neg = 0
    for i in range(1, len(yoy)):
        if yoy.iloc[i - 1] < 0:
            consecutive_neg += 1
        else:
            consecutive_neg = 0

        if yoy.iloc[i] > 0 and yoy.iloc[i - 1] < 0 and consecutive_neg >= 6:
            crossing_dates.append(yoy.index[i])
            consecutive_neg = 0

    if not crossing_dates:
        return mark_failed(sid, "No INDPRO YoY positive turns found after 6+ months negative")

    events = []
    pnl_parts = []
    hold_days = 126  # 6 months

    for cross_date in crossing_dates:
        # Entry on first trading day of the month after the crossing
        entry_month_start = cross_date + pd.offsets.MonthBegin(1)
        entry_mask = ret.index >= entry_month_start
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        if xli_r.iloc[window].isna().all():
            continue

        xli_window = xli_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xli_window)

        xli_cum = float((1 + xli_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "crossing_date": str(cross_date.date()),
            "entry_date": str(entry_idx.date()),
            "xli_6m_return": round(xli_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(xli_cum - spy_cum, 4),
            "indpro_yoy_at_cross": round(float(yoy.loc[cross_date]), 4),
        })

    if not events:
        return mark_failed(sid, "No INDPRO turning events with XLI price data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="INDPRO YoY Turn Positive -> Long XLI")

    avg_xli = np.mean([e["xli_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xli_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long XLI 6mo when FRED INDPRO YoY turns positive after 6+ months negative",
        "mechanism": "Industrial production recovery -> rail carloads inflect -> industrials re-rate as earnings estimates rise",
        "source": "FRED INDPRO (proxy for AAR carloads YoY); yfinance XLI",
        "n_events": len(events),
        "avg_xli_return": round(avg_xli, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XLI: {avg_xli*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xli_6m_return"] > 0 else "-"
        print(f"  {flag} {e['crossing_date']} (entry {e['entry_date']}): XLI {e['xli_6m_return']*100:+.1f}%, SPY {e['spy_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
