"""PL134_auto_loan_dq_trough_lenders — Auto Loan DQ Trough → Long ALLY+COF
When FRED DRCLACBS hits local minimum below 2.5%, long auto lenders 12mo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL134_auto_loan_dq_trough_lenders"
    try:
        px = load_prices(["ALLY", "COF", "SPY"], start="2005-01-01")
        dq = load_fred("DRCLACBS", start="1991-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Quarterly DQ rate — find local minima below 2.5%
    dq_q = dq.resample("Q").last().dropna()
    triggers = []

    for i in range(1, len(dq_q) - 1):
        v_prev = float(dq_q.iloc[i-1])
        v_curr = float(dq_q.iloc[i])
        v_next = float(dq_q.iloc[i+1])
        if np.isnan(v_curr) or np.isnan(v_prev) or np.isnan(v_next):
            continue
        if v_curr < v_prev and v_curr < v_next and v_curr < 2.5:
            # Avoid triggers too close together (require 2+ years gap)
            if not triggers or (dq_q.index[i] - triggers[-1]).days > 730:
                triggers.append(dq_q.index[i])

    events = []
    pnl_parts = []
    hold_days = 252

    for trig_date in triggers:
        # Entry = first trading day of the quarter AFTER the trough
        entry_date = trig_date + pd.DateOffset(months=3)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        # ALLY IPO'd Apr 2014; COF has longer history
        available = [t for t in ["ALLY", "COF"] if t in ret.columns and not ret[t].iloc[window].isna().all()]
        if not available:
            continue

        basket_r = ret[available].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trough_quarter": str(trig_date.date()),
            "dq_rate": round(float(dq_q.loc[trig_date]), 2),
            "entry": str(entry_idx.date()),
            "tickers_used": available,
            "basket_12m_return": round(basket_cum, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No auto loan DQ trough events below 2.5%")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Auto Loan DQ Trough → Long Lenders")

    avg_basket = np.mean([e["basket_12m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_12m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED DRCLACBS local minimum below 2.5% → long ALLY+COF 12 months",
        "mechanism": "Auto loan DQ trough = peak credit quality → provision releases + lending volume expansion → lender earnings beat",
        "source": "FRED DRCLACBS + yfinance",
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
        flag = "+" if e["basket_12m_return"] > 0 else "-"
        print(f"  {flag} {e['trough_quarter']} (DQ={e['dq_rate']}%): basket {e['basket_12m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
