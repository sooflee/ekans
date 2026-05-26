"""PL23_semi_b2b_specialty_chem — SEMI B2B Inflection → Long Specialty Chem + Equipment
When SEMI NA B2B crosses 1.05 from downcycle trough, long ENTG+AMAT+LRCX for 6 months.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL23_semi_b2b_specialty_chem"
    try:
        px = load_prices(["ENTG", "AMAT", "LRCX", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded SEMI B2B inflection dates (crosses above 1.05 from trough <0.95)
    inflections = [
        ("2013-07-15", "2013 recovery from 2012 memory/foundry downcycle"),
        ("2016-11-15", "2016 recovery from 2015-16 downcycle"),
        ("2019-12-15", "2019 recovery from 2018-19 memory downturn"),
        ("2024-03-15", "2024 recovery from 2022-23 downcycle"),
    ]

    events = []
    pnl_parts = []
    hold_days = 126  # ~6 months

    for date_str, desc in inflections:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        # Build basket from available tickers
        available = []
        for t in ["ENTG", "AMAT", "LRCX"]:
            if t in ret.columns and not ret[t].iloc[window].isna().all():
                available.append(t)
        if not available:
            continue

        basket_r = ret[available].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        per_ticker = {}
        for t in available:
            t_cum = float((1 + ret[t].iloc[window]).prod() - 1)
            per_ticker[t] = round(t_cum, 4)

        events.append({
            "inflection_date": date_str,
            "description": desc,
            "basket_6m_return": round(basket_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
            "per_ticker": per_ticker,
        })

    if not events:
        return mark_failed(sid, "No inflection events with price data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="SEMI B2B Inflection → Specialty Chem+Equip")

    avg_basket = np.mean([e["basket_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long ENTG+AMAT+LRCX equal-weight 6mo when SEMI B2B crosses 1.05 from trough",
        "mechanism": "B2B inflection signals fab reactivation → wafer starts ramp → consumable chemical demand rises linearly → specialty chem re-rates",
        "source": "SEMI B2B press releases (hand-coded inflection dates); yfinance",
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
        flag = "+" if e["basket_6m_return"] > 0 else "-"
        print(f"  {flag} {e['inflection_date']}: basket {e['basket_6m_return']*100:+.1f}%, spy {e['spy_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
