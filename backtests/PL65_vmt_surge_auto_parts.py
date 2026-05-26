"""PL65_vmt_surge_auto_parts — VMT +3% YoY for 3mo → Long Auto Parts Retailers
More driving = more wear = more replacement parts demand with 1-2Q lag.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL65_vmt_surge_auto_parts"
    try:
        px = load_prices(["AZO", "ORLY", "SPY"], start="2000-01-01")
        vmt = load_fred("TRFVOLUSM227NFWA", start="1998-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Monthly VMT, compute YoY growth
    vmt_m = vmt.resample("M").last().dropna()
    vmt_yoy = vmt_m.pct_change(12)

    # Find months where YoY > +3% for 3 consecutive months (first month of the 3rd consecutive)
    above_count = 0
    triggers = []
    last_trigger = None
    for i in range(len(vmt_yoy)):
        val = float(vmt_yoy.iloc[i])
        if np.isnan(val):
            above_count = 0
            continue
        if val > 0.03:
            above_count += 1
            if above_count == 3:
                trig = vmt_yoy.index[i]
                # Avoid triggering too close to previous (require 9+ months gap)
                if last_trigger is None or (trig - last_trigger).days > 270:
                    triggers.append(trig)
                    last_trigger = trig
        else:
            above_count = 0

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
        available = [t for t in ["AZO", "ORLY"] if t in ret.columns and not ret[t].iloc[window].isna().all()]
        if not available:
            continue

        basket_r = ret[available].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "vmt_yoy": round(float(vmt_yoy.loc[trig_date]), 4),
            "basket_6m_return": round(basket_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No VMT surge events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="VMT Surge → Long Auto Parts")

    avg_basket = np.mean([e["basket_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED TRFVOLUSM227NFWA YoY > +3% for 3 consecutive months → long AZO+ORLY 6mo",
        "mechanism": "More driving → more tire/brake/oil wear → replacement parts demand lags 1-2 quarters",
        "source": "FRED TRFVOLUSM227NFWA + yfinance",
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
    for e in events[:8]:
        flag = "+" if e["basket_6m_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (VMT YoY={e['vmt_yoy']*100:.1f}%): basket {e['basket_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if len(events) > 8:
        print(f"  ... and {len(events)-8} more events")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
