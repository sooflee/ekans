"""PL149_rescons_inflection_plumbing_hvac — Residential Construction 3 Positive MoM After Decline → Long WSO+WMS 63d
When FRED TLRESCONS (or proxy) shows 3 consecutive positive MoM after 2+ negative months, long WSO+WMS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL149_rescons_inflection_plumbing_hvac"
    try:
        px = load_prices(["WSO", "WMS", "SPY"], start="2000-01-01")
        # Try TLRESCONS first, fall back to PRRESCONS
        try:
            rescons = load_fred("TLRESCONS", start="1998-01-01").squeeze()
        except:
            try:
                rescons = load_fred("PRRESCONS", start="1998-01-01").squeeze()
            except:
                return mark_failed(sid, "Neither TLRESCONS nor PRRESCONS available on FRED")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["WSO", "WMS"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No plumbing/HVAC tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly construction spending, MoM
    res_m = rescons.resample("M").last().dropna()
    res_mom = res_m.pct_change()

    # Find 3 positive MoM after 2+ negative MoM
    neg_count = 0
    pos_count = 0
    triggers = []
    cooldown = 0
    for i in range(1, len(res_mom)):
        val = float(res_mom.iloc[i])
        if np.isnan(val):
            neg_count = 0
            pos_count = 0
            continue
        if cooldown > 0:
            cooldown -= 1
            neg_count = 0
            pos_count = 0
            continue
        if val < 0:
            neg_count += 1
            pos_count = 0
        elif val > 0:
            if neg_count >= 2:
                pos_count += 1
                if pos_count == 3:
                    triggers.append(res_mom.index[i])
                    neg_count = 0
                    pos_count = 0
                    cooldown = 3
            else:
                neg_count = 0
                pos_count = 0
        else:
            pos_count = 0

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
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "basket_63d_return": round(bask_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No residential construction inflection events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Residential Construction Inflection → Long WSO+WMS")

    avg_basket = np.mean([e["basket_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED TLRESCONS 3 positive MoM after 2+ negative months → long WSO+WMS 63 days",
        "mechanism": "Residential construction restart benefits plumbing/HVAC distributors first — they supply materials before framing trades",
        "source": "FRED TLRESCONS + yfinance",
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
        flag = "+" if e["basket_63d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_63d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
