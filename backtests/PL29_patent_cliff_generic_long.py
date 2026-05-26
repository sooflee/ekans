"""PL29_patent_cliff_generic_long — Major Patent Cliff → Long Generic Pharma
Long TEVA+MYL(VTRS) Jan-Dec in patent cliff years (>$20B branded revenue expiring).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL29_patent_cliff_generic_long"
    try:
        px = load_prices(["TEVA", "VTRS", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Patent cliff years (>$20B branded revenue facing expiry)
    # 2012: Lipitor ($10B), Plavix ($7B), Seroquel ($5B) = ~$22B
    # 2016: Abilify ($7B), Crestor ($5B), plus others = ~$20B
    # 2023: Humira ($21B biosimilar launch year)
    cliff_years = [2012, 2016, 2023]
    control_years = [y for y in range(2010, 2026) if y not in cliff_years]

    def year_return(year):
        """Jan 1 - Dec 31 return."""
        # Build generic basket: TEVA available all years; VTRS from Nov 2020 (was MYL before)
        mask = ret.index.year == year
        if mask.sum() < 100:
            return None, None, None

        available = ["TEVA"]
        if year >= 2021 and "VTRS" in ret.columns:
            available.append("VTRS")

        basket_r = ret[available][mask].mean(axis=1)
        spy_window = spy_r[mask]

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        return basket_cum, spy_cum, basket_r

    events = []
    pnl_parts = []
    control_events = []

    for yr in cliff_years:
        b_ret, s_ret, daily_pnl = year_return(yr)
        if b_ret is None:
            continue
        events.append({
            "year": yr,
            "type": "cliff",
            "generic_basket_return": round(b_ret, 4),
            "spy_return": round(s_ret, 4),
            "excess": round(b_ret - s_ret, 4),
        })
        pnl_parts.append(daily_pnl)

    for yr in control_years:
        b_ret, s_ret, _ = year_return(yr)
        if b_ret is None:
            continue
        control_events.append({
            "year": yr,
            "type": "control",
            "generic_basket_return": round(b_ret, 4),
            "spy_return": round(s_ret, 4),
        })

    if not events:
        return mark_failed(sid, "No cliff year events with data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Patent Cliff → Long Generic Pharma")

    cliff_avg = np.mean([e["generic_basket_return"] for e in events])
    ctrl_avg = np.mean([e["generic_basket_return"] for e in control_events]) if control_events else 0

    save_result(sid, m, extra={
        "rule": "Long TEVA+VTRS Jan-Dec in patent cliff years (>$20B branded revenue expiring)",
        "mechanism": "Major patent cliffs → generic ANDA launches → TEVA/VTRS capture market share from branded drugs",
        "source": "FDA Orange Book patent expiry (hand-coded cliff years); yfinance",
        "n_cliff_events": len(events),
        "cliff_avg_return": round(cliff_avg, 4),
        "control_avg_return": round(ctrl_avg, 4),
        "edge_vs_control": round(cliff_avg - ctrl_avg, 4),
        "events": events,
        "control_events": control_events[:5],
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} cliff years, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Cliff avg: {cliff_avg*100:.1f}%  Control avg: {ctrl_avg*100:.1f}%  Edge: {(cliff_avg-ctrl_avg)*100:.1f}%")
    for e in events:
        flag = "+" if e["generic_basket_return"] > 0 else "-"
        print(f"  {flag} {e['year']}: generic {e['generic_basket_return']*100:+.1f}%, spy {e['spy_return']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
