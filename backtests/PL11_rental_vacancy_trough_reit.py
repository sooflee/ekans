"""PL11_rental_vacancy_trough_reit — Rental Vacancy Trough → Long Apartment REITs
When vacancy bottoms, peak rents flow through to NOI with 1-2Q lag.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL11_rental_vacancy_trough_reit"
    try:
        px = load_prices(["EQR", "AVB", "MAA", "SPY"], start="2005-01-01")
        vac = load_fred("RRVRUSQ156N", start="2004-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    reit_basket = (ret["EQR"] + ret["AVB"] + ret["MAA"]) / 3
    spy_r = ret["SPY"]

    # Find vacancy troughs: lower than both neighbors and < 5.5%
    vac_q = vac.resample("Q").last().dropna()
    events = []
    pnl_parts = []
    hold_days = 252

    for i in range(1, len(vac_q) - 1):
        v_prev, v_curr, v_next = float(vac_q.iloc[i-1]), float(vac_q.iloc[i]), float(vac_q.iloc[i+1])
        if np.isnan(v_curr) or v_curr >= v_prev or v_curr >= v_next or v_curr >= 5.5:
            continue

        # Entry: first trading day of the NEXT quarter
        trough_date = vac_q.index[i]
        entry_date = trough_date + pd.DateOffset(months=3)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        daily_pnl = reit_basket.iloc[window]
        pnl_parts.append(daily_pnl)

        cum_ret = float((1 + daily_pnl).prod() - 1)
        spy_cum = float((1 + spy_r.iloc[window]).prod() - 1)
        events.append({
            "trough_quarter": str(trough_date.date()),
            "vacancy_rate": round(v_curr, 2),
            "entry": str(ret.index[entry_loc].date()),
            "reit_12m_return": round(cum_ret, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(cum_ret - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No vacancy trough events found with vacancy < 5.5%")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Vacancy Trough → Apartment REITs")
    save_result(sid, m, extra={
        "rule": "When FRED RRVRUSQ156N hits local min < 5.5%: long EQR+AVB+MAA 12 months",
        "mechanism": "Vacancy trough = peak rent growth already locked in; flows to NOI with 1-2Q lag",
        "source": "FRED RRVRUSQ156N + yfinance",
        "n_events": len(events),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
