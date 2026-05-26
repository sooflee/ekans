"""PL40_la_nina_winter_natgas — La Nina Winter → Long Natural Gas Oct-Mar
Long NG=F Oct 1 - Mar 31 in La Nina years. Compare to El Nino/neutral years.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL40_la_nina_winter_natgas"

    # Hand-coded La Nina winters (entry year Oct)
    la_nina_years = [2007, 2008, 2010, 2011, 2016, 2017, 2020, 2021, 2022]
    # El Nino years for comparison
    el_nino_years = [2009, 2014, 2015, 2018, 2023]
    # Neutral years
    neutral_years = [2012, 2013, 2019]

    try:
        px = load_prices(["NG=F", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "NG=F" not in px.columns:
        # Try UNG as fallback
        try:
            px = load_prices(["UNG", "SPY"], start="2006-01-01")
            ng_col = "UNG"
        except Exception as e2:
            return mark_failed(sid, f"NG=F and UNG both failed: {e2}")
    else:
        ng_col = "NG=F"

    ret = daily_returns(px)
    if ng_col not in ret.columns:
        return mark_failed(sid, f"missing {ng_col} returns")

    ng_ret = ret[ng_col]
    spy_ret = ret["SPY"]

    pnl_series = pd.Series(0.0, index=ng_ret.index)
    event_results = []

    def compute_seasonal_return(year, ret_series):
        """Compute Oct 1 - Mar 31 return"""
        start = pd.Timestamp(f"{year}-10-01")
        end = pd.Timestamp(f"{year+1}-03-31")
        mask = (ret_series.index >= start) & (ret_series.index <= end)
        period = ret_series[mask]
        if len(period) < 50:
            return None, period
        return float((1 + period).prod() - 1), period

    # La Nina years
    for yr in la_nina_years:
        cumret, period = compute_seasonal_return(yr, ng_ret)
        if cumret is None:
            continue
        # Mark positions in PnL
        mask = (pnl_series.index >= pd.Timestamp(f"{yr}-10-01")) & (pnl_series.index <= pd.Timestamp(f"{yr+1}-03-31"))
        pnl_series[mask] = ng_ret[mask].values if mask.sum() == ng_ret[mask].shape[0] else 0

        spy_cumret, _ = compute_seasonal_return(yr, spy_ret)
        event_results.append({
            "year": f"{yr}-{yr+1}",
            "type": "la_nina",
            "ng_oct_mar_return": round(cumret, 4),
            "spy_oct_mar_return": round(spy_cumret, 4) if spy_cumret else None,
            "n_days": len(period),
        })

    # El Nino / neutral for comparison (not traded)
    control_results = []
    for yr in el_nino_years + neutral_years:
        cumret, period = compute_seasonal_return(yr, ng_ret)
        if cumret is None:
            continue
        cat = "el_nino" if yr in el_nino_years else "neutral"
        control_results.append({
            "year": f"{yr}-{yr+1}",
            "type": cat,
            "ng_oct_mar_return": round(cumret, 4),
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no La Nina events with data")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="La Nina Winter → Long NatGas")
    la_nina_rets = [e["ng_oct_mar_return"] for e in event_results]
    control_rets = [e["ng_oct_mar_return"] for e in control_results]

    save_result(sid, m, extra={
        "rule": "Long NG=F Oct 1 - Mar 31 in La Nina years",
        "mechanism": "La Nina → colder US winters → higher natgas heating demand",
        "source": "NOAA CPC ENSO advisories (hand-coded); yfinance",
        "n_la_nina_events": len(event_results),
        "n_control_events": len(control_results),
        "avg_la_nina_return": round(float(np.mean(la_nina_rets)), 4),
        "avg_control_return": round(float(np.mean(control_rets)), 4) if control_rets else None,
        "events": event_results,
        "control_events": control_results,
    })
    print(f"Done: {len(event_results)} La Nina events, avg return={np.mean(la_nina_rets)*100:.2f}%")
    if control_rets:
        print(f"Control: {len(control_rets)} events, avg return={np.mean(control_rets)*100:.2f}%")


if __name__ == "__main__":
    main()
