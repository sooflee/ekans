"""PL58_natgas_storage_deficit_winter — NG Storage Deficit → Long NG=F Oct-Jan
NGTRSTUS not available on FRED CSV. Hand-code deficit years from EIA Weekly Natural Gas
Storage Report data where Oct 1 storage was >10% below the 5-year average.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL58_natgas_storage_deficit_winter"

    # Hand-coded from EIA Weekly Natural Gas Storage Report:
    # Years where Oct 1 working gas in storage was >10% below 5-year average
    deficit_years = [2005, 2007, 2014, 2018, 2022]
    # Surplus/normal years for comparison
    normal_years = [2006, 2008, 2009, 2010, 2011, 2012, 2013, 2015, 2016, 2017, 2019, 2020, 2021, 2023, 2024]

    print(f"Deficit years (hand-coded from EIA): {deficit_years}")
    print(f"Normal years: {normal_years}")

    try:
        px = load_prices(["NG=F", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "NG=F" not in px.columns:
        try:
            px = load_prices(["UNG", "SPY"], start="2007-01-01")
            ng_col = "UNG"
        except:
            return mark_failed(sid, "NG=F and UNG both unavailable")
    else:
        ng_col = "NG=F"

    ret = daily_returns(px)
    ng_ret = ret[ng_col]
    spy_ret = ret["SPY"]

    pnl_series = pd.Series(0.0, index=ng_ret.index)
    event_results = []
    control_results = []

    def compute_winter_return(year, ret_series):
        start = pd.Timestamp(f"{year}-10-01")
        end = pd.Timestamp(f"{year+1}-01-31")
        mask = (ret_series.index >= start) & (ret_series.index <= end)
        period = ret_series[mask]
        if len(period) < 40:
            return None, period
        return float((1 + period).prod() - 1), period

    # Deficit years (traded)
    for yr in deficit_years:
        cumret, period = compute_winter_return(yr, ng_ret)
        if cumret is None:
            continue
        mask = (pnl_series.index >= pd.Timestamp(f"{yr}-10-01")) & (pnl_series.index <= pd.Timestamp(f"{yr+1}-01-31"))
        matched = ng_ret[mask]
        if len(matched) > 0 and mask.sum() == len(matched):
            pnl_series[mask] = matched.values

        spy_cumret, _ = compute_winter_return(yr, spy_ret)
        event_results.append({
            "year": yr,
            "type": "deficit",
            "ng_oct_jan_return": round(cumret, 4),
            "spy_oct_jan_return": round(spy_cumret, 4) if spy_cumret else None,
        })

    # Normal years (comparison)
    for yr in normal_years:
        cumret, _ = compute_winter_return(yr, ng_ret)
        if cumret is None:
            continue
        control_results.append({
            "year": yr,
            "type": "normal",
            "ng_oct_jan_return": round(cumret, 4),
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no deficit year events with NG data")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="NG Storage Deficit → Long NatGas Winter")
    deficit_rets = [e["ng_oct_jan_return"] for e in event_results]
    control_rets = [e["ng_oct_jan_return"] for e in control_results]

    save_result(sid, m, extra={
        "rule": "Long NG=F Oct 1 - Jan 31 when Oct storage >10% below 5yr avg",
        "mechanism": "Thin storage cushion entering winter → prices must rise to ration demand",
        "source": "EIA Weekly NG Storage Report (hand-coded deficit years); yfinance",
        "n_deficit_events": len(event_results),
        "n_control_events": len(control_results),
        "avg_deficit_return": round(float(np.mean(deficit_rets)), 4),
        "avg_control_return": round(float(np.mean(control_rets)), 4) if control_rets else None,
        "events": event_results,
        "control_events": control_results,
    })
    print(f"Done: {len(event_results)} deficit events, avg return={np.mean(deficit_rets)*100:.2f}%")
    if control_rets:
        print(f"Control: {len(control_results)} normal events, avg return={np.mean(control_rets)*100:.2f}%")


if __name__ == "__main__":
    main()
