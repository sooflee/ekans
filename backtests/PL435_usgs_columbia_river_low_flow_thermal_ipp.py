"""PL435_usgs_columbia_river_low_flow_thermal_ipp
USGS Columbia River Streamflow Decline -> Hydro Shortfall -> Thermal IPP Revenue Surge

When USGS daily streamflow at The Dalles, OR (gauge 14105700) shows 30-day avg flow
below 25th percentile of historical June-September flows, long NRG+AES for 40 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd
import requests
import datetime as dt


def fetch_usgs_streamflow(site_no="14105700", start="2000-01-01", end=None):
    """Fetch daily mean streamflow from USGS NWIS for a given gauge."""
    cache_path = Path(__file__).resolve().parents[1] / "data" / f"usgs_{site_no}_{start}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    end = end or dt.date.today().isoformat()
    url = (
        f"https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={site_no}"
        f"&startDT={start}&endDT={end}"
        f"&parameterCd=00060&statCd=00003"  # 00060=discharge, 00003=mean
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    ts = data["value"]["timeSeries"]
    if not ts:
        raise ValueError(f"No timeSeries for site {site_no}")

    values = ts[0]["values"][0]["value"]
    records = []
    for v in values:
        try:
            flow = float(v["value"])
            if flow < 0:
                continue
            date = pd.Timestamp(v["dateTime"][:10])
            records.append({"date": date, "flow_cfs": flow})
        except (ValueError, KeyError):
            continue

    df = pd.DataFrame(records).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(cache_path)
    return df


def main():
    sid = "PL435_usgs_columbia_river_low_flow_thermal_ipp"

    # 1. Fetch USGS streamflow data for The Dalles
    try:
        flow_df = fetch_usgs_streamflow("14105700", start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"USGS data fetch: {e}")

    if len(flow_df) < 365:
        return mark_failed(sid, f"Insufficient USGS data: {len(flow_df)} rows")

    # 2. Load equity prices
    try:
        px = load_prices(["NRG", "AES", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    if "NRG" not in px.columns or px["NRG"].dropna().shape[0] < 200:
        return mark_failed(sid, "NRG price data insufficient")

    ret = daily_returns(px)

    # 3. Compute summer flow statistics (June-September only)
    flow_summer = flow_df[flow_df.index.month.isin([6, 7, 8, 9])].copy()
    flow_summer["flow_30d"] = flow_summer["flow_cfs"].rolling(30, min_periods=20).mean()

    # Compute 25th percentile of historical summer 30-day avg flows
    p25 = flow_summer["flow_30d"].quantile(0.25)
    print(f"Summer flow 30d avg - 25th pctl: {p25:,.0f} cfs, median: {flow_summer['flow_30d'].median():,.0f} cfs")

    # 4. Identify signal dates: 30d avg below 25th percentile during Jun-Sep
    low_flow_days = flow_summer[flow_summer["flow_30d"] < p25].copy()

    # Group into episodes with minimum 120-day gap between signals
    events = []
    last_signal = None
    for date in low_flow_days.index:
        if last_signal is not None and (date - last_signal).days < 120:
            continue
        events.append({"date": str(date.date()), "flow_30d_avg": float(low_flow_days.loc[date, "flow_30d"])})
        last_signal = date

    print(f"Found {len(events)} low-flow events (120-day min gap)")
    for ev in events:
        print(f"  {ev['date']}: 30d avg flow = {ev['flow_30d_avg']:,.0f} cfs")

    if len(events) < 3:
        return mark_failed(sid, f"Only {len(events)} events found (need >= 3)")

    # 5. Build event-study PnL
    nrg_ret = ret["NRG"].dropna() if "NRG" in ret.columns else None
    aes_ret = ret["AES"].dropna() if "AES" in ret.columns else None
    spy_ret = ret["SPY"].dropna()

    hold_days = 40
    pnl_series = pd.Series(0.0, index=spy_ret.index)
    positions = pd.Series(0.0, index=spy_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        # Use equal-weight NRG + AES (or NRG only if AES unavailable)
        if nrg_ret is not None and aes_ret is not None:
            combined = ret[["NRG", "AES"]].mean(axis=1).dropna()
        elif nrg_ret is not None:
            combined = nrg_ret
        else:
            continue

        mask = combined.index > entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = combined.index[mask][0]
        pos = combined.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(combined))

        event_rets = combined.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        # SPY same period
        spy_mask = spy_ret.index >= start_idx
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        # Fill into daily series
        for i in range(pos, end_pos):
            idx = combined.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = combined.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "portfolio_40d_return": round(cumret, 4),
            "spy_40d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    # 6. Compute metrics
    port_rets = np.array([e["portfolio_40d_return"] for e in ok_events])
    spy_rets = np.array([e["spy_40d_return"] for e in ok_events if e.get("spy_40d_return") is not None])
    excess_rets = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])

    avg_ret = float(port_rets.mean())
    avg_excess = float(excess_rets.mean()) if len(excess_rets) > 0 else None
    win_rate = float((port_rets > 0).mean())
    excess_win = float((excess_rets > 0).mean()) if len(excess_rets) > 0 else None

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="Columbia River Low Flow -> Thermal IPP", positions=positions[positions != 0])
    else:
        vol = float(in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0
        m = {
            "name": "Columbia River Low Flow -> Thermal IPP",
            "n_days": len(in_pos),
            "n_events": len(ok_events),
            "avg_event_return": round(avg_ret, 4),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "ann_vol": vol,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When USGS Columbia River at The Dalles 30d avg flow < 25th pctl (Jun-Sep), long NRG+AES 40d",
        "mechanism": "Low Columbia River flow -> hydro shortfall -> thermal power prices spike -> NRG/AES revenue surge",
        "source": "USGS NWIS gauge 14105700 + yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
        "excess_win_rate": round(excess_win, 4) if excess_win is not None else None,
        "flow_25th_pctl_cfs": float(p25),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"\nDone: {len(ok_events)} events")
    print(f"  Avg 40d return: {avg_ret*100:.2f}%")
    print(f"  Avg excess vs SPY: {avg_excess*100:.2f}%" if avg_excess else "  No excess computed")
    print(f"  Win rate: {win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in event_results:
        status = e["status"]
        if status == "ok":
            print(f"  {e['date']}: portfolio={e['portfolio_40d_return']*100:+.1f}%, SPY={e.get('spy_40d_return','N/A')}, excess={e.get('excess_return','N/A')}")
        else:
            print(f"  {e['date']}: {status} - {e.get('reason','')}")


if __name__ == "__main__":
    main()
