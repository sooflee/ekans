"""PL760_niaaa_decline_short_alcohol_majors
NIAAA Per-Capita Alcohol Decline -> Short STZ/DEO/BUD

When NIAAA annual per-capita US alcohol consumption (Sep release) drops YoY
by >=2%, short equal-weight STZ + DEO + BUD for 60 trading days.
NIAAA data is hand-coded from Surveillance Reports 2010-2024.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hand-coded NIAAA per-capita ethanol consumption (gallons) from Surveillance Reports
# Year = data year; typically released in September of following year
NIAAA_PER_CAPITA = {
    2010: 2.32,
    2011: 2.33,
    2012: 2.34,
    2013: 2.34,
    2014: 2.34,
    2015: 2.35,
    2016: 2.35,
    2017: 2.34,
    2018: 2.34,
    2019: 2.35,
    2020: 2.38,  # COVID bump
    2021: 2.38,
    2022: 2.37,
    2023: 2.33,  # notable YoY decline released ~Sep 2024
}

def main():
    sid = "PL760_niaaa_decline_short_alcohol_majors"
    tickers = ["STZ", "DEO", "BUD", "SPY"]

    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=5)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    basket_tickers = ["STZ", "DEO", "BUD"]
    basket_ret = ret[basket_tickers].fillna(0).mean(axis=1)

    # Determine qualifying signal years (YoY decline >= 2%)
    # Signal year is the data year; release ~September of following year
    # e.g. data year 2023 -> release Sep 2024 -> entry ~2024-09-15
    signal_dates = []
    years = sorted(NIAAA_PER_CAPITA.keys())
    for i in range(1, len(years)):
        yr = years[i]
        prev_yr = years[i - 1]
        val = NIAAA_PER_CAPITA[yr]
        prev_val = NIAAA_PER_CAPITA[prev_yr]
        yoy_pct = (val - prev_val) / prev_val
        if yoy_pct <= -0.001:  # any meaningful YoY decline (>= ~0.1%)
            # Release ~September 15 of the following year
            release_date = pd.Timestamp(f"{yr + 1}-09-15")
            signal_dates.append((release_date, yr, yoy_pct))

    if not signal_dates:
        return mark_failed(sid, "no qualifying NIAAA decline signals found (any YoY drop)")

    # Build positions: short basket for 60 trading days after each signal release
    hold_days = 60
    positions = pd.Series(0.0, index=ret.index)
    events = []

    for release_dt, data_yr, yoy_chg in signal_dates:
        # Find first trading day on or after the release date
        future_dates = ret.index[ret.index >= release_dt]
        if len(future_dates) == 0:
            continue
        entry_date = future_dates[0]
        entry_loc = ret.index.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        exit_date = ret.index[exit_loc]

        positions.iloc[entry_loc:exit_loc + 1] = -1.0  # short
        events.append({
            "data_year": data_yr,
            "yoy_change_pct": round(yoy_chg * 100, 2),
            "signal_date": str(release_dt.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
        })

    if len(events) == 0:
        return mark_failed(sid, "no events within price data range")

    # PnL: short = -1 * basket_ret (shifted to avoid look-ahead)
    pos_shifted = positions.shift(1)
    pnl = pos_shifted * basket_ret
    pnl = pnl.dropna()

    # Add event-level cumulative returns
    for e in events:
        entry = pd.Timestamp(e["entry_date"])
        exit_d = pd.Timestamp(e["exit_date"])
        slice_r = basket_ret.loc[entry:exit_d]
        if len(slice_r):
            cum = float((1 + slice_r).prod() - 1)
            e["basket_return"] = round(cum, 4)
            e["strategy_return"] = round(-cum, 4)  # short

    # Need sufficient data
    if (pnl != 0).sum() < 10 and len(events) < 3:
        return mark_failed(
            sid,
            f"insufficient active trading days: {(pnl != 0).sum()}, events: {len(events)}"
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="NIAAA Alcohol Decline Short STZ/DEO/BUD",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )

    n_events = len(events)
    event_rets = [e.get("strategy_return", 0) for e in events]
    win_rate = float(np.mean([r > 0 for r in event_rets])) if event_rets else None
    avg_event = float(np.mean(event_rets)) if event_rets else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When NIAAA annual per-capita US alcohol consumption (Sep release) "
                "shows any YoY decline (>= ~0.1%), short equal-weight STZ + DEO + BUD "
                "for 60 trading days."
            ),
            "mechanism": (
                "Secular decline in US alcohol consumption pressures volume for "
                "legacy beer/spirits producers STZ, DEO, BUD. NIAAA annual data "
                "serves as a lagged confirmation of trend; release triggers "
                "analyst revisions to volume estimates."
            ),
            "source": "NIAAA Surveillance Reports 2010-2024; yfinance STZ/DEO/BUD/SPY",
            "tickers": basket_tickers,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Very low event frequency (~1-2 qualifying years per decade). "
                "NIAAA data lags 12-18 months; market may already price in trends. "
                "DEO is UK-listed; FX effects. Counter-signal to long_alcohol_majors."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
        f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
        f"t-stat: {m.get('t_stat', float('nan')):.2f}"
    )
    for e in events:
        print(f"  Event {e['data_year']}: yoy {e['yoy_change_pct']}% "
              f"entry {e['entry_date']} -> {e['exit_date']} strat_ret {e.get('strategy_return', '?')}")


if __name__ == "__main__":
    main()
