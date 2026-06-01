"""PL742 — FBI NICS YoY Acceleration -> Long RGR/SWBI

When FBI monthly NICS background-check counts show YoY growth accelerating
by >= 10pp month-over-month (consecutive months with rising YoY%), go long
equal-weight RGR + SWBI for 30 trading days from the NICS release.

FBI NICS data is public. We hand-code approximate monthly NICS totals from
the official FBI NICS monthly statistics (2010-2025). Release is on the
final business day of the following month typically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# ---------------------------------------------------------------------------
# Hand-coded FBI NICS monthly background check totals (in thousands)
# Sources: FBI NICS monthly statistics (public)
# https://www.fbi.gov/file-repository/nics_firearm_checks_month_year.pdf
# ---------------------------------------------------------------------------
NICS_MONTHLY = {
    # year_month: total_checks (thousands)
    "2010-01": 952, "2010-02": 889, "2010-03": 1036, "2010-04": 953,
    "2010-05": 853, "2010-06": 891, "2010-07": 813, "2010-08": 855,
    "2010-09": 845, "2010-10": 975, "2010-11": 1164, "2010-12": 1390,
    "2011-01": 1023, "2011-02": 981, "2011-03": 1110, "2011-04": 1019,
    "2011-05": 891, "2011-06": 955, "2011-07": 894, "2011-08": 968,
    "2011-09": 971, "2011-10": 1144, "2011-11": 1534, "2011-12": 1527,
    "2012-01": 1178, "2012-02": 1177, "2012-03": 1408, "2012-04": 1171,
    "2012-05": 1032, "2012-06": 1018, "2012-07": 1045, "2012-08": 1044,
    "2012-09": 1045, "2012-10": 1256, "2012-11": 2006, "2012-12": 2783,
    "2013-01": 2495, "2013-02": 2126, "2013-03": 2025, "2013-04": 1592,
    "2013-05": 1408, "2013-06": 1371, "2013-07": 1241, "2013-08": 1280,
    "2013-09": 1258, "2013-10": 1450, "2013-11": 1980, "2013-12": 2042,
    "2014-01": 1778, "2014-02": 1571, "2014-03": 1782, "2014-04": 1512,
    "2014-05": 1273, "2014-06": 1283, "2014-07": 1191, "2014-08": 1277,
    "2014-09": 1225, "2014-10": 1450, "2014-11": 2005, "2014-12": 2328,
    "2015-01": 1817, "2015-02": 1679, "2015-03": 1976, "2015-04": 1630,
    "2015-05": 1418, "2015-06": 1665, "2015-07": 1531, "2015-08": 1651,
    "2015-09": 1681, "2015-10": 1976, "2015-11": 2869, "2015-12": 3314,
    "2016-01": 2545, "2016-02": 2513, "2016-03": 2523, "2016-04": 2145,
    "2016-05": 1870, "2016-06": 2197, "2016-07": 1872, "2016-08": 2049,
    "2016-09": 1992, "2016-10": 2333, "2016-11": 2909, "2016-12": 2718,
    "2017-01": 1986, "2017-02": 2052, "2017-03": 2350, "2017-04": 1850,
    "2017-05": 1651, "2017-06": 1880, "2017-07": 1742, "2017-08": 1901,
    "2017-09": 1977, "2017-10": 2290, "2017-11": 2742, "2017-12": 2700,
    "2018-01": 2032, "2018-02": 1940, "2018-03": 2167, "2018-04": 1764,
    "2018-05": 1554, "2018-06": 1660, "2018-07": 1592, "2018-08": 1677,
    "2018-09": 1616, "2018-10": 1927, "2018-11": 2374, "2018-12": 2500,
    "2019-01": 2034, "2019-02": 1857, "2019-03": 2102, "2019-04": 1817,
    "2019-05": 1605, "2019-06": 1753, "2019-07": 1700, "2019-08": 2068,
    "2019-09": 1795, "2019-10": 2076, "2019-11": 2595, "2019-12": 2937,
    "2020-01": 2712, "2020-02": 2805, "2020-03": 3740, "2020-04": 2905,
    "2020-05": 3069, "2020-06": 3931, "2020-07": 3637, "2020-08": 3419,
    "2020-09": 3263, "2020-10": 3277, "2020-11": 3562, "2020-12": 3945,
    "2021-01": 4317, "2021-02": 3542, "2021-03": 4692, "2021-04": 3403,
    "2021-05": 2879, "2021-06": 3295, "2021-07": 2942, "2021-08": 2981,
    "2021-09": 2721, "2021-10": 2916, "2021-11": 3394, "2021-12": 3296,
    "2022-01": 3200, "2022-02": 2866, "2022-03": 3008, "2022-04": 2440,
    "2022-05": 2278, "2022-06": 2508, "2022-07": 2182, "2022-08": 2413,
    "2022-09": 2195, "2022-10": 2461, "2022-11": 2871, "2022-12": 3174,
    "2023-01": 2688, "2023-02": 2328, "2023-03": 2579, "2023-04": 2215,
    "2023-05": 2003, "2023-06": 2250, "2023-07": 2072, "2023-08": 2280,
    "2023-09": 2126, "2023-10": 2558, "2023-11": 2919, "2023-12": 3091,
    "2024-01": 2594, "2024-02": 2331, "2024-03": 2536, "2024-04": 2174,
    "2024-05": 2009, "2024-06": 2203, "2024-07": 2115, "2024-08": 2364,
    "2024-09": 2217, "2024-10": 2533, "2024-11": 3038, "2024-12": 3312,
    "2025-01": 2601, "2025-02": 2290, "2025-03": 2500,
}

HOLD_DAYS = 30
ACCEL_THRESHOLD_PP = 10.0   # >= 10pp YoY acceleration MoM


def main():
    sid = "PL742_nics_accel_long_rgr_swbi"

    # Load price data
    try:
        px = load_prices(["RGR", "SWBI", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    if "RGR" not in px.columns or "SWBI" not in px.columns:
        return mark_failed(sid, f"missing tickers, got: {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)

    # Build NICS monthly series
    nics = pd.Series(
        {pd.Timestamp(k + "-01"): v for k, v in NICS_MONTHLY.items()},
        dtype=float
    ).sort_index()
    # Convert to month-end index
    nics.index = nics.index + pd.offsets.MonthEnd(0)

    # Compute YoY %
    nics_yoy = nics.pct_change(12) * 100

    # Month-over-month acceleration in YoY
    nics_accel = nics_yoy.diff(1)  # accel[t] = yoy[t] - yoy[t-1]

    # Release lag: NICS data is released the final business day of the following month
    # Approximate: 5 weeks after month-end
    events = []
    last_trigger = pd.Timestamp("1900-01-01")
    cooldown_days = 45

    for dt, accel in nics_accel.items():
        if pd.isna(accel):
            continue
        if accel < ACCEL_THRESHOLD_PP:
            continue
        if (dt - last_trigger).days < cooldown_days:
            continue

        # Release date: approximately final business day of the following month (~5 weeks out)
        release_dt = dt + pd.Timedelta(days=35)

        # Find next trading session after release
        future = px.index[px.index > release_dt]
        if len(future) == 0:
            continue
        entry_dt = future[0]

        yoy_val = float(nics_yoy.loc[dt]) if dt in nics_yoy.index else np.nan
        events.append({
            "data_month": str(dt.date()),
            "release_approx": str(release_dt.date()),
            "entry_date": str(entry_dt.date()),
            "yoy_pct": round(float(nics_yoy.loc[dt]), 1) if dt in nics_yoy.index else None,
            "accel_pp": round(float(accel), 1),
        })
        last_trigger = dt

    print(f"NICS acceleration events (>= {ACCEL_THRESHOLD_PP}pp): {len(events)}")
    if not events:
        return mark_failed(sid, "no qualifying NICS acceleration events")

    # Build daily PnL
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    rgr_r = ret["RGR"]
    swbi_r = ret["SWBI"]
    pnl = pd.Series(0.0, index=ret.index)

    event_log = []
    for ev in events:
        entry_dt = pd.Timestamp(ev["entry_date"])
        if entry_dt not in ret.index:
            # find next available
            future = ret.index[ret.index >= entry_dt]
            if len(future) == 0:
                continue
            entry_dt = future[0]

        ep = ret.index.get_loc(entry_dt)
        ex = min(ep + HOLD_DAYS, len(ret))

        rgr_sl = rgr_r.iloc[ep:ex]
        swbi_sl = swbi_r.iloc[ep:ex]
        spy_sl = spy_r.iloc[ep:ex]

        basket_sl = (rgr_sl.values[:ex-ep] + swbi_sl.values[:ex-ep]) / 2.0

        # Overlap check
        overlap = (pnl.iloc[ep:ex] != 0).sum()
        if overlap > HOLD_DAYS * 0.5:
            continue

        pnl.iloc[ep:ex] = pnl.iloc[ep:ex] + basket_sl

        rgr_ret = float((1 + rgr_sl).prod() - 1)
        swbi_ret = float((1 + swbi_sl).prod() - 1)
        basket_ret = (rgr_ret + swbi_ret) / 2.0
        spy_ret = float((1 + spy_sl).prod() - 1)

        event_log.append({
            **ev,
            "exit_date": str(ret.index[ex - 1].date()),
            "rgr_return": round(rgr_ret, 4),
            "swbi_return": round(swbi_ret, 4),
            "basket_return": round(basket_ret, 4),
            "spy_return": round(spy_ret, 4),
            "excess_vs_spy": round(basket_ret - spy_ret, 4),
        })

    n_valid = len(event_log)
    print(f"Valid events: {n_valid}")
    if n_valid == 0:
        return mark_failed(sid, "no valid events after overlap filter")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FBI NICS YoY Accel Long RGR/SWBI")

    basket_rets = [e["basket_return"] for e in event_log]
    save_result(sid, m, extra={
        "rule": (
            f"When FBI NICS monthly background checks show YoY growth accelerating "
            f">= {ACCEL_THRESHOLD_PP}pp MoM, go long equal-weight RGR + SWBI for "
            f"{HOLD_DAYS} trading days from approximate release date."
        ),
        "mechanism": (
            "NICS checks are a leading indicator of firearm sales. YoY acceleration "
            "signals a demand surge (driven by fear events, policy uncertainty, or "
            "seasonal buying) that flows directly into RGR (Ruger) and SWBI "
            "(Smith & Wesson) unit sales. Analysts raise revenue estimates, and the "
            "stocks typically outperform over the following 30 days."
        ),
        "source": (
            "Hand-coded FBI NICS monthly statistics (public); "
            "yfinance RGR, SWBI, SPY."
        ),
        "caveats": (
            "NICS checks are not a perfect proxy for gun sales (denied checks included). "
            "Release date is approximated at ~5 weeks after month-end. "
            "2020-2021 COVID surge dominates the sample."
        ),
        "n_events": n_valid,
        "avg_basket_return": round(float(np.mean(basket_rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in basket_rets])), 4),
        "events": event_log,
    })

    print(f"Done: {n_valid} events")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
