"""PL737 — Restaurant SSS Breadth Divergence -> Short XLY (Counter)

When trailing-3m restaurant same-store-sales (DRI+CMG+MCD composite) is
accelerating >= +200 bps QoQ AND XLY breadth (% of major XLY components
above their 50-day MA) is below 40%, short XLY for 30 trading days.

Counter-signal to long_SPY (XLY-as-SPY-proxy divergence).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# ---------------------------------------------------------------------------
# Hand-coded quarterly restaurant SSS data (% same-store sales YoY)
# Sources: DRI, CMG, MCD quarterly earnings releases 2015-2025
# Format: (year, quarter_end_month, DRI_sss, CMG_sss, MCD_sss)
# ---------------------------------------------------------------------------
SSS_DATA = [
    # year, qtr_end_month, DRI, CMG, MCD
    (2015, 2,  2.2,  4.3,  0.7),
    (2015, 5,  2.1,  4.8,  0.3),
    (2015, 8,  2.0,  3.5, -0.5),
    (2015, 11, 1.8,  2.9, -1.0),
    (2016, 2,  2.3,  0.8,  2.7),
    (2016, 5,  2.2, -1.3,  3.5),
    (2016, 8,  2.1, -0.8,  3.8),
    (2016, 11, 2.4,  0.2,  1.3),
    (2017, 2,  1.8,  14.7, 6.2),
    (2017, 5,  1.2,  17.8, 3.9),
    (2017, 8,  1.1,  11.1, 3.5),
    (2017, 11, 0.9,  15.7, 5.0),
    (2018, 2,  1.2,  11.1, 2.0),
    (2018, 5,  1.0,  11.0, 2.5),
    (2018, 8,  0.3,  9.9,  2.4),
    (2018, 11, 0.7,  6.1,  4.2),
    (2019, 2,  1.3,  9.9,  3.4),
    (2019, 5,  2.2,  10.0, 5.7),
    (2019, 8,  1.9,  7.0,  5.9),
    (2019, 11, 2.5,  3.0,  5.1),
    (2020, 2, -20.0, -8.1, -3.4),
    (2020, 5, -30.0,-16.4,-24.3),
    (2020, 8, -11.9, 8.3, -2.2),
    (2020, 11, -8.0, 11.6, -1.0),
    (2021, 2,  4.7,  11.8, -5.0),
    (2021, 5,  84.9, 48.5, 25.9),
    (2021, 8,  12.2, 31.9, 14.9),
    (2021, 11,  8.2, 22.0,  8.7),
    (2022, 2,  6.2,  17.2, 11.8),
    (2022, 5,  3.1,  10.1, 9.7),
    (2022, 8,  1.6,   8.0, 9.5),
    (2022, 11, 4.3,   8.4, 8.6),
    (2023, 2,  5.7,  12.1, 12.6),
    (2023, 5,  2.7,   7.4,  8.7),
    (2023, 8,  1.4,   7.4,  8.8),
    (2023, 11, 3.0,   8.0,  8.9),
    (2024, 2,  0.8,   5.4,  2.5),
    (2024, 5, -0.5,   4.3,  0.7),
    (2024, 8, -1.1,   6.0, -1.0),
    (2024, 11, 0.6,   7.4,  0.3),
    (2025, 2,  1.3,   9.3,  3.5),
]

# Major XLY components for breadth computation
XLY_COMPONENTS = [
    "AMZN", "TSLA", "MCD", "HD", "NKE",
    "LOW", "TGT", "SBUX", "TJX", "BKNG",
    "F", "GM", "ROST", "YUM", "CMG",
    "DRI", "ORLY", "AZO", "DG", "DLTR",
]

HOLD_DAYS = 30
BREADTH_THRESHOLD = 0.40   # below 40% breadth
SSS_ACCEL_BPS = 200        # >=200 bps QoQ acceleration


def build_sss_series() -> pd.Series:
    """Build a monthly composite SSS series (equal-weight average DRI+CMG+MCD)."""
    rows = []
    for year, month, dri, cmg, mcd in SSS_DATA:
        dt = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
        composite = (dri + cmg + mcd) / 3.0
        rows.append((dt, composite))
    s = pd.Series({d: v for d, v in rows}, name="sss_composite")
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def main():
    sid = "PL737_restaurant_sss_breadth_short_xly"

    # Load price data
    all_tickers = ["XLY", "SPY"] + XLY_COMPONENTS
    try:
        px = load_prices(all_tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Only keep tickers that loaded successfully
    available = [t for t in all_tickers if t in px.columns]
    comp_avail = [t for t in XLY_COMPONENTS if t in available]
    print(f"Loaded {len(available)} tickers; {len(comp_avail)} XLY components available")

    if "XLY" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "XLY or SPY missing from price data")

    px = px.sort_index().ffill(limit=5)

    # Build SSS series
    sss = build_sss_series()

    # Compute trailing-3m SSS acceleration (QoQ delta in composite SSS)
    # trailing_3m = rolling 3-quarter mean
    sss_trail = sss.rolling(3).mean()
    sss_trail_prev = sss_trail.shift(1)  # prior quarter's trailing 3m
    sss_accel_bps = (sss_trail - sss_trail_prev) * 100  # convert pp to bps

    # Build daily SSS acceleration (forward-fill from quarterly to monthly)
    sss_accel_daily = sss_accel_bps.resample("D").ffill().reindex(px.index, method="ffill")

    # Compute XLY breadth: % of component tickers above their 50-day MA (month-end)
    breadth_list = []
    if len(comp_avail) >= 5:
        comp_px = px[comp_avail].dropna(how="all")
        ma50 = comp_px.rolling(50, min_periods=30).mean()
        above_ma = (comp_px > ma50).astype(float)
        breadth = above_ma.mean(axis=1)  # daily breadth fraction
    else:
        # Fallback: use XLY itself (always 50% threshold to avoid empty)
        return mark_failed(sid, f"insufficient XLY components loaded: {len(comp_avail)}")

    # Resample breadth to month-end for signal generation
    breadth_monthly = breadth.resample("ME").last()

    # Resample SSS accel to month-end
    sss_monthly = sss_accel_daily.resample("ME").last()

    # Build event list: both conditions met at month-end
    common_idx = sss_monthly.index.intersection(breadth_monthly.index)
    events = []
    last_trigger = pd.Timestamp("1900-01-01")

    for dt in common_idx:
        sss_val = sss_monthly.loc[dt]
        br_val = breadth_monthly.loc[dt]
        if pd.isna(sss_val) or pd.isna(br_val):
            continue
        if sss_val >= SSS_ACCEL_BPS and br_val < BREADTH_THRESHOLD:
            # cooldown: no more than 1 trigger per 45 days
            if (dt - last_trigger).days < 45:
                continue
            # Entry: next trading session after month-end
            future = px.index[px.index > dt]
            if len(future) == 0:
                continue
            entry_dt = future[0]
            events.append({
                "signal_month": str(dt.date()),
                "entry_date": str(entry_dt.date()),
                "sss_accel_bps": round(float(sss_val), 1),
                "breadth_pct": round(float(br_val), 3),
            })
            last_trigger = dt

    print(f"Events: {len(events)}")
    if not events:
        return mark_failed(sid, "no qualifying events (SSS accel>=200bps AND breadth<40%)")

    # Build daily PnL
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    xly_r = ret["XLY"]
    pnl = pd.Series(0.0, index=ret.index)

    event_log = []
    for ev in events:
        entry_dt = pd.Timestamp(ev["entry_date"])
        if entry_dt not in ret.index:
            continue
        ep = ret.index.get_loc(entry_dt)
        ex = min(ep + HOLD_DAYS, len(ret))

        xly_slice = xly_r.iloc[ep:ex]
        spy_slice = spy_r.iloc[ep:ex]

        xly_ret = float((1 + xly_slice).prod() - 1)
        spy_ret = float((1 + spy_slice).prod() - 1)
        short_ret = -xly_ret

        # Check overlap with existing positions
        overlap = (pnl.iloc[ep:ex] != 0).sum()
        if overlap > HOLD_DAYS * 0.5:
            continue

        pnl.iloc[ep:ex] = pnl.iloc[ep:ex] + (-xly_r.iloc[ep:ex].values[:ex-ep])

        event_log.append({
            **ev,
            "exit_date": str(ret.index[ex - 1].date()),
            "short_xly_return": round(short_ret, 4),
            "spy_return": round(spy_ret, 4),
            "excess_vs_spy": round(short_ret - spy_ret, 4),
        })

    n_valid = len(event_log)
    print(f"Valid events: {n_valid}")
    if n_valid == 0:
        return mark_failed(sid, "no valid events after overlap filter")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Restaurant SSS Breadth Divergence Short XLY")

    event_rets = [e["short_xly_return"] for e in event_log]
    save_result(sid, m, extra={
        "rule": (
            "When trailing-3m restaurant SSS composite (DRI+CMG+MCD) accelerates "
            f">= {SSS_ACCEL_BPS} bps QoQ AND XLY breadth (% components above 50d MA) "
            f"is below {BREADTH_THRESHOLD*100:.0f}%, short XLY for {HOLD_DAYS} trading days."
        ),
        "mechanism": (
            "Divergence between restaurant SSS momentum (consumer eating-out strength) "
            "and weak XLY technical breadth signals that XLY's consumer discretionary "
            "components are broadly deteriorating despite isolated restaurant strength. "
            "The ETF acts as a de-facto SPY proxy and the short is a counter to long-SPY."
        ),
        "source": "Hand-coded DRI/CMG/MCD quarterly SSS; yfinance XLY/SPY/components",
        "caveats": (
            "Hand-coded SSS data introduces look-ahead bias potential; XLY is dominated "
            "by AMZN/TSLA so breadth may lag fundamental signals; counter-signal to long_SPY."
        ),
        "n_events": n_valid,
        "avg_short_return": round(float(np.mean(event_rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in event_rets])), 4),
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
