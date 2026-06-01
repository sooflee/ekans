"""PL1032_ssw_polar_vortex_sun_heating_oil
NOAA SSW Polar Vortex Displacement → Northeast Heating Oil Demand Surge → Long SUN

Entry: Sudden Stratospheric Warming (SSW) event confirmed (known NOAA CPC event dates).
       Supplementary filter: FRED DHOILNYH (NY Harbor Heating Oil spot) > 30-day MA at entry.
       Entry must be in heating season (Nov 1 – Mar 31).
Hold 30 calendar days. Stop-loss: SUN closes > 5% below entry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known NOAA CPC SSW event dates (in heating season, within SUN data window post-2012)
# Sources: Charlton & Polvani (2007) catalog, Butler et al. (2017), NOAA CPC bulletins
# Only including events with SUN data available (SUN IPO 2012-09-20)
SSW_EVENTS = [
    "2013-01-06",   # Major SSW, polar vortex displaced; led to late-Jan 2013 cold outbreak
    "2018-02-12",   # Strong SSW; led to "Beast from the East" in Europe; N. American cold lag
    "2019-01-02",   # SSW event, polar vortex disruption Jan-Feb 2019
    "2021-01-05",   # Strong SSW; led to Feb 2021 Arctic outbreak / Texas freeze
]


def main():
    sid = "PL1032_ssw_polar_vortex_sun_heating_oil"

    try:
        px = load_prices(["SUN", "SPY"], start="2012-09-20")
    except Exception as e:
        try:
            px = load_prices(["SUN", "SPY"], start="2012-09-20", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load prices: {e2}")

    try:
        fred_data = load_fred("DHOILNYH", start="2012-01-01")
    except Exception as e:
        try:
            fred_data = load_fred("DHOILNYH", start="2012-01-01", cache=False)
        except Exception as e2:
            fred_data = None
            print(f"Warning: DHOILNYH not available ({e2}); skipping heating oil filter")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["SUN", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    sun_r = ret["SUN"].dropna()
    sun_prices = px["SUN"]
    trading_dates = px.index

    # Build NY Harbor heating oil signal (30-day MA filter)
    if fred_data is not None:
        if isinstance(fred_data, pd.DataFrame):
            hoil = fred_data.iloc[:, 0].dropna()
        else:
            hoil = fred_data.dropna()
        hoil_ma30 = hoil.rolling(30, min_periods=15).mean()
        hoil_above_ma = hoil > hoil_ma30
    else:
        hoil_above_ma = None

    hold_cal_days = 30      # calendar days
    stop_loss_pct = 0.05    # 5% stop

    positions = pd.Series(0.0, index=trading_dates)
    events = []

    open_until_date = pd.Timestamp("1900-01-01")

    for event_str in SSW_EVENTS:
        event_date = pd.Timestamp(event_str)

        # Find first trading day on or after event date
        after = trading_dates[trading_dates >= event_date]
        if len(after) == 0:
            print(f"  Skipping {event_str}: no trading days after event")
            continue
        entry_date = after[0]

        # Must be in heating season (Nov 1 – Mar 31)
        month = entry_date.month
        if not (month >= 11 or month <= 3):
            print(f"  Skipping {entry_date.date()}: outside heating season (month={month})")
            continue

        # Skip if overlapping with a prior open position
        if entry_date <= open_until_date:
            print(f"  Skipping {entry_date.date()}: overlaps with prior trade (open until {open_until_date.date()})")
            continue

        # Heating oil filter: DHOILNYH > 30-day MA
        if hoil_above_ma is not None:
            hoil_signal = hoil_above_ma.asof(entry_date)
            if hoil_signal is None or (not hoil_signal):
                print(f"  Skipping {entry_date.date()}: DHOILNYH below 30-day MA (market already priced in)")
                continue

        # Entry price
        entry_price = sun_prices.get(entry_date, np.nan)
        if np.isnan(entry_price):
            print(f"  Skipping {entry_date.date()}: no SUN price")
            continue

        # Determine exit: 30 calendar days from entry, or stop-loss
        exit_cal_date = entry_date + pd.Timedelta(days=hold_cal_days)
        trade_window = trading_dates[(trading_dates >= entry_date) & (trading_dates <= exit_cal_date)]
        if len(trade_window) == 0:
            continue

        actual_exit_idx = len(trade_window) - 1
        exit_reason = "30_cal_day_hold"

        for j, td in enumerate(trade_window):
            cur_price = sun_prices.get(td, np.nan)
            if not np.isnan(cur_price) and not np.isnan(entry_price):
                chg = (cur_price - entry_price) / entry_price
                if chg <= -stop_loss_pct:
                    actual_exit_idx = j
                    exit_reason = "5pct_stop_loss"
                    break

        actual_exit_date = trade_window[actual_exit_idx]
        open_until_date = actual_exit_date

        # Mark positions
        trade_slice = trade_window[:actual_exit_idx + 1]
        positions.loc[trade_slice] = 1.0

        exit_price = sun_prices.get(actual_exit_date, np.nan)
        raw_ret = (exit_price - entry_price) / entry_price if not np.isnan(exit_price) else np.nan

        events.append({
            "ssw_event": event_str,
            "entry_date": str(entry_date.date()),
            "exit_date": str(actual_exit_date.date()),
            "exit_reason": exit_reason,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2) if not np.isnan(exit_price) else None,
            "raw_return": round(float(raw_ret), 4) if not np.isnan(raw_ret) else None,
        })
        print(f"  Trade: {entry_date.date()} → {actual_exit_date.date()} ({exit_reason}), ret={raw_ret:.2%}" if not np.isnan(raw_ret) else f"  Trade: {entry_date.date()} → {actual_exit_date.date()} ({exit_reason})")

    print(f"  Total events: {len(events)}")
    if len(events) < 2:
        return mark_failed(sid, f"insufficient events after filters: {len(events)} (SSW events={SSW_EVENTS})")

    # PnL: shift positions 1 day to avoid look-ahead
    pos_shifted = positions.shift(1).fillna(0)
    sun_r_aligned = sun_r.reindex(trading_dates).fillna(0)
    pnl_raw = pos_shifted * sun_r_aligned

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    in_pos_days = (pnl != 0).sum()
    print(f"  In-position days: {in_pos_days}")
    if in_pos_days < 10:
        return mark_failed(sid, f"insufficient in-position days: {in_pos_days}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="SSW Polar Vortex Long SUN",
        positions=abs(pos_shifted.reindex(pnl.index).fillna(0)),
        cost_bps=10,
    )

    ev_returns = [e["raw_return"] for e in events if e.get("raw_return") is not None]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Long SUN (Sunoco LP) when a Sudden Stratospheric Warming (SSW) event "
                "is confirmed by NOAA CPC during heating season (Nov–Mar), AND "
                "FRED DHOILNYH (NY Harbor heating oil spot) is above its 30-day MA "
                "(confirming market has not fully priced demand surge). "
                "Hold 30 calendar days. Stop-loss: SUN closes >5% below entry."
            ),
            "mechanism": (
                "SSW events displace the polar vortex, causing cold air outbreaks "
                "2-4 weeks later in the Northeast US (PADD 1). This boosts heating "
                "oil demand and distillate prices. SUN operates heating oil "
                "distribution in PADD 1; higher distillate pricing directly lifts "
                "SUN unit economics. The 2-4 week lag between SSW detection and "
                "surface cold air arrival provides an entry window before the "
                "full demand impact is priced."
            ),
            "source": "NOAA CPC SSW bulletins; FRED DHOILNYH; yfinance SUN/SPY",
            "tickers": ["SUN", "SPY"],
            "n_events": len(events),
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "ssw_dates_used": SSW_EVENTS,
            "caveats": (
                "Very small sample (4 SSW events in SUN data window post-2012). "
                "Event study with n=4 has low statistical power — all metrics should "
                "be interpreted with extreme caution. "
                "SSW→surface cold air lag varies (1-4 weeks); 2021 SSW led to TX "
                "freeze which disrupted SUN's own logistics. "
                "DHOILNYH is PADD 1 spot — not always a perfect proxy for SUN's "
                "realized margins (retail/distribution margins lag spot)."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {len(events)}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}  IS Sharpe: {m.get('is_sharpe'):.2f}")


if __name__ == "__main__":
    main()
