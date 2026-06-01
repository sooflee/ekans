"""PL735_marts_food_decel_short_casual_dining
MARTS Food Services Decel -> Short DRI/EAT/CAKE

When MARTS monthly food-services-and-drinking-places (NAICS 722) YoY
deceleration vs prior month is >= 3pp (slowdown in same-store-like growth
proxy), short equal-weight DRI + EAT + CAKE for 30 trading days from the
MARTS release date.

Data: FRED MRTSSM722USN (NSA monthly, $M). We compute YoY% each month, then
month-over-month change in YoY (deceleration). Release lag is approximately
2-3 weeks after month-end: we use the 3rd Friday of the following month as
a conservative publication date.

Source: FRED MRTSSM722USN; yfinance DRI, EAT, CAKE, SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


BASKET  = ["DRI", "EAT", "CAKE"]
HOLD_DAYS = 30
DECEL_THRESHOLD = 3.0   # pp deceleration in YoY
MIN_COOLDOWN_MONTHS = 2


def marts_release_date(data_month: pd.Timestamp) -> pd.Timestamp:
    """
    Approximate MARTS release date: 3rd Friday of the FOLLOWING month.
    This is a conservative approximation; actual advance retail sales
    (MARTS) usually release ~2-3 weeks after month-end.
    """
    # First day of next month
    if data_month.month == 12:
        next_month = pd.Timestamp(data_month.year + 1, 1, 1)
    else:
        next_month = pd.Timestamp(data_month.year, data_month.month + 1, 1)

    # Find 3rd Friday of next_month
    fridays = pd.date_range(next_month, periods=31, freq='D')
    fridays = fridays[fridays.dayofweek == 4]  # 4 = Friday
    if len(fridays) >= 3:
        return fridays[2]  # 0-indexed, so index 2 = 3rd Friday
    elif len(fridays) >= 1:
        return fridays[-1]
    return next_month + pd.Timedelta(days=14)


def main():
    sid = "PL735_marts_food_decel_short_casual_dining"
    equity_tickers = BASKET + ["SPY"]

    # Load FRED food services series
    try:
        food_df = load_fred("MRTSSM722USN", start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    food = food_df.squeeze().dropna()
    if len(food) < 24:
        return mark_failed(sid, f"insufficient FRED obs: {len(food)}")

    # Load equity prices
    try:
        px = load_prices(equity_tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in equity_tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Compute YoY % change and its MoM deceleration
    yoy  = food.pct_change(12) * 100   # YoY %
    decel = yoy.diff(-1)               # forward diff: decel[t] = yoy[t] - yoy[t+1]
                                       # wait — we want decel = yoy drop this month vs last month
    # MoM change in YoY (negative = deceleration in growth)
    yoy_change = yoy.diff(1)           # yoy_change[t] = yoy[t] - yoy[t-1]
    # Deceleration means yoy_change < -3pp (i.e., growth slowed by >= 3pp)
    decel_signal = yoy_change <= -DECEL_THRESHOLD

    # Build release dates for each data month, keep only decel months
    events_raw = []
    last_trigger_month = None
    for data_month, is_decel in decel_signal.items():
        if not is_decel or pd.isna(is_decel):
            continue
        # Cooldown: skip if within MIN_COOLDOWN_MONTHS of last trigger
        if last_trigger_month is not None:
            months_since = (
                (data_month.year - last_trigger_month.year) * 12
                + (data_month.month - last_trigger_month.month)
            )
            if months_since < MIN_COOLDOWN_MONTHS:
                continue
        release_date = marts_release_date(data_month)
        yoy_val      = float(yoy.loc[data_month]) if data_month in yoy.index else np.nan
        decel_val    = float(yoy_change.loc[data_month]) if data_month in yoy_change.index else np.nan
        events_raw.append({
            "data_month":   str(data_month.date()),
            "release_date": str(release_date.date()),
            "yoy_pct":      round(yoy_val, 2) if not np.isnan(yoy_val) else None,
            "yoy_decel_pp": round(decel_val, 2) if not np.isnan(decel_val) else None,
        })
        last_trigger_month = data_month

    print(f"Decel events (>= {DECEL_THRESHOLD}pp YoY drop): {len(events_raw)}")

    # Load equity daily returns
    ret     = daily_returns(px)
    spy_r   = ret["SPY"].fillna(0)
    idx     = ret.index
    bask_r  = ret[BASKET].fillna(0).mean(axis=1)

    pnl       = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    for ev in events_raw:
        release_dt = pd.Timestamp(ev["release_date"])
        # Enter at next session after release
        future = idx[idx > release_dt]
        if len(future) == 0:
            event_log.append({**ev, "status": "no_data_after_release"})
            continue
        entry_dt  = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos  = min(entry_pos + HOLD_DAYS, len(idx))
        exit_dt   = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        slice_bask = bask_r.iloc[entry_pos:exit_pos]
        slice_spy  = spy_r.iloc[entry_pos:exit_pos]
        gross_ev   = float((1 + (-slice_bask)).prod() - 1)
        spy_ev     = float((1 + slice_spy).prod() - 1)

        event_log.append({
            **ev,
            "entry_date":       str(entry_dt.date()),
            "exit_date":        str(exit_dt.date()),
            "n_hold_days":      int(exit_pos - entry_pos),
            "short_bask_return": round(gross_ev, 4),
            "spy_return":        round(spy_ev, 4),
            "excess_vs_spy":     round(gross_ev - spy_ev, 4),
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j]       = -bask_r.iloc[j]

    n_events = len([e for e in event_log if e.get("entry_date")])
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        return mark_failed(sid, f"insufficient held days: {len(held_pnl)}")

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="MARTS Food Decel >= 3pp → Short Casual Dining Basket (held-days)",
    )

    event_rets = [e["short_bask_return"] for e in event_log if e.get("short_bask_return") is not None]
    summary = {
        "n_events":            n_events,
        "avg_short_return":    round(float(np.mean(event_rets)), 4),
        "median_short_return": round(float(np.median(event_rets)), 4),
        "win_rate":            round(float(np.mean([r > 0 for r in event_rets])), 4),
        "best":                round(float(np.max(event_rets)), 4),
        "worst":               round(float(np.min(event_rets)), 4),
    }

    save_result(
        sid, m,
        extra={
            "status": "ok",
            "rule": (
                "When MARTS monthly food-services-and-drinking-places (NAICS 722) YoY "
                f"decelerates by >= {DECEL_THRESHOLD}pp vs prior month (released ~3rd Friday "
                "of the following month), short equal-weight DRI + EAT + CAKE at next session "
                f"open for {HOLD_DAYS} trading days."
            ),
            "mechanism": (
                "A sharp deceleration in food-services sales growth signals consumer trading "
                "down from casual dining to quick-service or at-home meals. This pressures "
                "comps for Darden (DRI: Olive Garden, LongHorn), Brinker (EAT: Chili's), and "
                "Cheesecake Factory (CAKE). Analysts revise same-store-sales and EBITDA estimates "
                "downward, causing the stocks to underperform SPY over the following 30 days."
            ),
            "source": (
                "FRED MRTSSM722USN (NAICS 722, Food Services and Drinking Places). "
                "Equity prices via yfinance DRI, EAT, CAKE, SPY. Release dates approximate "
                "using 3rd Friday of the following month."
            ),
            "caveats": (
                "MARTS NSA series has strong seasonal patterns; YoY computation removes "
                "seasonality but month-to-month YoY swings can be noisy. Release date "
                "approximation may introduce 1-3 day error. COVID months (2020) dominate "
                "the sample. CAKE has smaller market cap and lower liquidity than DRI/EAT."
            ),
            "tickers": equity_tickers,
            "basket": BASKET,
            "hold_days": HOLD_DAYS,
            "decel_threshold_pp": DECEL_THRESHOLD,
            "fred_series": "MRTSSM722USN",
            "n_events": n_events,
            "events": event_log,
            "summary": summary,
        },
        pnl=pnl[positions != 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary:  {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
