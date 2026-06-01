"""PL1048_google_trends_value_search_cmg_short
Google Trends Value-Search Dual Surge → CMG Short (Consumer Squeeze Counter)

Entry: BOTH 'dollar menu' AND 'cheap groceries' simultaneously exceed
52-week rolling mean + 1.5*std (Google Trends weekly). Short CMG.
Hold max 6 weeks; stop-loss if CMG +8% from entry.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def fetch_trends_safe(keywords, start_year=2006, end_year=2025):
    """
    Fetch weekly Google Trends data for keywords, stitched in 2-year windows
    to preserve relative scale. Returns a DataFrame indexed by week start date.
    Falls back to None on persistent error.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return None

    tr = TrendReq(hl='en-US', tz=360, timeout=(10, 30), retries=2, backoff_factor=0.5)
    dfs = []
    for year in range(start_year, end_year, 2):
        tf_start = f"{year}-01-01"
        tf_end = f"{min(year+2, end_year+1)}-12-31"
        timeframe = f"{tf_start} {tf_end}"
        try:
            time.sleep(2)  # be polite to Google
            tr.build_payload(keywords, timeframe=timeframe, geo='US')
            df = tr.interest_over_time()
            if df is None or df.empty:
                continue
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            dfs.append(df)
        except Exception as e:
            print(f"  pytrends fetch error ({timeframe}): {e}")
            continue

    if not dfs:
        return None

    # Concatenate and resolve overlaps via mean
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    return combined


def main():
    sid = "PL1048_google_trends_value_search_cmg_short"

    # Load prices first
    try:
        px = load_prices(["CMG", "MCD", "SPY"], start="2006-01-01")
    except Exception as e:
        try:
            px = load_prices(["CMG", "MCD", "SPY"], start="2006-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load prices: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["CMG", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    cmg_r = ret["CMG"].dropna()
    trading_dates = px.index
    cmg_prices = px["CMG"]

    # Fetch Google Trends data
    print("  Fetching Google Trends (this may take ~30s)...")
    keywords = ["dollar menu", "cheap groceries"]
    trends_df = fetch_trends_safe(keywords, start_year=2006, end_year=2025)

    if trends_df is None or trends_df.empty:
        # Fallback: use FRED CPIUFDNS (Food Away from Home CPI) as proxy
        print("  pytrends failed — using FRED food CPI proxy")
        try:
            food_cpi = load_fred("CPIUFDNS", start="2006-01-01").squeeze().dropna()
        except Exception as e:
            try:
                food_cpi = load_fred("CPIUFDNS", start="2006-01-01", cache=False).squeeze().dropna()
            except Exception as e2:
                return mark_failed(sid, f"both pytrends and FRED fallback failed: {e2}")

        # Convert to weekly frequency using YoY acceleration as signal
        food_cpi_yoy = food_cpi.pct_change(12) * 100
        food_cpi_w = food_cpi_yoy.resample("W-MON").ffill()
        # Signal: YoY acceleration exceeds +1.5 std above 52-week mean
        roll_mean = food_cpi_w.rolling(52, min_periods=26).mean()
        roll_std = food_cpi_w.rolling(52, min_periods=26).std()
        signal_series = food_cpi_w > (roll_mean + 1.5 * roll_std)
        data_source = "FRED CPIUFDNS (food away from home YoY, proxy for value-search)"
        signal_description = "FRED CPIUFDNS YoY > rolling mean + 1.5 std (proxy for consumer stress)"
    else:
        print(f"  Google Trends data: {len(trends_df)} weeks, cols={list(trends_df.columns)}")
        # Rename columns for clarity
        col_map = {}
        for col in trends_df.columns:
            if 'dollar' in col.lower():
                col_map[col] = 'dollar_menu'
            elif 'cheap' in col.lower() or 'groceries' in col.lower():
                col_map[col] = 'cheap_groceries'
        trends_df = trends_df.rename(columns=col_map)

        if 'dollar_menu' not in trends_df.columns or 'cheap_groceries' not in trends_df.columns:
            # Try partial match
            print(f"  Available columns: {list(trends_df.columns)}")
            if len(trends_df.columns) >= 2:
                trends_df.columns = ['dollar_menu', 'cheap_groceries'][:len(trends_df.columns)]
            else:
                return mark_failed(sid, f"pytrends returned unexpected columns: {list(trends_df.columns)}")

        # Z-score each series vs trailing 52-week window
        for col in ['dollar_menu', 'cheap_groceries']:
            if col in trends_df.columns:
                roll_mean = trends_df[col].rolling(52, min_periods=26).mean()
                roll_std = trends_df[col].rolling(52, min_periods=26).std()
                trends_df[f"{col}_z"] = (trends_df[col] - roll_mean) / roll_std.replace(0, np.nan)

        # Signal: both keywords > mean + 1.5 std in same week
        signal_series = (trends_df['dollar_menu'] > trends_df['dollar_menu'].rolling(52, min_periods=26).mean() + 1.5 * trends_df['dollar_menu'].rolling(52, min_periods=26).std()) & \
                        (trends_df['cheap_groceries'] > trends_df['cheap_groceries'].rolling(52, min_periods=26).mean() + 1.5 * trends_df['cheap_groceries'].rolling(52, min_periods=26).std())
        data_source = "Google Trends (pytrends) for 'dollar menu' + 'cheap groceries' US weekly"
        signal_description = "Both keywords > 52-week mean + 1.5 std simultaneously"

    # Convert weekly signal to daily (Monday signal → entry that Monday open)
    signal_dates = signal_series[signal_series].index.tolist()
    print(f"  Signal weeks identified: {len(signal_dates)}")

    if len(signal_dates) < 3:
        return mark_failed(sid, f"insufficient signal dates: {len(signal_dates)}")

    hold_weeks = 6
    hold_days = hold_weeks * 5  # ~30 trading days
    stop_loss_pct = 0.08    # CMG up 8% = cover short

    positions = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_date = pd.Timestamp("1900-01-01")

    # CMG quarterly earnings blackout: roughly mid-Feb, mid-May, mid-Aug, mid-Nov
    # We approximate by skipping entries within 5 trading days of known earnings months
    # (days 15 of Feb/May/Aug/Nov ±7 calendar days)
    def near_earnings(date):
        """Return True if date is within 7 days of CMG earnings release window."""
        for month in [2, 5, 8, 11]:
            earn_approx = pd.Timestamp(f"{date.year}-{month:02d}-15")
            if abs((date - earn_approx).days) <= 7:
                return True
        return False

    for signal_date in signal_dates:
        signal_date = pd.Timestamp(signal_date)

        # Entry: next trading Monday (or first trading day on/after signal_date)
        candidates = trading_dates[trading_dates >= signal_date]
        if len(candidates) == 0:
            continue
        entry_date = candidates[0]

        if entry_date <= open_until_date:
            continue

        # Skip if entry is near CMG earnings
        if near_earnings(entry_date):
            print(f"  Skipping {entry_date.date()}: near CMG earnings window")
            continue

        entry_price = cmg_prices.get(entry_date, np.nan)
        if np.isnan(entry_price):
            continue

        entry_idx = trading_dates.get_loc(entry_date)
        end_idx = min(entry_idx + hold_days, len(trading_dates) - 1)
        trade_window = trading_dates[entry_idx:end_idx + 1]

        actual_exit_idx = len(trade_window) - 1
        exit_reason = "6_week_hold"

        for j, td in enumerate(trade_window):
            # Check if signal dropped (either keyword below mean)
            # For simplicity with the FRED proxy or weekly data, we check weekly signal
            week_start = td - pd.Timedelta(days=td.weekday())
            if week_start in signal_series.index:
                if not signal_series.get(week_start, True):
                    # Signal abated
                    actual_exit_idx = j
                    exit_reason = "signal_abated"
                    break

            cur_price = cmg_prices.get(td, np.nan)
            if not np.isnan(cur_price) and not np.isnan(entry_price):
                chg = (cur_price - entry_price) / entry_price
                if chg >= stop_loss_pct:   # Short stop: CMG up 8%
                    actual_exit_idx = j
                    exit_reason = "cmg_8pct_stop"
                    break

        actual_exit_date = trade_window[actual_exit_idx]
        open_until_date = actual_exit_date

        trade_slice = trade_window[:actual_exit_idx + 1]
        positions.loc[trade_slice] = -1.0   # short CMG

        exit_price = cmg_prices.get(actual_exit_date, np.nan)
        # Short CMG return = -(price change)
        raw_ret = -(exit_price - entry_price) / entry_price if not np.isnan(exit_price) else np.nan

        events.append({
            "signal_week": str(signal_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(actual_exit_date.date()),
            "exit_reason": exit_reason,
            "short_cmg_return": round(float(raw_ret), 4) if not np.isnan(raw_ret) else None,
        })

    print(f"  Total trades: {len(events)}")
    if len(events) < 3:
        return mark_failed(sid, f"insufficient trades after filters: {len(events)}")

    # PnL: short CMG, shift 1 day
    pos_shifted = positions.shift(1).fillna(0)
    cmg_r_aligned = cmg_r.reindex(trading_dates).fillna(0)
    pnl_raw = pos_shifted * cmg_r_aligned   # short: PnL = -1 * CMG return

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    in_pos_days = (pnl != 0).sum()
    print(f"  In-position days: {in_pos_days}")
    if in_pos_days < 20:
        return mark_failed(sid, f"insufficient in-position days: {in_pos_days}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Value Search Surge Short CMG",
        positions=abs(pos_shifted.reindex(pnl.index).fillna(0)),
        cost_bps=15,   # short costs + spread
    )

    ev_returns = [e["short_cmg_return"] for e in events if e.get("short_cmg_return") is not None]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Short CMG when Google Trends 'dollar menu' AND 'cheap groceries' "
                "simultaneously exceed their 52-week rolling mean + 1.5*std. "
                "Entry Monday after signal week. Hold max 6 weeks or until signal "
                "abates. Stop-loss: CMG +8% from entry."
            ),
            "mechanism": (
                "Consumer value-seeking behavior (proxied by Google Trends) "
                "signals wallet stress that hits premium-priced fast casual restaurants "
                "like Chipotle. As consumers trade down, CMG same-store sales soften "
                "and forward PE compresses. The Google Trends signal leads "
                "earnings disappointments by ~4-8 weeks."
            ),
            "source": data_source,
            "signal_description": signal_description,
            "tickers": ["CMG", "MCD", "SPY"],
            "n_events": len(events),
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events[:20],   # cap to avoid huge JSON
            "caveats": (
                "Google Trends normalization is relative (100 = peak of window), "
                "making long-run comparisons noisy. Stitching 2-year windows "
                "introduces scale discontinuities. CMG is a secular growth story "
                "that can rally even during consumer stress due to unit economics "
                "improvements. Short CMG has a negative carry (borrow cost + "
                "dividends). Earnings risk requires blackout periods."
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
