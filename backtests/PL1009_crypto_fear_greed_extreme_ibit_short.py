"""PL1009_crypto_fear_greed_extreme_ibit_short — Crypto Fear & Greed Sustained Extreme Greed (>=85 for 5d) — Short IBIT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import urllib.request
import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def fetch_fear_greed() -> pd.Series:
    """Download full Alternative.me Fear & Greed index history."""
    url = "https://api.alternative.me/fng/?limit=3000&format=json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        d = json.loads(resp.read().decode())
    records = d["data"]
    dates = []
    values = []
    for r in records:
        ts = int(r["timestamp"])
        dt = pd.Timestamp(ts, unit="s", tz="UTC").normalize().tz_localize(None)
        dates.append(dt)
        values.append(int(r["value"]))
    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="fear_greed")
    s = s.sort_index()
    # deduplicate (keep last)
    s = s[~s.index.duplicated(keep="last")]
    return s


def main():
    sid = "PL1009_crypto_fear_greed_extreme_ibit_short"
    # ---- load prices ----
    try:
        # Use BTC-USD for full history; IBIT only from Jan 2024
        px = load_prices(["BTC-USD", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    # ---- load Fear & Greed index ----
    try:
        fg = fetch_fear_greed()
    except Exception as e:
        # retry once
        try:
            fg = fetch_fear_greed()
        except Exception as e2:
            return mark_failed(sid, f"fear_greed load: {e2}")

    # ---- align data ----
    ret = daily_returns(px)
    btc_r = ret["BTC-USD"]
    spy_r = ret["SPY"]

    # Reindex fear & greed to trading calendar (forward-fill weekends)
    fg_aligned = fg.reindex(btc_r.index, method="ffill")
    fg_aligned = fg_aligned.dropna()

    # ---- find entry signals ----
    # Entry: FG >= 85 for 5 consecutive calendar days → short BTC
    # Build a consecutive-days-above-85 counter
    fg_daily = fg.copy()
    fg_daily = fg_daily[fg_daily.index >= pd.Timestamp("2018-01-01")]

    # Count consecutive days at >= 85
    consec = pd.Series(0, index=fg_daily.index)
    cnt = 0
    for i, (dt, val) in enumerate(fg_daily.items()):
        if val >= 85:
            cnt += 1
        else:
            cnt = 0
        consec.iloc[i] = cnt

    # Signal fires on day when count first reaches 5
    # (i.e., 5th consecutive day above 85)
    signal_dates = consec.index[consec >= 5]
    # Find the first day of each contiguous run
    entry_dates = []
    prev_date = None
    for dt in signal_dates:
        if prev_date is None or (dt - prev_date).days > 3:
            entry_dates.append(dt)
        prev_date = dt

    # Also enforce minimum gap of 30 calendar days between entries
    filtered_entries = []
    last_entry = None
    for dt in entry_dates:
        if last_entry is None or (dt - last_entry).days >= 30:
            filtered_entries.append(dt)
            last_entry = dt

    print(f"Total signal dates above threshold: {len(entry_dates)}")
    print(f"Filtered entries (30d min gap): {len(filtered_entries)}")
    for d in filtered_entries:
        fg_val = fg_daily.get(d, np.nan)
        print(f"  {d.date()}  FG={fg_val}")

    if not filtered_entries:
        return mark_failed(sid, "no qualifying entry events")

    # ---- build PnL series ----
    # Short BTC for 21 calendar days or until FG drops below 60 for 2 days
    hold_calendar = 21
    pnl = pd.Series(0.0, index=btc_r.index)
    events = []

    for entry_dt in filtered_entries:
        # Find nearest trading day on or after entry
        future_mask = btc_r.index >= entry_dt
        if not future_mask.any():
            continue
        entry_idx = btc_r.index[future_mask][0]
        p = btc_r.index.get_loc(entry_idx)

        # Determine exit: 21 calendar days or FG < 60 for 2 consecutive days
        exit_p = None
        exit_reason = "hold_period"
        entry_calendar_end = entry_idx + pd.Timedelta(days=hold_calendar)

        fg_below60_cnt = 0
        for j in range(p + 1, min(p + 35, len(btc_r))):
            trade_dt = btc_r.index[j]
            if trade_dt > entry_calendar_end:
                exit_p = j
                exit_reason = "hold_period"
                break
            # check FG for exit
            fg_val_j = fg_aligned.get(trade_dt, np.nan)
            if not np.isnan(fg_val_j) and fg_val_j < 60:
                fg_below60_cnt += 1
                if fg_below60_cnt >= 2:
                    exit_p = j
                    exit_reason = "fg_below_60"
                    break
            else:
                fg_below60_cnt = 0

        if exit_p is None:
            exit_p = min(p + 22, len(btc_r))
            exit_reason = "hold_period"

        # Short = negative BTC returns
        window = btc_r.iloc[p:exit_p]
        short_ret = -window

        # Accumulate PnL (additive, position sizing = 1)
        for k, (idx, r) in enumerate(short_ret.items()):
            if pnl[idx] == 0.0:
                pnl[idx] += r
            # If already in a position (overlap), skip

        total_ret = float((1 + short_ret).prod() - 1)
        spy_loc = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else None
        spy_ret = None
        if spy_loc is not None:
            spy_window = spy_r.iloc[spy_loc:min(spy_loc + (exit_p - p), len(spy_r))]
            spy_ret = float((1 + spy_window).prod() - 1)

        events.append({
            "entry_date": str(entry_dt.date()),
            "exit_reason": exit_reason,
            "n_days_held": exit_p - p,
            "btc_short_return": round(total_ret, 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
            "fg_at_entry": int(fg_daily.get(entry_dt, np.nan)) if not np.isnan(fg_daily.get(entry_dt, 0)) else None,
        })

    print(f"\nEvents ({len(events)}):")
    for ev in events:
        print(f"  {ev}")

    # Filter to non-zero pnl days
    active_pnl = pnl[pnl != 0.0]
    print(f"\nActive trading days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Crypto F&G Extreme Greed → Short BTC")
    save_result(sid, m, extra={
        "rule": "Short BTC-USD when Crypto Fear & Greed Index reads >= 85 for 5+ consecutive days; hold 21 calendar days or until FG drops below 60 for 2 days",
        "mechanism": "Sustained extreme retail greed signals crowded long positioning and near-term mean reversion in BTC price",
        "source": "Alternative.me Crypto Fear & Greed Index (api.alternative.me/fng/); BTC-USD yfinance",
        "n_events": len(events),
        "avg_short_return": round(float(np.mean([e["btc_short_return"] for e in events])), 4),
        "win_rate": round(float(np.mean([e["btc_short_return"] > 0 for e in events])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events")


if __name__ == "__main__":
    main()
