"""PL1047_opentable_seated_diners_decline_dri_short — OpenTable Seated Diners Decline → Short DRI"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1047_opentable_seated_diners_decline_dri_short"

    # OpenTable historical data not freely available via API pre-2020.
    # Use FRED USRSGDFS (Food Services & Drinking Places sales, monthly) as proxy.
    # Also use FRED USLAH (Leisure & Hospitality employment) as second confirming signal.
    try:
        food_sales = load_fred("USRSGDFS", start="2000-01-01").squeeze().dropna()
    except Exception as e:
        try:
            # Try alternative: RSFSDP - Retail Sales Food Services
            food_sales = load_fred("RSFSDP", start="2000-01-01").squeeze().dropna()
        except Exception as e2:
            return mark_failed(sid, f"FRED food sales load: {e}, {e2}")

    if food_sales.empty:
        return mark_failed(sid, "no food services sales data")

    # Resample monthly
    food_m = food_sales.resample("MS").last()
    # Compute YoY change
    food_yoy = food_m.pct_change(12)

    # Signal proxy: food service sales YoY declining (below -2% is moderate weakness)
    # Use 4-month rolling average below -2% as proxy for 4-week consecutive seated diner decline
    food_roll4 = food_yoy.rolling(4).mean()

    # Find signal dates: rolling 4M avg < -0.02 (YoY decline of 2%+)
    triggers = []
    in_signal = False
    for i in range(len(food_roll4)):
        if pd.isna(food_roll4.iloc[i]):
            continue
        if food_roll4.iloc[i] < -0.02 and not in_signal:
            # Exclude COVID period (March-May 2020) - structural break
            dt = food_roll4.index[i]
            if pd.Timestamp("2020-03-01") <= dt <= pd.Timestamp("2020-07-01"):
                continue
            triggers.append(dt)
            in_signal = True
        elif food_roll4.iloc[i] >= -0.01:
            in_signal = False

    print(f"Signal events: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no signal events found")

    try:
        px = load_prices(["DRI", "SPY"], start="2000-01-01")
    except Exception as e:
        try:
            px = load_prices(["DRI", "SPY"], start="2000-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    ret = daily_returns(px)
    dri_r = ret["DRI"]
    spy_r = ret["SPY"]

    # DRI 12-week high check (avoid late entry if DRI already down >10%)
    dri_close = px["DRI"]
    dri_12wk_high = dri_close.rolling(60).max()

    hold_td = 30  # ~6 calendar weeks in trading days
    take_profit = 0.12  # DRI falls 12% from entry (short profits)
    stop_loss = 0.08   # DRI rises 8% from entry (stop)

    pnl = pd.Series(0.0, index=dri_r.index)
    events = []
    last_exit_date = None

    for td in triggers:
        # Entry: +2 trading days after signal month end
        entry_approx = td + pd.DateOffset(months=1, days=2)

        # Skip if still in position
        if last_exit_date is not None and entry_approx <= last_exit_date:
            continue

        mask = dri_r.index >= entry_approx
        if mask.sum() < hold_td:
            continue

        entry_idx = dri_r.index[mask][0]
        p = dri_r.index.get_loc(entry_idx)

        # Check: DRI has not already declined >10% from 12-week high
        if entry_idx in dri_12wk_high.index and entry_idx in dri_close.index:
            high = dri_12wk_high.loc[entry_idx]
            entry_price = dri_close.loc[entry_idx]
            if not pd.isna(high) and high > 0:
                decline_from_high = (entry_price - high) / high
                if decline_from_high < -0.10:
                    # Already declined too much - late entry, skip
                    continue

        exit_p = min(p + hold_td, len(dri_r))
        actual_exit = exit_p
        short_series = []
        cum_dri = 0.0

        for j in range(p, exit_p):
            r = dri_r.iloc[j]
            short_series.append(-r)
            cum_dri += r

            # Take profit: DRI down 12% from entry (short gains 12%)
            if cum_dri < -take_profit:
                actual_exit = j + 1
                break
            # Stop loss: DRI up 8% (short loses)
            if cum_dri > stop_loss:
                actual_exit = j + 1
                break

            # Signal reversion: food YoY rebounds above -1%
            check_dt = dri_r.index[j]
            recent_yoy = food_roll4[food_roll4.index <= check_dt]
            if len(recent_yoy) > 0 and recent_yoy.iloc[-1] > -0.01:
                yoy_at_entry = food_roll4[food_roll4.index <= entry_idx]
                if len(yoy_at_entry) > 0 and yoy_at_entry.iloc[-1] < -0.01:
                    actual_exit = j + 1
                    break

        seg_len = actual_exit - p
        if seg_len == 0 or not short_series:
            continue

        for j, val in enumerate(short_series[:seg_len]):
            pnl.iloc[p + j] = val

        last_exit_date = dri_r.index[actual_exit - 1]
        short_ret = float(sum(short_series[:seg_len]))
        spy_seg = spy_r.iloc[p:actual_exit]
        spy_ret = float((1 + spy_seg).prod() - 1) if len(spy_seg) > 0 else None

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": seg_len,
            "short_dri_return": round(short_ret, 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
        })

    print(f"Events executed: {len(events)}")
    if not events:
        return mark_failed(sid, "no executable events")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Food Services YoY Decline → Short DRI")
    returns_list = [e["short_dri_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short DRI 30td when FRED food services sales 4M rolling YoY avg < -2%, DRI not already down >10% from 12wk high",
        "mechanism": "Restaurant traffic decline (proxy via food services sales) signals DRI same-store-sales miss risk",
        "source": "FRED USRSGDFS/RSFSDP (food services sales); yfinance DRI/SPY; OpenTable proxy (true data limited availability)",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events,
        "implementation_note": "OpenTable weekly seated diners data (GitHub saulpw/opentable-data) only covers April 2020+; FRED food services sales used as longer-history proxy",
    })
    print(f"Done: {len(events)} events, metrics saved.")


if __name__ == "__main__":
    main()
