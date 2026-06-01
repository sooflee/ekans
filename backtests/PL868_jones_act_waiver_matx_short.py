"""PL868_jones_act_waiver_matx_short — DHS Jones Act Waiver Event -> Short MATX vs Long XTN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL868_jones_act_waiver_matx_short"

    # Known Jones Act waiver dates from strategy spec
    # Each date is the announcement/publication date; entry is day+1 (execution lag)
    known_events = [
        ("2005-09-08", "Hurricane Katrina - fuel routes"),
        ("2012-11-01", "Hurricane Sandy - fuel routes"),
        ("2017-08-31", "Hurricane Harvey - Gulf Coast petroleum"),
        ("2017-09-22", "Hurricane Maria - Puerto Rico"),
        ("2021-05-13", "Colonial Pipeline shutdown - Northeast fuel"),
    ]

    try:
        # MATX started trading in 2005; XTN inception is 2011
        px = load_prices(["MATX", "XTN", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or len(px) < 252:
        return mark_failed(sid, "insufficient price data")

    ret = daily_returns(px)
    matx_r = ret["MATX"] if "MATX" in ret.columns else None
    xtn_r = ret["XTN"] if "XTN" in ret.columns else None
    spy_r = ret["SPY"]

    if matx_r is None or matx_r.notna().sum() < 100:
        return mark_failed(sid, "MATX data insufficient")

    # For events before XTN inception (2011), use SPY as the long leg
    xtn_start = xtn_r.first_valid_index() if xtn_r is not None else pd.Timestamp("2011-01-01")

    hold_days = 20
    stop_loss = 0.08  # cover if MATX rallies >8% from entry

    pnl = pd.Series(0.0, index=spy_r.index)
    all_events = []
    last_exit = pd.Timestamp("2000-01-01")

    for event_date_str, label in known_events:
        trigger = pd.Timestamp(event_date_str)

        # Entry: day+1 after announcement
        next_dates = matx_r.index[matx_r.index > trigger]
        if len(next_dates) == 0:
            print(f"  Skipping {event_date_str} — no trading days after trigger")
            continue
        entry_date = next_dates[0]

        if entry_date <= last_exit:
            print(f"  Skipping {event_date_str} — overlaps previous position")
            continue

        if entry_date not in matx_r.index:
            print(f"  Skipping {event_date_str} — MATX not available at entry")
            continue

        entry_idx = matx_r.index.get_loc(entry_date)
        max_exit_idx = min(entry_idx + hold_days, len(matx_r) - 1)

        # Choose long leg: XTN if available, else SPY
        use_xtn = (xtn_r is not None and entry_date >= xtn_start and
                   entry_date in xtn_r.index and not pd.isna(xtn_r.loc[entry_date]))
        long_r = xtn_r if use_xtn else spy_r

        # Trade PnL = short MATX + long hedge (1:1 dollar notional)
        # pnl per day = long_return - matx_return
        pair_r_series = long_r - matx_r

        # Scan for stop-loss: MATX cumulative return > +8%
        cum_matx = 0.0
        actual_exit_idx = max_exit_idx
        exit_reason = "time_stop"

        for j in range(entry_idx, max_exit_idx + 1):
            mr = matx_r.iloc[j]
            if np.isnan(mr):
                continue
            cum_matx = (1 + cum_matx) * (1 + mr) - 1
            if cum_matx > stop_loss:
                actual_exit_idx = j
                exit_reason = "stop_loss"
                break

        window_pnl = pair_r_series.iloc[entry_idx:actual_exit_idx + 1]
        # Fill in global pnl series
        for j, idx_date in enumerate(window_pnl.index):
            if idx_date in pnl.index:
                pnl.loc[idx_date] += window_pnl.iloc[j]

        trade_return = float((1 + window_pnl).prod() - 1)
        matx_window = matx_r.iloc[entry_idx:actual_exit_idx + 1]
        matx_ret = float((1 + matx_window).prod() - 1)
        spy_window = spy_r.iloc[entry_idx:actual_exit_idx + 1]
        spy_ret = float((1 + spy_window).prod() - 1)

        def car_at(t):
            end = min(entry_idx + t, len(pair_r_series) - 1)
            w = pair_r_series.iloc[entry_idx:end + 1]
            return float((1 + w).prod() - 1) if len(w) > 0 else np.nan

        all_events.append({
            "event_date": event_date_str,
            "label": label,
            "entry_date": str(entry_date.date()),
            "exit_date": str(matx_r.index[actual_exit_idx].date()),
            "exit_reason": exit_reason,
            "days_held": actual_exit_idx - entry_idx + 1,
            "long_leg": "XTN" if use_xtn else "SPY",
            "car_t5": round(car_at(5), 4),
            "car_t10": round(car_at(10), 4),
            "car_t20": round(car_at(20), 4),
            "total_pair_return": round(trade_return, 4),
            "matx_return": round(matx_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

        last_exit = matx_r.index[actual_exit_idx]

    print(f"Events processed: {len(all_events)}")
    for ev in all_events:
        print(f"  {ev['event_date']} ({ev['label'][:30]}): "
              f"CAR(t5)={ev['car_t5']:.2%}, CAR(t10)={ev['car_t10']:.2%}, "
              f"CAR(t20)={ev['car_t20']:.2%}, MATX={ev['matx_return']:.2%}, "
              f"exit={ev['exit_reason']}, long={ev['long_leg']}")

    active_pnl = pnl[pnl != 0]
    if len(all_events) == 0:
        return mark_failed(sid, "no valid events processed")
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Jones Act Waiver Short MATX")
    m["n_events"] = len(all_events)

    avg_ret = float(np.mean([e["total_pair_return"] for e in all_events]))
    win_rate = float(np.mean([e["total_pair_return"] > 0 for e in all_events]))

    save_result(sid, m, extra={
        "rule": "Short MATX / Long XTN (or SPY pre-XTN) at day+1 after DHS/CBP Jones Act waiver publication. "
                "Hold 20 trading days; stop-loss if MATX rallies >8%.",
        "mechanism": "Jones Act waivers allow foreign-flag vessels on US coastwise routes, compressing MATX's "
                     "pricing power on domestic shipping lanes (Hawaii, Alaska, Puerto Rico, petroleum routes). "
                     "Pair vs XTN isolates shipping-rate compression from broad transport sector moves.",
        "source": "DHS/CBP Federal Register waiver notices; yfinance MATX/XTN/SPY; "
                  "events: Katrina 2005, Sandy 2012, Harvey 2017, Maria 2017, Colonial Pipeline 2021",
        "n_events": len(all_events),
        "avg_pair_return": round(avg_ret, 4),
        "win_rate": round(win_rate, 4),
        "events": all_events,
    })
    print(f"Done — {len(all_events)} events, avg_pair={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
