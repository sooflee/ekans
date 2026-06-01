"""PL1049_cargurus_dom_kmx_gpu_short — Used-Car Days-on-Market Surge → CarMax GPU Compression Short"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1049_cargurus_dom_kmx_gpu_short"

    try:
        used_car_cpi = load_fred("CUSR0000SETA02", start="1997-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED CUSR0000SETA02 load: {e}")

    if used_car_cpi.empty:
        return mark_failed(sid, "no Used Cars CPI data")

    # Resample to monthly
    used_car_m = used_car_cpi.resample("MS").last()
    mom = used_car_m.pct_change(1)  # month-over-month change

    # Signal: MoM negative for 2+ consecutive months AND absolute level > 102
    triggers = []
    consec_neg = 0
    in_signal = False

    for i in range(1, len(mom)):
        if pd.isna(mom.iloc[i]) or pd.isna(used_car_m.iloc[i]):
            consec_neg = 0
            continue

        if mom.iloc[i] < 0:
            consec_neg += 1
        else:
            consec_neg = 0
            in_signal = False

        if consec_neg >= 2 and not in_signal:
            # Check absolute level condition (elevated prices)
            level = float(used_car_m.iloc[i])
            if level > 102:
                triggers.append(mom.index[i])
                in_signal = True

    print(f"Signal events: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no signal events found")

    try:
        px = load_prices(["KMX", "SPY"], start="1997-01-01")
    except Exception as e:
        try:
            px = load_prices(["KMX", "SPY"], start="1997-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    ret = daily_returns(px)
    kmx_r = ret["KMX"]
    spy_r = ret["SPY"]

    # KMX 50-day SMA for entry filter
    kmx_close = px["KMX"]
    kmx_sma50 = kmx_close.rolling(50).mean()

    hold_td = 60  # ~12 calendar weeks in trading days
    stop_loss = 0.08  # 8% stop on short (KMX rises 8%)

    pnl = pd.Series(0.0, index=kmx_r.index)
    events = []
    last_exit_date = None

    for td in triggers:
        # Entry: first trading day of month after signal month
        entry_approx = td + pd.DateOffset(months=1)

        # Skip if still in a position
        if last_exit_date is not None and entry_approx <= last_exit_date:
            continue

        # Find next trading day
        mask = kmx_r.index >= entry_approx
        if mask.sum() < hold_td:
            continue

        entry_idx = kmx_r.index[mask][0]
        p = kmx_r.index.get_loc(entry_idx)

        # KMX below 50-day SMA filter
        if entry_idx in kmx_sma50.index:
            sma_val = kmx_sma50.loc[entry_idx]
            entry_price = kmx_close.loc[entry_idx]
            if not pd.isna(sma_val) and entry_price > sma_val:
                # Price above SMA - skip (not in downtrend)
                continue

        exit_p = min(p + hold_td, len(kmx_r))
        cum_loss = 0.0
        actual_exit = exit_p

        # Build short PnL with stop
        short_pnl_series = []
        for j in range(p, exit_p):
            r = kmx_r.iloc[j]
            short_r = -r
            short_pnl_series.append(short_r)
            cum_loss += r  # track KMX cumulative gain while short

            # Stop: KMX rises 8%
            if cum_loss > stop_loss:
                actual_exit = j + 1
                break

            # Exit: used car CPI MoM turns positive
            check_date = kmx_r.index[j]
            recent_mom = mom[mom.index <= check_date]
            if len(recent_mom) > 0 and recent_mom.iloc[-1] > 0:
                # Check that it wasn't already positive at entry
                mom_at_entry = mom[mom.index <= entry_idx]
                if len(mom_at_entry) > 0 and mom_at_entry.iloc[-1] > 0:
                    pass  # was already positive at entry, don't exit prematurely
                else:
                    actual_exit = j + 1
                    break

        seg_len = actual_exit - p
        if seg_len == 0 or not short_pnl_series:
            continue

        for j, val in enumerate(short_pnl_series[:seg_len]):
            pnl.iloc[p + j] = val

        last_exit_date = kmx_r.index[actual_exit - 1]
        short_ret = float(sum(short_pnl_series[:seg_len]))

        spy_seg = spy_r.iloc[p:actual_exit]
        spy_ret = float((1 + spy_seg).prod() - 1) if len(spy_seg) > 0 else None

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": seg_len,
            "short_kmx_return": round(short_ret, 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
        })

    print(f"Events executed: {len(events)}")
    if not events:
        return mark_failed(sid, "no executable events (all filtered by SMA or hold)")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Used-Car CPI Decline → Short KMX")
    returns_list = [e["short_kmx_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short KMX ~60td when Used Cars CPI MoM negative 2+ consecutive months AND level >102 AND KMX below 50d SMA",
        "mechanism": "Falling used car CPI signals inventory glut / rising DOM → CarMax GPU compression → earnings miss risk",
        "source": "FRED CUSR0000SETA02; yfinance KMX/SPY",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, metrics saved.")


if __name__ == "__main__":
    main()
