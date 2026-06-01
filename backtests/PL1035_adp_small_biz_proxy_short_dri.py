"""PL1035_adp_small_biz_proxy_short_dri — ADP/BLS Leisure-Hospitality Payroll Decel → Short DRI (Darden)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1035_adp_small_biz_proxy_short_dri"
    # Load FRED data
    try:
        uslah = load_fred("USLAH", start="2000-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED USLAH load: {e}")

    try:
        adp = load_fred("ADPMNUSNERSA", start="2010-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED ADP load: {e}")

    if uslah.empty or adp.empty:
        return mark_failed(sid, "missing FRED data")

    # Resample to monthly
    uslah_m = uslah.resample("MS").last()
    adp_m = adp.resample("MS").last()

    # Compute signals
    uslah_yoy = uslah_m.pct_change(12)
    adp_mom = adp_m.pct_change(1)

    # Align on common index
    common_idx = uslah_yoy.index.intersection(adp_mom.index)
    uslah_yoy_c = uslah_yoy.loc[common_idx]
    adp_mom_c = adp_mom.loc[common_idx]

    # Signal: USLAH YoY < -0.5% for 2+ consecutive months AND ADP MoM 2-month avg < -0.2%
    # Use 2-month rolling check
    uslah_cond = uslah_yoy_c < -0.005
    adp_cond = adp_mom_c.rolling(2).mean() < -0.002

    # Combined signal (both true)
    combined = uslah_cond & adp_cond
    # Identify start-of-signal dates: first month where combined turns True
    triggers = []
    in_signal = False
    consec = 0
    for i in range(len(combined)):
        if combined.iloc[i]:
            consec += 1
            if consec >= 2 and not in_signal:
                triggers.append(combined.index[i])
                in_signal = True
        else:
            consec = 0
            in_signal = False

    print(f"Signal events: {len(triggers)}")

    # Also try USLAH-only for additional coverage (pre-2010)
    uslah_cond_all = uslah_yoy < -0.005
    uslah_consec = 0
    triggers_uslah = []
    in_signal_u = False
    for i in range(len(uslah_cond_all)):
        if uslah_cond_all.iloc[i]:
            uslah_consec += 1
            if uslah_consec >= 2 and not in_signal_u:
                dt = uslah_cond_all.index[i]
                if dt < pd.Timestamp("2010-01-01"):
                    triggers_uslah.append(dt)
                    in_signal_u = True
        else:
            uslah_consec = 0
            in_signal_u = False

    all_triggers = sorted(set(triggers_uslah + triggers))
    print(f"Total triggers (USLAH pre-2010 + combined): {len(all_triggers)}")

    if not all_triggers:
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

    hold_td = 42  # ~8 calendar weeks in trading days
    cooldown_days = 60

    pnl = pd.Series(0.0, index=dri_r.index)
    events = []
    last_exit_date = None

    for td in all_triggers:
        # Approximate entry: +1 trading day after signal confirmation (~45 days after reference month)
        entry_approx = td + pd.DateOffset(days=45)

        # Cooldown check
        if last_exit_date is not None and (entry_approx - last_exit_date).days < cooldown_days:
            continue

        # Find next trading day
        mask = dri_r.index >= entry_approx
        if mask.sum() < hold_td:
            continue

        entry_idx = dri_r.index[mask][0]
        p = dri_r.index.get_loc(entry_idx)

        # Short DRI: PnL = -DRI_return, with stop/profit rules
        exit_p = min(p + hold_td, len(dri_r))
        cum_short_pnl = 0.0
        actual_exit = exit_p

        for j in range(p, exit_p):
            r = dri_r.iloc[j]
            cum_short_pnl += -r
            # Stop loss: DRI cumulative gain > 8% (short loses 8%)
            if cum_short_pnl < -0.08:
                actual_exit = j + 1
                break
            # Take profit: DRI cumulative decline > 15% (short gains 15%)
            if cum_short_pnl > 0.15:
                actual_exit = j + 1
                break

        seg_len = actual_exit - p
        if seg_len == 0:
            continue

        # Record PnL as short
        for j in range(p, actual_exit):
            pnl.iloc[j] = -dri_r.iloc[j]

        last_exit_date = dri_r.index[actual_exit - 1]

        short_ret = float((1 - dri_r.iloc[p:actual_exit]).prod() - 1)
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

    m = compute_metrics(active_pnl, benchmark=spy_r, name="L&H Payroll Decel → Short DRI")
    returns_list = [e["short_dri_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short DRI 42td when USLAH YoY < -0.5% for 2+ months AND ADP MoM 2M avg < -0.2% (pre-2010 USLAH-only)",
        "mechanism": "L&H payroll deceleration signals lower-income consumer stress → reduced dining out → DRI revenue compression",
        "source": "FRED USLAH, ADPMNUSNERSA; yfinance DRI/SPY",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, metrics saved.")


if __name__ == "__main__":
    main()
