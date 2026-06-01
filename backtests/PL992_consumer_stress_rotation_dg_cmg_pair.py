"""PL992_consumer_stress_rotation_dg_cmg_pair — Consumer Stress Rotation: Trade-Down Signal -> Long DG / Short CMG Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL992_consumer_stress_rotation_dg_cmg_pair"
    try:
        px = load_prices(["DG", "CMG", "SPY"], start="2009-11-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    if "DG" not in px.columns or "CMG" not in px.columns:
        return mark_failed(sid, "missing DG or CMG price data")

    ret = daily_returns(px)
    dg_r = ret["DG"]
    cmg_r = ret["CMG"]

    try:
        # DRTSCILM: net % banks tightening consumer lending standards (quarterly)
        drtscilm = load_fred("DRTSCILM", start="2005-01-01")
        # JTSQUR: private-sector quits rate (monthly)
        jtsqur = load_fred("JTSQUR", start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if drtscilm.empty or jtsqur.empty:
        return mark_failed(sid, "empty FRED data")

    drt = drtscilm.iloc[:, 0].sort_index()
    quits = jtsqur.iloc[:, 0].sort_index()

    # Compute JTSQUR 12-month trailing MA (monthly data)
    quits_ma12 = quits.rolling(12).mean()
    quits_below_ma = quits < quits_ma12

    # DRTSCILM: 2 consecutive positive quarters = tightening signal
    drt_pos = drt > 0
    drt_2consec = drt_pos & drt_pos.shift(1)
    tightening_quarters = drt.index[drt_2consec]

    # Build daily signal: align quarterly/monthly data to daily equity index
    # For each tightening_quarters date, check if the nearest monthly JTSQUR also confirms
    # Entry: within 3 trading days of the second quarterly DRTSCILM confirmation

    hold = 63  # 63 trading days (~1 quarter)
    stop_loss_pair = -0.08  # exit if DG-CMG pair spread moves -8% against trade

    pnl = pd.Series(0.0, index=spy_r.index)
    events = []
    used_entry_dates = set()

    for qt in tightening_quarters:
        # Find the closest monthly JTSQUR observation on or before this quarter date
        jtsqur_avail = quits_below_ma.index[quits_below_ma.index <= qt]
        if len(jtsqur_avail) == 0:
            continue
        closest_jtsqur_date = jtsqur_avail[-1]
        if not quits_below_ma.loc[closest_jtsqur_date]:
            continue  # Quits rate condition not met

        # Find next available trading day for DG (within 3 business days)
        future_dg = dg_r.index[dg_r.index >= qt]
        if len(future_dg) < hold:
            continue
        # Find entry within 3 trading days
        entry_candidates = future_dg[:3]
        entry_idx = entry_candidates[0]

        # Skip if we already used this entry date (avoid double entries)
        if entry_idx in used_entry_dates:
            continue
        # Also skip if we're still within a prior holding period
        if len(used_entry_dates) > 0:
            last_entry = max(used_entry_dates)
            if entry_idx in dg_r.index:
                ei = dg_r.index.get_loc(entry_idx)
                last_i = dg_r.index.get_loc(last_entry) if last_entry in dg_r.index else 0
                if ei - last_i < hold:
                    continue

        used_entry_dates.add(entry_idx)

        dp = dg_r.index.get_loc(entry_idx)
        cp = cmg_r.index.get_loc(entry_idx) if entry_idx in cmg_r.index else None
        if cp is None:
            continue

        cum_pair = 0.0
        active_days = 0
        event_pnl = []
        exit_reason = "max_hold"

        for j in range(dp, min(dp + hold, len(dg_r))):
            dt = dg_r.index[j]
            dg_day = dg_r.iloc[j]
            if dt not in cmg_r.index:
                continue
            cmg_day = cmg_r.loc[dt]

            # Pair PnL: long DG, short CMG
            pair_day = dg_day - cmg_day
            cum_pair = (1 + cum_pair) * (1 + pair_day) - 1
            active_days += 1
            event_pnl.append((dt, pair_day))

            # Check DRTSCILM easing (check if last available DRTSCILM < -5)
            drt_avail = drt.index[drt.index <= dt]
            if len(drt_avail) > 0 and drt.loc[drt_avail[-1]] < -5:
                exit_reason = "drt_easing"
                break

            # Stop-loss: pair spread moves -8% against trade
            if cum_pair <= stop_loss_pair:
                exit_reason = "stop_loss"
                break

        if not event_pnl:
            continue

        for dt, val in event_pnl:
            if dt in pnl.index:
                pnl.loc[dt] += val

        # SPY return over same period
        sp = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else None
        if sp is not None:
            spy_w = spy_r.iloc[sp:sp + active_days]
            spy_ret = float((1 + spy_w).prod() - 1)
        else:
            spy_ret = np.nan

        events.append({
            "trigger_date": str(qt.date()),
            "entry_date": str(entry_idx.date()),
            "pair_return": round(cum_pair, 4),
            "spy_return": round(spy_ret, 4) if not np.isnan(spy_ret) else None,
            "n_days": active_days,
            "exit_reason": exit_reason,
            "drtscilm_at_trigger": round(float(drt.loc[qt]), 1),
        })

    print(f"Events: {len(events)}")
    for e in events:
        spy_s = f"{e['spy_return']:.2%}" if e['spy_return'] is not None else "N/A"
        print(f"  {e['trigger_date']}: pair={e['pair_return']:.2%}, "
              f"SPY={spy_s}, DRTSCILM={e['drtscilm_at_trigger']}, exit={e['exit_reason']}")

    if not events:
        return mark_failed(sid, "no valid signal events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Consumer Stress: Long DG / Short CMG Pair")

    pair_returns = [e["pair_return"] for e in events]
    mean_pair = float(np.mean(pair_returns))
    win_rate = float(np.mean([r > 0 for r in pair_returns]))

    save_result(sid, m, extra={
        "rule": "Long DG / Short CMG equal-notional when DRTSCILM > 0 for 2 consecutive quarters AND JTSQUR below 12-month MA. Hold 63 days; exit on DRTSCILM < -5 or 8% stop.",
        "mechanism": "Consumer trade-down from premium fast-casual (CMG) to deep-discount retail (DG) during credit tightening episodes; bank lending standards lead consumer stress by 1-2 quarters",
        "source": "FRED DRTSCILM (consumer lending standards quarterly) + JTSQUR (quits rate monthly); equity prices via yfinance",
        "n_events": len(events),
        "mean_pair_return": round(mean_pair, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. n_events={len(events)}, mean_pair={mean_pair:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
