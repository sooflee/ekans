"""PL1001_eia_gasoline_demand_destruction_psx_short — EIA Gasoline Demand Destruction -> Short PSX (Refiner Margin Collapse)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1001_eia_gasoline_demand_destruction_psx_short"
    try:
        px = load_prices(["PSX", "VLO", "SPY"], start="2010-01-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    if "PSX" not in px.columns or "VLO" not in px.columns:
        return mark_failed(sid, "missing PSX or VLO price data")

    ret = daily_returns(px)
    psx_r = ret["PSX"]
    vlo_r = ret["VLO"]

    try:
        # GASREGCOVW: weekly US regular conventional gasoline retail price ($/gal)
        # Use as demand-destruction proxy: sustained YoY price spike reduces driving
        gas = load_fred("GASREGCOVW", start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED GASREGCOVW load: {e}")

    if gas is None or gas.empty:
        return mark_failed(sid, "GASREGCOVW is empty")

    gas = gas.iloc[:, 0].sort_index()

    # YoY change in gas price (52-week lag = 1 year for weekly data)
    gas_yoy = gas / gas.shift(52) - 1
    # Signal: YoY gas price >= 15% for 2 consecutive weeks (demand destruction proxy)
    # Exclude 2021 (distorted COVID base effects: 2021/2020 comparison)
    signal_raw = gas_yoy >= 0.15
    # Exclude 2021 (exclude base-effect years)
    excl_2021 = (gas_yoy.index.year == 2021)
    signal_raw[excl_2021] = False
    signal_2consec = signal_raw & signal_raw.shift(1)

    signal_dates = gas.index[signal_2consec].tolist()

    hold_days = 40  # 40 trading days (~8 weeks)
    stop_loss_psx = 0.06   # PSX rallies 6% = stop-loss (short PSX)
    signal_weakness = 0.05  # YoY declines to < 5% = signal weakening

    pnl = pd.Series(0.0, index=spy_r.index)
    events = []
    last_exit_date = None

    for sig_date in signal_dates:
        # Find next available equity trading day
        future_psx = psx_r.index[psx_r.index >= sig_date]
        if len(future_psx) < hold_days:
            continue
        entry_idx = future_psx[0]

        # Lockout: skip if within 6 weeks of last exit
        if last_exit_date is not None:
            last_ep = psx_r.index.get_loc(last_exit_date)
            entry_ep = psx_r.index.get_loc(entry_idx)
            if entry_ep - last_ep < 30:
                continue

        dp = psx_r.index.get_loc(entry_idx)
        vp = vlo_r.index.get_loc(entry_idx) if entry_idx in vlo_r.index else dp

        cum_psx = 0.0
        active_days = 0
        event_pnl = []
        exit_reason = "max_hold"
        exit_date = entry_idx

        for j in range(dp, min(dp + hold_days, len(psx_r))):
            dt = psx_r.index[j]
            psx_day = psx_r.iloc[j]
            # Short PSX: PnL = -psx_daily_return
            pair_pnl = -psx_day

            # Check VLO long leg for pair variant
            if dt in vlo_r.index:
                vlo_day = vlo_r.loc[dt]
                # Pair: short PSX (1x), long VLO (0.5x) per spec
                pair_pnl = vlo_day * 0.5 - psx_day

            cum_psx = (1 + cum_psx) * (1 + psx_r.iloc[j]) - 1
            active_days += 1
            event_pnl.append((dt, pair_pnl))
            exit_date = dt

            # Stop-loss: PSX up 6% from entry
            if cum_psx >= stop_loss_psx:
                exit_reason = "stop_loss"
                break

            # Signal weakening: YoY gas price drops below 5%
            nearest_gas = gas_yoy.index[gas_yoy.index <= dt]
            if len(nearest_gas) > 0:
                latest_yoy = gas_yoy.loc[nearest_gas[-1]]
                if latest_yoy < 0.05:
                    exit_reason = "signal_weak"
                    break

        if not event_pnl:
            continue

        for dt, val in event_pnl:
            if dt in pnl.index:
                pnl.loc[dt] += val

        last_exit_date = exit_date

        # Compute PSX and VLO cumulative returns
        psx_window = psx_r.iloc[dp:dp + active_days]
        psx_ret = float((1 + psx_window).prod() - 1)
        vlo_window = vlo_r.iloc[vp:vp + active_days]
        vlo_ret = float((1 + vlo_window).prod() - 1)

        # SPY
        sp = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else dp
        spy_w = spy_r.iloc[sp:sp + active_days]
        spy_ret = float((1 + spy_w).prod() - 1)

        pair_ret = vlo_ret * 0.5 - psx_ret  # combined pair return

        gas_yoy_avail = gas_yoy.index[gas_yoy.index <= sig_date]
        sig_yoy = float(gas_yoy.loc[gas_yoy_avail[-1]]) if len(gas_yoy_avail) > 0 else np.nan

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "gas_yoy": round(sig_yoy, 3),
            "psx_return": round(psx_ret, 4),
            "vlo_return": round(vlo_ret, 4),
            "pair_return": round(pair_ret, 4),  # short PSX + 0.5x long VLO
            "pnl_from_short": round(-psx_ret, 4),  # PnL from PSX short leg
            "spy_return": round(spy_ret, 4),
            "n_days": active_days,
            "exit_reason": exit_reason,
        })

    print(f"Events: {len(events)}")
    for e in events:
        print(f"  {e['signal_date']}: gas_yoy={e['gas_yoy']:.1%}, "
              f"PSX_short={e['pnl_from_short']:.2%}, pair={e['pair_return']:.2%}, "
              f"SPY={e['spy_return']:.2%}, exit={e['exit_reason']}")

    if not events:
        return mark_failed(sid, "no valid signal events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Gasoline Demand Destruction: Short PSX / Long VLO (0.5x)")

    pair_returns = [e["pair_return"] for e in events]
    mean_pair = float(np.nanmean(pair_returns))
    win_rate = float(np.nanmean([r > 0 for r in pair_returns]))

    save_result(sid, m, extra={
        "rule": "Short PSX (1x) / Long VLO (0.5x) when GASREGCOVW YoY >= 15% for 2 consecutive weeks (demand destruction). Exit on PSX +6% stop or signal weakening (YoY < 5%) or 8 weeks.",
        "mechanism": "High sustained gasoline prices compress gasoline demand and refiner crack spreads; PSX (higher gasoline exposure) underperforms VLO (higher distillate) when gasoline margins compress",
        "source": "FRED GASREGCOVW (weekly retail gasoline price, 2010-present) as EIA demand-destruction proxy; PSX, VLO via yfinance",
        "n_events": len(events),
        "mean_pair_return": round(mean_pair, 4),
        "win_rate": round(win_rate, 4),
        "events": events[:20],  # store first 20 for brevity
    })
    print(f"Done. n_events={len(events)}, mean_pair={mean_pair:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
