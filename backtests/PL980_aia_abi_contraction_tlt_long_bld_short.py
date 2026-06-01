"""PL980_aia_abi_contraction_tlt_long_bld_short — AIA Architecture Billings Index Sustained Contraction -> Duration Long + Construction Short"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL980_aia_abi_contraction_tlt_long_bld_short"
    # Known AIA ABI sustained contraction trigger dates (3+ consecutive months sub-45)
    # 2007-09-01: sustained contraction start
    # 2008-01-01: deep contraction
    # 2020-03-01: COVID collapse
    # 2022-09-01: post-hike construction slowdown
    trigger_dates = pd.to_datetime(["2007-09-01", "2008-01-01", "2020-03-01", "2022-09-01"])

    try:
        px = load_prices(["TLT", "IEF", "BLD", "SPY"], start="2002-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    missing = [t for t in ["TLT", "IEF", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing required tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    tlt_r = ret["TLT"]
    ief_r = ret["IEF"]

    # 50/50 TLT + IEF long basket
    duration_r = 0.5 * tlt_r + 0.5 * ief_r

    # BLD available from 2015-06 onward
    has_bld = "BLD" in ret.columns
    bld_r = ret["BLD"] if has_bld else None

    hold = 84  # ~4 calendar months in trading days
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for td in trigger_dates:
        future_mask = spy_r.index >= td
        if future_mask.sum() < hold:
            continue
        entry_idx = spy_r.index[future_mask][0]
        p = spy_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(spy_r))

        dur_future = duration_r[duration_r.index >= entry_idx]
        if len(dur_future) < 5:
            continue
        dur_window = dur_future.iloc[:hold]

        spy_window = spy_r.iloc[p:ep]
        dur_total = float((1 + dur_window).prod() - 1)
        spy_total = float((1 + spy_window).prod() - 1)

        # BLD short only for 2022+ events
        bld_total = None
        bld_short_total = None
        if has_bld and bld_r is not None:
            bld_future = bld_r[bld_r.index >= entry_idx]
            if len(bld_future) >= 5:
                bld_window = bld_future.iloc[:hold]
                bld_total = float((1 + bld_window).prod() - 1)
                bld_short_total = -bld_total  # short

        # Primary PnL is the duration long
        for dt, val in dur_window.items():
            if dt in pnl.index:
                pnl.loc[dt] += val

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "duration_return": round(dur_total, 4),
            "spy_return": round(spy_total, 4),
            "bld_return": round(bld_total, 4) if bld_total is not None else None,
            "bld_short_return": round(bld_short_total, 4) if bld_short_total is not None else None,
            "n_days": len(dur_window),
        })

    print(f"Events: {len(events)}")
    for e in events:
        bld_str = f", BLD_short={e['bld_short_return']:.2%}" if e['bld_short_return'] is not None else ""
        print(f"  {e['trigger_date']}: dur={e['duration_return']:.2%}, SPY={e['spy_return']:.2%}{bld_str}")

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r,
                        name="AIA ABI Contraction: Long TLT+IEF (Duration)")

    dur_returns = [e["duration_return"] for e in events]
    mean_dur = float(np.mean(dur_returns))
    win_rate = float(np.mean([r > 0 for r in dur_returns]))

    save_result(sid, m, extra={
        "rule": "Long 50/50 TLT+IEF 4 months when AIA ABI prints sub-45 for 3+ consecutive months",
        "mechanism": "Architecture billings lead construction 9-12 months; sustained ABI contraction signals construction recession -> flight to duration, rate cut expectations",
        "source": "AIA Architecture Billings Index event dates: 2007-09, 2008-01, 2020-03, 2022-09",
        "n_events": len(events),
        "mean_duration_return": round(mean_dur, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. Mean duration return: {mean_dur:.2%}, Win rate: {win_rate:.0%}")


if __name__ == "__main__":
    main()
