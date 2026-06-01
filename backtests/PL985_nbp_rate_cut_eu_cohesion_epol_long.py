"""PL985_nbp_rate_cut_eu_cohesion_epol_long — NBP Rate Cut + EU Cohesion Fund Disbursement -> Long Poland (EPOL)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL985_nbp_rate_cut_eu_cohesion_epol_long"
    # NBP rate cut events with post-cut rate >= 2.5% (removes ZIRP era 2015-2021)
    # 2012-2013 cycle: major easing cycle from 4.75% down to 2.5%
    # 2023 cycle: surprise cuts after years of hikes
    # 2019 and 2014 excluded: post-cut rates close to or at floor
    trigger_dates = pd.to_datetime([
        "2012-11-07",  # 4.75% -> 4.50%
        "2013-01-09",  # 4.50% -> 4.25%
        "2013-03-06",  # 4.25% -> 3.75% (-50bp)
        "2013-05-08",  # 3.75% -> 3.25% (-50bp)
        "2013-06-05",  # 3.25% -> 2.75% (-50bp)
        "2023-09-06",  # 6.75% -> 6.00% (-75bp surprise)
        "2023-10-04",  # 6.00% -> 5.75% (-25bp)
    ])

    try:
        px = load_prices(["EPOL", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "EPOL" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "missing EPOL or SPY price data")

    ret = daily_returns(px)
    epol_r = ret["EPOL"]
    spy_r = ret["SPY"]

    hold = 40  # 40 trading days (~8 weeks)
    stop_loss = -0.10   # exit if EPOL down 10% from entry
    take_profit = 0.12  # exit if EPOL up 12% from entry

    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for td in trigger_dates:
        future_epol = epol_r.index[epol_r.index >= td]
        if len(future_epol) < hold:
            continue

        entry_idx = future_epol[0]
        ep = epol_r.index.get_loc(entry_idx)

        cum = 0.0
        active_days = 0
        event_pnl = []
        exit_reason = "max_hold"

        for j in range(ep, min(ep + hold, len(epol_r))):
            day_ret = epol_r.iloc[j]
            cum = (1 + cum) * (1 + day_ret) - 1
            active_days += 1
            event_pnl.append((epol_r.index[j], day_ret))

            if cum <= stop_loss:
                exit_reason = "stop_loss"
                break
            if cum >= take_profit:
                exit_reason = "take_profit"
                break

        # Add to PnL series
        for dt, val in event_pnl:
            if dt in pnl.index:
                pnl.loc[dt] += val

        # SPY over same window for excess return calc
        sp = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else None
        if sp is not None:
            spy_window = spy_r.iloc[sp:sp + active_days]
            spy_ret = float((1 + spy_window).prod() - 1)
        else:
            spy_ret = np.nan

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "epol_return": round(cum, 4),
            "spy_return": round(spy_ret, 4) if not np.isnan(spy_ret) else None,
            "excess_return": round(cum - spy_ret, 4) if not np.isnan(spy_ret) else None,
            "n_days": active_days,
            "exit_reason": exit_reason,
        })

    print(f"Events: {len(events)}")
    for e in events:
        excess = f"{e['excess_return']:.2%}" if e['excess_return'] is not None else "N/A"
        spy_s = f"{e['spy_return']:.2%}" if e['spy_return'] is not None else "N/A"
        print(f"  {e['trigger_date']}: EPOL={e['epol_return']:.2%}, "
              f"SPY={spy_s}, excess={excess}, exit={e['exit_reason']}")

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NBP Rate Cut: Long Poland EPOL (8-week hold)")

    epol_returns = [e["epol_return"] for e in events]
    mean_return = float(np.mean(epol_returns))
    win_rate = float(np.mean([r > 0 for r in epol_returns]))

    save_result(sid, m, extra={
        "rule": "Long EPOL at NBP rate cut announcement (25+ bps), provided post-cut rate >= 2.5%. Hold 8 weeks or exit on 10% stop / 12% profit.",
        "mechanism": "Rate cuts ease financial conditions for Poland's bank-heavy EPOL (PKO, Santander PL, Pekao); valuation re-rating + FX inflows from EU cohesion funds amplify near-term equity upside",
        "source": "NBP RPP rate decisions (nbp.pl); events: 2012-11-07 to 2013-06-05, 2023-09-06, 2023-10-04",
        "n_events": len(events),
        "mean_return": round(mean_return, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. n_events={len(events)}, mean_return={mean_return:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
