"""PL139_advance_retail_sales_mom_streak_xrt — Advance Retail Sales 3 Consecutive Positive MoM → Long XRT 60d
When FRED RSAFS shows 3 consecutive positive MoM changes, long XRT for 60 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL139_advance_retail_sales_mom_streak_xrt"
    try:
        px = load_prices(["XRT", "SPY"], start="2006-01-01")
        rsafs = load_fred("RSAFS", start="2004-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xrt_r = ret["XRT"]
    spy_r = ret["SPY"]

    # Monthly retail sales, compute MoM
    rsafs_m = rsafs.resample("M").last().dropna()
    rsafs_mom = rsafs_m.pct_change()

    # Find 3-month positive streaks
    streak = 0
    triggers = []
    cooldown = 0
    for i in range(1, len(rsafs_mom)):
        val = float(rsafs_mom.iloc[i])
        if np.isnan(val):
            streak = 0
            continue
        if cooldown > 0:
            cooldown -= 1
            streak = 0
            continue
        if val > 0:
            streak += 1
            if streak == 3:
                triggers.append(rsafs_mom.index[i])
                streak = 0
                cooldown = 3  # wait 3 months before next trigger
        else:
            streak = 0

    events = []
    pnl_parts = []
    hold_days = 60

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        xrt_window = xrt_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xrt_window)

        xrt_cum = float((1 + xrt_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "rsafs_mom": round(float(rsafs_mom.loc[trig_date]) * 100, 2),
            "xrt_60d_return": round(xrt_cum, 4),
            "spy_60d_return": round(spy_cum, 4),
            "excess": round(xrt_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No retail sales positive streak events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Retail Sales 3mo Streak → Long XRT")

    avg_xrt = np.mean([e["xrt_60d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xrt_60d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED RSAFS 3 consecutive positive MoM → long XRT 60 trading days",
        "mechanism": "Consumer spending momentum — 3 consecutive positive months signals sustained demand, benefiting retail equities",
        "source": "FRED RSAFS (advance retail sales) + yfinance",
        "n_events": len(events),
        "avg_xrt_return": round(avg_xrt, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XRT: {avg_xrt*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xrt_60d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (MoM={e['rsafs_mom']:.1f}%): XRT {e['xrt_60d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
