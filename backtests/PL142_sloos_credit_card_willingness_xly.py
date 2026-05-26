"""PL142_sloos_credit_card_willingness_xly — SLOOS Credit Card Tightening Crosses Zero → Long XLY 63d
When FRED DRTSCLCC crosses from negative (tightening) to zero/positive, long XLY for 63 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL142_sloos_credit_card_willingness_xly"
    try:
        px = load_prices(["XLY", "SPY"], start="1998-01-01")
        cc = load_fred("DRTSCLCC", start="1996-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    xly_r = ret["XLY"]
    spy_r = ret["SPY"]

    cc = cc.dropna()

    # Find zero crossings from negative territory
    triggers = []
    cooldown = 0
    for i in range(1, len(cc)):
        if cooldown > 0:
            cooldown -= 1
            continue
        prev_val = float(cc.iloc[i-1])
        curr_val = float(cc.iloc[i])
        if prev_val < 0 and curr_val >= 0:
            triggers.append(cc.index[i])
            cooldown = 4  # skip ~4 quarters

    events = []
    pnl_parts = []
    hold_days = 63

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        xly_window = xly_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(xly_window)

        xly_cum = float((1 + xly_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "drtsclcc": round(float(cc.iloc[cc.index.get_loc(trig_date)]), 2),
            "xly_63d_return": round(xly_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(xly_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No SLOOS credit card easing events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="SLOOS CC Easing → Long XLY")

    avg_xly = np.mean([e["xly_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["xly_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED DRTSCLCC crosses from negative to zero/positive → long XLY 63 days",
        "mechanism": "Credit card lending easing signals bank willingness to extend consumer credit → discretionary spending benefits",
        "source": "FRED DRTSCLCC (SLOOS) + yfinance",
        "n_events": len(events),
        "avg_xly_return": round(avg_xly, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg XLY: {avg_xly*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["xly_63d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: XLY {e['xly_63d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
