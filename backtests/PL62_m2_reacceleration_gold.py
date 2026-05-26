"""PL62_m2_reacceleration_gold — M2 YoY Reaccelerates Above +5% → Long GLD 12mo
When M2SL YoY crosses above +5% after 6+ months below +3%: long GLD for 252 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL62_m2_reacceleration_gold"
    try:
        fred = load_fred("M2SL", start="1970-01-01")
        m2 = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if m2.empty or len(m2) < 24:
        return mark_failed(sid, "M2SL data insufficient")

    # Compute YoY change
    yoy = m2.pct_change(12) * 100
    yoy = yoy.dropna()

    # Find months where YoY crosses above +4% after having been below +2% for 6+ months
    # at some point in the past (before reaching +4%). Once fired, reset the flag.
    trigger_dates = []
    had_6mo_below2 = False
    below2_count = 0
    already_fired = False
    for i in range(len(yoy)):
        if yoy.iloc[i] < 2:
            below2_count += 1
            if below2_count >= 6:
                had_6mo_below2 = True
            already_fired = False
        else:
            if had_6mo_below2 and yoy.iloc[i] >= 4 and not already_fired:
                trigger_dates.append(yoy.index[i])
                already_fired = True
                had_6mo_below2 = False
            below2_count = 0

    print(f"M2 reacceleration events (>+4% after 6+ months <+2%): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: M2 YoY = {yoy.loc[d]:.2f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no M2 reacceleration events found")

    # Use GLD (from Nov 2004) and GC=F for earlier
    try:
        px_gld = load_prices(["GLD", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"GLD data load: {e}")

    try:
        px_gc = load_prices(["GC=F", "SPY"], start="1970-01-01")
    except:
        px_gc = None

    ret_gld = daily_returns(px_gld)
    if "GLD" in ret_gld.columns:
        gold_ret = ret_gld["GLD"]
    elif px_gc is not None and "GC=F" in daily_returns(px_gc).columns:
        gold_ret = daily_returns(px_gc)["GC=F"]
    else:
        return mark_failed(sid, "no gold price data")

    spy_ret = ret_gld["SPY"] if "SPY" in ret_gld.columns else daily_returns(px_gc)["SPY"]

    # Combine GC=F before GLD, GLD after
    if px_gc is not None and "GC=F" in daily_returns(px_gc).columns:
        gc_ret = daily_returns(px_gc)["GC=F"]
        combined_gold = pd.concat([gc_ret[gc_ret.index < gold_ret.index[0]], gold_ret])
    else:
        combined_gold = gold_ret

    hold_days = 252
    pnl_series = pd.Series(0.0, index=combined_gold.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = combined_gold.index >= entry_date
        if entry_mask.sum() < hold_days:
            if entry_mask.sum() < 30:
                continue
        entry_idx = combined_gold.index[entry_mask][0]
        pos = combined_gold.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(combined_gold))
        if end_pos - pos < 30:
            continue
        event_rets = combined_gold.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        # SPY comparison
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "m2_yoy": round(float(yoy.loc[td]), 2),
            "gold_12m_return": round(cumret, 4),
            "spy_12m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="M2 Reacceleration → Long Gold")
    rets_arr = [e["gold_12m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long GLD 252 days when M2 YoY crosses above +4% after 6+ months below +2%",
        "mechanism": "Monetary expansion drives gold with 6-12mo lag as liquidity flows to hard assets",
        "source": "FRED M2SL; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
