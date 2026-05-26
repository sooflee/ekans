"""PL55_stlfsi_stress_long_gold — STLFSI4 Spike > +1.0 → Long GLD
When STLFSI4 crosses above +1.0: long GLD for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL55_stlfsi_stress_long_gold"
    try:
        fred = load_fred("STLFSI4", start="1994-01-01")
        fsi = fred.squeeze()
    except Exception as e:
        # Try STLFSI2 as fallback
        try:
            fred = load_fred("STLFSI2", start="1994-01-01")
            fsi = fred.squeeze()
        except Exception as e2:
            return mark_failed(sid, f"FRED data load: {e} / STLFSI2: {e2}")

    if fsi.empty:
        return mark_failed(sid, "STLFSI data empty")

    # Find cross above +1.0 from below
    trigger_dates = []
    for i in range(1, len(fsi)):
        if fsi.iloc[i] > 1.0 and fsi.iloc[i-1] <= 1.0:
            # Avoid double-counting within same stress episode
            if len(trigger_dates) == 0 or (fsi.index[i] - trigger_dates[-1]).days > 90:
                trigger_dates.append(fsi.index[i])

    print(f"STLFSI stress events (>1.0): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: STLFSI = {fsi.loc[d]:.2f}")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no STLFSI stress events >1.0 found")

    try:
        px = load_prices(["GLD", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "GLD" not in ret.columns:
        # Try GC=F
        try:
            px = load_prices(["GC=F", "SPY"], start="2000-01-01")
            ret = daily_returns(px)
            gold_col = "GC=F"
        except Exception:
            return mark_failed(sid, "GLD and GC=F both unavailable")
    else:
        gold_col = "GLD"

    gold_ret = ret[gold_col]
    spy_ret = ret["SPY"]
    hold_days = 42

    pnl_series = pd.Series(0.0, index=gold_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_mask = gold_ret.index >= td
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = gold_ret.index[entry_mask][0]
        pos = gold_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(gold_ret))
        event_rets = gold_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "stlfsi": round(float(fsi.loc[td]), 2),
            "gold_2m_return": round(cumret, 4),
            "spy_2m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events (GLD starts Nov 2004)")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="STLFSI Stress → Long Gold")
    rets_arr = [e["gold_2m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long GLD for 42 days when STLFSI4 crosses above +1.0",
        "mechanism": "Flight-to-quality drives gold higher during financial stress",
        "source": "FRED STLFSI4; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
