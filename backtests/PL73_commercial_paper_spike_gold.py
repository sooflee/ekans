"""PL73_commercial_paper_spike_gold — CP Outstanding Spike > +10% 4wk → Long GLD
When COMPOUT 4-week rolling change exceeds +10%: long GLD for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL73_commercial_paper_spike_gold"
    try:
        fred = load_fred("COMPOUT", start="2001-01-01")
        cp = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if cp.empty or len(cp) < 10:
        return mark_failed(sid, "COMPOUT data insufficient")

    # Compute 4-week rolling % change
    cp_chg = cp.pct_change(4)
    cp_chg = cp_chg.dropna()

    # Find weeks where 4-week change first exceeds +10%
    trigger_dates = []
    cooldown = 0
    for i in range(len(cp_chg)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if cp_chg.iloc[i] > 0.10:
            trigger_dates.append(cp_chg.index[i])
            cooldown = 42  # avoid retriggering within same episode

    print(f"CP spike events (4wk chg > +10%): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: 4wk chg = {cp_chg.loc[d]*100:.1f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no CP spike events found")

    try:
        px = load_prices(["GLD", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "GLD" not in ret.columns:
        return mark_failed(sid, "GLD data missing")

    gold_ret = ret["GLD"]
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

        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "cp_4wk_chg": round(float(cp_chg.loc[td]) * 100, 1),
            "gld_return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="CP Spike → Long GLD")
    rets_arr = [e["gld_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long GLD 42 days when COMPOUT 4-week change > +10%",
        "mechanism": "CP market stress → backstage liquidity pressure → gold flight-to-quality",
        "source": "FRED COMPOUT; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
