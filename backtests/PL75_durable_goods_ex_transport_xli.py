"""PL75_durable_goods_ex_transport_xli — Durable Goods ex-Transport YoY Positive After Decline → Long XLI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL75_durable_goods_ex_transport_xli"
    try:
        fred = load_fred("NEWORDER", start="1990-01-01")
        no = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if no.empty or len(no) < 24:
        return mark_failed(sid, "NEWORDER data insufficient")

    yoy = no.pct_change(12)
    yoy = yoy.dropna()

    trigger_dates = []
    neg_count = 0
    fired = False
    for i in range(len(yoy)):
        val = float(yoy.iloc[i])
        if np.isnan(val):
            continue
        if val < 0:
            neg_count += 1
            fired = False
        elif val >= 0 and neg_count >= 6 and not fired:
            trigger_dates.append(yoy.index[i])
            fired = True
            neg_count = 0
        else:
            neg_count = 0

    print(f"Durable goods ex-transport YoY turn events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: YoY = {yoy.loc[d]*100:.1f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no events found")

    try:
        px = load_prices(["XLI", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    xli_ret = ret["XLI"]
    spy_ret = ret["SPY"]
    hold_days = 126

    pnl_series = pd.Series(0.0, index=xli_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = xli_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = xli_ret.index[entry_mask][0]
        pos = xli_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(xli_ret))
        event_rets = xli_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "yoy": round(float(yoy.loc[td]) * 100, 1),
            "xli_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Durable Goods ex-Transport Turn → Long XLI")
    rets_arr = [e["xli_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long XLI 126 days when NEWORDER YoY turns positive after 6+ months negative",
        "mechanism": "Core capex orders inflection → manufacturing demand recovery",
        "source": "FRED NEWORDER; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
