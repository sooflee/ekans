"""PL50_sloos_tightening_long_tlt — SLOOS Net Tightening > +40% → Long TLT
When DRTSCILM crosses above +40: long TLT for 189 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL50_sloos_tightening_long_tlt"
    try:
        fred = load_fred("DRTSCILM", start="1990-01-01")
        sloos = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if sloos.empty:
        return mark_failed(sid, "DRTSCILM data empty")

    # Find cross above +40 from below
    trigger_dates = []
    for i in range(1, len(sloos)):
        if sloos.iloc[i] > 40 and sloos.iloc[i-1] <= 40:
            trigger_dates.append(sloos.index[i])

    print(f"SLOOS tightening cross above +40: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: DRTSCILM = {sloos.loc[d]:.1f}")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no SLOOS tightening events found")

    try:
        px = load_prices(["TLT", "SPY"], start="2002-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "TLT" not in ret.columns or "SPY" not in ret.columns:
        return mark_failed(sid, "missing tickers")

    tlt_ret = ret["TLT"]
    spy_ret = ret["SPY"]
    hold_days = 189

    pnl_series = pd.Series(0.0, index=tlt_ret.index)
    event_results = []

    for td in trigger_dates:
        # Entry: first trading day of the following quarter
        entry_date = td + pd.offsets.QuarterBegin(1)
        entry_mask = tlt_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = tlt_ret.index[entry_mask][0]
        pos = tlt_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(tlt_ret))
        event_rets = tlt_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "drtscilm": round(float(sloos.loc[td]), 1),
            "tlt_9m_return": round(cumret, 4),
            "spy_9m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment (TLT starts 2002)")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="SLOOS Tightening → Long TLT")
    rets_arr = [e["tlt_9m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long TLT for 189 days when DRTSCILM crosses above +40",
        "mechanism": "Major credit tightening leads Fed rate cuts by 2-4 quarters",
        "source": "FRED DRTSCILM; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
