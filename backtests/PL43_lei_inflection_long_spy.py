"""PL43_lei_inflection_long_spy — Conference Board LEI 6mo RoC Turns Positive → Long SPY
When USSLIND 6-month annualized RoC turns positive after 6+ months negative: long SPY 252 days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL43_lei_inflection_long_spy"
    try:
        fred = load_fred("USSLIND", start="1970-01-01")
        lei = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if lei.empty or len(lei) < 12:
        return mark_failed(sid, "USSLIND data insufficient")

    # Compute 6-month annualized rate of change
    roc_6m = ((lei / lei.shift(6)) ** 2 - 1) * 100
    roc_6m = roc_6m.dropna()

    # Find months where RoC turns positive after 6+ consecutive negative months
    trigger_dates = []
    neg_count = 0
    for i in range(len(roc_6m)):
        if roc_6m.iloc[i] < 0:
            neg_count += 1
        else:
            if neg_count >= 6 and roc_6m.iloc[i] > 0:
                trigger_dates.append(roc_6m.index[i])
            neg_count = 0

    print(f"LEI 6mo RoC positive inflection (after 6+ neg months): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: RoC = {roc_6m.loc[d]:.2f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no LEI inflection events found")

    # Use ^GSPC for pre-1993, SPY for post
    try:
        px_spy = load_prices("SPY", start="1993-01-01")
        spy = px_spy.squeeze() if isinstance(px_spy, pd.DataFrame) else px_spy
    except Exception as e:
        return mark_failed(sid, f"SPY load: {e}")

    try:
        px_gspc = load_prices("^GSPC", start="1970-01-01")
        gspc = px_gspc.squeeze() if isinstance(px_gspc, pd.DataFrame) else px_gspc
    except Exception:
        gspc = None

    # Combine: use ^GSPC for earlier, SPY for later
    spy_ret = daily_returns(px_spy)
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]

    if gspc is not None:
        gspc_ret = daily_returns(px_gspc)
        if isinstance(gspc_ret, pd.DataFrame):
            gspc_ret = gspc_ret.iloc[:, 0]
        # Combine: gspc before spy start, spy after
        combined_ret = pd.concat([gspc_ret[gspc_ret.index < spy_ret.index[0]], spy_ret])
    else:
        combined_ret = spy_ret

    hold_days = 252
    pnl_series = pd.Series(0.0, index=combined_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = combined_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = combined_ret.index[entry_mask][0]
        pos = combined_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(combined_ret))
        event_rets = combined_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        event_results.append({
            "trigger_date": str(td.date()),
            "lei_roc_6m": round(float(roc_6m.loc[td]), 2),
            "spy_12m_return": round(cumret, 4),
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="LEI Inflection → Long SPY")
    rets_arr = [e["spy_12m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long SPY for 252 days when LEI 6mo RoC turns positive after 6+ negative months",
        "mechanism": "LEI trough signals recession bottom → SPY rallies in early recovery",
        "source": "FRED USSLIND; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
