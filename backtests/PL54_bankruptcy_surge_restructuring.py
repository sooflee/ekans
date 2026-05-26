"""PL54_bankruptcy_surge_restructuring — Bankruptcy Filings Surge → Long Restructuring Advisors
TNBCBSL not available on FRED CSV. Hand-code bankruptcy surge periods from ABI data.
Long HLI+LAZ+PJT for 252 days during bankruptcy surge periods.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL54_bankruptcy_surge_restructuring"

    # Hand-coded: quarters when US business bankruptcy filings surged >30% YoY
    # for 2+ consecutive quarters, from ABI/US Courts/S&P data.
    # These mark the start of major restructuring waves.
    events_handcoded = [
        ("2008-10-01", "GFC bankruptcy wave — Lehman cascade"),
        ("2009-04-01", "GFC continuation — GM/Chrysler"),
        ("2015-10-01", "Energy/commodity bankruptcy wave — oil crash"),
        ("2020-04-01", "COVID bankruptcy wave — retail/hospitality"),
        ("2023-04-01", "Post-ZIRP rate shock bankruptcy wave — SVB era"),
    ]

    trigger_dates = [pd.Timestamp(d) for d, _ in events_handcoded]

    print(f"Bankruptcy surge events (hand-coded): {len(trigger_dates)}")
    for d, desc in events_handcoded:
        print(f"  {d}: {desc}")

    try:
        px = load_prices(["HLI", "LAZ", "PJT", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    available = [t for t in ["HLI", "LAZ", "PJT"] if t in ret.columns]
    if len(available) == 0 or "SPY" not in ret.columns:
        return mark_failed(sid, f"missing tickers (available: {list(ret.columns)})")

    print(f"Basket tickers available: {available}")

    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 252

    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []

    for i, td in enumerate(trigger_dates):
        entry_mask = basket_ret.index >= td
        if entry_mask.sum() < hold_days:
            if entry_mask.sum() < 30:
                continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        if end_pos - pos < 30:
            continue
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:min(spy_pos + hold_days, len(spy_ret))]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "description": events_handcoded[i][1],
            "basket_used": available,
            "basket_12m_return": round(cumret, 4),
            "spy_12m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Bankruptcy Surge → Long Restructuring")
    rets_arr = [e["basket_12m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long HLI+LAZ+PJT for 252 days during bankruptcy surge periods",
        "mechanism": "Bankruptcy surge → restructuring advisory fee boom with 2-3 quarter lag",
        "source": "ABI/US Courts data (hand-coded surge periods); yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
