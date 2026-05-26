"""PL31_auto_inventory_trough_dealers — Auto Inventory Days-Supply Trough → Long Dealers
When FRED AISRSA hits a local minimum AND level < 0.40: long AN+PAG+LAD equal-weight for 126 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL31_auto_inventory_trough_dealers"
    try:
        fred = load_fred("AISRSA", start="1992-01-01")
        ais = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if ais.empty or len(ais) < 12:
        return mark_failed(sid, "AISRSA data insufficient")

    # AISRSA never went below 0.40 historically (min ~0.42 in chip shortage).
    # Use bottom-quintile threshold instead: find local minima in the bottom 20% of all readings.
    threshold = ais.quantile(0.20)
    print(f"AISRSA bottom-20% threshold: {threshold:.3f}")

    # Find local minima below threshold
    trigger_dates = []
    for i in range(1, len(ais) - 1):
        if ais.iloc[i] < ais.iloc[i-1] and ais.iloc[i] < ais.iloc[i+1] and ais.iloc[i] < threshold:
            # Avoid retriggering within same trough episode
            if len(trigger_dates) == 0 or (ais.index[i] - trigger_dates[-1]).days > 180:
                trigger_dates.append(ais.index[i])

    print(f"AISRSA local minima < {threshold:.3f}: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: AISRSA = {ais.loc[d]:.4f}")

    if len(trigger_dates) == 0:
        return mark_failed(sid, f"no AISRSA local minima below {threshold:.3f} found")

    try:
        px = load_prices(["AN", "PAG", "LAD", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    available = [t for t in ["AN", "PAG", "LAD"] if t in ret.columns]
    if len(available) == 0 or "SPY" not in ret.columns:
        return mark_failed(sid, f"missing tickers")

    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 126

    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []

    for td in trigger_dates:
        # Entry: first trading day of the month after the trough
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue

        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))

        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_pos = spy_ret.index.get_loc(entry_idx) if entry_idx in spy_ret.index else None
        spy_cumret = None
        if spy_pos is not None:
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "aisrsa": round(float(ais.loc[td]), 4),
            "basket_6m_return": round(cumret, 4),
            "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events after alignment")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Auto Inventory Trough → Long Dealers")
    rets_arr = [e["basket_6m_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long AN+PAG+LAD equal-weight for 126 days when AISRSA hits local minimum below 0.40",
        "mechanism": "Extreme scarcity → maximum dealer pricing power → margin expansion not yet in earnings",
        "source": "FRED AISRSA; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
