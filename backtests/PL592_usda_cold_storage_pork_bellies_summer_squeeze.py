"""PL592_usda_cold_storage_pork_bellies_summer_squeeze - USDA Cold Storage Belly Draw -> Long TSN
Curated May/June trigger events. Hold ~10 weeks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL592_usda_cold_storage_pork_bellies_summer_squeeze"
    # Trigger dates for May/June USDA Cold Storage report releases with >2σ low belly stocks
    events_raw = ["2011-05-23", "2014-06-23", "2017-06-22", "2018-06-22",
                  "2021-05-24", "2022-05-25", "2023-05-22"]
    hold = 50  # ~10 weeks
    tickers = ["TSN", "SPY"]
    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 20:
            continue
        win = ret["TSN"].iloc[loc:end]
        legs.append(win)
        event_results.append({"trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="Pork Belly Cold Storage Squeeze")
    save_result(sid, m, extra={
        "rule": "Long TSN after USDA Cold Storage shows >2σ-low pork belly stocks in May/June. Hold ~10 weeks.",
        "mechanism": "Summer BLT demand + tight bellies -> belly price spike -> TSN pork margin lift",
        "source": "USDA NASS Cold Storage + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
