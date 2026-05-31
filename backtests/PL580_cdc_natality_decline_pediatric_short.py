"""PL580_cdc_natality_decline_pediatric_short - CDC Natality 3mo YoY < -4% -> Pediatric Short / Elder Long
180-day hold pair on curated CDC NCHS release-triggered dates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL580_cdc_natality_decline_pediatric_short"
    events_raw = ["2020-11-15", "2021-03-15", "2022-06-15", "2024-03-15", "2024-12-15"]
    hold = 180
    short_basket = ["KMB", "PG", "ABT", "BFAM", "SNY", "PFE", "MRK"]
    long_basket = ["EHC"]
    tickers = sorted(set(short_basket + long_basket + ["SPY"]))
    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        avail_s = [t for t in short_basket if t in ret.columns]
        avail_l = [t for t in long_basket if t in ret.columns]
        if not avail_s or not avail_l:
            continue
        short_leg = -1.0 * ret[avail_s].mean(axis=1).iloc[loc:end]
        long_leg = ret[avail_l].mean(axis=1).iloc[loc:end]
        net = (short_leg + long_leg) / 2.0
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="CDC Natality Pediatric Short / Elder Long")
    save_result(sid, m, extra={
        "rule": "Short pediatric/staples basket + long elder-care when CDC natality 3mo YoY < -4%. Hold 180d.",
        "mechanism": "Birth dearth -> long-cycle pediatric TAM impairment / elder TAM expansion",
        "source": "CDC NCHS Vital Stats Rapid Release + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
