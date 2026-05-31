"""PL590_single_country_etf_central_bank_event_fade - BOJ Event EWJ Premium Fade
Short EWJ at close on day before BOJ rate decision. Hold 5 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL590_single_country_etf_central_bank_event_fade"
    boj_dates = [
        "2010-01-26","2010-04-30","2010-08-10","2010-12-21",
        "2011-04-28","2011-08-04","2011-12-21",
        "2012-04-27","2012-09-19","2013-04-04","2013-10-31",
        "2014-04-30","2014-10-31","2015-04-30","2015-12-18",
        "2016-01-29","2016-09-21","2017-04-27","2017-12-21",
        "2018-04-27","2018-12-20","2019-04-25","2019-12-19",
        "2020-04-27","2020-12-18","2021-04-27","2021-12-17",
        "2022-04-28","2022-12-20","2023-04-28","2023-12-19",
        "2024-03-19","2024-07-31","2024-12-19","2025-01-24",
    ]
    hold = 5
    tickers = ["EWJ", "SPY"]
    try:
        px = load_prices(tickers, start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    ewj = ret["EWJ"]

    legs = []
    event_results = []
    for d in boj_dates:
        dt = pd.Timestamp(d)
        # Short entry at close of day before
        mask = ewj.index < dt
        if mask.sum() < 1:
            continue
        entry_idx = ewj.index[mask][-1]
        loc = ewj.index.get_loc(entry_idx) + 1
        end = min(loc + hold, len(ewj))
        if end - loc < 3:
            continue
        win = -1.0 * ewj.iloc[loc:end]
        legs.append(win)
        event_results.append({"trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="BOJ EWJ Event Fade")
    save_result(sid, m, extra={
        "rule": "Short EWJ at close of day before BOJ meeting, exit at close T+5.",
        "mechanism": "BOJ event-day Japan timezone NAV mismatch -> premium fade",
        "source": "BOJ schedule + yfinance",
        "n_events": len(event_results),
        "events": event_results[:10],
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
