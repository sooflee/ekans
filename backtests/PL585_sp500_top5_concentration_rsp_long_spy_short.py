"""PL585_sp500_top5_concentration_rsp_long_spy_short - SP500 Top-5 Concentration -> Long RSP / Short SPY
9-month hold pair trade on curated trigger dates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL585_sp500_top5_concentration_rsp_long_spy_short"
    events_raw = ["2020-08-31", "2020-12-31", "2021-09-30", "2023-07-31",
                  "2024-06-28", "2024-12-31"]
    hold = 189
    tickers = ["RSP", "SPY"]
    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < 30:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        net = ret["RSP"].iloc[loc:end] - ret["SPY"].iloc[loc:end]
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="Top-5 Concentration RSP/SPY Pair")
    save_result(sid, m, extra={
        "rule": "Long RSP +1.0, short SPY -1.0 when top-5 concentration >27% & SPY TTM > +20%. Hold 9mo.",
        "mechanism": "Cap-weight concentration extreme -> mean reversion to equal-weight",
        "source": "S&P 500 component data + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
