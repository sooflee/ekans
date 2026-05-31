"""PL584_tic_foreign_equity_outflow_ath_proximity_defensive - TIC Outflow + SPY ATH -> Defensive Rotation
Short SPY, long GLD + TLT on signal dates. 60-day hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL584_tic_foreign_equity_outflow_ath_proximity_defensive"
    events_raw = ["2007-09-15", "2007-12-15", "2015-08-15", "2018-02-15",
                  "2018-10-15", "2020-02-15", "2021-12-15", "2022-04-15",
                  "2023-10-15"]
    hold = 60
    tickers = ["SPY", "GLD", "TLT"]
    try:
        px = load_prices(tickers, start="2005-01-01")
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
        spy_leg = -0.5 * ret["SPY"].iloc[loc:end]
        gld_leg = 0.25 * ret["GLD"].iloc[loc:end] if "GLD" in ret.columns else 0
        tlt_leg = 0.25 * ret["TLT"].iloc[loc:end] if "TLT" in ret.columns else 0
        net = spy_leg + gld_leg + tlt_leg
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="TIC Outflow Defensive Rotation")
    save_result(sid, m, extra={
        "rule": "Short SPY -0.5, long GLD +0.25, long TLT +0.25 when TIC 2-mo equity outflow + SPY near ATH.",
        "mechanism": "Foreign de-risking + sentiment top tells -> tail hedge premium",
        "source": "Treasury TIC + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
