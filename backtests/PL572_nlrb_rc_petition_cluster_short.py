"""PL572_nlrb_rc_petition_cluster_short - NLRB Form RC Petition Cluster -> Short Employer (Event Study)
Short the parent ticker after a 5+ petition cluster trigger. Hold 90 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL572_nlrb_rc_petition_cluster_short"
    events_raw = [
        ("SBUX", "2021-12-09"),
        ("AMZN", "2022-04-25"),
        ("AAPL", "2022-06-15"),
        ("CMG",  "2022-08-25"),
        ("TSLA", "2024-09-10"),
    ]
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    hold = 90

    legs = []  # per-event short PnL series, indexed by event window
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        # next-day entry
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        win = -1.0 * ret[tkr].iloc[loc:end]  # short = inverted
        legs.append(win)
        cum = float((1 + win).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "short_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")

    # Aggregate: equal-weight overlapping shorts
    df = pd.concat(legs, axis=1)
    df.columns = [f"leg_{i}" for i in range(len(legs))]
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="NLRB RC Petition Cluster Short")
    save_result(sid, m, extra={
        "rule": "Short parent ticker T+1 after NLRB RC petition cluster trigger (>=5 in trailing 90d). Equal-weight overlapping legs. Hold 90d.",
        "mechanism": "Labor organization clusters signal cost / reputational pressure -> margin compression",
        "source": "NLRB historical case search + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
