"""PL579_doi_insurance_withdrawal_aiz_long_homebuilder_short - DOI Insurance Withdrawal -> Long AIZ / Short Homebuilders
Pair trade triggered on state DOI announcement dates. Hold 90 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL579_doi_insurance_withdrawal_aiz_long_homebuilder_short"
    events_raw = ["2022-05-31", "2023-05-26", "2023-06-09", "2023-07-13",
                  "2024-03-20", "2024-07-15", "2023-09-15"]
    hold = 90
    tickers = ["AIZ", "PHM", "LEN", "SPY"]
    try:
        px = load_prices(tickers, start="2020-01-01")
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
        if end - loc < 30:
            continue
        aiz = ret["AIZ"].iloc[loc:end]
        builders = ret[["PHM", "LEN"]].mean(axis=1).iloc[loc:end]
        net = 0.5 * aiz - 0.5 * builders
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="DOI Insurance Withdrawal -> Long AIZ / Short Builders")
    save_result(sid, m, extra={
        "rule": "Long AIZ + short PHM/LEN basket at T+1 after DOI withdrawal announcement, hold 90d.",
        "mechanism": "Insurance withdrawal -> force-placed insurer (AIZ) revenue boost + homebuilder demand drag",
        "source": "State DOI public announcements + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
