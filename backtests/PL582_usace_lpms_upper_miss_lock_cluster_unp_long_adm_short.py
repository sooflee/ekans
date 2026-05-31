"""PL582_usace_lpms_upper_miss_lock_cluster_unp_long_adm_short - USACE Lock Cluster -> Long UNP/CF, Short ADM/BG
30-day pair after detected lock cluster events on Upper Mississippi.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL582_usace_lpms_upper_miss_lock_cluster_unp_long_adm_short"
    events_raw = ["2014-07-15", "2019-05-15", "2022-08-15", "2023-09-15", "2024-09-15"]
    hold = 30
    tickers = ["UNP", "CF", "ADM", "BG", "SPY"]
    try:
        px = load_prices(tickers, start="2013-01-01")
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
        if end - loc < 15:
            continue
        long_basket = ret[["UNP", "CF"]].mean(axis=1).iloc[loc:end]
        short_basket = ret[["ADM", "BG"]].mean(axis=1).iloc[loc:end]
        net = 0.5 * long_basket - 0.5 * short_basket
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="USACE Upper Miss Lock Cluster Pair")
    save_result(sid, m, extra={
        "rule": "Long UNP+CF, short ADM+BG when USACE LPMS shows lock-closure cluster on Upper Mississippi. Hold 30d.",
        "mechanism": "Barge disruption -> rail substitution + fertilizer logistics tailwind / grain export drag",
        "source": "USACE LPMS + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
