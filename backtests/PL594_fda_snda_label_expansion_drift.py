"""PL594_fda_snda_label_expansion_drift - FDA sNDA Label Expansion -> Sponsor 8-Week Long
40-day long after curated sNDA/sBLA expansion approvals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL594_fda_snda_label_expansion_drift"
    # (ticker, action_date) — curated label-expansion sNDA/sBLA approvals
    events_raw = [
        ("MRK",  "2017-05-23"),    # Keytruda 1L NSCLC
        ("MRK",  "2018-08-20"),    # Keytruda 1L NSCLC squamous
        ("ARGX", "2022-12-19"),    # Vyvgart gMG
        ("LLY",  "2023-11-08"),    # Zepbound
        ("NVO",  "2024-03-08"),    # Wegovy CV expansion
        ("MRK",  "2019-08-16"),    # Keytruda HNSCC
        ("BMY",  "2018-04-16"),    # Opdivo
        ("AZN",  "2020-12-30"),    # Tagrisso adjuvant
        ("GILD", "2022-02-14"),    # Trodelvy
        ("REGN", "2021-03-01"),    # Libtayo
        ("VRTX", "2019-10-21"),    # Trikafta
        ("PFE",  "2019-04-15"),    # Inlyta
        ("BIIB", "2017-12-22"),    # Spinraza expand
        ("LLY",  "2022-05-13"),    # Mounjaro initial (proxy)
        ("AMGN", "2022-08-12"),    # Tezspire pediatric
    ]
    hold = 40
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        win = ret[tkr].iloc[loc:end]
        legs.append(win)
        event_results.append({"ticker": tkr, "trigger_date": d,
                              "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA sNDA Label Expansion Drift")
    save_result(sid, m, extra={
        "rule": "Long sponsor at close of label-expansion sNDA/sBLA approval, hold 40d.",
        "mechanism": "Material TAM expansion under-priced by sell-side -> drift",
        "source": "Drugs@FDA + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
