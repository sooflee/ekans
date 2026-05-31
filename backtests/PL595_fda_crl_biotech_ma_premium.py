"""PL595_fda_crl_biotech_ma_premium - CRL on CMC -> 6w wait then 360-day Long
Long after FDA CMC-CRL; designed to capture later M&A premium.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL595_fda_crl_biotech_ma_premium"
    # (ticker, CRL date) — curated CMC/manufacturing CRLs where Phase 3 endpoint met
    events_raw = [
        ("IMGN", "2022-08-19"),    # ImmunoGen CMC CRL before AbbVie acquisition
        ("AIMT", "2019-08-05"),    # Aimmune Palforzia (later Nestle acquisition)
        ("ACAD", "2021-04-05"),    # Acadia Nuplazid dementia CRL
        ("SAGE", "2019-03-19"),    # Sage Zulresso/Zuranolone manufacturing
        ("KALA", "2018-07-25"),
        ("AKBA", "2019-09-09"),
        ("NVAX", "2020-08-20"),
        ("AGEN", "2021-12-20"),
        ("ATRA", "2020-12-15"),
        ("DCPH", "2022-10-20"),
        ("CRBP", "2021-07-19"),
        ("MNKD", "2018-02-15"),
    ]
    wait = 30
    hold = 180
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2017-01-01")
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
        mask = ret.index > dt
        idxs = ret.index[mask]
        if len(idxs) < wait + 30:
            continue
        entry = idxs[wait]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
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
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA CRL CMC -> M&A Premium")
    save_result(sid, m, extra={
        "rule": "Wait 30 trading days after CMC-only CRL, then long sponsor 180d.",
        "mechanism": "Post-panic discount + intact pipeline -> M&A premium re-rate",
        "source": "FDA CRL disclosures + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
