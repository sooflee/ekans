"""PL574_fda_clinical_hold_lift_8k_biotech_long - FDA Clinical Hold Lift 8-K -> Biotech Long T+1..T+15
Curated 8-K Item 8.01 clinical-hold-lift events. Long the issuer T+1..T+15, hedged with -50% XBI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL574_fda_clinical_hold_lift_8k_biotech_long"
    # (ticker, filing_date) -- curated representative dates from EDGAR full-text search 2015-2025
    events_raw = [
        ("SRPT", "2016-12-19"),
        ("BLUE", "2018-10-25"),
        ("BLUE", "2021-06-07"),
        ("RCKT", "2022-12-12"),
        ("TENX", "2017-09-15"),
        ("ATHX", "2019-07-29"),
        ("KRYS", "2020-05-04"),
        ("CRSP", "2021-04-15"),
        ("EDIT", "2018-05-30"),
        ("VYNE", "2020-06-08"),
        ("KNSA", "2023-02-21"),
        ("MGNX", "2019-08-19"),
        ("ARQT", "2022-03-31"),
        ("VSTM", "2024-05-13"),
        ("XBIT", "2018-11-13"),
    ]
    hold = 15
    tickers = sorted(set([t for t, _ in events_raw] + ["XBI", "SPY"]))
    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    xbi_r = ret["XBI"] if "XBI" in ret.columns else None

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 5:
            continue
        long_leg = ret[tkr].iloc[loc:end]
        hedge_leg = -0.5 * xbi_r.iloc[loc:end] if xbi_r is not None else 0
        net = long_leg + hedge_leg
        legs.append(net)
        cum = float((1 + net).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "net_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA Clinical Hold Lift Biotech Long")
    save_result(sid, m, extra={
        "rule": "Long issuer T+1..T+15 after 8-K Item 8.01 clinical hold lift, -50% XBI hedge",
        "mechanism": "Clinical hold lift -> renewed clinical optionality + short-cover pressure",
        "source": "SEC EDGAR full-text search + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
