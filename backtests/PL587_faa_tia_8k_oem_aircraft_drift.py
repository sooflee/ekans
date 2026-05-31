"""PL587_faa_tia_8k_oem_aircraft_drift - FAA TIA Issuance -> OEM Drift
30-day long after TIA 8-K, equal weight across active legs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL587_faa_tia_8k_oem_aircraft_drift"
    # (ticker, TIA event date) - curated TIA disclosures
    events_raw = [
        ("BA",   "2018-11-13"),   # 737 MAX TIA-era proxies
        ("BA",   "2020-09-30"),
        ("BA",   "2022-01-31"),   # 777X-related TIA progress
        ("TXT",  "2019-03-18"),
        ("TXT",  "2021-10-15"),
        ("JOBY", "2023-06-28"),
        ("JOBY", "2024-03-05"),
        ("ACHR", "2023-08-15"),
        ("ACHR", "2024-05-09"),
        ("EH",   "2024-04-19"),   # EHang TC progression
    ]
    hold = 30
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
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        win = ret[tkr].iloc[loc:end]
        legs.append(win)
        event_results.append({"ticker": tkr, "trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FAA TIA OEM Drift")
    save_result(sid, m, extra={
        "rule": "Long OEM T+1 after FAA TIA disclosure 8-K, hold 30d.",
        "mechanism": "TIA milestone de-risks certification -> equity rerating",
        "source": "EDGAR / press releases + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
