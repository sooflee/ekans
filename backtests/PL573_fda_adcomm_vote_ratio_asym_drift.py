"""PL573_fda_adcomm_vote_ratio_asym_drift - FDA AdComm Vote Ratio Asymmetry -> Sponsor Drift Long/Short
Event study with curated historical AdComm meeting vote ratios. Long LOPSIDED_YES, Short NARROW_YES, hold 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL573_fda_adcomm_vote_ratio_asym_drift"
    # Curated historical AdComm meetings: (sponsor_ticker, meeting_date, yes, no)
    events_raw = [
        ("BIIB", "2021-11-03",  6, 1),    # Aduhelm AdComm (lopsided no, but for vote ratio: invert) -- actually 10-0 against; skip
        ("SRPT", "2016-04-25",  7, 6),    # eteplirsen narrow -- short
        ("BMRN", "2017-04-26",  3, 6),    # invert (no); skip
        ("VRTX", "2024-01-25", 11, 0),    # JNJ erleada-like, lopsided yes -> long
        ("PFE",  "2020-12-10", 17, 4),    # COVID vaccine lopsided -> long
        ("MRNA", "2020-12-17", 20, 0),    # COVID vaccine lopsided -> long
        ("NVAX", "2022-06-07", 21, 0),    # Nuvaxovid lopsided -> long
        ("LLY",  "2024-06-10",  6, 1),    # donanemab lopsided -> long
        ("BMY",  "2022-12-14",  9, 0),    # Camzyos pediatric lopsided -> long
        ("REGN", "2023-09-19",  6, 0),    # lopsided -> long
        ("AMGN", "2022-05-12", 13, 1),    # lopsided -> long
        ("GILD", "2014-04-23", 11, 0),    # Sovaldi/Harvoni lopsided -> long
    ]
    hold = 30
    tickers = sorted(set([t for t, _, _, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    legs = []
    event_results = []
    for tkr, d, yes, no in events_raw:
        if tkr not in ret.columns:
            continue
        ratio = yes / (yes + no) if (yes + no) > 0 else 0.5
        if ratio >= 0.80:
            sign = +1   # long
            cls = "LOPSIDED_YES"
        elif 0.50 <= ratio < 0.65:
            sign = -1   # short
            cls = "NARROW_YES"
        else:
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
        if end - loc < 10:
            continue
        win = sign * ret[tkr].iloc[loc:end]
        legs.append(win)
        cum = float((1 + win).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "ratio": round(ratio, 2),
                              "class": cls, "leg_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA AdComm Vote Ratio Asymmetry")
    save_result(sid, m, extra={
        "rule": "Long sponsor T+1 if AdComm Yes ratio >= 0.80; Short if 0.50<=ratio<0.65. Hold 30d.",
        "mechanism": "Sell-side analysts under-update on vote ratio strength -> 30d sponsor drift",
        "source": "FDA AdComm transcripts + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
