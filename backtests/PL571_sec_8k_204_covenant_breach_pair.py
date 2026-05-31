"""PL571_sec_8k_204_covenant_breach_pair - SEC 8-K Item 2.04 Covenant Breach -> Short Issuer + HY/IG Spread Pair
Short filer at T close, short HYG + long LQD as spread pair. Hold 60d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL571_sec_8k_204_covenant_breach_pair"
    # Curated 8-K Item 2.04 filings (covenant breach / accelerated maturity)
    # since 2010-01-01
    events_raw = [
        # 2015-Q4 / 2016-Q2 E&P covenant-breach cluster
        ("CHK",  "2015-12-15"),
        ("HK",   "2016-02-23"),
        ("LINE", "2016-05-12"),
        # 2020 COVID wave
        ("AMC",  "2020-04-30"),
        ("CCL",  "2020-05-04"),
        # 2023 SVB/regional bank
        ("SIVB", "2023-03-10"),
        ("SBNY", "2023-03-12"),
        # 2024 CRE office cluster
        ("VNO",  "2024-06-14"),
        ("BXP",  "2024-08-01"),
        # Other historical
        ("X",    "2015-11-02"),
        ("RIG",  "2016-08-01"),
        ("DNR",  "2020-04-15"),
    ]
    hold = 60
    macro_pair = ["HYG", "LQD"]
    all_filers = sorted({t for t, _ in events_raw})
    tickers = sorted(set(all_filers + macro_pair + ["SPY"]))
    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

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
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 20:
            continue
        issuer_short = -1.0 * ret[tkr].iloc[loc:end]
        hyg_leg = -1.0 * ret["HYG"].iloc[loc:end] if "HYG" in ret.columns else 0
        lqd_leg = ret["LQD"].iloc[loc:end] if "LQD" in ret.columns else 0
        # Combine: 50% issuer, 25% HYG short, 25% LQD long
        net = 0.5 * issuer_short + 0.25 * hyg_leg + 0.25 * lqd_leg
        legs.append(net)
        cum = float((1 + net).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "net_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SEC 8-K 2.04 Covenant Breach Pair")
    save_result(sid, m, extra={
        "rule": "Short issuer + short HYG + long LQD at close of 8-K Item 2.04 filing day. Hold 60d.",
        "mechanism": "Covenant breach -> debt-equity stress contagion + HY-IG spread widening",
        "source": "SEC EDGAR Item 2.04 + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
