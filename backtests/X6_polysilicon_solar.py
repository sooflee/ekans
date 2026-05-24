"""
X6 Polysilicon/module-price bottoming -> long solar basket.

Original spec: PV Magazine weekly module price index. When module prices stabilize
after a major decline, long FSLR + JKS for 90 days.

Substitution: PV Magazine index requires scrape of paywalled weekly survey.
Public free-data proxy: use FSLR-vs-SPY relative drawdown regime as a *price-action*
substitute. FSLR is the dominant US solar pure-play; its drawdowns from 52w high
reflect both module price collapse AND broader sentiment. When FSLR has dropped
>25% in 90 days but realized 30-day vol has *fallen* below 30-day median (i.e.
stabilization), go long FSLR + JKS equal-weight for 90 trading days.

Mechanism (preserved): polysilicon/module pricing capitulation -> producer
supply discipline -> margin recovery in surviving names.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X6_polysilicon_solar"
    try:
        px = load_prices(["FSLR", "JKS", "CSIQ", "TAN", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load failed: {e}")

    if "FSLR" not in px.columns or "JKS" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, f"missing tickers; have {list(px.columns)}")

    fslr = px["FSLR"].dropna()
    if len(fslr) < 400:
        return mark_failed(sid, "insufficient FSLR history")

    # 90-day return
    ret_90d = fslr.pct_change(90)
    # vol regime: realized std over 20d vs trailing 60d median
    daily_r = fslr.pct_change()
    vol_20 = daily_r.rolling(20).std()
    vol_60_med = vol_20.rolling(60).median()
    vol_calm = vol_20 < vol_60_med  # below median = relative stabilization
    # 90-day drawdown threshold
    big_drop = ret_90d < -0.25

    trigger = big_drop & vol_calm
    first = trigger & ~trigger.shift(1, fill_value=False)
    entries = fslr.index[first]

    rets = daily_returns(px[["FSLR", "JKS", "SPY"]])
    basket = (rets["FSLR"] + rets["JKS"]) / 2

    hold = 90
    pos = pd.Series(0.0, index=basket.index)
    last_exit = -1
    event_rets = []
    used_entries = []
    for t in entries:
        if t not in pos.index:
            continue
        i = pos.index.get_loc(t)
        if i <= last_exit:
            continue
        i_entry = i + 1
        i_exit = min(i_entry + hold, len(pos) - 1)
        if i_exit <= i_entry + 5:
            continue
        pos.iloc[i_entry:i_exit] = 1.0
        last_exit = i_exit
        sub = basket.iloc[i_entry:i_exit].fillna(0)
        event_rets.append(float((1 + sub).prod() - 1))
        used_entries.append(str(t.date()))

    n_events = len(event_rets)
    if n_events < 3:
        return mark_failed(sid, f"only {n_events} trigger events",
                           extra={"n_events": n_events})

    pnl = (pos * basket).dropna()
    pnl = pnl.loc[pnl.ne(0).cumsum() > 0]
    if len(pnl) < 60:
        return mark_failed(sid, f"too few active days ({len(pnl)})")

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="X6 solar bottom long FSLR+JKS 90d")
    m["n_events"] = n_events
    m["entries"] = used_entries[:30]
    m["event_returns"] = event_rets
    m["event_mean"] = float(np.mean(event_rets))
    m["event_hit"] = float(np.mean([r > 0 for r in event_rets]))
    print(f"X6: n_events={n_events}, mean={m['event_mean']*100:+.2f}%, hit={m['event_hit']*100:.0f}%")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When FSLR 90-day return < -25% AND realized 20d vol < trailing 60d median, long FSLR+JKS equal-weight for 90 trading days.",
        "mechanism": "Polysilicon/module price capitulation -> supply discipline -> margin recovery in surviving names. Price-action proxy for fundamental bottoming.",
        "source": "yfinance daily closes 2012-2025.",
        "substitution_note": "Original PV Magazine weekly module price index not freely available; substituted with FSLR drawdown + vol-stabilization regime as price-action proxy.",
        "universe": ["FSLR", "JKS"],
        "hold_days": 90,
    })


if __name__ == "__main__":
    main()
