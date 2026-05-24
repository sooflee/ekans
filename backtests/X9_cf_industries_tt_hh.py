"""
X9 CF Industries TTF-HH divergence — long CF when EU gas (TTF) crashes vs HH.

Rule: When TTF drops > 30% in 60 sessions while HH is flat (+/-10%) over same
window, long CF for 90 trading days.

Mechanism: CF Industries' US ammonia/urea operations price off Henry Hub gas
feedstock, but global ammonia/urea prices are set by marginal EU producers
priced off TTF. When TTF crashes faster than HH, US producers like CF retain
their HH-based feedstock cost advantage AND see global ammonia spot stay high
on slow inventory rebuild -> margin window opens.

Source: yfinance TTF=F (delayed ICE TTF front-month), FRED DHHNGSP, yfinance CF.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X9_cf_industries_tt_hh"
    try:
        ttf = load_prices(["TTF=F"], start="2017-01-01").iloc[:, 0].rename("TTF")
        hh = load_fred("DHHNGSP", start="2017-01-01").iloc[:, 0].rename("HH")
        px = load_prices(["CF", "NTR", "MOS", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    if "CF" not in px.columns:
        return mark_failed(sid, "CF price missing")

    # Drop timezone
    if ttf.index.tz is not None:
        ttf.index = ttf.index.tz_localize(None)
    df = pd.concat([ttf, hh], axis=1).dropna()
    df["ttf_60d"] = df["TTF"].pct_change(60)
    df["hh_60d"] = df["HH"].pct_change(60)

    trigger = (df["ttf_60d"] < -0.30) & (df["hh_60d"].abs() < 0.10)
    first = trigger & ~trigger.shift(1, fill_value=False)
    entries = df.index[first]

    rets = daily_returns(px[["CF", "SPY"]]).reindex(df.index)

    hold = 90
    pos = pd.Series(0.0, index=rets.index)
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
        sub = rets["CF"].iloc[i_entry:i_exit].fillna(0)
        event_rets.append(float((1 + sub).prod() - 1))
        used_entries.append(str(t.date()))

    n_events = len(event_rets)
    if n_events < 3:
        return mark_failed(sid, f"only {n_events} trigger events",
                           extra={"n_events": n_events})

    pnl = (pos * rets["CF"]).dropna()
    pnl = pnl.loc[pnl.ne(0).cumsum() > 0]
    if len(pnl) < 60:
        return mark_failed(sid, f"too few active days ({len(pnl)})")

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="X9 CF TTF-crash long 90d")
    m["n_events"] = n_events
    m["entries"] = used_entries[:30]
    m["event_returns"] = event_rets
    m["event_mean"] = float(np.mean(event_rets))
    m["event_hit"] = float(np.mean([r > 0 for r in event_rets]))
    print(f"X9: events={n_events}, mean={m['event_mean']*100:+.2f}%, hit={m['event_hit']*100:.0f}%")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When TTF 60d return < -30% AND HH 60d return |.| < 10%, long CF 90 trading days.",
        "mechanism": "Global ammonia/urea priced off TTF (marginal EU producer); CF feedstock cost set by HH. Asymmetric crash opens margin window for CF before global ammonia spot adjusts.",
        "source": "yfinance TTF=F (ICE TTF front-month delayed) 2017-2025, FRED DHHNGSP, yfinance CF.",
        "universe": ["CF"],
        "hold_days": hold,
    })


if __name__ == "__main__":
    main()
