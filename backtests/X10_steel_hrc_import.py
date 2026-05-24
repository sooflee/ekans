"""
X10 US HRC import-parity divergence — long US steel when imports uneconomic.

Original spec: CME HRC settlements vs Mexican/world HRC export parity. Mexican
HRC export pricing not on free APIs.

Substitution: use HRC=F (CME Midwest HRC futures) absolute price and a momentum/
regime filter as proxy for "import parity is currently in favor of domestic
producers". When HRC=F rises > 20% in 60 sessions AND z-score of HRC vs its
trailing 252-day mean > +1, US-mill economics are clearly above import parity
(world prices typically lag). Long NUE+STLD+CLF equal-weight 60 trading days.

Source: yfinance HRC=F, yfinance steel equities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X10_steel_hrc_import"
    try:
        hrc = load_prices(["HRC=F"], start="2016-06-01").iloc[:, 0].rename("HRC")
        px = load_prices(["NUE", "STLD", "CLF", "MT", "SPY"], start="2016-06-01")
    except Exception as e:
        return mark_failed(sid, f"data load failed: {e}")

    needed = ["NUE", "STLD", "CLF"]
    if any(c not in px.columns for c in needed):
        return mark_failed(sid, f"missing tickers; have {list(px.columns)}")

    if hrc.index.tz is not None:
        hrc.index = hrc.index.tz_localize(None)
    hrc = hrc.dropna()
    # Regime filter
    hrc_60d = hrc.pct_change(60)
    win = 252
    hrc_z = (hrc - hrc.rolling(win).mean()) / hrc.rolling(win).std()
    trigger = (hrc_60d > 0.20) & (hrc_z > 1.0)
    first = trigger & ~trigger.shift(1, fill_value=False)
    entries = hrc.index[first]

    rets = daily_returns(px[["NUE", "STLD", "CLF", "SPY"]]).reindex(hrc.index)
    basket = (rets["NUE"] + rets["STLD"] + rets["CLF"]) / 3

    hold = 60
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
    m = compute_metrics(pnl, benchmark=bench, name="X10 HRC momentum long NUE+STLD+CLF 60d")
    m["n_events"] = n_events
    m["entries"] = used_entries[:30]
    m["event_returns"] = event_rets
    m["event_mean"] = float(np.mean(event_rets))
    m["event_hit"] = float(np.mean([r > 0 for r in event_rets]))
    print(f"X10: events={n_events}, mean={m['event_mean']*100:+.2f}%, hit={m['event_hit']*100:.0f}%")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When HRC=F 60d return > +20% AND HRC z-score over 252d > +1, long NUE+STLD+CLF equal-weight 60 trading days.",
        "mechanism": "Domestic HRC running well above its trailing distribution = imports uneconomic at current parity; US mills enjoy margin window before import response.",
        "source": "yfinance HRC=F (CME Midwest HRC futures) 2016-2025 + yfinance steel equities.",
        "substitution_note": "Original Mexican/world export-parity pricing not on free API; substituted HRC absolute-price regime filter.",
        "universe": ["NUE", "STLD", "CLF"],
        "hold_days": hold,
    })


if __name__ == "__main__":
    main()
