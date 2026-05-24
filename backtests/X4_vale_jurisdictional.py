"""
X4 VALE jurisdictional discount mean-reversion.

Original spec: VALE EV/EBITDA vs (BHP+RIO)/2 EV/EBITDA combined with no-ANM-news
filter. EV/EBITDA history not readily available without 10-Q/F-1 parsing.

Substitution: log(VALE/BHP) ratio z-score over 252 trading days. When z < -1.5,
go long VALE 90 trading days. (ANM news filter dropped; honest disclosure.)

Mechanism: VALE periodically trades at deep discount to Australian peers due to
Brazilian regulatory/dam-safety overhangs (Mariana 2015, Brumadinho 2019).
When the discount becomes extreme (>1.5 sd), the gap historically closes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X4_vale_jurisdictional"
    try:
        px = load_prices(["VALE", "BHP", "RIO", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load failed: {e}")

    if any(c not in px.columns for c in ["VALE", "BHP", "RIO", "SPY"]):
        return mark_failed(sid, f"missing tickers; have {list(px.columns)}")

    # log ratio vs BHP/RIO blend
    peer = (px["BHP"] + px["RIO"]) / 2
    ratio = np.log(px["VALE"] / peer).dropna()
    win = 252
    rmean = ratio.rolling(win).mean()
    rstd = ratio.rolling(win).std()
    z = (ratio - rmean) / rstd

    trigger = z < -1.5
    first = trigger & ~trigger.shift(1, fill_value=False)
    entries = ratio.index[first]

    rets = daily_returns(px[["VALE", "BHP", "RIO", "SPY"]])
    vale_ret = rets["VALE"].reindex(ratio.index)

    hold = 90
    pos = pd.Series(0.0, index=vale_ret.index)
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
        sub = vale_ret.iloc[i_entry:i_exit].fillna(0)
        event_rets.append(float((1 + sub).prod() - 1))
        used_entries.append(str(t.date()))

    n_events = len(event_rets)
    if n_events < 3:
        return mark_failed(sid, f"only {n_events} trigger events",
                           extra={"n_events": n_events, "z_min": float(z.min())})

    pnl = (pos * vale_ret).dropna()
    pnl = pnl.loc[pnl.ne(0).cumsum() > 0]
    if len(pnl) < 60:
        return mark_failed(sid, f"too few active days ({len(pnl)})")

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="X4 VALE/BHP z<-1.5 long 90d")
    m["n_events"] = n_events
    m["entries"] = used_entries[:30]
    m["event_returns"] = event_rets
    m["event_mean"] = float(np.mean(event_rets))
    m["event_hit"] = float(np.mean([r > 0 for r in event_rets]))
    print(f"X4: n_events={n_events}, mean={m['event_mean']*100:+.2f}%, hit={m['event_hit']*100:.0f}%")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When log(VALE / mean(BHP,RIO)) z-score over 252d falls below -1.5, long VALE for 90 trading days.",
        "mechanism": "Jurisdictional discount mean-reversion; structural BR overhang creates periodic dislocations that historically close.",
        "source": "yfinance daily closes 2008-2025.",
        "substitution_note": "Original EV/EBITDA spread + ANM news filter not feasible; substituted with price-ratio z-score and dropped ANM filter (honest disclosure).",
        "universe": ["VALE", "BHP", "RIO"],
        "hold_days": 90,
    })


if __name__ == "__main__":
    main()
