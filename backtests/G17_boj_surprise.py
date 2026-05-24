"""
G-17 BoJ surprise YCC tweaks.

Hardcoded surprise dates. For each: short USDJPY (i.e., long JPY) at T+1 NY open, exit T+5 close.
Use yfinance "JPY=X" which is USD/JPY -- so "short USDJPY" = -1 * pct change in JPY=X.
Equivalent: long JPYUSD; we just negate USDJPY returns.

Tiny N (3 events).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

EVENTS = [
    "2022-12-20",  # widened YCC band 0.5%
    "2023-10-31",  # made 1% upper bound a "reference"
    "2024-03-19",  # ended NIRP, dropped YCC framework
]


def main():
    try:
        px = load_prices(["JPY=X"], start="2010-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("G17_boj_surprise", f"FX load failed: {e}")
    rets = px.pct_change()
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    used = []
    for d in EVENTS:
        D = pd.Timestamp(d)
        loc = idx.searchsorted(D)
        if loc >= len(idx):
            continue
        # T+1 entry: shift to next trading day
        i_T = idx.get_loc(idx[loc]) if idx[loc] == D else loc
        if i_T + 1 >= len(idx):
            continue
        entry = i_T + 1
        exit_i = min(i_T + 5, len(idx) - 1)
        # Short USDJPY -> pos = -1 on USDJPY return
        pos.iloc[entry:exit_i + 1] = -1.0
        used.append(d)

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    spy = load_prices(["SPY"], start="2010-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="G17 BoJ surprise short USDJPY")
    print_metrics(m)
    save_result("G17_boj_surprise", m, extra={
        "status": "ok",
        "rule": "On each BoJ-surprise date D, short USDJPY (=long JPY) at T+1 open, exit T+5 close.",
        "universe": "JPY=X (USDJPY)",
        "events": used,
        "n_events": len(used),
        "source": "BoJ press releases / market commentary",
        "notes": "Very small N (3 events).",
    })


if __name__ == "__main__":
    main()
