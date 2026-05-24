"""
G-15 US Presidential Election VIX ramp (long VIXY/VXX equity).

Hardcoded Nov election dates 2008-2024. For each, long VIXY from T-120 to T-5 trading days; close T+10.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

ELECTIONS = [
    "2008-11-04",
    "2012-11-06",
    "2016-11-08",
    "2020-11-03",
    "2024-11-05",
]

# Prefer VIXY (launched 2011). For 2008 we don't have a vol ETF history -> skip 2008.
# For 2012+ use VXX (or VIXY); choose VIXY as primary.


def main():
    # try VIXY first; fall back to VXX
    try:
        px = load_prices(["VIXY"], start="2011-01-01")["VIXY"]
    except Exception:
        px = None
    if px is None or px.dropna().empty:
        try:
            px = load_prices(["VXX"], start="2009-01-01")["VXX"]
        except Exception as e:
            return mark_failed("G15_election_vix", f"VIXY/VXX load failed: {e}")
        ticker_used = "VXX"
    else:
        ticker_used = "VIXY"
    rets = px.pct_change()
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    used = []
    for d in ELECTIONS:
        D = pd.Timestamp(d)
        if D < idx[0] or D > idx[-1]:
            continue
        loc = idx.searchsorted(D)
        if loc >= len(idx):
            continue
        T = idx[loc] if idx[loc] == D else idx[min(loc, len(idx) - 1)]
        i = idx.get_loc(T)
        # Long T-120 to T-5; close on T+10 (so we are flat from T-4 to T+10? spec is "close T+10").
        # Interpret literally: long position from T-120 to T-5 (entry T-120, exit T-5 close);
        # AND separately: close T+10 — we interpret as exit at T+10. So position spans T-120 to T-5
        # only? That's strict. Alternative reading: long T-120 through T+10 with a small carve at T-5.
        # We follow the literal interpretation: long T-120 to T-5 INCLUSIVE, flat thereafter.
        start = max(0, i - 120)
        end = max(0, i - 5)
        pos.iloc[start:end + 1] = 1.0
        used.append(d)

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    spy = load_prices(["SPY"], start="2009-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="G15 Election VIX ramp")
    print_metrics(m)
    save_result("G15_election_vix", m, extra={
        "status": "ok",
        "rule": "Long VIXY (fallback VXX) from T-120 to T-5 trading days where T = first Tue of "
                "Nov election. Tiny sample; suffers from contango drag.",
        "universe": ticker_used,
        "n_events_used": len(used),
        "events": used,
        "pct_days_long": float(pos.mean()),
        "source": "Pre-election vol ramp anecdote",
        "notes": "Honest: VIXY/VXX have severe roll-cost drag, so even a true vol ramp may be eaten.",
    })


if __name__ == "__main__":
    main()
