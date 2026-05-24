"""
S9 WTI calendar spread tightness -> long USO (proxy).

Original rule: When WTI M1-M2 spread > $2/bbl backwardation, long USO 6 weeks.
yfinance exposes CL=F (front-month) only — no public free second-month feed.

Substitution (per spec): use a proxy for market tightness — momentum surge
in WTI: if CL=F front-month rises > 5% in the last 10 trading days
(2 weeks), enter long USO for 30 trading days (~6 weeks). Non-overlapping.

Mechanism: A sharp 2-week front-month rally is a robust correlate of
backwardation episodes (term-structure flips backwardated when physical
markets tighten); historically tends to continue for several weeks before
mean-reverting.

Source: yfinance CL=F, USO. Spec explicitly authorizes this momentum proxy:
"If next-month data hard, use a proxy: long USO when WTI rises > 5% in 2 weeks."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


def main():
    try:
        cl = load_prices(["CL=F"], start="2000-01-01").iloc[:, 0]
        uso = load_prices(["USO"], start="2006-04-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S9_wti_calendar_spread", f"yfinance load failed: {e}")

    df = pd.concat({"cl": cl, "uso": uso}, axis=1).dropna()
    if len(df) < 200:
        return mark_failed("S9_wti_calendar_spread", f"insufficient overlap ({len(df)} obs)")

    df["uso_ret"] = df["uso"].pct_change()
    df["cl_2w"] = df["cl"].pct_change(10)

    # Trigger: WTI front up > 5% over last 10 trading days
    triggers = df.index[df["cl_2w"] > 0.05].tolist()
    print(f"CL=F 2-week >5% days: {len(triggers)}")

    HOLD = 30
    pos = pd.Series(0.0, index=df.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d in triggers:
        idx = df.index.get_loc(d)
        start_idx = idx + 1
        if start_idx >= len(df.index):
            continue
        start = df.index[start_idx]
        if last_end is not None and start <= last_end:
            continue
        end_idx = min(start_idx + HOLD, len(df.index))
        for j in range(start_idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = df.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S9_wti_calendar_spread",
                           f"no qualifying triggers (max 2w cl chg={df['cl_2w'].max():.2%})")

    pnl = (pos.shift(1) * df["uso_ret"]).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = df["uso_ret"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S9 WTI 2w>+5% momentum proxy -> long USO 30d")
    m["n_events"] = n_events
    print(f"Non-overlap events: {n_events}; first: {event_dates[:5]}")
    print_metrics(m)

    save_result("S9_wti_calendar_spread", m, extra={
        "status": "ok",
        "rule": "Proxy: when WTI front-month (CL=F) is up > 5% over the prior 10 trading days, long USO next session for 30 trading days; non-overlapping events.",
        "mechanism": "A 2-week rally in WTI front-month is a robust correlate of backwardation / physical tightness episodes; spot strength historically persists for several weeks before term-structure reset.",
        "source": "yfinance CL=F, USO. Substitution from original spec: M2 contract not available via free public Yahoo feed; per spec instructions, used the 2-week WTI-up >5% momentum proxy for tightness.",
        "n_events": n_events,
        "first_events": event_dates[:5],
        "substitution_note": "Original rule (M1-M2 > +$2/bbl backwardation) requires next-month settlement data not free-available; used spec-authorized momentum proxy.",
    })


if __name__ == "__main__":
    main()
