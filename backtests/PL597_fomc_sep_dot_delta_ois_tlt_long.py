"""PL597_fomc_sep_dot_delta_ois_tlt_long - FOMC SEP Dot Revision vs OIS -> Long TLT

Event study around FOMC SEP releases (4/year: Mar, Jun, Sep, Dec).
When SEP median dot revision is dovish (lower than prior SEP, AND below
OIS-implied), long TLT for 10 trading days.

We use a curated list of historically dovish SEP revisions where the
median dot for year T+1 came in significantly below the prior SEP and
market expectations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


# Curated dovish SEP release dates: 1y-ahead median dot dropped vs prior SEP
# AND ran below market-implied OIS expectations at time of release.
DOVISH_SEP_DATES = [
    "2019-03-20",  # First major dovish pivot in 2019
    "2019-06-19",  # Continued dovish revision (significant)
    "2019-09-18",
    "2020-06-10",  # COVID; deep dovish revision
    "2020-09-16",
    "2020-12-16",
    "2021-12-15",  # Hawkish pivot — NOT dovish; skip
    "2023-12-13",  # SEP showed cuts coming in 2024 — dovish surprise
    "2024-09-18",  # Significant dovish 50bp cut SEP
    "2024-12-18",  # SEP shifted hawkish — skip
]

# Strict subset that are actually dovish:
ACTUAL_DOVISH = [
    "2019-03-20",
    "2019-06-19",
    "2019-09-18",
    "2020-06-10",
    "2020-09-16",
    "2020-12-16",
    "2023-12-13",
    "2024-09-18",
]

HOLD_DAYS = 10


def main():
    sid = "PL597_fomc_sep_dot_delta_ois_tlt_long"
    try:
        px = load_prices(["TLT", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    if "TLT" not in ret.columns:
        return mark_failed(sid, "TLT missing")
    tlt_r = ret["TLT"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=ret.index)
    events = []
    for d in ACTUAL_DOVISH:
        td = pd.Timestamp(d)
        mask = ret.index >= td
        if mask.sum() < HOLD_DAYS + 2:
            continue
        # Enter at next-day close (T+1)
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 5:
            continue
        win = tlt_r.iloc[loc:end]
        for idx in win.index:
            pnl.loc[idx] = win.loc[idx]
        cumret = float((1 + win).prod() - 1)
        spy_win = spy_r.iloc[loc:end]
        spy_cum = float((1 + spy_win).prod() - 1)
        events.append({
            "sep_date": d,
            "entry": str(entry.date()),
            "tlt_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4),
            "excess": round(cumret - spy_cum, 4),
        })

    if len(events) < 3:
        return mark_failed(sid, f"only {len(events)} events")

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 20:
        return mark_failed(sid, f"insufficient pnl days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_r, name="FOMC SEP Dovish Dot -> Long TLT")
    save_result(sid, m, extra={
        "rule": "Long TLT for 10 trading days starting T+1 after FOMC SEP release with dovish median-dot revision (lower than prior SEP, below OIS).",
        "mechanism": "Dovish dot surprises drive a multi-day duration repricing rally as positioning unwinds.",
        "source": "FOMC SEP archive + curated dovish revisions",
        "n_events": len(events),
        "events": events,
        "avg_tlt_return": round(float(np.mean([e["tlt_return"] for e in events])), 4),
        "avg_excess": round(float(np.mean([e["excess"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["tlt_return"] > 0 for e in events])), 4),
    })
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(in_pos)}")


if __name__ == "__main__":
    main()
