"""
AD-O3 Venezuela General License → CVX/REP/ENI long.

Rule: When Treasury issues / renews a Venezuela General License authorizing
oil-sector activity, long CVX 50% + REP.MC 25% + ENI.MI 25%; hold T+0 close
to T+15 close.

Hand-coded events:
- 2022-11-26 GL 41 issued (Chevron VEN authorization, Sat)
- 2023-10-18 GL 44 broad oil/gas authorization
- 2024-04-17 GL 41B / partial renewal
- 2024-10-18 GL 44A renewal (assumed mid-Oct based on Treasury news; verify)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


EVENTS = [
    {"date": "2022-11-28", "name": "GL 41 (Chevron VEN)"},   # Nov 26 was Sat
    {"date": "2023-10-18", "name": "GL 44 (broad oil/gas)"},
    {"date": "2024-04-17", "name": "GL 41B renewal"},
    {"date": "2024-10-18", "name": "GL 44A renewal (est.)"},
]

WEIGHTS = {"CVX": 0.5, "REP.MC": 0.25, "ENI.MI": 0.25}


def weighted_ret(px, d_event, hold_days=15):
    rets = {}
    for tk, w in WEIGHTS.items():
        if tk not in px.columns:
            continue
        s = px[tk].dropna()
        if s.empty:
            continue
        i = s.index.searchsorted(d_event, side="left")
        if i >= len(s) or i + hold_days >= len(s):
            continue
        p0 = s.iloc[i]
        p1 = s.iloc[i + hold_days]
        rets[tk] = float(p1 / p0 - 1)
    if not rets:
        return np.nan, {}
    # If some tickers missing, renormalize weights.
    total_w = sum(WEIGHTS[t] for t in rets)
    blended = sum(rets[t] * WEIGHTS[t] / total_w for t in rets)
    return float(blended), rets


def main():
    tickers = list(WEIGHTS.keys())
    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed("AD-O3", f"price load failed: {e}")
    if px.empty:
        return mark_failed("AD-O3", "no data")

    rows = []
    for ev in EVENTS:
        d = pd.Timestamp(ev["date"])
        r, parts = weighted_ret(px, d, hold_days=15)
        rows.append({
            "date": ev["date"],
            "name": ev["name"],
            "ret_pct": r * 100 if not np.isnan(r) else None,
            "parts_pct": {k: v * 100 for k, v in parts.items()},
        })

    valid = [r for r in rows if r["ret_pct"] is not None]
    if len(valid) < 3:
        return mark_failed("AD-O3", f"insufficient events ({len(valid)})")

    rets = np.array([r["ret_pct"] / 100.0 for r in valid])
    avg = float(rets.mean())
    sd = float(rets.std())
    se = sd / np.sqrt(len(rets)) if sd > 0 else np.nan
    t_stat = avg / se if se and se > 0 else 0.0
    hit = float((rets > 0).mean())
    sharpe = (avg / sd) * np.sqrt(252 / 15) if sd > 0 else 0.0

    print(f"AD-O3 Venezuela GL → CVX/REP/ENI basket, N={len(valid)} (T+15 hold)")
    for r in rows:
        print(f"  {r['date']} {r['name']}: ret={r['ret_pct']}  parts={r['parts_pct']}")
    print(f"  mean ret={avg*100:.2f}%  t-stat={t_stat:.2f}  hit={hit*100:.0f}%")

    metrics = {
        "name": "AD-O3 Venezuela GL basket",
        "n_events": len(valid),
        "mean_event_ret_pct": avg * 100,
        "stdev_event_ret_pct": sd * 100,
        "t_stat": t_stat,
        "sharpe_approx": sharpe,
        "hit_rate": hit,
    }
    extra = {
        "status": "ok",
        "rule": "On Venezuela GL issuance/renewal, long (CVX 50% + REP.MC 25% + "
                "ENI.MI 25%); hold T+0 to T+15 close.",
        "mechanism": "GLs unlock cash repatriation and oil-lift volumes; "
                     "named primes (Chevron / Repsol / Eni) and adjacent "
                     "European E&Ps benefit asymmetrically vs broader XLE.",
        "source": "Treasury OFAC press / GL pages (manually curated); yfinance prices.",
        "events": rows,
        "caveats": [
            "Small N (4).",
            "GL 44A renewal date estimated; verify in production use.",
            "REP.MC and ENI.MI quoted in EUR — FX exposure mixes into returns.",
        ],
    }
    save_result("AD-O3", metrics, extra=extra)


if __name__ == "__main__":
    main()
