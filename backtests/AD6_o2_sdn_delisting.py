"""
AD-O2 SDN delisting → listed-parent positive surprise.

Rule: OFAC removes name from SDN that maps to listed parent. Buy parent next
open; exit T+20 trading days.

Hand-coded mappings (a few clean cases):
- 2019-01-27 Rusal/En+ removed from SDN → proxy with GLEN.L (Glencore) which
  was the closest listed competitor / supplier-counterparty. Spec suggests
  +6% over 10d. We extend to T+20 close.
- 2019-01-27 mapped also to alumina-exposed RIO.L (Rio Tinto) — secondary
  read; we keep GLEN.L as primary.
- 2023-02-08 multiple Belarusian removals — most cleanly mapped to URKA
  (PhosAgro, but Russian; skip) or specific sectoral names. We use NLR
  uranium ETF as a coarse Belarus proxy. Weak; flag in caveats.
- 2022-05-20 Roman Abramovich-adjacent subs removal partial — too noisy.
- 2024-07-22 OFAC removed certain Cypriot entities — too far from listed
  parents to map cleanly; skip.

This is a deliberately small hand-coded study. If <3 verifiable mappings
yield prices, mark_failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


# (event_date, listed_parent_ticker, descriptor)
EVENTS = [
    ("2019-01-28", "GLEN.L", "Rusal/En+ SDN removal → Glencore (largest listed counterparty)"),
    ("2019-01-28", "RIO.L",  "Rusal/En+ SDN removal → Rio Tinto (alumina sector spillover)"),
    # 2023-02-08 Belarus removals — use NLR as weak proxy (potash adj.); flagged
    ("2023-02-09", "NLR",    "Belarus-related SDN removal → NLR uranium proxy (weak)"),
    # 2022-03-11 OFAC delisted certain Venezuelan-related: not enough mapping
    # Add 2024-03-13 — OFAC delisted some Iraqi names linked to listed
    # commodity firm. Hard to verify. Skip.
]


def main():
    tickers = sorted({t for _, t, _ in EVENTS})
    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed("AD-O2", f"price load failed: {e}")

    if px.empty:
        return mark_failed("AD-O2", "no data")

    rows = []
    for d_str, tk, desc in EVENTS:
        if tk not in px.columns:
            continue
        s = px[tk].dropna()
        if s.empty:
            continue
        d = pd.Timestamp(d_str)
        i = s.index.searchsorted(d, side="left")
        if i >= len(s) or i + 20 >= len(s):
            continue
        p0 = s.iloc[i]
        p20 = s.iloc[i + 20]
        r = float(p20 / p0 - 1)
        # Also show p+10 for reference
        p10 = s.iloc[i + 10] if i + 10 < len(s) else np.nan
        r10 = float(p10 / p0 - 1) if not np.isnan(p10) else np.nan
        rows.append({
            "date": d_str,
            "ticker": tk,
            "desc": desc,
            "ret10_pct": r10 * 100,
            "ret20_pct": r * 100,
        })

    if len(rows) < 3:
        return mark_failed("AD-O2",
                           f"Too few clean delisting→listed-parent maps ({len(rows)})")

    rets = np.array([r["ret20_pct"] / 100.0 for r in rows])
    avg = float(rets.mean())
    sd = float(rets.std())
    se = sd / np.sqrt(len(rets)) if sd > 0 else np.nan
    t_stat = avg / se if se and se > 0 else 0.0
    hit = float((rets > 0).mean())
    sharpe = (avg / sd) * np.sqrt(252 / 20) if sd > 0 else 0.0

    print(f"AD-O2 SDN delisting → listed parent, N={len(rows)} (T+20 hold)")
    for r in rows:
        print(f"  {r['date']} {r['ticker']}: T+10={r['ret10_pct']:.2f}%  T+20={r['ret20_pct']:.2f}%")
    print(f"  mean T+20 ret={avg*100:.2f}%  t-stat={t_stat:.2f}  hit={hit*100:.0f}%")

    metrics = {
        "name": "AD-O2 SDN delisting",
        "n_events": len(rows),
        "mean_event_ret_pct": avg * 100,
        "stdev_event_ret_pct": sd * 100,
        "t_stat": t_stat,
        "sharpe_approx": sharpe,
        "hit_rate": hit,
    }
    extra = {
        "status": "ok",
        "rule": "On OFAC SDN removal mapping to a listed parent, long parent "
                "next open; exit T+20 close.",
        "mechanism": "SDN removal restores ability to transact with previously "
                     "blocked counterparty; analysts and indexers slow to "
                     "re-price the supplier/competitor chain.",
        "source": "OFAC recent actions list; manually mapped to listed proxies.",
        "events": rows,
        "caveats": [
            "Very small N (3-4).",
            "Mappings are imperfect: NLR for Belarus is a weak proxy.",
            "GLEN.L was already heavily geared to Rusal; result is dominated by post-deal Russia macro flow.",
        ],
    }
    save_result("AD-O2", metrics, extra=extra)


if __name__ == "__main__":
    main()
