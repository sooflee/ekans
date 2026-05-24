"""
AD-O4 EO 14114 secondary-sanctions warnings → HK / China bank fade.

Rule: When Treasury publishes a press release with EO 14114 / secondary
sanctions / FFI / correspondent-account language naming Chinese banks,
short FXI 40% + EWH 40% + KWEB 20%; hold T+0 to T+10 close.

Hand-coded events (rough; verify in production):
- 2023-12-22 EO 14114 signed
- 2024-01-11 first OFAC FAQ on secondary sanctions for EO 14114
- 2024-03-19 Adeyemo Reuters interview warning Chinese banks (approx; use Mar
  18 as best public date)
- 2024-05-01 OFAC Russia-related action mentioning FFI risk (approx)

This is a SHORT basket; we report the return of the short basket as a
positive PnL when the basket goes down.
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
    {"date": "2023-12-22", "name": "EO 14114 signed"},
    {"date": "2024-01-11", "name": "OFAC FAQ on secondary sanctions"},
    {"date": "2024-03-18", "name": "Adeyemo warning"},
    {"date": "2024-05-01", "name": "OFAC Russia action w/ FFI ref"},
]

WEIGHTS = {"FXI": 0.40, "EWH": 0.40, "KWEB": 0.20}


def long_basket_ret(px, d_event, hold_days=10):
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
    total_w = sum(WEIGHTS[t] for t in rets)
    blended = sum(rets[t] * WEIGHTS[t] / total_w for t in rets)
    return float(blended), rets


def main():
    tickers = list(WEIGHTS.keys())
    try:
        px = load_prices(tickers, start="2023-01-01")
    except Exception as e:
        return mark_failed("AD-O4", f"price load failed: {e}")
    if px.empty:
        return mark_failed("AD-O4", "no data")

    rows = []
    for ev in EVENTS:
        d = pd.Timestamp(ev["date"])
        long_r, parts = long_basket_ret(px, d, hold_days=10)
        # short basket PnL = -long_r
        short_pnl = -long_r if not np.isnan(long_r) else np.nan
        rows.append({
            "date": ev["date"],
            "name": ev["name"],
            "long_basket_pct": long_r * 100 if not np.isnan(long_r) else None,
            "short_pnl_pct": short_pnl * 100 if not np.isnan(short_pnl) else None,
            "parts_pct": {k: v * 100 for k, v in parts.items()},
        })

    valid = [r for r in rows if r["short_pnl_pct"] is not None]
    if len(valid) < 3:
        return mark_failed("AD-O4", f"insufficient events ({len(valid)})")

    rets = np.array([r["short_pnl_pct"] / 100.0 for r in valid])
    avg = float(rets.mean())
    sd = float(rets.std())
    se = sd / np.sqrt(len(rets)) if sd > 0 else np.nan
    t_stat = avg / se if se and se > 0 else 0.0
    hit = float((rets > 0).mean())
    sharpe = (avg / sd) * np.sqrt(252 / 10) if sd > 0 else 0.0

    print(f"AD-O4 EO 14114 short basket (short FXI/EWH/KWEB), N={len(valid)} (T+10 hold)")
    for r in rows:
        print(f"  {r['date']} {r['name']}: long_basket={r['long_basket_pct']}  short_pnl={r['short_pnl_pct']}")
    print(f"  mean short PnL={avg*100:.2f}%  t-stat={t_stat:.2f}  hit={hit*100:.0f}%")

    metrics = {
        "name": "AD-O4 EO 14114 short HK/CN",
        "n_events": len(valid),
        "mean_event_ret_pct": avg * 100,
        "stdev_event_ret_pct": sd * 100,
        "t_stat": t_stat,
        "sharpe_approx": sharpe,
        "hit_rate": hit,
    }
    extra = {
        "status": "ok",
        "rule": "On Treasury press citing EO 14114 / secondary sanctions / FFI / "
                "correspondent account language naming Chinese banks, short "
                "FXI 40% + EWH 40% + KWEB 20%; hold T+0 to T+10.",
        "mechanism": "Markets under-discount cross-border banking-channel risk; "
                     "China-onshore-exposed ETFs drift down as banks dial down "
                     "dollar-clearing exposure in response.",
        "source": "Treasury / OFAC press archive (manually curated). yfinance prices.",
        "events": rows,
        "caveats": [
            "Small N (4).",
            "Adeyemo warning and May 1 dates are approximate.",
            "Macro China-policy noise dominates over single-press effects.",
        ],
    }
    save_result("AD-O4", metrics, extra=extra)


if __name__ == "__main__":
    main()
