"""
AD-O1 Crypto-mixer SDN add → compliance basket long.

Rule: When OFAC adds a mixer / privacy-tool to the SDN list, long an
equal-weight basket of compliance / mining names (COIN, RIOT, CLSK).
Optionally short XMR/USD as the privacy-tool short leg.

Hand-coded events:
- 2022-05-06 Blender.io SDN add (Treasury press)
- 2022-08-08 Tornado Cash SDN add
- 2023-11-29 Sinbad SDN add (Blender spin-up)
- 2024-04-24 Samourai Wallet DOJ unsealing (criminal case, OFAC-adjacent)

Hold: T+0 close → T+10 close.
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
    {"date": "2022-05-06", "name": "Blender.io"},
    {"date": "2022-08-08", "name": "Tornado Cash"},
    {"date": "2023-11-29", "name": "Sinbad"},
    {"date": "2024-04-24", "name": "Samourai Wallet (DOJ)"},
]

LONG_TICKERS = ["COIN", "RIOT", "CLSK"]
SHORT_TICKERS = ["XMR-USD"]


def basket_ret_at_event(px, tickers, d_event, hold_days=10):
    """Compute equal-weight basket return from T+0 close → T+hold_days close.
    T+0 is the trading day on or after d_event."""
    available = [t for t in tickers if t in px.columns]
    if not available:
        return np.nan, []
    rets = []
    used = []
    for t in available:
        s = px[t].dropna()
        if s.empty:
            continue
        i = s.index.searchsorted(d_event, side="left")
        if i >= len(s) or i + hold_days >= len(s):
            continue
        p0 = s.iloc[i]
        p1 = s.iloc[i + hold_days]
        if p0 <= 0:
            continue
        rets.append(float(p1 / p0 - 1))
        used.append(t)
    if not rets:
        return np.nan, []
    return float(np.mean(rets)), used


def main():
    all_t = list(set(LONG_TICKERS + SHORT_TICKERS))
    try:
        px = load_prices(all_t, start="2021-01-01")
    except Exception as e:
        return mark_failed("AD-O1", f"price load failed: {e}")

    if px.empty:
        return mark_failed("AD-O1", "no price data")

    rows = []
    for ev in EVENTS:
        d = pd.Timestamp(ev["date"])
        long_r, long_used = basket_ret_at_event(px, LONG_TICKERS, d, hold_days=10)
        short_r, short_used = basket_ret_at_event(px, SHORT_TICKERS, d, hold_days=10)
        pair = long_r - short_r if not np.isnan(short_r) else long_r
        rows.append({
            "date": d.date().isoformat(),
            "name": ev["name"],
            "long_ret_pct": long_r * 100 if not np.isnan(long_r) else None,
            "long_used": long_used,
            "short_ret_pct": short_r * 100 if not np.isnan(short_r) else None,
            "short_used": short_used,
            "pair_ret_pct": pair * 100 if not np.isnan(pair) else None,
        })

    valid = [r for r in rows if r["pair_ret_pct"] is not None]
    if len(valid) < 3:
        return mark_failed("AD-O1", f"insufficient events with data ({len(valid)})")

    rets = np.array([r["pair_ret_pct"] / 100.0 for r in valid])
    avg = float(rets.mean())
    sd = float(rets.std())
    se = sd / np.sqrt(len(rets)) if sd > 0 else np.nan
    t_stat = avg / se if se and se > 0 else 0.0
    hit = float((rets > 0).mean())
    sharpe = (avg / sd) * np.sqrt(252 / 10) if sd > 0 else 0.0

    print(f"AD-O1 mixer SDN-add basket (long COIN/RIOT/CLSK, short XMR), N={len(valid)}")
    for r in rows:
        print(f"  {r['date']} {r['name']}: long={r['long_ret_pct']}  short={r['short_ret_pct']}  pair={r['pair_ret_pct']}")
    print(f"  mean pair={avg*100:.2f}%  t-stat={t_stat:.2f}  hit={hit*100:.0f}%  sharpe~{sharpe:.2f}")

    metrics = {
        "name": "AD-O1 mixer SDN-add",
        "n_events": len(valid),
        "mean_event_ret_pct": avg * 100,
        "stdev_event_ret_pct": sd * 100,
        "t_stat": t_stat,
        "sharpe_approx": sharpe,
        "hit_rate": hit,
    }
    extra = {
        "status": "ok",
        "rule": "On OFAC mixer SDN designation, long (COIN+RIOT+CLSK)/3 "
                "and short XMR-USD; hold 10 trading days.",
        "mechanism": "Mixer designations push capital toward US-compliant "
                     "venues and accelerate institutional adoption of "
                     "monitored chains, while privacy coins face exchange "
                     "delisting risk.",
        "source": "OFAC press releases (manually curated); yfinance prices.",
        "events": rows,
        "caveats": [
            "Very small N (4).",
            "XMR/USD is illiquid via yfinance; short leg may not be implementable.",
            "Samourai is DOJ rather than OFAC.",
        ],
    }
    save_result("AD-O1", metrics, extra=extra)


if __name__ == "__main__":
    main()
