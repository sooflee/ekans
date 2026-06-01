"""PL889_brazil_arabica_frost_window_lapse_short_kc - Brazil Arabica Frost Window Lapse -> Short KC=F Premium Decay

Annual event study: Sep 1 entry each year when frost window closes without a major
Brazil frost (no sub-2C recorded). Short KC=F front-month + long HSY+SBUX as
COGS-relief beneficiary. Filter: KC=F price up >15% May-Aug as proxy for spec premium build.

Frost years (exclude): 1994, 1997, 2021 (significant), 2000, 2013 (minor scares but included).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


HOLD_DAYS = 25  # ~5 weeks trading days

# Known major frost years to exclude (significant Brazil frost events)
FROST_YEARS = {1994, 1997, 2021}


def main():
    sid = "PL889_brazil_arabica_frost_window_lapse_short_kc"
    try:
        px = load_prices(["KC=F", "HSY", "SBUX", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    if "SPY" not in ret.columns:
        return mark_failed(sid, "SPY missing")
    spy_r = ret["SPY"]

    if "KC=F" not in ret.columns:
        return mark_failed(sid, "KC=F (coffee futures) missing from yfinance")

    pnl_parts = []
    events_detail = []

    for year in range(2001, 2025):
        if year in FROST_YEARS:
            continue

        # Check KC=F premium build: price change May 1 - Aug 31
        may1 = pd.Timestamp(f"{year}-05-01")
        aug31 = pd.Timestamp(f"{year}-08-31")
        kc_window = px["KC=F"].loc[
            (px.index >= may1) & (px.index <= aug31)
        ].dropna()

        if len(kc_window) < 20:
            continue  # insufficient data

        kc_may = kc_window.iloc[0]
        kc_aug = kc_window.iloc[-1]
        kc_change = (kc_aug - kc_may) / kc_may

        # Filter: require >15% price increase May-Aug (spec premium build)
        if kc_change < 0.15:
            continue

        # Sep 1 entry
        sep1 = pd.Timestamp(f"{year}-09-01")
        mask = ret.index >= sep1
        if mask.sum() < 2:
            continue
        idxs = ret.index[mask]
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 15:
            continue

        window = ret.iloc[loc:end]

        # Build PnL: short KC=F + long 0.5 HSY + 0.5 SBUX
        kc_ret = -window["KC=F"] if "KC=F" in window.columns else pd.Series(0, index=window.index)

        long_legs = []
        for ticker in ["HSY", "SBUX"]:
            if ticker in window.columns and window[ticker].notna().sum() > 10:
                long_legs.append(window[ticker])

        if long_legs:
            long_basket = sum(l / len(long_legs) for l in long_legs)
            # PnL = 0.5 short KC=F + 0.5 long basket (equal dollar notional)
            pnl_window = 0.5 * kc_ret + 0.5 * long_basket
        else:
            # Only short KC=F vs SPY
            pnl_window = kc_ret - window["SPY"]

        pnl_parts.append(pnl_window)
        cumret = float((1 + pnl_window).prod() - 1)
        kc_cumret = float((1 + kc_ret).prod() - 1)

        events_detail.append({
            "year": year,
            "entry": str(entry.date()),
            "kc_may_aug_change": round(kc_change, 3),
            "n_days": len(pnl_window),
            "net_return": round(cumret, 4),
            "kc_short_return": round(kc_cumret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no qualifying events (no years with KC>15% May-Aug premium build)")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="Brazil Frost Window Lapse Short KC")
    save_result(sid, m, extra={
        "rule": "Sep 1: short KC=F + long HSY/SBUX equal-weight, hold 25 trading days, when frost window closes without Brazil frost AND KC=F gained >15% May-Aug (spec premium build).",
        "mechanism": "Without a frost, spec-driven weather premium in Coffee C futures decays into Sep-Oct harvest. Long HSY/SBUX benefits from lower green coffee input costs.",
        "source": "yfinance KC=F continuous coffee futures; CFTC COT proxy via KC=F price momentum",
        "n_events": len(events_detail),
        "events": events_detail,
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events_detail])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events_detail])), 4),
        "frost_years_excluded": sorted(FROST_YEARS),
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events_detail)} qualifying events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")
    for e in events_detail:
        print(f"  {e['year']}: KC May-Aug={e['kc_may_aug_change']:+.1%} -> net={e['net_return']:+.3f}, KC_short={e['kc_short_return']:+.3f}")


if __name__ == "__main__":
    main()
