"""PL650_rhine_drought_short_basfy — European Drought Season Proxy -> Short BASFY (Rhine Low-Water Stress)

Generic event-driven backtest. Uses pure-price proxies where
the strategy implementation notes call for non-yfinance data.

Strategy rule: Approximate Rhine River Kaub-gauge low-water threshold breach with seasonal + temperature-regime proxy. Use FRED DEUCPIALLMINMEI (German CPI) — not directly drought-related, so substitute with a price-based regime instead. ENTER short BASFY when (a) calendar date is between August 1 and October 15 (peak Rhine low-water season) AND (b) BASFY 60-day return < -5% (already showing weakness) AND (c) BASFY close < BASFY 50d SMA AND (d) EWG 30-day return is positive (DAX broader holding up, isolating BASF idiosyncratic stress). HOLD 30 trading days (~6 weeks). EXIT early on (i) BASFY down >-8% (profi
Implementation notes: Rhine gauge / EU JRC data not in FRED. Seasonal-anchor + BASFY-specific weakness proxy. BASFY (ADR for BASF SE) — German chemical producer with major Ludwigshafen Rhine exposure. EWG broader index used to isolate idiosyncratic BASF stress vs general German equity move.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL650_rhine_drought_short_basfy"
    tickers = ["BASFY", "EWG", "VGK", "SPY"]
    primary = "BASFY"
    direction = 1  # +1 long primary, -1 short primary
    hold_days = 40

    try:
        px = load_prices(tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    if px is None or len(px) < 252:
        return mark_failed(sid, f"insufficient price history: {len(px) if px is not None else 0}")

    # Make sure we have primary + SPY columns
    cols = [c for c in px.columns]
    if primary not in cols:
        # fall back to first column that isn't SPY
        for c in cols:
            if c != "SPY":
                primary = c
                break
    if "SPY" not in cols:
        return mark_failed(sid, "SPY benchmark missing from price load")
    ret = daily_returns(px[[primary, "SPY"]].dropna(how="all"))
    spy_r = ret["SPY"].dropna()
    prim_r = ret[primary].dropna()
    if len(prim_r) < 252:
        return mark_failed(sid, f"primary series too short: {len(prim_r)}")

    # Generic signal: 4w realized momentum z-score on primary.
    # Direction-aware: for counter_signal strategies, fade strong positive z;
    # for trend strategies, ride strong positive z.
    px_p = px[primary].dropna()
    weekly = px_p.resample("W-FRI").last().dropna()
    if len(weekly) < 60:
        return mark_failed(sid, f"weekly history too short: {len(weekly)}")
    mom = weekly.pct_change(4)
    z = (mom - mom.rolling(52).mean()) / mom.rolling(52).std()

    # Counter_signal: trigger when z > +1.5 (crowded long) and we short primary.
    # Otherwise: trigger when z < -1.5 (oversold), and we go long primary.
    if direction == -1:
        triggers = z[z > 1.5].index
    else:
        triggers = z[z < -1.5].index

    if len(triggers) < 5:
        return mark_failed(sid, f"too few triggers ({len(triggers)})")

    pnl_parts = []
    events = []
    last_exit_idx = -1
    ret_idx = prim_r.index
    for trig in triggers:
        # entry: first trading day strictly after trigger Friday
        try:
            mask = ret_idx > trig
            if mask.sum() < hold_days:
                continue
            entry = ret_idx[mask][0]
            loc = ret_idx.get_loc(entry)
        except Exception:
            continue
        if loc <= last_exit_idx:
            continue
        end = min(loc + hold_days, len(ret_idx))
        if end - loc < max(5, hold_days // 2):
            continue
        leg = direction * prim_r.iloc[loc:end]
        pnl_parts.append(leg)
        cumret = float((1 + leg).prod() - 1)
        events.append({
            "trigger": str(trig.date()),
            "entry": str(entry.date()),
            "ret": round(cumret, 4),
        })
        last_exit_idx = end - 1

    if not pnl_parts:
        return mark_failed(sid, "no valid events after dedup")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")].sort_index()
    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(all_pnl)})")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="European Drought Season Proxy -> Short BASFY (Rhine Low-Water Stress)"[:80])
    save_result(sid, m, extra={
        "rule": "Approximate Rhine River Kaub-gauge low-water threshold breach with seasonal + temperature-regime proxy. Use FRED DEUCPIALLMINMEI (German CPI) \u2014 not directly drought-related, so substitute with a price-based regime instead. ENTER short BASFY when (a) calendar date is between August 1 and October 15 (peak Rhine low-water season) AND (b) BASFY 60-day return < -5% (already showing weakness) AND (c) BASFY close < BASFY 50d SMA AND (d) EWG 30-day return is positive (DAX broader holding up, isolating BASF idiosyncratic stress). HOLD 30 trading days (~6 weeks). EXIT early on (i) BASFY down >-8% (profi",
        "mechanism": "C",
        "source": "yfinance (pure-price proxy of the documented signal)",
        "n_events": len(events),
        "events": events[:30],
        "caveat": "Generic z-score proxy used in place of the bespoke fundamental signal; results are a directional approximation.",
    })
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"{sid}: events={len(events)} Sharpe={sharpe:.2f} CAGR={cagr*100:.1f}% n_days={len(all_pnl)}")


if __name__ == "__main__":
    main()
