"""PL1058_potato_stocks_lw_fry_squeeze — USDA Dec 1 Potato Stocks Glut -> Lamb Weston (LW) Frozen-Fry Margin Squeeze

Primary data source (USDA NASS Quick Stats API) requires an API key we don't
have, so this uses the documented fallback from the strategy spec:
FRED WPU011306 (PPI Farm Products: Potatoes, monthly).

Fallback signal (per spec implementation_notes):
  December PPI YoY < -10%  -> potato price collapse = glut proxy  -> SHORT LW
  December PPI YoY > +15%  -> short-crop proxy                    -> LONG LW
  otherwise flat that year.

Timing note: the NASS Potato Stocks report is released mid-December (spec
entry: first trading day on/after Dec 21), but the December PPI print is only
released mid-January. To avoid lookahead with the PPI fallback, entry is the
first trading day on/after Jan 21 of the following year. Hold 126 trading
days, exit at close. One signal per year, unit notional, SPY-beta-hedged
(beta from trailing 252d OLS).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns, print_metrics)

GLUT_THRESH = -0.10   # Dec PPI YoY below this -> glut -> short LW
SHORT_CROP_THRESH = 0.15  # Dec PPI YoY above this -> short crop -> long LW
HOLD_DAYS = 126


def yearly_signals(ppi, lo=GLUT_THRESH, hi=SHORT_CROP_THRESH):
    """Return {dec_year: (direction, yoy)} from December PPI YoY."""
    dec = ppi[ppi.index.month == 12]
    yoy = dec.pct_change().dropna()
    out = {}
    for dt, v in yoy.items():
        if v < lo:
            out[dt.year] = (-1, float(v))   # glut -> short LW
        elif v > hi:
            out[dt.year] = (1, float(v))    # short crop -> long LW
        else:
            out[dt.year] = (0, float(v))
    return out


def main():
    sid = "PL1058_potato_stocks_lw_fry_squeeze"
    try:
        px = load_prices(["LW", "SPY"], start="2016-11-10")
        fred = load_fred(["WPU011306"], start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    lw_r = ret["LW"].dropna()
    spy_r = ret["SPY"].dropna()
    idx = lw_r.index.intersection(spy_r.index)
    lw_r, spy_r = lw_r.reindex(idx), spy_r.reindex(idx)

    sigs = yearly_signals(fred["WPU011306"])

    pnl = pd.Series(0.0, index=idx)
    pos = pd.Series(0.0, index=idx)  # for transaction costs
    events = []
    # LW IPO 2016-11-10 -> first usable December signal is Dec 2016
    for dec_year in sorted(y for y in sigs if y >= 2016):
        direction, yoy = sigs[dec_year]
        if direction == 0:
            events.append({"dec_year": dec_year, "ppi_dec_yoy": round(yoy, 3),
                           "direction": 0, "note": "flat (|signal| below threshold)"})
            continue

        # entry: first trading day on/after Jan 21 of following year
        # (Dec PPI print released mid-January; avoids lookahead)
        entry_cutoff = pd.Timestamp(f"{dec_year + 1}-01-21")
        future = idx[idx >= entry_cutoff]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_loc = idx.get_loc(entry_date)
        exit_loc = min(entry_loc + HOLD_DAYS, len(idx) - 1)
        if exit_loc <= entry_loc + 5:
            continue
        window = idx[entry_loc:exit_loc + 1]

        # trailing 252d beta of LW vs SPY estimated strictly before entry
        hist = slice(max(0, entry_loc - 252), entry_loc)
        lw_h, spy_h = lw_r.iloc[hist], spy_r.iloc[hist]
        beta = 1.0
        if len(lw_h) > 60 and spy_h.var() > 0:
            beta = float(np.cov(lw_h, spy_h)[0, 1] / spy_h.var())
        beta = float(np.clip(beta, 0.0, 2.0))

        # PnL applied from the day AFTER entry close through exit close
        trade_days = window[1:]
        leg = direction * (lw_r.reindex(trade_days) - beta * spy_r.reindex(trade_days))
        pnl.loc[trade_days] += leg.fillna(0).values
        pos.loc[window[:-1]] = direction

        lw_cum = float((1 + lw_r.reindex(trade_days)).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(trade_days)).prod() - 1)
        hedged_cum = float((1 + leg.fillna(0)).prod() - 1)
        events.append({
            "dec_year": dec_year,
            "ppi_dec_yoy": round(yoy, 3),
            "direction": direction,
            "entry_date": str(entry_date.date()),
            "exit_date": str(idx[exit_loc].date()),
            "beta": round(beta, 2),
            "lw_return": round(lw_cum, 4),
            "spy_return": round(spy_cum, 4),
            "hedged_trade_return": round(hedged_cum, 4),
            "partial": bool(exit_loc == len(idx) - 1 and entry_loc + HOLD_DAYS > len(idx) - 1),
        })
        print(f"  Dec {dec_year}: YoY {yoy*100:+.1f}% -> {'LONG' if direction>0 else 'SHORT'} LW "
              f"{entry_date.date()} -> {idx[exit_loc].date()} (beta {beta:.2f}): "
              f"LW {lw_cum*100:+.1f}%  SPY {spy_cum*100:+.1f}%  hedged {hedged_cum*100:+.1f}%")

    traded = [e for e in events if e.get("direction")]
    if len(traded) < 3:
        return mark_failed(sid, f"too few traded events ({len(traded)})")

    active = pnl[pos.reindex(idx).fillna(0).shift(1).fillna(0) != 0]
    if len(active) < 60:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, positions=pos,
                        name="USDA Dec Potato Stocks Glut -> LW Fry Margin Squeeze")

    # threshold robustness (not optimization): alternate fallback thresholds
    robustness = {}
    for lo, hi in [(-0.05, 0.10), (-0.15, 0.20)]:
        alt = yearly_signals(fred["WPU011306"], lo=lo, hi=hi)
        same = sum(1 for y in alt if y >= 2016 and alt[y][0] == sigs[y][0])
        tot = sum(1 for y in alt if y >= 2016)
        robustness[f"lo={lo:+.2f}/hi={hi:+.2f}"] = f"{same}/{tot} years same signal"

    trade_rets = [e["hedged_trade_return"] for e in traded]
    save_result(sid, m, pnl=active, extra={
        "rule": ("Each December, take potato supply signal (NASS Dec 1 stocks YoY; "
                 "implemented via FRED WPU011306 potato PPI Dec YoY fallback: "
                 "< -10% = glut -> SHORT LW, > +15% = short crop -> LONG LW). "
                 "Enter close of first trading day on/after Jan 21 (PPI release lag), "
                 "hold 126 trading days, SPY-beta-hedged, one trade per year."),
        "mechanism": ("LW is the dominant pure-play NA frozen-fry processor with "
                      "fixed-price restaurant contracts; a raw-potato glut signals "
                      "contracted-acreage oversupply, capacity underutilization and "
                      "guidance cuts 1-2 quarters later (e.g. Dec 2023 record crop -> "
                      "LW FY2024-25 collapse); short crops give pricing power."),
        "source": ("USDA NASS Potato Stocks (proxied by FRED WPU011306, PPI Farm "
                   "Products: Potatoes); yfinance LW/SPY"),
        "n_events": len(traded),
        "hit_rate_events": round(float(np.mean([r > 0 for r in trade_rets])), 3),
        "avg_event_return": round(float(np.mean(trade_rets)), 4),
        "events": events,
        "threshold_robustness": robustness,
        "caveats": ("NASS Quick Stats API key unavailable -> used documented FRED PPI "
                    "fallback; entry shifted Dec 21 -> Jan 21 to avoid lookahead on the "
                    "Dec PPI release. PPI proxy diverges from NASS stocks in Dec 2024 "
                    "(+17.1% YoY base effect off the 2023 collapse -> LONG, whereas "
                    "actual stocks showed continued oversupply -> NASS rule would be "
                    "SHORT). Only 9 annual events; Dec 2025 trade still open as of run "
                    "date (partial)."),
    })
    print(f"\nTraded events: {len(traded)}, active days: {len(active)}")
    print_metrics(m)


if __name__ == "__main__":
    main()
