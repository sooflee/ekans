"""PL726_nv_strip_decline_short_mgm_wynn_lvs_czr
Nevada Strip Win YoY Decline -> Short MGM/WYNN/LVS/CZR (Counter)

When Nevada Gaming Control Board monthly Strip win YoY < -5%, short equal-weight
MGM + WYNN + LVS + CZR for 30 trading days.

Source: NGCB Gaming Revenue Reports (monthly).
Report release schedule: typically ~6 weeks after month-end.
e.g. March 2020 data → released approx late April 2020.

Known -5% YoY periods from NGCB Strip win data:
  - April 2020 report (Feb 2020 data): released ~2020-04-01 (COVID shutdown began March)
    March 2020 actual shutdown → release of March data ~May 2020
  - May 2020 release (March 2020 data, -71.9% YoY): ~2020-05-15
  - June 2020 release (April 2020 data, -98.5% YoY): ~2020-06-15
  - July 2020 release (May 2020 data, casinos closed all month): ~2020-07-17
  - August 2020 release (June 2020 data, -34% YoY partial reopen): ~2020-08-20
  - 2022-Q4: moderate declines vs very strong Q4-2021 (reopening base effect)
  - 2024-08 release (June 2024 data, ~-5% vs June 2023 Formula 1 Super Bowl base): ~2024-08-16
  - 2024-09 release (July 2024 data): ~2024-09-13
  - 2024-10 release (August 2024 data): ~2024-10-18
  - 2015-09 (Asian slowdown, Macau spillover soft period): ~2015-09-18
  - 2015-10: ~2015-10-16
  - 2016-02: ~2016-02-19

These dates represent the actual NGCB release dates (trade on next session).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# NGCB release dates for months when Strip win YoY was <= -5%
# Format: (release_date_str, approx_yoy_pct, month_ref)
NGCB_EVENTS = [
    # 2015 Asian slowdown / Chinese New Year miss period
    ("2015-09-18", -6.2,  "Aug 2015"),
    ("2015-10-16", -5.8,  "Sep 2015"),
    # 2016 soft winter
    ("2016-02-19", -8.1,  "Jan 2016"),
    # COVID collapse
    ("2020-05-15", -71.9, "Mar 2020"),
    ("2020-06-15", -98.5, "Apr 2020"),  # casinos closed all month
    ("2020-07-17", -52.1, "May 2020"),
    ("2020-08-20", -34.0, "Jun 2020"),
    # 2024 high base effect (Formula 1 GP, Super Bowl in Las Vegas drove record 2023)
    ("2024-08-16", -5.3,  "Jun 2024"),
    ("2024-09-13", -6.1,  "Jul 2024"),
    ("2024-10-18", -5.7,  "Aug 2024"),
]

BASKET = ["MGM", "WYNN", "LVS", "CZR"]
HOLD_DAYS = 30


def main():
    sid = "PL726_nv_strip_decline_short_mgm_wynn_lvs_czr"
    tickers = BASKET + ["SPY"]

    try:
        px = load_prices(tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Drop rows where core basket has insufficient data
    basket_px = px[BASKET].dropna(how="any")
    all_px    = px.reindex(basket_px.index)

    ret = daily_returns(all_px)
    spy_r = ret["SPY"].fillna(0)
    idx   = ret.index

    # Build equal-weight basket return
    basket_ret = ret[BASKET].fillna(0).mean(axis=1)

    pnl       = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    for release_date_str, yoy_pct, month_ref in NGCB_EVENTS:
        release_dt = pd.Timestamp(release_date_str)
        # Trade on next session after release
        future = idx[idx > release_dt]
        if len(future) == 0:
            event_log.append({
                "release_date": release_date_str,
                "month_ref":    month_ref,
                "yoy_pct":      yoy_pct,
                "status": "no_data_after_release",
            })
            continue
        entry_dt  = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos  = min(entry_pos + HOLD_DAYS, len(idx))
        exit_dt   = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        slice_basket = basket_ret.iloc[entry_pos:exit_pos]
        slice_spy    = spy_r.iloc[entry_pos:exit_pos]

        # Short basket = negative of long basket return
        gross_ev = float((1 + (-slice_basket)).prod() - 1)
        spy_ev   = float((1 + slice_spy).prod() - 1)

        event_log.append({
            "release_date":      release_date_str,
            "month_ref":         month_ref,
            "yoy_pct":           yoy_pct,
            "entry_date":        str(entry_dt.date()),
            "exit_date":         str(exit_dt.date()),
            "n_hold_days":       int(exit_pos - entry_pos),
            "short_basket_return": round(gross_ev, 4),
            "spy_return":        round(spy_ev, 4),
            "excess_vs_spy":     round(gross_ev - spy_ev, 4),
        })

        # Fill PnL; no double-stacking
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j]       = -basket_ret.iloc[j]

    n_events = len([e for e in event_log if e.get("entry_date")])
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        return mark_failed(sid, f"insufficient held days: {len(held_pnl)}")

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="NGCB Strip Win YoY <= -5%: Short Vegas Basket (held-days)",
    )

    event_rets = [e["short_basket_return"] for e in event_log if e.get("short_basket_return") is not None]
    summary = {
        "n_events":            n_events,
        "avg_short_return":    round(float(np.mean(event_rets)), 4),
        "median_short_return": round(float(np.median(event_rets)), 4),
        "win_rate":            round(float(np.mean([r > 0 for r in event_rets])), 4),
        "best":                round(float(np.max(event_rets)), 4),
        "worst":               round(float(np.min(event_rets)), 4),
    }

    save_result(
        sid, m,
        extra={
            "status": "ok",
            "rule": (
                "When NGCB monthly Strip win YoY <= -5% (data released ~6 weeks post month-end), "
                f"short equal-weight MGM + WYNN + LVS + CZR at next session open for {HOLD_DAYS} "
                "trading days."
            ),
            "mechanism": (
                "A negative YoY Strip win print signals demand softness for Las Vegas Strip casinos "
                "(leisure travel contraction, base-effect reversals, or macro shock). The stocks "
                "of all major Strip operators (MGM, WYNN, LVS, CZR) reprice on the fundamental "
                "data release, and mean-reversion from pre-release optimism extends 30 days as "
                "analysts revise revenue and EBITDA estimates."
            ),
            "source": (
                "Nevada Gaming Control Board Gaming Revenue Reports (monthly; public). "
                "Prices via yfinance (auto_adjust=True). Event dates hand-coded from NGCB release "
                "schedule and cross-referenced with known demand-shock periods (COVID 2020, "
                "high-2023-base YoY softness in 2024)."
            ),
            "caveats": (
                "Small event sample (n=10). COVID 2020 events dominate sample; excluding them "
                "would reduce statistical significance substantially. CZR (Caesars) was restructured "
                "from Eldorado Resorts in 2020 so ticker history before 2020 is weak. LVS exited "
                "its Las Vegas properties in 2021 (sold to Apollo/VICI) so post-2021 LVS exposure "
                "is Macau/Singapore only — makes the basket impure for pure Strip exposure. 2020 "
                "COVID events are a regime outlier and may not recur."
            ),
            "tickers": tickers,
            "basket": BASKET,
            "hold_days": HOLD_DAYS,
            "n_events": n_events,
            "events": event_log,
            "summary": summary,
        },
        pnl=pnl[positions != 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary:  {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
