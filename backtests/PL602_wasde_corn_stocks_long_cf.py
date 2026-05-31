"""
PL602 — USDA WASDE Corn Stocks/Use < 10% -> Long CF Industries Drift

Implementation uses the project's specified pure-price proxy because WASDE
stocks/use is not in FRED. Proxy:
  - "WASDE event date" = 2nd Tuesday of each month (standard WASDE convention).
  - Fire entry on event date if:
      (a) ZC=F front-month close >= $5.00/bu, AND
      (b) ZC=F has risen >+15% over trailing 90 trading days (tightening regime).
  - Long CF at next-day open.
  - Exit on the earlier of 50 trading days elapsed OR CF down >10% from entry close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL602_wasde_corn_stocks_long_cf"
NAME = "WASDE Corn Stocks/Use <10% Proxy -> Long CF (50d)"


def second_tuesday_of_month(start, end):
    """Return list of pd.Timestamp for the 2nd Tuesday of every month in [start, end]."""
    dates = []
    cur = pd.Timestamp(start.year, start.month, 1)
    end_m = pd.Timestamp(end.year, end.month, 1)
    while cur <= end_m:
        # weekday(): Monday=0, Tuesday=1
        first = cur
        # find first Tuesday on/after the 1st
        offset = (1 - first.weekday()) % 7
        first_tue = first + pd.Timedelta(days=offset)
        second_tue = first_tue + pd.Timedelta(days=7)
        dates.append(second_tue)
        # advance one month
        if cur.month == 12:
            cur = pd.Timestamp(cur.year + 1, 1, 1)
        else:
            cur = pd.Timestamp(cur.year, cur.month + 1, 1)
    return dates


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["CF", "ZC=F", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    needed = ["CF", "ZC=F", "SPY"]
    if not all(t in px.columns for t in needed):
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px[needed].sort_index()
    # forward-fill small gaps
    for c in needed:
        px[c] = px[c].ffill(limit=5)
    px = px.dropna(subset=["ZC=F", "SPY"])
    cf = px["CF"].dropna()
    if len(cf) < 500:
        return mark_failed(sid, f"insufficient CF history: {len(cf)} days")

    ret = daily_returns(px)
    cf_r = ret["CF"]
    spy_r = ret["SPY"]

    # restrict to dates where CF has a price (CF IPO ~Aug 2005)
    cf_start = cf.index[0]
    px = px.loc[cf_start:]
    ret = ret.loc[cf_start:]

    # Build candidate "WASDE event" dates (2nd Tuesday of each month)
    event_dates = second_tuesday_of_month(px.index[0], px.index[-1])

    # snap each event date forward to the next trading day on or after it
    trading_idx = px.index
    snapped = []
    for d in event_dates:
        loc = trading_idx.searchsorted(d)
        if loc >= len(trading_idx):
            continue
        snapped.append(trading_idx[loc])

    zc_close = px["ZC=F"]
    zc_change_90 = zc_close / zc_close.shift(90) - 1.0

    hold_days = 50
    stop_loss = -0.10

    in_pos = False
    entry_loc = -1
    entry_price = np.nan
    positions = pd.Series(0.0, index=ret.index)
    events = []

    for d in snapped:
        if d not in ret.index:
            continue
        i = ret.index.get_loc(d)
        if in_pos:
            # only consider firing when flat
            continue
        zc_today = zc_close.iloc[i]
        zc_90 = zc_change_90.iloc[i]
        if pd.isna(zc_today) or pd.isna(zc_90):
            continue
        if zc_today < 5.00:
            continue
        if zc_90 < 0.15:
            continue
        # entry next trading day's close-to-close return -> mark position starting tomorrow
        if i + 1 >= len(ret.index):
            continue
        in_pos = True
        entry_loc = i + 1
        entry_price = cf.reindex(ret.index).iloc[entry_loc] \
            if entry_loc < len(ret.index) else np.nan
        # actually we hold from day i+1 through day i+hold_days or stop
        end_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        ev_start = ret.index[entry_loc]
        ev_end = ret.index[end_loc]

        # walk forward to evaluate stop-loss
        cf_window = cf.reindex(ret.index).iloc[entry_loc:end_loc + 1]
        if len(cf_window) == 0 or pd.isna(entry_price):
            in_pos = False
            continue
        # cumulative return from entry close
        cumret = cf_window / entry_price - 1.0
        exit_loc = end_loc
        for k, val in enumerate(cumret.values):
            if pd.notna(val) and val <= stop_loss:
                exit_loc = entry_loc + k
                break

        positions.iloc[entry_loc:exit_loc + 1] = 1.0

        # event diagnostics
        cf_ev_ret = float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)])
        spy_window = spy_r.iloc[entry_loc:exit_loc + 1]
        spy_cum = float((1 + spy_window).prod() - 1) if len(spy_window) else np.nan
        events.append({
            "trigger_date": str(d.date()),
            "entry_date": str(ev_start.date()),
            "exit_date": str(ret.index[exit_loc].date()),
            "cf_return": round(cf_ev_ret, 4),
            "spy_return": round(spy_cum, 4) if not np.isnan(spy_cum) else None,
            "excess": round(cf_ev_ret - spy_cum, 4) if not np.isnan(spy_cum) else None,
            "zc_close": float(zc_today),
            "zc_90d_chg": float(zc_90),
        })

        # release position
        in_pos = False
        entry_loc = -1

    if not events:
        return mark_failed(
            sid,
            "No WASDE-proxy events fired (ZC>=5 AND +15% 90d on 2nd Tuesday)",
            extra={
                "rule": "Long CF 50d when WASDE proxy fires (2nd Tue, ZC>=5, ZC +15% 90d)",
                "mechanism": "Tight corn stocks->use ratios coincide with elevated corn prices; CF Industries benefits from elevated nitrogen demand",
                "source": "PL602 catalog; WASDE convention proxy",
            },
        )

    # daily PnL = previous-day position * CF return (no look-ahead)
    pnl = positions.shift(1).fillna(0.0) * cf_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=positions,
        cost_bps=10,
    )

    m["n_events"] = len(events)
    m["pct_in_market"] = float(positions.mean())
    m["events"] = events[:30]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "On 2nd Tuesday of each month (WASDE convention), if ZC=F close >= $5/bu AND ZC=F up >+15% over trailing 90 trading days, go long CF at next day close for 50 trading days, with -10% stop.",
            "mechanism": "USDA WASDE corn stocks/use <10% is a tight-supply regime. Tight corn -> high prices -> US farmers maximize acreage and apply more nitrogen fertilizer. CF Industries is the largest US nitrogen producer and earnings lever directly on nitrogen pricing. Pure-price proxy (corn >$5 AND rising 15% over 90 days) approximates the WASDE tight-stocks regime because both historically coincide.",
            "source": "PL602 — WASDE Corn Stocks/Use idea; proxy implementation per project conventions (no WASDE PDF auto-ingest)",
            "caveats": "WASDE stocks/use not directly available; proxy uses corn front-month price regime. Stop-loss is one-sided (no upside take-profit). CF only IPO'd Aug 2005, so coverage starts then.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
