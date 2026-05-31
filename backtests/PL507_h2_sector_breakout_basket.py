"""PL507_h2_sector_breakout_basket
Hydrogen Sector Breakout Proxy for DOE Hydrogen Shot Cost-Validation

Use 4-stock hydrogen pure-play composite (PLUG, BE, BLDP, FCEL) 20-day return
crossing above +15% as a sector-regime trigger. On signal, long a 6-name
equal-weight basket (PLUG, BE, BLDP, FCEL, APD, LIN) for up to 60 trading days,
with early exit if the pure-play composite's 60-day return drops below -25%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL507_h2_sector_breakout_basket"
    pure_play = ["PLUG", "BE", "BLDP", "FCEL"]
    industrial = ["APD", "LIN"]
    basket = pure_play + industrial
    tickers = basket + ["SPY"]

    try:
        px = load_prices(tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Forward-fill mild missingness then drop anchor-NaN rows
    px = px.sort_index().ffill(limit=2)

    # ensure all tickers present
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Daily returns (auto-adjusted)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # 20-day total return per pure-play, then composite = mean across 4 names
    rolling_20d = px[pure_play].pct_change(20)
    composite_20 = rolling_20d.mean(axis=1)

    # 60-day total return composite (for invalidation rule)
    composite_60 = px[pure_play].pct_change(60).mean(axis=1)

    # Entry: previous-day composite_20 <= 0.15 AND today composite_20 > 0.15
    prev = composite_20.shift(1)
    cross_up = (prev <= 0.15) & (composite_20 > 0.15)
    entry_signal_dates = composite_20.index[cross_up.fillna(False)]

    # Build pnl series and positions
    hold_days = 60
    invalidation_thr = -0.25

    in_position = pd.Series(False, index=ret.index)
    daily_basket_pnl = pd.Series(0.0, index=ret.index)
    positions = pd.Series(0.0, index=ret.index)

    # Equal-weight basket of 6 names; if a name has missing return on a day,
    # treat as 0 for that day (no slippage assumed beyond that).
    basket_ret = ret[basket].fillna(0).mean(axis=1)

    events = []
    i = 0
    idx_list = list(ret.index)
    n = len(idx_list)

    # Walk through signals in chronological order; only one open at a time.
    pending = list(entry_signal_dates)
    open_until = None  # exclusive end position index
    in_pos_flag = False

    # We'll mark daily pnl and positions only on held days. Entry is at next
    # day's open ~ next session — we apply position to the NEXT trading day's
    # return (consistent with the harness's no-look-ahead convention).
    for j, dt in enumerate(idx_list):
        # If currently in a position, check invalidation at the close of dt.
        # Exit triggers at next-day open => position ends with today's return,
        # next-day return is no longer included.
        if in_pos_flag:
            # check exits
            comp60_today = composite_60.get(dt, np.nan)
            # End condition 1: reached planned end (60 hold days completed)
            # We track this by `open_until` (exclusive index position).
            today_return = basket_ret.iloc[j]
            daily_basket_pnl.iloc[j] = today_return
            positions.iloc[j] = 1.0

            scheduled_end = (j + 1 >= open_until)
            invalidate = (not np.isnan(comp60_today)) and (comp60_today < invalidation_thr)
            if scheduled_end or invalidate:
                # close position (next day, no return applied tomorrow)
                events[-1]["exit_date"] = str(dt.date())
                events[-1]["exit_reason"] = (
                    "invalidation" if invalidate and not scheduled_end else "scheduled_60d"
                )
                in_pos_flag = False
                open_until = None
        else:
            # Look for an entry signal whose signal-bar is <= dt-1 (so entry at dt's open)
            # We enter at next-day's open => signal generated on day k, position
            # earns return starting on day k+1.
            can_enter = j > 0 and pending and pending[0] <= idx_list[j - 1]
            if can_enter:
                sig_date = pending.pop(0)
                # discard any older queued signals
                while pending and pending[0] <= sig_date:
                    pending.pop(0)
                # Open position today (apply today's return as first hold day)
                today_return = basket_ret.iloc[j]
                daily_basket_pnl.iloc[j] = today_return
                positions.iloc[j] = 1.0
                in_pos_flag = True
                open_until = min(j + hold_days, n)  # exclusive
                events.append({
                    "signal_date": str(sig_date.date()),
                    "entry_date": str(dt.date()),
                    "composite_20d": float(composite_20.loc[sig_date]),
                })

    # Daily pnl series with cost-aware compute_metrics
    pnl = daily_basket_pnl
    # restrict to dates with returns (drop initial NaNs)
    pnl = pnl.dropna()

    if (pnl != 0).sum() < 30:
        return mark_failed(
            sid,
            f"insufficient in-position days: {(pnl != 0).sum()} (n_events={len(events)})",
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Hydrogen Sector Breakout Proxy",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level stats
    ev_returns = []
    for e in events:
        entry = pd.Timestamp(e["entry_date"])
        exit_d = pd.Timestamp(e.get("exit_date", entry))
        slice_r = basket_ret.loc[entry:exit_d]
        if len(slice_r):
            cum = float((1 + slice_r).prod() - 1)
            e["event_return"] = round(cum, 4)
            ev_returns.append(cum)

    n_events = len(events)
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When 4-stock H2 pure-play (PLUG/BE/BLDP/FCEL) 20d composite return "
                "crosses above +15%, long equal-weight 6-stock basket "
                "(PLUG/BE/BLDP/FCEL/APD/LIN) for 60 trading days; early exit if "
                "60d composite return falls below -25%."
            ),
            "mechanism": (
                "Sector-wide repricing in hydrogen pure-plays often follows DOE "
                "Hydrogen Shot cost-target milestones or policy validations. "
                "20d composite breakout above +15% captures regime shifts to "
                "constructive policy expectations; industrial-gas distributors "
                "(APD/LIN) participate via electrolyzer supply chains."
            ),
            "source": "yfinance auto-adjusted close; DOE Hydrogen Shot program context",
            "tickers": basket,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "PLUG/BLDP/FCEL are high-beta small-caps; auto_adjust handles "
                "reverse splits but vol regime is extreme. 60d hold + tight "
                "invalidation rule can stack overlapping draw-downs in bear "
                "regimes (2022-23). Sample size limited (sector born ~2010s)."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "net_sharpe" in m:
        print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
