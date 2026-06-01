"""PL779_nfp_negative_sahm_kbe_long_kre_short
NFP Negative + Sahm Rule Cross -> Long KBE / Short KRE Pair

When BOTH conditions met in same calendar month:
  (1) FRED PAYEMS MoM change < 0 (job losses)
  (2) FRED SAHMREALTIME >= 0.50 (Sahm rule triggered)

Enter: Long KBE (money-center banks) / Short KRE (regional banks), dollar-neutral.
Exit: Sahm drops below 0.30, OR 12-week time stop, OR 15% trailing stop on pair P&L.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL779_nfp_negative_sahm_kbe_long_kre_short"
    tickers = ["KBE", "KRE", "SPY"]

    try:
        px = load_prices(tickers, start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=5)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    try:
        fred_df = load_fred(["SAHMREALTIME", "PAYEMS"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load failed: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    kbe_r = ret["KBE"].dropna()
    kre_r = ret["KRE"].dropna()

    sahm = fred_df["SAHMREALTIME"].dropna()
    payems = fred_df["PAYEMS"].dropna()
    payems_mom = payems.diff()

    # Monthly signal: both conditions met within 30 days of each other
    # FRED monthly data is end-of-month dated, typically released ~first Friday following
    # Signal: PAYEMS MoM < 0 AND SAHMREALTIME >= 0.50, both in the same month
    signal_months = []

    for date in sahm.index:
        if sahm.get(date, 0) >= 0.50:
            # Check if PAYEMS was also negative in the same month or adjacent month
            month_start = date.replace(day=1)
            month_end = (date + pd.offsets.MonthEnd(1))

            # PAYEMS is monthly; look at same month
            payems_in_window = payems_mom[(payems_mom.index >= month_start - pd.Timedelta(days=31)) &
                                          (payems_mom.index <= month_end + pd.Timedelta(days=31))]
            if len(payems_in_window) > 0 and payems_in_window.min() < 0:
                signal_months.append(date)

    if not signal_months:
        return mark_failed(sid, "no months with NFP < 0 AND Sahm >= 0.50")

    # Convert signal months to trading dates
    # FRED monthly date -> find first trading day at or after that date + typical release lag (~5 days)
    hold_weeks = 12
    hold_days = hold_weeks * 5  # ~60 trading days
    trailing_stop = 0.15
    sahm_exit_threshold = 0.30

    positions_kbe = pd.Series(0.0, index=ret.index)
    positions_kre = pd.Series(0.0, index=ret.index)
    events = []

    processed_signals = []  # track to avoid overlapping entries
    last_exit_date = None

    for sig_dt in sorted(set(signal_months)):
        # Release lag: ~5 calendar days after month-end date
        release_dt = sig_dt + pd.Timedelta(days=5)
        future = ret.index[ret.index >= release_dt]
        if len(future) == 0:
            continue
        entry_date = future[0]

        # Don't re-enter if already in a position within 30 days
        if last_exit_date is not None and (entry_date - last_exit_date).days < 30:
            continue

        entry_loc = ret.index.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        # Walk holding period
        cum_pnl = 0.0
        peak_pnl = 0.0
        actual_exit = entry_loc
        exit_reason = "scheduled_12w"

        for j in range(entry_loc, exit_loc + 1):
            d = ret.index[j]
            kbe_day = kbe_r.get(d, 0.0) or 0.0
            kre_day = kre_r.get(d, 0.0) or 0.0
            # Long KBE, Short KRE
            day_pnl = kbe_day - kre_day
            cum_pnl += day_pnl
            peak_pnl = max(peak_pnl, cum_pnl)
            positions_kbe.iloc[j] = 1.0
            positions_kre.iloc[j] = -1.0

            # Check trailing stop: 15% drawdown from peak pair P&L
            if cum_pnl < -trailing_stop:  # absolute -15% from flat
                actual_exit = j
                exit_reason = "trailing_stop"
                break
            if peak_pnl > 0 and (cum_pnl - peak_pnl) < -trailing_stop:
                actual_exit = j
                exit_reason = "trailing_stop"
                break

            # Check Sahm exit: next available Sahm reading < 0.30
            sahm_at_d = sahm.asof(d)
            if not np.isnan(sahm_at_d) and sahm_at_d < sahm_exit_threshold and j > entry_loc + 5:
                actual_exit = j
                exit_reason = "sahm_recovery"
                break
        else:
            actual_exit = exit_loc

        exit_date = ret.index[actual_exit]
        last_exit_date = exit_date

        # Compute event stats
        kbe_event = kbe_r.loc[entry_date:exit_date]
        kre_event = kre_r.loc[entry_date:exit_date]
        kbe_cum = float((1 + kbe_event).prod() - 1) if len(kbe_event) else 0.0
        kre_cum = float((1 + kre_event).prod() - 1) if len(kre_event) else 0.0

        events.append({
            "signal_date": str(sig_dt.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "kbe_return": round(kbe_cum, 4),
            "kre_return": round(kre_cum, 4),
            "pair_pnl": round(cum_pnl, 4),
            "sahm_at_entry": float(sahm.asof(entry_date)),
            "payems_mom_at_entry": float(payems_mom.asof(entry_date) if len(payems_mom.asof(entry_date).shape if hasattr(payems_mom.asof(entry_date), 'shape') else []) == 0 else payems_mom.asof(entry_date)),
        })

    if not events:
        return mark_failed(sid, "no events within price data range")

    # Daily PnL (no look-ahead: shifted by 1)
    pos_kbe_shifted = positions_kbe.shift(1)
    pos_kre_shifted = positions_kre.shift(1)
    pnl = (pos_kbe_shifted.fillna(0) * kbe_r.reindex(ret.index).fillna(0) +
           pos_kre_shifted.fillna(0) * kre_r.reindex(ret.index).fillna(0))
    pnl = pnl.dropna()

    if (pnl != 0).sum() < 30:
        return mark_failed(sid, f"insufficient active days: {(pnl!=0).sum()}, events: {len(events)}")

    combined_pos = (positions_kbe.abs() + positions_kre.abs()) / 2
    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="NFP Negative + Sahm Cross: Long KBE / Short KRE",
        positions=combined_pos.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )

    n_events = len(events)
    ev_rets = [e["pair_pnl"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_rets])) if ev_rets else None
    avg_event = float(np.mean(ev_rets)) if ev_rets else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When FRED PAYEMS MoM < 0 (job losses) AND FRED SAHMREALTIME >= 0.50 "
                "in same calendar month: Long KBE / Short KRE dollar-neutral pair "
                "for 12 weeks. Exit if Sahm drops below 0.30, or 15% trailing stop."
            ),
            "mechanism": (
                "In recessions, money-center banks (KBE) benefit from flight-to-quality "
                "deposit flows and Fed support, while regional banks (KRE) face worse credit "
                "quality (CRE, SME) and deposit outflows. NFP < 0 + Sahm confirms recession onset."
            ),
            "source": "FRED SAHMREALTIME, FRED PAYEMS; yfinance KBE/KRE/SPY",
            "tickers": ["KBE", "KRE"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Very few events (~2-3). 2020 COVID analog is V-shaped recovery, atypical. "
                "SAHMREALTIME available only from 2007. KBE/KRE both launched 2005-2006. "
                "Trailing stop may trigger quickly in volatile periods (e.g. GFC)."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
        f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
        f"t-stat: {m.get('t_stat', float('nan')):.2f}"
    )
    for e in events:
        print(f"  Event {e['signal_date']}: kbe={e['kbe_return']:.2%} kre={e['kre_return']:.2%} "
              f"pair={e['pair_pnl']:.4f} [{e['exit_reason']}]")


if __name__ == "__main__":
    main()
