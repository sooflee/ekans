"""PL808_dsca_saudi_major_arms_lmt_long
DSCA Major Arms Sale Notification (Saudi Arabia) -> Long LMT / Short RTX Pair

When the DSCA publishes a Major Defense Sale notification naming Saudi Arabia
with value >= $5B, enter LONG LMT and SHORT RTX (0.7x notional) at the close
on the notification day. Hold 30 calendar days.

Known DSCA Saudi major-arms notification dates:
  - 2010-09-13: $60B Saudi arms package Congressional notification
  - 2017-05-20: Trump/Riyadh $110B framework announcement
  - 2022-11-04: DSCA Saudi notification for THAAD/Patriot replenishment

Benchmark: SPY. Also reported vs ITA (sector ETF).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -----------------------------------------------------------------------
# DSCA Saudi Arabia Major Arms Sale notification events
# Sourced from dsca.mil public press releases
# -----------------------------------------------------------------------
DSCA_EVENTS = [
    {
        "notification_date": "2010-09-13",
        "value_bn": 60.0,
        "programs": "F-15, Apache, Black Hawk, Patriot",
        "description": "US Congressional notification: $60B Saudi arms package (largest in history at time)",
    },
    {
        "notification_date": "2017-05-20",
        "value_bn": 110.0,
        "programs": "THAAD, Patriot, F-15, Munitions",
        "description": "Trump Riyadh visit: $110B Saudi arms framework announcement",
    },
    {
        "notification_date": "2022-11-04",
        "value_bn": 5.0,
        "programs": "Patriot, THAAD replenishment munitions",
        "description": "DSCA Saudi Patriot/THAAD munitions notification",
    },
]

HOLD_CALENDAR_DAYS = 30   # exit unconditionally T+30 calendar days
RTX_NOTIONAL_RATIO = 0.7  # short RTX at 0.7x of LMT long notional
STOP_LOSS_PCT = 0.06       # LMT falls >6% from entry -> close long leg
STOP_GAIN_PCT = 0.12       # LMT gains >12% from entry -> lock gain


def find_entry_date(notification_date: pd.Timestamp, trading_idx: pd.Index):
    """Return the first trading day >= notification_date (same day or next open)."""
    candidates = trading_idx[trading_idx >= notification_date]
    if len(candidates) == 0:
        return None
    return candidates[0]


def run_event(ev, ret, lmt_prices, rtx_prices, stop_loss=0.06, stop_gain=0.12,
              rtx_ratio=0.7, hold_calendar=30):
    """Simulate a single event: long LMT + short RTX (rtx_ratio x notional).
    Walk forward up to hold_calendar days. Return event record.
    """
    ann_dt = pd.Timestamp(ev["notification_date"])
    idx = ret.index

    entry_dt = find_entry_date(ann_dt, idx)
    if entry_dt is None:
        return {**ev, "status": "no_trading_data"}

    entry_pos = idx.get_loc(entry_dt)
    exit_calendar_target = ann_dt + pd.Timedelta(days=hold_calendar)
    # Find first trading day >= exit_calendar_target
    exit_candidates = idx[idx >= exit_calendar_target]
    if len(exit_candidates) == 0:
        exit_pos_exclusive = len(idx)
    else:
        exit_pos_exclusive = idx.get_loc(exit_candidates[0]) + 1

    entry_lmt_price = float(lmt_prices.iloc[entry_pos])
    exit_reason = "scheduled_30d"
    actual_exit_pos = exit_pos_exclusive

    # Daily returns for the hold window
    lmt_r = ret["LMT"].iloc[entry_pos:exit_pos_exclusive]
    rtx_r = ret["RTX"].iloc[entry_pos:exit_pos_exclusive]

    # Walk forward checking stop-loss / stop-gain on LMT
    lmt_cum = 1.0
    for j, (lm, rx) in enumerate(zip(lmt_r, rtx_r)):
        lmt_cum *= (1 + float(lm))
        lmt_gain = lmt_cum - 1.0
        if lmt_gain < -stop_loss:
            exit_reason = "stop_loss"
            actual_exit_pos = entry_pos + j + 1
            break
        if lmt_gain > stop_gain:
            exit_reason = "stop_gain"
            actual_exit_pos = entry_pos + j + 1
            break

    # Compute pair return on actual hold window
    lmt_slice = ret["LMT"].iloc[entry_pos:actual_exit_pos]
    rtx_slice = ret["RTX"].iloc[entry_pos:actual_exit_pos]

    lmt_cum_ret = float((1 + lmt_slice.fillna(0)).prod() - 1)
    rtx_cum_ret = float((1 + rtx_slice.fillna(0)).prod() - 1)
    pair_return = lmt_cum_ret - rtx_ratio * rtx_cum_ret

    exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt

    ev_record = dict(ev)
    ev_record["entry_date"] = str(entry_dt.date())
    ev_record["exit_date"] = str(exit_dt.date())
    ev_record["exit_reason"] = exit_reason
    ev_record["n_hold_days"] = int(actual_exit_pos - entry_pos)
    ev_record["lmt_return"] = round(lmt_cum_ret, 4)
    ev_record["rtx_return"] = round(rtx_cum_ret, 4)
    ev_record["pair_return"] = round(pair_return, 4)  # LMT - 0.7*RTX

    return ev_record


def main():
    sid = "PL808_dsca_saudi_major_arms_lmt_long"
    tickers = ["LMT", "RTX", "ITA", "SPY"]

    # RTX was formed from Raytheon + United Technologies merger in April 2020.
    # Before 2020, Raytheon Co. traded as RTN. yfinance may provide RTX back-adjusted.
    # LMT data goes back to 1995+.
    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    # RTX may not have full history back to 2010 due to merger; handle gracefully
    available_tickers = [t for t in tickers if t in px.columns]
    missing = [t for t in ["LMT", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing core tickers: {missing}")

    # If RTX not available for 2010-2020, we'll skip RTX hedge for those events
    has_rtx = "RTX" in px.columns

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    lmt_prices = px["LMT"]
    rtx_prices = px["RTX"] if has_rtx else None

    # Pre-check RTX availability per event
    print(f"RTX available: {has_rtx}")
    if has_rtx:
        rtx_start = ret["RTX"].dropna().index.min()
        print(f"RTX first non-NaN: {rtx_start.date()}")

    # ---- Run individual events ----
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)  # net position (LMT long + RTX short hedge)

    event_log = []
    for ev in DSCA_EVENTS:
        ann_dt = pd.Timestamp(ev["notification_date"])

        # Check data availability
        future_lmt = idx[idx >= ann_dt]
        if len(future_lmt) == 0:
            ev_record = dict(ev)
            ev_record["status"] = "no_data_after_notification"
            event_log.append(ev_record)
            continue

        ev_record = run_event(
            ev, ret, lmt_prices, rtx_prices if has_rtx else pd.Series(dtype=float),
            stop_loss=STOP_LOSS_PCT, stop_gain=STOP_GAIN_PCT,
            rtx_ratio=RTX_NOTIONAL_RATIO, hold_calendar=HOLD_CALENDAR_DAYS,
        )
        event_log.append(ev_record)

        # Fill PnL for this event
        if ev_record.get("entry_date") and ev_record.get("exit_date"):
            entry_dt = pd.Timestamp(ev_record["entry_date"])
            exit_dt = pd.Timestamp(ev_record["exit_date"])
            entry_pos = idx.get_loc(entry_dt) if entry_dt in idx else None
            exit_pos = (idx.get_loc(exit_dt) + 1) if exit_dt in idx else None
            if entry_pos is not None and exit_pos is not None:
                lmt_slice = ret["LMT"].iloc[entry_pos:exit_pos].fillna(0)
                if has_rtx and "RTX" in ret.columns:
                    rtx_avail = ret["RTX"].iloc[entry_pos:exit_pos].fillna(0)
                    pair_daily = lmt_slice - RTX_NOTIONAL_RATIO * rtx_avail
                else:
                    pair_daily = lmt_slice  # LMT only if RTX unavailable

                for j in range(entry_pos, exit_pos):
                    if positions.iloc[j] == 0.0:
                        idx_day = j - entry_pos
                        pnl.iloc[j] = float(pair_daily.iloc[idx_day])
                        positions.iloc[j] = 1.0  # in-position marker

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    print(f"\nTraded events: {n_events}")
    for e in event_log:
        if e.get("entry_date"):
            print(f"  {e['notification_date']} -> entry {e['entry_date']}, "
                  f"pair_return={e.get('pair_return'):.4f}, reason={e.get('exit_reason')}")
        else:
            print(f"  {e['notification_date']} -> NO TRADE: {e.get('status')}")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        pair_returns = [e.get("pair_return") for e in event_log if e.get("pair_return") is not None]
        avg_ret = float(np.mean(pair_returns)) if pair_returns else None
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={
                "events": event_log,
                "avg_pair_return": avg_ret,
                "has_rtx": has_rtx,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="DSCA Saudi Arms -> Long LMT / Short RTX Pair (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    pair_returns = [e.get("pair_return") for e in event_log if e.get("pair_return") is not None]
    win_rate = float(np.mean([r > 0 for r in pair_returns])) if pair_returns else None
    avg_pair_ret = float(np.mean(pair_returns)) if pair_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On DSCA Major Defense Sale notification naming Saudi Arabia >= $5B, "
                "enter LONG LMT + SHORT RTX (0.7x notional) at close on notification day. "
                f"Hold {HOLD_CALENDAR_DAYS} calendar days. Stop-loss: LMT -6%; stop-gain: LMT +12%."
            ),
            "mechanism": (
                "Saudi Arabia's $60-110B arms packages are disproportionately "
                "concentrated in LMT programs (F-35, THAAD, PAC-3). DSCA notification "
                "is the first public signal of Congressional approval; it precedes "
                "the formal Letter of Offer and Acceptance (LOA) and eventual contract "
                "bookings by 6-18 months. RTX short hedges defense-sector beta and "
                "Patriot program saturation. The pair isolates F-35/THAAD-specific "
                "order-intake premium over the defense sector."
            ),
            "source": (
                "DSCA Major Arms Sales archive (dsca.mil, public); "
                "prices via yfinance (auto_adjust=True)"
            ),
            "tickers": ["LMT", "RTX"],
            "n_events": n_events,
            "rtx_available": has_rtx,
            "rtx_notional_ratio": RTX_NOTIONAL_RATIO,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_pair_event_return": round(avg_pair_ret, 4) if avg_pair_ret is not None else None,
            "events": event_log,
            "caveats": (
                "Extremely small sample (N=3 events in ~14 years). "
                "RTX data via yfinance covers post-2020 merger; pre-merger "
                "Raytheon RTN data may not be available. "
                "The 2010 event predates the modern DSCA digital archive; "
                "notification date is approximate (Congressional notification "
                "vs formal DSCA press release may differ by days). "
                "2017 event is an announcement framework, not a signed contract. "
                "Sharpe/CAGR metrics on held-days only are not robust at N=3."
            ),
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_pair_return: {avg_pair_ret}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
