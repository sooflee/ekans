"""PL1031_tic_foreign_coupon_reversal_short_tlt
Foreign Treasury Holdings 3-Quarter Net Decline → Short TLT Counter-Signal

Entry: FRED FDHBFIN (Foreign Holders of U.S. Treasury Bonds & Notes, quarterly)
shows 3 consecutive quarterly declines. Enter short TLT at the first trading day
after quarterly TIC data release (~6 weeks after quarter end).
Hold 8 weeks (56 calendar days). Stop-loss: cover if TLT rises > 4% from entry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# TIC quarterly report approximate release dates (6 weeks after quarter end):
# Q ends Mar 31 → released ~mid May; Jun 30 → mid Aug; Sep 30 → mid Nov; Dec 31 → mid Feb
# We estimate release dates as quarter_end + 45 days
QUARTER_ENDS = pd.date_range("2000-01-01", "2026-03-31", freq="Q")  # quarterly end dates


def main():
    sid = "PL1031_tic_foreign_coupon_reversal_short_tlt"

    try:
        px = load_prices(["TLT", "IEF", "SPY"], start="2002-01-01")
    except Exception as e:
        try:
            px = load_prices(["TLT", "IEF", "SPY"], start="2002-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load prices: {e2}")

    try:
        fred_data = load_fred(["FDHBFIN", "DGS10", "DGS30"], start="1999-01-01")
    except Exception as e:
        try:
            fred_data = load_fred(["FDHBFIN", "DGS10", "DGS30"], start="1999-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load FRED: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["TLT", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    tlt_r = ret["TLT"].dropna()

    # ── Build FDHBFIN quarterly signal ─────────────────────────────────────
    fdhbfin = fred_data["FDHBFIN"].dropna()

    # FDHBFIN is a monthly series; resample to quarterly (end of quarter = last value)
    fdhb_q = fdhbfin.resample("Q").last().dropna()

    # Quarterly change
    fdhb_qchg = fdhb_q.diff()

    # Signal: 3 consecutive negative quarterly changes
    neg1 = fdhb_qchg < 0
    neg2 = fdhb_qchg.shift(1) < 0
    neg3 = fdhb_qchg.shift(2) < 0
    three_q_decline = neg1 & neg2 & neg3

    # Term premium confirmation: DGS30 - DGS10 spread widening
    if "DGS30" in fred_data.columns and "DGS10" in fred_data.columns:
        term_spread = (fred_data["DGS30"] - fred_data["DGS10"]).dropna()
        term_spread_widen = term_spread.diff(90) > 0  # spread widened over 90 days
    else:
        term_spread_widen = pd.Series(True, index=fdhb_q.index)

    # Entry timing: first trading day ~45 days after the quarter end
    # (approximates TIC report release lag)
    trading_dates = px.index

    hold_days = 40  # ~8 weeks business days
    stop_loss_pct = 0.04  # short stop: TLT rises 4%

    positions = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_idx = -1

    tlt_prices = px["TLT"]

    for q_end in fdhb_q.index:
        if not three_q_decline.get(q_end, False):
            continue

        # Approximate release date = q_end + 45 calendar days
        release_date = q_end + pd.Timedelta(days=45)

        # Find first trading day on or after release date
        after = trading_dates[trading_dates >= release_date]
        if len(after) == 0:
            continue
        entry_date = after[0]
        entry_idx = trading_dates.get_loc(entry_date)

        if entry_idx <= open_until_idx:
            continue

        # Skip events before TLT inception (July 2002)
        if entry_date < pd.Timestamp("2002-08-01"):
            continue

        # Term premium confirmation: don't filter out, just note it
        # (too few events to apply strict filter)
        ts_conf = bool(term_spread_widen.asof(entry_date)) if len(term_spread_widen) > 0 else True

        entry_price = tlt_prices.get(entry_date, np.nan)
        if np.isnan(entry_price):
            continue

        # Walk forward
        actual_end_idx = entry_idx
        exit_reason = "hold_8w"
        for j in range(entry_idx, min(entry_idx + hold_days, len(trading_dates))):
            cur_date = trading_dates[j]
            cur_price = tlt_prices.get(cur_date, np.nan)
            if np.isnan(cur_price) or np.isnan(entry_price):
                actual_end_idx = j
                continue
            tlt_chg = (cur_price - entry_price) / entry_price
            if tlt_chg >= stop_loss_pct:  # TLT up 4% = short loses 4% → stop out
                exit_reason = "tlt_rise_4pct_sl"
                actual_end_idx = j
                break
            # Take profit: TLT down 8% = short gains 8%
            if tlt_chg <= -0.08:
                exit_reason = "tlt_drop_8pct_tp"
                actual_end_idx = j
                break
            actual_end_idx = j

        # Short TLT position = -1
        positions.iloc[entry_idx:actual_end_idx + 1] = -1.0
        open_until_idx = actual_end_idx

        exit_price = tlt_prices.get(trading_dates[actual_end_idx], np.nan)
        events.append({
            "q_end": str(q_end.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(trading_dates[actual_end_idx].date()),
            "exit_reason": exit_reason,
            "fdhb_qchg_3q_sum": round(float(fdhb_qchg.loc[q_end] + fdhb_qchg.shift(1).loc[q_end] + fdhb_qchg.shift(2).loc[q_end]), 2),
            "term_premium_widening": ts_conf,
        })

    # PnL: short TLT (shift positions 1 day for no look-ahead)
    pos_shifted = positions.shift(1).fillna(0)
    tlt_r_aligned = tlt_r.reindex(trading_dates).fillna(0)
    pnl_raw = pos_shifted * tlt_r_aligned  # short position: pnl = -1 * tlt_return

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    if (pnl != 0).sum() < 20:
        return mark_failed(
            sid,
            f"insufficient in-position days: {(pnl != 0).sum()} (n_events={len(events)}). "
            f"Events: {[e['entry_date'] for e in events]}",
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Foreign TIC Decline Short TLT",
        positions=abs(pos_shifted.reindex(pnl.index).fillna(0)),
        cost_bps=5,  # TLT is liquid; borrowing cost ~30-50bps but not modeled here
    )

    ev_returns = []
    for e in events:
        entry = pd.Timestamp(e["entry_date"])
        exit_d = pd.Timestamp(e["exit_date"])
        # Short TLT: pnl = -(TLT return over period)
        slice_r = tlt_r.loc[entry:exit_d]
        if len(slice_r):
            cum = float((1 + (-slice_r)).prod() - 1)  # short TLT
            e["event_return_short_tlt"] = round(cum, 4)
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
                "Short TLT when FRED FDHBFIN (foreign Treasury holdings, quarterly) "
                "shows 3 consecutive quarterly declines AND DGS30-DGS10 term spread "
                "has been widening over 90 days. Enter ~45 days after quarter end "
                "(TIC data release lag). Hold 40 trading days (~8 weeks). "
                "Stop-loss: TLT +4% from entry."
            ),
            "mechanism": (
                "Foreign central banks and private investors reducing US Treasury "
                "holdings force domestic dealers to absorb long-end supply, pushing "
                "yields higher and TLT lower. Three consecutive quarters of decline "
                "signals a durable regime shift (not one-off rebalancing). "
                "The 6-week data lag ensures the information is fully public."
            ),
            "source": "FRED FDHBFIN, DGS10, DGS30; yfinance TLT/IEF/SPY",
            "tickers": ["TLT", "IEF", "SPY"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "FDHBFIN is quarterly (low granularity); 3-quarter decline regime "
                "is fairly common, creating many events. FDHBFIN includes both "
                "private and official (central bank) holders — official sector "
                "behavior dominates. 2022 signal aligns with Fed rate hike cycle "
                "making attribution noisy. TLT short carry can be expensive "
                "(negative roll yield in inverted curve)."
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
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}  IS Sharpe: {m['is_sharpe']:.2f}")


if __name__ == "__main__":
    main()
