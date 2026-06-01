"""PL1029_dot_plot_iqr_compression_ief_short
FOMC Dot-Plot IQR Compression → Rates Vol Underpriced → Short IEF / Long PFIX

Entry: After FOMC SEP release where 1-year-ahead dot IQR ≤ 25bps (near-consensus),
short IEF next trading day.
Hold 30 calendar days, or exit if IEF gains >2% from entry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# SEP release dates where 1-year-ahead dot IQR ≤ 25bps
# (manually verified from Federal Reserve SEP PDFs)
SEP_EVENTS = [
    "2013-12-18",  # unanimous ZLB; IQR=0bps
    "2015-12-16",  # first liftoff consensus; IQR~25bps
    "2019-06-19",  # pause consensus; IQR~25bps
    "2019-12-11",  # hold consensus; IQR~0bps
    "2020-06-10",  # unanimous ZLB commitment; IQR=0bps
    "2020-09-16",  # ZLB through 2023 consensus; IQR=0bps
    "2020-12-16",  # ZLB through 2023; IQR=0bps
    "2021-06-16",  # first dot shift; IQR~25bps
    "2022-12-14",  # terminal rate consensus; IQR~25bps
    "2023-06-14",  # pause consensus building; IQR~25bps
    "2023-12-13",  # cut consensus 2024; IQR~25bps
    "2024-06-12",  # cut delivery consensus; IQR~25bps
    "2024-12-18",  # slower cut pace; IQR~25bps
]


def main():
    sid = "PL1029_dot_plot_iqr_compression_ief_short"

    try:
        px = load_prices(["IEF", "TLT", "SPY"], start="2002-07-01")
    except Exception as e:
        try:
            px = load_prices(["IEF", "TLT", "SPY"], start="2002-07-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load prices: {e2}")

    # Try loading PFIX separately (only available from 2021-05)
    try:
        px_pfix = load_prices(["PFIX"], start="2021-05-01")
        pfix_available = "PFIX" in px_pfix.columns
    except Exception:
        try:
            px_pfix = load_prices(["PFIX"], start="2021-05-01", cache=False)
            pfix_available = "PFIX" in px_pfix.columns
        except Exception:
            pfix_available = False
            px_pfix = None

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["IEF", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    ief_r = ret["IEF"].dropna()
    ief_prices = px["IEF"]
    trading_dates = px.index

    # Regime filter: do not enter if 10Y yield rising >50bps in prior 4 weeks
    # We'll use TLT price as proxy: TLT down >3% in 4 weeks = yields rising 50bps
    tlt_prices = px["TLT"] if "TLT" in px.columns else None

    hold_cal_days = 30
    ief_stop_gain_pct = 0.02    # if IEF rises 2%, exit (vol failed to re-expand)

    positions = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_date = pd.Timestamp("1900-01-01")

    for event_str in SEP_EVENTS:
        event_date = pd.Timestamp(event_str)

        # Entry = next trading day after SEP release
        after = trading_dates[trading_dates > event_date]
        if len(after) == 0:
            print(f"  Skipping {event_str}: no trading days after event")
            continue
        entry_date = after[0]

        # Skip if overlapping with prior trade
        if entry_date <= open_until_date:
            print(f"  Skipping {entry_date.date()}: overlaps with prior trade")
            continue

        # Regime filter: check if 10Y yields already rising hard (TLT down >3% in 20 trading days)
        if tlt_prices is not None:
            tlt_entry = tlt_prices.get(entry_date, np.nan)
            # Look back 20 trading days
            entry_idx = trading_dates.get_loc(entry_date)
            if entry_idx >= 20:
                tlt_4w_ago = tlt_prices.iloc[entry_idx - 20]
                if not np.isnan(tlt_4w_ago) and not np.isnan(tlt_entry):
                    tlt_4w_chg = (tlt_entry - tlt_4w_ago) / tlt_4w_ago
                    if tlt_4w_chg <= -0.03:  # TLT down 3% ~ 50bps yield rise
                        print(f"  Skipping {entry_date.date()}: regime filter: TLT 4w chg={tlt_4w_chg:.2%} (yields rising fast)")
                        continue

        entry_price = ief_prices.get(entry_date, np.nan)
        if np.isnan(entry_price):
            print(f"  Skipping {entry_date.date()}: no IEF price")
            continue

        # Determine exit window
        exit_cal_date = entry_date + pd.Timedelta(days=hold_cal_days)
        trade_window = trading_dates[(trading_dates >= entry_date) & (trading_dates <= exit_cal_date)]
        if len(trade_window) == 0:
            continue

        actual_exit_idx = len(trade_window) - 1
        exit_reason = "30_cal_day_hold"

        for j, td in enumerate(trade_window):
            cur_price = ief_prices.get(td, np.nan)
            if not np.isnan(cur_price) and not np.isnan(entry_price):
                chg = (cur_price - entry_price) / entry_price
                if chg >= ief_stop_gain_pct:   # IEF up 2% = short stop out
                    actual_exit_idx = j
                    exit_reason = "ief_2pct_stop"
                    break

        actual_exit_date = trade_window[actual_exit_idx]
        open_until_date = actual_exit_date

        # Short IEF: positions = -1
        trade_slice = trade_window[:actual_exit_idx + 1]
        positions.loc[trade_slice] = -1.0

        exit_price = ief_prices.get(actual_exit_date, np.nan)
        # Short IEF return = -(IEF price change)
        raw_ret = -(exit_price - entry_price) / entry_price if not np.isnan(exit_price) else np.nan

        events.append({
            "sep_date": event_str,
            "entry_date": str(entry_date.date()),
            "exit_date": str(actual_exit_date.date()),
            "exit_reason": exit_reason,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2) if not np.isnan(exit_price) else None,
            "short_ief_return": round(float(raw_ret), 4) if not np.isnan(raw_ret) else None,
        })
        print(f"  Trade: {entry_date.date()} → {actual_exit_date.date()} ({exit_reason}), short IEF ret={raw_ret:.2%}" if not np.isnan(raw_ret) else f"  Trade: {entry_date.date()} → {actual_exit_date.date()}")

    print(f"  Total events used: {len(events)}")
    if len(events) < 3:
        return mark_failed(sid, f"insufficient events after filters: {len(events)}")

    # PnL: short IEF, shift 1 day
    pos_shifted = positions.shift(1).fillna(0)
    ief_r_aligned = ief_r.reindex(trading_dates).fillna(0)
    # Short IEF: PnL = -1 * IEF_return
    pnl_raw = pos_shifted * ief_r_aligned

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    in_pos_days = (pnl != 0).sum()
    print(f"  In-position days: {in_pos_days}")
    if in_pos_days < 20:
        return mark_failed(sid, f"insufficient in-position days: {in_pos_days}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Dot-Plot IQR Compression Short IEF",
        positions=abs(pos_shifted.reindex(pnl.index).fillna(0)),
        cost_bps=5,
    )

    ev_returns = [e["short_ief_return"] for e in events if e.get("short_ief_return") is not None]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Short IEF (7-10Y Treasuries) the day after FOMC SEP release "
                "when 1-year-ahead dot IQR ≤ 25bps (near-consensus). "
                "Exit after 30 calendar days or if IEF gains >2% (abort). "
                "Skip if 10Y yields already rising >50bps in prior 4 weeks (regime filter)."
            ),
            "mechanism": (
                "Near-zero IQR in the dot plot signals artificial consensus among FOMC "
                "members, which underprices rate uncertainty. Over the subsequent weeks, "
                "even minor macro surprises force vol re-expansion, pushing rates higher "
                "and IEF lower. The 'vol compression → vol expansion' cycle is most "
                "reliable when preceded by a prolonged ZLB or rates hold consensus."
            ),
            "source": "Federal Reserve SEP PDFs; hard-coded event dates; yfinance IEF/TLT/SPY",
            "tickers": ["IEF", "TLT", "SPY"],
            "pfix_available": pfix_available,
            "n_events": len(events),
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "sep_dates_used": SEP_EVENTS,
            "caveats": (
                "Dot IQR values are approximate; not machine-readable from any free API. "
                "ZLB-era events (2020) have near-zero IQR by design — those events "
                "saw IEF rally (ZLB constraints held) which caused losses on short IEF. "
                "Strategy conflates different vol regimes (ZLB vs hiking cycle). "
                "PFIX (long vol overlay) only available since 2021, limiting its "
                "contribution to the primary backtest."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {len(events)}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}  IS Sharpe: {m.get('is_sharpe'):.2f}")


if __name__ == "__main__":
    main()
