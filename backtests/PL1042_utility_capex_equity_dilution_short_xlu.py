"""PL1042_utility_capex_equity_dilution_short_xlu
Utility Sector Capex Excess + Equity Dilution Cycle → Short XLU Counter-Signal

Primary: 3 known utility equity-dilution cycle dates (hardcoded).
Secondary: FRED TLNRESCONS YoY > +15% for 3+ months → short XLU 126 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known utility equity-dilution cycle entry dates
KNOWN_EVENTS = [
    "2002-03-01",   # post-Enron restructuring equity raises
    "2009-10-01",   # regulated utility equity raises for ARRA stimulus grid
    "2022-12-01",   # AI/grid capex boom, ATM programs at NEE/AEP/DUK
]


def main():
    sid = "PL1042_utility_capex_equity_dilution_short_xlu"

    try:
        px = load_prices(["XLU", "SPY"], start="2000-01-01")
    except Exception as e:
        try:
            px = load_prices(["XLU", "SPY"], start="2000-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load prices: {e2}")

    try:
        tlnres = load_fred("TLNRESCONS", start="2000-01-01")
    except Exception as e:
        try:
            tlnres = load_fred("TLNRESCONS", start="2000-01-01", cache=False)
        except Exception as e2:
            tlnres = None
            print(f"  Warning: TLNRESCONS load failed ({e2}); using hardcoded events only")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["XLU", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    xlu_r = ret["XLU"].dropna()
    xlu_prices = px["XLU"]
    spy_prices = px["SPY"]
    trading_dates = px.index

    hold_days = 126        # ~6 calendar months
    stop_loss_pct = 0.10   # short stop: XLU up 10%
    take_profit_pct = 0.20 # take profit: XLU down 20%

    # Build event list: start with hardcoded known events
    event_dates_raw = [pd.Timestamp(d) for d in KNOWN_EVENTS]

    # Add rule-based TLNRESCONS events if data available
    if tlnres is not None:
        if isinstance(tlnres, pd.DataFrame):
            tlnres_s = tlnres.iloc[:, 0].dropna()
        else:
            tlnres_s = tlnres.dropna()

        # YoY % change
        tlnres_yoy = tlnres_s.pct_change(12) * 100
        # 3+ consecutive months above 15%
        above_15 = tlnres_yoy > 15
        # Rolling 3-month count
        consec_3 = above_15.rolling(3, min_periods=3).sum()
        triggered = consec_3 >= 3

        # Find first trigger dates per episode (6-month debounce)
        rule_events = []
        last_trigger = pd.Timestamp("1900-01-01")
        for date, val in triggered.items():
            if val and (date - last_trigger).days > 180:
                rule_events.append(date)
                last_trigger = date

        print(f"  TLNRESCONS rule-based events: {[str(d.date()) for d in rule_events]}")
        # Add if not already covered by known events (within 90 days)
        for re in rule_events:
            already_covered = any(abs((re - ke).days) < 90 for ke in event_dates_raw)
            if not already_covered:
                event_dates_raw.append(re)

    # Deduplicate and sort
    event_dates_raw = sorted(set(event_dates_raw))
    print(f"  Total event candidates: {len(event_dates_raw)}")

    positions = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_date = pd.Timestamp("1900-01-01")

    for event_date in event_dates_raw:
        # Find next trading day
        after = trading_dates[trading_dates >= event_date]
        if len(after) == 0:
            print(f"  Skipping {event_date.date()}: no trading days after event")
            continue
        entry_date = after[0]

        if entry_date <= open_until_date:
            print(f"  Skipping {entry_date.date()}: overlaps with prior trade")
            continue

        entry_idx = trading_dates.get_loc(entry_date)
        xlu_entry = xlu_prices.get(entry_date, np.nan)
        if np.isnan(xlu_entry):
            print(f"  Skipping {entry_date.date()}: no XLU price")
            continue

        # Walk forward hold_days trading days
        end_idx = min(entry_idx + hold_days, len(trading_dates) - 1)
        trade_window = trading_dates[entry_idx:end_idx + 1]

        actual_exit_idx = len(trade_window) - 1
        exit_reason = "126_day_hold"

        for j, td in enumerate(trade_window):
            cur_xlu = xlu_prices.get(td, np.nan)
            if not np.isnan(cur_xlu) and not np.isnan(xlu_entry):
                chg = (cur_xlu - xlu_entry) / xlu_entry
                if chg >= stop_loss_pct:     # Short stop: XLU up 10%
                    actual_exit_idx = j
                    exit_reason = "xlu_10pct_stop"
                    break
                if chg <= -take_profit_pct:  # Take profit: XLU down 20%
                    actual_exit_idx = j
                    exit_reason = "xlu_20pct_tp"
                    break

        actual_exit_date = trade_window[actual_exit_idx]
        open_until_date = actual_exit_date

        trade_slice = trade_window[:actual_exit_idx + 1]
        positions.loc[trade_slice] = -1.0   # short XLU

        xlu_exit = xlu_prices.get(actual_exit_date, np.nan)
        # Short XLU return = -(XLU price change)
        raw_ret = -(xlu_exit - xlu_entry) / xlu_entry if not np.isnan(xlu_exit) else np.nan

        # SPY return over same window for excess return calculation
        spy_entry = spy_prices.get(entry_date, np.nan)
        spy_exit = spy_prices.get(actual_exit_date, np.nan)
        spy_ret = (spy_exit - spy_entry) / spy_entry if not (np.isnan(spy_exit) or np.isnan(spy_entry)) else np.nan

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(actual_exit_date.date()),
            "exit_reason": exit_reason,
            "short_xlu_return": round(float(raw_ret), 4) if not np.isnan(raw_ret) else None,
            "spy_return": round(float(spy_ret), 4) if not np.isnan(spy_ret) else None,
            "excess_return": round(float(raw_ret - spy_ret), 4) if not (np.isnan(raw_ret) or np.isnan(spy_ret)) else None,
        })
        print(f"  Trade {entry_date.date()} → {actual_exit_date.date()} ({exit_reason}), "
              f"short XLU={raw_ret:.2%}, SPY={spy_ret:.2%}")

    print(f"  Total trades: {len(events)}")
    if len(events) < 2:
        return mark_failed(sid, f"insufficient trades: {len(events)}")

    # PnL: short XLU, shift 1 day
    pos_shifted = positions.shift(1).fillna(0)
    xlu_r_aligned = xlu_r.reindex(trading_dates).fillna(0)
    pnl_raw = pos_shifted * xlu_r_aligned   # short: PnL = -1 * XLU return

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    in_pos_days = (pnl != 0).sum()
    print(f"  In-position days: {in_pos_days}")
    if in_pos_days < 20:
        return mark_failed(sid, f"insufficient in-position days: {in_pos_days}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Utility Capex Dilution Short XLU",
        positions=abs(pos_shifted.reindex(pnl.index).fillna(0)),
        cost_bps=10,
    )

    ev_returns = [e["short_xlu_return"] for e in events if e.get("short_xlu_return") is not None]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Short XLU when utility sector enters equity-dilution cycle, "
                "identified either by (a) known episode dates (2002, 2009, 2022) "
                "or (b) FRED TLNRESCONS YoY > +15% for 3+ consecutive months "
                "(capex surge proxy). Hold 126 trading days (~6 months). "
                "Stop-loss: XLU +10%; take-profit: XLU -20%."
            ),
            "mechanism": (
                "Utility equity ATM programs (at-the-money issuance) during capex cycles "
                "are structurally dilutive. Sustained high capex-to-CFO ratios force "
                "equity raises, compressing EPS and ROE. The combination of rising "
                "capex needs (grid upgrades, generation investment) and rate regulation "
                "lag means utilities under-earn their cost of capital during peak "
                "investment cycles, creating persistent share price headwinds."
            ),
            "source": "FRED TLNRESCONS; hardcoded known dilution cycle dates; yfinance XLU/SPY",
            "tickers": ["XLU", "SPY"],
            "n_events": len(events),
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "known_events_used": KNOWN_EVENTS,
            "caveats": (
                "Very small sample (3 known events + possible TLNRESCONS-derived events). "
                "TLNRESCONS proxy is noisy (includes non-utility construction). "
                "2022 event overlaps with Fed rate hike cycle, making attribution unclear. "
                "2009 event occurred during broad recovery; XLU may have underperformed "
                "for macro (not dilution) reasons. Short XLU has negative carry in "
                "high-dividend-yield regimes."
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
