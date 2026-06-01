"""PL770_saudi_breakeven_gap_ksa_short
Brent 60d Avg Below Saudi IMF Fiscal Breakeven -> Short KSA ETF

When the trailing 60-calendar-day average of daily Brent crude close falls
more than $10/bbl below the IMF's current-year Saudi Arabia fiscal breakeven
Brent price, enter SHORT KSA (iShares MSCI Saudi Arabia ETF) the following
trading day. Hold 30 trading days.

IMF Saudi fiscal breakeven table sourced from IMF MENAP Regional Economic
Outlook annual editions (April and October issues). Values are hard-coded per
the strategy spec and known historical estimates.

Benchmark: SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -----------------------------------------------------------------------
# IMF Saudi Arabia fiscal breakeven Brent price by year (USD/bbl)
# Source: IMF World Economic Outlook / MENAP Regional Economic Outlook
# April-edition estimates (used for that calendar year)
# Approximate values based on public IMF publications
# -----------------------------------------------------------------------
IMF_SAUDI_BREAKEVEN = {
    2015: 95.0,
    2016: 79.0,
    2017: 83.0,
    2018: 88.0,
    2019: 85.0,
    2020: 76.0,
    2021: 68.0,
    2022: 72.0,
    2023: 80.0,
    2024: 96.0,
    2025: 91.0,
    2026: 90.0,
}

# Gap threshold: short when 60d avg Brent is this many $/bbl BELOW breakeven
GAP_THRESHOLD = 10.0   # per implementation_notes "> $X" gap
HOLD_DAYS = 30
MIN_DAYS_BETWEEN = 90  # calendar days
EARLY_EXIT_GAP = 5.0   # exit early if gap narrows to < $5
STOP_LOSS_PCT = 0.05   # KSA gains >5% from short entry -> cover


def get_breakeven(date: pd.Timestamp):
    """Return the IMF Saudi fiscal breakeven for the year of `date`."""
    return IMF_SAUDI_BREAKEVEN.get(date.year)


def main():
    sid = "PL770_saudi_breakeven_gap_ksa_short"

    # ---- Load prices ----
    tickers = ["KSA", "SPY"]
    try:
        px = load_prices(tickers, start="2015-09-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # ---- Load Brent spot from FRED ----
    try:
        brent_raw = load_fred("DCOILBRENTEU", start="2015-01-01")
        brent = brent_raw.squeeze().ffill().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED DCOILBRENTEU: {e}")

    # Align to trading days
    trading_idx = px.index

    # Compute rolling 60-calendar-day average of Brent
    # Resample Brent to daily, ffill gaps (weekends/holidays)
    brent_daily = brent.reindex(
        pd.date_range(brent.index.min(), brent.index.max(), freq="D")
    ).ffill()

    # rolling 60-day calendar average
    brent_60 = brent_daily.rolling(window=60, min_periods=30).mean()
    # Reindex to trading days
    brent_60_td = brent_60.reindex(trading_idx, method="ffill").dropna()

    # ---- Compute signal (gap = breakeven - brent_60avg > GAP_THRESHOLD) ----
    gap_series = pd.Series(np.nan, index=trading_idx)
    breakeven_series = pd.Series(np.nan, index=trading_idx)
    for dt in trading_idx:
        be = get_breakeven(dt)
        if be is None:
            continue
        b60 = brent_60_td.get(dt)
        if b60 is None or np.isnan(b60):
            continue
        gap_series[dt] = be - float(b60)
        breakeven_series[dt] = be

    # Signal fires on first day where gap > GAP_THRESHOLD (from below threshold)
    prev_gap = gap_series.shift(1)
    signal = (gap_series > GAP_THRESHOLD) & ((prev_gap <= GAP_THRESHOLD) | prev_gap.isna())

    signal_dates = gap_series.index[signal.fillna(False)]
    print(f"Signal dates (raw, before min-gap filter): {list(str(d.date()) for d in signal_dates)}")

    # ---- Event study ----
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    ksa_r = ret["KSA"].dropna()

    pnl = pd.Series(0.0, index=trading_idx)
    positions = pd.Series(0.0, index=trading_idx)  # -1 = short KSA

    events = []
    last_exit_date = None

    for sig_dt in signal_dates:
        # Enforce min 90 calendar days between signals
        if last_exit_date is not None:
            gap_days = (sig_dt - last_exit_date).days
            if gap_days < MIN_DAYS_BETWEEN:
                continue

        # Entry: next trading day
        future = trading_idx[trading_idx > sig_dt]
        if len(future) == 0:
            continue
        entry_dt = future[0]
        entry_pos = trading_idx.get_loc(entry_dt)

        # Get entry price for stop-loss
        ksa_prices = px["KSA"]
        entry_price = float(ksa_prices.iloc[entry_pos])

        # Walk forward holding up to HOLD_DAYS
        exit_pos = min(entry_pos + HOLD_DAYS, len(trading_idx))
        exit_reason = "scheduled_30d"
        actual_exit_pos = exit_pos  # exclusive

        # Check for early exit and stop-loss day by day
        for j in range(entry_pos, exit_pos):
            dt_j = trading_idx[j]
            ksa_price_j = float(ksa_prices.iloc[j])
            # Stop-loss: KSA gains >5% from entry (short position loses)
            if ksa_price_j > entry_price * (1 + STOP_LOSS_PCT):
                exit_reason = "stop_loss"
                actual_exit_pos = j + 1  # exclusive (include this day)
                break
            # Early exit: gap narrows to < EARLY_EXIT_GAP
            be_j = get_breakeven(dt_j)
            b60_j = brent_60_td.get(dt_j)
            if be_j is not None and b60_j is not None and not np.isnan(b60_j):
                gap_j = be_j - float(b60_j)
                if gap_j < EARLY_EXIT_GAP:
                    exit_reason = "gap_narrowed"
                    actual_exit_pos = j + 1
                    break

        # Apply SHORT KSA returns for hold window
        hold_ksa = ksa_r.iloc[entry_pos:actual_exit_pos]
        short_ksa_pnl = -hold_ksa  # short = negative of return

        ev_return = float((1 + hold_ksa).prod() - 1)  # KSA return (positive = loss for short)
        short_return = float((1 / (1 + hold_ksa)).prod() - 1)  # short return

        for j in range(entry_pos, actual_exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j] = float(-ksa_r.iloc[j])

        exit_dt = trading_idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt
        last_exit_date = exit_dt

        events.append({
            "signal_date": str(sig_dt.date()),
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "exit_reason": exit_reason,
            "n_hold_days": int(actual_exit_pos - entry_pos),
            "gap_at_signal": round(float(gap_series.get(sig_dt, np.nan)), 2),
            "breakeven_at_signal": round(float(breakeven_series.get(sig_dt, np.nan)), 1),
            "brent60_at_signal": round(float(brent_60_td.get(sig_dt, np.nan)), 2),
            "ksa_return": round(ev_return, 4),
            "short_ksa_return": round(short_return, 4),
        })

    n_events = len(events)
    print(f"\nTraded events: {n_events}")
    for e in events:
        print(f"  {e['signal_date']} -> entry {e['entry_date']}, gap={e['gap_at_signal']:.1f}, "
              f"short_return={e['short_ksa_return']:.4f}, reason={e['exit_reason']}")

    # ---- Compute metrics ----
    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        event_returns = [e["short_ksa_return"] for e in events]
        avg_ret = float(np.mean(event_returns)) if event_returns else None
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={
                "events": events,
                "avg_event_return": avg_ret,
                "n_events": n_events,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Saudi Breakeven Gap -> Short KSA (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event stats
    short_returns = [e["short_ksa_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in short_returns])) if short_returns else None
    avg_short_ret = float(np.mean(short_returns)) if short_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When trailing 60-calendar-day average of FRED Brent spot (DCOILBRENTEU) "
                f"falls more than ${GAP_THRESHOLD}/bbl below the IMF Saudi Arabia annual "
                "fiscal breakeven price (IMF MENAP REO), enter SHORT KSA ETF the next "
                f"trading day; hold {HOLD_DAYS} trading days. Early exit if gap narrows "
                f"to <${EARLY_EXIT_GAP}. Stop-loss: KSA +{STOP_LOSS_PCT*100:.0f}% from entry. "
                "Min 90 calendar days between signals."
            ),
            "mechanism": (
                "Saudi Arabia's budget requires ~$80-100/bbl Brent to balance. When "
                "Brent trades persistently below the fiscal breakeven, Saudi faces "
                "pressure to either cut spending (negative for GDP and corporate "
                "revenues → KSA lower) or draw down reserves (FX risk). The 60-day "
                "average smooths transient price dips; a $10 gap signals structural "
                "fiscal stress, which historically weighs on Saudi equities (Tadawul)."
            ),
            "source": (
                "FRED DCOILBRENTEU (daily Brent spot, free); "
                "IMF MENAP Regional Economic Outlook (annual, public); "
                "prices via yfinance (auto_adjust=True)"
            ),
            "tickers": ["KSA"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_short_event_return": round(avg_short_ret, 4) if avg_short_ret is not None else None,
            "gap_threshold_used": GAP_THRESHOLD,
            "imf_breakeven_table": IMF_SAUDI_BREAKEVEN,
            "events": events,
            "caveats": (
                "KSA ETF launched Sep 2015; only ~10 years of backtest history. "
                "IMF fiscal breakeven estimates are published annually (April) and "
                "subject to revision — hard-coded values are approximations. "
                "Brent futures roll costs not captured (using spot series). "
                "KSA ETF is thinly traded and has wide bid/ask spreads in short. "
                "Saudi Aramco IPO (Dec 2019) structurally changed KSA composition. "
                "Metrics computed on held-days only (short window). "
                "GAP_THRESHOLD of $10 chosen to match spec language; sensitivity "
                "to this parameter not tested here."
            ),
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_short_return: {avg_short_ret}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
