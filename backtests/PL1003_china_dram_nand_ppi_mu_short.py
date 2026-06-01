"""PL1003_china_dram_nand_ppi_mu_short
China DRAM/NAND Spot Undercut via BLS PPI Deflation — Short MU Gross Margin Compression

Entry: BLS PPI PCU334413334413 (Semiconductor & Related Devices) YoY <= -3%
for 2 consecutive months AND MU within 15% of 52-week high.
Short MU, long SOXX as partial hedge; hold 40 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1003_china_dram_nand_ppi_mu_short"
    try:
        px = load_prices(["MU", "SOXX", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        ppi_raw = load_fred(["PCU334413334413"], start="2004-01-01")
    except Exception as e:
        # retry once
        try:
            ppi_raw = load_fred(["PCU334413334413"], start="2004-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load FRED PCU334413334413: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["MU", "SOXX", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Daily returns
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    mu_r = ret["MU"].dropna()
    soxx_r = ret["SOXX"].dropna()

    # --- Build monthly PPI signal ---
    # PPI series is monthly; compute YoY change
    ppi = ppi_raw["PCU334413334413"].dropna()
    ppi_monthly = ppi.resample("MS").last()  # month-start frequency
    ppi_yoy = ppi_monthly.pct_change(12)  # YoY % change

    # Condition: YoY <= -3% for 2 consecutive months
    below_thresh = ppi_yoy <= -0.03
    # Signal triggers if current month AND prior month are both below threshold
    signal_months = below_thresh & below_thresh.shift(1)
    signal_months = signal_months.fillna(False)

    # Get the dates of signal trigger months (month-start dates)
    signal_dates = ppi_yoy.index[signal_months]

    # For each signal month, find the next trading day after that month-start as entry
    hold_days = 40
    # Position: short MU (weight -1), long SOXX (weight +0.5) as partial hedge
    # Net: delta-short MU, delta-long SOXX
    # PnL = -mu_return + 0.5 * soxx_return

    trading_dates = ret.index
    positions_mu = pd.Series(0.0, index=trading_dates)
    positions_soxx = pd.Series(0.0, index=trading_dates)

    # Also need MU near its 52-week high condition
    # Compute MU 252-day high (approx 1yr)
    mu_52wk_high = px["MU"].rolling(252, min_periods=126).max()

    events = []
    open_until_idx = -1  # exclusive end index into trading_dates

    for sig_month in signal_dates:
        # Find first trading day >= sig_month
        after = trading_dates[trading_dates >= sig_month]
        if len(after) == 0:
            continue
        entry_date = after[0]
        entry_idx = trading_dates.get_loc(entry_date)

        # Skip if already in a position
        if entry_idx <= open_until_idx:
            continue

        # Check MU within 15% of 52-week high at entry
        mu_price = px["MU"].get(entry_date, np.nan)
        high52 = mu_52wk_high.get(entry_date, np.nan)
        if np.isnan(mu_price) or np.isnan(high52) or high52 <= 0:
            continue
        ratio = mu_price / high52
        if ratio < 0.85:  # MU already down >15% from 52wk high, skip
            continue

        # Open position: short MU, long SOXX
        end_idx = min(entry_idx + hold_days, len(trading_dates))
        positions_mu.iloc[entry_idx:end_idx] = -1.0
        positions_soxx.iloc[entry_idx:end_idx] = 0.5
        open_until_idx = end_idx - 1

        events.append({
            "signal_month": str(sig_month.date()),
            "entry_date": str(entry_date.date()),
            "mu_price": round(float(mu_price), 2),
            "mu_pct_from_52wk_high": round(float(ratio - 1), 4),
            "ppi_yoy_at_signal": round(float(ppi_yoy.get(sig_month, np.nan)), 4),
        })

    # Compute daily PnL (no look-ahead: positions already set, no shift needed
    # because we set positions starting at entry_date, which means we're taking
    # that day's return — to be strict, shift by 1 so entry applies from next day)
    # Re-align: shift positions forward by 1 day (enter at next-open)
    pos_mu = positions_mu.shift(1).fillna(0)
    pos_soxx = positions_soxx.shift(1).fillna(0)

    # Daily PnL
    pnl_raw = pos_mu * mu_r + pos_soxx * soxx_r
    pnl = pnl_raw.dropna()
    pnl = pnl.reindex(spy_r.index).dropna()

    # Also compute combined positions for turnover-cost calc
    combined_pos = pd.Series(0.0, index=pnl.index)
    in_position_mask = (pos_mu.reindex(pnl.index) != 0) | (pos_soxx.reindex(pnl.index) != 0)
    combined_pos[in_position_mask] = 1.0

    if (pnl != 0).sum() < 30:
        return mark_failed(
            sid,
            f"insufficient in-position days: {(pnl != 0).sum()} (n_events={len(events)})",
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="China DRAM/NAND PPI Short MU",
        positions=combined_pos,
        cost_bps=10,
    )

    # Event-level stats
    ev_returns = []
    for e in events:
        entry = pd.Timestamp(e["entry_date"])
        # Find all returns from entry to entry+hold_days
        entry_iloc = trading_dates.get_loc(entry) if entry in trading_dates else None
        if entry_iloc is None:
            continue
        end_iloc = min(entry_iloc + hold_days, len(trading_dates))
        slice_dates = trading_dates[entry_iloc:end_iloc]
        slice_pnl = pnl_raw.reindex(slice_dates).dropna()
        if len(slice_pnl):
            cum = float((1 + slice_pnl).prod() - 1)
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
                "Short MU when BLS PPI PCU334413334413 (Semiconductor & Related Devices) "
                "YoY <= -3% for 2 consecutive months AND MU within 15% of 52-week high. "
                "Long SOXX at 0.5x as partial hedge to isolate MU commodity ASP exposure. "
                "Hold 40 trading days."
            ),
            "mechanism": (
                "China DRAM/NAND capacity expansion (CXMT/YMTC) floods commodity memory "
                "markets. BLS PPI semiconductor deflation (PCU334413334413 YoY <= -3%) "
                "is a lagged but clean backtestable proxy for DRAM/NAND spot ASP collapse. "
                "MU has ~60% commodity DRAM/NAND revenue concentration; margin compression "
                "follows ASP declines with ~1-2 quarter lag."
            ),
            "source": "FRED PCU334413334413 (BLS PPI Semiconductor & Related Devices); yfinance MU/SOXX/SPY",
            "tickers": ["MU", "SOXX", "SPY"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "MU has diversified into HBM (high-bandwidth memory) which partially "
                "offsets commodity DRAM/NAND headwinds; HBM surprise can break signal. "
                "PCU334413334413 is a broad semiconductor PPI, not pure DRAM/NAND. "
                "Sample has ~6 distinct episodes 2005-2025. SOXX hedge ratio of 0.5x "
                "is arbitrary; net exposure is -1 MU +0.5 SOXX = ~-0.5 net semi."
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
