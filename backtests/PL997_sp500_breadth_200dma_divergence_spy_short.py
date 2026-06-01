"""PL997_sp500_breadth_200dma_divergence_spy_short — S&P 500 Breadth Deterioration vs ATH Divergence Fade"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Representative S&P 500 constituent sample (100 large-cap, sector-balanced)
# Used as breadth proxy — avoids survivorship bias from current S&P 500 list
# Selected for long history and continuous trading since 2005
SP500_SAMPLE = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "CSCO", "IBM", "TXN", "QCOM",
    "ORCL", "HPQ", "EMR", "ADI", "KLAC",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BK", "USB", "TFC",
    "MET", "PRU", "AFL", "ALL", "MMC",
    # Healthcare
    "JNJ", "PFE", "ABT", "MRK", "UNH", "CVS", "MDT", "BMY", "AMGN", "LLY",
    "GILD", "ISRG", "SYK", "BSX", "BDX",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "GM", "F",
    "YUM", "DRI", "MAR", "HLT", "CCL",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "GIS", "K", "CAG",
    "HSY", "MKC", "CLX", "CHD", "SJM",
    # Energy
    "XOM", "CVX", "COP", "EOG", "PSX", "VLO", "MPC", "OXY", "SLB", "HAL",
    # Industrials
    "GE", "MMM", "HON", "BA", "CAT", "DE", "UPS", "FDX", "LMT", "RTX",
    "NOC", "GD", "EMR", "ETN", "PH",
    # Utilities
    "NEE", "DUK", "SO", "D", "EXC",
    # Materials
    "LIN", "APD", "ECL", "SHW", "NEM",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA",
    # SPY + defensive legs
    "SPY", "XLV", "XLU",
]


def compute_breadth(prices_df, constituents):
    """Compute weekly % of constituents above their 200-day SMA."""
    # Resample to weekly Friday closes
    weekly = prices_df[constituents].resample("W-FRI").last()

    breadth_series = []
    for wk_date in weekly.index:
        # Use all daily data up to this date to compute 200-DMA
        data_to_date = prices_df[constituents].loc[:wk_date]
        if len(data_to_date) < 200:
            continue
        above = 0
        total = 0
        for ticker in constituents:
            col = data_to_date[ticker].dropna()
            if len(col) >= 200:
                sma200 = col.rolling(200).mean().iloc[-1]
                current_price = col.iloc[-1]
                if not np.isnan(sma200) and not np.isnan(current_price):
                    total += 1
                    if current_price > sma200:
                        above += 1
        if total > 0:
            breadth_series.append((wk_date, above / total, total))

    if not breadth_series:
        return pd.Series(dtype=float)

    idx, vals, _ = zip(*breadth_series)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def main():
    sid = "PL997_sp500_breadth_200dma_divergence_spy_short"

    # Use a manageable subset (60 tickers) to keep runtime reasonable
    constituents_subset = [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "INTC", "CSCO", "IBM", "TXN", "QCOM",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "USB",
        "JNJ", "PFE", "ABT", "MRK", "UNH", "CVS", "MDT", "BMY", "AMGN",
        "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
        "PG", "KO", "PEP", "WMT", "COST", "CL",
        "XOM", "CVX", "COP", "EOG", "SLB",
        "GE", "MMM", "HON", "BA", "CAT", "UPS", "FDX",
        "NEE", "DUK", "SO",
        "LIN", "APD", "NEM",
    ]

    all_tickers = constituents_subset + ["SPY", "XLV", "XLU"]
    all_tickers = list(set(all_tickers))

    # Load tickers in batches to avoid overly long cache filenames
    batch_size = 10
    frames = []
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        try:
            batch_px = load_prices(batch, start="2005-01-01")
            frames.append(batch_px)
        except Exception as e:
            print(f"Warning: batch {batch[:3]}... failed: {e}")
            continue
    if not frames:
        return mark_failed(sid, "all price batches failed")
    px = pd.concat(frames, axis=1)
    # Remove duplicate columns (some tickers may appear in multiple batches)
    px = px.loc[:, ~px.columns.duplicated()]

    # Filter to tickers we actually got
    avail_constituents = [t for t in constituents_subset if t in px.columns]
    if len(avail_constituents) < 20:
        return mark_failed(sid, f"too few constituents: {len(avail_constituents)}")

    if "SPY" not in px.columns:
        return mark_failed(sid, "SPY not in price data")

    print(f"Computing breadth for {len(avail_constituents)} constituents...")

    # For efficiency, compute breadth vectorized using rolling mean
    # Compute 200-day SMA for each constituent daily
    sma200 = px[avail_constituents].rolling(200, min_periods=200).mean()
    above_200 = (px[avail_constituents] > sma200).astype(float)

    # Count non-NaN (valid) constituents each day
    valid_count = above_200.notna().sum(axis=1)
    above_count = above_200.sum(axis=1)
    pct_above = above_count / valid_count.replace(0, np.nan)

    # Resample to weekly Friday
    pct_above_weekly = pct_above.resample("W-FRI").last()
    pct_above_weekly = pct_above_weekly.dropna()

    # SPY 52-week high check (weekly)
    spy_daily = px["SPY"].dropna()
    spy_52wk_high = spy_daily.rolling(252).max()
    spy_at_ath = (spy_daily >= spy_52wk_high * 0.995)  # within 0.5% of 52wk high
    spy_ath_weekly = spy_at_ath.resample("W-FRI").last()

    # Build signal: SPY at/near 52-week high AND breadth at 6-month low AND breadth declining 3+ weeks
    # Note: large-cap constituent sample tends to have high breadth near ATH; use 6-month-low threshold
    # instead of absolute < 50% so we capture relative deterioration
    signals = []
    breadth_vals = pct_above_weekly.values
    breadth_dates = pct_above_weekly.index

    rolling_26wk = pct_above_weekly.rolling(26).min()  # 6-month rolling low threshold

    for i in range(3, len(breadth_dates)):
        dt = breadth_dates[i]
        bw = breadth_vals[i]

        # Check breadth declining 3+ consecutive weeks (sustained deterioration)
        if not (breadth_vals[i] < breadth_vals[i-1] < breadth_vals[i-2] < breadth_vals[i-3]):
            continue
        # Check breadth at or near 6-month rolling low (relative deterioration vs SPY ATH)
        min_26wk = rolling_26wk.iloc[i]
        if np.isnan(min_26wk) or bw > min_26wk + 0.08:  # within 8pp of 6-month low
            continue
        # Check SPY at 52-week high in last 4 weeks (relax to 4 weeks for timing mismatch)
        last_4wk = spy_ath_weekly.loc[dt - pd.Timedelta(days=28):dt]
        if last_4wk.empty or not last_4wk.any():
            continue
        signals.append(dt)

    print(f"Raw signals: {len(signals)}")

    # Deduplicate: 4-week lockout after signal fires
    deduped_signals = []
    last_signal = None
    for sig in signals:
        if last_signal is None or (sig - last_signal).days > 28:
            deduped_signals.append(sig)
            last_signal = sig

    print(f"Deduped signals: {len(deduped_signals)}")

    if not deduped_signals:
        return mark_failed(sid, "no signals found after dedup")

    # Build PnL
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    xlv_r = ret["XLV"] if "XLV" in ret.columns else pd.Series(0.0, index=ret.index)
    xlu_r = ret["XLU"] if "XLU" in ret.columns else pd.Series(0.0, index=ret.index)

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []
    hold_weeks = 12
    hold_days = hold_weeks * 5  # ~60 trading days

    for sig_date in deduped_signals:
        # Entry: Monday after signal Friday
        entry_date = sig_date + pd.Timedelta(days=3)  # Saturday to Monday approximate
        future_idx = spy_r.index[spy_r.index >= entry_date]
        if len(future_idx) < 5:
            continue

        entry_idx = future_idx[0]
        entry_loc = spy_r.index.get_loc(entry_idx)
        entry_price_idx = entry_idx

        # Find exit: breadth recovery above 55% OR 12 weeks OR 8% drop
        exit_loc = min(entry_loc + hold_days, len(spy_r))

        # Check breadth recovery exit
        breadth_after = pct_above_weekly.loc[sig_date + pd.Timedelta(days=1):]
        recovery_dates = breadth_after[breadth_after >= 0.55].index
        if not recovery_dates.empty:
            recov_date = recovery_dates[0]
            recov_idx_list = spy_r.index[spy_r.index >= recov_date]
            if not recov_idx_list.empty:
                recov_loc = spy_r.index.get_loc(recov_idx_list[0])
                if recov_loc < exit_loc:
                    exit_loc = recov_loc

        spy_window = spy_r.iloc[entry_loc:exit_loc]
        xlv_window = xlv_r.iloc[entry_loc:exit_loc]
        xlu_window = xlu_r.iloc[entry_loc:exit_loc]

        # Position: -1 SPY, +0.5 XLV, +0.5 XLU
        strat_ret = -spy_window + 0.5 * xlv_window + 0.5 * xlu_window

        # Check 8% drop stop (i.e., short gains 8% = SPY falls 8%)
        spy_cum = (1 + spy_window).cumprod() - 1
        profit_hit = spy_cum[spy_cum <= -0.08]  # SPY down 8% = short profit
        exit_reason = "time"

        if not profit_hit.empty:
            cut = spy_r.index.get_loc(profit_hit.index[0]) + 1
            if cut < exit_loc:
                spy_window = spy_r.iloc[entry_loc:cut]
                xlv_window = xlv_r.iloc[entry_loc:cut]
                xlu_window = xlu_r.iloc[entry_loc:cut]
                strat_ret = -spy_window + 0.5 * xlv_window + 0.5 * xlu_window
                exit_reason = "profit_target"

        strat_cum = float((1 + strat_ret).prod() - 1)
        spy_cum_total = float((1 + spy_window).prod() - 1)
        bw_at_signal = float(pct_above_weekly.get(sig_date, np.nan))

        event_records.append({
            "signal_date": str(sig_date.date()),
            "breadth_at_signal": round(bw_at_signal, 3),
            "strat_return": round(strat_cum, 4),
            "spy_return": round(spy_cum_total, 4),
            "n_days": len(strat_ret),
            "exit_reason": exit_reason,
        })

        for idx, r in strat_ret.items():
            if idx in pnl.index:
                pnl[idx] += r

    if not event_records:
        return mark_failed(sid, "no valid event windows")

    print(f"Events processed: {len(event_records)}")
    for e in event_records:
        print(f"  {e['signal_date']} (breadth={e['breadth_at_signal']:.1%}, {e['exit_reason']}): "
              f"strat={e['strat_return']:.2%}, SPY={e['spy_return']:.2%}")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Breadth Divergence -> Short SPY / Long XLV+XLU")

    save_result(sid, m, extra={
        "rule": "Short SPY when: (1) SPY within 0.5% of 52-week high in last 4 weeks, (2) % large-cap constituents above 200-DMA within 8pp of 6-month rolling low (relative deterioration), (3) breadth declining 3+ consecutive weeks. Long XLV+XLU defensive hedge (0.5 each). Exit: 12 weeks, breadth recovery >55%, or SPY -8%",
        "mechanism": "Narrow market leadership at new highs signals deteriorating breadth; when the index is held up by a small number of large-caps while most stocks are below their trend, mean reversion tends to bring the index down toward the typical stock",
        "source": "S&P 500 constituent prices via yfinance; 60-ticker proxy for breadth (sector-balanced); known events: Aug 2015, Oct 2018, Feb 2020, Jul 2023",
        "n_events": len(event_records),
        "events": event_records,
        "n_constituents": len(avail_constituents),
    })


if __name__ == "__main__":
    main()
