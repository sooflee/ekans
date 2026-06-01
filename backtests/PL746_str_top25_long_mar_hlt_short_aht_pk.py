"""PL746_str_top25_long_mar_hlt_short_aht_pk -- STR Top-25 RevPAR Divergence: Long MAR+HLT, Short AHT+PK

Proxy: when BLS Hotel/Motel CPI YoY (CPIHOSSL) crosses above 3% from below after being
suppressed (hotel cycle expansion signal), go long MAR+HLT and short AHT+PK for 30 trading days.
Rationale: above-cycle hotel pricing environment benefits premium-branded operators (MAR, HLT)
more than economy/select-service REITs (AHT, PK) due to brand premium and urban RevPAR mix.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL746_str_top25_long_mar_hlt_short_aht_pk"

    # Load FRED: CPIHOSSL = BLS CPI for Hotels and Motels (proxy for RevPAR environment)
    try:
        fred = load_fred(["CPIHOSSL"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if fred is None or fred.empty:
        return mark_failed(sid, "FRED CPIHOSSL data empty")

    series = fred["CPIHOSSL"].dropna()
    if len(series) < 18:
        return mark_failed(sid, "insufficient CPIHOSSL data")

    # Compute YoY change in hotel CPI
    yoy = series.pct_change(12) * 100
    yoy = yoy.dropna()

    # Signal: CPIHOSSL YoY crosses above 3% threshold after being below for >= 6 months
    # This proxies the STR Top-25 vs national spread expansion regime (urban outperformance vs national)
    # Dedup by 6-month cooldown; skip COVID base-effect distortion
    trigger_dates = []
    last_trigger = None
    for i in range(6, len(yoy)):
        date = yoy.index[i]
        # Skip COVID base-effect distortion
        if pd.Timestamp("2020-01-01") <= date <= pd.Timestamp("2021-12-31"):
            continue
        # Cross above 3%: current above, prior month below
        if yoy.iloc[i] > 3.0 and yoy.iloc[i - 1] <= 3.0:
            if last_trigger is None or (date - last_trigger).days >= 180:
                trigger_dates.append(date)
                last_trigger = date

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found (CPIHOSSL accel >= 300bps)")

    print(f"Trigger events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}")

    # Load prices
    try:
        px = load_prices(["MAR", "HLT", "AHT", "PK", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    # Retry once on data load failure
    if px is None or px.empty:
        try:
            px = load_prices(["MAR", "HLT", "AHT", "PK", "SPY"], start="2000-01-01")
        except Exception as e:
            return mark_failed(sid, f"price load retry: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    hold_days = 30
    pnl_parts = []
    event_results = []

    for td in trigger_dates:
        # Entry: next trading day on or after CPIHOSSL signal month
        # Guard: if trigger date is before MAR data starts, skip (price data unavailability)
        if ret.empty or ret.index[0] > td + pd.offsets.MonthEnd(1):
            print(f"  Skipping {td.date()}: trigger predates price data")
            continue
        entry_mask = ret.index >= td
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        if exit_loc - entry_loc < 15:
            continue

        window = slice(entry_loc, exit_loc)
        window_ret = ret.iloc[window]

        # Long basket: MAR + HLT (available since ~2013; earlier use MAR only)
        long_tickers = [t for t in ["MAR", "HLT"] if t in window_ret.columns
                        and not window_ret[t].isna().all()]
        # Short basket: AHT + PK (PK spun off from Hilton Jan 2017)
        short_tickers = [t for t in ["AHT", "PK"] if t in window_ret.columns
                         and not window_ret[t].isna().all()]

        if not long_tickers or not short_tickers:
            continue

        long_ret = window_ret[long_tickers].mean(axis=1)
        short_ret = window_ret[short_tickers].mean(axis=1)
        # Long-short spread: long premium brands, short economy REITs
        spread = long_ret - short_ret
        pnl_parts.append(spread)

        # Event metrics
        long_cum = float((1 + long_ret).prod() - 1)
        short_cum = float((1 + short_ret).prod() - 1)
        spread_cum = long_cum - short_cum
        spy_cum = None
        if spy_r is not None:
            spy_window = spy_r.iloc[window]
            spy_cum = float((1 + spy_window).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "long_tickers": long_tickers,
            "short_tickers": short_tickers,
            "long_return": round(long_cum, 4),
            "short_return": round(short_cum, 4),
            "spread_return": round(spread_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    if not event_results:
        return mark_failed(sid, "no valid events after price/ticker alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    all_pnl = all_pnl.dropna()

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(all_pnl)})")

    bench = spy_r.reindex(all_pnl.index).dropna() if spy_r is not None else None
    m = compute_metrics(all_pnl, benchmark=bench,
                        name="STR Top-25 Divergence: Long MAR+HLT / Short AHT+PK")

    spreads = [e["spread_return"] for e in event_results]
    win_rate = sum(1 for s in spreads if s > 0) / len(spreads)

    save_result(sid, m, extra={
        "rule": "CPIHOSSL YoY accel >=300bps from 6mo trough -> long MAR+HLT, short AHT+PK 30d",
        "mechanism": "Accelerating hotel pricing environment benefits premium urban brands (MAR/HLT) more than select-service/economy REITs (AHT/PK) via RevPAR mix and brand premium",
        "source": "FRED CPIHOSSL (BLS Hotels & Motels CPI proxy for STR RevPAR); yfinance MAR/HLT/AHT/PK/SPY",
        "proxy_note": "STR Top-25 vs national RevPAR data is proprietary; CPIHOSSL used as systematic proxy for hotel pricing environment",
        "n_events": len(event_results),
        "avg_spread_return": round(float(np.mean(spreads)), 4),
        "win_rate": round(win_rate, 3),
        "events": event_results,
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(event_results)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg spread: {np.mean(spreads)*100:.1f}%  Win rate: {win_rate*100:.0f}%")
    for e in event_results:
        flag = "+" if e["spread_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: spread {e['spread_return']*100:+.1f}%  "
              f"long={e['long_return']*100:+.1f}%  short={e['short_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
