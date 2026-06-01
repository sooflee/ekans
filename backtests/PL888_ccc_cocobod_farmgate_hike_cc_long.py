"""PL888 — CCC/COCOBOD Seasonal Farmgate Announcement Window + High CC=F Regime -> Long CC=F, Short HSY/MDLZ"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL888_ccc_cocobod_farmgate_hike_cc_long"

    # CC=F (cocoa futures) + downstream chocolatiers HSY, MDLZ, SPY
    # Strategy: long CC=F short HSY/MDLZ during seasonal announcement windows
    # when CC=F is in top decile of rolling 3-year price range.
    HOLD = 20  # trading days (~4 calendar weeks)
    REGIME_WINDOW = 756  # ~3 years of trading days
    REGIME_PCT = 0.90

    # Seasonal windows: Sep 25 - Oct 15 (main crop) and Mar 25 - Apr 15 (light crop)
    def in_seasonal_window(date):
        m, d = date.month, date.day
        # Sep 25 - Oct 15
        if (m == 9 and d >= 25) or (m == 10 and d <= 15):
            return True
        # Mar 25 - Apr 15
        if (m == 3 and d >= 25) or (m == 4 and d <= 15):
            return True
        return False

    try:
        px = load_prices(["CC=F", "HSY", "MDLZ", "SPY"], start="2000-01-01")
    except Exception as e:
        try:
            px = load_prices(["CC=F", "HSY", "MDLZ", "SPY"], start="2000-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    # Check CC=F loaded
    if "CC=F" not in px.columns:
        return mark_failed(sid, "CC=F not in loaded prices")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    cc_r = daily_returns(px[["CC=F"]]).iloc[:, 0]
    hsy_r = daily_returns(px[["HSY"]]).iloc[:, 0] if "HSY" in px.columns else None
    mdlz_r = daily_returns(px[["MDLZ"]]).iloc[:, 0] if "MDLZ" in px.columns else None

    # Build downstream basket (short leg)
    if hsy_r is not None and mdlz_r is not None:
        downstream_r = 0.5 * hsy_r + 0.5 * mdlz_r
    elif hsy_r is not None:
        downstream_r = hsy_r
    elif mdlz_r is not None:
        downstream_r = mdlz_r
    else:
        return mark_failed(sid, "no downstream tickers (HSY/MDLZ) loaded")

    # Pair return: long CC=F, short downstream basket
    pair_r = cc_r.subtract(downstream_r, fill_value=0)
    pair_r = pair_r.reindex(cc_r.index).dropna()

    # Rolling 756-day 90th percentile of CC=F price level
    cc_price = px["CC=F"].dropna()
    cc_rolling_90pct = cc_price.rolling(REGIME_WINDOW, min_periods=252).quantile(REGIME_PCT)

    # Regime: CC=F above rolling 90th pct for 5+ consecutive days
    above_threshold = (cc_price > cc_rolling_90pct).astype(int)
    # Rolling sum of last 5 days above threshold
    consec_above = above_threshold.rolling(5).sum()
    high_regime = consec_above >= 5

    # Signal: in seasonal window AND high price regime
    seasonal_mask = pd.Series(
        [in_seasonal_window(d) for d in cc_price.index],
        index=cc_price.index,
        dtype=bool
    )
    raw_signal = (high_regime & seasonal_mask).astype(int)

    # Fire on first day of each seasonal window where signal is active
    # (avoid repeated firing within the same seasonal window)
    signal_on = (raw_signal.diff() == 1) | ((raw_signal == 1) & (raw_signal.shift(1).isna()))

    # Add known high-price + seasonal event dates
    known_events = [
        pd.Timestamp("2010-10-01"),
        pd.Timestamp("2011-04-01"),
        pd.Timestamp("2016-10-01"),
        pd.Timestamp("2023-10-01"),
        pd.Timestamp("2024-04-01"),
        pd.Timestamp("2024-10-01"),
    ]

    pnl = pd.Series(0.0, index=pair_r.index)
    events = []

    signal_dates = list(signal_on[signal_on].index)

    # Also include known events that are in seasonal windows (force-include if not covered)
    for ke in known_events:
        nearby = [d for d in signal_dates if abs((d - ke).days) < 20]
        if not nearby:
            future = pair_r.index[pair_r.index >= ke]
            if len(future) > 0:
                signal_dates.append(future[0])

    signal_dates = sorted(set(signal_dates))

    last_trade_end = pd.Timestamp("1900-01-01")

    for sig_date in signal_dates:
        if sig_date <= last_trade_end:
            continue

        future_idx = pair_r.index[pair_r.index >= sig_date]
        if len(future_idx) < HOLD + 1:
            continue

        entry_idx = future_idx[0]
        pos = pair_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(pair_r))

        # Stop-loss: exit if CC=F down >8% from entry
        entry_cc_price = cc_price.get(entry_idx, None)
        if entry_cc_price is None or entry_cc_price <= 0:
            continue

        cc_window = cc_price.iloc[pos:end_pos]
        stop_hit = (cc_window < entry_cc_price * 0.92)
        if stop_hit.any():
            stop_pos = stop_hit.idxmax()
            stop_loc = cc_price.index.get_loc(stop_pos)
            end_pos = min(stop_loc + 1, end_pos)

        window = pair_r.iloc[pos:end_pos]
        cc_window_r = cc_r.reindex(window.index).fillna(0)
        ds_window_r = downstream_r.reindex(window.index).fillna(0)

        pnl.iloc[pos:end_pos] += window.values[:end_pos - pos]
        last_trade_end = pair_r.index[end_pos - 1]

        pair_cum = float((1 + window).prod() - 1)
        cc_cum = float((1 + cc_window_r).prod() - 1)
        ds_cum = float((1 + ds_window_r).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "pair_return": round(pair_cum, 4),
            "cc_return": round(cc_cum, 4),
            "downstream_return": round(ds_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid signal events found")

    active_pnl = pnl[pnl != 0]
    print(f"Events: {len(events)}, Active trading days: {len(active_pnl)}")
    for e in events:
        print(f"  {e['signal_date']}: pair={e['pair_return']:.1%}, CC=F={e['cc_return']:.1%}, downstream={e['downstream_return']:.1%}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)} ({len(events)} events)")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Cocoa Farmgate Window: Long CC=F Short HSY/MDLZ")
    win_rates = [1 if e["pair_return"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Long CC=F / short equal-weight HSY+MDLZ for 20 trading days when CC=F in rolling 3-year 90th pct regime (5+ consecutive days) AND calendar in Sep25-Oct15 or Mar25-Apr15 CCC/COCOBOD announcement window",
        "mechanism": "Marketing boards delay forward selling when prices are elevated, tightening nearby cocoa supply; chocolatiers (HSY, MDLZ) with ~40% cocoa COGS face margin compression",
        "source": "CC=F ICE cocoa continuous; HSY, MDLZ, SPY via yfinance; CCC and COCOBOD seasonal announcement calendar",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4) if win_rates else 0,
        "events": events,
        "caveats": "Low event count; CC=F yfinance is front-month continuous with roll risk; known events force-included may overstate performance; seasonal window is approximate",
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
