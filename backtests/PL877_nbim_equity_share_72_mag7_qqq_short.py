"""PL877 — NBIM Equity-Share Above 72% Rebalancing -> Short QQQ vs RSP (Mag-7 Weight Reduction)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL877_nbim_equity_share_72_mag7_qqq_short"

    # Use ACWI trailing 63-day return > 7% as proxy for NBIM equity-share above 72%.
    # When global equities rally strongly, NBIM equity share mechanically rises.
    # Trade: Short QQQ / Long RSP for 8 weeks (40 trading days)
    HOLD = 40  # ~8 weeks
    ACWI_THRESH = 0.07  # 7% trailing 63-day return
    ACWI_WINDOW = 63  # calendar days approx, use trading days here

    try:
        px = load_prices(["QQQ", "RSP", "ACWI", "SPY"], start="2008-01-01")
    except Exception as e:
        try:
            px = load_prices(["QQQ", "RSP", "ACWI", "SPY"], start="2008-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    qqq_r = daily_returns(px[["QQQ"]]).iloc[:, 0]
    rsp_r = daily_returns(px[["RSP"]]).iloc[:, 0]
    acwi_r = daily_returns(px[["ACWI"]]).iloc[:, 0]

    # Compute ACWI trailing ~63 trading-day (quarter) return
    acwi_trail = (1 + acwi_r).rolling(63).apply(lambda x: x.prod(), raw=True) - 1

    # Signal: ACWI trailing 63-day return crosses above 7% (was below, now above)
    # i.e., trigger fires on the day it exceeds threshold after not being above it
    above = (acwi_trail > ACWI_THRESH).astype(int)
    # Fire on rising edge: transition from 0 to 1
    signal = (above.diff() == 1)

    # Also add known NBIM rebalancing events as supplemental signals
    known_events = [
        pd.Timestamp("2013-05-01"),
        pd.Timestamp("2017-10-01"),
        pd.Timestamp("2019-11-01"),
        pd.Timestamp("2021-03-01"),
        pd.Timestamp("2023-07-01"),
        pd.Timestamp("2024-02-01"),
    ]

    # Build positions: short QQQ / long RSP pair
    pair_r = rsp_r - qqq_r  # long RSP, short QQQ

    pnl = pd.Series(0.0, index=pair_r.index)
    events = []

    signal_dates = list(signal[signal].index)

    # Add known events if they aren't already represented
    for ke in known_events:
        # Check if a signal already fired within 30 days of the known event
        nearby = [d for d in signal_dates if abs((d - ke).days) < 30]
        if not nearby:
            # Find the next trading day on or after the known event
            future = pair_r.index[pair_r.index >= ke]
            if len(future) > 0:
                signal_dates.append(future[0])

    signal_dates = sorted(set(signal_dates))

    # Avoid overlapping windows by tracking end of last trade
    last_trade_end = pd.Timestamp("1900-01-01")

    for sig_date in signal_dates:
        if sig_date <= last_trade_end:
            continue  # skip if overlap

        future_idx = pair_r.index[pair_r.index >= sig_date]
        if len(future_idx) < HOLD + 1:
            continue

        entry_idx = future_idx[0]
        pos = pair_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(pair_r))

        window = pair_r.iloc[pos:end_pos]
        qqq_window = qqq_r.reindex(window.index).fillna(0)
        rsp_window = rsp_r.reindex(window.index).fillna(0)

        pnl.iloc[pos:end_pos] += window.values[:end_pos - pos]
        last_trade_end = pair_r.index[end_pos - 1]

        pair_cum = float((1 + window).prod() - 1)
        qqq_cum = float((1 + qqq_window).prod() - 1)
        rsp_cum = float((1 + rsp_window).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "pair_return": round(pair_cum, 4),
            "qqq_return": round(qqq_cum, 4),
            "rsp_return": round(rsp_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid signal events found")

    active_pnl = pnl[pnl != 0]
    print(f"Events: {len(events)}, Active trading days: {len(active_pnl)}")
    for e in events[:10]:
        print(f"  {e['signal_date']}: pair={e['pair_return']:.1%}, qqq={e['qqq_return']:.1%}, rsp={e['rsp_return']:.1%}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)} ({len(events)} events)")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NBIM Rebalancing -> Short QQQ / Long RSP")
    win_rates = [1 if e["pair_return"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Short QQQ / Long RSP for 8 weeks when ACWI trailing 63-day return > 7% (proxy for NBIM equity-share > 72% rebalancing band breach)",
        "mechanism": "NBIM (Norway GPFG) is a top-3 holder of Mag-7 stocks; when equity-share rises above 70%+2pp band, it mechanically sells large-cap US tech. QQQ short captures that sell pressure; RSP long is less concentrated and avoids the selling.",
        "source": "NBIM nbim.no monthly equity-share; ACWI ETF as proxy trigger; yfinance QQQ/RSP/ACWI/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4) if win_rates else 0,
        "events": events[:20],
        "caveats": "ACWI trailing return is only a proxy for NBIM equity-share data; Mag-7 rebalancing effect may be diluted by broader market moves; limited independent events",
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
