"""PL998_cboe_total_pc_10dma_complacency_xly_xlp — CBOE Total Put/Call 10-DMA Complacency: Short XLY / Long XLP"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL998_cboe_total_pc_10dma_complacency_xly_xlp"

    try:
        px = load_prices(["XLY", "XLP", "SPY"], start="2010-01-01")
        vix = load_fred(["VIXCLS"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Proxy for CBOE total P/C ratio using VIX/VVIX divergence or direct VIX level
    # Since direct CBOE P/C data unavailable, use VIX as inverse proxy for complacency:
    # Low VIX = complacency (analogous to low P/C ratio)
    # Use 10-day rolling average of VIX as the signal
    vix_series = vix["VIXCLS"].reindex(ret.index, method="ffill").dropna()

    # Align VIX to price data
    common_idx = ret.index.intersection(vix_series.index)
    ret_aligned = ret.reindex(common_idx)
    vix_aligned = vix_series.reindex(common_idx)
    spy_r_aligned = spy_r.reindex(common_idx)

    # Compute 10-DMA of VIX
    vix_10dma = vix_aligned.rolling(10).mean()

    # Entry: VIX 10-DMA <= 10th percentile (complacency threshold)
    # Use 2010-2018 as in-sample to calibrate threshold
    is_mask = vix_10dma.index < pd.Timestamp("2019-01-01")
    is_vix = vix_10dma[is_mask].dropna()
    vix_threshold = float(np.percentile(is_vix.values, 10))
    print(f"VIX 10-DMA 10th-pctile threshold (in-sample 2010-2018): {vix_threshold:.2f}")

    # Additional filter: avoid extreme crisis (VIX raw > 30)
    vix_above_crisis = vix_aligned > 30

    hold_days = 21
    pnl = pd.Series(0.0, index=common_idx)
    in_position = pd.Series(False, index=common_idx)
    event_records = []

    # Generate signals: VIX 10-DMA drops to <= threshold and VIX < 30
    signal_dates = vix_10dma[
        (vix_10dma <= vix_threshold) & (~vix_above_crisis)
    ].index

    # Deduplicate: no overlapping windows — skip if already in position
    i = 0
    while i < len(signal_dates):
        sig_date = signal_dates[i]
        if in_position.get(sig_date, False):
            i += 1
            continue

        sig_loc = common_idx.get_loc(sig_date)
        entry_loc = sig_loc + 1  # enter next day
        exit_loc = min(entry_loc + hold_days, len(common_idx))

        if entry_loc >= len(common_idx):
            i += 1
            continue

        window_idx = common_idx[entry_loc:exit_loc]
        xly_ret = ret_aligned["XLY"].reindex(window_idx)
        xlp_ret = ret_aligned["XLP"].reindex(window_idx)

        # Long XLP, Short XLY (defensive over cyclical in complacent market)
        pair_ret = xlp_ret - xly_ret

        for idx in window_idx:
            pnl[idx] += pair_ret.get(idx, 0.0)
            in_position[idx] = True

        cum_pair = float((1 + pair_ret).prod() - 1)
        cum_spy = float((1 + spy_r_aligned.reindex(window_idx)).prod() - 1)
        event_records.append({
            "signal_date": str(sig_date.date()),
            "vix_10dma": round(float(vix_10dma.get(sig_date, np.nan)), 2),
            "pair_return": round(cum_pair, 4),
            "spy_return": round(cum_spy, 4),
            "n_days": len(window_idx),
        })

        # Skip forward by hold_days to avoid overlapping
        i = np.searchsorted(signal_dates, common_idx[min(exit_loc, len(common_idx)-1)])

    if not event_records:
        return mark_failed(sid, "no complacency signal events found")

    print(f"Events processed: {len(event_records)}")
    for e in event_records[:5]:
        print(f"  {e['signal_date']} VIX10d={e['vix_10dma']:.1f}: pair={e['pair_return']:.2%}, SPY={e['spy_return']:.2%}")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 50:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r_aligned, name="CBOE PC Complacency XLY/XLP Pair")

    save_result(sid, m, extra={
        "rule": "When VIX 10-DMA falls to bottom decile (<=10th pctile) and spot VIX<30, enter long XLP / short XLY for 21 trading days",
        "mechanism": "Low put/call (proxied by low VIX) signals complacency; in these regimes cyclicals (XLY) are vulnerable to mean-reversion while defensive consumer staples (XLP) provide relative safety",
        "source": "FRED VIXCLS; yfinance XLY, XLP, SPY; proxy for CBOE total P/C ratio",
        "n_events": len(event_records),
        "vix_threshold": round(vix_threshold, 2),
        "events_sample": event_records[:10],
    })


if __name__ == "__main__":
    main()
