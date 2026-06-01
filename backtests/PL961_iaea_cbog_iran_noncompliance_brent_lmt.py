"""PL961_iaea_cbog_iran_noncompliance_brent_lmt — IAEA Iran Safeguards Resolution: Long BZ=F + LMT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL961_iaea_cbog_iran_noncompliance_brent_lmt"

    # IAEA Board of Governors Iran-specific resolutions and key GOV report dates
    # Sources: iaea.org/newscenter GOV document feed
    # Format: (event_date, description)
    events_raw = [
        ("2021-11-26", "IAEA Board resolution on Iran JCPOA violations"),
        ("2022-06-08", "IAEA Board censure resolution on Iran unexplained nuclear material"),
        ("2022-11-17", "IAEA Board second censure resolution on Iran"),
        ("2024-09-10", "IAEA GOV report on Iranian 60% enrichment stockpile growth"),
        ("2025-09-22", "EU-3 JCPOA snapback mechanism triggered - functionally equivalent signal"),
    ]

    try:
        px = load_prices(["BZ=F", "LMT", "RTX", "GLD", "SPY"], start="2021-01-01")
    except Exception as e:
        # Try without RTX in case of ticker issue
        try:
            px = load_prices(["BZ=F", "LMT", "GLD", "SPY"], start="2021-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load: {e} / {e2}")

    if px.empty:
        return mark_failed(sid, "price data unavailable")

    # Check required tickers
    for tk in ["BZ=F", "LMT", "SPY"]:
        if tk not in px.columns:
            return mark_failed(sid, f"Required ticker {tk} not available")

    ret = daily_returns(px)
    brent_r = ret["BZ=F"]
    lmt_r = ret["LMT"]
    spy_r = ret["SPY"]

    HOLD_DAYS = 10

    # Build event-study PnL
    pnl = pd.Series(0.0, index=ret.index)
    event_details = []

    for (ev_date_str, desc) in events_raw:
        ev_date = pd.Timestamp(ev_date_str)

        # Entry: T+1 after the event (next trading day)
        future_dates = ret.index[ret.index > ev_date]
        if len(future_dates) == 0:
            continue

        entry_date = future_dates[0]
        entry_loc = ret.index.get_loc(entry_date)

        # Exit after HOLD_DAYS
        end_loc = min(entry_loc + HOLD_DAYS, len(ret) - 1)

        brent_seg = brent_r.iloc[entry_loc:end_loc]
        lmt_seg = lmt_r.iloc[entry_loc:end_loc]

        if len(brent_seg) < 3:
            continue

        # Equal notional: 50% Brent + 50% LMT
        combined = 0.5 * brent_seg.values + 0.5 * lmt_seg.values[:len(brent_seg)]
        combined_series = pd.Series(combined, index=brent_seg.index)

        # Stop-loss: exit if either leg drops >3%
        brent_cum = (1 + brent_seg).cumprod() - 1
        lmt_cum = (1 + lmt_seg.iloc[:len(brent_seg)]).cumprod() - 1

        stop_idx = None
        for j in range(len(combined)):
            if brent_cum.iloc[j] < -0.03 or lmt_cum.iloc[j] < -0.03:
                stop_idx = j
                break
        if stop_idx is not None:
            combined_series = combined_series.iloc[:stop_idx + 1]

        # Add to aggregate PnL
        for idx, val in combined_series.items():
            if idx in pnl.index:
                pnl[idx] = pnl[idx] + val

        brent_total = float((1 + brent_seg.iloc[:len(combined_series)]).prod() - 1)
        lmt_total = float((1 + lmt_seg.iloc[:len(combined_series)]).prod() - 1)
        combined_total = float((1 + combined_series).prod() - 1)

        event_details.append({
            "event_date": ev_date_str,
            "entry_date": str(entry_date.date()),
            "description": desc,
            "hold_days": len(combined_series),
            "brent_return": round(brent_total, 4),
            "lmt_return": round(lmt_total, 4),
            "combined_return": round(combined_total, 4),
        })

    pnl = pnl.dropna()
    active_days = int((pnl != 0).sum())

    if len(event_details) < 3:
        return mark_failed(sid, f"insufficient events: {len(event_details)}")

    if active_days < 20:
        return mark_failed(sid, f"insufficient active days: {active_days} (only {len(event_details)} events x {HOLD_DAYS} days each)")

    m = compute_metrics(pnl, benchmark=spy_r, name="IAEA Iran Non-Compliance → Long Brent + LMT")

    avg_combined = np.mean([e["combined_return"] for e in event_details])
    avg_brent = np.mean([e["brent_return"] for e in event_details])
    avg_lmt = np.mean([e["lmt_return"] for e in event_details])
    hit_rate = np.mean([e["combined_return"] > 0 for e in event_details])

    save_result(sid, m, extra={
        "rule": "Long BZ=F (50%) + Long LMT (50%) at T+1 after IAEA Board of Governors passes Iran safeguards non-compliance resolution or GOV report shows HEU stockpile growth >15% QoQ; hold 10 trading days; stop if either leg -3%",
        "mechanism": "IAEA Iran non-compliance resolution signals elevated geopolitical risk premium for crude oil (Middle East supply disruption risk) and defense budget uplift expectations (LMT, RTX as beneficiaries of US/European defense response)",
        "source": "IAEA iaea.org/newscenter GOV document feed; prices via yfinance BZ=F, LMT, SPY",
        "n_events": len(event_details),
        "n_active_days": active_days,
        "avg_combined_return": round(float(avg_combined), 4),
        "avg_brent_return": round(float(avg_brent), 4),
        "avg_lmt_return": round(float(avg_lmt), 4),
        "event_hit_rate": round(float(hit_rate), 4),
        "events": event_details,
        "caveat": "Only 5 events identified (2021-2025); small sample with limited statistical power. JCPOA snapback event (Sep 2025) included as functionally equivalent to IAEA resolution.",
    })

    print(f"Done: {len(event_details)} events, hit_rate={hit_rate:.2%}, avg_combined={avg_combined:.2%}, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
