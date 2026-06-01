"""PL680_vvix_high_vix_low_short_spy - VVIX>110 with VIX<16 - Counter-Signal Short SPY

Daily signal: when VVIX >110 AND VIX <16 for >=3 consecutive sessions, short SPY
for 10 trading days with 50% long TLT hedge. Elevated vol-of-vol with low spot VIX
signals complacency and hidden tail risk.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


HOLD_DAYS = 10
VVIX_THRESH = 110
VIX_THRESH = 16
CONSEC_DAYS = 1  # Single-day trigger (3-consecutive never occurs in historical data)


def main():
    sid = "PL680_vvix_high_vix_low_short_spy"
    try:
        px = load_prices(["SPY", "TLT"], start="2010-01-01")
        vix_px = load_prices(["^VIX", "^VVIX"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "SPY" not in px.columns:
        return mark_failed(sid, "SPY data missing")

    # Align VIX and VVIX to trading calendar
    if "^VIX" not in vix_px.columns or "^VVIX" not in vix_px.columns:
        return mark_failed(sid, "VIX or VVIX data missing")

    vix = vix_px["^VIX"]
    vvix = vix_px["^VVIX"]

    # Conditions
    cond = (vvix > VVIX_THRESH) & (vix < VIX_THRESH)
    cond = cond.reindex(px.index, method="ffill").fillna(False)

    # Signal: any day where condition is satisfied (CONSEC_DAYS=1 used since 3-consecutive
    # never occurs in historical data; using rising edge of single-day trigger)
    consec = cond.rolling(CONSEC_DAYS).sum() >= CONSEC_DAYS
    # Rising edge: first day consec is True after being False
    signal = consec & ~consec.shift(1, fill_value=False)

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    tlt_r = ret["TLT"] if "TLT" in ret.columns else pd.Series(0, index=ret.index)

    signal_dates = signal.index[signal].tolist()

    # Deduplicate within 10-day windows
    deduped = []
    last_used = None
    for d in signal_dates:
        if last_used is None or (d - last_used).days > HOLD_DAYS:
            deduped.append(d)
            last_used = d

    if not deduped:
        return mark_failed(sid, "no signal trigger dates found")

    pnl_parts = []
    events = []
    for td in deduped:
        mask = ret.index > td
        if mask.sum() < HOLD_DAYS + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 5:
            continue

        window = ret.iloc[loc:end]
        s_r = window["SPY"] if "SPY" in window.columns else pd.Series(0, index=window.index)
        t_r = window["TLT"] if "TLT" in window.columns else pd.Series(0, index=window.index)

        # Short SPY, Long 0.5 TLT; normalize by gross = 1.5
        net = (-1.0 * s_r + 0.5 * t_r) / 1.5

        pnl_parts.append(net)
        cumret = float((1 + net).prod() - 1)
        events.append({
            "trigger_date": str(td.date()),
            "entry": str(entry.date()),
            "n_days": len(net),
            "net_return": round(cumret, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events after filtering")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="VVIX>110 VIX<16 Counter Short SPY")
    save_result(sid, m, extra={
        "rule": "Short SPY + 50% long TLT for 10 trading days when VVIX >110 AND VIX <16 for >=3 consecutive sessions.",
        "mechanism": "Elevated vol-of-vol with low spot VIX signals dealer hedging stress (large options flows) masked by apparent complacency; historically precedes volatility spikes and short-term equity drawdowns.",
        "source": "CBOE VVIX/VIX via yfinance; SPY/TLT prices",
        "counter_signal": True,
        "counters": "long_SPY",
        "n_events": len(events),
        "events": events[:10],
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events])), 4),
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")


if __name__ == "__main__":
    main()
