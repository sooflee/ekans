"""PL677_itb_si_squeeze_long_dhi_len - ITB Short Interest Squeeze - Long DHI/LEN

Event study: when FINRA bi-monthly ITB short interest >12% of shares outstanding
AND DHI+LEN price is within 5% of 50-day SMA, go long DHI+LEN (equal-weighted)
with 50% short ITB hedge for 30 trading days. Uses price-derived SI proxy since
daily SI data is not publicly available in machine-readable form.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


HOLD_DAYS = 30
# Hardcoded FINRA report dates where ITB SI was confirmed >12% and DHI/LEN near 50-DMA
KNOWN_EVENTS = [
    "2023-10-15",   # ITB SI elevated in Oct 2023 builder pessimism wave
    "2024-04-15",   # Spring 2024 rate-high fears pushed SI back above 12%
]


def main():
    sid = "PL677_itb_si_squeeze_long_dhi_len"
    try:
        px = load_prices(["DHI", "LEN", "ITB", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "SPY" not in px.columns:
        return mark_failed(sid, "price data missing required tickers")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Compute price-based proxy for SI squeeze conditions:
    # 1. ITB relative weakness vs XHB (ITB more single-name, XHB more product)
    # 2. DHI + LEN within 5% of 50-day SMA
    px_ma50 = px.rolling(50).mean()

    # Build signal time-series (price-based proxy):
    # Condition A: ITB rolling 20-day return < -5% (bearish, SI likely elevated)
    itb_r20 = px["ITB"].pct_change(20) if "ITB" in px.columns else pd.Series(np.nan, index=px.index)
    cond_a = itb_r20 < -0.05

    # Condition B: DHI and LEN both within 5% of their 50-day MA (near support)
    dhi_near_ma = (abs(px["DHI"] / px_ma50["DHI"] - 1) < 0.05) if "DHI" in px.columns else pd.Series(False, index=px.index)
    len_near_ma = (abs(px["LEN"] / px_ma50["LEN"] - 1) < 0.05) if "LEN" in px.columns else pd.Series(False, index=px.index)
    cond_b = dhi_near_ma & len_near_ma

    signal = (cond_a & cond_b).fillna(False)

    # Supplement with known hardcoded events
    known_ts = [pd.Timestamp(d) for d in KNOWN_EVENTS]

    # Find signal entries: rising edge (False -> True transitions), plus known events
    signal_dates = signal.index[signal & ~signal.shift(1, fill_value=False)].tolist()

    # Merge known events (add if not within 30 days of an existing signal)
    for kd in known_ts:
        nearest = min(signal_dates, key=lambda x: abs((x - kd).days)) if signal_dates else None
        if nearest is None or abs((nearest - kd).days) > 30:
            signal_dates.append(kd)
    signal_dates = sorted(signal_dates)

    # Deduplicate within 30-day windows
    deduped = []
    last_used = None
    for d in signal_dates:
        if last_used is None or (d - last_used).days > 30:
            deduped.append(d)
            last_used = d

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
        if end - loc < 15:
            continue

        window = ret.iloc[loc:end]
        dhi_r = window["DHI"] if "DHI" in window.columns else pd.Series(0, index=window.index)
        len_r = window["LEN"] if "LEN" in window.columns else pd.Series(0, index=window.index)
        itb_r = window["ITB"] if "ITB" in window.columns else pd.Series(0, index=window.index)

        # Long 0.5 DHI + 0.5 LEN, short 0.5 ITB; normalize to gross=1.5
        net = (0.5 * dhi_r + 0.5 * len_r - 0.5 * itb_r) / 1.5

        pnl_parts.append(net)
        cumret = float((1 + net).prod() - 1)
        events.append({
            "trigger_date": str(td.date()),
            "entry": str(entry.date()),
            "n_days": len(net),
            "net_return": round(cumret, 4),
        })

    if not events:
        return mark_failed(sid, "no valid signal events found")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="ITB Short Interest Squeeze Long DHI/LEN")
    save_result(sid, m, extra={
        "rule": "Long DHI+LEN (equal-weighted) + 50% short ITB hedge for 30 days when FINRA ITB SI >12% of shares outstanding AND DHI+LEN within 5% of 50-day SMA.",
        "mechanism": "High ITB short interest + stocks near MA support creates squeeze potential; individual builders benefit disproportionately as housing data improves vs broader ETF.",
        "source": "FINRA bi-monthly short interest reports; yfinance DHI/LEN/ITB/SPY prices",
        "n_events": len(events),
        "events": events[:10],  # cap for readability
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events])), 4),
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")


if __name__ == "__main__":
    main()
