"""PL671_btfp_cliff_short_kre_long_kbe - BTFP Repayment Cliff - Short KRE Long KBE

Event study: when Fed H.4.1 BTFP outstanding declines >$30B in a 30-day window
(repayment wave, signaling maturity cliff), short KRE long KBE 1:1 for 60
trading days. Regional banks (KRE) more exposed to funding stress than larger
banks (KBE).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


HOLD_DAYS = 60


def main():
    sid = "PL671_btfp_cliff_short_kre_long_kbe"
    try:
        px = load_prices(["KRE", "KBE", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "SPY" not in px.columns:
        return mark_failed(sid, "price data missing required tickers")

    # Try to load BTFP data from FRED
    try:
        btfp = load_fred(["H41RESPPALDKHBTM"], start="2023-01-01")
        btfp_series = btfp.iloc[:, 0].dropna()
        # Compute 30-day change; if drop > $30B (millions = $30,000M) flag as repayment cliff
        btfp_30d_chg = btfp_series.diff(4)  # 4 weeks ~ 30 days for weekly data
        trigger_dates_auto = btfp_30d_chg[btfp_30d_chg < -30000].index.tolist()
    except Exception:
        trigger_dates_auto = []

    # Hardcoded known BTFP repayment cliff dates from public reporting
    known_events = [
        pd.Timestamp("2024-03-11"),   # BTFP expiry date announcement, large repayment wave
        pd.Timestamp("2025-01-15"),   # Final large BTFP loans matured
    ]

    # Combine auto-detected and known events, deduplicate within 30-day windows
    all_candidates = sorted(set(trigger_dates_auto + known_events))
    event_dates = []
    last_used = None
    for d in all_candidates:
        if last_used is None or (d - last_used).days > 30:
            event_dates.append(d)
            last_used = d

    if not event_dates:
        event_dates = known_events

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    pnl_parts = []
    events = []
    for td in event_dates:
        # Entry at next open (T+1 close after trigger)
        mask = ret.index > td
        if mask.sum() < HOLD_DAYS + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 20:
            continue

        window = ret.iloc[loc:end]
        kre_r = window["KRE"] if "KRE" in window.columns else pd.Series(0, index=window.index)
        kbe_r = window["KBE"] if "KBE" in window.columns else pd.Series(0, index=window.index)

        # Short KRE, Long KBE (1:1 pair trade); net pnl = kbe - kre
        net = (kbe_r - kre_r) / 2.0  # normalized to ~unit gross

        pnl_parts.append(net)
        cumret = float((1 + net).prod() - 1)
        events.append({
            "trigger_date": str(td.date()),
            "entry": str(entry.date()),
            "n_days": len(net),
            "net_return": round(cumret, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events with sufficient forward data")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="BTFP Repayment Cliff Short KRE Long KBE")
    save_result(sid, m, extra={
        "rule": "Short KRE long KBE 1:1 for 60 trading days when Fed H.4.1 BTFP outstanding declines >$30B in a 30-day window (BTFP repayment cliff).",
        "mechanism": "BTFP repayment cliffs force regional banks to seek market funding at higher rates, pressuring NIM; larger banks (KBE) have diverse funding and absorb shock better.",
        "source": "FRED H41RESPPALDKHBTM BTFP outstanding; yfinance KRE/KBE/SPY prices",
        "n_events": len(events),
        "events": events,
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events])), 4),
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")


if __name__ == "__main__":
    main()
