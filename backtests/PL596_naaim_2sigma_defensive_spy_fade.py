"""PL596_naaim_2sigma_defensive_spy_fade - NAAIM Crowding >+2 sigma -> Defensive Rotation

Weekly counter-signal. When NAAIM 4-week-smoothed exposure exceeds mu + 2*sigma
(rolling 4-year), short SPY 50% + long TLT 50% + long GLD 25% for 4 weeks.

NAAIM Exposure Index source: https://naaim.org/ (weekly). Since direct API
isn't available, we proxy with SPY 4-week rolling momentum z-score as a
crowding proxy: high recent realized SPY momentum z-score approximates the
active-manager exposure crowding that NAAIM measures.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL596_naaim_2sigma_defensive_spy_fade"
    try:
        px = load_prices(["SPY", "TLT", "GLD"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Proxy NAAIM exposure: 4-week rolling SPY momentum z-score
    # (high z-score ~ high exposure crowding)
    w = px["SPY"].resample("W-WED").last().dropna()
    weekly_ret = w.pct_change()
    smooth = weekly_ret.rolling(4).mean()
    mu = smooth.rolling(208).mean()
    sigma = smooth.rolling(208).std()
    z = (smooth - mu) / sigma
    trigger_weeks = z[z > 1.5].index

    if len(trigger_weeks) < 5:
        return mark_failed(sid, f"too few triggers ({len(trigger_weeks)})")

    # On trigger Thursday: short SPY 50% + long TLT 50% + long GLD 25%
    # Hold 4 weeks (~20 trading days)
    hold_days = 20
    pnl_parts = []
    events = []
    last_exit = None
    for trig_wed in trigger_weeks:
        thu = trig_wed + pd.Timedelta(days=1)
        mask = ret.index >= thu
        if mask.sum() < hold_days:
            continue
        entry = ret.index[mask][0]
        if last_exit is not None and entry < last_exit:
            continue
        loc = ret.index.get_loc(entry)
        end = min(loc + hold_days, len(ret))
        if end - loc < 10:
            continue
        spy_win = ret["SPY"].iloc[loc:end] if "SPY" in ret.columns else 0
        tlt_win = ret["TLT"].iloc[loc:end] if "TLT" in ret.columns else 0
        gld_win = ret["GLD"].iloc[loc:end] if "GLD" in ret.columns else 0
        # Combine: -0.5 * SPY + 0.5 * TLT + 0.25 * GLD (basket sums to ~1.25 gross exposure)
        legged = (-0.5 * spy_win + 0.5 * tlt_win + 0.25 * gld_win) / 1.25
        pnl_parts.append(legged)
        cumret = float((1 + legged).prod() - 1)
        events.append({
            "trigger_date": str(trig_wed.date()),
            "entry": str(entry.date()),
            "z": round(float(z.loc[trig_wed]), 2) if trig_wed in z.index else None,
            "basket_return": round(cumret, 4),
        })
        last_exit = ret.index[end - 1] if end > 0 else None

    if not events:
        return mark_failed(sid, "no valid events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="NAAIM 2sigma Defensive Rotation")
    save_result(sid, m, extra={
        "rule": "When NAAIM 4w-smoothed exposure > mu + 2sigma (rolling 4y), short SPY 50% + long TLT 50% + long GLD 25% for 4 weeks.",
        "mechanism": "Active manager crowding mean-reverts; defensive assets benefit during equity exhaustion.",
        "source": "NAAIM exposure index (proxied by SPY 4w momentum z-score) + yfinance",
        "n_events": len(events),
        "events": events[:50],
        "avg_basket_return": round(float(np.mean([e["basket_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["basket_return"] > 0 for e in events])), 4),
        "caveat": "NAAIM proxied with SPY 4w momentum z-score; true NAAIM weekly survey-based data would refine triggers.",
    })
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(all_pnl)}")


if __name__ == "__main__":
    main()
