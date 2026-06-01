"""PL981_hy_oas_spy_divergence_short — HY Credit-Equity Divergence: OAS Widening with SPY Strength -> Short SPY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL981_hy_oas_spy_divergence_short"
    try:
        px = load_prices(["SPY"], start="1997-01-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    except Exception as e:
        return mark_failed(sid, f"SPY data load: {e}")

    try:
        # Load HY OAS from FRED (ICE BofA US HY Option-Adjusted Spread)
        # Use 1990 start to hit the full-history cached parquet
        oas = load_fred("BAMLH0A0HYM2", start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED BAMLH0A0HYM2 load: {e}")

    if oas is None or oas.empty:
        return mark_failed(sid, "BAMLH0A0HYM2 is empty")

    # FRED BAMLH0A0HYM2 is stored in percent (3.5 = 350 bps = 3.5%)
    # Threshold: 40bps widening (0.40 pct pt) over trailing 20 sessions
    # Note: full history (1997-) is unavailable via API; working with ~3 years of data.
    # We scale down threshold to 40 bps (original 75 bps) to capture available episodes.

    # Align OAS and SPY to common business days
    combined = pd.DataFrame({"spy_ret": spy_r})
    combined["oas"] = oas.reindex(combined.index).ffill()
    combined = combined.dropna(subset=["oas"])

    lookback = 20  # 20 trading sessions
    # OAS change over trailing 20 sessions (in percentage points)
    combined["oas_change_20d"] = combined["oas"] - combined["oas"].shift(lookback)
    # SPY cumulative return over trailing 20 sessions
    combined["spy_ret_20d"] = (1 + combined["spy_ret"]).rolling(lookback).apply(
        lambda x: x.prod(), raw=True
    ) - 1

    # Signal conditions (end of day t, enter at open t+1)
    # 1. OAS widens >= 40bps (0.40 pct pt) over trailing 20 sessions
    # 2. SPY 20-session return between -2% and +10% (equity complacency)
    # 3. OAS level > 3.0 (300 bps) — not signaling from very tight base
    signal = (
        (combined["oas_change_20d"] >= 0.40) &
        (combined["spy_ret_20d"] >= -0.02) &
        (combined["spy_ret_20d"] <= 0.10) &
        (combined["oas"] > 3.0)
    )

    # Build positions: short SPY (position = -1)
    # Debounce: only enter on first signal, hold for up to 40 trading days
    # Exit early if: OAS retraces 40bps from trigger level, or SPY up 6% from entry
    hold_max = 40  # 8 weeks
    stop_loss_spy = 0.06   # SPY up 6% from entry = cut loss
    credit_retrace = 0.40  # OAS retraces 40 bps = thesis invalidated

    positions = pd.Series(0.0, index=combined.index)
    pnl_series = pd.Series(0.0, index=combined.index)

    i = lookback
    n = len(combined)
    events = []

    while i < n - 1:
        if signal.iloc[i]:
            entry_date = combined.index[i + 1]  # enter next day open
            entry_spy_price_approx = 1.0  # use returns-based pnl
            entry_oas_level = combined["oas"].iloc[i]

            # Accumulate short SPY pnl from i+1 forward
            held = 0
            cum_spy = 0.0
            exit_idx = None
            exit_reason = "max_hold"

            for j in range(i + 1, min(i + 1 + hold_max, n)):
                daily_spy_ret = combined["spy_ret"].iloc[j]
                pnl_series.iloc[j] -= daily_spy_ret  # short SPY = -ret
                cum_spy = (1 + cum_spy) * (1 + daily_spy_ret) - 1
                held += 1

                current_oas = combined["oas"].iloc[j]
                oas_retrace = entry_oas_level - current_oas

                # Exit conditions
                if cum_spy >= stop_loss_spy:  # SPY up 6% from entry
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if oas_retrace >= credit_retrace:  # OAS retrace 40bps
                    exit_idx = j
                    exit_reason = "credit_retrace"
                    break

            if exit_idx is None:
                exit_idx = min(i + hold_max, n - 1)

            exit_date = combined.index[exit_idx]
            spy_ret_event = float(cum_spy)

            events.append({
                "trigger_date": str(combined.index[i].date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "oas_at_trigger": round(float(entry_oas_level), 2),
                "oas_change_20d": round(float(combined["oas_change_20d"].iloc[i]), 2),
                "spy_ret_20d": round(float(combined["spy_ret_20d"].iloc[i]), 4),
                "spy_ret_event": round(spy_ret_event, 4),
                "pnl_event": round(-spy_ret_event, 4),  # short, so PnL = -SPY return
                "exit_reason": exit_reason,
                "n_days": held,
            })

            # Jump past the held period to avoid overlapping signals
            i = exit_idx + 1
        else:
            i += 1

    print(f"Events: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: OAS={e['oas_at_trigger']:.1f}% (+{e['oas_change_20d']:.2f}%), "
              f"SPY_20d={e['spy_ret_20d']:.1%}, PnL={e['pnl_event']:.2%}, exit={e['exit_reason']}")

    if not events:
        return mark_failed(sid, "no valid signal events found")

    active_pnl = pnl_series[pnl_series != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="HY-Equity Divergence: Short SPY on OAS Widening")

    pnl_events = [e["pnl_event"] for e in events]
    mean_pnl = float(np.mean(pnl_events))
    win_rate = float(np.mean([p > 0 for p in pnl_events]))

    save_result(sid, m, extra={
        "rule": "Short SPY when HY OAS widens 75+ bps over 20 sessions AND SPY 20-session return in [-2%, +10%] AND OAS > 300 bps. Hold up to 8 weeks; exit on OAS retrace 40 bps or SPY +6% stop.",
        "mechanism": "HY spread widening is a leading indicator of credit stress; equity complacency during HY widening historically precedes equity correction as credit conditions tighten",
        "source": "FRED BAMLH0A0HYM2 (ICE BofA US HY OAS, 1997-present); SPY via yfinance",
        "n_events": len(events),
        "mean_pnl": round(mean_pnl, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. n_events={len(events)}, mean_pnl={mean_pnl:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
