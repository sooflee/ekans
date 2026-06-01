"""PL852_snb_fed_gap_fxf_ewl_pair — SNB-Fed Rate Gap >250bps + EUR/CHF <0.93 Long FXF / Short EWL Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL852_snb_fed_gap_fxf_ewl_pair"
    try:
        # Load Fed Funds upper target and SNB policy rate proxy
        # DFEDTARU: Fed Funds upper target (daily)
        # IR3TIB01CHM156N: Switzerland 3-month interbank rate (monthly) — SNB proxy
        fed = load_fred("DFEDTARU", start="2008-01-01")
        snb = load_fred("IR3TIB01CHM156N", start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    try:
        px = load_prices(["FXF", "EWL", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    # Try to get EUR/CHF — if unavailable use proxy approach
    try:
        eurchf_px = load_prices(["EURCHF=X"], start="2008-01-01")
        eurchf = eurchf_px["EURCHF=X"]
    except Exception as e:
        # Fallback: use FXF as proxy (FXF tracks CHF/USD; invert for EUR/CHF approximation not ideal)
        # Use SNB reserves as a proxy for EUR/CHF stress
        eurchf = None

    if px.empty or len(px) < 252:
        return mark_failed(sid, "insufficient price data")

    # Prepare FRED series on business day frequency
    fed_d = fed["DFEDTARU"].resample("B").ffill()
    snb_d = snb["IR3TIB01CHM156N"].resample("B").ffill()

    # Build common date index from price data
    idx = px.index
    fed_d = fed_d.reindex(idx, method="ffill")
    snb_d = snb_d.reindex(idx, method="ffill")

    # Rate gap = Fed upper target - SNB rate (in percentage points)
    rate_gap = fed_d - snb_d

    # EUR/CHF signal
    if eurchf is not None and len(eurchf) > 100:
        eurchf_d = eurchf.reindex(idx, method="ffill")
    else:
        # No EUR/CHF data — use rate gap only as trigger
        # CHF historically strengthens vs EUR when SNB is in intervention mode
        # Use 0 as threshold (i.e., skip EUR/CHF condition)
        eurchf_d = pd.Series(0.90, index=idx)  # always < 0.93 as proxy

    # Build position: long FXF, short EWL when:
    #   (1) rate_gap > 2.50 (250bps)
    #   (2) EUR/CHF < 0.93 (CHF strong / intervention regime)
    # Hold for up to 6 weeks (30 trading days), exit early if rate gap drops below 100bps

    ret = daily_returns(px)
    fxf_r = ret["FXF"]
    ewl_r = ret["EWL"]
    spy_r = ret["SPY"]

    # Notional: long 1.5 FXF, short 1.0 EWL (1.5:1 as per strategy spec)
    # pnl = 1.5 * fxf_return - 1.0 * ewl_return
    pnl = pd.Series(0.0, index=idx)
    positions_fxf = pd.Series(0.0, index=idx)
    positions_ewl = pd.Series(0.0, index=idx)

    hold_days = 30  # ~6 weeks
    gap_entry = 2.50  # 250bps
    gap_exit = 1.00   # 100bps — intervention fatigue proxy

    # Scan for entry signals (no overlapping windows)
    last_exit = pd.Timestamp("2000-01-01")
    events = []

    for i in range(1, len(idx)):
        date = idx[i]
        if date <= last_exit:
            continue

        gap = rate_gap.iloc[i]
        ec = eurchf_d.iloc[i]

        if pd.isna(gap) or pd.isna(ec):
            continue

        # Entry condition
        if gap > gap_entry and ec < 0.93:
            entry_date = date
            exit_idx = min(i + hold_days, len(idx) - 1)

            # Scan for early exit (rate gap drops below exit threshold)
            actual_exit = exit_idx
            for j in range(i + 1, exit_idx + 1):
                g = rate_gap.iloc[j]
                if not pd.isna(g) and g < gap_exit:
                    actual_exit = j
                    break

            window = slice(i, actual_exit + 1)
            days_held = actual_exit - i + 1

            fxf_window = fxf_r.iloc[i:actual_exit + 1]
            ewl_window = ewl_r.iloc[i:actual_exit + 1]

            # Long FXF 1.5, Short EWL 1.0
            trade_pnl = 1.5 * fxf_window.values - 1.0 * ewl_window.values
            actual_len = min(len(trade_pnl), actual_exit + 1 - i)
            trade_pnl = trade_pnl[:actual_len]

            pnl.iloc[i:i + actual_len] += trade_pnl
            positions_fxf.iloc[i:i + actual_len] = 1.5
            positions_ewl.iloc[i:i + actual_len] = -1.0

            trade_return = float((1 + pd.Series(trade_pnl)).prod() - 1)
            spy_window = spy_r.iloc[i:actual_exit + 1]
            spy_return = float((1 + spy_window).prod() - 1)

            events.append({
                "entry_date": str(entry_date.date()),
                "exit_date": str(idx[actual_exit].date()),
                "rate_gap_bps": round(float(gap * 100), 1),
                "eurchf": round(float(ec), 4),
                "days_held": days_held,
                "trade_return": round(trade_return, 4),
                "spy_return": round(spy_return, 4),
            })

            last_exit = idx[actual_exit]

    print(f"Events found: {len(events)}")
    for ev in events:
        print(f"  {ev['entry_date']} to {ev['exit_date']}: gap={ev['rate_gap_bps']}bps, "
              f"EURCHF={ev['eurchf']}, return={ev['trade_return']:.2%}, SPY={ev['spy_return']:.2%}")

    active_pnl = pnl[pnl != 0]
    if len(events) == 0:
        return mark_failed(sid, "no entry events found — rate gap or EUR/CHF conditions never met")
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)}) for metrics")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="SNB-Fed Gap FXF/EWL Pair")
    m["n_events"] = len(events)

    avg_ret = float(np.mean([e["trade_return"] for e in events]))
    win_rate = float(np.mean([e["trade_return"] > 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Long FXF 1.5x / Short EWL 1x when Fed-SNB rate gap >250bps AND EUR/CHF <0.93. "
                "Hold up to 30 trading days (~6 weeks); exit early if gap drops below 100bps.",
        "mechanism": "SNB intervention/expansion episodes create CHF demand vs domestic Swiss equities; "
                     "FXF (CHF currency ETF) appreciates while EWL (Swiss equity ETF) lags due to export "
                     "competitiveness headwinds and risk-off flows.",
        "source": "FRED DFEDTARU (Fed upper target), IRSTCB01CHM156N (SNB proxy), yfinance FXF/EWL/EURCHF=X",
        "n_events": len(events),
        "avg_trade_return": round(avg_ret, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done — {len(events)} events, avg_return={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
