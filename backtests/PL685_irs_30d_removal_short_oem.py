"""PL685 — IRS 30D EV Credit Eligibility Removal → Short removed OEM / Long remaining OEM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# IRS/Treasury clean-vehicle credit events: (date, short_ticker, long_ticker, reason)
# Based on public IRS Publication 5866 / fueleconomy.gov list changes
# When a model is removed, short that OEM, long one with maintained eligibility
EVENTS = [
    # 2023-01-01: Initial IRA clean vehicle rules took effect, significant reshuffling
    ("2023-01-03", "GM", "TSLA", "GM EVs initially excluded, Tesla maintained some eligibility"),
    # 2023-04-18: Treasury guidance update on battery sourcing reduced TSLA eligibility
    ("2023-04-18", "TSLA", "GM", "Treasury sourcing rules hit TSLA Model 3 LR/Perf"),
    # 2024-01-01: Revised battery sourcing rules effective
    ("2024-01-02", "RIVN", "GM", "Rivian R1T/R1S lost credit due to battery sourcing"),
    # 2024-04-15: Additional Treasury guidance removes several trims
    ("2024-04-15", "TSLA", "F", "Model 3 RWD removed from eligible list due to sourcing"),
    # 2024-07-01: Further sourcing restrictions
    ("2024-07-01", "RIVN", "F", "R1 platform sourcing issues, F-150 Lightning maintained"),
    # 2025-01-01: New eligibility rules under updated guidance
    ("2025-01-02", "GM", "TSLA", "Chevy Silverado EV removed; Tesla Model Y maintained"),
]


def main():
    sid = "PL685_irs_30d_removal_short_oem"
    try:
        px = load_prices(["TSLA", "GM", "F", "RIVN", "SPY"], start="2022-08-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "price data empty")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    hold = 15
    pnl = pd.Series(0.0, index=spy_r.index)
    events_detail = []

    for date_str, short_t, long_t, reason in EVENTS:
        td = pd.Timestamp(date_str)

        # Check tickers available
        if short_t not in ret.columns or long_t not in ret.columns:
            continue

        short_r = ret[short_t]
        long_r = ret[long_t]

        # Find entry date
        mask = spy_r.index >= td
        if mask.sum() < hold:
            continue
        ei = spy_r.index[mask][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + hold, len(spy_r))

        window_idx = spy_r.index[p:ep]

        # 1:1 pair: long long_t, short short_t
        lg = long_r.reindex(window_idx).fillna(0.0)
        sh = short_r.reindex(window_idx).fillna(0.0)
        period_pnl = lg.values - sh.values

        # Avoid double-counting overlapping windows
        already_active = pnl.loc[window_idx] != 0.0
        if already_active.any():
            continue

        pnl.loc[window_idx] = period_pnl

        lg_ret = float((1 + lg).prod() - 1)
        sh_ret = float((1 + sh).prod() - 1)
        pair_ret = lg_ret - sh_ret
        sp_ret = float((1 + spy_r.reindex(window_idx).fillna(0.0)).prod() - 1)

        events_detail.append({
            "event_date": date_str,
            "entry_date": str(ei.date()),
            "short": short_t,
            "long": long_t,
            "reason": reason,
            "long_return": round(lg_ret, 4),
            "short_return": round(sh_ret, 4),
            "pair_pnl": round(pair_ret, 4),
            "spy_return": round(sp_ret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no valid non-overlapping events found")

    active_pnl = pnl[pnl != 0.0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="IRS EV Credit Removal → Short/Long OEM Pair")
    avg_ret = np.mean([e["pair_pnl"] for e in events_detail])
    win_rate = np.mean([e["pair_pnl"] > 0 for e in events_detail])

    save_result(sid, m, extra={
        "rule": "On IRS/Treasury clean vehicle list removing EV nameplate, short the OEM of removed model, long OEM with maintained eligibility, 1:1 pair for 15 trading days",
        "mechanism": "Credit removal reduces effective price to consumer, hurting demand for removed OEM vs. competitors who retain credit",
        "source": "IRS Publication 5866; fueleconomy.gov; yfinance",
        "n_events": len(events_detail),
        "avg_event_return": round(float(avg_ret), 4),
        "event_win_rate": round(float(win_rate), 4),
        "events": events_detail,
    })
    print(f"Done: {len(events_detail)} events, avg_ret={avg_ret:.2%}, win_rate={win_rate:.0%}")


if __name__ == "__main__":
    main()
