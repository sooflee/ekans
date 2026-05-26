"""PL51_maturity_wall_long_vcsh — High-Rate + Inverted Curve → Long VCSH
When DGS10 > 4.5% AND DGS2 > DGS10 (inverted): long VCSH for 252 days.
Exit early if DGS10 drops below 3.5%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL51_maturity_wall_long_vcsh"
    try:
        fred = load_fred(["DGS10", "DGS2"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if fred.empty:
        return mark_failed(sid, "yield data empty")

    dgs10 = fred["DGS10"].dropna()
    dgs2 = fred["DGS2"].dropna()

    # Align
    common = dgs10.index.intersection(dgs2.index)
    dgs10 = dgs10.loc[common]
    dgs2 = dgs2.loc[common]

    # Signal: DGS10 > 4.5 AND DGS2 > DGS10
    signal = (dgs10 > 4.5) & (dgs2 > dgs10)

    # Find first day signal fires in each regime
    trigger_dates = []
    was_off = True
    for i in range(len(signal)):
        if signal.iloc[i] and was_off:
            trigger_dates.append(signal.index[i])
            was_off = False
        elif not signal.iloc[i]:
            was_off = True

    print(f"High-rate inverted curve regimes: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: DGS10={dgs10.loc[d]:.2f}%, DGS2={dgs2.loc[d]:.2f}%")

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no high-rate inverted curve periods found")

    try:
        px = load_prices(["VCSH", "LQD", "SPY"], start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    ret = daily_returns(px)
    if "VCSH" not in ret.columns:
        return mark_failed(sid, "VCSH data not available")

    vcsh_ret = ret["VCSH"]
    spy_ret = ret["SPY"]
    hold_days = 252

    pnl_series = pd.Series(0.0, index=vcsh_ret.index)
    event_results = []

    for td in trigger_dates:
        entry_mask = vcsh_ret.index >= td
        if entry_mask.sum() < hold_days:
            # Use whatever is available
            if entry_mask.sum() < 30:
                continue
        entry_idx = vcsh_ret.index[entry_mask][0]
        pos = vcsh_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(vcsh_ret))

        # Early exit if DGS10 drops below 3.5
        for j in range(pos, end_pos):
            check_date = vcsh_ret.index[j]
            if check_date in dgs10.index and dgs10.loc[check_date] < 3.5:
                end_pos = j
                break

        if end_pos - pos < 30:
            continue

        event_rets = vcsh_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values

        lqd_cumret = None
        if "LQD" in ret.columns:
            lqd_event = ret["LQD"].iloc[pos:end_pos]
            lqd_cumret = float((1 + lqd_event).prod() - 1)

        spy_cumret = None
        spy_event = spy_ret.iloc[pos:end_pos]
        spy_cumret = float((1 + spy_event).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "dgs10": round(float(dgs10.loc[td]), 2),
            "dgs2": round(float(dgs2.loc[td]), 2),
            "hold_days": end_pos - pos,
            "vcsh_return": round(cumret, 4),
            "lqd_return": round(lqd_cumret, 4) if lqd_cumret is not None else None,
            "spy_return": round(spy_cumret, 4),
        })

    if len(event_results) == 0:
        return mark_failed(sid, "no valid events (VCSH starts 2009; need high-rate+inversion post-2009)")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="Maturity Wall → Long VCSH")
    rets_arr = [e["vcsh_return"] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long VCSH for 252 days when DGS10 > 4.5% AND curve inverted (DGS2 > DGS10)",
        "mechanism": "High-rate inverted curve → short-duration IG earns carry with less duration risk",
        "source": "FRED DGS10, DGS2; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
