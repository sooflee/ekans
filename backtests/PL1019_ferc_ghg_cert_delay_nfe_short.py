"""PL1019_ferc_ghg_cert_delay_nfe_short — FERC GHG Certificate Weighing — Short NFE / Long XLE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1019_ferc_ghg_cert_delay_nfe_short"

    # Key event: FERC PL18-1-000 policy issuance March 24, 2022
    # Entry: T+2 = March 28, 2022
    # Exit: 18 months = September 28, 2023 (or earlier on reversal)
    # Additional secondary events within the same policy window
    events_def = [
        # (entry_date, exit_date, note)
        ("2022-03-28", "2023-09-28", "FERC PL18-1-000 primary entry"),
        ("2022-11-01", "2024-05-01", "FERC delays on NFE pipeline certs"),
        ("2023-02-16", "2024-08-16", "FERC additional GHG certificate denial"),
    ]

    try:
        px = load_prices(["NFE", "XLE", "SPY"], start="2022-01-01")
    except Exception as e:
        try:
            px = load_prices(["NFE", "XLE", "SPY"], start="2022-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    ret = daily_returns(px)

    if "NFE" not in ret.columns:
        return mark_failed(sid, "NFE not in price data")
    if "XLE" not in ret.columns:
        return mark_failed(sid, "XLE not in price data")

    nfe_r = ret["NFE"]
    xle_r = ret["XLE"]
    spy_r = ret["SPY"]

    # Build pairs trade PnL: short NFE + long XLE (equal notional)
    # Daily PnL = -NFE_return + XLE_return
    combined_pnl = pd.Series(0.0, index=spy_r.index)
    events_summary = []
    active_windows = []

    for (entry_str, exit_str, note) in events_def:
        entry_dt = pd.Timestamp(entry_str)
        exit_dt = pd.Timestamp(exit_str)

        # Find entry trading day
        entry_mask = nfe_r.index >= entry_dt
        if not entry_mask.any():
            print(f"  SKIP {entry_str}: no trading days after entry")
            continue
        entry_idx = nfe_r.index[entry_mask][0]

        # Find exit trading day
        exit_mask = (nfe_r.index > entry_idx) & (nfe_r.index <= exit_dt)
        if not exit_mask.any():
            print(f"  SKIP {entry_str}: no trading days in window")
            continue

        window_nfe = nfe_r[exit_mask]
        window_xle = xle_r.reindex(window_nfe.index).fillna(0.0)
        window_spy = spy_r.reindex(window_nfe.index).fillna(0.0)

        # Pairs PnL: short NFE, long XLE
        daily_pair = -window_nfe + window_xle

        for idx, val in daily_pair.items():
            if idx not in [i for w in active_windows for i in w]:
                combined_pnl[idx] += val

        active_windows.append(list(window_nfe.index))

        total_nfe = float((1 + window_nfe).prod() - 1)
        total_xle = float((1 + window_xle).prod() - 1)
        total_spy = float((1 + window_spy).prod() - 1)
        pair_ret = -total_nfe + total_xle
        n_days = len(window_nfe)

        print(f"  {entry_str} -> {exit_dt.date()}: n={n_days}, "
              f"NFE={total_nfe:.3f}, XLE={total_xle:.3f}, pair={pair_ret:.3f}, note={note}")

        events_summary.append({
            "entry_date": entry_str,
            "exit_date": str(exit_dt.date()),
            "note": note,
            "n_days_held": n_days,
            "nfe_return": round(total_nfe, 4),
            "xle_return": round(total_xle, 4),
            "spy_return": round(total_spy, 4),
            "pair_return": round(pair_ret, 4),
        })

    if not events_summary:
        return mark_failed(sid, "no valid events found")

    active_pnl = combined_pnl[combined_pnl != 0.0]
    print(f"\nActive trading days: {len(active_pnl)}")
    print(f"Events: {len(events_summary)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FERC GHG Cert Delay → Short NFE / Long XLE")
    save_result(sid, m, extra={
        "rule": "Short NFE / Long XLE equal-notional within 5 days of FERC issuing GHG emissions weighing requirement for pipeline certificates; exit 18 months or on policy reversal",
        "mechanism": "FERC GHG weighing creates permitting uncertainty for LNG feed-gas pipelines, compressing NPV for highly-leveraged operators like NFE; XLE hedge isolates idiosyncratic regulatory risk",
        "source": "FERC eLibrary docket PL18-1-000; yfinance NFE, XLE, SPY",
        "n_events": len(events_summary),
        "avg_pair_return": round(float(np.mean([e["pair_return"] for e in events_summary])), 4),
        "win_rate": round(float(np.mean([e["pair_return"] > 0 for e in events_summary])), 4),
        "caveats": "Only 1 primary + 2 secondary events in history; NFE decline driven by multiple factors beyond FERC (over-leverage, cost overruns); treat as qualitative hypothesis with limited statistical power",
        "events": events_summary,
    })
    print(f"Done: {len(events_summary)} events")


if __name__ == "__main__":
    main()
