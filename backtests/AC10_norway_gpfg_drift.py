"""
AC-10 Norway GPFG Equity-Allocation Drift → Short MSCI World (URTH).

Rule: When reported equity allocation exceeds threshold (post-2017 strategic
band = 70%; "above by 2pp" → 72%), short URTH for ~65 trading days (≈ next
quarterly report).

Hand-coded NBIM quarterly equity allocation (%):
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


ALLOC = [
    ("2020Q1", "2020-04-30", 65.3),
    ("2020Q2", "2020-08-19", 71.8),
    ("2020Q3", "2020-10-28", 72.6),
    ("2020Q4", "2021-02-25", 72.8),
    ("2021Q1", "2021-04-29", 73.1),
    ("2021Q2", "2021-08-18", 72.4),
    ("2021Q3", "2021-10-27", 72.0),
    ("2021Q4", "2022-02-23", 72.0),
    ("2022Q1", "2022-04-28", 70.9),
    ("2022Q2", "2022-08-17", 68.5),
    ("2022Q3", "2022-10-26", 69.2),
    ("2022Q4", "2023-02-22", 69.8),
    ("2023Q1", "2023-04-27", 70.9),
    ("2023Q2", "2023-08-16", 71.3),
    ("2023Q3", "2023-10-25", 70.6),
    ("2023Q4", "2024-02-28", 70.9),
    ("2024Q1", "2024-04-25", 72.1),
    ("2024Q2", "2024-08-21", 71.4),
    ("2024Q3", "2024-10-23", 71.4),
    ("2024Q4", "2025-02-26", 71.4),
    ("2025Q1", "2025-04-23", 70.6),
]

THRESHOLD = 72.0
HOLD_DAYS = 65


def main():
    try:
        px = load_prices(["URTH", "ACWI"], start="2018-01-01")
    except Exception as e:
        return mark_failed("AC-10", f"data load failed: {e}")

    s = None
    if "URTH" in px.columns and px["URTH"].dropna().shape[0] >= 300:
        s = px["URTH"].dropna()
        ticker = "URTH"
    elif "ACWI" in px.columns and px["ACWI"].dropna().shape[0] >= 300:
        s = px["ACWI"].dropna()
        ticker = "ACWI"
    else:
        return mark_failed("AC-10", "neither URTH nor ACWI returned usable data")

    rets = s.pct_change()
    print(f"Using {ticker} (n={len(s)})")

    rows = []
    pnl_daily = pd.Series(0.0, index=s.index)
    bench_daily = pd.Series(0.0, index=s.index)

    for q, release_date, alloc in ALLOC:
        ts = pd.Timestamp(release_date)
        # Enter within 5 days of release; use first trading day on/after release+1
        i = s.index.searchsorted(ts, side="right")
        if i + HOLD_DAYS >= len(s):
            print(f"{q}: insufficient forward data; skip")
            continue
        win = s.index[i : i + HOLD_DAYS]
        evt_ret_long = float(s.loc[win[-1]] / s.loc[win[0]] - 1)
        # We are SHORTING when triggered
        short_ret = -evt_ret_long
        triggered = alloc > THRESHOLD
        rows.append({
            "quarter": q,
            "release_date": release_date,
            "alloc_pct": alloc,
            "triggered": triggered,
            "long_ret": evt_ret_long,
            "short_ret": short_ret,
        })
        if triggered:
            # short position daily series = negative of URTH daily ret
            r = rets.reindex(win).fillna(0.0)
            pnl_daily.loc[win] = -r.values
            bench_daily.loc[win] = r.values  # URTH buy-and-hold

    df = pd.DataFrame(rows)
    print("\nEvent table:")
    print(df.to_string(index=False))

    triggered = df[df["triggered"]]["short_ret"].values
    uncond = df["short_ret"].values
    print(f"\nTriggered (alloc > {THRESHOLD}%): N={len(triggered)}, "
          f"mean_short_ret={np.mean(triggered)*100 if len(triggered) else 0:.2f}%")
    print(f"Unconditional: N={len(uncond)}, "
          f"mean_short_ret={np.mean(uncond)*100:.2f}%")

    n_trig = len(triggered)
    if n_trig < 3:
        return mark_failed(
            "AC-10",
            f"only {n_trig} quarters triggered (alloc > {THRESHOLD}%); insufficient sample"
        )

    n = n_trig
    use = triggered
    mean_evt = float(np.mean(use))
    std_evt = float(np.std(use, ddof=1)) if n > 1 else 0.0
    t_evt = float(mean_evt / (std_evt / np.sqrt(n))) if std_evt > 0 else 0.0

    in_pos = (pnl_daily != 0.0)
    pnl_e = pnl_daily[in_pos]
    bench_e = bench_daily[in_pos]
    m = compute_metrics(pnl_e, benchmark=bench_e,
                        name="AC-10 GPFG drift short URTH")
    print_metrics(m)

    save_result("AC-10", m, extra={
        "status": "ok",
        "rule": f"Short URTH for {HOLD_DAYS} trading days after NBIM quarterly "
                f"release if reported equity alloc > {THRESHOLD}% (strategic 70% + 2pp).",
        "mechanism": "$1.5T+ AUM mechanical rebalancing acts as slow-moving DM headwind.",
        "source": "NBIM quarterly reports (hand-coded alloc); yfinance URTH.",
        "threshold_pct": THRESHOLD,
        "hold_trading_days": HOLD_DAYS,
        "ticker_used": ticker,
        "n_events_total": int(len(df)),
        "n_events_triggered": int(n_trig),
        "mean_event_ret": mean_evt,
        "t_stat_event": t_evt,
        "events": df.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
