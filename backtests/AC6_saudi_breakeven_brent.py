"""
AC-6 Saudi Budget Implicit Breakeven → Brent Bull.

Rule:
  - Saudi fiscal breakeven oil price (IMF Article IV hand-coded):
      2018:$83  2019:$86  2020:$76  2021:$77  2022:$80
      2023:$80  2024:$96  2025:$96
  - On Jan 2 each year, compute (breakeven − Brent spot). If > $12, long Brent
    (BZ=F) from Jan 2 through Mar 31 (≈ next OPEC+ JMMC window).
  - Build annual event-return series.

Mechanism: Saudi fiscal program is dominant constraint on OPEC+ supply policy.
When new-year budget assumes oil materially above market, MbS has fiscal &
political incentive to engineer cuts/extensions.

Source: IMF Article IV (hand-coded breakevens), yfinance BZ=F.
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


BREAKEVENS = {
    2018: 83.0,
    2019: 86.0,
    2020: 76.0,
    2021: 77.0,
    2022: 80.0,
    2023: 80.0,
    2024: 96.0,
    2025: 96.0,
    2026: 96.0,  # placeholder; treat as same as 2025 absent new IMF report
}

GAP_THRESHOLD = 12.0  # USD


def main():
    try:
        px = load_prices(["BZ=F"], start="2017-06-01")
    except Exception as e:
        return mark_failed("AC-6", f"data load failed: {e}")
    s = px["BZ=F"].dropna()
    if len(s) < 200:
        return mark_failed("AC-6", "Brent series too short")

    rows = []
    pnl_daily = pd.Series(0.0, index=s.index)
    bench_daily = pd.Series(0.0, index=s.index)  # benchmark = BZ=F buy-and-hold

    for y in sorted(BREAKEVENS.keys()):
        if y < 2018 or y > 2026:
            continue
        # find first trading day on/after Jan 2
        start_ts = pd.Timestamp(f"{y}-01-02")
        end_ts = pd.Timestamp(f"{y}-03-31")
        win_dates = s.index[(s.index >= start_ts) & (s.index <= end_ts)]
        if len(win_dates) < 5:
            continue
        spot_jan2 = float(s.loc[win_dates[0]])
        gap = BREAKEVENS[y] - spot_jan2
        triggered = gap > GAP_THRESHOLD
        evt_ret = float(s.loc[win_dates[-1]] / s.loc[win_dates[0]] - 1)
        rows.append({
            "year": y,
            "breakeven": BREAKEVENS[y],
            "jan2_spot": spot_jan2,
            "gap": gap,
            "triggered": triggered,
            "window_ret": evt_ret,
            "start": win_dates[0].date(),
            "end": win_dates[-1].date(),
        })
        if triggered:
            daily = s.reindex(win_dates).pct_change().fillna(0.0)
            pnl_daily.loc[win_dates] = daily.values
            bench_daily.loc[win_dates] = daily.values

    df = pd.DataFrame(rows)
    print("\nEvent table:")
    print(df.to_string(index=False))

    triggered_rets = df[df["triggered"]]["window_ret"].values
    n_trig = len(triggered_rets)
    print(f"\nTriggered events (gap > ${GAP_THRESHOLD}): N={n_trig}")
    if n_trig > 0:
        print(f"  mean_ret={np.mean(triggered_rets)*100:.2f}%, "
              f"hit={np.mean(triggered_rets > 0)*100:.1f}%")

    uncond = df["window_ret"].values
    print(f"Unconditional all years: N={len(uncond)}, "
          f"mean={np.mean(uncond)*100:.2f}%, hit={np.mean(uncond > 0)*100:.1f}%")

    if n_trig < 3:
        # still save with honest small-N caveat
        use = triggered_rets if n_trig > 0 else uncond
        cut = "triggered" if n_trig > 0 else "unconditional_fallback"
    else:
        use = triggered_rets
        cut = "triggered"

    if len(use) == 0:
        return mark_failed("AC-6", "no events to evaluate")

    n = len(use)
    mean_evt = float(np.mean(use))
    std_evt = float(np.std(use, ddof=1)) if n > 1 else 0.0
    t_evt = float(mean_evt / (std_evt / np.sqrt(n))) if std_evt > 0 else 0.0

    in_pos = (pnl_daily != 0.0)
    pnl_e = pnl_daily[in_pos]
    bench_e = bench_daily[in_pos]
    if len(pnl_e) >= 30 and n_trig >= 3:
        m = compute_metrics(pnl_e, benchmark=bench_e,
                            name="AC-6 Saudi breakeven Brent bull")
    else:
        # event-level summary (very small N)
        compound = float(np.prod(1.0 + use) - 1.0)
        years_span = max(1.0, n)  # 1 event/year
        m = {
            "name": "AC-6 Saudi breakeven Brent bull",
            "start": str(df["start"].min()),
            "end": str(df["end"].max()),
            "n_days": int(len(pnl_e)),
            "n_events": int(n),
            "cagr": float((1 + compound) ** (1 / years_span) - 1),
            "ann_vol": float(std_evt * np.sqrt(4.0)),  # ~Q1 window ≈ 1/4 year
            "sharpe": float(mean_evt / std_evt * np.sqrt(4.0)) if std_evt > 0 else 0.0,
            "max_dd": float(np.min(use)) if len(use) else 0.0,
            "calmar": None,
            "hit_rate": float(np.mean(use > 0)) if len(use) else 0.0,
            "t_stat": t_evt,
        }
    print_metrics(m)

    save_result("AC-6", m, extra={
        "status": "ok",
        "rule": "Long Brent (BZ=F) Jan 2 → Mar 31 when (Saudi breakeven − Jan2 spot) > $12.",
        "mechanism": "Saudi fiscal pressure → OPEC+ supply discipline / cuts.",
        "source": "Hand-coded IMF Article IV breakevens; yfinance BZ=F.",
        "gap_threshold_usd": GAP_THRESHOLD,
        "cut": cut,
        "n_events_triggered": int(n_trig),
        "n_events_total": int(len(df)),
        "mean_triggered_ret": float(np.mean(triggered_rets)) if n_trig else None,
        "mean_unconditional_ret": float(np.mean(uncond)),
        "t_stat_event": t_evt,
        "events": df.assign(start=df["start"].astype(str), end=df["end"].astype(str))
                     .to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
