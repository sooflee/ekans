"""
G-1 GPR Threats (Caldara-Iacoviello) defense drift.

Rule:
- GPRT = daily GPRD_THREAT series.
- Compute trailing 252-day (~12m) mean and std on calendar (i.e., index of daily series).
- Trigger when GPRT closes > 1.5 * std above the 12m mean for 5 consecutive days.
- On trigger date (last of the 5), go long an equal-weight basket of {LMT, RTX, NOC, GD, LHX, ITA} at
  next close (next trading day's close, since GPRT is daily including weekends; we map to next
  available trading day).
- Hold up to 60 trading days OR until GPRT mean-reverts back to within 0.0 sigma of the 12m mean
  (i.e., GPRT[t] <= mean12m[t]).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed
from _gpr import load_gpr


def main():
    try:
        gpr = load_gpr()
    except Exception as e:
        return mark_failed("G1_gprt_defense", f"GPR load failed: {e}")

    gprt = gpr["GPRD_THREAT"].dropna()
    # 12m trailing on daily-calendar series
    m12 = gprt.rolling(365, min_periods=200).mean()
    s12 = gprt.rolling(365, min_periods=200).std()
    z = (gprt - m12) / s12
    trig_day = (z > 1.5)
    # 5 consecutive days
    five_consec = trig_day & trig_day.shift(1).fillna(False) & trig_day.shift(2).fillna(False) \
                  & trig_day.shift(3).fillna(False) & trig_day.shift(4).fillna(False)
    trigger_dates = gprt.index[five_consec.values]
    if len(trigger_dates) == 0:
        return mark_failed("G1_gprt_defense", "no GPRT triggers")

    # Load defense basket
    tickers = ["LMT", "RTX", "NOC", "GD", "LHX", "ITA"]
    px = load_prices(tickers, start="2005-01-01")
    rets = px.pct_change()

    # Equal-weight basket return
    bsk_ret = rets.mean(axis=1)
    idx = bsk_ret.index

    pos = pd.Series(0.0, index=idx)
    held_periods = []
    last_exit_idx = -1
    # We'll just iterate trigger dates and for each, set pos=1 starting from next trading day,
    # exit if GPRT mean-reverts (gprt <= m12) on a trading day, or after 60 trading days.
    # Overlapping events: take union (max position == 1).

    triggers_used = 0
    gprt_aligned = gprt.reindex(idx, method="ffill")
    m12_aligned = m12.reindex(idx, method="ffill")

    for d in trigger_dates:
        # Find next trading day strictly after d
        ploc = idx.searchsorted(d, side="right")
        if ploc >= len(idx):
            continue
        entry_i = ploc
        # Walk forward
        end_i = min(entry_i + 60, len(idx) - 1)
        exit_i = end_i
        for j in range(entry_i, end_i + 1):
            if gprt_aligned.iloc[j] <= m12_aligned.iloc[j]:
                exit_i = j
                break
        pos.iloc[entry_i:exit_i + 1] = 1.0
        held_periods.append((idx[entry_i], idx[exit_i]))
        triggers_used += 1

    pnl = (pos.shift(1).fillna(0) * bsk_ret).dropna()
    spy = load_prices(["SPY"], start="2005-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="G1 GPRT defense drift")
    print_metrics(m)
    save_result("G1_gprt_defense", m, extra={
        "status": "ok",
        "rule": "GPRT > mean+1.5*sigma (rolling 12m) for 5 consecutive days -> long EW basket "
                "{LMT,RTX,NOC,GD,LHX,ITA}; exit on GPRT<=12m-mean or 60 trading days.",
        "universe": "Defense basket; ITA ETF + 5 names",
        "n_triggers": int(len(trigger_dates)),
        "n_triggers_used": triggers_used,
        "pct_days_long": float(pos.mean()),
        "source": "Caldara & Iacoviello GPR (FRBSF data)",
    })


if __name__ == "__main__":
    main()
