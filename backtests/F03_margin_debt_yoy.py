"""
F03 Margin debt YoY change
FRED series BOGZ1FL663067003Q (Brokers/Dealers Margin Loans) or fallbacks.
Rule: monthly YoY > -20% -> reduce exposure 6 months; YoY back through 0 -> re-enter.
Interpretation: when YoY < -20% (deep credit contraction), go flat for 6 months;
when YoY crosses back above 0%, re-enter long.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    series_candidates = ["BOGZ1FL663067003Q", "BOGZ1FL663067005Q"]
    md = None
    src = None
    for s in series_candidates:
        try:
            df = load_fred(s, start="1980-01-01")
            md = df.iloc[:, 0].dropna()
            if len(md) > 20:
                src = s
                break
        except Exception:
            continue

    if md is None:
        return mark_failed("F03_margin_debt_yoy", "no FRED margin-debt series accessible")

    # resample to monthly (forward-fill quarterly to month-end)
    md_m = md.resample("M").last().ffill()
    yoy = md_m.pct_change(12)

    spy = load_prices(["SPY"], start="1995-01-01").iloc[:, 0].rename("SPY")
    rets = spy.pct_change()

    # state machine on monthly grid
    months = yoy.dropna().index
    state = 1  # default long
    flat_remaining = 0
    pos_m = pd.Series(0.0, index=months)
    for i, m_end in enumerate(months):
        y = yoy.loc[m_end]
        if state == 1:
            if y < -0.20:
                state = 0
                flat_remaining = 6
        else:
            # flat
            flat_remaining -= 1
            if y > 0:
                state = 1
            elif flat_remaining <= 0:
                state = 1
        pos_m.iloc[i] = state

    daily_pos = pos_m.reindex(spy.index, method="ffill").fillna(1.0)  # default long pre-history
    pnl = (daily_pos.shift(1) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets, name="F03 Margin debt YoY")
    print_metrics(m)
    save_result("F03_margin_debt_yoy", m, extra={
        "status": "ok",
        "rule": "Margin-debt YoY < -20% -> flat 6 months; re-enter when YoY > 0.",
        "data_source": f"FRED {src}",
    })


if __name__ == "__main__":
    main()
