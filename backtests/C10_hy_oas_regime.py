"""
C10 HY OAS regime
FRED BAMLH0A0HYM2 (HY OAS) and BAMLC0A0CM (IG OAS).
When HY OAS rises 100bps from trailing 6-month low AND breaches 500bps, go to cash.
Re-risk when it rolls over from local peak by 100bps.
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
    try:
        hy = load_fred("BAMLH0A0HYM2", start="1996-12-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("C10_hy_oas_regime", f"FRED load failed: {e}")

    # NOTE: FRED's free CSV/datareader endpoint truncates this series to the most
    # recent ~3 years; full history requires a FRED API key. Backtest is therefore
    # limited to 2023-present.
    hy = hy.astype(float)
    start_overlap = max(hy.index.min(), pd.Timestamp("1996-12-01"))
    spy = load_prices(["SPY"], start=str(start_overlap.date())).iloc[:, 0].rename("SPY")
    df = pd.concat([hy.rename("HY"), spy], axis=1).dropna()
    rets = df["SPY"].pct_change()

    rolling_min_6m = df["HY"].rolling(126).min()      # ~6 months
    # local peak: rolling max since last regime change; track in state machine
    pos = pd.Series(1.0, index=df.index)  # default long
    state = 1  # 1 long, 0 cash
    peak = df["HY"].iloc[0]
    for i in range(len(df)):
        oas = df["HY"].iloc[i]
        lo = rolling_min_6m.iloc[i]
        if state == 1:
            if not np.isnan(lo) and (oas - lo) >= 1.0 and oas >= 5.0:
                state = 0
                peak = oas
        else:
            if oas > peak:
                peak = oas
            if (peak - oas) >= 1.0:
                state = 1
        pos.iloc[i] = state

    pnl = (pos.shift(1) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets, name="C10 HY OAS regime")
    print_metrics(m)
    save_result("C10_hy_oas_regime", m, extra={
        "status": "ok",
        "rule": "Cash when HY OAS rises 100bps from 6m low AND > 500bps; re-risk when it rolls 100bps off peak.",
        "data_source": "FRED BAMLH0A0HYM2",
    })


if __name__ == "__main__":
    main()
