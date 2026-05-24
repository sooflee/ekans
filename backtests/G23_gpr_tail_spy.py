"""
G-23 GPR tail-spike -> contrarian BUY SPY (expansion-conditioned).

Rule:
- GPR = daily GPRD.
- On any day t, if GPRD[t] > mean36m + 3 * std36m AND USREC (FRED) = 0 at month containing t
  (expansion regime) -> long SPY at next close.
- Exit when (a) GPRD within 1 sigma of 36m mean OR (b) 60 trading days after entry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed
from _gpr import load_gpr


def main():
    try:
        gpr = load_gpr()
    except Exception as e:
        return mark_failed("G23_gpr_tail_spy", f"GPR load failed: {e}")
    gprd = gpr["GPRD"].dropna()
    # 36m on daily calendar = ~1095 days
    m36 = gprd.rolling(1095, min_periods=400).mean()
    s36 = gprd.rolling(1095, min_periods=400).std()
    z = (gprd - m36) / s36
    trig = z > 3.0

    try:
        usrec = load_fred(["USREC"], start="1990-01-01")["USREC"]
    except Exception as e:
        return mark_failed("G23_gpr_tail_spy", f"FRED USREC load failed: {e}")
    usrec_daily = usrec.reindex(gprd.index, method="ffill").fillna(0)

    spy = load_prices(["SPY"], start="1995-01-01")["SPY"]
    rets = spy.pct_change()
    idx = rets.index

    gprd_aligned = gprd.reindex(idx, method="ffill")
    m36_aligned = m36.reindex(idx, method="ffill")
    s36_aligned = s36.reindex(idx, method="ffill")
    usrec_aligned = usrec_daily.reindex(idx, method="ffill").fillna(0)
    trig_aligned = trig.reindex(idx, method="ffill").fillna(False)

    pos = pd.Series(0.0, index=idx)
    n_trig = 0
    last_exit = -1
    i = 0
    while i < len(idx):
        if trig_aligned.iloc[i] and usrec_aligned.iloc[i] == 0 and i > last_exit:
            entry = i + 1
            if entry >= len(idx):
                break
            end = min(entry + 60, len(idx) - 1)
            exit_i = end
            for j in range(entry, end + 1):
                # Within 1 sigma of 36m mean
                m = m36_aligned.iloc[j]
                s = s36_aligned.iloc[j]
                g = gprd_aligned.iloc[j]
                if abs(g - m) <= s:
                    exit_i = j
                    break
            pos.iloc[entry:exit_i + 1] = 1.0
            n_trig += 1
            last_exit = exit_i
            i = exit_i + 1
        else:
            i += 1

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets.dropna(), name="G23 GPR tail BUY SPY")
    print_metrics(m)
    save_result("G23_gpr_tail_spy", m, extra={
        "status": "ok",
        "rule": "GPRD > 36m_mean + 3*sigma AND USREC=0 -> long SPY next close; exit when GPR within "
                "1 sigma of 36m mean OR 60 trading days.",
        "universe": "SPY",
        "n_triggers": n_trig,
        "pct_days_long": float(pos.mean()),
        "source": "Caldara & Iacoviello GPR; contrarian tail",
    })


if __name__ == "__main__":
    main()
