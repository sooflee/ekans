"""
N4 SIA worldwide semi billings 3MMA-YoY MoM acceleration.
SIA monthly press releases (semiconductors.org) not free as machine-readable time series.
Substitute proxy: Census Manufacturers' New Orders: Computers and Electronic Products
(FRED A36SNO, monthly, available 1992+). Compute 3MMA-YoY MoM acceleration; long SOXX
60 trading days when accel > +2pp MoM. Documented substitution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    try:
        # A36SNO: Manufacturers' New Orders: Computers and Electronic Products ($M)
        ord_ = load_fred("A36SNO", start="2000-01-01").iloc[:, 0].rename("orders")
        soxx = load_prices(["SOXX"], start="2001-07-01").iloc[:, 0].rename("SOXX")
    except Exception as e:
        return mark_failed("N4_sia_billings", f"data load failed: {e}")

    # 3-month moving average
    ma3 = ord_.rolling(3).mean()
    # YoY %
    yoy = ma3.pct_change(12) * 100.0
    # MoM acceleration in YoY
    accel = yoy.diff()  # change in YoY pct points

    soxx_rets = soxx.pct_change()
    pos = pd.Series(0.0, index=soxx_rets.index)

    triggers = []
    for d, a in accel.dropna().items():
        if a > 2.0:
            # the orders print typically lands mid-month for prior month; trade next day
            # Use a 4-business-day lag to be safe
            entry = d + pd.tseries.offsets.BDay(4)
            loc = soxx_rets.index.searchsorted(entry)
            for k in range(0, 60):
                if loc + k < len(pos):
                    pos.iloc[loc + k] = 1.0
            triggers.append((d.date().isoformat(), float(a)))

    if not triggers:
        return mark_failed("N4_sia_billings", "no MoM accel > +2pp triggers",
                           extra={"data_substitution": "A36SNO proxy"})

    pnl_full = (pos * soxx_rets).dropna()
    pnl = pnl_full.loc[pnl_full.ne(0).cummax()]
    bench = soxx_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="N4 SIA proxy (A36SNO accel) -> SOXX")
    m["n_triggers"] = len(triggers)
    print(f"Triggers: {len(triggers)}")
    print_metrics(m)
    save_result("N4_sia_billings", m, extra={
        "status": "ok",
        "rule": "Long SOXX 60 sessions when 3MMA YoY of computers-and-electronics new orders accelerates > +2pp MoM.",
        "mechanism": "Order acceleration leads chip-revenue cycle by ~3-6 months -> semi equities re-rate",
        "universe": "SOXX",
        "source": "Census M3 (FRED A36SNO) as proxy for SIA Worldwide Semi Billings",
        "data_substitution": "SIA WSTS monthly billings paywalled / press-release only; using A36SNO (Census M3 New Orders: Computer & Electronics) as a public proxy. Spec rule preserved (3MMA-YoY MoM accel > +2pp).",
        "triggers_sample": triggers[:8],
    })


if __name__ == "__main__":
    main()
