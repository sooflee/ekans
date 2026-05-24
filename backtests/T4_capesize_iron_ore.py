"""
T4 Baltic Capesize index spike -> iron ore proxy.

Rule: When BCI (Baltic Capesize Index) WoW return > +35%, long iron-ore proxy.
Hold: 4 weeks.

Data substitutions:
  - BCI is not free daily; FRED has no `BALTIC` series; Baltic Exchange
    distribution is subscription-only.
  - BDRY (Breakwave Dry Bulk Shipping ETF) is the closest free public proxy
    on yfinance (started 2018-03-22). It tracks a basket of FFA rates
    weighted Cape/Pana/Supra, so it is correlated but not identical to BCI.
  - SGX FEF iron-ore future not on yfinance; we use VALE as proxy.

Signal mechanic with BDRY:
  Weekly resampled BDRY return (W-FRI). When wk return > +20% (relaxed
  from +35% because BDRY blends three vessel classes, smoothing extremes),
  go long VALE on Monday open for 4 weeks (~20 trading days).

We also try a stricter +35% to be faithful to the original rule.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def run_threshold(bdry_w_ret, vale, vale_r, threshold, hold=20):
    sig = bdry_w_ret > threshold
    pos = pd.Series(0.0, index=vale.index)
    for d in sig[sig].index:
        i = vale.index.searchsorted(d) + 1
        if i >= len(vale.index):
            continue
        end = min(i + hold, len(vale.index))
        pos.iloc[i:end] = 1.0
    pnl = (pos.shift(1) * vale_r).dropna()
    bench = vale_r.reindex(pnl.index)
    return compute_metrics(pnl, benchmark=bench,
                            name=f"T4 BDRY WoW > {threshold:.0%} -> long VALE 4w"), int(sig.sum())


def main():
    try:
        bdry = load_prices(["BDRY"], start="2018-03-22").iloc[:, 0].rename("BDRY")
        vale = load_prices(["VALE"], start="2018-03-22").iloc[:, 0].rename("VALE")
    except Exception as e:
        return mark_failed("T4_capesize_iron_ore", f"price load: {e}")

    if bdry.empty or vale.empty:
        return mark_failed("T4_capesize_iron_ore", "BDRY or VALE empty")

    bdry_w = bdry.resample("W-FRI").last()
    bdry_w_ret = bdry_w.pct_change()
    vale_r = vale.pct_change()

    m_relaxed, n_rel = run_threshold(bdry_w_ret, vale, vale_r, 0.20)
    print(f"--- Relaxed +20% threshold ({n_rel} triggers) ---")
    print_metrics(m_relaxed)
    m_strict, n_str = run_threshold(bdry_w_ret, vale, vale_r, 0.35)
    print(f"--- Strict +35% threshold ({n_str} triggers) ---")
    print_metrics(m_strict)

    # Save the +20% version as the primary; record +35% in extras.
    save_result("T4_capesize_iron_ore", m_relaxed, extra={
        "status": "ok",
        "rule": ("Weekly BDRY return > +20% (relaxed proxy for BCI > +35%); "
                 "long VALE next Monday for 4 weeks."),
        "mechanism": "Dry-bulk freight spike => iron-ore exporters bid; spillover trade.",
        "source": "yfinance BDRY (proxy for Baltic Cape), VALE (proxy for SGX FEF)",
        "n_triggers_relaxed_20pct": n_rel,
        "n_triggers_strict_35pct": n_str,
        "metrics_strict_35pct": m_strict,
        "caveats": ("BDRY launched 2018-03-22, so sample is short. BDRY is a Cape+Pana+Supra "
                    "blend, smoothing pure-Cape spikes. VALE substituted for SGX FEF."),
    })


if __name__ == "__main__":
    main()
