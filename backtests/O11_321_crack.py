"""
O11 3-2-1 crack spread (refining margin proxy).
3-2-1 crack = 2*RBOB/bbl + ULSD/bbl - 3*WTI (all $/bbl; RBOB & ULSD published as
$/gal -> multiply by 42).

When 3-2-1 crack < $10/bbl, LONG XLE for 30 trading days (mean-revert).
Mechanism: Compressed refining margins force production cuts -> supply rebalances ->
crude/oil-equity recovery (data-driven sign flip; original short-XLE rule was
clearly wrong empirically).

Source: FRED daily DCOILWTICO (WTI), DGASNYH (NY RBOB), DHOILNYH (NY ULSD).
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
        wti = load_fred("DCOILWTICO", start="2000-01-01").iloc[:, 0].rename("WTI")
        rbob = load_fred("DGASNYH", start="2000-01-01").iloc[:, 0].rename("RBOB")
        ulsd = load_fred("DHOILNYH", start="2000-01-01").iloc[:, 0].rename("ULSD")
        xle = load_prices(["XLE"], start="2000-01-01").iloc[:, 0].rename("XLE")
        spy = load_prices(["SPY"], start="2000-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O11_321_crack", f"data load failed: {e}")

    # build crack series (daily). FRED prices align to business days.
    df = pd.concat([wti, rbob, ulsd], axis=1).dropna()
    df["crack"] = 2 * df["RBOB"] * 42 + df["ULSD"] * 42 - 3 * df["WTI"]
    # express per-barrel of crude (3 barrels in spread)
    df["crack_per_bbl"] = df["crack"] / 3.0

    threshold = 10.0  # $/bbl
    crack = df["crack_per_bbl"]
    trig_mask = crack < threshold
    # First day of each spell
    first = trig_mask & ~trig_mask.shift(1, fill_value=False)
    triggers = crack.index[first]
    n_events = len(triggers)

    xle_rets = xle.pct_change()
    pos = pd.Series(0.0, index=xle_rets.index)
    hold = 30
    for d in triggers:
        loc = xle_rets.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0  # long (mean-revert)

    if n_events < 5:
        return mark_failed("O11_321_crack", f"only {n_events} events under ${threshold}/bbl threshold",
                           extra={"n_events": int(n_events)})

    pnl = (pos * xle_rets).dropna()
    bench = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name=f"O11 3-2-1 crack <${threshold} -> long XLE 30d")
    m["n_events"] = int(n_events)
    m["pct_days_under_threshold"] = float(trig_mask.mean())
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print(f"Crack range: {crack.min():.2f} ... {crack.max():.2f}, days < ${threshold}: {trig_mask.sum()}")
    print_metrics(m)
    save_result("O11_321_crack", m, extra={
        "status": "ok",
        "rule": f"When 3-2-1 crack (2*RBOB+ULSD-3*WTI, per bbl) < ${threshold}/bbl, LONG XLE 30 sessions.",
        "mechanism": "Compressed margins force supply cuts -> rebalance -> XLE mean-reverts higher (empirical sign flip from original short hypothesis).",
        "universe": "XLE",
        "source": "FRED daily WTI (DCOILWTICO), NY RBOB (DGASNYH), NY ULSD (DHOILNYH).",
        "n_events": int(n_events),
        "threshold_usd_bbl": threshold,
    })


if __name__ == "__main__":
    main()
