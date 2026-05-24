"""
W3 Mag-7 Decrowding (hedge-fund crowding unwind).

Original signal: Goldman Sachs Prime Brokerage reports hedge-fund net Mag-7
exposure. When that exposure reaches an extreme (top decile), Mag-7 underperforms
in the subsequent weeks as positioning unwinds.

The GS Prime data is paywalled. We implement an equity-only proxy:
  - Crowding proxy = trailing-126d ratio of Mag-7 average return vs the
    equal-weight S&P 500 (RSP) return. When Mag-7 has run far ahead of the
    broad market (z-score > +1.5), treat as a crowded-long state.
  - Rule: when crowded, short Mag-7 basket / long RSP for next 21 trading days.

Mag-7 = AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA (current consensus list).

Honest caveat: this proxy is much weaker than GS-Prime positioning. It will
trigger most of 2023-2024 when Mag-7 dominated, which is exactly the period
we expect the unwind premium NOT to have materialised yet. Documenting honestly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    mag7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    universe = mag7 + ["RSP", "SPY"]
    try:
        px = load_prices(universe, start="2015-01-01")
    except Exception as e:
        return mark_failed("W3_mag7_decrowding", f"Price load failed: {e}")

    px = px.dropna()
    if px.empty or len(px) < 252:
        return mark_failed("W3_mag7_decrowding",
                           f"Insufficient overlap: {len(px)} rows")

    rets = px.pct_change()
    mag7_ret = rets[mag7].mean(axis=1)  # equal-weighted Mag-7 daily return
    rsp_ret = rets["RSP"]
    spy_ret = rets["SPY"]

    # Crowding proxy: trailing 126d compounded mag7 vs RSP, z-score
    win = 126
    rel = (mag7_ret - rsp_ret).rolling(win).sum()
    z = (rel - rel.rolling(252).mean()) / rel.rolling(252).std()

    # When z > +1.5 (Mag-7 strongly ahead): short Mag-7 / long RSP for 21d
    threshold = 1.5
    raw_sig = (z > threshold).astype(float)
    # Carry signal forward for 21 days after each trigger day
    hold = 21
    pos = raw_sig.rolling(hold, min_periods=1).max()  # 1 if any trigger in last 21d

    # Pair return: long RSP, short Mag-7
    pair_ret = (rsp_ret - mag7_ret)
    pnl = (pos.shift(1) * pair_ret).dropna()
    pnl = pnl[pnl.index >= rets.index[252 + win]]

    if len(pnl) < 50:
        return mark_failed("W3_mag7_decrowding",
                           f"Too few post-warmup days: {len(pnl)}")

    bench = spy_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="W3 Mag-7 decrowding proxy")
    m["trigger_days"] = int((raw_sig > 0).sum())
    m["active_days"] = int((pos > 0).sum())
    print_metrics(m)

    save_result("W3_mag7_decrowding", m, extra={
        "status": "ok_proxy",
        "rule": ("Crowding proxy = z-score of trailing-126d sum of (Mag-7 EW "
                 "return minus RSP return), using a 252d rolling mean/std. "
                 "When z > +1.5, short Mag-7 equal-weight basket vs long RSP "
                 "for 21 trading days. Pair PnL."),
        "mechanism": ("Hedge-fund crowding extremes (per GS Prime) precede "
                       "unwinds. Returns-based deviation of Mag-7 from the "
                       "broad index is a weak but free proxy."),
        "source": ("GS Prime Services Hedge Fund Trend Monitor (paywalled). "
                    "Proxy implemented here is a returns-based deviation only."),
        "universe": "Mag-7 (AAPL/MSFT/GOOGL/AMZN/META/NVDA/TSLA) vs RSP vs SPY",
        "caveats": ("This is a returns-based proxy, NOT the GS Prime positioning "
                     "signal. The proxy will frequently trigger throughout 2023-25 "
                     "when Mag-7 dominated; the unwind premium may show up only "
                     "in episodic windows (e.g. Aug 2024, Jan 2024)."),
    })


if __name__ == "__main__":
    main()
