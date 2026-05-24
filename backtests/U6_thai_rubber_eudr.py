"""
U6 EUDR deadline window for Thai rubber -> long Bridgestone (5108.T) / Sri Trang (STGT.BK).

Rule: 90 trading days BEFORE each EUDR effective-date milestone (Dec 30 2025
for large operators per Reg 2023/1115 + 2024 postponement), long Bridgestone
(5108.T) and STGT.BK equally weighted for 60 trading days. Non-overlapping.

Mechanism: EUDR forces deforestation-free traceability for natural rubber
into the EU. Thai supply (Sri Trang, Thai Hua) holds DDS-compliant inventory;
compliant inventory commands a premium ahead of cutover. Bridgestone passes
input cost; STGT directly benefits from compliant-premium spread.

Source: EU Regulation 2023/1115 (EUDR), Regulation 2024/3234 (postponement).
Curated deadline anchor dates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated EUDR-related milestone anchor dates.
# Strategy entry = anchor minus ~90 trading days.
ANCHORS = [
    ("2024-12-30", "Original EUDR effective date (postponed)"),
    ("2025-12-30", "Postponed EUDR effective date for large operators"),
]


def main():
    try:
        bs = load_prices(["5108.T"], start="2018-01-01").iloc[:, 0].dropna()
        stgt = load_prices(["STGT.BK"], start="2020-07-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U6_thai_rubber_eudr", f"yfinance load failed: {e}")

    if len(bs) < 200 or len(stgt) < 100:
        return mark_failed("U6_thai_rubber_eudr", "insufficient 5108.T or STGT.BK history")

    px = pd.concat({"5108": bs, "STGT": stgt}, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 60
    LEAD = 90
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    n_events = 0
    last_end = None
    used = []
    for d_str, lbl in ANCHORS:
        anchor = pd.Timestamp(d_str)
        if anchor < rets.index[0] or anchor > rets.index[-1]:
            # if anchor in future, still try - we can backtest the entry window only
            pass
        # find the trading day closest to anchor - LEAD
        target = anchor - pd.tseries.offsets.BDay(LEAD)
        nxt = rets.index[rets.index >= target]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + HOLD, len(rets.index))
        if end_idx - idx < 20:
            continue
        pos.iloc[idx:end_idx, :] = 0.5
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U6_thai_rubber_eudr", "no EUDR window landed in price history")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U6 EUDR -T-90d -> long 5108+STGT {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U6_thai_rubber_eudr", m, extra={
        "status": "ok",
        "rule": "Starting 90 trading days before each EUDR effective-date milestone (Dec 30 2024 original, Dec 30 2025 postponed), long Bridgestone (5108.T) + STGT.BK 50/50 for 60 trading days; non-overlapping.",
        "mechanism": "EUDR forces deforestation-free traceability for natural-rubber imports into the EU; compliant Thai inventory (Sri Trang, Thai Hua) carries a premium ahead of cutover. Bridgestone passes through, STGT captures the compliant-premium spread.",
        "source": "EU Regulation 2023/1115 + Regulation 2024/3234 postponement. Prices via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N very small (1-2); single-cycle deadline-driven event.",
    })


if __name__ == "__main__":
    main()
