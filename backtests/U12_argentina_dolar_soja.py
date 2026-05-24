"""
U12 Argentina 'dolar soja' export-incentive windows -> long ZM=F + BG (rebound leg).

Rule: For each Argentina 'dolar soja' / 'dolar agro' window (preferential
FX rate for soy exporters), enter the trade 60 calendar days AFTER the
window CLOSES, long CBOT soy meal (ZM=F) + Bunge (BG) equally weighted for
40 trading days (~8 weeks). Non-overlapping.

Mechanism: Dolar-soja windows pull forward soybean sales from Argentine
farmers, depressing prices during the window. After the window closes,
Argentina liquidates fewer beans and the global crush margin tightens -
ZM rallies and Bunge captures crush spread improvement.

Source: Curated from Reuters / La Nacion / BCRA press releases.
Hardcoded window-close dates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated 'dolar soja' / 'dolar agro' window-close dates.
WINDOWS = [
    ("2022-09-30", "Dolar soja 1 closes (ARS 200/USD soy)"),
    ("2022-12-30", "Dolar soja 2 closes"),
    ("2023-05-31", "Dolar soja 3 / 'dolar agro' (Massa) closes"),
    ("2023-08-23", "Dolar soja 4 closes (Massa pre-PASO)"),
    ("2023-11-17", "Dolar exportador 50/50 split closes (Caputo pre-takeover)"),
]


def main():
    try:
        zm = load_prices(["ZM=F"], start="2015-01-01").iloc[:, 0].dropna()
        bg = load_prices(["BG"], start="2010-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U12_argentina_dolar_soja", f"yfinance load failed: {e}")

    if len(zm) < 200 or len(bg) < 200:
        return mark_failed("U12_argentina_dolar_soja", "insufficient ZM/BG history")

    px = pd.concat({"ZM": zm, "BG": bg}, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 40
    LAG = 60  # calendar days after window close
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    n_events = 0
    last_end = None
    used = []
    for d_str, lbl in WINDOWS:
        close = pd.Timestamp(d_str)
        entry = close + pd.Timedelta(days=LAG)
        nxt = rets.index[rets.index >= entry]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + HOLD, len(rets.index))
        pos.iloc[idx:end_idx, :] = 0.5
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U12_argentina_dolar_soja", "no events landed")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.mean(axis=1).reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U12 dolar-soja close + {LAG}d -> long ZM+BG {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U12_argentina_dolar_soja", m, extra={
        "status": "ok",
        "rule": "60 calendar days after each Argentina 'dolar soja' window closes, long an equal-weight basket of ZM=F + BG for 40 trading days (~8 weeks); non-overlapping (rebound leg).",
        "mechanism": "Dolar-soja windows pull forward Argentine soy sales, depressing prices during the window; after closure, lower Argentine liquidations tighten global crush, lifting ZM and Bunge.",
        "source": "Curated window-close dates from Reuters / La Nacion / BCRA press releases. ZM=F + BG via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~5); single-political-cycle phenomenon.",
    })


if __name__ == "__main__":
    main()
