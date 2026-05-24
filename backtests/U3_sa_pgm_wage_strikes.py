"""
U3 South Africa LRA Section 64 strike notices at PGM majors -> long platinum + SBSW.

Rule: When a South African platinum-belt union (AMCU / NUM) files a Section
64 strike notice at Amplats / Implats / Sibanye-Stillwater, long PL=F
(platinum futures) and SBSW (Sibanye) equally weighted for 30 trading days
(~6 weeks). Non-overlapping.

Mechanism: South Africa produces ~70% of mined platinum; Section 64 notices
precede strikes that historically halt 30-50% of global Pt supply. Even when
strikes are short / settled quickly, the option-on-disruption is bid up.

Source: Curated from press archives (Reuters / Bloomberg / Miningmx) for
Section 64 filings 2014-2024. PL=F and SBSW via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated SA platinum-belt Section 64 strike-notice / strike-start events.
EVENTS = [
    ("2014-01-23", "AMCU 5-month strike begins (Amplats/Implats/Lonmin)"),
    ("2014-06-24", "AMCU strike settlement"),
    ("2018-11-21", "Sibanye gold-wing strike begins (spillover)"),
    ("2019-04-17", "Sibanye gold strike ends; PGM division wage talks"),
    ("2021-03-09", "AMCU files Section 64 at Sibanye PGM"),
    ("2022-03-09", "Sibanye gold strike begins; PGM contagion bid"),
    ("2022-09-19", "Implats 3-month wage strike threat"),
    ("2024-03-04", "Implats Section 64 (Rustenburg)"),
]


def main():
    try:
        pl = load_prices(["PL=F"], start="2010-01-01").iloc[:, 0].dropna()
        sbsw = load_prices(["SBSW"], start="2018-01-01").iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed("U3_sa_pgm_wage_strikes", f"yfinance load failed: {e}")

    if len(pl) < 200 or len(sbsw) < 200:
        return mark_failed("U3_sa_pgm_wage_strikes", "insufficient PL=F or SBSW history")

    cols = {"PL": pl, "SBSW": sbsw}
    px = pd.concat(cols, axis=1).sort_index()
    rets = px.pct_change()

    HOLD = 30
    pos = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    n_events = 0
    last_end = None
    used = []
    for d_str, lbl in EVENTS:
        d = pd.Timestamp(d_str)
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + HOLD, len(rets.index))
        # SBSW only valid post-listing; PL=F always valid
        for col_i, col in enumerate(rets.columns):
            col_rets = rets[col].iloc[idx:end_idx]
            if col_rets.notna().sum() > 5:
                pos.iloc[idx:end_idx, col_i] = 0.5
        last_end = rets.index[end_idx - 1]
        n_events += 1
        used.append((str(start.date()), lbl))

    if n_events == 0:
        return mark_failed("U3_sa_pgm_wage_strikes", "no events landed in price history")

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets["PL"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench,
                        name=f"U3 SA PGM Section 64 -> long PL+SBSW {HOLD}d")
    m["n_events"] = n_events
    print(f"Events used: {n_events}")
    print_metrics(m)

    save_result("U3_sa_pgm_wage_strikes", m, extra={
        "status": "ok",
        "rule": "When a South African platinum-belt union (AMCU/NUM) files a Section 64 strike notice or begins a strike at Amplats/Implats/Sibanye-Stillwater, long equal-weight PL=F + SBSW for 30 trading days (~6 weeks); non-overlapping.",
        "mechanism": "South Africa produces ~70% of mined platinum; Section 64 notices precede strikes that historically halt 30-50% of global Pt supply. The option-on-disruption is bid up at filing.",
        "source": "Curated from press archives (Reuters, Bloomberg, Miningmx) for SA Section 64 filings 2014-2024. PL=F and SBSW via yfinance.",
        "n_events": n_events,
        "events": used,
        "small_sample_warning": "N small (~6-8); covers two distinct regimes (pre-2018 strikes, post-2020 PGM rallies).",
    })


if __name__ == "__main__":
    main()
