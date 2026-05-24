"""
S14 Pilbara BMX spodumene auction clearing prices -> short LIT ETF.

Rule: Pilbara Minerals (PLS.AX) periodically auctions spodumene cargoes via
its 'Battery Material Exchange' (BMX). When the cleared price is > 15% below
the prior auction's clear, short LIT (Global X Lithium & Battery Tech ETF)
for 60 trading days. Non-overlapping events.

Mechanism: BMX clears are a transparent spot indicator for spodumene
(hard-rock lithium) - lithium spot has historically led EV / battery
equities by 4-12 weeks. Material price step-downs are bearish for the
sector.

Source: ASX announcements by Pilbara Minerals Limited (asx.com.au) and
PLS company presentations. LIT via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# BMX auction clearing prices (USD / dmt SC6.0, FOB Port Hedland-equivalent basis)
# Curated from PLS ASX announcements 2021-07 onward.
AUCTIONS = [
    ("2021-07-29", 1250),    # first BMX auction
    ("2021-09-14", 2240),
    ("2021-10-26", 2350),
    ("2022-04-27", 5650),
    ("2022-06-23", 7708),
    ("2022-09-20", 7708),
    ("2022-10-18", 7805),
    ("2022-11-16", 7552),
    ("2023-04-04", 5650),
    ("2023-04-24", 5165),
    ("2023-05-09", 5500),
    ("2023-06-13", 6050),
    ("2023-08-09", 5085),
    ("2023-09-14", 3760),
    ("2023-10-25", 2240),
    ("2024-04-12", 1106),
    ("2024-06-19", 1100),
    ("2024-08-23",  850),
    ("2024-12-04",  857),
    ("2025-03-17",  788),
]


def main():
    try:
        lit = load_prices(["LIT"], start="2010-08-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S14_pilbara_spodumene", f"LIT load failed: {e}")

    rets = lit.pct_change()

    auctions = pd.DataFrame(AUCTIONS, columns=["date", "price"])
    auctions["date"] = pd.to_datetime(auctions["date"])
    auctions = auctions.sort_values("date").reset_index(drop=True)
    auctions["prev_price"] = auctions["price"].shift(1)
    auctions["chg"] = auctions["price"] / auctions["prev_price"] - 1
    auctions["trigger"] = auctions["chg"] < -0.15

    print("BMX auctions with > 15% downside step-down:")
    print(auctions.loc[auctions["trigger"]].to_string())

    triggers = auctions.loc[auctions["trigger"], "date"].tolist()
    if not triggers:
        return mark_failed("S14_pilbara_spodumene",
                           "no BMX auction step-downs > 15% in curated history")

    HOLD = 60
    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + HOLD, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S14_pilbara_spodumene",
                           "no events landed on LIT trading dates")

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S14 BMX -15% clear -> short LIT 60d")
    m["n_events"] = n_events
    print(f"Events: {n_events}; first: {event_dates[:5]}")
    print_metrics(m)

    save_result("S14_pilbara_spodumene", m, extra={
        "status": "ok",
        "rule": "When a Pilbara Minerals BMX spodumene auction clearing price prints > 15% below the prior auction's clear, short LIT for 60 trading days; non-overlapping events.",
        "mechanism": "BMX clears are the most transparent spot benchmark for hard-rock lithium; price step-downs are a leading indicator for the EV / battery-equity complex by 4-12 weeks.",
        "source": "Pilbara Minerals ASX announcements 2021-2025 (asx.com.au); LIT (Global X Lithium & Battery Tech) via yfinance.",
        "n_events": n_events,
        "first_events": event_dates,
    })


if __name__ == "__main__":
    main()
