"""
S15 Urals-Brent discount widens -> long BZ=F / short CL=F.

Rule: Russian Urals crude trades at a discount to Brent post-2022 sanctions.
When the Urals-Brent discount widens past $20/bbl (Urals < Brent - $20),
long BZ=F and short CL=F for 60 trading days; non-overlapping events.

Curated monthly Urals price levels from Russia MinFin official tax-base
disclosures (Russian Ministry of Finance: 'Average price of Urals crude').
Brent benchmark levels from monthly averages of BZ=F.

Mechanism: A blown-out Urals discount signals stress on Russian export
logistics (shadow fleet, sanctions enforcement) and a relative scarcity of
Brent-quality (non-Russian) waterborne crude. Brent has historically
outperformed WTI during these dislocation periods because Brent represents
the marginal seaborne barrel for European buyers losing Urals access.

Source: Russia Ministry of Finance press releases (minfin.gov.ru) -
monthly Urals average prices. Brent monthly avg from BZ=F. yfinance BZ=F, CL=F.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Russia MinFin reported monthly Urals average ($/bbl)
# (publication date ~ first week of NEXT month, used as signal date for tradeability)
URALS_MONTHLY = [
    # publish_date  urals_avg_$/bbl   month_label
    ("2022-02-04", 84.07, "2022-01"),
    ("2022-03-04", 88.94, "2022-02"),  # pre-invasion
    ("2022-04-04", 78.81, "2022-03"),  # post-invasion shock
    ("2022-05-05", 70.52, "2022-04"),
    ("2022-06-03", 78.81, "2022-05"),
    ("2022-07-04", 87.25, "2022-06"),
    ("2022-08-04", 78.41, "2022-07"),
    ("2022-09-02", 83.16, "2022-08"),
    ("2022-10-04", 70.62, "2022-09"),
    ("2022-11-03", 70.62, "2022-10"),
    ("2022-12-02", 66.47, "2022-11"),
    ("2023-01-13", 50.47, "2022-12"),  # Dec 5 EU price cap kicks in
    ("2023-02-15", 49.48, "2023-01"),  # discount blowout
    ("2023-03-15", 49.56, "2023-02"),  # discount blowout
    ("2023-04-14", 47.85, "2023-03"),  # discount blowout
    ("2023-05-12", 51.15, "2023-04"),  # discount blowout
    ("2023-06-12", 53.34, "2023-05"),
    ("2023-07-13", 55.28, "2023-06"),
    ("2023-08-14", 64.37, "2023-07"),
    ("2023-09-13", 74.00, "2023-08"),
    ("2023-10-13", 83.08, "2023-09"),
    ("2023-11-13", 81.52, "2023-10"),
    ("2023-12-13", 72.84, "2023-11"),
    ("2024-01-12", 64.23, "2023-12"),
    ("2024-02-15", 65.59, "2024-01"),
    ("2024-03-15", 69.42, "2024-02"),
    ("2024-04-15", 70.34, "2024-03"),
    ("2024-05-14", 74.98, "2024-04"),
    ("2024-06-14", 67.27, "2024-05"),
    ("2024-07-12", 70.27, "2024-06"),
    ("2024-08-13", 70.34, "2024-07"),
    ("2024-09-13", 70.27, "2024-08"),
    ("2024-10-14", 63.56, "2024-09"),
    ("2024-11-14", 64.72, "2024-10"),
    ("2024-12-13", 63.42, "2024-11"),
    ("2025-01-15", 69.69, "2024-12"),
    ("2025-02-14", 67.66, "2025-01"),
    ("2025-03-14", 61.99, "2025-02"),
    ("2025-04-14", 58.99, "2025-03"),
    ("2025-05-14", 54.76, "2025-04"),  # discount blowout after April Trump tariffs
]


def main():
    try:
        bz = load_prices(["BZ=F"], start="2020-01-01").iloc[:, 0]
        cl = load_prices(["CL=F"], start="2020-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S15_urals_discount", f"yfinance load failed: {e}")

    px = pd.concat({"bz": bz, "cl": cl}, axis=1).dropna()
    bz_ret = px["bz"].pct_change()
    cl_ret = px["cl"].pct_change()
    spread_ret = bz_ret - cl_ret  # long BZ, short CL

    # Build Urals monthly series and compare to Brent monthly avg
    urals = pd.DataFrame(URALS_MONTHLY, columns=["pub_date", "urals", "month_label"])
    urals["pub_date"] = pd.to_datetime(urals["pub_date"])
    urals["urals"] = pd.to_numeric(urals["urals"])
    # Compute Brent monthly avg matching the month label
    px["mo"] = px.index.to_period("M")
    brent_monthly = px.groupby("mo")["bz"].mean()
    urals["brent_avg"] = urals["month_label"].apply(
        lambda s: brent_monthly.get(pd.Period(s, "M"), np.nan)
    )
    urals = urals.dropna(subset=["brent_avg"]).copy()
    urals["discount"] = urals["urals"] - urals["brent_avg"]
    urals["disc_blowout"] = urals["discount"] < -20.0

    print("Urals monthly (after Brent merge):")
    print(urals[["pub_date", "month_label", "urals", "brent_avg", "discount"]].to_string())

    triggers = urals.loc[urals["disc_blowout"], "pub_date"].tolist()
    print(f"\nTriggers with discount < -$20: {len(triggers)}")

    HOLD = 60
    pos = pd.Series(0.0, index=spread_ret.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d in triggers:
        nxt = spread_ret.index[spread_ret.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = spread_ret.index.get_loc(start)
        end_idx = min(idx + HOLD, len(spread_ret.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = spread_ret.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("S15_urals_discount",
                           "no Urals-Brent discount < -$20 events with Brent overlap")

    pnl = (pos.shift(1) * spread_ret).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    # Benchmark: long BZ buy-and-hold
    bench = bz_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="S15 Urals disc < -$20 -> long BZ, short CL 60d")
    m["n_events"] = n_events
    print_metrics(m)

    save_result("S15_urals_discount", m, extra={
        "status": "ok",
        "rule": "When the MinFin-reported monthly Urals average crude price minus same-month Brent average is < -$20/bbl, long BZ=F and short CL=F (1:1) for 60 trading days; non-overlapping events.",
        "mechanism": "A blown-out Urals discount reflects sanctions / shadow-fleet stress restricting Russian export realisation. Brent represents the marginal seaborne replacement barrel for European buyers losing Urals access, so Brent has historically outperformed WTI during dislocation episodes.",
        "source": "Russia Ministry of Finance press releases (minfin.gov.ru) for monthly Urals average; Brent monthly avg from BZ=F; CL=F via yfinance.",
        "n_events": n_events,
        "first_events": event_dates[:8],
    })


if __name__ == "__main__":
    main()
