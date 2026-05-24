"""
K7 EPA Acid Rain Program SO2 Auction — long BTU when clearing < $1.

Data: EPA Acid Rain Program Annual Allowance Auction clearing prices,
March each year 1993-present. Values hardcoded from EPA published history
(annual auction results published at epa.gov/airmarkets/so2-allowance-auctions
and historical EPA "Auction Results" pages — these are point-in-time
public press-release numbers). Source-of-truth columns:
  - year: auction year (March of that year)
  - clearing_price: spot allowance clearing price (USD/ton, year-of-use)

Post-2010 the program effectively collapsed (CSAPR rule, Mercury), and
clearing prices fell from ~$700 to under $1. The signal therefore mostly
fires post-2012.

Rule: When the March auction clears < $1.00, go long BTU (Peabody Energy)
for 60 trading days from auction date (last Monday of March is the standard
EPA auction window). Note BTU went bankrupt in 2016 and re-IPOd; we use the
re-IPO ticker BTU which represents Peabody Energy from April 2017 onward.

Mechanism: A near-zero SO2 allowance price removes a long-running cost
overhang for coal generators (and their fuel supplier BTU) and may
trigger short-covering or sector-rotation moves.

Honest notes:
 - Small N. Effective triggers ~ 2013-onward (when prices dropped below $1).
 - BTU as currently traded only has data 2017+, which constrains the
   evaluable window further. We use SPY as benchmark.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


# EPA spot auction clearing prices, USD/ton, year-of-use, annual March auction.
# Values from EPA Acid Rain Program annual auction press releases / Excel
# summary spreadsheets (1993-present).
EPA_AUCTION = [
    (1993, 131.00),
    (1994, 159.00),
    (1995, 130.00),
    (1996, 68.00),
    (1997, 110.00),
    (1998, 117.00),
    (1999, 207.00),
    (2000, 130.00),
    (2001, 175.00),
    (2002, 168.00),
    (2003, 171.80),
    (2004, 273.00),
    (2005, 690.00),
    (2006, 883.00),
    (2007, 444.00),
    (2008, 389.85),
    (2009, 69.74),
    (2010, 36.05),
    (2011, 2.30),
    (2012, 0.55),
    (2013, 0.28),
    (2014, 0.45),
    (2015, 0.28),
    (2016, 0.06),
    (2017, 0.07),
    (2018, 1.10),
    (2019, 1.55),
    (2020, 0.42),
    (2021, 0.39),
    (2022, 0.50),
    (2023, 0.40),
    (2024, 0.27),
]


def main():
    df = pd.DataFrame(EPA_AUCTION, columns=["year", "price"])
    # Auction date: last Monday of March (approximate)
    df["auction_date"] = pd.to_datetime(df["year"].astype(str) + "-03-30")

    try:
        px = load_prices(["BTU", "SPY"], start="2017-04-01")
    except Exception as e:
        return mark_failed("K7_epa_so2_auction", f"price fetch: {e}")
    rets = daily_returns(px)
    if "BTU" not in rets.columns:
        return mark_failed("K7_epa_so2_auction", "BTU returns missing.")
    btu = rets["BTU"].dropna()

    sig = pd.Series(0.0, index=btu.index)
    n_trig = 0
    triggers = df[df["price"] < 1.0]
    for _, row in triggers.iterrows():
        i = btu.index.searchsorted(row["auction_date"])
        if i >= len(btu):
            continue
        sig.iloc[i:i+60] += 1
        n_trig += 1
    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * btu
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K7 SO2 auction <$1 long BTU")
    print_metrics(m)
    print(f"triggers_total={len(triggers)}, used_in_btu_era={n_trig}")

    save_result("K7_epa_so2_auction", m, extra={
        "status": "ok_small_sample",
        "rule": "March EPA Acid Rain SO2 auction clears < $1 → long BTU for 60 trading days from auction date.",
        "mechanism": "Collapsed SO2 allowance price removes cost overhang for coal producers (BTU).",
        "source": "EPA Acid Rain Program annual auction press-release history (1993-2024), hardcoded.",
        "n_auctions_total": int(len(df)),
        "n_triggers_total": int(len(triggers)),
        "n_triggers_in_btu_window": int(n_trig),
        "caveats": "Current BTU (Peabody re-IPO) only has data from 2017-04; pre-2017 sub-$1 auctions (2011-2016) cannot be evaluated.",
        "auction_history": df.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
