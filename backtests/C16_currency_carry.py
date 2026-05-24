"""
C16 Currency carry
Simplest G4-vs-USD carry. Use FRED 3M interbank rates IR3TIB01{EZ,JP,GB,CH}M156N (monthly)
versus IR3TIB01USM156N. Trade currency exposure via ETFs FXE, FXY, FXB, FXF.
Monthly: long the (long-USD or long-foreign) side with the higher 3M rate; size to 1 unit gross.
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

CCY_MAP = {
    "EUR": ("FXE", "IR3TIB01EZM156N"),
    "JPY": ("FXY", "IR3TIB01JPM156N"),
    "GBP": ("FXB", "IR3TIB01GBM156N"),
    "CHF": ("FXF", "IR3TIB01CHM156N"),
}


def main():
    try:
        usd_rate = load_fred("IR3TIB01USM156N", start="2005-01-01").iloc[:, 0].rename("USD")
        rate_frames = {ccy: load_fred(code, start="2005-01-01").iloc[:, 0].rename(ccy)
                       for ccy, (_, code) in CCY_MAP.items()}
        etf_tickers = [tup[0] for tup in CCY_MAP.values()]
        etfs = load_prices(etf_tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed("C16_currency_carry", f"data load failed: {e}")

    rates = pd.concat([usd_rate] + list(rate_frames.values()), axis=1)
    rates.columns = ["USD"] + list(rate_frames.keys())
    rates = rates.sort_index().ffill()

    # Build daily forward-filled rates aligned to ETF dates
    rates_d = rates.reindex(etfs.index, method="ffill")
    px = etfs.dropna()
    rates_d = rates_d.reindex(px.index, method="ffill").dropna()
    px = px.loc[rates_d.index]

    # For each currency: differential = foreign - USD. Positive => long foreign ETF, negative => short.
    monthly_idx = px.resample("ME").last().index
    pos_m = pd.DataFrame(0.0, index=monthly_idx, columns=etf_tickers)

    for d in monthly_idx:
        if d not in rates_d.index:
            d_use = rates_d.index[rates_d.index <= d]
            if len(d_use) == 0:
                continue
            d_use = d_use[-1]
        else:
            d_use = d
        row = rates_d.loc[d_use]
        if row.isna().any():
            continue
        diffs = {}
        for ccy, (etf, _) in CCY_MAP.items():
            diffs[etf] = row[ccy] - row["USD"]
        # Equal weight 1/N long-short by sign
        signs = {etf: np.sign(d) for etf, d in diffs.items()}
        gross = sum(abs(s) for s in signs.values())
        if gross == 0:
            continue
        for etf, s in signs.items():
            pos_m.loc[d, etf] = s / gross  # weight by 1/N

    pos_d = pos_m.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    pnl = (pos_d * rets).sum(axis=1).dropna()

    bench = rets["FXE"].reindex(pnl.index)  # not a clean benchmark; just for context
    m = compute_metrics(pnl, benchmark=bench, name="C16 Currency carry (G4 vs USD)")
    print_metrics(m)
    save_result("C16_currency_carry", m, extra={
        "status": "ok",
        "rule": "Monthly: per currency, long ETF if foreign 3M rate > USD 3M rate, short if lower. Equal weight; total gross 1.",
        "universe": "FXE, FXY, FXB, FXF",
        "source": "Lustig-Verdelhan (2007); simplified G4 carry via FRED 3M interbank rates",
        "caveat": "FRED foreign 3M interbank series are monthly and lag (~1mo); not transactable signals.",
    })


if __name__ == "__main__":
    main()
