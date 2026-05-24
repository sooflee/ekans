"""
F13 "Buy-the-dip" anti-signal (AQR)
SPY monthly DCA ($1 per month) vs SPY monthly DCA conditioned on "wait until SPY is
10% below its 12-month high before deploying."
Compare terminal wealth and Sharpe over 1990-present.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result,
)


def main():
    # Use ^GSPC for the full 1990 window since SPY only started 1993
    px = load_prices(["^GSPC"], start="1990-01-01").iloc[:, 0].rename("SPX")
    rets = px.pct_change().fillna(0)

    # Monthly investment dates: first trading day of each month
    monthly_idx = px.resample("MS").first().index
    # map each monthly date to first available trading day in px on/after that date
    contrib_dates = []
    for d in monthly_idx:
        loc = px.index.searchsorted(d, side="left")
        if loc < len(px.index):
            contrib_dates.append(px.index[loc])
    contrib_dates = pd.DatetimeIndex(contrib_dates).unique()

    # Strategy 1: invest $1 every month
    shares_dca = 0.0
    cash_dca = 0.0
    invested_dca = 0.0
    nav_dca = pd.Series(index=px.index, dtype=float)
    for d in px.index:
        if d in contrib_dates:
            shares_dca += 1.0 / px.loc[d]
            invested_dca += 1.0
        nav_dca.loc[d] = shares_dca * px.loc[d]

    # Strategy 2: hold cash until SPY is 10% below trailing 12-month high
    high_12m = px.rolling(252).max()
    drawdown = px / high_12m - 1.0

    shares_dip = 0.0
    cash_dip = 0.0
    invested_dip = 0.0
    nav_dip = pd.Series(index=px.index, dtype=float)
    for d in px.index:
        if d in contrib_dates:
            cash_dip += 1.0
            invested_dip += 1.0
        # deploy any cash if drawdown <= -10%
        dd = drawdown.loc[d]
        if not np.isnan(dd) and dd <= -0.10 and cash_dip > 0:
            shares_dip += cash_dip / px.loc[d]
            cash_dip = 0.0
        nav_dip.loc[d] = shares_dip * px.loc[d] + cash_dip

    # Terminal wealth
    term_dca = nav_dca.iloc[-1]
    term_dip = nav_dip.iloc[-1]
    inv = invested_dca

    # Daily returns of each strategy NAV (treat contributions as inflows by computing
    # returns of NAV minus contributions)
    # Build a clean return series by tracking growth-of-invested basis: use weighted IRR via
    # treating NAV like a unit-trust where contributions don't count as returns.
    # Method: define daily PnL = (NAV_t - NAV_{t-1} - contrib_t) / NAV_{t-1}
    def strat_returns(nav, contrib_dates):
        contrib = pd.Series(0.0, index=nav.index)
        contrib.loc[contrib.index.intersection(contrib_dates)] = 1.0
        prev = nav.shift(1).replace(0, np.nan)
        r = (nav - nav.shift(1) - contrib) / prev
        return r.fillna(0)

    r_dca = strat_returns(nav_dca, contrib_dates)
    r_dip = strat_returns(nav_dip, contrib_dates)

    m_dca = compute_metrics(r_dca.replace([np.inf, -np.inf], 0).dropna(),
                            benchmark=rets, name="F13 DCA every month")
    m_dip = compute_metrics(r_dip.replace([np.inf, -np.inf], 0).dropna(),
                            benchmark=rets, name="F13 Wait-for-10%-dip DCA")
    print_metrics(m_dca)
    print_metrics(m_dip)

    extra = {
        "status": "ok",
        "rule": "DCA $1/mo into SPY vs hoard cash until SPY 10% below 12m high.",
        "terminal_dca": float(term_dca),
        "terminal_dip": float(term_dip),
        "total_invested": float(inv),
        "wealth_diff_dca_minus_dip": float(term_dca - term_dip),
        "variant_dip": m_dip,
        "data_source": "yfinance ^GSPC",
        "source": "AQR Asness 'Buy-the-Dip' note",
    }
    save_result("F13_buy_the_dip", m_dca, extra=extra)


if __name__ == "__main__":
    main()
