"""
G01 Google Trends "recession"
pytrends weekly worldwide "recession" trend; rule: 4-week MA z > 1.5sigma above 5y mean ->
long SPY 6 months.
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
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return mark_failed("G01_google_trends_recession",
                          "pytrends not installed (and rate-limit risk); cite Preis-Moat-Stanley 2013")

    try:
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        pytrends.build_payload(["recession"], cat=0, timeframe="all", geo="", gprop="")
        df = pytrends.interest_over_time()
        if df.empty:
            raise RuntimeError("pytrends empty payload")
    except Exception as e:
        return mark_failed("G01_google_trends_recession",
                          f"pytrends rate-limited or failed; cite Preis-Moat-Stanley 2013. ({e})")

    s = df["recession"].astype(float)
    s.index = pd.to_datetime(s.index)

    # Google Trends 'all' timeframe returns monthly resolution. Adapt windows:
    # 4-period MA (rule of 4 weeks -> 4 months when monthly), 5y trailing = 60 months.
    ma4 = s.rolling(4).mean()
    mean5y = s.rolling(60).mean()
    std5y = s.rolling(60).std()
    z = (ma4 - mean5y) / std5y

    trig = z > 1.5

    spy = load_prices(["SPY"], start="2004-01-01").iloc[:, 0].rename("SPY")
    rets = spy.pct_change()

    pos_monthly = pd.Series(0.0, index=s.index)
    rem = 0
    for i in range(len(s)):
        if bool(trig.iloc[i]):
            rem = 6  # 6 months
        if rem > 0:
            pos_monthly.iloc[i] = 1.0
            rem -= 1

    daily_pos = pos_monthly.reindex(spy.index, method="ffill").fillna(0.0)
    pnl = (daily_pos.shift(1) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets, name="G01 Google Trends recession")
    print_metrics(m)

    n_trig = int(trig.fillna(False).sum())
    save_result("G01_google_trends_recession", m, extra={
        "status": "ok",
        "rule": "4-month MA Google Trends 'recession' z>1.5 over 5y trailing -> long SPY 6 months.",
        "data_note": "pytrends 'all' timeframe returns monthly; rule adapted from weekly to monthly grid.",
        "n_triggers": n_trig,
        "source": "Preis-Moat-Stanley 2013",
    })


if __name__ == "__main__":
    main()
