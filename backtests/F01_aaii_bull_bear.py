"""
F01 AAII bull-bear extremes
Pull AAII sentiment.xls. (Bulls - Bears) < -20 for 2 wks -> long SPY 12 wks; > +30 for 3 wks -> flat 12 wks.
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


def fetch_aaii():
    import requests
    url = "https://www.aaii.com/files/surveys/sentiment.xls"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.aaii.com/sentimentsurvey/sent_results",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.content, url


def main():
    try:
        raw, src = fetch_aaii()
    except Exception as e:
        return mark_failed("F01_aaii_bull_bear",
                          f"AAII data behind protection or unreachable: {e}")

    # Header is row 3 (0-indexed), data starts row 5
    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name="SENTIMENT", header=3)
    except Exception as e:
        return mark_failed("F01_aaii_bull_bear", f"AAII xls parse failed: {e}")

    # Pick Date, Bullish, Bearish columns
    cols = [c for c in df.columns]
    # Best-effort: find them by exact match (post header=3)
    date_col = next((c for c in cols if "date" in str(c).lower()), None)
    bull_col = next((c for c in cols if "bull" in str(c).lower()), None)
    bear_col = next((c for c in cols if "bear" in str(c).lower()), None)
    if not (date_col and bull_col and bear_col):
        return mark_failed("F01_aaii_bull_bear",
                          f"AAII columns unrecognized: {cols[:10]}")

    df = df[[date_col, bull_col, bear_col]].copy()
    df.columns = ["Date", "Bull", "Bear"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Bull"] = pd.to_numeric(df["Bull"], errors="coerce")
    df["Bear"] = pd.to_numeric(df["Bear"], errors="coerce")
    df = df.dropna().sort_values("Date").set_index("Date")
    # values come in 0..1 form; convert to percent
    if df["Bull"].max() <= 1.5:
        df["Bull"] *= 100
        df["Bear"] *= 100

    spread = df["Bull"] - df["Bear"]

    bullish_extreme = (spread > 30).rolling(3).sum() == 3
    bearish_extreme = (spread < -20).rolling(2).sum() == 2

    spy = load_prices(["SPY"], start="1993-01-01").iloc[:, 0].rename("SPY")
    rets = spy.pct_change()

    weekly_pos = pd.Series(0.0, index=spread.index)
    rem_long = 0
    rem_flat = 0
    for i in range(len(spread)):
        if bearish_extreme.iloc[i]:
            rem_long = 12
        if bullish_extreme.iloc[i]:
            rem_flat = 12
        if rem_flat > 0:
            weekly_pos.iloc[i] = 0.0
            rem_flat -= 1
            if rem_long > 0:
                rem_long -= 1
        elif rem_long > 0:
            weekly_pos.iloc[i] = 1.0
            rem_long -= 1

    daily_pos = weekly_pos.reindex(spy.index, method="ffill").fillna(0.0)
    pnl = (daily_pos.shift(1) * rets).dropna()

    m = compute_metrics(pnl, benchmark=rets, name="F01 AAII bull-bear")
    print_metrics(m)
    save_result("F01_aaii_bull_bear", m, extra={
        "status": "ok",
        "rule": "(Bull-Bear) < -20 for 2 wks -> long SPY 12 wks; > +30 for 3 wks -> flat 12 wks.",
        "data_source": src,
    })


if __name__ == "__main__":
    main()
