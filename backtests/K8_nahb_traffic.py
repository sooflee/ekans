"""
K8 NAHB Traffic of Prospective Buyers subindex — long ITB on 1σ surges.

Rule: When the monthly Traffic-of-Prospective-Buyers subindex prints 1σ
above its trailing 12-month mean, go long ITB (homebuilders) for 30
trading days starting the day after the official mid-month release.

Mechanism: Walk-in foot traffic at model homes leads contract signings by
3-6 weeks; jumps above trend predict near-term homebuilder revenue beats.

Source: NAHB/Wells Fargo HMI Components history (XLS, Table 3, "Traffic of
Prospective Buyers" block — seasonally adjusted monthly back to 1985).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA, rolling_zscore,
)


# NAHB publishes the XLS in dated subfolders; their latest report folder is
# named YYYY-MM. We try the current month and fall back several months.
def fetch_nahb():
    cache = DATA / "nahb_components.xls"
    if cache.exists():
        return cache
    today = pd.Timestamp.today()
    for back in range(0, 13):
        d = today - pd.DateOffset(months=back)
        ym = f"{d.year:04d}-{d.month:02d}"
        ym_short = f"{d.year:04d}{d.month:02d}"
        url = (f"https://www.nahb.org/-/media/NAHB/news-and-economics/docs/"
               f"housing-economics/hmi/{ym}/"
               f"t3-national-hmi-components-history-{ym_short}.xls")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                cache.write_bytes(r.content)
                return cache
        except Exception:
            continue
    return None


def parse_block(df, start_row):
    """Parse 12 months × N years block starting at `start_row` (year header)."""
    # row start_row is years across columns 1..N
    years = df.iloc[start_row, 1:].dropna().astype(int).tolist()
    months = df.iloc[start_row+1:start_row+13, 0].tolist()
    block = df.iloc[start_row+1:start_row+13, 1:1+len(years)]
    block.columns = years
    block.index = months
    long_ = block.stack().reset_index()
    long_.columns = ["month_label", "year", "value"]
    months_map = {m: i+1 for i, m in enumerate(
        ["Jan.", "Feb.", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."])}
    long_["month"] = long_["month_label"].map(months_map)
    long_ = long_.dropna(subset=["month"])
    long_["date"] = pd.to_datetime(long_.apply(lambda r: f"{int(r.year)}-{int(r.month):02d}-15", axis=1))
    return long_[["date", "value"]].sort_values("date").reset_index(drop=True)


def main():
    p = fetch_nahb()
    if p is None:
        return mark_failed("K8_nahb_traffic", "Could not fetch NAHB HMI components XLS.")
    df = pd.read_excel(p, header=None)
    # Find Traffic of Prospective Buyers section
    traffic_row = None
    for i in range(len(df)):
        if "Traffic of Prospective Buyers" in str(df.iloc[i, 0]):
            traffic_row = i
            break
    if traffic_row is None:
        return mark_failed("K8_nahb_traffic", "Traffic block not found in XLS.")
    series = parse_block(df, traffic_row + 1).set_index("date")["value"].astype(float)
    series = series.dropna()
    if len(series) < 100:
        return mark_failed("K8_nahb_traffic", f"Traffic series too short: {len(series)}")

    # 12-month rolling mean & std
    mean12 = series.rolling(12).mean()
    std12 = series.rolling(12).std()
    sigma = (series - mean12) / std12

    try:
        px = load_prices(["ITB", "SPY"], start="2006-05-01")
    except Exception as e:
        return mark_failed("K8_nahb_traffic", f"price fetch: {e}")
    rets = daily_returns(px)
    itb = rets["ITB"].dropna()

    # NAHB HMI released ~3rd week of month (~17th typically). Use month-end of
    # release month as the earliest tradeable date.
    sig = pd.Series(0.0, index=itb.index)
    n_trig = 0
    for ts, z in sigma.dropna().items():
        if z > 1.0:
            # publish ~17th; start trading next business day
            trade_date = ts + pd.offsets.MonthEnd(0)  # safe: end-of-publish-month
            i = itb.index.searchsorted(trade_date)
            if i >= len(itb):
                continue
            sig.iloc[i:i+30] += 1
            n_trig += 1

    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * itb
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K8 NAHB Traffic > 1σ long ITB")
    print_metrics(m)
    print(f"n_obs={len(series)}, n_trig={n_trig}, exposure={pos.mean():.2%}")

    save_result("K8_nahb_traffic", m, extra={
        "status": "ok",
        "rule": "Monthly NAHB Traffic >1σ above its trailing-12m mean → long ITB 30 trading days from month-end.",
        "mechanism": "Foot traffic at model homes leads contract signings; predicts homebuilder near-term revenue.",
        "source": "NAHB/Wells Fargo HMI Components history XLS, Traffic of Prospective Buyers block",
        "n_observations": int(len(series)),
        "n_triggers": int(n_trig),
        "exposure_pct": float(pos.mean()),
    })


if __name__ == "__main__":
    main()
