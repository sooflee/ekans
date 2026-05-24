"""
O9 Indeed Hiring Lab US job postings index. When YoY % change < -8%, short XLY for
60 trading days.
Mechanism: Job postings lead actual employment / wages -> consumer discretionary
spend pulls back when hiring momentum collapses.

Source: github.com/hiring-lab/job_postings_tracker (free CSV, daily, since Feb 2020).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_indeed():
    fp = DATA / "indeed_us_postings.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/US/aggregate_job_postings_US.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    df = df[df["variable"] == "total postings"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["indeed_job_postings_index_SA"]]
    df.columns = ["idx"]
    df = df.sort_index()
    df.to_parquet(fp)
    return df


def main():
    try:
        indeed = fetch_indeed()
        xly = load_prices(["XLY"], start="2019-01-01").iloc[:, 0].rename("XLY")
        spy = load_prices(["SPY"], start="2019-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O9_indeed_jobs", f"data load failed: {e}")

    idx = indeed["idx"]
    # Resample to weekly (Friday); YoY = 52w change
    weekly = idx.resample("W-FRI").last().dropna()
    yoy = weekly.pct_change(52) * 100  # in %

    # Trigger weeks: YoY < -8%
    trig_mask = yoy < -8.0
    # Mark first week of each spell
    first = trig_mask & ~trig_mask.shift(1, fill_value=False)
    triggers = yoy.index[first]
    n_events = len(triggers)

    xly_rets = xly.pct_change()
    pos = pd.Series(0.0, index=xly_rets.index)

    hold = 60
    for d in triggers:
        loc = xly_rets.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = -1.0  # short

    if n_events < 1:
        return mark_failed("O9_indeed_jobs", f"no qualifying YoY < -8% events", extra={"n_events": 0})

    pnl = (pos * xly_rets).dropna()
    bench = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O9 Indeed jobs YoY<-8% -> short XLY 60d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O9_indeed_jobs", m, extra={
        "status": "ok",
        "rule": "When Indeed Hiring Lab US postings YoY (52w) < -8%, short XLY for 60 sessions.",
        "mechanism": "Postings lead employment/wages -> discretionary spending rolls over when hiring collapses.",
        "universe": "XLY",
        "source": "github.com/hiring-lab/job_postings_tracker (aggregate_job_postings_US.csv, total postings, SA index).",
        "n_events": int(n_events),
    })


if __name__ == "__main__":
    main()
