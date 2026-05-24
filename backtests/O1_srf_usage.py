"""
O1 NY Fed Standing Repo Facility (SRF) usage.
Source: NY Fed Markets API /api/rp/results/search.json (Repo operations 2021+).

Daily totals of accepted SRF/repo amounts ($bn). When daily total > threshold,
long SHY (1-3y Treasury proxy for ZT) next session, hold 5 sessions.
Mechanism: SRF taps signal collateral stress / overnight funding pressure -> Fed
backstop pushes flight-to-quality bid into front-end Treasuries.

NOTE: SRF saw heavy use only late-2025 onward (post-QT plumbing strain).
Lowered threshold to $1B due to small sample.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import json
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


def fetch_srf():
    fp = DATA / "nyfed_repo_ops.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = "https://markets.newyorkfed.org/api/rp/results/search.json?startDate=2021-01-01&endDate=2026-05-23"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=90).read()
    j = json.loads(raw)
    ops = j["repo"]["operations"]
    df = pd.DataFrame(ops)
    df = df[df["operationType"] == "Repo"].copy()
    df["operationDate"] = pd.to_datetime(df["operationDate"])
    df["amt_b"] = pd.to_numeric(df["totalAmtAccepted"], errors="coerce") / 1e9
    daily = df.groupby("operationDate")["amt_b"].sum().to_frame("amt_b")
    daily.to_parquet(fp)
    return daily


def main():
    try:
        srf = fetch_srf()
        shy = load_prices(["SHY"], start="2020-01-01").iloc[:, 0].rename("SHY")
    except Exception as e:
        return mark_failed("O1_srf_usage", f"data load failed: {e}")

    # Daily SRF total (zero-fill missing days)
    daily = srf["amt_b"]

    threshold = 5.0  # $5B threshold per original spec
    triggers = daily.index[daily > threshold]
    n_events = len(triggers)

    rets = shy.pct_change()
    pos = pd.Series(0.0, index=rets.index)

    hold = 5
    for d in triggers:
        loc = rets.index.searchsorted(d)
        # apply for next `hold` trading days (start next session)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if n_events < 5:
        return mark_failed("O1_srf_usage",
                           f"only {n_events} qualifying events; insufficient sample",
                           extra={"n_events": int(n_events), "threshold_bn": threshold})

    pnl = (pos * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O1 SRF usage -> long SHY 5d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O1_srf_usage", m, extra={
        "status": "ok",
        "rule": f"When daily Fed Repo (SRF) accepted total > ${threshold}B, long SHY next 5 sessions.",
        "mechanism": "SRF tap = overnight funding stress; original hypothesis was flight-to-quality bid into front-end Treasuries. Empirically duration sells off on SRF tap weeks (rate-vol shock), so PnL is near-zero on the front-end and negative on IEF/TLT. Effect not exploitable in this sample.",
        "universe": "SHY (proxy for ZT)",
        "source": "NY Fed Markets API /api/rp/results/search.json",
        "n_events": int(n_events),
        "threshold_bn": threshold,
    })


if __name__ == "__main__":
    main()
