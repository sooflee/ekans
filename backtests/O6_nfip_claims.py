"""
O6 FEMA NFIP (National Flood Insurance Program) claims weekly aggregate.

When weekly aggregate paid-claims (building + contents damage) > $500M, short XHB
for 30 trading days.
Mechanism: Major flood events -> regional homebuilder & supplier headwind; raw
materials / labor diverted to repair; demand destruction in flooded markets.

Source: FEMA OpenFEMA FimaNfipClaims API (paginated 10k at a time).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import urllib.request
import urllib.parse

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_nfip_weekly():
    fp = DATA / "fema_nfip_weekly_claims.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    page_size = 10000
    filt = "dateOfLoss ge '2010-01-01T00:00:00.000Z'"
    select = "dateOfLoss,buildingDamageAmount,contentsDamageAmount"
    all_chunks = []
    skip = 0
    while True:
        params = urllib.parse.urlencode({
            "$filter": filt,
            "$select": select,
            "$top": page_size,
            "$skip": skip,
        })
        url = f"https://www.fema.gov/api/open/v2/FimaNfipClaims?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=90).read()
        j = json.loads(raw)
        records = j.get("FimaNfipClaims", [])
        if not records:
            break
        all_chunks.append(records)
        if len(records) < page_size:
            break
        skip += page_size
        if skip > 3_000_000:
            break  # safety cap
    if not all_chunks:
        raise RuntimeError("NFIP returned no records")
    flat = [r for chunk in all_chunks for r in chunk]
    print(f"Fetched {len(flat)} NFIP claim rows")
    df = pd.DataFrame(flat)
    df["dateOfLoss"] = pd.to_datetime(df["dateOfLoss"], errors="coerce", utc=True).dt.tz_convert(None)
    df["bldg"] = pd.to_numeric(df["buildingDamageAmount"], errors="coerce").fillna(0)
    df["cont"] = pd.to_numeric(df["contentsDamageAmount"], errors="coerce").fillna(0)
    df["paid"] = df["bldg"] + df["cont"]
    df = df.dropna(subset=["dateOfLoss"]).set_index("dateOfLoss")
    weekly = df["paid"].resample("W-FRI").sum().to_frame("paid_usd")
    weekly.to_parquet(fp)
    return weekly


def main():
    try:
        weekly = fetch_nfip_weekly()
        xhb = load_prices(["XHB"], start="2010-01-01").iloc[:, 0].rename("XHB")
        spy = load_prices(["SPY"], start="2010-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O6_nfip_claims", f"data load failed: {e}")

    # strip tz from weekly index for searchsorted alignment with XHB
    if weekly.index.tz is not None:
        weekly.index = weekly.index.tz_convert(None)

    print(f"Weekly claims rows: {len(weekly)}; max week paid: ${weekly['paid_usd'].max()/1e6:.1f}M")

    threshold = 500e6
    trig_mask = weekly["paid_usd"] > threshold
    triggers = weekly.index[trig_mask]
    n_events = len(triggers)
    if n_events < 3:
        # Lower threshold and try again
        threshold = 100e6
        trig_mask = weekly["paid_usd"] > threshold
        triggers = weekly.index[trig_mask]
        n_events = len(triggers)
    if n_events < 3:
        return mark_failed("O6_nfip_claims", f"only {n_events} events at $100M threshold; data likely lags / suppressed",
                           extra={"n_events": int(n_events)})

    xhb_rets = xhb.pct_change()
    pos = pd.Series(0.0, index=xhb_rets.index)
    hold = 30
    for d in triggers:
        loc = xhb_rets.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = -1.0

    pnl = (pos * xhb_rets).dropna()
    bench = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name=f"O6 NFIP weekly>${threshold/1e6:.0f}M -> short XHB 30d")
    m["n_events"] = int(n_events)
    m["threshold_usd"] = float(threshold)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O6_nfip_claims", m, extra={
        "status": "ok",
        "rule": f"When weekly NFIP paid-claims sum > ${threshold/1e6:.0f}M, short XHB for 30 sessions.",
        "mechanism": "Major flood weeks drag homebuilder/supplier demand & divert labor/materials to repair.",
        "universe": "XHB",
        "source": "FEMA OpenFEMA FimaNfipClaims API (since 2010, paginated).",
        "n_events": int(n_events),
        "threshold_usd": float(threshold),
    })


if __name__ == "__main__":
    main()
