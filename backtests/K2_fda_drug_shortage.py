"""
K2 FDA Drug Shortages — long IHE on net additions weeks.

Rule: For each ISO week, count distinct generic_name entries whose
'initial_posting_date' fell that week. When the weekly count > 5, go long
IHE (pharma ETF) for 10 trading days starting Monday of the following week
(to ensure data is observable post-publication).

Mechanism: A burst of new shortages tightens supply for branded/generic
substitutes; pharma producers with capacity gain pricing power and market
share. Holding IHE captures the basket effect rather than picking single
manufacturers.

Source: openFDA /drug/shortages.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA,
)


def pull_shortages():
    cache = DATA / "fda_shortages.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    skip, limit = 0, 1000
    while True:
        url = f"https://api.fda.gov/drug/shortages.json?limit={limit}&skip={skip}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            break
        j = r.json()
        data = j.get("results", [])
        if not data:
            break
        for row in data:
            rows.append({
                "initial_posting_date": row.get("initial_posting_date"),
                "generic_name": row.get("generic_name"),
                "status": row.get("status"),
                "update_date": row.get("update_date"),
            })
        if len(data) < limit:
            break
        skip += limit
        if skip > 50000:
            break
    df = pd.DataFrame(rows)
    df.to_parquet(cache)
    return df


def main():
    try:
        df = pull_shortages()
    except Exception as e:
        return mark_failed("K2_fda_drug_shortage", f"openFDA fetch: {e}")

    if df.empty:
        return mark_failed("K2_fda_drug_shortage", "openFDA returned empty.")

    df["initial_posting_date"] = pd.to_datetime(df["initial_posting_date"], errors="coerce")
    df = df.dropna(subset=["initial_posting_date", "generic_name"])
    df = df.drop_duplicates(subset=["generic_name", "initial_posting_date"])

    weekly = df.set_index("initial_posting_date").assign(n=1)["n"].resample("W-MON").sum()

    try:
        px = load_prices(["IHE", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed("K2_fda_drug_shortage", f"price fetch: {e}")

    rets = daily_returns(px)
    ihe = rets["IHE"].dropna()

    triggers = weekly[weekly > 5].index
    sig = pd.Series(0.0, index=ihe.index)
    n_trig = 0
    for d in triggers:
        # enter the following Monday's trading day
        i = ihe.index.searchsorted(d + pd.Timedelta(days=1))
        if i >= len(ihe):
            continue
        sig.iloc[i:i+10] += 1
        n_trig += 1

    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * ihe
    pnl = pnl.dropna()

    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K2 FDA shortages long IHE")
    print_metrics(m)
    print(f"weeks_obs={len(weekly)}, weeks_>5={int((weekly>5).sum())}, n_trig={n_trig}, exposure={pos.mean():.2%}")

    save_result("K2_fda_drug_shortage", m, extra={
        "status": "ok",
        "rule": "Weekly count of new openFDA shortage postings > 5 → long IHE for 10 trading days from following Mon.",
        "mechanism": "Shortage clusters give pharma producers pricing/share gains; IHE captures basket effect.",
        "source": "openFDA /drug/shortages.json (initial_posting_date)",
        "n_triggers": int(n_trig),
        "exposure_pct": float(pos.mean()),
        "caveats": ("openFDA returns current entries (with historical posting dates) — "
                    "not a true point-in-time snapshot, so survivorship/coverage drift is possible "
                    "for very old observations."),
    })


if __name__ == "__main__":
    main()
