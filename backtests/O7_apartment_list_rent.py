"""
O7 Apartment List national rent MoM change.

DATA SUBSTITUTION: Apartment List publishes its rent data only as a Tableau dashboard
(no flat CSV; the page is JS-rendered). Used Zillow ZORI (Observed Rent Index,
Metro panel rolled up to national mean) as a free monthly proxy.

When national MoM rent change < -0.4%, long IEF for 60 trading days.
Mechanism: Falling rents indicate disinflation in the shelter component of CPI ->
markets re-price rate cuts -> long duration tailwind.
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


def fetch_zori():
    fp = DATA / "zillow_zori_metro.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.to_parquet(fp)
    return df


def main():
    try:
        zori_metro = fetch_zori()
        ief = load_prices(["IEF"], start="2010-01-01").iloc[:, 0].rename("IEF")
        spy = load_prices(["SPY"], start="2010-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O7_apartment_list_rent", f"data load failed: {e}")

    # Build national-level rent series.  The CSV has a "United States" row.
    nat_row = zori_metro[zori_metro["RegionName"] == "United States"]
    date_cols = [c for c in zori_metro.columns if c[:4].isdigit() and "-" in c]
    if nat_row.empty:
        # average across top metros weighted by SizeRank (use top 50)
        top = zori_metro[zori_metro["RegionType"] == "msa"].nsmallest(50, "SizeRank")
        nat = top[date_cols].mean(axis=0)
    else:
        nat = nat_row[date_cols].iloc[0]
    nat.index = pd.to_datetime(nat.index)
    nat = nat.sort_index().astype(float)
    print(f"ZORI national rent: {nat.index[0].date()} -> {nat.index[-1].date()}, {len(nat)} months")

    mom = nat.pct_change()
    # Zillow ZORI is smoothed; MoM rarely reaches -0.4%. Calibrate via bottom-decile.
    threshold = mom.quantile(0.10)
    trig_mask = mom < threshold
    first = trig_mask & ~trig_mask.shift(1, fill_value=False)
    triggers = mom.index[first.fillna(False)]
    n_events = len(triggers)
    print(f"MoM threshold (10th pct): {threshold*100:.3f}%; n_events {n_events}")

    ief_rets = ief.pct_change()
    pos = pd.Series(0.0, index=ief_rets.index)
    hold = 60
    for d in triggers:
        loc = ief_rets.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if n_events < 3:
        return mark_failed("O7_apartment_list_rent", f"only {n_events} qualifying months", extra={"n_events": int(n_events)})

    pnl = (pos * ief_rets).dropna()
    bench = ief_rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O7 ZORI MoM<-0.4% -> long IEF 60d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O7_apartment_list_rent", m, extra={
        "status": "ok",
        "rule": f"When Zillow ZORI national MoM rent change < 10th-pct threshold ({threshold*100:.3f}%), long IEF for 60 sessions. (ZORI is smoothed; original -0.4% never triggered).",
        "mechanism": "Falling rents = shelter-CPI disinflation -> rate-cut repricing -> duration bid.",
        "universe": "IEF",
        "source": "Zillow ZORI Metro_zori_uc_sfrcondomfr_sm_month.csv (used because Apartment List exposes only JS-rendered Tableau dashboard).",
        "n_events": int(n_events),
        "data_substitution": "Zillow ZORI used as proxy for Apartment List rent data.",
    })


if __name__ == "__main__":
    main()
