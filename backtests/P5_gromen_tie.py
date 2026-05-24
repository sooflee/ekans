"""
P5 Gromen "TIE" — Treasury-Interest-Entitlements ratio.

Compute trailing-12-month TIE = (Interest + Social Security + Medicare outlays TTM)
                               / Total Receipts TTM.

When TIE > 100%, long GLD until ratio < 100%.

Data:
- Treasury MTS (fiscaldata.treasury.gov). Table 5 has agency outlays; Table 1
  has total receipts. We use the monthly "current_month_gross_outly_amt"
  values for the line items:
    * "Total--Interest on the Public Debt"
    * "Total--Social Security Administration"
    * "Total--Centers for Medicare and Medicaid Services"
  And monthly receipts from FRED MTSR133FMS (Net Receipts) which matches the
  MTS top line.

Sample from 2010+. Hold GLD via daily-bar simulation; position updates on the
1st trading day of each month using the most-recent fully-published TTM.
"""
import sys
import json
import datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

CACHE_FP = DATA / "mts_outlays_ttm.parquet"

TARGETS = {
    "interest": "Total--Interest on the Public Debt",
    "ss": "Total--Social Security Administration",
    "medicare": "Total--Centers for Medicare and Medicaid Services",
}


def fetch_mts_outlays(start_year=2009):
    """Pull monthly gross outlays for our 3 line items from MTS Table 5."""
    if CACHE_FP.exists():
        return pd.read_parquet(CACHE_FP)

    url = ("https://api.fiscaldata.treasury.gov/services/api/"
           "fiscal_service/v1/accounting/mts/mts_table_5")
    descs = list(TARGETS.values())
    # Filter by description IN (...) is not supported, so we filter client-side.
    # Page through all rows from start_year forward, fetching only needed fields.
    fields = ("record_date,classification_desc,"
              "current_month_gross_outly_amt")
    rows = []
    page = 1
    while True:
        params = {
            "fields": fields,
            "filter": f"record_date:gte:{start_year}-09-01",
            "page[size]": 10000,
            "page[number]": page,
            "sort": "record_date,classification_desc",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        chunk = data.get("data", [])
        if not chunk:
            break
        for row in chunk:
            d = row.get("classification_desc", "")
            if d in descs and row.get("current_month_gross_outly_amt") not in (None, "null"):
                rows.append({
                    "record_date": row["record_date"],
                    "desc": d,
                    "amt": float(row["current_month_gross_outly_amt"]),
                })
        if len(chunk) < 10000:
            break
        page += 1
        if page > 20:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["record_date"] = pd.to_datetime(df["record_date"])
    wide = df.pivot_table(index="record_date", columns="desc", values="amt", aggfunc="first")
    rename = {v: k for k, v in TARGETS.items()}
    wide = wide.rename(columns=rename).sort_index()
    wide.to_parquet(CACHE_FP)
    return wide


def main():
    try:
        outlays = fetch_mts_outlays(start_year=2009)
    except Exception as e:
        mark_failed("P5_gromen_tie", f"MTS API fetch failed: {e}")
        return

    if outlays.empty or any(c not in outlays.columns for c in ("interest", "ss", "medicare")):
        mark_failed("P5_gromen_tie", "MTS data incomplete: missing one of interest/ss/medicare line items.")
        return

    # Monthly receipts from FRED (MTSR133FMS = Federal Government: Receipts, monthly $M)
    rcpt = load_fred(["MTSR133FMS"], start="2009-01-01")["MTSR133FMS"]
    # MTS outlays are in dollars (we just parsed raw $); FRED is millions $.
    rcpt = rcpt * 1e6  # convert to dollars
    rcpt.index = rcpt.index + pd.offsets.MonthEnd(0)  # align to month-end like outlays

    # Align outlays to month-end (they already are EOM dates).
    outlays.index = outlays.index + pd.offsets.MonthEnd(0)
    # Combine + take TTM sums
    interest_ttm = outlays["interest"].rolling(12).sum()
    ss_ttm = outlays["ss"].rolling(12).sum()
    medicare_ttm = outlays["medicare"].rolling(12).sum()
    rcpt_ttm = rcpt.rolling(12).sum()

    df = pd.concat([
        interest_ttm.rename("int_ttm"),
        ss_ttm.rename("ss_ttm"),
        medicare_ttm.rename("med_ttm"),
        rcpt_ttm.rename("rcpt_ttm"),
    ], axis=1).dropna()

    df["tie"] = (df["int_ttm"] + df["ss_ttm"] + df["med_ttm"]) / df["rcpt_ttm"]

    # GLD prices
    gld = load_prices(["GLD"], start="2005-01-01")["GLD"]
    spy = load_prices(["SPY"], start="2005-01-01")["SPY"]
    rets = gld.pct_change()
    spy_rets = spy.pct_change()
    idx = gld.index

    # Publication lag: MTS for month M is published mid-following-month (~17th).
    # Use a conservative 25-day lag from EOM publication date to be safe.
    pub_lag = pd.Timedelta(days=25)

    pos = pd.Series(0.0, index=idx)
    last_signal = 0.0
    tie_signal_dates = []
    for date_eom, row in df.iterrows():
        eff_date = date_eom + pub_lag
        ix = idx.searchsorted(eff_date, side="left")
        if ix >= len(idx):
            continue
        sig = 1.0 if row["tie"] > 1.0 else 0.0
        # Persist until next signal arrives, but rule says "until ratio < 100%".
        # Equivalent: set pos = sig at each new monthly publication date.
        pos.iloc[ix] = sig
        tie_signal_dates.append((str(idx[ix].date()), float(row["tie"]), sig))

    # Forward-fill so position remains as last signal between monthly updates
    pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
    # But we need to actually emit 0s when sig flips to 0. Above pattern doesn't do that.
    # Re-do: collect sequence of (eff_date, sig) and ffill explicitly.
    sig_series = pd.Series(dtype=float)
    for date_eom, row in df.iterrows():
        eff_date = date_eom + pub_lag
        ix = idx.searchsorted(eff_date, side="left")
        if ix >= len(idx):
            continue
        d = idx[ix]
        sig = 1.0 if row["tie"] > 1.0 else 0.0
        sig_series.loc[d] = sig
    sig_series = sig_series.sort_index()
    pos = sig_series.reindex(idx).ffill().fillna(0.0)

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=spy_rets.dropna(), name="P5 Gromen TIE (long GLD if TIE>100%)")
    print_metrics(m)
    last_tie = float(df["tie"].iloc[-1])
    n_months_on = int((sig_series > 0).sum())
    save_result("P5_gromen_tie", m, extra={
        "status": "ok",
        "rule": "Compute TTM TIE = (interest + SS + Medicare outlays) / total receipts. Long GLD when > 100%, else flat.",
        "mechanism": "Luke Gromen: when entitlement+interest costs exceed receipts, fiscal repression / gold backing pressure.",
        "universe": "GLD; signal: Treasury MTS Table 5 outlays + FRED MTSR133FMS receipts.",
        "latest_tie": last_tie,
        "n_signal_months_on": n_months_on,
        "n_total_months": int(len(df)),
        "source": "Luke Gromen, FFTT (YouTube/podcasts, Phase 1P)",
    })


if __name__ == "__main__":
    main()
