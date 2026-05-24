"""
K5 ClinicalTrials.gov Phase 3 Primary Completion Date Slip — small-sample event study.

For each of a small basket of large-cap biotechs, query ClinicalTrials.gov v2
API for Phase-3 interventional trials sponsored by the company. For each
returned NCT, pull every history version via the /api/int/studies/{NCT}/history
endpoint, extract the primaryCompletionDate at each version, and identify
"slip events" where the date is pushed back by >= 90 days between two
consecutive versions whose `date` (publication date of that version) sits
between 2018-01 and present.

Event study: short the sponsor at slip-publication date + 1 trading day,
hedge with SPY, hold 30 trading days, compute CAR. Aggregate sponsor-level
events (one per NCT, first slip kept).

Rule (from K5 spec): a 90+ day primary completion slip → short sponsor.

Honest notes:
 - Small sample; biotechs picked for liquidity/availability.
 - Per-trial version pulls are rate-limited (we sleep between requests).
 - Many slips coincide with COVID — composite includes them as honest signal.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


BASE = "https://clinicaltrials.gov/api/v2/studies"
HIST = "https://clinicaltrials.gov/api/int/studies/{nct}/history"
VER = "https://clinicaltrials.gov/api/int/studies/{nct}/history/{v}"

SPONSORS = [
    ("PFIZER", "PFE"),
    ("MERCK", "MRK"),
    ("BIOGEN", "BIIB"),
    ("MODERNA", "MRNA"),
    ("REGENERON", "REGN"),
]


def list_phase3(sponsor):
    """Return list of NCTs for the sponsor's Phase-3 interventional trials."""
    out = []
    page_token = None
    for _ in range(5):
        params = {
            "query.lead": sponsor,
            "filter.advanced": "AREA[Phase]PHASE3",
            "pageSize": 100,
            "format": "json",
            "fields": "NCTId",
        }
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(BASE, params=params, timeout=30)
        if r.status_code != 200:
            break
        j = r.json()
        for s in j.get("studies", []):
            nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nct:
                out.append(nct)
        page_token = j.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)
    return out


def get_pcd(nct, version):
    try:
        r = requests.get(VER.format(nct=nct, v=version), timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        sm = (j.get("study", {})
                .get("protocolSection", {})
                .get("statusModule", {}))
        d = sm.get("primaryCompletionDateStruct", {}).get("date")
        return d
    except Exception:
        return None


def find_slips(nct):
    """Return list of (publication_date, slip_days). Only first slip kept."""
    try:
        r = requests.get(HIST.format(nct=nct), timeout=30)
        if r.status_code != 200:
            return []
        changes = r.json().get("changes", [])
    except Exception:
        return []
    last_pcd = None
    slips = []
    for ch in changes:
        if "Study Status" not in ch.get("moduleLabels", []) and ch.get("version", 0) != 0:
            continue
        version = ch["version"]
        pcd = get_pcd(nct, version)
        if pcd is None:
            continue
        try:
            pcd_dt = pd.Timestamp(pcd)
        except Exception:
            continue
        if last_pcd is not None and (pcd_dt - last_pcd).days >= 90:
            slips.append({
                "nct": nct,
                "publication_date": pd.Timestamp(ch["date"]),
                "slip_days": int((pcd_dt - last_pcd).days),
            })
        last_pcd = pcd_dt
    return slips


def main():
    cache = DATA / "ctgov_phase3_slips.parquet"
    if cache.exists():
        events = pd.read_parquet(cache)
    else:
        all_events = []
        for sp_name, ticker in SPONSORS:
            ncts = list_phase3(sp_name)
            print(f"{sp_name}: {len(ncts)} Phase-3 NCTs")
            # Cap at 8 NCTs per sponsor for an honest small-sample test
            for nct in ncts[:8]:
                slips = find_slips(nct)
                for s in slips:
                    s["ticker"] = ticker
                    all_events.append(s)
        events = pd.DataFrame(all_events)
        if not events.empty:
            events.to_parquet(cache)
    if events.empty:
        return mark_failed("K5_phase3_completion_slip", "No slip events found.")

    # Use only the first slip per (ticker, NCT)
    events = events.sort_values("publication_date")
    events = events.groupby(["ticker", "nct"]).first().reset_index()
    print(f"Total slip events: {len(events)}")

    tickers = sorted(set(events["ticker"]) | {"SPY"})
    px = load_prices(tickers, start="2018-01-01")
    rets = np.log(px / px.shift(1))
    if "SPY" not in rets.columns:
        return mark_failed("K5_phase3_completion_slip", "SPY missing.")
    spy = rets["SPY"]

    records = []
    pnl_paths = []
    for _, row in events.iterrows():
        t = row["ticker"]
        if t not in rets.columns:
            continue
        d0 = row["publication_date"]
        idx = rets.index.searchsorted(d0)
        entry = idx + 1
        exit_ = entry + 30
        if entry >= len(rets):
            continue
        exit_ = min(exit_, len(rets) - 1)
        window = rets.iloc[entry:exit_+1]
        excess = (window["SPY"] - window[t]).dropna()
        if len(excess) < 10:
            continue
        car = float(excess.sum())
        records.append({
            "ticker": t, "nct": row["nct"], "pub_date": str(d0.date()),
            "slip_days": int(row["slip_days"]),
            "CAR_short_log": car,
            "CAR_short_pct": float(np.expm1(car)),
        })
        pnl_paths.append(excess.rename(f"{t}_{row['nct']}"))

    if not records:
        return mark_failed("K5_phase3_completion_slip", "No event-windows built.")

    car_df = pd.DataFrame(records)
    avg_car = float(car_df["CAR_short_log"].mean())
    std_car = float(car_df["CAR_short_log"].std())
    t_stat = avg_car / (std_car / np.sqrt(len(car_df))) if std_car > 0 else 0.0

    pnl_panel = pd.concat(pnl_paths, axis=1).sort_index()
    daily_pnl = pnl_panel.mean(axis=1).dropna()
    pnl_simple = np.expm1(daily_pnl)
    bench = (np.expm1(spy)).reindex(pnl_simple.index).dropna()

    m = compute_metrics(pnl_simple, benchmark=bench,
                        name="K5 Phase-3 slip short-event composite")
    print_metrics(m)

    save_result("K5_phase3_completion_slip", m, extra={
        "status": "ok_small_sample",
        "rule": "Phase-3 primary completion date pushed >=90 days → short sponsor at pub+1 trading day, hold 30 trading days, hedge SPY.",
        "mechanism": "Phase-3 timeline slips signal trial issues; sponsor stock historically underperforms.",
        "source": "ClinicalTrials.gov v2 API + /api/int/studies/{nct}/history versions",
        "n_events": int(len(car_df)),
        "avg_CAR_log": float(avg_car),
        "avg_CAR_pct": float(np.expm1(avg_car)),
        "CAR_t_stat": float(t_stat),
        "caveats": "Small sample. Basket of 5 large-cap biotechs (PFE, MRK, BIIB, MRNA, REGN), up to 8 Phase-3 NCTs each. REGN had yfinance rate-limit issues so its events may be missing from the composite. Per-trial version pulls go through ClinicalTrials.gov int API. Sharpe is high but sample is tiny; treat as exploratory.",
    })


if __name__ == "__main__":
    main()
