"""
AD-F3 CMS Friday / pre-holiday payment-rule timing.

Rule: When CMS publishes an IPPS/OPPS/MPFS/MA Final Rule on Friday OR pre-
holiday → take the OPPOSITE of the typical insurer/provider reaction T+0
to T+3. Simplification per spec:

  Pair = LONG (UNH + HUM + CVS + ELV)/4  vs  SHORT (HCA + THC + UHS + CYH)/4

We compute the pair return for the 3 trading days after publication.

Data: federalregister.gov API filtered to CMS + RULE; title regex on
"IPPS|OPPS|Physician Fee Schedule|Medicare Advantage|Rate Announcement|
SNF PPS|IRF PPS". Then keep events on Friday OR business-day-before-federal-
holiday.
"""
import sys
import json
import time
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse

from harness import (
    load_prices, compute_metrics, save_result, mark_failed, DATA,
)


CACHE = DATA / "AD_F3_cms_rules.json"
API = "https://www.federalregister.gov/api/v1/articles.json"

TITLE_RE = re.compile(
    r"IPPS|OPPS|Physician Fee Schedule|Medicare Advantage|Rate Announcement|"
    r"SNF PPS|IRF PPS|Hospital Outpatient Prospective|Hospital Inpatient "
    r"Prospective|Home Health Prospective|Hospice Wage Index|"
    r"End-Stage Renal Disease Prospective",
    re.IGNORECASE,
)

INSURERS = ["UNH", "HUM", "CVS", "ELV"]
PROVIDERS = ["HCA", "THC", "UHS", "CYH"]

# US federal holidays a quick lookup via pandas USFederalHolidayCalendar
def federal_holidays():
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    return set(d.normalize() for d in cal.holidays(start="2017-01-01", end="2026-12-31"))


def is_pre_holiday(d, holidays):
    """True if the next business day is a federal holiday."""
    next_bd = d + pd.tseries.offsets.BDay(1)
    return pd.Timestamp(next_bd).normalize() in holidays


def fetch_cms_rules():
    if CACHE.exists():
        with open(CACHE) as f:
            return json.load(f)
    results = []
    page = 1
    while True:
        params = {
            "conditions[agencies][]": "centers-for-medicare-medicaid-services",
            "conditions[type][]": "RULE",
            "conditions[publication_date][gte]": "2018-01-01",
            "conditions[publication_date][lte]": "2026-05-24",
            "per_page": "200",
            "page": str(page),
        }
        parts = []
        for k, v in params.items():
            parts.append(f"{urllib.parse.quote(k)}={urllib.parse.quote(v)}")
        url = API + "?" + "&".join(parts)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  fetch page {page} failed: {e}", flush=True)
            break
        results.extend(data.get("results", []))
        total = data.get("count", 0)
        next_page = data.get("next_page_url")
        print(f"  page {page}: got {len(data.get('results', []))} of {total} total", flush=True)
        if not next_page:
            break
        page += 1
        time.sleep(0.5)
        if page > 30:
            break
    with open(CACHE, "w") as f:
        json.dump(results, f)
    return results


def main():
    print("AD-F3: fetching CMS Final Rules...", flush=True)
    try:
        rules = fetch_cms_rules()
    except Exception as e:
        return mark_failed("AD-F3", f"FR API fetch failed: {e}")
    print(f"  total CMS rules pulled: {len(rules)}", flush=True)

    holidays = federal_holidays()

    payment_events = []
    for r in rules:
        title = r.get("title", "")
        if not TITLE_RE.search(title):
            continue
        d_str = r.get("publication_date")
        if not d_str:
            continue
        d = pd.Timestamp(d_str)
        is_fri = d.dayofweek == 4
        is_pre = is_pre_holiday(d, holidays)
        if not (is_fri or is_pre):
            continue
        payment_events.append({
            "date": d_str,
            "title": title[:120],
            "fri": is_fri,
            "pre_holiday": is_pre,
        })
    print(f"  CMS payment rules on Friday or pre-holiday: {len(payment_events)}", flush=True)

    if len(payment_events) < 5:
        return mark_failed("AD-F3",
                           f"too few events ({len(payment_events)})")

    # dedup by date
    by_date = {}
    for r in payment_events:
        if r["date"] not in by_date:
            by_date[r["date"]] = r
    events = sorted(by_date.values(), key=lambda x: x["date"])
    print(f"  unique-date events: {len(events)}", flush=True)

    # Prices
    tickers = INSURERS + PROVIDERS
    try:
        px = load_prices(tickers, start="2017-12-01")
    except Exception as e:
        return mark_failed("AD-F3", f"price load failed: {e}")

    def basket_ret(d_event, names, hold=3):
        rets = []
        for tk in names:
            if tk not in px.columns:
                continue
            s = px[tk].dropna()
            if s.empty:
                continue
            i = s.index.searchsorted(d_event, side="right") - 1
            if i < 0 or i + hold >= len(s):
                continue
            r = float(s.iloc[i + hold] / s.iloc[i] - 1)
            rets.append(r)
        return float(np.mean(rets)) if rets else np.nan

    rows = []
    for ev in events:
        d = pd.Timestamp(ev["date"])
        ins_r = basket_ret(d, INSURERS, hold=3)
        prov_r = basket_ret(d, PROVIDERS, hold=3)
        if np.isnan(ins_r) or np.isnan(prov_r):
            continue
        pair = ins_r - prov_r  # long insurers vs short providers
        rows.append({
            "date": ev["date"],
            "title": ev["title"],
            "fri": ev["fri"],
            "pre_holiday": ev["pre_holiday"],
            "ins_pct": ins_r * 100,
            "prov_pct": prov_r * 100,
            "pair_pct": pair * 100,
        })

    if len(rows) < 5:
        return mark_failed("AD-F3", f"too few testable events ({len(rows)})")

    rets = np.array([r["pair_pct"] / 100.0 for r in rows])
    avg = float(rets.mean())
    sd = float(rets.std())
    se = sd / np.sqrt(len(rets)) if sd > 0 else np.nan
    t_stat = avg / se if se and se > 0 else 0.0
    hit = float((rets > 0).mean())
    sharpe = (avg / sd) * np.sqrt(252 / 3) if sd > 0 else 0.0
    years = (pd.Timestamp(rows[-1]["date"]) - pd.Timestamp(rows[0]["date"])).days / 365.25
    evt_per_yr = len(rows) / years if years > 0 else 0
    ann_est = avg * evt_per_yr

    print(f"AD-F3 CMS Friday/pre-holiday → insurer-long, provider-short, N={len(rows)}, T+3 hold")
    print(f"  span: {rows[0]['date']} -> {rows[-1]['date']}  (~{evt_per_yr:.1f}/yr)")
    print(f"  mean pair ret={avg*100:.2f}%  stdev={sd*100:.2f}%  t-stat={t_stat:.2f}")
    print(f"  hit={hit*100:.0f}%  sharpe~{sharpe:.2f}  ann_est={ann_est*100:.2f}%")
    print(f"  worst: {min(rows, key=lambda r: r['pair_pct'])['date']} {min(rows, key=lambda r: r['pair_pct'])['pair_pct']:.2f}%")
    print(f"  best:  {max(rows, key=lambda r: r['pair_pct'])['date']} {max(rows, key=lambda r: r['pair_pct'])['pair_pct']:.2f}%")

    metrics = {
        "name": "AD-F3 CMS Friday/pre-holiday payment rule",
        "n_events": len(rows),
        "events_per_year": evt_per_yr,
        "mean_event_ret_pct": avg * 100,
        "stdev_event_ret_pct": sd * 100,
        "t_stat": t_stat,
        "sharpe_approx": sharpe,
        "hit_rate": hit,
        "ann_return_est_pct": ann_est * 100,
        "cagr": ann_est,
    }
    extra = {
        "status": "ok",
        "rule": "On CMS payment-rate Final Rule (IPPS/OPPS/MPFS/MA/SNF/IRF) "
                "published Friday or pre-holiday, long insurer basket "
                "(UNH/HUM/CVS/ELV) - short provider basket (HCA/THC/UHS/CYH); "
                "hold T+0 close to T+3 close.",
        "mechanism": "Payment-rate Final Rules published on bad-news-burying "
                     "days tend to be insurer-friendly or to fade the headline-"
                     "negative interpretation; market over-reacts intraday and "
                     "snaps back over the following 3 sessions.",
        "source": "Federal Register API filtered to CMS + RULE + title regex "
                  "for payment systems; yfinance prices.",
        "events_sample": rows[:5] + rows[-5:],
        "n_total_cms_rules": len(rules),
        "caveats": [
            "Many CMS 'payment system' Final Rules are routine technical "
            "corrections, not the headline annual rate-setting rules.",
            "Pre-holiday filter uses USFederalHolidayCalendar; not perfect.",
            "Simplified per spec: no surprise-vs-Advance Notice gate, no "
            "direction-flip logic.",
        ],
    }
    save_result("AD-F3", metrics, extra=extra)


if __name__ == "__main__":
    main()
