"""
AD-F1 Friday-evening EPA final-rule → XLE long / XLU short pair.

Rule: EPA Final Rule (CFR title 40) published on a Friday → short XLU + long
XLE pair from Friday close to T+5 close.

Data: federalregister.gov API
  /api/v1/articles.json?conditions[agencies][]=environmental-protection-agency
   &conditions[type][]=RULE&per_page=1000

We filter to type='Rule', publication_date on Friday, dates 2018-01-01 to today.
Cache the API responses to data/AD_F1_epa_rules.json so repeated runs are fast.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


CACHE = DATA / "AD_F1_epa_rules.json"
API = "https://www.federalregister.gov/api/v1/articles.json"


def fetch_epa_rules():
    if CACHE.exists():
        with open(CACHE) as f:
            return json.load(f)
    results = []
    page = 1
    while True:
        params = {
            "conditions[agencies][]": "environmental-protection-agency",
            "conditions[type][]": "RULE",
            "conditions[publication_date][gte]": "2018-01-01",
            "conditions[publication_date][lte]": "2026-05-24",
            "per_page": "200",
            "page": str(page),
            "fields[]": ["publication_date", "title", "document_number",
                         "type", "significant", "html_url", "abstract"],
        }
        # rebuild manually because of repeated keys
        parts = []
        for k, v in params.items():
            if isinstance(v, list):
                for vv in v:
                    parts.append(f"{urllib.parse.quote(k)}={urllib.parse.quote(vv)}")
            else:
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
        if page > 30:  # safety
            break
    with open(CACHE, "w") as f:
        json.dump(results, f)
    return results


def main():
    print("AD-F1: fetching EPA Final Rules from Federal Register API...", flush=True)
    try:
        rules = fetch_epa_rules()
    except Exception as e:
        return mark_failed("AD-F1", f"FR API fetch failed: {e}")
    print(f"  total EPA rules pulled: {len(rules)}")

    # Filter to Fridays
    fri_rules = []
    for r in rules:
        d_str = r.get("publication_date")
        if not d_str:
            continue
        d = pd.Timestamp(d_str)
        if d.dayofweek != 4:  # Friday
            continue
        fri_rules.append({
            "date": d_str,
            "title": r.get("title", "")[:120],
            "doc": r.get("document_number"),
            "sig": r.get("significant"),
        })
    print(f"  EPA Final Rules published on a Friday: {len(fri_rules)}")

    if len(fri_rules) < 5:
        return mark_failed("AD-F1", f"too few Friday EPA rules ({len(fri_rules)})")

    # De-dup by date — multiple rules same Friday = one trading event
    by_date = {}
    for r in fri_rules:
        by_date.setdefault(r["date"], []).append(r["title"])
    fri_dates = sorted(by_date.keys())
    print(f"  unique Friday dates with EPA rules: {len(fri_dates)}")

    # Prices
    try:
        px = load_prices(["XLE", "XLU", "SPY"], start="2017-12-01")
    except Exception as e:
        return mark_failed("AD-F1", f"price load failed: {e}")

    xle = px["XLE"].dropna()
    xlu = px["XLU"].dropna()
    if xle.empty or xlu.empty:
        return mark_failed("AD-F1", "no XLE/XLU data")

    rows = []
    for d_str in fri_dates:
        d = pd.Timestamp(d_str)
        # Entry: Friday close. If d is a market holiday, snap to prior close.
        i_xle = xle.index.searchsorted(d, side="right") - 1
        i_xlu = xlu.index.searchsorted(d, side="right") - 1
        if i_xle < 0 or i_xlu < 0:
            continue
        # T+5 close
        if i_xle + 5 >= len(xle) or i_xlu + 5 >= len(xlu):
            continue
        xle0 = xle.iloc[i_xle]
        xle5 = xle.iloc[i_xle + 5]
        xlu0 = xlu.iloc[i_xlu]
        xlu5 = xlu.iloc[i_xlu + 5]
        r_xle = float(xle5 / xle0 - 1)
        r_xlu = float(xlu5 / xlu0 - 1)
        pair = r_xle - r_xlu  # long XLE - short XLU
        rows.append({
            "date": d_str,
            "n_rules": len(by_date[d_str]),
            "first_title": by_date[d_str][0][:80],
            "xle_pct": r_xle * 100,
            "xlu_pct": r_xlu * 100,
            "pair_pct": pair * 100,
        })

    if len(rows) < 5:
        return mark_failed("AD-F1", f"too few testable events ({len(rows)})")

    rets = np.array([r["pair_pct"] / 100.0 for r in rows])
    avg = float(rets.mean())
    sd = float(rets.std())
    se = sd / np.sqrt(len(rets)) if sd > 0 else np.nan
    t_stat = avg / se if se and se > 0 else 0.0
    hit = float((rets > 0).mean())
    sharpe_per_event = (avg / sd) * np.sqrt(252 / 5) if sd > 0 else 0.0
    # Annualization: ~20-40 events/year if every Friday; we measure per-event.
    # Approx annual return = avg * (events_per_year). Rough events/yr:
    years = (pd.Timestamp(rows[-1]["date"]) - pd.Timestamp(rows[0]["date"])).days / 365.25
    evt_per_yr = len(rows) / years if years > 0 else 0
    ann_est = avg * evt_per_yr

    print(f"AD-F1 EPA Friday rule → long XLE / short XLU, N={len(rows)}, T+5 hold")
    print(f"  span: {rows[0]['date']} -> {rows[-1]['date']}  (~{evt_per_yr:.1f} events/yr)")
    print(f"  mean pair ret={avg*100:.2f}%  stdev={sd*100:.2f}%  t-stat={t_stat:.2f}")
    print(f"  hit={hit*100:.0f}%  sharpe~{sharpe_per_event:.2f}  ann_est={ann_est*100:.2f}%")
    print(f"  worst: {min(rows, key=lambda r: r['pair_pct'])}")
    print(f"  best:  {max(rows, key=lambda r: r['pair_pct'])}")

    metrics = {
        "name": "AD-F1 EPA Friday → XLE/XLU pair",
        "n_events": len(rows),
        "events_per_year": evt_per_yr,
        "mean_event_ret_pct": avg * 100,
        "stdev_event_ret_pct": sd * 100,
        "t_stat": t_stat,
        "sharpe_approx": sharpe_per_event,
        "hit_rate": hit,
        "ann_return_est_pct": ann_est * 100,
        "cagr": ann_est,  # approximate
    }
    extra = {
        "status": "ok",
        "rule": "On any EPA Final Rule (type=RULE) published on a Friday, "
                "long XLE / short XLU pair from Friday close to T+5 close.",
        "mechanism": "Friday dumps signal regulatory burden the agency wants "
                     "buried; energy producers re-rate as cost-of-compliance "
                     "shows, utilities re-rate as monopoly-cost-pass-through "
                     "value drops. Pair captures the relative move.",
        "source": "Federal Register API (https://www.federalregister.gov/api/v1/articles.json) "
                  "filtered to EPA + Final Rule + Friday publication_date; yfinance.",
        "events_sample": rows[:5] + rows[-5:],
        "n_total_epa_rules": len(rules),
        "n_friday_epa_rules": len(fri_rules),
        "caveats": [
            "All EPA Final Rules included (no 'economically significant' filter — the FR API doesn't expose that field reliably).",
            "Many Friday EPA rules are routine state-plan approvals, not major regulations; signal-to-noise is low.",
        ],
    }
    save_result("AD-F1", metrics, extra=extra)


if __name__ == "__main__":
    main()
