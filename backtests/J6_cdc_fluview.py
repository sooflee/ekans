"""
J6 CDC FluView surge -> long WMT+CVS / short JETS (use LUV as airline proxy due to JETS history).

Data source: Carnegie Mellon DELPHI FluSurv-NET API (mirrors CDC weekly hospitalization).
Endpoint: https://api.delphi.cmu.edu/epidata/flusurv/?locations=network_all&epiweeks=...
Field: 'rate_overall' (per 100k pop, weekly hospitalization rate).

Rule:
- During Oct-Mar, when weekly rate rises > 30% WoW, go long-defensive (WMT, CVS equal weight)
  and short LUV (long-history airline; JETS only since 2015), held for 4 weeks (20 trading days).
- Net pnl = 0.5 (WMT_ret + CVS_ret) - LUV_ret, position scaled to 1.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA
)


def fetch_flusurv():
    cache = DATA / "delphi_flusurv_network_all.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    # Pull season-by-season to keep URLs short -- but a single big call works
    url = "https://api.delphi.cmu.edu/epidata/flusurv/?locations=network_all&epiweeks=200940-202520"
    r = requests.get(url, timeout=60)
    j = r.json()
    if j.get("result") != 1:
        raise RuntimeError(f"Delphi error: {j}")
    df = pd.DataFrame(j["epidata"])
    # Convert epiweek to a date (Wednesday of that MMWR week, approx)
    def ew_to_date(ew):
        y, w = ew // 100, ew % 100
        return pd.to_datetime(f"{y}-W{w:02d}-3", format="%G-W%V-%u", errors="coerce")
    df["date"] = df["epiweek"].apply(ew_to_date)
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df.to_parquet(cache)
    return df


def main():
    try:
        flu = fetch_flusurv()
    except Exception as e:
        return mark_failed("J6_cdc_fluview", f"Delphi FluSurv fetch failed: {e}")
    if flu.empty:
        return mark_failed("J6_cdc_fluview", "Empty FluSurv data")

    rate = flu["rate_overall"].astype(float).dropna()
    # WoW pct change
    wow = rate.pct_change()
    in_season = rate.index.month.isin([10, 11, 12, 1, 2, 3])
    triggers = wow[(wow > 0.30) & in_season & (rate > 0.5)].index  # avoid noise at near-zero rates

    px = load_prices(["WMT", "CVS", "LUV"], start="2010-01-01")
    if px.empty or any(c not in px.columns for c in ["WMT", "CVS", "LUV"]):
        return mark_failed("J6_cdc_fluview", "Stock load failed")
    rets = px.pct_change()

    daily_pos_long = pd.Series(0.0, index=rets.index)  # weight on WMT/CVS basket
    daily_pos_short = pd.Series(0.0, index=rets.index)  # weight on LUV (negative)
    n_events = 0
    last_end = None
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 20, len(rets.index))
        for j in range(idx, end_idx):
            daily_pos_long.iloc[j] = 1.0
            daily_pos_short.iloc[j] = -1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1

    basket_ret = 0.5 * (rets["WMT"] + rets["CVS"])
    pnl = (daily_pos_long.shift(1) * basket_ret + daily_pos_short.shift(1) * rets["LUV"]).dropna()
    # Use SPY as benchmark (broad market)
    spy = load_prices(["SPY"], start="2010-01-01")["SPY"].pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy, name="J6 FluView surge -> defensives/short airlines")
    print_metrics(m)
    print(f"\nTrigger events (deduplicated): {n_events}")

    save_result("J6_cdc_fluview", m, extra={
        "status": "ok",
        "rule": ("CDC FluSurv-NET weekly hospitalization rate WoW > +30% during Oct-Mar "
                 "-> long WMT+CVS equal-weight / short LUV for 4 weeks."),
        "mechanism": "Flu surges drive retail/pharmacy spending up (WMT, CVS) and dampen discretionary travel (LUV).",
        "source": "https://api.delphi.cmu.edu/epidata/flusurv/ (CDC FluSurv-NET mirror via Carnegie Mellon DELPHI)",
        "n_events": n_events,
        "caveats": "Used LUV instead of JETS (JETS only since 2015); FluSurv coverage starts ~2009.",
    })


if __name__ == "__main__":
    main()
