"""
K9 DoD Treasury Daily Outlays — long ITA on big-DoD-spending days.

Rule: Sum daily DoD-related withdrawals from the Treasury Daily Statement
(deposits/withdrawals operating cash table). On days where the total DoD
outlay > $5B, go long ITA (aerospace & defense ETF) for 10 trading days.

Sub-categories summed: 'Dept of Defense (DoD) - misc',
'DoD - Military Active Duty Pay', 'DoD - Military Retirement',
'IAP - Foreign Military Sales'.

Mechanism: A burst of DoD outlays signals contract awards/draws and may
front-run reported revenue at primes (LMT, RTX, NOC, GD, BA).

Source: api.fiscaldata.treasury.gov DTS deposits_withdrawals_operating_cash.
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


DOD_CATS = {
    'Dept of Defense (DoD) - misc',
    'DoD - Military Active Duty Pay',
    'DoD - Military Retirement',
    'IAP - Foreign Military Sales',
}


def pull_dts(start="2005-10-01"):
    cache = DATA / "treasury_dts_dod.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    all_rows = []
    # Page through year-by-year to keep response sizes bounded
    base = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
            "accounting/dts/deposits_withdrawals_operating_cash"
            "?fields=record_date,transaction_catg,transaction_today_amt"
            "&filter=transaction_type:eq:Withdrawals,record_date:gte:{s},record_date:lt:{e}"
            "&page[size]=10000&page[number]={p}")
    years = list(range(2005, 2027))
    for y in years:
        s = f"{y}-01-01"
        e = f"{y+1}-01-01"
        page = 1
        while True:
            url = base.format(s=s, e=e, p=page)
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                j = r.json()
            except Exception as ex:
                print(f"warn {y} p{page}: {ex}")
                break
            data = j.get("data", [])
            if not data:
                break
            for row in data:
                if row["transaction_catg"] in DOD_CATS:
                    all_rows.append(row)
            if len(data) < 10000:
                break
            page += 1
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["amt"] = pd.to_numeric(df["transaction_today_amt"], errors="coerce")
    df.to_parquet(cache)
    return df


def main():
    try:
        df = pull_dts()
    except Exception as e:
        return mark_failed("K9_dod_treasury_outlays", f"DTS fetch: {e}")

    if df.empty:
        return mark_failed("K9_dod_treasury_outlays", "DTS empty.")

    daily_dod = df.groupby("record_date")["amt"].sum()  # in $millions
    daily_dod_b = daily_dod / 1000.0  # $billions

    try:
        px = load_prices(["ITA", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed("K9_dod_treasury_outlays", f"price fetch: {e}")

    rets = daily_returns(px)
    ita = rets["ITA"].dropna()

    trig = daily_dod_b[daily_dod_b > 5].index
    sig = pd.Series(0.0, index=ita.index)
    n_trig = 0
    for d in trig:
        i = ita.index.searchsorted(d)
        if i >= len(ita):
            continue
        sig.iloc[i:i+10] += 1
        n_trig += 1
    pos = (sig > 0).astype(float)
    pnl = pos.shift(1) * ita
    pnl = pnl.dropna()

    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K9 DoD outlays long ITA")
    print_metrics(m)
    print(f"n_trig={n_trig}, exposure={pos.mean():.2%}")

    save_result("K9_dod_treasury_outlays", m, extra={
        "status": "ok",
        "rule": "Daily DoD withdrawals (sum of 4 DTS categories) > $5B → long ITA for 10 trading days.",
        "mechanism": "Spike in DoD outlays may front-run defense-contractor revenue.",
        "source": "Treasury Fiscal Data DTS deposits_withdrawals_operating_cash",
        "n_triggers": int(n_trig),
        "exposure_pct": float(pos.mean()),
    })


if __name__ == "__main__":
    main()
