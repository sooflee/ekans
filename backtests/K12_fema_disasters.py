"""
K12 FEMA Disaster Declarations — rolling 30d major disaster count

Rule: When the rolling 30-day count of major disaster declarations (DR-type,
PA program declared) exceeds 5, short an equal-weight basket of P&C insurers
(TRV + ALL + CB) for 20 trading days.

Mechanism: Concentrated disaster periods elevate near-term claim expectations
and pressure P&C earnings; the market often re-prices these names on a 2-4
week horizon.

Source: FEMA OpenFEMA v2 API (DisasterDeclarationsSummaries).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd
import requests

from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed, DATA,
)


FEMA = ("https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
        "?$select=disasterNumber,declarationDate,declarationType,paProgramDeclared,incidentType,state"
        "&$filter=declarationDate ge '2000-01-01T00:00:00.000Z'"
        "&$orderby=declarationDate"
        "&$top={top}&$skip={skip}")


def pull_fema():
    cache = DATA / "fema_disasters.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows, skip, top = [], 0, 5000
    while True:
        url = FEMA.format(top=top, skip=skip)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json().get("DisasterDeclarationsSummaries", [])
        if not data:
            break
        rows.extend(data)
        if len(data) < top:
            break
        skip += top
    df = pd.DataFrame(rows)
    df.to_parquet(cache)
    return df


def main():
    try:
        df = pull_fema()
    except Exception as e:
        return mark_failed("K12_fema_disasters", f"FEMA fetch: {e}")

    if df.empty:
        return mark_failed("K12_fema_disasters", "Empty FEMA dataset.")

    df["declarationDate"] = pd.to_datetime(df["declarationDate"], utc=True).dt.tz_localize(None)
    # Major disaster = DR type with Public Assistance program declared
    major = df[(df["declarationType"] == "DR") & (df["paProgramDeclared"] == True)].copy()
    # Unique disaster per disasterNumber (multiple county/area rows)
    major = major.groupby("disasterNumber").first().reset_index()
    major = major.sort_values("declarationDate")

    daily = major.set_index("declarationDate").assign(n=1)["n"].resample("D").sum().fillna(0)
    rolling30 = daily.rolling(30).sum()

    try:
        px = load_prices(["TRV", "ALL", "CB", "SPY"], start="2001-01-01")
    except Exception as e:
        return mark_failed("K12_fema_disasters", f"price fetch: {e}")

    rets = daily_returns(px)
    basket = rets[["TRV", "ALL", "CB"]].mean(axis=1).dropna()

    # Build signal: short for 20 trading days from any trigger date
    trigger_dates = rolling30[rolling30 > 5].index
    sig = pd.Series(0.0, index=basket.index)
    n_trig = 0
    for d in trigger_dates:
        i = basket.index.searchsorted(d)
        if i >= len(basket):
            continue
        sig.iloc[i:i+20] += 1
        n_trig += 1
    pos = -(sig > 0).astype(float)  # short
    pnl = pos.shift(1) * basket
    pnl = pnl.dropna()

    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K12 FEMA disasters short P&C")
    print_metrics(m)
    print(f"n_triggers={n_trig}, days_short={int((pos<0).sum())}/{len(pos)}")

    save_result("K12_fema_disasters", m, extra={
        "status": "ok",
        "rule": "Rolling 30d major disaster (DR, PA declared) count > 5 → short equal-weight TRV+ALL+CB for 20 trading days.",
        "mechanism": "Catastrophe clusters pressure P&C insurer earnings.",
        "source": "OpenFEMA v2 DisasterDeclarationsSummaries",
        "n_triggers": int(n_trig),
        "exposure_pct": float((pos != 0).mean()),
    })


if __name__ == "__main__":
    main()
