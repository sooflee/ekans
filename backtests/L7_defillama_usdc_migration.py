"""
L7 DefiLlama USDC supply share on Ethereum -> long ETH-USD.

Rule:
- Pull daily USDC circulating supply on Ethereum and total USDC supply
  across all chains from DefiLlama.
- Compute the WoW change in (USDC_Ethereum / USDC_Total). When the ratio
  grows by > 1 percentage point WoW (i.e. USDC migrating *back to* /
  growing faster on Ethereum than other L2s/chains), go long ETH-USD for
  14 calendar days.

Mechanism:
- A relative rebound in USDC share on Ethereum mainnet often coincides
  with rising on-chain DeFi activity / DEX volumes (gas demand), which
  is a tailwind for ETH price.

Source: https://stablecoins.llama.fi/stablecoincharts/<chain>?stablecoin=2
        (stablecoin id 2 = USDC)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import requests
import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


def pull_series(chain):
    cache = DATA / f"usdc_{chain}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = (f"https://stablecoins.llama.fi/stablecoincharts/{chain}?stablecoin=2"
           if chain != "all" else
           "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=2")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = []
    for item in r.json():
        ts = int(item.get("date"))
        circ = item.get("totalCirculating", {}).get("peggedUSD")
        if circ is None:
            continue
        rows.append({"date": pd.to_datetime(ts, unit="s"), "supply": float(circ)})
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").set_index("date")
    df.to_parquet(cache)
    return df


def main():
    try:
        eth = pull_series("Ethereum")
        tot = pull_series("all")
    except Exception as e:
        return mark_failed("L7_defillama_usdc_migration", f"DefiLlama fetch: {e}")
    if eth.empty or tot.empty:
        return mark_failed("L7_defillama_usdc_migration", "Empty DefiLlama dataset")

    df = pd.concat({"eth": eth["supply"], "tot": tot["supply"]}, axis=1).dropna()
    df["share"] = df["eth"] / df["tot"]
    # weekly resample
    weekly = df.resample("W-FRI").last().dropna()
    weekly["share_dlt"] = weekly["share"].diff()
    # trigger when share grew by > 1 pp WoW
    triggers = weekly[weekly["share_dlt"] > 0.01].index

    # ETH-USD price (yfinance)
    try:
        px = load_prices(["ETH-USD", "BTC-USD"], start="2019-01-01")
    except Exception as e:
        return mark_failed("L7_defillama_usdc_migration", f"price fetch: {e}")
    rets = px.pct_change()

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        i = rets.index.get_loc(start)
        end = min(i + 14, len(rets.index))
        for j in range(i, end):
            pos.iloc[j] = 1.0
        last_end = rets.index[end - 1]
        n_events += 1

    pnl = (pos.shift(1) * rets["ETH-USD"]).dropna()
    bench = rets["BTC-USD"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="L7 USDC Ethereum-share rise -> long ETH")
    print_metrics(m)
    print(f"\nWeeks: {len(weekly)} ; triggers: {len(triggers)} ; events: {n_events}")

    save_result("L7_defillama_usdc_migration", m, extra={
        "status": "ok",
        "rule": ("Weekly (USDC_Ethereum / USDC_Total) rises > 1 pp WoW -> long ETH-USD for 14 calendar days. "
                 "Dedup overlapping events."),
        "mechanism": "USDC re-concentrating on Ethereum signals revived mainnet DeFi/gas demand, supportive of ETH.",
        "source": "DefiLlama stablecoincharts (USDC, id=2)",
        "n_events": int(n_events),
        "n_weeks": int(len(weekly)),
    })


if __name__ == "__main__":
    main()
