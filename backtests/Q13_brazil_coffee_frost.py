"""
Q13 Brazil coffee frost (Minas Gerais / SP / PR) -> long KC=F.

NOAA NCEI GHCN-Daily by_station CSVs. Stations identified via ghcnd-inventory.txt
restricted to Brazilian coffee belt (lat -25 to -18, lon -52 to -42) with TMIN
coverage spanning 2005-2020.

Rule: For June-August (Brazilian winter), any station Tmin < 2 degC ->
long KC=F next session for 30 trading days. Non-overlapping events.

Note: Tmin = 2 deg at high elevation is enough to imply localized frost in valleys.
"""
import sys
import io
import gzip
import urllib.request
import socket
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA
)


# pre-vetted via ghcnd-inventory.txt
STATIONS = [
    "BR001943012",  # Sete Lagoas region MG
    "BR001946002",  # Patrocinio MG
    "BR002245032",  # Campos do Jordao SP (high alt)
    "BR002351003",  # Londrina PR
    "BR00C4-0630",  # Ribeirao Preto SP
    "BR00D6-0010",  # Aguas de Sao Pedro / Bauru SP
    "BR00E2-0220",  # Sao Jose dos Campos
    "BR00E3-0520",  # Sao Paulo
    "BR00E3-1520",  # Sao Paulo
    "BR00E4-1230",  # Sorocaba SP
    "BRM00083566",  # Belo Horizonte
    "BRM00083587",  # Belo Horizonte / PampulhaTimer
    "BRM00083746",  # Rio de Janeiro Galeao
]

URL_TMPL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{sid}.csv.gz"


def fetch_station(sid, timeout=25):
    cache = DATA / f"ghcnd_{sid}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    url = URL_TMPL.format(sid=sid)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    socket.setdefaulttimeout(timeout)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"  {sid}: download failed ({e})")
        return None
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  {sid}: decompress failed ({e})")
        return None
    df = pd.read_csv(io.StringIO(text), header=None,
                     names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obs_time"],
                     dtype=str)
    if df.empty:
        return None
    df = df[df["element"] == "TMIN"]
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["tmin_c"] = pd.to_numeric(df["value"], errors="coerce") / 10.0
    df = df.dropna(subset=["date", "tmin_c"])[["date", "tmin_c"]].set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(cache)
    return df


def main():
    found = {}
    for sid in STATIONS:
        try:
            d = fetch_station(sid)
        except Exception as e:
            print(f"  {sid} exception: {e}")
            d = None
        if d is not None and len(d) > 365:
            found[sid] = d
            print(f"  {sid}: {len(d)} TMIN obs, "
                  f"{d.index.min().date()}..{d.index.max().date()}, "
                  f"min={d['tmin_c'].min():.1f}C")
    if not found:
        return mark_failed("Q13_brazil_coffee_frost", "no GHCN-D Brazil TMIN stations loaded")

    combo = pd.concat({sid: d["tmin_c"] for sid, d in found.items()}, axis=1)
    daily_min = combo.min(axis=1).dropna()
    mask_winter = daily_min.index.month.isin([6, 7, 8])
    mask_years = (daily_min.index.year >= 2000) & (daily_min.index.year <= 2024)
    winter = daily_min[mask_winter & mask_years]

    THRESH = 2.0
    triggers = winter[winter < THRESH].index
    print(f"\nWinter Tmin obs: {len(winter)}; min observed: {winter.min():.2f} C")
    print(f"Days < {THRESH}C: {len(triggers)}")

    try:
        kc = load_prices(["KC=F"], start="2000-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("Q13_brazil_coffee_frost", f"KC=F load failed: {e}")
    rets = kc.pct_change()

    pos = pd.Series(0.0, index=rets.index)
    n_events = 0
    last_end = None
    event_dates = []
    for d in triggers:
        nxt = rets.index[rets.index > d]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        if last_end is not None and start <= last_end:
            continue
        idx = rets.index.get_loc(start)
        end_idx = min(idx + 30, len(rets.index))
        for j in range(idx, end_idx):
            pos.iloc[j] = 1.0
        last_end = rets.index[end_idx - 1]
        n_events += 1
        event_dates.append(str(start.date()))

    if n_events == 0:
        return mark_failed("Q13_brazil_coffee_frost",
                           f"no qualifying triggers (min winter Tmin={winter.min():.2f}C, threshold={THRESH}C)",
                           extra={"stations_used": list(found.keys()), "n_winter_obs": int(len(winter))})

    pnl = (pos.shift(1) * rets).dropna()
    pnl = pnl.loc[pnl.ne(0).cummax()]
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Q13 Brazil frost -> long KC=F 30d")
    m["n_events"] = n_events
    print(f"\nEvents: {n_events} (non-overlap)")
    print(f"First few entries: {event_dates[:5]}")
    print_metrics(m)

    save_result("Q13_brazil_coffee_frost", m, extra={
        "status": "ok",
        "rule": f"If any Brazilian coffee-region GHCN-D station Tmin < {THRESH} degC during Jun-Aug, long KC=F next session for 30 trading days; non-overlapping events.",
        "mechanism": "Frost in Minas Gerais / Parana coffee belt damages arabica buds and trees -> forward-supply shock -> arabica futures rally.",
        "source": "NOAA NCEI GHCN-Daily by_station CSVs (stations vetted via ghcnd-inventory.txt for coffee-belt coverage); yfinance KC=F.",
        "n_events": n_events,
        "stations_used": list(found.keys()),
        "first_events": event_dates[:5],
    })


if __name__ == "__main__":
    main()
