"""
K13 SBA 7(a) Weekly Lending — short IWM on lending crunches.

Data: SBA FOIA datasets (foia-7a-fy*.csv) — every approved 7(a) loan with
approval date and gross amount. Available 1991-present.

Rule: Aggregate gross approval $ volume by ISO-week. When WoW drop > 25%,
short IWM (small caps) for 20 trading days starting Monday following the
week.

Mechanism: SBA 7(a) is the main credit pipeline to US small businesses;
a sharp WoW contraction signals tightening main-street credit, which
pressures small-cap earnings.

Source: data.sba.gov dataset "7-a-504-foia".
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


CSVS = [
    # We use only fy2010-present panels to keep memory bounded and align with IWM history.
    ("https://data.sba.gov/en/dataset/0ff8e8e9-b967-4f4e-987c-6ac78c575087/resource/"
     "3f838176-6060-44db-9c91-b4acafbcb28c/download/foia-7a-fy2010-fy2019-asof-260331.csv"),
    ("https://data.sba.gov/en/dataset/0ff8e8e9-b967-4f4e-987c-6ac78c575087/resource/"
     "d67d3ccb-2002-4134-a288-481b51cd3479/download/foia-7a-fy2020-present-asof-260331.csv"),
]


def load_sba():
    cache = DATA / "sba_7a_weekly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for u in CSVS:
        fp = DATA / u.split("/")[-1]
        if not fp.exists():
            r = requests.get(u, timeout=180)
            r.raise_for_status()
            fp.write_bytes(r.content)
        # Read only the needed cols, low_memory
        df = pd.read_csv(fp, usecols=lambda c: c.lower() in {"approvaldate", "grossapproval"},
                         low_memory=False, dtype=str)
        df.columns = [c.lower() for c in df.columns]
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["approvaldate"] = pd.to_datetime(raw["approvaldate"], errors="coerce")
    raw["grossapproval"] = pd.to_numeric(raw["grossapproval"], errors="coerce")
    raw = raw.dropna(subset=["approvaldate", "grossapproval"])
    weekly = raw.set_index("approvaldate")["grossapproval"].resample("W-FRI").sum()
    weekly = weekly.to_frame("vol")
    weekly.to_parquet(cache)
    return weekly


def main():
    try:
        wk = load_sba()
    except Exception as e:
        return mark_failed("K13_sba_7a_loans", f"SBA fetch: {e}")
    if wk.empty:
        return mark_failed("K13_sba_7a_loans", "SBA empty.")

    vol = wk["vol"]
    wow = vol.pct_change()

    try:
        px = load_prices(["IWM", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed("K13_sba_7a_loans", f"price fetch: {e}")
    rets = daily_returns(px)
    iwm = rets["IWM"].dropna()

    sig = pd.Series(0.0, index=iwm.index)
    n_trig = 0
    for ts, change in wow.dropna().items():
        if change < -0.25:
            i = iwm.index.searchsorted(ts + pd.Timedelta(days=3))  # next Mon
            if i >= len(iwm):
                continue
            sig.iloc[i:i+20] += 1
            n_trig += 1
    pos = -(sig > 0).astype(float)
    pnl = pos.shift(1) * iwm
    pnl = pnl.dropna()
    bench = rets["SPY"].reindex(pnl.index).dropna()
    m = compute_metrics(pnl, benchmark=bench, name="K13 SBA 7a WoW drop short IWM")
    print_metrics(m)
    print(f"weeks={len(vol)}, n_trig={n_trig}, exposure_short={(pos<0).mean():.2%}")

    save_result("K13_sba_7a_loans", m, extra={
        "status": "ok",
        "rule": "Weekly SBA 7(a) gross approval $ volume drops > 25% WoW → short IWM 20 trading days from next Mon.",
        "mechanism": "SBA 7(a) is the primary credit pipeline to small business; contractions pressure small-cap earnings.",
        "source": "data.sba.gov 7-a-504-foia (foia-7a-fy2010-fy2019, fy2020-present)",
        "weeks_observed": int(len(vol)),
        "n_triggers": int(n_trig),
        "exposure_pct": float((pos != 0).mean()),
    })


if __name__ == "__main__":
    main()
