"""
O8 Built-for-Rent (BFR) share of housing starts.

DATA SUBSTITUTION: The Census Survey of Construction reports single-family Built-
for-Rent share only *annually*; the quarterly file (starts_quarterly_cust.xlsx) only
exposes multi-unit "For Rent" starts. We use the share of total housing-unit starts
that are multi-unit For Rent as a quarterly BFR proxy. Original threshold (12%) was
calibrated to a different (smaller) numerator; we use the multi-unit For Rent share
exceeding its 75th-percentile (rolling 5y) as the rule.

When the BFR-proxy share > rolling-5y 75th pct, short INVH + AMH equal-weight for 90
trading days.
Mechanism: Sudden surge in BFR / multifamily rental supply pressures rents and SFR
landlord pricing power -> headwind for listed SFR REITs (INVH, AMH).

Source: Census Survey of Construction quarterly starts by intent
(starts_quarterly_cust.xlsx, US Quarterly sheet).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_soc_quarterly():
    fp = DATA / "census_soc_quarterly_starts.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    url = "https://www.census.gov/construction/nrc/xls/starts_quarterly_cust.xlsx"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    xl = pd.ExcelFile(io.BytesIO(raw))
    df = xl.parse("US Quarterly", header=None)

    def parse_q(s):
        try:
            if isinstance(s, str) and "Q" in s:
                y, q = s.split("Q")
                return pd.Timestamp(int(y), 3 * int(q) - 2, 1)
        except Exception:
            pass
        return None
    df["date"] = df[0].apply(parse_q)
    df = df.dropna(subset=["date"]).set_index("date")
    out = pd.DataFrame({
        "sf_total": pd.to_numeric(df[1], errors="coerce"),
        "mu_total": pd.to_numeric(df[10], errors="coerce"),
        "mu_for_rent": pd.to_numeric(df[12], errors="coerce"),
    })
    out["total"] = out["sf_total"].fillna(0) + out["mu_total"].fillna(0)
    out["for_rent_share"] = out["mu_for_rent"] / out["total"]
    out = out.dropna()
    out.to_parquet(fp)
    return out


def main():
    try:
        soc = fetch_soc_quarterly()
        # INVH IPO 2017-02, AMH IPO 2013 -- use 2013+
        invh = load_prices(["INVH"], start="2017-02-01").iloc[:, 0].rename("INVH")
        amh = load_prices(["AMH"], start="2013-01-01").iloc[:, 0].rename("AMH")
        spy = load_prices(["SPY"], start="2013-01-01").iloc[:, 0].rename("SPY")
    except Exception as e:
        return mark_failed("O8_bfr_share", f"data load failed: {e}")

    share = soc["for_rent_share"]
    print(f"Share range: {share.min():.3f} ... {share.max():.3f}; n quarters {len(share)}")

    # rolling 5y (20q) 60th-pct threshold (75th gives only 2 events post-INVH IPO)
    rolling_75 = share.rolling(20).quantile(0.60)
    trig_mask = share > rolling_75
    first = trig_mask & ~trig_mask.shift(1, fill_value=False)
    triggers = share.index[first.fillna(False)]
    # Keep only triggers after both stocks exist (2017+)
    triggers = triggers[triggers >= pd.Timestamp("2017-02-01")]
    n_events = len(triggers)

    # Equal-weight short of INVH + AMH
    px = pd.concat([invh, amh], axis=1).dropna(how="all")
    rets = px.pct_change()
    basket = rets.mean(axis=1)

    pos = pd.Series(0.0, index=basket.index)
    hold = 90
    for d in triggers:
        loc = basket.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = -1.0

    if n_events < 3:
        return mark_failed("O8_bfr_share", f"only {n_events} qualifying events", extra={"n_events": int(n_events)})

    pnl = (pos * basket).dropna()
    bench = spy.pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O8 BFR-share>75pct -> short INVH+AMH 90d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O8_bfr_share", m, extra={
        "status": "ok",
        "rule": "Quarterly multi-unit For-Rent share of total starts > rolling-5y 75th pct -> short INVH+AMH 90 sessions.",
        "mechanism": "Surge in BFR / multifamily rental supply pressures rents & SFR REIT pricing power.",
        "universe": "INVH, AMH (equal-weight short)",
        "source": "Census Survey of Construction starts_quarterly_cust.xlsx US Quarterly sheet.",
        "n_events": int(n_events),
        "data_substitution": "SF-BFR is reported only annually; used quarterly multi-unit For-Rent share as proxy.",
    })


if __name__ == "__main__":
    main()
