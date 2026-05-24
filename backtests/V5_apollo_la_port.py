"""
V5 Apollo / LA Port containers regime
Scrapes Port of Los Angeles monthly TEU statistics
(portoflosangeles.org). Signal originally specified weekly 4w-MA YoY < -15%
for 3 consecutive weeks; Port of LA publishes monthly totals only, so we use
the closest equivalent: 3 consecutive months with YoY < -15% in total TEUs.
Rule:
  Track monthly total TEUs. Compute YoY change.
  When YoY < -15% for 3 consecutive months: open the trade for 90 trading days.
  Trade = long XLP + XLU, short XLI + XLY + XTN (50/50 each leg, market-neutral).
Mechanism (Apollo / Torsten Slok): West-Coast import volume is a leading
indicator for US goods consumption + industrial demand. A collapse in LA imports
typically front-runs an industrial / discretionary slowdown by ~2 quarters,
favouring defensives (staples, utilities) over cyclicals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (
    load_prices, compute_metrics, print_metrics,
    save_result, mark_failed,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA_DIR / "la_port_teu_monthly.parquet"

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]


def fetch_year(year):
    url = f"https://www.portoflosangeles.org/Business/statistics/Container-Statistics/Historical-TEU-Statistics-{year}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return []
    cells = re.findall(r'<td[^>]*>([^<]+)</td>', r.text)
    cells = [c.strip().replace("&nbsp;","").replace(",","") for c in cells]
    rows = []
    i = 0
    while i < len(cells):
        if cells[i] in MONTHS:
            month = cells[i]
            # Per-year page layout: month, loaded inbound, empty inbound, total inbound, loaded outbound, empty outbound, total outbound, GRAND TOTAL, YoY%
            # We grab month index + GRAND TOTAL (cells[i+7])
            try:
                total = float(cells[i+7])
                rows.append((year, MONTHS.index(month)+1, total))
            except Exception:
                pass
            i += 8
        else:
            i += 1
    return rows


def build_teu_history(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    rows = []
    for y in range(1995, 2027):
        try:
            rows.extend(fetch_year(y))
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["year","month","teu"])
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1)) + pd.offsets.MonthEnd(0)
    df = df.sort_values("date").set_index("date")[["teu"]]
    df.to_parquet(CACHE)
    return df


def main():
    try:
        teu = build_teu_history()
    except Exception as e:
        return mark_failed("V5_apollo_la_port", f"scrape failed: {e}")
    if teu is None or teu.empty:
        return mark_failed("V5_apollo_la_port", "no TEU data scraped")

    yoy = teu["teu"].pct_change(12)
    # Flag months with yoy < -15
    trig = yoy < -0.15
    # 3 consecutive months
    trig3 = trig & trig.shift(1).fillna(False) & trig.shift(2).fillna(False)

    # Load ETF universe
    try:
        px = load_prices(["XLP","XLU","XLI","XLY","XTN","SPY"], start="1998-12-22")
    except Exception as e:
        return mark_failed("V5_apollo_la_port", f"ETF load: {e}")
    px = px.dropna()
    if px.empty:
        return mark_failed("V5_apollo_la_port", "no ETF overlap")

    rets = px[["XLP","XLU","XLI","XLY","XTN"]].pct_change()

    # Build daily on/off mask: 90 trading days after each trigger month-end
    trigger_dates = trig3[trig3].index
    daily_on = pd.Series(False, index=px.index)
    for td in trigger_dates:
        # find first trading day after the month-end
        future = px.index[px.index > td]
        if len(future) == 0:
            continue
        first = future[0]
        first_pos = px.index.get_loc(first)
        last_pos = min(first_pos + 90, len(px.index)-1)
        daily_on.iloc[first_pos:last_pos+1] = True

    # Position frame
    pos = pd.DataFrame(0.0, index=px.index, columns=["XLP","XLU","XLI","XLY","XTN"])
    pos.loc[daily_on, "XLP"] = 0.5
    pos.loc[daily_on, "XLU"] = 0.5
    pos.loc[daily_on, "XLI"] = -1/3
    pos.loc[daily_on, "XLY"] = -1/3
    pos.loc[daily_on, "XTN"] = -1/3

    pnl = (pos.shift(1) * rets).sum(axis=1).dropna()
    bench = px["SPY"].pct_change().reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="V5 Apollo LA Port containers")
    print_metrics(m)
    n_trigs = int(trig3.sum())
    n_on = int(daily_on.sum())
    save_result("V5_apollo_la_port", m, extra={
        "status": "ok",
        "rule": "3 consecutive months of LA Port TEU YoY < -15% -> long XLP+XLU vs short XLI+XLY+XTN (50/50 vs 1/3 each) for 90 trading days.",
        "mechanism": "West-Coast imports are a leading indicator of US goods consumption / industrial demand; collapse favours defensives over cyclicals ~2 quarters out.",
        "source": "Apollo / Torsten Slok, YouTube interview round 2 (Phase 1V). Data: portoflosangeles.org monthly TEU pages.",
        "substitution": "Weekly 4w-MA replaced with monthly TEU YoY (Port of LA does not publish weekly).",
        "n_triggers": n_trigs,
        "n_on_days": n_on,
    })


if __name__ == "__main__":
    main()
