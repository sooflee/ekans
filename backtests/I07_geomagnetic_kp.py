"""
I-7 Geomagnetic storm "mood" trade.

Idea (Krivelyova-Robotti 2003 "Playing the field"): geomagnetic storms have
been linked in some papers to lower next-day equity returns ("bad mood").
Test: when 5-day rolling Kp is in the top decile, exit equity for 10 trading
days; otherwise hold SPY. Compare to SPY buy-and-hold.

Data: GFZ Potsdam Kp index since 1932 (3-hourly).
   https://kp.gfz.de/app/files/Kp_ap_since_1932.txt
Aggregated to daily mean Kp.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import requests
import pandas as pd
import numpy as np

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)

KP_URL = "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt"


def load_kp():
    cache_fp = DATA / "kp_gfz_1932.parquet"
    if cache_fp.exists():
        df = pd.read_parquet(cache_fp)
        latest = df.index.max()
        if (pd.Timestamp.utcnow().tz_localize(None) - latest).days < 7:
            return df["kp_daily"]

    try:
        r = requests.get(KP_URL, timeout=60)
        r.raise_for_status()
    except Exception as e:
        if cache_fp.exists():
            return pd.read_parquet(cache_fp)["kp_daily"]
        raise

    rows = []
    for ln in r.text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 8:
            continue
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            kp = float(parts[7])
        except Exception:
            continue
        if kp < 0:
            continue
        rows.append((dt_date(y, m, d), kp))
    df_raw = pd.DataFrame(rows, columns=["date", "kp"])
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    daily = df_raw.groupby("date")["kp"].mean()
    daily.name = "kp_daily"
    daily.to_frame().to_parquet(cache_fp)
    return daily


def dt_date(y, m, d):
    import datetime as _dt
    return _dt.date(y, m, d)


def main():
    try:
        kp = load_kp()
    except Exception as e:
        return mark_failed("I07_geomagnetic_kp", f"Kp load failed: {e}")

    if kp is None or len(kp) < 1000:
        return mark_failed("I07_geomagnetic_kp", "Kp series too short")

    # 5-day rolling mean Kp
    kp_roll = kp.rolling(5).mean()

    # Top-decile threshold computed in a *rolling* way (5y window) to avoid
    # look-ahead from a full-sample quantile.
    threshold = kp_roll.rolling(252 * 5, min_periods=252).quantile(0.90)
    storm_signal = kp_roll > threshold

    # Position: hold SPY (=1) unless storm has fired in the last 10 trading days.
    spy = load_prices(["SPY"], start="1995-01-01")["SPY"].dropna()
    rets = spy.pct_change()

    storm_daily = storm_signal.reindex(spy.index, method="ffill").fillna(False)
    # When storm fires today, we exit for the next 10 trading days.
    # Build position: pos[t] = 0 if any storm in [t-10, t-1], else 1.
    exit_flag = storm_daily.shift(1).rolling(10, min_periods=1).max().fillna(0)
    pos = 1.0 - exit_flag.clip(0, 1)

    pnl = (pos.shift(1) * rets).dropna()
    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="I-7 Kp storm exit -> SPY")
    print_metrics(m)
    print(f"\nFraction days flat: {1 - pos.mean():.2%}")

    save_result("I07_geomagnetic_kp", m, extra={
        "status": "ok",
        "rule": ("Exit SPY for the 10 trading days after the 5-day rolling Kp "
                 "crosses above its trailing 5y 90th percentile."),
        "data_source": "GFZ Potsdam Kp index (3-hourly, averaged to daily)",
        "fraction_days_flat": float(1 - pos.mean()),
        "source_paper": "Krivelyova & Robotti 2003.",
    })


if __name__ == "__main__":
    main()
