"""
G03 WSB / Reddit mention velocity (ApeWisdom).

Plan: ApeWisdom provides current and 24h-ago mention counts. We capture the
current snapshot of all stocks, compute a mention-velocity z-score (today's
mentions vs 24h-ago), and form a contrarian short basket of the top decile
by velocity. We then proxy past performance using *prior* 15 trading days of
returns (this is a degenerate one-shot test because the API does not expose
historical time series).

Honest notes:
- ApeWisdom REST API exposes only a current page snapshot plus the 24h-ago
  field. We do NOT have a 30-day mention history (the docstring claim of
  "last ~30 days" in the task description does not match the public API).
- Because of this we cannot run a panel back-test. We do a one-shot cross-
  sectional study: take the top-10 velocity names today, look at their
  realised returns over the last 15 trading days vs SPY. This shows what a
  contrarian "fade the recent attention spike" trade would have made on the
  one observation we have. Marked small-sample / proof-of-concept.
- If we cannot reach ApeWisdom or fewer than 30 valid tickers come back, we
  mark_failed.
"""
import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


APE_BASE = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"


def fetch_snapshot(max_pages=11):
    rows = []
    for p in range(1, max_pages + 1):
        try:
            r = requests.get(APE_BASE.format(page=p), timeout=20)
            r.raise_for_status()
            data = r.json()
            rows.extend(data.get("results", []))
            if p >= data.get("pages", 1):
                break
            time.sleep(0.15)
        except Exception:
            break
    return pd.DataFrame(rows)


def main():
    try:
        snap = fetch_snapshot()
    except Exception as e:
        return mark_failed("G03_wsb_reddit_velocity", f"ApeWisdom fetch failed: {e}")

    if snap.empty or len(snap) < 30:
        return mark_failed("G03_wsb_reddit_velocity",
                           f"ApeWisdom returned too few rows: {len(snap)}")

    # Save snapshot for reproducibility
    snap_path = DATA / f"apewisdom_snapshot_{pd.Timestamp.utcnow().strftime('%Y%m%d')}.parquet"
    try:
        snap.to_parquet(snap_path)
    except Exception:
        pass

    # Velocity score: (mentions - mentions_24h_ago) / sqrt(mentions_24h_ago + 1)
    snap["mentions"] = pd.to_numeric(snap["mentions"], errors="coerce")
    snap["mentions_24h_ago"] = pd.to_numeric(snap["mentions_24h_ago"], errors="coerce").fillna(0)
    snap = snap.dropna(subset=["mentions"])
    snap = snap[snap["mentions"] >= 5]  # filter noise
    snap["velocity"] = (snap["mentions"] - snap["mentions_24h_ago"]) / np.sqrt(snap["mentions_24h_ago"] + 1.0)

    # Take top decile by velocity (high attention spike). Restrict to tickers
    # that look like single equities (4-5 chars, all letters, no ETFs we know).
    KNOWN_ETFS = {"SPY", "QQQ", "VOO", "VTI", "IWM", "DIA", "SOXL", "SOXX",
                  "SMH", "TQQQ", "SQQQ", "ARKK", "GLD", "SLV", "USO", "UNG",
                  "FXI", "ASHR", "EWH", "EWS", "EEM", "EFA", "TLT", "HYG",
                  "JNK", "VXX", "UVXY", "VGT", "VXUS", "VT", "JUST", "VYM",
                  "SCHD", "QTUM", "JEPI", "JEPQ", "SOXS", "SGOV", "BND",
                  "AGG", "LQD", "SPCX", "SOFI"}
    cand = snap[~snap["ticker"].isin(KNOWN_ETFS)].copy()
    cand = cand[cand["ticker"].str.match(r"^[A-Z]{1,5}$")]

    if len(cand) < 20:
        return mark_failed("G03_wsb_reddit_velocity",
                           f"Too few equity-like tickers after filtering: {len(cand)}")

    top_n = max(5, len(cand) // 10)
    short_basket = cand.nlargest(top_n, "velocity")["ticker"].tolist()

    # Fetch prior 20 trading days of prices for those names + SPY
    universe = list(dict.fromkeys(short_basket + ["SPY"]))
    today = pd.Timestamp.utcnow().normalize()
    start = (today - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    try:
        import yfinance as yf
        px = yf.download(universe, start=start, progress=False, auto_adjust=True)
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"]
        px = px.dropna(how="all")
    except Exception as e:
        return mark_failed("G03_wsb_reddit_velocity", f"Price fetch failed: {e}")

    if "SPY" not in px.columns or px.empty:
        return mark_failed("G03_wsb_reddit_velocity", "Could not load SPY prices")

    spy = px["SPY"].dropna()
    px = px[[c for c in short_basket if c in px.columns]]
    if px.shape[1] < 5:
        return mark_failed("G03_wsb_reddit_velocity",
                           f"Only {px.shape[1]} short-basket prices available")

    # One-shot proof-of-concept: simulate the contrarian rule retroactively
    # by applying it on the LAST 20 trading days as if the velocity ranking
    # had been the same throughout the window. This is an unfair test (a
    # single snapshot) but is the only public ApeWisdom information available.
    px = px.tail(25).copy()
    rets = px.pct_change().dropna(how="all")
    spy_rets = spy.reindex(rets.index).pct_change().dropna()
    # Equal-weight short basket: position = -1 / n on each name; net long SPY
    n = px.shape[1]
    basket_ret = rets.mean(axis=1)  # equal-weight avg return of the short names
    # Short basket vs long SPY (dollar-neutral pair) over the 20-day window
    pair_ret = (spy_rets - basket_ret).reindex(rets.index).dropna()

    if len(pair_ret) < 5:
        return mark_failed("G03_wsb_reddit_velocity",
                           f"Too few aligned days ({len(pair_ret)})")

    # Build a tiny "metrics" record honest about the sample size.
    cum = float((1 + pair_ret).prod() - 1)
    mean = float(pair_ret.mean())
    std = float(pair_ret.std())
    sharpe = float(mean / std * np.sqrt(252)) if std > 0 else 0.0
    hit = float((pair_ret > 0).mean())

    # Use cumulative return as a proxy "cagr" so the print helper doesn't break;
    # honest interpretation is just total return over the tiny window.
    ann_factor = 252.0
    proxy_cagr = (1 + cum) ** (ann_factor / max(len(pair_ret), 1)) - 1 if cum > -1 else -1.0
    eq = (1 + pair_ret).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    metrics = {
        "name": "G03 WSB mention-velocity contrarian (1-snapshot proof-of-concept)",
        "start": str(pair_ret.index[0].date()),
        "end": str(pair_ret.index[-1].date()),
        "n_days": int(len(pair_ret)),
        "cagr": float(proxy_cagr),
        "ann_vol": float(std * np.sqrt(252)),
        "sharpe": sharpe,
        "max_dd": dd,
        "calmar": float(proxy_cagr / abs(dd)) if dd < 0 else None,
        "hit_rate": hit,
        "t_stat": float(mean / (std / math.sqrt(len(pair_ret)))) if std > 0 else 0.0,
        "cumulative_pair_return": cum,
    }
    print_metrics(metrics)
    print("Short basket:", short_basket)

    save_result("G03_wsb_reddit_velocity", metrics, extra={
        "status": "small_sample",
        "rule": ("Identify ApeWisdom top-decile mention-velocity tickers; short "
                 "equal-weight, long SPY equal-weight (dollar-neutral). "
                 "Spec called for hold 15 days after a 5-day wait, but no "
                 "historical velocity feed is publicly available."),
        "data_source": "ApeWisdom (current snapshot + 24h-ago mentions field)",
        "short_basket": short_basket,
        "snapshot_date": str(today.date()),
        "snapshot_universe_size": int(len(snap)),
        "caveats": ("ApeWisdom public API exposes only the current snapshot and "
                    "a 24h-ago count, not a 30-day time series. Result is a "
                    "single-snapshot proof of concept, not a real backtest. "
                    "Sign and magnitude not statistically meaningful."),
    })


if __name__ == "__main__":
    main()
