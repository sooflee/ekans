"""
AC-4 India Union Budget Pre-Run Defense Tilt.

Rule: Long equal-weight {HAL.NS, BEL.NS, BDL.NS, MAZDOCK.NS} from Jan 10 through
Feb 1 open each year (close-of-Jan-10 to open-of-Feb-1, approximated by
close-of-Jan-9 to close-of-Jan-31 using free EOD data).
Skip 2024 (interim/Vote-on-Account year — general election).

Benchmark: NIFTYBEES.NS over same window (NIFTY 50 ETF), fallback to ^NSEI.

Mechanism: Defense capital outlay = fastest-growing major budget line since 2020
(Atmanirbhar Bharat); retail front-runs Feb 1 budget; sell-the-news after.

Source: yfinance, India Union Budget calendar (Feb 1 since 2017).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


TICKERS = ["HAL.NS", "BEL.NS", "BDL.NS", "MAZDOCK.NS"]
# Years (skip 2024 interim — Vote on Account)
YEARS = [2020, 2021, 2022, 2023, 2025, 2026]
# Note: HAL listed Mar 2018, BDL listed Mar 2018, MAZDOCK listed Oct 2020.
# So full basket only available from 2021 onward.


def main():
    try:
        # Try NIFTYBEES first; fall back later if needed
        px = load_prices(TICKERS + ["NIFTYBEES.NS"], start="2017-01-01")
    except Exception as e:
        return mark_failed("AC-4", f"data load failed: {e}")

    print("Columns loaded:", list(px.columns))
    print("First non-NaN per ticker:")
    for c in px.columns:
        s = px[c].dropna()
        if len(s):
            print(f"  {c}: {s.index[0].date()} → {s.index[-1].date()} (n={len(s)})")
        else:
            print(f"  {c}: NO DATA")

    basket_px = px[[c for c in TICKERS if c in px.columns]]
    bench = px["NIFTYBEES.NS"] if "NIFTYBEES.NS" in px.columns else None
    if bench is None or bench.dropna().shape[0] < 100:
        # fallback to ^NSEI
        try:
            ns = load_prices(["^NSEI"], start="2017-01-01")
            bench = ns["^NSEI"]
            print("Using ^NSEI as benchmark")
        except Exception:
            bench = None

    rows = []
    pnl_daily = pd.Series(0.0, index=basket_px.index)
    bench_daily = pd.Series(0.0, index=basket_px.index)

    for y in YEARS:
        # Window: from first trading day ≥ Jan 10 to last trading day ≤ Jan 31
        # (Feb 1 open ≈ close of last trading day of January for daily close data)
        start_ts = pd.Timestamp(f"{y}-01-10")
        end_ts = pd.Timestamp(f"{y}-01-31")
        win = basket_px.index[(basket_px.index >= start_ts) &
                              (basket_px.index <= end_ts)]
        if len(win) < 5:
            print(f"Year {y}: too few trading days ({len(win)}); skip")
            continue
        sub = basket_px.loc[win]
        # Equal-weight basket: drop columns missing at start
        first = sub.iloc[0]
        last = sub.iloc[-1]
        valid = first.notna() & last.notna()
        if valid.sum() == 0:
            print(f"Year {y}: no valid tickers; skip")
            continue
        per_ticker_ret = (last[valid] / first[valid] - 1)
        b = float(per_ticker_ret.mean())
        # benchmark return same window
        if bench is not None:
            b_sub = bench.reindex(win).dropna()
            br = float(b_sub.iloc[-1] / b_sub.iloc[0] - 1) if len(b_sub) >= 2 else np.nan
        else:
            br = np.nan
        # daily series
        daily_eq = sub.loc[:, valid[valid].index].pct_change().mean(axis=1).fillna(0.0)
        pnl_daily.loc[win] = daily_eq.values
        if bench is not None:
            b_sub_full = bench.reindex(win).pct_change().fillna(0.0)
            bench_daily.loc[win] = b_sub_full.values
        rows.append({
            "year": y,
            "start": win[0].date(),
            "end": win[-1].date(),
            "n_days_win": len(win),
            "basket_ret": b,
            "bench_ret": br,
            "names": ",".join(valid[valid].index.tolist()),
            "n_names": int(valid.sum()),
        })

    df = pd.DataFrame(rows)
    print("\nAnnual events:")
    print(df.to_string(index=False))

    if len(df) < 3:
        return mark_failed("AC-4", f"too few annual events ({len(df)})")

    # Event-level stats
    rets = df["basket_ret"].values
    brets = df["bench_ret"].dropna().values
    n = len(rets)
    mean_b = float(np.mean(rets))
    mean_bench = float(np.mean(brets)) if len(brets) else float("nan")
    excess = rets[: len(brets)] - brets if len(brets) == n else None
    mean_excess = float(np.mean(excess)) if excess is not None else float("nan")
    std_b = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    t_b = float(mean_b / (std_b / np.sqrt(n))) if std_b > 0 else 0.0
    t_excess = float(np.mean(excess) /
                     (np.std(excess, ddof=1) / np.sqrt(len(excess)))) \
        if excess is not None and len(excess) > 1 and np.std(excess, ddof=1) > 0 else 0.0

    print(f"\nEvent: N={n}, basket_mean={mean_b*100:.2f}%, "
          f"bench_mean={mean_bench*100:.2f}%, excess={mean_excess*100:.2f}%, "
          f"t_basket={t_b:.2f}, t_excess={t_excess:.2f}, "
          f"hit_pos={np.mean(rets > 0)*100:.1f}%")

    in_pos = (pnl_daily != 0.0)
    pnl_e = pnl_daily[in_pos]
    bench_e = bench_daily[in_pos]
    if len(pnl_e) >= 30:
        m = compute_metrics(pnl_e, benchmark=bench_e,
                            name="AC-4 India Union Budget defense tilt")
    else:
        # event-level summary
        compound = float(np.prod(1.0 + rets) - 1.0)
        years_span = max(1.0, (df["end"].max() - df["start"].min()).days / 365.25)
        m = {
            "name": "AC-4 India Union Budget defense tilt",
            "start": str(df["start"].min()),
            "end": str(df["end"].max()),
            "n_days": int(len(pnl_e)),
            "n_events": int(n),
            "cagr": float((1 + compound) ** (1 / years_span) - 1),
            "ann_vol": float(std_b * np.sqrt(252.0 / df["n_days_win"].mean())),
            "sharpe": float((mean_b / std_b) * np.sqrt(252.0 / df["n_days_win"].mean())) if std_b > 0 else 0.0,
            "max_dd": float(np.min(rets)),
            "calmar": None,
            "hit_rate": float(np.mean(rets > 0)),
            "t_stat": t_b,
        }
    print_metrics(m)

    save_result("AC-4", m, extra={
        "status": "ok",
        "rule": "Long EW {HAL.NS,BEL.NS,BDL.NS,MAZDOCK.NS} Jan 10 → Feb 1 open each year.",
        "mechanism": "Atmanirbhar defense capex growth; pre-budget retail front-running.",
        "source": "yfinance NSE; budget calendar (Feb 1 since 2017); skip 2024 (Vote on Account).",
        "n_events": int(n),
        "mean_basket_ret": mean_b,
        "mean_bench_ret": mean_bench,
        "mean_excess_ret": mean_excess,
        "t_stat_basket": t_b,
        "t_stat_excess": t_excess,
        "annual_rets": [float(x) for x in rets],
        "annual_bench_rets": [float(x) for x in df["bench_ret"].values],
        "annual_years": [int(y) for y in df["year"].values],
    })


if __name__ == "__main__":
    main()
