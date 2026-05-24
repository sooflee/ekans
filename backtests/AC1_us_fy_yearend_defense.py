"""
AC-1 US Fiscal-Year-End Contractor Obligation Surge.

Rule: Long equal-weight {LMT, GD, LDOS, BAH, SAIC} from the 5th trading day of
September through Sep 30 each year. Baseline = full window (no early-exit gate).

Mechanism: ~16-18% of annual DoD contract obligations land in the final fiscal
month as program offices burn unobligated balances; federal-IT primes benefit
because revenue recognizes on task-order issuance.

Benchmark: SPY same-window return each year.

Output: build annual event-return series, compute metrics, save.
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


TICKERS = ["LMT", "GD", "LDOS", "BAH", "SAIC"]


def september_window(year, idx):
    """Return (start_date, end_date) trading-day window: 5th TD of Sep through Sep 30."""
    sep = idx[(idx.year == year) & (idx.month == 9)]
    if len(sep) < 6:
        return None
    start = sep[4]   # 5th trading day (0-indexed = 4)
    end = sep[-1]    # last trading day of September
    return start, end


def basket_return(prices, start, end):
    """Equal-weight return of basket between start and end (inclusive close-to-close)."""
    sub = prices.loc[start:end].dropna(how="all")
    if len(sub) < 2:
        return np.nan
    first = sub.iloc[0]
    last = sub.iloc[-1]
    ret_each = (last / first - 1).dropna()
    if len(ret_each) == 0:
        return np.nan
    return float(ret_each.mean())


def main():
    try:
        px = load_prices(TICKERS + ["SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed("AC-1", f"data load failed: {e}")

    # Restrict to dates where SPY trades; build basket DataFrame
    px = px.sort_index()
    basket_px = px[TICKERS]
    spy_px = px["SPY"]

    # Find first year where all 5 names have data in September
    sep_first_seen = {}
    for t in TICKERS:
        s = basket_px[t].dropna()
        if len(s) == 0:
            continue
        sep_first_seen[t] = s.index[0].year
    print(f"First year present per ticker: {sep_first_seen}")

    start_year = max(sep_first_seen.values()) if sep_first_seen else 2007
    # SAIC IPO'd Sep 2013 (post-Leidos split). Use 2014 as first full-basket year.
    # But to be honest about composition: also report a "available-names" version.
    print(f"Computed start_year (all 5 present in Sept): {start_year}")

    idx = px.index
    end_year = idx[-1].year - 1 if idx[-1].month < 10 else idx[-1].year

    rows = []
    for y in range(2007, end_year + 1):
        win = september_window(y, idx)
        if win is None:
            continue
        s, e = win
        b = basket_return(basket_px, s, e)
        spy_sub = spy_px.loc[s:e].dropna()
        if len(spy_sub) < 2:
            sp = np.nan
        else:
            sp = float(spy_sub.iloc[-1] / spy_sub.iloc[0] - 1)
        # Names that contributed (had data both endpoints)
        sub = basket_px.loc[s:e].dropna(how="all")
        names_used = [t for t in TICKERS
                      if t in sub.columns and not np.isnan(sub[t].iloc[0])
                      and not np.isnan(sub[t].iloc[-1])]
        rows.append({
            "year": y,
            "start": s.date(),
            "end": e.date(),
            "basket_ret": b,
            "spy_ret": sp,
            "n_names": len(names_used),
            "names": ",".join(names_used),
        })

    df = pd.DataFrame(rows).dropna(subset=["basket_ret"])
    print("\nAnnual event returns:")
    print(df.to_string(index=False))

    # Build daily-equivalent series for metrics: one "event return" per year.
    # We'll convert each annual event return to a daily-equivalent over the
    # event window so compute_metrics works on the daily PnL series.
    pnl_daily = pd.Series(0.0, index=idx)
    bench_daily = pd.Series(0.0, index=idx)
    for r in rows:
        if np.isnan(r["basket_ret"]):
            continue
        # Daily basket return = equal-weight average of basket constituents' daily rets
        win_dates = idx[(idx >= pd.Timestamp(r["start"])) & (idx <= pd.Timestamp(r["end"]))]
        if len(win_dates) < 2:
            continue
        # actual daily returns of equal-weight basket
        sub_px = basket_px.loc[win_dates]
        daily_eq = sub_px.pct_change().mean(axis=1)  # equal-weight cross-sectional avg
        pnl_daily.loc[win_dates] = daily_eq.fillna(0.0)
        # SPY daily
        spy_sub = spy_px.loc[win_dates].pct_change().fillna(0.0)
        bench_daily.loc[win_dates] = spy_sub

    # Only keep dates where we were actually in-position (event windows)
    in_position = (pnl_daily != 0.0) | (bench_daily != 0.0)
    pnl_e = pnl_daily[in_position]
    bench_e = bench_daily[in_position]

    if len(pnl_e) < 30:
        return mark_failed("AC-1", f"too few in-position days ({len(pnl_e)})")

    m = compute_metrics(pnl_e, benchmark=bench_e, name="AC-1 US FY-yearend defense surge")
    # Add explicit event-level stats
    annual_rets = df["basket_ret"].values
    annual_spy = df["spy_ret"].values
    excess = annual_rets - annual_spy
    n_events = len(annual_rets)
    mean_basket = float(np.mean(annual_rets))
    mean_spy = float(np.mean(annual_spy))
    mean_excess = float(np.mean(excess))
    hit_rate_evt = float(np.mean(annual_rets > 0))
    hit_vs_spy = float(np.mean(excess > 0))
    t_stat_excess = float(np.mean(excess) / (np.std(excess, ddof=1) / np.sqrt(n_events))) \
        if n_events > 1 and np.std(excess, ddof=1) > 0 else 0.0

    print(f"\nEvent-level: N={n_events}, mean_basket={mean_basket*100:.2f}%, "
          f"mean_spy={mean_spy*100:.2f}%, mean_excess={mean_excess*100:.2f}%, "
          f"hit_pos={hit_rate_evt*100:.1f}%, hit_vs_spy={hit_vs_spy*100:.1f}%, "
          f"t_excess={t_stat_excess:.2f}")
    print_metrics(m)

    save_result("AC-1", m, extra={
        "status": "ok",
        "rule": "Long EW {LMT,GD,LDOS,BAH,SAIC} from 5th TD of Sep through Sep 30.",
        "mechanism": "Use-it-or-lose-it federal contracting; ~16-18% of DoD obligations in FY-end month.",
        "source": "yfinance; calendar of US fiscal year end (Sep 30).",
        "n_events": int(n_events),
        "mean_basket_ret": mean_basket,
        "mean_spy_ret": mean_spy,
        "mean_excess_ret": mean_excess,
        "hit_rate_event": hit_rate_evt,
        "hit_rate_vs_spy": hit_vs_spy,
        "t_stat_excess": t_stat_excess,
        "annual_rets": [float(x) for x in annual_rets],
        "annual_spy_rets": [float(x) for x in annual_spy],
        "annual_years": [int(r["year"]) for r in rows if not np.isnan(r["basket_ret"])],
    })


if __name__ == "__main__":
    main()
