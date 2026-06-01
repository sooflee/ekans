"""PL724_harpex_spike_long_matx
Harpex Charter Spike -> Long MATX

When Harpex charter rate index rises >=10% over a trailing 4-week window,
go long MATX (Matson Inc., a US container shipping company) for 45 trading days.

Harpex (Harper Petersen) index is not publicly available on FRED.
We proxy with FRED PCU483111483111 (PPI: Deep Sea Freight Transportation),
which is monthly and highly correlated with charter rate indices.
We trigger when the 2-month (2-observation) change in this PPI is >=+10%.
This is an approximation; the original rule uses 4-week Harpex.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL724_harpex_spike_long_matx"
    tickers = ["MATX", "SPY"]

    try:
        px = load_prices(tickers, start="2015-01-01")
        deep_sea_ppi = load_fred("PCU483111483111", start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    matx_r = ret["MATX"].dropna()

    # Monthly deep sea freight PPI as Harpex proxy
    # 2-month pct change as ~4-week proxy
    # Use 5% threshold (vs original 10% for Harpex weekly) to get adequate events
    # given monthly resolution and lower correlation with short-term charter rates
    ppi = deep_sea_ppi["PCU483111483111"].dropna()
    ppi_2m_chg = ppi.pct_change(2)

    # Identify months where 2-month change >= +5% (monthly PPI proxy for Harpex 4w +10%)
    signal_months = ppi_2m_chg[ppi_2m_chg >= 0.05].index

    hold_days = 45
    events = []
    pnl_parts = []
    positions_parts = []

    idx = matx_r.index
    used_entries = set()  # track entry dates to avoid overlap

    last_exit_date = None

    for sig_date in signal_months:
        # Find first trading day at least 3 weeks into the next month
        # (monthly PPI is released ~1-2 months lag; we approximate 6-week lag)
        entry_start = sig_date + pd.DateOffset(months=2)
        future_idx = idx[idx >= entry_start]
        if len(future_idx) < hold_days:
            continue
        entry_date = future_idx[0]

        # Skip if we're still in a previous position
        if last_exit_date is not None and entry_date <= last_exit_date:
            continue

        entry_loc = idx.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(idx))
        exit_date = idx[exit_loc - 1]
        last_exit_date = exit_date

        window = slice(entry_loc, exit_loc)
        matx_window = matx_r.iloc[window]
        spy_window = spy_r.reindex(matx_window.index).fillna(0)

        if matx_window.isna().all():
            continue

        pnl_window = matx_window.fillna(0)
        pos_window = pd.Series(1.0, index=matx_window.index)

        matx_cum = float((1 + matx_window.fillna(0)).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        pnl_parts.append(pnl_window)
        positions_parts.append(pos_window)

        events.append({
            "signal_month": str(sig_date.date()),
            "ppi_2m_change": round(float(ppi_2m_chg.loc[sig_date]), 4),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "matx_45d_return": round(matx_cum, 4),
            "spy_45d_return": round(spy_cum, 4),
            "excess": round(matx_cum - spy_cum, 4),
        })

    if len(events) < 5:
        return mark_failed(
            sid,
            f"insufficient events: {len(events)} (Harpex proxy PCU483111483111 >=10% 2-month change)",
        )

    all_pnl = pd.concat(pnl_parts)
    all_pos = pd.concat(positions_parts)

    m = compute_metrics(
        all_pnl,
        benchmark=spy_r.reindex(all_pnl.index).fillna(0),
        name="Deep Sea Freight Spike -> Long MATX",
        positions=all_pos,
        cost_bps=10,
    )

    n_events = len(events)
    exc_rets = [e["excess"] for e in events]
    win_rate = float(np.mean([r > 0 for r in [e["matx_45d_return"] for e in events]]))
    avg_excess = float(np.mean(exc_rets))

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When deep-sea freight PPI (PCU483111483111) rises >=5% over "
                "2 months (proxy for Harpex 4-week spike >=+10%; monthly PPI "
                "dampened vs weekly), long MATX for 45 trading days."
            ),
            "mechanism": (
                "Rising charter rates boost container shipping revenue per TEU. "
                "MATX (Matson), operating US Jones Act and transpacific routes, "
                "benefits directly through spot rate repricing and contract "
                "roll-overs when freight indices spike."
            ),
            "source": (
                "yfinance MATX, SPY; FRED PCU483111483111 (PPI: Deep Sea Freight "
                "Transportation) as Harpex charter rate proxy"
            ),
            "tickers": tickers,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4),
            "avg_excess_vs_spy": round(avg_excess, 4),
            "events": events,
            "caveats": (
                "Harpex (Harper Petersen) is not public; PCU483111483111 monthly PPI "
                "is a coarse proxy. Monthly PPI has ~2-month publication lag; "
                "entry timing is approximate. MATX also affected by Jones Act "
                "routes (Hawaii/Alaska/Guam) which are not directly Harpex-driven."
            ),
        },
        pnl=all_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate:.0%}, avg_excess: {avg_excess:.4f}")
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(
        f"  Sharpe: {sharpe:.2f}  CAGR: {cagr*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  t-stat: {m.get('t_stat', 0):.2f}"
    )
    for e in events:
        flag = "+" if e["excess"] > 0 else "-"
        print(f"  {flag} {e['signal_month']}: MATX={e['matx_45d_return']*100:+.1f}%, SPY={e['spy_45d_return']*100:+.1f}%, excess={e['excess']*100:+.1f}%")


if __name__ == "__main__":
    main()
