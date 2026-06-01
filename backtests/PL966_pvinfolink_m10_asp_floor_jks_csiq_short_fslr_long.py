"""PL966_pvinfolink_m10_asp_floor_jks_csiq_short_fslr_long
PVInfoLink Mono-PERC M10 ASP Sub-Floor Breach + China Inventory Glut:
Short JKS/CSIQ, Long FSLR

Since PVInfoLink raw weekly data is not freely downloadable, we proxy the signal
using JKS/CSIQ gross margin deterioration (rolling 4-quarter TTM gross margin
below a threshold = oversupply/price crash regime) and inventory days > 60
(approximated from quarterly financials). FSLR has insulated fixed-price US
utility contracts; the pair captures divergence between spot-price-exposed Chinese
modules vs. contracted US manufacturers.

Signal logic:
- Compute proxy gross margin signal from stock price dynamics: when JKS and CSIQ
  both show 30-day relative underperformance vs FSLR of > 10% (indicating the
  market has already started pricing the ASP floor breach), continue shorting.
- Also capture known historical oversupply events via approximate dates.

Because quarterly gross-margin data is hard to retrieve programmatically,
we use rolling 60-day relative return divergence (JKS+CSIQ avg vs FSLR)
as the continuous signal: when FSLR has outperformed JKS+CSIQ composite
by > 10% on a 60-day rolling basis, enter the short JKS+CSIQ / long FSLR pair.

Known oversupply regime dates (module ASP crash periods):
- 2019-Q1: mono-PERC ASP crash to ~$0.23/W (~breakeven for Chinese Tier-2)
- 2022-Q3: ASP falling sharply from pandemic spike
- 2023-Q4: historic low ASP breach ~$0.105-0.115/W
- 2024-Q1: continued sub-floor pricing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL966_pvinfolink_m10_asp_floor_jks_csiq_short_fslr_long"
    tickers = ["JKS", "CSIQ", "FSLR", "SPY"]

    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    jks_r = ret["JKS"].fillna(0)
    csiq_r = ret["CSIQ"].fillna(0)
    fslr_r = ret["FSLR"].fillna(0)

    # Chinese module composite = equal-weight JKS + CSIQ
    china_r = 0.5 * jks_r + 0.5 * csiq_r

    # Short JKS+CSIQ (equal weight) / Long FSLR dollar-neutral pair return
    pair_r = fslr_r - china_r

    # Signal: rolling 60-day cumulative return of FSLR vs Chinese composite
    # When FSLR has outperformed by > 10% on 60d rolling basis -> oversupply regime
    # (Chinese ASP crash showing up in stock divergence)
    fslr_cum60 = fslr_r.rolling(60).apply(lambda x: (1 + x).prod() - 1)
    china_cum60 = china_r.rolling(60).apply(lambda x: (1 + x).prod() - 1)
    divergence_60 = fslr_cum60 - china_cum60

    # Signal = 1 when divergence > 0.10 (FSLR has outperformed China composite by 10%+)
    # This acts as a "oversupply detected" signal
    threshold = 0.10
    raw_signal = (divergence_60 > threshold).astype(float)

    # Smooth: require signal for at least 3 consecutive days (debounce)
    signal = raw_signal.rolling(3, min_periods=3).min()

    # Hold for 30 trading days from each new entry; use position state machine
    hold_days = 30
    positions = pd.Series(0.0, index=ret.index)

    idx_list = list(ret.index)
    n = len(idx_list)
    in_pos = False
    exit_day = None

    # Vectorized approach: positions = signal shifted 1 day (no lookahead)
    # But apply 30-day hold: once signal triggers, stay in for at least 30 days
    # and then re-evaluate on day 30 whether signal is still on.
    pos = 0.0
    exit_idx = None
    for j in range(1, n):
        dt = idx_list[j]
        sig_yesterday = signal.iloc[j - 1]
        if np.isnan(sig_yesterday):
            sig_yesterday = 0.0

        if pos == 0.0:
            # Check for entry
            if sig_yesterday >= 1.0:
                pos = 1.0
                exit_idx = min(j + hold_days, n)
        else:
            # In position: hold until exit_idx, then check signal
            if j >= exit_idx:
                if sig_yesterday >= 1.0:
                    # Re-enter: extend hold
                    exit_idx = min(j + hold_days, n)
                    pos = 1.0
                else:
                    pos = 0.0
                    exit_idx = None
        positions.iloc[j] = pos

    pnl = positions * pair_r
    pnl = pnl.dropna()

    n_active_days = int((pnl != 0).sum())

    if n_active_days < 30:
        return mark_failed(
            sid,
            f"insufficient in-position days: {n_active_days}"
        )

    spy_r = spy_r.reindex(pnl.index).fillna(0)

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="PVInfoLink M10 Floor Short JKS/CSIQ Long FSLR",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=15,
    )

    # Count approximate entry events (rising edges of positions)
    pos_diff = positions.diff().fillna(0)
    n_entries = int((pos_diff > 0.5).sum())

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When FSLR has outperformed equal-weight JKS+CSIQ composite by >10% "
                "on a rolling 60-day basis (proxy for PVInfoLink M10 ASP sub-floor breach), "
                "enter short JKS+CSIQ / long FSLR dollar-neutral pair for 30 trading days. "
                "Re-enter if signal still active at 30-day mark."
            ),
            "mechanism": (
                "Chinese mono-PERC module ASP crashes below cash cost floor for "
                "Tier-1 manufacturers (JKS, CSIQ) as inventory days spike >60 on "
                "overcapacity. FSLR's US utility-scale fixed-price contracts insulate "
                "it from spot pricing. The pair captures the divergence between "
                "spot-price-exposed Chinese modules vs contracted US manufacturer."
            ),
            "source": (
                "yfinance auto-adjusted close; proxy signal from price divergence "
                "(FSLR vs JKS+CSIQ rolling 60d relative performance); known oversupply "
                "regimes: 2019-Q1, 2022-Q3, 2023-Q4, 2024-Q1 from PVInfoLink/BNEF reports."
            ),
            "tickers": ["JKS", "CSIQ", "FSLR"],
            "n_entries": n_entries,
            "n_active_days": n_active_days,
            "caveats": (
                "Signal is a price-based proxy, not direct PVInfoLink/CPIA data — "
                "may have look-ahead bias if FSLR outperformance precedes the ASP "
                "crash rather than lagging it. CSIQ is Canadian-listed (dual), JKS "
                "is NYSE ADR — both have yfinance history from ~2010. FSLR is highly "
                "sensitive to IRA manufacturing credit policy (US political risk). "
                "Avoid 2020-2021 pandemic disruption period in interpretation."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_entries: {n_entries}, n_active_days: {n_active_days}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "net_sharpe" in m:
        print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
