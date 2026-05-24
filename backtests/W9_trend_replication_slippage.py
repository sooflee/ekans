"""
W9 Trend Replication Slippage.

Hypothesis: managed-futures ETFs (KMLM, DBMF, CTA) advertise themselves as
'CTA replication' vehicles but in practice carry costs, queue lag, and
broader-basket weights. A vanilla in-house TSMOM on liquid ETFs should
out-perform the replicator products. We test:
  - Long: equal-weight basket of {KMLM, DBMF, CTA}
  - Short: simple monthly TSMOM on {SPY, TLT, GLD, DBC, EFA, EEM} with
    12m-lookback (our C14 implementation)
  - Spread PnL = TSMOM_internal - ETF_basket

Source: Asness-Frazzini-Pedersen QPB notes 2023 + Resolve Asset Management's
        'replication slippage' commentary 2024.

Note: KMLM (KFA Mount Lucas) inception 2020-12; DBMF (iMGP DBi Managed
Futures) inception 2019-05; CTA (Simplify Managed Futures) inception 2022-03.
Spread back-test only meaningful from ~2022-03 onward when all three trade.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed,
)


def tsmom_pnl(px, lookback_months=12):
    """Simple monthly TSMOM PnL on a price dataframe.
    For each asset: if trailing 12m return > 0, long at 1/N; else flat that bucket.
    """
    n = px.shape[1]
    monthly = px.resample("ME").last()
    r12 = monthly.pct_change(lookback_months)
    sig = (r12 > 0).astype(float) / n
    daily_sig = sig.reindex(px.index, method="ffill").shift(1)
    rets = px.pct_change()
    return (daily_sig * rets).sum(axis=1)


def main():
    cta_tickers = ["KMLM", "DBMF", "CTA"]
    tsmom_tickers = ["SPY", "TLT", "GLD", "DBC", "EFA", "EEM"]
    try:
        cta_px = load_prices(cta_tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed("W9_trend_replication_slippage",
                            f"CTA ETF load failed: {e}")
    try:
        tsmom_px = load_prices(tsmom_tickers, start="2003-01-01")
    except Exception as e:
        return mark_failed("W9_trend_replication_slippage",
                            f"TSMOM basket load failed: {e}")

    cta_px = cta_px.dropna(how="any")
    if cta_px.empty:
        return mark_failed("W9_trend_replication_slippage",
                            "No overlapping CTA ETF data")

    # ETF basket: equal-weight daily return
    cta_ret_basket = cta_px.pct_change().mean(axis=1)

    # Internal TSMOM PnL (constructed on tsmom_px)
    internal_pnl = tsmom_pnl(tsmom_px, lookback_months=12)
    internal_pnl = internal_pnl.dropna()

    # Spread: long internal TSMOM, short ETF basket
    aligned = pd.concat([internal_pnl.rename("internal"),
                          cta_ret_basket.rename("etf_basket")], axis=1).dropna()
    if len(aligned) < 100:
        return mark_failed("W9_trend_replication_slippage",
                            f"Too few overlapping days: {len(aligned)}")

    aligned["spread"] = aligned["internal"] - aligned["etf_basket"]

    try:
        spy_px = load_prices(["SPY"], start="2003-01-01")
    except Exception:
        spy_px = None
    spy_ret = spy_px["SPY"].pct_change().reindex(aligned.index) if spy_px is not None else None

    m = compute_metrics(aligned["spread"], benchmark=spy_ret,
                        name="W9 internal TSMOM minus CTA-ETF basket")
    m_internal = compute_metrics(aligned["internal"], benchmark=spy_ret,
                                  name="W9 internal TSMOM (long-only)")
    m_etf = compute_metrics(aligned["etf_basket"], benchmark=spy_ret,
                             name="W9 ETF basket (long-only)")
    print_metrics(m)
    print_metrics(m_internal)
    print_metrics(m_etf)

    save_result("W9_trend_replication_slippage", m, extra={
        "status": "ok",
        "rule": ("Long the C14 TSMOM PnL (monthly 12m-lookback on SPY/TLT/GLD/"
                  "DBC/EFA/EEM, equal-weight signal). Short equal-weight basket "
                  "of KMLM + DBMF + CTA. Daily spread PnL."),
        "mechanism": ("Managed-futures replication ETFs target CTA returns but "
                       "deviate from a vanilla TSMOM due to broader baskets, "
                       "fee drag, queue lag, and explicit ranking schemes. The "
                       "spread captures replication slippage."),
        "source": ("Yfinance for KMLM/DBMF/CTA price series; methodology echoes "
                    "Asness-Frazzini-Pedersen and Resolve's 2024 'replication "
                    "slippage' analyses."),
        "universe": ("Long: SPY/TLT/GLD/DBC/EFA/EEM TSMOM (12m). "
                      "Short: KMLM + DBMF + CTA equal-weight basket."),
        "internal_metrics": {k: v for k, v in m_internal.items()
                                if not isinstance(v, dict)},
        "etf_metrics": {k: v for k, v in m_etf.items()
                          if not isinstance(v, dict)},
    })


if __name__ == "__main__":
    main()
